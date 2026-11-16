#!/usr/bin/env python3
"""Branchy quality benchmark: how good is a branched model compared with the sequential model of the same size?

  quality_bench.py --models seq=.../seq/inference.pt flat8=.../flat8/inference.pt --reference seq --out docs/results/Q1

Four parts, all on the same inputs for every model, each reported next to the reference with a paired 95% interval:

1. Held-out loss    1000 windows x 512 tokens of the TinyStories validation file (float32): loss, top-token accuracy,
                    loss by position in the context (does the model use long context?).
2. Rare-word reuse  top-token accuracy on rare tokens (names, objects) that already appeared at least 64 tokens earlier in
                    the window: a probe for long-range memory, the first thing narrow branches could lose.
3. Grammar          BLiMP: 67 phenomena x 1000 minimal pairs; the model should give the grammatical sentence the higher
                    probability. Chance is 50%. Mostly outside the training vocabulary, so it also tests robustness.
4. Stories          the same 200 story beginnings continued by every model (same sampling seed): repetition and
                    diversity measures here; the judged comparison is done by judge_stories.py on the saved stories.

Models are BranchyN inference files (`config` + `model`), as written by src/proxy/train_300m.py.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'proxy'))
from branchyn_model import BranchyN, Config  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'runs/rocm_300m_train/data'
BLIMP = ROOT / 'runs/quality/blimp'
BUCKETS = [(0, 16), (16, 64), (64, 256), (256, 512)]


def load(path, device):
    ck = torch.load(path, map_location='cpu', weights_only=True)
    model = BranchyN(Config(**ck['config']))
    model.load_state_dict(ck['model'])
    return model.eval().to(device)


def interval(delta):
    """Mean and 95% half-width of paired differences."""
    d = np.asarray(delta, dtype=np.float64)
    return float(d.mean()), float(1.96 * d.std(ddof=1) / math.sqrt(len(d)))


@torch.no_grad()
def part_loss(model, valid, starts, rare, device, batch=8, context=512):
    per_window, acc_window, bucket = [], [], np.zeros((len(starts), len(BUCKETS)))
    reuse_hit, reuse_n = np.zeros(len(starts)), np.zeros(len(starts))
    for i in range(0, len(starts), batch):
        idx = starts[i:i + batch]
        arr = np.stack([valid[s:s + context + 1].astype(np.int64) for s in idx])
        x, y = torch.from_numpy(arr[:, :-1]).to(device), torch.from_numpy(arr[:, 1:]).to(device)
        logits = model(x).float()
        loss = F.cross_entropy(logits.transpose(1, 2), y, reduction='none')            # [B, T]
        hit = logits.argmax(-1) == y
        per_window += loss.mean(1).tolist()
        acc_window += hit.float().mean(1).tolist()
        for k, (a, b) in enumerate(BUCKETS):
            bucket[i:i + len(idx), k] = loss[:, a:b].mean(1).cpu().numpy()
        # rare target tokens that occurred at least 64 positions earlier in the same window
        full = torch.from_numpy(arr).to(device)                                           # [B, T+1]; target of position t is full[t+1]
        seen = torch.zeros(len(idx), context, dtype=torch.bool, device=device)
        for t in range(64, context):
            seen[:, t] = (full[:, :t + 1 - 64] == y[:, t:t + 1]).any(1)
        mask = seen & rare[y]
        reuse_hit[i:i + len(idx)] = (hit & mask).sum(1).cpu().numpy()
        reuse_n[i:i + len(idx)] = mask.sum(1).cpu().numpy()
    return dict(loss=np.array(per_window), acc=np.array(acc_window), bucket=bucket, reuse_hit=reuse_hit, reuse_n=reuse_n)


@torch.no_grad()
def sentence_scores(model, tok, sentences, device, batch=256):
    """Sum of token log-probabilities of each sentence, given a preceding full stop."""
    prefix = tok.encode('.').ids
    enc = [prefix + tok.encode(' ' + s).ids for s in sentences]
    out = np.zeros(len(enc))
    order = np.argsort([len(e) for e in enc])
    for i in range(0, len(enc), batch):
        idx = order[i:i + batch]
        width = max(len(enc[j]) for j in idx)
        x = torch.zeros(len(idx), width, dtype=torch.long, device=device)
        m = torch.zeros(len(idx), width, dtype=torch.bool, device=device)
        for r, j in enumerate(idx):
            x[r, :len(enc[j])] = torch.tensor(enc[j], device=device)
            m[r, len(prefix):len(enc[j])] = True
        logp = F.log_softmax(model(x[:, :-1]).float(), -1).gather(-1, x[:, 1:, None])[..., 0]
        out[idx] = (logp * m[:, 1:]).sum(1).cpu().numpy()
    return out


def part_grammar(model, tok, device, per_set):
    import pyarrow.parquet as pq
    rows = []
    for f in sorted(glob.glob(str(BLIMP / '*' / '*.parquet'))):
        t = pq.read_table(f, columns=['sentence_good', 'sentence_bad', 'field', 'UID']).slice(0, per_set).to_pylist()
        good = sentence_scores(model, tok, [r['sentence_good'] for r in t], device)
        bad = sentence_scores(model, tok, [r['sentence_bad'] for r in t], device)
        rows += [dict(uid=r['UID'], field=r['field'], ok=bool(g > b)) for r, g, b in zip(t, good, bad)]
    return rows


@torch.no_grad()
def part_stories(model, prompts, new_tokens, device, seed, temperature=0.8, top_p=0.95, batch=50):
    out = []
    gen = torch.Generator(device=device).manual_seed(seed)
    for i in range(0, len(prompts), batch):
        x = torch.tensor(prompts[i:i + batch], dtype=torch.long, device=device)
        for _ in range(new_tokens):
            logits = model(x)[:, -1].float() / temperature
            probs = F.softmax(logits, -1)
            sp, si = probs.sort(-1, descending=True)
            keep = sp.cumsum(-1) - sp < top_p
            sp = sp * keep
            pick = torch.multinomial(sp / sp.sum(-1, keepdim=True), 1, generator=gen)
            x = torch.cat([x, si.gather(-1, pick)], 1)
        out += x[:, len(prompts[0]):].tolist()
    return out


def story_stats(stories):
    def ngrams(s, n):
        return [tuple(s[i:i + n]) for i in range(len(s) - n + 1)]
    repeat = [1 - len(set(ngrams(s, 4))) / max(1, len(ngrams(s, 4))) for s in stories]
    all3 = [g for s in stories for g in ngrams(s, 3)]
    return dict(repeated_4gram_share=float(np.mean(repeat)), distinct_3gram_share=len(set(all3)) / max(1, len(all3)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--models', nargs='+', required=True, help='name=path/to/inference.pt')
    ap.add_argument('--reference', required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--windows', type=int, default=1000)
    ap.add_argument('--blimp-per-set', type=int, default=1000)
    ap.add_argument('--stories', type=int, default=200)
    ap.add_argument('--prompt-tokens', type=int, default=48)
    ap.add_argument('--new-tokens', type=int, default=160)
    ap.add_argument('--seed', type=int, default=20261113)
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(DATA / 'tokenizer.json'))
    device = torch.device(args.device)
    valid = np.fromfile(DATA / 'valid.bin', dtype=np.uint16)
    train = np.fromfile(DATA / 'train.bin', dtype=np.uint16)
    counts = np.bincount(train, minlength=8192)
    rare = torch.from_numpy(counts < np.sort(counts)[-1000]).to(device)                 # not among the 1000 most frequent tokens
    rng = np.random.default_rng(args.seed)
    starts = np.sort(rng.choice(len(valid) - 513, size=args.windows, replace=False))
    text = (DATA / 'TinyStoriesV2-GPT4-valid.txt').read_text(errors='ignore').split('<|endoftext|>')[1:]
    prompts, texts = [], []
    for story in text:
        ids = tok.encode(story.strip()).ids
        if len(ids) >= args.prompt_tokens + 100:
            prompts.append(ids[:args.prompt_tokens])
            texts.append(tok.decode(ids[:args.prompt_tokens]))
        if len(prompts) == args.stories:
            break
    args.out.mkdir(parents=True, exist_ok=True)
    names = [m.split('=', 1)[0] for m in args.models]
    assert args.reference in names
    raw, report = {}, dict(reference=args.reference, windows=args.windows, tokens=args.windows * 512, stories=len(prompts),
                           blimp_pairs=None, seed=args.seed, models={})
    for spec in args.models:
        name, path = spec.split('=', 1)
        t0 = time.time()
        model = load(path, device)
        params = sum(p.numel() for p in model.parameters())
        loss = part_loss(model, valid, starts, rare, device)
        grammar = part_grammar(model, tok, device, args.blimp_per_set)
        stories = part_stories(model, prompts, args.new_tokens, device, args.seed)
        raw[name] = dict(loss=loss, grammar=grammar)
        with open(args.out / f'stories_{name}.jsonl', 'w') as f:
            for p, s in zip(texts, stories):
                f.write(json.dumps(dict(beginning=p, continuation=tok.decode(s))) + '\n')
        report['models'][name] = dict(path=path, parameters=params, config=model.cfg.kind, stories=story_stats(stories))
        report['blimp_pairs'] = len(grammar)
        print(f'{name}: loss {loss["loss"].mean():.4f}, grammar {np.mean([g["ok"] for g in grammar]) * 100:.1f}%, '
              f'{time.time() - t0:.0f}s', flush=True)
        del model
        torch.cuda.empty_cache()
    ref = raw[args.reference]
    fields = sorted({g['field'] for g in ref['grammar']})
    for name in names:
        r, m = raw[name], report['models'][name]
        d, ci = interval(r['loss']['loss'] - ref['loss']['loss'])
        m['loss'] = dict(value=float(r['loss']['loss'].mean()), perplexity=float(math.exp(r['loss']['loss'].mean())), vs_reference=d, ci95=ci)
        d, ci = interval(r['loss']['acc'] - ref['loss']['acc'])
        m['accuracy'] = dict(value=float(r['loss']['acc'].mean()), vs_reference=d, ci95=ci)
        m['loss_by_position'] = {f'{a + 1}-{b}': dict(value=float(r['loss']['bucket'][:, k].mean()),
                                                      vs_reference=interval(r['loss']['bucket'][:, k] - ref['loss']['bucket'][:, k])[0])
                                 for k, (a, b) in enumerate(BUCKETS)}
        keep = ref['loss']['reuse_n'] > 0
        rate = lambda x: x['loss']['reuse_hit'][keep] / x['loss']['reuse_n'][keep]      # noqa: E731
        d, ci = interval(rate(r) - rate(ref))
        m['rare_reuse'] = dict(accuracy=float(r['loss']['reuse_hit'].sum() / r['loss']['reuse_n'].sum()), cases=int(r['loss']['reuse_n'].sum()),
                               vs_reference=d, ci95=ci)
        ok, ok_ref = np.array([g['ok'] for g in r['grammar']], float), np.array([g['ok'] for g in ref['grammar']], float)
        d, ci = interval(ok - ok_ref)
        m['grammar'] = dict(accuracy=float(ok.mean()), vs_reference=d, ci95=ci,
                            by_field={f: float(np.mean([g['ok'] for g in r['grammar'] if g['field'] == f])) for f in fields})
    (args.out / 'quality.json').write_text(json.dumps(report, indent=1))
    pm = lambda v, c: f'{v:+.3f} ±{c:.3f}'                                               # noqa: E731
    md = ['# Branchy quality benchmark', '',
          f"Reference: `{args.reference}`. {args.windows} windows x 512 held-out tokens; {report['blimp_pairs']:,} grammar pairs; "
          f"{len(prompts)} story beginnings. Differences are paired, with a 95% interval over the test items (not over training seeds).", '',
          '| Model | Held-out loss | vs reference | Top-token accuracy | Rare-word reuse | vs reference | Grammar pairs | vs reference | Repeated 4-grams in stories |',
          '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for name in names:
        m = report['models'][name]
        md.append(f"| {name} | {m['loss']['value']:.3f} | {pm(m['loss']['vs_reference'], m['loss']['ci95'])} | {m['accuracy']['value'] * 100:.2f}% | "
                  f"{m['rare_reuse']['accuracy'] * 100:.1f}% | {pm(m['rare_reuse']['vs_reference'] * 100, m['rare_reuse']['ci95'] * 100)} | "
                  f"{m['grammar']['accuracy'] * 100:.1f}% | {pm(m['grammar']['vs_reference'] * 100, m['grammar']['ci95'] * 100)} | "
                  f"{m['stories']['repeated_4gram_share'] * 100:.1f}% |")
    md += ['', '## Loss by position in the context (difference to the reference)', '',
           '| Model | ' + ' | '.join(f'tokens {a + 1}-{b}' for a, b in BUCKETS) + ' |', '|---|' + '---:|' * len(BUCKETS)]
    for name in names:
        m = report['models'][name]['loss_by_position']
        md.append(f'| {name} | ' + ' | '.join(f"{v['value']:.3f} ({v['vs_reference']:+.3f})" for v in m.values()) + ' |')
    md += ['', '## Grammar by field (share of pairs where the grammatical sentence is preferred)', '',
           '| Model | ' + ' | '.join(fields) + ' |', '|---|' + '---:|' * len(fields)]
    for name in names:
        g = report['models'][name]['grammar']['by_field']
        md.append(f'| {name} | ' + ' | '.join(f'{g[f] * 100:.1f}%' for f in fields) + ' |')
    (args.out / 'quality.md').write_text('\n'.join(md) + '\n')
    print(f'saved {args.out / "quality.md"}', flush=True)


if __name__ == '__main__':
    main()
