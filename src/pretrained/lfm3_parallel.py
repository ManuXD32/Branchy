#!/usr/bin/env python3
"""E6: turn a trained sequential model (LFM3-1.7B-Base) into a Branchy-shaped one WITHOUT training and measure the damage.

Mapping: trunk = first layers (unchanged); branches = groups of consecutive middle layers, every group starts from
the trunk output and runs its own layers in order; fusion = trunk output + scale * sum of the branches' updates;
final = last layers (unchanged). Controls: the untouched model, and "skip" (the region's layers deleted).

  lfm3_parallel.py --out ../../docs/results/E6_lfm3_parallel            (full sweep, GPU)
  lfm3_parallel.py --device cpu --windows 2 --context 128 --quick       (functional check)

The layers are wrapped, not rewritten: the model's own forward loop is used, each wrapper decides which tensor
its layer reads and what it hands on. No KV cache (full-sequence evaluation only).
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / 'runs/pretrained/LFM3-1.7B-Base'
TOPK = 50
TINY = ROOT / 'runs/rocm_300m_train/data/TinyStoriesV2-GPT4-valid.txt'


class Plan:
    """Shared state for one forward pass."""

    def __init__(self):
        self.roles = {}       # layer index -> (first_of_region, first_of_branch, last_of_branch, last_of_region)
        self.scale = 1.0
        self.skip = set()     # layer indices that are deleted (identity)
        self.teacher = False  # True: layers that have a frozen original copy (Wrapped.alt) use it
        self.h0 = None
        self.acc = None

    def clear(self):
        self.roles, self.skip, self.scale = {}, set(), 1.0

    def set_branches(self, branches, fusion='sum'):
        """branches: list of (start, end) half-open layer ranges, contiguous and in order."""
        self.clear()
        for b, (start, end) in enumerate(branches):
            for i in range(start, end):
                self.roles[i] = (b == 0 and i == start, i == start, i == end - 1, b == len(branches) - 1 and i == end - 1)
        self.scale = 1.0 if fusion == 'sum' else 1.0 / len(branches)

    def set_skip(self, start, end):
        self.clear()
        self.skip = set(range(start, end))


class Wrapped(nn.Module):
    def __init__(self, layer, index, plan: Plan):
        super().__init__()
        self.layer, self.index, self.plan = layer, index, plan
        self.alt = None       # frozen copy of the original layer (set when the layer itself is being retrained)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(super().__getattr__('layer'), name)

    def forward(self, hidden_states, *args, **kwargs):
        plan = self.plan
        if self.index in plan.skip:
            return hidden_states
        layer = self.alt if (plan.teacher and self.alt is not None) else self.layer
        role = plan.roles.get(self.index)
        if role is None:
            return layer(hidden_states, *args, **kwargs)
        first_region, first_branch, last_branch, last_region = role
        if first_region:
            plan.h0, plan.acc = hidden_states, torch.zeros_like(hidden_states)
        out = layer(plan.h0 if first_branch else hidden_states, *args, **kwargs)
        if last_branch:
            plan.acc = plan.acc + (out - plan.h0)
        if last_region:
            out = plan.h0 + plan.scale * plan.acc
            plan.h0 = plan.acc = None
        return out


def load(device, dtype, model_dir=MODEL_DIR):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    remote = (Path(model_dir) / 'modeling_lfm3.py').exists()      # only LFM3 ships its own (reviewed) model code
    tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=remote)
    model = AutoModelForCausalLM.from_pretrained(model_dir, trust_remote_code=remote, dtype=dtype).to(device).eval()
    plan = Plan()
    model.model.layers = nn.ModuleList([Wrapped(layer, i, plan) for i, layer in enumerate(model.model.layers)])
    return tok, model, plan


def windows_from_text(tok, text, count, context):
    ids = tok(text, add_special_tokens=False)['input_ids']
    head = [] if tok.bos_token_id is None else [tok.bos_token_id]          # Qwen has no BOS token
    body = context - len(head)
    out = [head + ids[i * body:(i + 1) * body] for i in range(count) if (i + 1) * body <= len(ids)]
    return torch.tensor(out, dtype=torch.long)


def datasets(tok, count, context, names=('tinystories', 'wikitext2')):
    """Evaluation windows per kind of text. The first two are the E6/E7 sets (unchanged); more kinds: textsets.py."""
    import textsets
    data = {}
    for name in names:
        data[name] = windows_from_text(tok, textsets.eval_text(name, count * context * 8), count, context)
        assert len(data[name]) == count, f'{name}: only {len(data[name])} evaluation windows'
    return data


@torch.no_grad()
def evaluate(model, windows, device, batch, reference=None, autocast=False):
    """Returns metrics and, when reference is None, the untouched model's top-50 log-probs per token (used for
    top-token agreement and a KL estimate restricted to those tokens)."""
    nll = total = agree = 0
    kl = 0.0
    keep = []
    for n, i in enumerate(range(0, len(windows), batch)):
        ids = windows[i:i + batch].to(device)
        with torch.autocast('cuda', dtype=torch.float16, enabled=autocast and ids.is_cuda):
            logits = model(input_ids=ids, use_cache=False).logits[:, :-1]
        logits = logits.float()
        logp = F.log_softmax(logits, dim=-1)
        target = ids[:, 1:]
        nll += float(-logp.gather(-1, target[..., None]).sum())
        total += target.numel()
        if reference is None:
            top = logp.topk(TOPK, dim=-1)                       # the full distributions would not fit in RAM
            keep.append((top.values.cpu(), top.indices.cpu()))
        else:
            ref_logp, ref_idx = (t.to(device) for t in reference[n])
            agree += int((logp.argmax(-1) == ref_idx[..., 0]).sum())
            kl += float((ref_logp.exp() * (ref_logp - logp.gather(-1, ref_idx))).sum())
        del logits, logp
    out = dict(nll=nll / total, ppl=math.exp(nll / total), tokens=total)
    if reference is not None:
        out.update(top1_agreement=agree / total, kl_top50_from_original=kl / total)
    return out, keep


def configurations(n_layers, quick, regions=None):
    regions = regions or ([(8, 16), (6, 18)] if quick else [(8, 16), (6, 18), (4, 20), (2, 22), (4, 12), (12, 20), (2, 18), (6, 22)])
    for start, end in regions:
        length = end - start
        yield dict(kind='skip', start=start, end=end)
        for size in (1, 2, 3, 4, 6, 8):
            if length % size == 0 and length // size >= 2:
                branches = [(s, s + size) for s in range(start, end, size)]
                for fusion in ('sum', 'mean'):
                    yield dict(kind='parallel', start=start, end=end, layers_per_branch=size, branches=len(branches),
                               fusion=fusion, ranges=branches)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model', type=Path, default=MODEL_DIR)
    ap.add_argument('--regions', default=None, help='e.g. "8-16,6-18,4-20" (default: the eight E6 regions)')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--dtype', default=None, help='float16 | bfloat16 | float32 (default: float16 on GPU, float32 on CPU)')
    ap.add_argument('--windows', type=int, default=48)
    ap.add_argument('--context', type=int, default=512)
    ap.add_argument('--batch', type=int, default=2)
    ap.add_argument('--quick', action='store_true')
    ap.add_argument('--out', type=Path, default=ROOT / 'docs/results/E6_lfm3_parallel')
    args = ap.parse_args()
    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype or ('float16' if device.type == 'cuda' else 'float32'))
    t0 = time.time()
    tok, model, plan = load(device, dtype, args.model)
    cfg = model.config
    print(f'loaded {args.model.name}: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B parameters, {cfg.num_hidden_layers} layers '
          f'{getattr(cfg, "layer_types", "")}, {dtype} on {device} ({time.time() - t0:.0f}s)', flush=True)
    data = datasets(tok, args.windows, args.context)
    results = dict(model=str(args.model), revision=(args.model / 'REVISION').read_text().strip(), dtype=str(dtype),
                   windows=args.windows, context=args.context, layer_types=getattr(cfg, 'layer_types', None), rows=[])
    reference = {}
    plan.clear()
    base = {}
    for name, windows in data.items():
        base[name], reference[name] = evaluate(model, windows, device, args.batch)
        print(f'  original           {name:12s} NLL {base[name]["nll"]:.4f} PPL {base[name]["ppl"]:.2f} ({base[name]["tokens"]} tokens)', flush=True)
    results['original'] = base
    regions = [tuple(int(v) for v in r.split('-')) for r in args.regions.split(',')] if args.regions else None
    for conf in configurations(cfg.num_hidden_layers, args.quick, regions):
        if conf['kind'] == 'skip':
            plan.set_skip(conf['start'], conf['end'])
            label = f"skip layers {conf['start']}-{conf['end'] - 1}"
        else:
            plan.set_branches(conf['ranges'], conf['fusion'])
            label = f"layers {conf['start']}-{conf['end'] - 1} as {conf['branches']} branches x {conf['layers_per_branch']} ({conf['fusion']})"
        row = dict(conf, label=label, metrics={})
        for name, windows in data.items():
            row['metrics'][name] = evaluate(model, windows, device, args.batch, reference[name])[0]
        results['rows'].append(row)
        print(f'  {label:52s} ' + ' | '.join(f"{n}: NLL {m['nll']:.3f} (+{m['nll'] - base[n]['nll']:.3f}) agree {m['top1_agreement'] * 100:.0f}%"
                                              for n, m in row['metrics'].items()), flush=True)
    plan.clear()
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / 'sweep.json').write_text(json.dumps(results, indent=1))
    names = list(data)
    md = [f'# {args.model.name} with middle layers run as parallel branches (no training)', '',
          f"Model revision {results['revision'][:12]}, {dtype}, {args.windows} windows x {args.context} tokens per dataset. "
          'NLL in nats per token; "+" is the increase over the untouched model.', '',
          '| Configuration | Trunk layers | Branches | Final layers | ' + ' | '.join(f'{n} NLL | + | same top token' for n in names) + ' |',
          '|---|---:|---|---:|' + '---:|---:|---:|' * len(names),
          '| original (sequential) | 24 | - | - | ' + ' | '.join(f"{base[n]['nll']:.3f} | 0 | 100%" for n in names) + ' |']
    for r in results['rows']:
        branches = 'deleted' if r['kind'] == 'skip' else f"{r['branches']} x {r['layers_per_branch']} layers, {r['fusion']}"
        md.append(f"| {r['label']} | {r['start']} | {branches} | {cfg.num_hidden_layers - r['end']} | " +
                  ' | '.join(f"{r['metrics'][n]['nll']:.3f} | {r['metrics'][n]['nll'] - base[n]['nll']:+.3f} | "
                             f"{r['metrics'][n]['top1_agreement'] * 100:.0f}%" for n in names) + ' |')
    (args.out / 'sweep.md').write_text('\n'.join(md) + '\n')
    print(f'saved {args.out / "sweep.md"} ({(time.time() - t0) / 60:.1f} min)', flush=True)


if __name__ == '__main__':
    main()
