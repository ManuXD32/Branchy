#!/usr/bin/env python3
"""Train one small-proxy BranchyN variant on TinyStories and evaluate it on fixed held-out windows.

  train_proxy.py --variant flat8 --out ../../docs/results/E2_proxy --ckpt ../../runs/e2_proxy

Every variant gets the same data, tokenizer, token budget, optimizer and (within about 1%) parameter count;
branch width is solved to match the flat4 reference. Results: <out>/<name>/result.json + curve.csv.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from branchyn_model import (BranchyN, Config, count, fixed_groups, initialize, match_branch_width, random_partition,
                            unit_bytes)

ROOT_DIR = Path(__file__).resolve().parents[2]      # the repository

DATA = (ROOT_DIR / 'runs/rocm_300m_train/data')


def variant_config(name: str, context: int) -> Config:
    base = dict(context=context)
    ref = Config(**base)                                   # flat4: the parameter-count reference
    target = count(BranchyN(ref))
    if name == 'seq':
        return Config(kind='sequential', **base)
    if name == 'flat4':
        return ref
    parts = name.split('-')
    head, opts = parts[0], parts[1:]
    kw = dict(base)
    for o in opts:
        if o.startswith('delay'):
            kw['delay'] = int(o[5:])
        elif o.startswith('drop'):
            kw['drop_p'] = int(o[4:]) / 100
        elif o.startswith('g'):
            kw['groups'] = int(o[1:])
        else:
            raise ValueError(f'unknown option {o}')
    for kind in ('flat', 'grouped', 'shared'):
        if head.startswith(kind):
            kw.update(fusion=kind, n_branches=int(head[len(kind):]))
            break
    else:
        raise ValueError(f'unknown variant {name}')
    cfg = Config(**kw)
    if cfg.n_branches == 4 and cfg.fusion == 'flat':
        return cfg
    return match_branch_width(cfg, target)[0]


class Data:
    def __init__(self, device, context):
        manifest = json.loads((DATA / 'manifest.json').read_text())
        assert manifest['vocab_actual'] == 8192
        self.train = torch.from_numpy(np.fromfile(DATA / 'train.bin', dtype=np.uint16).astype(np.int16)).to(device)
        self.valid = torch.from_numpy(np.fromfile(DATA / 'valid.bin', dtype=np.uint16).astype(np.int16)).to(device)
        self.context, self.device = context, device
        self.offsets = torch.arange(context + 1, device=device)

    def batch(self, source, starts):
        chunk = source[starts[:, None] + self.offsets].long()
        return chunk[:, :-1], chunk[:, 1:]

    def train_batch(self, size, gen):
        starts = torch.randint(0, len(self.train) - self.context - 1, (size,), generator=gen).to(self.device)
        return self.batch(self.train, starts)

    def valid_windows(self, count_, seed=314159):
        gen = torch.Generator().manual_seed(seed)
        return torch.randint(0, len(self.valid) - self.context - 1, (count_,), generator=gen).to(self.device)


@torch.no_grad()
def evaluate(model, data, starts, batch=32, partition=None, keep_fn=None, reference=None):
    """Token-weighted NLL / accuracy over fixed windows. keep_fn(B) -> bool [B, N] drops branches.
    With `reference` (list of argmax tensors from another run) also returns argmax agreement."""
    model.eval()
    nll = correct = total = agree = 0
    preds = []
    for i in range(0, len(starts), batch):
        x, y = data.batch(data.valid, starts[i:i + batch])
        keep = keep_fn(x.shape[0]) if keep_fn else None
        with torch.autocast('cuda', dtype=torch.float16, enabled=x.is_cuda):
            logits = model(x, partition=partition, keep=keep)
        logits = logits.float()
        nll += float(F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1), reduction='sum'))
        arg = logits.argmax(-1)
        correct += int((arg == y).sum())
        total += y.numel()
        if reference is not None:
            agree += int((arg == reference[len(preds)]).sum())
        preds.append(arg)
    model.train()
    out = dict(nll=nll / total, ppl=math.exp(nll / total), acc=correct / total, tokens=total)
    if reference is not None:
        out['agreement_with_full'] = agree / total
    return out, preds


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--variant', required=True)
    ap.add_argument('--name')
    ap.add_argument('--steps', type=int, default=5000)
    ap.add_argument('--batch', type=int, default=8, help='micro-batch (sequences per forward pass)')
    ap.add_argument('--accum', type=int, default=2, help='micro-batches per optimizer step')
    ap.add_argument('--context', type=int, default=256)
    ap.add_argument('--lr', type=float, default=5e-4)
    ap.add_argument('--warmup', type=int, default=200)
    ap.add_argument('--seed', type=int, default=1337)
    ap.add_argument('--eval-every', type=int, default=500)
    ap.add_argument('--final-windows', type=int, default=128)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--out', type=Path, default=(ROOT_DIR / 'docs/results/E2_proxy'))
    ap.add_argument('--ckpt', type=Path, default=(ROOT_DIR / 'runs/e2_proxy'))
    args = ap.parse_args()
    name = args.name or (args.variant if args.seed == 1337 else f'{args.variant}_s{args.seed}')
    out = args.out / name
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    cfg = variant_config(args.variant, args.context)
    model = initialize(BranchyN(cfg), args.seed).to(device)
    params = count(model)
    unit_mib = unit_bytes(cfg) / 2**20 if cfg.kind == 'branched' else None
    print(f'{name}: {params:,} parameters; {json.dumps(asdict(cfg))}', flush=True)
    if unit_mib:
        print(f'  branch unit: width {cfg.branch_width}, {unit_mib:.1f} MiB fp32 each, {cfg.n_branches} units', flush=True)
    data = Data(device, args.context)
    decay = [p for n, p in model.named_parameters() if p.dim() >= 2]
    no_decay = [p for n, p in model.named_parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([dict(params=decay, weight_decay=0.1), dict(params=no_decay, weight_decay=0.0)],
                            lr=args.lr, betas=(0.9, 0.95))
    scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda')
    gen = torch.Generator().manual_seed(args.seed + 41)
    prng = random.Random(args.seed + 7)
    quick = data.valid_windows(16)
    final = data.valid_windows(args.final_windows)
    curve = []
    tokens_per_step = args.batch * args.accum * args.context
    model.train()
    t0 = time.perf_counter()
    running = 0.0
    for step in range(1, args.steps + 1):
        lr = args.lr * step / args.warmup if step <= args.warmup else \
            args.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * (step - args.warmup) / max(1, args.steps - args.warmup))))
        for group in opt.param_groups:
            group['lr'] = lr
        opt.zero_grad(set_to_none=True)
        step_loss = 0.0
        for _ in range(args.accum):
            x, y = data.train_batch(args.batch, gen)
            partition = None
            if cfg.kind == 'branched' and cfg.fusion == 'shared':
                partition = random_partition(cfg.n_branches, prng)
            with torch.autocast('cuda', dtype=torch.float16, enabled=device.type == 'cuda'):
                logits = model(x, partition=partition)
                loss = F.cross_entropy(logits.reshape(-1, cfg.vocab).float(), y.reshape(-1))
            scaler.scale(loss / args.accum).backward()
            step_loss += float(loss) / args.accum
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        running = step_loss if step == 1 else 0.98 * running + 0.02 * step_loss
        if not math.isfinite(running):
            raise SystemExit('loss diverged')
        if step % args.eval_every == 0 or step == args.steps or step == 1:
            val, _ = evaluate(model, data, quick)
            elapsed = time.perf_counter() - t0
            tps = step * tokens_per_step / elapsed
            curve.append(dict(step=step, tokens=step * tokens_per_step, train_loss=running, val_nll=val['nll'], elapsed_s=elapsed))
            vram = torch.cuda.max_memory_allocated() / 2**20 if device.type == 'cuda' else 0
            print(f'  step {step}/{args.steps} train {running:.4f} val {val["nll"]:.4f} | {tps:,.0f} tok/s | VRAM {vram:.0f} MiB | '
                  f'{elapsed / 60:.1f} min', flush=True)
    train_s = time.perf_counter() - t0
    # ---------------------------------------------------------------- final evaluation
    result = dict(name=name, variant=args.variant, seed=args.seed, config=asdict(cfg), parameters=params, unit_mib=unit_mib,
                  steps=args.steps, batch=args.batch, accum=args.accum, context=args.context, train_tokens=args.steps * tokens_per_step,
                  train_minutes=train_s / 60, tokens_per_s=args.steps * tokens_per_step / train_s,
                  peak_vram_mib=torch.cuda.max_memory_allocated() / 2**20 if device.type == 'cuda' else 0, curve=curve)
    full, full_preds = evaluate(model, data, final)
    result['final'] = full
    print(f'  FINAL {name}: NLL {full["nll"]:.4f} PPL {full["ppl"]:.3f} acc {full["acc"] * 100:.2f}% on {full["tokens"]} held-out tokens', flush=True)
    if cfg.kind == 'branched':
        n, local = cfg.n_branches, cfg.local_branches
        if cfg.fusion == 'shared':
            parts = {'one_group': [list(range(n))], 'singletons': [[i] for i in range(n)]}
            for g in (2, 4, 8):
                if g < n:
                    parts[f'{g}_equal_groups'] = fixed_groups(n, g)
            if n >= 16:
                parts['plan_11_4_1'] = [list(range(0, 11)), list(range(11, 15)), [15]]
            result['partitions'] = {k: evaluate(model, data, final, partition=p)[0] for k, p in parts.items()}
            for k, v in result['partitions'].items():
                print(f'    partition {k:16s} NLL {v["nll"]:.4f}', flush=True)
        if cfg.fusion == 'flat':
            # quality when k of the non-local branches are missing (zeros), averaged over 3 random subsets
            drops = {}
            g2 = torch.Generator().manual_seed(99)
            for k in sorted({0, 1, 2, (n - local) // 2, n - local}):
                rows = []
                for _ in range(1 if k in (0, n - local) else 3):
                    gone = (torch.randperm(n - local, generator=g2)[:k] + local).tolist()

                    def keep_fn(b, gone=gone):
                        keep = torch.ones(b, n, dtype=torch.bool, device=device)
                        keep[:, gone] = False
                        return keep
                    rows.append(evaluate(model, data, final, keep_fn=keep_fn, reference=full_preds)[0])
                drops[str(k)] = dict(nll=sum(r['nll'] for r in rows) / len(rows), acc=sum(r['acc'] for r in rows) / len(rows),
                                     agreement_with_full=sum(r['agreement_with_full'] for r in rows) / len(rows))
                print(f'    {k} of {n - local} remote branches missing: NLL {drops[str(k)]["nll"]:.4f}, '
                      f'same top token as the full model {drops[str(k)]["agreement_with_full"] * 100:.1f}%', flush=True)
            result['missing_branches'] = drops
    (out / 'result.json').write_text(json.dumps(result, indent=1))
    with open(out / 'curve.csv', 'w') as f:
        f.write('step,tokens,train_loss,val_nll,elapsed_s\n')
        for c in curve:
            f.write(f"{c['step']},{c['tokens']},{c['train_loss']:.5f},{c['val_nll']:.5f},{c['elapsed_s']:.1f}\n")
    args.ckpt.mkdir(parents=True, exist_ok=True)
    torch.save(dict(config=asdict(cfg), model=model.state_dict()), args.ckpt / f'{name}.pt')
    print(f'  saved {out / "result.json"} ({train_s / 60:.1f} min)', flush=True)


if __name__ == '__main__':
    main()
