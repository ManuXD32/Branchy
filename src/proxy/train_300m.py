#!/usr/bin/env python3
"""E5: train a ~310M BranchyN model with the recipe of the inherited 300M runs (sequential 313.4M, 2-way Y 313.8M):
TinyStories, vocabulary 8192, context 1024, 10,000 steps x 4 sequences (40.96M tokens), AdamW 3e-4 (0.9, 0.95),
weight decay 0.1, 100 warmup steps, cosine decay to 10%, gradient clip 1.0, float16 autocast, seed 1337.

  train_300m.py --branches 8 --drop 0.2 --name flat8-drop20

Resumable (checkpoint every --save-every steps). Final evaluation on the same 96 windows x 512 tokens as the
inherited quality comparison (I14: sequential 1.5716, 2-way 1.5908), in float32, plus the missing-branch curve.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from branchyn_model import BranchyN, Config, count, initialize, pick_heads, unit_bytes

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'runs/rocm_300m_train/data'
TARGET = 313_438_464          # the inherited sequential model


def build_config(args):
    if args.sequential:
        cfg = Config(kind='sequential', vocab=8192, context=args.context, width=args.seq_width, heads=args.seq_heads,
                     seq_blocks=args.seq_blocks)
        return cfg, None
    base = dict(kind='branched', vocab=8192, context=args.context, width=args.width, heads=args.heads, shared_blocks=2,
                fuse_blocks=2, n_branches=args.branches, branch_blocks=args.branch_blocks, fusion=args.fusion, groups=args.groups,
                local_blocks=1, local_width=args.width, local_branches=1, delay=args.delay, drop_p=args.drop)

    def make(w):
        return Config(**base, branch_width=w, branch_heads=args.heads if w == args.width else pick_heads(w))

    if args.branch_width:
        return make(args.branch_width), None
    lo, hi = 8, args.width // 8                  # largest multiple of 8 whose model stays within the target (binary search)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if count(BranchyN(make(mid * 8))) <= args.target:
            lo = mid
        else:
            hi = mid - 1
    return make(lo * 8), None


def atomic_save(path, state):
    tmp = path.with_suffix(path.suffix + '.tmp')
    torch.save(state, tmp)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--name', required=True)
    ap.add_argument('--sequential', action='store_true', help='the plain sequential baseline (inherited geometry: width 1152, 19 blocks)')
    ap.add_argument('--seq-width', type=int, default=1152)
    ap.add_argument('--seq-heads', type=int, default=12)
    ap.add_argument('--seq-blocks', type=int, default=19)
    ap.add_argument('--fusion', choices=['flat', 'grouped'], default='flat')
    ap.add_argument('--groups', type=int, default=4)
    ap.add_argument('--no-checkpoint', action='store_true', help='keep all activations (faster; needs more GPU memory)')
    ap.add_argument('--branches', type=int, default=8)
    ap.add_argument('--drop', type=float, default=0.2)
    ap.add_argument('--delay', type=int, default=0)
    ap.add_argument('--width', type=int, default=880)
    ap.add_argument('--heads', type=int, default=10)
    ap.add_argument('--branch-blocks', type=int, default=7)
    ap.add_argument('--branch-width', type=int, default=0, help='0 = largest width that keeps the model within --target')
    ap.add_argument('--target', type=int, default=TARGET)
    ap.add_argument('--context', type=int, default=1024)
    ap.add_argument('--steps', type=int, default=10000)
    ap.add_argument('--accum', type=int, default=4)
    ap.add_argument('--microbatch', type=int, default=1)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--warmup', type=int, default=100)
    ap.add_argument('--seed', type=int, default=1337)
    ap.add_argument('--log-every', type=int, default=20)
    ap.add_argument('--eval-every', type=int, default=500)
    ap.add_argument('--save-every', type=int, default=250)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--train-file', type=Path, default=DATA / 'train.bin', help='uint16 training tokens (default: the inherited 24M-token prefix)')
    ap.add_argument('--out', type=Path, default=ROOT / 'docs/results/E5_300m')
    ap.add_argument('--workdir', type=Path, default=ROOT / 'runs/e5_300m')
    args = ap.parse_args()
    out, work = args.out / args.name, args.workdir / args.name
    out.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    cfg, _ = build_config(args)
    model = initialize(BranchyN(cfg), args.seed)
    params = count(model)
    branched = cfg.kind == 'branched'
    if branched:
        print(f'{args.name}: {params:,} parameters; trunk/fusion width {cfg.width}, {cfg.n_branches} branches x {cfg.branch_blocks} blocks '
              f'of width {cfg.branch_width} ({cfg.branch_heads} heads), {unit_bytes(cfg) / 2**20:.1f} MiB fp32 per branch, '
              f'fusion {cfg.fusion}' + (f' in {cfg.groups} groups' if cfg.fusion == 'grouped' else '') +
              f', dropout {cfg.drop_p}, delay {cfg.delay}', flush=True)
    else:
        print(f'{args.name}: {params:,} parameters; sequential, {cfg.seq_blocks} blocks of width {cfg.width}', flush=True)
    assert json.loads((DATA / 'manifest.json').read_text())['vocab_actual'] == cfg.vocab
    train = np.memmap(args.train_file, dtype=np.uint16, mode='r')
    print(f'training text: {args.train_file} ({len(train):,} tokens); budget {args.steps * args.accum * args.microbatch * cfg.context:,} tokens', flush=True)
    valid = np.memmap(DATA / 'valid.bin', dtype=np.uint16, mode='r')
    model.to(device)
    model.checkpoint_branches = not args.no_checkpoint
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1, foreach=False)
    scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda')
    gen = torch.Generator(device='cpu').manual_seed(args.seed + 41)
    drop_gen = torch.Generator(device='cpu').manual_seed(args.seed + 97)
    latest = work / 'latest.pt'
    settings = dict(config=asdict(cfg), steps=args.steps, accum=args.accum, microbatch=args.microbatch, lr=args.lr)
    step0, previous, curve = 0, 0.0, []
    if latest.is_file():
        state = torch.load(latest, map_location='cpu', weights_only=True)
        if state['settings'] != settings:
            raise SystemExit('resume settings changed; use another --name')
        model.load_state_dict(state['model'])
        opt.load_state_dict(state['optimizer'])
        scaler.load_state_dict(state['scaler'])
        gen.set_state(state['batch_rng'])
        drop_gen.set_state(state['drop_rng'])
        step0, previous, curve = state['step'], state['elapsed_s'], state['curve']
        print(f'  resumed at step {step0} ({previous / 60:.0f} min so far)', flush=True)
        del state

    def window(source, starts, length):
        arr = np.stack([np.asarray(source[int(i):int(i) + length + 1], dtype=np.int64) for i in starts])
        return torch.from_numpy(arr[:, :-1].copy()).to(device), torch.from_numpy(arr[:, 1:].copy()).to(device)

    def keep_mask(batch):
        if not branched or cfg.drop_p <= 0:
            return None
        keep = torch.rand(batch, cfg.n_branches, generator=drop_gen) >= cfg.drop_p
        keep[:, :cfg.local_branches] = True
        return keep.to(device)

    @torch.no_grad()
    def quick_eval():          # the inherited trainers' validation windows (8 x context 1024)
        model.eval()
        total = 0.0
        for j in range(8):
            x, y = window(valid, [(j * 15485863) % (len(valid) - cfg.context - 1)], cfg.context)
            with torch.autocast('cuda', dtype=torch.float16, enabled=device.type == 'cuda'):
                logits = model(x)
            total += float(F.cross_entropy(logits.reshape(-1, cfg.vocab).float(), y.reshape(-1))) / 8
        model.train()
        return total

    model.train()
    t0 = time.perf_counter()
    val = float('nan')
    for step in range(step0 + 1, args.steps + 1):
        lr = args.lr * step / args.warmup if step <= args.warmup else \
            args.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * (step - 1 - args.warmup) / max(1, args.steps - args.warmup))))
        for group in opt.param_groups:
            group['lr'] = lr
        opt.zero_grad(set_to_none=True)
        total = 0.0
        for _ in range(args.accum):
            starts = torch.randint(0, len(train) - cfg.context - 1, (args.microbatch,), generator=gen).tolist()
            x, y = window(train, starts, cfg.context)
            with torch.autocast('cuda', dtype=torch.float16, enabled=device.type == 'cuda'):
                logits = model(x, keep=keep_mask(x.shape[0]))
                loss = F.cross_entropy(logits.reshape(-1, cfg.vocab).float(), y.reshape(-1))
            scaler.scale(loss / args.accum).backward()
            total += float(loss.detach()) / args.accum
            del x, y, logits, loss
        if not math.isfinite(total):
            raise SystemExit(f'loss is not finite at step {step}')
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        elapsed = previous + time.perf_counter() - t0
        if step == 1 or step % args.eval_every == 0 or step == args.steps:
            val = quick_eval()
            curve.append(dict(step=step, train_loss=total, val_loss=val, elapsed_s=elapsed))
        if step == 1 or step % args.log_every == 0 or step == args.steps:
            done = step - step0
            eta = (args.steps - step) * (time.perf_counter() - t0) / done / 60
            vram = torch.cuda.max_memory_allocated() / 2**20 if device.type == 'cuda' else 0
            print(f'  step {step}/{args.steps} loss {total:.4f} val {val:.4f} | {step * args.microbatch * args.accum * cfg.context:,} tokens | '
                  f'VRAM {vram:.0f} MiB | {elapsed / 60:.0f} min, about {eta:.0f} min left', flush=True)
        if step % args.save_every == 0 or step == args.steps:
            atomic_save(latest, dict(settings=settings, model=model.state_dict(), optimizer=opt.state_dict(), scaler=scaler.state_dict(),
                                     batch_rng=gen.get_state(), drop_rng=drop_gen.get_state(), step=step, elapsed_s=elapsed, curve=curve))
    train_s = previous + time.perf_counter() - t0
    # ------------------------------------------------------------------ final evaluation (float32, 96 x 512 windows of I14)
    model.eval()
    torch.save(dict(config=asdict(cfg), model={k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}),
               work / 'inference.pt')
    starts = np.random.default_rng(314159).choice(len(valid) - 512 - 1, size=96, replace=False).astype(np.int64)

    @torch.no_grad()
    def held_out(gone=()):
        losses, acc, preds = [], [], []
        for st in starts:
            x, y = window(valid, [st], 512)
            keep = None
            if gone and branched:
                keep = torch.ones(1, cfg.n_branches, dtype=torch.bool, device=device)
                keep[:, list(gone)] = False
            logits = model(x, keep=keep).float()
            losses.append(float(F.cross_entropy(logits.transpose(1, 2), y)))
            arg = logits.argmax(-1)
            acc.append(float((arg == y).float().mean()))
            preds.append(arg)
        return dict(nll=float(np.mean(losses)), ppl=float(math.exp(np.mean(losses))), acc=float(np.mean(acc)), per_window=losses), preds

    full, full_preds = held_out()
    print(f'  FINAL {args.name}: held-out NLL {full["nll"]:.5f} (PPL {full["ppl"]:.3f}, accuracy {full["acc"] * 100:.2f}%) on 96 x 512 tokens; '
          f'inherited sequential 1.57161, 2-way 1.59078', flush=True)
    missing = {}
    n, local = cfg.n_branches, cfg.local_branches
    pick = torch.Generator().manual_seed(99)
    for k in sorted({1, 2, (n - local) // 2, n - local}) if branched else []:
        rows = []
        for _ in range(1 if k == n - local else 3):
            gone = (torch.randperm(n - local, generator=pick)[:k] + local).tolist()
            res, preds = held_out(gone)
            res['agreement_with_full'] = float(np.mean([float((a == b).float().mean()) for a, b in zip(preds, full_preds)]))
            rows.append(res)
        missing[str(k)] = dict(nll=float(np.mean([r['nll'] for r in rows])), acc=float(np.mean([r['acc'] for r in rows])),
                               agreement_with_full=float(np.mean([r['agreement_with_full'] for r in rows])))
        print(f'    {k} of {n - local} remote branches missing: NLL {missing[str(k)]["nll"]:.4f}, same top token as the full model '
              f'{missing[str(k)]["agreement_with_full"] * 100:.1f}%', flush=True)
    result = dict(name=args.name, config=asdict(cfg), parameters=params, unit_mib=unit_bytes(cfg) / 2**20 if branched else None, steps=args.steps,
                  microbatch=args.microbatch, accum=args.accum, machine=os.uname().nodename + ' ' + (torch.cuda.get_device_name(0) if device.type == 'cuda' else 'cpu'),
                  train_tokens=args.steps * args.microbatch * args.accum * cfg.context, train_minutes=train_s / 60, seed=args.seed,
                  peak_vram_mib=torch.cuda.max_memory_allocated() / 2**20 if device.type == 'cuda' else 0,
                  final=full, missing_branches=missing, curve=curve,
                  inherited=dict(sequential_nll=1.57161, two_way_nll=1.59078, source='EXPERIMENT_LOG I14'))
    (out / 'result.json').write_text(json.dumps(result, indent=1))
    print(f'  saved {out / "result.json"} ({train_s / 60:.0f} min training)', flush=True)


if __name__ == '__main__':
    main()
