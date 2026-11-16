#!/usr/bin/env python3
"""Q2: what does rounding the trained weights to 8, 6 or 4 bits cost, without any retraining?

  quant_check.py --models seq=.../seq/inference.pt flat8=.../inference.pt --out docs/results/Q2_quantization

The weights are rounded and written back as floats (simulated quantization), then the model is measured with the
held-out loss of Q1 on the same windows as its unrounded self (paired difference). Rounding follows the block layout of
the GGUF formats (blocks of 32 weights with one scale each), but it is not llama.cpp's exact code:

  q8    8 bits, symmetric, one fp16 scale per 32 weights            8.5 bits per weight (like Q8_0)
  q6    6 bits, symmetric                                           6.5
  q4    4 bits, symmetric                                           4.5 (like Q4_0)
  q4a   4 bits between the block's minimum and maximum (two fp16)   5.0 (like Q4_1)

Rounded: every weight matrix (token and position tables included); biases and normalisation weights stay float.
"branches" rounds only the branch stacks, the part that would sit on the devices.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from quality_bench import DATA, interval, load, part_loss

SCHEMES = {'q8': (8, True, 8.5), 'q6': (6, True, 6.5), 'q4': (4, True, 4.5), 'q4a': (4, False, 5.0)}


def round_blocks(w, bits, symmetric, block=32):
    flat = w.reshape(-1)
    pad = (-flat.numel()) % block
    f = F.pad(flat, (0, pad)).view(-1, block)
    if symmetric:
        top = 2 ** (bits - 1) - 1
        scale = (f.abs().amax(1, keepdim=True) / top).half().float().clamp_min(1e-12)
        out = (f / scale).round().clamp(-top - 1, top) * scale
    else:
        lo = f.amin(1, keepdim=True).half().float()
        scale = ((f.amax(1, keepdim=True) - lo) / (2 ** bits - 1)).half().float().clamp_min(1e-12)
        out = ((f - lo) / scale).round().clamp(0, 2 ** bits - 1) * scale + lo
    return out.view(-1)[:flat.numel()].view_as(w)


@torch.no_grad()
def apply(model, original, scheme, scope):
    """Restore the float weights, then round the matrices in scope. Returns (rounded parameters, relative error)."""
    bits, symmetric, _ = SCHEMES[scheme]
    model.load_state_dict(original)
    count, err, norm = 0, 0.0, 0.0
    for name, p in model.named_parameters():
        if p.dim() < 2 or (scope == 'branches' and not name.startswith('branches.')):
            continue
        q = round_blocks(p.data, bits, symmetric)
        err += float((q - p.data).pow(2).sum())
        norm += float(p.data.pow(2).sum())
        p.data.copy_(q)
        count += p.numel()
    return count, (err / norm) ** 0.5


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--models', nargs='+', required=True, help='name=path/to/inference.pt')
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--windows', type=int, default=300)
    ap.add_argument('--seed', type=int, default=20261113)
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()
    device = torch.device(args.device)
    valid = np.fromfile(DATA / 'valid.bin', dtype=np.uint16)
    train = np.fromfile(DATA / 'train.bin', dtype=np.uint16)
    counts = np.bincount(train, minlength=8192)
    rare = torch.from_numpy(counts < np.sort(counts)[-1000]).to(device)
    starts = np.sort(np.random.default_rng(args.seed).choice(len(valid) - 513, size=args.windows, replace=False))
    args.out.mkdir(parents=True, exist_ok=True)
    report = dict(windows=args.windows, tokens=args.windows * 512, seed=args.seed, schemes={k: v[2] for k, v in SCHEMES.items()}, models={})
    rows = []
    for spec in args.models:
        name, path = spec.split('=', 1)
        model = load(path, device)
        original = {k: v.clone() for k, v in model.state_dict().items()}
        total = sum(p.numel() for p in model.parameters())
        base = part_loss(model, valid, starts, rare, device)
        reuse = lambda r: float(r['reuse_hit'].sum() / r['reuse_n'].sum())          # noqa: E731
        entry = dict(parameters=total, float=dict(loss=float(base['loss'].mean()), accuracy=float(base['acc'].mean()), rare_reuse=reuse(base)), rounded=[])
        print(f'{name}: float loss {base["loss"].mean():.4f}', flush=True)
        rows.append((name, 'float (as trained)', '32', entry['float']['loss'], '', entry['float']['accuracy'], entry['float']['rare_reuse'], ''))
        scopes = ['all'] + (['branches'] if any(k.startswith('branches.') for k in original) else [])
        for scope in scopes:
            for scheme in SCHEMES:
                if scope == 'branches' and scheme in ('q8', 'q6'):
                    continue
                t0 = time.time()
                count, err = apply(model, original, scheme, scope)
                r = part_loss(model, valid, starts, rare, device)
                d, ci = interval(r['loss'] - base['loss'])
                item = dict(scheme=scheme, scope=scope, bits_per_weight=SCHEMES[scheme][2], rounded_parameters=count, share=count / total,
                            weight_error=err, loss=float(r['loss'].mean()), vs_float=d, ci95=ci, accuracy=float(r['acc'].mean()), rare_reuse=reuse(r))
                entry['rounded'].append(item)
                print(f'  {scheme} {scope}: loss {item["loss"]:.4f} ({d:+.4f} ±{ci:.4f}), weight error {err * 100:.2f}%, {time.time() - t0:.0f}s', flush=True)
                rows.append((name, f'{scheme}, {"all weights" if scope == "all" else "branches only"} ({count / total * 100:.0f}% of the model)',
                             f'{SCHEMES[scheme][2]}', item['loss'], f'{d:+.4f} ±{ci:.4f}', item['accuracy'], item['rare_reuse'], f'{err * 100:.2f}%'))
        report['models'][name] = entry
        del model
        torch.cuda.empty_cache()
    (args.out / 'quant.json').write_text(json.dumps(report, indent=1))
    md = ['# Q2: rounding the trained weights without retraining', '',
          f'{args.windows} windows x 512 held-out tokens; differences are paired against the same model in float, 95% interval over windows.', '',
          '| Model | Weights | Bits per weight | Held-out loss | vs float | Top-token accuracy | Rare-word reuse | Weight error |', '|---|---|---:|---:|---:|---:|---:|---:|']
    md += [f'| {n} | {s} | {b} | {l:.4f} | {d} | {a * 100:.2f}% | {rr * 100:.1f}% | {e} |' for n, s, b, l, d, a, rr, e in rows]
    (args.out / 'quant.md').write_text('\n'.join(md) + '\n')
    print(f'saved {args.out / "quant.md"}', flush=True)


if __name__ == '__main__':
    main()
