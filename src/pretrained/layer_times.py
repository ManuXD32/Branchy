#!/usr/bin/env python3
"""How much speed could the branch layouts of a reshaped pretrained model (E6: LFM3-1.7B) give? An estimate, not a
device measurement: the nodes cannot run LFM3 layers.

Times every layer of the untouched model on this machine's CPU while it writes tokens one at a time (cached decode),
then adds the layer times up along the token path of each layout tested in E6:

  sequential           all 24 layers, one after the other
  branched, ideal      layers before the region + the slowest branch + layers after it (branches on separate, equally
                       fast devices, no network)
  branched, with a round trip of R ms added once per token (the coordinator waits for the branches)

  layer_times.py --out docs/results/E6_lfm3_parallel/speed_estimate
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

from lfm3_parallel import MODEL_DIR

LAYOUTS = [  # (first layer, one past the last, branches, layers per branch, loss increase after healing: TinyStories / WikiText-2)
    (8, 16, 2, 4, '+0.13 / +0.09'), (8, 16, 4, 2, '+0.31 / +0.19'), (8, 16, 8, 1, '+0.43 / +0.29'),
    (6, 18, 2, 6, '+0.28 / +0.16'), (4, 20, 4, 4, '+0.86 / +0.61'), (4, 20, 8, 2, '+1.10 / +0.82'), (4, 20, 16, 1, '+1.29 / +0.96')]


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model', default=str(MODEL_DIR))
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--threads', type=int, default=8)
    ap.add_argument('--tokens', type=int, default=96)
    ap.add_argument('--round-trips', default='20,45')
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, trust_remote_code=True, dtype=torch.float32).eval()
    layers = model.model.layers
    times = [[] for _ in layers]
    started = {}

    def before(i):
        return lambda module, inputs: started.__setitem__(i, time.perf_counter())

    def after(i):
        return lambda module, inputs, output: times[i].append(time.perf_counter() - started[i])
    for i, layer in enumerate(layers):
        layer.register_forward_pre_hook(before(i))
        layer.register_forward_hook(after(i))
    ids = tok('Once upon a time, there was a little dragon who lived in a cave near a quiet village.', return_tensors='pt').input_ids
    t0 = time.perf_counter()
    out = model.generate(ids, max_new_tokens=args.tokens, do_sample=False, use_cache=True)
    wall = time.perf_counter() - t0
    new = out.shape[1] - ids.shape[1]
    # first call = the whole prompt; the following ones are single-token steps. Skip the prompt and two warm-up steps.
    per_layer = [statistics.median(t[3:]) * 1000 for t in times]
    kinds = list(getattr(model.config, 'layer_types', ['?'] * len(layers)))
    total = sum(per_layer)
    step_ms = wall * 1000 / new
    outside = step_ms - total            # embedding, output layer over 128,000 words, sampling: paid once per token in every layout
    rows = []
    for a, b, n, each, loss in LAYOUTS:
        branches = [sum(per_layer[a + k * each:a + (k + 1) * each]) for k in range(n)]
        path = sum(per_layer[:a]) + max(branches) + sum(per_layer[b:])
        row = dict(layers=f'{a}-{b - 1}', layout=f'{n} x {each}', path_layers=a + each + len(layers) - b, loss_after_healing=loss,
                   sequential_ms=step_ms, path_ms=path + outside, ideal_speedup=step_ms / (path + outside))
        for r in [float(v) for v in args.round_trips.split(',')]:
            row[f'speedup_with_{r:g}ms'] = step_ms / (path + outside + r)
        rows.append(row)
    report = dict(model=args.model, threads=args.threads, new_tokens=new, wall_tokens_per_second=new / wall, step_ms=step_ms,
                  layers_ms=per_layer, layer_kinds=kinds, layers_total_ms=total, outside_layers_ms=step_ms - total, layouts=rows,
                  text=tok.decode(out[0, ids.shape[1]:])[:300])
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / 'layer_times.json').write_text(json.dumps(report, indent=1))
    trips = [float(v) for v in args.round_trips.split(',')]
    md = ['# LFM3-1.7B: speed estimate for the branch layouts of E6', '',
          f'Layer times measured on this server ({args.threads} threads, float32, cached single-token decode, median of {new - 3} steps): '
          f'the 24 layers take {total:.1f} ms of a {step_ms:.1f} ms token ({new / wall:.2f} tokens per second). '
          'Estimates below add the layer times along the token path, plus the time outside the layers (output layer over 128,000 words, '
          'embedding, sampling), which every layout pays in full; nothing here ran on the devices.', '',
          '| Layers | Layout | Layers on the token path | Loss increase after healing (TinyStories / WikiText-2) | Sequential token, ms | Branched token, ms | Speed-up, no network | '
          + ' | '.join(f'Speed-up with a {r:g} ms round trip' for r in trips) + ' |', '|---|---|---:|---|---:|---:|---:|' + '---:|' * len(trips)]
    for r in rows:
        md.append(f"| {r['layers']} | {r['layout']} | {r['path_layers']} | {r['loss_after_healing']} | {r['sequential_ms']:.1f} | {r['path_ms']:.1f} | "
                  f"{r['ideal_speedup']:.2f}x | " + ' | '.join(f"{r[f'speedup_with_{t:g}ms']:.2f}x" for t in trips) + ' |')
    md += ['', '## Time per layer (ms)', '', '| Layer | Kind | ms |', '|---:|---|---:|'] + \
          [f'| {i} | {kinds[i] if i < len(kinds) else "?"} | {t:.2f} |' for i, t in enumerate(per_layer)]
    (args.out / 'layer_times.md').write_text('\n'.join(md) + '\n')
    print('\n'.join(md[:14 + len(rows)]))


if __name__ == '__main__':
    main()
