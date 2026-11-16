#!/usr/bin/env python3
"""Tables of the low-memory experiment (E16) from its summary files -> docs/results/E16_low_memory/tables.md."""
from __future__ import annotations

import json
from pathlib import Path

OUT = Path(__file__).resolve().parents[2] / 'docs/results/E16_low_memory'


def rows_of(summary, model):
    budgets = sorted({r['budget_mb'] for r in summary['rows'] if r.get('model') == model}, key=lambda b: (b != 0, -b))
    for b in budgets:
        yield b, {r['placement']: r for r in summary['rows'] if r.get('model') == model and r.get('budget_mb') == b and not r.get('failed')}


def table(summary, model, alone, splits, title):
    md = [f'### {title}', '',
          '| Memory for weights per device, MB | Weight pages held by device A / B at the end, MB | One device, tokens/s (median / slowest 5% of token times, ms) | Read from storage per token, MB | '
          + ' | '.join(f'Two devices, {d} ms network, tokens/s' for d in ('0', '20', '45')) + ' | Read per token in the split (A / B), MB |', '|---:|---:|---:|---:|---:|---:|---:|---:|']
    for b, r in rows_of(summary, model):
        if alone not in r:
            md.append(f"| {b + 100 if b else 'no limit'} | failed | | | | | | |")
            continue
        a = r[alone]
        first = r.get(splits[0])
        held_b = next((v['file'] for k, v in (first or {}).get('memory_mb', {}).items() if k != 'local'), None)
        read_split = '' if not first else f"{first['read_mb_per_token'].get('local', 0):,.0f} / {sum(v for k, v in first['read_mb_per_token'].items() if k != 'local'):,.0f}"
        md.append(f"| {f'{b + 100:,}' if b else 'no limit'} | {a['memory_mb']['local']['file']:,.0f} / {held_b:,.0f} | {a['decode_tps']:.2f} ({a['step_p50_ms']:.0f} / {a['step_p95_ms']:.0f}) | "
                  f"{sum(a['read_mb_per_token'].values()):,.0f} | " + ' | '.join(f"{r[p]['decode_tps']:.2f}" if p in r else '' for p in splits) + f' | {read_split} |')
    return md + ['']


def main():
    md = ['# E16: tables', '', 'Written by `src/lowmem/report.py` from the summary files. "Memory for weights" is the budget of the run plus the',
          '96 MB margin (rounded): the page cache the limit leaves to the device besides what its programs need.', '']
    for name, label in (('pass2/summary.json', 'Second pass (3 runs of 32 tokens per cell)'), ('summary.json', 'First pass (2 runs of 24 tokens; the story judging used llama-swap during it)')):
        path = OUT / name
        if not path.exists():
            continue
        s = json.loads(path.read_text())
        md += [f'## 300M models. {label}', '', f"Memory the programs need besides weights (MB): {s['overhead_mb']}", '']
        md += table(s, 'sequential', 'alone', ['split-0ms', 'split-20ms', 'split-45ms'], 'Sequential 313M model (1,195 MB of weights); split = layer split, blocks 0-9 on A and 10-18 on B')
        md += table(s, 'branched', 'alone', ['split-0ms', 'split-20ms', 'split-45ms'], '8-branch 309M model (1,180 MB); split = branches 0-3 on A and 4-7 on B, delayed execution')
    path = OUT / 'lfm3' / 'summary.json'
    if path.exists():
        s = json.loads(path.read_text())
        md += ['## LFM3-1.7B (6,595 MB of weights; 2 runs of 16 tokens per cell)', '', f"Memory the programs need besides weights (MB): {s['overhead_mb']}", '']
        md += table(s, 'lfm3', 'alone', ['layer-split-0ms', 'layer-split-20ms', 'layer-split-45ms'], 'Original layout; split = layers 0-11 on A and 12-23 on B')
        md += table(s, 'lfm3', '4x4-alone', ['4x4-split-0ms', '4x4-split-20ms', '4x4-split-45ms'], 'Layers 4-19 as 4 branches of 4 layers (E6 conversion); split = one branch on A and three on B, the coordinator waits for them')
    (OUT / 'tables.md').write_text('\n'.join(md) + '\n')
    print('\n'.join(md))


if __name__ == '__main__':
    main()
