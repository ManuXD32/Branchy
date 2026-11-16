#!/usr/bin/env python3
"""Collect docs/results/E6_lfm3_parallel/heal/*/result.json into heal/summary.md (one row per healing run)."""
import json
from pathlib import Path

import sys

ROOT_DIR = Path(__file__).resolve().parents[2]      # the repository
ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else str(ROOT_DIR / 'docs/results/E6_lfm3_parallel/heal'))
rows = [json.loads(p.read_text()) for p in sorted(ROOT.glob('*/result.json'))]
rows.sort(key=lambda r: (r['region'], r['kind'] == 'skip', -(r['layers_per_branch'] or 0), r['name']))
names = ['tinystories', 'wikitext2']
md = [f'# Healing runs ({ROOT.parent.name})', '',
      'Loss increase over the untouched model (nats per token) before -> after healing, and the share of tokens where the',
      'healed model picks the same top token as the original. 24,528 scored tokens per dataset.', '',
      '| Run | Layers | Layout | Mode | Steps | Rank | lr | Hidden loss | TinyStories | WikiText-2 | Same top token after | Minutes |',
      '|---|---|---|---|---:|---:|---:|---:|---|---|---|---:|']
for r in rows:
    start, end = r['region']
    layout = 'deleted' if r['kind'] == 'skip' else f"{r['branches']} x {r['layers_per_branch']}"
    cells = []
    for n in names:
        o = r['original'][n]['nll']
        cells.append(f"+{r['before'][n]['nll'] - o:.3f} -> +{r['after'][n]['nll'] - o:.3f}")
    agree = ' / '.join(f"{r['after'][n]['top1_agreement'] * 100:.0f}%" for n in names)
    mode = f"full, lr {r['full_lr']:g}" if r.get('full') else 'adapters'
    md.append(f"| {r['name']} | {start}-{end - 1} | {layout} | {mode} | {r['steps']} | {r['rank']} | {r['lr']:g} | {r.get('hidden_weight', 0):g} | "
              f"{cells[0]} | {cells[1]} | {agree} | {r['train_minutes']:.0f} |")
(ROOT / 'summary.md').write_text('\n'.join(md) + '\n')
print('\n'.join(md))
