#!/usr/bin/env python3
"""E0b: does turning on OpenBLAS threads help (and stay numerically correct) on a node?

For each (omp threads, openblas threads, attention kernel): restart the node with those settings,
run the self-test three times (the Termux corruption is random), and if clean, profile one branch.
  e0b_blas_threads.py --nodes phone,phone2 --out ../../docs/results/E0b_blas_threads
"""
import argparse
import json
import time
from pathlib import Path

import nodectl
from nodeclient import NodeError, load_cluster

COMBOS = [(4, 1), (1, 1), (1, 4), (4, 4), (1, 6), (6, 6)]


def run_node(node, args):
    info = node.info()
    original = dict(threads=info['threads'], kernels=info['kernels'], linear=info.get('linear', 'gemm'),
                    env=info['state'].get('env') or {})
    cpus = info['cpu_count'] or 8
    rows = []
    for kernels in args.kernels.split(','):
        for omp, blas in COMBOS:
            if max(omp, blas) > cpus:
                continue
            row = dict(kernels=kernels, omp=omp, openblas=blas)
            try:
                node.restart(threads=omp, kernels=kernels, linear='gemm', env={'OPENBLAS_NUM_THREADS': str(blas)})
                tests = [t for _ in range(3) for t in nodectl.selftest_node(node)]
                row['numerics_ok'] = all(t['ok'] for t in tests)
                row['worst_vs_reference'] = max(t['max_abs_vs_reference'] for t in tests)
                row['worst_repeat'] = max(t['repeat_max_abs'] for t in tests)
                if row['numerics_ok']:
                    prof = node.call('profile', timeout=1800, width=880, heads=10, blocks=7, prefills=[32, 256], decode_steps=24)[0]
                    a, b = prof['rows']
                    row.update(decode_ms=a['decode_ms_median'], decode_ms_ctx256=b['decode_ms_median'],
                               prefill32_tps=a['prefill_tps'], prefill256_tps=b['prefill_tps'])
                row['thermal_max_c'] = node.info().get('thermal_max_c')
            except NodeError as exc:
                row['error'] = str(exc)
            rows.append(row)
            print(f"  {node.name:7s} {kernels:9s} omp={omp} blas={blas}: numerics={'ok' if row.get('numerics_ok') else 'BAD'} "
                  f"(vs ref {row.get('worst_vs_reference', float('nan')):.1e}, repeat {row.get('worst_repeat', float('nan')):.1e}) "
                  f"decode={row.get('decode_ms')} ms prefill256={row.get('prefill256_tps')} tok/s {row.get('error', '')}", flush=True)
    node.restart(**original)
    return dict(info={k: info.get(k) for k in ('android', 'cpu_model', 'torch', 'cpu_max_mhz')}, restored=original, rows=rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--cluster'); ap.add_argument('--nodes', required=True); ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--kernels', default='math-attn,native')
    args = ap.parse_args()
    nodes = load_cluster(args.cluster, only=args.nodes.split(','))
    results = nodectl.each(nodes, lambda n: run_node(n, args))
    report = dict(date=time.strftime('%Y-%m-%d %H:%M:%S'),
                  nodes={k: (dict(error=str(v)) if isinstance(v, Exception) else v) for k, v in results.items()})
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / 'blas_threads.json').write_text(json.dumps(report, indent=1))
    md = ['# E0b: OpenBLAS threads on the nodes', '', f"Date {report['date']}. One 7-block width-880 branch; self-test run 3 times per setting.", '',
          '| Node | Attention | OMP threads | OpenBLAS threads | Numerics | worst vs reference | worst repeat diff | Decode ms | Decode ms @256 | Prefill 32 tok/s | Prefill 256 tok/s | Max temp C |',
          '|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|']
    for name, rep in report['nodes'].items():
        for r in rep.get('rows', []):
            f = lambda k, d=1: '' if r.get(k) is None else f'{r[k]:.{d}f}'
            md.append(f"| {name} | {r['kernels']} | {r['omp']} | {r['openblas']} | {'ok' if r.get('numerics_ok') else 'FAIL'} | "
                      f"{r.get('worst_vs_reference', float('nan')):.1e} | {r.get('worst_repeat', float('nan')):.1e} | {f('decode_ms', 2)} | "
                      f"{f('decode_ms_ctx256', 2)} | {f('prefill32_tps')} | {f('prefill256_tps')} | {r.get('thermal_max_c') or ''} |")
    (args.out / 'blas_threads.md').write_text('\n'.join(md) + '\n')
    print('saved', args.out / 'blas_threads.md')


if __name__ == '__main__':
    main()
