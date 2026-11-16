#!/usr/bin/env python3
"""Placement benchmark: one Branchy model, branches spread over this machine and remote nodes.

  A placement may leave branches out with "none:6,7" (models trained with branch dropout): no parity gate for it,
  the agreement of its top token with the reference is reported instead.

  bench_cluster.py --placement "all-local=local:0,1,2,3" \
                   --placement "4dev=local:0;fedora:1;phone:2;tablet:3" --out docs/results/E1_x

Model: --synthetic N (untrained N-way, default geometry = Branchy4 309M; nodes rebuild
the same weights themselves, nothing is uploaded) or --checkpoint FILE (trained 2-way Y /
Branchy4; branch weights are uploaded once and cached on the node by sha256).

The first placement is the reference: it decodes greedily, every other placement is
teacher-forced with the same tokens and must reproduce its logits within --tolerance.
(Exact token equality is not used: untrained logits are nearly flat, so harmless 1e-6
cross-device rounding could flip an argmax.)
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import statistics
import time
import uuid
from pathlib import Path

import torch

import nodeclient
from nodeclient import load_cluster

import stacks  # noqa: E402  (payload dir is put on sys.path by nodeclient)
import cluster_model  # noqa: E402
from cluster_model import LocalExecutor, Model, perf, run_trial, summarize  # noqa: F401
from coordinator import DistributedExecutor, aggregate, deploy, parse_placement, run_placements  # noqa: E402,F401


def write_report(report, out):
    """bench.json and bench.md for a placement benchmark report."""
    out = Path(out)
    results = report['results']
    out.mkdir(parents=True, exist_ok=True)
    (out / 'bench.json').write_text(json.dumps(report, indent=1))
    md = [f'# Placement benchmark: {report['model']}', '', f"Date {report['date']}. {report['note']}", '',
          f"Coordinator: {report['coordinator'].get('name', 'this server')}, {report['coordinator']['threads']} threads, linear={report['coordinator']['linear']}, wire={report['wire']}, {report['runs']} runs per cell, "
          f"teacher-forced parity tolerance {report['tolerance']}.", '',
          '| Placement | Devices | Prompt | Decode | Prefill tok/s | Decode tok/s | Step p50 ms | Step p95 ms | Shared | Local | Wait | Fuse | Parity (max abs) |',
          '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|']
    for r in results:
        md.append(f"| {r['placement']} | {r['devices']} | {r['prompt']} | {r['decode']} | {r['prefill_tps']:.1f} | {r['decode_tps']:.2f} | "
                  f"{r['step_total_ms']['p50']:.1f} | {r['step_total_ms']['p95']:.1f} | {r['step_shared_ms']['p50']:.1f} | "
                  f"{r['step_local_ms']['p50']:.1f} | {r['step_wait_ms']['p50']:.1f} | {r['step_fuse_ms']['p50']:.1f} | "
                  f"{r['parity']} ({r['max_abs_logit_diff']:.1e}) |")
    md += ['', '## Remote legs per decode step (p50 / p95 ms)', '',
           '| Placement | Prompt | Decode | Node | Round trip | Node compute | Network + framing | Bytes/step |', '|---|---:|---:|---|---:|---:|---:|---:|']
    for r in results:
        for n, x in r['remote'].items():
            md.append(f"| {r['placement']} | {r['prompt']} | {r['decode']} | {n} | {x['rpc_ms']['p50']:.1f} / {x['rpc_ms']['p95']:.1f} | "
                      f"{x['compute_ms']['p50']:.1f} / {x['compute_ms']['p95']:.1f} | {x['net_ms']['p50']:.1f} / {x['net_ms']['p95']:.1f} | {x['bytes_per_step']} |")
    sequential = report.get('sequential')
    if sequential:
        md += ['', f"## Sequential baseline on the coordinator ({sequential['label']}, {sequential['parameters']:,} parameters)", '',
               '| Prompt | Decode | Prefill tok/s | Decode tok/s | Step p50 ms | Step p95 ms |', '|---:|---:|---:|---:|---:|---:|']
        for r in sequential['rows']:
            md.append(f"| {r['prompt']} | {r['decode']} | {r['prefill_tps']:.1f} | {r['decode_tps']:.2f} | {r['step_ms']['p50']:.1f} | {r['step_ms']['p95']:.1f} |")
    (out / 'bench.md').write_text('\n'.join(md) + '\n')
    print(f"saved {out / 'bench.json'} and bench.md")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--cluster', default=None)
    ap.add_argument('--placement', action='append', required=True, help='label=local:0,1;nodeA:2;nodeB:3 (first one is the reference)')
    ap.add_argument('--synthetic', type=int, default=0, help='number of branches of an untrained synthetic model')
    ap.add_argument('--geometry', default='880,10,2,7,2', help='width,heads,shared,branch,fuse for --synthetic')
    ap.add_argument('--checkpoint', type=Path)
    ap.add_argument('--sequential', default='', help='also time a sequential baseline here: "synthetic" or a checkpoint path')
    ap.add_argument('--prompts', default='32,256,768')
    ap.add_argument('--decodes', default='16,64')
    ap.add_argument('--runs', type=int, default=5)
    ap.add_argument('--master-threads', type=int, default=4)
    ap.add_argument('--linear', choices=['gemm', 'mv'], default='gemm')
    ap.add_argument('--wire', choices=['f4', 'f2'], default='f4')
    ap.add_argument('--tolerance', type=float, default=2e-3)
    ap.add_argument('--delay-ms', type=float, default=None, help='override injected delay (simulation only)')
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--note', default='')
    args = ap.parse_args()
    torch.set_num_threads(args.master_threads)
    torch.set_grad_enabled(False)
    stacks.set_linear_mode(args.linear)
    prompts = [int(v) for v in args.prompts.split(',')]
    decodes = [int(v) for v in args.decodes.split(',')]
    placements = [parse_placement(p) for p in args.placement]
    used = sorted({name for _, a in placements for name in a if name not in ('local', 'none')})
    nodes = load_cluster(args.cluster, only=used, delay_ms=args.delay_ms) if used else {}
    missing = [n for n in used if n not in nodes]
    if missing:
        ap.error(f'nodes not in cluster file: {missing}')
    if args.checkpoint:
        model = Model.from_checkpoint(args.checkpoint)
        tag = 'ck' + hashlib.sha256(str(args.checkpoint.resolve()).encode()).hexdigest()[:6] + '-'
    else:
        w, h, s, b, f = (int(v) for v in args.geometry.split(','))
        model = Model.synthetic(args.synthetic or 4, w, h, s, b, f)
        tag = f'syn{args.synthetic or 4}w{w}b{b}-'
    print(f'model: {model.label}, {model.parameters():,} parameters; coordinator threads={args.master_threads} linear={args.linear}', flush=True)
    info_before = {name: node.info() for name, node in nodes.items()}
    for name, info in info_before.items():
        print(f"  node {name}: threads={info['threads']} kernels={info['kernels']} linear={info.get('linear')} "
              f"temp={info.get('thermal_max_c')} load={info['loadavg']}")
    for note in deploy(model, placements, nodes, tag):
        print('  ' + note, flush=True)
    results = run_placements(model, placements, nodes, prompts, decodes, args.runs, args.wire, args.tolerance, tag)
    sequential = None
    if args.sequential:
        seq = Model.synthetic_sequential() if args.sequential == 'synthetic' else Model.from_checkpoint(args.sequential)
        print(f'sequential baseline on the coordinator: {seq.label}, {seq.parameters():,} parameters', flush=True)
        sequential = dict(label=seq.label, parameters=seq.parameters(), rows=cluster_model.time_local(seq, prompts, decodes, args.runs))
        for r in sequential['rows']:
            print(f"  sequential     prompt {r['prompt']:>4} decode {r['decode']:>3}: prefill {r['prefill_tps']:7.1f} tok/s | "
                  f"decode {r['decode_tps']:6.2f} tok/s (step p50 {r['step_ms']['p50']:.1f} p95 {r['step_ms']['p95']:.1f} ms)")
    info_after = {name: node.info() for name, node in nodes.items()}
    keep = ('name', 'version', 'torch', 'threads', 'kernels', 'linear', 'cpu_model', 'android', 'thermal_max_c', 'loadavg',
            'battery_pct', 'affinity', 'env')
    report = dict(date=time.strftime('%Y-%m-%d %H:%M:%S'), note=args.note, model=model.label, parameters=model.parameters(),
                  coordinator=dict(threads=args.master_threads, linear=args.linear, torch=torch.__version__,
                                   affinity=sorted(os.sched_getaffinity(0)), loadavg=os.getloadavg()),
                  wire=args.wire, runs=args.runs, tolerance=args.tolerance,
                  nodes_before={n: {k: i.get(k) for k in keep} for n, i in info_before.items()},
                  nodes_after={n: {k: i.get(k) for k in keep} for n, i in info_after.items()},
                  results=results, sequential=sequential)
    write_report(report, args.out)


if __name__ == '__main__':
    main()
