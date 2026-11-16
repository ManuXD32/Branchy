#!/usr/bin/env python3
"""LFM3-1.7B on the Branchy nodes with a node as coordinator: sequential, layer split and branch layouts.

  remote_lfm3.py --coordinator fedora --peers phone --cases cases.json --out docs/results/E15_x

cases.json: [{"label": ..., "plan": [...], "reference": null | label}, ...]  (plans as in payload/lfm3_model.py).
The weight blobs come from src/pretrained/lfm3_export.py (runs/pretrained/lfm3_stacks). Each stack is uploaded to the
node that runs it (the coordinator for 'local'), from this server; nothing is uploaded twice (sha256-addressed).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from nodeclient import NodeError, load_cluster

from bench_cluster import write_report  # noqa: E402

STACKS = Path(__file__).resolve().parents[2] / 'runs/pretrained/lfm3_stacks'        # written by src/pretrained/lfm3_export.py


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--cluster', default=None)
    ap.add_argument('--coordinator', required=True)
    ap.add_argument('--peers', default='')
    ap.add_argument('--cases', type=Path, required=True)
    ap.add_argument('--prompts', default='32')
    ap.add_argument('--decodes', default='32')
    ap.add_argument('--runs', type=int, default=2)
    ap.add_argument('--threads', type=int, default=6)
    ap.add_argument('--kernels', default='native')
    ap.add_argument('--linear', default='gemm')
    ap.add_argument('--timeout', type=int, default=7200)
    ap.add_argument('--ram-mb', type=float, default=0, help='RAM budget for weights on the coordinator, enforced by the process itself')
    ap.add_argument('--peer-ram-mb', type=float, default=0, help='the same budget on every peer')
    ap.add_argument('--blas-threads', type=int, default=0, help='set OPENBLAS_NUM_THREADS for the job (0 = leave the node default)')
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--note', default='')
    ap.add_argument('--upload-only', action='store_true', help='copy the weight files to the nodes and stop (no measurement)')
    args = ap.parse_args()
    import lfm3_model
    index = json.loads((STACKS / 'index.json').read_text())
    cases = json.loads(args.cases.read_text())
    peers = [p for p in args.peers.split(',') if p]
    cluster_path = Path(args.cluster) if args.cluster else Path(__file__).resolve().parent / 'cluster.json'
    spec = json.loads(cluster_path.read_text())['nodes']
    nodes = load_cluster(args.cluster, only=[args.coordinator] + peers)
    info = nodes[args.coordinator].info()
    print(f"coordinator {args.coordinator}: version {info['version']}, torch {info['torch']}", flush=True)
    wanted = {('local', 'embed.bin')}
    used = {}
    for case in cases:
        for where, first, end in lfm3_model.plan_stacks(case['plan']):
            name = f'L{first}-{end}.bin'
            if name not in index['files']:
                raise SystemExit(f'{name} was not exported; available: {sorted(index["files"])}')
            wanted.add((where, name))
            used[f'{first}-{end}'] = index['files'][name]['sha256']
    for where, name in sorted(wanted):
        node = nodes[args.coordinator if where == 'local' else where]
        sha = index['files'][name]['sha256']
        if node.call('blob_stat', sha=sha)[0]['have']:
            print(f'  {name} already on {node.name}', flush=True)
            continue
        t0, shown = time.time(), [-1]

        def progress(done, total, name=name, node=node):
            step = 4 * done // total
            if step != shown[0]:
                shown[0] = step
                print(f'  {name} to {node.name}: {100 * done // total}% of {total / 2**20:.0f} MiB ({time.time() - t0:.0f}s)', flush=True)
        node.upload_blob((STACKS / name).read_bytes(), progress)
    if args.upload_only:
        print('upload finished', flush=True)
        return
    job_args = dict(kinds=index['kinds'], embed=index['files']['embed.bin']['sha256'], stacks=used, tag='lfm', cases=cases,
                    peers={p: dict(host=spec[p]['host'], port=spec[p].get('port', 8730), delay_ms=spec[p].get('delay_ms', 0)) for p in peers},
                    prompts=[int(v) for v in args.prompts.split(',')], decodes=[int(v) for v in args.decodes.split(',')],
                    runs=args.runs, threads=args.threads, kernels=args.kernels, linear=args.linear,
                    ram_mb=args.ram_mb, peer_ram_mb=args.peer_ram_mb, mmap=bool(args.ram_mb or args.peer_ram_mb))
    node = nodes[args.coordinator]
    job = node.call('job_start', json.dumps(job_args).encode(), module='coord_lfm3', timeout_s=args.timeout,
                    env={'OPENBLAS_NUM_THREADS': str(args.blas_threads)} if args.blas_threads else {})[0]['job']
    printed = 0
    while True:
        status, out, _ = node.call('job_status', job=job)
        tail = [l for l in status.get('stderr_tail', '').splitlines() if l.startswith('  ') or l.startswith('model:')]
        for line in tail:
            if line not in main.seen:
                main.seen.add(line)
                print(line, flush=True)
                printed += 1
        if not status['running']:
            break
        time.sleep(3.0)
    if status['returncode'] != 0:
        raise NodeError(f"{args.coordinator}: coord_lfm3 failed ({status['returncode']}): {status.get('stderr_tail', '')[-1500:]}")
    report = json.loads(bytes(out))
    report['note'] = args.note
    report['coordinator']['name'] = args.coordinator
    write_report(report, args.out)


main.seen = set()
if __name__ == '__main__':
    main()
