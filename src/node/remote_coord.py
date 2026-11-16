#!/usr/bin/env python3
"""Placement benchmark with a NODE as the coordinator (this server only starts it and collects the report).

  remote_coord.py --coordinator fedora --peers phone --checkpoint runs/.../inference.pt \
      --placement "fed-only=local:0,1,2,3,4,5,6,7" --placement "fed5+phone3=local:0,1,2,3,4;phone:5,6,7" --out docs/results/E13_x

"local" in a placement is the coordinating node. The model file is uploaded once into the coordinator's blob cache
(sha256-addressed); the peers' branches are uploaded by the coordinator if they are not loaded there already.
Peers are reached at the addresses of cluster.json, so they must be reachable from the coordinator.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

from nodeclient import NodeError, load_cluster

from bench_cluster import write_report  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--cluster', default=None)
    ap.add_argument('--coordinator', required=True)
    ap.add_argument('--peers', default='')
    ap.add_argument('--checkpoint', type=Path, required=True)
    ap.add_argument('--placement', action='append', required=True)
    ap.add_argument('--prompts', default='32,256')
    ap.add_argument('--decodes', default='64')
    ap.add_argument('--runs', type=int, default=3)
    ap.add_argument('--threads', type=int, default=6)
    ap.add_argument('--kernels', default='native')
    ap.add_argument('--linear', default='gemm')
    ap.add_argument('--wire', default='f4')
    ap.add_argument('--tolerance', type=float, default=2e-3)
    ap.add_argument('--timeout', type=int, default=3600)
    ap.add_argument('--predeploy', action='store_true', help='load the peers\' parts of the model from this server before the job starts')
    ap.add_argument('--blas-threads', type=int, default=0, help='set OPENBLAS_NUM_THREADS for the job (0 = leave the node default)')
    ap.add_argument('--ram-mb', type=float, default=0, help='RAM budget for weights on the coordinator, enforced by the process itself (weights memory-mapped)')
    ap.add_argument('--peer-ram-mb', type=float, default=0, help='the same budget on every peer')
    ap.add_argument('--mmap', action='store_true', help='memory-map the weights without a budget')
    ap.add_argument('--no-deploy', action='store_true', help='the peers already hold their parts of the model: do not send them again')
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--note', default='')
    args = ap.parse_args()
    peers = [p for p in args.peers.split(',') if p]
    cluster_path = Path(args.cluster) if args.cluster else Path(__file__).resolve().parent / 'cluster.json'
    spec = json.loads(cluster_path.read_text())['nodes']
    nodes = load_cluster(args.cluster, only=[args.coordinator] + peers)
    node = nodes[args.coordinator]
    info = node.info()
    print(f"coordinator {args.coordinator}: version {info['version']}, torch {info['torch']}, load {info.get('loadavg')}", flush=True)
    data = args.checkpoint.read_bytes()
    sha = hashlib.sha256(data).hexdigest()
    t0, shown = time.time(), [-1]

    def progress(done, total):
        step = 10 * done // total
        if step != shown[0]:
            shown[0] = step
            print(f'  model file to {args.coordinator}: {100 * done // total}% of {total / 2**20:.0f} MiB ({time.time() - t0:.0f}s)', flush=True)
    node.upload_blob(data, progress)
    del data
    print(f'  model file on {args.coordinator}: sha256 {sha[:12]}', flush=True)
    tag = 'ck' + hashlib.sha256(str(args.checkpoint.resolve()).encode()).hexdigest()[:6] + '-'
    job_args = dict(checkpoint=sha, tag=tag, peers={p: dict(host=spec[p]['host'], port=spec[p].get('port', 8730), delay_ms=spec[p].get('delay_ms', 0)) for p in peers},
                    placements=args.placement, prompts=[int(v) for v in args.prompts.split(',')], decodes=[int(v) for v in args.decodes.split(',')],
                    runs=args.runs, threads=args.threads, kernels=args.kernels, linear=args.linear, wire=args.wire, tolerance=args.tolerance,
                    deploy=not args.no_deploy, ram_mb=args.ram_mb, peer_ram_mb=args.peer_ram_mb, mmap=args.mmap or bool(args.ram_mb or args.peer_ram_mb))
    if args.predeploy and peers:
        import cluster_model
        import coordinator
        model = cluster_model.Model.from_checkpoint(args.checkpoint)
        for note in coordinator.deploy(model, [coordinator.parse_placement(p) for p in args.placement], {p: nodes[p] for p in peers}, tag):
            print('  from this server: ' + note, flush=True)
        del model
    reply = node.call('job_start', json.dumps(job_args).encode(), module='coord_bench', timeout_s=args.timeout,
                      env={'OPENBLAS_NUM_THREADS': str(args.blas_threads)} if args.blas_threads else {})[0]
    job, seen = reply['job'], ''
    while True:
        status, out, _ = node.call('job_status', job=job)
        tail = status.get('stderr_tail', '')
        new = tail[len(seen):] if tail.startswith(seen) else tail[-(len(tail) - tail.rfind(seen[-200:]) - len(seen[-200:])):] if seen[-200:] and seen[-200:] in tail else tail
        if new:
            print(new, end='' if new.endswith('\n') else '\n', flush=True)
        seen = tail
        if not status['running']:
            break
        time.sleep(2.0)
    if status['returncode'] != 0:
        raise NodeError(f"{args.coordinator}: coord_bench failed ({status['returncode']}): {tail[-1500:]}")
    report = json.loads(bytes(out))
    report['note'] = args.note
    report['coordinator']['name'] = args.coordinator
    report['coordinator']['info'] = {k: info.get(k) for k in ('cpu_model', 'android', 'threads', 'kernels', 'version')}
    report['checkpoint'] = str(args.checkpoint)
    write_report(report, args.out)


if __name__ == '__main__':
    main()
