#!/usr/bin/env python3
"""E16: two simulated low-memory devices on this server.

Each "device" is a Branchy node in its own control group with a memory limit (systemd-run, MemoryMax, no swap) and its
own CPU cores. Weights are memory-mapped (BRANCHY_MMAP=1), so a device whose limit is below the model size keeps
running but reads weights back from storage for every token, as llama.cpp does with mmap. Device A coordinates
(`coord_bench` job inside A's limit), device B is the worker; the network between them is the loopback plus an
injected round-trip delay (aliases devB20 / devB45 of device B).

  run_lowmem.py --out docs/results/E16_low_memory --weights-mb 0,1000,800,600     (0 = no limit)

For every memory budget and both models (sequential, 8 branches with delay): the model alone on device A, and split
over A and B with 0, 20 and 45 ms of network delay (layer split for the sequential model, branch split for the other).
The limit of a device = what it needs without weights (measured in the unlimited pass) + the budget for weights.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
NODE = ROOT / 'src' / 'node'
sys.path[:0] = [str(NODE), str(NODE / 'payload')]
from nodeclient import load_cluster  # noqa: E402

PY = sys.executable          # the Python that runs this script must have torch
BASE = ROOT / 'runs' / 'lowmem'
DEVICES = {'devA': dict(port=8741, cpus=None, unit='branchy-lowmem-a'),        # cpus: set with --cpus-a / --cpus-b
           'devB': dict(port=8742, cpus=None, unit='branchy-lowmem-b')}
MODELS = {'sequential': (ROOT / 'runs/e11_long_300m/seq-full/inference.pt',
                         ['alone=local:0-18', 'split-0ms=local:0-9;devB:10-18', 'split-20ms=local:0-9;devB20:10-18',
                          'split-45ms=local:0-9;devB45:10-18']),
          'branched': (ROOT / 'runs/e11_long_300m/flat8-delay1-drop20-full/inference.pt',
                       ['alone=local:0,1,2,3,4,5,6,7', 'split-0ms=local:0,1,2,3;devB:4,5,6,7',
                        'split-20ms=local:0,1,2,3;devB20:4,5,6,7', 'split-45ms=local:0,1,2,3;devB45:4,5,6,7'])}
MARGIN_MB = 96          # program files of Python and torch that must stay in memory next to the weights


def sh(*cmd, check=True):
    return subprocess.run(list(cmd), check=check, capture_output=True, text=True)


def start(threads):
    token = (ROOT / '.secrets' / 'node_token').read_text()
    nodes = {}
    for name, d in DEVICES.items():
        home = BASE / name
        (home / 'cache').mkdir(parents=True, exist_ok=True)
        (home / 'token').write_text(token)
        os.chmod(home / 'token', 0o600)
        (home / 'node_state.json').write_text(json.dumps(dict(
            name=name, port=d['port'], bind='127.0.0.1', threads=threads, kernels='math-attn', linear='mv',
            affinity=d['cpus'], env={}, cache_gib=32), indent=1))
        sh('systemctl', 'stop', d['unit'], check=False)
        for blob in (home / 'cache').rglob('*.bin'):      # cached pages of an earlier run belong to no device: drop them,
            fd = os.open(blob, os.O_RDONLY)               # or the new device would use them without being charged
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            os.close(fd)
        sh('systemd-run', f"--unit={d['unit']}", '--collect', '--quiet', '-p', 'MemoryMax=infinity', '-p', 'MemorySwapMax=0',
           '--setenv=BRANCHY_MMAP=1', f'--setenv=OPENBLAS_NUM_THREADS={threads}', f'--working-directory={NODE / "payload"}',
           PY, str(NODE / 'payload' / 'branchy_node.py'), '--home', str(home))
        nodes[name] = dict(host='127.0.0.1', port=d['port'])
    nodes['devB20'] = dict(nodes['devB'], delay_ms=20)
    nodes['devB45'] = dict(nodes['devB'], delay_ms=45)
    (BASE / 'cluster.json').write_text(json.dumps(dict(token_file=str(BASE / 'devA' / 'token'), nodes=nodes), indent=1))
    live = load_cluster(str(BASE / 'cluster.json'), only=['devA', 'devB'])
    for name, node in live.items():
        info = node.wait_up(60)
        print(f"{name}: up, threads {info['threads']}, cores {info['affinity']}", flush=True)


def stop():
    for d in DEVICES.values():
        sh('systemctl', 'stop', d['unit'], check=False)


def set_limit(name, megabytes):
    value = 'infinity' if megabytes is None else f'{int(megabytes)}M'
    sh('systemctl', 'set-property', '--runtime', DEVICES[name]['unit'] + '.service', f'MemoryMax={value}')


def bench(model, out, args, deploy=True, quick=False):
    checkpoint, placements = MODELS[model]
    cmd = [PY, '-u', str(NODE / 'remote_coord.py'), '--cluster', str(BASE / 'cluster.json'), '--coordinator', 'devA',
           '--peers', 'devB,devB20,devB45', '--checkpoint', str(checkpoint), '--prompts', '8' if quick else args.prompts,
           '--decodes', '2' if quick else args.decodes, '--runs', '1' if quick else str(args.runs), '--threads', str(args.threads), '--kernels', 'math-attn', '--linear', 'mv',
           '--blas-threads', str(args.threads), '--timeout', '7200', '--out', str(out)]
    for p in placements:
        cmd += ['--placement', p]
    if not deploy:
        cmd.append('--no-deploy')
    env = dict(os.environ, PYTHONPATH=str(NODE / 'payload'), OPENBLAS_NUM_THREADS='2')
    proc = subprocess.Popen(cmd, cwd=NODE, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in proc.stdout:
        if not line.lstrip().startswith('model file to'):
            print('   ' + line.rstrip()[:260], flush=True)
    if proc.wait() != 0:
        return None
    return json.loads((out / 'bench.json').read_text())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--weights-mb', default='0,1000,800,600', help='memory budgets for weights per device; 0 = no limit (must come first)')
    ap.add_argument('--models', default='sequential,branched')
    ap.add_argument('--prompts', default='32')
    ap.add_argument('--decodes', default='24')
    ap.add_argument('--runs', type=int, default=2)
    ap.add_argument('--threads', type=int, default=3)
    ap.add_argument('--seq', type=Path, default=MODELS['sequential'][0], help='inference file of the sequential model')
    ap.add_argument('--branched', type=Path, default=MODELS['branched'][0], help='inference file of the 8-branch model')
    ap.add_argument('--cpus-a', default='', help='CPU cores of device A, e.g. 12,13,14 (default: not pinned)')
    ap.add_argument('--cpus-b', default='', help='CPU cores of device B')
    args = ap.parse_args()
    MODELS['sequential'] = (args.seq.resolve(), MODELS['sequential'][1])
    MODELS['branched'] = (args.branched.resolve(), MODELS['branched'][1])
    DEVICES['devA']['cpus'] = [int(c) for c in args.cpus_a.split(',') if c] or None
    DEVICES['devB']['cpus'] = [int(c) for c in args.cpus_b.split(',') if c] or None
    budgets = [int(v) for v in args.weights_mb.split(',')]
    args.out = args.out.resolve()        # the benchmark runs from another folder
    args.out.mkdir(parents=True, exist_ok=True)
    summary_path = args.out / 'summary.json'
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else dict(overhead_mb={}, rows=[])
    models = args.models.split(',')

    def load_workers():
        """Without limits: device B loads its parts of every model (sending them needs memory on device A)."""
        for name in DEVICES:
            set_limit(name, None)
        for model in models:
            print(f'== {time.strftime("%T")} {model}: loading the worker parts (no limits, not measured)', flush=True)
            if bench(model, args.out / f'_deploy_{model}', args, quick=True) is None:
                raise SystemExit('loading the worker parts failed')
            shutil.rmtree(args.out / f'_deploy_{model}', ignore_errors=True)

    start(args.threads)
    try:
        load_workers()
        for budget in budgets:
            for model in models:
                if budget:
                    need = summary['overhead_mb'].get(model)
                    if not need:
                        raise SystemExit('run the unlimited pass (budget 0) first: it measures what a device needs without weights')
                    limits = {name: need[name] + MARGIN_MB + budget for name in DEVICES}
                else:
                    limits = {name: None for name in DEVICES}
                for name, limit in limits.items():
                    set_limit(name, limit)
                print(f"== {time.strftime('%T')} {model}, weights budget {budget or 'unlimited'} MB per device, limits {limits}", flush=True)
                report = bench(model, args.out / f"{model}_{budget or 'unlimited'}", args, deploy=False)
                if report is None:
                    print('   FAILED (see above); most likely the device ran out of memory', flush=True)
                    summary['rows'] = [r for r in summary['rows'] if (r['model'], r['budget_mb']) != (model, budget)]
                    summary['rows'].append(dict(model=model, budget_mb=budget, failed=True))
                    summary_path.write_text(json.dumps(summary, indent=1))
                    start(args.threads)
                    load_workers()
                    continue
                if not budget:      # memory a device uses besides the weight files: everything that is not page cache
                    def other(res):
                        return (res.get('anon', 0) + res.get('kernel', 0)) / 2**20
                    a = max(other(r['resources'].get('local', {})) for r in report['results'])
                    b = max((other(v) for r in report['results'] for k, v in r['resources'].items() if k != 'local'), default=0)
                    summary['overhead_mb'][model] = dict(devA=round(a), devB=round(b))
                    print(f"   memory besides weights: device A {a:.0f} MB, device B {b:.0f} MB", flush=True)
                summary['rows'] = [r for r in summary['rows'] if (r['model'], r['budget_mb']) != (model, budget)]
                for r in report['results']:
                    res = r.get('resources', {})
                    passes = max(1, res.get('local', {}).get('passes', 1))
                    read = {k: v.get('io_read_bytes', 0) / passes / 2**20 for k, v in res.items()}
                    refault = {k: v.get('workingset_refault_file', 0) * 4096 / passes / 2**20 for k, v in res.items()}
                    summary['rows'].append(dict(
                        model=model, budget_mb=budget, limit_mb=limits, placement=r['placement'], prompt=r['prompt'],
                        decode_tps=r['decode_tps'], prefill_tps=r['prefill_tps'], step_p50_ms=r['step_total_ms']['p50'],
                        step_p95_ms=r['step_total_ms']['p95'], parity=r['parity'], max_abs_logit_diff=r['max_abs_logit_diff'],
                        read_mb_per_token=read, refault_mb_per_token=refault,
                        memory_mb={k: dict(current=v.get('current', 0) / 2**20, file=v.get('file', 0) / 2**20,
                                           anon=v.get('anon', 0) / 2**20) for k, v in res.items()}))
                summary_path.write_text(json.dumps(summary, indent=1))
    finally:
        stop()
    print(f"== {time.strftime('%T')} finished", flush=True)


if __name__ == '__main__':
    main()
