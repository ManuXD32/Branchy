#!/usr/bin/env python3
"""E16 part 2: LFM3-1.7B (6.6 GB of float32 weights) on the two simulated low-memory devices of run_lowmem.py.

Cases: the model alone on device A; layer split 12 + 12; the 4 x 4 branch layout of E6 alone and with three branches on
device B (the coordinator waits for them; no delay training for this model); each split with 0, 20 and 45 ms network.

  run_lowmem_lfm3.py --out docs/results/E16_low_memory/lfm3 --weights-mb 0,4000
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_lowmem as R  # noqa: E402

SEQ = [['seq', 'local', a, a + 4] for a in range(0, 24, 4)]


def cases():
    out = [dict(label='alone', reference=None, plan=SEQ)]
    for name, tag in (('devB', '0ms'), ('devB20', '20ms'), ('devB45', '45ms')):
        out.append(dict(label=f'layer-split-{tag}', reference='alone',
                        plan=SEQ[:3] + [['seq', name, a, a + 4] for a in (12, 16, 20)]))
    out.append(dict(label='4x4-alone', reference=None,
                    plan=[SEQ[0], ['par', [['local', a, a + 4] for a in (4, 8, 12, 16)]], SEQ[5]]))
    for name, tag in (('devB', '0ms'), ('devB20', '20ms'), ('devB45', '45ms')):
        out.append(dict(label=f'4x4-split-{tag}', reference='4x4-alone',
                        plan=[SEQ[0], ['par', [['local', 4, 8]] + [[name, a, a + 4] for a in (8, 12, 16)]], SEQ[5]]))
    return out


def bench(out, args):
    out.mkdir(parents=True, exist_ok=True)
    (out / 'cases.json').write_text(json.dumps(cases(), indent=1))
    cmd = [R.PY, '-u', str(R.NODE / 'remote_lfm3.py'), '--cluster', str(R.BASE / 'cluster.json'), '--coordinator', 'devA',
           '--peers', 'devB,devB20,devB45', '--cases', str(out / 'cases.json'), '--prompts', args.prompts, '--decodes', args.decodes,
           '--runs', str(args.runs), '--threads', str(args.threads), '--kernels', 'math-attn', '--linear', 'mv',
           '--blas-threads', str(args.threads), '--timeout', '10800', '--out', str(out)]
    env = dict(os.environ, PYTHONPATH=str(R.NODE / 'payload'), OPENBLAS_NUM_THREADS='2')
    proc = subprocess.Popen(cmd, cwd=R.NODE, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in proc.stdout:
        if ' MiB (' not in line:
            print('   ' + line.rstrip()[:300], flush=True)
    if proc.wait() != 0 or not (out / 'bench.json').exists():
        return None
    return json.loads((out / 'bench.json').read_text())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--weights-mb', default='0,4000')
    ap.add_argument('--prompts', default='16')
    ap.add_argument('--decodes', default='8')
    ap.add_argument('--runs', type=int, default=1)
    ap.add_argument('--threads', type=int, default=3)
    args = ap.parse_args()
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    summary_path = args.out / 'summary.json'
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else dict(overhead_mb={}, rows=[])
    R.start(args.threads)
    try:
        for budget in [int(v) for v in args.weights_mb.split(',')]:
            if budget:
                need = summary['overhead_mb'].get('lfm3')
                if not need:
                    raise SystemExit('run the unlimited pass (budget 0) first')
                limits = {name: need[name] + R.MARGIN_MB + budget for name in R.DEVICES}
            else:
                limits = {name: None for name in R.DEVICES}
            for name, limit in limits.items():
                R.set_limit(name, limit)
            print(f"== {time.strftime('%T')} LFM3-1.7B, weights budget {budget or 'unlimited'} MB per device, limits {limits}", flush=True)
            report = bench(args.out / f"lfm3_{budget or 'unlimited'}", args)
            summary['rows'] = [r for r in summary['rows'] if r['budget_mb'] != budget]
            if report is None:
                print('   FAILED (see above)', flush=True)
                summary['rows'].append(dict(model='lfm3', budget_mb=budget, failed=True))
                summary_path.write_text(json.dumps(summary, indent=1))
                R.start(args.threads)
                continue
            if not budget:
                def other(res):
                    return (res.get('anon', 0) + res.get('kernel', 0)) / 2**20
                a = max(other(r.get('resources', {}).get('local', {})) for r in report['results'])
                b = max((other(v) for r in report['results'] for k, v in r.get('resources', {}).items() if k != 'local'), default=0)
                summary['overhead_mb']['lfm3'] = dict(devA=round(a), devB=round(b))
                print(f'   memory besides weights: device A {a:.0f} MB, device B {b:.0f} MB', flush=True)
            for r in report['results']:
                res = r.get('resources', {})
                passes = max(1, res.get('local', {}).get('passes', 1))
                summary['rows'].append(dict(
                    model='lfm3', budget_mb=budget, limit_mb=limits, placement=r['placement'], prompt=r['prompt'],
                    decode_tps=r['decode_tps'], prefill_tps=r['prefill_tps'], step_p50_ms=r['step_total_ms']['p50'],
                    step_p95_ms=r['step_total_ms']['p95'], parity=r['parity'], max_abs_logit_diff=r['max_abs_logit_diff'],
                    read_mb_per_token={k: v.get('io_read_bytes', 0) / passes / 2**20 for k, v in res.items()},
                    refault_mb_per_token={k: v.get('workingset_refault_file', 0) * 4096 / passes / 2**20 for k, v in res.items()},
                    memory_mb={k: dict(current=v.get('current', 0) / 2**20, file=v.get('file', 0) / 2**20,
                                       anon=v.get('anon', 0) / 2**20) for k, v in res.items()}))
            summary_path.write_text(json.dumps(summary, indent=1))
    finally:
        R.stop()
    print(f"== {time.strftime('%T')} finished", flush=True)


if __name__ == '__main__':
    main()
