"""Job: how many branch stacks can this device decode at the same time before memory bandwidth saturates?

For k in `ks`, start k separate processes (1 thread each by default), each holding its own copy of one
synthetic stack, and let them decode single tokens during the same time window.
args: {"width":880,"heads":10,"blocks":7,"ks":[1,2,3,4],"seconds":4,"threads":1,"kernels":"math-attn","linear":"gemm"}
"""
import json
import os
import statistics
import subprocess
import sys
import time

args = json.load(sys.stdin)
spec = dict(width=int(args.get('width', 880)), heads=int(args.get('heads', 10)), blocks=int(args.get('blocks', 7)),
            seconds=float(args.get('seconds', 4)), threads=int(args.get('threads', 1)),
            kernels=args.get('kernels', 'math-attn'), linear=args.get('linear', 'gemm'))
per_block = 12 * spec['width'] ** 2 * 4
weight_mib = per_block * spec['blocks'] / 2**20
env = dict(os.environ, OMP_NUM_THREADS=str(spec['threads']), OPENBLAS_NUM_THREADS=str(spec['threads']),
           MKL_NUM_THREADS=str(spec['threads']))
rows = []
for k in args.get('ks', [1, 2, 3, 4]):
    procs = [subprocess.Popen([sys.executable, '-m', 'jobs.concurrency_worker', json.dumps(spec)], env=env,
                              stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True) for _ in range(k)]
    try:
        for p in procs:
            json.loads(p.stdout.readline())               # wait until every worker has built its stack
        start = time.time() + 0.5
        for p in procs:
            p.stdin.write(f'{start}\n')
            p.stdin.flush()
        results = [json.loads(p.stdout.readline()) for p in procs]
    finally:
        for p in procs:
            try:
                p.wait(timeout=20)
            except subprocess.TimeoutExpired:
                p.kill()
    medians = [statistics.median(r['times']) for r in results]
    tokens = sum(len(r['times']) for r in results)
    rows.append(dict(k=k, median_ms_each=[round(m, 2) for m in medians], median_ms=round(statistics.median(medians), 2),
                     worst_ms=round(max(medians), 2), tokens_per_s_total=round(tokens / spec['seconds'], 1),
                     gib_per_s_total=round(tokens / spec['seconds'] * weight_mib / 1024, 1),
                     outputs_agree=len({round(r['checksum'], 2) for r in results}) == 1))
print(json.dumps(dict(spec=spec, weight_mib=round(weight_mib, 1), rows=rows)))
