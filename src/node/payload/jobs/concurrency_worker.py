"""Child process of concurrency_probe: decode one synthetic stack in a loop during a shared time window."""
import json
import os
import sys
import time

args = json.loads(sys.argv[1])
os.environ.setdefault('OMP_NUM_THREADS', '1')
import torch  # noqa: E402

torch.set_num_threads(int(args.get('threads', 1)))
torch.set_grad_enabled(False)
import cpu_kernels  # noqa: E402
import stacks  # noqa: E402

cpu_kernels.install_kernels(args.get('kernels', 'math-attn'))
stacks.set_linear_mode(args.get('linear', 'gemm'))
width, heads, blocks = args['width'], args['heads'], args['blocks']
stack = stacks.build_stack(width, heads, blocks)
stacks.fill_synthetic(stack, 'probe')
x = stacks.synthetic_tensor([1, 4096, width], 'probe/x', 1.0)
_, cache0 = stacks.run_stack(stack, x[:, :32], None)
y, _ = stacks.run_stack(stack, x[:, 32:33], cache0)          # warm-up + reference output
print(json.dumps(dict(ready=True)), flush=True)
start = float(sys.stdin.readline())                           # parent sends the common start time
while time.time() < start:
    time.sleep(0.001)
end = start + float(args['seconds'])
times, cache, i = [], cache0, 32
while time.time() < end:
    t = time.perf_counter()
    out, cache = stacks.run_stack(stack, x[:, i:i + 1], cache)
    times.append((time.perf_counter() - t) * 1000)
    i += 1
    if i >= 96:                                               # keep the context short and constant-ish
        cache, i = cache0, 32
print(json.dumps(dict(times=[round(t, 3) for t in times], checksum=float(y.abs().sum()))), flush=True)
