"""Job: this node is the coordinator of one text generation (the dashboard's "Try it" tab with a device as coordinator).

args: {"checkpoint": "<sha256 of an inference file in this node's blob cache>", "tag": "ck123abc-", "ids": [prompt tokens],
       "assignment": {"local": [0, 1], "phone": [6, 7], "none": []}, "peers": {"phone": {"host": "PHONE_HOST", "port": 8730}},
       "max_tokens": 120, "temperature": 0.8, "seed": 1, "threads": 6, "kernels": "native", "linear": "gemm"}
stderr carries the progress, one line each: "S <status text>" and "T <index> <token id>"; stdout the final JSON.
"""
import json
import os
import sys
from pathlib import Path

args = json.load(sys.stdin)
import torch  # noqa: E402

torch.set_num_threads(int(args.get('threads', 4)))
torch.set_grad_enabled(False)
import cpu_kernels  # noqa: E402
import stacks  # noqa: E402

cpu_kernels.install_kernels(args.get('kernels', 'native'))
stacks.set_linear_mode(args.get('linear', 'gemm'))
import cluster_model  # noqa: E402
import coordinator  # noqa: E402
from peerclient import Node  # noqa: E402


def say(kind, text):
    print(f'{kind} {text}', file=sys.stderr, flush=True)


home = Path(os.environ['BRANCHY_HOME'])
say('S', 'loading the model')
model = cluster_model.Model.from_checkpoint(home / 'cache' / 'blobs' / f"{args['checkpoint']}.bin")
assignment = {k: [int(i) for i in v] for k, v in (args.get('assignment') or {'local': []}).items()}
used = [k for k in assignment if k not in ('local', 'none') and assignment[k]]
token = (home / 'token').read_text().strip()
nodes = {name: Node(name, args['peers'][name]['host'], args['peers'][name].get('port', 8730), token) for name in used}
tag = args.get('tag', 'b')
for worker in used:
    say('S', f'connecting to {worker}')
    nodes[worker].connect(timeout=8)
    for i in assignment[worker]:
        say('S', f'checking branch {i} on {worker}')
        branch = model.branches[i]
        nodes[worker].ensure_stack(f'{tag}{i}', branch.state_dict(), model.branch_width, model.branch_heads, len(branch),
                                   lambda done, total, worker=worker, i=i: say('S', f'sending branch {i} to {worker}: {100 * done // total}%'))
if model.branches and (used or assignment.get('none')):
    exe = coordinator.DistributedExecutor(model, {k: v for k, v in assignment.items() if v or k == 'local'}, nodes, 'f4', tag)
else:
    exe = cluster_model.LocalExecutor(model)
say('S', 'reading the beginning')
try:
    stats = coordinator.generate(exe, args['ids'], int(args.get('max_tokens', 120)), float(args.get('temperature', 0.8)),
                                 int(args.get('seed', 1)), on_token=lambda out: say('T', f'{len(out) - 1} {out[-1]}'))
finally:
    exe.close()
print(json.dumps(stats))
