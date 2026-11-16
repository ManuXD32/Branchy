"""Job: this node is the coordinator of a placement benchmark (trunk, fusion and its own branches here, the other
branches on peer nodes). Same loop and parity gate as bench_cluster.py on the controller.

args: {"checkpoint": "<sha256 of an inference file in this node's blob cache>", "tag": "ck123abc-",
       "peers": {"phone": {"host": "PHONE_HOST", "port": 8730}}, "placements": ["all=local:0,1,2,3,4,5,6,7", "x=local:0,1;phone:2,3"],
       "prompts": [32], "decodes": [64], "runs": 3, "threads": 6, "kernels": "native", "linear": "gemm", "wire": "f4",
       "tolerance": 0.002}
Progress lines go to stderr (visible through job_status), the JSON report to stdout.
"""
import json
import os
import sys
import time
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

home = Path(os.environ['BRANCHY_HOME'])
lines = []


def log(text):
    lines.append(text)
    print(text, file=sys.stderr, flush=True)


# low-memory experiments: weights paged in from the file; ram_mb = RAM budget for weights enforced by the process itself
mapped = bool(args.get('mmap')) or bool(args.get('ram_mb')) or os.environ.get('BRANCHY_MMAP') == '1'
model = cluster_model.Model.from_checkpoint(home / 'cache' / 'blobs' / f"{args['checkpoint']}.bin", mmap=mapped)
model.label = model.label.split(' (')[0]
log(f"model: {model.label}, {model.parameters():,} parameters; coordinator threads={torch.get_num_threads()} "
    f"linear={stacks.LINEAR_MODE} kernels={args.get('kernels', 'native')}" + (' weights memory-mapped' if mapped else ''))
token = (home / 'token').read_text().strip()
nodes = {name: Node(name, peer['host'], peer.get('port', 8730), token, delay_ms=float(peer.get('delay_ms', 0)))
         for name, peer in args.get('peers', {}).items()}
placements = [coordinator.parse_placement(p) for p in args['placements']]
keep = ('name', 'version', 'torch', 'threads', 'kernels', 'linear', 'cpu_model', 'android', 'thermal_max_c', 'loadavg', 'battery_pct')
before = {name: {k: node.info().get(k) for k in keep} for name, node in nodes.items()}
# deploy=false: the workers already hold their stacks (a low-memory coordinator cannot serialise them again)
notes = coordinator.deploy(model, placements, nodes, args['tag'], mmap=mapped) if args.get('deploy', True) else ['deployment skipped']
budgets = {}
if args.get('ram_mb'):
    stacks.set_budget(float(args['ram_mb']))
    log(f"  RAM budget for weights: {args['ram_mb']} MB here" + (f", {args['peer_ram_mb']} MB on each peer" if args.get('peer_ram_mb') else ''))
for node in nodes.values():
    node.call('set_budget', mb=float(args.get('peer_ram_mb') or 0))
for note in notes:
    log('  ' + note)
try:
    results = coordinator.run_placements(model, placements, nodes, [int(v) for v in args.get('prompts', [32])],
                                         [int(v) for v in args.get('decodes', [64])], int(args.get('runs', 3)), args.get('wire', 'f4'),
                                         float(args.get('tolerance', 2e-3)), args['tag'], log)
finally:                                   # never leave a budget behind on a worker
    for name, node in nodes.items():
        try:
            budgets[name] = node.call('set_budget', mb=0)[0].get('previous')
        except Exception:  # noqa: BLE001
            pass
    budgets['local'] = stacks.set_budget(0)
after = {name: {k: node.info().get(k) for k in keep} for name, node in nodes.items()}
print(json.dumps(dict(date=time.strftime('%Y-%m-%d %H:%M:%S'), model=model.label, parameters=model.parameters(),
                      coordinator=dict(threads=torch.get_num_threads(), linear=stacks.LINEAR_MODE, kernels=args.get('kernels', 'native'),
                                       torch=torch.__version__, mmap=mapped, loadavg=os.getloadavg() if hasattr(os, 'getloadavg') else None),
                      wire=args.get('wire', 'f4'), runs=int(args.get('runs', 3)), tolerance=float(args.get('tolerance', 2e-3)),
                      nodes_before=before, nodes_after=after, deploy=notes, ram_mb=args.get('ram_mb'), peer_ram_mb=args.get('peer_ram_mb'), budgets=budgets, results=results, sequential=None, log=lines)))
