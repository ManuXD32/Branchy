"""Job: this node coordinates a benchmark of LFM3-1.7B: sequential, layer split and branch layouts (lfm3_model plans).

args: {"kinds": [24 layer kinds], "embed": "<sha of embed.bin>", "tag": "lfm",
       "stacks": {"0-4": "<sha>", ...}            every stack any case uses (blob must be on the node that runs it),
       "peers": {"phone": {"host": "PHONE_HOST", "port": 8730}},
       "cases": [{"label": "sequential", "plan": [["seq", "local", 0, 4], ...], "reference": null},
                 {"label": "4x4, 2 on the phone", "plan": [...], "reference": "4x4 here"}],
       "prompts": [32], "decodes": [32], "runs": 2, "threads": 6, "kernels": "native", "linear": "gemm", "tolerance": 0.02}
A case with a reference is teacher-forced with the reference's tokens and must reproduce its logits.
"""
import json
import os
import statistics
import sys
import time
from pathlib import Path

args = json.load(sys.stdin)
import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(int(args.get('threads', 4)))
torch.set_grad_enabled(False)
import cpu_kernels  # noqa: E402
import stacks  # noqa: E402

cpu_kernels.install_kernels(args.get('kernels', 'native'))
stacks.set_linear_mode(args.get('linear', 'gemm'))
import coordinator  # noqa: E402
import lfm3_model  # noqa: E402
from cluster_model import run_trial  # noqa: E402
from peerclient import Node  # noqa: E402

home = Path(os.environ['BRANCHY_HOME'])
blobs = home / 'cache' / 'blobs'
lines = []


def log(text):
    lines.append(text)
    print(text, file=sys.stderr, flush=True)


kinds, tag = args['kinds'], args.get('tag', 'lfm')
# low-memory experiments: weights are paged in from the files; ram_mb = RAM budget enforced by the process itself
mapped = bool(args.get('mmap')) or bool(args.get('ram_mb')) or os.environ.get('BRANCHY_MMAP') == '1'
width = 2048
if mapped:
    path = blobs / f"{args['embed']}.bin"
    flat = torch.from_file(str(path), shared=False, size=path.stat().st_size // 4, dtype=torch.float32)
    model = lfm3_model.Lfm3(flat[:-width].view(-1, width), flat[-width:].clone(), kinds)
else:
    raw = np.fromfile(blobs / f"{args['embed']}.bin", dtype='<f4')
    model = lfm3_model.Lfm3(torch.from_numpy(raw[:-width].reshape(-1, width)), torch.from_numpy(raw[-width:].copy()), kinds)
needed = {s for case in args['cases'] for s in lfm3_model.plan_stacks(case['plan'])}
token = (home / 'token').read_text().strip()
nodes = {name: Node(name, peer['host'], peer.get('port', 8730), token, delay_ms=float(peer.get('delay_ms', 0)))
         for name, peer in args.get('peers', {}).items()}
for where, first, end in sorted(needed, key=lambda s: (s[1], s[0])):
    sha = args['stacks'][f'{first}-{end}']
    t = time.perf_counter()
    if where == 'local':
        model.add_stack(first, end, path=blobs / f'{sha}.bin', mmap=mapped)
    else:
        nodes[where].call('load_stack', timeout=900, name=lfm3_model.stack_name(tag, first, end), width=width, heads=16,
                          blocks=end - first, sha=sha, kind='lfm3', layers=kinds[first:end], **({'mmap': True} if mapped else {}))
    log(f'  layers {first}-{end - 1} ready on {where} ({time.perf_counter() - t:.1f}s)')
log(f'model: {model.label}; coordinator threads={torch.get_num_threads()} linear={stacks.LINEAR_MODE} kernels={args.get("kernels", "native")}'
    + (' weights memory-mapped' if mapped else ''))
results, references = [], {}
if args.get('ram_mb'):
    stacks.set_budget(float(args['ram_mb']))
    log(f"  RAM budget for weights: {args['ram_mb']} MB here" + (f", {args['peer_ram_mb']} MB on each peer" if args.get('peer_ram_mb') else ''))
for node in nodes.values():
    node.call('set_budget', mb=float(args.get('peer_ram_mb') or 0))
for p in [int(v) for v in args.get('prompts', [32])]:
    ids = (stacks.synthetic_tensor([1, p], f'prompt/{p}', 1.0).abs() * 1e6).long() % model.vocab
    for d in [int(v) for v in args.get('decodes', [32])]:
        for case in args['cases']:
            exe = lfm3_model.Lfm3Executor(model, case['plan'], nodes, 'f4', tag)
            try:
                ref = references.get((case.get('reference'), p, d))
                forced = None if ref is None else ref['tokens']
                run_trial(exe, ids, 2, forced)
                used = sorted({w for w, _, _ in lfm3_model.plan_stacks(case['plan']) if w != 'local'})
                res0 = coordinator.measure_resources(nodes, used) if hasattr(coordinator, 'measure_resources') else {}
                trials = [run_trial(exe, ids, d, forced) for _ in range(int(args.get('runs', 2)))]
                res1 = coordinator.measure_resources(nodes, used) if res0 else {}
            finally:
                exe.close()
            row = coordinator.aggregate(trials, p, d)
            if res1:
                import resources
                row['resources'] = {k: dict(resources.delta(res0.get(k, {}), v), passes=int(args.get('runs', 2)) * (d + 1)) for k, v in res1.items() if v}
            row.update(placement=case['label'], plan=case['plan'], reference=case.get('reference'),
                       devices=len({w for w, _, _ in lfm3_model.plan_stacks(case['plan'])}))
            if ref is None:
                references[(case['label'], p, d)] = trials[0]
                row.update(parity='reference', max_abs_logit_diff=0.0, argmax_agreement=1.0)
            else:
                diff = max(float((t['logits'] - ref['logits']).abs().max()) for t in trials)
                agree = statistics.fmean(float((t['logits'].argmax(-1) == ref['logits'].argmax(-1)).float().mean()) for t in trials)
                row.update(parity='ok' if diff <= float(args.get('tolerance', 2e-2)) else 'FAIL', max_abs_logit_diff=diff, argmax_agreement=agree)
            results.append(row)
            waits = ' '.join(f"{n}:rpc {r['rpc_ms']['p50']:.0f} (compute {r['compute_ms']['p50']:.0f} net {r['net_ms']['p50']:.0f})" for n, r in row['remote'].items())
            log(f"  {case['label']:34s} prompt {p:>4} decode {d:>3}: prefill {row['prefill_tps']:6.1f} tok/s | decode {row['decode_tps']:5.2f} tok/s "
                f"(step p50 {row['step_total_ms']['p50']:.0f} p95 {row['step_total_ms']['p95']:.0f} ms; here {row['step_shared_ms']['p50']:.0f} "
                f"wait {row['step_wait_ms']['p50']:.0f} output {row['step_fuse_ms']['p50']:.0f}) | parity {row['parity']} {row['max_abs_logit_diff']:.1e} | {waits}")
budgets = {'local': stacks.set_budget(0)}
for name, node in nodes.items():
    try:
        budgets[name] = node.call('set_budget', mb=0)[0].get('previous')
    except Exception:  # noqa: BLE001
        pass
print(json.dumps(dict(date=time.strftime('%Y-%m-%d %H:%M:%S'), model=model.label, parameters=1_730_000_000, ram_mb=args.get('ram_mb'), peer_ram_mb=args.get('peer_ram_mb'), budgets=budgets,
                      coordinator=dict(threads=torch.get_num_threads(), linear=stacks.LINEAR_MODE, kernels=args.get('kernels', 'native'),
                                       torch=torch.__version__, loadavg=os.getloadavg() if hasattr(os, 'getloadavg') else None),
                      wire='f4', runs=int(args.get('runs', 2)), tolerance=float(args.get('tolerance', 2e-2)), nodes_before={}, nodes_after={},
                      results=results, sequential=None, log=lines)))
