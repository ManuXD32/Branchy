"""Coordinator side of a Branchy cluster: the executor that spreads the branches of one model over this machine and
remote nodes, and the placement benchmark loop.

Lives in the payload so that any node can be the coordinator (jobs/coord_bench.py); the controller imports it too
(bench_cluster.py).
"""
from __future__ import annotations

import concurrent.futures
import hashlib
import statistics
import uuid

import torch

import resources
import stacks
from cluster_model import Lag, LocalExecutor, perf, run_trial, summarize
from peerclient import NodeError


class DistributedExecutor:
    def __init__(self, model: Model, assignment: dict, nodes: dict, wire='f4', tag='b'):
        self.model, self.nodes, self.wire = model, nodes, wire
        self.local = sorted(assignment.get('local', []))
        self.missing = sorted(assignment.get('none', []))     # branches nobody computes (zeros at fusion, as in dropout training)
        self.remote = {name: sorted(ids) for name, ids in assignment.items() if name not in ('local', 'none') and ids}
        covered = sorted(self.local + self.missing + [i for ids in self.remote.values() for i in ids])
        if covered != list(range(model.n_branches)):
            raise ValueError(f'placement must cover every branch exactly once, got {covered}')
        self.stack_name = lambda i: f'{tag}{i}'
        self.session = uuid.uuid4().hex[:12]
        self.cache = None
        # one thread per node: calls to a node stay in order (its cache depends on it) and never share the socket
        self.pools = {name: concurrent.futures.ThreadPoolExecutor(max_workers=1) for name in self.remote}
        self.lag = Lag(model)
        self.late = {}             # delay 1: replies of the previous token that have not been collected yet
        if model.delay and any(i < model.local_branches for ids in list(self.remote.values()) + [self.missing] for i in ids):
            raise ValueError('with delay, the first (never delayed) branch must stay on the coordinator')

    def drain(self):
        """Delay 1: wait for the replies of the last token. Nobody fuses them, but the node connections are shared with
        the other placements, and a request still in flight would collide with the next placement's first request
        (seen in E12: 'bad header size' on the node's socket)."""
        for future in self.late.values():
            future.result()
        self.late = {}

    def close(self):
        for pool in self.pools.values():
            pool.shutdown(wait=True)
        for name in self.remote:
            try:
                self.nodes[name].call('drop_session', session=self.session)
            except NodeError:
                pass

    @torch.inference_mode()
    def step(self, ids, reset=False):
        m = self.model
        cache = None if reset else self.cache
        t0 = perf()
        h, shared = m.trunk(ids, None if cache is None else cache['shared'])
        t1 = perf()
        def node_input(branch_ids):          # one tensor for all stacks, or one per stack if branches have their own projection
            return h if m.inproj is None else torch.cat([m.branch_input(i, h) for i in branch_ids], dim=0)
        if reset:
            for future in self.late.values():      # a new sequence: let the previous one finish on the nodes first
                future.result()
            self.late = {}
        futures = {name: self.pools[name].submit(self.nodes[name].forward, [self.stack_name(i) for i in branch_ids],
                                                 node_input(branch_ids), reset, self.session, self.wire, self.wire)
                   for name, branch_ids in self.remote.items()}
        outs = [None] * m.n_branches
        local_caches = {}
        for i in self.local:
            y, local_caches[i] = stacks.run_stack(m.branches[i], m.branch_input(i, h),
                                                  None if cache is None else cache['branches'][i])
            outs[i] = self.lag.shift(i, y, reset)
        t2 = perf()
        remote_stats = []
        if m.delay and not reset:
            # Delay 1: this token is fused with the remote outputs of the PREVIOUS token (already requested one step ago);
            # the requests just sent are collected at the next step, so the network is off the per-token path.
            collect, self.late = self.late, futures
            for name, future in collect.items():
                result, stats = future.result()
                stats['done_at_s'] = perf() - t1
                remote_stats.append(stats)
                for i in self.remote[name]:
                    self.lag.held[i] = result[self.stack_name(i)][:, -1:]
            for name in self.remote:
                for i in self.remote[name]:
                    outs[i] = self.lag.held[i]
        else:
            for name, future in futures.items():
                result, stats = future.result()
                stats['done_at_s'] = perf() - t1
                remote_stats.append(stats)
                for i in self.remote[name]:
                    outs[i] = self.lag.shift(i, result[self.stack_name(i)], reset)
        t3 = perf()
        logits, fused = m.head(outs, h, None if cache is None else cache['fused'])
        t4 = perf()
        self.cache = dict(shared=shared, branches=local_caches, fused=fused)
        return logits, dict(total_s=t4 - t0, shared_s=t1 - t0, local_s=t2 - t1, wait_s=t3 - t2, fuse_s=t4 - t3,
                            remote=remote_stats)


class PipelineExecutor:
    """Layer split of a SEQUENTIAL model, the way llama.cpp's RPC backend offloads layers: consecutive blocks live on
    different machines and the coordinator passes the hidden state through them in order, so every token costs one
    round trip per remote segment and nothing runs in parallel. `assignment` maps 'local' or a node name to a
    contiguous range of block numbers, e.g. {'local': [0..9], 'phone': [10..18]}."""

    def __init__(self, model, assignment: dict, nodes: dict, wire='f4', tag='s'):
        self.model, self.nodes, self.wire, self.tag = model, nodes, wire, tag
        self.segments = pipeline_segments(model, assignment)
        self.blocks = {(a, b): model.shared[a:b] for where, a, b in self.segments if where == 'local'}
        self.session = uuid.uuid4().hex[:12]
        self.cache, self.position = {}, 0

    def close(self):
        for where in {w for w, _, _ in self.segments if w != 'local'}:
            try:
                self.nodes[where].call('drop_session', session=self.session)
            except NodeError:
                pass

    @torch.inference_mode()
    def step(self, ids, reset=False):
        m = self.model
        if reset:
            self.cache, self.position = {}, 0
        if self.position + ids.shape[1] > m.context:
            raise ValueError('sequence exceeds context')
        t0 = perf()
        h = m.tok[ids] + m.pos[self.position:self.position + ids.shape[1]]
        self.position += ids.shape[1]
        local_s = wait_s = 0.0
        remote_stats = []
        for where, a, b in self.segments:
            t = perf()
            if where == 'local':
                h, self.cache[(a, b)] = stacks.run_stack(self.blocks[(a, b)], h, self.cache.get((a, b)))
                local_s += perf() - t
            else:
                name = f'{self.tag}s{a}-{b}'
                result, stats = self.nodes[where].forward([name], h, reset, self.session, self.wire, self.wire)
                h = result[name]
                stats['done_at_s'] = perf() - t0
                remote_stats.append(stats)
                wait_s += perf() - t
        t1 = perf()
        logits, _ = m.head([], h, None)
        t2 = perf()
        return logits, dict(total_s=t2 - t0, shared_s=local_s + (t1 - t0 - local_s - wait_s), local_s=0.0, wait_s=wait_s, fuse_s=t2 - t1,
                            remote=remote_stats)


def pipeline_segments(model, assignment):
    """[(where, first block, one past the last)] in block order; every block exactly once, each device one range."""
    segments = sorted((ids[0], ids[-1] + 1, where) for where, ids in assignment.items() if ids)
    covered = [i for a, b, _ in segments for i in range(a, b)]
    if covered != list(range(len(model.shared))) or any(sorted(ids) != list(range(ids[0], ids[-1] + 1)) for ids in assignment.values() if ids):
        raise ValueError(f'a layer split needs one contiguous range of blocks per device covering all {len(model.shared)} blocks')
    return [(where, a, b) for a, b, where in segments]


def parse_placement(text):
    """'label=local:0,1,2;phone:3-7' -> ('label', {'local': [0, 1, 2], 'phone': [3, 4, 5, 6, 7]})"""
    label, _, spec = text.partition('=')
    assignment = {}
    for part in spec.split(';'):
        where, _, ids = part.partition(':')
        numbers = []
        for v in ids.split(','):
            v = v.strip()
            if '-' in v:
                first, last = v.split('-')
                numbers += list(range(int(first), int(last) + 1))
            elif v != '':
                numbers.append(int(v))
        assignment[where.strip()] = numbers
    return label.strip(), assignment


def deploy(model: Model, placements, nodes, tag, mmap=False):
    """Make sure every remote branch is loaded on its node; returns upload/verify notes."""
    notes = []
    local_sha = {}
    extra = {'mmap': True} if mmap else {}     # workers keep the weights memory-mapped (low-memory experiments)
    if not model.branches:                    # sequential model, layer split: each remote range of blocks is one stack
        done = set()
        for _, assignment in placements:
            if set(assignment) == {'local'}:
                continue
            for name, a, b in pipeline_segments(model, assignment):
                if name == 'local' or (name, a, b) in done:
                    continue
                done.add((name, a, b))
                t = perf()
                before = nodes[name].uploaded
                part = torch.nn.ModuleList(list(model.shared[a:b]))
                reply = nodes[name].ensure_stack(f'{tag}s{a}-{b}', part.state_dict(), model.width, model.heads, b - a, **extra)
                sent = nodes[name].uploaded - before
                notes.append(f'{name}: blocks {a}-{b - 1} ' + ('already loaded' if reply.get('cached') and not sent else
                             f'uploaded {sent / 2**20:.0f} MiB, ready in {perf() - t:.1f}s'))
        return notes
    for _, assignment in placements:
        for name, branch_ids in assignment.items():
            if name in ('local', 'none'):
                continue
            for i in branch_ids:
                stack_name = f'{tag}{i}'
                branch = model.branches[i]
                blocks = len(branch)
                if model.branch_seeds:
                    reply = nodes[name].synth_stack(stack_name, model.branch_width, model.branch_heads, blocks, model.branch_seeds[i], model.depth)
                    if reply.get('wsha'):
                        if i not in local_sha:
                            local_sha[i] = hashlib.sha256(stacks.state_to_blob(branch.state_dict())).hexdigest()
                        if reply['wsha'] != local_sha[i]:
                            raise RuntimeError(f'{name}: synthetic weights of branch {i} differ from the coordinator')
                    notes.append(f'{name}: branch {i} synthetic, weights verified identical' + (' (cached)' if reply.get('cached') else ''))
                else:
                    t = perf()
                    before = nodes[name].uploaded
                    reply = nodes[name].ensure_stack(stack_name, branch.state_dict(), model.branch_width, model.branch_heads, blocks, **extra)
                    sent = nodes[name].uploaded - before
                    notes.append(f'{name}: branch {i} ' + ('already loaded' if reply.get('cached') and not sent else
                                 f'uploaded {sent / 2**20:.0f} MiB, ready in {perf() - t:.1f}s'))
    return notes


def aggregate(trials, prompt, decode):
    steps = [s for t in trials for s in t['steps']]
    row = dict(prompt=prompt, decode=decode,
               prefill_s=statistics.median(t['prefill']['total_s'] for t in trials),
               decode_s=statistics.median(sum(s['total_s'] for s in t['steps']) for t in trials))
    row['prefill_tps'] = prompt / row['prefill_s']
    row['decode_tps'] = decode / row['decode_s']
    row['total_s'] = row['prefill_s'] + row['decode_s']
    for key in ('total_s', 'shared_s', 'local_s', 'wait_s', 'fuse_s'):
        row['step_' + key[:-2] + '_ms'] = {k: v * 1000 for k, v in summarize([s[key] for s in steps]).items()}
    remote = {}
    for s in steps:
        for r in s['remote']:
            remote.setdefault(r['node'], []).append(r)
    row['remote'] = {}
    for name, items in remote.items():
        row['remote'][name] = {key[:-2] + '_ms': {k: v * 1000 for k, v in summarize([i[key] for i in items]).items()}
                               for key in ('rpc_s', 'srv_s', 'compute_s', 'net_s', 'queue_s', 'deser_s', 'ser_s')}
        row['remote'][name]['bytes_per_step'] = items[0]['up_bytes'] + items[0]['down_bytes']
    pre = trials[len(trials) // 2]['prefill']
    row['prefill_breakdown_ms'] = dict(shared=pre['shared_s'] * 1000, local=pre['local_s'] * 1000, wait=pre['wait_s'] * 1000,
                                       fuse=pre['fuse_s'] * 1000,
                                       remote={r['node']: dict(rpc=r['rpc_s'] * 1000, compute=r['compute_s'] * 1000,
                                                               net=r['net_s'] * 1000, mbytes=(r['up_bytes'] + r['down_bytes']) / 2**20)
                                               for r in pre['remote']})
    return row


def _say(text):
    print(text, flush=True)


def measure_resources(nodes, names):
    """Memory and storage counters of this process's control group and of the named nodes (empty where unsupported)."""
    out = {'local': resources.snapshot()}
    for name in names:
        try:
            out[name] = nodes[name].info().get('resources') or {}
        except Exception:  # noqa: BLE001
            out[name] = {}
    return out


def run_placements(model, placements, nodes, prompts, decodes, runs, wire='f4', tolerance=2e-3, tag='b', log=_say):
    """Time every placement of `model` (first one = reference, the others are teacher-forced against it and must
    reproduce its logits). Returns the result rows. Used by bench_cluster.py on the controller and by the
    coord_bench job when a node is the coordinator."""
    split = DistributedExecutor if model.branches else PipelineExecutor       # sequential model: layer split
    executors = [(label, LocalExecutor(model) if set(a) == {'local'} else split(model, a, nodes, wire, tag), a)
                 for label, a in placements]
    results = []
    try:
        for p in prompts:
            ids = (stacks.synthetic_tensor([1, p], f'prompt/{p}', 1.0).abs() * 1e6).long() % model.vocab
            for d in decodes:
                if p + d > model.context:
                    continue
                reference = None
                for label, exe, assignment in executors:
                    run_trial(exe, ids, min(2, d), None if reference is None else reference['tokens'])   # warm-up
                    if hasattr(exe, 'drain'):        # a delayed request still in flight would collide with the info call below
                        exe.drain()
                    used = [k for k, v in assignment.items() if v and k not in ('local', 'none')]
                    res0 = measure_resources(nodes, used)
                    trials = [run_trial(exe, ids, d, None if reference is None else reference['tokens']) for _ in range(runs)]
                    if hasattr(exe, 'drain'):
                        exe.drain()
                    res1 = measure_resources(nodes, used)
                    row = aggregate(trials, p, d)
                    row['resources'] = {k: dict(resources.delta(res0.get(k, {}), v), passes=runs * (d + 1)) for k, v in res1.items() if v}
                    row.update(placement=label, assignment=assignment,
                               devices=len([k for k, v in assignment.items() if v and k != 'none']))
                    if reference is None:
                        reference = trials[0]
                        row.update(parity='reference', max_abs_logit_diff=0.0, argmax_agreement=1.0)
                    else:
                        diff = max(float((t['logits'] - reference['logits']).abs().max()) for t in trials)
                        agree = statistics.fmean(float((t['logits'].argmax(-1) == reference['logits'].argmax(-1)).float().mean())
                                                 for t in trials)
                        if assignment.get('none'):       # branches left out on purpose: not the same function, report agreement only
                            row.update(parity=f'n/a, {len(assignment["none"])} branches missing, same top token {agree * 100:.0f}%',
                                       max_abs_logit_diff=diff, argmax_agreement=agree)
                        else:
                            row.update(parity='ok' if diff <= tolerance else 'FAIL', max_abs_logit_diff=diff, argmax_agreement=agree)
                    results.append(row)
                    waits = ' '.join(f"{n}:rpc {r['rpc_ms']['p50']:.1f} (compute {r['compute_ms']['p50']:.1f} net {r['net_ms']['p50']:.1f})"
                                     for n, r in row['remote'].items())
                    log(f"  {label:14s} prompt {p:>4} decode {d:>3}: prefill {row['prefill_tps']:7.1f} tok/s | decode {row['decode_tps']:6.2f} tok/s "
                          f"(step p50 {row['step_total_ms']['p50']:.1f} p95 {row['step_total_ms']['p95']:.1f} ms; shared {row['step_shared_ms']['p50']:.1f} "
                          f"local {row['step_local_ms']['p50']:.1f} wait {row['step_wait_ms']['p50']:.1f} fuse {row['step_fuse_ms']['p50']:.1f}) "
                          f"| parity {row['parity']} {row['max_abs_logit_diff']:.1e} | {waits}")
                    if row['parity'] == 'FAIL':
                        raise RuntimeError(f'{label}: logits differ from the reference by {row["max_abs_logit_diff"]:.3g}; stopping')
    finally:
        for _, exe, _ in executors:
            exe.close()
    return results


@torch.inference_mode()
def generate(exe, ids, max_new, temperature=0.8, seed=1, on_token=None, should_stop=None, eos=1, top_p=0.95):
    """Continue the token list `ids` with executor `exe` (sampling with temperature and top-p; temperature 0 = greedy).
    Calls on_token(tokens_so_far) after every token. Returns the new tokens and timing statistics."""
    def med(values):
        return statistics.median(values) * 1000 if values else None
    gen = torch.Generator().manual_seed(int(seed))
    t_start = perf()
    logits, stat = exe.step(torch.tensor([list(ids)]), reset=True)
    prefill_s = stat['total_s']
    out, steps, ended, stopped = [], [], False, False
    for _ in range(int(max_new)):
        if should_stop is not None and should_stop():
            stopped = True
            break
        if temperature <= 0:
            nxt = int(logits.argmax())
        else:
            probs = torch.softmax(logits.float() / temperature, -1)
            sp, si = probs.sort(descending=True)
            sp = sp * ((sp.cumsum(-1) - sp) < top_p)
            nxt = int(si[torch.multinomial(sp / sp.sum(), 1, generator=gen)])
        if nxt == eos:
            ended = True
            break
        out.append(nxt)
        logits, stat = exe.step(torch.tensor([[nxt]]), reset=False)
        steps.append(stat)
        if on_token is not None:
            on_token(out)
    timed = steps[2:] if len(steps) > 6 else steps          # the first steps warm the caches up
    remote = {}
    for s in timed:
        for r in s['remote']:
            remote.setdefault(r['node'], []).append(r)
    return dict(tokens=out, ended=ended, stopped=stopped, prefill_s=prefill_s, total_s=perf() - t_start,
                step_ms=med([s['total_s'] for s in timed]),
                decode_tps=len(timed) / sum(s['total_s'] for s in timed) if timed else None,
                breakdown_ms={k: med([s[k + '_s'] for s in timed]) for k in ('shared', 'local', 'wait', 'fuse')},
                remote={n: dict(round_trip_ms=med([r['rpc_s'] for r in v]), compute_ms=med([r['compute_s'] for r in v]),
                                network_ms=med([r['net_s'] for r in v])) for n, v in remote.items()})
