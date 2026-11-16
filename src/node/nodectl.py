#!/usr/bin/env python3
"""Control Branchy nodes from the coordinating machine.

  nodectl.py info                      status table of every node in cluster.json
  nodectl.py ping [--count 30]         round-trip / bandwidth statistics
  nodectl.py selftest                  numerical check of every node against this machine
  nodectl.py profile [--prefills ..]   time a synthetic branch stack on every node
  nodectl.py restart NODE --threads N --kernels MODE [--affinity 4,5,6,7]
  nodectl.py push                      push src/node/payload to every node (auto rollback on failure)
  nodectl.py calibrate --out DIR       E0: thread x kernel sweep (selftest + profile) + ping, saved to DIR
  nodectl.py log NODE | probe | job NODE MODULE [JSON] | evict NODE | shutdown NODE
Use --cluster FILE and --nodes a,b to select targets.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import statistics
import sys
import time
from pathlib import Path

import nodeclient
from nodeclient import NodeError, load_cluster, payload_tarball

SELFTESTS = [dict(width=256, heads=4, blocks=3, prefill=64, decode=8, seed='st-small'),
             dict(width=880, heads=10, blocks=2, prefill=128, decode=4, seed='st-880')]
TOLERANCE = 1e-3


def pct(values, q):
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(round(q * (len(ordered) - 1))))]


def each(nodes, fn):
    """Run fn(node) on all nodes in parallel; returns {name: result or Exception}."""
    out = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, len(nodes))) as pool:
        futures = {name: pool.submit(fn, node) for name, node in nodes.items()}
        for name, future in futures.items():
            try:
                out[name] = future.result()
            except Exception as exc:  # noqa: BLE001 - reported per node
                out[name] = exc
    return out


def cmd_info(nodes, args):
    results = each(nodes, lambda n: n.info())
    for name, info in results.items():
        if isinstance(info, Exception):
            print(f'{name:8s} DOWN  {info}')
            continue
        print(f"{name:8s} UP    v{info['version']} thr={info['threads']} kern={info['kernels']} torch={info['torch']} "
              f"py={info['python']} cpus={info['cpu_count']} aff={len(info['affinity'] or [])} "
              f"load={info['loadavg'][0] if info['loadavg'] else '?'} memfree={info.get('mem_available_mb')}MB "
              f"temp={info.get('thermal_max_c')} batt={info.get('battery_pct')} busy={info['busy']} "
              f"stacks={list(info['stacks'])} pending={info['pending_update']}")
        if args.verbose:
            print(json.dumps(info, indent=1))
    return {k: (str(v) if isinstance(v, Exception) else v) for k, v in results.items()}


def ping_node(node, count, sizes):
    rows = []
    for up, down in sizes:
        blob = b'\x5a' * up
        rpc, net = [], []
        for _ in range(count):
            reply, _, elapsed = node.call('ping', blob, reply_bytes=down)
            rpc.append(elapsed * 1000)
            net.append((elapsed - reply['srv_s']) * 1000)
        rows.append(dict(up_bytes=up, down_bytes=down, count=count, rtt_ms_min=min(rpc), rtt_ms_p50=pct(rpc, .5),
                         rtt_ms_p95=pct(rpc, .95), rtt_ms_max=max(rpc), rtt_ms_mean=statistics.fmean(rpc),
                         mbit_s=(up + down) * 8 / 1e6 / (pct(rpc, .5) / 1000) if up + down > 100000 else None))
    return rows


def cmd_ping(nodes, args):
    sizes = [(0, 0), (3520, 3520), (3520, 3 * 3520), (2703360, 0), (0, 2703360)]
    results = each(nodes, lambda n: ping_node(n, args.count, sizes)) if args.parallel else \
        {name: (lambda n: _safe(lambda: ping_node(n, args.count, sizes)))(node) for name, node in nodes.items()}
    for name, rows in results.items():
        if isinstance(rows, Exception):
            print(f'{name}: {rows}')
            continue
        for r in rows:
            rate = f" ~{r['mbit_s']:.0f} Mbit/s" if r['mbit_s'] else ''
            print(f"{name:8s} up={r['up_bytes']:>8} down={r['down_bytes']:>8}  min={r['rtt_ms_min']:.1f} p50={r['rtt_ms_p50']:.1f} "
                  f"p95={r['rtt_ms_p95']:.1f} max={r['rtt_ms_max']:.1f} ms{rate}")
    return {k: (str(v) if isinstance(v, Exception) else v) for k, v in results.items()}


def _safe(fn):
    try:
        return fn()
    except Exception as exc:  # noqa: BLE001
        return exc


_reference_cache = {}


def reference(test):
    """Selftest outputs computed in this process with native kernels, 1 thread."""
    key = json.dumps(test, sort_keys=True)
    if key not in _reference_cache:
        import torch
        import stacks
        threads = torch.get_num_threads()
        torch.set_num_threads(1)
        _reference_cache[key] = stacks.selftest_run(**test)
        torch.set_num_threads(threads)
    return _reference_cache[key]


def selftest_node(node, tests=SELFTESTS):
    import numpy as np
    rows = []
    for test in tests:
        ref_meta, ref = reference(test)
        reply, out, _ = node.call('selftest', timeout=600, **test)
        got = np.frombuffer(out, dtype='<f4').reshape(reply['shape'])
        diff = float(np.abs(got - ref.numpy()).max())
        ok = bool(reply['finite'] and diff < TOLERANCE and reply['repeat_max_abs'] < TOLERANCE
                  and reply['cache_vs_full_max_abs'] < TOLERANCE)
        rows.append(dict(test=test, ok=ok, max_abs_vs_reference=diff, bit_identical=reply['sha'] == ref_meta['sha'],
                         repeat_max_abs=reply['repeat_max_abs'], cache_vs_full_max_abs=reply['cache_vs_full_max_abs'],
                         out_abs_mean=reply['out_abs_mean'], run_s=reply['run_s']))
    return rows


def cmd_selftest(nodes, args):
    results = each(nodes, selftest_node)
    for name, rows in results.items():
        if isinstance(rows, Exception):
            print(f'{name}: ERROR {rows}')
            continue
        for r in rows:
            print(f"{name:8s} {'PASS' if r['ok'] else 'FAIL'} width={r['test']['width']:<4} vs_ref={r['max_abs_vs_reference']:.2e} "
                  f"repeat={r['repeat_max_abs']:.1e} cache_vs_full={r['cache_vs_full_max_abs']:.1e} "
                  f"bit_identical={r['bit_identical']} ({r['run_s']:.2f}s)")
    return {k: (str(v) if isinstance(v, Exception) else v) for k, v in results.items()}


def profile_node(node, args):
    reply = node.call('profile', timeout=1800, width=args.width, heads=args.heads, blocks=args.blocks,
                      prefills=[int(v) for v in args.prefills.split(',')], decode_steps=args.decode_steps)[0]
    return {k: reply.get(k) for k in ('width', 'heads', 'blocks', 'weight_mb', 'threads', 'kernels', 'linear', 'rows')}


def show_profile(name, prof):
    for r in prof['rows']:
        gbs = prof['weight_mb'] / 1024 / (r['decode_ms_median'] / 1000)
        print(f"{name:8s} thr={prof['threads']} {prof['kernels']:9s} {prof.get('linear') or 'gemm':4s} prefill {r['prefill']:>4}: {r['prefill_tps']:8.1f} tok/s | "
              f"decode median {r['decode_ms_median']:7.2f} ms (min {r['decode_ms_min']:.2f}, p90 {r['decode_ms_p90']:.2f}) "
              f"= {r['decode_ms_median'] / prof['blocks']:.2f} ms/block, weights read at ~{gbs:.1f} GiB/s")


def cmd_profile(nodes, args):
    results = each(nodes, lambda n: profile_node(n, args))
    for name, prof in results.items():
        if isinstance(prof, Exception):
            print(f'{name}: ERROR {prof}')
        else:
            show_profile(name, prof)
    return {k: (str(v) if isinstance(v, Exception) else v) for k, v in results.items()}


def cmd_restart(nodes, args):
    settings = {}
    if args.threads:
        settings['threads'] = args.threads
    if args.kernels:
        settings['kernels'] = args.kernels
    if args.linear:
        settings['linear'] = args.linear
    if args.env:
        settings['env'] = json.loads(args.env)
    if args.affinity is not None:
        settings['affinity'] = None if args.affinity in ('', 'none') else [int(v) for v in args.affinity.split(',')]
    for name, node in nodes.items():
        info = node.restart(**settings)
        print(f"{name}: restarted, threads={info['threads']} kernels={info['kernels']} linear={info.get('linear')} "
              f"affinity={info['affinity']} env={info['env']}")


def cmd_push(nodes, args):
    tarball, version = payload_tarball()
    print(f'payload version {version} ({len(tarball)} bytes)')
    results = each(nodes, lambda n: n.push_code(tarball, version))
    for name, result in results.items():
        print(f'{name}: {result}')
    return any(isinstance(r, Exception) for r in results.values())


def calibrate_node(node, args):
    """Thread x kernel sweep: restart, selftest, and (if numerically sound) profile."""
    info = node.info()
    original = dict(threads=info['threads'], kernels=info['kernels'], linear=info.get('linear', 'gemm'))
    cpus = info['cpu_count'] or 4
    grid = [t for t in (int(v) for v in args.threads_grid.split(',')) if t <= cpus]
    report = dict(info={k: info.get(k) for k in ('name', 'version', 'torch', 'numpy', 'python', 'platform', 'machine',
                                                 'cpu_count', 'cpu_model', 'cpu_max_mhz', 'android', 'mem_total_mb',
                                                 'affinity', 'native_rms_norm', 'parallel_info')}, runs=[])
    combos = [(k, l, t) for k in args.kernel_modes.split(',') for l in args.linear_modes.split(',') for t in grid]
    for kernels, linear, threads in combos:
        if True:
            run = dict(threads=threads, kernels=kernels, linear=linear)
            try:
                node.restart(threads=threads, kernels=kernels, linear=linear)
                run['selftest'] = selftest_node(node)
                run['numerics_ok'] = all(r['ok'] for r in run['selftest'])
                if run['numerics_ok']:
                    run['profile'] = profile_node(node, args)
                state = node.info()
                run['thermal_max_c'], run['loadavg'] = state.get('thermal_max_c'), state.get('loadavg')
            except NodeError as exc:
                run['error'] = str(exc)
            report['runs'].append(run)
            ms = run.get('profile', {}).get('rows', [{}])[0].get('decode_ms_median')
            print(f"  {node.name:8s} {kernels:9s} {linear:4s} thr={threads:<2} numerics={'ok' if run.get('numerics_ok') else 'BAD'} "
                  f"decode={ms if ms is None else round(ms, 2)} ms {run.get('error', '')}", flush=True)
    good = [r for r in report['runs'] if r.get('numerics_ok') and 'profile' in r]
    if good:
        best = min(good, key=lambda r: r['profile']['rows'][0]['decode_ms_median'])
        report['best'] = dict(threads=best['threads'], kernels=best['kernels'], linear=best['linear'],
                              decode_ms_median=best['profile']['rows'][0]['decode_ms_median'])
        node.restart(threads=best['threads'], kernels=best['kernels'], linear=best['linear'])
    else:
        node.restart(**original)
    return report


def cmd_calibrate(nodes, args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print('== ping (sequential, quiet network)')
    args.parallel = False
    pings = cmd_ping(nodes, args)
    print('== thread x kernel sweep (all nodes in parallel)')
    sweeps = each(nodes, lambda n: calibrate_node(n, args))
    report = dict(date=time.strftime('%Y-%m-%d %H:%M:%S'), stack=dict(width=args.width, heads=args.heads, blocks=args.blocks),
                  ping=pings, nodes={k: (dict(error=str(v)) if isinstance(v, Exception) else v) for k, v in sweeps.items()})
    (out / 'calibration.json').write_text(json.dumps(report, indent=1))
    lines = ['# E0 calibration', '', f"Date: {report['date']}. Stack profiled: {args.blocks} blocks, width {args.width}, "
             f"heads {args.heads} (one Branchy4 branch).", '', '## Network (controller to node)', '',
             '| Node | Up bytes | Down bytes | min ms | p50 ms | p95 ms | max ms | Mbit/s |', '|---|---:|---:|---:|---:|---:|---:|---:|']
    for name, rows in pings.items():
        for r in rows if isinstance(rows, list) else []:
            lines.append(f"| {name} | {r['up_bytes']} | {r['down_bytes']} | {r['rtt_ms_min']:.1f} | {r['rtt_ms_p50']:.1f} | "
                         f"{r['rtt_ms_p95']:.1f} | {r['rtt_ms_max']:.1f} | {'' if not r['mbit_s'] else round(r['mbit_s'])} |")
    lines += ['', '## Compute and numerics per thread count', '',
              '| Node | Kernels | Linear | Threads | Numerics | max abs vs x86 ref | Decode ms/token (median) | ms/block | Prefill tok/s (first length) |',
              '|---|---|---|---:|---|---:|---:|---:|---:|']
    for name, rep in report['nodes'].items():
        for run in rep.get('runs', []):
            worst = max((r['max_abs_vs_reference'] for r in run.get('selftest', [])), default=float('nan'))
            row = run.get('profile', {}).get('rows', [None])[0]
            lines.append(f"| {name} | {run['kernels']} | {run['linear']} | {run['threads']} | {'ok' if run.get('numerics_ok') else 'FAIL'} | {worst:.1e} | "
                         + (f"{row['decode_ms_median']:.2f} | {row['decode_ms_median'] / args.blocks:.2f} | {row['prefill_tps']:.1f} |"
                            if row else ' | | |'))
        if 'best' in rep:
            lines.append(f"| **{name} best** | {rep['best']['kernels']} | {rep['best']['linear']} | {rep['best']['threads']} | ok | | "
                         f"**{rep['best']['decode_ms_median']:.2f}** | | |")
    (out / 'calibration.md').write_text('\n'.join(lines) + '\n')
    print(f"saved {out / 'calibration.json'} and calibration.md")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--cluster', default=None)
    ap.add_argument('--nodes', default=None, help='comma separated subset')
    ap.add_argument('--out', default=None, help='write JSON result here (file, or folder for calibrate)')
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('info'); p.add_argument('-v', '--verbose', action='store_true')
    p = sub.add_parser('ping'); p.add_argument('--count', type=int, default=30); p.add_argument('--parallel', action='store_true')
    sub.add_parser('selftest')
    for name in ('profile', 'calibrate'):
        p = sub.add_parser(name)
        p.add_argument('--width', type=int, default=880); p.add_argument('--heads', type=int, default=10)
        p.add_argument('--blocks', type=int, default=7); p.add_argument('--prefills', default='32,256')
        p.add_argument('--decode-steps', type=int, default=24)
        if name == 'calibrate':
            p.add_argument('--kernel-modes', default='math-attn,native'); p.add_argument('--count', type=int, default=30)
            p.add_argument('--linear-modes', default='gemm,mv'); p.add_argument('--threads-grid', default='1,2,4,6,8,12')
    p = sub.add_parser('restart'); p.add_argument('--threads', type=int); p.add_argument('--kernels'); p.add_argument('--affinity')
    p.add_argument('--linear', choices=['gemm', 'mv']); p.add_argument('--env', help='JSON dict of extra environment variables')
    sub.add_parser('push')
    p = sub.add_parser('log'); p.add_argument('--which', default='node'); p.add_argument('--bytes', type=int, default=6000)
    sub.add_parser('probe')
    p = sub.add_parser('job'); p.add_argument('module'); p.add_argument('json', nargs='?', default='{}')
    sub.add_parser('evict'); sub.add_parser('shutdown'); sub.add_parser('wait')
    p = sub.add_parser('sh'); p.add_argument('command'); p.add_argument('--timeout', type=float, default=60)
    p = sub.add_parser('stage0'); p.add_argument('--no-tmux', action='store_true')
    args = ap.parse_args()
    nodes = load_cluster(args.cluster, only=args.nodes.split(',') if args.nodes else None)
    if not nodes:
        ap.error('no matching nodes in the cluster file')
    result = None
    if args.cmd == 'info':
        result = cmd_info(nodes, args)
    elif args.cmd == 'ping':
        result = cmd_ping(nodes, args)
    elif args.cmd == 'selftest':
        result = cmd_selftest(nodes, args)
    elif args.cmd == 'profile':
        result = cmd_profile(nodes, args)
    elif args.cmd == 'restart':
        cmd_restart(nodes, args)
    elif args.cmd == 'push':
        sys.exit(1 if cmd_push(nodes, args) else 0)
    elif args.cmd == 'calibrate':
        cmd_calibrate(nodes, args)
        return
    elif args.cmd == 'log':
        for name, node in nodes.items():
            print(f'--- {name}'); print(bytes(node.call('get_log', which=args.which, bytes=args.bytes)[1]).decode('utf-8', 'replace'))
    elif args.cmd == 'probe':
        spec = json.loads(Path(args.cluster or nodeclient.HERE / 'cluster.json').read_text())
        targets = [[e['host'], e.get('port', 8730)] for e in spec['nodes'].values()] + \
                  [[e['lan'], e.get('port', 8730)] for e in spec['nodes'].values() if e.get('lan')]
        result = each(nodes, lambda n: n.call('peers_probe', targets=targets, count=8)[0]['peers'])
        for name, peers in result.items():
            print(name, json.dumps(peers) if not isinstance(peers, Exception) else peers)
        result = {k: str(v) if isinstance(v, Exception) else v for k, v in result.items()}
    elif args.cmd == 'job':
        result = {}
        for name, node in nodes.items():
            result[name] = node.run_job(args.module, json.loads(args.json))
            print(name, json.dumps(result[name]))
    elif args.cmd == 'sh':
        result = each(nodes, lambda n: n.run_job('sh', dict(cmd=args.command, timeout=args.timeout), timeout=args.timeout + 10))
        for name, r in result.items():
            if isinstance(r, Exception):
                print(f'--- {name}: {r}')
            else:
                print(f"--- {name} (rc={r['rc']})\n{r['out']}{r['err']}", end='')
        result = {k: str(v) if isinstance(v, Exception) else v for k, v in result.items()}
    elif args.cmd == 'stage0':
        files = {n: (nodeclient.HERE / 'stage0' / n).read_text() for n in ('start.sh', 'stop.sh', 'supervisor.py')}

        def relaunch(node):
            node.call('job_start', json.dumps(dict(files=files, env={'BRANCHY_NO_TMUX': '1'} if args.no_tmux else {})).encode(),
                      module='stage0_update')
            node.close()
            time.sleep(6)
            info = node.wait_up(120)
            mode = node.run_job('sh', dict(cmd='tmux has-session -t "=$(cat .tmux_session)" 2>/dev/null && echo tmux:$(cat .tmux_session) || echo background'))
            return f"up, version {info['version']}, running in {mode['out'].strip()}"
        for name, r in each(nodes, relaunch).items():
            print(f'{name}: {r}')
    elif args.cmd == 'evict':
        for name, node in nodes.items():
            print(name, node.call('unload', name='all')[0], node.call('blob_evict', sha='all')[0])
    elif args.cmd == 'shutdown':
        for name, node in nodes.items():
            print(name, node.call('shutdown')[0])
    elif args.cmd == 'wait':
        for name, node in nodes.items():
            print(name, 'up, version', node.wait_up(600)['version'])
    if args.out and result is not None:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(result, indent=1, default=str))


if __name__ == '__main__':
    main()
