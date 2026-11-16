#!/usr/bin/env python3
"""Branchy coordinator web UI: compare ways of running a model over your devices from a browser.

This machine is the coordinator. It loads the models, talks to the worker nodes of cluster.json and serves one page:

  Cluster     which workers answer, their settings and round-trip time
  Compare     write text with several configurations on the same beginning and see speed and text side by side
              (sequential model here, its layer split over workers, the branched model with branches on workers)
  Benchmark   time a list of placements with a parity check against the first one (same loop as bench_cluster.py)

  python src/webui/webui.py --models models --cluster src/node/cluster.json --port 8780 --threads 8

--models DIR is searched for *.pt inference files (and tokenizer.json). One job runs at a time. The page has no login:
it binds to 127.0.0.1 unless you pass --bind, so open it only to networks you trust.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'src' / 'node'))

import torch  # noqa: E402

from nodeclient import load_cluster  # noqa: E402

import coordinator  # noqa: E402
import stacks  # noqa: E402
from cluster_model import LocalExecutor, Model  # noqa: E402
from coordinator import DistributedExecutor, PipelineExecutor  # noqa: E402

ARGS = None
MODELS: dict = {}          # name -> dict(path, kind, label, blocks, branches, delay, parameters)
loaded: dict = {}
tokenizer = None
lock = threading.Lock()
job = dict(id=0, state='idle', kind=None)
BUSY = ('loading', 'deploying', 'running')
MAX_NEW, MAX_PROMPT = 400, 400


# ------------------------------------------------------------------------------------------------ models and cluster
def discover_models(folder: Path):
    for path in sorted(folder.rglob('*.pt')):
        try:
            ck = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
            cfg = ck['config']
        except Exception:  # noqa: BLE001
            continue
        branched = cfg.get('kind') != 'sequential' and 'n_branches' in cfg
        name = path.parent.name if path.name == 'inference.pt' else path.stem
        MODELS[name] = dict(path=str(path), kind='branched' if branched else 'sequential',
                            blocks=int(cfg.get('seq_blocks', 0)) if not branched else int(cfg.get('branch_blocks', 0)),
                            branches=int(cfg['n_branches']) if branched else 0, delay=int(cfg.get('delay', 0)) if branched else 0,
                            width=int(cfg['width']), parameters=sum(v.numel() for v in ck['model'].values()),
                            label=(f"{cfg['n_branches']} branches" + (', one-token delay' if cfg.get('delay') else '')) if branched
                            else f"sequential, {cfg.get('seq_blocks')} blocks")
        del ck


def get_model(name):
    if name not in loaded:
        loaded[name] = Model.from_checkpoint(MODELS[name]['path'])
    return loaded[name]


def get_tokenizer():
    global tokenizer
    if tokenizer is None:
        from tokenizers import Tokenizer
        tokenizer = Tokenizer.from_file(str(ARGS.tokenizer))
    return tokenizer


def cluster_spec():
    path = Path(ARGS.cluster)
    return json.loads(path.read_text())['nodes'] if path.exists() else {}


def node_status(name):
    try:
        node = load_cluster(ARGS.cluster, only=[name])[name]
        node.connect(timeout=4)
        info = node.info()
        times = []
        for _ in range(5):
            times.append(node.call('ping', b'x' * 4096, reply_bytes=4096)[2] * 1000)
        node.close()
        return dict(name=name, up=True, round_trip_ms=sorted(times)[len(times) // 2], version=info.get('version'), torch=info.get('torch'),
                    threads=info.get('threads'), kernels=info.get('kernels'), linear=info.get('linear'),
                    cpu=info.get('android') or info.get('cpu_model'), mem_available_mb=info.get('mem_available_mb'),
                    stacks=len(info.get('stacks') or {}), busy=info.get('busy'))
    except Exception as exc:  # noqa: BLE001
        return dict(name=name, up=False, error=f'{type(exc).__name__}: {exc}'[:200])


def nodes_status():
    names = list(cluster_spec())
    if not names:
        return []
    with ThreadPoolExecutor(max_workers=len(names)) as pool:
        return list(pool.map(node_status, names))


# ------------------------------------------------------------------------------------------------ placements
def assignment_for(model, cfg):
    """A configuration of the page -> {'local': [...], worker: [...], 'none': [...]} (block or branch numbers).

    Sequential model: 'layers' = [[device, number of blocks], ...] in order; missing blocks go to the coordinator first.
    Branched model: 'workers' = {worker: number of branches}; branch 0 always stays on the coordinator (never delayed);
    remote branches are numbered from the top; 'leave_out' branches are computed by nobody."""
    info = MODELS[cfg['model']]
    if info['kind'] == 'sequential':
        total = len(model.shared)
        parts = [(str(d), int(n)) for d, n in (cfg.get('layers') or []) if int(n) > 0 and d != 'local']
        remote = sum(n for _, n in parts)
        if remote >= total:
            raise ValueError(f'{remote} blocks on workers: the coordinator must keep at least one of the {total}')
        out, start = {'local': list(range(total - remote))}, total - remote
        for device, n in parts:
            if device in out:
                raise ValueError('each worker can hold one range of blocks')
            out[device] = list(range(start, start + n))
            start += n
        return out
    n = info['branches']
    counts = {str(k): int(v) for k, v in (cfg.get('workers') or {}).items() if int(v) > 0}
    leave_out = int(cfg.get('leave_out') or 0)
    remote = sum(counts.values())
    if remote + leave_out > n - 1:
        raise ValueError(f'{remote} branches on workers plus {leave_out} left out is more than the {n - 1} that can leave the coordinator')
    top, out = n, {}
    for name in sorted(counts):
        out[name] = list(range(top - counts[name], top))
        top -= counts[name]
    local = n - remote - leave_out
    out['local'] = list(range(local))
    if leave_out:
        out['none'] = list(range(local, local + leave_out))
    return out


def describe(assignment, kind):
    unit = 'blocks' if kind == 'sequential' else 'branches'
    parts = [f"{'this machine' if k == 'local' else 'left out' if k == 'none' else k}: {len(v)}" for k, v in assignment.items() if v]
    return f"{unit} — " + ', '.join(parts)


def tag_for(name):
    return 'ui' + hashlib.sha256(str(Path(MODELS[name]['path']).resolve()).encode()).hexdigest()[:6] + '-'


def connect(used, me):
    nodes = load_cluster(ARGS.cluster, only=used) if used else {}
    unknown = [k for k in used if k not in nodes]
    if unknown:
        raise ValueError(f'unknown workers: {unknown}')
    for worker in used:
        me['status_text'] = f'connecting to {worker}'
        try:
            nodes[worker].connect(timeout=6)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f'{worker} does not answer ({exc}). Is its node running?') from exc
    return nodes


def executor_for(model, name, assignment, nodes, me):
    used = [k for k in assignment if k not in ('local', 'none') and assignment[k]]
    tag = tag_for(name)
    if used:
        me.update(state='deploying', status_text='sending weights to the workers (only the first time)')
        coordinator.deploy(model, [('ui', assignment)], nodes, tag)
        return (DistributedExecutor if model.branches else PipelineExecutor)(model, assignment, nodes, 'f4', tag)
    if model.branches and assignment.get('none'):
        return DistributedExecutor(model, assignment, {}, 'f4', tag)
    return LocalExecutor(model)


# ------------------------------------------------------------------------------------------------ jobs
def run_compare(params, me):
    """Every configuration writes from the same beginning with the same seed, one after the other."""
    tok = get_tokenizer()
    ids = tok.encode(params['prompt']).ids[:MAX_PROMPT]
    if not ids:
        raise ValueError('the beginning is empty')
    max_new = max(1, min(MAX_NEW, int(params.get('max_tokens', 120))))
    me['prompt_tokens'] = len(ids)
    for slot, cfg in zip(me['results'], params['configs']):
        if me.get('stop'):
            break
        exe, nodes = None, {}
        try:
            slot.update(state='loading')
            me.update(state='loading', status_text=f"{slot['label']}: loading the model")
            model = get_model(cfg['model'])
            assignment = assignment_for(model, cfg)
            slot.update(assignment=assignment, placement=describe(assignment, MODELS[cfg['model']]['kind']))
            used = [k for k in assignment if k not in ('local', 'none') and assignment[k]]
            nodes = connect(used, me)
            exe = executor_for(model, cfg['model'], assignment, nodes, me)
            me.update(state='running', status_text=f"{slot['label']}: writing")
            slot.update(state='running')
            t_first = time.perf_counter()

            def on_token(out, slot=slot, t_first=t_first):
                wall = time.perf_counter() - t_first
                slot.update(text=tok.decode(out), tokens=len(out), wall_tps=len(out) / wall if wall > 0 else None)
            stats = coordinator.generate(exe, ids, min(max_new, model.context - len(ids)), float(params.get('temperature', 0.8)),
                                         int(params.get('seed', 1)), on_token=on_token, should_stop=lambda: me.get('stop'))
            tokens = stats.pop('tokens')
            slot.update(stats, state='stopped' if stats['stopped'] else 'done', text=tok.decode(tokens), tokens=len(tokens))
        except Exception as exc:  # noqa: BLE001
            slot.update(state='error', error=f'{type(exc).__name__}: {exc}')
        finally:
            try:
                if exe is not None:
                    exe.close()
            except Exception:  # noqa: BLE001
                pass
            for node in nodes.values():
                node.close()


def run_benchmark(params, me):
    name = params['model']
    me.update(state='loading', status_text='loading the model')
    model = get_model(name)
    placements = [coordinator.parse_placement(line) for line in params['placements'] if line.strip()]
    if not placements:
        raise ValueError('no placements')
    if set(placements[0][1]) - {'local'}:
        raise ValueError('the first placement is the reference and must run on this machine only (local:...)')
    used = sorted({k for _, a in placements for k in a if k not in ('local', 'none') and a[k]})
    nodes = connect(used, me)
    try:
        tag = tag_for(name)
        me.update(state='deploying', status_text='sending weights to the workers (only the first time)')
        for note in coordinator.deploy(model, placements, nodes, tag):
            me['log'].append(note)

        def log(text):
            me['log'].append(text.strip())
            me['status_text'] = text.strip()[:140]
        me.update(state='running', status_text='measuring')
        rows = coordinator.run_placements(model, placements, nodes, [int(params.get('prompt_tokens', 32))], [int(params.get('decode', 48))],
                                          int(params.get('runs', 2)), 'f4', 2e-3, tag, log)
        me['rows'] = [dict(placement=r['placement'], devices=r['devices'], decode_tps=r['decode_tps'], prefill_tps=r['prefill_tps'],
                           step_p50_ms=r['step_total_ms']['p50'], step_p95_ms=r['step_total_ms']['p95'], wait_ms=r['step_wait_ms']['p50'],
                           parity=r['parity'], max_abs_logit_diff=r['max_abs_logit_diff'],
                           remote={n: dict(round_trip_ms=v['rpc_ms']['p50'], compute_ms=v['compute_ms']['p50'], network_ms=v['net_ms']['p50'])
                                   for n, v in r['remote'].items()}) for r in rows]
    finally:
        for node in nodes.values():
            node.close()


def run(kind, params, me):
    try:
        torch.set_num_threads(ARGS.threads)
        torch.set_grad_enabled(False)
        stacks.set_linear_mode(ARGS.linear)
        (run_compare if kind == 'compare' else run_benchmark)(params, me)
        if me['state'] in BUSY:
            me.update(state='stopped' if me.get('stop') else 'done', status_text='')
    except Exception as exc:  # noqa: BLE001
        me.update(state='error', error=f'{type(exc).__name__}: {exc}', status_text='')
    finally:
        me['finished'] = time.time()


# ------------------------------------------------------------------------------------------------ HTTP
class Handler(BaseHTTPRequestHandler):
    server_version = 'BranchyWebUI/1'

    def log_message(self, *args):
        pass

    def reply(self, obj, code=200, ctype='application/json'):
        data = obj if isinstance(obj, bytes) else json.dumps(obj).encode()
        self.send_response(code)
        self.send_header('Content-Type', ctype)
        self.send_header('Content-Length', str(len(data)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        path = self.path.split('?')[0]
        if path in ('/', '/index.html'):
            return self.reply((HERE / 'index.html').read_bytes(), ctype='text/html; charset=utf-8')
        if path == '/api/info':
            return self.reply(dict(models=MODELS, workers=list(cluster_spec()), threads=ARGS.threads, linear=ARGS.linear,
                                   torch=torch.__version__, max_new=MAX_NEW))
        if path == '/api/nodes':
            return self.reply(nodes_status())
        if path == '/api/job':
            return self.reply({k: v for k, v in job.items() if k != 'stop'})
        self.reply(dict(error='not found'), 404)

    def do_POST(self):
        global job
        path = self.path.split('?')[0]
        body = self.rfile.read(min(200000, int(self.headers.get('Content-Length') or 0)))
        if path == '/api/stop':
            job['stop'] = True
            return self.reply(dict(ok=True))
        if path not in ('/api/compare', '/api/benchmark'):
            return self.reply(dict(error='not found'), 404)
        kind = path.rsplit('/', 1)[1]
        try:
            params = json.loads(body or b'{}')
            if kind == 'compare':
                configs = params.get('configs') or []
                if not configs or len(configs) > 6:
                    raise ValueError('give 1 to 6 configurations')
                if any(c.get('model') not in MODELS for c in configs):
                    raise ValueError('unknown model in a configuration')
                if not str(params.get('prompt', '')).strip():
                    raise ValueError('the beginning is empty')
            elif params.get('model') not in MODELS:
                raise ValueError('unknown model')
        except (ValueError, TypeError) as exc:
            return self.reply(dict(error=str(exc)), 400)
        with lock:
            if job.get('state') in BUSY:
                return self.reply(dict(error='another job is still running'), 409)
            job = dict(id=job['id'] + 1, kind=kind, state='loading', status_text='starting', started=time.time(), log=[])
            if kind == 'compare':
                job.update(prompt=params['prompt'], results=[dict(label=str(c.get('label') or f'configuration {i + 1}')[:60], model=c['model'],
                                                                  state='waiting', text='', tokens=0) for i, c in enumerate(params['configs'])])
            else:
                job.update(model=params['model'], rows=[])
            threading.Thread(target=run, args=(kind, params, job), daemon=True).start()
        return self.reply(dict(ok=True, id=job['id']))


def main():
    global ARGS
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--models', type=Path, default=ROOT / 'models', help='folder with inference files (*.pt) and tokenizer.json')
    ap.add_argument('--tokenizer', type=Path, default=None, help='tokenizer.json (default: the one in --models)')
    ap.add_argument('--cluster', default=str(ROOT / 'src' / 'node' / 'cluster.json'))
    ap.add_argument('--bind', default='127.0.0.1')
    ap.add_argument('--port', type=int, default=8780)
    ap.add_argument('--threads', type=int, default=8)
    ap.add_argument('--linear', default='gemm', choices=['gemm', 'mv'], help="'mv' is faster on torch builds that use OpenBLAS")
    ARGS = ap.parse_args()
    if ARGS.tokenizer is None:
        ARGS.tokenizer = next(iter(sorted(ARGS.models.rglob('tokenizer.json'))), ARGS.models / 'tokenizer.json')
    discover_models(ARGS.models)
    if not MODELS:
        sys.exit(f'no inference files (*.pt) found under {ARGS.models}')
    print(f'Branchy coordinator UI on http://{ARGS.bind}:{ARGS.port}  models: {", ".join(MODELS)}  workers: {", ".join(cluster_spec()) or "none"}', flush=True)
    ThreadingHTTPServer((ARGS.bind, ARGS.port), Handler).serve_forever()


if __name__ == '__main__':
    main()
