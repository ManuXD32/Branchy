#!/usr/bin/env python3
"""Branchy node daemon: a remotely controlled block-stack executor.

One process per device, started by stage0/supervisor.py. Network handlers run in
threads; ALL torch work runs on the main thread (one OpenMP team, the mode that
was validated on Termux). Thread count and kernel mode are fixed per process; the
`restart` op rewrites node_state.json and exits so the supervisor relaunches.

Ops (see docs/NODE.md): info ping confirm get_log peers_probe restart shutdown
update_code blob_stat blob_put blob_list blob_evict job_start job_status job_kill
load_stack synth_stack unload forward truncate drop_session selftest profile
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import logging
import logging.handlers
import os
import platform
import queue
import re
import shutil
import socket
import socketserver
import subprocess
import sys
import tarfile
import threading
import time
import uuid
from pathlib import Path

EXIT_RESTART = 75
EXIT_SHUTDOWN = 76
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
log = logging.getLogger('node')
perf = time.perf_counter
SAFE_NAME = re.compile(r'^[A-Za-z0-9_][A-Za-z0-9_.-]{0,63}$')
HEX64 = re.compile(r'^[0-9a-f]{64}$')


def read_text(path, default=None):
    try:
        return Path(path).read_text().strip()
    except OSError:
        return default


def atomic_write(path: Path, text: str):
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(text)
    os.replace(tmp, path)


class Future:
    def __init__(self):
        self.event = threading.Event()
        self.result = None
        self.error = None
        self.queued = perf()
        self.queue_s = 0.0


class Node:
    def __init__(self, home: Path):
        self.home = home
        self.payload_root = home / 'payload'
        self.version = HERE.name
        self.state_path = home / 'node_state.json'
        self.state = json.loads(self.state_path.read_text())
        self.name = str(self.state.get('name') or platform.node() or 'node')
        self.token = (home / 'token').read_text().strip().encode()
        if len(self.token) < 16:
            raise SystemExit('token file missing or too short')
        self.blob_dir = home / 'cache' / 'blobs'
        self.blob_dir.mkdir(parents=True, exist_ok=True)
        self.job_dir = home / 'jobs'
        self.job_dir.mkdir(exist_ok=True)
        self.queue: queue.Queue = queue.Queue()
        self.stacks = {}
        self.sessions = {}
        self.jobs = {}
        self.busy = None
        self.exit_code = None
        self.started = time.time()
        self.blob_lock = threading.Lock()
        self.static = {}
        self.ping_noise = os.urandom(1 << 20)
        self.activity = {}
        self.activity_lock = threading.Lock()

    # ------------------------------------------------------------------ setup
    def setup_torch(self):
        affinity = self.state.get('affinity')
        if affinity and hasattr(os, 'sched_setaffinity'):
            try:
                os.sched_setaffinity(0, set(int(c) for c in affinity))
            except OSError as exc:
                log.warning('affinity %s refused: %s', affinity, exc)
        import numpy
        import torch
        import cpu_kernels
        threads = int(self.state.get('threads', 4))
        torch.set_num_threads(threads)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass
        torch.set_grad_enabled(False)
        kernels = cpu_kernels.install_kernels(self.state.get('kernels', 'native'))
        import stacks
        linear = stacks.set_linear_mode(self.state.get('linear', 'gemm'))
        freqs = []
        for i in range(os.cpu_count() or 1):
            f = read_text(f'/sys/devices/system/cpu/cpu{i}/cpufreq/cpuinfo_max_freq')
            freqs.append(int(f) // 1000 if f and f.isdigit() else None)
        model = None
        for line in (read_text('/proc/cpuinfo', '') or '').splitlines():
            if line.lower().startswith(('model name', 'hardware')):
                model = line.split(':', 1)[1].strip()
                break
        android = None
        if shutil.which('getprop'):
            try:
                android = ' '.join(subprocess.run(['getprop', k], capture_output=True, text=True, timeout=3).stdout.strip()
                                   for k in ('ro.product.manufacturer', 'ro.product.model', 'ro.soc.model',
                                             'ro.build.version.release'))
            except Exception:
                pass
        mem_total = None
        for line in (read_text('/proc/meminfo', '') or '').splitlines():
            if line.startswith('MemTotal'):
                mem_total = int(line.split()[1]) // 1024
        self.static = dict(
            name=self.name, version=self.version, python=platform.python_version(), torch=torch.__version__,
            numpy=numpy.__version__, platform=platform.platform(), machine=platform.machine(),
            cpu_count=os.cpu_count(), cpu_model=model, cpu_max_mhz=freqs, android=android, mem_total_mb=mem_total,
            affinity=sorted(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else None,
            threads=torch.get_num_threads(), kernels=kernels, linear=linear, native_rms_norm=hasattr(torch.nn.functional, 'rms_norm'),
            parallel_info=torch.__config__.parallel_info()[:1200], pid=os.getpid(),
            env={k: os.environ.get(k) for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')})
        log.info('node %s v%s torch %s threads=%s kernels=%s linear=%s', self.name, self.version, torch.__version__,
                 self.static['threads'], kernels, linear)

    # ------------------------------------------------------------------ dispatch
    def submit(self, fn, *args):
        fut = Future()
        self.queue.put((fn, args, fut))
        fut.event.wait()
        if fut.error is not None:
            raise fut.error
        return fut

    def dispatch(self, op, hdr, payload):
        if op in CONTROL:
            return CONTROL[op](self, hdr, payload)
        if op in COMPUTE:
            fut = self.submit(COMPUTE[op], self, hdr, payload)
            reply, out = fut.result
            reply['queue_s'] = fut.queue_s
            return reply, out
        raise ValueError(f'unknown op {op!r}')

    def compute_loop(self):
        import torch
        while True:
            item = self.queue.get()
            if item is None:
                break
            fn, args, fut = item
            fut.queue_s = perf() - fut.queued
            self.busy = (fn.__name__, time.time())
            try:
                with torch.inference_mode():
                    fut.result = fn(*args)
            except BaseException as exc:  # reported to the caller, daemon keeps running
                log.exception('compute op failed')
                fut.error = exc
            self.busy = None
            fut.event.set()
        log.info('exiting with code %s', self.exit_code)
        logging.shutdown()
        os._exit(self.exit_code if self.exit_code is not None else 1)

    # ------------------------------------------------------------------ human-readable activity log
    def note(self, op, hdr, reply, n_in, n_out, seconds, peer):
        """High-rate ops are summarized every few seconds; everything else gets its own line."""
        if op in QUIET and reply.get('ok'):
            with self.activity_lock:
                a = self.activity.setdefault(op, dict(n=0, s=0.0, bytes=0, compute=0.0, tokens=0, cached=0, peer=peer))
                a['n'] += 1
                a['s'] += seconds
                a['bytes'] += n_in + n_out
                if op == 'forward':
                    a['compute'] += sum(reply.get('compute_s', []))
                    a['tokens'] += int(hdr.get('shape', [0, 0])[1])
                    a['cached'] = reply.get('cached_len', 0)
                    a['stacks'] = ','.join(hdr.get('stacks', []))
            return
        detail = ' '.join(f'{k}={hdr[k]}' for k in ('name', 'stacks', 'width', 'heads', 'blocks', 'prefill', 'prefills', 'threads',
                                                    'kernels', 'linear', 'affinity', 'module', 'job', 'session', 'sha')
                          if k in hdr)
        outcome = 'ok' if reply.get('ok') else 'FAILED: ' + str(reply.get('error'))
        log.info('%-12s %s -> %s (%.2fs, from %s)', op, detail[:160], outcome, seconds, peer)

    def activity_loop(self, period=5.0):
        while True:
            time.sleep(period)
            self.flush_activity(period)

    def flush_activity(self, period=5.0):
        if True:
            with self.activity_lock:
                snapshot, self.activity = self.activity, {}
            for op, a in snapshot.items():
                if op == 'forward':
                    log.info('forward      %d calls, %d tokens in %.0fs: compute %.1f ms/call, %.1f KB/call, stacks [%s], context now %d (from %s)',
                             a['n'], a['tokens'], period, 1000 * a['compute'] / a['n'], a['bytes'] / a['n'] / 1024,
                             a.get('stacks', ''), a['cached'], a['peer'])
                elif op in ('ping', 'blob_put'):
                    log.info('%-12s %d calls, %.2f MB total (from %s)', op, a['n'], a['bytes'] / 2**20, a['peer'])

    def request_exit(self, code):
        self.flush_activity()
        self.exit_code = code
        threading.Timer(0.4, lambda: self.queue.put(None)).start()

    # ------------------------------------------------------------------ control ops
    def op_info(self, hdr, payload):
        dyn = dict(time=time.time(), uptime_s=time.time() - self.started,
                   loadavg=os.getloadavg() if hasattr(os, 'getloadavg') else None,
                   busy=self.busy, pending_update=read_text(self.payload_root / 'PENDING'),
                   state=self.state,
                   stacks={k: v['meta'] for k, v in self.stacks.items()}, sessions=sorted(self.sessions),
                   jobs={k: self._job_brief(v) for k, v in self.jobs.items()})
        for line in (read_text('/proc/meminfo', '') or '').splitlines():
            if line.startswith('MemAvailable'):
                dyn['mem_available_mb'] = int(line.split()[1]) // 1024
        temps = []
        for zone in sorted(Path('/sys/class/thermal').glob('thermal_zone*/temp'))[:64]:
            value = read_text(zone)
            if value and value.lstrip('-').isdigit() and 1000 < int(value) < 150000:
                temps.append(int(value) / 1000)
        dyn['thermal_max_c'] = max(temps) if temps else None
        for supply in Path('/sys/class/power_supply').glob('*/capacity'):
            dyn['battery_pct'] = read_text(supply)
            dyn['battery_status'] = read_text(supply.with_name('status'))
            break
        blobs = list(self.blob_dir.glob('*.bin'))
        dyn['blobs'] = dict(count=len(blobs), mb=sum(b.stat().st_size for b in blobs) // 2**20)
        import resources
        dyn['resources'] = resources.snapshot()
        return dict(self.static, **dyn), b''

    def op_ping(self, hdr, payload):
        want = min(int(hdr.get('reply_bytes', 0)), 64 << 20)
        noise = self.ping_noise
        return dict(got=len(payload)), (noise * (want // len(noise) + 1))[:want] if want else b''

    def op_confirm(self, hdr, payload):
        pending = self.payload_root / 'PENDING'
        had = pending.exists()
        if had:
            pending.unlink()
            log.info('update to %s confirmed by controller', self.version)
        return dict(confirmed=had, version=self.version), b''

    def op_get_log(self, hdr, payload):
        which = {'node': 'node.log', 'daemon': 'daemon.out', 'supervisor': 'supervisor.out'}[hdr.get('which', 'node')]
        path = self.home / 'logs' / which
        size = min(int(hdr.get('bytes', 20000)), 1 << 20)
        try:
            with path.open('rb') as f:
                f.seek(0, 2)
                f.seek(max(0, f.tell() - size))
                return dict(file=which), f.read()
        except OSError as exc:
            return dict(file=which, missing=str(exc)), b''

    def op_peers_probe(self, hdr, payload):
        out = {}
        for host, port in hdr['targets'][:16]:
            times = []
            err = None
            for _ in range(min(int(hdr.get('count', 5)), 50)):
                t = perf()
                try:
                    socket.create_connection((host, int(port)), timeout=float(hdr.get('timeout', 3))).close()
                    times.append(round((perf() - t) * 1000, 3))
                except OSError as exc:
                    err = str(exc)
                    break
            out[f'{host}:{port}'] = dict(connect_ms=times, error=err)
        return dict(peers=out), b''

    def op_restart(self, hdr, payload):
        new = dict(self.state)
        if 'threads' in hdr:
            threads = int(hdr['threads'])
            if not 1 <= threads <= 64:
                raise ValueError('threads out of range')
            new['threads'] = threads
        if 'kernels' in hdr:
            if hdr['kernels'] not in ('native', 'math-attn', 'math-gelu', 'math-both'):
                raise ValueError('bad kernel mode')
            new['kernels'] = hdr['kernels']
        if 'linear' in hdr:
            if hdr['linear'] not in ('gemm', 'mv'):
                raise ValueError('bad linear mode')
            new['linear'] = hdr['linear']
        if 'affinity' in hdr:
            new['affinity'] = None if hdr['affinity'] is None else [int(c) for c in hdr['affinity']]
        if 'env' in hdr:
            new['env'] = {str(k): str(v) for k, v in dict(hdr['env']).items()}
        if 'cache_gib' in hdr:              # size limit of the uploaded-weights cache on this node
            if not 1 <= float(hdr['cache_gib']) <= 64:
                raise ValueError('cache_gib out of range')
            new['cache_gib'] = float(hdr['cache_gib'])
        atomic_write(self.state_path, json.dumps(new, indent=1))
        log.info('restart requested: %s', {k: new.get(k) for k in ('threads', 'kernels', 'linear', 'affinity', 'env')})
        self.request_exit(EXIT_RESTART)
        return dict(restarting=True, state=new), b''

    def op_shutdown(self, hdr, payload):
        log.info('shutdown requested by controller')
        self.request_exit(EXIT_SHUTDOWN)
        return dict(stopping=True), b''

    def op_update_code(self, hdr, payload):
        if 'psha' not in hdr or not payload:
            raise ValueError('update needs a hashed payload')
        version = hdr['psha'][:12]
        if version == self.version:
            return dict(version=version, unchanged=True), b''
        dest = self.payload_root / version
        if dest.exists():
            shutil.rmtree(dest)
        tmp = self.payload_root / f'.incoming-{uuid.uuid4().hex[:8]}'
        total = 0
        try:
            with tarfile.open(fileobj=io.BytesIO(bytes(payload)), mode='r:gz') as tar:
                for member in tar:
                    parts = member.name.split('/')
                    if member.isdir():
                        continue
                    if not member.isfile() or not all(SAFE_NAME.match(p) for p in parts) or len(parts) > 4:
                        raise ValueError(f'unsafe archive member {member.name!r}')
                    total += member.size
                    if total > 16 << 20:
                        raise ValueError('update too large')
                    target = tmp.joinpath(*parts)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(tar.extractfile(member).read())
            if not (tmp / 'branchy_node.py').is_file():
                raise ValueError('update has no branchy_node.py')
            os.replace(tmp, dest)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        atomic_write(self.payload_root / 'PENDING', self.version)
        atomic_write(self.payload_root / 'CURRENT', version)
        for old in self.payload_root.iterdir():
            if old.is_dir() and old.name not in (version, self.version):
                shutil.rmtree(old, ignore_errors=True)
        log.info('update staged: %s -> %s (%d bytes)', self.version, version, total)
        self.request_exit(EXIT_RESTART)
        return dict(version=version, previous=self.version, restarting=True), b''

    # blobs -------------------------------------------------------------
    def _blob(self, sha):
        if not HEX64.match(str(sha)):
            raise ValueError('bad blob id')
        return self.blob_dir / f'{sha}.bin', self.blob_dir / f'{sha}.part'

    def op_blob_stat(self, hdr, payload):
        final, part = self._blob(hdr['sha'])
        return dict(have=final.is_file(), partial=part.stat().st_size if part.is_file() else 0), b''

    def op_blob_put(self, hdr, payload):
        final, part = self._blob(hdr['sha'])
        total, offset = int(hdr['total']), int(hdr['offset'])
        with self.blob_lock:
            if final.is_file():
                return dict(done=True, size=final.stat().st_size), b''
            used = sum(b.stat().st_size for b in self.blob_dir.iterdir() if b != part)     # the part being uploaded is in `total`
            if used + total > float(self.state.get('cache_gib', 6)) * 2**30:
                raise RuntimeError('blob cache quota exceeded; evict first')
            have = part.stat().st_size if part.is_file() else 0
            if offset != have:
                return dict(done=False, partial=have, resync=True), b''
            with part.open('ab') as f:
                f.write(payload)
            have += len(payload)
            if have < total:
                return dict(done=False, partial=have), b''
            digest = hashlib.sha256()
            with part.open('rb') as f:
                while chunk := f.read(1 << 22):
                    digest.update(chunk)
            if have != total or digest.hexdigest() != hdr['sha']:
                part.unlink()
                raise RuntimeError('blob failed verification; upload discarded')
            os.replace(part, final)
            log.info('blob %s stored (%d MiB)', hdr['sha'][:12], total // 2**20)
            return dict(done=True, size=total), b''

    def op_blob_list(self, hdr, payload):
        return dict(blobs={b.name: b.stat().st_size for b in self.blob_dir.iterdir()}), b''

    def op_blob_evict(self, hdr, payload):
        removed = []
        with self.blob_lock:
            for b in list(self.blob_dir.iterdir()):
                if hdr.get('sha') in ('all', b.name.split('.')[0]):
                    b.unlink()
                    removed.append(b.name)
        return dict(removed=removed), b''

    # jobs --------------------------------------------------------------
    def _job_brief(self, job):
        code = job['proc'].poll()
        return dict(module=job['module'], running=code is None, returncode=code,
                    elapsed_s=round((job.get('ended') or time.time()) - job['started'], 2))

    def op_job_start(self, hdr, payload):
        module = str(hdr['module'])
        if not re.match(r'^[a-z][a-z0-9_]{0,40}$', module) or not (HERE / 'jobs' / f'{module}.py').is_file():
            raise ValueError('unknown job module')
        if payload and 'psha' not in hdr:
            raise ValueError('job arguments must be hashed')
        job_id = uuid.uuid4().hex[:10]
        folder = self.job_dir / job_id
        folder.mkdir(parents=True)
        (folder / 'args.json').write_bytes(bytes(payload) or b'{}')
        env = dict(os.environ, BRANCHY_HOME=str(self.home), BRANCHY_JOB_DIR=str(folder), PYTHONPATH=str(HERE))
        env.update({str(k): str(v) for k, v in dict(hdr.get('env') or {}).items()})
        proc = subprocess.Popen([sys.executable, '-m', f'jobs.{module}'], cwd=str(HERE), env=env,
                                stdin=(folder / 'args.json').open('rb'), stdout=(folder / 'stdout').open('wb'),
                                stderr=(folder / 'stderr').open('wb'))
        job = dict(proc=proc, module=module, started=time.time(), folder=folder)
        self.jobs[job_id] = job
        timeout = float(hdr.get('timeout_s', 3600))

        def reap():
            try:
                proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                proc.kill()
                log.warning('job %s timed out', job_id)
            job['ended'] = time.time()
        threading.Thread(target=reap, daemon=True).start()
        log.info('job %s started: %s', job_id, module)
        return dict(job=job_id), b''

    def op_job_status(self, hdr, payload):
        job = self.jobs[hdr['job']]
        brief = self._job_brief(job)

        def tail(name, limit):
            data = (job['folder'] / name).read_bytes()
            return data[-limit:]
        brief['stderr_tail'] = tail('stderr', 4000).decode('utf-8', 'replace')
        out = tail('stdout', 8 << 20) if not brief['running'] else b''
        return brief, out

    def op_job_kill(self, hdr, payload):
        job = self.jobs[hdr['job']]
        job['proc'].kill()
        return self._job_brief(job), b''

    # ------------------------------------------------------------------ compute ops (main thread)
    def _register(self, name, module, meta):
        if not SAFE_NAME.match(str(name)):
            raise ValueError('bad stack name')
        self.stacks[name] = dict(module=module, meta=meta)
        for key in [k for k in self.sessions if k.split('/', 1)[1] == name]:
            del self.sessions[key]

    def c_load_stack(self, hdr, payload):
        import stacks
        meta = dict(width=int(hdr['width']), heads=int(hdr['heads']), blocks=int(hdr['blocks']), source=hdr['sha'])
        if hdr.get('kind') == 'lfm3':          # a stack of LFM3 layers (lfm3_blocks) instead of Branchy blocks
            meta.update(kind='lfm3', layers=[str(k) for k in hdr['layers']])
        # low-memory experiments: weights stay in the file and are paged in when used (request field or node environment)
        mapped = bool(hdr.get('mmap')) or os.environ.get('BRANCHY_MMAP') == '1'
        if mapped:
            meta['mmap'] = True
        old = self.stacks.get(hdr['name'])
        if old and old['meta'] == meta:
            return dict(cached=True, meta=meta), b''
        final, _ = self._blob(hdr['sha'])
        if not final.is_file():
            raise FileNotFoundError('blob not uploaded')
        t = perf()
        if meta.get('kind') == 'lfm3':
            import lfm3_blocks
            module = lfm3_blocks.build_stack(meta['layers'])
        else:
            module = stacks.build_stack(meta['width'], meta['heads'], meta['blocks'], empty=mapped)
        stacks.load_blob_file(module, final, mmap=mapped)
        self._register(hdr['name'], module, meta)
        return dict(cached=False, meta=meta, load_s=perf() - t), b''

    def c_set_budget(self, hdr, payload):
        """RAM budget for memory-mapped weights in this node (stacks.Budget); mb 0 removes it. Returns the old budget's counters."""
        import stacks
        old = stacks.set_budget(float(hdr.get('mb') or 0))
        return dict(previous=old, budget_mb=float(hdr.get('mb') or 0)), b''

    def c_synth_stack(self, hdr, payload):
        import stacks
        meta = dict(width=int(hdr['width']), heads=int(hdr['heads']), blocks=int(hdr['blocks']),
                    source=f"synthetic:{hdr['seed']}:{int(hdr.get('depth', 11))}")
        old = self.stacks.get(hdr['name'])
        if old and old['meta'] == meta:
            return dict(cached=True, meta=meta, wsha=old.get('wsha')), b''
        t = perf()
        module = stacks.build_stack(meta['width'], meta['heads'], meta['blocks'])
        stacks.fill_synthetic(module, str(hdr['seed']), int(hdr.get('depth', 11)))
        wsha = hashlib.sha256(stacks.state_to_blob(module.state_dict())).hexdigest() if hdr.get('hash', True) else None
        self._register(hdr['name'], module, meta)
        self.stacks[hdr['name']]['wsha'] = wsha
        return dict(cached=False, meta=meta, wsha=wsha, build_s=perf() - t), b''

    def c_unload(self, hdr, payload):
        names = list(self.stacks) if hdr.get('name') == 'all' else [hdr['name']]
        for name in names:
            self.stacks.pop(name, None)
            for key in [k for k in self.sessions if k.split('/', 1)[1] == name]:
                del self.sessions[key]
        import gc
        gc.collect()
        return dict(unloaded=names), b''

    def c_forward(self, hdr, payload):
        import numpy as np
        import torch
        import stacks
        t0 = perf()
        shape = [int(v) for v in hdr['shape']]
        din = '<f2' if hdr.get('in_dtype') == 'f2' else '<f4'
        dout = '<f2' if hdr.get('out_dtype') == 'f2' else '<f4'
        if len(shape) != 3 or shape[0] not in (1, len(hdr['stacks'])):
            raise ValueError('expected shape [1, T, width] or one input per stack [n, T, width]')
        x = torch.from_numpy(np.frombuffer(payload, dtype=din).astype(np.float32).reshape(shape))
        t1 = perf()
        session, reset = str(hdr.get('session', 'default')), bool(hdr.get('reset'))
        chain = bool(hdr.get('chain'))
        outs, times, length = [], [], 0
        x_all = x
        for index, name in enumerate(hdr['stacks']):
            entry = self.stacks[name]
            if entry['meta']['width'] != shape[2]:
                raise ValueError('width mismatch')
            x = x_all[index:index + 1] if shape[0] > 1 else x_all
            if chain and index:                 # stacks one after the other: each reads the previous one's output
                x = outs.pop()
            key = f'{session}/{name}'
            cache = None if reset else self.sessions.get(key)
            if cache is None and not reset:
                raise RuntimeError(f'no cache for {key}; send reset=true with the full prefix')
            ta = perf()
            y, new = stacks.run_stack(entry['module'], x, cache)
            times.append(perf() - ta)
            self.sessions[key] = new
            length = new[0][0].shape[2]
            outs.append(y)
        t2 = perf()
        raw = b''.join(y.numpy().astype(dout, copy=False).tobytes() for y in outs)
        return dict(compute_s=times, deser_s=t1 - t0, ser_s=perf() - t2, cached_len=length), raw

    def c_truncate(self, hdr, payload):
        import stacks
        session, length = str(hdr.get('session', 'default')), int(hdr['length'])
        done = []
        for key in list(self.sessions):
            if key.split('/', 1)[0] == session:
                self.sessions[key] = stacks.truncate_cache(self.sessions[key], length)
                done.append(key)
        return dict(truncated=done, length=length), b''

    def c_drop_session(self, hdr, payload):
        session = str(hdr.get('session', 'default'))
        gone = [k for k in self.sessions if hdr.get('session') == 'all' or k.split('/', 1)[0] == session]
        for key in gone:
            del self.sessions[key]
        return dict(dropped=gone), b''

    def c_selftest(self, hdr, payload):
        """Seeded prefill + cached decode on synthetic weights; returns the outputs for cross-device comparison."""
        import stacks
        meta, first = stacks.selftest_run(int(hdr.get('width', 256)), int(hdr.get('heads', 4)), int(hdr.get('blocks', 3)),
                                          int(hdr.get('prefill', 64)), int(hdr.get('decode', 8)), str(hdr.get('seed', 'selftest')))
        return meta, first.numpy().astype('<f4', copy=False).tobytes()

    def c_profile(self, hdr, payload):
        """Time synthetic block stacks: prefill at several lengths, then single-token cached decode."""
        import statistics
        import stacks
        width, heads, blocks = int(hdr.get('width', 880)), int(hdr.get('heads', 10)), int(hdr.get('blocks', 7))
        steps, seed = int(hdr.get('decode_steps', 24)), str(hdr.get('seed', 'profile'))
        name = hdr.get('stack')
        if name:
            module = self.stacks[name]['module']
        else:
            module = stacks.build_stack(width, heads, blocks)
            stacks.fill_synthetic(module, seed)
        rows = []
        for length in [int(v) for v in hdr.get('prefills', [32])]:
            x = stacks.synthetic_tensor([1, length + steps, width], f'{seed}/x{length}', 1.0)
            if hdr.get('warm', True):
                stacks.run_stack(module, x[:, :min(length, 8)], None)
            t = perf()
            _, cache = stacks.run_stack(module, x[:, :length], None)
            prefill_s = perf() - t
            per = []
            for i in range(length, length + steps):
                t = perf()
                _, cache = stacks.run_stack(module, x[:, i:i + 1], cache)
                per.append((perf() - t) * 1000)
            ordered = sorted(per)
            rows.append(dict(prefill=length, prefill_s=prefill_s, prefill_tps=length / prefill_s,
                             decode_ms_median=statistics.median(per), decode_ms_min=ordered[0],
                             decode_ms_p90=ordered[int(0.9 * (len(ordered) - 1))], decode_ms=[round(v, 3) for v in per]))
        return dict(width=width, heads=heads, blocks=blocks, weight_mb=stacks.stack_bytes(width, blocks) / 2**20,
                    threads=self.static.get('threads'), kernels=self.static.get('kernels'),
                    linear=self.static.get('linear'), rows=rows), b''


QUIET = {'forward', 'ping', 'blob_put', 'blob_stat', 'info', 'job_status', 'confirm', 'get_log', 'drop_session'}
CONTROL = {name[3:]: getattr(Node, name) for name in dir(Node) if name.startswith('op_')}
COMPUTE = {name[2:]: getattr(Node, name) for name in dir(Node) if name.startswith('c_')}
PRIVILEGED = {'update_code', 'job_start', 'restart', 'shutdown'}


class Handler(socketserver.BaseRequestHandler):
    def handle(self):
        import nodewire
        node: Node = self.server.node
        sock = self.request
        nodewire.tune(sock)
        sock.settimeout(20)
        try:
            channel = nodewire.server_handshake(sock, node.token, node.name)
        except Exception as exc:
            quiet = 'disconnected' in str(exc)      # port probes (start.sh, scanners) just connect and leave
            log.log(logging.DEBUG if quiet else logging.WARNING, 'handshake from %s failed: %s', self.client_address[0], exc)
            return
        sock.settimeout(None)
        log.debug('controller connected from %s', self.client_address[0])
        while True:
            try:
                hdr, payload = channel.recv()
            except Exception as exc:
                log.debug('connection from %s closed: %s', self.client_address[0], exc)
                return
            t_in = perf()
            op = hdr.get('op')
            try:
                reply, out = node.dispatch(op, hdr, payload)
                reply = dict(reply, ok=True)
            except Exception as exc:
                if op not in COMPUTE:
                    log.exception('op %s failed', op)
                reply, out = dict(ok=False, error=f'{type(exc).__name__}: {exc}'[:800]), b''
            reply['srv_s'] = perf() - t_in
            try:
                channel.send(reply, out)
            except Exception as exc:
                log.info('send to %s failed: %s', self.client_address[0], exc)
                return
            try:
                node.note(op, hdr, reply, len(payload), len(out), reply['srv_s'], self.client_address[0])
            except Exception:
                log.exception('activity note failed')


class Server(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True
    request_queue_size = 16


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--home', type=Path, required=True)
    args = ap.parse_args()
    home = args.home.resolve()
    (home / 'logs').mkdir(exist_ok=True)
    handler = logging.handlers.RotatingFileHandler(home / 'logs' / 'node.log', maxBytes=2 << 20, backupCount=3)
    handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(logging.Formatter('%(asctime)s %(message)s', '%H:%M:%S'))
    logging.basicConfig(level=logging.INFO, handlers=[handler, console])
    node = Node(home)
    node.setup_torch()
    server = Server((node.state.get('bind', '0.0.0.0'), int(node.state.get('port', 8730))), Handler)
    server.node = node
    threading.Thread(target=server.serve_forever, daemon=True).start()
    threading.Thread(target=node.activity_loop, daemon=True).start()
    st = node.static
    log.info('=' * 78)
    log.info(' BRANCHY NODE "%s"   code %s   port %s', node.name, node.version, server.server_address[1])
    log.info(' %s | torch %s | python %s', st.get('android') or st.get('cpu_model') or st.get('machine'), st['torch'], st['python'])
    log.info(' threads=%s  attention=%s  linear=%s  cpus visible=%s', st['threads'], st['kernels'], st['linear'],
             len(st['affinity'] or []) or st['cpu_count'])
    log.info(' waiting for the coordinator... (activity is summarized every 5 s)')
    log.info('=' * 78)
    node.compute_loop()


if __name__ == '__main__':
    main()
