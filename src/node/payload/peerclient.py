"""Client for Branchy nodes (pure Python + nodewire). In the payload so that a node can talk to other nodes when it is
the coordinator; the controller uses it through nodeclient.py."""
from __future__ import annotations

import hashlib
import json
import socket
import time

import nodewire

CHUNK = 4 << 20
IDEMPOTENT = {'info', 'ping', 'confirm', 'get_log', 'peers_probe', 'blob_stat', 'blob_list', 'blob_evict',
              'load_stack', 'synth_stack', 'unload', 'drop_session', 'selftest', 'profile', 'job_status', 'set_budget'}
HASHED = {'update_code', 'job_start'}


class NodeError(RuntimeError):
    pass



class Node:
    def __init__(self, name, host, port, token, timeout=120.0, delay_ms=0.0):
        self.name, self.host, self.port = name, host, int(port)
        self.token = token.encode() if isinstance(token, str) else token
        self.timeout = timeout
        self.delay_s = delay_ms / 1000.0     # artificial extra round-trip time (local simulations only)
        self.channel = None
        self.sock = None
        self.hello = None
        self.uploaded = 0

    # ---------------------------------------------------------------- transport
    def connect(self, timeout=None):
        self.close()
        sock = socket.create_connection((self.host, self.port), timeout=timeout or 10)
        nodewire.tune(sock)
        sock.settimeout(timeout or 10)
        self.channel, self.hello = nodewire.client_handshake(sock, self.token)
        sock.settimeout(self.timeout)
        self.sock = sock
        return self

    def close(self):
        if self.sock is not None:
            try:
                self.sock.close()
            except OSError:
                pass
        self.sock = self.channel = None

    def wait_up(self, timeout=90.0, expect_version=None):
        deadline = time.time() + timeout
        last = None
        while time.time() < deadline:
            try:
                self.connect(timeout=5)
                info = self.call('info')[0]
                if expect_version in (None, info['version']):
                    return info
                last = f"running {info['version']}, expected {expect_version}"
            except (OSError, nodewire.WireError, NodeError) as exc:
                last = exc
            time.sleep(1.0)
        raise NodeError(f'{self.name}: not reachable after {timeout:.0f}s ({last})')

    def call(self, op, payload=b'', timeout=None, **fields):
        """Returns (reply, payload, rpc_seconds). Reconnects once for idempotent ops."""
        attempts = 2 if op in IDEMPOTENT else 1
        for attempt in range(attempts):
            try:
                if self.channel is None:
                    self.connect()
                if timeout:
                    self.sock.settimeout(timeout)
                header = dict(fields, op=op)
                if op in HASHED:
                    header['psha'] = hashlib.sha256(payload).hexdigest()
                t = time.perf_counter()
                if self.delay_s:
                    time.sleep(self.delay_s / 2)
                self.channel.send(header, payload)
                reply, out = self.channel.recv()
                if self.delay_s:
                    time.sleep(self.delay_s / 2)
                elapsed = time.perf_counter() - t
                if timeout:
                    self.sock.settimeout(self.timeout)
                if not reply.get('ok'):
                    raise NodeError(f"{self.name}: {op}: {reply.get('error')}")
                return reply, out, elapsed
            except (OSError, nodewire.WireError) as exc:
                self.close()
                if attempt + 1 >= attempts:
                    raise NodeError(f'{self.name}: connection lost during {op}: {exc}') from exc
                time.sleep(0.5)

    # ---------------------------------------------------------------- helpers
    def info(self):
        return self.call('info')[0]

    def upload_blob(self, data: bytes, progress=None) -> str:
        sha = hashlib.sha256(data).hexdigest()
        stat = self.call('blob_stat', sha=sha)[0]
        if stat['have']:
            return sha
        offset = stat['partial'] if stat['partial'] <= len(data) else 0
        if stat['partial'] > len(data):
            self.call('blob_evict', sha=sha)
        while True:
            chunk = data[offset:offset + CHUNK]
            reply = self.call('blob_put', chunk, timeout=600, sha=sha, total=len(data), offset=offset)[0]
            if reply.get('resync'):
                offset = reply['partial']
                continue
            offset += len(chunk)
            self.uploaded += len(chunk)
            if progress:
                progress(offset, len(data))
            if reply['done']:
                return sha

    def ensure_stack(self, name, state_dict, width, heads, blocks, progress=None, **extra):
        import stacks
        sha = self.upload_blob(stacks.state_to_blob(state_dict), progress)
        return self.call('load_stack', timeout=600, name=name, width=width, heads=heads, blocks=blocks, sha=sha, **extra)[0]

    def ensure_blob_stack(self, name, raw, sha, width, heads, blocks, progress=None, **extra):
        """Like ensure_stack for weights that are already a blob (bytes or None if the node is known to have it)."""
        if raw is not None:
            self.upload_blob(raw, progress)
        return self.call('load_stack', timeout=900, name=name, width=width, heads=heads, blocks=blocks, sha=sha, **extra)[0]

    def synth_stack(self, name, width, heads, blocks, seed, depth=11, hash=True):
        return self.call('synth_stack', timeout=900, name=name, width=width, heads=heads, blocks=blocks,
                         seed=seed, depth=depth, hash=hash)[0]

    def forward(self, stack_names, h, reset, session='default', in_dtype='f4', out_dtype='f4', chain=False):
        """h: torch tensor [1, T, W] (same input for every stack) or [n, T, W] (one input per stack).
        Returns ({stack: tensor [1, T, W]}, stats). With chain=True the stacks run one after the other on the node
        (each reads the previous one's output) and only the last output comes back, under the last stack's name."""
        import numpy as np
        import torch
        array = h.detach().to('cpu', torch.float32).contiguous().numpy()
        raw = array.astype('<f2' if in_dtype == 'f2' else '<f4', copy=False).tobytes()
        reply, out, rpc_s = self.call('forward', raw, stacks=list(stack_names), shape=list(array.shape), reset=bool(reset),
                                      session=session, in_dtype=in_dtype, out_dtype=out_dtype, chain=bool(chain))
        returned = list(stack_names)[-1:] if chain else list(stack_names)
        dt = np.dtype('<f2' if out_dtype == 'f2' else '<f4')
        one = (1,) + tuple(array.shape[1:])
        count = array.shape[1] * array.shape[2]
        each = count * dt.itemsize
        if len(out) != each * len(returned):
            raise NodeError(f'{self.name}: unexpected reply size')
        outs = {}
        for i, name in enumerate(returned):
            arr = np.frombuffer(out, dtype=dt, count=count, offset=i * each).astype(np.float32).reshape(one)
            outs[name] = torch.from_numpy(arr)
        stats = dict(node=self.name, rpc_s=rpc_s, srv_s=reply['srv_s'], compute_s=sum(reply['compute_s']),
                     deser_s=reply['deser_s'], ser_s=reply['ser_s'], queue_s=reply.get('queue_s', 0.0),
                     net_s=rpc_s - reply['srv_s'], up_bytes=len(raw), down_bytes=len(out))
        return outs, stats

    def restart(self, wait=90.0, **settings):
        self.call('restart', **settings)
        self.close()
        time.sleep(1.5)
        info = self.wait_up(wait)
        for key, value in settings.items():
            if key in ('threads', 'kernels', 'linear') and info[key] != value:
                raise NodeError(f'{self.name}: after restart {key}={info[key]}, wanted {value}')
        return info

    def push_code(self, tarball: bytes, version: str, wait=120.0):
        """Push a payload version, wait for the node to come back on it, then confirm (else it rolls back)."""
        reply = self.call('update_code', tarball, timeout=120)[0]
        if reply.get('unchanged'):
            return dict(version=version, unchanged=True)
        self.close()
        time.sleep(1.5)
        info = self.wait_up(wait, expect_version=version)
        self.call('confirm')
        return dict(version=info['version'], previous=reply.get('previous'), unchanged=False)

    def run_job(self, module, args=None, timeout=3600, poll=1.0, env=None):
        reply = self.call('job_start', json.dumps(args or {}).encode(), module=module, timeout_s=timeout, env=env or {})[0]
        job = reply['job']
        while True:
            status, out, _ = self.call('job_status', job=job)
            if not status['running']:
                if status['returncode'] != 0:
                    raise NodeError(f"{self.name}: job {module} failed ({status['returncode']}): {status['stderr_tail'][-1500:]}")
                return json.loads(bytes(out)) if out else None
            time.sleep(poll)
