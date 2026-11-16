#!/usr/bin/env python3
"""Branchy node supervisor (stage 0). Keeps the daemon alive; never imports torch.

Runs in the foreground of a tmux pane (preferred, so the owner can watch) or
detached under nohup. Everything is also written to logs/.

Exit codes from the daemon: 75 = planned restart (settings or code changed),
76 = stop. Anything else is a crash and is retried with backoff. A freshly
pushed code version must be confirmed by the controller within CONFIRM_S,
otherwise (or if it crashes first) the previous version is restored.
"""
import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

HOME = Path(__file__).resolve().parent
PAYLOAD = HOME / 'payload'
CONFIRM_S = int(os.environ.get('BRANCHY_CONFIRM_S', 180))
TTY = sys.stdout.isatty()
child = None


def say(msg):
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} [supervisor] {msg}\n"
    with open(HOME / 'logs' / 'supervisor.out', 'a') as f:
        f.write(line)
    if TTY:
        sys.stdout.write(line)
        sys.stdout.flush()


def pump(stream, path):
    """Copy the daemon's output to its log file and, when watched, to the terminal."""
    with open(path, 'ab') as out:
        for line in iter(stream.readline, b''):
            out.write(line)
            out.flush()
            if TTY:
                try:
                    sys.stdout.buffer.write(line)
                    sys.stdout.flush()
                except OSError:
                    pass


def rollback(reason):
    pending = PAYLOAD / 'PENDING'
    if not pending.exists():
        return False
    previous = pending.read_text().strip()
    say(f'ROLLBACK to {previous}: {reason}')
    (PAYLOAD / 'CURRENT').write_text(previous)
    pending.unlink()
    return True


def stop(signum, frame):
    say(f'signal {signum}: stopping')
    if child and child.poll() is None:
        child.terminate()
        try:
            child.wait(timeout=10)
        except subprocess.TimeoutExpired:
            child.kill()
    (HOME / 'supervisor.pid').unlink(missing_ok=True)
    sys.exit(0)


def main():
    global child
    (HOME / 'logs').mkdir(exist_ok=True)
    (HOME / 'supervisor.pid').write_text(str(os.getpid()))
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGHUP, stop)
    backoff = 1
    while True:
        version = (PAYLOAD / 'CURRENT').read_text().strip()
        state = json.loads((HOME / 'node_state.json').read_text())
        threads = str(state.get('threads', 4))
        env = dict(os.environ, OMP_NUM_THREADS=threads, OPENBLAS_NUM_THREADS=threads, MKL_NUM_THREADS=threads,
                   PYTHONUNBUFFERED='1')
        env.update({str(k): str(v) for k, v in (state.get('env') or {}).items()})
        say(f'starting daemon {version} threads={threads} kernels={state.get("kernels")} linear={state.get("linear")} '
            f'env={state.get("env")}')
        started = time.time()
        on_trial = (PAYLOAD / 'PENDING').exists()   # this launch runs a freshly pushed, unconfirmed version
        out_path = HOME / 'logs' / 'daemon.out'
        if out_path.exists() and out_path.stat().st_size > 4 << 20:
            os.replace(out_path, out_path.with_suffix('.out.1'))
        child = subprocess.Popen([sys.executable, str(PAYLOAD / version / 'branchy_node.py'), '--home', str(HOME)],
                                 env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
        reader = threading.Thread(target=pump, args=(child.stdout, out_path), daemon=True)
        reader.start()
        while child.poll() is None:
            time.sleep(1)
            if on_trial and not (PAYLOAD / 'PENDING').exists():
                on_trial = False            # confirmed by the controller
            # A PENDING file only concerns this process while CURRENT still names the version it runs;
            # once this daemon stages a newer update, PENDING belongs to that next launch.
            if on_trial and (PAYLOAD / 'CURRENT').read_text().strip() == version \
                    and time.time() - started > CONFIRM_S:
                child.terminate()
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    child.kill()
                rollback(f'no controller confirmation within {CONFIRM_S}s')
        reader.join(timeout=3)
        code = child.returncode
        uptime = time.time() - started
        say(f'daemon exited code={code} after {uptime:.0f}s')
        if code == 76:
            (HOME / 'supervisor.pid').unlink(missing_ok=True)
            return
        if code == 75:
            backoff = 1
            continue
        if on_trial and (PAYLOAD / 'CURRENT').read_text().strip() == version:
            rollback(f'new version crashed with code {code}')
        if uptime > 60:
            backoff = 1
        time.sleep(backoff)
        backoff = min(30, backoff * 2)


if __name__ == '__main__':
    main()
