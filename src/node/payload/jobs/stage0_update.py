"""Job: replace stage 0 files (start.sh, stop.sh, supervisor.py) and relaunch the node through start.sh.

args: {"files": {"start.sh": "<text>", ...}, "env": {"BRANCHY_NO_TMUX": "1"}}
The relaunch is detached from this job; the node goes down for a few seconds and comes back
under the new supervisor (inside tmux when available).
"""
import json
import os
import subprocess
import sys
from pathlib import Path

args = json.load(sys.stdin)
home = Path(os.environ['BRANCHY_HOME'])
allowed = {'start.sh', 'stop.sh', 'supervisor.py'}
written = []
for name, text in args.get('files', {}).items():
    if name not in allowed:
        raise SystemExit(f'refusing to write {name}')
    backup = home / (name + '.prev')
    if (home / name).exists():
        backup.write_bytes((home / name).read_bytes())
    (home / name).write_text(text)
    os.chmod(home / name, 0o755)
    written.append(name)
env = dict(os.environ, BRANCHY_NO_ATTACH='1')
for key in ('BRANCHY_JOB_DIR', 'PYTHONPATH', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    env.pop(key, None)
env.update({str(k): str(v) for k, v in (args.get('env') or {}).items()})
(home / 'logs').mkdir(exist_ok=True)
log = open(home / 'logs' / 'relaunch.out', 'ab')
subprocess.Popen(['bash', str(home / 'start.sh')], cwd=str(home), env=env, stdin=subprocess.DEVNULL, stdout=log,
                 stderr=subprocess.STDOUT, start_new_session=True)
print(json.dumps(dict(written=written, relaunching=True)))
