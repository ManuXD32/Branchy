"""Job: run one shell command on the node and return its output. args: {"cmd": "...", "timeout": 60}

Same privilege as update_code (the controller can already run any code here); used for maintenance,
for example checking that tmux exists or installing a Termux package.
"""
import json
import os
import subprocess
import sys

args = json.load(sys.stdin)
try:
    done = subprocess.run(['bash', '-c', args['cmd']], capture_output=True, text=True, timeout=float(args.get('timeout', 60)),
                          cwd=os.environ.get('BRANCHY_HOME'))
    print(json.dumps(dict(rc=done.returncode, out=done.stdout[-6000:], err=done.stderr[-3000:])))
except subprocess.TimeoutExpired:
    print(json.dumps(dict(rc=-1, out='', err='timeout')))
