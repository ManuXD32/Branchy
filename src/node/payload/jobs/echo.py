"""Sample job: proves that the controller can run pushed code on a node and get JSON back."""
import json, os, platform, sys

args = json.load(sys.stdin)
print(json.dumps(dict(echo=args, python=platform.python_version(), home=os.environ.get('BRANCHY_HOME'))))
