#!/usr/bin/env bash
# Prepare this machine as the coordinator: shared secret, worker bundle, cluster file.
#   bash scripts/setup_coordinator.sh 192.168.1.10        (the address your devices can reach this machine at)
set -e
cd "$(dirname "$0")/.."
ADDR="${1:?give the address of this machine as seen from your devices, e.g. 192.168.1.10}"
PORT="${2:-8765}"
PY="${PYTHON:-python3}"
"$PY" -c "import torch, numpy" || { echo "!! $PY has no torch/numpy: pip install -r requirements.txt"; exit 1; }
mkdir -p .secrets dist
if [ ! -s .secrets/node_token ]; then
  ( umask 077; "$PY" -c "import secrets; print(secrets.token_hex(24), end='')" > .secrets/node_token )
  echo ">> created the shared secret .secrets/node_token (never commit it)"
fi
[ -f src/node/cluster.json ] || { cp src/node/cluster.example.json src/node/cluster.json; echo ">> created src/node/cluster.json: edit it and list your workers"; }
( cd src/node && "$PY" build_bundle.py --base "http://$ADDR:$PORT" )
cat <<MSG

Next:
1. Serve the bundle to your devices (leave it running while you install workers):
     $PY -m http.server $PORT --bind $ADDR --directory dist
2. On every worker (Linux shell or Termux on Android), run:
     curl -fsSL http://$ADDR:$PORT/go.sh | BRANCHY_TOKEN=$(cat .secrets/node_token) BRANCHY_NAME=<name> bash
   (Linux with a firewall: open TCP port 8730 first, e.g. sudo firewall-cmd --add-port=8730/tcp)
3. Put each worker's name and address into src/node/cluster.json, then check:
     cd src/node && $PY nodectl.py info && $PY nodectl.py selftest
4. Start the web UI:  bash scripts/start_webui.sh
MSG
