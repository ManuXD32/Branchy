#!/usr/bin/env bash
# Coordinator web UI on http://127.0.0.1:8780 (pass --bind 0.0.0.0 to reach it from other machines; it has no login).
#   bash scripts/start_webui.sh [--threads 8] [--linear mv] [--bind 0.0.0.0] [--port 8780]
cd "$(dirname "$0")/.."
PY="${PYTHON:-python3}"
[ -d models ] || { echo "!! no models/ folder: see MODELS.md (download the two inference files and tokenizer.json)"; exit 1; }
# torch builds that compute with OpenBLAS oversubscribe the CPU unless OpenBLAS gets the same thread count as torch
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-8}"
exec "$PY" -u src/webui/webui.py --models models --cluster src/node/cluster.json "$@"
