#!/usr/bin/env bash
# Branchy node installer:  curl -fsSL http://SERVER:8765/go.sh | BRANCHY_TOKEN=... BRANCHY_NAME=phone bash
set -e
BASE="${BRANCHY_BASE:-__BASE__}"
DEST="${BRANCHY_DEST:-$HOME/branchy-node}"
[ -n "$BRANCHY_TOKEN" ] || { echo "!! BRANCHY_TOKEN is not set; copy the full command again"; exit 1; }
mkdir -p "$DEST"; cd "$DEST"
echo ">> Downloading $BASE/branchy-node.zip"
curl -fsSL "$BASE/branchy-node.zip" -o bundle.zip
[ -f stop.sh ] && bash ./stop.sh quiet || true
if command -v unzip >/dev/null 2>&1; then unzip -oq bundle.zip
elif command -v python3 >/dev/null 2>&1; then python3 -m zipfile -e bundle.zip .
elif command -v python >/dev/null 2>&1; then python -m zipfile -e bundle.zip .
elif command -v pkg >/dev/null 2>&1; then pkg install -y unzip && unzip -oq bundle.zip
else echo "!! need unzip or python to unpack"; exit 1; fi
rm -f bundle.zip payload/PENDING
( umask 077; printf '%s' "$BRANCHY_TOKEN" > token )
if [ -n "$BRANCHY_NAME" ] && [ -f node_state.json ]; then rm -f node_state.json; fi
exec bash ./start.sh
