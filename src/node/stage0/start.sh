#!/usr/bin/env bash
# Branchy node: find a Python with torch, then run the supervisor in the background.
cd "$(dirname "$0")" || exit 1
HOME_DIR="$PWD"
mkdir -p logs cache
is_termux() { [ -n "$TERMUX_VERSION" ] || case "$PREFIX" in *com.termux*) true;; *) false;; esac; }
has_torch() { "$1" -c 'import torch, numpy' >/dev/null 2>&1; }

PY=""
for cand in "$BRANCHY_PYTHON" "$(cat .python 2>/dev/null)" "$HOME_DIR/venv/bin/python" python3 python \
            "$HOME"/Escritorio/Branchy/y_llm_poc/.venv/bin/python "$HOME"/Escritorio/Branchy/y_llm_poc/venv/bin/python \
            "$HOME"/miniconda3/envs/*/bin/python "$HOME"/anaconda3/envs/*/bin/python "$HOME"/.venv/bin/python; do
  [ -n "$cand" ] || continue
  full="$(command -v "$cand" 2>/dev/null)" || continue
  if has_torch "$full"; then PY="$full"; break; fi
done
if [ -z "$PY" ]; then
  if is_termux; then
    echo ">> No Python with torch found. Installing Termux packages (python, python-numpy, python-torch)..."
    pkg install -y python python-numpy python-torch || { echo "!! pkg install failed"; exit 1; }
    PY="$(command -v python)"
  else
    echo ">> No Python with torch found. Creating a private venv in $HOME_DIR/venv (CPU-only torch, nothing system-wide)..."
    python3 -m venv venv && venv/bin/pip install -q --upgrade pip \
      && venv/bin/pip install -q torch --index-url https://download.pytorch.org/whl/cpu \
      && venv/bin/pip install -q numpy || { echo "!! venv setup failed"; exit 1; }
    PY="$HOME_DIR/venv/bin/python"
  fi
  has_torch "$PY" || { echo "!! torch still not importable with $PY"; exit 1; }
fi
echo "$PY" > .python
echo ">> Python: $PY ($("$PY" -c 'import torch,platform;print(platform.python_version(),"torch",torch.__version__)'))"

if [ ! -f node_state.json ]; then
  # Termux default = the configuration validated on the Redmagic (math attention, OpenBLAS single-threaded).
  if is_termux; then KERN=math-attn; ENVJ='{"OPENBLAS_NUM_THREADS": "1"}'; else KERN=native; ENVJ='{}'; fi
  NAME="${BRANCHY_NAME:-$(hostname 2>/dev/null || echo node)}"
  printf '{\n "name": "%s",\n "port": %s,\n "threads": 4,\n "kernels": "%s",\n "linear": "gemm",\n "affinity": null,\n "env": %s,\n "cache_gib": 6\n}\n' \
    "$NAME" "${BRANCHY_PORT:-8730}" "$KERN" "$ENVJ" > node_state.json
fi
[ -s token ] || { echo "!! token file missing: re-run the install command (it carries BRANCHY_TOKEN)"; exit 1; }

bash ./stop.sh quiet
is_termux && command -v termux-wake-lock >/dev/null 2>&1 && termux-wake-lock && echo ">> Termux wake lock taken"
PORT="$("$PY" -c 'import json;print(json.load(open("node_state.json")).get("port",8730))')"
if [ "$PORT" = 8730 ]; then SESSION=branchy; else SESSION="branchy-$PORT"; fi
echo "$SESSION" > .tmux_session

launch_tmux() {
  command -v tmux >/dev/null 2>&1 || return 1
  tmux kill-session -t "=$SESSION" 2>/dev/null
  tmux new-session -d -s "$SESSION" -n node -x 200 -y 50 \
    "cd '$HOME_DIR' && '$PY' supervisor.py; echo; echo '>> Branchy node stopped. Start again: bash $HOME_DIR/start.sh  (Enter closes this window)'; read _" || return 1
  tmux set-option -t "$SESSION" status-left "[branchy node] " >/dev/null 2>&1
  tmux set-option -t "$SESSION" history-limit 20000 >/dev/null 2>&1
  return 0
}
launch_background() {
  if command -v setsid >/dev/null 2>&1; then
    nohup setsid "$PY" supervisor.py > /dev/null 2>&1 < /dev/null &
  else
    nohup "$PY" supervisor.py > /dev/null 2>&1 < /dev/null &
  fi
}
wait_up() {
  for i in $(seq 1 40); do
    sleep 1
    "$PY" -c "import socket,sys;s=socket.create_connection(('127.0.0.1',$PORT),2);sys.exit(0 if b'branchy-node' in s.recv(200) else 1)" 2>/dev/null && return 0
  done
  return 1
}

MODE=""
if [ -z "$BRANCHY_NO_TMUX" ] && launch_tmux; then
  if wait_up; then MODE=tmux; else echo "!! node did not come up inside tmux, falling back to background mode"; tmux kill-session -t "=$SESSION" 2>/dev/null; bash ./stop.sh quiet; fi
fi
if [ -z "$MODE" ]; then
  [ -n "$BRANCHY_NO_TMUX" ] || echo ">> tmux not available: running in the background instead (install tmux and run start.sh again to watch it)"
  launch_background
  if wait_up; then MODE=background; fi
fi
if [ -z "$MODE" ]; then
  echo "!! Node did not come up in 40 s. Last log lines:"; tail -n 30 logs/supervisor.out logs/daemon.out 2>/dev/null
  exit 1
fi
echo ""
echo "=============================================================="
echo " Branchy node is RUNNING on port $PORT (name: $("$PY" -c 'import json;print(json.load(open("node_state.json"))["name"])'))"
if [ "$MODE" = tmux ]; then
  echo " Watch it:  tmux attach -t $SESSION      (detach again with Ctrl-b d)"
else
  echo " Running in the background. Live log: tail -f $HOME_DIR/logs/daemon.out"
fi
echo " Logs: $HOME_DIR/logs/   Stop: bash $HOME_DIR/stop.sh"
if command -v firewall-cmd >/dev/null 2>&1 && ! firewall-cmd --list-ports 2>/dev/null | grep -q "$PORT/tcp"; then
  echo " NOTE: firewalld is active. Open the port with:  sudo firewall-cmd --add-port=$PORT/tcp"
fi
echo "=============================================================="
if [ "$MODE" = tmux ] && [ -t 1 ] && [ -z "$TMUX" ] && [ -z "$BRANCHY_NO_ATTACH" ] && [ -e /dev/tty ]; then
  sleep 1; exec tmux attach -t "=$SESSION" < /dev/tty
fi
exit 0
