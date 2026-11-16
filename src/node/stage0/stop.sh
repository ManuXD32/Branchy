#!/usr/bin/env bash
# Stop the Branchy node (supervisor + daemon + its tmux session).
cd "$(dirname "$0")" || exit 1
if [ -f supervisor.pid ] && kill -0 "$(cat supervisor.pid)" 2>/dev/null; then
  kill "$(cat supervisor.pid)"; sleep 2
  [ "$1" = quiet ] || echo "Branchy node stopped."
else
  [ "$1" = quiet ] || echo "Branchy node was not running."
fi
rm -f supervisor.pid
SESSION="$(cat .tmux_session 2>/dev/null)"
[ -n "$SESSION" ] && command -v tmux >/dev/null 2>&1 && tmux kill-session -t "=$SESSION" 2>/dev/null
case "$PREFIX" in *com.termux*) [ "$1" = quiet ] || { command -v termux-wake-unlock >/dev/null 2>&1 && termux-wake-unlock; };; esac
exit 0
