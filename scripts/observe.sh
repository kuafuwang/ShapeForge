#!/usr/bin/env bash
# Start or restart the local ShapeForge dashboard without leaving duplicate servers.
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PORT="${OBSERVE_PORT:-8765}"
URL="http://127.0.0.1:${PORT}"
PID_DIR="${TMPDIR:-/tmp}/shapeforge-observe-${UID}"
PID_FILE="${PID_DIR}/server.pid"
LOG_FILE="${PID_DIR}/server.log"
export OBSERVE_PORT="$PORT"

command -v lsof >/dev/null 2>&1 || {
  echo "error: lsof is required to detect an existing dashboard" >&2
  exit 1
}
command -v curl >/dev/null 2>&1 || {
  echo "error: curl is required for the dashboard health check" >&2
  exit 1
}
test -x .venv/bin/python || {
  echo "error: .venv/bin/python is missing; run 'uv sync' first" >&2
  exit 1
}
mkdir -p "$PID_DIR"

stop_existing() {
  local pids=()
  while IFS= read -r pid; do
    [[ "$pid" =~ ^[0-9]+$ ]] && pids+=("$pid")
  done < <(lsof -nP -iTCP:"$PORT" -sTCP:LISTEN -t 2>/dev/null || true)

  if ((${#pids[@]} == 0)); then
    if [[ -f "$PID_FILE" ]]; then
      local recorded
      recorded="$(cat "$PID_FILE")"
      if [[ "$recorded" =~ ^[0-9]+$ ]] && kill -0 "$recorded" 2>/dev/null; then
        # The recorded process was already stopped outside this script.
        if ! kill -TERM "$recorded" 2>/dev/null; then
          echo "error: cannot stop recorded dashboard process $recorded" >&2
          return 1
        fi
        pids+=("$recorded")
      fi
    fi
  fi

  if ((${#pids[@]} > 0)); then
    echo "Stopping dashboard on port $PORT (pid(s): ${pids[*]})"
    kill -TERM "${pids[@]}" 2>/dev/null || true
    local deadline=$((SECONDS + 10))
    while ((SECONDS < deadline)); do
      local alive=0 pid
      for pid in "${pids[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then alive=1; fi
      done
      ((alive == 0)) && break
      sleep 0.2
    done
    local pid
    for pid in "${pids[@]}"; do
      if kill -0 "$pid" 2>/dev/null; then
        echo "Force-stopping dashboard process $pid"
        kill -KILL "$pid" 2>/dev/null || true
      fi
    done
  fi

  # Do not bind a new server until the old listener has released the port.
  local deadline=$((SECONDS + 10))
  while lsof -nP -iTCP:"$PORT" -sTCP:LISTEN -t >/dev/null 2>&1; do
    if ((SECONDS >= deadline)); then
      echo "error: port $PORT is still in use; not starting a duplicate" >&2
      return 1
    fi
    sleep 0.2
  done
}

stop_existing
rm -f "$PID_FILE"
: >"$LOG_FILE"
.venv/bin/python -m observe >>"$LOG_FILE" 2>&1 &
NEW_PID=$!
echo "$NEW_PID" >"$PID_FILE"

cleanup() {
  if [[ -n "${NEW_PID:-}" ]] && kill -0 "$NEW_PID" 2>/dev/null; then
    echo "Startup failed; stopping new dashboard process $NEW_PID" >&2
    kill -TERM "$NEW_PID" 2>/dev/null || true
    wait "$NEW_PID" 2>/dev/null || true
  fi
  rm -f "$PID_FILE"
}
trap cleanup ERR

echo "Starting dashboard: $URL (pid $NEW_PID, log: $LOG_FILE)"
for _ in {1..60}; do
  if curl --silent --show-error --fail --max-time 2 "$URL/api/v2/overview" >/dev/null 2>&1; then
    trap - ERR
    echo "Dashboard is ready: $URL"
    echo "Restart later with: scripts/observe.sh"
    exit 0
  fi
  if ! kill -0 "$NEW_PID" 2>/dev/null; then
    echo "error: dashboard exited during startup; see $LOG_FILE" >&2
    tail -40 "$LOG_FILE" >&2 || true
    exit 1
  fi
  sleep 0.5
done

echo "error: dashboard did not become healthy within 30 seconds; see $LOG_FILE" >&2
tail -40 "$LOG_FILE" >&2 || true
exit 1
