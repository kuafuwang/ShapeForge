#!/usr/bin/env python3
"""Strict SSH forced-command entry for the NPU simulation evaluator.

Only a fixed command name and two bounded identifiers are accepted. Candidate
source is never accepted as an argument, shell fragment or path. The command
always starts the same container with the real RemoteService core and the
 test-only FakeNpuBackend composition root.
"""

from __future__ import annotations

import os
import re
import select
import shlex
import signal
import subprocess
import sys
import threading
import uuid
from pathlib import Path

SAFE_ID = re.compile(r"^[A-Za-z0-9._:-]{1,64}$")
ENTRY_NAMES = {"/usr/local/bin/taic-remote-evaluator", "taic-remote-evaluator"}


def _arguments() -> tuple[str, str] | None:
    original = os.environ.get("SSH_ORIGINAL_COMMAND", "")
    try:
        parts = shlex.split(original)
    except ValueError:
        return None
    if not parts or parts[0] not in ENTRY_NAMES:
        return None
    values = {"run_id": "run-1", "session_id": "session-1"}
    if len(parts) == 1:
        return values["run_id"], values["session_id"]
    if len(parts) != 5:
        return None
    index = 1
    seen = set()
    while index < len(parts):
        flag = parts[index]
        if flag not in {"--run-id", "--session-id"} or flag in seen:
            return None
        seen.add(flag)
        index += 1
        if index >= len(parts):
            return None
        value = parts[index]
        if not SAFE_ID.fullmatch(value):
            return None
        values[flag[2:].replace("-", "_")] = value
        index += 1
    return values["run_id"], values["session_id"]


def main() -> int:
    values = _arguments()
    if values is None:
        print("forced command accepts only taic-remote-evaluator with bounded run/session ids", file=sys.stderr)
        return 64
    run_id, session_id = values
    container = "taic-npu-sim-" + uuid.uuid4().hex[:16]
    state = Path.home() / "taic-state"
    state.mkdir(parents=True, exist_ok=True)
    argv = [
        "/usr/bin/docker", "run", "--rm", "-i", "--name", container,
        "--network", "none", "--read-only", "--cap-drop", "ALL",
        "--security-opt", "no-new-privileges", "--pids-limit", "128",
        "--user", f"{os.getuid()}:{os.getgid()}",
        "--mount", f"type=bind,src={state},dst=/var/lib/tcenv",
        "--tmpfs", "/tmp:rw,noexec,nosuid,size=128m",
        "--env", "PYTHONPATH=/opt/taic",
        "--env", "HOME=/tmp",
        "--env", "NPU_SIM_COUNTER=/var/lib/tcenv/counter.txt",
        "taic-remote-sim:local",
        "python3", "-m", "tests.support.npu_sim.service_entry",
        "--state-dir", "/var/lib/tcenv/state",
        "--run-id", run_id,
        "--session-id", session_id,
        "--token", "test-session-token",
    ]
    proc = subprocess.Popen(argv)
    removing = threading.Event()

    def remove_container() -> None:
        if removing.is_set():
            return
        removing.set()
        try:
            subprocess.run(["/usr/bin/docker", "rm", "-f", container], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=10, check=False)
        except (OSError, subprocess.TimeoutExpired):
            pass

    def watch_disconnect() -> None:
        # POLLHUP tells us the SSH writer went away without consuming any frame
        # bytes. docker run -i can otherwise leave the container alive after the
        # client process is killed.
        poll = select.poll()
        try:
            poll.register(sys.stdin.fileno(), select.POLLHUP | select.POLLERR)
            while proc.poll() is None and not removing.is_set():
                if poll.poll(250):
                    remove_container()
                    return
        except (OSError, ValueError):
            return

    def stop(signum, _frame):
        remove_container()
        raise SystemExit(128 + signum)

    for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(signum, stop)
    watcher = threading.Thread(target=watch_disconnect, name="taic-remote-disconnect", daemon=True)
    watcher.start()
    try:
        return proc.wait()
    finally:
        remove_container()


if __name__ == "__main__":
    raise SystemExit(main())
