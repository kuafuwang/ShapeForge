"""A fake ACP agent for the real-hmz selfcheck fence acceptance.

The script is intentionally standalone: hmz launches it as an added ACP CLI
under the same fence as a real coding agent.  It never calls a model, GPU or
network service.  On its first turn it writes a submission, runs the exact
``taic.selfcheck_bridge`` command rendered in ``TASK.md``, and records the
observable facts in ``NOTES.md`` for the host test.

stdout is reserved for ACP JSON-RPC messages.  Diagnostics go into NOTES.md
instead, so this file stays usable with the real hmz process supervisor.
"""

from __future__ import annotations

import json
import os
import re
import socket
import stat
import subprocess
import sys
from pathlib import Path
from typing import Any


def _send(document: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(document, ensure_ascii=False, separators=(",", ":")) + "\n")
    sys.stdout.flush()


def _read() -> dict[str, Any] | None:
    line = sys.stdin.readline()
    if not line:
        return None
    value = json.loads(line)
    return value if isinstance(value, dict) else None


def _request(socket_path: str, token: str, mode: str) -> dict[str, Any]:
    payload = json.dumps(
        {"mode": mode, "source": "kernel void demo() {}\n", "token": token},
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8") + b"\n"
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
        sock.settimeout(5.0)
        sock.connect(socket_path)
        sock.sendall(payload)
        raw = b""
        while b"\n" not in raw:
            chunk = sock.recv(65_536)
            if not chunk:
                break
            raw += chunk
    value = json.loads(raw.split(b"\n", 1)[0].decode("utf-8"))
    return value if isinstance(value, dict) else {"ok": False, "error": {"code": "invalid_response"}}


def _submission(prompt: str) -> str:
    match = re.search(r"Write the finished kernel to ([A-Za-z0-9._-]+)", prompt)
    return match.group(1).rstrip(".") if match else "relu-fp16-small-init.metal"


def _probe(prompt: str) -> dict[str, Any]:
    root = Path.cwd()
    task = (root / "TASK.md").read_text(encoding="utf-8")
    submit = _submission(prompt)
    (root / submit).write_text("kernel void demo() {}\n", encoding="utf-8")

    commands = [
        line.strip()
        for line in task.splitlines()
        if line.strip().startswith("uv ")
        and "taic.selfcheck_bridge" in line
        and "--quick" in line
    ]
    socket_path = os.environ.get("TAIC_SELFCHECK_SOCKET", "")
    token = os.environ.get("TAIC_SELFCHECK_TOKEN", "")
    command = commands[0] if commands else ""
    completed = subprocess.run(
        command,
        shell=True,
        cwd=root,
        env=dict(os.environ),
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    canary = Path(socket_path).parent / f"agent-write-canary-{os.getpid()}"
    outside_write_refused = False
    try:
        canary.write_text("agent reached the host socket directory", encoding="utf-8")
    except OSError:
        outside_write_refused = True
    finally:
        canary.unlink(missing_ok=True)

    tcp_port = os.environ.get("TAIC_TEST_HMZ_TCP_PORT", "")
    network_connect_succeeded = False
    if tcp_port:
        try:
            with socket.create_connection(("127.0.0.1", int(tcp_port)), timeout=0.5):
                network_connect_succeeded = True
        except OSError:
            pass

    other_socket = os.environ.get("TAIC_TEST_HMZ_OTHER_SOCKET", "")
    cross_session = (
        _request(other_socket, token, "quick")
        if other_socket and token
        else {"ok": False, "error": {"code": "not_configured"}}
    )
    formal = _request(socket_path, token, "formal_eval") if socket_path and token else {"ok": False, "error": {"code": "not_configured"}}
    wrong = _request(socket_path, "wrong-token", "quick") if socket_path else {"ok": False, "error": {"code": "not_configured"}}

    ppid = os.getppid()
    try:
        ppid_command = subprocess.run(
            ["ps", "-o", "command=", "-p", str(ppid)],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        ).stdout.strip()[-500:]
    except OSError:
        ppid_command = ""
    return {
        "prompt_count": 1,
        "cwd": str(root),
        "ppid": ppid,
        "ppid_command": ppid_command,
        "hmz_environment": sorted(name for name in os.environ if name.startswith("HMZ") or name.startswith("HUMANIZE")),
        "command_found": bool(command),
        "selfcheck_rc": completed.returncode,
        "selfcheck_stdout_tail": completed.stdout[-2000:],
        "selfcheck_stderr_tail": completed.stderr[-2000:],
        "socket_env_present": bool(socket_path),
        "token_env_present": bool(token),
        "token_file_env_present": bool(os.environ.get("TAIC_SELFCHECK_TOKEN_FILE")),
        "socket_in_task": bool(socket_path and socket_path in task),
        "token_in_task": bool(token and token in task),
        "socket_in_command": bool(socket_path and socket_path in command),
        "token_file_in_command": "--token-file" in command,
        "socket_arg_in_command": "--socket" in command,
        "workspace_token_file_exists": (root / ".selfcheck" / "token").exists(),
        "socket_mode": stat.S_IMODE(Path(socket_path).stat().st_mode) if socket_path else None,
        "socket_dir_mode": stat.S_IMODE(Path(socket_path).parent.stat().st_mode) if socket_path else None,
        "outside_write_refused": outside_write_refused,
        "network_connect_succeeded": network_connect_succeeded,
        "cross_session_error": (cross_session.get("error") or {}).get("code"),
        "formal_mode_error": (formal.get("error") or {}).get("code"),
        "wrong_token_error": (wrong.get("error") or {}).get("code"),
    }


def _serve() -> int:
    session_id = "fake-acp-session"
    prompted = False
    while True:
        message = _read()
        if message is None:
            return 0
        method = str(message.get("method") or "")
        at = message.get("id")
        if method == "initialize":
            _send(
                {
                    "jsonrpc": "2.0",
                    "id": at,
                    "result": {
                        "protocolVersion": 1,
                        "agentCapabilities": {"loadSession": False, "sessionCapabilities": {}},
                    },
                }
            )
        elif method == "session/new":
            _send({"jsonrpc": "2.0", "id": at, "result": {"sessionId": session_id}})
        elif method == "session/prompt":
            if not prompted:
                prompted = True
                try:
                    evidence = _probe(str((message.get("params") or {}).get("prompt") or ""))
                except Exception as exc:  # noqa: BLE001 - the host needs the failure as evidence
                    evidence = {"probe_error": f"{type(exc).__name__}: {exc}"}
                Path("NOTES.md").write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            _send(
                {
                    "jsonrpc": "2.0",
                    "method": "session/update",
                    "params": {
                        "sessionId": session_id,
                        "update": {
                            "sessionUpdate": "agent_message_chunk",
                            "content": {"type": "text", "text": "fake ACP selfcheck probe complete"},
                        },
                    },
                }
            )
            _send({"jsonrpc": "2.0", "id": at, "result": {"stopReason": "end_turn"}})
        elif method == "session/cancel":
            continue
        elif at is not None:
            _send(
                {
                    "jsonrpc": "2.0",
                    "id": at,
                    "error": {"code": -32601, "message": f"unsupported method {method}"},
                }
            )


if __name__ == "__main__":
    raise SystemExit(_serve())
