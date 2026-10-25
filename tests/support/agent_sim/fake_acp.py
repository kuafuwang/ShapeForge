"""A fake ACP agent that runs inside the isolated Docker host.

It writes a fixed kernel into its workspace, reads the short-lived credential the
control-host helper provisioned at ``/run/taic/credential.json``, and calls the
control-host fake provider over ``host.docker.internal``.  The evidence (provider status and
body) is written to ``NOTES.md`` so the host runner can check it after collection.

This is a deterministic test fixture: it never contacts a real provider, never reads a real
key, and never invokes a model.
"""

from __future__ import annotations

import json
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

CREDENTIAL_PATH = Path("/run/taic/credential.json")


def _send(document: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(document, ensure_ascii=False, separators=(",", ":")) + "\n")
    sys.stdout.flush()


def _read() -> dict[str, Any] | None:
    line = sys.stdin.readline()
    if not line:
        return None
    try:
        value = json.loads(line)
    except ValueError:
        return None
    return value if isinstance(value, dict) else None


def _submission(prompt: str) -> str:
    match = re.search(r"Write the finished kernel to ([A-Za-z0-9._-]+)", prompt)
    return match.group(1).rstrip(".") if match else "submit.metal"


def _provider_call(document: dict[str, Any]) -> dict[str, Any]:
    """One request to the control-host fake provider, with no proxy and no long-term key."""
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    base_url = document.get("base_url") or document["provider_base_url"]
    request = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=json.dumps({"prompt": "isolated agent probe"}).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {document['token']}",
        },
        method="POST",
    )
    try:
        with opener.open(request, timeout=10) as response:
            body = json.loads(response.read().decode("utf-8"))
            return {
                "provider_status": response.status,
                "provider_body": body.get("choices", [{}])[0].get("message", {}).get("content"),
            }
    except urllib.error.HTTPError as exc:
        return {"provider_status": exc.code, "provider_error": exc.read().decode("utf-8", "replace")[:200]}
    except Exception as exc:  # noqa: BLE001 - recorded as evidence, not raised
        return {"provider_error": f"{type(exc).__name__}: {exc}"}


def _probe(prompt: str) -> dict[str, Any]:
    root = Path.cwd()
    submit = _submission(prompt)
    (root / submit).write_text("kernel void demo() { /* isolated fake agent */ }\n", encoding="utf-8")
    evidence: dict[str, Any] = {"cwd": str(root), "submit": submit}
    if CREDENTIAL_PATH.is_file():
        document = json.loads(CREDENTIAL_PATH.read_text(encoding="utf-8"))
        evidence["credential_present"] = True
        evidence.update(_provider_call(document))
        evidence["credential_token_prefix"] = ""  # never persist even a prefix
    else:
        evidence["credential_present"] = False
    (root / "NOTES.md").write_text(json.dumps(evidence, sort_keys=True) + "\n", encoding="utf-8")
    return evidence


def serve() -> int:
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
                    "result": {"protocolVersion": 1, "agentCapabilities": {"loadSession": False, "sessionCapabilities": {}}},
                }
            )
        elif method == "session/new":
            _send({"jsonrpc": "2.0", "id": at, "result": {"sessionId": session_id}})
        elif method == "session/prompt":
            if not prompted:
                prompted = True
                try:
                    _probe(str((message.get("params") or {}).get("prompt") or ""))
                except Exception as exc:  # noqa: BLE001 - the host needs the failure as evidence
                    Path("NOTES.md").write_text(json.dumps({"probe_error": f"{type(exc).__name__}: {exc}"}) + "\n", encoding="utf-8")
            _send(
                {
                    "jsonrpc": "2.0",
                    "method": "session/update",
                    "params": {
                        "sessionId": session_id,
                        "update": {
                            "sessionUpdate": "agent_message_chunk",
                            "content": {"type": "text", "text": "isolated fake agent done"},
                        },
                    },
                }
            )
            _send({"jsonrpc": "2.0", "id": at, "result": {"stopReason": "end_turn"}})
        elif method == "session/cancel":
            continue
        elif at is not None:
            _send({"jsonrpc": "2.0", "id": at, "error": {"code": -32601, "message": f"unsupported method {method}"}})


if __name__ == "__main__":
    raise SystemExit(serve())
