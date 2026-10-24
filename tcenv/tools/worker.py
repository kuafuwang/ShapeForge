"""Isolated tool worker. The parent supervises heartbeats and writes the result."""

from __future__ import annotations

import hashlib
import json
import sys
import traceback
from pathlib import Path


def main() -> int:
    from tcenv.schema import write_json_atomic
    from tcenv.worker import _install_heartbeat

    raw = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    out = Path(raw["out"])
    from .manifest import ToolError, validated_result
    from .registry import get_tool, load_handler, role_capabilities

    try:
        manifest = get_tool(raw["tool_id"])
        required = set(manifest.capabilities)
        if not required <= role_capabilities(raw["session_role"]):
            raise ToolError("the worker role lacks the approved tool capabilities")
        if hashlib.sha256(json.dumps(raw["request"], sort_keys=True, separators=(",", ":")).encode()).hexdigest() != raw["request_digest"]:
            raise ToolError("request digest mismatch")
        _install_heartbeat(raw.get("heartbeat"))
        handler = load_handler(manifest)
        result = validated_result(manifest, handler(raw["request"], manifest=manifest))
        write_json_atomic(out, result)
        return 0 if result["status"] == "ok" else 1
    except Exception as exc:  # noqa: BLE001 - a tool crash is data, never a host crash
        result = {
            "schema_version": 1, "tool_id": str(raw.get("tool_id") or ""), "version": "",
            "request_digest": str(raw.get("request_digest") or ""),
            "tool_manifest_sha256": "", "status": "error", "metrics": {}, "units": {},
            "target_id": "", "protocol_version": "", "stdout_excerpt": "",
            "error": f"{type(exc).__name__}: {exc}\n" + "".join(traceback.format_exc().splitlines(True)[-4:]),
        }
        write_json_atomic(out, result)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
