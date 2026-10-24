"""Host-side typed invocation, resource limits, process cleanup and evidence output."""

from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
import time
from pathlib import Path

from tcenv.children import forget, kill_group, popen_group, scrubbed_env
from tcenv.gpulock import released
from tcenv.schema import write_json_atomic

from .manifest import ToolError, canonical_digest, validate_schema
from .registry import ROOT, get_tool, role_capabilities

MAX_SOURCE_BYTES = 128_000
MAX_STDOUT_BYTES = 4_000
MAX_ERROR_BYTES = 4_000
DEFAULT_TIMEOUT_S = 120.0


def request_digest(request: dict) -> str:
    return canonical_digest(request)


def validate_request(manifest, request: dict, *, session_role: str = "") -> dict:
    if "role" in request:
        raise ToolError("role is assigned by the host, never by the model request")
    validate_schema(request, manifest.request_schema, what="tool request")
    if request["tool_id"] != manifest.tool_id or request["version"] != manifest.version:
        raise ToolError("request and approved tool identity disagree")
    if not manifest.capabilities or not set(manifest.capabilities) <= role_capabilities(session_role):
        raise ToolError("the host role has not granted this tool's required capabilities")
    if request.get("manifest_sha256") != manifest.digest:
        raise ToolError("request does not bind the current approved tool manifest")
    limits = manifest.limits
    if len(request["source"].encode("utf-8")) > min(MAX_SOURCE_BYTES, int(limits["max_source_bytes"])):
        raise ToolError("probe source exceeds the manifest byte limit")
    if not 1 <= request["iters"] <= int(limits["max_iters"]):
        raise ToolError("probe iteration count exceeds the manifest limit")
    if not 1 <= request["buffer_bytes"] <= int(limits["max_buffer_bytes"]) or request["buffer_bytes"] % 4:
        raise ToolError("probe buffer size must be a bounded multiple of four")
    for name in ("grid", "threads"):
        shape = request[name]
        if len(shape) != 3 or any(isinstance(v, bool) or not isinstance(v, int) or v < 1 or v > int(limits[f"max_{name}_dim"]) for v in shape):
            raise ToolError(f"{name} must contain three bounded positive integer dimensions")
    if request["grid"][0] * request["grid"][1] * request["grid"][2] > int(limits["max_threadgroups"]):
        raise ToolError("probe grid exceeds the manifest threadgroup limit")
    if not isinstance(request["kernel_name"], str) or not request["kernel_name"].isidentifier():
        raise ToolError("kernel_name must be one identifier")
    if not isinstance(request["target_id"], str) or not request["target_id"]:
        raise ToolError("target_id is required")
    return request


def invoke(request: dict, *, session_role: str = "", timeout_s: float = DEFAULT_TIMEOUT_S) -> dict:
    """Invoke a registered tool. No LLM or GPU is called during validation alone."""
    manifest = get_tool(str(request.get("tool_id") or ""))
    request = validate_request(manifest, request, session_role=session_role)
    if not isinstance(timeout_s, (int, float)) or isinstance(timeout_s, bool) or not 0 < timeout_s <= 120:
        raise ToolError("tool timeout must be in (0, 120] seconds")
    digest = request_digest(request)
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="tcenv_tool_") as td:
        folder = Path(td)
        result_path = folder / "result.json"
        heartbeat = folder / "heartbeat"
        request_path = folder / "request.json"
        child_request = {"tool_id": manifest.tool_id, "session_role": session_role, "request": request, "request_digest": digest, "out": str(result_path), "heartbeat": str(heartbeat)}
        write_json_atomic(request_path, child_request)
        env = scrubbed_env()
        env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
        for key in ("TCENV_TOOL_REGISTRY", "TCENV_TOOL_REGISTRY_TEST"):
            env.pop(key, None)
        with (folder / "stdout.txt").open("w") as stdout, (folder / "stderr.txt").open("w") as stderr:
            proc = popen_group([sys.executable, "-m", "tcenv.tools.worker", str(request_path)], cwd=ROOT, env=env, stdout=stdout, stderr=stderr)
            try:
                from tcenv.evaluate import supervise
                with released():
                    reason = supervise(proc, heartbeat, launch_deadline_s=2.0, timeout=timeout_s, compile_deadline_s=min(90.0, timeout_s))
            finally:
                kill_group(proc)
                forget(proc)
        try:
            result = json.loads(result_path.read_text(encoding="utf-8")) if result_path.is_file() else None
        except (OSError, json.JSONDecodeError):
            result = None
        stdout_excerpt = (folder / "stdout.txt").read_text(encoding="utf-8", errors="replace")[-MAX_STDOUT_BYTES:]
        if reason:
            status, error = "timeout", f"tool supervisor stopped the worker: {reason}"
        elif not isinstance(result, dict):
            status, error = "error", "tool worker left no valid host-readable result"
        else:
            from .manifest import validated_result
            result = validated_result(manifest, result)
            if result["request_digest"] != digest or result["tool_manifest_sha256"] != manifest.digest or result["tool_id"] != manifest.tool_id or result["version"] != manifest.version:
                raise ToolError("tool result does not bind the request and approved manifest")
            status, error = result["status"], result["error"]
        result = {
            "schema_version": 1, "tool_id": manifest.tool_id, "version": manifest.version,
            "request_digest": digest, "tool_manifest_sha256": manifest.digest,
            "handler_source_sha256": manifest.handler_source_sha256,
            "status": status, "metrics": result.get("metrics") if isinstance(result, dict) else {},
            "units": result.get("units") if isinstance(result, dict) else {},
            "target_id": request["target_id"], "protocol_version": "3",
            "stdout_excerpt": stdout_excerpt, "error": str(error)[:MAX_ERROR_BYTES],
            "elapsed_s": round(time.perf_counter() - started, 6),
        }
        return result


def write_evidence(run_dir: Path, request: dict, result: dict) -> Path:
    """Store host-produced evidence atomically and bind it to the source/request digest."""
    from .manifest import get_tool
    manifest = get_tool(result["tool_id"])
    from .manifest import validated_result
    validated_result(manifest, result)
    if result["request_digest"] != request_digest(request):
        raise ToolError("cannot record evidence for a different request")
    folder = Path(run_dir) / "tool_evidence"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{result['request_digest']}.json"
    write_json_atomic(path, {"schema_version": 1, "request": request, "result": result})
    return path
