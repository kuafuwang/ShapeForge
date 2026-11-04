"""Static, versioned tool manifests. Model input cannot add or replace a handler."""

from __future__ import annotations

import hashlib
import json
import re
import tomllib
from dataclasses import dataclass
from pathlib import Path

SCHEMA_VERSION = 1
CAPABILITIES = frozenset({"probe:execute", "tool:invoke", "tool:propose", "harness:implement"})


class ToolError(ValueError):
    """A manifest, invocation, result or capability check failed."""


@dataclass(frozen=True)
class ToolManifest:
    tool_id: str
    version: str
    handler: str
    request_schema: dict
    result_schema: dict
    capabilities: tuple[str, ...]
    limits: dict
    digest: str
    handler_source: str
    handler_source_sha256: str


def canonical_digest(document: dict) -> str:
    return hashlib.sha256(json.dumps(document, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def load_manifest(path: Path, *, expected_digest: str | None = None) -> ToolManifest:
    """Load TOML without importing a module. Unknown capabilities or handlers fail closed."""
    path = Path(path)
    try:
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
        impl = tomllib.loads(path.with_name("tool_impl.toml").read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise ToolError(f"cannot read tool manifest {path}: {exc}") from exc
    if raw.get("schema_version") != SCHEMA_VERSION or impl.get("schema_version") != SCHEMA_VERSION:
        raise ToolError("unsupported tool manifest schema")
    tool_id, version, handler = (raw.get(key) for key in ("tool_id", "version", "handler"))
    if not isinstance(tool_id, str) or not tool_id or any(ch not in "abcdefghijklmnopqrstuvwxyz0123456789_-." for ch in tool_id):
        raise ToolError("invalid tool_id")
    if not isinstance(version, str) or not version or not isinstance(handler, str) or handler.count(":") != 1:
        raise ToolError("tool version and module:function handler are required")
    module, function = handler.split(":")
    if module != "tcenv.backends.metal.probes" or function != "time_msl":
        raise ToolError("a tool handler must name an approved repository probe")
    if impl.get("tool_id") != tool_id or impl.get("handler") != handler or impl.get("version") != version:
        raise ToolError("tool manifest and implementation binding disagree")
    handler_source = impl.get("handler_source")
    source_digest = impl.get("handler_source_sha256")
    source_path = Path(handler_source) if isinstance(handler_source, str) else Path(".")
    if not isinstance(handler_source, str) or source_path.is_absolute() or ".." in source_path.parts or not isinstance(source_digest, str) or not re.fullmatch(r"[0-9a-f]{64}", source_digest):
        raise ToolError("handler source must be a repository-relative file with a sha256")
    if module.replace(".", "/") != handler_source.removesuffix(".py"):
        raise ToolError("handler source path does not match the registered module")
    try:
        actual_source = (Path(__file__).resolve().parents[2] / source_path).read_bytes()
    except OSError as exc:
        raise ToolError(f"cannot read registered handler source: {exc}") from exc
    if hashlib.sha256(actual_source).hexdigest() != source_digest:
        raise ToolError("handler source differs from its registered digest")
    request_schema, result_schema = raw.get("request_schema"), raw.get("result_schema")
    if not isinstance(request_schema, dict) or not isinstance(result_schema, dict):
        raise ToolError("tool schemas must be objects")
    if request_schema.get("schema_version") != 1 or result_schema.get("schema_version") != 1:
        raise ToolError("unsupported request/result schema")
    capabilities = raw.get("capabilities")
    if not isinstance(capabilities, list) or not capabilities or any(capability not in CAPABILITIES for capability in capabilities):
        raise ToolError("unknown or missing tool capability")
    limits = raw.get("limits")
    if not isinstance(limits, dict):
        raise ToolError("tool limits must be an object")
    for key in ("max_source_bytes", "max_iters", "max_buffer_bytes", "max_grid_dim", "max_threads_dim", "max_threadgroups"):
        if not isinstance(limits.get(key), int) or isinstance(limits.get(key), bool) or limits[key] < 1:
            raise ToolError(f"invalid tool limit {key}")
    if limits["max_source_bytes"] > 128_000 or limits["max_iters"] > 20 or limits["max_buffer_bytes"] > 1_048_576 or limits["max_threadgroups"] > 65_536 or limits["max_grid_dim"] > 4096 or limits["max_threads_dim"] > 1024:
        raise ToolError("tool limits exceed the broker's hard caps")
    digest = str(impl.get("manifest_sha256") or "")
    if digest != canonical_digest(raw):
        raise ToolError("tool manifest digest does not match its content")
    if expected_digest is not None and digest != expected_digest:
        raise ToolError("the approved tool manifest digest differs from the registry")
    return ToolManifest(tool_id, version, handler, request_schema, result_schema, tuple(capabilities), limits, digest, handler_source, source_digest)


def validate_schema(document: dict, schema: dict, *, what: str) -> None:
    """A deliberately small typed-request schema; no arbitrary JSON-Schema evaluation."""
    if not isinstance(document, dict) or document.get("schema_version") != schema.get("schema_version"):
        raise ToolError(f"{what} has an unsupported schema_version")
    allowed = set(schema.get("fields") or {})
    required = set(schema.get("required") or ())
    if not allowed or not required <= allowed:
        raise ToolError(f"{what} schema must declare its fields")
    if set(document) - allowed or required - set(document):
        raise ToolError(f"{what} has unknown or missing fields")
    types = schema.get("types") or {}
    mapping = {"string": str, "integer": int, "number": (int, float), "boolean": bool, "array": list, "object": dict}
    if set(allowed) - set(types):
        raise ToolError(f"{what} schema does not type every field")
    for name, kind in types.items():
        if name not in document:
            continue
        if kind not in mapping:
            raise ToolError(f"{what} schema has an unknown type {kind}")
        value = document[name]
        expected = mapping[kind]
        if isinstance(value, bool) and kind != "boolean" or not isinstance(value, expected):
            raise ToolError(f"{what}.{name} must be {kind}")
    for name, limit in (schema.get("limits") or {}).items():
        if name in document and isinstance(limit, int) and isinstance(document[name], (str, list, dict)) and len(document[name]) > limit:
            raise ToolError(f"{what}.{name} exceeds its size limit")


def validated_result(manifest: ToolManifest, result: dict) -> dict:
    validate_schema(result, manifest.result_schema, what="tool result")
    if result["status"] not in {"ok", "rejected", "error", "timeout"}:
        raise ToolError("unknown tool result status")
    if len(result["stdout_excerpt"]) > 4000 or len(result["error"]) > 4000:
        raise ToolError("tool text result exceeds its bound")
    if set(result["metrics"]) != set(result["units"]):
        raise ToolError("every tool metric needs a unit")
    for name, value in result["metrics"].items():
        values = value if isinstance(value, list) else [value]
        if not values or any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in values):
            raise ToolError(f"tool metric {name} must be numeric")
    return result
