"""Repository-owned tool registry; only reviewed handlers can be loaded."""

from __future__ import annotations

import importlib
import json
import os
from pathlib import Path

from .manifest import ToolError, ToolManifest, load_manifest

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REGISTRY = ROOT / "backends" / "metal" / "tools.json"


def load_registry(path: Path | None = None) -> dict:
    path = Path(os.environ.get("TCENV_TOOL_REGISTRY", path or DEFAULT_REGISTRY)).resolve()
    # The worker inherits a scrubbed environment, but a user may not redirect a broker to
    # an unreviewed manifest. A custom registry is for offline tests and must opt in explicitly.
    if path != DEFAULT_REGISTRY.resolve() and os.environ.get("TCENV_TOOL_REGISTRY_TEST") != "1":
        raise ToolError("a custom tool registry is only allowed in offline tests")
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ToolError(f"cannot read tool registry: {exc}") from exc
    if doc.get("schema_version") != 1 or not isinstance(doc.get("tools"), dict):
        raise ToolError("unsupported tool registry")
    return doc


def get_tool(tool_id: str, *, registry_path: Path | None = None) -> ToolManifest:
    document = load_registry(registry_path)
    entry = document["tools"].get(tool_id)
    if not isinstance(entry, dict) or entry.get("tool_id") != tool_id:
        raise ToolError(f"unknown tool {tool_id}")
    manifest_path = Path(str(entry.get("manifest") or ""))
    if manifest_path.is_absolute() or ".." in manifest_path.parts:
        raise ToolError("tool manifest path must stay inside the repository")
    manifest = load_manifest(ROOT / manifest_path, expected_digest=entry.get("manifest_sha256"))
    if str(entry.get("version") or "") != manifest.version:
        raise ToolError("registry and tool versions disagree")
    return manifest


def load_handler(manifest: ToolManifest):
    """The module path comes from the verified manifest, never from the request."""
    module_name, function = manifest.handler.split(":")
    try:
        handler = getattr(importlib.import_module(module_name), function)
    except (ImportError, AttributeError) as exc:
        raise ToolError(f"cannot load approved tool handler: {exc}") from exc
    if not callable(handler):
        raise ToolError("registered tool handler is not callable")
    return handler


def role_capabilities(role: str) -> frozenset[str]:
    """Capabilities assigned by the host to a session role, not by model text."""
    if role == "kernel_once":
        return frozenset({"probe:execute", "tool:invoke"})
    if role == "code_patch":
        return frozenset({"harness:implement", "tool:propose"})
    if role == "offline_test":
        return frozenset(CAPABILITY for CAPABILITY in ("probe:execute", "tool:invoke", "tool:propose", "harness:implement"))
    return frozenset()
