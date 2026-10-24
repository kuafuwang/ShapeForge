"""Offline tool-proposal scaffolds and reviewed manifest digest maintenance."""

from __future__ import annotations

import hashlib
import re
import tomllib
from pathlib import Path

from .manifest import ToolError, canonical_digest


def scaffold_proposal(proposal: dict, out: Path) -> dict:
    """Write inert drafts only. No registry edit, import, execution or capability grant."""
    if not isinstance(proposal, dict) or not isinstance(proposal.get("action"), str):
        raise ToolError("a tool scaffold needs a proposal with an action")
    evidence = proposal.get("evidence") if isinstance(proposal.get("evidence"), dict) else {}
    if not evidence.get("reproduction") or not evidence.get("expected"):
        raise ToolError("a tool scaffold needs reproduction and expected evidence")
    scope = proposal.get("scope") if isinstance(proposal.get("scope"), dict) else {}
    if not scope.get("target_id") or not scope.get("task"):
        raise ToolError("a tool scaffold needs target/task scope")
    out = Path(out)
    if out.exists():
        raise ToolError("the scaffold output must be a new directory")
    out.mkdir(parents=True)
    tool_id = "draft.time_msl"
    (out / "proposal.json").write_text(__import__("json").dumps(proposal, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    handler_text = '"""Draft only. Review and register explicitly before execution."""\n\n\ndef probe(request, *, manifest):\n    raise NotImplementedError("draft handler has no execution permission")\n'
    (out / "handler.py").write_text(handler_text, encoding="utf-8")
    manifest_text = f'schema_version = 1\ntool_id = "{tool_id}"\nversion = "draft"\nhandler = "handler:probe"\ncapabilities = []\n'
    (out / "manifest.toml").write_text(manifest_text, encoding="utf-8")
    import json
    binding = {
        "schema_version": 1, "tool_id": tool_id, "version": "draft", "handler": "handler:probe",
        "handler_source": "handler.py", "handler_source_sha256": hashlib.sha256(handler_text.encode()).hexdigest(),
        "manifest_sha256": canonical_digest(tomllib.loads(manifest_text)),
    }
    (out / "tool_impl.toml").write_text("\n".join(f"{key} = {json.dumps(value)}" for key, value in binding.items()) + "\n", encoding="utf-8")
    (out / "fixture.json").write_text(json.dumps({
        "schema_version": 1,
        "request": {"schema_version": 1, "tool_id": tool_id, "version": "draft"},
        "expected": {"status": "draft", "executable": False},
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out / "protocol_test.py").write_text('"""Move reviewed tests to tests/backends/metal/ before registration."""\n\nfrom pathlib import Path\nimport tomllib\n\n\ndef test_draft_has_no_capabilities_or_permission():\n    manifest = tomllib.loads(Path("manifest.toml").read_text())\n    assert manifest["capabilities"] == []\n    assert manifest["version"] == "draft"\n', encoding="utf-8")
    return {"tool_id": tool_id, "out": str(out), "registered": False, "capabilities": [], "executable": False}


def refresh_binding(registry: Path) -> dict:
    """Recompute registration digests for an existing reviewed Metal tool, never add one."""
    from .registry import ROOT

    registry = Path(registry)
    document = tomllib.loads(registry.read_text(encoding="utf-8")) if registry.suffix == ".toml" else __import__("json").loads(registry.read_text(encoding="utf-8"))
    tools = document.get("tools") if isinstance(document, dict) else None
    if not isinstance(tools, dict) or len(tools) != 1:
        raise ToolError("digest refresh supports exactly one registered tool")
    tool_id, entry = next(iter(tools.items()))
    manifest_path = ROOT / str(entry["manifest"])
    raw = manifest_path.read_text(encoding="utf-8")
    binding_path = manifest_path.with_name("tool_impl.toml")
    binding = tomllib.loads(binding_path.read_text(encoding="utf-8"))
    handler = binding.get("handler")
    if not isinstance(handler, str) or ":" not in handler:
        raise ToolError("registered handler must be module:function")
    module = handler.split(":", 1)[0]
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.]*", module):
        raise ToolError("invalid handler module")
    source = Path(*module.split(".")).with_suffix(".py")
    source_path = ROOT / source
    if not source_path.is_file():
        raise ToolError("registered handler source is missing")
    binding["schema_version"] = 1
    binding["tool_id"] = tool_id
    binding["version"] = entry["version"]
    binding["handler_source"] = source.as_posix()
    binding["handler_source_sha256"] = hashlib.sha256(source_path.read_bytes()).hexdigest()
    binding["manifest_sha256"] = canonical_digest(tomllib.loads(raw))
    binding_path.write_text("\n".join(f'{key} = {__import__("json").dumps(value, ensure_ascii=False)}' for key, value in binding.items()) + "\n", encoding="utf-8")
    entry["manifest_sha256"] = binding["manifest_sha256"]
    registry.write_text(__import__("json").dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    from .registry import get_tool
    get_tool(tool_id)
    return {"tool_id": tool_id, "manifest_sha256": binding["manifest_sha256"], "handler_source_sha256": binding["handler_source_sha256"]}
