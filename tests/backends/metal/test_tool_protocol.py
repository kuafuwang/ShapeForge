"""Metal typed tool manifests and host-side request validation; no GPU."""

from __future__ import annotations

import json

import pytest

from tcenv.backends.metal import probes
from tcenv.tools.broker import invoke, validate_request
from tcenv.tools.manifest import ToolError, canonical_digest, load_manifest, validate_schema
from tcenv.tools.registry import get_tool, role_capabilities


def _request():
    manifest = get_tool("metal.time_msl")
    return {
        "schema_version": 1, "tool_id": "metal.time_msl", "version": "1",
        "manifest_sha256": manifest.digest, "target_id": "apple-m4-max", "kernel_name": "probe",
        "source": "kernel void probe(device float* out [[buffer(0)]], uint i [[thread_position_in_grid]]) { out[i] = 1.0f; }",
        "grid": [1, 1, 1], "threads": [32, 1, 1], "buffer_bytes": 4096, "iters": 4,
    }


def test_manifest_binds_registry_and_types():
    manifest = get_tool("metal.time_msl")
    assert manifest.handler == "tcenv.backends.metal.probes:time_msl"
    assert validate_request(manifest, _request(), session_role="kernel_once")["iters"] == 4
    assert role_capabilities("kernel_once") == {"probe:execute", "tool:invoke"}


def test_model_cannot_name_a_handler_or_grant_itself_a_scope():
    manifest = get_tool("metal.time_msl")
    request = _request()
    request["handler"] = "os:system"
    with pytest.raises(ToolError, match="unknown or missing fields"):
        validate_request(manifest, request, session_role="kernel_once")
    request = _request()
    request["role"] = "kernel_once"
    with pytest.raises(ToolError, match="host"):
        validate_request(manifest, request, session_role="kernel_once")
    request = _request()
    with pytest.raises(ToolError, match="not granted"):
        validate_request(manifest, request)
    request = _request()
    request["manifest_sha256"] = "0" * 64
    with pytest.raises(ToolError, match="current approved"):
        validate_request(manifest, request, session_role="kernel_once")


@pytest.mark.parametrize("field,value", [("iters", 21), ("buffer_bytes", 1048577), ("grid", [0, 1, 1]), ("threads", [1025, 1, 1]), ("iters", True)])
def test_probe_resources_are_bounded(field, value):
    request = _request()
    request[field] = value
    with pytest.raises(ToolError):
        validate_request(get_tool("metal.time_msl"), request, session_role="kernel_once")


def test_a_changed_manifest_is_rejected(tmp_path):
    source = get_tool("metal.time_msl")
    original = __import__("pathlib").Path("backends/metal/tools/metal_time_msl.toml")
    target = tmp_path / "metal_time_msl.toml"
    target.write_text(original.read_text())
    impl = __import__("pathlib").Path("backends/metal/tools/tool_impl.toml")
    (tmp_path / "tool_impl.toml").write_text(impl.read_text())
    load_manifest(target, expected_digest=source.digest)
    changed = target.read_text().replace("max_iters = 20", "max_iters = 200")
    target.write_text(changed)
    with pytest.raises(ToolError, match="hard caps|digest"):
        load_manifest(target, expected_digest=source.digest)


def test_schema_requires_a_type_for_every_field():
    with pytest.raises(ToolError, match="type every field"):
        validate_schema({"schema_version": 1}, {"schema_version": 1, "fields": ["schema_version", "answer"], "required": ["schema_version"], "types": {"schema_version": "integer"}}, what="request")


def test_screen_rejection_is_a_structured_probe_result():
    manifest = get_tool("metal.time_msl")
    request = _request()
    request["source"] = "kernel void probe(device float* out [[buffer(0)]]) { while (true) {} }"
    result = probes.time_msl(request, manifest=manifest)
    assert result["status"] == "rejected" and "unconditional loop" in result["error"]
    assert result["request_digest"] == canonical_digest(request)


def test_broker_cleans_up_the_worker_and_never_imports_a_request_handler(monkeypatch):
    class Proc:
        pid = 12345
        returncode = 0
    calls = []
    monkeypatch.setattr("tcenv.tools.broker.popen_group", lambda *args, **kwargs: calls.append(("spawn", args, kwargs)) or Proc())
    monkeypatch.setattr("tcenv.tools.broker.kill_group", lambda proc: calls.append(("kill", proc.pid)))
    monkeypatch.setattr("tcenv.tools.broker.forget", lambda proc: calls.append(("forget", proc.pid)))
    class Lock:
        def __enter__(self):
            calls.append(("release",))
        def __exit__(self, *args):
            pass
    monkeypatch.setattr("tcenv.tools.broker.released", lambda: Lock())
    monkeypatch.setattr("tcenv.evaluate.supervise", lambda *args, **kwargs: calls.append(("supervise", kwargs)) or "")
    result = invoke(_request(), session_role="kernel_once")
    assert result["status"] == "error" and "no valid host-readable result" in result["error"]
    assert calls[0][0] == "spawn" and calls[1] == ("release",) and calls[2][0] == "supervise"
    assert ("kill", 12345) in calls and ("forget", 12345) in calls
    spawn = calls[0][2]
    assert spawn["cwd"].name == "ai_lowerling" and "TCENV_TOOL_REGISTRY" not in spawn["env"]


def test_handler_source_drift_is_rejected(tmp_path):
    source = get_tool("metal.time_msl")
    original = __import__("pathlib").Path("backends/metal/tools/metal_time_msl.toml")
    impl = __import__("pathlib").Path("backends/metal/tools/tool_impl.toml")
    target = tmp_path / "metal_time_msl.toml"
    target.write_text(original.read_text())
    (tmp_path / "tool_impl.toml").write_text(impl.read_text().replace(source.handler_source_sha256, "0" * 64))
    with pytest.raises(ToolError, match="source differs"):
        load_manifest(target)


def test_scaffold_is_inert_and_never_registers_a_handler(tmp_path):
    from tcenv.tools.scaffold import scaffold_proposal
    result = scaffold_proposal({"action": "add a diagnostic tool", "scope": {"target_id": "apple-m4-max", "task": "conv2d"}, "evidence": {"reproduction": "r", "expected": "e"}}, tmp_path / "draft")
    assert result["registered"] is False and result["executable"] is False
    assert (tmp_path / "draft" / "handler.py").is_file()
    assert (tmp_path / "draft" / "tool_impl.toml").is_file()
    assert (tmp_path / "draft" / "fixture.json").is_file()
    assert "test_draft_has_no_capabilities_or_permission" in (tmp_path / "draft" / "protocol_test.py").read_text()
    assert "draft.time_msl" not in __import__("pathlib").Path("backends/metal/tools.json").read_text()
    with pytest.raises(ToolError, match="new directory"):
        scaffold_proposal({"action": "again", "scope": {"target_id": "apple-m4-max", "task": "conv2d"}, "evidence": {"reproduction": "r", "expected": "e"}}, tmp_path / "draft")
