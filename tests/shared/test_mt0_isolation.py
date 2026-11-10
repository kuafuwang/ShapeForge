"""MT0-isolation evidence boundaries; no real hmz/Seatbelt execution."""

from __future__ import annotations

import ast
import copy
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "tests/support/golden/shared/多DSL-Track-阶段验收/MT0-isolation"


def _load(name):
    return json.loads((PACKAGE / name).read_text(encoding="utf-8"))


def _verifier():
    spec = importlib.util.spec_from_file_location("mt0_isolation_verify", PACKAGE / "verify.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    from tests.support.boundaries import rebind_acceptance_root

    rebind_acceptance_root(module)
    return module


def test_scope_is_only_hmz_adaptation_and_static_validation():
    document = _load("static-contract.json")
    assert document["round_done"] is True
    assert document["round_scope"] == "hmz_fence_adaptation_and_static_contract_validation"
    assert document["static_contract_status"] == "passed"
    assert document["mt0_status"] == "blocked"
    assert document["embedded_dsl_production_ready"] is False
    assert document["evidence_boundary"]["real_attack_evidence_status"] == "pending_outer_probe"
    for stage in ("MT1b", "MT2", "MT3", "MT4", "MT6", "MT7"):
        assert document["next_stages"][stage] == "not_started_blocked"


def test_hmz_identity_and_implicit_write_blocker_are_explicit():
    document = _load("hmz-installation.json")
    assert document["version"] == "0.2.2"
    assert len(document["source_sha256"]) == 7
    assert document["source_findings"]["local_implicit_write_roots"]
    assert document["runtime_compatibility"]["status"] == "backend_capability_unavailable"
    assert document["runtime_compatibility"]["candidate_launch_allowed"] is False
    assert document["runtime_compatibility"]["unfenced_fallback"] is False
    assert document["sandbox_exec"]["application_attempted"] is False


def test_outer_probe_is_pending_and_has_positive_negative_controls():
    document = _load("outer-probe.json")
    assert document["status"] == "pending_outer_probe"
    assert document["executed_this_round"] is False
    assert document["real_attack_evidence"] is False
    assert "outside every Seatbelt sandbox" in document["execution_location"]
    assert "Operation not permitted" in document["why_not_in_rlar"]
    assert len(document["required_positive_cases"]) >= 3
    assert len(document["required_negative_cases"]) >= 7
    assert any(word.endswith("outer_probe.py") for word in document["command"])
    tree = ast.parse((PACKAGE / "outer_probe.py").read_text())
    fixtures = {
        target.id: node.value.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str)
        for target in node.targets
        if isinstance(target, ast.Name) and target.id in {"source", "sleeping", "detached"}
    }
    assert set(fixtures) == {"source", "sleeping", "detached"}
    for name, source in fixtures.items():
        compile(source, f"<outer-fixture:{name}>", "exec")


def test_closed_mt0_mt1a_mt5_bytes_are_unchanged_and_outer_script_is_not_run():
    _verifier().verify_history()
    result_name = "outer-probe-result.json"
    assert not (PACKAGE / result_name).exists()
    assert "outer-probe-result.json" not in (PACKAGE / "manifest.sha256").read_text()


@pytest.mark.parametrize("mutation", ["production", "attack", "mt0", "fallback", "mt1b"])
def test_verifier_rejects_scope_and_evidence_overclaim(mutation):
    document = copy.deepcopy(_load("static-contract.json"))
    if mutation == "production":
        document["embedded_dsl_production_ready"] = True
    elif mutation == "attack":
        document["evidence_boundary"]["real_attack_evidence_status"] = "passed"
    elif mutation == "mt0":
        document["mt0_status"] = "complete"
    elif mutation == "fallback":
        document["current_hmz_runtime"]["unfenced_fallback"] = True
    else:
        document["next_stages"]["MT1b"] = "started"
    with pytest.raises(ValueError):
        _verifier().verify_document(document)
