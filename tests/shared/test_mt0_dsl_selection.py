"""MT0 read-only DSL selection evidence; no frontend build, model, GPU, or NPU."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "tests/support/golden/shared/多DSL-Track-阶段验收/MT0"


def _load(name):
    return json.loads((PACKAGE / name).read_text())


def test_mt0_is_honestly_blocked_and_dependent_stages_do_not_claim_authorization():
    document = _load("dsl-selection.json")
    assert document["status"] == "blocked"
    assert document["audited_candidate_count"] == 2
    assert document["qualified_candidate_count"] == 0
    assert document["qualified_candidate_ids"] == []
    for stage in ("MT1b", "MT2", "MT3", "MT4", "MT6", "MT7"):
        assert document["next_stages"][stage] == "blocked"
    assert document["next_stages"]["MT1a"] == "allowed_independent_work"
    assert document["next_stages"]["MT5"] == "allowed_independent_work"
    boundary = document["evidence_boundary"]
    assert boundary["read_only_upstream_audit"] is True
    for key in ("local_frontend_help_or_version_probe", "local_kernel_compilation", "gpu_or_npu_execution", "model_or_paid_search", "publication_allowed"):
        assert boundary[key] is False
    assert boundary["performance_claim"] is None


def test_two_concrete_candidates_have_pinned_identities_and_real_upstream_paths():
    document = _load("dsl-selection.json")
    candidates = {row["dsl_id"]: row for row in document["candidates"]}
    assert set(candidates) == {"tilelang", "halide"}
    expected = {
        "tilelang": ("0.1.15", "a35f8ddf45eba16c21211ec8822d56ce5363036f"),
        "halide": ("22.1.0", "a8c592b056c06eeeeb064778e6f83e3c6002338f"),
    }
    for dsl_id, (version, commit) in expected.items():
        row = candidates[dsl_id]
        assert row["version"] == version and row["commit"] == commit
        assert row["repo"].startswith("https://github.com/") and row["license"]
        assert row["source_form"]["syntax"]
        assert row["candidate_execution_model"] == "embedded_host_code"
        assert row["candidate_roles"] and row["host_scheduler_ownership"]
        assert row["frontend_compiler"]["name"] and row["frontend_compiler"]["version"] == version
        assert row["build_artifacts"]["upstream_observed"]
        assert row["device_execution_path"]
        assert row["worker_isolation_requirements"] and row["cache_lifecycle_requirements"]
        assert row["reuses_current_msl_single_or_bundle_contract"] is False
        assert row["reuses_current_msl_single_or_bundle_adapter"] is False
        assert row["availability"] == "interface_only" and row["qualified"] is False


def test_candidates_do_not_turn_upstream_artifacts_into_local_evidence():
    for row in _load("dsl-selection.json")["candidates"]:
        local = row["local_frontend"]
        assert local["installed"] is False
        assert local["distribution_version"] is None
        assert local["import_spec_found"] is False
        assert local["offline_compile_run"] is False
        assert local["device_execution_run"] is False
        assert row["build_artifacts"]["local_observed"] == []
        assert row["build_artifacts"]["air_or_metallib_observed"] is False
        assert row["blockers"]


def test_independent_frontend_scheduler_and_delivery_evidence_is_explicit():
    for row in _load("dsl-selection.json")["candidates"]:
        evidence = row["independence_evidence"]
        assert set(evidence) == {"frontend", "optimizer_scheduler", "delivery_identity", "shared_substrate_boundary"}
        assert all(evidence.values())
        assert "HostTilingPlan" in evidence["shared_substrate_boundary"]
    matrix = _load("capability-matrix.json")
    assert matrix["rows"][-1]["tilelang"] == matrix["rows"][-1]["halide"] == "blocked"
    assert any(row["severity"] == "release_blocker" for row in matrix["risk_register"])


def test_upstream_ledger_has_verifiable_source_excerpt_hashes_and_no_performance_dependency():
    ledger = _load("upstream-map.json")
    ids = {row["source_id"] for row in ledger["sources"]}
    assert {
        "tilelang-tag-api", "tilelang-metal-internals", "tilelang-codegen", "tilelang-tvm-metal-runtime",
        "halide-tag-api", "halide-metal-codegen", "halide-metal-runtime", "halide-license",
    } <= ids
    for row in ledger["sources"]:
        assert row["evidence_excerpt"]
        assert hashlib.sha256(row["evidence_excerpt"].encode()).hexdigest() == row["evidence_excerpt_sha256"]
        assert row["retrieval_reference"]
    tilelang = next(row for row in ledger["sources"] if row["source_id"] == "tilelang-metal-internals")
    assert "Performance section 5.6 is not used" in tilelang["evidence_excerpt"]


def _verifier_module():
    spec = importlib.util.spec_from_file_location("mt0_acceptance_verify", PACKAGE / "verify.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    from tests.support.boundaries import rebind_acceptance_root

    rebind_acceptance_root(module)
    return module


def test_vertical_slices_are_design_only_and_rejections_do_not_count_toys_or_forbidden_routes():
    slices = _load("vertical-slices.json")
    assert slices["status"] == "design_only_not_authorized_for_MT1b_or_MT2"
    assert {row["id"] for row in slices["slices"]} == {"tilelang", "halide"}
    for row in slices["slices"]:
        assert row["current_blocker"]
        assert row["required_positive_evidence"] and row["required_negative_evidence"]
    rejected = _load("dsl-selection.json")["rejected_routes"]
    assert any("Toy JSON" in row["route"] for row in rejected)
    assert any("preserved Triton v1" in row["route"] for row in rejected)
    assert any("MPSGraph" in row["route"] for row in rejected)
    assert any("CUDA/CANN" in row["route"] for row in rejected)


@pytest.mark.parametrize("mutation", ["available", "adapter", "complete", "local", "mt2"])
def test_verifier_rejects_each_fixed_negative_mutation(mutation):
    document = copy.deepcopy(_load("dsl-selection.json"))
    if mutation == "available":
        document["candidates"][0]["availability"] = "available"
    elif mutation == "adapter":
        document["candidates"][0]["reuses_current_msl_single_or_bundle_adapter"] = True
    elif mutation == "complete":
        document["status"] = "complete"
    elif mutation == "local":
        document["candidates"][0]["build_artifacts"]["local_observed"] = ["invented.metallib"]
    else:
        document["next_stages"]["MT2"] = "allowed"
    with pytest.raises(ValueError):
        _verifier_module().verify_document(document)
