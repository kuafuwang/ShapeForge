"""MT0-isolation-profile static and preserved outer-probe evidence; no probe execution."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "tests/support/golden/shared/多DSL-Track-阶段验收/MT0-isolation-profile"


def _load(name):
    return json.loads((PACKAGE / name).read_text(encoding="utf-8"))


def _verifier():
    spec = importlib.util.spec_from_file_location("mt0_isolation_profile_verify", PACKAGE / "verify.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    from tests.support.boundaries import rebind_acceptance_root

    rebind_acceptance_root(module)
    return module


def test_static_epoch_remains_limited_and_outer_epoch_does_not_complete_mt0():
    document = _load("static-contract.json")
    assert document["round_done"] is True
    assert document["round_scope"] == "hmz_lower_level_profile_adaptation_and_static_cpu_validation"
    assert document["static_contract_status"] == "passed"
    assert document["mt0_status"] == "blocked"
    assert document["embedded_dsl_production_ready"] is False
    assert document["decision"]["production_route_count"] == 1
    assert document["decision"]["candidate_build_online"] is False
    assert document["decision"]["mds_required"] is False
    assert document["evidence_boundary"]["real_attack_evidence_status"] == "pending_outer_probe"
    for stage in ("MT1b", "MT2", "MT3", "MT4", "MT6", "MT7"):
        assert document["next_stages"][stage] == "not_started_blocked"
    review = _load("outer-probe-review.json")
    assert review["parent_static_epoch"] == document["epoch"]
    assert review["current_status"]["mt0_status"] == "blocked"


def test_source_identity_binds_driver_interpreter_and_no_mds():
    document = _load("source-identity.json")
    assert document["version"] == "0.2.2" and document["python_version"] == "3.14.4"
    assert len(document["source_sha256"]) == 8
    assert document["driver"]["sha256"] and document["trusted_shim"]["sha256"]
    assert document["protocol_id"] == "hmz-lower-level-seatbelt-profile-proxy-1"
    assert document["source_findings"]["implicit_mds_write_roots"] == []
    assert document["runtime_compatibility"]["status"] == "static_identity_valid_outer_probe_pending"
    assert document["runtime_compatibility"]["candidate_launch_allowed_this_round"] is False
    assert document["runtime_compatibility"]["unfenced_fallback"] is False
    assert document["sandbox_exec"]["application_attempted"] is False


def test_outer_probe_plan_preserves_offline_port_and_ipc_boundary():
    document = _load("outer-probe.json")
    assert document["status"] == "pending_outer_probe"
    assert document["executed_this_round"] is False and document["real_attack_evidence"] is False
    assert document["route"]["online"] is False and document["route"]["proxy_hosts"] == []
    assert "actual" in document["route"]["profile_port"] and "never None" in document["route"]["profile_port"]
    assert document["route"]["mds_expected_write_roots"] == []
    assert "do not prove" in document["route"]["local_ipc_claim"]
    assert len(document["required_positive_cases"]) >= 5
    assert len(document["required_negative_cases"]) >= 8
    assert any(word.endswith("outer_probe.py") for word in document["command"])
    clang = next(row for row in document["required_positive_cases"] if row["id"] == "python_and_clang_children")
    assert "xcrun -n -f clang" in clang["expected"]


def test_closed_internal_fence_package_and_both_outer_results_retained():
    _verifier().verify_history()
    manifest = (PACKAGE / "manifest.sha256").read_text()
    assert "outer-probe-result.json" in manifest
    assert "outer-probe-result-r2.json" in manifest
    assert "outer-probe-review.json" in manifest
    assert _load("outer-probe-result.json")["status"] == "outer_probe_failed"
    assert _load("outer-probe-result-r2.json")["status"] == "outer_probe_observed"


def test_real_cpu_shim_watchdog_evidence_is_not_attack_evidence():
    document = _load("cpu-shim-watchdog.json")
    assert document["status"] == "passed" and len(document["nodes"]) == 4
    assert document["real_os_fence_executed"] is False
    assert document["attack_containment_proven"] is False
    assert document["observations"]["ordinary_grandchild"]["stdout"] == "grandchild-ok\n"
    assert document["observations"]["ordinary_timeout_process_group"]["setsid_or_detached_attack_tested"] is False


def test_r1_failure_and_absent_clang_diagnostics_are_immutable():
    document = _load("outer-probe-result.json")
    _verifier().verify_outer_result(document, expected_status="outer_probe_failed")
    assert document["real_os_fence_executed"] is True
    assert len(document["checks"]) == 18
    assert {name for name, value in document["checks"].items() if value is False} == {
        "clang_child_usable", "temporary_home",
    }
    assert document["checks"]["clang_child_usable"] is False
    assert document["checks"]["temporary_home"] is False
    assert document["main_observations"]["clang_returncode"] == 69
    assert "clang_stdout" not in document["main_observations"]
    assert "clang_stderr" not in document["main_observations"]
    record = _load("outer-probe-review.json")["raw_results"]["r1"]
    assert record["failed_checks"] == ["clang_child_usable", "temporary_home"]
    assert record["sha256"] == hashlib.sha256((PACKAGE / "outer-probe-result.json").read_bytes()).hexdigest()


def test_r2_exact_raw_check_set_is_18_of_18():
    document = _load("outer-probe-result-r2.json")
    verifier = _verifier()
    verifier.verify_outer_result(document, expected_status="outer_probe_observed")
    assert len(document["checks"]) == 18
    assert set(document["checks"]) == verifier.EXPECTED_OUTER_CHECKS
    assert {name for name, value in document["checks"].items() if value is not True} == set()
    assert document["real_os_fence_executed"] is True
    assert document["status"] == "outer_probe_observed"
    assert document["independent_review_required"] is True
    review = _load("outer-probe-review.json")
    assert review["raw_results"]["r2"]["raw_check_count"] == 18
    assert "not 17" in review["raw_results"]["count_note"]


def test_outer_review_rebinds_raw_probe_adapter_driver_shim_and_hmz_sources():
    review = _load("outer-probe-review.json")
    verifier = _verifier()
    verifier.verify_source_identity()
    verifier.verify_outer_review(review)
    identity = review["source_identity"]
    assert identity["outer_probe_sha256"] == hashlib.sha256((PACKAGE / "outer_probe.py").read_bytes()).hexdigest()
    assert identity["adapter_sha256"] == hashlib.sha256((ROOT / "tcenv/worker_isolation/hmz_fence.py").read_bytes()).hexdigest()
    assert identity["driver_sha256"] == hashlib.sha256((ROOT / "tcenv/worker_isolation/hmz_profile_driver.py").read_bytes()).hexdigest()
    assert identity["trusted_shim_sha256"] == hashlib.sha256((ROOT / "tcenv/worker_isolation/child.py").read_bytes()).hexdigest()
    assert identity["hmz_source_sha256"] == _load("source-identity.json")["source_sha256"]
    assert _load("outer-probe-result.json")["hmz"] == _load("outer-probe-result-r2.json")["hmz"]


def test_home_path_membership_uses_resolved_private_var_aliases():
    verifier = _verifier()
    r2_home = Path(_load("outer-probe-result-r2.json")["main_observations"]["home"])
    build = r2_home.parent
    alias_home = Path(str(r2_home).replace("/private/var/", "/var/", 1))
    alias_build = Path(str(build).replace("/private/var/", "/var/", 1))
    assert alias_home != r2_home and alias_build != build
    assert verifier.path_within(alias_home, alias_build) is True
    assert verifier.path_within(r2_home, build) is True
    assert verifier.path_within(build.parent / "outside-home", build) is False
    r1 = _load("outer-probe-result.json")
    assert r1["checks"]["temporary_home"] is False
    # The corrected comparison would accept r1's real path too; its recorded
    # false result remains immutable evidence of the pre-correction failure.
    assert verifier.path_within(Path(r1["main_observations"]["home"]), Path(r1["main_observations"]["home"]).parent) is True


def test_clang_comes_from_xcrun_real_path_and_r2_saves_streams():
    verifier = _verifier()
    review = _load("outer-probe-review.json")
    assert verifier.verify_probe_source() == review["source_identity"]["outer_probe_sha256"]
    source = (PACKAGE / "outer_probe.py").read_text()
    assert 'subprocess.check_output(["xcrun", "-n", "-f", "clang"], text=True).strip()' in source
    assert 'observed["clang_stdout"] = clang.stdout' in source
    assert 'observed["clang_stderr"] = clang.stderr' in source
    assert '"/usr/bin/clang"' not in source
    r2 = _load("outer-probe-result-r2.json")["main_observations"]
    assert r2["clang_returncode"] == 0
    assert r2["clang_stdout"].startswith("Apple clang version 21.0.0 (clang-2100.3.34.2)\n")
    assert "/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin" in r2["clang_stdout"]
    assert r2["clang_stderr"] == ""
    r1 = _load("outer-probe-result.json")["main_observations"]
    assert r1["clang_returncode"] == 69 and "clang_stderr" not in r1


def test_mds_outside_and_non_proxy_loopback_denials_with_controls():
    verifier = _verifier()
    for name in ("outer-probe-result.json", "outer-probe-result-r2.json"):
        document = _load(name)
        verifier.verify_outer_result(
            document,
            expected_status="outer_probe_failed" if name == "outer-probe-result.json" else "outer_probe_observed",
        )
        observed = document["main_observations"]
        assert observed["outside_read"]["result"] == "denied"
        assert observed["outside_write"]["result"] == "denied"
        assert observed["mds_write"] == {"result": "denied", "error": "PermissionError"}
        assert observed["non_proxy_loopback"] == {"result": "denied", "error": "PermissionError"}
        assert observed["empty_host_proxy"] == {"result": "succeeded", "value": 403}
        assert document["checks"]["mds_parent_negative_control"] is True
        assert document["negative_controls"] == verifier.EXPECTED_NEGATIVE_CONTROLS
        assert document["negative_controls"]["artificial_mds_parent_was_writable"] is True
        assert document["negative_controls"]["loopback_listener_bound_and_accepted_outer_control"] is True


def test_loopback_unix_mach_residual_is_not_claimed_eliminated():
    review = _load("outer-probe-review.json")
    residual = review["residual_boundaries"]
    assert residual["loopback_unix_socket_and_mach_service_residual_remains"] is True
    assert residual["all_local_ipc_eliminated"] is False
    for name in ("outer-probe-result.json", "outer-probe-result-r2.json"):
        document = _load(name)
        assert document["local_ipc_boundary"] == "source-audited Unix-socket/Mach-service residual; no all-local-IPC-denied claim"
        assert "no claim that all local IPC disappears" in document["hmz"]["local_ipc_boundary"]
    findings = _load("source-identity.json")["source_findings"]["local_ipc_residual"]
    assert "Unix-socket" in findings and "Mach" in findings


def test_setsid_survives_process_group_but_remains_fenced_and_is_not_closed():
    document = _load("outer-probe-result-r2.json")
    assert document["checks"]["ordinary_child_reaped"] is True
    assert document["checks"]["setsid_outside_write_denied"] is True
    assert document["checks"]["setsid_fence_inherited"] is True
    assert document["checks"]["setsid_process_group_gap_explicitly_measured"] is True
    residual = _load("outer-probe-review.json")["residual_boundaries"]
    assert residual["setsid_process_group_gap_explicitly_measured"] is True
    assert residual["setsid_process_group_gap_closed"] is False
    assert residual["setsid_descendant_retains_profile_and_denies_outside_write"] is True
    assert residual["outer_detached_group_cleanup_required"] is True


def test_outer_observation_does_not_complete_mt0_dsl_frontend_metal_or_publication():
    review = _load("outer-probe-review.json")
    current = review["current_status"]
    assert current["real_attack_evidence_status"] == "observed_and_independently_reviewed_not_production_pass"
    assert current["mt0_status"] == "blocked"
    assert current["production_ready"] is False and current["publication_allowed"] is False
    assert current["frontend_acceptance"] == "not_installed_not_accepted"
    assert current["mt1b_status"] == "not_started_blocked"
    assert current["metal_dsl_execution_or_performance_acceptance"] is False
    selection = json.loads((PACKAGE.parent / "MT0" / "dsl-selection.json").read_text())
    assert selection["status"] == "blocked" and selection["qualified_candidate_count"] == 0
    assert all(row["availability"] == "interface_only" and row["qualified"] is False for row in selection["candidates"])


@pytest.mark.parametrize("mutation", ["production", "attack", "mt0", "fallback", "online", "dual_path", "mds", "mt1b"])
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
    elif mutation == "online":
        document["decision"]["candidate_build_online"] = True
    elif mutation == "dual_path":
        document["decision"]["production_route_count"] = 2
    elif mutation == "mds":
        document["decision"]["implicit_mds_write_roots"] = ["/private/var/folders/example/C/mds"]
    else:
        document["next_stages"]["MT1b"] = "started"
    with pytest.raises(ValueError):
        _verifier().verify_document(document)


@pytest.mark.parametrize("mutation", [
    "r1_status", "r1_hide_failure", "r1_invent_stderr", "r2_drop_check",
    "r2_extra_check", "r2_mds_succeeded", "r2_no_fence", "r2_bad_negative_control",
    "r2_drop_stderr", "r2_ipc_overclaim", "r2_hide_setsid_gap",
])
def test_verifier_rejects_raw_result_mutation(mutation):
    is_r1 = mutation.startswith("r1_")
    document = copy.deepcopy(_load("outer-probe-result.json" if is_r1 else "outer-probe-result-r2.json"))
    expected_status = "outer_probe_failed" if is_r1 else "outer_probe_observed"
    if mutation == "r1_status":
        document["status"] = "outer_probe_observed"
    elif mutation == "r1_hide_failure":
        document["checks"]["temporary_home"] = True
    elif mutation == "r1_invent_stderr":
        document["main_observations"]["clang_stderr"] = "invented"
    elif mutation == "r2_drop_check":
        del document["checks"]["mds_write_denied"]
    elif mutation == "r2_extra_check":
        document["checks"]["extra"] = True
    elif mutation == "r2_mds_succeeded":
        document["main_observations"]["mds_write"] = {"result": "succeeded"}
    elif mutation == "r2_no_fence":
        document["real_os_fence_executed"] = False
    elif mutation == "r2_bad_negative_control":
        document["negative_controls"]["artificial_mds_parent_was_writable"] = False
    elif mutation == "r2_drop_stderr":
        del document["main_observations"]["clang_stderr"]
    elif mutation == "r2_ipc_overclaim":
        document["local_ipc_boundary"] = "all local IPC denied"
    else:
        document["checks"]["setsid_process_group_gap_explicitly_measured"] = False
    with pytest.raises(ValueError):
        _verifier().verify_outer_result(document, expected_status=expected_status)


@pytest.mark.parametrize("mutation", [
    "production", "mt0", "frontend", "all_ipc", "setsid_closed", "review_pending",
    "r2_count_17", "r1_removed", "probe_drift", "raw_digest_drift", "negative_control", "probe_rerun",
])
def test_verifier_rejects_outer_review_overclaim(mutation):
    document = copy.deepcopy(_load("outer-probe-review.json"))
    if mutation == "production":
        document["current_status"]["production_ready"] = True
    elif mutation == "mt0":
        document["current_status"]["mt0_status"] = "complete"
    elif mutation == "frontend":
        document["current_status"]["frontend_acceptance"] = "accepted"
    elif mutation == "all_ipc":
        document["residual_boundaries"]["all_local_ipc_eliminated"] = True
    elif mutation == "setsid_closed":
        document["residual_boundaries"]["setsid_process_group_gap_closed"] = True
    elif mutation == "review_pending":
        document["independent_review"]["status"] = "pending"
    elif mutation == "r2_count_17":
        document["raw_results"]["r2"]["raw_check_count"] = 17
    elif mutation == "r1_removed":
        document["raw_results"].pop("r1")
    elif mutation == "probe_drift":
        document["source_identity"]["outer_probe_sha256"] = "0" * 64
    elif mutation == "raw_digest_drift":
        document["raw_results"]["r2"]["sha256"] = "0" * 64
    elif mutation == "negative_control":
        document["required_negative_controls"]["artificial_secret_was_readable"] = False
    else:
        document["independent_review"]["outer_probe_rerun"] = True
    with pytest.raises(ValueError):
        _verifier().verify_outer_review(document)
