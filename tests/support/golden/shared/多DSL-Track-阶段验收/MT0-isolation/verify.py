#!/usr/bin/env python3
"""Structural verifier for hmz adaptation/static evidence; never starts a fence."""

from __future__ import annotations

import ast
import copy
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent


def _load(name):
    return json.loads((HERE / name).read_text(encoding="utf-8"))


def _nodes():
    document = _load("nodes.json")
    return tuple(
        document["adapter_positive"] + document["adapter_negative"]
        + document["evidence_positive"] + document["evidence_negative"]
    )


def verify_document(document: dict) -> None:
    if document.get("schema_id") != "shapeforge.mt0-isolation.static-contract.1":
        raise ValueError("wrong MT0-isolation schema")
    if document.get("round_scope") != "hmz_fence_adaptation_and_static_contract_validation":
        raise ValueError("the completed scope must be hmz adaptation plus static validation only")
    if document.get("round_done") is not True or document.get("static_contract_status") != "passed":
        raise ValueError("the static adaptation scope is not honestly recorded")
    if document.get("mt0_status") != "blocked":
        raise ValueError("MT0 remains blocked")
    if document.get("embedded_dsl_production_ready") is not False or document.get("publication_allowed") is not False:
        raise ValueError("static validation cannot make embedded DSL production-ready")
    boundary = document["evidence_boundary"]
    if boundary.get("real_attack_evidence_status") != "pending_outer_probe":
        raise ValueError("real attack evidence must remain pending_outer_probe")
    for key in ("cpu_shim_and_static_contract_without_fence", "real_cpu_shim_executed", "real_process_group_watchdog_executed"):
        if boundary.get(key) is not True:
            raise ValueError(f"missing real CPU/static contract check: {key}")
    for key in ("mock_or_source_existence_is_isolation_proof", "sandbox_exec_existence_is_isolation_proof"):
        if boundary.get(key) is not False:
            raise ValueError(f"invalid evidence boundary: {key}")
    for key in ("real_hmz_process_launched", "real_fence_probe_executed", "direct_sandbox_exec_attack_probe_executed", "gpu_npu_model_paid_search"):
        if boundary.get(key) is not False:
            raise ValueError(f"unexecuted work is being claimed: {key}")
    decision = document["decision"]
    if decision["os_fence_owner"] != "hmz macOS Seatbelt/fence":
        raise ValueError("the selected owner is hmz's OS fence")
    for key in ("second_seatbelt_renderer", "direct_sandbox_exec_launcher", "hmz_internal_python_import"):
        if decision.get(key) is not False:
            raise ValueError(f"duplicate or implicit fence dependency is forbidden: {key}")
    runtime = document["current_hmz_runtime"]
    if runtime.get("status") != "backend_capability_unavailable" or not runtime.get("fail_closed"):
        raise ValueError("the local implicit-write incompatibility must fail closed")
    if "hmz_implicit_mds_write_outside_attempt_build" not in runtime.get("blockers", []):
        raise ValueError("the hmz implicit-write blocker cannot be hidden")
    if runtime.get("unfenced_fallback") is not False:
        raise ValueError("an unfenced fallback is forbidden")
    if document["next_stages"].get("MT0") != "blocked":
        raise ValueError("MT0 was widened")
    for stage in ("MT1b", "MT2", "MT3", "MT4", "MT6", "MT7"):
        if document["next_stages"].get(stage) != "not_started_blocked":
            raise ValueError(f"{stage} cannot start")


def verify_hmz() -> None:
    from tcenv.identity import fingerprint
    from tcenv.worker_isolation.hmz_fence import HMZ_SOURCE_SHA256, HMZ_VERSION

    document = _load("hmz-installation.json")
    if document["version"] != HMZ_VERSION or document["source_sha256"] != HMZ_SOURCE_SHA256:
        raise ValueError("recorded hmz identity differs from the adapter's pinned protocol")
    digest = fingerprint({
        "protocol_id": document["protocol_id"], "version": document["version"],
        "source_sha256": document["source_sha256"],
    })
    if digest != document["protocol_digest"]:
        raise ValueError("hmz protocol digest was not recomputed")
    if document["sandbox_exec"]["application_attempted"] is not False:
        raise ValueError("the static package cannot claim a Seatbelt application probe")
    compatibility = document["runtime_compatibility"]
    if compatibility["candidate_launch_allowed"] is not False or compatibility["unfenced_fallback"] is not False:
        raise ValueError("the incompatible local hmz must not launch candidates")
    if not document["source_findings"]["local_implicit_write_roots"]:
        raise ValueError("the implicit hmz write root is missing from the audit")


def verify_outer_plan() -> None:
    document = _load("outer-probe.json")
    if document["status"] != "pending_outer_probe" or document["executed_this_round"] is not False:
        raise ValueError("outer attacks remain pending")
    if document["real_attack_evidence"] is not False:
        raise ValueError("a plan is not attack evidence")
    if "Operation not permitted" not in document["why_not_in_rlar"]:
        raise ValueError("the nested-Seatbelt reason is not recorded")
    command = document["command"]
    if not any(str(word).endswith("outer_probe.py") for word in command) or not (HERE / "outer_probe.py").is_file():
        raise ValueError("the guarded outer-probe command is missing")
    positive = {row["id"] for row in document["required_positive_cases"]}
    negative = {row["id"] for row in document["required_negative_cases"]}
    if not {"benign_artifact", "python_and_clang_children", "temporary_environment"} <= positive:
        raise ValueError("outer positive controls are incomplete")
    if not {
        "outside_read_write", "implicit_mds_write", "loopback_and_network",
        "ordinary_timeout_process_group", "setsid_descendant", "artifact_symlink", "capability_drift",
    } <= negative:
        raise ValueError("outer negative controls are incomplete")
    tree = ast.parse((HERE / "outer_probe.py").read_text(encoding="utf-8"))
    fixtures = {
        target.id: node.value.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str)
        for target in node.targets
        if isinstance(target, ast.Name) and target.id in {"source", "sleeping", "detached"}
    }
    if set(fixtures) != {"source", "sleeping", "detached"}:
        raise ValueError("outer probe fixture sources are missing")
    for name, source in fixtures.items():
        compile(source, f"<outer-fixture:{name}>", "exec")


def verify_history() -> None:
    start = _load("start-state.json")
    for relative in (
        "docs/架构/多DSL-Track-阶段验收/MT0/acceptance.md",
        "docs/架构/多DSL-Track-阶段验收/MT0/manifest.sha256",
        "docs/架构/多DSL-Track-阶段验收/MT1a/manifest.sha256",
        "docs/架构/多DSL-Track-阶段验收/MT5/manifest.sha256",
    ):
        expected = start["scoped_file_sha256"][relative]
        if hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() != expected:
            raise ValueError(f"a closed historical package was rewritten: {relative}")


def verify_results() -> None:
    document = _load("verification.json")
    if document["scope"] != "hmz_fence_adaptation_and_static_contract_validation":
        raise ValueError("verification result has the wrong scope")
    if document["fixed_nodes"]["mt0_isolation"] != 53:
        raise ValueError("the fixed MT0-isolation node count is not 53")
    if document["fixed_nodes"]["historical_mt0"] != 11:
        raise ValueError("the historical MT0 anchor count drifted")
    if document["fixed_nodes"]["historical_mt1a_regression"] != 86 or document["fixed_nodes"]["historical_mt1a_new"] != 23:
        raise ValueError("the historical MT1a anchor counts drifted")
    if document["fixed_nodes"]["historical_mt5"] != 45 or document["fixed_nodes"]["test_structure"] != 6:
        raise ValueError("the historical MT5 or test-structure anchors drifted")
    if document["fixed_nodes"].get("failure_codes") != 3:
        raise ValueError("the typed failure-code runbook tests were not recorded")
    if document["real_hmz_or_seatbelt_or_attack_probe"] is not False:
        raise ValueError("verification results cannot invent a real probe")
    if document["real_cpu_shim_and_process_group_watchdog"] is not True:
        raise ValueError("the real CPU shim/watchdog tests must be recorded")
    if document["gpu_npu_model_or_paid_search"] is not False:
        raise ValueError("verification results cannot invent device/model work")
    for relative, expected in document["scoped_source_sha256"].items():
        if hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() != expected:
            raise ValueError(f"scoped source digest mismatch: {relative}")


def verify_manifest() -> None:
    for line in (HERE / "manifest.sha256").read_text().splitlines():
        if not line.strip():
            continue
        expected, relative = line.split("  ", 1)
        path = HERE / relative
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"manifest mismatch: {relative}")


def self_test() -> None:
    original = _load("static-contract.json")
    production = copy.deepcopy(original)
    production["embedded_dsl_production_ready"] = True
    attack = copy.deepcopy(original)
    attack["evidence_boundary"]["real_attack_evidence_status"] = "passed"
    mt0 = copy.deepcopy(original)
    mt0["mt0_status"] = "complete"
    fallback = copy.deepcopy(original)
    fallback["current_hmz_runtime"]["unfenced_fallback"] = True
    mt1b = copy.deepcopy(original)
    mt1b["next_stages"]["MT1b"] = "started"
    for index, changed in enumerate((production, attack, mt0, fallback, mt1b)):
        try:
            verify_document(changed)
        except ValueError:
            continue
        raise AssertionError(f"verifier self-test mutation {index} was accepted")


def _pytest() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-n", "0", "-m", "not gpu and not llm and not slow", *_nodes()],
        cwd=ROOT, text=True, capture_output=True, check=False,
    )
    output = completed.stdout + completed.stderr
    if completed.returncode != 0:
        print(output, end="")
        raise ValueError("MT0-isolation fixed nodes failed; read logs/pytest/triage/triage_summary.md")
    match = re.search(r"(\d+) passed", output)
    if match is None or int(match.group(1)) != 53 or re.search(r"[1-9]\d* (?:skipped|xfailed)", output):
        raise ValueError(f"expected exactly 53 passed without skip/xfail, got: {output[-1000:]}")


def main() -> int:
    verify_document(_load("static-contract.json"))
    verify_hmz()
    verify_outer_plan()
    verify_history()
    verify_results()
    self_test()
    if "--check" in sys.argv:
        verify_manifest()
        _pytest()
    print(json.dumps({
        "done": True,
        "scope": "hmz_fence_adaptation_and_static_contract_validation",
        "fixed_nodes_passed": 53 if "--check" in sys.argv else None,
        "manifest": "--check" in sys.argv,
        "mt0_status": "blocked",
        "embedded_dsl_production_ready": False,
        "real_attack_evidence_status": "pending_outer_probe",
        "current_hmz_runtime": "backend_capability_unavailable: implicit mds write outside attempt/build",
        "real_cpu_shim_and_process_group_watchdog": True,
        "real_hmz_or_sandbox_exec_launched": False,
    }, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
