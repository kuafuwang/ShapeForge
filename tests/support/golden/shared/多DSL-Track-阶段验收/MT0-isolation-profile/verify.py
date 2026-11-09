#!/usr/bin/env python3
"""Verify static and preserved outer-probe epochs without launching hmz or a fence."""

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
EXPECTED_NODES = 100
OUTER_REVIEW_EPOCH = "mt0-hmz-lower-level-profile-outer-review-1"
OUTER_SCHEMA = "shapeforge.mt0-isolation-profile.outer-probe-result.1"
OUTER_REVIEW_SCHEMA = "shapeforge.mt0-isolation-profile.outer-probe-review.1"
R1_FILE = "outer-probe-result.json"
R2_FILE = "outer-probe-result-r2.json"

# This is the exact set in both raw files.  The earlier prose count of 17 was
# wrong: the raw objects contain 18 entries, and evidence is never edited to
# match a prose count.
EXPECTED_OUTER_CHECKS = frozenset({
    "actual_proxy_port_matches_evidence",
    "artifact_symlink_rejected",
    "clang_child_usable",
    "empty_host_proxy_refused_upstream",
    "host_environment_not_inherited",
    "mds_parent_negative_control",
    "mds_write_denied",
    "non_proxy_loopback_denied",
    "ordinary_child_reaped",
    "outside_read_denied",
    "outside_write_denied",
    "positive_artifact",
    "python_child_still_fenced",
    "setsid_fence_inherited",
    "setsid_outside_write_denied",
    "setsid_process_group_gap_explicitly_measured",
    "single_lower_level_driver_identity",
    "temporary_home",
})
R1_FAILED_CHECKS = frozenset({"clang_child_usable", "temporary_home"})
EXPECTED_NEGATIVE_CONTROLS = {
    "artificial_mds_parent_was_writable": True,
    "artificial_outside_directory_was_writable": True,
    "artificial_secret_was_readable": True,
    "loopback_listener_bound_and_accepted_outer_control": True,
    "real_credentials_or_external_network_used": False,
}
EXPECTED_OUTER_FIELDS = frozenset({
    "checks", "hmz", "independent_review_required", "local_ipc_boundary",
    "main_observations", "mt0_status", "negative_controls", "production_ready",
    "publication_allowed", "real_os_fence_executed", "schema_id", "status",
})
COMMON_OBSERVATION_FIELDS = frozenset({
    "clang_returncode", "empty_host_proxy", "home", "host_canary", "mds_write",
    "non_proxy_loopback", "outside_read", "outside_write", "proxy_port",
    "python_child_returncode",
})
EXPECTED_MANIFEST_MEMBERS = frozenset({
    "acceptance.md", "cpu-shim-watchdog.json", "nodes.json", R1_FILE, R2_FILE,
    "outer-probe.json", "outer_probe.py", "outer-probe-review.json",
    "source-identity.json", "start-state.json", "start-worktree.sha256",
    "static-contract.json", "verification.json", "verify.py",
})


def _load(name):
    return json.loads((HERE / name).read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _nodes():
    return tuple(_load("nodes.json")["node_ids"])


def path_within(path: str | Path, root: str | Path) -> bool:
    """Resolve both sides before comparing /var and /private/var aliases."""

    return Path(path).resolve(strict=False).is_relative_to(Path(root).resolve(strict=False))


def verify_document(document: dict) -> None:
    if document.get("schema_id") != "shapeforge.mt0-isolation-profile.static-contract.1":
        raise ValueError("wrong MT0-isolation-profile schema")
    if document.get("epoch") != "mt0-hmz-lower-level-profile-1":
        raise ValueError("the static epoch identity drifted")
    if document.get("round_scope") != "hmz_lower_level_profile_adaptation_and_static_cpu_validation":
        raise ValueError("completed scope must be lower-level adaptation plus static/ordinary CPU validation")
    if document.get("round_done") is not True or document.get("static_contract_status") != "passed":
        raise ValueError("the limited adaptation scope is not honestly recorded")
    if document.get("mt0_status") != "blocked":
        raise ValueError("MT0 remains blocked")
    if document.get("embedded_dsl_production_ready") is not False or document.get("publication_allowed") is not False:
        raise ValueError("static validation cannot make embedded DSL production-ready")
    decision = document["decision"]
    if decision.get("production_route_count") != 1 or decision.get("internal_fence_wrap_used") is not False:
        raise ValueError("exactly one lower-level production route is required")
    for key in ("hmz_source_modified", "hmz_config_modified", "hmz_forked", "second_seatbelt_renderer", "direct_sandbox_exec_launcher", "project_python_imports_hmz", "mds_required", "candidate_build_online"):
        if decision.get(key) is not False:
            raise ValueError(f"forbidden route/contract widening: {key}")
    if decision.get("implicit_mds_write_roots") != []:
        raise ValueError("the lower-level route must not add mds write roots")
    if "actual_proxy_port" not in decision.get("offline_mechanism", "") or "port=None" not in decision["offline_mechanism"]:
        raise ValueError("offline must bind an empty-host Proxy and its actual port, not None")
    if "no claim" not in decision.get("local_ipc_assumption", ""):
        raise ValueError("the local IPC residual cannot be hidden")
    if "separate trusted phase" not in decision.get("dependency_preparation", ""):
        raise ValueError("dependency preparation must stay separate from untrusted offline construction")
    boundary = document["evidence_boundary"]
    for key in ("static_contract_and_cpu_mock_without_fence", "read_only_real_installation_identity_inspection", "real_cpu_shim_executed", "real_process_group_watchdog_executed"):
        if boundary.get(key) is not True:
            raise ValueError(f"missing static/ordinary CPU check: {key}")
    for key in ("mock_or_source_existence_is_isolation_proof", "sandbox_exec_existence_is_isolation_proof", "real_hmz_process_launched", "real_driver_or_profile_launched", "real_fence_probe_executed", "direct_sandbox_exec_attack_probe_executed", "gpu_npu_model_paid_search"):
        if boundary.get(key) is not False:
            raise ValueError(f"unexecuted work is being claimed in the static epoch: {key}")
    if boundary.get("real_attack_evidence_status") != "pending_outer_probe":
        raise ValueError("the static epoch must retain its pending_outer_probe boundary")
    runtime = document["current_hmz_runtime"]
    if runtime.get("status") != "static_identity_valid_outer_probe_pending" or runtime.get("outer_seatbelt_capability_proven") is not False:
        raise ValueError("static identity must not be relabelled as outer Seatbelt capability")
    if runtime.get("fail_closed") is not True or runtime.get("unfenced_fallback") is not False or runtime.get("candidate_launched_this_round") is not False:
        raise ValueError("fail-closed/no-launch boundary drift")
    if document["next_stages"].get("MT0") != "blocked":
        raise ValueError("MT0 was widened")
    for stage in ("MT1b", "MT2", "MT3", "MT4", "MT6", "MT7"):
        if document["next_stages"].get(stage) != "not_started_blocked":
            raise ValueError(f"{stage} cannot start")


def verify_source_identity() -> None:
    from tcenv.identity import fingerprint
    from tcenv.worker_isolation import hmz_fence as live

    document = _load("source-identity.json")
    if document["version"] != live.HMZ_VERSION or document["python_version"] != live.HMZ_PYTHON_VERSION:
        raise ValueError("recorded hmz/interpreter version differs from the adapter")
    if document["protocol_id"] != live.HMZ_PROTOCOL_ID or document["source_sha256"] != live.HMZ_SOURCE_SHA256:
        raise ValueError("recorded lower-level protocol/source identity differs from the adapter")
    if document["entrypoint_sha256"] != live.HMZ_ENTRYPOINT_SHA256:
        raise ValueError("recorded hmz entrypoint identity drift")
    if len(document["source_sha256"]) != 8 or not any(path.endswith("fence/wrap.py") for path in document["source_sha256"]):
        raise ValueError("the eight-source audit anchor is incomplete")
    if any("/cli/" in path for path in document["source_sha256"]):
        raise ValueError("the lower-level route must not pin an internal CLI production dependency")
    driver = document["driver"]
    shim = document["trusted_shim"]
    if driver["sha256"] != live.DRIVER_SOURCE_SHA256 or shim["sha256"] != live.CHILD_SOURCE_SHA256:
        raise ValueError("recorded driver/shim identity differs from the adapter")
    if _sha256(ROOT / "tcenv/worker_isolation/hmz_profile_driver.py") != driver["sha256"]:
        raise ValueError("live driver bytes differ from the recorded identity")
    if _sha256(ROOT / "tcenv/worker_isolation/child.py") != shim["sha256"]:
        raise ValueError("live shim bytes differ from the recorded identity")
    digest = fingerprint({
        "protocol_id": document["protocol_id"], "layout": document["layout"],
        "version": document["version"], "python_version": document["python_version"],
        "entrypoint_sha256": document["entrypoint_sha256"], "source_sha256": document["source_sha256"],
        "driver_sha256": driver["sha256"], "child_sha256": shim["sha256"],
    })
    if digest != document["protocol_digest"] or digest != _load("static-contract.json")["current_hmz_runtime"]["protocol_digest"]:
        raise ValueError("lower-level protocol digest was not recomputed consistently")
    findings = document["source_findings"]
    if findings["implicit_mds_write_roots"] != [] or "never" not in findings["profile_port_none"]:
        raise ValueError("mds or offline source findings were weakened")
    if "Unix-socket" not in findings["local_ipc_residual"] or "Mach" not in findings["local_ipc_residual"]:
        raise ValueError("local IPC source boundary is missing")
    if document["sandbox_exec"]["application_attempted"] is not False:
        raise ValueError("the static package cannot claim a Seatbelt application")
    argv = document["controlled_driver_argv"] + document["controlled_fenced_argv"]
    if "internal" in argv or "fence" in argv:
        raise ValueError("the controlled argv revived the internal-fence path")


def verify_outer_plan() -> None:
    document = _load("outer-probe.json")
    if document["status"] != "pending_outer_probe" or document["executed_this_round"] is not False:
        raise ValueError("the preserved static-epoch plan must retain its original pending scope")
    if document["real_attack_evidence"] is not False:
        raise ValueError("a plan is not attack evidence")
    route = document["route"]
    if route["production_route_count"] != 1 or route["online"] is not False or route["proxy_hosts"] != []:
        raise ValueError("outer plan widened the offline single-route contract")
    if "actual" not in route["profile_port"] or "never None" not in route["profile_port"] or route["mds_expected_write_roots"] != []:
        raise ValueError("outer plan lost the actual-port or no-mds requirement")
    if "do not prove" not in route["local_ipc_claim"]:
        raise ValueError("outer plan overclaims local IPC containment")
    command = document["command"]
    if not any(str(word).endswith("outer_probe.py") for word in command) or not (HERE / "outer_probe.py").is_file():
        raise ValueError("the guarded outer-probe command is missing")
    positive = {row["id"] for row in document["required_positive_cases"]}
    negative = {row["id"] for row in document["required_negative_cases"]}
    if not {"benign_artifact", "python_and_clang_children", "temporary_environment", "driver_identity_and_argv", "empty_host_proxy_positive_control"} <= positive:
        raise ValueError("outer positive controls are incomplete")
    if not {
        "outside_read_write", "mds_write", "non_proxy_loopback_and_network", "environment_secret",
        "ordinary_timeout_process_group", "setsid_descendant", "artifact_symlink", "capability_drift",
    } <= negative:
        raise ValueError("outer negative controls are incomplete")
    clang_case = next(row for row in document["required_positive_cases"] if row["id"] == "python_and_clang_children")
    if "xcrun -n -f clang" not in clang_case["expected"] or "/usr/bin/clang" in clang_case["expected"]:
        raise ValueError("the compiler control must use the real xcrun-resolved Clang path")
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
        compile(source, f"<outer-profile-fixture:{name}>", "exec")
    compile((HERE / "outer_probe.py").read_text(encoding="utf-8"), str(HERE / "outer_probe.py"), "exec")


def verify_cpu_evidence() -> None:
    document = _load("cpu-shim-watchdog.json")
    if document["status"] != "passed" or len(document["nodes"]) != 4:
        raise ValueError("the four real ordinary shim/watchdog nodes are required")
    if document["real_os_fence_executed"] is not False or document["attack_containment_proven"] is not False:
        raise ValueError("ordinary CPU observations are not attack evidence")
    observed = document["observations"]
    if observed["ordinary_echo"]["stdout"] != "ok\n" or observed["ordinary_grandchild"]["stdout"] != "grandchild-ok\n":
        raise ValueError("ordinary positive process controls are missing")
    if observed["required_limit_failure"]["returncode"] != 125 or observed["required_limit_failure"]["frontend_exec_attempted"] is not False:
        raise ValueError("required-limit failure did not prevent frontend execution")
    if observed["ordinary_timeout_process_group"]["sentinel_exists_after_reap_and_0_8_second_wait"] is not False:
        raise ValueError("ordinary timeout descendant was not reaped")


def verify_history() -> None:
    start = _load("start-state.json")
    old = HERE.parent / "MT0-isolation"
    for relative, expected in start["closed_internal_fence_package_sha256"].items():
        path = old / relative
        if not path.is_file() or _sha256(path) != expected:
            raise ValueError(f"the closed internal-fence package was rewritten: {relative}")
    for relative, expected in start["other_closed_sha256"].items():
        if _sha256(ROOT / relative) != expected:
            raise ValueError(f"a closed historical package was rewritten: {relative}")


def _verify_result_hmz(hmz: dict) -> None:
    from tcenv.worker_isolation import hmz_fence as live

    source = _load("source-identity.json")
    _require(hmz.get("schema_id") == live.CAPABILITY_SCHEMA_ID, "outer result has the wrong installation schema")
    _require(hmz.get("protocol_id") == live.HMZ_PROTOCOL_ID, "outer result protocol identity drift")
    _require(hmz.get("protocol_digest") == source["protocol_digest"], "outer result protocol digest drift")
    _require(hmz.get("version") == live.HMZ_VERSION and hmz.get("python_version") == live.HMZ_PYTHON_VERSION, "outer hmz/interpreter version drift")
    _require(hmz.get("entrypoint_sha256") == live.HMZ_ENTRYPOINT_SHA256, "outer hmz entrypoint drift")
    _require(hmz.get("source_sha256") == live.HMZ_SOURCE_SHA256, "outer hmz source set or digest drift")
    _require(hmz.get("driver_sha256") == live.DRIVER_SOURCE_SHA256 == source["driver"]["sha256"], "outer driver identity drift")
    _require(hmz.get("child_sha256") == live.CHILD_SOURCE_SHA256 == source["trusted_shim"]["sha256"], "outer shim identity drift")
    _require(hmz.get("driver_path") == source["driver"]["path"] and hmz.get("child_path") == source["trusted_shim"]["path"], "outer driver/shim path drift")
    _require(hmz.get("executable") == source["executable"] and hmz.get("resolved_interpreter") == source["resolved_interpreter"], "outer hmz executable/interpreter path drift")
    _require(hmz.get("implicit_mds_write_roots") == [], "outer result added an mds write root")
    _require(hmz.get("internal_fence_or_wrap_used") is False, "outer result used the closed internal wrapper")
    _require(hmz.get("sandbox_exec_present_executable") is True, "outer result lacks the real Seatbelt executable observation")
    _require("no claim that all local IPC disappears" in hmz.get("local_ipc_boundary", ""), "outer hmz evidence hides the IPC residual")


def verify_outer_result(document: dict, *, expected_status: str) -> None:
    """Independently validate one raw result, including semantics behind every check."""

    _require(expected_status in {"outer_probe_failed", "outer_probe_observed"}, "unknown expected outer status")
    _require(set(document) == EXPECTED_OUTER_FIELDS, "outer result top-level fields differ from the preserved raw schema")
    _require(document.get("schema_id") == OUTER_SCHEMA, "wrong outer result schema")
    _require(document.get("status") == expected_status, "outer result status was rewritten")
    _require(document.get("real_os_fence_executed") is True, "outer result does not record a real OS fence")
    _require(document.get("production_ready") is False and document.get("publication_allowed") is False, "outer result is not a production/publication pass")
    _require(document.get("mt0_status") == "blocked", "outer result cannot complete MT0")
    _require(document.get("independent_review_required") is True, "raw outer observation still names independent review as required")
    _require(
        document.get("local_ipc_boundary") == "source-audited Unix-socket/Mach-service residual; no all-local-IPC-denied claim",
        "outer result hides the Unix-socket/Mach-service residual",
    )
    checks = document.get("checks")
    _require(isinstance(checks, dict) and set(checks) == EXPECTED_OUTER_CHECKS, "outer checks differ from the exact raw 18-key set")
    for name in sorted(EXPECTED_OUTER_CHECKS):
        _require(type(checks[name]) is bool, f"outer check is not a boolean: {name}")
    expected_failed = R1_FAILED_CHECKS if expected_status == "outer_probe_failed" else frozenset()
    actual_failed = frozenset(name for name, value in checks.items() if value is False)
    _require(actual_failed == expected_failed, f"outer failed-check set differs: {sorted(actual_failed)}")
    _require(document.get("negative_controls") == EXPECTED_NEGATIVE_CONTROLS, "outer negative controls are missing, invalid or widened")

    observed = document.get("main_observations")
    expected_observations = COMMON_OBSERVATION_FIELDS if expected_status == "outer_probe_failed" else COMMON_OBSERVATION_FIELDS | {"clang_stdout", "clang_stderr"}
    _require(isinstance(observed, dict) and set(observed) == expected_observations, "outer main-observation fields differ; r1 diagnostics cannot be invented and r2 streams cannot be dropped")
    denied = {"result": "denied", "error": "PermissionError"}
    for name in ("outside_read", "outside_write", "mds_write", "non_proxy_loopback"):
        _require(observed[name] == denied, f"{name} was not denied with the recorded PermissionError")
    _require(observed["empty_host_proxy"] == {"result": "succeeded", "value": 403}, "empty-host Proxy did not return the required 403")
    _require(observed["host_canary"] is None, "host environment canary leaked into the attempt")
    _require(observed["python_child_returncode"] == 13, "Python child did not report the expected fenced-write denial")
    port = observed["proxy_port"]
    _require(type(port) is int and 1 <= port <= 65535, "actual Proxy port is invalid")
    home = observed["home"]
    _require(isinstance(home, str) and home.startswith("/private/var/folders/"), "raw temporary HOME is not the preserved resolved path")
    home_path = Path(home)
    build = home_path.parent
    _require(home_path.name == "home" and build.name == "build" and build.parent.name == "attempt" and build.parent.parent.name == "main", "temporary HOME is outside the expected main/attempt/build layout")
    _require(path_within(home_path, build), "resolved temporary HOME is outside its attempt/build root")
    expected_clang_returncode = 69 if expected_status == "outer_probe_failed" else 0
    _require(observed["clang_returncode"] == expected_clang_returncode, "raw Clang return code was rewritten")
    if expected_status == "outer_probe_observed":
        stdout = observed["clang_stdout"]
        _require(stdout.startswith("Apple clang version 21.0.0 (clang-2100.3.34.2)\n"), "r2 does not preserve the observed Apple Clang 21.0.0 stdout")
        _require("Target: arm64-apple-darwin25.6.0\n" in stdout, "r2 Clang target stdout is missing")
        _require("InstalledDir: /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin\n" in stdout, "r2 Clang real toolchain location is missing")
        _require(observed["clang_stderr"] == "", "r2 Clang stderr was changed")
    _verify_result_hmz(document["hmz"])


def verify_probe_source() -> str:
    """Bind r2 to the corrected probe and independently enumerate its check keys."""

    path = HERE / "outer_probe.py"
    text = path.read_text(encoding="utf-8")
    tree = ast.parse(text)
    check_keys = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == "checks" and isinstance(node.value, ast.Dict):
                check_keys.update(key.value for key in node.value.keys if isinstance(key, ast.Constant) and isinstance(key.value, str))
            if isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name) and target.value.id == "checks":
                if isinstance(target.slice, ast.Constant) and isinstance(target.slice.value, str):
                    check_keys.add(target.slice.value)
    _require(check_keys == EXPECTED_OUTER_CHECKS, "outer_probe.py check construction differs from the exact raw 18-key set")
    required_snippets = (
        "root = Path(directory).resolve()",
        "clang = subprocess.check_output([\"xcrun\", \"-n\", \"-f\", \"clang\"], text=True).strip()",
        "Path(observed[\"home\"]).is_relative_to(root / \"main\" / \"attempt\" / \"build\")",
        "observed[\"clang_stdout\"] = clang.stdout",
        "observed[\"clang_stderr\"] = clang.stderr",
        "(outside / \"secret.txt\").write_text",
        "outside_write.write_text(\"precheck\")",
        "mds_write.write_text(\"precheck\")",
        "listener.bind((\"127.0.0.1\", 0))",
        "with socket.create_connection((\"127.0.0.1\", port), timeout=0.5):",
        "start_new_session=True",
        "os.killpg(detached_pid, 9)",
        "except ArtifactContractError:",
    )
    for snippet in required_snippets:
        _require(snippet in text, f"corrected probe source/control is missing: {snippet}")
    _require("\"/usr/bin/clang\"" not in text, "the probe revived the wrapper Clang instead of the xcrun-resolved path")
    for forbidden in ("xcodebuild -license", "xcodebuild -runFirstLaunch", "hmz_flow(", "Agent(", "subagent("):
        _require(forbidden not in text, f"forbidden outer-probe action appears: {forbidden}")
    compile(text, str(path), "exec")
    return _sha256(path)


def verify_outer_review(document: dict) -> None:
    _require(document.get("schema_id") == OUTER_REVIEW_SCHEMA, "wrong outer-review schema")
    _require(document.get("epoch") == OUTER_REVIEW_EPOCH and document.get("parent_static_epoch") == "mt0-hmz-lower-level-profile-1", "outer-review epoch lineage drift")
    _require(document.get("scope") == "real_outer_lower_level_profile_probe_independent_review", "outer-review scope was widened")
    _require(document.get("status") == "outer_probe_observed_independently_reviewed" and document.get("review_done") is True, "the scoped independent review is not complete")
    _require("not an MT0, frontend, production DSL, Metal, performance or whole-plan pass" in document.get("done_semantics", ""), "outer-review done semantics were widened")

    r1 = _load(R1_FILE)
    r2 = _load(R2_FILE)
    verify_outer_result(r1, expected_status="outer_probe_failed")
    verify_outer_result(r2, expected_status="outer_probe_observed")
    _require(r1["hmz"] == r2["hmz"], "r1/r2 do not bind the same hmz installation identity")
    records = document.get("raw_results")
    _require(isinstance(records, dict) and set(records) == {"r1", "r2", "count_note"}, "the review must retain exactly r1, r2 and the raw-count note")
    _require("18" in records["count_note"] and "not 17" in records["count_note"], "the raw 18-vs-prose-17 distinction was hidden")
    for key, raw, filename, status, failed, clang_rc, streams in (
        ("r1", r1, R1_FILE, "outer_probe_failed", sorted(R1_FAILED_CHECKS), 69, False),
        ("r2", r2, R2_FILE, "outer_probe_observed", [], 0, True),
    ):
        record = records[key]
        _require(set(record) == {"file", "sha256", "status", "real_os_fence_executed", "raw_check_count", "failed_checks", "clang_returncode", "clang_stdout_stderr_fields_present", "retained_unmodified"}, f"{key} review record fields drifted")
        _require(record["file"] == filename and record["sha256"] == _sha256(HERE / filename), f"{key} raw file identity drift")
        _require(record["status"] == status == raw["status"] and record["real_os_fence_executed"] is True, f"{key} status/fence summary drift")
        _require(record["raw_check_count"] == len(raw["checks"]) == 18 and record["failed_checks"] == failed, f"{key} raw check accounting drift")
        _require(record["clang_returncode"] == clang_rc == raw["main_observations"]["clang_returncode"], f"{key} Clang return-code summary drift")
        _require(record["clang_stdout_stderr_fields_present"] is streams, f"{key} Clang stream-presence summary drift")
        _require(record["retained_unmodified"] is True, f"{key} was not retained")

    corrections = document.get("probe_corrections")
    _require(corrections == {
        "probe_root_resolved_before_home_membership": True,
        "private_var_and_var_alias_compare_after_resolve": True,
        "compiler_resolved_with": ["xcrun", "-n", "-f", "clang"],
        "resolved_clang_path_passed_through_PROBE_CLANG": True,
        "r2_clang_stdout_stderr_saved": True,
        "r1_missing_clang_diagnostics_fabricated": False,
        "profile_write_set_expanded": False,
        "hmz_modified": False,
        "xcode_license_accepted": False,
        "gpu_npu_model_or_external_network_used": False,
    }, "probe correction scope or no-widening boundaries drifted")
    identity = document.get("source_identity")
    from tcenv.worker_isolation import hmz_fence as live
    _require(identity.get("outer_probe_sha256") == verify_probe_source(), "outer-review probe source identity drift")
    _require(identity.get("adapter_sha256") == _sha256(ROOT / "tcenv/worker_isolation/hmz_fence.py"), "outer-review adapter identity drift")
    _require(identity.get("driver_sha256") == live.DRIVER_SOURCE_SHA256 and identity.get("trusted_shim_sha256") == live.CHILD_SOURCE_SHA256, "outer-review driver/shim identity drift")
    _require(identity.get("hmz_entrypoint_sha256") == live.HMZ_ENTRYPOINT_SHA256 and identity.get("hmz_source_sha256") == live.HMZ_SOURCE_SHA256, "outer-review hmz source identity drift")
    _require(identity.get("protocol_id") == live.HMZ_PROTOCOL_ID and identity.get("protocol_digest") == _load("source-identity.json")["protocol_digest"], "outer-review protocol identity drift")
    _require(document.get("required_negative_controls") == EXPECTED_NEGATIVE_CONTROLS == r1["negative_controls"] == r2["negative_controls"], "review/raw negative controls differ")

    review = document.get("independent_review")
    _require(review.get("status") == "completed", "independent review is still pending")
    for key in ("outer_probe_rerun", "nested_hmz_or_agent_or_subagent", "raw_result_files_modified", "frontend_installed", "gpu_npu_real_model_or_paid_search"):
        _require(review.get(key) is False, f"independent review boundary drift: {key}")
    _require("read-only" in review.get("method", "") and "negative-control" in review["method"], "independent review method is not the recorded read-only source/raw/control audit")

    residual = document.get("residual_boundaries")
    _require(residual == {
        "loopback_unix_socket_and_mach_service_residual_remains": True,
        "all_local_ipc_eliminated": False,
        "setsid_process_group_gap_explicitly_measured": True,
        "setsid_process_group_gap_closed": False,
        "setsid_descendant_retains_profile_and_denies_outside_write": True,
        "outer_detached_group_cleanup_required": True,
        "memory_cap_or_full_resource_isolation_proven": False,
    }, "loopback/IPC, setsid or resource residuals were hidden")
    _require(r2["checks"]["setsid_process_group_gap_explicitly_measured"] is True and r2["checks"]["setsid_fence_inherited"] is True and r2["checks"]["setsid_outside_write_denied"] is True, "r2 no longer distinguishes setsid survival from inherited-fence denial")

    current = document.get("current_status")
    _require(current.get("real_attack_evidence_status") == "observed_and_independently_reviewed_not_production_pass", "outer evidence was relabelled as a production pass")
    _require(current.get("mt0_status") == "blocked" and current.get("production_ready") is False and current.get("publication_allowed") is False, "MT0/production/publication overclaim")
    _require(current.get("frontend_acceptance") == "not_installed_not_accepted" and current.get("mt1b_status") == "not_started_blocked", "frontend/MT1b was falsely completed")
    _require(current.get("metal_dsl_execution_or_performance_acceptance") is False and current.get("dsl_selection_status") == "blocked_zero_qualified_interface_only", "DSL/Metal acceptance overclaim")
    _require(len(current.get("remaining_blockers", [])) == 6, "remaining MT0/frontend/MT1b/Metal/process/IPC blockers were dropped")
    selection = json.loads((HERE.parent / "MT0" / "dsl-selection.json").read_text(encoding="utf-8"))
    _require(selection["status"] == "blocked" and selection["qualified_candidate_count"] == 0 and not selection["qualified_candidate_ids"], "MT0 DSL selection was falsely completed")
    _require(all(row["availability"] == "interface_only" and row["qualified"] is False for row in selection["candidates"]), "an unaccepted DSL was marked available/qualified")


def verify_nodes() -> None:
    document = _load("nodes.json")
    nodes = _nodes()
    if len(nodes) != EXPECTED_NODES or len(set(nodes)) != EXPECTED_NODES:
        raise ValueError("fixed node registry must contain 100 unique node ids")
    adapter = sum(node.startswith("tests/shared/test_worker_isolation_hmz.py::") for node in nodes)
    evidence = sum(node.startswith("tests/shared/test_mt0_isolation_profile.py::") for node in nodes)
    if (adapter, evidence) != (55, 45) or document["expected_counts"] != {"adapter_and_cpu": 55, "evidence": 45, "total": 100}:
        raise ValueError("fixed node category counts drifted")
    if "does not rerun" not in document["execution"] or "preserved real outer" not in document["execution"]:
        raise ValueError("node registry hides the no-rerun/preserved-outer-evidence boundary")


def verify_results() -> None:
    document = _load("verification.json")
    if document["schema_id"] != "shapeforge.mt0-isolation-profile.verification.2" or document["epoch"] != OUTER_REVIEW_EPOCH:
        raise ValueError("verification result is not the outer-review epoch")
    if document["scope"] != "real_outer_lower_level_profile_probe_independent_review":
        raise ValueError("verification result has the wrong scope")
    if document["fixed_nodes"] != {
        "adapter_and_cpu": 55, "evidence": 45, "total": 100,
        "structure_and_historical_internal_fence": 15, "failure_codes": 3,
        "historical_mt0_dsl_selection": 11, "final_focused_regression": 29,
    }:
        raise ValueError("verification counts drifted")
    if document["read_only_real_installation_inspection"] is not True or document["real_cpu_shim_and_process_group_watchdog"] is not True:
        raise ValueError("required read-only identity and real ordinary CPU checks are missing")
    if document["real_outer_profile_observation_available"] is not True or document["outer_probe_rerun_during_review"] is not False:
        raise ValueError("preserved outer evidence and no-rerun scope were confused")
    if document["r1_preserved_failed"] is not True or document["r2_observed_raw_check_count"] != 18:
        raise ValueError("r1 failure or the exact r2 raw check count was hidden")
    if document["independent_review_status"] != "completed" or document["real_attack_evidence_status"] != "observed_and_independently_reviewed_not_production_pass":
        raise ValueError("independent review state is missing or overclaimed")
    if document["gpu_npu_model_or_paid_search"] is not False or document["mt0_status"] != "blocked":
        raise ValueError("verification invents unauthorized work or completes MT0")
    if document["production_ready"] is not False or document["publication_allowed"] is not False:
        raise ValueError("verification cannot publish or mark production-ready")
    for relative, expected in document["scoped_source_sha256"].items():
        if _sha256(ROOT / relative) != expected:
            raise ValueError(f"scoped source digest mismatch: {relative}")


def verify_manifest() -> None:
    rows = {}
    for line in (HERE / "manifest.sha256").read_text().splitlines():
        if not line.strip():
            continue
        expected, relative = line.split("  ", 1)
        _require(relative not in rows, f"duplicate manifest member: {relative}")
        rows[relative] = expected
    _require(set(rows) == EXPECTED_MANIFEST_MEMBERS, f"manifest file set differs: {sorted(set(rows) ^ EXPECTED_MANIFEST_MEMBERS)}")
    for relative, expected in rows.items():
        path = HERE / relative
        if not path.is_file() or _sha256(path) != expected:
            raise ValueError(f"manifest mismatch: {relative}")


def self_test() -> None:
    original = _load("static-contract.json")
    mutations = []
    production = copy.deepcopy(original)
    production["embedded_dsl_production_ready"] = True
    mutations.append(production)
    attack = copy.deepcopy(original)
    attack["evidence_boundary"]["real_attack_evidence_status"] = "passed"
    mutations.append(attack)
    mt0 = copy.deepcopy(original)
    mt0["mt0_status"] = "complete"
    mutations.append(mt0)
    fallback = copy.deepcopy(original)
    fallback["current_hmz_runtime"]["unfenced_fallback"] = True
    mutations.append(fallback)
    online = copy.deepcopy(original)
    online["decision"]["candidate_build_online"] = True
    mutations.append(online)
    dual = copy.deepcopy(original)
    dual["decision"]["production_route_count"] = 2
    mutations.append(dual)
    mds = copy.deepcopy(original)
    mds["decision"]["mds_required"] = True
    mutations.append(mds)
    mt1b = copy.deepcopy(original)
    mt1b["next_stages"]["MT1b"] = "started"
    mutations.append(mt1b)
    for index, changed in enumerate(mutations):
        try:
            verify_document(changed)
        except ValueError:
            continue
        raise AssertionError(f"static verifier self-test mutation {index} was accepted")

    r1 = _load(R1_FILE)
    r2 = _load(R2_FILE)
    raw_mutations = []
    changed = copy.deepcopy(r1)
    changed["status"] = "outer_probe_observed"
    raw_mutations.append((changed, "outer_probe_failed"))
    changed = copy.deepcopy(r1)
    changed["checks"]["temporary_home"] = True
    raw_mutations.append((changed, "outer_probe_failed"))
    changed = copy.deepcopy(r1)
    changed["main_observations"]["clang_stderr"] = "invented"
    raw_mutations.append((changed, "outer_probe_failed"))
    changed = copy.deepcopy(r2)
    del changed["checks"]["mds_write_denied"]
    raw_mutations.append((changed, "outer_probe_observed"))
    changed = copy.deepcopy(r2)
    changed["checks"]["extra"] = True
    raw_mutations.append((changed, "outer_probe_observed"))
    changed = copy.deepcopy(r2)
    changed["main_observations"]["mds_write"] = {"result": "succeeded"}
    raw_mutations.append((changed, "outer_probe_observed"))
    changed = copy.deepcopy(r2)
    changed["real_os_fence_executed"] = False
    raw_mutations.append((changed, "outer_probe_observed"))
    changed = copy.deepcopy(r2)
    changed["negative_controls"]["artificial_mds_parent_was_writable"] = False
    raw_mutations.append((changed, "outer_probe_observed"))
    changed = copy.deepcopy(r2)
    del changed["main_observations"]["clang_stderr"]
    raw_mutations.append((changed, "outer_probe_observed"))
    changed = copy.deepcopy(r2)
    changed["local_ipc_boundary"] = "all local IPC denied"
    raw_mutations.append((changed, "outer_probe_observed"))
    changed = copy.deepcopy(r2)
    changed["checks"]["setsid_process_group_gap_explicitly_measured"] = False
    raw_mutations.append((changed, "outer_probe_observed"))
    for index, (changed, status) in enumerate(raw_mutations):
        try:
            verify_outer_result(changed, expected_status=status)
        except ValueError:
            continue
        raise AssertionError(f"raw-result verifier self-test mutation {index} was accepted")

    original_review = _load("outer-probe-review.json")
    review_mutations = []
    for key, value in (
        ("production_ready", True), ("mt0_status", "complete"),
        ("frontend_acceptance", "accepted"), ("metal_dsl_execution_or_performance_acceptance", True),
    ):
        changed = copy.deepcopy(original_review)
        changed["current_status"][key] = value
        review_mutations.append(changed)
    changed = copy.deepcopy(original_review)
    changed["residual_boundaries"]["all_local_ipc_eliminated"] = True
    review_mutations.append(changed)
    changed = copy.deepcopy(original_review)
    changed["residual_boundaries"]["setsid_process_group_gap_closed"] = True
    review_mutations.append(changed)
    changed = copy.deepcopy(original_review)
    changed["independent_review"]["status"] = "pending"
    review_mutations.append(changed)
    changed = copy.deepcopy(original_review)
    changed["raw_results"]["r2"]["raw_check_count"] = 17
    review_mutations.append(changed)
    changed = copy.deepcopy(original_review)
    changed["raw_results"].pop("r1")
    review_mutations.append(changed)
    changed = copy.deepcopy(original_review)
    changed["source_identity"]["outer_probe_sha256"] = "0" * 64
    review_mutations.append(changed)
    changed = copy.deepcopy(original_review)
    changed["raw_results"]["r2"]["sha256"] = "0" * 64
    review_mutations.append(changed)
    changed = copy.deepcopy(original_review)
    changed["required_negative_controls"]["artificial_secret_was_readable"] = False
    review_mutations.append(changed)
    changed = copy.deepcopy(original_review)
    changed["independent_review"]["outer_probe_rerun"] = True
    review_mutations.append(changed)
    for index, changed in enumerate(review_mutations):
        try:
            verify_outer_review(changed)
        except ValueError:
            continue
        raise AssertionError(f"outer-review verifier self-test mutation {index} was accepted")


def _pytest() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-n", "0", "-m", "not gpu and not llm and not slow", *_nodes()],
        cwd=ROOT, text=True, capture_output=True, check=False,
    )
    output = completed.stdout + completed.stderr
    if completed.returncode != 0:
        print(output, end="")
        raise ValueError("MT0-isolation-profile fixed nodes failed; read logs/pytest/triage/triage_summary.md")
    match = re.search(r"(\d+) passed", output)
    if match is None or int(match.group(1)) != EXPECTED_NODES or re.search(r"[1-9]\d* (?:skipped|xfailed)", output):
        raise ValueError(f"expected exactly 100 passed without skip/xfail, got: {output[-1000:]}")


def main() -> int:
    verify_document(_load("static-contract.json"))
    verify_source_identity()
    verify_outer_plan()
    verify_cpu_evidence()
    verify_history()
    verify_outer_review(_load("outer-probe-review.json"))
    verify_nodes()
    verify_results()
    self_test()
    if "--check" in sys.argv:
        verify_manifest()
        _pytest()
    print(json.dumps({
        "done": True,
        "scope": "real_outer_lower_level_profile_probe_independent_review",
        "epoch": OUTER_REVIEW_EPOCH,
        "fixed_nodes_passed": EXPECTED_NODES if "--check" in sys.argv else None,
        "manifest": "--check" in sys.argv,
        "r1_status": "outer_probe_failed_preserved",
        "r2_status": "outer_probe_observed_independently_reviewed",
        "r2_raw_checks": "18/18",
        "mt0_status": "blocked",
        "production_ready": False,
        "publication_allowed": False,
        "real_attack_evidence_status": "observed_and_independently_reviewed_not_production_pass",
        "outer_probe_rerun_during_review": False,
        "frontend_or_metal_acceptance": False,
        "local_ipc_and_setsid_residuals_remain": True,
    }, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
