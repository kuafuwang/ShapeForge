"""Fixed tests for the MT0 TileLang-first install and source-only probe package.

No tilelang import, no sandbox-exec, no GPU/NPU, no model and no paid search.
The package records a completed source-only probe with four preserved failures.
These tests prevent anyone from rewording that into MT0/Metal/production success,
from dropping the raw negative cases, or from erasing the failed rounds.
"""

from __future__ import annotations

import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import types

import pytest

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "tests" / "support" / "golden" / "shared" / "多DSL-Track-阶段验收" / "MT0-tilelang-probe"
ARTIFACT_NAMES = (
    "frontend_identity.json",
    "kernel.metal",
    "device_ir.json",
    "kernel_params.json",
    "host_module.txt",
    "build_identity.json",
)
BUILD_ARTIFACT_NAMES = ARTIFACT_NAMES[:-1]
EVIDENCE_DIR = PACKAGE / "evidence" / "source-only-r5"
FAILED_RESULTS = {
    "r1": "probe-result.json",
    "r2": "probe-result-r2.json",
    "r3": "probe-result-r3.json",
    "r4": "probe-result-r4.json",
}


def _load(name: str) -> dict:
    return json.loads((PACKAGE / name).read_text(encoding="utf-8"))


def _module(name: str, path: Path) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    from tests.support.boundaries import rebind_acceptance_root

    rebind_acceptance_root(module)
    return module


def _verifier() -> types.ModuleType:
    return _module("mt0_tilelang_probe_verify", PACKAGE / "verify.py")


def _runner() -> types.ModuleType:
    return _module("mt0_tilelang_probe_runner", PACKAGE / "probe" / "run_probe.py")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _identity_mutation_fixture(tmp_path: Path, mutation: str):
    """Build a tiny installed tree and drive the real frozen entry, not a mock verifier.

    The install document is mutated before the runner's real pinned-document
    projection.  The entry then performs its normal package/dist-info checks;
    no TileLang import or lowering is needed for any of these negative cases.
    """

    entry = _module("mt0_tilelang_frontend_entry_negative", PACKAGE / "probe/frontend_entry.py")
    runner = _runner()
    site_packages = tmp_path / "venv/lib/python3.13/site-packages"
    package = site_packages / "tilelang"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("__version__ = '0.1.15'\n", encoding="utf-8")
    (package / "module.py").write_text("# small but real package payload\n", encoding="utf-8")

    install = copy.deepcopy(_load("frontend-install.json"))
    if mutation == "pinned_version":
        # The directory follows the requested pin, while METADATA describes the
        # actually installed 0.1.15 distribution.  This reaches version checking
        # rather than failing early on a missing dist-info directory.
        install["pinned"]["version"] = "0.1.16"
    dist_info = site_packages / f"tilelang-{install['pinned']['version']}.dist-info"
    dist_info.mkdir()
    metadata = dist_info / "METADATA"
    record = dist_info / "RECORD"
    metadata.write_text("Metadata-Version: 2.1\nName: tilelang\nVersion: 0.1.15\n", encoding="utf-8")
    record.write_text("tilelang/__init__.py,,\n", encoding="utf-8")

    files, total, digest = entry.package_manifest(package)
    installed = install["installed"]
    installed.update({
        "tilelang_package_files": files,
        "tilelang_package_bytes": total,
        "tilelang_package_manifest_sha256": digest,
        "dist_info_metadata_sha256": _sha(metadata),
        "dist_info_record_sha256": _sha(record),
    })
    if mutation == "package_manifest":
        installed["tilelang_package_manifest_sha256"] = "0" * 64
    elif mutation == "metadata":
        installed["dist_info_metadata_sha256"] = "0" * 64
    elif mutation == "record":
        installed["dist_info_record_sha256"] = "0" * 64
    elif mutation != "pinned_version":
        raise AssertionError(f"unknown identity mutation: {mutation}")
    return entry, install, package, runner.pinned_frontend_document(install)


@pytest.mark.parametrize("mutation", ["pinned_version", "package_manifest", "metadata", "record"])
def test_frontend_identity_drift_rejected_before_full_build(tmp_path, monkeypatch, capsys, mutation):
    entry, install, package, pinned = _identity_mutation_fixture(tmp_path, mutation)
    pinned_path = tmp_path / "pinned.json"
    pinned_path.write_text(json.dumps(pinned), encoding="utf-8")
    candidate = tmp_path / "candidate.py"
    candidate.write_text("# identity failure must happen before candidate loading\n", encoding="utf-8")
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()

    monkeypatch.setattr(
        entry.importlib.util,
        "find_spec",
        lambda name: types.SimpleNamespace(submodule_search_locations=[str(package)]),
    )
    lowering_calls = []

    def lowering_must_not_run(*args, **kwargs):
        lowering_calls.append((args, kwargs))
        raise AssertionError("lowering/full build was reached after identity drift")

    monkeypatch.setattr(entry, "lower_source_only", lowering_must_not_run)
    returncode = entry.main([
        "--candidate", str(candidate), "--pinned", str(pinned_path), "--artifacts", str(artifacts),
    ])
    stderr = capsys.readouterr().err
    expected_detail = {
        "pinned_version": "dist-info version drift",
        "package_manifest": "package manifest digest drift",
        "metadata": "dist-info METADATA drift",
        "record": "dist-info RECORD drift",
    }[mutation]
    assert returncode == entry.EXIT_IDENTITY_DRIFT
    assert "frontend-probe: IdentityDrift:" in stderr
    assert expected_detail in stderr
    assert lowering_calls == []
    # The real entry writes build_identity.json only on its full-build success
    # path.  A typed non-zero exit with no artifact is therefore no full build
    # and emits no production-ready success result.
    full_build = (artifacts / "build_identity.json").exists()
    production_ready = returncode == 0 and full_build
    assert full_build is False
    assert production_ready is False
    assert list(artifacts.iterdir()) == []


def test_probe_completed_status_keeps_mt0_blocked_and_does_not_overclaim_mt0():
    document = _load("probe-status.json")
    assert document["status"] == "source_only_probe_completed_mt0_blocked"
    assert document["typed_code"] == "source_only_probe_completed"
    assert document["prior_typed_code"]["code"] == "backend_capability_unavailable"
    probe = document["probe"]
    assert probe["executed"] is True
    assert probe["full_build"] is True
    assert probe["attempts"] == 5
    assert probe["failed_attempts"] == ["r1", "r2", "r3", "r4"]
    assert [row["path"] for row in probe["artifacts_observed"]] == list(ARTIFACT_NAMES)
    assert probe["metal_source_generated"] is True
    assert probe["device_dispatch_executed"] is False
    assert probe["device_dispatch_calls"] == 0
    assert probe["enable_device_compile"] is False
    assert probe["online"] is False and probe["hosts"] == []
    assert probe["mds_write_roots"] == []
    current = document["current_status"]
    assert current["mt0_status"] == "blocked"
    assert current["mt1b_status"] == "not_started_blocked"
    assert current["production_ready"] is False
    assert current["publication_allowed"] is False
    assert current["metal_dsl_execution_or_performance_acceptance"] is False
    assert current["metal_dispatch_or_library_executed"] is False
    assert len(current["mt0_blockers"]) == 1
    assert "second non-MSL DSL" in current["mt0_blockers"][0]
    mt1b_blockers = " ".join(current["mt1b_blockers"])
    assert "MT0" in mt1b_blockers and "selection" in mt1b_blockers
    assert "does not mean MT0" in document["done_semantics"]
    assert any("second non-MSL DSL" in gap for gap in document["gaps"])


def test_round1_is_tilelang_only_and_halide_is_absent():
    decision = _load("probe-status.json")["round1_dsl_decision"]
    assert decision["selected_first_round_dsl"] == "tilelang"
    assert decision["pinned_version"] == "0.1.15"
    assert decision["second_non_msl_dsl"] == "deferred_to_later_round"
    assert decision["toy_or_substitute_routes_used"] is False
    halide = decision["halide"]
    assert halide["installed"] is False
    assert halide["built"] is False
    assert halide["probed"] is False
    assert halide["code_or_dependency_added"] is False


def test_frontend_install_binds_pinned_wheel_license_and_python():
    document = _load("frontend-install.json")
    assert document["schema_id"] == "shapeforge.mt0-tilelang-probe.frontend-install.1"
    assert document["status"] == "staged_and_verified"
    assert document["pinned"]["version"] == "0.1.15"
    assert document["pinned"]["upstream_commit"] == "a35f8ddf45eba16c21211ec8822d56ce5363036f"
    assert document["pinned"]["tvm_commit"] == "4211874e9e4b8770fd20ad0d98ab80afd1df5028"
    wheel = document["wheel"]
    assert wheel["filename"] == "tilelang-0.1.15-cp39-abi3-macosx_11_0_arm64.whl"
    assert wheel["sha256"] == wheel["pypi_published_sha256"]
    assert wheel["sha256"] == "dcffc9aa8d4cd086dd93090f5afeeaa38aa621d2bb80998c4e5f5ee07be116aa"
    assert wheel["bytes"] == 33667284
    assert wheel["canonical_url"].startswith("https://files.pythonhosted.org/")
    assert document["download"]["mirror_url"].startswith("https://pypi.tuna.tsinghua.edu.cn/")
    assert document["license"]["spdx"] == "MIT"
    assert document["license"]["sha256"] == "f70e5cdc24c3797af741930440086f9a0ad36930b05a3dfc25b420936dc2cfa1"
    assert "Microsoft" in document["license"]["note"]
    assert document["python"]["version"] == "3.13.1"
    assert document["python"]["venv_python"] == document["installed"]["staging_root"] + "/venv/bin/python"


def test_install_identity_counts_and_provenance_boundary():
    document = _load("frontend-install.json")
    assert document["download"]["all_wheel_sha256_verified_against_pypi"] is True
    assert document["download"]["wheel_count"] == 19 == len(document["dependencies"])
    assert len({row["project"] for row in document["dependencies"]}) == 19
    installed = document["installed"]
    assert installed["record_files_checked"] == installed["record_files_verified"] == 21199
    assert installed["record_mismatches"] == {} and installed["record_missing"] == {}
    assert installed["tilelang_package_files"] == 3651
    assert installed["tilelang_package_manifest_sha256"] == "06f3b0adee7abce77af1111cb36e704b4b23c3cbe38d095dd62cce86509c11cc"
    provenance = document["source_provenance"]
    assert provenance["tilelang_python_identical"] is True
    assert provenance["tilelang_python_files_in_tag"] == 390
    assert provenance["tilelang_python_files_compared"] == 390
    assert provenance["tilelang_python_files_in_wheel"] == 1137
    assert provenance["tilelang_python_only_in_wheel"] == 747
    assert provenance["tvm_python_identical"] is True
    assert provenance["tvm_python_files_in_commit"] == 745
    assert provenance["native_libraries_rebuilt_from_source"] is False
    assert "not rebuilt" in provenance["boundary"]
    boundary = document["execution_boundary"]
    assert boundary["tilelang_imported"] is False
    assert boundary["codegen_executed"] is False
    assert boundary["gpu_npu_executed"] is False
    assert document["isolation_of_install"]["repo_contains_wheel_bytes"] is False
    assert document["installed"]["staging_root_mode"] == "0700"
    assert document["isolation_of_install"]["staging_root_not_group_or_world_accessible"] is True


def test_context_capability_records_nested_sandbox_denial():
    document = _load("context-capability.json")
    observations = document["observations"]
    assert observations["minimal_profile"]["returncode"] == 71
    assert observations["minimal_profile"]["stderr"] == "sandbox-exec: sandbox_apply: Operation not permitted\n"
    assert observations["deny_network_profile"]["returncode"] == 71
    assert observations["deny_network_profile"]["stderr"] == observations["minimal_profile"]["stderr"]
    assert "Operation not permitted" in observations["ps_denied"]["error"]
    assert observations["writability"]["home"]["writable"] is False
    assert observations["writability"]["tmp"]["writable"] is True
    assert observations["writability"]["repo"]["writable"] is True
    interpretation = document["interpretation"]
    assert interpretation["typed_code"] == "backend_capability_unavailable"
    assert interpretation["real_probe_executed"] is False
    assert interpretation["frontend_imported"] is False
    assert interpretation["candidate_executed"] is False
    assert interpretation["full_build"] is False
    assert interpretation["unfenced_fallback_allowed"] is False
    assert interpretation["capability_probe_only"] is True
    assert interpretation["attack_or_candidate_fixture_executed"] is False
    assert interpretation["probe_shell_or_true_executed"] is False
    assert "outer control plane" in interpretation["why_deferred"]


def test_context_capability_is_not_outer_isolation_evidence():
    document = _load("context-capability.json")
    hmz = document["hmz_installation_read_only_inspection"]
    assert hmz["performed"] is True and hmz["launched"] is False
    assert hmz["protocol_digest"] == "c6a6a61052d8e7a19b30a9674b685b75378d8999791eff0ef1c6c570cb762c83"
    assert hmz["version"] == "0.2.2" and hmz["python_version"] == "3.14.4"
    assert hmz["implicit_mds_write_roots"] == []
    assert hmz["internal_fence_or_wrap_used"] is False
    assert hmz["runtime_attack_evidence"] == "not established by installation inspection"
    joined = " ".join(document["not_evidence_for"])
    assert "outer Seatbelt" in joined
    assert "MT0 completion" in joined
    assert "Metal execution or performance" in joined


def test_probe_plan_is_source_only_and_explicit_target():
    document = _load("probe-plan.json")
    assert document["candidate"]["execution_model"] == "embedded_host_code"
    assert document["candidate"]["sha256"] == _sha(PACKAGE / "probe/candidate.py")
    assert document["frontend"]["entry_sha256"] == _sha(PACKAGE / "probe/frontend_entry.py")
    assert document["runner"]["sha256"] == _sha(PACKAGE / "probe/run_probe.py")
    frontend = document["frontend"]
    assert frontend["api"] == "tilelang.engine.lower"
    assert frontend["target"] == {"kind": "metal", "keys": ["metal", "gpu"], "target_host": "c"}
    assert frontend["options"] == {"enable_host_codegen": True, "enable_device_compile": False}
    note = frontend["import_time_mps_query"]
    assert "device presence query" in note and "never initiates" in note
    assert "zero dispatches" in note
    forbidden = set(frontend["forbidden_calls"])
    assert "tilelang.compile" in forbidden
    assert "tilelang.jit" in forbidden
    assert "MTLDevice/MTLLibrary/MTLComputePipelineState" in forbidden
    isolation = document["isolation"]
    assert isolation["online"] is False
    assert isolation["hosts"] == []
    assert isolation["implicit_mds_write_roots"] == []
    assert isolation["frontend_root_mounted_read_only"] is True
    assert isolation["candidate_has_network"] is False
    assert isolation["limits"]["wall_seconds"] == 300
    assert isolation["limits"]["artifact_total_bytes"] == 33554432
    evidence = document["evidence"]
    assert evidence["real_probe_result"] == "probe-result-r5.json"
    assert evidence["artifact_dir"] == "evidence/source-only-r5"
    assert evidence["attempts"] == 5
    assert evidence["failed_rounds"] == ["r1", "r2", "r3", "r4"]
    assert "decorator" in evidence["fix_scope"] and "with target:" in evidence["fix_scope"]


def test_probe_plan_artifact_contract_and_negative_matrix():
    document = _load("probe-plan.json")
    contract = document["artifact_contract"]
    assert tuple(contract["exact_set"]) == ARTIFACT_NAMES
    assert contract["per_file_bytes"] == 16777216
    assert contract["total_bytes"] == 33554432
    required = " ".join(contract["full_build_requires"])
    for needle in ("Metal source", "device IR", "kernel params", "host artifact", "frontend identity", "build identity"):
        assert needle in required
    matrix = {row["id"]: row for row in document["negative_matrix"]}
    assert set(matrix) == {
        "frontend_version_drift", "frontend_source_drift", "candidate_sensitive_read",
        "candidate_outside_write", "candidate_network_connect", "undeclared_artifact",
        "codegen_timeout", "candidate_nonzero", "missing_toolchain",
    }
    for row in matrix.values():
        assert "no full build" in row["expected"]
    real = set(document["evidence"]["negative_results"]["real_executed"])
    static = set(document["evidence"]["negative_results"]["static_or_adapter_covered"])
    assert real == {"candidate_sensitive_read", "candidate_outside_write", "candidate_network_connect", "undeclared_artifact"}
    assert static == {"frontend_version_drift", "frontend_source_drift", "codegen_timeout", "candidate_nonzero", "missing_toolchain"}
    assert real | static == set(matrix)
    fixtures = document["fixtures"]
    assert set(fixtures) == {"read_outside", "write_outside", "network_connect", "undeclared_artifact"}


def test_probe_sources_and_fixtures_are_frozen():
    document = _load("probe-plan.json")
    for relative, expected in (
        ("probe/candidate.py", document["candidate"]["sha256"]),
        ("probe/frontend_entry.py", document["frontend"]["entry_sha256"]),
        ("probe/run_probe.py", document["runner"]["sha256"]),
    ):
        assert hashlib.sha256((PACKAGE / relative).read_bytes()).hexdigest() == expected
    for name, expected in document["fixtures"].items():
        filename = {"read_outside": "read_outside.py", "write_outside": "write_outside.py",
                    "network_connect": "network_connect.py", "undeclared_artifact": "undeclared_artifact.py"}[name]
        assert hashlib.sha256((PACKAGE / "probe" / "fixtures" / filename).read_bytes()).hexdigest() == expected


def test_entry_and_runner_do_not_execute_device_code():
    entry_text = (PACKAGE / "probe/frontend_entry.py").read_text(encoding="utf-8")
    entry_tree = ast.parse(entry_text)
    module_imports = set()
    for node in entry_tree.body:
        if isinstance(node, ast.Import):
            module_imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            module_imports.add(node.module)
    assert "tilelang" not in module_imports and "torch" not in module_imports
    all_imports = set()
    for node in ast.walk(entry_tree):
        if isinstance(node, ast.Import):
            all_imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            all_imports.add(node.module)
    assert not ({"subprocess", "socket", "urllib", "http"} & {name.split(".")[0] for name in all_imports})
    forbidden_attributes = {"jit", "compile", "build", "device_codegen", "newLibraryWithSource",
                            "dispatchThreadgroups", "makeCommandBuffer"}
    for node in ast.walk(entry_tree):
        if isinstance(node, ast.Attribute):
            assert node.attr not in forbidden_attributes
    assert entry_text.count("with target:") == 1
    assert entry_text.index("with target:") < entry_text.index("tilelang_lower(")
    runner_tree = ast.parse((PACKAGE / "probe/run_probe.py").read_text(encoding="utf-8"))
    runner_imports = set()
    for node in ast.walk(runner_tree):
        if isinstance(node, ast.Import):
            runner_imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            runner_imports.add(node.module)
    assert not any(name.startswith("tilelang") for name in runner_imports)
    assert "tcenv.worker_isolation" in runner_imports


def test_runner_builds_the_offline_attempt_contract(tmp_path):
    runner = _runner()
    install = {
        "pinned": {
            "version": "0.1.15", "upstream_commit": "a" * 40, "tvm_commit": "b" * 40,
        },
        "wheel": {"sha256": "c" * 64},
        "python": {"venv_python": str(tmp_path / "venv/bin/python")},
        "installed": {
            "staging_root": str(tmp_path / "frontend"),
            "tilelang_package_files": 3651, "tilelang_package_bytes": 134219728,
            "tilelang_package_manifest_sha256": "d" * 64,
            "dist_info_metadata_sha256": "e" * 64, "dist_info_record_sha256": "f" * 64,
        },
    }
    args = types.SimpleNamespace(attempt_root=tmp_path / "attempt", wall_seconds=300.0)
    spec, plan = runner.build_spec(args, install)
    assert spec.argv[0] == install["python"]["venv_python"]
    assert spec.argv[1] == "-I"
    assert spec.argv[2] == str(tmp_path / "attempt/input/frontend_entry.py")
    assert spec.argv[-1] == str(tmp_path / "attempt/build/artifacts")
    assert spec.artifacts == ARTIFACT_NAMES
    assert set(spec.input_files) == {"frontend_entry.py", "candidate.py", "pinned-frontend.json"}
    assert spec.read_only_roots == (tmp_path / "frontend",)
    assert set(spec.extra_env) == {"TILELANG_CACHE_DIR", "TILELANG_DISABLE_CACHE", "TILELANG_PRINT_ON_COMPILATION"}
    assert spec.limits.wall_seconds == 300.0
    assert spec.limits.cpu_seconds == 330
    assert spec.limits.artifact_total_bytes == 32 * 1024**2
    assert plan["artifacts"] == ARTIFACT_NAMES
    pinned_doc = json.loads(spec.input_files["pinned-frontend.json"])
    assert pinned_doc["wheel_sha256"] == "c" * 64
    assert pinned_doc["tilelang_package_manifest_sha256"] == "d" * 64


def test_runner_preflight_fails_closed_without_unfenced_fallback():
    text = (PACKAGE / "probe/run_probe.py").read_text(encoding="utf-8")
    assert "PROFILE_PROBE" in text
    assert '"sandbox_apply_possible": completed.returncode == 0' in text
    assert 'if not capability["sandbox_apply_possible"]:' in text
    assert "return 3" in text
    assert "--no-sandbox" not in text
    start = text.index("if not capability[\"sandbox_apply_possible\"]:")
    assert "executor.run(spec)" in text[start:]
    assert text.index("executor.run(spec)") > start


def test_verifier_accepts_the_current_package():
    verifier = _verifier()
    verifier.verify_frontend_install(_load("frontend-install.json"))
    verifier.verify_context_capability(_load("context-capability.json"))
    verifier.verify_plan(_load("probe-plan.json"))
    verifier.verify_status(_load("probe-status.json"))
    verifier.verify_probe_result(_load("probe-result-r5.json"))
    verifier.verify_evidence()
    verifier.verify_negative_cases()
    verifier.verify_probe_history()
    verifier.verify_fix_chain()
    verifier.verify_review(_load("review.json"))
    verifier.verify_sources()
    verifier.verify_nodes()
    verifier.verify_history()
    verifier.verify_history_anchor_drift(_load("history-anchor-drift.json"))


@pytest.mark.parametrize(
    "mutation",
    ["mt0", "mt1b", "production", "second_dsl", "dispatch", "device_compile", "online", "mds", "no_evidence"],
)
def test_verifier_rejects_status_overclaim(mutation):
    verifier = _verifier()
    document = copy.deepcopy(_load("probe-status.json"))
    if mutation == "mt0":
        document["current_status"]["mt0_status"] = "complete"
    elif mutation == "mt1b":
        document["current_status"]["mt1b_status"] = "started"
    elif mutation == "production":
        document["current_status"]["production_ready"] = True
    elif mutation == "second_dsl":
        document["round1_dsl_decision"]["second_non_msl_dsl"] = "halide"
    elif mutation == "dispatch":
        document["probe"]["device_dispatch_calls"] = 1
    elif mutation == "device_compile":
        document["probe"]["enable_device_compile"] = True
    elif mutation == "online":
        document["probe"]["online"] = True
    elif mutation == "mds":
        document["probe"]["mds_write_roots"] = ["/Users/x/Library/Caches"]
    else:
        document["probe"]["artifacts_observed"] = []
    with pytest.raises(ValueError):
        verifier.verify_status(document)


@pytest.mark.parametrize("mutation", ["device_compile", "metal4", "online", "drop_negative", "drop_exclusion", "evidence_digest"])
def test_verifier_rejects_plan_mutation(mutation):
    verifier = _verifier()
    document = copy.deepcopy(_load("probe-plan.json"))
    if mutation == "device_compile":
        document["frontend"]["options"]["enable_device_compile"] = True
    elif mutation == "metal4":
        document["frontend"]["target"]["keys"] = ["metal", "gpu", "metal4"]
    elif mutation == "online":
        document["isolation"]["online"] = True
    elif mutation == "drop_negative":
        document["negative_matrix"] = [row for row in document["negative_matrix"] if row["id"] != "missing_toolchain"]
    elif mutation == "drop_exclusion":
        document["completion_predicate"]["explicitly_not_implied"] = []
    else:
        document["evidence"]["artifact_set_sha256"] = "0" * 64
    with pytest.raises(ValueError):
        verifier.verify_plan(document)


@pytest.mark.parametrize("mutation", ["wheel", "record", "native_rebuild", "imported", "license", "dependency_publish"])
def test_verifier_rejects_install_mutation(mutation):
    verifier = _verifier()
    document = copy.deepcopy(_load("frontend-install.json"))
    if mutation == "wheel":
        document["wheel"]["sha256"] = "0" * 64
    elif mutation == "record":
        document["installed"]["record_files_verified"] = 21198
    elif mutation == "native_rebuild":
        document["source_provenance"]["native_libraries_rebuilt_from_source"] = True
    elif mutation == "imported":
        document["execution_boundary"]["tilelang_imported"] = True
    elif mutation == "dependency_publish":
        document["dependencies"][0]["pypi_published_sha256"] = "0" * 64
    else:
        document["license"]["sha256"] = "0" * 64
    with pytest.raises(ValueError):
        verifier.verify_frontend_install(document)


def test_archived_source_only_artifacts_match_result_and_identity():
    verifier = _verifier()
    expected_rows = []
    for name in BUILD_ARTIFACT_NAMES:
        path = EVIDENCE_DIR / name
        expected_rows.append({"path": name, "bytes": path.stat().st_size, "sha256": _sha(path)})
    result = _load("probe-result-r5.json")
    assert result["build"]["artifact_rows"] == expected_rows
    identity = _load("evidence/source-only-r5/build_identity.json")
    assert identity["artifacts"] == expected_rows
    digest = hashlib.sha256(
        json.dumps(expected_rows, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    assert digest == result["build"]["artifact_set_sha256"] == identity["artifact_set_sha256"]
    assert identity["candidate_sha256"] == _sha(PACKAGE / "probe/candidate.py")
    assert identity["artifact_set_sha256"].startswith("bc6a7300")
    verifier.verify_evidence()
    host = (EVIDENCE_DIR / "host_module.txt").read_text(encoding="utf-8")
    guard_start = host.index("#ifdef __OBJC__")
    guard_end = host.index("#endif", guard_start)
    guarded_headers = host[guard_start:guard_end]
    for header in ("<Metal/Metal.h>", "<Foundation/Foundation.h>", "<torch/mps.h>"):
        assert header in guarded_headers
    for device_text in ("torch::mps::get_dispatch_queue()", "dispatch_sync", "torch::mps::get_command_buffer()", "metal.SetStream"):
        assert device_text in host and host.index(device_text) > guard_end
    # A guard-confused source is a real verifier rejection, not just a wording
    # assertion: move one later call into the header-only guard.
    call = "  auto serialQueue = torch::mps::get_dispatch_queue();\n"
    misguarded = host.replace(call, "", 1).replace(
        "#include <torch/mps.h>\n#endif", "#include <torch/mps.h>\n" + call + "#endif", 1,
    )
    with pytest.raises(ValueError):
        verifier._check_host_source(misguarded)
    for name in ARTIFACT_NAMES:
        path = EVIDENCE_DIR / name
        assert path.is_file() and not path.is_symlink() and path.stat().st_nlink == 1


def test_probe_result_binds_zero_dispatch_offline_and_no_mds():
    verifier = _verifier()
    document = _load("probe-result-r5.json")
    verifier.verify_probe_result(document)
    assert document["status"] == "source_only_frontend_probe_completed"
    assert document["gpu_npu_executed"] is False
    assert document["device_dispatch_calls"] == 0
    assert document["build"]["device_execution"]["dispatch_calls"] == 0
    assert document["build"]["device_execution"]["device_library_created"] is False
    assert document["build"]["device_execution"]["pipeline_state_created"] is False
    assert document["build"]["device_execution"]["buffer_allocated"] is False
    assert document["isolation"]["policy_online"] is False
    assert document["isolation"]["policy_hosts"] == []
    assert document["isolation"]["implicit_mds_write_roots"] == []
    assert document["isolation"]["driver_report"]["mds_write_root_added"] is False
    assert document["isolation"]["driver_report"]["attack_containment_proven"] is False
    write_roots = document["isolation"]["driver_report"]["profile"]["write"]
    expected_roots = {
        "/dev/null", "/dev/zero", "/dev/random", "/dev/urandom",
        "/dev/tty", "/dev/ptmx", "/dev/fd", "/dev/dtracehelper",
        *{f"/private/tmp/shapeforge-mt0-tilelang-attempt-r5/build/{name}"
          for name in ("work", "home", "cache", "config", "data", "tmp", "pycache", "artifacts")},
    }
    assert set(write_roots) == expected_roots and len(write_roots) == len(expected_roots)
    assert "Library" not in " ".join(write_roots)
    assert document["mt0_status"] == "blocked"
    assert document["production_ready"] is False and document["publication_allowed"] is False


def test_failure_history_r1_to_r4_is_preserved():
    verifier = _verifier()
    verifier.verify_probe_history()
    history = _load("probe-history.json")
    rounds = [row["round"] for row in history["attempts"]]
    assert rounds == ["r1", "r2", "r3", "r4", "r5"]
    for round_id, filename in FAILED_RESULTS.items():
        raw = _load(filename)
        assert raw["full_build"] is False
        assert raw["probe_executed"] is False
        assert raw["isolation"]["returncode"] != 0
        assert raw["mt0_status"] == "blocked"
        assert raw["status"] == "full_build_rejected"
        assert raw["typed_code"] == "backend_capability_unavailable"
        row = next(item for item in history["attempts"] if item["round"] == round_id)
        assert row["full_build"] is False and row["root_cause"]
        assert row["status"] == raw["status"]
        assert row["typed_code"] == raw["typed_code"]
        assert row["stderr_sha256"] == _sha(PACKAGE / row["stderr_log"])
    r3_log = (PACKAGE / "evidence/failure-logs/r3-stderr.txt").read_text(encoding="utf-8")
    r4_log = (PACKAGE / "evidence/failure-logs/r4-stderr.txt").read_text(encoding="utf-8")
    assert "The parser does not understand the decorator" in r3_log
    assert "Target context required" in r4_log
    assert history["attempts"][-1]["candidate_sha256"] == _sha(PACKAGE / "probe/candidate.py")
    assert history["attempts"][-1]["frontend_entry_sha256"] == _sha(PACKAGE / "probe/frontend_entry.py")


def test_fix_chain_changes_only_decorator_and_target_context():
    verifier = _verifier()
    verifier.verify_fix_chain()
    candidate = (PACKAGE / "probe/candidate.py").read_text(encoding="utf-8")
    assert "@T.prim_func" in candidate
    assert "@prim_func\n" not in candidate
    assert "from tilelang.language.tir import prim_func" not in candidate
    assert "N = 1024" in candidate and "THREADS = 256" in candidate
    entry = (PACKAGE / "probe/frontend_entry.py").read_text(encoding="utf-8")
    assert entry.count("with target:") == 1
    assert entry.index("with target:") < entry.index("tilelang_lower(")
    assert "enable_device_compile=False" in entry
    history = _load("probe-history.json")
    assert history["fix_chain"]["candidate"]["chosen_after_round"] == "r4"
    assert history["fix_chain"]["frontend_entry"]["r3_to_r4_changed"] == []
    assert "No algorithm" in history["fix_chain"]["frontend_entry"]["not_a_rewrite"]


def test_review_verdict_is_narrow_and_evidence_bound():
    verifier = _verifier()
    document = _load("review.json")
    verifier.verify_review(document)
    verdict = document["verdict"]
    assert verdict["done"] is True
    assert "source-only isolated build" in verdict["done_scope"]
    assert "negative cases" in verdict["done_scope"]
    not_done = " ".join(verdict["not_done"])
    for needle in ("MT0", "MT1b", "second non-MSL DSL", "production DSL", "real Metal correctness", "publication"):
        assert needle in not_done
    assert "does not mean MT0" in document["notes"]
    findings = " ".join(document["findings"])
    assert "not an OS-level trace" in findings
    assert "PermissionError" in findings
    assert "first network" in findings
    assert "installed_tree_manifest_sha256" in findings


def test_negative_cases_split_real_execution_from_static_coverage():
    verifier = _verifier()
    verifier.verify_negative_cases()
    status = _load("probe-status.json")
    real = [row["id"] for row in status["negative_cases"]["real_executed"]]
    static = [row["id"] for row in status["negative_cases"]["static_or_adapter_covered"]]
    assert set(real) == {"candidate_sensitive_read", "candidate_outside_write", "candidate_network_connect", "undeclared_artifact"}
    assert set(static) == {"frontend_version_drift", "frontend_source_drift", "codegen_timeout", "candidate_nonzero", "missing_toolchain"}
    assert status["negative_cases"]["authority"] == "fixture_observation"
    assert "fixture_observation" in status["negative_cases"]["top_level_status_note"]
    for filename in ("negative-read-outside.json", "negative-write-outside.json", "negative-network.json", "negative-artifact.json"):
        raw = _load(filename)
        assert raw["status"] == "blocked_backend_capability_unavailable"
        assert raw["fixture_observation"]["matched"] is True
        assert raw["full_build"] is False and raw["probe_executed"] is False
        assert "build" not in raw
    for filename in ("negative-read-outside.json", "negative-write-outside.json"):
        raw = _load(filename)
        assert raw["fixture_observation"]["returncode"] == 13
        assert raw["fixture_observation"]["artifact_count"] == 0
        assert raw["fixture_observation"]["outside_canary_unchanged"] is True
    network = _load("negative-network.json")
    assert network["fixture_observation"]["returncode"] == 13
    assert network["fixture_observation"]["artifact_count"] == 0
    artifact = _load("negative-artifact.json")
    assert artifact["fixture_observation"]["status"] == "artifact_contract_rejected"
    assert "isolation" not in artifact


@pytest.mark.parametrize("mutation", ["kernel_bytes", "artifact_set", "dispatch", "missing_artifact"])
def test_verifier_rejects_artifact_mutation(mutation):
    verifier = _verifier()
    if mutation == "kernel_bytes":
        text = (EVIDENCE_DIR / "kernel.metal").read_text(encoding="utf-8")
        with pytest.raises(ValueError):
            verifier._check_kernel_source(text.replace("2.000000e+00f", "3.000000e+00f"))
        return
    if mutation == "missing_artifact":
        with pytest.raises(ValueError):
            verifier._check_evidence_members([name for name in ARTIFACT_NAMES if name != "kernel.metal"])
        return
    rows = verifier._evidence_rows()
    install = _load("frontend-install.json")
    candidate_sha = _sha(PACKAGE / "probe/candidate.py")
    document = copy.deepcopy(_load("evidence/source-only-r5/build_identity.json"))
    if mutation == "artifact_set":
        document["artifacts"][0]["sha256"] = "0" * 64
    else:
        document["device_execution"]["dispatch_calls"] = 1
    with pytest.raises(ValueError):
        verifier.verify_build_identity_document(document, rows, install, candidate_sha)


@pytest.mark.parametrize("mutation", ["write_extra", "write_other_attempt", "write_missing"])
def test_verifier_rejects_probe_result_mutation(mutation):
    verifier = _verifier()
    document = copy.deepcopy(_load("probe-result-r5.json"))
    write_roots = document["isolation"]["driver_report"]["profile"]["write"]
    if mutation == "write_extra":
        write_roots.append("/private/tmp/unrelated/build/escape")
    elif mutation == "write_other_attempt":
        write_roots[-1] = "/private/tmp/shapeforge-mt0-tilelang-attempt/build/artifacts"
    else:
        write_roots.pop()
    with pytest.raises(ValueError):
        verifier.verify_probe_result(document)


@pytest.mark.parametrize("mutation", ["matched_false", "artifacts_present", "success_status"])
def test_verifier_rejects_negative_case_overclaim(mutation):
    verifier = _verifier()
    document = copy.deepcopy(_load("negative-read-outside.json"))
    if mutation == "matched_false":
        document["fixture_observation"]["matched"] = False
    elif mutation == "artifacts_present":
        document["fixture_observation"]["artifact_count"] = 1
    else:
        document["status"] = "source_only_frontend_probe_completed"
    with pytest.raises(ValueError):
        verifier.verify_negative_case("candidate_sensitive_read", document)


@pytest.mark.parametrize("mutation", ["drop_round", "fake_success", "root_cause", "status_drift", "typed_code_drift"])
def test_verifier_rejects_history_rewrite(mutation):
    verifier = _verifier()
    document = copy.deepcopy(_load("probe-history.json"))
    if mutation == "drop_round":
        document["attempts"] = document["attempts"][1:]
    elif mutation == "fake_success":
        document["attempts"][0]["full_build"] = True
    elif mutation == "status_drift":
        document["attempts"][0]["status"] = "blocked_backend_capability_unavailable"
    elif mutation == "typed_code_drift":
        document["attempts"][0]["typed_code"] = "process_failed"
    else:
        document["attempts"][0]["root_cause"] = ""
    with pytest.raises(ValueError):
        verifier.verify_history_document(document)


def test_history_packages_remain_closed_and_mt1b_blocked():
    selection = json.loads((PACKAGE.parent / "MT0/dsl-selection.json").read_text(encoding="utf-8"))
    assert selection["status"] == "blocked"
    assert selection["qualified_candidate_count"] == 0
    assert selection["next_stages"]["MT1b"] == "blocked"
    assert all(row["availability"] == "interface_only" for row in selection["candidates"])
    review = json.loads(
        (PACKAGE.parent / "MT0-isolation-profile/outer-probe-review.json").read_text(encoding="utf-8")
    )
    assert review["review_done"] is True
    assert review["current_status"]["mt0_status"] == "blocked"
    closed = _load("start-state.json")["other_closed_sha256"]
    for relative, expected in closed.items():
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == expected


def test_history_anchor_drift_is_recorded_without_editing_the_old_package():
    document = _load("history-anchor-drift.json")
    assert document["classification"] == "expected_living_document_anchor_drift_not_evidence_rewrite"
    for row in document["living_documents"]:
        current = hashlib.sha256((ROOT / row["relative"]).read_bytes()).hexdigest()
        assert row["current_sha256"] == current
        assert row["current_sha256"] != row["pinned_sha256"]
    unchanged = document["old_package_unchanged"]
    verifier_snapshot = unchanged["verification.json"]
    assert hashlib.sha256(
        (ROOT / "tests/support/golden/shared/多DSL-Track-阶段验收/MT0-isolation-profile/verification.json").read_bytes()
    ).hexdigest() == verifier_snapshot
    for key in ("verifier_modified", "raw_modified", "review_modified", "manifest_modified"):
        assert unchanged[key] is False
    joined = " ".join(document["not_done"])
    assert "editing the closed" in joined
    assert "claiming the old verifier still passes" in joined
    isolated = document["isolated_old_verifier_checks"]
    assert isolated["failed"] == ["verify_results"]
    assert "verify_manifest" in isolated["passed"]
    assert "scoped source digest mismatch" in isolated["failure_reason"]
    assert isolated["fixed_pytest_nodes_executed"] is False


def test_completion_predicate_excludes_mt0_metal_and_publication():
    completion = _load("probe-plan.json")["completion_predicate"]
    required = " ".join(completion["required"])
    assert "outer preflight" in required
    assert "independent reviewer" in required
    satisfied = " ".join(completion["satisfied_in_this_package"])
    for needle in ("outer preflight", "six-artifact set", "wheel sha256", "negative", "independent read-only review"):
        assert needle in satisfied
    outstanding = " ".join(completion["outstanding_in_this_package"])
    for needle in ("second non-MSL DSL", "MT1b", "Metal"):
        assert needle in outstanding
    excluded = " ".join(completion["explicitly_not_implied"])
    for needle in ("MT0 completion", "second non-MSL DSL", "MT1b/MT2", "Metal correctness or performance", "publication"):
        assert needle in excluded
    unauthorized = " ".join(_load("probe-plan.json")["not_authorized_here"])
    for needle in ("GPU or NPU", "paid search", "Halide", "MT1b"):
        assert needle in unauthorized
