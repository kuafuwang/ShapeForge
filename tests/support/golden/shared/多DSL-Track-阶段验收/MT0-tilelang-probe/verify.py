#!/usr/bin/env python3
"""Verify the MT0 TileLang-first install + source-only probe evidence package.

This verifier is a structure/claim regression tool.  It does not import
``tilelang``, does not launch ``sandbox-exec``, does not run the probe and does
not touch a GPU/NPU.  The recorded context-capability observation is this
epoch's fact; on the outer control plane the runner (not this verifier) performs
the real probe.

Run ``--check`` from the repository root to also execute the fixed pytest nodes
and recompute ``manifest.sha256``.
"""

from __future__ import annotations

import ast
import copy
import difflib
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
EXPECTED_NODES = 75
EPOCH = "mt0-tilelang-source-only-probe-1"

ARTIFACT_NAMES = (
    "frontend_identity.json",
    "kernel.metal",
    "device_ir.json",
    "kernel_params.json",
    "host_module.txt",
    "build_identity.json",
)
# build_identity.json carries the rows and the digest of the five files it describes.
BUILD_ARTIFACT_NAMES = ARTIFACT_NAMES[:-1]
EVIDENCE_DIR = "evidence/source-only-r5"
HISTORY_FILE = "probe-history.json"
PROBE_RESULT = "probe-result-r5.json"
EMPTY_SHA256 = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
FAILED_ROUNDS = ("r1", "r2", "r3", "r4")
EXPECTED_ATTEMPT_RESULTS = {
    "r1": ("probe-result.json", 73),
    "r2": ("probe-result-r2.json", 73),
    "r3": ("probe-result-r3.json", 73),
    "r4": ("probe-result-r4.json", 72),
    "r5": ("probe-result-r5.json", 0),
}
# The successful attempt's exact Profile write set: non-persistent devices plus this
# attempt's build subdirectories only. No other attempt and no mds/persistent root.
R5_ATTEMPT_ROOT = "/private/tmp/shapeforge-mt0-tilelang-attempt-r5"
R5_DEVICE_ROOTS = (
    "/dev/null", "/dev/zero", "/dev/random", "/dev/urandom",
    "/dev/tty", "/dev/ptmx", "/dev/fd", "/dev/dtracehelper",
)
R5_BUILD_ROOTS = tuple(
    f"{R5_ATTEMPT_ROOT}/build/{name}"
    for name in ("work", "home", "cache", "config", "data", "tmp", "pycache", "artifacts")
)
R5_WRITE_ROOTS = frozenset((*R5_DEVICE_ROOTS, *R5_BUILD_ROOTS))
NEGATIVE_FILES = {
    "candidate_sensitive_read": ("read_outside", "negative-read-outside.json"),
    "candidate_outside_write": ("write_outside", "negative-write-outside.json"),
    "candidate_network_connect": ("network_connect", "negative-network.json"),
    "undeclared_artifact": ("undeclared_artifact", "negative-artifact.json"),
}
NEGATIVE_IDS = frozenset({
    "frontend_version_drift",
    "frontend_source_drift",
    "candidate_sensitive_read",
    "candidate_outside_write",
    "candidate_network_connect",
    "undeclared_artifact",
    "codegen_timeout",
    "candidate_nonzero",
    "missing_toolchain",
})
REAL_NEGATIVE_IDS = frozenset(NEGATIVE_FILES)
STATIC_NEGATIVE_IDS = NEGATIVE_IDS - REAL_NEGATIVE_IDS
FIXTURE_FILES = {
    "read_outside": "read_outside.py",
    "write_outside": "write_outside.py",
    "network_connect": "network_connect.py",
    "undeclared_artifact": "undeclared_artifact.py",
}
EXPECTED_MANIFEST_MEMBERS = frozenset({
    "acceptance.md", "context-capability.json", "frontend-install.json", "history-anchor-drift.json",
    "nodes.json", "probe-history.json", "probe-plan.json", "probe-status.json", "review.json",
    "negative-artifact.json", "negative-network.json", "negative-read-outside.json",
    "negative-write-outside.json",
    "probe-result.json", "probe-result-r2.json", "probe-result-r3.json", "probe-result-r4.json",
    "probe-result-r5.json",
    "start-state.json", "start-worktree.sha256",
    "probe/candidate.py", "probe/frontend_entry.py", "probe/run_probe.py",
    "probe/fixtures/network_connect.py", "probe/fixtures/read_outside.py",
    "probe/fixtures/undeclared_artifact.py", "probe/fixtures/write_outside.py",
    "verify.py",
    *{f"{EVIDENCE_DIR}/{name}" for name in ARTIFACT_NAMES},
    *{f"evidence/failure-inputs/r{round}-{name}.py"
      for round in ("1", "2", "3", "4", "5") for name in ("candidate", "frontend_entry")},
    *{f"evidence/failure-logs/r{index}-stderr.txt" for index in (1, 2, 3, 4)},
})
HEX256 = re.compile(r"^[0-9a-f]{64}$")


def _load(name: str, here: Path = HERE) -> dict:
    return json.loads((here / name).read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _evidence_rows(here: Path = HERE) -> list[dict]:
    rows: list[dict] = []
    for name in BUILD_ARTIFACT_NAMES:
        path = here / EVIDENCE_DIR / name
        rows.append({"path": name, "bytes": path.stat().st_size, "sha256": _sha256(path)})
    return rows


def _set_digest(rows: list[dict]) -> str:
    return hashlib.sha256(
        json.dumps(rows, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _tree_manifest(root: Path) -> tuple[int, int, str]:
    """The pinned ``package_manifest`` rule: regular non-pyc files in sorted path order."""

    rows: list[tuple[str, int, str]] = []
    total = 0
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if "__pycache__" in relative.parts or path.suffix == ".pyc":
            continue
        if not path.is_file() or path.is_symlink():
            continue
        size = path.stat().st_size
        total += size
        rows.append((relative.as_posix(), size, _sha256(path)))
    manifest = "\n".join(f"{name}  {size}  {digest}" for name, size, digest in rows)
    return len(rows), total, hashlib.sha256(manifest.encode("utf-8")).hexdigest()


def _verify_record_files(site_packages: Path, venv_root: Path, expected_count: int) -> None:
    """Re-hash every sha256 entry in every installed RECORD (urlsafe-base64 hashes)."""

    import base64
    import csv as _csv

    venv_root = venv_root.resolve()
    checked = 0
    for record in sorted(site_packages.glob("*.dist-info/RECORD")):
        with record.open(newline="") as handle:
            for row in _csv.reader(handle):
                if len(row) < 3 or not row[1].startswith("sha256="):
                    continue
                target = (site_packages / row[0]).resolve()
                _require(target.is_file() and target.is_relative_to(venv_root), f"RECORD entry is missing or escapes the venv: {row[0]}")
                data = target.read_bytes()
                digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode("ascii")
                _require(digest == row[1].split("=", 1)[1] and str(len(data)) == row[2], f"RECORD entry does not match the file: {row[0]}")
                checked += 1
    _require(checked == expected_count, f"RECORD verification count drifted: {checked} != {expected_count}")


def _check_evidence_members(names) -> None:
    _require(set(names) == set(ARTIFACT_NAMES), f"evidence artifact set differs: {sorted(set(names) ^ set(ARTIFACT_NAMES))}")


def _nodes(here: Path = HERE) -> tuple[str, ...]:
    return tuple(_load("nodes.json", here)["node_ids"])


def verify_frontend_install(document: dict, here: Path = HERE) -> None:
    _require(document.get("schema_id") == "shapeforge.mt0-tilelang-probe.frontend-install.1", "wrong frontend-install schema")
    _require(document.get("status") == "staged_and_verified", "the frontend install is not recorded as staged and verified")
    pinned = document.get("pinned", {})
    _require(pinned.get("dsl_id") == "tilelang" and pinned.get("version") == "0.1.15", "the pinned DSL/version drifted")
    _require(pinned.get("upstream_commit") == "a35f8ddf45eba16c21211ec8822d56ce5363036f", "the pinned TileLang commit drifted")
    _require(pinned.get("tvm_commit") == "4211874e9e4b8770fd20ad0d98ab80afd1df5028", "the pinned TVM commit drifted")
    wheel = document.get("wheel", {})
    _require(wheel.get("filename") == "tilelang-0.1.15-cp39-abi3-macosx_11_0_arm64.whl", "the fixed macOS arm64 wheel is not the recorded build")
    _require(wheel.get("bytes") == 33667284, "the wheel byte size drifted")
    _require(wheel.get("sha256") == wheel.get("pypi_published_sha256") == "dcffc9aa8d4cd086dd93090f5afeeaa38aa621d2bb80998c4e5f5ee07be116aa", "the wheel sha256 is not the PyPI-published value")
    _require(wheel.get("abi") == "cp39-abi3" and wheel.get("platform_tag") == "macosx_11_0_arm64", "the wheel compatibility tags drifted")
    _require(wheel.get("requires_python") == ">=3.10", "the recorded requires_python drifted")
    _require(wheel.get("canonical_url", "").startswith("https://files.pythonhosted.org/") and wheel["canonical_url"].endswith(wheel["filename"]), "the canonical wheel URL is missing")
    _require(document.get("download", {}).get("mirror_url", "").startswith("https://pypi.tuna.tsinghua.edu.cn/") and document["download"]["mirror_url"].endswith(wheel["filename"]), "the mirror wheel URL is missing")
    license_row = document.get("license", {})
    _require(license_row.get("spdx") == "MIT" and license_row.get("bytes") == 1283, "the license identity drifted")
    _require(license_row.get("sha256") == "f70e5cdc24c3797af741930440086f9a0ad36930b05a3dfc25b420936dc2cfa1", "the wheel LICENSE bytes drifted")
    _require("Microsoft" in license_row.get("note", ""), "the historical collaboration note was dropped from the license record")
    download = document.get("download", {})
    _require(download.get("all_wheel_sha256_verified_against_pypi") is True, "wheel hashes are not all verified")
    _require(download.get("wheel_count") == 19 == len(document.get("dependencies", [])), "the dependency wheel count drifted")
    _require(download.get("total_wheel_bytes") == sum(row["bytes"] for row in document["dependencies"]), "the dependency byte total was not recomputed")
    _require("pythonhosted" in download.get("why_mirror", ""), "the mirror decision is not recorded honestly")
    projects = [row["project"] for row in document["dependencies"]]
    _require(projects.count("tilelang") == 1 and len(set(projects)) == 19, "dependency projects are duplicated or missing")
    for row in document["dependencies"]:
        _require(isinstance(row.get("version"), str) and row["version"], "dependency version is missing")
        _require(isinstance(row.get("filename"), str) and row["filename"].endswith(".whl"), "dependency wheel filename is missing")
        _require(isinstance(row.get("bytes"), int) and row["bytes"] > 0, "dependency byte size is invalid")
        _require(HEX256.fullmatch(row.get("sha256", "")) is not None, "dependency sha256 is not a lower-case hex digest")
        _require(row.get("sha256") == row.get("pypi_published_sha256"), "dependency wheel is not bound to its PyPI-published digest")
    python = document.get("python", {})
    _require(re.fullmatch(r"3\.13\.\d+", python.get("version", "")) is not None, "the recorded frontend Python is not 3.13.x")
    _require(python.get("short_version") == "3.13" and python.get("implementation") == "CPython", "the Python identity drifted")
    staging = document["installed"]["staging_root"]
    _require(python.get("venv_python") == staging + "/venv/bin/python", "the venv python is not rooted in the staging directory")
    provenance = document.get("source_provenance", {})
    _require(provenance.get("tilelang_python_identical") is True and provenance.get("tilelang_python_files_mismatched") == [], "TileLang Python source provenance is not identical")
    _require(
        provenance.get("tilelang_python_files_in_tag") == 390
        and provenance.get("tilelang_python_files_compared") == 390
        and provenance.get("tilelang_python_files_in_wheel") == 1137
        and provenance.get("tilelang_python_only_in_wheel") == 747,
        "TileLang Python source counts drifted",
    )
    _require(provenance.get("tvm_python_identical") is True and not provenance.get("tvm_python_files_mismatched") and not provenance.get("tvm_python_only_in_commit") and not provenance.get("tvm_python_only_in_wheel"), "TVM Python source provenance is not identical")
    _require(provenance.get("tvm_python_files_in_commit") == 745 and provenance.get("tvm_python_files_compared") == 745, "TVM Python source count drifted")
    _require(provenance.get("native_libraries_rebuilt_from_source") is False, "native libraries are being claimed as rebuilt")
    _require("not rebuilt" in provenance.get("boundary", ""), "the native-library provenance boundary was hidden")
    _require(HEX256.fullmatch(provenance.get("tilelang_tag_tarball", {}).get("sha256", "")) is not None, "the tag tarball hash is missing")
    _require(HEX256.fullmatch(provenance.get("tvm_commit_tarball", {}).get("sha256", "")) is not None, "the TVM tarball hash is missing")
    installed = document.get("installed", {})
    _require(staging.startswith("/tmp/shapeforge-frontends/tilelang-0.1.15"), "the staging root is not the controlled ephemeral path")
    _require("ephemeral" in installed.get("scope", ""), "the ephemeral staging scope is not recorded")
    _require(installed.get("staging_root_mode") == "0700", "the staging root is not recorded as owner-only")
    _require(document.get("isolation_of_install", {}).get("staging_root_not_group_or_world_accessible") is True, "the staging access mode claim is missing")
    _require(installed.get("record_distributions") == 20, "the RECORD distribution count drifted")
    _require(installed.get("record_files_checked") == installed.get("record_files_verified") == 21199, "RECORD verification is incomplete")
    _require(installed.get("record_mismatches") == {} and installed.get("record_missing") == {}, "a RECORD mismatch or missing file was recorded")
    _require(installed.get("installed_tree_files") == 21186 and installed.get("installed_tree_bytes") == 853505126, "the installed tree summary drifted")
    _require(installed.get("installed_tree_manifest_sha256") == "ed87747484c00473356eae644062bf37e137a55ef208dacc940cfb075ed55dad", "the installed tree digest drifted")
    _require(installed.get("tilelang_package_files") == 3651 and installed.get("tilelang_package_bytes") == 134219728, "the tilelang package summary drifted")
    _require(installed.get("tilelang_package_manifest_sha256") == "06f3b0adee7abce77af1111cb36e704b4b23c3cbe38d095dd62cce86509c11cc", "the tilelang package manifest digest drifted")
    expected_dist_info = {
        "dist_info_metadata_sha256": "380d3486286aeb708dc4846cfdc4b438985c84199a160b21238cadf3e4e3fa19",
        "dist_info_record_sha256": "6ba83c2ed267154eedc9dec6d916e315c553a8e515744f0feb7ca05dfa388c34",
    }
    for key, expected in expected_dist_info.items():
        _require(HEX256.fullmatch(installed.get(key, "")) is not None and installed.get(key) == expected, f"{key} drifted")
    isolation = document.get("isolation_of_install", {})
    for key in ("project_venv_untouched", "hmz_uv_tool_untouched", "system_python_untouched", "global_config_untouched", "home_used_only_inside_staging", "repo_contains_wheel_bytes"):
        _require(isolation.get(key) is (key != "repo_contains_wheel_bytes"), f"install isolation claim drifted: {key}")
    boundary = document.get("execution_boundary", {})
    for key in ("tilelang_imported", "codegen_executed", "kernel_generated", "gpu_npu_executed"):
        _require(boundary.get(key) is False, f"staging inventing execution: {key}")
    _require(boundary.get("network_used_only_for_verified_download") is True, "the staging network boundary was widened")
    if Path(staging).is_dir():
        import stat as _stat
        staging_root = Path(staging)
        _require(_stat.S_IMODE(staging_root.stat().st_mode) == 0o700, "the staging root is group/world accessible")
        _require((staging_root / "venv/bin/python").is_file(), "recorded staging root lost its venv python")
        _require(_sha256(Path(wheel["local_path"])) == wheel["sha256"], "staged wheel bytes no longer match the pinned sha256")
        site_packages = staging_root / "venv" / "lib" / f"python{python['short_version']}" / "site-packages"
        package = site_packages / "tilelang"
        _require(package.is_dir(), "the staged tilelang package is missing")
        files, total, digest = _tree_manifest(package)
        _require((files, total, digest) == (
            installed["tilelang_package_files"],
            installed["tilelang_package_bytes"],
            installed["tilelang_package_manifest_sha256"],
        ), "the staged tilelang package manifest was not recomputed consistently")
        dist_info = site_packages / f"tilelang-{pinned['version']}.dist-info"
        _require(_sha256(dist_info / "METADATA") == installed["dist_info_metadata_sha256"], "staged dist-info METADATA drift")
        _require(_sha256(dist_info / "RECORD") == installed["dist_info_record_sha256"], "staged dist-info RECORD drift")
        wheels = staging_root / "wheels"
        for row in document["dependencies"]:
            path = wheels / row["filename"]
            _require(path.is_file(), f"a verified dependency wheel is missing from staging: {row['filename']}")
            _require(path.stat().st_size == row["bytes"], f"dependency wheel byte size drifted: {row['filename']}")
            _require(_sha256(path) == row["sha256"] == row["pypi_published_sha256"], f"dependency wheel digest drifted: {row['filename']}")
        _verify_record_files(site_packages, staging_root / "venv", installed["record_files_verified"])


def verify_context_capability(document: dict) -> None:
    _require(document.get("schema_id") == "shapeforge.mt0-tilelang-probe.context-capability.1", "wrong context-capability schema")
    _require(document.get("epoch") == "mt0-tilelang-source-only-probe-context-1", "context epoch drifted")
    _require(document.get("context", {}).get("pi_coding_agent") == "true", "the nested agent context is not recorded")
    observations = document.get("observations", {})
    minimal = observations.get("minimal_profile", {})
    denied = observations.get("deny_network_profile", {})
    expected_stderr = "sandbox-exec: sandbox_apply: Operation not permitted\n"
    for row, label in ((minimal, "minimal"), (denied, "deny-network")):
        _require(row.get("returncode") == 71, f"the {label} profile attempt is not recorded as denied")
        _require(row.get("stderr") == expected_stderr and row.get("stdout") == "", f"the {label} profile stderr drifted")
        _require(row.get("argv", [])[:3] == ["/usr/bin/sandbox-exec", "-p", "(version 1)(allow default)"] or label == "deny-network", "the minimal profile command drifted")
    _require("Operation not permitted" in observations.get("ps_denied", {}).get("error", ""), "the enclosing-sandbox ps denial is missing")
    writability = observations.get("writability", {})
    _require(writability.get("home", {}).get("writable") is False, "HOME writability is not the observed denial")
    _require(writability.get("library_app_support", {}).get("writable") is False, "Library/Application Support writability is not denied")
    _require(writability.get("tmp", {}).get("writable") is True and writability.get("repo", {}).get("writable") is True, "the writable roots were misrecorded")
    _require("Apple clang version" in observations.get("apple_clang", ""), "the Apple Clang identity is missing")
    _require("Metal.xctoolchain" in observations.get("metal_toolchain_path", ""), "the Metal toolchain observation is missing")
    interpretation = document.get("interpretation", {})
    _require(interpretation.get("typed_code") == "backend_capability_unavailable", "the typed capability code drifted")
    _require(interpretation.get("minimal_and_network_profiles_both_denied") is True, "the two profile denials are not both recorded")
    _require(interpretation.get("enclosing_sandbox_observed") is True, "the enclosing sandbox observation is missing")
    _require(interpretation.get("home_writable") is False and interpretation.get("tmp_writable") is True and interpretation.get("repository_writable") is True, "the writability interpretation drifted")
    _require(interpretation.get("unfenced_fallback_allowed") is False, "an unfenced fallback is being admitted")
    _require(interpretation.get("capability_probe_only") is True, "the context observation is not recorded as a capability probe only")
    _require(interpretation.get("attack_or_candidate_fixture_executed") is False, "an attack/candidate fixture was executed")
    _require(interpretation.get("probe_shell_or_true_executed") is False, "the denied shell/true probe is claimed as executed")
    _require("outer control plane" in interpretation.get("why_deferred", ""), "the probe deferral does not point at the outer control plane")
    for key in ("real_probe_executed", "frontend_imported", "candidate_executed", "full_build", "gpu_npu_executed"):
        _require(interpretation.get(key) is False, f"this context claims work it did not do: {key}")
    hmz = document.get("hmz_installation_read_only_inspection", {})
    _require(hmz.get("performed") is True and hmz.get("launched") is False, "hmz inspection must be read-only and unlaunched")
    _require(hmz.get("protocol_id") == "hmz-lower-level-seatbelt-profile-proxy-1", "hmz protocol identity drifted")
    _require(hmz.get("protocol_digest") == "c6a6a61052d8e7a19b30a9674b685b75378d8999791eff0ef1c6c570cb762c83", "hmz protocol digest drifted")
    _require(hmz.get("version") == "0.2.2" and hmz.get("python_version") == "3.14.4", "hmz/interpreter version drifted")
    _require(len(hmz.get("source_sha256", {})) == 8 and any(path.endswith("fence/wrap.py") for path in hmz["source_sha256"]), "the eight-source hmz audit anchor is incomplete")
    _require(hmz.get("implicit_mds_write_roots") == [] and hmz.get("internal_fence_or_wrap_used") is False, "an mds/internal-wrap exception was added")
    _require(hmz.get("sandbox_exec_present_executable") is True, "the Seatbelt executable observation is missing")
    _require(hmz.get("runtime_attack_evidence") == "not established by installation inspection", "read-only inspection is being relabelled as attack evidence")
    _require("run_probe.py" in document.get("outer_command", ""), "the outer probe command is missing")
    not_evidence = document.get("not_evidence_for", [])
    _require(any("outer Seatbelt" in row for row in not_evidence), "the outer-isolation boundary was dropped")
    _require(any("MT0 completion" in row for row in not_evidence), "the MT0 boundary was dropped")


def verify_plan(document: dict, here: Path = HERE) -> None:
    _require(document.get("schema_id") == "shapeforge.mt0-tilelang-probe.plan.1", "wrong probe-plan schema")
    _require(document.get("epoch") == EPOCH and document.get("scope") == "tilelang_source_only_frontend_probe", "probe-plan epoch/scope drifted")
    install = _load("frontend-install.json", here)
    candidate = document.get("candidate", {})
    _require(candidate.get("sha256") == _sha256(here / "probe/candidate.py"), "candidate source hash drifted")
    _require(candidate.get("execution_model") == "embedded_host_code", "the candidate execution model was not classified as embedded_host_code")
    _require("N=1024" in candidate.get("fixed_program", "") and "4 x 256" in candidate.get("fixed_program", ""), "the fixed candidate shape is not recorded")
    frontend = document.get("frontend", {})
    _require(frontend.get("entry_sha256") == _sha256(here / "probe/frontend_entry.py"), "frontend entry hash drifted")
    _require(frontend.get("version") == install["pinned"]["version"] and frontend.get("wheel_sha256") == install["wheel"]["sha256"], "the plan and install identities disagree")
    _require(frontend.get("upstream_commit") == install["pinned"]["upstream_commit"] and frontend.get("tvm_commit") == install["pinned"]["tvm_commit"], "the plan does not bind the pinned commits")
    _require(frontend.get("api") == "tilelang.engine.lower", "the source-only frontend entry point drifted")
    _require(frontend.get("target") == {"kind": "metal", "keys": ["metal", "gpu"], "target_host": "c"}, "the target must be explicit and must not use the host-dependent metal4 detector")
    _require(frontend.get("options") == {"enable_host_codegen": True, "enable_device_compile": False}, "the build options are not source-only")
    import_note = frontend.get("import_time_mps_query", "")
    _require("device presence query" in import_note and "never initiates" in import_note, "the import-time MPS query boundary is not recorded")
    _require("zero dispatches" in import_note and "no MTLLibrary" in import_note, "the import-time device-object boundary was widened")
    forbidden = set(frontend.get("forbidden_calls", []))
    for needle in ("tilelang.compile", "tilelang.jit", "JITKernel", "MTLDevice/MTLLibrary/MTLComputePipelineState", "command encoder/commit/dispatch"):
        _require(needle in forbidden, f"a forbidden call category is missing: {needle}")
    runner = document.get("runner", {})
    _require(runner.get("sha256") == _sha256(here / "probe/run_probe.py"), "runner hash drifted")
    _require(runner.get("imports_tilelang") is False, "the trusted runner must not import tilelang")
    _require("backend_capability_unavailable" in runner.get("preflight", "") and "sandbox-exec" in runner["preflight"], "the runner preflight is not typed or does not probe Seatbelt")
    isolation = document.get("isolation", {})
    _require(isolation.get("online") is False and isolation.get("hosts") == [] and isolation.get("proxy_host_allowlist") == [], "the probe is not offline with an empty host allowlist")
    _require(isolation.get("implicit_mds_write_roots") == [] and isolation.get("internal_fence_or_wrap_used") is False, "an mds/internal-wrap exception was added")
    _require(isolation.get("frontend_root_mounted_read_only") is True and isolation.get("candidate_has_network") is False, "the frontend read-only/network boundary drifted")
    limits = isolation.get("limits", {})
    _require(limits == {
        "wall_seconds": 300, "cpu_seconds": 330, "file_bytes": 67108864, "open_files": 256,
        "address_space_bytes": None, "processes": None, "artifact_total_bytes": 33554432,
    }, "resource limits drifted")
    contract = document.get("artifact_contract", {})
    _require(tuple(contract.get("exact_set", ())) == ARTIFACT_NAMES, "the declared artifact set drifted")
    _require(contract.get("per_file_bytes") == 16777216 and contract.get("total_bytes") == 33554432, "artifact caps drifted")
    rules = " ".join(contract.get("rules", []))
    for needle in ("exact declared set", "symlink", "hardlink", "regular files"):
        _require(needle in rules, f"artifact rule missing: {needle}")
    required = " ".join(contract.get("full_build_requires", []))
    for needle in ("Metal source", "device IR", "kernel params", "host artifact", "frontend identity", "build identity"):
        _require(needle in required, f"full-build requirement missing: {needle}")
    steps = document.get("build_steps", [])
    _require(len(steps) >= 6 and any("enable_device_compile=False" in step for step in steps), "the source-only build steps are incomplete")
    matrix = document.get("negative_matrix", [])
    ids = {row.get("id") for row in matrix}
    _require(ids == NEGATIVE_IDS, f"negative case matrix drifted: {sorted(ids ^ NEGATIVE_IDS)}")
    for row in matrix:
        _require(isinstance(row.get("expected"), str) and row["expected"], f"negative case has no expected outcome: {row.get('id')}")
        _require("full build" in row["expected"] or "no full build" in row["expected"], f"negative case does not require no-full-build: {row.get('id')}")
    identity_evidence = {row["id"]: row.get("evidence", "") for row in matrix if row.get("kind") == "identity"}
    _require("frontend_entry.main()" in identity_evidence.get("frontend_version_drift", "") and "[pinned_version]" in identity_evidence["frontend_version_drift"], "the version-drift row is only a coverage-list claim")
    source_evidence = identity_evidence.get("frontend_source_drift", "")
    _require("frontend_entry.main()" in source_evidence and all(f"[{name}]" in source_evidence for name in ("package_manifest", "metadata", "record")), "the source-drift row does not name all real entry mutations")
    fixtures = document.get("fixtures", {})
    _require(set(fixtures) == set(FIXTURE_FILES), "the fixture set drifted")
    for name, filename in FIXTURE_FILES.items():
        _require(fixtures[name] == _sha256(here / "probe/fixtures" / filename), f"fixture hash drifted: {name}")
    completion = document.get("completion_predicate", {})
    required_steps = " ".join(completion.get("required", []))
    for needle in ("outer preflight", "exact artifact set", "wheel sha256", "negative cases", "independent reviewer"):
        _require(needle in required_steps, f"completion predicate missing: {needle}")
    excluded = " ".join(completion.get("explicitly_not_implied", []))
    for needle in ("MT0 completion", "second non-MSL DSL", "Metal correctness or performance", "publication"):
        _require(needle in excluded, f"completion predicate overclaims: {needle}")
    unauthorized = " ".join(document.get("not_authorized_here", []))
    for needle in ("GPU or NPU", "paid search", "Halide", "MT1b"):
        _require(needle in unauthorized, f"authorization boundary missing: {needle}")
    evidence = document.get("evidence", {})
    _require(evidence.get("real_probe_result") == PROBE_RESULT, "the plan does not name the real probe result")
    _require(evidence.get("history") == HISTORY_FILE, "the plan does not name the probe history document")
    _require(evidence.get("artifact_dir") == EVIDENCE_DIR, "the plan does not name the evidence directory")
    _require(evidence.get("artifact_set_sha256") == _set_digest(_evidence_rows(here)), "the plan artifact-set digest is not the archived bytes")
    _require(evidence.get("attempts") == 5 and evidence.get("failed_rounds") == list(FAILED_ROUNDS), "the plan hides the failed probe rounds")
    negatives = evidence.get("negative_results", {})
    _require(set(negatives.get("real_executed", [])) == set(REAL_NEGATIVE_IDS), "the real negative cases drifted")
    _require(set(negatives.get("static_or_adapter_covered", [])) == set(STATIC_NEGATIVE_IDS), "the static/adapter negative coverage drifted")
    fix_scope = evidence.get("fix_scope", "")
    for needle in ("decorator", "with target:", "unchanged"):
        _require(needle in fix_scope, f"the fix-scope statement is missing: {needle}")
    satisfied = " ".join(document["completion_predicate"].get("satisfied_in_this_package", []))
    for needle in ("outer preflight", "six-artifact set", "wheel sha256", "negative", "independent read-only review"):
        _require(needle in satisfied, f"satisfied requirement missing: {needle}")
    outstanding = " ".join(document["completion_predicate"].get("outstanding_in_this_package", []))
    for needle in ("second non-MSL DSL", "MT1b", "Metal"):
        _require(needle in outstanding, f"outstanding requirement missing: {needle}")
    round_status = document.get("round_status", {})
    _require(round_status.get("source_only_probe") == "completed_and_independently_reviewed", "the source-only probe is not recorded as completed")
    _require(round_status.get("mt0") == "blocked", "MT0 was completed in the plan")
    _require(round_status.get("mt1b") == "not_started_blocked", "MT1b was unblocked in the plan")
    _require(round_status.get("metal_or_performance") == "not_started" and round_status.get("production_dsl") is False, "Metal/production is claimed in the plan")
    blockers = round_status.get("mt0_blockers", [])
    _require(len(blockers) == 1 and "second non-MSL DSL" in blockers[0], "the plan does not name the deferred second DSL as the only MT0 blocker")
    _require("MT0" in " ".join(round_status.get("mt1b_blockers", [])) and "selection" in " ".join(round_status.get("mt1b_blockers", [])), "the plan hides the MT1b dependency/selection blockers")


def verify_status(document: dict) -> None:
    _require(document.get("schema_id") == "shapeforge.mt0-tilelang-probe.status.1", "wrong probe-status schema")
    _require(document.get("epoch") == EPOCH, "probe-status epoch drifted")
    _require(document.get("status") == "source_only_probe_completed_mt0_blocked", "the status does not record the completed probe with MT0 still blocked")
    _require(document.get("typed_code") == "source_only_probe_completed", "the current typed code drifted")
    prior = document.get("prior_typed_code", {})
    _require(prior.get("code") == "backend_capability_unavailable" and prior.get("evidence") == "context-capability.json", "the prior nested-context typed code was dropped")
    decision = document.get("round1_dsl_decision", {})
    _require(decision.get("selected_first_round_dsl") == "tilelang" and decision.get("pinned_version") == "0.1.15", "the first-round DSL decision drifted")
    _require(decision.get("second_non_msl_dsl") == "deferred_to_later_round", "the second DSL must stay deferred")
    _require(decision.get("toy_or_substitute_routes_used") is False, "a toy/substitute route is being used")
    halide = decision.get("halide", {})
    for key in ("installed", "built", "probed", "code_or_dependency_added"):
        _require(halide.get(key) is False, f"Halide is being claimed as present in a TileLang-only round: {key}")
    installation = document.get("installation", {})
    _require(installation.get("status") == "staged_and_verified", "the installation status drifted")
    _require(installation.get("wheel_sha256_verified_against_pypi") is True, "the wheel verification claim was weakened")
    _require(installation.get("record_files_verified") == 21199 and installation.get("record_file_mismatches") == 0, "the RECORD summary drifted")
    _require("not rebuilt" in installation.get("source_provenance_boundary", ""), "the native-library boundary was hidden")
    _require(installation.get("probe_used_this_install") is True, "the probe is not bound to this install")
    probe = document.get("probe", {})
    _require(probe.get("executed") is True and probe.get("full_build") is True, "the completed probe is not recorded")
    _require(probe.get("result") == PROBE_RESULT and probe.get("history") == HISTORY_FILE, "the probe result/history pointers drifted")
    _require(probe.get("evidence_dir") == EVIDENCE_DIR, "the probe evidence directory drifted")
    _require(probe.get("attempts") == 5 and probe.get("failed_attempts") == list(FAILED_ROUNDS), "the failed attempts were hidden")
    observed = probe.get("artifacts_observed", [])
    _require([row.get("path") for row in observed] == list(ARTIFACT_NAMES), "the observed artifact set drifted")
    for row in observed:
        path = HERE / EVIDENCE_DIR / row["path"]
        _require(row.get("bytes") == path.stat().st_size and row.get("sha256") == _sha256(path), f"observed artifact does not match the archived bytes: {row['path']}")
    _require(probe.get("artifact_set_sha256") == _set_digest(_evidence_rows(HERE)), "the status artifact-set digest is not the archived bytes")
    _require(probe.get("metal_source_generated") is True and probe.get("kernel_metal_bytes") == 732, "the Metal source fact was dropped")
    _require(probe.get("device_dispatch_executed") is False and probe.get("device_dispatch_calls") == 0, "a device dispatch is being claimed")
    for key in ("device_library_created", "pipeline_state_created", "buffer_allocated", "gpu_npu_executed", "metal_correctness_or_performance_claim", "cpu_or_npu_simulation_of_metal"):
        _require(probe.get(key) is False, f"the probe claims work it did not do: {key}")
    _require(probe.get("enable_device_compile") is False and probe.get("enable_host_codegen") is True, "the source-only build options drifted")
    _require(probe.get("online") is False and probe.get("hosts") == [] and probe.get("mds_write_roots") == [], "the probe is no longer offline/mds-free")
    _require("run_probe.py" in probe.get("outer_command", ""), "the outer probe command is missing")
    negatives = document.get("negative_cases", {})
    real = negatives.get("real_executed", [])
    _require({row.get("id") for row in real} == set(REAL_NEGATIVE_IDS), "the real negative cases drifted")
    for row in real:
        _require(row.get("matched") is True and row.get("no_full_build") is True, f"a real negative case is claimed without its typed outcome: {row.get('id')}")
        _require(row.get("result") == NEGATIVE_FILES[row["id"]][1], f"the negative result pointer drifted: {row.get('id')}")
    static = negatives.get("static_or_adapter_covered", [])
    _require({row.get("id") for row in static} == set(STATIC_NEGATIVE_IDS), "the static/adapter negative coverage drifted")
    static_by_id = {row["id"]: row.get("covered_by", "") for row in static}
    _require("frontend_entry.main()" in static_by_id.get("frontend_version_drift", "") and "[pinned_version]" in static_by_id["frontend_version_drift"], "status reduced the version-drift negative to a list entry")
    _require("frontend_entry.main()" in static_by_id.get("frontend_source_drift", "") and all(f"[{name}]" in static_by_id["frontend_source_drift"] for name in ("package_manifest", "metadata", "record")), "status lost a real source-drift mutation")
    _require(negatives.get("total") == 9 and negatives.get("all_require_no_full_build") is True, "the negative matrix lost a case")
    _require(negatives.get("authority") == "fixture_observation", "the negative authority is not fixture_observation")
    _require("fixture_observation" in negatives.get("top_level_status_note", "") and "blocked_backend_capability_unavailable" in negatives["top_level_status_note"], "the generic fixture top-level status is not explained")
    current = document.get("current_status", {})
    _require(current.get("mt0_status") == "blocked", "MT0 was completed")
    _require(current.get("mt1b_status") == "not_started_blocked", "MT1b was started")
    blockers = current.get("mt0_blockers", [])
    _require(len(blockers) == 1 and "second non-MSL DSL" in blockers[0], "the only remaining MT0 blocker is not the deferred second DSL")
    _require(not any("probe has not" in blocker or "has not run" in blocker for blocker in blockers), "the completed probe is still listed as an MT0 blocker")
    mt1b_blockers = " ".join(current.get("mt1b_blockers", []))
    _require("MT0" in mt1b_blockers and "selection" in mt1b_blockers, "the MT1b dependency/selection blockers are missing")
    _require(current.get("production_ready") is False and current.get("publication_allowed") is False, "production/publication was claimed")
    _require(current.get("metal_dsl_execution_or_performance_acceptance") is False and current.get("metal_dispatch_or_library_executed") is False, "Metal acceptance was claimed")
    _require(current.get("real_model_or_paid_search") is False and current.get("gpu_or_npu_executed") is False, "unauthorized work is claimed")
    _require(current.get("dsl_selection_status") == "blocked_one_active_candidate_interface_only", "the DSL selection status was widened")
    _require("does not mean MT0" in document.get("done_semantics", ""), "the done semantics hide the MT0 boundary")
    _require(len(document.get("gaps", [])) >= 5, "the remaining gaps were dropped")
    gaps = " ".join(document.get("gaps", []))
    for needle in ("second non-MSL DSL", "MT1b", "source-only", "native libraries were not rebuilt"):
        _require(needle in gaps, f"the gaps hide a remaining limitation: {needle}")
    not_claimed = " ".join(document.get("not_claimed", []))
    for needle in ("MT0 completion", "second non-MSL DSL", "MT1b", "real Metal correctness", "outer Seatbelt attack containment"):
        _require(needle in not_claimed, f"not_claimed is missing: {needle}")


def verify_probe_result(document: dict) -> None:
    """The raw r5 result: a source-only full build with zero device work."""

    _require(document.get("schema_id") == "shapeforge.mt0-tilelang-probe.probe-result.1", "wrong probe-result schema")
    _require(document.get("status") == "source_only_frontend_probe_completed", "the r5 result is not the completed source-only probe")
    _require(document.get("probe_executed") is True and document.get("full_build") is True, "the r5 result does not record a full build")
    _require(document.get("gpu_npu_executed") is False and document.get("device_dispatch_calls") == 0, "the r5 result claims device work")
    _require(document.get("publication_allowed") is False and document.get("production_ready") is False, "the r5 result claims production/publication")
    _require(document.get("metal_correctness_or_performance_claim") is False, "the r5 result claims Metal correctness/performance")
    _require(document.get("mt0_status") == "blocked" and document.get("mt1b_status") == "not_started_blocked", "the r5 result completes MT0 or MT1b")
    _require(document.get("unfenced_fallback_used", False) is False, "the r5 result admits an unfenced fallback")
    build = document.get("build", {})
    _require(build.get("full_build") is True, "the r5 build block does not claim a full build")
    _require(build.get("build_options") == {
        "enable_device_compile": False,
        "enable_host_codegen": True,
        "frontend_entry": "tilelang.engine.lower",
        "host_artifact_format": "c-source",
        "target": {"keys": ["metal", "gpu"], "kind": "metal", "target_host": "c"},
    }, "the r5 build options drifted")
    _require(build.get("device_execution") == {
        "buffer_allocated": False,
        "device_library_created": False,
        "dispatch_calls": 0,
        "pipeline_state_created": False,
    }, "the r5 device-execution record drifted")
    rows = build.get("artifact_rows", [])
    _require([row.get("path") for row in rows] == list(BUILD_ARTIFACT_NAMES), "the r5 artifact rows drifted")
    _require(_set_digest(rows) == build.get("artifact_set_sha256"), "the r5 artifact-set digest does not match its own rows")
    kernel_row = next(row for row in rows if row["path"] == "kernel.metal")
    _require(build.get("kernel_metal_sha256") == kernel_row["sha256"] and build.get("kernel_metal_bytes") == kernel_row["bytes"], "the r5 kernel row is inconsistent")
    _require(kernel_row["bytes"] == 732, "the r5 Metal source size drifted")
    frontend = build.get("frontend_identity", {})
    install = _load("frontend-install.json")
    _require(frontend.get("package_manifest_sha256") == install["installed"]["tilelang_package_manifest_sha256"], "r5 does not bind the pinned package manifest")
    _require(frontend.get("package_files") == install["installed"]["tilelang_package_files"], "r5 package file count drifted")
    _require(frontend.get("package_bytes") == install["installed"]["tilelang_package_bytes"], "r5 package byte count drifted")
    for key in ("dist_info_metadata_sha256", "dist_info_record_sha256"):
        _require(frontend.get(key) == install["installed"][key], f"r5 {key} drifted")
    _require(frontend.get("tilelang_version") == install["pinned"]["version"], "r5 frontend version drifted")
    _require(document.get("frontend", {}).get("wheel_sha256") == install["wheel"]["sha256"], "r5 does not bind the pinned wheel")
    _require(document.get("staged_frontend", {}).get("package_sha256") == install["installed"]["tilelang_package_manifest_sha256"], "r5 staged frontend identity drifted")
    isolation = document.get("isolation", {})
    _require(isolation.get("status") == "completed" and isolation.get("returncode") == 0, "the r5 isolation run did not complete")
    _require(isolation.get("policy_online") is False and isolation.get("policy_hosts") == [], "the r5 isolation is not offline")
    _require(isolation.get("implicit_mds_write_roots") == [] and isolation.get("internal_fence_or_wrap_used") is False, "an mds/internal-wrap exception appeared in r5")
    driver = isolation.get("driver_report", {})
    _require(driver.get("schema_id") == "tcenv.worker-isolation.hmz-profile.driver-report.1", "the r5 driver report schema drifted")
    _require(driver.get("fenced_returncode") == 0 and driver.get("profile_command_spawned") is True, "the r5 profile was not applied")
    _require(driver.get("proxy_stopped") is True and driver.get("online") is False and driver.get("hosts") == [], "the r5 proxy boundary drifted")
    _require(driver.get("mds_write_root_added") is False and driver.get("internal_fence_or_wrap_used") is False, "the r5 driver added an mds/internal-wrap exception")
    _require(driver.get("attack_containment_proven") is False, "r5 is claiming outer attack containment")
    _require(driver.get("protocol_id") == "hmz-lower-level-seatbelt-profile-proxy-1", "the r5 protocol id drifted")
    _require(driver.get("protocol_digest") == "c6a6a61052d8e7a19b30a9674b685b75378d8999791eff0ef1c6c570cb762c83", "the r5 protocol digest drifted")
    write_roots = driver.get("profile", {}).get("write", [])
    _require(isinstance(write_roots, list) and len(write_roots) == len(R5_WRITE_ROOTS), f"r5 write-root count drifted: {write_roots}")
    _require(set(write_roots) == R5_WRITE_ROOTS, f"r5 write roots are not exactly /dev plus this attempt/build: {sorted(set(write_roots) ^ R5_WRITE_ROOTS)}")
    _require(not any("Library" in root or "mds" in root.lower() for root in write_roots), "r5 wrote into an mds/persistent location")
    _require(build.get("torch_mps_availability_queries") == {"torch.backends.mps.is_available": 2, "torch.mps.is_available": 0}, "the r5 import-time MPS query counters drifted")


def verify_build_identity_document(document: dict, rows: list[dict], install: dict, candidate_sha256: str) -> None:
    """Pure check of build_identity.json against the recomputed artifact rows."""

    _require(document.get("schema_id") == "shapeforge.mt0-tilelang-probe.build-identity.1", "wrong build-identity schema")
    _require(document.get("full_build") is True, "build identity does not claim a full build")
    _require(document.get("pinned_wheel_sha256") == install["wheel"]["sha256"], "build identity does not bind the pinned wheel")
    _require(document.get("tilelang_package_manifest_sha256") == install["installed"]["tilelang_package_manifest_sha256"], "build identity does not bind the package manifest")
    _require(document.get("candidate_sha256") == candidate_sha256, "build identity does not bind the current candidate")
    _require(document.get("frontend_entry") == "tilelang.engine.lower", "build identity frontend entry drifted")
    _require(document.get("build_options") == {
        "enable_device_compile": False,
        "enable_host_codegen": True,
        "frontend_entry": "tilelang.engine.lower",
        "host_artifact_format": "c-source",
        "target": {"keys": ["metal", "gpu"], "kind": "metal", "target_host": "c"},
    }, "build identity options are not source-only")
    _require(document.get("device_execution") == {
        "buffer_allocated": False,
        "device_library_created": False,
        "dispatch_calls": 0,
        "pipeline_state_created": False,
    }, "build identity records device objects or a dispatch")
    _require(document.get("artifacts") == rows, "build identity artifact rows were not recomputed consistently")
    _require(document.get("artifact_set_sha256") == _set_digest(rows), "build identity artifact-set digest drift")


def _check_kernel_source(text: str) -> None:
    """The MSL must be the frozen 4x256 vector-scale kernel and must be source only."""

    _require(text.startswith("// Function: scale_kernel_kernel"), "kernel.metal does not start with the kernel comment")
    for needle in (
        "#include <metal_stdlib>",
        "[[kernel, max_total_threads_per_threadgroup(256)]] void scale_kernel_kernel(",
        "const device float* __restrict a [[ buffer(0) ]]",
        "device float* __restrict b [[ buffer(1) ]]",
        "  b[((((int)blockIdx.x) * 256) + ((int)threadIdx.x))] = (a[((((int)blockIdx.x) * 256) + ((int)threadIdx.x))] * 2.000000e+00f);",
    ):
        _require(needle in text, f"kernel.metal lost the frozen MSL structure: {needle}")
    for forbidden in ("MTLCommandBuffer", "dispatchThreadgroups", "newLibraryWithSource", "makeComputePipelineState", "MTLDevice"):
        _require(forbidden not in text, f"kernel.metal contains an execution-side construct: {forbidden}")


def _check_device_ir(document: dict) -> None:
    _require(document.get("metadata", {}).get("tvm_version") == "0.25.dev0", "device IR tvm_version drifted")
    nodes = document.get("nodes")
    _require(isinstance(nodes, list) and nodes, "device IR has no node table")
    strings = [node.get("data") for node in nodes if node.get("type") == "ffi.String"]
    for needle in ("global_symbol", "thread_extent", "scale_kernel_kernel", "metal", "max_threads_per_block"):
        _require(needle in strings, f"device IR lost {needle}")
    index = next(i for i, node in enumerate(nodes) if node.get("type") == "ffi.String" and node.get("data") == "thread_extent")
    extents = [node.get("data", {}).get("value") for node in nodes[index + 1:index + 5]]
    _require(extents == [4, 256, 1, 1], f"device IR thread extent drifted: {extents}")
    _require(any(node.get("type") == "ir.PrimType" and node.get("data", {}).get("dtype") == "float32" for node in nodes), "device IR lost float32")
    _require(any(node.get("type") == "int" and node.get("data") == 256 for node in nodes), "device IR lost the 256 thread constant")


def _check_kernel_params(document: dict) -> None:
    nodes = document.get("nodes") or []
    strings = [node.get("data") for node in nodes if node.get("type") == "ffi.String"]
    integers = [node.get("data") for node in nodes if node.get("type") == "int"]
    _require("dtype" in strings and "shape" in strings, "kernel params lost dtype/shape")
    _require("float32" in strings, "kernel params lost float32")
    _require(1024 in integers, "kernel params lost the static 1024 shape")
    _require(document.get("metadata", {}).get("tvm_version") == "0.25.dev0", "kernel params tvm_version drifted")


def _check_host_source(text: str) -> None:
    """Check the actual generated-source layout, without treating it as an OS trace.

    Only the early device headers are inside ``#ifdef __OBJC__``.  The later
    ``get_dispatch_queue``/``dispatch_sync``/``get_command_buffer`` and
    ``metal.SetStream`` text is outside that guard.  This generated C source was
    never compiled or executed; the recorded counters are code-path facts, not
    an OS-level trace, and device text must not be deleted to make the claim.
    """

    _require(bool(text.strip()), "host artifact is empty")
    _require("__tvm_ffi_scale_kernel" in text and "scale_kernel_kernel" in text, "host artifact lost the kernel entry")
    _require("TVMBackendGetFuncFromEnv" in text, "host artifact is not the generated host wrapper")
    _require(text.count("#ifdef __OBJC__") == 1, "host artifact Objective-C guard count drifted")
    guard_start = text.index("#ifdef __OBJC__")
    guard_end = text.index("#endif", guard_start)
    guarded = text[guard_start:guard_end]
    for header in ("tvm/runtime/device_api.h", "tvm/ffi/function.h", "<Metal/Metal.h>", "<Foundation/Foundation.h>", "<torch/mps.h>"):
        _require(header in guarded, f"an early device header left the Objective-C guard: {header}")
        _require(header not in text[:guard_start] + text[guard_end:], f"a device header was duplicated outside the guard: {header}")
    for device_text in ("torch::mps::get_dispatch_queue()", "dispatch_sync", "torch::mps::get_command_buffer()", "metal.SetStream"):
        _require(device_text in text, f"generated host device text was deleted: {device_text}")
        _require(device_text not in guarded, f"device call text was incorrectly moved into the Objective-C guard: {device_text}")
        _require(text.index(device_text) > guard_end, f"device call text is not in the later unguarded source: {device_text}")


def verify_frontend_identity_document(document: dict, install: dict, result: dict) -> None:
    _require(document.get("schema_id") == "shapeforge.mt0-tilelang-probe.frontend-identity.2", "wrong frontend-identity schema")
    verified = document.get("verified", {})
    installed = install["installed"]
    for key, source in (
        ("dist_info_metadata_sha256", "dist_info_metadata_sha256"),
        ("dist_info_record_sha256", "dist_info_record_sha256"),
        ("package_manifest_sha256", "tilelang_package_manifest_sha256"),
    ):
        _require(verified.get(key) == installed[source], f"frontend identity {key} drifted")
    _require(verified.get("package_files") == installed["tilelang_package_files"], "frontend identity package file count drifted")
    _require(verified.get("package_bytes") == installed["tilelang_package_bytes"], "frontend identity package byte count drifted")
    _require(verified.get("tilelang_version") == install["pinned"]["version"], "frontend identity version drifted")
    _require(verified.get("site_packages", "").endswith("/site-packages") and verified.get("package", "").endswith("/site-packages/tilelang"), "frontend identity package path drifted")
    pinned = document.get("pinned", {})
    _require(pinned.get("wheel_sha256") == install["wheel"]["sha256"], "frontend identity does not bind the wheel")
    _require(pinned.get("tvm_commit") == install["pinned"]["tvm_commit"] and pinned.get("upstream_commit") == install["pinned"]["upstream_commit"], "frontend identity commits drifted")
    _require(document.get("build_options") == result["build"]["build_options"], "frontend identity build options disagree with the result")
    _require(document.get("device_execution") == result["build"]["device_execution"], "frontend identity device execution disagrees with the result")
    _require(document.get("import", {}).get("torch_mps_availability_queries") == result["build"]["torch_mps_availability_queries"], "frontend identity MPS query counters drifted")
    _require(document.get("import", {}).get("tilelang_module", "").endswith("/site-packages/tilelang/__init__.py"), "frontend identity imported module path drifted")
    _require(document.get("python", {}).get("version", "").startswith("3.13."), "frontend identity python version drifted")


def verify_evidence(here: Path = HERE) -> None:
    """Recompute every archived artifact byte and bind it to result + identity."""

    directory = here / EVIDENCE_DIR
    _require(directory.is_dir(), "the archived evidence directory is missing")
    _check_evidence_members(sorted(path.name for path in directory.iterdir()))
    for name in ARTIFACT_NAMES:
        path = directory / name
        _require(path.is_file() and not path.is_symlink(), f"evidence member is not a regular file: {name}")
        _require(path.stat().st_nlink == 1, f"evidence member is hardlinked: {name}")
    install = _load("frontend-install.json", here)
    result = _load(PROBE_RESULT, here)
    rows = _evidence_rows(here)
    _require(rows == result["build"]["artifact_rows"], "the archived artifacts do not match the raw r5 rows")
    _require(_set_digest(rows) == result["build"]["artifact_set_sha256"], "the archived artifact-set digest drifted")
    identity = _load(f"{EVIDENCE_DIR}/build_identity.json", here)
    verify_build_identity_document(identity, rows, install, _sha256(here / "probe/candidate.py"))
    _check_kernel_source((directory / "kernel.metal").read_text(encoding="utf-8"))
    _check_device_ir(_load(f"{EVIDENCE_DIR}/device_ir.json", here))
    _check_kernel_params(_load(f"{EVIDENCE_DIR}/kernel_params.json", here))
    _check_host_source((directory / "host_module.txt").read_text(encoding="utf-8"))
    verify_frontend_identity_document(_load(f"{EVIDENCE_DIR}/frontend_identity.json", here), install, result)


def verify_negative_case(fixture_id: str, document: dict) -> None:
    """One real negative fixture document, judged by ``fixture_observation``.

    The frozen runner reuses its generic result document for fixture runs, so the
    top-level ``status`` stays ``blocked_backend_capability_unavailable`` even when
    the fixture was correctly denied.  ``fixture_observation`` is authoritative and
    the document must never contain a build block.
    """

    _require(fixture_id in NEGATIVE_FILES, f"unknown negative fixture: {fixture_id}")
    fixture, _ = NEGATIVE_FILES[fixture_id]
    _require(document.get("schema_id") == "shapeforge.mt0-tilelang-probe.probe-result.1", f"wrong negative schema: {fixture_id}")
    _require(document.get("status") == "blocked_backend_capability_unavailable", f"the frozen fixture document shape changed: {fixture_id}")
    _require(document.get("probe_executed") is False and document.get("full_build") is False, f"a negative case claims a build: {fixture_id}")
    _require("build" not in document, f"a negative case carries build evidence: {fixture_id}")
    observation = document.get("fixture_observation", {})
    _require(observation.get("fixture") == fixture, f"negative fixture identity drifted: {fixture_id}")
    _require(observation.get("matched") is True, f"negative fixture did not match its expected outcome: {fixture_id}")
    if fixture_id == "undeclared_artifact":
        _require(observation.get("expected") == "artifact_contract_rejected", "the undeclared-artifact expectation drifted")
        _require(observation.get("status") == "artifact_contract_rejected", "the undeclared artifact was not rejected")
        _require("undeclared=['artifact.txt']" in observation.get("detail", ""), "the undeclared artifact detail drifted")
        _require("isolation" not in document, "the collector rejection should not carry an isolation run")
        return
    _require(observation.get("expected") == "denied_no_artifacts", f"negative expectation drifted: {fixture_id}")
    _require(observation.get("status") == "process_failed" and observation.get("returncode") == 13, f"negative fixture is not the denied outcome: {fixture_id}")
    _require(observation.get("artifact_count") == 0, f"negative fixture produced artifacts: {fixture_id}")
    isolation = document.get("isolation", {})
    _require(isolation.get("status") == "process_failed" and isolation.get("returncode") == 13, f"negative isolation outcome drifted: {fixture_id}")
    _require(isolation.get("policy_online") is False and isolation.get("policy_hosts") == [], f"negative fixture was not offline: {fixture_id}")
    _require(isolation.get("implicit_mds_write_roots") == [] and isolation.get("internal_fence_or_wrap_used") is False, f"negative fixture added an mds/wrap exception: {fixture_id}")
    _require(isolation.get("driver_report", {}).get("fenced_returncode") == 13, f"negative driver returncode drifted: {fixture_id}")
    _require(isolation.get("driver_report", {}).get("mds_write_root_added") is False, f"negative driver added an mds root: {fixture_id}")
    if fixture_id in {"candidate_sensitive_read", "candidate_outside_write"}:
        _require(observation.get("outside_canary_unchanged") is True, f"outside canary changed: {fixture_id}")
        _require(observation.get("outside_canary_sha256") == "637af8ae02a133816e448860ee500475672d9a87ba6136572b7c650ef4d8a686", f"canary identity drifted: {fixture_id}")
    else:
        _require("outside_canary_unchanged" not in observation, "the network fixture recorded a canary it does not have")


def verify_negative_cases(here: Path = HERE) -> None:
    for fixture_id, (_, filename) in NEGATIVE_FILES.items():
        document = _load(filename, here)
        verify_negative_case(fixture_id, document)


def verify_history_document(document: dict, here: Path = HERE) -> None:
    """The failed-round history, bound to the raw results and archived inputs."""

    _require(document.get("schema_id") == "shapeforge.mt0-tilelang-probe.history.1", "wrong probe-history schema")
    _require(document.get("epoch") == EPOCH and document.get("scope") == "tilelang_source_only_frontend_probe", "probe history epoch/scope drifted")
    _require(document.get("record_epoch") == f"{EPOCH}/evidence-r5", "the evidence record epoch drifted")
    attempts = document.get("attempts", [])
    _require([row.get("round") for row in attempts] == ["r1", "r2", "r3", "r4", "r5"], "the probe history lost or reordered a round")
    for row in attempts:
        round_id = row["round"]
        result_name, returncode = EXPECTED_ATTEMPT_RESULTS[round_id]
        _require(row.get("result") == result_name, f"{round_id} result pointer drifted")
        _require(row.get("result_sha256") == _sha256(here / result_name), f"{round_id} raw result bytes changed")
        _require(row.get("returncode") == returncode, f"{round_id} returncode drifted")
        _require(row.get("candidate_sha256") == _sha256(here / row["candidate_input"]), f"{round_id} candidate input bytes changed")
        _require(row.get("frontend_entry_sha256") == _sha256(here / row["frontend_entry_input"]), f"{round_id} frontend entry input bytes changed")
        raw = _load(result_name, here)
        _require(raw.get("isolation", {}).get("returncode") == returncode, f"{round_id} raw isolation returncode drifted")
        if round_id == "r5":
            _require(raw.get("probe_executed") is True and raw.get("full_build") is True, "r5 raw result is not the completed build")
            _require(row.get("isolation_status") == "completed" and row.get("status") == "source_only_frontend_probe_completed", "r5 is not recorded as the completed round")
            _require(row.get("full_build") is True and row.get("probe_executed") is True and row.get("artifacts") == 6, "r5 does not record the full build")
            _require(row.get("stderr_log") is None and row.get("stderr_sha256") == EMPTY_SHA256, "r5 recorded stderr that does not exist")
            _require(row.get("root_cause") is None, "r5 is given a failure root cause")
            _require(row.get("status") == raw.get("status") and row.get("typed_code") == raw.get("typed_code"), "r5 status/typed_code disagrees with the raw result")
        else:
            _require(raw.get("probe_executed") is False and raw.get("full_build") is False, f"{round_id} raw result claims a build")
            _require(row.get("isolation_status") == "process_failed", f"{round_id} is not recorded as a process failure")
            _require(raw.get("status") == "full_build_rejected", f"{round_id} raw status is not full_build_rejected")
            _require(raw.get("typed_code") == "backend_capability_unavailable", f"{round_id} raw typed_code drifted")
            _require(row.get("status") == raw.get("status"), f"{round_id} status disagrees with the raw result")
            _require(row.get("typed_code") == raw.get("typed_code"), f"{round_id} typed_code disagrees with the raw result")
            _require(row.get("full_build") is False and row.get("probe_executed") is False and row.get("artifacts") == 0, f"{round_id} is falsely recorded as a build")
            _require(isinstance(row.get("root_cause"), str) and row["root_cause"], f"{round_id} hides its root cause")
            _require(row.get("stderr_log") == f"evidence/failure-logs/{round_id}-stderr.txt", f"{round_id} log pointer drifted")
            _require(row.get("stderr_sha256") == _sha256(here / row["stderr_log"]), f"{round_id} stderr bytes changed")
    logs = {
        round_id: (here / f"evidence/failure-logs/{round_id}-stderr.txt").read_text(encoding="utf-8", errors="replace")
        for round_id in FAILED_ROUNDS
    }
    _require("line_map size: 9" in logs["r1"] and "line_map size: 9" in logs["r2"], "the r1/r2 decorator-render crash is missing from the logs")
    _require("The parser does not understand the decorator" in logs["r3"], "the r3 parser diagnosis is missing from the log")
    _require("LayoutInference" in logs["r4"] and "Target context required" in logs["r4"], "the r4 Target-context diagnosis is missing from the log")
    final = document.get("final_probe", {})
    _require(final.get("result") == PROBE_RESULT and final.get("evidence_dir") == EVIDENCE_DIR, "the final probe pointers drifted")
    _require(final.get("attempt_root") == R5_ATTEMPT_ROOT, "the final attempt root is not the archived one")
    _require(set(final.get("artifact_names", [])) == set(ARTIFACT_NAMES), "the final artifact names drifted")
    _require(final.get("artifact_set_sha256") == _set_digest(_evidence_rows(here)), "the final artifact-set digest is not the archived bytes")
    _require(final.get("device_execution") == {
        "buffer_allocated": False,
        "device_library_created": False,
        "dispatch_calls": 0,
        "pipeline_state_created": False,
    }, "the final probe records device work")
    _require(final.get("device_dispatch_calls") == 0 and final.get("enable_device_compile") is False, "the final probe is not source-only")
    _require(final.get("online") is False and final.get("hosts") == [] and final.get("implicit_mds_write_roots") == [], "the final probe is not offline/mds-free")
    _require(final.get("gpu_npu_executed") is False and final.get("metal_correctness_or_performance_claim") is False, "the final probe claims device work")
    collector = final.get("collector", {})
    _require(collector.get("fenced_returncode") == 0 and collector.get("artifact_count") == 6, "the collector observation drifted")
    host_note = final.get("host_artifact_is_source_only", "")
    _require("generated C source" in host_note and "MTLCommandBuffer" in host_note, "the generated-host-source nuance was dropped")
    _require("early headers" in host_note and "outside" in host_note and "dispatch_sync" in host_note, "the host-source guard layout was misstated")
    incidents = document.get("process_incidents", [])
    _require(any(row.get("id") == "network_fixture_missing_port" and "exited 2" in row.get("what", "") and "negative-network.json" in row.get("resolution", "") for row in incidents), "the first network-fixture invocation (exit 2) was hidden")
    negatives = document.get("negative_evidence", {})
    real = negatives.get("real_executed", [])
    _require({row.get("id") for row in real} == set(REAL_NEGATIVE_IDS), "the history real negative set drifted")
    for row in real:
        _require(row.get("sha256") == _sha256(here / row["file"]), f"negative raw bytes changed: {row.get('file')}")
        _require(row.get("fixture_observation", {}).get("matched") is True, f"history records an unmatched negative: {row.get('id')}")
    _require(set(negatives.get("static_or_adapter_covered", [])) == set(STATIC_NEGATIVE_IDS), "the history static negative set drifted")
    identity_tests = negatives.get("identity_mutation_tests", {})
    _require(identity_tests.get("frontend_version_drift", "").endswith("[pinned_version]"), "the pinned-version identity mutation node is missing")
    _require(identity_tests.get("frontend_source_drift", []) == [
        "tests/shared/test_mt0_tilelang_probe.py::test_frontend_identity_drift_rejected_before_full_build[package_manifest]",
        "tests/shared/test_mt0_tilelang_probe.py::test_frontend_identity_drift_rejected_before_full_build[metadata]",
        "tests/shared/test_mt0_tilelang_probe.py::test_frontend_identity_drift_rejected_before_full_build[record]",
    ], "the package/METADATA/RECORD identity mutation nodes drifted")
    _require("IdentityDrift" in negatives.get("identity_mutation_outcome", "") and "no production-ready" in negatives["identity_mutation_outcome"], "the identity mutation outcome is not fail-closed")
    _require("fixture_observation" in negatives.get("top_level_status_quirk", ""), "the generic fixture status quirk is not explained")
    _require("probe_executed=false" in negatives.get("not_a_full_build", ""), "the negative cases are not pinned to no-full-build")
    _require(set(document.get("preserved_raw_files", [])) == {
        EXPECTED_ATTEMPT_RESULTS[round_id][0] for round_id in EXPECTED_ATTEMPT_RESULTS
    } | {filename for _, filename in NEGATIVE_FILES.values()}, "a raw result file left the preserved set")
    not_claimed = " ".join(document.get("not_claimed", []))
    for needle in ("MT0 completion", "second non-MSL DSL", "real Metal correctness"):
        _require(needle in not_claimed, f"the history overclaims: {needle}")
    chain = document.get("fix_chain", {})
    candidate = chain.get("candidate", {})
    _require(set(candidate.get("diff_lines", [])) == {
        "-from tilelang.language.tir import prim_func", "-@prim_func", "+@T.prim_func",
    }, "the candidate fix is not recorded as the decorator-only change")
    _require("AST" in candidate.get("algorithm_unchanged", "") and "identical" in candidate.get("algorithm_unchanged", ""), "the algorithm-unchanged argument was dropped")
    entry = chain.get("frontend_entry", {})
    _require(set(entry.get("r1_to_r2_added", [])) == {"import traceback", "        traceback.print_exc()"}, "the r2 diagnostic-only change drifted")
    _require(entry.get("r3_to_r4_changed") == [], "r3->r4 is not recorded as unchanged")
    _require("with target:" in entry.get("r4_to_r5", ""), "the r5 Target-context fix is not recorded")
    _require("enable_device_compile=False" in entry.get("options_unchanged", ""), "the unchanged source-only options are not recorded")
    _require("No algorithm" in entry.get("not_a_rewrite", ""), "the history does not deny an algorithm rewrite")


def verify_probe_history(here: Path = HERE) -> None:
    document = _load(HISTORY_FILE, here)
    verify_history_document(document, here)
    r5 = document["attempts"][-1]
    _require(r5["candidate_sha256"] == _sha256(here / "probe/candidate.py"), "the successful round is not the current candidate")
    _require(r5["frontend_entry_sha256"] == _sha256(here / "probe/frontend_entry.py"), "the successful round is not the current frontend entry")
    _require(document.get("unchanged_boundaries_across_rounds", {}).get("runner", "").find(_sha256(here / "probe/run_probe.py")) >= 0, "the runner digest is not the frozen one")


def _diff_lines(left: str, right: str) -> tuple[list[str], list[str]]:
    diff = difflib.unified_diff(left.splitlines(), right.splitlines(), lineterm="", n=0)
    added: list[str] = []
    removed: list[str] = []
    for line in diff:
        if line.startswith("+++") or line.startswith("---") or line.startswith("@@"):
            continue
        if line.startswith("+"):
            added.append(line[1:])
        elif line.startswith("-"):
            removed.append(line[1:])
    return added, removed


def _reconstruct_without_target(text: str) -> str:
    lines = text.splitlines(keepends=True)
    start = next(index for index, line in enumerate(lines) if line.strip() == "with target:")
    end = next(index for index in range(start + 1, len(lines)) if lines[index] == "            )\n")
    body = [line[4:] if line.startswith("    ") else line for line in lines[start + 1:end + 1]]
    return "".join(lines[:start] + body + lines[end + 1:])


def _candidate_ast(text: str) -> tuple[list[str], list[str]]:
    tree = ast.parse(text)
    decorators: list[str] = []
    body: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module == "tilelang.language.tir":
            continue
        if isinstance(node, ast.FunctionDef) and node.name == "scale_kernel":
            decorators = [ast.unparse(item) for item in node.decorator_list]
            node = ast.FunctionDef(
                name=node.name, args=node.args, body=node.body, decorator_list=[],
                returns=node.returns, type_comment=node.type_comment,
                type_params=getattr(node, "type_params", []),
            )
        body.append(ast.dump(node, include_attributes=False))
    return decorators, body


def verify_fix_chain(here: Path = HERE) -> None:
    """Prove the two fixes are frontend call-site changes, not algorithm changes."""

    inputs = here / "evidence/failure-inputs"
    r1_candidate = (inputs / "r1-candidate.py").read_text(encoding="utf-8")
    r5_candidate = (inputs / "r5-candidate.py").read_text(encoding="utf-8")
    r1_decorators, r1_body = _candidate_ast(r1_candidate)
    r5_decorators, r5_body = _candidate_ast(r5_candidate)
    _require(r1_decorators == ["prim_func"], "the r1 candidate is not the unqualified @prim_func candidate")
    _require(r5_decorators == ["T.prim_func"], "the r5 candidate is not the @T.prim_func candidate")
    _require(r1_body == r5_body, "the candidate algorithm changed between r1 and r5")
    for needle in ("N = 1024", "THREADS = 256", "T.Kernel(N // THREADS, threads=THREADS)", "T.Parallel(THREADS)", "* 2.0"):
        _require(needle in r1_candidate and needle in r5_candidate, f"the frozen algorithm lost {needle}")
    _require((here / "probe/candidate.py").read_bytes() == (inputs / "r5-candidate.py").read_bytes(), "the package candidate is not the successful r5 input")
    _require((here / "probe/frontend_entry.py").read_bytes() == (inputs / "r5-frontend_entry.py").read_bytes(), "the package frontend entry is not the successful r5 input")
    r1_entry = (inputs / "r1-frontend_entry.py").read_text(encoding="utf-8")
    r2_entry = (inputs / "r2-frontend_entry.py").read_text(encoding="utf-8")
    r3_entry = (inputs / "r3-frontend_entry.py").read_text(encoding="utf-8")
    r4_entry = (inputs / "r4-frontend_entry.py").read_text(encoding="utf-8")
    r5_entry = (inputs / "r5-frontend_entry.py").read_text(encoding="utf-8")
    added, removed = _diff_lines(r1_entry, r2_entry)
    _require(added == ["import traceback", "        traceback.print_exc()"] and removed == [], "r1->r2 is not diagnostic-only")
    added, removed = _diff_lines(r2_entry, r3_entry)
    _require(removed == ["    function = load_candidate(candidate_path)"], "r2->r3 changed more than candidate loading")
    _require(added == ["    try:", "        function = load_candidate(candidate_path)", "    except Exception:", "        traceback.print_exc()", "        raise"], "r2->r3 is not diagnostic-only")
    added, removed = _diff_lines(r3_entry, r4_entry)
    _require((added, removed) == ([], []), "r3->r4 changed the frontend entry")
    _require(r4_entry.count("with target:") == 0 and r5_entry.count("with target:") == 1, "the Target-context wrapper count drifted")
    _require(_reconstruct_without_target(r5_entry) == r4_entry, "r4->r5 changed more than the Target-context wrapper")
    _require(r5_entry.index("with target:") < r5_entry.index("tilelang_lower("), "the r5 lower call is not inside the Target context")
    _require("enable_device_compile=False" in r1_entry and "enable_device_compile=False" in r5_entry, "the source-only option changed across rounds")


def verify_review(document: dict) -> None:
    """The reviewer verdict must stay narrow and name the residual limits."""

    _require(document.get("schema_id") == "shapeforge.mt0-tilelang-probe.review.1", "wrong review schema")
    _require(document.get("epoch") == EPOCH and document.get("record_epoch") == f"{EPOCH}/evidence-r5", "review epoch drifted")
    _require(document.get("scope") == "pinned_tilelang_install_and_source_only_isolated_build_with_negatives", "the review scope was widened")
    reviewer = document.get("reviewer", {})
    _require(reviewer.get("read_only_during_the_review_pass") is True and reviewer.get("took_the_actor_summary_on_faith") is False, "the review was not read-only/raw-file based")
    for key in ("probe_rerun", "nested_agents_or_hmz_flows_used", "gpu_npu_model_or_paid_search", "raw_or_archived_evidence_modified_during_the_review_pass"):
        _require(reviewer.get(key) is False, f"the reviewer claims forbidden work: {key}")
    recomputed = " ".join(document.get("recomputed", []))
    for needle in ("artifact_set_sha256", "package manifest", "thread_extent", "r1-r4", "fixture_observation", "other_closed_sha256"):
        _require(needle in recomputed, f"the review does not re-derive: {needle}")
    findings = " ".join(document.get("findings", []))
    for needle in ("host_module.txt", "outside", "dispatch_sync", "not an OS-level trace", "PermissionError", "first network", "installed_tree_manifest_sha256", "75 fixed nodes"):
        _require(needle in findings, f"the review hides a residual limit: {needle}")
    _require("IdentityDrift" in recomputed and "pinned version" in recomputed and "METADATA" in recomputed and "RECORD" in recomputed, "the review did not independently re-derive every identity mutation")
    verdict = document.get("verdict", {})
    _require(verdict.get("done") is True, "the reviewer did not accept the narrow scope")
    done_scope = verdict.get("done_scope", "")
    _require("source-only isolated build" in done_scope and "negative cases" in done_scope, "the done scope is not the source-only build + negatives")
    _require("MT0" not in done_scope.replace("MT0-tilelang", ""), "the done scope claims MT0")
    not_done = " ".join(verdict.get("not_done", []))
    for needle in ("MT0", "MT1b", "second non-MSL DSL", "production DSL", "real Metal correctness", "publication"):
        _require(needle in not_done, f"the verdict hides an unfinished item: {needle}")
    _require("does not mean MT0" in document.get("notes", ""), "the review notes hide the MT0 boundary")


def _module_level_imports(tree: ast.Module) -> set[str]:
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return names


def _all_imports(tree: ast.Module) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return names


def verify_sources(here: Path = HERE) -> None:
    candidate_path = here / "probe/candidate.py"
    entry_path = here / "probe/frontend_entry.py"
    runner_path = here / "probe/run_probe.py"
    for path in (candidate_path, entry_path, runner_path, *sorted((here / "probe/fixtures").glob("*.py"))):
        compile(path.read_text(encoding="utf-8"), str(path), "exec")

    candidate_text = candidate_path.read_text(encoding="utf-8")
    for needle in ("tilelang.metal.language", "@T.prim_func", "T.Kernel", "T.Parallel", "N = 1024", "THREADS = 256", "* 2.0"):
        _require(needle in candidate_text, f"candidate is not the frozen Metal-dialect program: {needle}")
    for forbidden in ("from tilelang.language.tir import prim_func", "@prim_func\n"):
        _require(forbidden not in candidate_text, "the candidate revives the bare @prim_func decorator that failed r1-r3")

    entry_text = entry_path.read_text(encoding="utf-8")
    entry_tree = ast.parse(entry_text)
    module_imports = _module_level_imports(entry_tree)
    _require("tilelang" not in module_imports and "torch" not in module_imports, "the frontend entry imports tilelang/torch at module scope")
    all_imports = _all_imports(entry_tree)
    for forbidden in ("subprocess", "socket", "urllib", "shutil", "http"):
        _require(forbidden not in {name.split(".")[0] for name in all_imports}, f"the frontend entry imports {forbidden}")
    for node in ast.walk(entry_tree):
        if isinstance(node, ast.Attribute):
            _require(node.attr not in {"jit", "compile", "build", "device_codegen", "newLibraryWithSource", "dispatchThreadgroups", "makeCommandBuffer"}, f"the frontend entry reaches a forbidden call: {node.attr}")
    _require(entry_text.count("write_bytes(") == 1, "only write_artifact may write files")
    _require('if name not in ARTIFACT_NAMES:' in entry_text, "write_artifact does not enforce the declared set")
    _require("enable_device_compile=False" in entry_text, "the frontend entry does not disable device compilation")
    _require(entry_text.count("with target:") == 1, "the frontend entry must lower inside exactly one Target context")
    _require(entry_text.index("with target:") < entry_text.index("tilelang_lower("), "the lowering call is outside the Target context")
    _require("tilelang_lower(" in entry_text and "kernel_source" in entry_text, "the source-only lowering call is missing")
    _require("ARTIFACT_NAMES = (" in entry_text, "the artifact declaration is missing")
    for name in ARTIFACT_NAMES:
        _require(f'"{name}"' in entry_text, f"artifact name missing from the entry: {name}")

    runner_text = runner_path.read_text(encoding="utf-8")
    runner_tree = ast.parse(runner_text)
    runner_imports = _all_imports(runner_tree)
    _require(not any(name.startswith("tilelang") for name in runner_imports), "the trusted runner imports tilelang")
    _require("tcenv.worker_isolation" in runner_imports, "the runner does not use the single lower-level isolation adapter")
    _require("preflight_seatbelt" in runner_text and "PROFILE_PROBE" in runner_text, "the runner has no Seatbelt capability preflight")
    _require("backend_capability_unavailable" in runner_text, "the runner is not typed on capability failure")
    _require("FrontendIdentityDrift" in runner_text and "frontend_identity_drift" in runner_text, "the runner has no typed frontend-identity drift result")
    _require("FrontendStateError" in runner_text and "frontend_not_staged" in runner_text, "the runner has no typed missing-frontend result")
    _require("--no-sandbox" not in runner_text and "unfenced" not in runner_text.split("unfenced_fallback_used")[0], "the runner revives an unfenced path")
    _require("HmzFenceExecutor" in runner_text, "the runner bypasses the hmz fence")

    fixtures = here / "probe/fixtures"
    read_fixture = (fixtures / "read_outside.py").read_text(encoding="utf-8")
    write_fixture = (fixtures / "write_outside.py").read_text(encoding="utf-8")
    network_fixture = (fixtures / "network_connect.py").read_text(encoding="utf-8")
    undeclared = (fixtures / "undeclared_artifact.py").read_text(encoding="utf-8")
    for fixture in (read_fixture, write_fixture, network_fixture):
        _require("PermissionError" in fixture and "return 13" in fixture and "return 14" in fixture, "a denial fixture does not distinguish denied from allowed")
    _require("socket.create_connection" in network_fixture and "127.0.0.1" in network_fixture, "the network fixture is not a loopback connection")
    _require('"artifact.txt"' in undeclared and "write_bytes" in undeclared, "the undeclared-artifact fixture is missing")
    for fixture in (read_fixture, write_fixture, network_fixture, undeclared):
        for forbidden in (".ssh", "id_rsa", "API_KEY", "api_key", "token", "$HOME", "openai", "aws"):
            _require(forbidden not in fixture, f"a fixture references a real credential location: {forbidden}")


def verify_history(here: Path = HERE) -> None:
    selection = json.loads((here.parent / "MT0/dsl-selection.json").read_text(encoding="utf-8"))
    _require(selection.get("status") == "blocked", "the historical MT0 selection was rewritten")
    _require(selection.get("qualified_candidate_count") == 0 and selection.get("qualified_candidate_ids") == [], "the historical MT0 selection was falsely completed")
    _require(selection.get("next_stages", {}).get("MT1b") == "blocked", "MT1b was unblocked in the historical selection")
    _require(all(row.get("availability") == "interface_only" and row.get("qualified") is False for row in selection.get("candidates", [])), "a historical candidate was relabelled available")
    review = json.loads((here.parent / "MT0-isolation-profile/outer-probe-review.json").read_text(encoding="utf-8"))
    _require(review.get("review_done") is True and review.get("current_status", {}).get("mt0_status") == "blocked", "the closed isolation-profile epoch was rewritten")
    closed = _load("start-state.json", here).get("other_closed_sha256", {})
    for relative, expected in closed.items():
        _require(_sha256(ROOT / relative) == expected, f"a closed historical file was rewritten: {relative}")


def verify_history_anchor_drift(document: dict, here: Path = HERE) -> None:
    _require(document.get("schema_id") == "shapeforge.mt0-tilelang-probe.history-anchor-drift.1", "wrong history-anchor-drift schema")
    _require(document.get("epoch") == EPOCH, "history-anchor-drift epoch drifted")
    _require(document.get("classification") == "expected_living_document_anchor_drift_not_evidence_rewrite", "the drift classification was widened")
    old = ROOT / document.get("old_package", "")
    _require((old / "manifest.sha256").is_file() and (old / "verify.py").is_file(), "the closed old package is missing")
    documents = document.get("living_documents", [])
    _require(len(documents) == 2, "the drift must cover exactly the two updated living documents")
    expected_pinned = {
        "docs/架构/多DSL-Track实施计划.md": "71525f5fd0c074d91cc23303b7223a0c1435a59cf397a27b5ce087b90c8a95d5",
        "docs/可信harness/KDA-CAKE2-与ShapeForge的多Track交付对比.md": "9135995514b04cb824aa0a8982ecd2261d6c823ecd84a8841c5d3673b02ca25b",
    }
    for row in documents:
        relative = row.get("relative")
        _require(relative in expected_pinned, f"unexpected living document in drift: {relative}")
        _require(row.get("pinned_sha256") == expected_pinned[relative], f"pinned living-document hash drifted: {relative}")
        _require(row.get("current_sha256") == _sha256(ROOT / relative), f"current living-document hash was not recomputed: {relative}")
        _require(row.get("current_sha256") != row.get("pinned_sha256"), "the living document was not updated")
        _require(isinstance(row.get("reason"), str) and row["reason"], "the living-document update reason is missing")
    unchanged = document.get("old_package_unchanged", {})
    closed = _load("start-state.json", here)["other_closed_sha256"]
    pairs = {
        "manifest.sha256": "docs/架构/多DSL-Track-阶段验收/MT0-isolation-profile/manifest.sha256",
        "outer-probe-result-r2.json": "docs/架构/多DSL-Track-阶段验收/MT0-isolation-profile/outer-probe-result-r2.json",
        "outer-probe-review.json": "docs/架构/多DSL-Track-阶段验收/MT0-isolation-profile/outer-probe-review.json",
    }
    for key, relative in pairs.items():
        _require(unchanged.get(key) == closed[relative] == _sha256(ROOT / relative), f"the closed package file changed: {key}")
    _require(unchanged.get("outer-probe-result.json") == _sha256(old / "outer-probe-result.json"), "the r1 raw result changed")
    _require(unchanged.get("verification.json") == _sha256(old / "verification.json"), "the old verification.json changed")
    for key in ("verifier_modified", "raw_modified", "review_modified", "manifest_modified"):
        _require(unchanged.get(key) is False, f"the closed package was edited: {key}")
    old_verification = json.loads((old / "verification.json").read_text(encoding="utf-8"))
    scoped = old_verification.get("scoped_source_sha256", {})
    for row in documents:
        _require(scoped.get(row["relative"]) == row["pinned_sha256"], "the old verifier's pinned value was edited")
        _require(scoped.get(row["relative"]) != row["current_sha256"], "the old verifier was back-filled with the new hash")
    isolated = document.get("isolated_old_verifier_checks", {})
    _require(isolated.get("passed") == [
        "verify_document", "verify_source_identity", "verify_outer_plan", "verify_cpu_evidence",
        "verify_history", "verify_outer_review", "verify_nodes", "self_test", "verify_manifest",
    ], "the isolated old-verifier check list drifted")
    _require(isolated.get("failed") == ["verify_results"], "the old-verifier failure scope drifted")
    _require("scoped source digest mismatch" in isolated.get("failure_reason", ""), "the old-verifier failure reason is missing")
    _require(isolated.get("fixed_pytest_nodes_executed") is False, "the old fixed nodes were claimed as executed")
    not_done = " ".join(document.get("not_done", []))
    for needle in ("editing the closed", "skipping", "claiming the old verifier still passes"):
        _require(needle in not_done, f"drift record hides a forbidden action: {needle}")
    remains = " ".join(document.get("remains_valid", []))
    for needle in ("byte-identical", "scoped_source_sha256"):
        _require(needle in remains, f"drift record overclaims closure: {needle}")


def verify_nodes(document: dict | None = None, here: Path = HERE) -> None:
    if document is None:
        document = _load("nodes.json", here)
    nodes = tuple(document.get("node_ids", ()))
    _require(len(nodes) == EXPECTED_NODES and len(set(nodes)) == EXPECTED_NODES, f"fixed node registry must contain {EXPECTED_NODES} unique ids")
    counts = document.get("expected_counts", {})
    categories = {
        "new_package": "tests/shared/test_mt0_tilelang_probe.py::",
        "inherited_adapter": "tests/shared/test_worker_isolation_hmz.py::",
        "inherited_profile": "tests/shared/test_mt0_isolation_profile.py::",
        "inherited_selection": "tests/shared/test_mt0_dsl_selection.py::",
    }
    actual = {key: sum(node.startswith(prefix) for node in nodes) for key, prefix in categories.items()}
    actual["total"] = len(nodes)
    _require(counts == actual, f"node category counts drifted: {counts} != {actual}")
    _require(actual["new_package"] == 62 and actual["total"] == 75, "the fixed new-package/total node counts drifted")
    if "no tilelang import" not in document.get("execution", "") or "GPU" not in document.get("execution", ""):
        raise ValueError("node registry hides the no-import/no-device execution boundary")


def verify_manifest(here: Path = HERE) -> None:
    rows: dict[str, str] = {}
    for line in (here / "manifest.sha256").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, relative = line.split("  ", 1)
        _require(relative not in rows, f"duplicate manifest member: {relative}")
        rows[relative] = expected
    _require(set(rows) == EXPECTED_MANIFEST_MEMBERS, f"manifest file set differs: {sorted(set(rows) ^ EXPECTED_MANIFEST_MEMBERS)}")
    for relative, expected in rows.items():
        path = here / relative
        _require(path.is_file() and _sha256(path) == expected, f"manifest mismatch: {relative}")


def self_test(here: Path = HERE) -> None:
    install = _load("frontend-install.json", here)
    status = _load("probe-status.json", here)
    context = _load("context-capability.json", here)
    plan = _load("probe-plan.json", here)
    mutations = []
    changed = copy.deepcopy(install)
    changed["wheel"]["sha256"] = "0" * 64
    mutations.append((verify_frontend_install, changed))
    changed = copy.deepcopy(install)
    changed["dependencies"][0]["sha256"] = "not-a-hash"
    mutations.append((verify_frontend_install, changed))
    changed = copy.deepcopy(install)
    changed["installed"]["record_files_verified"] = 21198
    mutations.append((verify_frontend_install, changed))
    changed = copy.deepcopy(install)
    changed["source_provenance"]["native_libraries_rebuilt_from_source"] = True
    mutations.append((verify_frontend_install, changed))
    changed = copy.deepcopy(install)
    changed["execution_boundary"]["tilelang_imported"] = True
    mutations.append((verify_frontend_install, changed))
    changed = copy.deepcopy(install)
    changed["dependencies"][0]["pypi_published_sha256"] = "0" * 64
    mutations.append((verify_frontend_install, changed))
    # These are real identity mutations, not merely names in a coverage list.
    for mutation in ("pinned_version", "package_manifest", "metadata", "record"):
        changed = copy.deepcopy(install)
        if mutation == "pinned_version":
            changed["pinned"]["version"] = "0.1.16"
        elif mutation == "package_manifest":
            changed["installed"]["tilelang_package_manifest_sha256"] = "0" * 64
        elif mutation == "metadata":
            changed["installed"]["dist_info_metadata_sha256"] = "0" * 64
        else:
            changed["installed"]["dist_info_record_sha256"] = "0" * 64
        mutations.append((verify_frontend_install, changed))
    changed = copy.deepcopy(status)
    changed["status"] = "probe_completed"
    mutations.append((verify_status, changed))
    changed = copy.deepcopy(status)
    changed["probe"]["executed"] = False
    mutations.append((verify_status, changed))
    changed = copy.deepcopy(status)
    changed["current_status"]["mt0_status"] = "complete"
    mutations.append((verify_status, changed))
    changed = copy.deepcopy(status)
    changed["current_status"]["mt1b_status"] = "started"
    mutations.append((verify_status, changed))
    changed = copy.deepcopy(status)
    changed["current_status"]["production_ready"] = True
    mutations.append((verify_status, changed))
    changed = copy.deepcopy(status)
    changed["round1_dsl_decision"]["second_non_msl_dsl"] = "halide"
    mutations.append((verify_status, changed))
    changed = copy.deepcopy(status)
    changed["round1_dsl_decision"]["halide"]["installed"] = True
    mutations.append((verify_status, changed))
    changed = copy.deepcopy(status)
    changed["probe"]["artifacts_observed"] = [{"path": "kernel.metal"}]
    mutations.append((verify_status, changed))
    changed = copy.deepcopy(status)
    changed["not_claimed"] = []
    mutations.append((verify_status, changed))
    changed = copy.deepcopy(context)
    changed["interpretation"]["typed_code"] = "ok"
    mutations.append((verify_context_capability, changed))
    changed = copy.deepcopy(context)
    changed["observations"]["minimal_profile"]["returncode"] = 0
    mutations.append((verify_context_capability, changed))
    changed = copy.deepcopy(context)
    changed["interpretation"]["full_build"] = True
    mutations.append((verify_context_capability, changed))
    changed = copy.deepcopy(context)
    changed["hmz_installation_read_only_inspection"]["launched"] = True
    mutations.append((verify_context_capability, changed))
    changed = copy.deepcopy(context)
    changed["hmz_installation_read_only_inspection"]["implicit_mds_write_roots"] = ["/Users/x/Library/Caches"]
    mutations.append((verify_context_capability, changed))
    changed = copy.deepcopy(plan)
    changed["frontend"]["options"]["enable_device_compile"] = True
    mutations.append((verify_plan, changed))
    changed = copy.deepcopy(plan)
    changed["frontend"]["target"]["keys"] = ["metal", "gpu", "metal4"]
    mutations.append((verify_plan, changed))
    changed = copy.deepcopy(plan)
    changed["isolation"]["online"] = True
    mutations.append((verify_plan, changed))
    changed = copy.deepcopy(plan)
    changed["artifact_contract"]["exact_set"] = list(ARTIFACT_NAMES[:-1])
    mutations.append((verify_plan, changed))
    changed = copy.deepcopy(plan)
    changed["negative_matrix"] = [row for row in plan["negative_matrix"] if row["id"] != "missing_toolchain"]
    mutations.append((verify_plan, changed))
    changed = copy.deepcopy(plan)
    changed["completion_predicate"]["explicitly_not_implied"] = []
    mutations.append((verify_plan, changed))
    changed = copy.deepcopy(plan)
    changed["negative_matrix"][0]["evidence"] = "frontend_version_drift coverage list only"
    mutations.append((verify_plan, changed))
    changed = copy.deepcopy(status)
    changed["negative_cases"]["static_or_adapter_covered"][0]["covered_by"] = "coverage list only"
    mutations.append((verify_status, changed))
    # Moving the real device call into the header-only Objective-C guard must
    # fail; otherwise a reviewer could repeat the old incorrect guard claim.
    host_text = (here / EVIDENCE_DIR / "host_module.txt").read_text(encoding="utf-8")
    call = "  auto serialQueue = torch::mps::get_dispatch_queue();\n"
    changed_host = host_text.replace(call, "", 1).replace(
        "#include <torch/mps.h>\n#endif", "#include <torch/mps.h>\n" + call + "#endif", 1,
    )
    mutations.append((_check_host_source, changed_host))
    changed = copy.deepcopy(_load("nodes.json", here))
    changed["node_ids"] = changed["node_ids"][:-1]
    mutations.append((verify_nodes, changed))
    changed = copy.deepcopy(_load("nodes.json", here))
    changed["expected_counts"]["new_package"] = 61
    mutations.append((verify_nodes, changed))
    changed = copy.deepcopy(_load("history-anchor-drift.json", here))
    changed["living_documents"][0]["current_sha256"] = "0" * 64
    mutations.append((verify_history_anchor_drift, changed))
    changed = copy.deepcopy(_load("history-anchor-drift.json", here))
    changed["old_package_unchanged"]["verification.json"] = "0" * 64
    mutations.append((verify_history_anchor_drift, changed))
    changed = copy.deepcopy(_load("history-anchor-drift.json", here))
    changed["not_done"] = []
    mutations.append((verify_history_anchor_drift, changed))
    for mutation in ("mt0", "mt1b", "production", "second_dsl", "dispatch", "device_compile", "online", "mds", "no_evidence"):
        changed = copy.deepcopy(_load("probe-status.json", here))
        if mutation == "mt0":
            changed["current_status"]["mt0_status"] = "complete"
        elif mutation == "mt1b":
            changed["current_status"]["mt1b_status"] = "started"
        elif mutation == "production":
            changed["current_status"]["production_ready"] = True
        elif mutation == "second_dsl":
            changed["round1_dsl_decision"]["second_non_msl_dsl"] = "halide"
        elif mutation == "dispatch":
            changed["probe"]["device_dispatch_calls"] = 1
        elif mutation == "device_compile":
            changed["probe"]["enable_device_compile"] = True
        elif mutation == "online":
            changed["probe"]["online"] = True
        elif mutation == "mds":
            changed["probe"]["mds_write_roots"] = ["/Users/x/Library/Caches"]
        else:
            changed["probe"]["artifacts_observed"] = []
        mutations.append((verify_status, changed))
    changed = copy.deepcopy(_load("probe-plan.json", here))
    changed["evidence"]["artifact_set_sha256"] = "0" * 64
    mutations.append((verify_plan, changed))
    for mutation in ("dispatch", "online", "mds", "write_extra", "write_other_attempt", "write_missing"):
        changed = copy.deepcopy(_load(PROBE_RESULT, here))
        if mutation == "dispatch":
            changed["device_dispatch_calls"] = 1
        elif mutation == "online":
            changed["isolation"]["policy_online"] = True
        elif mutation == "mds":
            changed["isolation"]["implicit_mds_write_roots"] = ["/Users/x/Library/Caches"]
        elif mutation == "write_extra":
            changed["isolation"]["driver_report"]["profile"]["write"].append("/private/tmp/unrelated/build/escape")
        elif mutation == "write_other_attempt":
            changed["isolation"]["driver_report"]["profile"]["write"][-1] = "/private/tmp/shapeforge-mt0-tilelang-attempt/build/artifacts"
        else:
            changed["isolation"]["driver_report"]["profile"]["write"].pop()
        mutations.append((verify_probe_result, changed))
    rows = _evidence_rows(here)
    install_document = _load("frontend-install.json", here)
    candidate_sha256 = _sha256(here / "probe/candidate.py")
    for mutation in ("artifact_set", "dispatch", "candidate", "device_compile"):
        changed = copy.deepcopy(_load(f"{EVIDENCE_DIR}/build_identity.json", here))
        if mutation == "artifact_set":
            changed["artifacts"][0]["sha256"] = "0" * 64
        elif mutation == "dispatch":
            changed["device_execution"]["dispatch_calls"] = 1
        elif mutation == "candidate":
            changed["candidate_sha256"] = "0" * 64
        else:
            changed["build_options"]["enable_device_compile"] = True
        mutations.append((
            lambda document, _rows=rows, _install=install_document, _sha=candidate_sha256: verify_build_identity_document(document, _rows, _install, _sha),
            changed,
        ))
    for mutation in ("matched_false", "artifacts_present", "success_status"):
        changed = copy.deepcopy(_load("negative-read-outside.json", here))
        if mutation == "matched_false":
            changed["fixture_observation"]["matched"] = False
        elif mutation == "artifacts_present":
            changed["fixture_observation"]["artifact_count"] = 1
        else:
            changed["status"] = "source_only_frontend_probe_completed"
        mutations.append((lambda document, _fixture="candidate_sensitive_read": verify_negative_case(_fixture, document), changed))
    for mutation in ("drop_round", "fake_success", "root_cause", "status_drift", "typed_code_drift"):
        changed = copy.deepcopy(_load(HISTORY_FILE, here))
        if mutation == "drop_round":
            changed["attempts"] = changed["attempts"][1:]
        elif mutation == "fake_success":
            changed["attempts"][0]["full_build"] = True
        elif mutation == "status_drift":
            changed["attempts"][0]["status"] = "blocked_backend_capability_unavailable"
        elif mutation == "typed_code_drift":
            changed["attempts"][0]["typed_code"] = "process_failed"
        else:
            changed["attempts"][0]["root_cause"] = ""
        mutations.append((lambda document, _here=here: verify_history_document(document, _here), changed))
    for mutation in ("mt0_done", "drop_limit", "scope_widened"):
        changed = copy.deepcopy(_load("review.json", here))
        if mutation == "mt0_done":
            changed["verdict"]["not_done"] = [row for row in changed["verdict"]["not_done"] if row != "MT0"]
        elif mutation == "drop_limit":
            changed["findings"] = [row for row in changed["findings"] if "PermissionError" not in row]
        else:
            changed["scope"] = "mt0_completed"
        mutations.append((verify_review, changed))
    for index, (verifier, changed) in enumerate(mutations):
        try:
            verifier(changed)
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
        raise ValueError("MT0-tilelang-probe fixed nodes failed; read logs/pytest/triage/triage_summary.md")
    match = re.search(r"(\d+) passed", output)
    if match is None or int(match.group(1)) != EXPECTED_NODES or re.search(r"[1-9]\d* (?:skipped|xfailed)", output):
        raise ValueError(f"expected exactly {EXPECTED_NODES} passed without skip/xfail, got: {output[-1000:]}")


def main() -> int:
    verify_frontend_install(_load("frontend-install.json"))
    verify_context_capability(_load("context-capability.json"))
    verify_plan(_load("probe-plan.json"))
    verify_status(_load("probe-status.json"))
    verify_probe_result(_load(PROBE_RESULT))
    verify_evidence()
    verify_negative_cases()
    verify_probe_history()
    verify_fix_chain()
    verify_review(_load("review.json"))
    verify_sources()
    verify_nodes()
    verify_history()
    verify_history_anchor_drift(_load("history-anchor-drift.json"))
    self_test()
    if "--check" in sys.argv:
        verify_manifest()
        _pytest()
    print(json.dumps({
        "verifier_passed": True,
        "done": False,
        "reviewer_done": True,
        "reviewer_done_scope": "pinned_tilelang_install_and_source_only_isolated_build_with_negatives",
        "scope": "tilelang_source_only_frontend_probe",
        "epoch": EPOCH,
        "installation": "staged_and_verified",
        "probe_executed": True,
        "full_build": True,
        "artifact_set_sha256": _set_digest(_evidence_rows()),
        "device_dispatch_calls": 0,
        "enable_device_compile": False,
        "online": False,
        "mds_write_roots": [],
        "failed_rounds_preserved": list(FAILED_ROUNDS),
        "negative_cases_real": sorted(REAL_NEGATIVE_IDS),
        "negative_cases_static": sorted(STATIC_NEGATIVE_IDS),
        "typed_code": "source_only_probe_completed",
        "mt0_status": "blocked",
        "mt1b_status": "not_started_blocked",
        "second_dsl": "deferred",
        "metal_or_gpu_acceptance": False,
        "production_ready": False,
        "publication_allowed": False,
    }, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
