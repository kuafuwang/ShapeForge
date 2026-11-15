"""Static governance inventory must describe closed packages without executing them."""

from __future__ import annotations

import ast
from collections import Counter
import hashlib
import json
from pathlib import Path

import pytest

from tests.support.acceptance_git import GitObjects, tracked_paths
from tests.support.acceptance_retirement import (
    _package_aliases, _tree_size, load_retired, retirement_check, validate_retired,
)
from tests.support.acceptance_inventory import ROOT, _references, build_inventory
from tests.support.acceptance_manifest import GOLDEN_SHARED, scan_manifests
from tests.support.acceptance_status import build_status, load_epochs


@pytest.fixture(scope="module")
def inventory() -> dict[str, object]:
    return build_inventory(ROOT)


def _records(inventory: dict[str, object]) -> dict[str, dict[str, object]]:
    return {record["path"]: record for record in inventory["packages"]}


def _phase(name: str) -> str:
    return f"tests/support/golden/shared/shape-portfolio-阶段验收/{name}"


def _track(name: str) -> str:
    return f"tests/support/golden/shared/多DSL-Track-阶段验收/{name}"


def test_inventory_separates_archived_product_code_from_acceptance(inventory):
    records = _records(inventory)
    assert "tests/support/golden/shared/v1-track/tcenv/verify.py" not in records
    assert "tests/support/golden/shared/v1-track/tcenv/backends/metal/verify.py" not in records
    assert records["docs/架构/清微NPU数据模拟"]["kind"] == "verifier"
    assert not any(path.startswith("tests/support/golden/shared/varlen-v6-chunked重录-") for path in records)


def test_prompt_dependency_set_is_exact(inventory):
    records = _records(inventory)
    expected = {
        _track("MT1a"), _track("MT1b"), _track("MT4"), _track("MT6"),
        *(_phase(name) for name in ("P2-Q", "P2-S", "P2-V")),
    }
    assert {path for path, record in records.items() if "prompt" in record["depends"]} == expected


def test_dynamic_verifier_imports_are_verifier_self_tests(inventory):
    records = _records(inventory)
    expected = {
        _track("MT0"): {"tests/shared/test_mt0_dsl_selection.py"},
        _track("MT0-isolation"): {"tests/shared/test_mt0_isolation.py"},
        _track("MT0-isolation-profile"): {"tests/shared/test_mt0_isolation_profile.py"},
        _track("MT0-tilelang-probe"): {"tests/shared/test_mt0_tilelang_probe.py"},
        _track("MT1a"): {"tests/shared/test_track_mt1a.py"},
        _track("MT1b"): {"tests/backends/metal/test_migration_gpu_evidence.py"},
    }
    actual = {
        path: {item["path"] for item in record["imported_by_live"]}
        for path, record in records.items() if record["imported_by_live"]
    }
    assert actual == expected


def test_closure_dependencies_cover_the_scratch_geometry_epochs(inventory):
    records = _records(inventory)
    closure = {path for path, record in records.items() if "closure" in record["depends"]}
    assert _phase("P2-I") in closure


def test_library_imports_and_cross_hardware_inbound_are_named(inventory):
    records = _records(inventory)
    for package in (_phase("P2-0"), _phase("P2-I")):
        assert records[package]["imported_by_live"] == []
    for package in (
        "tests/support/golden/shared/跨硬件算子生成分工-阶段验收/A5",
        "tests/support/golden/shared/跨硬件算子生成分工-G0",
    ):
        assert "tests/shared/test_a6_retirement.py" in records[package]["inbound"]["active_tests"]


def test_git_index_and_package_history_are_not_truncated(inventory):
    records = _records(inventory)
    tracked = tracked_paths(ROOT)
    phase = _phase("P2-I")
    assert records[phase]["tracked_files"] == 23
    assert records[phase]["first_commit_date"] == records[phase]["last_commit_date"] == "2026-11-12"
    owned = [path for path in tracked if path.startswith(phase + "/")]
    assert records[phase]["tracked_bytes"] == sum((ROOT / path).stat().st_size for path in owned)
    retired = {row["path"] for row in load_retired(ROOT)["packages"]}
    assert _phase("P0") in retired and _phase("P1") in retired
    assert not any(path.startswith(_phase("P0") + "/") or path.startswith(_phase("P1") + "/") for path in tracked)


def test_package_references_and_pins_respect_path_boundaries(inventory):
    records = _records(inventory)
    p2 = ROOT / _phase("P2")
    mt0 = ROOT / _track("MT0")
    assert _references("tests/.../P2/verify.py", p2, ROOT)
    assert _references("P2.md", p2, ROOT)
    assert not _references("tests/.../P2-I/verify.py", p2, ROOT)
    assert not _references("P2-J", p2, ROOT)
    assert _references("tests/.../MT0/verify.py", mt0, ROOT)
    assert not _references("tests/.../MT0-isolation/verify.py", mt0, ROOT)
    p2j_pins = {item["package"] for item in records[_phase("P2-J")]["pins_other_packages"]}
    assert _phase("P2") not in p2j_pins
    p2_inbound = {item["package"] for item in records[_phase("P2")]["inbound"]["closed_packages"]}
    assert _phase("P2-J") not in p2_inbound


def test_structured_history_manifest_pins_are_kept_without_markdown_false_positives(inventory):
    records = _records(inventory)
    p2q_pins = {item["package"] for item in records[_phase("P2-Q")]["pins_other_packages"]}
    assert p2q_pins == {_phase(f"P2-{letter}") for letter in "IJKLN"}
    p2s_to_i = next(
        item for item in records[_phase("P2-S")]["pins_other_packages"]
        if item["package"] == _phase("P2-I")
    )
    assert f"{_phase('P2-S')}/acceptance-baseline.json" in p2s_to_i["files"]
    assert f"{_phase('P2-S')}/S.md" not in p2s_to_i["files"]


def test_verifier_loop_and_named_path_pins_are_not_lost(inventory):
    records = _records(inventory)
    p2j_pins = {item["package"] for item in records[_phase("P2-J")]["pins_other_packages"]}
    assert _phase("P2-I") in p2j_pins
    p2n_pins = {item["package"] for item in records[_phase("P2-N")]["pins_other_packages"]}
    assert {_phase(f"P2-{letter}") for letter in "IJKL"} <= p2n_pins
    p2v_to_s = next(
        item for item in records[_phase("P2-V")]["pins_other_packages"]
        if item["package"] == _phase("P2-S")
    )
    assert f"{_phase('P2-S')}/evidence/experience-overlay.md" in p2v_to_s["pinned_paths"]
    probe_to_profile = next(
        item for item in records[_track("MT0-tilelang-probe")]["pins_other_packages"]
        if item["package"] == _track("MT0-isolation-profile")
    )
    assert _track("MT0-isolation-profile") not in probe_to_profile["pinned_paths"]


def test_direct_shared_packages_keep_their_inbound_references(inventory):
    records = _records(inventory)
    g0 = "tests/support/golden/shared/跨硬件算子生成分工-G0"
    assert records[g0]["inbound"]["AGENTS.md"] == ["AGENTS.md"]
    retired = {row["path"] for row in load_retired(ROOT)["packages"]}
    for name in ("varlen-v6-chunked验收", "varlen-v6-chunked重录-20261110"):
        path = f"tests/support/golden/shared/{name}"
        assert path in retired
        assert path not in records


def test_n0_manifest_path_resolution_and_hash_buckets_are_reproduced():
    baseline = next(epoch for epoch in load_epochs(ROOT) if epoch["epoch"] == "20261115-baseline")
    mismatches = {
        (row["package"], row["detail"]["path"])
        for row in baseline["expected_drift"] if row["check"] == "manifest_sha256"
    }
    counts = Counter()
    for entry in scan_manifests(ROOT):
        package = entry.package.resolve().relative_to(ROOT).as_posix()
        matches = (package, entry.name) not in mismatches
        if entry.manifest == ROOT / "docs/架构/清微NPU数据模拟/manifest.sha256" and entry.name == "README.md":
            # Reproduce N0's pre-D3 bucket without hiding the current bytes:
            # the live status test below requires the approved new manifest hash.
            assert entry.expected_sha256 == entry.actual_sha256 == (
                "b772a0b5737aeda33453a5c1f3e721bede64877b578484ac836d33e3b6dcd414"
            )
            matches = False
        counts[(entry.resolution, matches)] += 1
    assert counts == {
        ("package_internal", True): 1894,
        ("package_internal", False): 1,
        ("repo_root", True): 276,
        ("repo_root", False): 17,
        ("relocated_docs", True): 46,
        ("same_tree", True): 4,
        ("legacy_four_parent", True): 3,
        ("legacy_four_parent", False): 14,
    }


def test_node_pins_come_from_registries_not_historical_prose(inventory):
    records = _records(inventory)
    # These packages mention test nodes in reviews, prose or failure evidence,
    # but their verifiers do not register a node list that would be invalidated
    # by renaming those tests.
    for package in (
        "tests/support/golden/shared/跨硬件算子生成分工-G0",
    ):
        assert records[package]["pinned_live_nodes"] == []
        assert records[package]["referenced_live_nodes"]

    mt4 = records[_track("MT4")]
    expected = {
        f"tests/backends/metal/test_tilelang_mt4.py::{name}"
        for name in (
            "test_mt4_prompt_and_signature_describe_tilelang_not_an_msl_submission",
            "test_mt4_packer_preserves_bytes_and_never_executes_candidate",
            "test_mt4_projection_and_preflight_keep_dsl_identity_and_msl_guard",
            "test_mt4_dsl_confirmation_is_explicitly_deferred_not_relabeled_as_msl",
            "test_mt4_resume_uses_taic_not_a_fake_metal_extension",
            "test_mt4_evaluate_candidate_uses_the_production_portfolio_runner",
            "test_mt4_fake_agent_drives_real_orchestrator_and_resume_without_api",
        )
    }
    assert set(mt4["pinned_live_nodes"]) == expected
    historical_failure = (
        "tests/backends/metal/test_shape_portfolio_confirmation.py::"
        "test_p2_b_suite_rejects_expected_and_report_coordinated_forgery"
    )
    assert historical_failure not in mt4["pinned_live_nodes"]
    assert historical_failure in mt4["referenced_live_nodes"]

    mt1b = records[_track("MT1b")]
    nodes = json.loads((ROOT / _track("MT1b") / "nodes.json").read_text(encoding="utf-8"))
    assert set(mt1b["pinned_live_nodes"]) == set(nodes["positive"])
    for historical in (
        "tests/shared/test_mt0_tilelang_probe.py::test_history_anchor_drift_is_recorded_without_editing_the_old_package",
        "tests/shared/test_mt0_tilelang_probe.py::test_verifier_accepts_the_current_package",
        "tests/shared/test_remote_job_engine.py::test_formal_priority_allows_aged_cross_run_selfcheck",
    ):
        assert historical not in mt1b["pinned_live_nodes"]
        assert historical in mt1b["referenced_live_nodes"]


def _write_epoch(root: Path, rows: list[dict[str, object]]) -> None:
    directory = root / GOLDEN_SHARED / "acceptance-epochs"
    directory.mkdir(parents=True)
    document = {
        "epoch": "20261115-baseline",
        "date": "2026-11-15",
        "reason": "CPU governance fixture",
        "commit": "0" * 40,
        "approved_by": "test human",
        "change_kind": "worktree",
        "before": {},
        "after": {},
        "expected_drift": rows,
        "manifest_pins": {
            str(manifest.relative_to(root)): hashlib.sha256(manifest.read_bytes()).hexdigest()
            for manifest in (root / GOLDEN_SHARED).glob("**/manifest.sha256")
        },
        "manifest_rows": {
            str(manifest.relative_to(root)): {
                line.split("  ", 1)[1]: line.split("  ", 1)[0]
                for line in manifest.read_text(encoding="utf-8").splitlines() if line
            }
            for manifest in (root / GOLDEN_SHARED).glob("**/manifest.sha256")
        },
    }
    (directory / "20261115-baseline.json").write_text(json.dumps(document), encoding="utf-8")


def _write_manifest_package(root: Path, name: str, contents: bytes, expected: str) -> Path:
    package = root / GOLDEN_SHARED / "P"
    package.mkdir(parents=True)
    (package / name).write_bytes(contents)
    (package / "manifest.sha256").write_text(f"{expected}  {name}\n", encoding="utf-8")
    return package


def test_status_is_clean_and_d3_is_an_approved_document_update():
    document = build_status(ROOT)
    assert document["summary"]["rewritten"] == 0
    assert document["summary"]["unregistered"] == 0
    assert document["summary"]["environment_unregistered"] == 0
    assert document["document_updates"] == [{
        "package": "docs/架构/清微NPU数据模拟",
        "path": "README.md",
        "epoch": "20261115-baseline",
        "approved_by": "用户（本计划与 2026-11-14 基线/D3 确认）",
        "before": "24791109d9628c87157db13706ae17bb93c10e1dc751bcb1d7ab1fa2bcb20298",
        "after": "b772a0b5737aeda33453a5c1f3e721bede64877b578484ac836d33e3b6dcd414",
    }]


def test_status_never_lets_an_epoch_excuse_internal_evidence_rewrite(tmp_path):
    actual = hashlib.sha256(b"changed").hexdigest()
    _write_manifest_package(tmp_path, "evidence.json", b"changed", "0" * 64)
    _write_epoch(tmp_path, [{
        "package": "tests/support/golden/shared/P",
        "check": "manifest_sha256",
        "detail": {
            "path": "evidence.json", "before": "0" * 64, "after": actual,
            "resolution": "package_internal",
        },
    }])
    document = build_status(tmp_path)
    assert document["summary"]["rewritten"] == 1
    assert document["summary"]["registered"] == 0


def test_document_update_requires_the_manifest_to_name_the_new_bytes(tmp_path):
    after = hashlib.sha256(b"readable\n").hexdigest()
    _write_manifest_package(tmp_path, "README.md", b"readable\n", "0" * 64)
    _write_epoch(tmp_path, [{
        "package": "tests/support/golden/shared/P",
        "check": "document_update",
        "detail": {
            "path": "README.md", "before": "0" * 64, "after": after,
            "resolution": "package_internal", "classification": "文档更新，不是证据篡改",
        },
    }])
    document = build_status(tmp_path)
    assert document["summary"]["rewritten"] == 1
    assert document["summary"]["invalid_document_updates"] == 1
    assert document["summary"]["document_updates"] == 0


def test_registered_outside_drift_does_not_authorize_a_third_version(tmp_path):
    target = tmp_path / "tcenv/live.py"
    target.parent.mkdir(parents=True)
    target.write_text("second\n", encoding="utf-8")
    package = tmp_path / GOLDEN_SHARED / "P"
    package.mkdir(parents=True)
    (package / "manifest.sha256").write_text(f"{'0' * 64}  tcenv/live.py\n", encoding="utf-8")
    second = hashlib.sha256(b"second\n").hexdigest()
    _write_epoch(tmp_path, [{
        "package": "tests/support/golden/shared/P",
        "check": "manifest_sha256",
        "detail": {
            "path": "tcenv/live.py", "before": "0" * 64, "after": second,
            "resolution": "repo_root",
        },
    }])
    assert build_status(tmp_path)["summary"]["registered"] == 1
    target.write_text("third\n", encoding="utf-8")
    document = build_status(tmp_path)
    assert document["summary"]["registered"] == 0
    assert document["summary"]["unregistered"] == 1


def test_deleted_internal_evidence_cannot_fall_back_to_a_root_file(tmp_path):
    contents = b"original\n"
    package = _write_manifest_package(tmp_path, "evidence.json", contents, hashlib.sha256(contents).hexdigest())
    (package / "evidence.json").unlink()
    (tmp_path / "evidence.json").write_bytes(contents)
    _write_epoch(tmp_path, [{
        "package": "tests/support/golden/shared/P",
        "check": "manifest_sha256",
        "detail": {
            "path": "evidence.json", "before": hashlib.sha256(contents).hexdigest(),
            "after": None, "resolution": "package_internal",
        },
    }])
    document = build_status(tmp_path)
    assert document["summary"]["rewritten"] == 1
    assert document["summary"]["registered"] == 0
    assert document["packages"][0]["rewritten"][0]["path"] == "evidence.json"


@pytest.mark.parametrize("name", ["evidence.json", "README.md"])
def test_rewriting_bytes_and_manifest_together_is_still_detected(tmp_path, name):
    before = hashlib.sha256(b"before\n").hexdigest()
    package = _write_manifest_package(tmp_path, name, b"before\n", before)
    _write_epoch(tmp_path, [])
    assert build_status(tmp_path)["summary"]["rewritten"] == 0
    after = hashlib.sha256(b"after\n").hexdigest()
    (package / name).write_bytes(b"after\n")
    (package / "manifest.sha256").write_text(f"{after}  {name}\n", encoding="utf-8")
    document = build_status(tmp_path)
    assert document["summary"]["rewritten"] == 1
    assert document["packages"][0]["rewritten"] == [{
        "path": "manifest.sha256",
        "expected": "baseline or approved document update",
        "actual": hashlib.sha256((package / "manifest.sha256").read_bytes()).hexdigest(),
    }]


def _loose_object(root: Path, kind: str, body: bytes) -> str:
    import zlib

    raw = f"{kind} {len(body)}\0".encode() + body
    oid = hashlib.sha1(raw).hexdigest()
    path = root / ".git/objects" / oid[:2] / oid[2:]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(zlib.compress(raw))
    return oid


def _git_tree(entries: list[tuple[str, str, str]]) -> bytes:
    chunks = []
    for mode, name, oid in sorted(entries, key=lambda item: item[1]):
        chunks.append(f"{mode} {name}\0".encode() + bytes.fromhex(oid))
    return b"".join(chunks)


def _remove_package(package: Path) -> None:
    for child in package.iterdir():
        child.unlink()
    package.rmdir()


def test_an_invalid_retirement_record_cannot_hide_a_deleted_manifest(tmp_path):
    """A path-only row is not a retirement. Status must fail closed instead of reporting a clean tree."""
    contents = b"before\n"
    package = _write_manifest_package(tmp_path, "evidence.json", contents, hashlib.sha256(contents).hexdigest())
    _write_epoch(tmp_path, [])
    retired = tmp_path / "tests/support/golden/shared/retired.json"
    retired.parent.mkdir(parents=True, exist_ok=True)
    retired.write_text(
        json.dumps({"schema_version": 1, "packages": [{"path": "tests/support/golden/shared/P"}]}),
        encoding="utf-8",
    )
    _remove_package(package)
    with pytest.raises(ValueError, match="missing required fields"):
        build_status(tmp_path)


def test_a_validated_retirement_still_exempts_a_missing_manifest(tmp_path):
    """A record that validates against Git still means a missing manifest is not a rewrite."""
    contents = b"before\n"
    package = _write_manifest_package(tmp_path, "evidence.json", contents, hashlib.sha256(contents).hexdigest())
    _write_epoch(tmp_path, [])
    package_path = "tests/support/golden/shared/P"
    epoch_path = tmp_path / GOLDEN_SHARED / "acceptance-epochs/20261115-baseline.json"
    epoch = json.loads(epoch_path.read_text(encoding="utf-8"))
    epoch["package_roots"] = [package_path]
    epoch_path.write_text(json.dumps(epoch), encoding="utf-8")
    blob = _loose_object(tmp_path, "blob", contents)
    current = _loose_object(tmp_path, "tree", _git_tree([("100644", "evidence.json", blob)]))
    package_tree = current
    for name in ("P", "shared", "golden", "support", "tests"):
        parent = _loose_object(tmp_path, "tree", _git_tree([("40000", name, current)]))
        if name == "P":
            package_tree = current
        current = parent
    commit_body = (
        f"tree {current}\n"
        "author Test <test@example.com> 1700000000 +0000\n"
        "committer Test <test@example.com> 1700000000 +0000\n"
        "\n"
        "fixture\n"
    ).encode()
    commit = _loose_object(tmp_path, "commit", commit_body)
    retired = tmp_path / "tests/support/golden/shared/retired.json"
    retired.write_text(json.dumps({
        "schema_version": 1,
        "packages": [{
            "path": package_path,
            "retired_at": "2026-11-15",
            "reason": "fixture package left the temporary worktree",
            "successor": None,
            "approved_by": "test human",
            "last_commit": commit,
            "tree": package_tree,
            "files": 1,
            "bytes": len(contents),
            "restore": f"git checkout {commit} -- {package_path}",
            "change": "20261115-retire-closed-packages",
        }],
    }), encoding="utf-8")
    _remove_package(package)
    assert build_status(tmp_path)["summary"]["rewritten"] == 0


def test_deleting_a_pinned_manifest_cannot_hide_evidence_rewrite(tmp_path):
    package = _write_manifest_package(tmp_path, "evidence.json", b"before\n", hashlib.sha256(b"before\n").hexdigest())
    _write_epoch(tmp_path, [])
    (package / "manifest.sha256").unlink()
    (package / "evidence.json").write_bytes(b"after\n")
    document = build_status(tmp_path)
    assert document["summary"]["rewritten"] == 1
    assert document["packages"][0]["rewritten"][0]["actual"] is None


def _write_document_epoch(
    root: Path, *, readme_before: str, readme_after: str, manifest_before: str, manifest_after: str,
) -> None:
    document = {
        "epoch": "20261116-document",
        "date": "2026-11-16",
        "reason": "approved Markdown-only fixture",
        "commit": "1" * 40,
        "approved_by": "test human",
        "change_kind": "worktree",
        "before": {},
        "after": {},
        "expected_drift": [{
            "package": "tests/support/golden/shared/P",
            "check": "document_update",
            "detail": {
                "path": "README.md", "before": readme_before, "after": readme_after,
                "resolution": "package_internal", "classification": "文档更新，不是证据篡改",
                "manifest_before": manifest_before, "manifest_after": manifest_after,
            },
        }],
    }
    path = root / GOLDEN_SHARED / "acceptance-epochs/20261116-document.json"
    path.write_text(json.dumps(document), encoding="utf-8")


@pytest.mark.parametrize("change_evidence", [False, True])
def test_document_update_changes_exactly_one_manifest_row(tmp_path, change_evidence):
    package = tmp_path / GOLDEN_SHARED / "P"
    package.mkdir(parents=True)
    readme_before = hashlib.sha256(b"readme before\n").hexdigest()
    evidence_before = hashlib.sha256(b"evidence before\n").hexdigest()
    (package / "README.md").write_bytes(b"readme before\n")
    (package / "evidence.json").write_bytes(b"evidence before\n")
    manifest = package / "manifest.sha256"
    manifest.write_text(f"{readme_before}  README.md\n{evidence_before}  evidence.json\n", encoding="utf-8")
    _write_epoch(tmp_path, [])
    manifest_before = hashlib.sha256(manifest.read_bytes()).hexdigest()

    readme_after = hashlib.sha256(b"readme after\n").hexdigest()
    evidence_after = hashlib.sha256(b"evidence after\n").hexdigest() if change_evidence else evidence_before
    (package / "README.md").write_bytes(b"readme after\n")
    if change_evidence:
        (package / "evidence.json").write_bytes(b"evidence after\n")
    manifest.write_text(f"{readme_after}  README.md\n{evidence_after}  evidence.json\n", encoding="utf-8")
    _write_document_epoch(
        tmp_path, readme_before=readme_before, readme_after=readme_after,
        manifest_before=manifest_before, manifest_after=hashlib.sha256(manifest.read_bytes()).hexdigest(),
    )
    document = build_status(tmp_path)
    assert document["summary"]["document_updates"] == 1
    assert document["summary"]["rewritten"] == int(change_evidence)


def test_non_baseline_cannot_carry_even_null_manifest_pins(tmp_path):
    _write_manifest_package(tmp_path, "evidence.json", b"same\n", hashlib.sha256(b"same\n").hexdigest())
    _write_epoch(tmp_path, [])
    path = tmp_path / GOLDEN_SHARED / "acceptance-epochs/20261115-baseline.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    document["epoch"] = "20261116-other"
    document["date"] = "2026-11-16"
    document["manifest_pins"] = None
    other = path.with_name("20261116-other.json")
    other.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="only the N0 baseline"):
        load_epochs(tmp_path)


def test_epoch_loader_rejects_a_latest_alias(tmp_path):
    directory = tmp_path / GOLDEN_SHARED / "acceptance-epochs"
    directory.mkdir(parents=True)
    (directory / "latest.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid acceptance epoch filename"):
        load_epochs(tmp_path)


def test_live_anchors_are_current_and_closed_verifiers_never_read_them():
    from taic.trust import current_prompt_sha1
    from tests.support.live_anchors import load_live_anchors, measure_pack_closure

    anchors = load_live_anchors(ROOT)
    assert current_prompt_sha1() == anchors["prompt_sha1"]
    files, digest = measure_pack_closure(ROOT)
    assert list(files) == anchors["pack_closure"]["files"]
    assert digest == anchors["pack_closure"]["sha256"]
    verifiers = list((ROOT / GOLDEN_SHARED).glob("**/verify.py"))
    assert len(verifiers) == 23
    verifiers.append(ROOT / "docs/架构/清微NPU数据模拟/verify.py")
    assert all("live_anchors" not in path.read_text(encoding="utf-8") for path in verifiers)


RETIRED_PATHS = (
    "tests/support/golden/shared/shape-portfolio-阶段验收/P0",
    "tests/support/golden/shared/shape-portfolio-阶段验收/P1",
    "tests/support/golden/shared/shape-portfolio-阶段验收/P2-G",
    "tests/support/golden/shared/shape-portfolio-阶段验收/P2-H",
    "tests/support/golden/shared/shape-portfolio-阶段验收/P2-T",
    "tests/support/golden/shared/shape-portfolio-阶段验收/P2-U",
    "tests/support/golden/shared/varlen-v6-chunked重录-20261110",
    "tests/support/golden/shared/varlen-v6-chunked验收",
    "tests/support/golden/shared/多DSL-Track-阶段验收/MT5",
    "tests/support/golden/shared/跨硬件算子生成分工-阶段验收/A1",
    "tests/support/golden/shared/跨硬件算子生成分工-阶段验收/A2",
    "tests/support/golden/shared/跨硬件算子生成分工-阶段验收/A3",
    "tests/support/golden/shared/跨硬件算子生成分工-阶段验收/A4",
    "tests/support/golden/shared/跨硬件算子生成分工-阶段验收/A6",
    "tests/support/golden/shared/shape-portfolio-阶段验收/P2-M",
    "tests/support/golden/shared/跨硬件算子生成分工-阶段验收/B-evidence-rerecord-2026-11-07",
    "tests/support/golden/shared/shape-portfolio-阶段验收/P2-O",
    "tests/support/golden/shared/shape-portfolio-阶段验收/P2-P",
    "tests/support/golden/shared/shape-portfolio-阶段验收/P2-R",
    "tests/support/golden/shared/shape-portfolio-阶段验收/P3",
)


def test_retirement_registry_matches_git_history_without_restoring():
    """Recoverability is the recorded tree object. This test does not check the files back out."""
    document = load_retired(ROOT)
    assert "limits" not in document
    assert [row["path"] for row in document["packages"]] == list(RETIRED_PATHS)
    assert {row["change"] for row in document["packages"]} == {"20261115-retire-closed-packages"}
    assert {row["approved_by"] for row in document["packages"]} == {
        "用户（2026-11-15 授权修改闸门并移出工作区）",
    }
    for row in document["packages"]:
        assert row["files"] > 0
        assert not (ROOT / row["path"]).exists()
    try:
        validate_retired(document, ROOT)
    except KeyError:
        if not (ROOT / ".git/shallow").is_file():
            raise
        pytest.skip("the shallow clone does not contain a retired tree; fetch history before retiring")


def _retirement_row(path: str, change: str, day: str) -> dict[str, object]:
    return {
        "path": path, "retired_at": day, "reason": "test", "successor": None,
        "approved_by": "test human", "last_commit": "0" * 40, "tree": "1" * 40,
        "files": 1, "bytes": 1,
        "restore": f"git checkout {'0' * 40} -- {path}", "change": change,
    }


def test_retirement_has_no_frequency_cap():
    document = {
        "schema_version": 1,
        "packages": [
            _retirement_row("tests/support/golden/shared/X1", "20260101-change-a", "2026-01-01"),
            _retirement_row("tests/support/golden/shared/X2", "20260101-change-a", "2026-01-02"),
            *[
                _retirement_row(
                    f"tests/support/golden/shared/Y{index}",
                    f"20260101-change-{index}",
                    "2026-01-01",
                )
                for index in range(4)
            ],
        ],
    }
    validate_retired(document)


def test_retirement_tree_record_reproduces_a_real_git_tree():
    path = _phase("P2-G")
    git = GitObjects(ROOT)
    commit = git.head()
    tree = git.path_oid(commit, path)
    files, size = _tree_size(git, tree)
    document = {
        "schema_version": 1,
        "packages": [{
            **_retirement_row(path, "20261115-test-only-not-a-retirement", "2026-11-15"),
            "last_commit": commit, "tree": tree, "files": files, "bytes": size,
            "restore": f"git checkout {commit} -- {path}",
        }],
    }
    validate_retired(document, ROOT)
    on_disk = {row["path"] for row in load_retired(ROOT)["packages"]}
    assert path in on_disk
    assert on_disk == set(RETIRED_PATHS)


def test_retirement_paths_and_change_ids_are_canonical():
    for path in ("", ".", "tests/support/golden/shared/a/./b", "README.md"):
        document = {
            "schema_version": 1,
            "packages": [_retirement_row(path, "20261115-change", "2026-11-15")],
        }
        with pytest.raises(ValueError, match="path"):
            validate_retired(document)
    document["packages"] = [_retirement_row("tests/support/golden/shared/P", " 20261115-change ", "2026-11-15")]
    with pytest.raises(ValueError, match="normalized"):
        validate_retired(document)


def test_retirement_does_not_reject_package_names():
    """Names such as G0, A5 and P2 are not a ban. G0 stays because tests still read it."""
    g0 = "tests/support/golden/shared/跨硬件算子生成分工-G0"
    assert "G0" in _package_aliases(g0)
    document = {
        "schema_version": 1,
        "packages": [
            _retirement_row(path, "20261115-name-check", "2026-11-15")
            for path in (
                g0,
                _phase("P2"),
                _phase("P2-0"),
                "tests/support/golden/shared/跨硬件算子生成分工-阶段验收/A5",
            )
        ],
    }
    validate_retired(document)


def test_retirement_rejects_subtrees_and_parent_directories():
    git = GitObjects(ROOT)
    commit = git.head()
    for path in (
        _phase("P2") + "/evidence",
        "tests/support/golden/shared/shape-portfolio-阶段验收",
    ):
        tree = git.path_oid(commit, path)
        files, size = _tree_size(git, tree)
        document = {
            "schema_version": 1,
            "packages": [{
                **_retirement_row(path, "20261115-subtree", "2026-11-15"),
                "last_commit": commit, "tree": tree, "files": files, "bytes": size,
                "restore": f"git checkout {commit} -- {path}",
            }],
        }
        with pytest.raises(ValueError, match="exact inventoried package root"):
            validate_retired(document, ROOT)


def test_open_package_retirement_check_reports_blockers_without_deleting():
    path = "tests/support/golden/shared/跨硬件算子生成分工-G0"
    report = retirement_check(ROOT, path)
    assert report["eligible"] is False
    assert report["deletion_executed"] is False
    checks = {item["name"]: item for item in report["checks"]}
    assert checks["D1_history_rule"]["status"] == "pass"
    assert "tests/shared/test_v1_identity_g0.py" in checks["active_references"]["detail"]
    assert checks["archive_roots"]["status"] == "blocked"
    assert checks["AGENTS_unique_source_human_review"]["status"] == "human_required"
    assert (ROOT / path / "manifest.sha256").is_file()


def test_inventory_does_not_execute_verifiers_or_spawn_processes():
    for name in ("acceptance_inventory.py", "acceptance_manifest.py", "acceptance_git.py", "acceptance_status.py"):
        path = ROOT / "tests/support" / name
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                assert all(alias.name != "subprocess" for alias in node.names), (name, node.lineno)
                assert all(not alias.name.endswith("verify") for alias in node.names), (name, node.lineno)
            elif isinstance(node, ast.ImportFrom):
                assert node.module != "subprocess", (name, node.lineno)
                assert not (node.module or "").endswith("verify"), (name, node.lineno)
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                assert node.func.id not in {"exec", "eval"}, (name, node.lineno)
