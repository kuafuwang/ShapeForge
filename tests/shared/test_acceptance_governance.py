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
from tests.support.acceptance_inventory import ROOT, _package_paths, _references, build_inventory
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
    assert {path for path, record in records.items() if "prompt" in record["depends"]} == set()
    for name in ("MT1a", "MT1b", "MT4", "MT6"):
        assert _track(name) not in records


def test_dynamic_verifier_imports_are_verifier_self_tests(inventory):
    """Live tests load the tests/support copies, not the package verify.py files."""
    records = _records(inventory)
    actual = {
        path: {item["path"] for item in record["imported_by_live"]}
        for path, record in records.items() if record["imported_by_live"]
    }
    assert actual == {}


def test_closure_dependencies_cover_the_scratch_geometry_epochs(inventory):
    records = _records(inventory)
    closure = {path for path, record in records.items() if "closure" in record["depends"]}
    removed = {_phase(name) for name in ("P2-I", "P2-J", "P2-K", "P2-L", "P2-N", "P2-Q", "P2-S", "P2-V")}
    assert removed.isdisjoint(records)
    assert closure == set()
    assert _track("MT1b") not in records and _track("MT2") not in records


def test_library_imports_and_cross_hardware_inbound_are_named(inventory):
    records = _records(inventory)
    for package in (_phase("P2-0"), _phase("P2-I")):
        assert package not in records
    assert "tests/support/golden/shared/跨硬件算子生成分工-阶段验收/A5" not in records
    package = "tests/support/golden/shared/跨硬件算子生成分工-G0"
    assert "tests/shared/test_a6_retirement.py" in records[package]["inbound"]["active_tests"]


def test_restored_b_packages_and_fixture_copies_stay_out_of_inventory(tmp_path: Path):
    """A checkout of B1-B4 must not re-enter the live inventory, and fixture copies are not packages."""
    stage = tmp_path / "tests/support/golden/shared/跨硬件算子生成分工-阶段验收"
    fixtures = tmp_path / "tests/support/golden/shared/acceptance-epochs/fixtures"
    for name in ("B1", "B2", "B3", "B4"):
        boundary = stage / name / "boundary.json"
        boundary.parent.mkdir(parents=True)
        boundary.write_text("{}\n", encoding="utf-8")
    for name in ("b1", "b2", "b3", "b4"):
        copy = fixtures / name / "boundary.json"
        copy.parent.mkdir(parents=True)
        copy.write_text("{}\n", encoding="utf-8")
    live = tmp_path / "tests/support/golden/shared/跨硬件算子生成分工-G0/verify.py"
    live.parent.mkdir(parents=True)
    live.write_text("# live\n", encoding="utf-8")
    found = {path.resolve().relative_to(tmp_path.resolve()).as_posix() for path in _package_paths(tmp_path)}
    assert found == {"tests/support/golden/shared/跨硬件算子生成分工-G0"}


def test_git_index_and_package_history_are_not_truncated(inventory):
    records = _records(inventory)
    tracked = tracked_paths(ROOT)
    removed = {
        _phase(name)
        for name in ("P2", "P2-0", "P2-I", "P2-J", "P2-K", "P2-L", "P2-N", "P2-Q", "P2-S", "P2-V", "P4")
    }
    assert removed.isdisjoint(records)
    leftovers = [path for path in tracked if any(path.startswith(phase + "/") for phase in removed)]
    assert leftovers == []
    retired = {row["path"] for row in load_retired(ROOT)["packages"]}
    assert removed <= retired
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
    assert _phase("P2") not in records and _phase("P2-J") not in records


def _remaining_portfolio_epoch() -> dict:
    return next(
        item for item in load_epochs(ROOT)
        if item["epoch"] == "20261115-retire-remaining-portfolio"
    )


def _epoch_pin_targets(source: str) -> set[str]:
    return {
        row["detail"]["retired_package"]
        for row in _remaining_portfolio_epoch()["expected_drift"]
        if row["check"] == "retirement_pin" and row["package"] == source
    }


def test_structured_history_manifest_pins_are_kept_without_markdown_false_positives(inventory):
    records = _records(inventory)
    assert {_phase("P2-Q"), _phase("P2-S"), _phase("P2-I")}.isdisjoint(records)
    assert _epoch_pin_targets(_phase("P2-Q")) == {_phase(f"P2-{letter}") for letter in "IJKLN"}
    pin = next(
        row for row in _remaining_portfolio_epoch()["expected_drift"]
        if row["check"] == "retirement_pin"
        and row["package"] == _phase("P2-S")
        and row["detail"]["retired_package"] == _phase("P2-I")
    )
    assert f"{_phase('P2-S')}/acceptance-baseline.json" in pin["detail"]["source_files"]
    assert f"{_phase('P2-S')}/S.md" not in pin["detail"]["source_files"]


def test_verifier_loop_and_named_path_pins_are_not_lost(inventory):
    records = _records(inventory)
    removed = {_phase(name) for name in ("P2-J", "P2-N", "P2-V", "P2-S", "P2-I")}
    assert removed.isdisjoint(records)
    assert _phase("P2-I") in _epoch_pin_targets(_phase("P2-J"))
    assert {_phase(f"P2-{letter}") for letter in "IJKL"} <= _epoch_pin_targets(_phase("P2-N"))
    p2v_to_s = next(
        row for row in _remaining_portfolio_epoch()["expected_drift"]
        if row["check"] == "retirement_pin"
        and row["package"] == _phase("P2-V")
        and row["detail"]["retired_package"] == _phase("P2-S")
    )
    assert f"{_phase('P2-S')}/evidence/experience-overlay.md" in p2v_to_s["detail"]["pinned_paths"]
    assert _track("MT0-tilelang-probe") not in records
    assert _track("MT0-isolation-profile") not in records


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
        ("package_internal", True): 1489,
        ("package_internal", False): 1,
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

    assert _track("MT4") not in records
    assert _track("MT1b") not in records


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
    verifiers = [
        path for path in (ROOT / GOLDEN_SHARED).glob("**/verify.py")
        if "acceptance-epochs" not in path.parts
    ]
    assert len(verifiers) == 1
    verifiers.append(ROOT / "docs/架构/清微NPU数据模拟/verify.py")
    assert all("live_anchors" not in path.read_text(encoding="utf-8") for path in verifiers)


# Byte copies of evidence that live tests still parse. The digest is the lock.
# A missing file fails the read; existence alone is not the assertion.
PORTFOLIO_FIXTURE_SHA256 = (
    ("tests/support/golden/shared/acceptance-epochs/fixtures/p2/evidence/unseen-confirmations.json", "cdae2c9e5964a8415fd5d316111027b8c22a33001a9dc1957509668a49b6167c"),
    ("tests/support/golden/shared/acceptance-epochs/fixtures/p20/golden.json", "70ccf5d34e788ee1d74213c216ae758c2d9ca6a9e38893ea4c21f32f4285d86f"),
    ("tests/support/golden/shared/acceptance-epochs/fixtures/p2i/golden.json", "901d2a0194da7b7d80bf6e30002da19c6722fe370e6d9ae2124fe4998a638fb8"),
    ("tests/support/golden/shared/acceptance-epochs/fixtures/p2i/evidence/unseen-confirmations.json", "eb0cbd62d916d6d9d755ad84d5741418eb25d9dfe67264bdfbc78a8d4bf68bbd"),
    ("tests/support/golden/shared/acceptance-epochs/fixtures/p2i/evidence/tune-confirmations.json", "6f0ea66c32b38d956e86016e45d0a8533b29a639dc027c6a9f58ca174dbdd271"),
    ("tests/support/golden/shared/acceptance-epochs/fixtures/p2i/evidence/conv2d-portfolio-small.json", "61746ced5ea6fafea8f59b2b7cbda48db807606d964b7a609c9773c2ba4bbd9d"),
    ("tests/support/golden/shared/acceptance-epochs/fixtures/p2i/evidence/conv2d-portfolio-paper.json", "5e9cf74f0f4e6579b918bc020968592819e7ef0f09668ed8c6362cc082552f4a"),
    ("tests/support/golden/shared/acceptance-epochs/fixtures/p2i/evidence/conv2d-ablation.json", "ef74eb6b9d8c84664e8ccb4686afb95fbaa9951f8c70044acec208b33fe901af"),
    ("tests/support/golden/shared/acceptance-epochs/fixtures/p2i/evidence/conv2d-mma-f4-confirmation.json", "ba0483857eb5cf0ad9c17bd609739f48600db45a6a3d7d0fbd25e8a1f8a55814"),
    ("tests/support/golden/shared/acceptance-epochs/fixtures/p2j/evidence/launch-floor.json", "42d1255dfeba5def1e18951adcb2246d2b4769b57d36e3cfff0a93d0621355ad"),
    ("tests/support/golden/shared/acceptance-epochs/fixtures/p2j/evidence/small-run.json", "d58ab5fbe0422534151c636d60867740d78adec39cbd728f7d24120d1b6f84fc"),
    ("tests/support/golden/shared/acceptance-epochs/fixtures/p2j/evidence/paper-run.json", "f74d12e566cd0046d0a73b0ce1eebf5319e02c5d25fbd7ac778ada4b30d4c548"),
    ("tests/support/golden/shared/acceptance-epochs/fixtures/p2j/evidence/small-confirmation.json", "595fc4321f3f75fd4fd165ac28c0cbb20336f400741872496b0fd0b102bf4af8"),
    ("tests/support/golden/shared/acceptance-epochs/fixtures/p2j/evidence/headroom.json", "296021e3d9ddfb13a5e45d0d5a8df84ff6c67cfeee9b00a520206339fe0c9578"),
    ("tests/support/golden/shared/acceptance-epochs/fixtures/p2k/evidence/fresh-suite.json", "b77ddee21190c998c566d97816d7c1187a7777478d00cef4e31f27f14c326546"),
    ("tests/support/golden/shared/acceptance-epochs/fixtures/p2k/evidence/validated-manifest.json", "982a0867aa0a178c8f9b135cda404594d45fe4f798722e6f91da2a1b6904ba55"),
    ("tests/support/golden/shared/acceptance-epochs/fixtures/p2k/evidence/packed-half-probe.json", "522d40cab176d95f76036ee838deaf1a39ff578ad72902fcaa8d13926c1fba0d"),
    ("tests/support/golden/shared/acceptance-epochs/fixtures/p2l/evidence/launch-floor.json", "1e792f82f8617390a1c168a26032eee1ac4bc855c3e293088a7a077b952720fb"),
    ("tests/support/golden/shared/acceptance-epochs/fixtures/p2l/evidence/analysis.json", "de14fe5c085792de67da60e2d210175f7aca61c8f9bbffbffe875e8ee44adb3b"),
    ("tests/support/golden/shared/acceptance-epochs/fixtures/p2n/evidence/decision.json", "b9d9b39cd16b6bd0f57de32b339dbf795ce4b003260e07d131a748a892cb5d69"),
    ("tests/support/golden/shared/acceptance-epochs/fixtures/p2n/evidence/weight-cache.metal", "62e1ac92b7bbc15f936d819e3a41da9aca979cadf88a2af265430351b76651eb"),
    ("tests/support/golden/shared/acceptance-epochs/fixtures/p2n/evidence/weight-cache-report.json", "e1ba9312dc65afe1d58ba39a4f3d50c6034622bf7d8be1d4adf1b43cef8a478b"),
    ("tests/support/golden/shared/acceptance-epochs/fixtures/p2q/Q.md", "54985349db51ac44a2577d7055292780c8ba45ca69d48140ea80223628bbdeaa"),
    ("tests/support/golden/shared/acceptance-epochs/fixtures/p2q/evidence/confirmations.json", "96fd0d700871e24429583fdefc5ca85a89b2fb3818f30e535157f420ab6600cb"),
    ("tests/support/golden/shared/acceptance-epochs/fixtures/p2q/evidence/decision.json", "9a6d1d0b50176db1ce3f68a5d92cc77d6a8a89a66f0eab643d6e331e33a9f6ba"),
    ("tests/support/golden/shared/acceptance-epochs/fixtures/p2q/evidence/seq-handoff-measurements.json", "d7e30af88a4b7f67f2d378a111b8371e4b3424407f85ced0713ac8e08355df46"),
    ("tests/support/golden/shared/acceptance-epochs/fixtures/p2q/evidence/seq-handoff-probe.json", "28c22547f5c7440967586d2170684c9fada1d289bb31bcbd95a4d5c7abd02cff"),
    ("tests/support/golden/shared/acceptance-epochs/fixtures/p2q/evidence/seq-handoff-scratch-attempt.json", "3a1dacd40e7456634d6c3f9fdd60619cc55e4deb5d2a84af5e3a5cfa1eb9d520"),
    ("tests/support/golden/shared/acceptance-epochs/fixtures/p2s/S.md", "5c6043e9a398fb7622a3d59e0a7aaaea04bc1f497db681c9b1d65f343ee5c114"),
    ("tests/support/golden/shared/acceptance-epochs/fixtures/p2s/evidence/experience-overlay.md", "8e3b9f975cb3988e03af09a12e131bfdb8a699631c8780d544d3836f4bdb97cc"),
    ("tests/support/golden/shared/acceptance-epochs/fixtures/p2s/evidence/prompt-boundary.json", "670e3195ebc8fcacf410fc8f9f6ba9357ac8f8679efa4f3126086b3472407684"),
    ("tests/support/golden/shared/acceptance-epochs/fixtures/p2s/evidence/dry-run-plan.json", "8d0178d66d94a7c3444d28732e5cf4ff6420b0c7114cc732aafaf4af2b31f496"),
    ("tests/support/golden/shared/acceptance-epochs/fixtures/p2s/evidence/catalog.json", "c498d195333350a1da2d4f2d2986b3c0847a13251186ef208a4690d126633c50"),
    ("tests/support/golden/shared/acceptance-epochs/fixtures/p2s/evidence/approval.json", "e94cea0e0cd943d025c8e04fd36bc28981bd24229846e27c60301da1f8d2316a"),
    ("tests/support/golden/shared/acceptance-epochs/fixtures/p2v/evidence/compact-overlay.md", "fd72abf078d9d5a7cd2118ee5263a3c83e5695c044c0bc0cc6ca86758a0aa3d1"),
    ("tests/support/golden/shared/acceptance-epochs/fixtures/p2v/evidence/dry-run-plan.json", "d3781f05f0a5647cdb58b57582c83cbd1a6e567275e562f36bb24325c9e5525e"),
    ("tests/support/golden/shared/acceptance-epochs/fixtures/p4/evidence/simulation.json", "c9c7f32d692df4392946beb83d414da8362dbfce88817cf628890c32e46b870f"),
)


def test_portfolio_fixture_copies_keep_the_source_bytes():
    """Each copied evidence file is the original bytes, identified by path and SHA-256."""
    assert len(PORTFOLIO_FIXTURE_SHA256) == 37
    for relative, digest in PORTFOLIO_FIXTURE_SHA256:
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == digest, relative


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
    "tests/support/golden/shared/shape-portfolio-阶段验收/P2",
    "tests/support/golden/shared/shape-portfolio-阶段验收/P2-0",
    "tests/support/golden/shared/shape-portfolio-阶段验收/P2-I",
    "tests/support/golden/shared/shape-portfolio-阶段验收/P2-J",
    "tests/support/golden/shared/shape-portfolio-阶段验收/P2-K",
    "tests/support/golden/shared/shape-portfolio-阶段验收/P2-L",
    "tests/support/golden/shared/shape-portfolio-阶段验收/P2-N",
    "tests/support/golden/shared/shape-portfolio-阶段验收/P2-Q",
    "tests/support/golden/shared/shape-portfolio-阶段验收/P2-S",
    "tests/support/golden/shared/shape-portfolio-阶段验收/P2-V",
    "tests/support/golden/shared/shape-portfolio-阶段验收/P4",
    "tests/support/golden/shared/多DSL-Track-阶段验收/MT0",
    "tests/support/golden/shared/多DSL-Track-阶段验收/MT0-isolation",
    "tests/support/golden/shared/多DSL-Track-阶段验收/MT0-isolation-profile",
    "tests/support/golden/shared/多DSL-Track-阶段验收/MT0-tilelang-probe",
    "tests/support/golden/shared/多DSL-Track-阶段验收/MT1a",
    "tests/support/golden/shared/多DSL-Track-阶段验收/MT1b",
    "tests/support/golden/shared/多DSL-Track-阶段验收/MT2",
    "tests/support/golden/shared/多DSL-Track-阶段验收/MT4",
    "tests/support/golden/shared/多DSL-Track-阶段验收/MT6",
    "tests/support/golden/shared/多DSL-Track-阶段验收/MT7",
    "tests/support/golden/shared/跨硬件算子生成分工-阶段验收/A5",
    "tests/support/golden/shared/跨硬件算子生成分工-阶段验收/B1",
    "tests/support/golden/shared/跨硬件算子生成分工-阶段验收/B2",
    "tests/support/golden/shared/跨硬件算子生成分工-阶段验收/B3",
    "tests/support/golden/shared/跨硬件算子生成分工-阶段验收/B4",
)


def test_retirement_registry_matches_git_history_without_restoring():
    """Recoverability is the recorded tree object. This test does not check the files back out."""
    document = load_retired(ROOT)
    assert "limits" not in document
    assert [row["path"] for row in document["packages"]] == list(RETIRED_PATHS)
    assert {row["change"] for row in document["packages"]} == {
        "20261115-retire-closed-packages",
        "20261115-retire-remaining-portfolio",
        "20261115-retire-mt-a-b",
    }
    assert {row["approved_by"] for row in document["packages"]} == {
        "用户（2026-11-15 授权修改闸门并移出工作区）",
        "用户（2026-11-15 授权移出剩余 shape-portfolio 验收包）",
        "用户（2026-11-15 授权移出 MT、A、B 验收包）",
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


def test_retired_restore_commands_name_trees_that_exist():
    """Each restore command names a tree that is still in git. This does not check the files out."""
    git = GitObjects(ROOT)
    for row in load_retired(ROOT)["packages"]:
        assert row["restore"] == f"git checkout {row['last_commit']} -- {row['path']}"
        assert git.path_oid(row["last_commit"], row["path"]) == row["tree"]
        kind, _body = git.read(row["tree"])
        assert kind == "tree"


def test_retired_checkout_commands_name_complete_trees():
    """git checkout of the recorded tree would restore every file. This does not check them out."""
    git = GitObjects(ROOT)
    for row in load_retired(ROOT)["packages"]:
        assert row["restore"] == f"git checkout {row['last_commit']} -- {row['path']}"
        assert _tree_size(git, row["tree"]) == (row["files"], row["bytes"])
        assert row["files"] > 0


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
    recorded = next(row for row in load_retired(ROOT)["packages"] if row["path"] == path)
    commit = recorded["last_commit"]
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
    # HEAD after this retirement no longer contains the parent directory.
    # The commit that still has both trees is the P2 row's last_commit.
    commit = next(
        row["last_commit"] for row in load_retired(ROOT)["packages"] if row["path"] == _phase("P2")
    )
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


# Byte copies of MT/A/B evidence that live tests still parse. The digest is the lock.
# A missing file fails the read; existence alone is not the assertion.
MT_AB_FIXTURE_SHA256 = (
    ("tests/support/golden/shared/acceptance-epochs/fixtures/a5/evidence/perf_gate.json", "0b01b22eb740d2dea32d9e3c98d74ca0adfab406cf4eb74a9c5fff0664119f4f"),
    ("tests/support/golden/shared/acceptance-epochs/fixtures/a5/evidence/perf_v1_paper.json", "a9947dbf9f4439bf0b086c5bfe5e3571ca14b764431e94143cfb922c9a6da400"),
    ("tests/support/golden/shared/acceptance-epochs/fixtures/a5/evidence/perf_v1_paper_r2.json", "32518a062aee0c3c7fd5317d5ecbdf59a249f380e815c99ba24f5de32ecd1f4d"),
    ("tests/support/golden/shared/acceptance-epochs/fixtures/a5/evidence/perf_v1_paper_r3.json", "f35fc79d7fe9e0c23791a311acdac9390560970aee0a6e2554c0562f3bd0144a"),
    ("tests/support/golden/shared/acceptance-epochs/fixtures/a5/evidence/perf_v1_small.json", "38ccf7234ac6897131a143bac692f8063199b5b9c50c93aa3a7ac8436ebea17d"),
    ("tests/support/golden/shared/acceptance-epochs/fixtures/a5/evidence/perf_v1_small_r2.json", "60e1f4b8e18631b9cc120989eeb64af0d8db5081f754c177bd7aede34e6fe276"),
    ("tests/support/golden/shared/acceptance-epochs/fixtures/a5/evidence/perf_v1_small_r3.json", "5bb2ad72a1bcfe2bce0a6ee9f29d1dc710a0300308b773ad0303ef5c8b7ad6e9"),
    ("tests/support/golden/shared/acceptance-epochs/fixtures/a5/evidence/perf_v2_paper.json", "79b1a13610815fd224265b6db8f619dc52a0c19219ec045cdd06e13f7a55a683"),
    ("tests/support/golden/shared/acceptance-epochs/fixtures/a5/evidence/perf_v2_paper_r2.json", "1ab28349d491b0b661be347c483728f457acb8393d99302c4fa67a396bad4300"),
    ("tests/support/golden/shared/acceptance-epochs/fixtures/a5/evidence/perf_v2_paper_r3.json", "b8396000ae51b306390abce400b64d9170f3b8d70eb7170fc8b8fea3fb5ab9e5"),
    ("tests/support/golden/shared/acceptance-epochs/fixtures/a5/evidence/perf_v2_small.json", "40df6a812c70648c5b05e3fd70f8d29572ff076190b21b2ddbe5e2f4b8bff3ed"),
    ("tests/support/golden/shared/acceptance-epochs/fixtures/a5/evidence/perf_v2_small_r2.json", "2194217f20bee6e11076747497c25cba136c571d80772614cedd0f2a20cb20d0"),
    ("tests/support/golden/shared/acceptance-epochs/fixtures/a5/evidence/perf_v2_small_r3.json", "3a07254e43d9c2da5b955924005bff3544a0b5672709ca747767f29f56818447"),
    ("tests/support/golden/shared/acceptance-epochs/fixtures/a5/evidence/slices/gemm.json", "dc3f3b0024e23a58cd1ac01899c0359c0a71e67168a767cc9e955acf84cf1deb"),
    ("tests/support/golden/shared/acceptance-epochs/fixtures/a5/evidence/slices/gen_swiglu.json", "2df472fa9b64e00f9419cd411cef56845e5de3f5cf094c186c539afc8177b662"),
    ("tests/support/golden/shared/acceptance-epochs/fixtures/a5/evidence/slices/reduction_sum.json", "1781530e7bdb94cf0ebd2745867f9ff44b5937aa038e927d72156e21716dc159"),
    ("tests/support/golden/shared/acceptance-epochs/fixtures/a5/evidence/slices/rmsnorm.json", "d960f0354f3ebe8c3c300eeeacee950dba583fd4fb1ae4780867fedd75e93b4d"),
    ("tests/support/golden/shared/acceptance-epochs/fixtures/a5/evidence/slices/sem_gelu_named_relu.json", "767bf92107e2d78647f90a8061ab6886b4662024d7f8ada2c64f7378abca5138"),
    ("tests/support/golden/shared/acceptance-epochs/fixtures/a5/evidence/slices/sigmoid.json", "72ca58b8e7d614a5dc2276605e97e1fff521a0db67a854ac1911c2b16da77a11"),
    ("tests/support/golden/shared/acceptance-epochs/fixtures/a5/manifest.sha256", "b0668b8786952d1171fa51ae8d90fa0d50678c9ec87d484578a9a128a146d406"),
    ("tests/support/golden/shared/acceptance-epochs/fixtures/b1/boundary.json", "2a91ccf757077347e5d0abe832c865388a1c3e74d13baabd759403fe196da7d7"),
    ("tests/support/golden/shared/acceptance-epochs/fixtures/b1/evidence/candidates/gemv-naive.msl", "9ca25d1127ad6b5b6785937479d9243b9aec957b49d1df8e7b1f1d2ada3c165d"),
    ("tests/support/golden/shared/acceptance-epochs/fixtures/b1/evidence/candidates/rope-naive.msl", "e035b940d5ca800dacd55efcea75d08806da14031a6ce0aa5826e730c1f19ab6"),
    ("tests/support/golden/shared/acceptance-epochs/fixtures/b1/evidence/v1-frozen.json", "824756c98fa5d8884b6f67eb44de0b3ea7e65320057b4d48f0bf45988a5e7f0d"),
    ("tests/support/golden/shared/acceptance-epochs/fixtures/b2/boundary.json", "617041c32821bd313b4c14cb8414f412a3b8a61d0dd0ab516e6da8e984376b2a"),
    ("tests/support/golden/shared/acceptance-epochs/fixtures/b2/evidence/candidates/conv2d-naive.msl", "9e2d6e94949cae0d05de4deee9b66e5b462007372c74e780f97c8df7c3550754"),
    ("tests/support/golden/shared/acceptance-epochs/fixtures/b2/evidence/candidates/lion-naive.msl", "54db3405d9f1b6cc572652abc9169ecacf405b13ae50893f8ff7ff9de35f306c"),
    ("tests/support/golden/shared/acceptance-epochs/fixtures/b2/evidence/v1-frozen.json", "b6ae95d7d17fb5fa72061d0afac52d0ae2c217d15cf50c384a1338be8c07cd13"),
    ("tests/support/golden/shared/acceptance-epochs/fixtures/b3/boundary.json", "e0ee3a4c916f43bac770a53da7d29a0f5b2b280f444488e8148fcff943a1df5d"),
    ("tests/support/golden/shared/acceptance-epochs/fixtures/b3/evidence/candidates/bitdelta_gemm-naive.msl", "55915f28881e202159c8df30643105e8ce7f1e181f11af4982db017b34a4695a"),
    ("tests/support/golden/shared/acceptance-epochs/fixtures/b3/evidence/candidates/flash_attention-naive.msl", "2e7e307f98b3dfb20813539532b17a00f774d8fcb11a1aee1ff588cafc02fcfa"),
    ("tests/support/golden/shared/acceptance-epochs/fixtures/b3/evidence/candidates/mamba2_chunk_scan-naive.msl", "a74c1455aa12cf6b6ff6aa046a7a984b36926da284b521464e90f5d82d57232d"),
    ("tests/support/golden/shared/acceptance-epochs/fixtures/b3/evidence/v1-frozen.json", "2b8ee7fd4463ccdb138e309a99d8518dfef4762da6b4ce56985e41f5d38b67a5"),
    ("tests/support/golden/shared/acceptance-epochs/fixtures/b4/boundary.json", "055876d4cdafdf7cf58244831de787ab37014c5013a4fab161a1666fe3ad2436"),
    ("tests/support/golden/shared/acceptance-epochs/fixtures/b4/evidence/candidates/sem_attention_odd_scale-naive.msl", "50cf5c6c5aaea7b267e094973eff39537463250cfd4cc3e579dad84fe976bbb3"),
    ("tests/support/golden/shared/acceptance-epochs/fixtures/b4/evidence/candidates/sem_int_007-naive.msl", "f2a86fc8b1d10652952397d7d982a378e59c47525a28eb87d9241282c8ac1b12"),
    ("tests/support/golden/shared/acceptance-epochs/fixtures/b4/evidence/candidates/sem_int_1-naive.msl", "d1db4d51a4bfebd95f3a2b7d1d376dcdd5028d66dd486ce4f63739f7409f022e"),
    ("tests/support/golden/shared/acceptance-epochs/fixtures/b4/evidence/candidates/sem_int_2-naive.msl", "1e5c59e96bc8b5804e6c73d59e66d4ab318c0225ac9b2481f454bd2ac8e1d158"),
    ("tests/support/golden/shared/acceptance-epochs/fixtures/b4/evidence/candidates/sem_int_3-naive.msl", "6c0c1f3324e62bf80260d0017b4c3c138fc0c8a9b97260fd928f809648cc8d02"),
    ("tests/support/golden/shared/acceptance-epochs/fixtures/b4/evidence/candidates/sem_int_4-naive.msl", "d8e9b317c118770bd2b7f19516b1782dc154009ac0536b2796810420cd2d37c9"),
    ("tests/support/golden/shared/acceptance-epochs/fixtures/b4/evidence/candidates/sem_int_5-naive.msl", "372e9c95902dd3a0e14c6e2f4006c87d32056871f7b308db31b7f1c09ebe6f60"),
    ("tests/support/golden/shared/acceptance-epochs/fixtures/b4/evidence/sem_int-frozen.json", "c8472c5803e6375a87f9fd4468d7bd2071d721e069ccd0b0295034fba6e88e09"),
    ("tests/support/golden/shared/acceptance-epochs/fixtures/b4/evidence/v1-frozen.json", "a1ae204a232ea3514a363512070d1d5e205c9c7788baeb2b3d5b7f24e0bfd434"),
    ("tests/support/golden/shared/acceptance-epochs/fixtures/hw-stage/A5.md", "f08e239c86c44a92b69f3e16e5d32d355ff871be8b94bbb8372ee59e037f172d"),
    ("tests/support/golden/shared/acceptance-epochs/fixtures/hw-stage/G0.md", "cce9fb17f87828d0c6176e1d5c06b801d50e6c1690acadf6631cd04bdde1b883"),
    ("tests/support/golden/shared/acceptance-epochs/fixtures/mt0-isolation-profile/acceptance.md", "a2d30de4c250fece378238044e38ea07f0a1cc28dd4a99b5c664f9a7f68680d1"),
    ("tests/support/golden/shared/acceptance-epochs/fixtures/mt0-isolation-profile/cpu-shim-watchdog.json", "a852ff31696b1c0b2eba3dc8287530bfe5a629b4b2e630f6f7e76e9acacf17a9"),
    ("tests/support/golden/shared/acceptance-epochs/fixtures/mt0-isolation-profile/manifest.sha256", "c3b52f630fa27ac76cfbfce0303a334fba11d4bc685573b361fbd29313d96a6d"),
    ("tests/support/golden/shared/acceptance-epochs/fixtures/mt0-isolation-profile/nodes.json", "a4c425008dcc56625ca0818d82c87b188716ccfb187c6f65f66cc992d8ac3e8d"),
    ("tests/support/golden/shared/acceptance-epochs/fixtures/mt0-isolation-profile/outer-probe-result-r2.json", "69f21a5da32a4051396adca94702195533e716bf071c1d117e5da53b0d164229"),
    ("tests/support/golden/shared/acceptance-epochs/fixtures/mt0-isolation-profile/outer-probe-result.json", "a8c50f471d04ac3418cdf83c0441cdab4d2a781fae9197c607ad7c349f0cf88d"),
    ("tests/support/golden/shared/acceptance-epochs/fixtures/mt0-isolation-profile/outer-probe-review.json", "7105d5e51e6eb2bb40ce02053bf776d92c23d765353719803a2f20128c4182d8"),
    ("tests/support/golden/shared/acceptance-epochs/fixtures/mt0-isolation-profile/outer-probe.json", "4004da719797eda9ed841f5e036a2b481caf7ed7d27073d7e9b5d73332fdc335"),
    ("tests/support/golden/shared/acceptance-epochs/fixtures/mt0-isolation-profile/outer_probe.py", "5a7f63ea24d06c1b4e17d114ce70a8836ea0b426ee4106fe15f0f72bdf365ef6"),
    ("tests/support/golden/shared/acceptance-epochs/fixtures/mt0-isolation-profile/source-identity.json", "027c68011747cf3ce5cbaeae1469f24c0aa0af1fa6cc36eb0317fb05dc02ae09"),
    ("tests/support/golden/shared/acceptance-epochs/fixtures/mt0-isolation-profile/start-state.json", "eafc09747ed2ec9f6426175e8a6d27c36cdeee53d5fa4b0db1452d516e05219c"),
    ("tests/support/golden/shared/acceptance-epochs/fixtures/mt0-isolation-profile/start-worktree.sha256", "1f3311122132b5ca89de375bdd1f3908b165ae8a0d231e3a6e4aaba8d957cd8a"),
    ("tests/support/golden/shared/acceptance-epochs/fixtures/mt0-isolation-profile/static-contract.json", "81974402490f6e8b5a3442fd03e21b84d84489f75ee28af664686c86f43b542b"),
    ("tests/support/golden/shared/acceptance-epochs/fixtures/mt0-isolation-profile/verification.json", "139cbcddb8dec7d482ccfaa0a079263534a7aa51725f14a105d782bccc7d9df7"),
    ("tests/support/golden/shared/acceptance-epochs/fixtures/mt0-isolation/acceptance.md", "45fda16a5fe8b917549f80855bc495a88b4daf429015ce326297b2867b951859"),
    ("tests/support/golden/shared/acceptance-epochs/fixtures/mt0-isolation/hmz-installation.json", "9de3a5004d7b10cf6429660c2c8060cc287aa4416edea25231828202193202cf"),
    ("tests/support/golden/shared/acceptance-epochs/fixtures/mt0-isolation/manifest.sha256", "aa23a2662dd6c61f0ae2a8f3f4d01ad98142dbf4f73d4be2a7fd385311696309"),
    ("tests/support/golden/shared/acceptance-epochs/fixtures/mt0-isolation/nodes.json", "4ad4d0116b58b2d63656b19ea699e5a55359d3d666a04d94addb4c71861c673c"),
    ("tests/support/golden/shared/acceptance-epochs/fixtures/mt0-isolation/outer-probe.json", "595cc559e544e3bec0df9255c93e52e128730b5151c27e3ab3e15a834263dd73"),
    ("tests/support/golden/shared/acceptance-epochs/fixtures/mt0-isolation/outer_probe.py", "7c36cddc62802c8c63659c791b558a513120b6cc6d6d4bf5c7c067bb3625fcba"),
    ("tests/support/golden/shared/acceptance-epochs/fixtures/mt0-isolation/start-state.json", "76d560acc07ced6bbc4fff4dcba3e227dbaca38f0056c39cea01994765689a52"),
    ("tests/support/golden/shared/acceptance-epochs/fixtures/mt0-isolation/start-worktree.sha256", "6540b23b46d1d5eb73b743e6f4a350b6047ae9b4fba47c769ffa7ab33b2f5c58"),
    ("tests/support/golden/shared/acceptance-epochs/fixtures/mt0-isolation/static-contract.json", "5cc17d248a13e348e448af592e287e8d96824ece21f967ce8b3b821be711352e"),
    ("tests/support/golden/shared/acceptance-epochs/fixtures/mt0-isolation/verification.json", "788552bcfa94db517ba926ba1b7746103a5a14fc4d584dc987e5209b597e97aa"),
    ("tests/support/golden/shared/acceptance-epochs/fixtures/mt0-tilelang-probe/acceptance.md", "d007f45b2a853864b8d93472c59cffcaa08bac297919388b2e6716aae0ae739b"),
    ("tests/support/golden/shared/acceptance-epochs/fixtures/mt0-tilelang-probe/context-capability.json", "82dcebbba48835e5a36a3edae8ebe2e5cca3947340168ea2ddffbff1c0d49771"),
    ("tests/support/golden/shared/acceptance-epochs/fixtures/mt0-tilelang-probe/evidence/failure-inputs/r1-candidate.py", "1c839a8156c8939546613627aac2d46054fa253c1c6f38bc2b4f59649884efa4"),
    ("tests/support/golden/shared/acceptance-epochs/fixtures/mt0-tilelang-probe/evidence/failure-inputs/r1-frontend_entry.py", "7782cf19c47c08fea1c6411b91260df1ae7dfbdba8e69d3003513f1e35545cf6"),
    ("tests/support/golden/shared/acceptance-epochs/fixtures/mt0-tilelang-probe/evidence/failure-inputs/r2-candidate.py", "1c839a8156c8939546613627aac2d46054fa253c1c6f38bc2b4f59649884efa4"),
    ("tests/support/golden/shared/acceptance-epochs/fixtures/mt0-tilelang-probe/evidence/failure-inputs/r2-frontend_entry.py", "2422bb6ed15c7823a7f5ba95486ede71daa76752634848121e04fda3750beecf"),
    ("tests/support/golden/shared/acceptance-epochs/fixtures/mt0-tilelang-probe/evidence/failure-inputs/r3-candidate.py", "1c839a8156c8939546613627aac2d46054fa253c1c6f38bc2b4f59649884efa4"),
    ("tests/support/golden/shared/acceptance-epochs/fixtures/mt0-tilelang-probe/evidence/failure-inputs/r3-frontend_entry.py", "d2546349d52170ae2d5f4c0142232906caa12a4edb591cbb5753ca92fa96adca"),
    ("tests/support/golden/shared/acceptance-epochs/fixtures/mt0-tilelang-probe/evidence/failure-inputs/r4-candidate.py", "3ebead1927e3fe4885b98151ee11f92ddd2e8e13314e9d0a530fc11506e32a8c"),
    ("tests/support/golden/shared/acceptance-epochs/fixtures/mt0-tilelang-probe/evidence/failure-inputs/r4-frontend_entry.py", "d2546349d52170ae2d5f4c0142232906caa12a4edb591cbb5753ca92fa96adca"),
    ("tests/support/golden/shared/acceptance-epochs/fixtures/mt0-tilelang-probe/evidence/failure-inputs/r5-candidate.py", "3ebead1927e3fe4885b98151ee11f92ddd2e8e13314e9d0a530fc11506e32a8c"),
    ("tests/support/golden/shared/acceptance-epochs/fixtures/mt0-tilelang-probe/evidence/failure-inputs/r5-frontend_entry.py", "f953a0e84174a617e9423555dfa9490f6ed1b1943ad735d12a09aa4cf1931940"),
    ("tests/support/golden/shared/acceptance-epochs/fixtures/mt0-tilelang-probe/evidence/failure-logs/r1-stderr.txt", "b44bdb303099aa0bd40ce4a2c94507264f4fa69579d1238ffc998d248da4fb85"),
    ("tests/support/golden/shared/acceptance-epochs/fixtures/mt0-tilelang-probe/evidence/failure-logs/r2-stderr.txt", "b44bdb303099aa0bd40ce4a2c94507264f4fa69579d1238ffc998d248da4fb85"),
    ("tests/support/golden/shared/acceptance-epochs/fixtures/mt0-tilelang-probe/evidence/failure-logs/r3-stderr.txt", "a828a1afc3978756c1012ef7d94b5be7e65f668beefd5248913ebe52a33574e1"),
    ("tests/support/golden/shared/acceptance-epochs/fixtures/mt0-tilelang-probe/evidence/failure-logs/r4-stderr.txt", "bd0ff6c72e6e65b6ed8626802285d399fb9c99b4c14a147035f3e8a6d2c6ddfe"),
    ("tests/support/golden/shared/acceptance-epochs/fixtures/mt0-tilelang-probe/evidence/source-only-r5/build_identity.json", "e0d3b94c919ce68f7b173f47b6384bf7663079cee21ecc5e0730b48a71cafab5"),
    ("tests/support/golden/shared/acceptance-epochs/fixtures/mt0-tilelang-probe/evidence/source-only-r5/device_ir.json", "16b53c656ca94b6506a97e5898fee0ca4c7f2af0087fcf757a50d0748b94c3e0"),
    ("tests/support/golden/shared/acceptance-epochs/fixtures/mt0-tilelang-probe/evidence/source-only-r5/frontend_identity.json", "f31474453cd08dd58fa557568a5b8cfa3cc13caadead791427df1deec07c8ed7"),
    ("tests/support/golden/shared/acceptance-epochs/fixtures/mt0-tilelang-probe/evidence/source-only-r5/host_module.txt", "a3152a4b5d5bcd89461f36ef2ede118d64c86cafa8494fe6af4ba632ffc5792e"),
    ("tests/support/golden/shared/acceptance-epochs/fixtures/mt0-tilelang-probe/evidence/source-only-r5/kernel.metal", "a53f916fac4806462ab541ba9d06b9ad8a03415a58f3f270c984b483dfcf1ca8"),
    ("tests/support/golden/shared/acceptance-epochs/fixtures/mt0-tilelang-probe/evidence/source-only-r5/kernel_params.json", "51e90320a5f7057ca3f93fe61fbe4fbae711524f10d7820e608f5a66de19f58f"),
    ("tests/support/golden/shared/acceptance-epochs/fixtures/mt0-tilelang-probe/frontend-install.json", "910d3f8f862ec70c1cd7b18e27d067075d5b3c0918fc4b0c4c308f3ca0367aed"),
    ("tests/support/golden/shared/acceptance-epochs/fixtures/mt0-tilelang-probe/history-anchor-drift.json", "023a73f89e70ef5bd188a4ac83658f898c2e13648b5edcb354864d5b3d24fb27"),
    ("tests/support/golden/shared/acceptance-epochs/fixtures/mt0-tilelang-probe/manifest.sha256", "b112150f9279cd43e53b93eed87032f00a024403915fa1eb8ecd83920169bafc"),
    ("tests/support/golden/shared/acceptance-epochs/fixtures/mt0-tilelang-probe/negative-artifact.json", "d3a96856abed5b3409267697440c3bc1d164c28117e16f6948ee8d98cce703b4"),
    ("tests/support/golden/shared/acceptance-epochs/fixtures/mt0-tilelang-probe/negative-network.json", "6644776ddd30011ced81b8897fad4e8254e26e09ac02e58114027037652f5ac9"),
    ("tests/support/golden/shared/acceptance-epochs/fixtures/mt0-tilelang-probe/negative-read-outside.json", "fa71e934bab362bef097c88ec2bff9770ab99104d9dc35ffa20b3af798e51ffd"),
    ("tests/support/golden/shared/acceptance-epochs/fixtures/mt0-tilelang-probe/negative-write-outside.json", "d6415227e1a9627630a2c646eee2db3d6bc7a85856ed4d9a2bf7601973a7bfdb"),
    ("tests/support/golden/shared/acceptance-epochs/fixtures/mt0-tilelang-probe/nodes.json", "7bf5b2a7605ca878f36fd00371a8bceeb3412e25898a9921f82fca5daaa30409"),
    ("tests/support/golden/shared/acceptance-epochs/fixtures/mt0-tilelang-probe/probe-history.json", "8700675074f689287dc9cad215fa2b04a48f6cf398c8d4454bf9e9d9cd36daa4"),
    ("tests/support/golden/shared/acceptance-epochs/fixtures/mt0-tilelang-probe/probe-plan.json", "81cff9173b65c19b04f9c2674e1067453017a2240e56f4dc4f801f0996dabced"),
    ("tests/support/golden/shared/acceptance-epochs/fixtures/mt0-tilelang-probe/probe-result-r2.json", "b1fe9fb99ceb45cb4744ca40ad3d6136ae8153f5d08c9f929b13a7e21b632504"),
    ("tests/support/golden/shared/acceptance-epochs/fixtures/mt0-tilelang-probe/probe-result-r3.json", "70279b2ff192201740da1f4578a23074a36e24b13d8687924c44691a0fe6f1be"),
    ("tests/support/golden/shared/acceptance-epochs/fixtures/mt0-tilelang-probe/probe-result-r4.json", "4583ce598251977dae5842757e2d5565dd975f75a0450395b7ea3e07375ca2a8"),
    ("tests/support/golden/shared/acceptance-epochs/fixtures/mt0-tilelang-probe/probe-result-r5.json", "243871eb2328a485777c7580e83995b61e18d04b14f31309df95dee898e3381a"),
    ("tests/support/golden/shared/acceptance-epochs/fixtures/mt0-tilelang-probe/probe-result.json", "3a8f00a3356cc1a755d87707f51fa069e687a9cd75874fce6b81007f1282f857"),
    ("tests/support/golden/shared/acceptance-epochs/fixtures/mt0-tilelang-probe/probe-status.json", "14539f3bbb595432a291cad134e6441f734475c92248cd913515f984c6968dc8"),
    ("tests/support/golden/shared/acceptance-epochs/fixtures/mt0-tilelang-probe/probe/candidate.py", "3ebead1927e3fe4885b98151ee11f92ddd2e8e13314e9d0a530fc11506e32a8c"),
    ("tests/support/golden/shared/acceptance-epochs/fixtures/mt0-tilelang-probe/probe/fixtures/network_connect.py", "fb50c0fd62cf3966a910f1c55a62dbc0ee3c5dd7ab3dcfa237c430f9f866b81d"),
    ("tests/support/golden/shared/acceptance-epochs/fixtures/mt0-tilelang-probe/probe/fixtures/read_outside.py", "b69dbe17aa905423a73695ee33eb669f61f2cf955e37fe6c4e2310daa6678c98"),
    ("tests/support/golden/shared/acceptance-epochs/fixtures/mt0-tilelang-probe/probe/fixtures/undeclared_artifact.py", "2964f48946d6e9a66c69a6292ef5a20bafc549a5427eee3e8077784f7eb54121"),
    ("tests/support/golden/shared/acceptance-epochs/fixtures/mt0-tilelang-probe/probe/fixtures/write_outside.py", "dc2b80a47103b1fbab6495fceee75f494a5f94fbd44f0807a4c6ebe726256955"),
    ("tests/support/golden/shared/acceptance-epochs/fixtures/mt0-tilelang-probe/probe/frontend_entry.py", "f953a0e84174a617e9423555dfa9490f6ed1b1943ad735d12a09aa4cf1931940"),
    ("tests/support/golden/shared/acceptance-epochs/fixtures/mt0-tilelang-probe/probe/run_probe.py", "205e82532f6fc420bd39ca0425c8ea94fb0573842f37d5aac073f9264cf6df3c"),
    ("tests/support/golden/shared/acceptance-epochs/fixtures/mt0-tilelang-probe/review.json", "09bf3aabfc22c5da1abfa0595e632f7b1b67c8313cb31f2e0ff6effad73c31d2"),
    ("tests/support/golden/shared/acceptance-epochs/fixtures/mt0-tilelang-probe/start-state.json", "8b533b11cf34c4166de5e6d788b327c7c616d28c095b98696c6aa82f871bec13"),
    ("tests/support/golden/shared/acceptance-epochs/fixtures/mt0-tilelang-probe/start-worktree.sha256", "18c895f1c32e05626ad87136d6b15f7e81e6b440cf7ca37140859cd864d10d9b"),
    ("tests/support/golden/shared/acceptance-epochs/fixtures/mt0/acceptance.md", "83f25e9de9b66f254b9bb727d1d154a2c91abfd946f8d361aee633e7f799f4e7"),
    ("tests/support/golden/shared/acceptance-epochs/fixtures/mt0/capability-matrix.json", "3c330b72261011c0b8cf218a1f7e3542a23ada2bcf01ccef0b123af6af6bac8b"),
    ("tests/support/golden/shared/acceptance-epochs/fixtures/mt0/dsl-selection.json", "f938eebe2d4a1f95837ee29f9df072b736c7211f7b97c2a1555a0bc893cbedb8"),
    ("tests/support/golden/shared/acceptance-epochs/fixtures/mt0/local-toolchain.json", "0b66624214b51474e79a07ff9e733717cdab30df2e58bb773a5be0e9f1226407"),
    ("tests/support/golden/shared/acceptance-epochs/fixtures/mt0/manifest.sha256", "a9e3bdc65c720059bee0d35802697d8eaaaa1688bd5cd81cb30df1199650e9a6"),
    ("tests/support/golden/shared/acceptance-epochs/fixtures/mt0/nodes.json", "fca44189ccbfb8c4f1af575696fa9e11f11c2bfc119d327f8b648a4b512683c5"),
    ("tests/support/golden/shared/acceptance-epochs/fixtures/mt0/upstream-map.json", "ca0cb42f6d782416d8e1afd77d33de80d2a4a66675d7f278ac82d6d0edd97032"),
    ("tests/support/golden/shared/acceptance-epochs/fixtures/mt0/vertical-slices.json", "ef7ca22a497d72bd7c89e9be47fc97892044dcace9cc651bcf0df7a63da5986b"),
    ("tests/support/golden/shared/acceptance-epochs/fixtures/mt1a/acceptance.md", "d07640ebc20db01eed28dc478e4bfe2fead0263f3ea59d14d4f51268b0107aa6"),
    ("tests/support/golden/shared/acceptance-epochs/fixtures/mt1a/after.json", "9be07928bc93845d6a12ecfdeecbbff5949f433a0faee94016ac18385e87b775"),
    ("tests/support/golden/shared/acceptance-epochs/fixtures/mt1a/baseline.json", "c9da60fe35a9a106eef8300c014fb6907b0b8c2f0c320f69a84a6c31f218dc1a"),
    ("tests/support/golden/shared/acceptance-epochs/fixtures/mt1a/golden/gemm-msl-bundle-TASK.md", "b27994245bce89bf562cf085739c847528231c9c0f22c36f4482f58f747b7aa6"),
    ("tests/support/golden/shared/acceptance-epochs/fixtures/mt1a/golden/gemm-msl-single-TASK.md", "4e6f84ee0e9521b084c2805602a45e8788d056164519399208caa87a6f8eb732"),
    ("tests/support/golden/shared/acceptance-epochs/fixtures/mt1a/manifest.sha256", "d66abee33124595c15feae2d32ab2e445d5a834827ed9fa7bb507bb4fd631a41"),
    ("tests/support/golden/shared/acceptance-epochs/fixtures/mt1a/nodes.json", "69fffc140c7358b0e008e7bb10957dd3086acdce686a37906321e81b6e202f57"),
    ("tests/support/golden/shared/acceptance-epochs/fixtures/mt1b/acceptance.md", "fdf0107516a34bc9d480174ff840634efee5dd63ff3703f7c3d44d1c9fca5f32"),
    ("tests/support/golden/shared/acceptance-epochs/fixtures/mt1b/b-evidence-drift.json", "d442cdf1414d3859e3f9e3bd8cc69cc8be0f2097484bfc082cd47f389f6fe411"),
    ("tests/support/golden/shared/acceptance-epochs/fixtures/mt1b/b-evidence-resolution.json", "661a8a452c083e04c96402326986265fd0503bc1937b923da4775f71766ae5c8"),
    ("tests/support/golden/shared/acceptance-epochs/fixtures/mt1b/contract.md", "3b9a0fac98985fbcc5961f69acf8e75c6d4c7d4dcafa009e422b02ab972f8f27"),
    ("tests/support/golden/shared/acceptance-epochs/fixtures/mt1b/evidence/closure-before.json", "64773c8b32b3e64696714f35d6c8f11df6c8acd6da7fefe7f876997a4f637fe9"),
    ("tests/support/golden/shared/acceptance-epochs/fixtures/mt1b/evidence/tracks-v1-epoch.json", "00c1f53c0a51f2a910c490519362e52330fabe9f68e4240b5b33c9490e567f93"),
    ("tests/support/golden/shared/acceptance-epochs/fixtures/mt1b/history-anchor-drift.json", "28c7bb4b39cd5c2413567ad92443131d028c00464b4a97c9ff16e7441d532d64"),
    ("tests/support/golden/shared/acceptance-epochs/fixtures/mt1b/identity.json", "a88c8acf7970cdd1a788716d9ca7fa55c4a732ffa8d31f4c8f3c0192b0bc76e3"),
    ("tests/support/golden/shared/acceptance-epochs/fixtures/mt1b/manifest.sha256", "020e3966dd2f0ee10c7dc12a0b464b5ccc921d3e197e5e31917d69fb998526a8"),
    ("tests/support/golden/shared/acceptance-epochs/fixtures/mt1b/nodes.json", "cc305e161ab892ce3d5faa7ba65501f3f4a992bf92604002045ad0d046e4854a"),
    ("tests/support/golden/shared/acceptance-epochs/fixtures/mt1b/p2-epoch-drift.json", "21fc9645e101a17c241e15d32fbe2b3629749c3a0e48435864ab31d9f71ee076"),
    ("tests/support/golden/shared/acceptance-epochs/fixtures/mt4/evidence/deepseek-relu-r1/best.taic", "e871e86a9ef31888d131c09a5437eb94b913ed29b7cc45fd2a08dc8caafda924"),
    ("tests/support/golden/shared/acceptance-epochs/fixtures/mt4/evidence/deepseek-relu-r1/cli-resume-boundary.json", "9a63427151aedc99524833f1c65845668a83e5724a0feb45db39f36b340327d2"),
    ("tests/support/golden/shared/acceptance-epochs/fixtures/mt4/evidence/deepseek-relu-r1/replay.json", "a81c6885c5665e973c8216c7b716d562d3793bddebdcfc0014837a6f51e4da00"),
    ("tests/support/golden/shared/acceptance-epochs/fixtures/mt4/evidence/deepseek-relu-r1/report.json", "34a8b77ced3aa14158cf77f9b44f917ea6b200e6e4c2d5015181dbe16acc70f7"),
    ("tests/support/golden/shared/acceptance-epochs/fixtures/mt4/evidence/deepseek-relu-r1/result.json", "f2765192410ac59954d3b009a57181e9464a629884e3370a0fdcdcf811348d7e"),
    ("tests/support/golden/shared/acceptance-epochs/fixtures/mt4/evidence/deepseek-relu-r1/resume-dry-run.txt", "f9d73201affbd2c0d280cd54d746f4ce7fd7a526de0fbd4ab75233444993b921"),
    ("tests/support/golden/shared/acceptance-epochs/fixtures/mt4/evidence/deepseek-relu-r1/run_config.json", "b28fe31e3ab25a5a43dc2f99b36f3e3b80ee94f179b8c7e74ae4cd54be346cf6"),
    ("tests/support/golden/shared/acceptance-epochs/fixtures/mt4/prompt-epoch.json", "463ca9eb1ae6f0f3461982f29959322cbeb2a4056f855c318da1f5b66dba9b89"),
    ("tests/support/golden/shared/acceptance-epochs/fixtures/mt6-registry-live.json", "0826be13638312d6428c4166ac88775e039b683fb8ee880fa99695e7a955e71c"),
    ("tests/support/golden/shared/acceptance-epochs/fixtures/mt6/evidence/track-explore-r15/observations.json", "49ebdb2cb8d62b19440ccf96af644447349a648e059ca6f3720e23187af52709"),
    ("tests/support/golden/shared/acceptance-epochs/fixtures/mt6/evidence/track-explore-r15/sources/affine-bundle.taic", "e9f8194b089e1d7c43b0ac724b8509c30747cb6e313c4c4259def2cc5c867544"),
    ("tests/support/golden/shared/acceptance-epochs/fixtures/mt6/evidence/track-explore-r15/sources/affine-kernel.py", "e2c355ca2dddebf331dc3503ad6683f0b6349407526870867760371640e63828"),
    ("tests/support/golden/shared/acceptance-epochs/fixtures/mt6/evidence/track-explore-r15/sources/affine-single.metal", "43e89241b2e06854a40b400b1ecc151dcd59e2451eb8039e6189572be86fbdef"),
    ("tests/support/golden/shared/acceptance-epochs/fixtures/mt6/evidence/track-explore-r15/sources/affine-tilelang.taic", "e7f7ba7c85c09ba493f3e198788aaf77feb201d70b523001a405cbb1362ce699"),
    ("tests/support/golden/shared/acceptance-epochs/fixtures/mt6/evidence/track-explore-r15/summary.json", "cc0afa79ce70255ee43de5efdfc41e22eec09b7084ba3e35ff8d827015e3cb08"),
    ("tests/support/golden/shared/acceptance-epochs/fixtures/mt6/evidence/track-explore-r15/track_explore.json", "4564e487cdb984c20fee58e6a01930e998c5019f22ea1bf6db378b32c3f186a8"),
    ("tests/support/golden/shared/acceptance-epochs/fixtures/mt6/registry-epoch.json", "55f75d406fd04f54d0c209c7f999dc4b1e6368f459db6b264388a71d8d97230e"),
    ("tests/support/golden/shared/acceptance-epochs/fixtures/b1/evidence/gpu-gemv.json", "9546d8343a95a82c04c918af5ef5e73571d366842171ce47601913fd15436509"),
    ("tests/support/golden/shared/acceptance-epochs/fixtures/b1/evidence/gpu-rope.json", "c7a535816e1ee22bb13d9b500ddafdce619585693c9e593b5ad672063752f88d"),
    ("tests/support/golden/shared/acceptance-epochs/fixtures/b2/evidence/gpu-conv2d.json", "35a7fc9bcb562ae8e206f734e923f2186ab95cbec8c451131d334664c2cc3b5a"),
    ("tests/support/golden/shared/acceptance-epochs/fixtures/b2/evidence/gpu-lion.json", "ebb0789a6b460bbd1a90364af821b52f8dcb3df10b568ddfd04b161c46f6896b"),
    ("tests/support/golden/shared/acceptance-epochs/fixtures/b3/evidence/gpu-bitdelta_gemm.json", "5ba3f61c879a79e4d8384aaa3fdec1713f76b730d19cad5bb0e1726be1ddf4aa"),
    ("tests/support/golden/shared/acceptance-epochs/fixtures/b3/evidence/gpu-flash_attention.json", "03c775073c5b2bc6870883e5cef5d53d4031395f447e3021d8a4454ce8d3c665"),
    ("tests/support/golden/shared/acceptance-epochs/fixtures/b3/evidence/gpu-mamba2_chunk_scan.json", "60dcab8a04c8c858e4385599bd3319a3fbb46a18f388d2e5682b330482fc970b"),
    ("tests/support/golden/shared/acceptance-epochs/fixtures/b4/evidence/gpu-sem_attention_odd_scale.json", "fce209d919a0b62bb7956c49b70686e4f42a89770791484f7ee455978a9de037"),
    ("tests/support/golden/shared/acceptance-epochs/fixtures/b4/evidence/gpu-sem_int_5.json", "10437c40ec0580c49615d0ce69c03f3c140aa8e30acd932ccd369820812478f7"),
)

MT_AB_MODULE_SHA256 = (
    ("tests/support/mt0_acceptance.py", "7f66f34efb5aa972bb4e9098316d6f658f8516c2f81114f7a629f33b138eb95f"),
    ("tests/support/mt0_isolation_verify.py", "728c609132e0098a26cf5742b34d5c2cf96aa2ebf14e303cafb0b9b1378756e4"),
    ("tests/support/mt0_isolation_profile_verify.py", "24d85c5d0a1183fe31783b148eee4db49e28d4085f76682150de0879bddd0a42"),
    ("tests/support/mt0_tilelang_probe_verify.py", "7f9b2c81759c44d9121f5f76d67fba6e0c49f0de47ad0a6fdc1d16d198c86113"),
    ("tests/support/mt1a_acceptance.py", "d6fcac1e337980f42f3c06732b2a024d9f0b7868b7eafef0722ba1f5202bf7c7"),
    ("tests/support/mt1b_acceptance.py", "d6f678f6df3ebb4368a8f8664c3a45e8e2d7549fade5010aa813baea353c25d4"),
    ("tests/support/a5_g0_compare.py", "652d7be11ae8b816e12bb10970da8c0ecc6ea635efeb7b2f4e4907aaa1116bc0"),
    ("tests/support/a5_perf.py", "4b8066bfbb2714bfca27f6ee4288ec7dfabbdf1baf9b34157710397098f20831"),
    ("tests/support/b1_compare.py", "0ff3bee5c8334b281950fafa36065061d581639ca89a3dbd2e2b1bf664962a7c"),
    ("tests/support/b2_compare.py", "d241d4ef279e72569ace143d9466fe2d401081cdf7013c9740931cb0e04c3593"),
    ("tests/support/b3_compare.py", "79f58a5629d78fde313c46fd8749b5889dd7c99a52ddc096b5ee0d9f42f90dda"),
    ("tests/support/b4_compare.py", "a90c3d87046018d2c5be2fb365481b7fb67084b9886a801e103e3575535843a9"),
    ("tests/support/b4_compare_sem_int.py", "ced49c6c4078701fcf84bb6fe74369cbb684b16f8323366c16701a662b5413ab"),
)


def test_mt_ab_fixture_copies_keep_the_source_bytes():
    """Each copied evidence file is the original bytes, identified by path and SHA-256."""
    assert len(MT_AB_FIXTURE_SHA256) == 169
    for relative, digest in MT_AB_FIXTURE_SHA256:
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == digest, relative


def test_mt_ab_modules_differ_only_by_evidence_paths():
    """Copied modules keep the package bytes except the evidence-path constants."""
    header = (
        b"ROOT = Path(__file__).resolve().parents[4]\n"
        b"HERE = Path(__file__).resolve().parent\n"
    )
    fixtures = b"tests/support/golden/shared/acceptance-epochs/fixtures"
    edits = {
        "tests/support/mt0_acceptance.py": (
            (b'ROOT = Path(__file__).resolve().parents[2]\nHERE = ROOT / "' + fixtures + b'/mt0"\n', header),
        ),
        "tests/support/mt0_isolation_verify.py": (
            (b'ROOT = Path(__file__).resolve().parents[2]\nHERE = ROOT / "' + fixtures + b'/mt0-isolation"\n', header),
        ),
        "tests/support/mt0_isolation_profile_verify.py": (
            (b'ROOT = Path(__file__).resolve().parents[2]\nHERE = ROOT / "' + fixtures + b'/mt0-isolation-profile"\n', header),
            (b'old = ROOT / "' + fixtures + b'/mt0-isolation"', b'old = HERE.parent / "MT0-isolation"'),
            (
                b'selection = json.loads((ROOT / "' + fixtures + b'/mt0/dsl-selection.json").read_text(encoding="utf-8"))',
                b'selection = json.loads((HERE.parent / "MT0" / "dsl-selection.json").read_text(encoding="utf-8"))',
            ),
        ),
        "tests/support/mt0_tilelang_probe_verify.py": (
            (b'ROOT = Path(__file__).resolve().parents[2]\nHERE = ROOT / "' + fixtures + b'/mt0-tilelang-probe"\n', header),
            (
                b'selection = json.loads((ROOT / "' + fixtures + b'/mt0/dsl-selection.json").read_text(encoding="utf-8"))',
                b'selection = json.loads((here.parent / "MT0/dsl-selection.json").read_text(encoding="utf-8"))',
            ),
            (
                b'review = json.loads((ROOT / "' + fixtures + b'/mt0-isolation-profile/outer-probe-review.json").read_text(encoding="utf-8"))',
                b'review = json.loads((here.parent / "MT0-isolation-profile/outer-probe-review.json").read_text(encoding="utf-8"))',
            ),
        ),
        "tests/support/mt1a_acceptance.py": (
            (b'ROOT = Path(__file__).resolve().parents[2]\nHERE = ROOT / "' + fixtures + b'/mt1a"\n', header),
        ),
        "tests/support/mt1b_acceptance.py": (
            (b'ROOT = Path(__file__).resolve().parents[2]\nHERE = ROOT / "' + fixtures + b'/mt1b"\n', header),
            (
                b'R5 = ROOT / "' + fixtures + b'/mt0-tilelang-probe"',
                'R5 = ROOT / "docs/架构/多DSL-Track-阶段验收/MT0-tilelang-probe"'.encode(),
            ),
        ),
        "tests/support/a5_g0_compare.py": (
            (b'ROOT = Path(__file__).resolve().parents[2]\nHERE = ROOT / "' + fixtures + b'/a5"\n', header),
        ),
        "tests/support/a5_perf.py": (
            (b'ROOT = Path(__file__).resolve().parents[2]\nHERE = ROOT / "' + fixtures + b'/a5"\n', header),
        ),
    }
    for name in ("b1", "b2", "b3", "b4"):
        current = (
            b"ROOT = Path(__file__).resolve().parents[2]\n"
            + name.upper().encode()
            + b' = ROOT / "' + fixtures + b"/" + name.encode() + b'"\n'
        )
        original = (
            name.upper().encode() + b" = Path(__file__).resolve().parent\n"
            + b"ROOT = " + name.upper().encode() + b".parents[3]\n"
        )
        edits[f"tests/support/{name}_compare.py"] = ((current, original),)
    edits["tests/support/b4_compare_sem_int.py"] = edits["tests/support/b4_compare.py"]
    assert len(MT_AB_MODULE_SHA256) == len(edits) == 13
    for relative, digest in MT_AB_MODULE_SHA256:
        text = (ROOT / relative).read_bytes()
        for current, original in edits[relative]:
            assert current in text, relative
            text = text.replace(current, original, 1)
        assert hashlib.sha256(text).hexdigest() == digest, relative
