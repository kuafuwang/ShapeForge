#!/usr/bin/env python3
"""P0 acceptance verifier: closed contracts, candidate projection, offline exact dispatch.

P0 starts no model, no search, no run/resume and no GPU.  It verifies the *result* of the offline
slice:

* the complete P0 manifest (every delivered file hashed, no extra/missing entry);
* the acceptance-epoch baseline (`acceptance-baseline.json`, reviewer blocker 12 closed by a real new
  baseline rather than a waiver) still describes this worktree: the epoch's own bytes are pinned
  here, every covered production/test file and `tracks.json` is re-hashed from the worktree, the
  registry/Track digests are recomputed, and the record has to reproduce the earlier audit-point
  digests — so a refreshed snapshot cannot hide drift;
* the historical A4-A6 acceptance artifacts (and the Track registry/`tracks.json`) are unchanged —
  P0 adds contracts and tests, it does not move a Track, a digest or an evidence anchor;
* the closed contracts hold without any fixture self-reference: canonical guard digests, the
  structural-only canonicalization (`a-b<0` is not rewritten to `a<b`), forged/non-canonical
  digests, ``trunc_div``/``rem`` truncation and zero-divisor refusal, the unchanged exact ``/``,
  the non-finite refusal, the parent-never-a-Track-candidate rule, the projection conditions,
  duplicate geometry / path collision / dropped include / parent-SHA reuse, and every dispatch
  outcome (including resource preflight and a non-chaining fallback);
* the fixed P0 CPU nodes all pass with no skip/xfail and an exact pass count;
* the related safe regressions pass (A2 plane, evaluate_task, task assembly, structure).

No GPU, performance, active-manifest or publication evidence is produced: P0 cannot claim any of
them.  A skipped phase is diagnostic only and P0 is not accepted.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.util
import io
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
P0 = Path(__file__).resolve().parent
MANIFEST = P0 / "manifest.sha256"

DOMAIN_TEST = "tests/shared/test_shape_portfolio_domain.py"
CANDIDATE_TEST = "tests/shared/test_shape_portfolio_candidate.py"
DISPATCH_TEST = "tests/shared/test_shape_portfolio_dispatch.py"
P0_TEST = "docs/架构/shape-portfolio-阶段验收/P0/verify.py"

DELIVERABLES: tuple[str, ...] = (
    "tcenv/portfolio/__init__.py",
    "tcenv/portfolio/domain.py",
    "tcenv/portfolio/candidate.py",
    "tcenv/portfolio/dispatch.py",
    "tcenv/portfolio/schema/semantic-domain-plan.v1.json",
    "tcenv/portfolio/schema/candidate-portfolio.v1.json",
    "tcenv/portfolio/schema/portfolio-dispatch-record.v1.json",
    "tcenv/track/expr.py",
    "tcenv/backends/metal/launch.py",
    "tests/support/shape_portfolio.py",
    "tests/support/README.md",
    DOMAIN_TEST,
    CANDIDATE_TEST,
    DISPATCH_TEST,
    "docs/架构/shape-portfolio实施计划.md",
    "docs/架构/shape-portfolio-阶段验收/P0.md",
    "docs/架构/shape-portfolio-阶段验收/P0/baseline_audit.sha256",
    "docs/架构/shape-portfolio-阶段验收/P0/acceptance-baseline.json",
    "docs/架构/shape-portfolio-阶段验收/P0/acceptance-baseline.sha256",
    "docs/架构/shape-portfolio-阶段验收/P0/baseline_epoch.py",
    P0_TEST,
)

#: The acceptance-epoch record (reviewer blocker 12, route 2: a real new baseline, no waiver).
BASELINE_JSON = P0 / "acceptance-baseline.json"
BASELINE_MANIFEST = P0 / "acceptance-baseline.sha256"
BASELINE_AUDIT = P0 / "baseline_audit.sha256"
BASELINE_GENERATOR = P0 / "baseline_epoch.py"

#: The record's own bytes are pinned here, so refreshing the snapshot to hide later drift is an edit
#: to the verifier (and to ``manifest.sha256``) instead of a silent re-baseline.
ACCEPTANCE_BASELINE_SHA256 = "f463eeb37d6e8868123c6de918b88344b4fbe09aa11962be19cb66a4f3e19046"
ACCEPTANCE_BASELINE_EPOCH = "p0-reacceptance-baseline-1"
ACCEPTANCE_BASELINE_HEAD = "4a93ed2a4934fe526eedfcf4577de5dbd366e24a"
ACCEPTANCE_BASELINE_FILE_COUNT = 44
#: sha256 over the newline-joined, sorted covered path list: dropping a covered file from the epoch
#: surface changes this fingerprint even if every remaining digest still matches.
ACCEPTANCE_BASELINE_SURFACE_SHA256 = "c4e073a368ec784a80c560147a13f15b5bbce182af5162545d7f601b8a0367b4"

#: Fixed P0 nodes.  Every planned positive/negative rule has one stable node id; the verifier
#: refuses a missing node, a renamed node, an extra node or a skip.
FIXED_NODE_IDS: tuple[str, ...] = (
    f"{DOMAIN_TEST}::test_p0_a_finite_domain_plan_normalizes_explicit_named_instances",
    f"{DOMAIN_TEST}::test_p0_a_non_finite_domain_is_refused_instead_of_treated_as_sampled_finite",
    f"{DOMAIN_TEST}::test_p0_a_dtype_or_size_outside_the_opspec_is_refused",
    f"{DOMAIN_TEST}::test_p0_a_guard_digest_hashes_the_canonical_ast_not_the_spelling",
    f"{DOMAIN_TEST}::test_p0_a_guard_canonicalization_is_pure_structure_and_idempotent",
    f"{DOMAIN_TEST}::test_p0_parse_guard_rejects_forged_and_non_canonical_digests",
    f"{DOMAIN_TEST}::test_p0_trunc_div_and_rem_truncate_toward_zero_with_the_dividend_sign",
    f"{DOMAIN_TEST}::test_p0_exact_division_keeps_the_a2_semantics",
    f"{DOMAIN_TEST}::test_p0_a_v2_variant_is_still_refused",
    f"{DOMAIN_TEST}::test_p0_the_guard_name_space_is_symbols_and_integer_attrs_only",
    f"{CANDIDATE_TEST}::test_p0_the_envelope_is_parsed_before_any_track_sees_it",
    f"{CANDIDATE_TEST}::test_p0_each_variant_and_fallback_projects_to_an_ordinary_candidate",
    f"{CANDIDATE_TEST}::test_p0_a_shared_source_is_copied_into_every_projection",
    f"{CANDIDATE_TEST}::test_p0_msl_single_requires_exactly_one_self_contained_device_source",
    f"{CANDIDATE_TEST}::test_p0_msl_bundle_requires_host_tiling_and_a_declared_workspace",
    f"{CANDIDATE_TEST}::test_p0_includes_match_the_existing_v2_screen",
    f"{CANDIDATE_TEST}::test_p0_malformed_envelopes_are_refused_closed",
    f"{CANDIDATE_TEST}::test_p0_a_legacy_bundle_without_a_coverage_manifest_is_refused",
    f"{CANDIDATE_TEST}::test_p0_a_projection_path_collision_and_parent_sha_reuse_are_refused",
    f"{CANDIDATE_TEST}::test_p0_variant_identity_is_stable_author_neutral_and_geometry_free",
    f"{CANDIDATE_TEST}::test_p0_the_track_registry_and_tracks_json_are_unchanged",
    f"{CANDIDATE_TEST}::test_p0_the_domain_fallback_is_not_a_timing_baseline",
    f"{CANDIDATE_TEST}::test_p0_the_portfolio_package_revives_no_forbidden_architecture",
    f"{DISPATCH_TEST}::test_p0_unparseable_requests_are_semantic_unresolved",
    f"{DISPATCH_TEST}::test_p0_a_request_outside_the_finite_domain_never_falls_back",
    f"{DISPATCH_TEST}::test_p0_a_request_document_round_trips_and_binds_to_the_authoritative_instance",
    f"{DISPATCH_TEST}::test_p0_an_undecidable_guard_or_evidence_is_dispatch_unknown",
    f"{DISPATCH_TEST}::test_p0_threshold_odd_tail_and_large_points_dispatch_exactly_one_variant",
    f"{DISPATCH_TEST}::test_p0_multiple_matches_are_dispatch_overlap_without_priority",
    f"{DISPATCH_TEST}::test_p0_a_coverage_hole_uses_the_declared_fallback_or_is_unmatched",
    f"{DISPATCH_TEST}::test_p0_resource_preflight_diverts_before_the_variant_runs",
    f"{DISPATCH_TEST}::test_p0_a_fallback_resource_failure_never_chains",
    f"{DISPATCH_TEST}::test_p0_a_declared_geometry_requires_the_cpu_preflight",
    f"{DISPATCH_TEST}::test_p0_a_domain_plan_cannot_relabel_another_opspec_or_derived_abi",
    f"{DISPATCH_TEST}::test_p0_finite_registration_reports_points_overlaps_and_empty_variants",
    f"{DISPATCH_TEST}::test_p0_the_record_is_a_composition_decision_not_runtime_or_perf_evidence",
)
FIXED_EXPECTED_PASSED = len(FIXED_NODE_IDS)

#: Related safe regressions (files, not pinned counts: another author may add a case to one of
#: these files without invalidating P0).  No GPU/LLM/slow case is selected.
REGRESSION_FILES: tuple[str, ...] = (
    "tests/shared/test_track_geometry.py",
    "tests/shared/test_track_delivery.py",
    "tests/shared/test_track_bundle.py",
    "tests/shared/test_track_abi.py",
    "tests/shared/test_evaluate_task.py",
    "tests/shared/test_cross_hardware_boundary_a2.py",
    "tests/shared/test_structure.py",
    "tests/shared/test_variant_family.py",
    "tests/shared/test_task_assembly_structure.py",
    "tests/backends/metal/test_launch.py",
)

#: A4-A6 acceptance artifacts.  P0 must not touch a historical record, a Track or an anchor.
HISTORICAL_DIGESTS: dict[str, str] = {
    "docs/架构/跨硬件算子生成分工-阶段验收/A4.md": "ca26ca3e0320a8b2f35760ef02b91262c5eb1ab2c3f23bf33738c427a9604b20",
    "docs/架构/跨硬件算子生成分工-阶段验收/A4/verify.py": "389398e9bdb54bf132fedfdf06795b6e2e028f02a9affc8aa68c984f61dd3259",
    "docs/架构/跨硬件算子生成分工-阶段验收/A4/manifest.sha256": "4690a55e5fa8cd835c1ae364935864858ee96ecc36e6747efe148ef05b9b3bcc",
    "docs/架构/跨硬件算子生成分工-阶段验收/A5.md": "f08e239c86c44a92b69f3e16e5d32d355ff871be8b94bbb8372ee59e037f172d",
    "docs/架构/跨硬件算子生成分工-阶段验收/A5/verify.py": "146c7d6e3e707244d46a49929625a7848d9f64cd0a8257d494dd7664e44e7b85",
    "docs/架构/跨硬件算子生成分工-阶段验收/A5/manifest.sha256": "b0668b8786952d1171fa51ae8d90fa0d50678c9ec87d484578a9a128a146d406",
    "docs/架构/跨硬件算子生成分工-阶段验收/A6.md": "f22dfed7d9ba20b060bd7f2d44dbd90303e22c47d684d3b1faabe4fc951a4bb8",
    "docs/架构/跨硬件算子生成分工-阶段验收/A6/verify.py": "05a38ee168dcd633dd7e8841c4452547db07fc3123148189950f62e1fd1b9ba6",
    "docs/架构/跨硬件算子生成分工-阶段验收/A6/manifest.sha256": "ff8e5d9e136e9aa2e311732b28f5d35d17f8651213a3eabf8a51b7c7b96251f5",
}
#: Stages whose manifests list the frozen evidence files.  The full historical manifest covers
#: source files that legitimately drifted after that stage, so only the ``evidence/`` entries are
#: re-checked (the nine top-level records above stay pinned by their own digests).
HISTORICAL_EVIDENCE_ROOT = "docs/架构/跨硬件算子生成分工-阶段验收"
HISTORICAL_EVIDENCE_STAGES: tuple[str, ...] = ("A4", "A5", "A6")
TRACKS_PATH = "backends/metal/tracks.json"
TRACKS_SHA256 = "00c1f53c0a51f2a910c490519362e52330fabe9f68e4240b5b33c9490e567f93"
REGISTRY_DIGEST = "9d568eec90bab265cdc5d498a7f31baeb0f1406240617e028f9a5da4dd6044cb"
TRACK_DIGESTS = {
    "msl-single": "7d19a788cb2df0b25816b9f4200d9676f8d7eec317d2411ef0a09c1b8c703af8",
    "msl-bundle": "31b10ee5b33bc6d418106cb85154f6d14d04e08fb92d5382462dbe212610b9ba",
}
FORBIDDEN_ARCHITECTURE_NAMES = (
    "PatternResolver",
    "LoweringRuleRegistry",
    "StagePartitioner",
    "TileConnector",
    "LoweringProfile",
    "@variant",
    "msl-portfolio",
)


def require(condition: bool, message: str) -> None:
    """@brief 断言一个验收条件。"""
    if not condition:
        raise AssertionError(message)


def sha256(path: Path) -> str:
    """@brief 文件字节 sha256。"""
    return hashlib.sha256(path.read_bytes()).hexdigest()


#: One pytest summary line ends with ``N passed``; the count is parsed as an integer so a
#: substring test can never accept ``136 passed`` for a 36-node gate.
_PASSED_COUNT = re.compile(r"(\d+) passed\b")


def _passed_count(summary: str, *, where: str) -> int:
    """@brief 从 pytest summary 行解析整数 passed 数。

    @param summary pytest 最后一行摘要。
    @param where 诊断位置。
    @return passed 数。
    @raises AssertionError 摘要里不是恰好一个 passed 计数。
    """
    matches = _PASSED_COUNT.findall(summary)
    require(len(matches) == 1, f"{where}: expected exactly one '<n> passed' count in {summary!r}")
    return int(matches[0])


def _pytest(
    command: list[str],
    *,
    where: str,
    expect_passed: int | None = None,
    allow_deselected: bool = False,
) -> str:
    """@brief 运行一次 pytest 并校验通过/跳过/数量。"""
    environment = os.environ.copy()
    environment.setdefault("PYTHONPATH", str(ROOT))
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, env=environment)
    output = (result.stdout or "") + (result.stderr or "")
    require(result.returncode == 0, f"{where} failed:\n{output[-8000:]}")
    summary = [line for line in output.splitlines() if re.search(r"\b(passed|failed|error)\b", line)]
    require(summary, f"{where}: no pytest summary line")
    last = summary[-1]
    require(" skipped" not in last and "xfailed" not in last, f"{where} skipped a fixed node: {last}")
    if not allow_deselected:
        require(" deselected" not in last, f"{where} deselected a selected node: {last}")
    if expect_passed is not None:
        passed = _passed_count(last, where=where)
        require(
            passed == expect_passed,
            f"{where}: expected exactly {expect_passed} passed, got {passed} ({last})",
        )
    return last


# ---------------------------------------------------------------------------------------------
# Manifest and historical artifacts
# ---------------------------------------------------------------------------------------------


def write_manifest() -> None:
    """@brief 重写 P0 manifest.sha256。"""
    lines = [f"{sha256(ROOT / name)}  {name}" for name in DELIVERABLES]
    MANIFEST.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"OK: wrote {MANIFEST.relative_to(ROOT)} ({len(lines)} files)")


def verify_manifest() -> None:
    """@brief 校验 manifest 的摘要与文件集合完全一致。"""
    lines = MANIFEST.read_text(encoding="utf-8").splitlines()
    seen: dict[str, str] = {}
    for line in lines:
        digest, name = line.split("  ", 1)
        require(bool(re.fullmatch(r"[0-9a-f]{64}", digest)), f"bad digest: {name}")
        require(name not in seen, f"duplicate manifest entry: {name}")
        seen[name] = digest
    expected = set(DELIVERABLES)
    require(
        set(seen) == expected,
        f"manifest file set differs: missing={sorted(expected - set(seen))} extra={sorted(set(seen) - expected)}",
    )
    for name, digest in seen.items():
        path = ROOT / name
        require(path.is_file(), f"manifest file is missing: {name}")
        require(sha256(path) == digest, f"manifest mismatch: {name}")
    require(P0_TEST in seen, "the verifier itself is not pinned by the manifest")
    print(f"OK: manifest.sha256 ({len(seen)} files)")


def verify_historical_evidence() -> int:
    """@brief 逐个校验 A4-A6 manifest 中列出的 evidence 文件冻结 SHA-256。

    The rest of those manifests lists source files that are expected to drift after the stage
    (the current worktree is post-A6), so this checks exactly the ``evidence/`` entries, file by
    file, instead of replaying a whole historical manifest against today's tree.

    @return 校验的 evidence 文件数。
    """
    checked = 0
    for stage in HISTORICAL_EVIDENCE_STAGES:
        manifest = ROOT / HISTORICAL_EVIDENCE_ROOT / stage / "manifest.sha256"
        require(manifest.is_file(), f"the {stage} manifest is missing: {manifest}")
        prefix = f"{HISTORICAL_EVIDENCE_ROOT}/{stage}/evidence/"
        for line in manifest.read_text(encoding="utf-8").splitlines():
            record = line.strip()
            if not record:
                continue
            parts = record.split("  ", 1)
            require(len(parts) == 2, f"the {stage} manifest has a malformed line: {record!r}")
            digest, name = parts
            if not name.startswith(prefix):
                continue
            path = ROOT / name
            require(path.is_file(), f"historical evidence is missing: {name}")
            require(sha256(path) == digest, f"historical evidence changed: {name}")
            checked += 1
    require(checked > 0, "no historical evidence files were checked")
    return checked


def verify_historical() -> None:
    """@brief A4-A6 验收记录、evidence 文件与 Track/registry digest 未变。"""
    for name, digest in HISTORICAL_DIGESTS.items():
        path = ROOT / name
        require(path.is_file(), f"historical evidence is missing: {name}")
        require(sha256(path) == digest, f"historical evidence changed: {name}")
    evidence = verify_historical_evidence()
    require(sha256(ROOT / TRACKS_PATH) == TRACKS_SHA256, "backends/metal/tracks.json changed")
    from tcenv.identity import fingerprint
    from tcenv.track.registry import load_track_registry

    registry = load_track_registry()
    require(fingerprint(registry.document()) == REGISTRY_DIGEST, "the Track registry digest changed")
    require({track.track_id for track in registry.tracks} == {"msl-single", "msl-bundle"}, "a Track was added/removed")
    for track in registry.tracks:
        require(fingerprint(track.document()) == TRACK_DIGESTS[track.track_id], f"track {track.track_id} digest changed")
    print(
        f"OK: A4-A6 records untouched ({len(HISTORICAL_DIGESTS)} files) + {evidence} frozen evidence files, "
        "tracks.json/registry/Track digests unchanged"
    )


# ---------------------------------------------------------------------------------------------
# Acceptance-epoch baseline (reviewer blocker 12, route 2: real new baseline, no waiver)
# ---------------------------------------------------------------------------------------------


def _baseline_module() -> object:
    """@brief 加载 epoch 生成器模块（路径清单的单一来源，不提供摘要）。"""
    spec = importlib.util.spec_from_file_location("p0_baseline_epoch", BASELINE_GENERATOR)
    require(spec is not None and spec.loader is not None, "the epoch baseline generator cannot be loaded")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _parse_sha256_manifest(path: Path) -> dict[str, str]:
    """@brief 解析 ``<sha>  <path>`` 清单（忽略 ``#`` 说明头）。"""
    entries: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        record = line.strip()
        if not record or record.startswith("#"):
            continue
        parts = record.split("  ", 1)
        require(len(parts) == 2, f"{path}: malformed manifest line {record!r}")
        digest, name = parts
        require(bool(re.fullmatch(r"[0-9a-f]{64}", digest)), f"{path}: bad digest for {name}")
        require(name not in entries, f"{path}: duplicate entry {name}")
        entries[name] = digest
    return entries


def _baseline_drift(record: dict, root: Path) -> list[str]:
    """@brief 重算记录中每个文件摘要，返回漂移/缺失清单。"""
    drift: list[str] = []
    for name, digest in record["files"].items():
        path = root / name
        if not path.is_file():
            drift.append(f"missing: {name}")
        elif sha256(path) != digest:
            drift.append(f"changed: {name}")
    return drift


def _verify_baseline_files(record: dict, root: Path) -> list[str]:
    """@brief 校验基线文件集、manifest、审计交叉证据与当前工作树一致。

    @param record 已读入的 epoch 记录。
    @param root 待校验的仓库根（自检沙箱可传入副本）。
    @return 已校验的覆盖文件名（已排序）。
    @raises AssertionError 文件集、摘要、manifest 或审计交叉证据不一致。
    """
    module = _baseline_module()
    surface = tuple(module.CONTRACT_SURFACE)
    files = record.get("files")
    require(isinstance(files, dict) and bool(files), "the epoch baseline records no file digests")
    require(
        len(files) == ACCEPTANCE_BASELINE_FILE_COUNT,
        f"the epoch baseline covers {len(files)} files, expected {ACCEPTANCE_BASELINE_FILE_COUNT}",
    )
    require(
        tuple(sorted(files)) == tuple(sorted(surface)),
        "the epoch baseline file set differs from the generator's contract surface",
    )
    require(
        hashlib.sha256("\n".join(sorted(files)).encode()).hexdigest() == ACCEPTANCE_BASELINE_SURFACE_SHA256,
        "the epoch baseline covers a different path list than the pinned surface",
    )
    groups = record["contract_surface"]["groups"]
    require(set(groups) == set(module.GROUPS), f"epoch baseline groups differ: {sorted(groups)}")
    for name, paths in groups.items():
        require(bool(paths), f"epoch baseline group {name} is empty")
        require(len(set(paths)) == len(paths), f"epoch baseline group {name} repeats a path")
        require(set(paths) <= set(files), f"epoch baseline group {name} lists an uncovered file")
    union = set().union(*(set(paths) for paths in groups.values()))
    require(
        union == set(files) - {TRACKS_PATH},
        "the epoch baseline groups do not add up to the covered file set",
    )
    require(TRACKS_PATH in files, "the epoch baseline does not cover backends/metal/tracks.json")
    require(
        any(name.startswith("tcenv/portfolio/") for name in files),
        "the epoch baseline does not cover the portfolio production code",
    )
    # The epoch pins a closed set, not just the digests of a known list: a *new* module dropped into
    # an owned directory is drift too, and so is a new `test_shape_portfolio_*` file.
    for directory, suffix in (("tcenv/portfolio", ".py"), ("tcenv/portfolio/schema", ".json")):
        present = {
            str(path.relative_to(root))
            for path in (root / directory).iterdir()
            if path.is_file() and path.suffix == suffix and "__pycache__" not in path.parts
        }
        covered_here = {
            name
            for name in files
            if name.startswith(f"{directory}/") and "/" not in name[len(directory) + 1 :]
        }
        require(
            present == covered_here,
            f"{directory} contains files outside the epoch surface: {sorted(present ^ covered_here)}",
        )
    covered_tests = {name for name in files if name.startswith("tests/shared/test_shape_portfolio_")}
    present_tests = {f"tests/shared/{path.name}" for path in (root / "tests/shared").glob("test_shape_portfolio_*.py")}
    require(
        present_tests == covered_tests,
        f"the P0 test set drifted from the epoch surface: {sorted(present_tests ^ covered_tests)}",
    )
    drift = _baseline_drift(record, root)
    require(not drift, f"the contract surface drifted from the acceptance-epoch baseline: {drift}")
    p0_relative = P0.relative_to(ROOT)
    manifest_path = root / p0_relative / BASELINE_MANIFEST.name
    require(manifest_path.is_file(), "the acceptance-epoch manifest is missing")
    manifest_entries = _parse_sha256_manifest(manifest_path)
    require(
        manifest_entries == files,
        f"the epoch manifest disagrees with the record: {sorted(set(manifest_entries) ^ set(files))}",
    )
    require(sha256(manifest_path) == record["contract_surface"]["manifest_sha256"], "the epoch manifest bytes changed")
    # Independent corroboration: `baseline_audit.sha256` was written at an earlier audit point (it is
    # untouched by this epoch) and recorded the same bytes for every covered file, so the new record
    # reproduces an older snapshot instead of inventing digests now.
    audit_path = root / p0_relative / BASELINE_AUDIT.name
    require(audit_path.is_file(), "the audit-point snapshot is missing")
    audit = _parse_sha256_manifest(audit_path)
    require(set(files) <= set(audit), f"the audit snapshot does not cover {sorted(set(files) - set(audit))}")
    corroborated = sorted(name for name, digest in files.items() if audit[name] != digest)
    require(not corroborated, f"the epoch baseline disagrees with the audit-point snapshot: {corroborated}")
    for name, digest in record["observations"]["documents"].items():
        if name in audit:
            require(audit[name] == digest, f"the epoch snapshot of {name} disagrees with the audit point")
    return sorted(files)


def _git_commit_exists(head: str) -> bool | None:
    """@brief 记录中的 HEAD 是否解析到一个提交（git 不可用时返回 None）。"""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--verify", "--quiet", f"{head}^{{commit}}"],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
    except OSError:
        return None
    return result.returncode == 0 and result.stdout.strip() == head


def _verify_baseline_identity(record: dict, root: Path) -> None:
    """@brief 校验基线标注、HEAD/状态证据与 tracks.json、registry、Track digest。"""
    require(
        record.get("schema_id") == "tcenv.p0.acceptance-baseline.1",
        f"unexpected epoch baseline schema_id: {record.get('schema_id')!r}",
    )
    require(record.get("epoch") == ACCEPTANCE_BASELINE_EPOCH, "the epoch baseline name changed")
    require(record.get("baseline_kind") == "acceptance-epoch-start", "the record is not an epoch-start baseline")
    require(record.get("is_p0_start_baseline") is False, "the record claims to be the P0-start baseline")
    require(record.get("supersedes_blocker") == 12, "the record does not name reviewer blocker 12")
    note = str(record.get("note") or "").lower()
    require(
        "not the missing p0-start baseline" in note,
        "the record does not state that it is not the missing P0-start baseline",
    )
    require("new p0 acceptance epoch" in note, "the record does not name the new acceptance epoch")
    head = record.get("head")
    require(
        isinstance(head, str) and bool(re.fullmatch(r"[0-9a-f]{40}", head)),
        "the epoch baseline HEAD is not a full commit id",
    )
    require(head == ACCEPTANCE_BASELINE_HEAD, "the epoch baseline records a different HEAD than the pinned one")
    resolved = _git_commit_exists(head)
    require(resolved is not False, "the recorded epoch HEAD does not resolve to a commit")
    status = record.get("git_status") or {}
    entries = status.get("entries")
    codes = status.get("codes") or {}
    require(isinstance(entries, int) and entries > 0, "the epoch baseline records no git status entries")
    require(isinstance(codes, dict) and sum(codes.values()) == entries, "the baseline git status counts do not add up")
    require(
        bool(re.fullmatch(r"[0-9a-f]{64}", str(status.get("porcelain_z_sha256") or ""))),
        "the epoch baseline records no git status digest",
    )
    tracks = record.get("tracks") or {}
    require(tracks.get("path") == TRACKS_PATH, "the epoch baseline does not pin backends/metal/tracks.json")
    require(tracks.get("sha256") == TRACKS_SHA256, "the baseline tracks.json digest is not the frozen A4-A6 one")
    require(sha256(root / TRACKS_PATH) == tracks["sha256"], "backends/metal/tracks.json drifted from the epoch baseline")
    from tcenv.identity import fingerprint
    from tcenv.track.registry import load_track_registry

    registry = load_track_registry()
    recomputed = fingerprint(registry.document())
    require(
        recomputed == tracks.get("registry_digest") == REGISTRY_DIGEST,
        "the registry digest drifted from the epoch baseline",
    )
    digests = {track.track_id: fingerprint(track.document()) for track in registry.tracks}
    require(digests == tracks.get("track_digests"), f"a Track digest drifted from the epoch baseline: {digests}")
    require(digests == TRACK_DIGESTS, "a Track digest drifted from the frozen A4-A6 value")
    if resolved is None:
        print("NOTE: git is unavailable; the recorded epoch HEAD was not cross-checked against the repository")


def verify_acceptance_baseline(root: Path = ROOT) -> None:
    """@brief 新验收 epoch 基线：记录真实、覆盖合同面，且当前工作树未漂移。

    The enforced anti-drift property is the per-file digest map (production, tests, `tracks.json`)
    plus the recomputed registry/Track digests.  The recorded HEAD and `git status` snapshot are
    evidence of the epoch start, not a live gate: a later commit must not invalidate a historical
    stage, while the covered bytes still have to match.

    @param root 待校验的仓库根。
    @raises AssertionError 记录被重生、标注不对、或合同面漂移。
    """
    p0_relative = P0.relative_to(ROOT)
    baseline_path = root / p0_relative / BASELINE_JSON.name
    require(baseline_path.is_file(), "the acceptance-epoch baseline is missing")
    require(
        sha256(baseline_path) == ACCEPTANCE_BASELINE_SHA256,
        "the acceptance-epoch baseline was regenerated (its bytes differ from the pinned epoch digest); "
        "only a new epoch plus a full re-acceptance may change that pin",
    )
    record = json.loads(baseline_path.read_text(encoding="utf-8"))
    require(isinstance(record, dict), "the acceptance-epoch baseline is not a JSON object")
    covered = _verify_baseline_files(record, root)
    _verify_baseline_identity(record, root)
    print(
        f"OK: acceptance-epoch baseline {ACCEPTANCE_BASELINE_EPOCH} ({len(covered)} files: production, tests, "
        f"their contract dependencies and tracks.json), HEAD {ACCEPTANCE_BASELINE_HEAD}"
    )
    print(
        "OK: contract surface + tracks.json + registry/Track digests unchanged since the epoch start; "
        "the record reproduces the earlier audit-point digests"
    )


# ---------------------------------------------------------------------------------------------
# Contract negatives, no GPU
# ---------------------------------------------------------------------------------------------


def verify_contracts(tmp_path: Path) -> None:
    """@brief 离线合同负例：规范化、除法、投影条件、dispatch 八态、Track 不变。"""
    sys.path.insert(0, str(ROOT))
    from tcenv.contract_errors import GeometryError, TaskContractError, UnsupportedContractCapability
    from tcenv.identity import fingerprint
    from tcenv.opspec.v2 import instantiate_v2
    from tcenv.portfolio.candidate import (
        PORTFOLIO_ENTRY_PATH,
        EvaluationBinding,
        parse_portfolio,
        project_provider,
    )
    from tcenv.portfolio.dispatch import (
        DISPATCH_RECORD_SCHEMA_PATH,
        DispatchRequest,
        ExactDispatchResolver,
        GeometryPreflight,
        PortfolioDispatchRecord,
        preflight_geometry,
    )
    from tcenv.portfolio.domain import (
        canonical_guard,
        guard_ast_document,
        guard_digest,
        normalize_semantic_domain_plan,
        parse_guard,
    )
    from tcenv.opspec.validator import validate
    from tcenv.target import DEFAULT_TARGET_ID, load_target
    from tcenv.taskcontract import load_task_contract
    from tcenv.track.expr import evaluate_term
    from tcenv.track.plane import CandidateArtifact, CandidateEntry
    from tcenv.track.registry import load_track_registry
    from tests.support.shape_portfolio import (
        bundle_fallback_document,
        evidence_all_valid,
        geometry_bytes,
        geometry_document,
        load_portfolio_task,
        parent_entries,
        portfolio_document,
        portfolio_instances,
        portfolio_opspec,
        write_portfolio_task,
    )

    checks: list[str] = []
    schema = json.loads(DISPATCH_RECORD_SCHEMA_PATH.read_text(encoding="utf-8"))

    # The fixed-node gate parses the passed count as an integer; a substring test would accept
    # "136 passed" as "36 passed".
    require(_passed_count("36 passed in 0.7s", where="count self-test") == 36, "the passed-count parser lost 36")
    require(
        _passed_count("136 passed in 0.7s", where="count self-test") == 136,
        "the passed-count parser truncated 136",
    )
    require(
        _passed_count("136 passed in 0.7s", where="count self-test") != FIXED_EXPECTED_PASSED,
        "a 136-passed summary would satisfy the 36-node gate",
    )
    checks.append("exact passed-count parsing (136 is not 36)")

    # 1. Canonical guard digests: equivalent spellings agree, pure-structure alternatives differ.
    require(guard_digest(["gt", "N", 63]) == guard_digest(["lt", 63, "N"]), "gt/lt canonicalization broke")
    require(
        guard_digest(["and", ["ge", "N", 64], ["lt", "N", 128]])
        == guard_digest(["and", ["lt", "N", 128], ["le", 64, "N"]]),
        "and operand ordering/flattening broke",
    )
    require(
        guard_digest(["lt", ["-", "N", 63], 0]) != guard_digest(["lt", 63, "N"]),
        "a-b<0 was algebraically rewritten",
    )
    sub = canonical_guard(["lt", ["-", "N", 63], 0])
    require(guard_ast_document(sub.node)["args"][0]["call"] == "-", "the subtraction node was rewritten")
    checks.append("canonical guard: gt/ge re-orient, and/or flatten+sort, no algebraic rewrite")

    # 2. Forged digest (raw spelling) and mismatched digest are refused.
    import hashlib as _hashlib

    raw = _hashlib.sha256(json.dumps(["and", ["ge", "N", 64], ["lt", "N", 128]]).encode()).hexdigest()
    for declared in (raw, "f" * 64):
        try:
            parse_guard(
                {"expr": ["and", ["ge", "N", 64], ["lt", "N", 128]], "digest": declared}
            )
        except TaskContractError as exc:
            require("canonical AST digest" in str(exc), f"unexpected forged-digest refusal: {exc}")
        else:
            raise AssertionError("a non-canonical guard digest was accepted")
    checks.append("forged/non-canonical guard digest refused")

    # 3. trunc_div/rem truncate toward zero with the dividend sign; zero divisor is a GeometryError.
    for term, expected in (
        (["trunc_div", "-7", "2"], -3),
        (["trunc_div", "7", "-2"], -3),
        (["rem", "-7", "2"], -1),
        (["rem", "7", "-2"], 1),
        (["rem", "-1", "2"], -1),
        (["rem", "63", "2"], 1),
    ):
        require(evaluate_term(term, {}, where="p0", expect="int") == expected, f"{term} evaluated wrong")
    for term in (["trunc_div", "1", "0"], ["rem", "1", "0"]):
        try:
            evaluate_term(term, {}, where="p0", expect="int")
        except GeometryError:
            pass
        else:
            raise AssertionError(f"{term} did not refuse a zero divisor")
    checks.append("trunc_div/rem: truncation, dividend sign, zero divisor")

    # 4. ``/`` keeps exact A2 division semantics (no truncation for the portfolio).
    require(evaluate_term(["/", "-8", "2"], {}, where="p0", expect="int") == -4, "/ changed")
    for term in (["/", "7", "2"], ["/", "7", "0"]):
        try:
            evaluate_term(term, {}, where="p0", expect="int")
        except GeometryError as exc:
            require("divide evenly" in str(exc), f"unexpected division refusal: {exc}")
        else:
            raise AssertionError(f"{term} was accepted as exact division")
    checks.append("exact / preserved")

    # 5. non_finite is registered vocabulary but not activated.
    contract = load_portfolio_task(tmp_path / "domain")
    instances = portfolio_instances(contract)
    plan = normalize_semantic_domain_plan(contract.spec, instances, target=load_target(DEFAULT_TARGET_ID))
    try:
        normalize_semantic_domain_plan(
            contract.spec, instances, target=load_target(DEFAULT_TARGET_ID), domain_kind="non_finite"
        )
    except UnsupportedContractCapability:
        pass
    else:
        raise AssertionError("a non-finite domain was activated")
    document = plan.document()
    document["domain_kind"] = "non_finite"
    try:
        from tcenv.portfolio.domain import parse_semantic_domain_plan

        parse_semantic_domain_plan(document)
    except UnsupportedContractCapability:
        pass
    else:
        raise AssertionError("a non-finite domain document was accepted")
    checks.append("finite-only domain activation")

    # 6. Projection: bundle/single tracks, parent never a Track candidate, geometry binding.
    from tcenv.portfolio.domain import DomainAuthority

    authority = DomainAuthority(
        plan=plan,
        spec=contract.spec,
        target=load_target(DEFAULT_TARGET_ID),
        instances=instances,
        contract=contract,
    )
    portfolio = parse_portfolio(parent_entries(portfolio_document(plan.digest)), domain_plan=plan)
    bundle = project_provider(portfolio, portfolio.variants[0], authority=authority)
    single = project_provider(portfolio, portfolio.variants[1], authority=authority)
    fallback = project_provider(portfolio, portfolio.fallback, authority=authority)
    require(bundle.track_id == "msl-bundle" and single.track_id == "msl-single", "projection split tracks wrong")
    require(fallback.provider == "fallback" and fallback.binding.variant_id is None, "fallback was credited as a variant")
    require(
        fallback.binding.geometry_origin_path is None and fallback.binding.geometry_path is None,
        "a single-source fallback invented a geometry",
    )
    require(
        bundle.binding.geometry_origin_path == "host/geometry_a.json"
        and bundle.binding.geometry_path == "host/execution_geometry.json",
        "the original geometry declaration is not bound",
    )
    require(
        bundle.binding.geometry_origin_sha256
        == _hashlib.sha256(portfolio.entry("host/geometry_a.json").data).hexdigest(),
        "the original geometry content SHA-256 is not bound",
    )
    require(bundle.candidate.candidate_sha256 != portfolio.parent_sha256, "parent SHA reused as a child candidate SHA")
    from tcenv.backends.metal.v2_runtime import HostTilingPlan

    require(
        HostTilingPlan.from_candidate(bundle.candidate, contract) is not None,
        "the projected bundle's Host Tiling plan is not legal for the existing closed parser",
    )
    registry = load_track_registry()
    for track_id in ("msl-single", "msl-bundle"):
        parent = CandidateArtifact(track_id=track_id, entries=portfolio.entries)
        try:
            parent.validate_for(registry.get(track_id))
        except TaskContractError:
            pass
        else:
            raise AssertionError("the outer portfolio passed CandidateArtifact.validate_for")
    checks.append("projection tracks + parent never a Track candidate + geometry binding")

    # 7. Projection refusals: duplicate geometry, dropped include, workspace missing, path collision.
    from tests.support.shape_portfolio import GUARD_C, manifest_entry, rename_variant

    duplicate = json.loads(json.dumps(portfolio_document(plan.digest)))
    duplicate["fallback"]["geometry_entry"] = "host/geometry_a.json"
    try:
        parse_portfolio(parent_entries(duplicate), domain_plan=plan)
    except TaskContractError as exc:
        require("duplicate geometry" in str(exc), f"unexpected duplicate-geometry refusal: {exc}")
    else:
        raise AssertionError("a duplicate geometry declaration was accepted")
    include_data = b'#include "device/common.metal"\n/* taic-launch */\nkernel void includes_common() {}\n'
    include_document = json.loads(json.dumps(portfolio_document(plan.digest)))
    include_document["variants"][1]["entries"] = ["device/includes_common.metal"]
    include_portfolio = parse_portfolio(
        parent_entries(
            include_document,
            extra_sources={"device/includes_common.metal": ("device_kernel", "text/plain", include_data)},
        ),
        domain_plan=plan,
    )
    try:
        project_provider(include_portfolio, include_portfolio.variants[1], authority=authority)
    except TaskContractError as exc:
        require("quoted #include" in str(exc), f"unexpected quoted-include refusal: {exc}")
    else:
        raise AssertionError("a quoted include was accepted although the v2 screen refuses it")
    # Angle-bracket includes are closed at the projection boundary: a host path never survives
    # to a build, while a real Metal standard header still passes.
    bare_include = b"#include </etc/passwd>\n/* taic-launch */\nkernel void k(device half *y) {}\n"
    angle_document = json.loads(json.dumps(portfolio_document(plan.digest)))
    angle_document["variants"][1]["entries"] = ["device/angle_host.metal"]
    angle_portfolio = parse_portfolio(
        parent_entries(
            angle_document,
            extra_sources={"device/angle_host.metal": ("device_kernel", "text/plain", bare_include)},
        ),
        domain_plan=plan,
    )
    try:
        project_provider(angle_portfolio, angle_portfolio.variants[1], authority=authority)
    except TaskContractError as exc:
        require("Metal standard header" in str(exc), f"unexpected angle-include refusal: {exc}")
    else:
        raise AssertionError("an angle-bracket host include was accepted")
    metal_include = b"#include <metal_stdlib>\n/* taic-launch */\nkernel void k(device half *y) {}\n"
    metal_document = json.loads(json.dumps(portfolio_document(plan.digest)))
    metal_document["variants"][1]["entries"] = ["device/angle_metal.metal"]
    metal_portfolio = parse_portfolio(
        parent_entries(
            metal_document,
            extra_sources={"device/angle_metal.metal": ("device_kernel", "text/plain", metal_include)},
        ),
        domain_plan=plan,
    )
    require(
        project_provider(metal_portfolio, metal_portfolio.variants[1], authority=authority).track_id == "msl-single",
        "a Metal standard header include no longer passes the projection",
    )
    # The projection boundary and the existing v2 source screen agree on the same bytes.
    from tcenv.backends.metal.screen import screen_v2_source

    for projection in (bundle, single):
        for entry in projection.candidate.entries:
            if entry.role == "device_kernel":
                require(
                    screen_v2_source(bytes(entry.data).decode("utf-8")) == [],
                    f"a projected device source fails the existing v2 screen: {entry.path}",
                )
    # A fake marker is not a launch block: the projection reuses the backend's launch scan, so a
    # line comment, a string literal and a second block are all refused here.
    for label, source in (
        ("line", b"// taic-launch\nkernel void fake(device half *y) {}\n"),
        ("string", b'const char *m = "taic-launch";\nkernel void fake(device half *y) {}\n'),
        ("twice", b"/* taic-launch */\n/* taic-launch */\nkernel void fake(device half *y) {}\n"),
        ("prefix", b"/* taic-launch-fake */\nkernel void fake(device half *y) {}\n"),
        (
            "misplaced",
            b"/* taic-launch */\n/* not-a-launch taic-launch */\nkernel void fake(device half *y) {}\n",
        ),
    ):
        name = f"device/fake_{label}.metal"
        fake_document = json.loads(json.dumps(portfolio_document(plan.digest)))
        fake_document["variants"][1]["entries"] = [name]
        fake_portfolio = parse_portfolio(
            parent_entries(fake_document, extra_sources={name: ("device_kernel", "text/plain", source)}),
            domain_plan=plan,
        )
        try:
            project_provider(fake_portfolio, fake_portfolio.variants[1], authority=authority)
        except TaskContractError as exc:
            require("taic-launch" in str(exc), f"unexpected fake-launch refusal ({label}): {exc}")
        else:
            raise AssertionError(f"a fake taic-launch marker was accepted ({label})")
    # A real block still projects (variant_b.metal keeps `/* taic-launch */`).
    require(
        project_provider(portfolio, portfolio.variants[1], authority=authority).track_id == "msl-single",
        "a real taic-launch block no longer passes the projection",
    )
    # A host_tiling role is not enough: a non-JSON plan is refused by the existing parser.
    bad_host = b"// not a host plan\n"
    bad_document = json.loads(json.dumps(portfolio_document(plan.digest)))
    bad_document["variants"][0]["entries"] = ["device/variant_a.metal", "host/tiling_bad.metal"]
    bad_portfolio = parse_portfolio(
        parent_entries(
            bad_document,
            extra_sources={"host/tiling_bad.metal": ("host_tiling", "application/json", bad_host)},
        ),
        domain_plan=plan,
    )
    try:
        project_provider(bad_portfolio, bad_portfolio.variants[0], authority=authority)
    except TaskContractError as exc:
        require("is not JSON" in str(exc), f"unexpected host-tiling refusal: {exc}")
    else:
        raise AssertionError("a non-JSON Host Tiling plan was accepted")

    # A declared geometry is never delivered as a source: neither by its own provider nor by a
    # different provider that lists that path in its entries.
    from tests.support.shape_portfolio import bundle_fallback_document

    for label in ("fallback", "cross-provider"):
        leak = bundle_fallback_document(plan.digest)
        if label == "fallback":
            leak["fallback"]["entries"].append("host/geometry_generic.json")
        else:
            leak["variants"][0]["entries"].append("host/geometry_generic.json")
        try:
            parse_portfolio(parent_entries(leak), domain_plan=plan)
        except TaskContractError:
            pass
        else:
            raise AssertionError(f"a geometry entry was accepted as a source entry ({label})")
    # A workspace-less OpSpec refuses msl-bundle; the workspace fact is derived from the
    # authoritative DomainAuthority, so there is no boolean declaration left to forge.
    bare = load_portfolio_task(tmp_path / "bare", workspace=False)
    bare_instances = portfolio_instances(bare)
    bare_plan = normalize_semantic_domain_plan(
        bare.spec, bare_instances, target=load_target(DEFAULT_TARGET_ID)
    )
    bare_authority = DomainAuthority(
        plan=bare_plan,
        spec=bare.spec,
        target=load_target(DEFAULT_TARGET_ID),
        instances=bare_instances,
        contract=bare,
    )
    require(not bare_authority.workspace_declared, "a workspace-less OpSpec was reported as declaring workspace")
    bare_portfolio = parse_portfolio(
        parent_entries(portfolio_document(bare_plan.digest)), domain_plan=bare_plan
    )
    try:
        project_provider(bare_portfolio, bare_portfolio.variants[0], authority=bare_authority)
    except TaskContractError as exc:
        require("declares no workspace tensor" in str(exc), f"unexpected workspace refusal: {exc}")
    else:
        raise AssertionError("a bundle without an authoritative workspace was accepted")
    try:
        project_provider(bare_portfolio, bare_portfolio.variants[0], authority=authority)
    except TaskContractError as exc:
        require("different semantic domain plan" in str(exc), f"unexpected borrowed-authority refusal: {exc}")
    else:
        raise AssertionError("a projection borrowed another domain's workspace authority")
    collision = portfolio_document(plan.digest)
    collision["variants"][0]["entries"].append("host/execution_geometry.json")
    extra = {
        "host/execution_geometry.json": (
            "host_tiling",
            "application/json",
            geometry_bytes(geometry_document("variant_a")),
        )
    }
    collision_portfolio = parse_portfolio(
        parent_entries(collision, extra_sources=extra), domain_plan=plan
    )
    try:
        project_provider(collision_portfolio, collision_portfolio.variants[0], authority=authority)
    except TaskContractError as exc:
        require("would overwrite" in str(exc), f"unexpected path-collision refusal: {exc}")
    else:
        raise AssertionError("a projected path collision was accepted")
    binding = bundle.binding.document()
    binding["candidate_sha256"] = portfolio.parent_sha256
    try:
        EvaluationBinding(**binding)
    except TaskContractError:
        pass
    else:
        raise AssertionError("the parent SHA was accepted as a projected candidate identity")
    checks.append(
        "projection refusals: duplicate geometry, geometry-as-source, non-JSON Host Tiling, quoted/angle include, "
        "fake launch marker, workspace, collision, parent SHA"
    )

    # 8. Every dispatch outcome is reachable and every record passes its registered schema.
    resolver = ExactDispatchResolver(
        domain_plan=plan,
        portfolio=portfolio,
        evidence=evidence_all_valid(portfolio),
        spec=contract.spec,
        target=load_target(DEFAULT_TARGET_ID),
        instances=instances,
    )

    from tcenv.track.abi import abi_limits_from_target

    limits = abi_limits_from_target(load_target(DEFAULT_TARGET_ID))

    def hook_for(instance, abi):
        return lambda geometry: preflight_geometry(geometry, instance=instance, contract=abi, limits=limits)

    records: list[PortfolioDispatchRecord] = []
    records.append(resolver.resolve({}))
    outside = DispatchRequest(
        op_name="portfolio_pointwise",
        dtype="fp16",
        size="not_a_named_size",
        usage="golden",
        symbols={"N": 1},
        attrs={},
        semantic_instance_digest="a" * 64,
        target_abi_digest="b" * 64,
    )
    records.append(resolver.resolve(outside))
    unknown_guard = json.loads(json.dumps(portfolio_document(plan.digest)))
    unknown_guard["variants"][2]["guard"] = {
        "expr": ["eq", "K", 3],
        "digest": guard_digest(["eq", "K", 3], where="variant-c-tail"),
    }
    unknown_guard["coverage_plan"]["buckets"][2]["guard_digest"] = guard_digest(["eq", "K", 3], where="odd_tail")
    unknown_portfolio = parse_portfolio(parent_entries(unknown_guard), domain_plan=plan)
    tail_request, tail_instance, tail_abi = _request(contract, plan, "tail")
    records.append(
        ExactDispatchResolver(
            domain_plan=plan,
            portfolio=unknown_portfolio,
            evidence=evidence_all_valid(unknown_portfolio),
            spec=contract.spec,
            target=load_target(DEFAULT_TARGET_ID),
            instances=instances,
        )
        .resolve(tail_request, preflight=hook_for(tail_instance, tail_abi))
    )
    overlap = json.loads(json.dumps(portfolio_document(plan.digest)))
    overlap["coverage_plan"]["buckets"].append(
        {"bucket_key": "tail_plain", "guard_digest": guard_digest(["lt", "N", 32], where="tail_plain")}
    )
    overlap["variants"].append(
        {
            "variant_key": "variant-d-overlap",
            "bucket_key": "tail_plain",
            "guard": {"expr": ["lt", "N", 32], "digest": guard_digest(["lt", "N", 32], where="variant-d")},
            "entries": ["device/variant_b.metal"],
            "geometry_entry": None,
        }
    )
    overlap_portfolio = parse_portfolio(parent_entries(overlap), domain_plan=plan)
    records.append(
        ExactDispatchResolver(
            domain_plan=plan,
            portfolio=overlap_portfolio,
            evidence=evidence_all_valid(overlap_portfolio),
            spec=contract.spec,
            target=load_target(DEFAULT_TARGET_ID),
            instances=instances,
        )
        .resolve(tail_request, preflight=hook_for(tail_instance, tail_abi))
    )
    unmatched_document = json.loads(json.dumps(portfolio_document(plan.digest)))
    unmatched_document["fallback"]["reasons"] = ["resource_unavailable"]
    unmatched_portfolio = parse_portfolio(parent_entries(unmatched_document), domain_plan=plan)
    below_request, below_instance, below_abi = _request(contract, plan, "below")
    records.append(
        ExactDispatchResolver(
            domain_plan=plan,
            portfolio=unmatched_portfolio,
            evidence=evidence_all_valid(unmatched_portfolio),
            spec=contract.spec,
            target=load_target(DEFAULT_TARGET_ID),
            instances=instances,
        )
        .resolve(below_request, preflight=hook_for(below_instance, below_abi))
    )
    threshold_request, threshold_instance, threshold_abi = _request(contract, plan, "threshold")
    records.append(resolver.resolve(threshold_request, preflight=hook_for(threshold_instance, threshold_abi)))
    records.append(resolver.resolve(below_request, preflight=hook_for(below_instance, below_abi)))
    over_limit = geometry_bytes(geometry_document("variant_a", threads=2048))
    over_entries = {"host/geometry_a.json": ("host_tiling", "application/json", over_limit)}
    over_document = json.loads(json.dumps(portfolio_document(plan.digest)))
    over_document["fallback"]["reasons"] = ["outside_candidate_coverage"]
    over_portfolio = parse_portfolio(parent_entries(over_document, extra_sources=over_entries), domain_plan=plan)
    records.append(
        ExactDispatchResolver(
            domain_plan=plan,
            portfolio=over_portfolio,
            evidence=evidence_all_valid(over_portfolio),
            spec=contract.spec,
            target=load_target(DEFAULT_TARGET_ID),
            instances=instances,
        )
        .resolve(threshold_request, preflight=hook_for(threshold_instance, threshold_abi))
    )
    expected_outcomes = {
        "semantic_unresolved",
        "unsupported_contract_capability",
        "dispatch_unknown",
        "dispatch_overlap",
        "dispatch_unmatched",
        "resolved",
        "fallback",
        "variant_unavailable",
    }
    require({record.outcome for record in records} == expected_outcomes, "a dispatch outcome was not exercised")
    # A legitimate replay document parses back to the request and resolves identically.
    replay = DispatchRequest.parse(threshold_request.document())
    require(replay == threshold_request, "a request document does not round-trip")
    require(replay.document() == threshold_request.document(), "a replay document is not stable")
    expected_resolution = resolver.resolve(threshold_request, preflight=hook_for(threshold_instance, threshold_abi))
    require(expected_resolution.outcome == "resolved", "the round-trip case did not resolve")
    require(
        resolver.resolve(replay.document(), preflight=hook_for(threshold_instance, threshold_abi)).document()
        == expected_resolution.document(),
        "a replayed document does not resolve like the original request",
    )
    # A request that contradicts the authoritative instance never resolves or falls back.
    for field, value in (
        ("op_name", "completely_other_op"),
        ("symbols", {"N": 1024}),
        ("attrs", {"EPS": 1}),
        ("semantic_instance_digest", "c" * 64),
        ("target_abi_digest", "d" * 64),
    ):
        forged = threshold_request.document()
        forged[field] = value
        refusal = resolver.resolve(forged)
        require(
            refusal.outcome == "semantic_unresolved" and refusal.provider is None and refusal.fallback_id is None,
            f"a request contradicting the authoritative instance via {field} was not refused",
        )
        validate(refusal.document(), schema)
    # The plan itself is not self-certifying: relabelling the opspec/generation/ABI identity and
    # rebinding the portfolio to the forged plan is refused by the authoritative normalization.
    from tcenv.portfolio.candidate import PORTFOLIO_ENTRY_PATH
    from tcenv.portfolio.domain import parse_semantic_domain_plan

    for field, value in (("opspec_digest", "f" * 64), ("target_generation_digest", "f" * 64)):
        forged_document = plan.document()
        forged_document[field] = value
        forged_plan = parse_semantic_domain_plan(forged_document)
        forged_portfolio = parse_portfolio(
            parent_entries(portfolio_document(forged_plan.digest)), domain_plan=forged_plan
        )
        try:
            ExactDispatchResolver(
                domain_plan=forged_plan,
                portfolio=forged_portfolio,
                evidence=evidence_all_valid(forged_portfolio),
                spec=contract.spec,
                target=load_target(DEFAULT_TARGET_ID),
                instances=instances,
            )
        except TaskContractError:
            pass
        else:
            raise AssertionError(f"a plan with a forged {field} was accepted")
    forged_document = plan.document()
    for point in forged_document["instances"]:
        point["target_abi_digest"] = "a" * 64
    forged_plan = parse_semantic_domain_plan(forged_document)
    forged_portfolio = parse_portfolio(
        parent_entries(portfolio_document(forged_plan.digest)), domain_plan=forged_plan
    )
    try:
        ExactDispatchResolver(
            domain_plan=forged_plan,
            portfolio=forged_portfolio,
            evidence=evidence_all_valid(forged_portfolio),
            spec=contract.spec,
            target=load_target(DEFAULT_TARGET_ID),
            instances=instances,
        )
    except TaskContractError:
        pass
    else:
        raise AssertionError("a plan with forged derived ABI digests was accepted")
    # A legacy bundle without the fixed coverage manifest never enters the portfolio path.
    legacy_entries = tuple(
        entry for entry in parent_entries(portfolio_document(plan.digest)) if entry.path != PORTFOLIO_ENTRY_PATH
    )
    for bundle in (legacy_entries, tuple(entry for entry in legacy_entries if entry.path == "device/variant_b.metal")):
        try:
            parse_portfolio(bundle, domain_plan=plan)
        except TaskContractError as exc:
            require("requires the fixed entry" in str(exc), f"unexpected legacy-bundle refusal: {exc}")
        else:
            raise AssertionError("a legacy bundle without coverage_plan/portfolio.json was accepted")
    checks.append("plan authority (opspec/generation/ABI) + legacy bundle refusal")

    # A request outside the record schema's field caps is semantic_unresolved, never a record
    # with an illegal $.request; a long request-controlled detail is clamped, not rejected.
    from tcenv.portfolio.domain import instance_key as _instance_key

    oversized = threshold_request.document()
    oversized["size"] = "s" * 65
    oversized["instance_key"] = _instance_key("fp16", oversized["size"], "golden")
    refusal = resolver.resolve(oversized)
    require(
        refusal.outcome == "semantic_unresolved" and refusal.request is None,
        "an unrecordable request was recorded instead of refused",
    )
    validate(refusal.document(), schema)
    huge_symbol = threshold_request.document()
    huge_symbol["symbols"] = {"K" * 5000: 1}
    refusal = resolver.resolve(huge_symbol)
    require(refusal.outcome == "semantic_unresolved", "a huge symbol contradiction was not refused")
    require(len(refusal.detail or "") <= 4000, "a refusal detail exceeded the record schema's detail cap")
    validate(refusal.document(), schema)
    checks.append("record-schema-consistent requests (field caps, clamped detail)")

    # A forged derived key is not accepted.
    require(
        resolver.resolve({**threshold_request.document(), "instance_key": "fp16/large/golden"}).outcome
        == "semantic_unresolved",
        "a forged instance_key was accepted",
    )
    require(PortfolioDispatchRecord.__name__ == "PortfolioDispatchRecord", "the composition record was renamed")
    require(not hasattr(__import__("tcenv.portfolio.dispatch", fromlist=["x"]), "DispatchRecord"), "the runtime record name was reused")
    for record in records:
        validate(record.document(), schema)
        require(record.identity == fingerprint(record.document()), "a record identity does not cover its document")
        for forbidden in ("speedup", "median_us", "active", "publication_allowed"):
            require(forbidden not in record.document(), f"a dispatch record carries {forbidden}")
    fallback_record = next(record for record in records if record.outcome == "fallback")
    require(fallback_record.provider == "fallback" and fallback_record.variant_id is None, "fallback was credited as a variant")
    resource_record = next(record for record in records if record.outcome == "variant_unavailable")
    require(resource_record.preflight.status == "resource_unavailable", "the resource preflight did not divert the variant")
    checks.append(
        "dispatch: 8/8 outcomes, request round-trip, authoritative-instance binding, preflight diversion, "
        "fallback not credited, schema-valid records"
    )

    # 9. v2 ``@variant`` is still refused; forbidden architecture names are absent.
    try:
        load_task_contract(write_portfolio_task(tmp_path / "variant", op_name="portfolio_pointwise"), variant="x")
    except UnsupportedContractCapability:
        pass
    else:
        raise AssertionError("a v2 @variant was assembled")
    package = ROOT / "tcenv" / "portfolio"
    for path in sorted(package.glob("*.py")):
        text = path.read_text(encoding="utf-8")
        for token in FORBIDDEN_ARCHITECTURE_NAMES:
            require(token not in text, f"{path.name} revives the forbidden architecture name {token}")
    checks.append("v2 @variant refused; no forbidden architecture names in tcenv/portfolio")

    # 10. The registry is unchanged (historical check repeats it at the file level).
    require(sha256(ROOT / TRACKS_PATH) == TRACKS_SHA256, "tracks.json changed during the contract check")
    checks.append("Track registry/tracks.json unchanged")

    # 11. The acceptance-epoch baseline is evidence, not a rubber stamp: a faithful copy verifies,
    # one changed byte in a covered contract file fails, a deleted covered file fails, refreshing the
    # snapshot after a change is caught by the manifest/audit cross-checks, and a regenerated record
    # is refused by the pinned bytes.
    record = json.loads(BASELINE_JSON.read_text(encoding="utf-8"))
    p0_relative = P0.relative_to(ROOT)
    sandbox = tmp_path / "epoch-sandbox"

    def _stage(source: Path, relative: str) -> None:
        target = sandbox / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())

    for name in record["files"]:
        _stage(ROOT / name, name)
    for name in (BASELINE_JSON.name, BASELINE_MANIFEST.name, BASELINE_AUDIT.name):
        _stage(P0 / name, str(p0_relative / name))
    require(
        _verify_baseline_files(record, sandbox) == sorted(record["files"]),
        "a faithful copy of the epoch baseline did not verify",
    )
    require(
        _verify_baseline_files(json.loads(BASELINE_JSON.read_text(encoding="utf-8")), sandbox)
        == sorted(record["files"]),
        "the epoch baseline file check is not reproducible",
    )
    with contextlib.redirect_stdout(io.StringIO()):
        verify_acceptance_baseline(sandbox)
    victim = sandbox / "tcenv/portfolio/dispatch.py"
    original = victim.read_bytes()
    victim.write_bytes(original + b"\n")
    try:
        _verify_baseline_files(record, sandbox)
    except AssertionError as exc:
        require("drifted" in str(exc), f"unexpected drift refusal: {exc}")
    else:
        raise AssertionError("a one-byte change to a covered contract file passed the epoch baseline")
    victim.write_bytes(original)
    removed = sandbox / "tcenv/track/expr.py"
    removed_bytes = removed.read_bytes()
    removed.unlink()
    try:
        _verify_baseline_files(record, sandbox)
    except AssertionError as exc:
        require("drifted" in str(exc), f"unexpected missing-file refusal: {exc}")
    else:
        raise AssertionError("a deleted covered contract file passed the epoch baseline")
    removed.write_bytes(removed_bytes)
    # A new module dropped into the owned package is drift as well (the epoch pins a closed set).
    intruder = sandbox / "tcenv/portfolio/extra.py"
    intruder.write_text("# not part of the epoch surface\n", encoding="utf-8")
    try:
        _verify_baseline_files(record, sandbox)
    except AssertionError as exc:
        require("outside the epoch surface" in str(exc), f"unexpected added-file refusal: {exc}")
    else:
        raise AssertionError("a new module in the P0 package passed the epoch baseline")
    intruder.unlink()
    # "Refresh the snapshot and its manifest" still loses: the audit-point cross-check binds the
    # digests to a snapshot that predates this epoch.
    refreshed = json.loads(json.dumps(record))
    refreshed["files"]["tcenv/portfolio/dispatch.py"] = _hashlib.sha256(original + b"\n").hexdigest()
    refreshed_manifest = _baseline_module().manifest_text(refreshed["files"])
    (sandbox / p0_relative / BASELINE_MANIFEST.name).write_text(refreshed_manifest, encoding="utf-8")
    refreshed["contract_surface"]["manifest_sha256"] = _hashlib.sha256(refreshed_manifest.encode()).hexdigest()
    victim.write_bytes(original + b"\n")
    try:
        _verify_baseline_files(refreshed, sandbox)
    except AssertionError as exc:
        require("audit-point" in str(exc), f"unexpected refreshed-baseline refusal: {exc}")
    else:
        raise AssertionError("a refreshed baseline hid a contract-surface change")
    victim.write_bytes(original)
    _stage(BASELINE_MANIFEST, str(p0_relative / BASELINE_MANIFEST.name))
    # A regenerated record (different bytes in the sandbox) is refused before it is even compared.
    sandbox_record = sandbox / p0_relative / BASELINE_JSON.name
    sandbox_record.write_bytes(sandbox_record.read_bytes().replace(b'"stage": "P0"', b'"stage": "P0!"'))
    with contextlib.redirect_stdout(io.StringIO()):
        try:
            verify_acceptance_baseline(sandbox)
        except AssertionError as exc:
            require("regenerated" in str(exc), f"unexpected regenerated-baseline refusal: {exc}")
        else:
            raise AssertionError("a regenerated epoch baseline was accepted")
    checks.append(
        "acceptance-epoch baseline: pinned bytes, 44-file contract surface, tracks.json/registry/Track digests, "
        "drift and re-baseline are both refused"
    )

    print(f"OK: contract self-test ({len(checks)} checks)")
    for name in checks:
        print(f"  - {name}")


def _request(contract, plan, size: str, usage: str = "golden"):
    """@brief 现场构造一个请求（与测试 helper 同规则）。"""
    from tcenv.opspec.v2 import instantiate_v2
    from tcenv.portfolio.dispatch import DispatchRequest
    from tcenv.target import DEFAULT_TARGET_ID, load_target
    from tcenv.track.abi import abi_limits_from_target, derive_tensor_memory_contract

    profile = load_target(DEFAULT_TARGET_ID)
    instance = instantiate_v2(contract.spec, "fp16", size)
    abi = derive_tensor_memory_contract(instance, target=profile)
    point = plan.find("fp16", size, usage)
    require(point is not None, f"{size}/{usage} is not a domain point")
    return (
        DispatchRequest.from_instance(instance, target_abi_digest=point.target_abi_digest, usage=usage),
        instance,
        abi,
    )


# ---------------------------------------------------------------------------------------------
# Pytest phases
# ---------------------------------------------------------------------------------------------


def run_fixed_cpu() -> None:
    """@brief 运行 P0 固定 CPU 节点（精确数量、无 skip/xfail）。"""
    summary = _pytest(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-n",
            "0",
            "-p",
            "no:cacheprovider",
            "-m",
            "not gpu and not llm and not slow",
            *FIXED_NODE_IDS,
        ],
        where="fixed P0 CPU pytest",
        expect_passed=FIXED_EXPECTED_PASSED,
    )
    print(f"OK: fixed P0 CPU pytest ({FIXED_EXPECTED_PASSED} nodes)\n{summary}")


def run_regression() -> None:
    """@brief 相关 safe regression（A2 平面、evaluate_task、装配、结构）。"""
    summary = _pytest(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-n",
            "0",
            "-p",
            "no:cacheprovider",
            "-m",
            "not gpu and not llm and not slow",
            *REGRESSION_FILES,
        ],
        where="related safe regression",
        allow_deselected=True,
    )
    print(f"OK: related safe regression ({len(REGRESSION_FILES)} files, no failure/skip)\n{summary}")


def main() -> None:
    """@brief 验收入口。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--write-manifest", action="store_true")
    parser.add_argument("--skip-pytest", action="store_true", help="static/contract only; not a P0 acceptance")
    args = parser.parse_args()
    if args.write_manifest:
        write_manifest()
        return
    verify_manifest()
    verify_acceptance_baseline()
    verify_historical()
    with tempfile.TemporaryDirectory(prefix="p0-contract-") as temporary:
        verify_contracts(Path(temporary))
    if not args.skip_pytest:
        run_fixed_cpu()
        run_regression()
        print(f"OK: P0 acceptance ({FIXED_EXPECTED_PASSED} fixed nodes + {len(REGRESSION_FILES)} regression files)")
    else:
        print("WARNING: the pytest phases were skipped; this is diagnostic only and P0 is not accepted")
        raise SystemExit(2)


if __name__ == "__main__":
    main()
