#!/usr/bin/env python3
"""P1 stage-start baseline: the real P0-completed worktree P1 continues from.

实施计划 §4/§5 require every stage to start by recording the full HEAD, the worktree manifest and
SHA-256 digests, and the Track digest — a later HEAD may never stand in for the stage's start.
P0 closed its own start-baseline gap with a real new epoch (``P0.md`` §6.10); P1 inherits that
*tree* and records its own real start here, before a single P1 file exists.

What is recorded:

* the full HEAD (verified to resolve to a commit) and a ``git status --porcelain=v1 -z`` snapshot;
* a **complete worktree manifest** (``start-worktree.sha256``: every file outside ``.git/``,
  ``logs/``, caches and this manifest itself), so "worktree manifest/SHA-256" is a real full
  snapshot and not a hand-picked list;
* the enforced contract surface (P0 production/tests/regressions + the P1-reused execution
  modules + ``backends/metal/tracks.json``) byte for byte;
* ``tracks.json`` plus the freshly recomputed registry and per-Track digests.

Independence: P0's own acceptance record is *not* trusted here.  The generator and
``verify.py`` both compare every covered file against ``P0/acceptance-baseline.json`` (whose own
bytes are pinned inside ``P0/verify.py``), so a baseline invented today cannot pass: it has to
reproduce an older, independently pinned snapshot.  ``start-worktree.sha256`` is start-time
evidence and is deliberately *not* required to stay unchanged (P1 adds files); the contract
surface is the enforced part.

This record is evidence, not a gate.  Its bytes are pinned in ``verify.py``; regenerating it to
hide drift therefore requires editing the verifier and the manifest too, and a second baseline is
a new epoch that must redo the whole P1 acceptance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
P1 = Path(__file__).resolve().parent
BASELINE_JSON = P1 / "acceptance-baseline.json"
BASELINE_MANIFEST = P1 / "acceptance-baseline.sha256"
START_MANIFEST = P1 / "start-worktree.sha256"
P0_BASELINE = ROOT / "docs" / "架构" / "shape-portfolio-阶段验收" / "P0" / "acceptance-baseline.json"

SCHEMA_ID = "tcenv.p1.acceptance-baseline.1"
EPOCH = "p1-real-slice-baseline-1"
BASELINE_KIND = "acceptance-stage-start"
STAGE = "P1"
TRACKS_PATH = "backends/metal/tracks.json"

#: P0's production surface: every file P0 created plus the one backend file it modified.
P0_PRODUCTION: tuple[str, ...] = (
    "tcenv/portfolio/__init__.py",
    "tcenv/portfolio/domain.py",
    "tcenv/portfolio/candidate.py",
    "tcenv/portfolio/dispatch.py",
    "tcenv/portfolio/schema/semantic-domain-plan.v1.json",
    "tcenv/portfolio/schema/candidate-portfolio.v1.json",
    "tcenv/portfolio/schema/portfolio-dispatch-record.v1.json",
    "tcenv/track/expr.py",
    "tcenv/backends/metal/launch.py",
)

#: Contract code P0 does not own but executes its contracts against (identical to P0's epoch).
P0_CONTRACT_DEPENDENCIES: tuple[str, ...] = (
    "tcenv/identity.py",
    "tcenv/contract_errors.py",
    "tcenv/target.py",
    "tcenv/schema.py",
    "tcenv/opspec/model.py",
    "tcenv/opspec/v2.py",
    "tcenv/opspec/validator.py",
    "tcenv/taskcontract/__init__.py",
    "tcenv/taskcontract/loader.py",
    "tcenv/taskcontract/baselines.py",
    "tcenv/track/abi.py",
    "tcenv/track/geometry.py",
    "tcenv/track/plane.py",
    "tcenv/track/registry.py",
    "tcenv/track/delivery.py",
    "tcenv/backends/metal/v2_runtime.py",
    "tcenv/backends/metal/screen.py",
)

#: P0's test surface: the three fixed-node files, their fixture module and the test harness.
P0_TESTS: tuple[str, ...] = (
    "tests/shared/test_shape_portfolio_domain.py",
    "tests/shared/test_shape_portfolio_candidate.py",
    "tests/shared/test_shape_portfolio_dispatch.py",
    "tests/support/shape_portfolio.py",
    "tests/support/v2_tasks.py",
    "tests/support/README.md",
    "tests/conftest.py",
)

#: P0's safe regressions (files, not counts).  P1 re-runs these unchanged.
P0_REGRESSION: tuple[str, ...] = (
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

#: Execution modules P1 reuses without modifying.  P1 only *adds* a composition seam on top of
#: ``evaluate_task``; if any of these bytes changes, the P1 baseline has to be redone as a new
#: epoch instead of silently accepting a different execution protocol.
P1_REUSED_PRODUCTION: tuple[str, ...] = (
    "tcenv/taskcontract/runner.py",
    "tcenv/taskcontract/locator.py",
    "tcenv/taskcontract/reference.py",
    "tcenv/taskcontract/sampling.py",
    "tcenv/backends/metal/track_adapter.py",
    "tcenv/backends/metal/runtime.py",
    "tcenv/backends/metal/source_builder.py",
    "tcenv/backends/metal/stage_binding.py",
    "tcenv/backends/metal/baselines.py",
    "tcenv/backends/metal/testing.py",
    "tcenv/backends/registry.py",
    "tcenv/evaluate.py",
    "tcenv/build_projection.py",
    "tcenv/bench.py",
    "tcenv/v2_worker.py",
    "tcenv/remote/artifact_builder.py",
    "tcenv/remote/artifact_bundle.py",
    "tcenv/remote/execution_geometry.py",
    "tcenv/track/bundle.py",
    "tcenv/track/lifecycle.py",
    "tcenv/track/protocol.py",
    "tcenv/track/__init__.py",
)

#: Historical acceptance records P1 must leave alone.  These are the same nine pinned by P0's
#: verify plus P0's own four acceptance files; P1 changes none of them.
HISTORICAL_RECORDS: tuple[str, ...] = (
    "docs/架构/跨硬件算子生成分工-阶段验收/A4.md",
    "docs/架构/跨硬件算子生成分工-阶段验收/A4/verify.py",
    "docs/架构/跨硬件算子生成分工-阶段验收/A4/manifest.sha256",
    "docs/架构/跨硬件算子生成分工-阶段验收/A5.md",
    "docs/架构/跨硬件算子生成分工-阶段验收/A5/verify.py",
    "docs/架构/跨硬件算子生成分工-阶段验收/A5/manifest.sha256",
    "docs/架构/跨硬件算子生成分工-阶段验收/A6.md",
    "docs/架构/跨硬件算子生成分工-阶段验收/A6/verify.py",
    "docs/架构/跨硬件算子生成分工-阶段验收/A6/manifest.sha256",
    "docs/架构/shape-portfolio-阶段验收/P0.md",
    "docs/架构/shape-portfolio-阶段验收/P0/verify.py",
    "docs/架构/shape-portfolio-阶段验收/P0/manifest.sha256",
    "docs/架构/shape-portfolio-阶段验收/P0/acceptance-baseline.json",
)

GROUPS: dict[str, tuple[str, ...]] = {
    "p0_production": P0_PRODUCTION,
    "p0_contract_dependencies": P0_CONTRACT_DEPENDENCIES,
    "p0_tests": P0_TESTS,
    "p0_safe_regression": P0_REGRESSION,
    "p1_reused_production": P1_REUSED_PRODUCTION,
    "historical_records": HISTORICAL_RECORDS,
}

#: Everything the stage-start baseline enforces, in a stable order.
CONTRACT_SURFACE: tuple[str, ...] = (
    P0_PRODUCTION
    + P0_CONTRACT_DEPENDENCIES
    + P0_TESTS
    + P0_REGRESSION
    + P1_REUSED_PRODUCTION
    + HISTORICAL_RECORDS
    + (TRACKS_PATH,)
)

#: Documents that change *while P1 is implemented*.  Recorded as start-time evidence only.
OBSERVED_DOCUMENTS: tuple[str, ...] = (
    "docs/架构/shape-portfolio实施计划.md",
)

#: The full-worktree manifest covers the *source* worktree.  Generated/derived trees (virtualenvs,
#: run databases, measurement data, caches) are excluded and named here, so the recorded manifest is
#: an auditable source snapshot instead of a dump of the local machine's run history.
WORKTREE_EXCLUDED_DIRS: tuple[str, ...] = (
    ".git",
    "logs",
    "runs",
    "data",
    ".venvs",
    ".venv",
    "node_modules",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".idea",
)
WORKTREE_EXCLUDED_SUFFIXES: tuple[str, ...] = (".pyc", ".pyo")
WORKTREE_EXCLUDED_FILES: tuple[str, ...] = (
    "docs/架构/shape-portfolio-阶段验收/P1/start-worktree.sha256",
)

#: The declared P1 change to the stage-start surface (the same allow-list ``verify.py`` enforces).
ALLOWED_DRIFT: tuple[str, ...] = ("tcenv/portfolio/dispatch.py",)

NOTE = (
    "Real start of the P1 (shape-portfolio vertical slice) acceptance stage on the P0-completed "
    "worktree, taken before any P1 file existed. The HEAD is the real current commit and is never "
    "replaced by an older one; the full worktree manifest is start-time evidence. The enforced "
    "surface is the P0 production/test/regression contract, the execution modules P1 reuses, the "
    "historical A4-A6 and P0 records, and tracks.json; P1's own documents are listed under "
    "observations because they change while the stage runs. Regenerating this file is a new epoch "
    "and requires the whole P1 acceptance to be redone; it must never be used to hide code drift."
)


def sha256(path: Path) -> str:
    """@brief 文件字节 sha256。

    @param path 文件路径。
    @return 十六进制摘要。
    """
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git(*arguments: str) -> str:
    """@brief 在仓库根目录运行 git 并返回 stdout。

    @param arguments git 参数。
    @return stdout。
    """
    result = subprocess.run(["git", *arguments], cwd=ROOT, capture_output=True, text=True, check=True)
    return result.stdout


def head_commit() -> tuple[str, bool]:
    """@brief 当前 HEAD 与其是否解析到一个提交。

    @return ``(head, resolves)``。
    """
    head = _git("rev-parse", "HEAD").strip()
    exists = subprocess.run(
        ["git", "rev-parse", "--verify", "--quiet", f"{head}^{{commit}}"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    return head, exists.returncode == 0 and exists.stdout.strip() == head


def git_status() -> dict[str, object]:
    """@brief 工作树状态快照：条目数、状态码计数与 ``--porcelain=v1 -z`` 摘要。

    @return 字段字典。
    """
    raw = subprocess.run(
        ["git", "status", "--porcelain=v1", "-z"], cwd=ROOT, capture_output=True, check=True
    ).stdout
    codes: dict[str, int] = {}
    for line in _git("status", "--porcelain=v1").splitlines():
        code = line[:2]
        codes[code] = codes.get(code, 0) + 1
    return {
        "entries": sum(codes.values()),
        "codes": dict(sorted(codes.items())),
        "porcelain_z_sha256": hashlib.sha256(raw).hexdigest(),
    }


def track_identity() -> dict[str, object]:
    """@brief 现场重算 tracks.json、registry 与每个 Track 的摘要。

    @return 字段字典。
    """
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from tcenv.identity import fingerprint
    from tcenv.track.registry import load_track_registry

    registry = load_track_registry()
    return {
        "path": TRACKS_PATH,
        "sha256": sha256(ROOT / TRACKS_PATH),
        "registry_digest": fingerprint(registry.document()),
        "track_digests": {track.track_id: fingerprint(track.document()) for track in registry.tracks},
    }


def worktree_files() -> dict[str, str]:
    """@brief 采集完整工作树清单（排除生成/派生目录与自身）。

    @return 路径 → sha256。
    """
    excluded = {str((P1 / name).relative_to(ROOT)) for name in (START_MANIFEST.name,)}
    excluded |= set(WORKTREE_EXCLUDED_FILES)
    files: dict[str, str] = {}
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file() or path.is_symlink():
            continue
        relative = path.relative_to(ROOT)
        parts = set(relative.parts[:-1])
        if parts & set(WORKTREE_EXCLUDED_DIRS):
            continue
        if relative.suffix in WORKTREE_EXCLUDED_SUFFIXES:
            continue
        name = str(relative)
        if name in excluded:
            continue
        files[name] = sha256(path)
    if not files:
        raise SystemExit("the worktree manifest is empty")
    return files


def _manifest_text(header: list[str], files: dict[str, str]) -> str:
    """@brief 生成 ``<sha>  <path>`` 清单文本。

    @param header 说明行。
    @param files 路径 → 摘要。
    @return 文本。
    """
    return "\n".join([*header, *(f"{digest}  {name}" for name, digest in files.items())]) + "\n"


def manifest_text(files: dict[str, str]) -> str:
    """@brief 生成验收表面 manifest 文本（带说明头）。

    @param files 路径 → 摘要。
    @return 文本。
    """
    header = [
        f"# P1 stage-start baseline ({EPOCH}) - generated by P1/baseline_epoch.py",
        "# enforced contract surface at the real P1 start (see P1.md); re-baselining is a new epoch",
        f"# files {len(files)}",
    ]
    return _manifest_text(header, files)


def start_manifest_text(files: dict[str, str]) -> str:
    """@brief 生成完整工作树 manifest 文本。

    @param files 路径 → 摘要。
    @return 文本。
    """
    header = [
        f"# P1 stage-start FULL worktree manifest ({EPOCH}) - generated by P1/baseline_epoch.py",
        "# start-time evidence only: P1 adds files, so this snapshot is not required to stay unchanged",
        f"# files {len(files)}",
    ]
    return _manifest_text(header, files)


def p0_cross_check(files: dict[str, str]) -> dict[str, object]:
    """@brief 与 P0 验收记录逐条比对（P0 记录本身的字节由 P0/verify.py 钉住）。

    @param files 本阶段覆盖文件的摘要。
    @return 交叉证据。
    @raises SystemExit P0 记录缺失或与当前工作树矛盾。
    """
    if not P0_BASELINE.is_file():
        raise SystemExit(f"the P0 acceptance record is missing: {P0_BASELINE}")
    document = json.loads(P0_BASELINE.read_text(encoding="utf-8"))
    p0_files = document.get("files")
    if not isinstance(p0_files, dict) or not p0_files:
        raise SystemExit("the P0 acceptance record has no file digests")
    compared = sorted(set(files) & set(p0_files))
    mismatched = [name for name in compared if files[name] != p0_files[name]]
    if mismatched:
        raise SystemExit(
            f"the worktree does not reproduce the P0-pinned bytes for {mismatched}; P1 must start from "
            "the P0-completed tree, not from a rewritten one"
        )
    return {
        "record": str(P0_BASELINE.relative_to(ROOT)),
        "record_sha256": sha256(P0_BASELINE),
        "p0_covered_files": len(p0_files),
        "compared_files": len(compared),
        "mismatches": [],
        "note": (
            "every file this baseline shares with P0's record has the P0-pinned bytes; P0's own record "
            "digest is pinned inside P0/verify.py, so both records cannot be regenerated together unnoticed"
        ),
    }


def snapshot(created_at: str) -> tuple[dict[str, object], dict[str, str], dict[str, str]]:
    """@brief 采集当前工作树的阶段起点记录（尚未写盘）。

    @param created_at UTC 时间戳。
    @return ``(record, contract files, worktree files)``。
    """
    missing = [name for name in CONTRACT_SURFACE if not (ROOT / name).is_file()]
    if missing:
        raise SystemExit(f"contract surface file is missing: {missing}")
    duplicate = sorted({name for name in CONTRACT_SURFACE if CONTRACT_SURFACE.count(name) > 1})
    if duplicate:
        raise SystemExit(f"duplicate contract surface path: {duplicate}")
    files = {name: sha256(ROOT / name) for name in CONTRACT_SURFACE}
    worktree = worktree_files()
    cross = p0_cross_check(files)
    head, resolves = head_commit()
    record: dict[str, object] = {
        "schema_id": SCHEMA_ID,
        "epoch": EPOCH,
        "baseline_kind": BASELINE_KIND,
        "stage": STAGE,
        "created_at_utc": created_at,
        "note": NOTE,
        "is_p1_start_baseline": True,
        "p0_status_at_start": "completed",
        "head": head,
        "head_resolves_to_commit": resolves,
        "git_status": git_status(),
        "tracks": track_identity(),
        "contract_surface": {
            "files": len(files),
            "groups": {name: list(paths) for name, paths in GROUPS.items()},
            "manifest": str(BASELINE_MANIFEST.relative_to(ROOT)),
            "manifest_sha256": hashlib.sha256(manifest_text(files).encode("utf-8")).hexdigest(),
            "enforced": (
                "every listed file is re-hashed from the worktree by P1/verify.py; tracks.json is also "
                "checked against the frozen digest and the registry/Track digests are recomputed"
            ),
        },
        "worktree": {
            "manifest": str(START_MANIFEST.relative_to(ROOT)),
            "manifest_sha256": hashlib.sha256(start_manifest_text(worktree).encode("utf-8")).hexdigest(),
            "files": len(worktree),
            "excluded_dirs": list(WORKTREE_EXCLUDED_DIRS),
            "role": (
                "full start-time worktree snapshot (HEAD + every file's SHA-256); it is evidence of the "
                "starting point and is not enforced later, because P1 adds files"
            ),
        },
        "files": files,
        "p0_cross_check": cross,
        "observations": {
            "purpose": "start-time evidence for the documents that change while P1 is implemented",
            "documents": {name: sha256(ROOT / name) for name in OBSERVED_DOCUMENTS},
        },
    }
    return record, files, worktree


def write(created_at: str, *, force: bool) -> None:
    """@brief 写 manifest 与 JSON 记录。

    @param created_at UTC 时间戳。
    @param force 覆盖已有基线（只用于刻意的新 epoch）。
    """
    if BASELINE_JSON.exists() and not force:
        raise SystemExit(
            f"{BASELINE_JSON.relative_to(ROOT)} already exists: taking another baseline is a NEW epoch and "
            "requires the whole P1 acceptance to be redone (and a new pin in verify.py), not a silent "
            "refresh. Pass --force only for a deliberate new epoch."
        )
    record, files, worktree = snapshot(created_at)
    BASELINE_MANIFEST.write_text(manifest_text(files), encoding="utf-8")
    START_MANIFEST.write_text(start_manifest_text(worktree), encoding="utf-8")
    BASELINE_JSON.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    digest = sha256(BASELINE_JSON)
    print(f"OK: wrote {BASELINE_JSON.relative_to(ROOT)} (epoch {EPOCH}, {len(files)} files)")
    print(f"OK: wrote {BASELINE_MANIFEST.relative_to(ROOT)}")
    print(f"OK: wrote {START_MANIFEST.relative_to(ROOT)} ({len(worktree)} worktree files)")
    print(f"OK: pin this digest in verify.py: ACCEPTANCE_BASELINE_SHA256 = \"{digest}\"")


def check() -> int:
    """@brief 独立复核：记录与当前工作树、manifest、Track 摘要是否一致。

    This is deliberately a second implementation of the comparison (``verify.py`` has its own), so
    a reviewer can run it without trusting the verifier that enforces the gate.

    @return 进程退出码（0 表示一致）。
    """
    problems: list[str] = []
    record = json.loads(BASELINE_JSON.read_text(encoding="utf-8"))
    files = record["files"]
    if record["epoch"] != EPOCH:
        problems.append(f"epoch is {record['epoch']!r}, expected {EPOCH!r}")
    if record.get("is_p1_start_baseline") is not True:
        problems.append("the record does not claim to be the real P1 stage start")
    if set(files) != set(CONTRACT_SURFACE):
        problems.append(
            f"file set differs: missing={sorted(set(CONTRACT_SURFACE) - set(files))} "
            f"extra={sorted(set(files) - set(CONTRACT_SURFACE))}"
        )
    drifted: list[str] = []
    for name, digest in files.items():
        path = ROOT / name
        if not path.is_file():
            problems.append(f"missing: {name}")
        elif sha256(path) != digest:
            drifted.append(name)
    # P1 declares exactly one change to the P1-start surface: the opt-in ``source_preflight``
    # extension in P0's dispatch resolver.  It is not hidden here — the drift is printed, and any
    # other changed file is a failure.  ``verify.py`` enforces the same allow-list and re-pins the
    # file in the stage manifest, so allowing it here does not widen the gate.
    undeclared = sorted(set(drifted) - set(ALLOWED_DRIFT))
    problems.extend(f"drift: {name}" for name in undeclared)
    manifest = BASELINE_MANIFEST.read_text(encoding="utf-8")
    expected_manifest = manifest_text(files)
    if manifest != expected_manifest:
        problems.append("acceptance-baseline.sha256 is not in sync with the baseline file map")
    if hashlib.sha256(expected_manifest.encode("utf-8")).hexdigest() != record["contract_surface"]["manifest_sha256"]:
        problems.append("the recorded manifest digest does not cover the recorded file map")
    start = START_MANIFEST.read_text(encoding="utf-8")
    if hashlib.sha256(start.encode("utf-8")).hexdigest() != record["worktree"]["manifest_sha256"]:
        problems.append("the recorded full-worktree manifest digest does not cover start-worktree.sha256")
    start_entries: dict[str, str] = {}
    for line in start.splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        digest, name = line.split("  ", 1)
        start_entries[name] = digest
    mismatched = sorted(name for name, digest in files.items() if start_entries.get(name) != digest)
    if mismatched:
        problems.append(f"the full worktree manifest disagrees with the contract surface: {mismatched}")
    if P0_BASELINE.is_file():
        p0 = json.loads(P0_BASELINE.read_text(encoding="utf-8"))
        shared = sorted(set(files) & set(p0.get("files") or {}))
        drift = [name for name in shared if files[name] != p0["files"][name]]
        if drift:
            problems.append(f"the P0-pinned bytes do not reproduce: {drift}")
        if not shared:
            problems.append("no file is shared with the P0 acceptance record")
    else:
        problems.append("the P0 acceptance record is missing")
    tracks = track_identity()
    if tracks["sha256"] != record["tracks"]["sha256"]:
        problems.append("tracks.json changed")
    if tracks["registry_digest"] != record["tracks"]["registry_digest"]:
        problems.append("the Track registry digest changed")
    if tracks["track_digests"] != record["tracks"]["track_digests"]:
        problems.append("a Track digest changed")
    head, resolves = head_commit()
    if head != record["head"] or resolves != record["head_resolves_to_commit"]:
        problems.append(f"HEAD moved: {head} (recorded {record['head']})")
    if problems:
        print(f"FAIL: P1 stage-start baseline ({EPOCH})")
        for problem in problems:
            print(f"  - {problem}")
        return 1
    print(f"OK: P1 stage-start baseline ({EPOCH}) matches the worktree ({len(files)} files)")
    for name in drifted:
        print(f"  - declared P1 extension (re-pinned by P1/manifest.sha256): {name}")
    print(f"  - head {record['head']} (resolves to a commit: {record['head_resolves_to_commit']})")
    print(
        f"  - git status entries {record['git_status']['entries']} "
        f"{record['git_status']['codes']} porcelain_z_sha256 {record['git_status']['porcelain_z_sha256']}"
    )
    print(f"  - full worktree manifest {record['worktree']['files']} files, {record['worktree']['manifest_sha256'][:16]}…")
    print(
        f"  - tracks.json {record['tracks']['sha256'][:16]}… registry {record['tracks']['registry_digest'][:16]}… "
        f"tracks {sorted(record['tracks']['track_digests'])}"
    )
    print(f"  - P0 cross-check: {record['p0_cross_check']['compared_files']} shared files reproduce the P0-pinned bytes")
    return 0


def main() -> None:
    """@brief 命令行入口。"""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--write", action="store_true", help="record the stage-start baseline for the current worktree")
    parser.add_argument("--force", action="store_true", help="overwrite: only for a deliberate NEW epoch")
    parser.add_argument("--created-at", default="", help="UTC timestamp to record (default: now)")
    args = parser.parse_args()
    if args.write:
        created_at = args.created_at.strip()
        if not created_at:
            from datetime import datetime, timezone

            created_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        write(created_at, force=args.force)
        return
    raise SystemExit(check())


if __name__ == "__main__":
    main()
