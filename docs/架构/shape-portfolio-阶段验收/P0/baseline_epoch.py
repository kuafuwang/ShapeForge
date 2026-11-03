#!/usr/bin/env python3
"""P0 acceptance-epoch baseline: the real start of a NEW acceptance stage (blocker 12, route 2).

实施计划 §P0 requires the *start* of the stage to be pinned (full HEAD, worktree manifest,
SHA-256).  P0's original start was never recorded, so that snapshot cannot be rebuilt afterwards.
P0.md §7.2 allows exactly two dispositions: an explicit written waiver by the plan owner, or a real
new baseline with the P0 acceptance redone as a new stage.  No waiver was given, so this module
implements the second route: it snapshots the contract surface of the **current** worktree, names
the epoch, and states explicitly that it is **not** the missing P0-start baseline.

What is enforced (by ``verify.py``, which re-hashes the same paths itself):

* every P0 production/test file and every direct contract dependency, byte for byte;
* ``backends/metal/tracks.json`` plus the freshly recomputed registry/Track digests.

What is *not* part of the enforced surface: the acceptance documents themselves (``P0.md``,
``verify.py``, ``manifest.sha256``, the plan).  They necessarily change while blocker 12 is closed,
so they are recorded under ``observations`` as epoch-start evidence and stay unenforced.

This record is evidence, not a gate.  Its own bytes are pinned in ``verify.py``; regenerating it to
hide drift therefore requires editing the verifier and the manifest too, instead of silently
refreshing a snapshot.  A second baseline is a new epoch and must redo the whole acceptance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
P0 = Path(__file__).resolve().parent
BASELINE_JSON = P0 / "acceptance-baseline.json"
BASELINE_MANIFEST = P0 / "acceptance-baseline.sha256"

SCHEMA_ID = "tcenv.p0.acceptance-baseline.1"
EPOCH = "p0-reacceptance-baseline-1"
BASELINE_KIND = "acceptance-epoch-start"
STAGE = "P0"
TRACKS_PATH = "backends/metal/tracks.json"

#: The P0 production surface: every file P0 created plus the one backend file it modified.
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

#: Contract code that P0 does not own but that its contracts are executed against.  Drift here
#: would change what the P0 contracts mean, so the epoch pins it as well.
CONTRACT_DEPENDENCIES: tuple[str, ...] = (
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

#: The P0 test surface: the three fixed-node files, their fixture module and the test harness.
P0_TESTS: tuple[str, ...] = (
    "tests/shared/test_shape_portfolio_domain.py",
    "tests/shared/test_shape_portfolio_candidate.py",
    "tests/shared/test_shape_portfolio_dispatch.py",
    "tests/support/shape_portfolio.py",
    "tests/support/v2_tasks.py",
    "tests/support/README.md",
    "tests/conftest.py",
)

#: The safe regressions the acceptance runs (files, not counts).
SAFE_REGRESSION_SURFACE: tuple[str, ...] = (
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

GROUPS: dict[str, tuple[str, ...]] = {
    "p0_production": P0_PRODUCTION,
    "contract_dependencies": CONTRACT_DEPENDENCIES,
    "p0_tests": P0_TESTS,
    "safe_regression": SAFE_REGRESSION_SURFACE,
}

#: Everything the epoch enforces, in a stable order.  ``tracks.json`` is added explicitly: it is
#: the Track registry source and has its own named section in the record.
CONTRACT_SURFACE: tuple[str, ...] = P0_PRODUCTION + CONTRACT_DEPENDENCIES + P0_TESTS + SAFE_REGRESSION_SURFACE + (
    TRACKS_PATH,
)

#: Documents that change *while closing blocker 12*.  Recorded as epoch-start evidence only.
OBSERVED_DOCUMENTS: tuple[str, ...] = (
    "docs/架构/shape-portfolio-阶段验收/P0/verify.py",
    "docs/架构/shape-portfolio-阶段验收/P0.md",
    "docs/架构/shape-portfolio-阶段验收/P0/manifest.sha256",
    "docs/架构/shape-portfolio-阶段验收/P0/baseline_audit.sha256",
    "docs/架构/shape-portfolio实施计划.md",
)

NOTE = (
    "Real start of a NEW P0 acceptance epoch, taken from the current post-A6 worktree before any "
    "change that closes reviewer blocker 12. This is NOT the missing P0-start baseline: that "
    "snapshot was never recorded and cannot be reconstructed, and the historical gap stays open in "
    "P0.md §7.2. The enforced surface is the P0 production/test contract code plus tracks.json; the "
    "acceptance documents are listed under observations because they change while the blocker is "
    "closed. Regenerating this file is a new epoch and requires the whole acceptance to be redone; "
    "it must never be used to hide later code drift."
)


def sha256(path: Path) -> str:
    """@brief 文件字节 sha256。"""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git(*arguments: str) -> str:
    """@brief 在仓库根目录运行 git 并返回 stdout。"""
    result = subprocess.run(
        ["git", *arguments], cwd=ROOT, capture_output=True, text=True, check=True
    )
    return result.stdout


def head_commit() -> tuple[str, bool]:
    """@brief 当前 HEAD 与其是否解析到一个提交。"""
    head = _git("rev-parse", "HEAD").strip()
    exists = subprocess.run(
        ["git", "rev-parse", "--verify", "--quiet", f"{head}^{{commit}}"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    return head, exists.returncode == 0 and exists.stdout.strip() == head


def git_status() -> dict[str, object]:
    """@brief 工作树状态快照：条目数、状态码计数与 --porcelain=v1 -z 的摘要。"""
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
    """@brief 现场重算 tracks.json、registry 与每个 Track 的摘要。"""
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


def manifest_text(files: dict[str, str]) -> str:
    """@brief 生成 .sha256 manifest 文本（带说明头）。"""
    header = [
        f"# P0 acceptance-epoch baseline ({EPOCH}) - generated by P0/baseline_epoch.py",
        "# real start of a NEW acceptance stage; NOT the missing P0-start baseline (see P0.md §7.2)",
        f"# files {len(files)}",
    ]
    return "\n".join([*header, *(f"{digest}  {name}" for name, digest in files.items())]) + "\n"


def snapshot(created_at: str) -> dict[str, object]:
    """@brief 采集当前工作树的验收 epoch 记录（尚未写盘）。"""
    missing = [name for name in CONTRACT_SURFACE if not (ROOT / name).is_file()]
    if missing:
        raise SystemExit(f"contract surface file is missing: {missing}")
    duplicate = sorted({name for name in CONTRACT_SURFACE if CONTRACT_SURFACE.count(name) > 1})
    if duplicate:
        raise SystemExit(f"duplicate contract surface path: {duplicate}")
    files = {name: sha256(ROOT / name) for name in CONTRACT_SURFACE}
    head, resolves = head_commit()
    return {
        "schema_id": SCHEMA_ID,
        "epoch": EPOCH,
        "baseline_kind": BASELINE_KIND,
        "stage": STAGE,
        "created_at_utc": created_at,
        "note": NOTE,
        "is_p0_start_baseline": False,
        "supersedes_blocker": 12,
        "disposition": (
            "P0.md §7.2 route 2: a real new baseline plus a full P0 acceptance redone as a new stage; "
            "no owner waiver, and no claim that the original start snapshot exists"
        ),
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
                "every listed file is re-hashed from the worktree by P0/verify.py; tracks.json is also "
                "checked against the frozen digest and the registry/Track digests are recomputed"
            ),
        },
        "files": files,
        "observations": {
            "purpose": (
                "epoch-start evidence for the documents that change while blocker 12 is closed; "
                "they are deliberately outside the enforced surface"
            ),
            "documents": {name: sha256(ROOT / name) for name in OBSERVED_DOCUMENTS},
            "baseline_audit": (
                "P0/baseline_audit.sha256 stays at its own audit point and is left unchanged; it is "
                "still not a start baseline"
            ),
        },
    }


def write(created_at: str, *, force: bool) -> None:
    """@brief 写 manifest 与 JSON 记录。"""
    if BASELINE_JSON.exists() and not force:
        raise SystemExit(
            f"{BASELINE_JSON.relative_to(ROOT)} already exists: taking another baseline is a NEW epoch "
            "and requires the whole acceptance to be redone (and a new pin in verify.py), not a silent "
            "refresh. Pass --force only for a deliberate new epoch."
        )
    record = snapshot(created_at)
    BASELINE_MANIFEST.write_text(manifest_text(record["files"]), encoding="utf-8")
    BASELINE_JSON.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    digest = sha256(BASELINE_JSON)
    print(f"OK: wrote {BASELINE_JSON.relative_to(ROOT)} (epoch {EPOCH}, {len(record['files'])} files)")
    print(f"OK: wrote {BASELINE_MANIFEST.relative_to(ROOT)}")
    print(f"OK: pin this digest in verify.py: ACCEPTANCE_BASELINE_SHA256 = \"{digest}\"")


def check() -> int:
    """@brief 独立复核：记录与当前工作树、manifest、Track 摘要是否一致。

    This is intentionally a second implementation of the comparison (``verify.py`` has its own);
    a reviewer can run it without trusting the verifier that enforces the gate.

    @return 进程退出码（0 表示一致）。
    """
    problems: list[str] = []
    record = json.loads(BASELINE_JSON.read_text(encoding="utf-8"))
    files = record["files"]
    if record["epoch"] != EPOCH:
        problems.append(f"epoch is {record['epoch']!r}, expected {EPOCH!r}")
    if set(files) != set(CONTRACT_SURFACE):
        problems.append(
            f"file set differs: missing={sorted(set(CONTRACT_SURFACE) - set(files))} "
            f"extra={sorted(set(files) - set(CONTRACT_SURFACE))}"
        )
    for name, digest in files.items():
        path = ROOT / name
        if not path.is_file():
            problems.append(f"missing: {name}")
        elif sha256(path) != digest:
            problems.append(f"drift: {name}")
    manifest = BASELINE_MANIFEST.read_text(encoding="utf-8")
    expected_manifest = manifest_text(files)
    if manifest != expected_manifest:
        problems.append("manifest.sha256 is not in sync with the baseline file map")
    if hashlib.sha256(expected_manifest.encode("utf-8")).hexdigest() != record["contract_surface"]["manifest_sha256"]:
        problems.append("the recorded manifest digest does not cover the recorded file map")
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
        print(f"FAIL: acceptance-epoch baseline ({EPOCH})")
        for problem in problems:
            print(f"  - {problem}")
        return 1
    print(f"OK: acceptance-epoch baseline ({EPOCH}) matches the worktree ({len(files)} files)")
    print(f"  - head {record['head']} (resolves to a commit: {record['head_resolves_to_commit']})")
    print(
        f"  - git status entries {record['git_status']['entries']} "
        f"{record['git_status']['codes']} porcelain_z_sha256 {record['git_status']['porcelain_z_sha256']}"
    )
    print(
        f"  - tracks.json {record['tracks']['sha256'][:16]}… registry {record['tracks']['registry_digest'][:16]}… "
        f"tracks {sorted(record['tracks']['track_digests'])}"
    )
    return 0


def main() -> None:
    """@brief 命令行入口。"""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--write", action="store_true", help="record the epoch baseline for the current worktree")
    parser.add_argument("--force", action="store_true", help="overwrite: only for a deliberate NEW epoch")
    parser.add_argument("--created-at", default="", help="UTC timestamp to record (default: now)")
    args = parser.parse_args()
    if args.write:
        created_at = args.created_at.strip()
        if not created_at:
            created_at = subprocess.run(
                ["date", "-u", "+%Y-%m-%dT%H:%M:%SZ"], capture_output=True, text=True, check=True
            ).stdout.strip()
        write(created_at, force=args.force)
        return
    raise SystemExit(check())


if __name__ == "__main__":
    main()
