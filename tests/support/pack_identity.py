"""Exact task-pack identity for the v2 standard entry.

Two different digests are involved and neither may be accepted on shape alone:

* ``contract.pack_digest`` is the *track* pack digest over the repo-relative task files
  (``tcenv.track.pack_digest(contract.pack_files)``);
* ``TaskPackClosure.digest`` is the deployment closure digest over the task files plus the
  hardware-neutral code import closure (``tcenv.taskcontract.pack``).

They are different domains, so a test cannot compare them with each other and must not fall
back to ``len(digest) == 64``.  This helper recomputes each one from its own inputs: the
contract digest through the track's canonical pack digest, and the closure digest through the
documented RFC 8785 manifest (task side: task-relative path; code side: repo-relative path).
"""

from __future__ import annotations

import hashlib
from pathlib import Path

#: Standard task files in registration order (``tcenv.taskcontract.pack.STANDARD_TASK_FILES``).
STANDARD_TASK_FILES: tuple[str, ...] = (
    "opspec.json",
    "reference.py",
    "reference.json",
    "baselines.json",
    "baselines.py",
    "display.json",
    "known_good.metal",
)
PACK_SCHEMA_ID = "tcenv.task.standard-pack.1"
PACK_SCHEMA_VERSION = 1
FORBIDDEN_CODE_ROOTS = frozenset({"backends", "taic", "tasks", "tests", "observe", "experiments"})


def assert_task_pack_identity(contract, closure, *, task_dir: Path | str, repo_root: Path | str) -> None:
    """@brief 精确核对 contract pack 与 deployment closure（文件集合 + 各自摘要域）。

    @param contract v2 ``TaskContract``。
    @param closure ``task_pack_closure`` 结果。
    @param task_dir 任务目录。
    @param repo_root 仓库根。
    @raises AssertionError 任一摘要/集合不对。
    """
    from tcenv.identity import canonical_bytes
    from tcenv.track import pack_digest

    root = Path(task_dir).resolve()
    repo = Path(repo_root).resolve()
    expected_task_files = tuple(name for name in STANDARD_TASK_FILES if (root / name).is_file())
    assert expected_task_files, f"{root}: no standard task files"
    assert tuple(closure.task_files) == expected_task_files, (
        f"{root.name}: closure task files {closure.task_files} != standard files {expected_task_files}"
    )

    expected_pack = tuple(f"tasks/{root.name}/{name}" for name in expected_task_files)
    assert tuple(contract.pack_files) == expected_pack, (
        f"{root.name}: contract pack files {contract.pack_files} != {expected_pack}"
    )
    assert contract.pack_digest == pack_digest(contract.pack_files), (
        f"{root.name}: contract.pack_digest is not the track pack digest of its own file set"
    )

    entries: list[dict] = []
    for side, paths, base in (("task", closure.task_files, root), ("code", closure.code_files, repo)):
        assert paths, f"{root.name}: the {side} side of the closure is empty"
        for relative in paths:
            path = base / relative
            assert path.is_file() and not path.is_symlink(), f"{root.name}: {side} pack file {relative} is not a regular file"
            data = path.read_bytes()
            entries.append({
                "side": side,
                "path": str(relative),
                "size": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            })
    manifest = {
        "schema_id": PACK_SCHEMA_ID,
        "schema_version": PACK_SCHEMA_VERSION,
        "entries": sorted(entries, key=lambda item: (item["side"], item["path"])),
    }
    assert closure.digest == hashlib.sha256(canonical_bytes(manifest)).hexdigest(), (
        f"{root.name}: closure.digest does not recompute from its file sets"
    )

    for relative in closure.code_files:
        parts = Path(relative).parts
        assert parts and parts[0] == "tcenv", f"{root.name}: code pack escaped tcenv/: {relative}"
        assert len(parts) > 1 and parts[1] not in FORBIDDEN_CODE_ROOTS, (
            f"{root.name}: code pack followed a forbidden root: {relative}"
        )
    for entry in ("tcenv/taskcontract/__init__.py", "tcenv/opspec/__init__.py"):
        assert entry in closure.code_files, f"{root.name}: code pack lost its entry module {entry}"


__all__ = ["assert_task_pack_identity"]
