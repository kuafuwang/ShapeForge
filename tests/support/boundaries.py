"""The *current* migration boundary: the frozen G0 boundary plus each later batch's own file.

G0's ``migration-boundary.json`` is closed evidence and is never edited.  A later migration
batch (B1, B2, ...) writes its own ``boundary.json`` next to its own evidence, so "which tasks
are still on the v1 path" is always derivable without touching a completed phase:

    preserved now = G0 preserved_v1_out_of_scope - every later batch's migrated tasks
    migrated now  = G0 migration_candidates + every later batch's migrated tasks

Test support reads this module instead of hard-coding a list, so migrating a task without
adding its batch boundary fails the inventory tests instead of passing silently.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
G0_BOUNDARY = ROOT / "docs" / "架构" / "跨硬件算子生成分工-G0" / "migration-boundary.json"
ACCEPTANCE = ROOT / "docs" / "架构" / "跨硬件算子生成分工-阶段验收"


def _g0() -> dict:
    """@brief 读取冻结的 G0 边界。

    @return 边界文档。
    """
    return json.loads(G0_BOUNDARY.read_text(encoding="utf-8"))


def later_batches() -> list[tuple[str, dict]]:
    """@brief 按阶段名顺序读出 G0 之后各批次自己的边界文件。

    @return ``(阶段名, 边界文档)`` 列表。
    """
    batches: list[tuple[str, dict]] = []
    if ACCEPTANCE.is_dir():
        for path in sorted(ACCEPTANCE.glob("B*/boundary.json")):
            batches.append((path.parent.name, json.loads(path.read_text(encoding="utf-8"))))
    return batches


def migrated_tasks() -> tuple[str, ...]:
    """@brief 当前已在 v2 标准入口的题名（G0 的 16 题 + 后续批次）。

    @return 题名元组，顺序为先 G0 后批次。
    """
    names = [str(item["task"]) for item in _g0()["migration_candidates"]]
    for _phase, document in later_batches():
        names.extend(str(item["task"]) for item in document.get("migrated", []))
    return tuple(names)


def preserved_v1() -> tuple[str, ...]:
    """@brief 当前仍走 v1 装配的保留项（含 ``sem_int dynamic family`` 这一族条目）。

    @return 保留项元组。
    """
    migrated = set()
    for _phase, document in later_batches():
        migrated.update(str(item["task"]) for item in document.get("migrated", []))
    return tuple(name for name in _g0()["preserved_v1_out_of_scope"] if name not in migrated)
