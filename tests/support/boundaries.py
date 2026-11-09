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
_G0_LEGACY = "docs/架构/跨硬件算子生成分工-G0"
_G0_LIVE = "tests/support/golden/shared/跨硬件算子生成分工-G0"
_PORTFOLIO_LEGACY = "docs/架构/shape-portfolio-阶段验收"
_PORTFOLIO_LIVE = "tests/support/golden/shared/shape-portfolio-阶段验收"
_HW_LEGACY = "docs/架构/跨硬件算子生成分工-阶段验收"
_HW_LIVE = "tests/support/golden/shared/跨硬件算子生成分工-阶段验收"
_DSL_LEGACY = "docs/架构/多DSL-Track-阶段验收"
_DSL_LIVE = "tests/support/golden/shared/多DSL-Track-阶段验收"
G0_BOUNDARY = ROOT / _G0_LIVE / "migration-boundary.json"
ACCEPTANCE = ROOT / "tests" / "support" / "golden" / "shared" / "跨硬件算子生成分工-阶段验收"


def _rewrite_legacy_path(path: Path) -> Path:
    """@brief 把按旧目录深度拼出来的绝对路径改到现行夹具。"""
    posix = path.as_posix()
    for legacy, live in (
        (_G0_LEGACY, _G0_LIVE),
        (_PORTFOLIO_LEGACY, _PORTFOLIO_LIVE),
        (_HW_LEGACY, _HW_LIVE),
        (_DSL_LEGACY, _DSL_LIVE),
    ):
        marker = f"/{legacy}"
        index = posix.find(marker)
        if index < 0:
            continue
        suffix = posix[index + len(marker):].lstrip("/")
        return ROOT / live / suffix if suffix else ROOT / live
    return path


def _relocate_acceptance(text: str) -> str:
    """@brief 把冻结脚本里的旧 docs 验收路径改到现行夹具目录。"""
    for legacy, live in (
        (_G0_LEGACY, _G0_LIVE),
        (_PORTFOLIO_LEGACY, _PORTFOLIO_LIVE),
        (_HW_LEGACY, _HW_LIVE),
        (_DSL_LEGACY, _DSL_LIVE),
    ):
        if text == legacy or text.startswith(legacy + "/"):
            return live + text[len(legacy):]
    return text


class _AcceptanceRoot(Path):
    """@brief 仓库根；把冻结脚本里的旧验收文档路径改到现行夹具目录。"""

    def __truediv__(self, other):
        text = other.as_posix() if isinstance(other, Path) else str(other).replace("\\", "/")
        return Path(self, _relocate_acceptance(text))


def rebind_acceptance_root(module) -> None:
    """@brief 把验收脚本按旧目录深度算出的仓库根改到当前仓库。

    脚本字节是冻结证据。它们从 ``docs/架构/...`` 用固定的 parent 层数找仓库根；
    目录挪到 ``tests/support/golden/shared/`` 之后，这个层数落在 ``tests/support``。
    函数体里写死的旧 G0、shape-portfolio 和多 DSL 验收路径也改到现行夹具目录。调用方在 ``exec_module`` 之后立刻改模块全局。

    @param module 刚载入的验收脚本模块。
    """
    for name, value in list(vars(module).items()):
        if isinstance(value, Path) and name != "ROOT":
            rewritten = _rewrite_legacy_path(value)
            if rewritten != value:
                setattr(module, name, rewritten)
    module.ROOT = _AcceptanceRoot(ROOT)
    g0 = getattr(module, "G0", None)
    if isinstance(g0, Path) and g0.name == "index.json":
        module.G0 = ROOT / _G0_LIVE / "corpus" / "index.json"
    elif isinstance(g0, Path) and g0.name == "跨硬件算子生成分工-G0":
        module.G0 = ROOT / _G0_LIVE
    if hasattr(module, "CORPUS") and isinstance(getattr(module, "G0", None), Path) and module.G0.name == "index.json":
        module.CORPUS = module.G0.parent


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
    if any(name.startswith("sem_int_") for name in migrated):
        # sem_int 动态族按 seed 冻结为固定 v2 目录后，G0 里的族条目也随之退役。
        migrated.add("sem_int dynamic family")
    return tuple(name for name in _g0()["preserved_v1_out_of_scope"] if name not in migrated)
