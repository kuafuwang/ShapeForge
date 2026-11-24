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

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
_G0_LEGACY = "docs/架构/跨硬件算子生成分工-G0"
_G0_LIVE = "tests/support/golden/shared/跨硬件算子生成分工-G0"
_PORTFOLIO_LEGACY = "docs/架构/shape-portfolio-阶段验收"
_PORTFOLIO_LIVE = "tests/support/golden/shared/shape-portfolio-阶段验收"
_HW_LEGACY = "docs/架构/跨硬件算子生成分工-阶段验收"
_HW_LIVE = "tests/support/golden/shared/跨硬件算子生成分工-阶段验收"
# The closed MT1b resolution stores this incident path. The package itself is
# retired; the CPU test still checks these observation bytes from the fixture.
_B_EVIDENCE_LEGACY = _HW_LEGACY + "/B-evidence-rerecord-2026-11-07"
_B_EVIDENCE_FIXTURE = (
    "tests/support/golden/shared/acceptance-epochs/fixtures/b-evidence-rerecord-2026-11-07"
)
_DSL_LEGACY = "docs/架构/多DSL-Track-阶段验收"
_DSL_LIVE = "tests/support/golden/shared/多DSL-Track-阶段验收"
_FIXTURE = "tests/support/golden/shared/acceptance-epochs/fixtures"
# Longest prefix first. A shorter name must not swallow MT0-isolation or MT6-registry-live.
_RETIRED_PACKAGE_FIXTURES = (
    ("docs/架构/多DSL-Track-阶段验收/MT0-isolation-profile", f"{_FIXTURE}/mt0-isolation-profile"),
    ("docs/架构/多DSL-Track-阶段验收/MT0-isolation", f"{_FIXTURE}/mt0-isolation"),
    ("docs/架构/多DSL-Track-阶段验收/MT0-tilelang-probe", f"{_FIXTURE}/mt0-tilelang-probe"),
    ("docs/架构/多DSL-Track-阶段验收/MT0", f"{_FIXTURE}/mt0"),
    ("docs/架构/多DSL-Track-阶段验收/MT1b", f"{_FIXTURE}/mt1b"),
    ("docs/架构/多DSL-Track-阶段验收/MT1a", f"{_FIXTURE}/mt1a"),
    ("docs/架构/多DSL-Track-阶段验收/MT6-registry-live.json", f"{_FIXTURE}/mt6-registry-live.json"),
    ("docs/架构/多DSL-Track-阶段验收/MT6", f"{_FIXTURE}/mt6"),
    ("docs/架构/多DSL-Track-阶段验收/MT4", f"{_FIXTURE}/mt4"),
)


def _move_b_evidence(text: str) -> str | None:
    """@brief 把已退役的 B-evidence 观察路径改到逐字节副本。"""
    if text == _B_EVIDENCE_LEGACY or text.startswith(_B_EVIDENCE_LEGACY + "/"):
        return _B_EVIDENCE_FIXTURE + text[len(_B_EVIDENCE_LEGACY):]
    return None


def _move_retired_package(text: str) -> str | None:
    """@brief 把本轮退役包的冻结 docs 路径改到逐字节副本。"""
    for legacy, live in _RETIRED_PACKAGE_FIXTURES:
        if text == legacy or text.startswith(legacy + "/"):
            return live + text[len(legacy):]
    return None


def _rewrite_legacy_path(path: Path) -> Path:
    """@brief 把按旧目录深度拼出来的绝对路径改到现行夹具。"""
    posix = path.as_posix()
    marker = "/" + _B_EVIDENCE_LEGACY
    index = posix.find(marker)
    if index >= 0:
        return ROOT / _move_b_evidence(posix[index + 1:])
    for legacy, live in _RETIRED_PACKAGE_FIXTURES:
        marker = f"/{legacy}"
        index = posix.find(marker)
        if index < 0:
            continue
        end = index + len(marker)
        if end != len(posix) and posix[end] != "/":
            continue
        suffix = posix[end:].lstrip("/")
        return ROOT / live / suffix if suffix else ROOT / live
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
    moved = _move_b_evidence(text)
    if moved is not None:
        return moved
    moved = _move_retired_package(text)
    if moved is not None:
        return moved
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
