"""Structural check N-1: ``tasks/**`` no longer owns a Contract or the launch geometry.

The plan forbids task assembly from constructing a ``Contract`` or writing its own
``grid``/``threads``/``num_warps`` facts (§2.1, §4.4, D07).  This is an AST check, not a
text search: comments and docstrings that merely *mention* ``Contract(`` do not count,
and a real call, keyword argument or assignment does (that is the exact failure mode the
discarded draft's regex check could not see).

R1 migrates RMSNorm only.  The remaining modules are an explicit *transitional inventory*:
the test fails when a new offender appears, and R2 must shrink the inventory to empty and
delete the list together with the legacy path.
"""

from __future__ import annotations

import ast
from pathlib import Path

TASKS = Path(__file__).resolve().parents[2] / "tasks"

#: Modules that still construct ``Contract`` themselves.  R1 moved ``rmsnorm`` out and R2 moved
#: every task module *and* deleted the shared elementwise builder, so the inventory is empty:
#: ``tasks/**`` may not build a ``Contract`` (the registered LoweringProfile is the only source).
TRANSITIONAL_CONTRACT_OWNERS: frozenset[str] = frozenset()


def _offences(path: Path) -> list[str]:
    """@brief 扫描一个模块里对启动事实的构造。

    @param path 模块路径。
    @return ``行号:说明`` 形式的违规列表。
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            name = getattr(func, "id", None) or getattr(func, "attr", None)
            if name == "Contract":
                found.append(f"{node.lineno}:Contract(")
            for keyword in node.keywords:
                if keyword.arg in ("grid", "threads"):
                    if name in ("Contract", "replace", "AbiArg2"):
                        found.append(f"{node.lineno}:{keyword.arg}=")
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in ("num_warps", "grid", "threads"):
                    found.append(f"{node.lineno}:{target.id}=")
    return found


def _owners() -> dict[str, list[str]]:
    """@brief 收集所有自己构造 ``Contract`` 的模块。

    @return 仓库相对路径 → 违规列表。
    """
    result: dict[str, list[str]] = {}
    for path in sorted(TASKS.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        offences = [item for item in _offences(path) if item.endswith("Contract(")]
        if offences:
            result[path.relative_to(TASKS.parent).as_posix()] = offences
    return result


def test_rmsnorm_assembly_owns_no_contract_and_no_launch_geometry():
    """@brief RMSNorm 的装配不构造 Contract，也不写 grid/threads/num_warps（注释不算调用）。"""
    for path in sorted((TASKS / "rmsnorm").glob("*.py")):
        assert _offences(path) == [], path


def test_the_remaining_contract_owners_are_the_registered_transitional_inventory():
    """@brief 其他 task 尚未迁移是显式清单；新增违规会让本测试失败，R2 必须清空。"""
    owners = _owners()
    assert frozenset(owners) == TRANSITIONAL_CONTRACT_OWNERS
    assert "tasks/rmsnorm/task.py" not in owners
