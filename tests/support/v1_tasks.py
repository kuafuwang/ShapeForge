"""The tasks a v1-mechanism test may assemble, and the one place that names them.

A5 moved the 16 G0 migration tasks to the v2 standard entry and deleted their v1 assembly in
the same change, so ``tasks.get_task(<migrated name>)`` now refuses instead of falling back.
A test that exercises a *v1 mechanism* (registry, resume, publication, schema 4, the remote job
engine, timeouts, ...) therefore has to name a task the migration boundary keeps on v1
(``migration-boundary.json.preserved_v1_out_of_scope``) instead of reviving a migrated task
through a fallback, a second registry or an archived prompt.

This module exists so those call sites share one name instead of scattering literals, and so
the choice is documented where it is made.  It is test support only: nothing in production
imports it, and it never builds a task the v1 registry does not already own.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BOUNDARY = ROOT / "docs/架构/跨硬件算子生成分工-G0/migration-boundary.json"
from tests.support.boundaries import preserved_v1 as _current_preserved

_PRESERVED = tuple(str(name) for name in _current_preserved())

#: The default v1 specimen.  A v1-mechanism test wants "a real v1 task", not this task's
#: mathematics, so a test that also asserts a shape-dependent fact must say so explicitly and
#: use its own task.  B1 moved rope/gemv to v2 (default became conv2d); B2 moved conv2d/lion,
#: so the default is flash_attention — the remaining static task with registered baselines and
#: a variant family.  The boundary files remain the only inventory.
V1_TASK = "flash_attention"
#: The static tasks the boundary keeps on the v1 path.  This is derived from the frozen
#: migration boundary, not a second list: adding/removing a preserved task is a boundary edit.
V1_TASKS: tuple[str, ...] = tuple(name for name in _PRESERVED if not name.startswith("sem_int "))
#: The dynamic ``sem_int`` family has no static directory and stays v1.  The boundary names
#: the family once; these are representative seed names for tests that need a concrete task.
V1_DYNAMIC: tuple[str, ...] = ("sem_int_5", "sem_int_3", "sem_int_007")
assert V1_TASK in V1_TASKS, "V1_TASK must be a preserved v1 task"
assert "sem_int dynamic family" in _PRESERVED, "the boundary no longer names the dynamic sem_int family"


def v1_kernel_name(name: str = V1_TASK, dtype: str = "fp16", size: str = "small") -> str:
    """@brief 保留 v1 题在 v1 装配下的内核函数名。

    测试夹具常需要一段能被该题 ABI 接受的候选源码；内核名必须来自同一份 v1 装配，不能在测试里另写
    一个字面量再指望它恰好匹配。

    @param name 任务名（必须是 ``preserved_v1_out_of_scope`` 之一）。
    @param dtype 元素类型短名。
    @param size 问题规模名。
    @return 契约声明的内核函数名。
    """
    return v1_task(name, dtype, size).contract.kernel_name


def v1_task(name: str = V1_TASK, dtype: str = "fp16", size: str = "small"):
    """@brief 用 v1 registry 装配一道保留下来的 v1 题。

    @param name 任务名（必须是 ``preserved_v1_out_of_scope`` 之一）。
    @param dtype 元素类型短名。
    @param size 问题规模名。
    @return ``tcenv.task.Task``。
    @raises AssertionError 当名字其实是迁移题（测试自身写错了题名）。
    """
    from tasks import get_task

    dynamic = name in V1_DYNAMIC or (
        name.startswith("sem_int_") and "sem_int dynamic family" in _PRESERVED
    )
    assert name in V1_TASKS or dynamic, f"{name!r} is not a preserved v1 task"
    return get_task(name, dtype, size)
