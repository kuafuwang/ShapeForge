"""Semantic-recovery tasks (paper Section 4.1): does the compiler follow the *code*?

The one preserved fixed task (``sem_attention_odd_scale``) owns its semantics in
``tasks/<name>/opspec.json``; the dynamic integer family ``sem_int_<seed>`` materialises an
OpSpec document from the registered generator in :mod:`tasks.sem_int`.  This module only
dispatches names: it constructs no ``Contract`` and holds no second semantic contract.

A5 moved the other three fixed tasks (``sem_gemm_neg``, ``sem_relu_gelu_comment``,
``sem_gelu_named_relu``) to the v2 standard entry; A6 deleted their v1 name entries here so a
migrated name can only reach the v2 loader (no fallback, no second registry).
"""

from __future__ import annotations

from tcenv.task import Task

#: 仍由 v1 装配的固定语义题。sem_int_<种子> 不在这个名单里，走 tasks.sem_int 的生成器；
#: A5 迁移的三道固定题也不在这里（v2 标准目录是它们唯一的装配入口）。
FIXED_NAMES: tuple[str, ...] = ()  # B4: sem_attention_odd_scale 已迁 v2


def build(name: str, dtype: str, size: str) -> Task:
    """@brief 语义恢复题的总装派发。

    @param name 题目名，``sem_`` 前缀。
    @param dtype 元素类型；整数族只接受登记的别名拼写。
    @param size 尺寸档位，``paper`` 或 ``small``。
    @return 装配好的 ``Task``。
    @raises KeyError 名字不在本模块认识的范围内时抛出。
    """
    if name.startswith("sem_int_"):
        from tcenv.contract_errors import TaskContractError

        raise TaskContractError(
            f"{name}: the sem_int dynamic family is now a set of fixed v2 task directories "
            "(tasks/sem_int_<n>/, with sem_int_007 spelled sem_int_7). Use "
            "tcenv.taskcontract.load_task_contract / discover_standard_tasks; there is no v1 fallback."
        )
    from ._common import check_dtype

    check_dtype(dtype)
    if name in FIXED_NAMES:
        from importlib import import_module

        module = import_module(f"tasks.{name}.task")
        return module.build(dtype, size)
    raise KeyError(name)
