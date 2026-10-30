"""Semantic-recovery tasks (paper Section 4.1): does the compiler follow the *code*?

The four fixed tasks (``sem_gemm_neg``, ``sem_relu_gelu_comment``, ``sem_gelu_named_relu``,
``sem_attention_odd_scale``) own their semantics in ``tasks/<name>/opspec.json``; the dynamic
integer family ``sem_int_<seed>`` materialises an OpSpec document from the registered
generator in :mod:`tasks.sem_int`.  This module only dispatches names: it constructs no
``Contract`` and holds no second semantic contract.
"""

from __future__ import annotations

from tcenv.task import Task

#: 四个固定错位题。sem_int_<种子> 不在这里，走 tasks.sem_int 的生成器。
FIXED_NAMES = ("sem_gemm_neg", "sem_relu_gelu_comment", "sem_gelu_named_relu", "sem_attention_odd_scale")


def build(name: str, dtype: str, size: str) -> Task:
    """@brief 语义恢复题的总装派发。

    @param name 题目名，``sem_`` 前缀。
    @param dtype 元素类型；整数族只接受登记的别名拼写。
    @param size 尺寸档位，``paper`` 或 ``small``。
    @return 装配好的 ``Task``。
    @raises KeyError 名字不在本模块认识的范围内时抛出。
    """
    if name.startswith("sem_int_"):
        from tasks.sem_int.task import build as build_int

        return build_int(name, dtype, size)
    from ._common import check_dtype

    check_dtype(dtype)
    if name in FIXED_NAMES:
        from importlib import import_module

        module = import_module(f"tasks.{name}.task")
        return module.build(dtype, size)
    raise KeyError(name)
