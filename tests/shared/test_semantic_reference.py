"""The exact integer reference of the semantic-recovery tasks. No GPU."""

from __future__ import annotations

import torch

from tasks.sem_int.document import opspec_document
from tasks.sem_int.program import BLOCK, SCALAR_K, evaluate, wrap


def test_every_integer_operation_stays_inside_int32():
    """@brief 验证随机整数程序在 int32 边界输入下仍只产出 int32，且同一输入两次求值逐元素相同。

    输入取 int32 的最大值与最小值，正是溢出最敏感的两个取值。
    """
    document = opspec_document(1, "small")["program"]
    x = torch.full((BLOCK,), 2**31 - 1, dtype=torch.int64)
    y = torch.full((BLOCK,), -(2**31), dtype=torch.int64)
    out = evaluate(document, x, y, SCALAR_K)
    assert out.dtype == torch.int32
    assert int(out.min()) >= -(2**31) and int(out.max()) <= 2**31 - 1
    assert torch.equal(out, evaluate(document, x, y, SCALAR_K))


def test_wrap_keeps_the_int32_boundaries():
    """@brief 验证 ``wrap`` 按 int32 环绕：上溢回绕成负数、下溢保持下界，边界值本身不变。"""
    assert wrap(torch.tensor([-(2**31), 2**31, 2**31 - 1, 2**31 + 1])).tolist() == [-(2**31), -(2**31), 2**31 - 1, -(2**31) + 1]
