"""The exact integer reference of the semantic-recovery tasks. No GPU."""

from __future__ import annotations

import torch

from tasks.semantic import BLOCK, IntProgram, wrap


def test_every_integer_operation_stays_inside_int32():
    """@brief 验证随机整数程序在 int32 边界输入下仍只产出 int32，且同一输入两次求值逐元素相同。

    输入取 int32 的最大值与最小值，正是溢出最敏感的两个取值。
    """
    program = IntProgram.generate(1)
    x = torch.full((BLOCK,), 2**31 - 1, dtype=torch.int64)
    y = torch.full((BLOCK,), -(2**31), dtype=torch.int64)
    out = program.evaluate(x, y)
    assert out.dtype == torch.int32
    assert int(out.min()) >= -(2**31) and int(out.max()) <= 2**31 - 1
    assert torch.equal(out, program.evaluate(x, y))


def test_wrap_keeps_the_int32_boundaries():
    """@brief 验证 ``wrap`` 按 int32 环绕：上溢回绕成负数、下溢保持下界，边界值本身不变。"""
    assert wrap(torch.tensor([-(2**31), 2**31, 2**31 - 1, 2**31 + 1])).tolist() == [-(2**31), -(2**31), 2**31 - 1, -(2**31) + 1]
