"""The exact integer reference of the semantic-recovery tasks. No GPU."""

from __future__ import annotations

import torch

from tasks.semantic import BLOCK, IntProgram, wrap


def test_every_integer_operation_stays_inside_int32():
    program = IntProgram.generate(1)
    x = torch.full((BLOCK,), 2**31 - 1, dtype=torch.int64)
    y = torch.full((BLOCK,), -(2**31), dtype=torch.int64)
    out = program.evaluate(x, y)
    assert out.dtype == torch.int32
    assert int(out.min()) >= -(2**31) and int(out.max()) <= 2**31 - 1
    assert torch.equal(out, program.evaluate(x, y))


def test_wrap_keeps_the_int32_boundaries():
    assert wrap(torch.tensor([-(2**31), 2**31, 2**31 - 1, 2**31 + 1])).tolist() == [-(2**31), -(2**31), 2**31 - 1, -(2**31) + 1]
