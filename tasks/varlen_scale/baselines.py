"""varlen_scale baselines: one runner, the framework-native elementwise scale.

A baseline is a timing comparison object, never the correctness oracle.  ``BaselineRunner`` passes
exactly ``task.contract.inputs`` as positional arguments, so ``run`` takes ``x`` and ``cu_seqlens``;
the scale is elementwise, so ``cu_seqlens`` is accepted for ABI completeness but not read.  The final
``.to(x.dtype)`` is the one and only output rounding (``numerics.rounding = single_round_at_output``).
"""

from __future__ import annotations

import torch

SCALE = 3


def run(x, cu_seqlens):
    """@brief Framework-native elementwise scale over the packed varlen batch.

    @param x 声明 dtype 的打包输入（fp16）。
    @param cu_seqlens 分段偏移（int32；elementwise 算子不需要读它）。
    @return 同 dtype 的输出。
    """
    return (SCALE * x.to(torch.float32)).to(x.dtype)


BASELINES = {"torch_mps": run}
