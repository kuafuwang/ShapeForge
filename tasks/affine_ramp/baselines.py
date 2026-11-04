"""affine_ramp baselines: one runner, the framework-native ramp product.

A baseline is a timing comparison object, never the correctness oracle.  ``BaselineRunner`` passes
exactly ``task.contract.inputs`` as positional arguments, and a workspace tensor is excluded from
that list, so ``run`` takes ``x`` only and builds the ramp itself (in fp32) -- it can never read the
candidate's workspace.  The final ``.to(x.dtype)`` is the one and only output rounding, which is what
``numerics.rounding = single_round_at_output`` means for this task.

``SEG``/``BASE``/``STEP`` are repeated here on purpose: a baseline may not import the harness.  The
task's own test asserts they equal ``instance.resolved_attrs`` so the two copies cannot drift.
"""

from __future__ import annotations

import torch

SEG, BASE, STEP = 64, 2049, 2


def run(x):
    """@brief Framework-native ramp product.

    @param x 声明 dtype 的输入（fp16/bf16）。
    @return 同 dtype 的输出。
    """
    n = int(x.shape[0])
    ns = (n + SEG - 1) // SEG
    tab = BASE + STEP * torch.arange(ns, dtype=torch.float32, device=x.device)
    index = torch.arange(n, device=x.device) // SEG
    return (tab[index] * x.to(torch.float32)).to(x.dtype)


BASELINES = {"torch_mps": run}
