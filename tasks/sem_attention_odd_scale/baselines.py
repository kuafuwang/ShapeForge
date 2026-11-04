"""Performance baselines for sem_attention_odd_scale: the vendor attention calls with the SUM scale mode.

A baseline is a timing comparison object, never the correctness oracle.  The scale is the same
sum_with_log2e form the v1 track registered for both baselines.
"""

from __future__ import annotations

_LOG2E = 1.4426950408889634
_LN2 = 0.6931471805599453
_SM_SCALE = 0.125  # the frozen OpSpec attr default (1/sqrt(D) with D = 64)
_SCALE = (_SM_SCALE + _LOG2E) * _LN2


def mlx(q, k, v):
    """v1 baseline mlx (``mx.fast.scaled_dot_product_attention``, sum_with_log2e)."""
    import mlx.core as mx
    return mx.fast.scaled_dot_product_attention(q, k, v, scale=_SCALE)


def torch_mps(q, k, v):
    """v1 baseline torch_mps (``F.scaled_dot_product_attention``, sum_with_log2e)."""
    import torch.nn.functional as F
    return F.scaled_dot_product_attention(q, k, v, scale=_SCALE)


BASELINES = {"mlx": mlx, "torch_mps": torch_mps}
