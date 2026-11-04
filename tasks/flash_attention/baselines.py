"""Performance baselines for flash_attention: the vendor/library calls the v1 track registered.

A baseline is a timing comparison object, never the correctness oracle.  Functions receive
the framework-native arrays BaselineRunner prepared and return the framework-native result.
"""

from __future__ import annotations

_SM_SCALE = 0.125  # the frozen OpSpec attr default (1/sqrt(D) with D = 64)


def mlx(q, k, v):
    """v1 baseline mlx (``mx.fast.scaled_dot_product_attention``)."""
    import mlx.core as mx
    return mx.fast.scaled_dot_product_attention(q, k, v, scale=_SM_SCALE)


def torch_mps(q, k, v):
    """v1 baseline torch_mps (``F.scaled_dot_product_attention``)."""
    import torch.nn.functional as F
    return F.scaled_dot_product_attention(q, k, v, scale=_SM_SCALE)

BASELINES = {"mlx": mlx, "torch_mps": torch_mps}
