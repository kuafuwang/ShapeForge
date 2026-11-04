"""Performance baselines for conv2d: the vendor/library calls the v1 track registered.

A baseline is a timing comparison object, never the correctness oracle.  Functions receive
the framework-native arrays BaselineRunner prepared and return the framework-native result.
"""

from __future__ import annotations

def mlx(x, w):
    """v1 baseline mlx (``mx.conv2d``)."""
    import mlx.core as mx
    return mx.conv2d(x, w)


def torch_mps(x, w):
    """v1 baseline torch_mps (NHWC -> NCHW -> conv2d -> NHWC)."""
    import torch.nn.functional as F
    return F.conv2d(x.permute(0, 3, 1, 2), w.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)

BASELINES = {"mlx": mlx, "torch_mps": torch_mps}
