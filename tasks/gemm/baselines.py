"""Performance baselines for gemm: the G0-registered vendor/library calls.

A baseline is a timing comparison object, never the correctness oracle.  Each function
receives the framework-native arrays BaselineRunner prepared (mlx array or torch MPS
tensor) and returns the framework-native result.  The call is the one the v1 track
registered, so a v1/v2 timing comparison measures the same comparison object.
"""

from __future__ import annotations

def mlx(a, b):
    """G0 baseline mlx."""
    import mlx.core as mx
    return a @ b

def torch_mps(a, b):
    """G0 baseline torch_mps."""
    import torch
    return torch.matmul(a, b)

BASELINES = {
    "mlx": mlx,
    "torch_mps": torch_mps,
}
