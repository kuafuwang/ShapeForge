"""Performance baselines for gemv: the vendor/library calls the v1 track registered.

A baseline is a timing comparison object, never the correctness oracle.  Each function
receives the framework-native arrays BaselineRunner prepared (mlx array or torch MPS
tensor) and returns the framework-native result.  The call is the one the v1 track
registered, so a v1/v2 timing comparison measures the same comparison object.
"""

from __future__ import annotations

def mlx(a, x):
    """v1 baseline mlx (``matmul``)."""
    return a @ x


def torch_mps(a, x):
    """v1 baseline torch_mps (``torch.mv``)."""
    import torch
    return torch.mv(a, x)

BASELINES = {"mlx": mlx, "torch_mps": torch_mps}
