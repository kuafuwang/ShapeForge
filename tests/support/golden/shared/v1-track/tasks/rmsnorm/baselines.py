"""Performance baselines for rmsnorm: the G0-registered vendor/library calls.

A baseline is a timing comparison object, never the correctness oracle.  Each function
receives the framework-native arrays BaselineRunner prepared (mlx array or torch MPS
tensor) and returns the framework-native result.  The call is the one the v1 track
registered, so a v1/v2 timing comparison measures the same comparison object.
"""

from __future__ import annotations

_EPS = 1e-6  # the frozen OpSpec attr default (EPS has no per-instance override)
def mlx(x, w):
    """G0 baseline mlx."""
    import mlx.core as mx
    return mx.fast.rms_norm(x, w, _EPS)

def torch_mps(x, w):
    """G0 baseline torch_mps."""
    import torch
    return torch.nn.functional.rms_norm(x, (x.shape[-1],), w, _EPS)

BASELINES = {
    "mlx": mlx,
    "torch_mps": torch_mps,
}
