"""Performance baselines for rope: the vendor/library calls the v1 track registered.

A baseline is a timing comparison object, never the correctness oracle.  Each function
receives the framework-native arrays BaselineRunner prepared (mlx array or torch MPS
tensor) and returns the framework-native result.  The call is the one the v1 track
registered, so a v1/v2 timing comparison measures the same comparison object.
"""

from __future__ import annotations

_BASE = 10000.0  # the frozen OpSpec attr default (BASE has no per-instance override)


def mlx(x):
    """v1 baseline mlx (``mx.fast.rope`` with the registered traditional/scale/offset)."""
    import mlx.core as mx
    return mx.fast.rope(x, x.shape[-1], traditional=False, base=_BASE, scale=1.0, offset=0)

BASELINES = {"mlx": mlx}
