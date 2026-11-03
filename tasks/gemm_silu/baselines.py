"""Performance baselines for gemm_silu: the G0-registered vendor/library calls.

A baseline is a timing comparison object, never the correctness oracle.  Each function
receives the framework-native arrays BaselineRunner prepared (mlx array or torch MPS
tensor) and returns the framework-native result.  The call is the one the v1 track
registered, so a v1/v2 timing comparison measures the same comparison object.
"""

from __future__ import annotations

_COMPILED: dict = {}


def _compiled(key, builder):
    """@brief 每个 key 只 ``mx.compile`` 一次（与 v1 ``mlx_compiled`` 相同）。"""
    fn = _COMPILED.get(key)
    if fn is None:
        import mlx.core as mx

        fn = _COMPILED[key] = mx.compile(builder(mx))
    return fn

def mlx(a, b):
    """G0 baseline mlx."""
    return _compiled("silu", lambda mx: (lambda t: t * mx.sigmoid(t)))(a @ b)

def torch_mps(a, b):
    """G0 baseline torch_mps."""
    import torch
    return torch.nn.functional.silu(torch.matmul(a, b))

BASELINES = {
    "mlx": mlx,
    "torch_mps": torch_mps,
}
