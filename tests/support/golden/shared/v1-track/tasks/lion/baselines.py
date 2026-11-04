"""Performance baselines for lion: the vendor/library calls the v1 track registered.

A baseline is a timing comparison object, never the correctness oracle.  Functions receive
the framework-native arrays BaselineRunner prepared and return the framework-native result.
"""

from __future__ import annotations

# Frozen OpSpec attr defaults (LR/WD/BETA1/BETA2 have no per-instance override).
_LR, _WD, _BETA1, _BETA2 = 0.001, 0.1, 0.9, 0.99
_COMPILED: dict = {}


def _compiled(key, builder):
    """@brief 每个 key 只编译一次（与 v1 ``mlx_compiled``/``torch_compiled`` 相同）。"""
    fn = _COMPILED.get(key)
    if fn is None:
        fn = _COMPILED[key] = builder()
    return fn


def _mlx_step(mx):
    def step(p, g, m):
        u = mx.sign(_BETA1 * m + (1 - _BETA1) * g)
        return p * (1 - _LR * _WD) - _LR * u, _BETA2 * m + (1 - _BETA2) * g
    return step


def mlx(p, g, m):
    """v1 baseline mlx: the whole step fused with ``mx.compile``."""
    import mlx.core as mx
    return _compiled("lion-mlx", lambda: mx.compile(_mlx_step(mx)))(p, g, m)


def _torch_step(p, g, m):
    import torch
    u = torch.sign(_BETA1 * m + (1 - _BETA1) * g)
    return p * (1 - _LR * _WD) - _LR * u, _BETA2 * m + (1 - _BETA2) * g


def torch_mps(p, g, m):
    """v1 baseline torch_mps: eager ops."""
    return _torch_step(p, g, m)


def torch_compile(p, g, m):
    """v1 baseline torch_compile: Inductor fuses the whole step into one kernel."""
    import torch
    return _compiled("lion-torch", lambda: torch.compile(_torch_step))(p, g, m)

BASELINES = {"mlx": mlx, "torch_mps": torch_mps, "torch_compile": torch_compile}
