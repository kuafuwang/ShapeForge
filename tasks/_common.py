"""Shared helpers for task definitions."""

from __future__ import annotations

from pathlib import Path

import torch

from tcenv.dtypes import get_dtype

TASKS_DIR = Path(__file__).resolve().parent

TL_DTYPES = {"fp16": "tl.float16", "bf16": "tl.bfloat16", "fp32": "tl.float32"}
TORCH_DTYPES = {"fp16": "torch.float16", "bf16": "torch.bfloat16", "fp32": "torch.float32"}


def load_text(task_dir: str, filename: str, dtype: str, extra: dict[str, str] | None = None) -> str:
    """Read a per-task text file and substitute dtype placeholders.

    ``{{T}}``        MSL element type (half / bfloat)
    ``{{TL}}``       Triton dtype (tl.float16 / tl.bfloat16)
    ``{{TORCH}}``    torch dtype (torch.float16 / torch.bfloat16)
    """
    s = (TASKS_DIR / task_dir / filename).read_text()
    s = s.replace("{{T}}", get_dtype(dtype).msl).replace("{{TL}}", TL_DTYPES[dtype]).replace("{{TORCH}}", TORCH_DTYPES[dtype])
    for k, v in (extra or {}).items():
        s = s.replace("{{" + k + "}}", v)
    return s


def check_dtype(dtype: str) -> None:
    if dtype not in ("fp16", "bf16"):
        raise ValueError(f"dtype must be 'fp16' or 'bf16' (Apple GPUs have no fp8), got {dtype!r}")


_MLX_COMPILED: dict = {}


_TORCH_COMPILED: dict = {}


def torch_compiled(key: str, fn):
    """Lazily ``torch.compile`` ``fn`` once per process. On MPS this goes through Inductor's Metal backend, i.e. a real
    compiler-generated fused Metal kernel, which makes it the strongest available baseline for unfused composite operators."""
    c = _TORCH_COMPILED.get(key)
    if c is None:
        c = _TORCH_COMPILED[key] = torch.compile(fn)
    return c


def mlx_compiled(key: str, builder):
    """Lazily build (once per process) an ``mx.compile``d function; keeps ``import mlx`` out of module import time."""
    fn = _MLX_COMPILED.get(key)
    if fn is None:
        import mlx.core as mx

        fn = _MLX_COMPILED[key] = mx.compile(builder(mx))
    return fn
