"""Shared helpers for task definitions."""

# 各题 task.py 共用的读题面和基线编译。搜索和验证拿到的 Task 里，题面和融合基线就是经这里准备的。
# load_text 读取 kernel_triton.py / naive.metal，把 {{T}}、{{TL}}、{{TORCH}} 以及 extra 里的 {{键}} 换成这次 dtype 和形状的具体写法。
# torch_compiled、mlx_compiled 按 key 缓存一次编译。lion、bitdelta_gemm、gemm_silu、mamba2、silu、swiglu 用它们做融合后的库基线。
from __future__ import annotations

from pathlib import Path

import torch

from tcenv.dtypes import get_dtype

TASKS_DIR = Path(__file__).resolve().parent

TL_DTYPES = {"fp16": "tl.float16", "bf16": "tl.bfloat16", "fp32": "tl.float32"}
TORCH_DTYPES = {"fp16": "torch.float16", "bf16": "torch.bfloat16", "fp32": "torch.float32"}


# 题面正文留在各题目录的文本文件里。这里把 {{T}}、{{TL}}、{{TORCH}} 和 extra 里的 {{键}} 换成具体写法。
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


# 题目只接受 fp16 / bf16。Apple GPU 没有 fp8。
def check_dtype(dtype: str) -> None:
    if dtype not in ("fp16", "bf16"):
        raise ValueError(f"dtype must be 'fp16' or 'bf16' (Apple GPUs have no fp8), got {dtype!r}")


_MLX_COMPILED: dict = {}


_TORCH_COMPILED: dict = {}


# 每个 key 在进程里只 torch.compile 一次。MPS 上走 Inductor 的 Metal 后端，给未融合的复合算子当强基线。
def torch_compiled(key: str, fn):
    """Lazily ``torch.compile`` ``fn`` once per process. On MPS this goes through Inductor's Metal backend, i.e. a real
    compiler-generated fused Metal kernel, which makes it the strongest available baseline for unfused composite operators."""
    c = _TORCH_COMPILED.get(key)
    if c is None:
        c = _TORCH_COMPILED[key] = torch.compile(fn)
    return c


# 每个 key 只 mx.compile 一次，并且把 import mlx 推迟到第一次真正调用。
def mlx_compiled(key: str, builder):
    """Lazily build (once per process) an ``mx.compile``d function; keeps ``import mlx`` out of module import time."""
    fn = _MLX_COMPILED.get(key)
    if fn is None:
        import mlx.core as mx

        fn = _MLX_COMPILED[key] = mx.compile(builder(mx))
    return fn
