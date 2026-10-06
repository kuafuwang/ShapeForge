"""Task registry. ``get_task(name, dtype, size)`` builds a fully specified ``tcenv.task.Task``."""

from __future__ import annotations

import importlib
from functools import lru_cache

# "paper": the size used for every experiment. "small": quick tests. "plan": the exact size in the implementation plan, defined only
# where "paper" deviates from it (reduction_sum: 256x4096 is launch-bound, see the README).
SIZES = ("paper", "small", "plan")

TASK_NAMES = (
    "sigmoid",
    "relu",
    "silu",
    "swiglu",
    "rmsnorm",
    "softmax",
    "reduction_sum",
    "rope",
    "gemv",
    "gemm",
    "gemm_silu",
    "conv2d",
)

# Tasks added in phase 2 (recent-paper kernels).
PHASE2_TASK_NAMES: tuple[str, ...] = ("flash_attention", "lion", "bitdelta_gemm", "mamba2_chunk_scan")


def list_tasks(include_phase2: bool = False) -> tuple[str, ...]:
    return TASK_NAMES + (PHASE2_TASK_NAMES if include_phase2 else ())


SEMANTIC_NAMES = ("sem_gemm_neg", "sem_relu_gelu_comment", "sem_gelu_named_relu", "sem_attention_odd_scale")


@lru_cache(maxsize=64)
def get_task(name: str, dtype: str = "fp16", size: str = "paper"):
    if name.startswith("sem_"):  # semantic-recovery tasks (paper Section 4.1)
        if size not in ("paper", "small"):
            raise ValueError("size must be 'paper' or 'small'")
        from . import semantic

        return semantic.build(name, dtype, size)
    base, _, variant = name.partition("@")  # "gemm@128x64x32t256": the same operator under another launch contract (tile / threads)
    if base not in TASK_NAMES + PHASE2_TASK_NAMES:
        raise KeyError(f"unknown task {name!r}; known: {list(TASK_NAMES + PHASE2_TASK_NAMES)}")
    if size not in SIZES:
        raise ValueError(f"size must be one of {SIZES}")
    mod = importlib.import_module(f"tasks.{base}.task")
    if size == "plan" and "plan" not in getattr(mod, "SHAPES", {}):
        raise ValueError(f"task {base!r} has no 'plan' size (only tasks whose 'paper' size deviates from the implementation plan define one)")
    if variant:
        if not hasattr(mod, "build_variant"):
            raise ValueError(f"task {base!r} has no contract variants (got {name!r})")
        return mod.build_variant(dtype, size, variant)
    return mod.build(dtype, size)
