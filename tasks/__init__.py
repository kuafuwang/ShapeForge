"""Task registry. ``get_task(name, dtype, size)`` builds a fully specified ``tcenv.task.Task``."""

# 搜索（taic.orchestrator）和验证（tcenv.evaluate）都只调用 get_task，拿到同一份 Task。
# 题面在 triton_source，对错看 reference，速度跟 baselines 比；contract 的 kernel_name、grid、threads、args、constants 由环境锁定，模型改不了。
# 普通题加载 tasks/<名字>/task.py 的 build。名字以 sem_ 开头交给 semantic.build。名字以 gen_ 开头交给 tasks/generated/<op>/task.py。名字里 @ 之后是另一套 tile/threads，走该题的 build_variant。
from __future__ import annotations

import importlib
from functools import lru_cache

# "paper": the size used for every experiment. "small": quick tests. "plan": the exact size in the implementation plan, defined only
# where "paper" deviates from it (reduction_sum: 256x4096 is launch-bound, see the README).
# paper 给正式实验，small 给快速检查，plan 只在 paper 和实现方案不一致的题上出现（目前是 reduction_sum）。
SIZES = ("paper", "small", "plan")

# 第一阶段算子。目录名必须和这里一致，get_task 靠 importlib 加载 tasks.<名字>.task。
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
# 第二阶段算子。get_task 接受它们，list_tasks 默认不列出。
PHASE2_TASK_NAMES: tuple[str, ...] = ("flash_attention", "lion", "bitdelta_gemm", "mamba2_chunk_scan")


def list_tasks(include_phase2: bool = False) -> tuple[str, ...]:
    """@brief 列出实验要枚举的题目名。

    @param include_phase2 是否把第二阶段算子接在后面。
    @return 题目名元组，顺序与 ``TASK_NAMES`` 一致。
    """
    # 给实验枚举题目名。include_phase2 为真时把 PHASE2_TASK_NAMES 接在 TASK_NAMES 后面。
    return TASK_NAMES + (PHASE2_TASK_NAMES if include_phase2 else ())


# 四道固定的语义恢复题。它们不在 TASK_NAMES 里，get_task 靠 sem_ 前缀认出来。
SEMANTIC_NAMES = ("sem_gemm_neg", "sem_relu_gelu_comment", "sem_gelu_named_relu", "sem_attention_odd_scale")

# 生成式逐元素题（``gen_<op>``）：每题一份 tasks/generated/<op>/opspec.json + track.json，
# 题面/known-good 由登记的生成器渲染。清单在这里显式登记（不从文件系统扫描），
# 冻结的 R0 TOML 清单是独立对照，两边漂移会让等价用例失败。
GENERATED_NAMES = ("relu", "sigmoid", "silu", "swiglu")


@lru_cache(maxsize=64)
def get_task(name: str, dtype: str = "fp16", size: str = "paper"):
    """@brief 按名字、dtype 与档位装配一个完整的 ``tcenv.task.Task``。

    搜索（taic.orchestrator）和验证（tcenv.evaluate）都只调用本函数，拿到同一份 Task。
    名字以 ``sem_`` 开头交给 ``semantic.build``，以 ``gen_`` 开头交给 ``tasks/generated/<op>/task.py``，``@`` 之后是启动契约变体。

    @param name 题目名，可带 ``sem_`` / ``gen_`` 前缀或 ``@<变体>`` 后缀。
    @param dtype 元素类型，``fp16`` 或 ``bf16``。
    @param size 尺寸档位，``paper``、``small``，个别题多一档 ``plan``。
    @return 装配好的 ``Task``；同一 ``(name, dtype, size)`` 命中缓存。
    @raises ValueError 档位与题型不匹配、档位非法，或该题没有 ``plan`` 档、没有启动契约变体时抛出。
    @raises KeyError 题目名不在 ``TASK_NAMES`` 与 ``PHASE2_TASK_NAMES`` 里时抛出。
    """
    # 同一 (name, dtype, size) 命中缓存。size="plan" 时，该题的 SHAPES 里必须真有 "plan" 这一档。
    if name.startswith("sem_"):  # semantic-recovery tasks (paper Section 4.1)
        if size not in ("paper", "small"):
            raise ValueError("size must be 'paper' or 'small'")
        from . import semantic

        return semantic.build(name, dtype, size)
    if name.startswith("gen_"):
        if size not in ("paper", "small"):
            raise ValueError("size must be 'paper' or 'small'")
        stem = name[len("gen_") :]
        if stem not in GENERATED_NAMES:
            known = ", ".join("gen_" + entry for entry in GENERATED_NAMES)
            raise KeyError(f"unknown generated task {name!r}; known: {known}")
        mod = importlib.import_module(f"tasks.generated.{stem}.task")
        return mod.build(dtype, size)
    # @ 前是算子名，后面是启动契约变体（tile / threads）。公式和 reference 不变。
    base, _, variant = name.partition("@")  # "gemm@128x64x32t256": the same operator under another launch contract (tile / threads)
    if base not in TASK_NAMES + PHASE2_TASK_NAMES:
        raise KeyError(f"unknown task {name!r}; known: {list(TASK_NAMES + PHASE2_TASK_NAMES)}")
    if size not in SIZES:
        raise ValueError(f"size must be one of {SIZES}")
    mod = importlib.import_module(f"tasks.{base}.task")
    # Migrated tasks declare their sizes in opspec.json, so ``instantiate`` is the single
    # authority for the accepted size domain; the module-level SHAPES table is transitional.
    shapes = getattr(mod, "SHAPES", None)
    if shapes is not None and size == "plan" and "plan" not in shapes:
        raise ValueError(f"task {base!r} has no 'plan' size (only tasks whose 'paper' size deviates from the implementation plan define one)")
    if variant:
        if not hasattr(mod, "build_variant"):
            raise ValueError(f"task {base!r} has no contract variants (got {name!r})")
        return mod.build_variant(dtype, size, variant)
    return mod.build(dtype, size)
