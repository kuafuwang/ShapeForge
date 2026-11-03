"""Task registry. ``get_task(name, dtype, size)`` builds a fully specified ``tcenv.task.Task``."""

# 搜索（taic.orchestrator）和验证（tcenv.evaluate）都只调用 get_task，拿到同一份 Task。
# 题面在 triton_source，对错看 reference，速度跟 baselines 比；contract 的 kernel_name、grid、threads、args、constants 由环境锁定，模型改不了。
# 这是**保留的 v1 装配**：普通题加载 tasks/<名字>/task.py 的 build，名字以 sem_ 开头交给 semantic.build，
# 名字里 @ 之后是另一套 tile/threads，走该题的 build_variant。A5 迁移到 v2 标准目录的 16 道题不在
# 这里登记（A6 删除它们的名字条目）；v2 入口是 tcenv.taskcontract 的自动发现，本模块永不读取它。
from __future__ import annotations

import importlib
import json
import re
from functools import lru_cache
from pathlib import Path

# "paper": the size used for every experiment. "small": quick tests. "plan": the exact size in the implementation plan, defined only
# where "paper" deviates from it (reduction_sum: 256x4096 is launch-bound, see the README).
# paper 给正式实验，small 给快速检查，plan 只在 paper 和实现方案不一致的题上出现（目前是 reduction_sum）。
SIZES = ("paper", "small", "plan")

TASKS_ROOT = Path(__file__).resolve().parent

#: Standard-entry markers.  ``get_task`` is the *v1* assembly entry: a directory that answers
#: ``spec_version == 2`` is refused here instead of being imported (A5 dispatches by
#: ``spec_version``; the v1 registry never reads a v2 task and never falls back).
V2_ENTRY_FILES: tuple[str, ...] = ("reference.py", "reference.json", "baselines.json", "baselines.py", "display.json")


# 第一阶段算子。目录名必须和这里一致，get_task 靠 importlib 加载 tasks.<名字>.task。
# 只登记仍由 v1 装配的题（``preserved_v1_out_of_scope``）。A5 迁移的 9 道普通题已删除 v1 task.py，
# A6 同时删除它们的名字条目：列一张本 registry 装不出来的名单，和 import 回退一样是假的多路径。
TASK_NAMES = (
    "rope",
    "gemv",
    "conv2d",
)

# Tasks added in phase 2 (recent-paper kernels).
# 第二阶段算子。get_task 接受它们，list_tasks 默认不列出。
PHASE2_TASK_NAMES: tuple[str, ...] = ("flash_attention", "lion", "bitdelta_gemm", "mamba2_chunk_scan")


def _keeps_v1(name: str) -> bool:
    """@brief 这道题是否仍由 v1 装配（标准目录不是 ``spec_version`` 2）。

    ``list_tasks`` enumerates tasks for the v1 tools (``tcenv.cli tasks``/``triton-baseline``)
    and for tests that walk "every task".  A5 moved 16 tasks to the v2 standard entry and the v1
    registry refuses them, so they must not be enumerated here either: a list that names tasks
    this registry cannot build is the same fallback illusion as a refused import.

    @param name 题目名。
    @return 仍是 v1 时为 True。
    """
    for directory in _task_dir_candidates(name):
        spec = directory / "opspec.json"
        try:
            if not spec.is_file():
                continue
            return json.loads(spec.read_text(encoding="utf-8")).get("spec_version") != 2
        except (OSError, ValueError):
            return True
    return True


def list_tasks(include_phase2: bool = False) -> tuple[str, ...]:
    """@brief 列出实验要枚举的题目名（只列 v1 registry 真能装配的题）。

    @param include_phase2 是否把第二阶段算子接在后面。
    @return 题目名元组，顺序与 ``TASK_NAMES`` 一致。
    """
    # 给实验枚举题目名。include_phase2 为真时把 PHASE2_TASK_NAMES 接在 TASK_NAMES 后面。
    # 已迁移（v2 标准目录）的题不再出现在 v1 的枚举里。
    names = TASK_NAMES + (PHASE2_TASK_NAMES if include_phase2 else ())
    return tuple(name for name in names if _keeps_v1(name))


# 固定的语义恢复题登记在 :data:`tasks.semantic.FIXED_NAMES`（get_task 靠 sem_ 前缀分派）。
# 生成族（``gen_<op>``）在 A5 整体迁移到 v2 标准目录，A6 删除旧解析器 ``tasks/generated/``
# 和它在这里的名字清单：生成族只有一个权威读取器，即每题自己的 ``opspec.json`` + ``reference.py``。

#: A task name is a single directory name under ``tasks/``.  The name arrives from run state,
#: ledgers and CLI input, so it is *data*, not a trusted path fragment: a name with a separator
#: would probe outside ``tasks/`` and a name longer than ``NAME_MAX`` makes ``stat`` raise
#: ``OSError(ENAMETOOLONG)`` instead of answering "no such task".
_TASK_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")
_NAME_MAX = 255


def _probeable_name(base: str) -> bool:
    """@brief 这个题名能否作为 ``tasks/`` 下的单层目录名去探测。

    @param base 去掉 ``@<variant>`` 后的题名。
    @return 是单层、长度合法的目录名时为 True。
    """
    return (
        bool(_TASK_NAME.fullmatch(base))
        and base not in {".", ".."}
        and len(base.encode("utf-8")) <= _NAME_MAX
    )


def _task_dir_candidates(name: str) -> tuple[Path, ...]:
    """@brief 一个题名可能对应的标准目录（v2 目录名等于 OpSpec ``op_name``）。

    非法的题名（空、含分隔符、``.``/``..``、超过 ``NAME_MAX``）没有候选目录：它们不可能
    是 ``tasks/`` 下的题，探测它们只会越界或触发 ``ENAMETOOLONG``。

    @param name 题目名（可带 ``@<variant>``）。
    @return 目录候选；非法题名返回空元组。
    """
    base = name.split("@", 1)[0]
    if not _probeable_name(base):
        return ()
    return (TASKS_ROOT / base,)


def _refuse_migrated(name: str) -> None:
    """@brief v1 注册表显式拒绝已迁移的 v2 标准目录（不 import、不回退）。

    ``get_task`` is the v1 assembly: it reads ``track.json``/``task.py``/``kernel_triton.py``.
    A5 moved 16 tasks to the v2 standard entry (``opspec.json`` with ``spec_version`` 2 +
    ``reference.py``), so the version decides the reader.  Trying the v1 import would raise an
    opaque ``ModuleNotFoundError`` and could look like a missing file instead of a deliberate
    dispatch, so the version is checked first.

    @param name 题目名。
    @raises tcenv.contract_errors.TaskContractError 目标是 v2 标准目录。
    """
    for directory in _task_dir_candidates(name):
        spec = directory / "opspec.json"
        try:
            if not spec.is_file():
                continue
            version = json.loads(spec.read_text(encoding="utf-8")).get("spec_version")
        except (OSError, ValueError):
            continue
        if version == 2:
            from tcenv.contract_errors import TaskContractError

            raise TaskContractError(
                f"task {name!r} ({directory.name}) is a v2 standard entry (spec_version 2); the v1 task "
                "registry never reads it. Use the v2 path: tcenv.taskcontract.load_task_contract / "
                "discover_standard_tasks + evaluate_task. There is no fallback in either direction."
            )


@lru_cache(maxsize=64)
def get_task(name: str, dtype: str = "fp16", size: str = "paper"):
    """@brief 按名字、dtype 与档位装配一个完整的 ``tcenv.task.Task``。

    搜索（taic.orchestrator）和验证（tcenv.evaluate）都只调用本函数，拿到同一份 Task。
    名字以 ``sem_`` 开头交给 ``semantic.build``，``@`` 之后是启动契约变体。

    @param name 题目名，可带 ``sem_`` 前缀或 ``@<变体>`` 后缀。
    @param dtype 元素类型，``fp16`` 或 ``bf16``。
    @param size 尺寸档位，``paper``、``small``，个别题多一档 ``plan``。
    @return 装配好的 ``Task``；同一 ``(name, dtype, size)`` 命中缓存。
    @raises ValueError 档位与题型不匹配、档位非法，或该题没有 ``plan`` 档、没有启动契约变体时抛出。
    @raises KeyError 题目名不在 ``TASK_NAMES`` 与 ``PHASE2_TASK_NAMES`` 里时抛出。
    @raises tcenv.contract_errors.TaskContractError 目标是已迁移的 v2 标准目录。
    """
    _refuse_migrated(name)
    # 同一 (name, dtype, size) 命中缓存。size="plan" 时，该题的 SHAPES 里必须真有 "plan" 这一档。
    if name.startswith("sem_"):  # semantic-recovery tasks (paper Section 4.1)
        if size not in ("paper", "small"):
            raise ValueError("size must be 'paper' or 'small'")
        from . import semantic

        return semantic.build(name, dtype, size)
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
