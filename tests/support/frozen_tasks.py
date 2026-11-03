"""Frozen R0 task oracle shared by the shared and Metal test layers (R2 H-15/H-19).

The oracle is the **commit recorded in the R0 baseline manifest**, never the floating head and
never the live tree.  It is materialised by exporting ``tasks/`` from that commit and installing
it as the ``tasks`` package inside :func:`frozen_tasks`, so the old modules' absolute imports
(``tasks._common``, ``tasks._elementwise``, ``tasks.gemm.task`` …) resolve inside the snapshot —
including ``load_text``'s template root, the elementwise builder and ``gemm_silu``'s re-exported
GEMM builder.  The live package is restored on exit.

Why this lives in ``tests/support``: the shared equivalence harness and the Metal tolerance
verdict test must use *the same* frozen oracle, and test modules must not import each other.
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import subprocess
import sys
import tarfile
import tomllib
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Iterator

ROOT = Path(__file__).resolve().parents[2]
R0_MANIFEST = ROOT / "docs" / "架构" / "构建执行与证据语义升级-R0" / "manifest.json"
SIZES = ("paper", "small", "plan")


def r0_commit() -> str:
    """@brief R0 基线包登记的完整提交（oracle 只能固定到它）。

    @return 40 位十六进制提交号。
    """
    recorded = json.loads(R0_MANIFEST.read_text(encoding="utf-8"))["head"]["commit"]
    assert len(recorded) == 40 and all(character in "0123456789abcdef" for character in recorded), recorded
    return recorded


#: Static entries that still build their own Contract.  Both the equivalence harness and the
#: Metal tolerance sweep import this single inventory, so a batch cannot be migrated in one place
#: and silently stay "remaining" (or vice versa) in the other.
#: A5 moved the 16 G0 migration candidates to the v2 standard entry.  Everything else stays on
#: the v1 path, so the inventory is exactly ``migration-boundary.json.preserved_v1_out_of_scope``.
REMAINING_STATIC = frozenset({
    "rope", "gemv", "conv2d", "flash_attention", "lion", "bitdelta_gemm", "mamba2_chunk_scan",
    "sem_attention_odd_scale",
})
#: ``gen_*`` family migrated to per-op OpSpec documents; the frozen TOML list stays the oracle.
REMAINING_GEN = False
#: The dynamic ``sem_int`` family has no standard directory and stays v1.
REMAINING_SEM_INT = True
#: ``flash_attention@…`` stays v1; the ``gemm@…`` variants retired with the v1 gemm assembly.
REMAINING_VARIANTS = True
#: Deterministic seed corpus for the frozen generators.
SEEDS = (0, 7, 1234)
#: ``Task.meta`` keys the retired TOML path wrote and that R2 removes (N-32): the ``gen_*``
#: family tagged ``meta["source"]`` with ``tasks/generated/elementwise.toml#<sha12>``.  Once the
#: manifest is gone that tag cannot be reproduced, and the provenance it carried now lives in
#: the track/OpSpec identity, so the comparison drops it on both sides *and* asserts that a
#: migrated task no longer carries it (``RETIRED_META_KEYS`` is also checked by the harness).
RETIRED_META_KEYS = ("source",)


@dataclass(frozen=True)
class FrozenSnapshot:
    """@brief 冻结快照的接受域（在与 live 隔离的上下文里读出）。"""

    root: Path
    tasks: tuple[str, ...]
    phase2: tuple[str, ...]
    semantic: tuple[str, ...]
    generated: tuple[str, ...]
    sizes: dict[str, tuple[str, ...]]

    def static_cases(self) -> list[tuple[str, str, str]]:
        """@brief 静态全集 × 两个 dtype 的全部组合。

        @return ``(task, dtype, size)`` 列表。
        """
        cases: list[tuple[str, str, str]] = []
        for task in self.tasks + self.phase2:
            for size in self.sizes[task]:
                for dtype in ("fp16", "bf16"):
                    cases.append((task, dtype, size))
        for task in self.semantic + self.generated:
            for size in ("paper", "small"):
                for dtype in ("fp16", "bf16"):
                    cases.append((task, dtype, size))
        return cases

    def migrated_cases(self) -> list[tuple[str, str, str]]:
        """@brief 只保留真正迁移的静态组合（判定与 ``is_remaining`` 共用同一实现）。

        @return 组合列表。
        """
        return [case for case in self.static_cases() if not is_remaining(case[0])]


@contextmanager
def frozen_tasks(root: Path) -> Iterator[None]:
    """@brief 期间把整个 ``tasks`` 包替换成冻结快照（绝对导入也落在冻结树）。

    @param root 冻结的 ``tasks`` 目录。
    """
    saved = {name: module for name, module in sys.modules.items() if name == "tasks" or name.startswith("tasks.")}
    for name in saved:
        del sys.modules[name]
    saved_flag = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec = importlib.util.spec_from_file_location(
            "tasks", root / "__init__.py", submodule_search_locations=[str(root)]
        )
        assert spec is not None and spec.loader is not None
        package = importlib.util.module_from_spec(spec)
        sys.modules["tasks"] = package
        spec.loader.exec_module(package)
        yield
    finally:
        for name in [name for name in sys.modules if name == "tasks" or name.startswith("tasks.")]:
            del sys.modules[name]
        sys.modules.update(saved)
        sys.dont_write_bytecode = saved_flag


def old_build(task: str, dtype: str, size: str):
    """@brief 在冻结隔离里装配一道旧题（必须在外层 :func:`frozen_tasks` 内调用）。

    @param task 任务名。
    @param dtype 元素类型。
    @param size 档位。
    @return 冻结快照里的 ``Task``。
    """
    if task.startswith("sem_"):
        return importlib.import_module("tasks.semantic").build(task, dtype, size)
    if task.startswith("gen_"):
        return importlib.import_module("tasks.generated.elementwise").build(task, dtype, size)
    return importlib.import_module(f"tasks.{task}.task").build(dtype, size)


def _read_domain(root: Path) -> FrozenSnapshot:
    """@brief 在冻结隔离里读取接受域（不读 live tree）。

    @param root 冻结的 ``tasks`` 目录。
    @return :class:`FrozenSnapshot`。
    """
    with frozen_tasks(root):
        registry = importlib.import_module("tasks")
        elementwise = importlib.import_module("tasks._elementwise")
        sizes: dict[str, tuple[str, ...]] = {}
        for task in registry.TASK_NAMES + registry.PHASE2_TASK_NAMES:
            module = importlib.import_module(f"tasks.{task}.task")
            shapes = getattr(module, "SHAPES", None)
            sizes[task] = tuple(shapes) if shapes is not None else tuple(elementwise.SIZES)
        with (root / "generated" / "elementwise.toml").open("rb") as handle:
            generated = tuple(sorted(entry["task"] for entry in tomllib.load(handle)["kernel"]))
        return FrozenSnapshot(
            root=root,
            tasks=tuple(registry.TASK_NAMES),
            phase2=tuple(registry.PHASE2_TASK_NAMES),
            semantic=tuple(registry.SEMANTIC_NAMES),
            generated=generated,
            sizes=sizes,
        )


def extract_snapshot(target: Path) -> FrozenSnapshot:
    """@brief 从 R0 登记提交导出 ``tasks/`` 并读出接受域。

    @param target 用于解包的目录。
    @return :class:`FrozenSnapshot`。
    """
    target.mkdir(parents=True, exist_ok=True)
    archive = target / "r0.tar"
    commit = r0_commit()
    subprocess.run(
        ["git", "archive", "--format=tar", "-o", str(archive), commit, "tasks"],
        cwd=ROOT,
        check=True,
        stdout=subprocess.DEVNULL,
    )
    with tarfile.open(archive) as tar:
        tar.extractall(target, filter="data")
    return _read_domain(target / "tasks")


def is_remaining(task: str) -> bool:
    """@brief 一个静态入口是否仍在过渡清单里（两个 harness 共用的唯一判定）。

    @param task 任务名。
    @return 未迁移时为 True。
    """
    if task.startswith("gen_"):
        return REMAINING_GEN
    return task in REMAINING_STATIC


def raw_bytes(tensor) -> bytes:
    """@brief 取张量的连续字节视图（能区分 ``-0.0``/``+0.0``，能匹配同位 NaN）。

    @param tensor torch 张量。
    @return 原始字节。
    """
    import torch

    return tensor.detach().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()


def fields_of(task) -> dict:
    """@brief 取一道题全部可比字段（不含实现闭包本身）。

    @param task 任务。
    @return 字段字典。
    """
    import dataclasses

    return {
        "task_identity": (task.name, task.dtype, task.size, task.key),
        # Known-good source per backend, read through the *live* accessor.  The frozen
        # ``sem_gemm_neg`` patched ``naive_msl`` after constructing the ``Task``, so its
        # ``reference_sources['metal']`` copy stayed stale (a second, unused copy of the
        # source).  R2 deletes that post-build patch, so the comparable value is the one
        # ``reference_for`` returns on each side; the migrated side's single-copy rule is
        # asserted separately in the equivalence sweep (H-21).
        "known_good_sources": {
            backend: task.reference_for(backend)
            for backend in ("metal",) + tuple(sorted(set(task.reference_sources) - {"metal"}))
        },
        "contract": dataclasses.asdict(task.contract),
        "describe": task.contract.describe(),
        "header": task.contract.render_header(),
        "abi": task.contract.abi_stub(),
        "assemble": task.contract.assemble("kernel void probe() {}\n"),
        "triton_source": task.triton_source,
        "naive_msl": task.naive_msl,
        "description": task.description,
        "cases": tuple(task.cases),
        "tolerances": (task.rtol, task.atol_rel, task.exact),
        # Work accounting is materialised by the registered rule as float (R1/reset).  A
        # pre-R1 task may still carry an int (``flops_per_elem * n``); the value must match
        # exactly and the migrated side must use the registered float materialisation.
        "work": (float(task.flops), float(task.min_bytes)),
        # Registration identity is a deliberate R1/R2 addition (the frozen task has none),
        # so it is checked separately instead of being compared against the old object.
        "baselines": [(b.name, b.framework) for b in task.baselines],
        "error_scale_registered": task.error_scale is not None,
        # Registered intentional change (N-32): the retired TOML provenance tag is dropped from
        # both sides.  ``RETIRED_META_KEYS`` is asserted to be *absent* on the migrated task.
        "meta": {key: value for key, value in (task.meta or {}).items() if key not in RETIRED_META_KEYS},
    }


def logical_cases(task) -> list[str]:
    """@brief 生成器真正接管的 case（``ieee`` 由 harness 经 ``ieee_base`` 物化）。

    @param task 任务。
    @return case 名列表。
    """
    base = (task.meta or {}).get("ieee_base", "normal")
    return [base if name == "ieee" else name for name in task.cases]


def assert_tensor_equal(old_tensor, new_tensor, where: str) -> None:
    """@brief dtype/shape 之后做真正的字节比较。

    @param old_tensor 冻结实现的张量。
    @param new_tensor 迁移实现的张量。
    @param where 诊断前缀。
    """
    assert new_tensor.dtype == old_tensor.dtype, f"{where}: dtype {new_tensor.dtype} != {old_tensor.dtype}"
    assert tuple(new_tensor.shape) == tuple(old_tensor.shape), f"{where}: shape"
    assert raw_bytes(new_tensor) == raw_bytes(old_tensor), f"{where}: bytes"


def materialised_scale(task, inputs, reference) -> dict:
    """@brief 物化一个 task 的误差尺度：登记实现优先，否则用本侧 reference 的 RMS。

    两侧必须各自调用，绝不共用一侧的结果（H-18）。

    @param task 任务。
    @param inputs 该 task 自己的输入。
    @param reference 该 task 自己的参考输出。
    @return 输出名 → scale 张量。
    """
    import torch

    if task.error_scale is not None:
        return task.error_scale(inputs, reference)
    return {name: torch.sqrt(torch.mean(value.double() ** 2)) for name, value in reference.items()}


def module_of(relative: str) -> ModuleType:
    """@brief 在冻结隔离里 import 一个子模块（供隔离回归使用）。

    @param relative ``tasks/`` 下的相对模块路径，如 ``"_common.py"``。
    @return 模块。
    """
    return importlib.import_module("tasks." + relative[:-3].replace("/", "."))


__all__ = [
    "ROOT",
    "R0_MANIFEST",
    "REMAINING_GEN",
    "REMAINING_SEM_INT",
    "REMAINING_STATIC",
    "RETIRED_META_KEYS",
    "REMAINING_VARIANTS",
    "SEEDS",
    "SIZES",
    "FrozenSnapshot",
    "assert_tensor_equal",
    "extract_snapshot",
    "fields_of",
    "frozen_tasks",
    "is_remaining",
    "logical_cases",
    "materialised_scale",
    "module_of",
    "old_build",
    "r0_commit",
    "raw_bytes",
]
