"""read_write / workspace / guard / timing protocol of the v2 candidate plane (A2).

The plan fixes this protocol in words; this module is the one machine-readable statement of it,
so the correctness runner, the benchmark runner and the evidence reader cannot each invent their
own version:

*Every correctness launch* (``out1``/``out2``/``out3``) first **restores the read_write initial
values**.  A ``read_write`` buffer is part of the reference input contract, so it is never
poisoned or zeroed, and it is never counted by the "still holds the poison pattern" statistic.
Pure ``write`` outputs are poisoned for ``out1``/``out3`` and zero-filled for ``out2`` — that is
what makes "depends on the initial contents of the output buffer" observable.  Every buffer's
guard bands are ``GUARD_PATTERN`` (``0xCD``) on every trial; a guard hit is a safety violation,
not a numeric one.

*Workspace.*  ``workspace_fills`` only clears a workspace the candidate **declared zero**.
Undeclared workspaces get no clearing kernel in the benchmark, while correctness poisons them
(``out1``/``out3``) or zeroes them (``out2``) so a kernel that depends on garbage is caught.

*Benchmark repetitions.*  After the first fills, consecutive repetitions may check only the
guard bands, the out-of-bounds bytes and the hardware state — no per-rep value diff and no blanket
overflow assertion.  What must be registered is which timing discipline is in use:
``fresh_rotation`` (fresh clones / rotating buffer sets per repetition) or ``stateful`` (a
registered state identity, because the kernel mutates state across reps).  The two are mutually
exclusive; mixing them would publish rotation-free timing as if it were fresh.

*Prepare / Host Tiling.*  Input-dependent work must be inside the timed window.  Host Tiling may
only produce geometry, resources and launch arguments; it may not compute the reference, pre-
process inputs or do the device kernel's math.  A declaration that hides input-dependent prepare
outside timing is refused instead of being silently billed to nobody.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from ..contract_errors import GeometryError, TaskContractError
from .geometry import ResolvedGeometry, WorkspaceUse
from .plane import TensorMemoryContract

#: Guard band byte pattern.  Same value the Metal buffer layer uses, so a guard hit means the same
#: thing in v1 and v2 evidence.
GUARD_PATTERN = 0xCD
#: One-sided guard band width of a buffer, in bytes.
GUARD_BYTES = 16384
#: Poison pattern: NaN for fp16/bf16/fp32, ``-1`` for integers.
POISON_PATTERN = 0xFF

CORRECTNESS_TRIALS: tuple[str, ...] = ("out1", "out2", "out3")

RESTORE_INITIAL = "restore_initial"
POISON = "poison"
ZERO = "zero"
UNTOUCHED = "untouched"


@dataclass(frozen=True)
class TensorFill:
    """@brief 一次启动前对某个缓冲区的填充动作。"""

    name: str
    reason: str
    action: str

    def document(self) -> dict[str, Any]:
        """@brief 该动作的 JSON 文档。

        @return 字段字典。
        """
        return {"name": self.name, "reason": self.reason, "action": self.action}


@dataclass(frozen=True)
class LaunchFillPlan:
    """@brief 一次受测启动的完整填充计划（含 guard band 图案）。"""

    phase: str
    tensors: tuple[TensorFill, ...]
    guard_pattern: int = GUARD_PATTERN
    guard_bytes: int = GUARD_BYTES

    def action(self, name: str) -> str:
        """@brief 取某个缓冲区的动作。

        @param name 缓冲区名。
        @return 动作字符串。
        @raises KeyError 没有该缓冲区。
        """
        for fill in self.tensors:
            if fill.name == name:
                return fill.action
        raise KeyError(name)

    def document(self) -> dict[str, Any]:
        """@brief 计划的 JSON 文档。

        @return 字段字典。
        """
        return {
            "phase": self.phase,
            "guard_pattern": int(self.guard_pattern),
            "guard_bytes": int(self.guard_bytes),
            "tensors": [fill.document() for fill in self.tensors],
        }


def correctness_launch_plan(
    contract: TensorMemoryContract,
    geometry: ResolvedGeometry,
    trial: str,
) -> LaunchFillPlan:
    """@brief 一次正确性启动（out1/out2/out3）的填充计划。

    @param contract ABI 合同。
    @param geometry 已求值几何（提供 workspace 使用）。
    @param trial ``out1``/``out2``/``out3``。
    @return 填充计划。
    @raises TaskContractError trial 未登记。
    """
    if trial not in CORRECTNESS_TRIALS:
        raise TaskContractError(f"unknown correctness trial {trial!r}; registered trials are {list(CORRECTNESS_TRIALS)}")
    zero_trial = trial == "out2"
    fills: list[TensorFill] = []
    for tensor in contract.tensors:
        if tensor.space == "workspace":
            # Correctness fills every workspace regardless of its declaration: a kernel that
            # reads an uncleared scratch buffer must be caught, and out2 vs out1/out3 is the
            # asymmetry that catches it.
            fills.append(TensorFill(tensor.name, "workspace", ZERO if zero_trial else POISON))
            continue
        if tensor.access == "read_write":
            # A read_write buffer's initial value is part of the contract: restore it, never
            # poison it, and never count it as unwritten.
            fills.append(TensorFill(tensor.name, "read_write", RESTORE_INITIAL))
            continue
        if tensor.access == "read":
            fills.append(TensorFill(tensor.name, "input", RESTORE_INITIAL))
            continue
        fills.append(TensorFill(tensor.name, "output", ZERO if zero_trial else POISON))
    return LaunchFillPlan(phase=f"correctness:{trial}", tensors=tuple(fills))


def benchmark_launch_plan(
    contract: TensorMemoryContract,
    geometry: ResolvedGeometry,
    *,
    discipline: "TimingDiscipline",
) -> LaunchFillPlan:
    """@brief 一次基准启动（每轮）的填充计划。

    @param contract ABI 合同。
    @param geometry 已求值几何（提供 workspace 使用与声明）。
    @param discipline 计时纪律。
    @return 填充计划。
    @raises GeometryError workspace 声明冲突。
    """
    uses = {use.name: use for use in geometry.workspace_uses}
    fills: list[TensorFill] = []
    for tensor in contract.tensors:
        if tensor.space == "workspace":
            declared = uses.get(tensor.name)
            # Only a workspace the candidate declared zero is cleared inside the timed window.
            clears = declared is not None and declared.zero
            fills.append(
                TensorFill(
                    tensor.name,
                    "workspace:declared_zero" if clears else "workspace",
                    ZERO if clears else UNTOUCHED,
                )
            )
            continue
        if tensor.access == "read_write":
            # fresh rotation restores the initial value every repetition; a stateful run must
            # register the state instead of pretending the state is the original input.
            action = RESTORE_INITIAL if discipline.mode == "fresh_rotation" else UNTOUCHED
            fills.append(TensorFill(tensor.name, "read_write", action))
            continue
        if tensor.access == "read":
            fills.append(
                TensorFill(tensor.name, "input", RESTORE_INITIAL if discipline.mode == "fresh_rotation" else UNTOUCHED)
            )
            continue
        # A pure write output needs no fill inside the timed window: the kernel writes every
        # element, and terminal values are not diffed in the benchmark.
        fills.append(TensorFill(tensor.name, "output", UNTOUCHED))
    return LaunchFillPlan(phase=f"benchmark:{discipline.mode}", tensors=tuple(fills))


def workspace_fills(
    contract: TensorMemoryContract,
    geometry: ResolvedGeometry,
    *,
    phase: str,
) -> tuple[WorkspaceUse, ...]:
    """@brief 需要清零的 workspace（只含候选声明 ``zero`` 的那些）。

    @param contract ABI 合同。
    @param geometry 已求值几何。
    @param phase ``benchmark`` 或 ``correctness``。
    @return 需要清零的 workspace 使用。
    @raises TaskContractError phase 未登记。
    """
    if phase not in ("benchmark", "correctness"):
        raise TaskContractError(f"unknown phase {phase!r}")
    if phase == "correctness":
        return tuple(WorkspaceUse(name=tensor.name, zero=True) for tensor in contract.tensors if tensor.space == "workspace")
    return tuple(use for use in geometry.workspace_uses if use.zero)


def unwritten_check_applies(contract: TensorMemoryContract, name: str) -> bool:
    """@brief 某个张量是否参与"从未写入"统计（read_write 不参与）。

    @param contract ABI 合同。
    @param name 张量名。
    @return 纯 write 输出为 True。
    @raises KeyError 合同没有该张量。
    """
    tensor = contract.tensor(name)
    if tensor.space == "workspace":
        return False
    return tensor.access == "write"


@dataclass(frozen=True)
class TimingDiscipline:
    """@brief 计时纪律：``fresh_rotation`` 或登记过的 ``stateful``，两者互斥。"""

    mode: str
    rotation_sets: int = 1
    stateful_declaration: str | None = None

    def __post_init__(self) -> None:
        """@brief 校验纪律。

        @raises TaskContractError 未知 mode 或旋转组数非法。
        @raises GeometryError ``fresh_rotation`` 与 stateful 声明同时出现。
        """
        if self.mode not in ("fresh_rotation", "stateful"):
            raise TaskContractError(f"unknown timing discipline {self.mode!r}; use fresh_rotation or stateful")
        if isinstance(self.rotation_sets, bool) or not isinstance(self.rotation_sets, int) or self.rotation_sets < 1:
            raise TaskContractError("TimingDiscipline.rotation_sets must be an integer >= 1")
        if self.mode == "fresh_rotation" and self.stateful_declaration is not None:
            raise GeometryError(
                "fresh rotation and a stateful declaration cannot describe the same run: publishing "
                "rotation-free timing as if every repetition started from the original input is not evidence"
            )
        if self.mode == "stateful" and not self.stateful_declaration:
            raise TaskContractError(
                "stateful timing must register the declaration that makes the mutated state explicit"
            )

    @classmethod
    def fresh_rotation(cls, rotation_sets: int = 1) -> "TimingDiscipline":
        """@brief 每轮使用 fresh 输入副本的纪律。

        @param rotation_sets 轮转组数。
        @return 纪律。
        """
        return cls(mode="fresh_rotation", rotation_sets=rotation_sets)

    @classmethod
    def stateful(cls, declaration: str) -> "TimingDiscipline":
        """@brief 登记为 stateful 的纪律。

        @param declaration state 的登记说明（进入证据）。
        @return 纪律。
        """
        return cls(mode="stateful", rotation_sets=1, stateful_declaration=declaration)

    def post_rep_checks(self) -> tuple[str, ...]:
        """@brief 连续 reps 之后允许做的检查。

        @return 检查名元组（guard band、越界字节、硬件状态，仅此三项）。
        """
        return ("guard_bands", "out_of_bounds", "hardware_state")

    @property
    def fresh(self) -> bool:
        """@brief 是否为 fresh rotation。

        @return ``mode == "fresh_rotation"``。
        """
        return self.mode == "fresh_rotation"

    @classmethod
    def resolve(
        cls,
        *,
        fresh_rotation: bool,
        rotation_sets: int = 1,
        stateful_declaration: str | None = None,
    ) -> "TimingDiscipline":
        """@brief 由调用方的请求解析出**唯一**纪律。

        @param fresh_rotation 调用方是否要求每轮 fresh 输入。
        @param rotation_sets 轮转组数。
        @param stateful_declaration stateful 登记的声明。
        @return 纪律。
        @raises GeometryError 同时要求 fresh rotation 与 stateful。
        @raises TaskContractError stateful 未登记。
        """
        if fresh_rotation and stateful_declaration is not None:
            raise GeometryError(
                "a run cannot be both fresh-rotation and stateful: the timing discipline must be one of them"
            )
        if fresh_rotation:
            return cls(mode="fresh_rotation", rotation_sets=rotation_sets)
        if stateful_declaration is None:
            raise TaskContractError(
                "timing must be registered as fresh_rotation or stateful; an unregistered run neither rotates "
                "fresh inputs nor declares the mutated state"
            )
        return cls(mode="stateful", rotation_sets=1, stateful_declaration=stateful_declaration)

    def document(self) -> dict[str, Any]:
        """@brief 纪律的 JSON 文档。

        @return 字段字典。
        """
        document: dict[str, Any] = {
            "mode": self.mode,
            "rotation_sets": int(self.rotation_sets),
            "post_rep_checks": list(self.post_rep_checks()),
        }
        if self.stateful_declaration is not None:
            document["stateful_declaration"] = self.stateful_declaration
        return document


@dataclass(frozen=True)
class PrepareDeclaration:
    """@brief 候选 prepare / Host Tiling 的一步声明（计时归属必须显式）。"""

    name: str
    input_dependent: bool
    cached: bool = False
    inside_timing: bool = False
    cache_identity: str | None = None
    computes_reference: bool = False
    preprocesses_inputs: bool = False

    def document(self) -> dict[str, Any]:
        """@brief 声明的 JSON 文档。

        @return 字段字典。
        """
        return {
            "name": self.name,
            "input_dependent": bool(self.input_dependent),
            "cached": bool(self.cached),
            "inside_timing": bool(self.inside_timing),
            "cache_identity": self.cache_identity,
            "computes_reference": bool(self.computes_reference),
            "preprocesses_inputs": bool(self.preprocesses_inputs),
        }


def check_prepare_timing(declarations: Iterable[PrepareDeclaration]) -> tuple[PrepareDeclaration, ...]:
    """@brief 校验 Host Tiling/prepare 的计时边界。

    The rule is deliberately absolute for input-dependent work: it must be inside the timed
    window.  Caching an input-dependent step outside timing is still hiding input work from the
    measurement (a "cache" that only exists because the inputs were read once is not a cold-start
    exclusion), so it is refused rather than accepted with a label.  An input-independent cached
    step is reusable preparation and must carry the identity that binds it to the candidate/bundle,
    target and instance.

    @param declarations 候选/构建适配器声明的步骤。
    @return 原声明元组。
    @raises GeometryError 输入相关工作被藏到计时外，或 Host Tiling 替内核做了数学。
    """
    checked: list[PrepareDeclaration] = []
    for declaration in declarations:
        if declaration.computes_reference:
            raise GeometryError(
                f"prepare step {declaration.name!r} computes the reference; Host Tiling may only produce geometry, "
                "resources and launch arguments"
            )
        if declaration.preprocesses_inputs and not declaration.inside_timing:
            raise GeometryError(
                f"prepare step {declaration.name!r} preprocesses inputs outside the timed window; input-dependent "
                "work must be billed to the measured run"
            )
        if declaration.input_dependent and not declaration.inside_timing:
            raise GeometryError(
                f"prepare step {declaration.name!r} is input-dependent and outside the timed window; input-dependent "
                "prepare must be billed to the measured run, never hidden behind a cache or a pre-run step"
            )
        if declaration.cached and not declaration.cache_identity:
            raise GeometryError(
                f"prepare step {declaration.name!r} is cached with no cache identity; reusable preparation must bind "
                "the candidate/bundle, target and instance it belongs to"
            )
        checked.append(declaration)
    return tuple(checked)


__all__ = [
    "CORRECTNESS_TRIALS",
    "GUARD_BYTES",
    "GUARD_PATTERN",
    "POISON",
    "POISON_PATTERN",
    "RESTORE_INITIAL",
    "UNTOUCHED",
    "ZERO",
    "LaunchFillPlan",
    "PrepareDeclaration",
    "TensorFill",
    "TimingDiscipline",
    "benchmark_launch_plan",
    "check_prepare_timing",
    "correctness_launch_plan",
    "unwritten_check_applies",
    "workspace_fills",
]
