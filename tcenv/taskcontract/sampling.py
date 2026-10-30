"""Default v2 input sampling (hardware-neutral, deterministic, no global RNG).

The standard v2 entry only *requires* ``reference(inputs, attrs)``; a task may keep its
input generator in ``reference.py`` (``gen_inputs(case, seed, instance)``) or declare a
sampling profile in ``reference.json``.  When neither exists, this module produces the
initial values, including output/read_write buffers' initial values.  It is the v2 default only: v1 task modules keep their own generators and the G0
seed stream, cases and error scale are untouched.

The sampling policy adapts the upstream *methodology* (normal/uniform draws from a seeded
generator) without importing the upstream vendor-licensed skill code: that helper lives in a
hardware-vendor skill tree and mutates the process-global torch RNG, which cannot preserve
the parallel/replay guarantees this repository needs.  The local policy uses a private
``torch.Generator`` per tensor, so the stream does not depend on what ran before it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from ..dtypes import get_dtype
from ..opspec.v2 import SemanticInstanceV2
from .errors import TaskContractError

#: Stride used to derive a per-tensor stream from one case seed (the same idea as the frozen
#: G0 corpus ``base_seed + 7919 * index``, but applied to the tensor index, not the case index).
TENSOR_SEED_STRIDE = 7919

_DEFAULT_FLOAT = {"kind": "normal", "mean": 0.0, "std": 1.0}


@dataclass(frozen=True)
class SamplingProfile:
    """@brief 一个输入分布 profile。"""

    kind: str
    mean: float = 0.0
    std: float = 1.0
    low: float = 0.0
    high: float = 1.0

    def document(self) -> dict[str, Any]:
        """@brief 该 profile 的 JSON 文档。

        @return 字段字典。
        """
        if self.kind == "normal":
            return {"kind": "normal", "mean": float(self.mean), "std": float(self.std)}
        return {"kind": "uniform", "low": float(self.low), "high": float(self.high)}

    def sample(self, shape: tuple[int, ...], dtype: str, *, seed: int):
        """@brief 按该 profile 生成一个张量（私有 generator，不碰全局 RNG）。

        @param shape 形状。
        @param dtype 元素类型短名。
        @param seed 该张量的种子。
        @return ``torch.Tensor``。
        @raises TaskContractError 整数类型用了正态分布，或分布退化。
        """
        import torch

        spec = get_dtype(dtype)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed))
        if spec.is_float:
            if self.kind == "normal":
                values = torch.randn(shape, generator=generator, dtype=torch.float32) * float(self.std) + float(self.mean)
            else:
                if not self.high >= self.low:
                    raise TaskContractError(f"uniform sampling needs high >= low, got [{self.low}, {self.high}]")
                values = torch.rand(shape, generator=generator, dtype=torch.float32) * float(self.high - self.low) + float(
                    self.low
                )
            return values.to(spec.torch)
        # Integer tensors need an explicit uniform profile: a rounded normal draw would make
        # the expected value distribution an accident of the generator implementation.
        if self.kind != "uniform":
            raise TaskContractError(
                f"integer tensor sampling needs an explicit uniform profile for dtype {dtype!r}, got {self.kind!r}"
            )
        low, high = int(self.low), int(self.high)
        if float(low) != float(self.low) or float(high) != float(self.high):
            raise TaskContractError(f"integer uniform bounds must be integers, got [{self.low}, {self.high}]")
        if high < low:
            raise TaskContractError(f"uniform sampling needs high >= low, got [{low}, {high}]")
        return torch.randint(low, high + 1, shape, generator=generator, dtype=torch.int64).to(spec.torch)


@dataclass(frozen=True)
class SamplingPlan:
    """@brief 一个实例的完整采样计划：default + case 覆盖 + tensor 覆盖。"""

    default: SamplingProfile
    cases: Mapping[str, SamplingProfile]
    tensors: Mapping[str, SamplingProfile]

    def document(self) -> dict[str, Any]:
        """@brief 采样计划的 JSON 文档（用于 reference/任务身份）。

        @return 字段字典。
        """
        return {
            "default": self.default.document(),
            "cases": {name: profile.document() for name, profile in sorted(self.cases.items())},
            "tensors": {name: profile.document() for name, profile in sorted(self.tensors.items())},
        }

    def profile_for(self, *, case: str, tensor: str) -> SamplingProfile:
        """@brief 取某 (case, tensor) 的分布：tensor > case > default。

        @param case 用例名。
        @param tensor 张量名。
        @return 分布 profile。
        """
        return self.tensors.get(tensor) or self.cases.get(case) or self.default


def _profile_from_document(document: Mapping[str, Any], where: str) -> SamplingProfile:
    """@brief 由一个已通过 schema 的 profile 文档构造对象。

    @param document profile 文档。
    @param where 诊断位置。
    @return profile。
    @raises TaskContractError 类型未知。
    """
    kind = str(document.get("kind"))
    if kind == "normal":
        return SamplingProfile(kind="normal", mean=float(document["mean"]), std=float(document["std"]))
    if kind == "uniform":
        return SamplingProfile(kind="uniform", low=float(document["low"]), high=float(document["high"]))
    raise TaskContractError(f"{where}: unknown sampling kind {kind!r}")


def sampling_plan_from_document(
    sampling: Mapping[str, Any] | None,
    *,
    instance: SemanticInstanceV2,
    where: str = "reference.json.sampling",
) -> SamplingPlan:
    """@brief 由 ``reference.json.sampling`` 构造并校验一个采样计划。

    @param sampling 已通过 schema 的 ``sampling`` 块；``None`` 表示使用默认正态分布。
    @param instance 语义实例（用于校验 case/tensor 名字和整数 dtype）。
    @param where 诊断位置。
    @return 采样计划。
    @raises TaskContractError case/tensor 未声明，或整数张量没有显式均匀分布。
    """
    declared_cases = set(instance.case_domain.cases)
    initial_names = list(instance.reference_inputs)
    initial_set = set(initial_names)
    if sampling is None:
        default = SamplingProfile(**_DEFAULT_FLOAT)
        cases: dict[str, SamplingProfile] = {}
        tensors: dict[str, SamplingProfile] = {}
    else:
        default = _profile_from_document(sampling["default"], f"{where}.default")
        cases = {
            str(case): _profile_from_document(profile, f"{where}.cases.{case}")
            for case, profile in sampling.get("cases", {}).items()
        }
        tensors = {
            str(tensor): _profile_from_document(profile, f"{where}.tensors.{tensor}")
            for tensor, profile in sampling.get("tensors", {}).items()
        }
        unknown_cases = sorted(set(cases) - declared_cases)
        if unknown_cases:
            raise TaskContractError(f"{where}.cases: unknown cases {unknown_cases}; declared {sorted(declared_cases)}")
        unknown_tensors = sorted(set(tensors) - initial_set)
        if unknown_tensors:
            raise TaskContractError(
                f"{where}.tensors: {unknown_tensors} are not declared reference-initial tensors {sorted(initial_names)}"
            )
    plan = SamplingPlan(default=default, cases=cases, tensors=tensors)
    integer_inputs = [
        binding
        for binding in instance.tensors
        if binding.name in initial_set and not get_dtype(binding.dtype).is_float
    ]
    # The effective profile is tensor > case > default.  Validate every declared case ×
    # every integer initial tensor, not just the default: a legal per-case uniform override
    # must pass, and a case that falls back to a normal profile must fail at assembly rather
    # than at generate() time.
    for case in instance.case_domain.cases:
        for binding in integer_inputs:
            profile = plan.profile_for(case=case, tensor=binding.name)
            if profile.kind != "uniform":
                raise TaskContractError(
                    f"{where}: integer input {binding.name!r} ({binding.dtype}) in case {case!r} "
                    f"needs an explicit uniform profile, got {profile.kind!r}; the default normal draw "
                    "is only defined for float tensors"
                )
    return plan


def generate_inputs(instance: SemanticInstanceV2, plan: SamplingPlan, case: str, seed: int) -> dict[str, Any]:
    """@brief 为一个 case/seed 生成全部 reference 初值（含 output/read_write 初值）。

    @param instance 语义实例。
    @param plan 已校验的采样计划。
    @param case 用例名。
    @param seed 基础种子。
    @return 张量名 → ``torch.Tensor``。
    @raises TaskContractError case 未声明。
    """
    if case not in instance.case_domain.cases:
        raise TaskContractError(f"{instance.op_name}: case {case!r} is not declared")
    inputs: dict[str, Any] = {}
    index = 0
    initial_set = set(instance.reference_inputs)
    for binding in instance.tensors:
        if binding.name not in initial_set:
            continue
        profile = plan.profile_for(case=case, tensor=binding.name)
        inputs[binding.name] = profile.sample(binding.shape, binding.dtype, seed=seed + TENSOR_SEED_STRIDE * index)
        index += 1
    return inputs


__all__ = [
    "SamplingPlan",
    "SamplingProfile",
    "TENSOR_SEED_STRIDE",
    "generate_inputs",
    "sampling_plan_from_document",
]
