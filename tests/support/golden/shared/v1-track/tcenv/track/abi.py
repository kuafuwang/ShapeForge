"""``TargetAbiDeriver``: derive the ``TensorMemoryContract`` from an instance and a target (A2).

A1 froze the contract *shape* and refused to invent an ABI.  A2 derives it: the deriver is the
one place that turns

    (v2 SemanticInstance, target profile + its registered limits)

into the ``TensorMemoryContract`` the renderer shows, the builder binds and the evaluator
checks.  It locks exactly what the plan allows it to lock:

* tensor identity — name, role, dtype, memory space, access, concrete shape/strides;
* the accumulator dtype and the numeric tolerances of the instance;
* the buffer-slot budget of the target (a candidate's arguments must fit ``max_buffer_index``).

It deliberately contains **no** tile, grid, threads, num_warps, launch constant or fallback
geometry: those belong to the candidate's ``ExecutionGeometry``.  A workspace tensor is a
first-class ABI argument here (``space == "workspace"``); it is a scratch buffer, so it may not
be a reference input or a reference result.

The deriver reads only *generic* target fields (``raw_limits``).  A backend-specific limits view
(``TargetProfile.limits`` is the legacy Metal one) is not imported, and no chip knowledge is
embedded: the numbers always come from the target profile that the caller passes in.  Deriving
without a registered limit is an error, never a silent default.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from ..contract_errors import GeometryError, TaskContractError
from ..dtypes import get_dtype
from ..identity import fingerprint
from ..opspec.v2 import SemanticInstanceV2
from ..target import TargetProfile
from .plane import TensorMemoryArg, TensorMemoryContract

ABI_DERIVER_VERSION = 1
#: The only target limits an ABI derivation needs.  They are generic device facts, not chip
#: knowledge: how many argument slots exist, how many threads a group may hold, how much
#: threadgroup memory is available.
REQUIRED_TARGET_LIMITS: tuple[str, ...] = (
    "max_buffer_index",
    "max_threads_per_threadgroup",
    "max_threadgroup_memory_bytes",
)


@dataclass(frozen=True)
class TargetAbiLimits:
    """@brief 派生 ABI 所需的目标上限（来自 target 档案，不含芯片知识）。"""

    max_buffer_index: int
    max_threads_per_threadgroup: int
    max_threadgroup_memory_bytes: int
    #: Device buffer length.  ``None`` means "the caller has no device report yet" (CPU
    #: derivation); the size check is then skipped instead of inventing a limit.
    max_buffer_length: int | None = None

    def __post_init__(self) -> None:
        """@brief 校验上限为正整数。

        @raises TaskContractError 非法。
        """
        for name in ("max_buffer_index", "max_threads_per_threadgroup", "max_threadgroup_memory_bytes"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise TaskContractError(f"TargetAbiLimits.{name} must be an integer >= 1, got {value!r}")
        if self.max_buffer_length is not None:
            value = self.max_buffer_length
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise TaskContractError(f"TargetAbiLimits.max_buffer_length must be an integer >= 1, got {value!r}")

    @property
    def buffer_slots(self) -> int:
        """@brief 可用的 buffer 槽位数（下标 ``0..max_buffer_index``）。

        @return 槽位数量。
        """
        return self.max_buffer_index + 1

    def document(self) -> dict[str, Any]:
        """@brief 上限的 JSON 文档。

        @return 字段字典。
        """
        document: dict[str, Any] = {
            "max_buffer_index": int(self.max_buffer_index),
            "max_threads_per_threadgroup": int(self.max_threads_per_threadgroup),
            "max_threadgroup_memory_bytes": int(self.max_threadgroup_memory_bytes),
        }
        if self.max_buffer_length is not None:
            document["max_buffer_length"] = int(self.max_buffer_length)
        return document


def abi_limits_from_target(profile: TargetProfile, *, max_buffer_length: int | None = None) -> TargetAbiLimits:
    """@brief 由 target 档案的通用上限派生 :class:`TargetAbiLimits`。

    @param profile 目标档案。
    @param max_buffer_length 设备报告的缓冲长度；CPU 派生时省略。
    @return 上限。
    @raises TaskContractError 档案缺少 ABI 派生所需的通用上限。
    """
    limits: Mapping[str, Any] = profile.raw_limits or {}
    missing = [key for key in REQUIRED_TARGET_LIMITS if key not in limits]
    if missing:
        raise TaskContractError(
            f"target {profile.id!r} does not register the ABI limits {missing}; the deriver does not invent "
            "a device limit"
        )
    return TargetAbiLimits(
        max_buffer_index=int(limits["max_buffer_index"]),
        max_threads_per_threadgroup=int(limits["max_threads_per_threadgroup"]),
        max_threadgroup_memory_bytes=int(limits["max_threadgroup_memory_bytes"]),
        max_buffer_length=max_buffer_length,
    )


@dataclass(frozen=True)
class TargetAbiDeriver:
    """@brief 唯一的 ABI 派生器：``(SemanticInstance, target, limits) -> TensorMemoryContract``。"""

    backend_id: str
    target_id: str
    limits: TargetAbiLimits
    version: int = ABI_DERIVER_VERSION

    def __post_init__(self) -> None:
        """@brief 校验派生器身份。

        @raises TaskContractError 身份为空或版本不支持。
        """
        if not self.backend_id or not self.target_id:
            raise TaskContractError("TargetAbiDeriver needs a non-empty backend_id and target_id")
        if self.version != ABI_DERIVER_VERSION:
            raise TaskContractError(f"unsupported TargetAbiDeriver version {self.version}")

    @classmethod
    def for_target(cls, profile: TargetProfile, *, max_buffer_length: int | None = None) -> "TargetAbiDeriver":
        """@brief 由目标档案构造派生器。

        @param profile 目标档案。
        @param max_buffer_length 设备缓冲长度；CPU 派生时省略。
        @return 派生器。
        """
        return cls(
            backend_id=profile.backend,
            target_id=profile.id,
            limits=abi_limits_from_target(profile, max_buffer_length=max_buffer_length),
        )

    def derive(self, instance: SemanticInstanceV2) -> TensorMemoryContract:
        """@brief 由语义实例派生 ABI 合同。

        @param instance v2 语义实例。
        @return 派生出的 ``TensorMemoryContract``。
        @raises TaskContractError 实例与该目标的槽位/尺寸上限不兼容。
        """
        if not isinstance(instance, SemanticInstanceV2):
            raise TaskContractError("TargetAbiDeriver.derive needs a v2 SemanticInstance")
        tensors = tuple(_arg(binding) for binding in instance.tensors)
        if len(tensors) > self.limits.buffer_slots:
            raise TaskContractError(
                f"{instance.op_name}: {len(tensors)} ABI tensors do not fit the {self.limits.buffer_slots} buffer "
                f"slots of target {self.target_id}"
            )
        for tensor in tensors:
            span = tensor_span_bytes(tensor.shape, tensor.strides, get_dtype(tensor.dtype).nbytes)
            if self.limits.max_buffer_length is not None and span > self.limits.max_buffer_length:
                raise TaskContractError(
                    f"{instance.op_name}.{tensor.name}: {span} bytes exceed the device buffer length "
                    f"{self.limits.max_buffer_length} of target {self.target_id}"
                )
        self._check_workspace(instance, tensors)
        try:
            return TensorMemoryContract(
                abi_id=f"abi.{instance.op_name}.{instance.dtype}.{instance.size}.{self.target_id}",
                backend_id=self.backend_id,
                target_id=self.target_id,
                tensors=tensors,
                acc_dtype=instance.acc_dtype,
                rtol=float(instance.tolerances.rtol),
                atol_rel=float(instance.tolerances.atol_rel),
                exact=bool(instance.tolerances.exact),
            )
        except TaskContractError as exc:
            raise TaskContractError(f"{instance.op_name}.{instance.dtype}.{instance.size}: {exc}") from exc

    @staticmethod
    def _check_workspace(instance: SemanticInstanceV2, tensors: tuple[TensorMemoryArg, ...]) -> None:
        """@brief 校验 workspace 张量不进入 reference 合同。

        @param instance 语义实例。
        @param tensors 派生出的张量。
        @raises TaskContractError workspace 张量被声明为 reference 初值/结果。
        """
        workspace = [tensor.name for tensor in tensors if tensor.space == "workspace"]
        if not workspace:
            return
        inputs = set(instance.reference_inputs)
        outputs = set(instance.reference_outputs)
        leaked = sorted(set(workspace) & (inputs | outputs))
        if leaked:
            raise TaskContractError(
                f"{instance.op_name}: workspace tensors {leaked} are scratch buffers and cannot be a reference "
                "input or reference result"
            )


def _arg(binding: Any) -> TensorMemoryArg:
    """@brief 把一个实例张量绑定投影成 ABI 参数。

    @param binding ``TensorBinding``。
    @return ABI 参数。
    @raises GeometryError 形状/步长不是具体整数。
    """
    shape = tuple(_concrete(term, f"{binding.name}.shape") for term in binding.shape)
    strides = tuple(_concrete(term, f"{binding.name}.strides") for term in binding.strides)
    return TensorMemoryArg(
        name=binding.name,
        role=binding.role,
        dtype=binding.dtype,
        space=binding.space,
        access=binding.access,
        shape=shape,
        strides=strides,
        doc=binding.doc,
    )


def _concrete(term: Any, where: str) -> int:
    """@brief 要求一个实例维度/步长是具体整数。

    @param term 术语。
    @param where 诊断位置。
    @return 整数。
    @raises GeometryError 术语不是整数。
    """
    if isinstance(term, bool) or not isinstance(term, int):
        raise GeometryError(f"{where}: the ABI derivation needs a concrete integer, got {term!r}")
    if term < 0:
        raise GeometryError(f"{where}: the ABI derivation needs a non-negative dimension, got {term!r}")
    return int(term)


def derive_tensor_memory_contract(
    instance: SemanticInstanceV2,
    *,
    target: TargetProfile,
    max_buffer_length: int | None = None,
) -> TensorMemoryContract:
    """@brief 唯一入口：``(instance, target) -> TensorMemoryContract``。

    @param instance v2 语义实例。
    @param target 目标档案。
    @param max_buffer_length 设备报告的缓冲长度；CPU 派生时省略。
    @return 派生出的 ABI 合同。
    @raises TaskContractError 目标缺上限或实例不兼容。
    """
    return TargetAbiDeriver.for_target(target, max_buffer_length=max_buffer_length).derive(instance)


def derived_abi_digest(contract: TensorMemoryContract) -> str:
    """@brief 派生 ABI 的稳定摘要（ABI 文档本身，不含几何）。

    @param contract ABI 合同。
    @return 十六进制 sha256。
    """
    return fingerprint(
        {
            "fingerprint_kind": "tensor-memory-contract",
            "fingerprint_version": 1,
            "contract": contract.document(),
        }
    )


def tensor_span_bytes(shape: tuple[int, ...], strides: tuple[int, ...], elem_size: int) -> int:
    """@brief 一个（可能 strided 的）张量实际可寻址的字节跨度。

    ``numel * elem_size`` is only the span of a contiguous tensor.  A strided ABI tensor needs
    ``max_offset + 1`` elements, otherwise the buffer-length check and the binding record would
    under-count exactly the buffers a candidate can address.

    @param shape 具体维度。
    @param strides 具体步长（元素数）。
    @param elem_size 单元素字节数。
    @return 字节跨度；空/零元张量为 0。
    @raises GeometryError 维度/步长/元素尺寸非法。
    """
    if len(shape) != len(strides):
        raise GeometryError("a tensor span needs one stride per dimension")
    if isinstance(elem_size, bool) or not isinstance(elem_size, int) or elem_size < 1:
        raise GeometryError(f"tensor element size must be a positive integer, got {elem_size!r}")
    if not shape:
        return 0
    max_offset = 0
    for dim, stride in zip(shape, strides):
        if isinstance(dim, bool) or not isinstance(dim, int) or dim < 0:
            raise GeometryError(f"tensor dimension must be a non-negative integer, got {dim!r}")
        if isinstance(stride, bool) or not isinstance(stride, int) or stride < 0:
            raise GeometryError(f"tensor stride must be a non-negative integer, got {stride!r}")
        if dim == 0:
            return 0
        max_offset += (dim - 1) * stride
    return (max_offset + 1) * elem_size


__all__ = [
    "ABI_DERIVER_VERSION",
    "REQUIRED_TARGET_LIMITS",
    "TargetAbiDeriver",
    "TargetAbiLimits",
    "abi_limits_from_target",
    "derive_tensor_memory_contract",
    "derived_abi_digest",
    "tensor_span_bytes",
]
