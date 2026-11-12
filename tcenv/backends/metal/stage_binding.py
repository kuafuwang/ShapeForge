"""MSL-reflection driven per-stage argument binding for the v2 candidate plane (A2).

The v1 path checks one contract against one compiled kernel.  A v2 candidate owns its geometry, so
each stage has its **own** argument list: some stages bind only a subset of the tensors, some bind
workspace, and some bind extra named ``Derived Scalars`` that the harness must set by name with
``setBytes``.  This module turns the compiler's own reflection into that binding:

* every reflected buffer parameter must resolve to a contract tensor or to a derived scalar of
  *this* stage — an unknown name is a hard error, never "extra argument";
* a stage's parameter slots must be the contiguous range ``0..n-1`` (the plan's contiguous-slot
  rule), with no duplicate slot and no duplicate name;
* access is checked per tensor: ``read`` must be ``device const`` (reflection access ``read``),
  ``write`` must be writable, ``read_write`` must be ``readwrite``;
* widths are checked: a buffer may be vectorized (``half``/``half2``/``half4``) but the base
  scalar type must be the tensor's own MSL type, and a derived scalar must be a 4-byte ``int``/
  ``uint`` scalar set by name;
* every result tensor (pure write output or ``read_write`` tensor) must be written by at least
  one active stage — an unwritten output is caught before the GPU runs;
* inactive stages are not part of the instance ABI at all (a zero-active-stage geometry is already
  refused by the geometry resolver).

The record type here is deliberately a plain frozen dataclass rather than the runtime's
``Binding``: the binding layer must be testable on CPU with no Metal import, and the build adapter
(A4) converts ``MTLDataType`` into the MSL scalar name once, at the edge.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence

from ...contract_errors import GeometryError
from ...dtypes import get_dtype
from ...track.abi import TargetAbiLimits, tensor_span_bytes
from ...scratch import ResolvedScratch
from ...track.geometry import ResolvedGeometry, ResolvedStage
from ...track.plane import TensorMemoryArg, TensorMemoryContract

BINDING_SCHEMA_VERSION = 1
#: MSL scalar type names a derived scalar may use: it is an integer, set with ``setBytes``.
DERIVED_SCALAR_TYPES = frozenset({"int", "uint", "long", "ulong"})
#: Vector widths a pointer parameter may use over the same buffer.  ``3`` is absent on purpose:
#: MSL ``half3``/``float3`` are padded (size 8/16), so a 3-wide element would not be
#: ``base * width`` bytes; the padded size is still accepted as the corresponding 4-wide vector.
VECTOR_WIDTHS = (1, 2, 4, 8)
_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


@dataclass(frozen=True)
class ReflectedArgument:
    """@brief 编译器反射出的一个参数（与 Metal 库解耦的纯记录）。"""

    index: int
    name: str
    kind: str
    access: str
    scalar_type: str | None = None
    elem_size: int | None = None
    used: bool = True

    def __post_init__(self) -> None:
        """@brief 校验记录字段。

        @raises GeometryError 非法 kind/access 或字段缺失。
        """
        if self.kind not in ("buffer", "threadgroup", "texture", "other"):
            raise GeometryError(f"reflection {self.name!r}: unknown parameter kind {self.kind!r}")
        if self.access not in ("read", "write", "readwrite"):
            raise GeometryError(f"reflection {self.name!r}: unknown access {self.access!r}")
        if self.index < 0:
            raise GeometryError(f"reflection {self.name!r}: negative slot {self.index}")
        if self.kind in ("buffer", "texture"):
            if not self.name or not _IDENTIFIER.fullmatch(self.name):
                raise GeometryError(f"reflection: {self.kind} parameter name {self.name!r} is not an identifier")
        if self.kind == "buffer":
            if not self.scalar_type:
                raise GeometryError(f"reflection {self.name!r}: a buffer parameter needs its scalar type")
            if not self.elem_size or self.elem_size < 1:
                raise GeometryError(f"reflection {self.name!r}: a buffer parameter needs a positive element size")


@dataclass(frozen=True)
class StageArgument:
    """@brief 一个 stage 的一个已绑定参数。"""

    slot: int
    name: str
    role: str
    dtype: str
    access: str
    nbytes: int
    value: int | None = None
    binding_kind: str = "buffer"
    """``buffer`` 或 ``texture``。``document()`` 只在非 buffer 时写出，既有 buffer 绑定文档不变。"""

    def document(self) -> dict:
        """@brief 该参数的 JSON 文档。

        @return 字段字典。
        """
        document = {
            "slot": int(self.slot),
            "name": self.name,
            "role": self.role,
            "dtype": self.dtype,
            "access": self.access,
            "nbytes": int(self.nbytes),
        }
        if self.value is not None:
            document["value"] = int(self.value)
        if self.binding_kind != "buffer":
            document["binding"] = self.binding_kind
        return document


@dataclass(frozen=True)
class StageBinding:
    """@brief 一个 stage 的完整参数绑定。"""

    stage_index: int
    kernel: str
    arguments: tuple[StageArgument, ...]
    schema_version: int = BINDING_SCHEMA_VERSION

    @property
    def set_bytes(self) -> tuple[tuple[int, str, int], ...]:
        """@brief 需要按名 ``setBytes`` 的派生标量。

        @return ``(slot, name, value)`` 元组。
        """
        return tuple(
            (argument.slot, argument.name, int(argument.value))
            for argument in self.arguments
            if argument.role == "derived_scalar"
        )

    def document(self) -> dict:
        """@brief 绑定的 JSON 文档。

        @return 字段字典。
        """
        return {
            "schema_version": self.schema_version,
            "stage_index": int(self.stage_index),
            "kernel": self.kernel,
            "arguments": [argument.document() for argument in self.arguments],
        }


def bind_stage_arguments(
    stage: ResolvedStage,
    *,
    contract: TensorMemoryContract,
    reflection: Sequence[ReflectedArgument],
    limits: TargetAbiLimits,
    where: str = "stage",
    scratch: Sequence[ResolvedScratch] = (),
    poison_producers: frozenset[str] = frozenset(),
) -> StageBinding:
    """@brief 用 MSL 反射为一个 stage 生成按名绑定。

    @param stage 已求值 stage。
    @param contract ABI 合同。
    @param reflection 该 kernel 的反射参数。
    @param limits 目标上限（buffer 槽位数）。
    @param where 诊断前缀。
    @return 绑定。
    @raises GeometryError 任何一项不一致。
    """
    if not stage.active:
        raise GeometryError(
            f"{where}[{stage.index}]: an inactive stage is not part of the instance ABI and must not be bound"
        )
    prefix = f"{where}[{stage.index}:{stage.kernel}]"
    buffers: dict[int, ReflectedArgument] = {}
    textures: dict[int, ReflectedArgument] = {}
    names: dict[str, int] = {}
    for argument in reflection:
        table = buffers if argument.kind == "buffer" else textures if argument.kind == "texture" else None
        if table is None:
            if argument.kind == "threadgroup":
                raise GeometryError(
                    f"{prefix}: dynamic threadgroup parameter {argument.name!r} is not part of the ABI; declare "
                    "`threadgroup T name[N];` inside the kernel body instead"
                )
            raise GeometryError(f"{prefix}: parameter {argument.name!r} is neither a buffer, a texture, nor a threadgroup array")
        if argument.index in table:
            raise GeometryError(f"{prefix}: two parameters share {argument.kind} slot {argument.index}")
        if argument.name in names:
            raise GeometryError(f"{prefix}: parameter name {argument.name!r} is bound twice")
        if argument.index > limits.max_buffer_index:
            raise GeometryError(
                f"{prefix}: {argument.kind} slot {argument.index} exceeds the target limit {limits.max_buffer_index}"
            )
        table[argument.index] = argument
        names[argument.name] = argument.index

    derived = dict(stage.derived_scalars)
    scratch_by_name = {spec.name: spec for spec in scratch}
    arguments: list[StageArgument] = []
    for slot in sorted(buffers):
        reflected = buffers[slot]
        if reflected.name in derived:
            arguments.append(_derived_argument(prefix, reflected, derived[reflected.name]))
            continue
        if reflected.name in scratch_by_name:
            arguments.append(
                _scratch_argument(
                    prefix,
                    reflected,
                    stage,
                    scratch_by_name[reflected.name],
                    poison_producer=reflected.name in poison_producers,
                )
            )
            continue
        tensor = _tensor(contract, reflected.name)
        if tensor is None:
            raise GeometryError(
                f"{prefix}: buffer {reflected.name!r} is neither an ABI tensor nor a derived scalar of this stage "
                f"(tensors: {sorted(t.name for t in contract.tensors)}; derived: {sorted(derived)})"
            )
        if tensor.space == "workspace" and tensor.name not in {use.name for use in stage.workspace}:
            raise GeometryError(
                f"{prefix}: workspace {tensor.name!r} is bound by the kernel but not declared by this stage; "
                "a workspace must be declared in the stage's workspace plan so its zero/fill policy and dispatch "
                "evidence are authoritative"
            )
        arguments.append(_tensor_argument(prefix, reflected, tensor))
    for slot in sorted(textures):
        reflected = textures[slot]
        spec = scratch_by_name.get(reflected.name)
        if spec is None:
            raise GeometryError(
                f"{prefix}: texture {reflected.name!r} is not a scratch buffer of this geometry; "
                "ABI tensors stay in the buffer binding space"
            )
        arguments.append(
            _scratch_argument(
                prefix,
                reflected,
                stage,
                spec,
                poison_producer=reflected.name in poison_producers,
            )
        )

    for kind, table in (("buffer", buffers), ("texture", textures)):
        slots = sorted(table)
        if slots != list(range(len(slots))):
            raise GeometryError(
                f"{prefix}: {kind} slots must be contiguous from 0; the kernel uses {slots}. A stage owns its own "
                "contiguous slot range in each binding space"
            )
    # Result-write completeness is a pipeline property, not a per-stage one: one stage may
    # legitimately bind only a subset of the tensors (see bind_pipeline_arguments).
    unbound_scratch = sorted(use.name for use in stage.scratch if use.name not in names)
    if unbound_scratch:
        raise GeometryError(
            f"{prefix}: scratch {unbound_scratch} is declared by this stage but not bound by the kernel"
        )
    unbound_derived = sorted(name for name in derived if name not in names)
    if unbound_derived:
        raise GeometryError(
            f"{prefix}: derived scalars {unbound_derived} are declared by the geometry but not bound by the kernel; "
            "every derived scalar must reach the device through its named scalar parameter"
        )
    return StageBinding(stage_index=stage.index, kernel=stage.kernel, arguments=tuple(arguments))


def _tensor(contract: TensorMemoryContract, name: str) -> TensorMemoryArg | None:
    """@brief 按名字取 ABI 张量。

    @param contract ABI 合同。
    @param name 张量名。
    @return 张量或 None。
    """
    try:
        return contract.tensor(name)
    except KeyError:
        return None


def _tensor_argument(prefix: str, reflected: ReflectedArgument, tensor: TensorMemoryArg) -> StageArgument:
    """@brief 校验并构造一个张量参数。

    @param prefix 诊断前缀。
    @param reflected 反射记录。
    @param tensor ABI 张量。
    @return 参数。
    @raises GeometryError 访问/类型/宽度不一致。
    """
    dtype = get_dtype(tensor.dtype)
    if tensor.access == "read" and reflected.access != "read":
        raise GeometryError(
            f"{prefix}: input {tensor.name!r} is writable in the kernel (reflection says {reflected.access}); an "
            "input must stay `device const`"
        )
    if tensor.access == "write" and reflected.access == "read":
        raise GeometryError(f"{prefix}: output {tensor.name!r} is read-only in the kernel; it must be writable")
    if tensor.access == "read_write" and reflected.access != "readwrite":
        raise GeometryError(
            f"{prefix}: read_write tensor {tensor.name!r} reflects access {reflected.access}; a read_write buffer "
            "must be `device` (non-const) and read before it is written"
        )
    if reflected.scalar_type != dtype.msl:
        raise GeometryError(
            f"{prefix}: {tensor.name!r} is {tensor.dtype} (`{dtype.msl}` in MSL) but the kernel declares "
            f"`{reflected.scalar_type}`; the base scalar type must match even when the pointer is vectorized"
        )
    elem = int(reflected.elem_size or 0)
    base = int(dtype.nbytes)
    if elem not in {base * width for width in VECTOR_WIDTHS}:
        raise GeometryError(
            f"{prefix}: {tensor.name!r} reflects a {elem}-byte element, which is not a `{dtype.msl}` scalar or a "
            f"`{dtype.msl}N` vector (base element is {base} bytes)"
        )
    nbytes = tensor_span_bytes(tuple(int(dim) for dim in tensor.shape), tuple(int(s) for s in tensor.strides), base)
    return StageArgument(
        slot=reflected.index,
        name=tensor.name,
        role="tensor",
        dtype=tensor.dtype,
        access=tensor.access,
        nbytes=nbytes,
    )


def _scratch_argument(
    prefix: str,
    reflected: ReflectedArgument,
    stage: ResolvedStage,
    spec: ResolvedScratch,
    *,
    poison_producer: bool,
) -> StageArgument:
    """@brief 校验并构造一个 scratch 参数。角色是 ``scratch``，不算结果张量。

    ``uninitialized_poison`` 的第一次使用必须是声明 ``write`` 且反射 ``write``。这台机器上
    非 const 的 ``device T*`` 一律反射成 ``readwrite``，所以 producer 要用
    ``texture2d<float, access::write>``。``device_zero`` / ``host_written`` 的 buffer 写仍接受
    ``write`` 或 ``readwrite``，因为那条规则只约束 poison 的第一次使用。
    """
    declared = next((use for use in stage.scratch if use.name == spec.name), None)
    if declared is None:
        raise GeometryError(
            f"{prefix}: scratch {spec.name!r} is bound by the kernel but not declared by this stage"
        )
    if poison_producer and (declared.access != "write" or reflected.access != "write"):
        raise GeometryError(
            f"{prefix}: the first use of uninitialized_poison scratch {spec.name!r} is declared "
            f"{declared.access} and reflects {reflected.access}; it must be bound write "
            "(a device pointer reflects readwrite and is refused)"
        )
    if not poison_producer and declared.access == "write" and reflected.access == "read":
        raise GeometryError(
            f"{prefix}: scratch {spec.name!r} is declared write but the kernel reflects read"
        )
    if declared.access == "read" and reflected.access != "read":
        raise GeometryError(
            f"{prefix}: scratch {spec.name!r} is declared read but the kernel reflects {reflected.access}"
        )
    if declared.access == "read_write" and reflected.access != "readwrite":
        raise GeometryError(
            f"{prefix}: scratch {spec.name!r} is declared read_write but the kernel reflects {reflected.access}"
        )
    if reflected.kind == "texture":
        if spec.dtype != "fp32":
            raise GeometryError(
                f"{prefix}: scratch {spec.name!r} is bound as a texture but its dtype is {spec.dtype}; "
                "only fp32 scratch has a linear R32Float view"
            )
        return StageArgument(
            slot=reflected.index,
            name=spec.name,
            role="scratch",
            dtype=spec.dtype,
            access=declared.access,
            nbytes=int(spec.nbytes),
            binding_kind="texture",
        )
    dtype = get_dtype(spec.dtype)
    if reflected.scalar_type != dtype.msl:
        raise GeometryError(
            f"{prefix}: scratch {spec.name!r} is {spec.dtype} (`{dtype.msl}`) but the kernel declares "
            f"`{reflected.scalar_type}`"
        )
    elem = int(reflected.elem_size or 0)
    base = int(dtype.nbytes)
    if elem not in {base * width for width in VECTOR_WIDTHS}:
        raise GeometryError(
            f"{prefix}: scratch {spec.name!r} reflects a {elem}-byte element, not a `{dtype.msl}` scalar or vector"
        )
    return StageArgument(
        slot=reflected.index,
        name=spec.name,
        role="scratch",
        dtype=spec.dtype,
        access=declared.access,
        nbytes=int(spec.nbytes),
    )


def _derived_argument(prefix: str, reflected: ReflectedArgument, value: int) -> StageArgument:
    """@brief 校验并构造一个派生标量参数。

    @param prefix 诊断前缀。
    @param reflected 反射记录。
    @param value 派生标量的值。
    @return 参数。
    @raises GeometryError 类型/宽度/权限不一致。
    """
    if reflected.scalar_type not in DERIVED_SCALAR_TYPES:
        raise GeometryError(
            f"{prefix}: derived scalar {reflected.name!r} must be an integer scalar "
            f"({sorted(DERIVED_SCALAR_TYPES)}), got `{reflected.scalar_type}`"
        )
    width = {"int": 4, "uint": 4, "long": 8, "ulong": 8}[reflected.scalar_type]
    if int(reflected.elem_size or 0) != width:
        raise GeometryError(
            f"{prefix}: derived scalar {reflected.name!r} reflects a {reflected.elem_size}-byte element; "
            f"`{reflected.scalar_type}` is {width} bytes"
        )
    if reflected.access != "read":
        raise GeometryError(
            f"{prefix}: derived scalar {reflected.name!r} must be `constant`/read-only; setBytes writes it once per "
            "dispatch and the kernel must not modify it"
        )
    return StageArgument(
        slot=reflected.index,
        name=reflected.name,
        role="derived_scalar",
        dtype=reflected.scalar_type,
        access="read",
        nbytes=width,
        value=int(value),
    )


def _first_poison_producers(geometry: ResolvedGeometry) -> dict[int, frozenset[str]]:
    """@brief 每个 ``uninitialized_poison`` scratch 第一次被激活 stage 使用时的 stage 下标。"""
    specs = {spec.name: spec for spec in geometry.scratch}
    first: dict[str, int] = {}
    for stage in geometry.active_stages:
        for use in getattr(stage, "scratch", ()) or ():
            spec = specs.get(use.name)
            if spec is None or spec.init_policy != "uninitialized_poison" or use.name in first:
                continue
            first[use.name] = stage.index
    grouped: dict[int, set[str]] = {}
    for name, index in first.items():
        grouped.setdefault(index, set()).add(name)
    return {index: frozenset(names) for index, names in grouped.items()}


def bind_pipeline_arguments(
    geometry: ResolvedGeometry,
    *,
    contract: TensorMemoryContract,
    reflection: dict[str, Sequence[ReflectedArgument]],
    limits: TargetAbiLimits,
    where: str = "pipeline",
) -> tuple[StageBinding, ...]:
    """@brief 绑定整条流水线的所有激活 stage，并检查结果张量写入完整性。

    @param geometry 已求值几何。
    @param contract ABI 合同。
    @param reflection kernel 名 → 反射参数。
    @param limits 目标上限。
    @param where 诊断前缀。
    @return 逐 stage 绑定。
    @raises GeometryError 某 stage 反射缺失、绑定不一致或某个结果没有被任何激活 stage 写入。
    """
    bindings: list[StageBinding] = []
    written: set[str] = set()
    producers = _first_poison_producers(geometry)
    for stage in geometry.active_stages:
        items = reflection.get(stage.kernel)
        if items is None:
            raise GeometryError(f"{where}: stage {stage.index} needs kernel {stage.kernel!r}, which has no reflection")
        binding = bind_stage_arguments(
            stage,
            contract=contract,
            reflection=items,
            limits=limits,
            where=where,
            scratch=geometry.scratch,
            poison_producers=producers.get(stage.index, frozenset()),
        )
        bindings.append(binding)
        written |= {
            argument.name for argument in binding.arguments if argument.role == "tensor" and argument.access in ("write", "read_write")
        }
    missing = sorted(
        tensor.name
        for tensor in contract.tensors
        if tensor.space != "workspace" and tensor.access in ("write", "read_write") and tensor.name not in written
    )
    if missing:
        raise GeometryError(
            f"{where}: no active stage writes the result tensors {missing}; an unwritten result is refused before "
            "the GPU runs"
        )
    return tuple(bindings)


__all__ = [
    "BINDING_SCHEMA_VERSION",
    "DERIVED_SCALAR_TYPES",
    "VECTOR_WIDTHS",
    "ReflectedArgument",
    "StageArgument",
    "StageBinding",
    "bind_pipeline_arguments",
    "bind_stage_arguments",
]
