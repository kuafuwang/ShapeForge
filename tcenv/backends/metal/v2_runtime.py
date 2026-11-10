"""A4: real Metal execution of the v2 ABI/geometry/lifecycle plane.

The v1 evaluator still owns one ``Contract`` and one legacy ``ArgSet``.  A v2 candidate owns a
``TensorMemoryContract``, a resolved ``ExecutionGeometry`` and a candidate Host Tiling plan, so
this module supplies the missing execution half without changing v1:

* buffers are sized from the ABI (including a workspace tensor's own span), never from a
  candidate's ``M*2`` launch guess;
* ``read_write`` is a first-class access: every correctness trial restores its initial bytes,
  it is compared as a result and it is excluded from the "never written" statistic;
* correctness fills pure outputs poison/zero/poison; benchmark zeroes only a workspace the
  candidate declared ``zero``;
* stage arguments come from ``bind_pipeline_arguments`` and the compiler reflection, so a
  ``Derived Scalar`` reaches the GPU by ``setBytes`` at its reflected slot;
* every dispatch is recorded by the runtime loop that actually calls
  ``dispatchThreadgroups`` (not by re-parsing the source), including the actual provider and
  workspace names;
* the candidate Host Tiling entry is a declarative host plan executed on the CPU before the
  device stages; it produces workspace resources/launch-side data and never a reference or an
  input preprocessing step.

Nothing here knows an operator name, a shape, a tile or a grid.  The only algorithm facts come
from the candidate, its ABI and its resolved geometry.
"""

from __future__ import annotations

import hashlib
import json
import os
import struct
import time
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import torch

from ...contract_errors import GeometryError, TaskContractError, VarlenContractError
from ...dtypes import get_dtype
from ...portfolio.domain import evaluation_input_digest
from ...taskcontract.loader import TaskContract
from ...taskcontract.reference import broadcasts_scale
from ...taskcontract.runtime_shape import (
    cu_seqlens_problems,
    declared_dim_terms,
    logical_region,
    padding_byte_problems,
    padding_slabs,
    require_capacity_axes,
    resolved_shape_for_case,
)
from ...taskcontract.varlen import is_varlen, segment_lengths
from ...taskcontract.varlen_cases import case_realization
from ...track import (
    CORRECTNESS_TRIALS,
    GUARD_BYTES,
    GUARD_PATTERN,
    POISON_PATTERN,
    RESTORE_INITIAL,
    ZERO,
    ResolvedGeometry,
    ResolvedStage,
    TensorMemoryContract,
    TimingDiscipline,
    WorkspaceUse,
    benchmark_launch_plan,
    correctness_launch_plan,
    tensor_span_bytes,
    workspace_fills,
)
from ...track.expr import INT, evaluate_term
from ...tensors import from_raw, round_to, to_raw
from .launch import LaunchSpec
from .host_program import HostProgramFailed, HostProgramTiming, MetalHostProgram
from .sanitize import GuardedBuffer
from .stage_binding import ReflectedArgument, StageArgument, StageBinding, bind_pipeline_arguments
from .testing import CaseResult, _floor, scaled_error

HOST_TILING_SCHEMA_ID = "tcenv.metal.host-tiling.1"
HOST_TILING_SCHEMA_VERSION = 1
HOST_TILING_FIELDS = frozenset({"schema_id", "schema_version", "writes"})
HOST_TILING_WRITE_FIELDS = frozenset({"workspace", "kind", "start", "step", "count", "stride", "offset", "dtype"})


def _names(instance: Any) -> dict[str, int]:
    names = {str(k): int(v) for k, v in dict(instance.symbols).items()}
    names.update(
        {
            str(k): int(v)
            for k, v in dict(instance.resolved_attrs).items()
            if isinstance(v, int) and not isinstance(v, bool)
        }
    )
    return names


def _integer(term: Any, instance: Any, where: str) -> int:
    value = evaluate_term(term, _names(instance), where=where, expect=INT)
    return int(value)


@dataclass(frozen=True)
class HostTilingWrite:
    """@brief 一条 host 侧写入：把线性序列写到 workspace 的指定步长位置。"""

    workspace: str
    kind: str
    start: int
    step: int
    count: int
    stride: int
    offset: int
    dtype: str

    def document(self) -> dict[str, Any]:
        return {
            "workspace": self.workspace,
            "kind": self.kind,
            "start": int(self.start),
            "step": int(self.step),
            "count": int(self.count),
            "stride": int(self.stride),
            "offset": int(self.offset),
            "dtype": self.dtype,
        }


def host_write_span(write: HostTilingWrite, abi_nbytes: int) -> list[str]:
    """@brief Check one host-tiling write against the ABI workspace span (pure, CPU only).

    The preflight and :meth:`HostTilingPlan.execute` share this one function so the rules cannot
    drift apart.  The span rule matches the execution semantics exactly:

    * ``stride >= 1`` unconditionally -- ``count == 0`` does not relax it;
    * ``count == 0`` means ``span == 0`` (the naive ``offset + (count-1)*stride + 1`` would be
      negative and would wrongly reject an accepted empty write);
    * ``count > 0`` means ``span = offset + (count-1)*stride + 1`` and requires ``offset >= 0``;
    * ``span * nbytes <= abi_nbytes``.

    ``offset >= 0`` is stricter than the historical execution check (which accepted a negative
    first address that produced a non-negative span); the first write would land before the
    buffer, so the preflight refuses it instead of letting the runtime write out of bounds.

    @param write one parsed host write.
    @param abi_nbytes the ABI workspace span in bytes.
    @return problem strings; empty when the write is legal.
    """
    nbytes = get_dtype(write.dtype).nbytes
    problems: list[str] = []
    if write.count < 0:
        problems.append(f"host tiling write targets workspace {write.workspace!r} with a negative count {write.count}")
    if write.stride < 1:
        problems.append(
            f"host tiling write targets workspace {write.workspace!r} with stride {write.stride}; the stride must be "
            "at least 1 even when count is 0"
        )
    if write.count > 0 and write.offset < 0:
        problems.append(
            f"host tiling write targets workspace {write.workspace!r} with a negative first index {write.offset}"
        )
    span = write.offset + (write.count - 1) * write.stride + 1 if write.count > 0 else 0
    if write.count < 0 or write.stride < 1 or span < 0:
        # The shape of the write itself is already refused above; do not add a span problem for
        # an expression whose span is meaningless.
        return problems
    if span * nbytes > abi_nbytes:
        problems.append(
            f"host tiling write targets {span * nbytes} bytes of workspace {write.workspace!r}, but the ABI "
            f"workspace is {abi_nbytes} bytes"
        )
    return problems


@dataclass(frozen=True)
class HostTilingPlan:
    """@brief 候选 Host Tiling entry 的闭集声明（资源/launch-side 数据）。"""

    writes: tuple[HostTilingWrite, ...]
    source_sha256: str
    path: str = "host/tiling.json"

    @classmethod
    def from_candidate(cls, candidate: Any, contract: TaskContract) -> "HostTilingPlan | None":
        """@brief 读取候选唯一的 host_tiling entry；没有时返回 None。

        @param candidate CandidateArtifact。
        @param contract v2 任务合同。
        @return plan 或 None。
        @raises TaskContractError 多条 entry、非 UTF-8/JSON、未知字段、非 workspace 写入或类型不符。
        """
        entries = [entry for entry in candidate.entries if entry.role == "host_tiling"]
        if not entries:
            return None
        if len(entries) != 1:
            raise TaskContractError(f"an A4 bundle needs exactly one host_tiling entry, got {len(entries)}")
        entry = entries[0]
        try:
            document = json.loads(entry.data.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise TaskContractError(f"host tiling entry {entry.path!r} is not JSON: {exc}") from exc
        except ValueError as exc:
            # ``json.loads`` raises a bare ``ValueError`` for an integer field longer than the
            # interpreter's digit limit (and a bare ``RecursionError`` for a deeply nested one).
            raise TaskContractError(f"host tiling entry {entry.path!r} is not decodable JSON: {exc}") from exc
        except RecursionError as exc:
            raise TaskContractError(f"host tiling entry {entry.path!r} is nested too deeply: {exc}") from exc
        if not isinstance(document, Mapping) or set(document) != HOST_TILING_FIELDS:
            raise TaskContractError(
                f"host tiling entry {entry.path!r} must have exactly {sorted(HOST_TILING_FIELDS)}"
            )
        if document.get("schema_id") != HOST_TILING_SCHEMA_ID:
            raise TaskContractError(f"unsupported host tiling schema {document.get('schema_id')!r}")
        try:
            version = int(document.get("schema_version", 0))
        except (TypeError, ValueError) as exc:
            # ``int(None)`` is a ``TypeError`` and a non-numeric string a ``ValueError``; both would
            # otherwise escape the typed contract boundary and kill the search.
            raise TaskContractError(
                f"host tiling entry {entry.path!r} has a non-integer schema_version {document.get('schema_version')!r}"
            ) from exc
        if version != HOST_TILING_SCHEMA_VERSION:
            raise TaskContractError("unsupported host tiling schema_version")
        writes = document.get("writes")
        if not isinstance(writes, list) or not writes:
            raise TaskContractError("host tiling plan needs at least one write")
        workspaces = {t.name: t for t in contract.instance.tensors if t.space == "workspace"}
        parsed: list[HostTilingWrite] = []
        for index, item in enumerate(writes):
            if not isinstance(item, Mapping) or set(item) != HOST_TILING_WRITE_FIELDS:
                raise TaskContractError(f"host tiling write[{index}] must have exactly {sorted(HOST_TILING_WRITE_FIELDS)}")
            if item.get("kind") != "linear":
                raise TaskContractError(f"host tiling write[{index}] kind must be 'linear'")
            name = str(item["workspace"])
            tensor = workspaces.get(name)
            if tensor is None:
                raise TaskContractError(f"host tiling write[{index}] targets unknown workspace {name!r}")
            if str(item["dtype"]) != tensor.dtype:
                raise TaskContractError(
                    f"host tiling write[{index}] dtype {item['dtype']!r} does not match workspace {name!r} dtype {tensor.dtype!r}"
                )
            parsed.append(
                HostTilingWrite(
                    workspace=name,
                    kind="linear",
                    start=_integer(item["start"], contract.instance, f"host tiling write[{index}].start"),
                    step=_integer(item["step"], contract.instance, f"host tiling write[{index}].step"),
                    count=_integer(item["count"], contract.instance, f"host tiling write[{index}].count"),
                    stride=_integer(item["stride"], contract.instance, f"host tiling write[{index}].stride"),
                    offset=_integer(item["offset"], contract.instance, f"host tiling write[{index}].offset"),
                    dtype=tensor.dtype,
                )
            )
        return cls(writes=tuple(parsed), source_sha256=entry.sha256, path=entry.path)

    def document(self) -> dict[str, Any]:
        """@brief host plan 的规范文档（进入执行证据）。"""
        return {
            "schema_id": HOST_TILING_SCHEMA_ID,
            "schema_version": HOST_TILING_SCHEMA_VERSION,
            "writes": [write.document() for write in self.writes],
        }

    def execute(self, argset: "V2ArgSet") -> dict[str, Any]:
        """@brief 在 host 上执行计划，把字节写进已分配的 workspace。

        @param argset v2 参数组。
        @return 实际 host 执行记录（字节数、哈希）；计时边界由 lifecycle 的 inside_timing 决定。
        """
        started = time.perf_counter()
        digest = hashlib.sha256(json.dumps([w.document() for w in self.writes], sort_keys=True).encode()).hexdigest()
        total = 0
        for write in self.writes:
            data = argset.workspace(write.workspace)
            nbytes = get_dtype(write.dtype).nbytes
            problems = host_write_span(write, data.nbytes)
            if problems:
                # The same helper the CPU preflight uses, so an out-of-bounds plan is refused
                # identically before compilation and at execution.
                raise GeometryError("; ".join(problems))
            span = write.offset + (write.count - 1) * write.stride + 1 if write.count > 0 else 0
            for i in range(write.count):
                value = write.start + i * write.step
                raw = _pack_scalar(value, write.dtype)
                at = (write.offset + i * write.stride) * nbytes
                data.mem[data.guard + at : data.guard + at + nbytes] = raw
                total += nbytes
        return {
            "path": self.path,
            "source_sha256": self.source_sha256,
            "plan_sha256": digest,
            "writes": total,
        }


def _pack_scalar(value: int, dtype: str) -> bytes:
    d = get_dtype(dtype)
    if d.name == "fp16":
        return struct.pack("<e", float(value))
    if d.name == "bf16":
        return torch.tensor([float(value)], dtype=torch.bfloat16).view(torch.int16).numpy().tobytes()
    if d.name == "fp32":
        return struct.pack("<f", float(value))
    fmt = {"i32": "<i", "u32": "<I", "i8": "<b", "u8": "<B"}.get(d.name)
    if fmt is None:
        raise GeometryError(f"host tiling cannot write dtype {dtype!r}")
    return struct.pack(fmt, int(value))


@dataclass(frozen=True)
class V2Case:
    """@brief v2 一个 case 的输入初值/参考终值/误差尺度。"""

    case: str
    seed: int
    initial: Mapping[str, bytes]
    rounded: Mapping[str, torch.Tensor]
    ref: Mapping[str, torch.Tensor]
    scale: Mapping[str, torch.Tensor]
    #: L1/L2 的实际输入身份（case + seed + 实际 ``cu_seqlens`` 字节 + 输入张量身份 + L2 的 resolved
    #: 形状），失败也能重放。固定 shape 实例保持空串：它们的输入身份仍由既有实例指纹与证据字段表达。
    input_digest: str = ""
    #: L2 本次调用的 runtime resolved scalar（``SfHostResolved`` / 证据用）。固定/L1 为空字典。
    resolved: Mapping[str, int] = field(default_factory=dict)
    #: L2 本次调用的 resolved 形状文档（固定/L1 为 ``None``）。
    resolved_document: Mapping[str, Any] | None = None
    #: L2 本次调用的 ``ResolvedShape``（logical/padding 区域从它派生；固定/L1 为 ``None``）。
    resolved_shape: Any = None


def prepare_v2_case(contract: TaskContract, case: str, seed: int) -> V2Case:
    """@brief 由 v2 reference 生成全部 reference inputs 与终值。

    ``read_write`` tensors are reference inputs *and* results; they are in ``initial`` and
    ``ref``.  There is no use of the legacy ``Contract.inputs`` list, which cannot represent
    read_write.
    """
    instance = contract.instance
    logical = contract.reference.generate(instance, case, int(seed))
    shapes = {t.name: t.shape for t in contract.instance.tensors}
    dtypes = {t.name: t.dtype for t in contract.instance.tensors}
    rounded: dict[str, torch.Tensor] = {}
    initial: dict[str, bytes] = {}
    for name, value in logical.items():
        if name not in shapes:
            continue
        tensor = round_to(value.reshape(tuple(int(v) for v in shapes[name])), dtypes[name])
        rounded[name] = tensor
        initial[name] = to_raw(tensor, dtypes[name])
    ref = contract.reference.execute(instance, rounded)
    resolved_shape = resolved_shape_for_case(instance, case)
    if resolved_shape is not None and resolved_shape.kind != "l2":
        # L1's resolved shape is its declared total/count, already in ``symbols``/``SfHostShapes``.
        # Only a capacity tier has a runtime resolved shape that is not a static symbol.
        resolved_shape = None
    if resolved_shape is not None:
        # The logical/padding decomposition is defined once, here, and every consumer (the scale, the
        # input poison check, the comparator) derives its region from it.  A tensor axis that cannot be
        # proven resolved-or-invariant is refused now instead of silently comparing the whole capacity
        # buffer (``tcenv.opspec.model.resolve_axis_kind`` is the same rule the document validator uses).
        require_capacity_axes(instance)
    region: dict[str, tuple[slice, ...]] = {}
    if resolved_shape is not None:
        for name in (*instance.reference_inputs, *instance.reference_outputs):
            region[name] = logical_region(instance, resolved_shape, name)
    scale: dict[str, torch.Tensor] = {}
    if contract.reference.error_scale_kind not in ("none", ""):
        # For L2 the hook sees the logical region only: the unused capacity rows hold the declared
        # poison, and a scale computed over them would be inflated by bytes the honest candidate never
        # wrote.  That inflation is exactly how "output all right, final_state all wrong" used to pass.
        scale = contract.reference.error_scale(
            instance, rounded, ref, logical=region or None
        )
    else:
        for name, value in ref.items():
            # L2: the default RMS scale is taken over the *logical* region of the result, never over the
            # capacity buffer (which contains the poison padding of a partial batch).
            logical = region.get(name)
            finite_source = value if logical is None else value[logical]
            finite = torch.where(torch.isfinite(finite_source), finite_source, torch.zeros_like(finite_source)) if finite_source.is_floating_point() else finite_source
            scale[name] = torch.sqrt(torch.mean(finite.to(torch.float64) ** 2))
    if not region:
        # Fixed/L1: the scale has always been defined on the declared tensors.  The broadcast check is
        # still applied so a task whose scale cannot describe its result can never fall through to a
        # runtime exception inside ``scaled_error`` (which would be misreported as infrastructure).
        for name, value in scale.items():
            target = tuple(int(dim) for dim in instance.tensor(name).shape)
            if not broadcasts_scale(tuple(int(dim) for dim in value.shape), target):
                raise TaskContractError(
                    f"{contract.reference.path}: error scale for {name!r} has shape "
                    f"{tuple(int(dim) for dim in value.shape)}, which does not broadcast to the declared "
                    f"result shape {target}"
                )
    if resolved_shape is not None:
        # L2: the generator filled the capacity buffers, so the *unused* input region must already be the
        # declared poison and the cu_seqlens tail the declared sentinel.  This is the harness's own check
        # (`gen_inputs` cannot certify it), and it runs before any device byte is written.
        # Byte-level, on the storage bytes that reach the device (``initial``), not on the widened
        # float32 the oracle works with: the fp16 padding pattern has a different float32 encoding.
        payload = {name: value for name, value in initial.items() if name != "cu_seqlens"}
        problems = padding_byte_problems(
            instance,
            resolved_shape,
            payload,
            {name: tuple(int(v) for v in instance.tensor(name).shape) for name in payload},
            {name: get_dtype(instance.tensor(name).dtype).nbytes for name in payload},
            where=f"{instance.op_name}.{instance.size}/{case}",
        )
        if problems:
            raise VarlenContractError("; ".join(problems))
    return V2Case(
        case=case,
        seed=int(seed),
        initial=initial,
        rounded=rounded,
        ref=ref,
        scale=scale,
        input_digest=(
            evaluation_input_digest(instance, case, int(seed), rounded) if is_varlen(instance) else ""
        ),
        resolved={} if resolved_shape is None else resolved_shape.resolved_scalars(),
        resolved_document=None if resolved_shape is None else resolved_shape.document(),
        resolved_shape=resolved_shape,
    )


class V2ArgSet:
    """@brief 由 ``TensorMemoryContract`` 分配的真实绑定集合。

    Workspace allocation is the union over the geometry's *possible* stages (declared stages,
    active or not), not every workspace tensor the ABI declares.  A candidate that never binds a
    scratch tensor must not pay for it: ``varlen_kda.log_decay_tab`` is about 384 MiB at
    T=8192/H=96/D=128, so allocating it for a naive recurrence would be pure waste.  The union is
    computed once here, before any dispatch, because an active set may only change between cases
    (or, in Dyn-1, between launches) and a workspace allocation inside the timed window is
    forbidden.  A workspace no stage can ever bind is not allocated, and its correctness fill is
    skipped: nothing can read it.
    """

    def __init__(self, ctx: Any, contract: TensorMemoryContract, geometry: ResolvedGeometry, *, tight: bool = False) -> None:
        self.ctx = ctx
        self.contract = contract
        self.geometry = geometry
        self.buffers: dict[str, GuardedBuffer] = {}
        self.workspace_bufs: dict[str, GuardedBuffer] = {}
        self.possible_workspaces: tuple[WorkspaceUse, ...] = possible_workspace_uses(geometry)
        possible = {use.name for use in self.possible_workspaces}
        for tensor in contract.tensors:
            nbytes = tensor_span_bytes(tuple(int(v) for v in tensor.shape), tuple(int(v) for v in tensor.strides), get_dtype(tensor.dtype).nbytes)
            pattern = POISON_PATTERN if tensor.access == "read" else GUARD_PATTERN
            target = self.workspace_bufs if tensor.space == "workspace" else self.buffers
            if tensor.space == "workspace" and tensor.name not in possible:
                continue
            target[tensor.name] = GuardedBuffer(ctx, nbytes, guard=GUARD_BYTES, tight=tight, pattern=pattern)
            if tensor.space == "workspace":
                # Keep the ABI span available for the workspace-size cross-check/evidence.
                target[tensor.name].abi_nbytes = nbytes
        self.workspace_zero = {use.name: bool(use.zero) for use in self.possible_workspaces}
        #: 本 case 的 runtime resolved scalar（L2；固定/L1 为空字典）。Host Program 与证据都读这里，
        #: 保证“候选看到的 resolved 形状”和“报告里记录的”是同一份。
        self.resolved_scalars: dict[str, int] = {}
        #: 本 case 的 resolved 形状文档（L2；固定/L1 为 ``None``）。进 HostRunEvidence 与报告。
        self.resolved_shape_document: dict[str, Any] | None = None
        #: 本 case 的实际输入身份（L1/L2；固定 shape 为空串）。
        self.evaluation_input_digest: str = ""
        self._initial: dict[str, bytes] = {}
        self._read_snapshot: dict[str, bytes] = {}
        self.workspace_bytes: dict[str, int] = {name: gb.nbytes for name, gb in self.workspace_bufs.items()}

    def workspace(self, name: str) -> GuardedBuffer:
        if name not in self.workspace_bufs:
            raise GeometryError(f"workspace {name!r} is not declared by the ABI")
        return self.workspace_bufs[name]

    def load_initial(self, initial: Mapping[str, bytes]) -> None:
        """@brief 写入全部 reference inputs（含 read_write）并留只读快照。"""
        self._initial = dict(initial)
        for name, raw in initial.items():
            if name in self.buffers:
                self.buffers[name].write(raw)
            else:
                raise GeometryError(f"reference input {name!r} is not a global ABI tensor")
        self._read_snapshot = {
            name: self.buffers[name].read()
            for name, tensor in ((t.name, t) for t in self.contract.tensors)
            if tensor.access == "read" and tensor.space != "workspace" and name in self.buffers
        }

    def _restore(self, name: str) -> None:
        if name not in self._initial:
            raise GeometryError(f"cannot restore {name!r}: no reference initial value was supplied")
        self.buffers[name].write(self._initial[name])

    def apply_fill_plan(self, plan: Any, *, reset_guards: bool = True) -> None:
        """@brief 执行 A2 ``LaunchFillPlan``（正确性逐 trial / benchmark）。

        @param plan A2 fill plan。
        @param reset_guards 是否先重置 guard；benchmark 连续 reps 必须为 ``False``，否则
            第二/后续 launch 的越界痕迹会被擦除。
        """
        if reset_guards:
            self.reset_guards_all()
        for fill in plan.tensors:
            if fill.name in self.workspace_bufs:
                gb = self.workspace_bufs[fill.name]
                if fill.action == ZERO:
                    gb.mem[gb.guard : gb.guard + gb.nbytes] = bytes(gb.nbytes)
                elif fill.action == "poison":
                    gb.poison()
                elif fill.action == RESTORE_INITIAL:
                    raise GeometryError("workspace cannot be restored from a reference input")
                continue
            if fill.reason.startswith("workspace"):
                # A workspace tensor that no possible stage binds was deliberately not
                # allocated; it has no reader, so its fill action has nothing to act on.
                continue
            gb = self.buffers[fill.name]
            if fill.action == RESTORE_INITIAL:
                self._restore(fill.name)
            elif fill.action == ZERO:
                gb.mem[gb.guard : gb.guard + gb.nbytes] = bytes(gb.nbytes)
            elif fill.action == "poison":
                gb.poison()
            elif fill.action == "untouched":
                pass
            else:
                raise GeometryError(f"unknown fill action {fill.action!r} for {fill.name!r}")

    def load_capacity_padding(self, instance: Any, resolved: Any) -> None:
        """@brief 把声明的 padding poison 写进 L2 输出的**未用**区域（每个 trial 重建）。

        A pure ``write`` output is pre-filled with the generic ``0xFF`` poison pattern, which is not the
        capacity tier's declared padding byte.  Without this step the post-run padding check would
        compare against a value the run never had, so the harness writes the declared byte into exactly
        the region it will later verify — and a candidate that writes anything there (including zeros)
        is detected.  Read-only inputs and ``read_write`` initial values keep the generator's own
        padding, which `prepare_v2_case` already validated.

        Correctness/benchmark *setup* only: it rewrites the whole output buffer, so it must never be
        called inside a timed sample (it is not).

        @param instance 语义实例。
        @param resolved 本 case 的 ``ResolvedShape``。
        """
        if not getattr(instance, "is_capacity", False) or resolved is None:
            return
        expected = int(instance.capacity.padding_poison_byte)
        for tensor in self.contract.tensors:
            if tensor.space == "workspace" or tensor.access != "write":
                continue
            gb = self.buffers.get(tensor.name)
            if gb is None:  # pragma: no cover - every declared tensor is allocated
                continue
            shape = tuple(int(v) for v in tensor.shape)
            terms = declared_dim_terms(instance, tensor.name)
            _, slabs = padding_slabs(instance, resolved, shape, terms)
            if not slabs:
                continue
            import numpy as np

            # ``bytearray`` keeps the numpy view writable (``bytes`` would be read-only).
            buffer = bytearray(gb.mem[gb.guard : gb.guard + gb.nbytes])
            array = np.frombuffer(buffer, dtype=np.uint8).reshape(shape + (get_dtype(tensor.dtype).nbytes,))
            for slab in slabs:
                array[tuple(slab) + (slice(0, int(array.shape[-1])),)] = expected
            gb.mem[gb.guard : gb.guard + gb.nbytes] = bytes(buffer)

    def reset_guards_all(self) -> None:
        for gb in (*self.buffers.values(), *self.workspace_bufs.values()):
            gb.reset_guards()

    def stage_args(self, binding: StageBinding) -> list[tuple[int, Any]]:
        """@brief 按已校验的 per-stage binding 生成真实 Metal 参数槽。"""
        from .runtime import BufferArg, ScalarArg

        out: list[tuple[int, Any]] = []
        for argument in binding.arguments:
            if argument.role == "tensor":
                gb = self.buffers.get(argument.name) or self.workspace_bufs.get(argument.name)
                if gb is None:
                    raise GeometryError(f"stage binding refers to unallocated tensor {argument.name!r}")
                out.append((argument.slot, BufferArg(gb.buf, gb.guard)))
            elif argument.role == "derived_scalar":
                out.append((argument.slot, ScalarArg(_derived_bytes(argument))))
            else:
                raise GeometryError(f"unsupported stage argument role {argument.role!r}")
        return out

    def workspace_fills_declared_zero(self, *, exclude: Sequence[str] = ()) -> list[tuple]:
        """@brief 计时窗内只清候选声明 ``zero`` 的 workspace。

        @param exclude 由 Host Program 自己写的 workspace：它们的内存初始化归 host，device 不再
            fill。host 写与 device_zero 同块在 envelope 阶段已经拒绝，这里只是把两条路径分开。
        """
        skip = set(exclude)
        return [
            (gb.buf, gb.guard, gb.nbytes)
            for name, gb in self.workspace_bufs.items()
            if self.workspace_zero.get(name, False) and name not in skip
        ]

    def read_results(self) -> dict[str, bytes]:
        return {
            t.name: self.buffers[t.name].read()
            for t in self.contract.tensors
            if t.space != "workspace" and t.access in ("write", "read_write")
        }

    def safety_violations(self) -> list[str]:
        out: list[str] = []
        for name, gb in (*self.buffers.items(), *self.workspace_bufs.items()):
            out.extend(f"buffer `{name}`: {v}" for v in gb.guard_violations())
        for name, raw in self._read_snapshot.items():
            if self.buffers[name].read() != raw:
                out.append(f"input `{name}` was modified by the kernel")
        return out

    def guard_violations(self) -> list[str]:
        """@brief benchmark reps 后的真实 guard 检查（不做输出终值差分）。"""
        out: list[str] = []
        for name, gb in (*self.buffers.items(), *self.workspace_bufs.items()):
            out.extend(f"buffer `{name}`: {v}" for v in gb.guard_violations())
        return out

    def result_tensors(self) -> tuple[Any, ...]:
        return tuple(
            t for t in self.contract.tensors
            if t.space != "workspace" and t.access in ("write", "read_write")
        )

    def stage_workspace_names(self, stage: ResolvedStage) -> tuple[str, ...]:
        return tuple(use.name for use in stage.workspace)

    def actual_buffers(self) -> dict[str, dict[str, int]]:
        return {
            "workspace": {name: gb.nbytes for name, gb in self.workspace_bufs.items()},
            "global": {name: gb.nbytes for name, gb in self.buffers.items()},
        }


def _derived_bytes(argument: StageArgument) -> bytes:
    mapping = {"int": "<i", "uint": "<I", "long": "<q", "ulong": "<Q"}
    fmt = mapping.get(argument.dtype)
    if fmt is None:
        raise GeometryError(f"derived scalar {argument.name!r} has unsupported dtype {argument.dtype!r}")
    return struct.pack(fmt, int(argument.value))


def _msl_scalar_type(code: int | None) -> str | None:
    """@brief 把 Metal reflection dtype code 转成 MSL 标量名（A4 edge conversion）。"""
    import Metal

    mapping = {
        Metal.MTLDataTypeHalf: "half",
        Metal.MTLDataTypeFloat: "float",
        Metal.MTLDataTypeBFloat: "bfloat",
        Metal.MTLDataTypeInt: "int",
        Metal.MTLDataTypeUInt: "uint",
        Metal.MTLDataTypeLong: "long",
        Metal.MTLDataTypeULong: "ulong",
        Metal.MTLDataTypeShort: "short",
        Metal.MTLDataTypeUShort: "ushort",
        Metal.MTLDataTypeChar: "char",
        Metal.MTLDataTypeUChar: "uchar",
    }
    return mapping.get(int(code)) if code is not None else None


def reflection_for_kernel(kernel: Any) -> list[ReflectedArgument]:
    """@brief 把 runtime 的 ``CompiledKernel.bindings`` 转成 A2 反射记录。"""
    out: list[ReflectedArgument] = []
    for binding in kernel.bindings:
        out.append(
            ReflectedArgument(
                index=int(binding.index),
                name=str(binding.name),
                kind=str(binding.kind),
                access=str(binding.access),
                scalar_type=_msl_scalar_type(binding.dtype_code) if binding.kind == "buffer" else None,
                elem_size=None if binding.elem_size is None else int(binding.elem_size),
                used=bool(getattr(binding, "used", True)),
            )
        )
    return out


@dataclass
class DispatchRecord:
    """@brief runtime 实际执行的一次 stage dispatch。"""

    stage: int
    kernel: str
    grid: tuple[int, int, int]
    threads: tuple[int, int, int]
    threadgroup_bytes: int
    workspace: tuple[str, ...]
    provider: str
    sequence: int

    def document(self) -> dict[str, Any]:
        return {
            "stage": int(self.stage),
            "kernel": self.kernel,
            "grid": list(self.grid),
            "threads": list(self.threads),
            "threadgroup_bytes": int(self.threadgroup_bytes),
            "workspace": list(self.workspace),
            "provider": self.provider,
            "sequence": int(self.sequence),
        }


@dataclass
class V2Benchmark:
    """@brief 一次 v2 计时结果：候选、登记基线、重试噪声与真实 dispatch。

    候选与基线来自同一次 ``tcenv.bench`` 交错计时（同一个 ``interleaved`` 调用、同一个批量下限、
    同一套重试与选择规则），所以候选中位数是在与 v1 相同的测量边界下得到的。
    """

    stats: Any
    baselines: dict[str, Any]
    rotation_sets: int
    dispatch: list[DispatchRecord]
    host: dict[str, Any] | None
    guard_violations: list[str]
    noise_cv: float
    attempts: int
    #: L2 的 ordered continuous-batching mixture（每批次一条，固定/L1 为空）。每条带 case、
    #: resolved 形状与本次调用的 evaluation input digest；它不是单形状测量，报告不得把它当成
    #: 与单形状基线同尺度的比值。
    batches: list[dict[str, Any]] = field(default_factory=list)

    @property
    def measurement_scope(self) -> str:
        """@brief 本次计时的边界：单形状还是声明的连续批次 mixture。

        The distinction is not cosmetic: a mixture median cannot be divided by a single-shape baseline
        median and presented as a like-for-like speedup, so the report has to know which one it holds.

        @return ``"continuous_batching_mixture"`` 或 ``"single_shape"``。
        """
        totals = set()
        counts = set()
        for batch in self.batches:
            document = batch.get("resolved_shape")
            if not isinstance(document, Mapping):
                continue
            totals.add(int(document.get("total_tokens", -1)))
            counts.add(int(document.get("num_seqs", -1)))
        return "continuous_batching_mixture" if len(totals) > 1 or len(counts) > 1 else "single_shape"

    def document(self) -> dict[str, Any]:
        return {
            "candidate": self.stats.to_json(),
            "baselines": {name: value.to_json() for name, value in self.baselines.items()},
            "rotation_sets": int(self.rotation_sets),
            "noise_cv": float(self.noise_cv),
            "attempts": int(self.attempts),
            "guard_violations": list(self.guard_violations),
        }


def benchmark_datasets(
    contract: TaskContract,
    cases: Sequence[str],
    rotation: int,
    seed: int,
) -> tuple[tuple[str, ...], dict[str, "V2Case"]]:
    """@brief 计时 mixture 的 case 顺序与每 set 使用的 ``V2Case``。

    The order is the declared continuous-batching plan (:func:`benchmark_mixture`); each distinct case is
    prepared once, so a rotation set carries its own resolved shape, input bytes, declared padding and
    evaluation-input digest.  Preparing a dataset is *not* part of the timed window: the caller builds all
    rotation sets (inputs, padding, Host Program storage) before the interleaved loop.

    @param contract 任务合同。
    @param cases 声明的 case 名（声明顺序）。
    @param rotation 记忆足迹给出的轮转组数（下限 1）。
    @param seed 输入种子。
    @return ``(mixture_cases, case -> V2Case)``。
    """
    plan = benchmark_mixture(contract.instance, cases, rotation)
    datasets: dict[str, V2Case] = {}
    for case in plan:
        if case not in datasets:
            datasets[case] = prepare_v2_case(contract, case, int(seed) + 1)
    return plan, datasets


def benchmark_mixture(instance: Any, cases: Sequence[str], rotation: int) -> tuple[str, ...]:
    """@brief 计时轮转集要跑的 case 序列（L2 为声明的 continuous-batching 顺序）。

    For an L2 capacity instance the rotation mixture must cover the declared cases in declared order:
    set ``i`` runs case ``i % len(cases)``, and there are at least ``len(cases)`` sets so every declared
    workload profile is timed.  The count is capped at the backend's ring maximum (a memory bound, not
    a measurement choice); the caller records the dropped cases in ``benchmark_mixture_notes``, and a cap
    that drops the full-capacity batch is a loud ``benchmark_mixture_problems`` failure rather than a
    silent "capacity measured" claim.  Fixed-shape and L1 instances keep the historical behaviour
    (every set is the first case) so their timing boundary and evidence do not change.

    @param instance 语义实例。
    @param cases 声明的 case 名（声明顺序）。
    @param rotation 记忆足迹给出的轮转组数（下限 1）。
    @return 每个 rotation set 的 case 名元组。
    @raises TaskContractError 没有声明 case。
    """
    declared = tuple(str(case) for case in cases)
    if not declared:
        raise TaskContractError("a benchmark needs at least one declared case")
    count = max(1, int(rotation))
    if not bool(getattr(instance, "is_capacity", False)):
        return tuple(declared[0] for _ in range(count))
    from .baselines import RING_MAX

    count = min(max(count, len(declared)), int(RING_MAX))
    return tuple(declared[index % len(declared)] for index in range(count))


def benchmark_mixture_problems(
    instance: Any,
    batches: Sequence[Mapping[str, Any]],
    *,
    cases: Sequence[str] | None = None,
) -> tuple[str, ...]:
    """@brief 校验一份 benchmark mixture 证据确实是声明的连续批次，而不是一个满 capacity 点。

    The evidence is only worth recording if it can be checked.  A valid L2 mixture:

    * is not empty and follows the **declared case order** cyclically from the first declared case;
    * carries, per batch, a resolved shape equal to the declared workload profile of that case;
    * covers every declared workload profile, so the timed totals/counts are exactly the declared ones
      (a task that declares several profiles cannot time only the capacity point — that is an L1 tier
      under an L2 name — while a task that declares a single profile is not failed for the
      declaration it wrote);
    * has a full-capacity batch whenever the declaration contains a full-capacity profile;
    * carries an evaluation-input digest per batch;
    * attributes every batch to the candidate provider, never to a fallback (fallback time is not the
      variant's, and a mixture that mixes providers would publish it as one).

    @param instance 语义实例。
    @param batches ``benchmark_batches`` 证据。
    @param cases 声明的 case 名（缺省取实例的）。
    @return 问题描述元组；为空表示合法。
    """
    if not bool(getattr(instance, "is_capacity", False)):
        return () if not batches else ("a non-capacity instance must not publish a continuous-batching mixture",)
    from ...taskcontract.runtime_shape import resolved_shape_for_case

    declared = tuple(str(case) for case in (cases if cases is not None else instance.case_domain.cases))
    problems: list[str] = []
    if not batches:
        return ("an L2 benchmark published no continuous-batching mixture",)
    totals: set[int] = set()
    counts: set[int] = set()
    profiles: set[str] = set()
    full = False
    for ordinal, batch in enumerate(batches):
        expected_case = declared[ordinal % len(declared)]
        where = f"batch[{ordinal}]"
        case = str(batch.get("case"))
        if case != expected_case:
            problems.append(
                f"{where} ran case {case!r}, but the declared continuous-batching order puts {expected_case!r} there"
            )
            continue
        if str(batch.get("provider")) != "candidate":
            problems.append(
                f"{where} was attributed to provider {batch.get('provider')!r}; a fallback's time is not the "
                "generated variant's and cannot be mixed into its mixture"
            )
        digest = batch.get("evaluation_input_digest")
        if not isinstance(digest, str) or len(digest) != 64:
            problems.append(f"{where} carries no evaluation_input_digest")
        declared_shape = resolved_shape_for_case(instance, case)
        document = batch.get("resolved_shape")
        if not isinstance(document, Mapping) or declared_shape is None:
            problems.append(f"{where} carries no resolved_shape")
            continue
        if (int(document.get("total_tokens", -1)), int(document.get("num_seqs", -1))) != (
            declared_shape.total_tokens,
            declared_shape.num_seqs,
        ):
            problems.append(
                f"{where} ran resolved [{document.get('total_tokens')} tokens / {document.get('num_seqs')} "
                f"sequences], but case {case!r} declares [{declared_shape.total_tokens} / {declared_shape.num_seqs}]"
            )
        if str(document.get("workload_profile_id")) != str(declared_shape.workload_profile_id):
            problems.append(f"{where} names workload profile {document.get('workload_profile_id')!r}, not the declared one")
        totals.add(int(document.get("total_tokens", -1)))
        counts.add(int(document.get("num_seqs", -1)))
        profiles.add(str(document.get("workload_profile_id")))
        if bool(document.get("full_capacity")):
            full = True
    declared_profiles = {}
    for case in declared:
        profile = resolved_shape_for_case(instance, case)
        if profile is not None:
            declared_profiles[str(profile.workload_profile_id)] = profile
    expected_totals = {profile.total_tokens for profile in declared_profiles.values()}
    expected_counts = {profile.num_seqs for profile in declared_profiles.values()}
    missing = sorted(set(declared_profiles) - profiles)
    if missing:
        problems.append(f"the mixture never timed declared workload profile(s) {missing}")
    if totals != expected_totals:
        problems.append(
            f"the mixture timed resolved totals {sorted(totals)}; the declaration covers "
            f"{sorted(expected_totals)}"
        )
    if counts != expected_counts:
        problems.append(
            f"the mixture timed resolved counts {sorted(counts)}; the declaration covers "
            f"{sorted(expected_counts)}"
        )
    if any(profile.is_full_capacity for profile in declared_profiles.values()) and not full:
        problems.append(
            "the mixture never timed the declared full-capacity profile, so the capacity point itself is unmeasured"
        )
    return tuple(problems)


def possible_workspace_uses(geometry: ResolvedGeometry) -> tuple[WorkspaceUse, ...]:
    """@brief 整份几何中所有 *可能* stage（含 ``when`` 为假者）声明的 workspace 使用并集。

    Dyn-0 的 stage 全部静态激活，所以它的并集与 ``geometry.workspace_uses`` 逐项相同；保留
    “possible” 这个名字是因为分配点必须在任何 dispatch 之前：Dyn-1 的某个 case 可能激活一个
    本 case 未激活的 stage，届时在计时圈里补分配已经太晚。并集同时按名去重并校验同一 workspace
    的 zero 策略唯一，否则“什么时候清零”会取决于 dispatch 顺序。

    @param geometry 已求值几何。
    @return workspace 使用元组（按首次出现顺序去重）。
    @raises GeometryError 同一 workspace 被不同 stage 声明成 zero 与非 zero。
    """
    seen: dict[str, bool] = {}
    for stage in geometry.stages:
        for use in stage.workspace:
            if seen.get(use.name, use.zero) != use.zero:
                raise GeometryError(
                    f"workspace `{use.name}` is declared both zero and not-zero by the geometry's stages; the "
                    "clearing policy of one workspace must be one value"
                )
            seen.setdefault(use.name, use.zero)
    return tuple(WorkspaceUse(name=name, zero=zero) for name, zero in seen.items())


def workspace_size_problems(abi: TensorMemoryContract, geometry: ResolvedGeometry, spec: LaunchSpec | None) -> list[str]:
    """@brief Compare launch workspace declarations with the ABI span (CPU-checkable).

    A launch expression such as ``M*2`` is a candidate guess; the ABI tensor's shape/strides
    are the authority.  A mismatch is refused before any buffer is allocated.
    """
    declared = {} if spec is None else {w.name: int(w.nbytes) for w in spec.workspaces}
    actual = {
        t.name: tensor_span_bytes(tuple(int(v) for v in t.shape), tuple(int(v) for v in t.strides), get_dtype(t.dtype).nbytes)
        for t in abi.tensors if t.space == "workspace"
    }
    problems: list[str] = []
    for name, nbytes in declared.items():
        if name not in actual:
            problems.append(f"launch declares workspace {name!r} that the ABI does not contain")
        elif int(nbytes) != int(actual[name]):
            problems.append(
                f"workspace {name!r}: launch declares {nbytes} bytes, ABI TensorMemoryContract spans {actual[name]} bytes"
            )
    for use in geometry.workspace_uses:
        if use.name not in declared:
            problems.append(f"resolved geometry uses workspace {use.name!r}, but the launch does not declare it")
        if use.name not in actual:
            problems.append(f"resolved geometry uses workspace {use.name!r}, but the ABI does not contain it")
    return problems

class MetalV2Runtime:
    """@brief 真实 Metal v2 执行器：反射绑定 + 协议 + host tiling + dispatch 记录。"""

    def __init__(
        self,
        ctx: Any,
        *,
        contract: TaskContract,
        abi: TensorMemoryContract,
        geometry: ResolvedGeometry,
        kernels: Sequence[Any],
        launch_spec: LaunchSpec | None,
        host_plan: HostTilingPlan | None,
        host_inside_timing: bool,
        limits: Any,
        selection: Any,
        host_program: MetalHostProgram | None = None,
    ) -> None:
        self.ctx = ctx
        self.contract = contract
        self.abi = abi
        self.geometry = geometry
        self.selection = selection
        # The provider is the geometry the runtime actually executes, not a second copy of
        # the evidence's label.  A worker that swaps in the fallback changes this value.
        self.provider = str(geometry.source)
        self.host_plan = host_plan
        self.host_program = host_program
        self.host_inside_timing = bool(host_inside_timing)
        self.limits = limits
        by_name = {str(k.name): k for k in kernels}
        self.kernels = by_name
        self._check_workspace_sizes(launch_spec)
        self.bindings = self._bind(by_name)
        self.argset = V2ArgSet(ctx, abi, geometry)
        self.last_host_run: dict[str, Any] | None = None
        self.last_host_timing: HostProgramTiming | None = None
        self.last_gpu_ns: int = 0
        self.stage_args_builds: int = 0
        self.read_only_restores: int = 0
        self.candidate_command_buffers: int = 0
        self.benchmark_diagnostics: dict[str, int] = {}
        self.workspace_probe: dict[str, str] = {}
        #: L2 的 ordered continuous-batching mixture（每个 rotation set 一条）；固定/L1 为空。
        self.benchmark_batches: list[dict[str, Any]] = []
        #: 计时 mixture 的声明/实际/被上限丢弃的 case（不让“只跑了 cases[0]”静默通过）。
        self.benchmark_mixture_notes: dict[str, Any] = {}
        #: 每一次真实调用（正确性按 ordered case、benchmark 按 mixture 批次）的"启动记录"。
        #: 固定/L1 实例不增长（旧证据形状不变）；L2 的连续调用每一批各一条，不按静态 dispatch
        #: 签名去重——Dyn-0 的 grid 是冻结的，去重会把“同一个 build 跑了多个批次”吞掉。
        self.launch_runs: list[dict[str, Any]] = []
        self._check_host_plan()
        self._check_host_program()

    def _note_launch(
        self,
        *,
        phase: str,
        ordinal: int,
        case: str,
        case_ordinal: int,
        data: V2Case,
        trial: str | None = None,
        dispatch: Sequence[DispatchRecord] = (),
        audited: bool = True,
    ) -> None:
        """@brief 记录一次真实调用（L2 连续批次的身份）；固定/L1 不记录。

        A Dyn-0 build uses one frozen grid for every batch, so the static dispatch signature is identical
        across calls and a signature-keyed dedup cannot tell two batches apart.  This record is per call:
        ordered phase, ordinal, case, provider, resolved shape and the evaluation-input digest the call
        actually ran.  It is launch *evidence*, not a dispatch-parity claim (the frozen-geometry parity
        check still runs on the dispatch records).

        @param phase ``correctness``/``benchmark``。
        @param ordinal 该 phase 内的**调用**序号（0 起，全局唯一；一次 case 的每个 correctness
            trial 都是独立一次调用）。
        @param case 调用使用的 case 名。
        @param case_ordinal 该 case 在声明顺序里的序号（同一 case 的多个 trial 相同）。
        @param data 该调用的 ``V2Case``。
        @param trial correctness trial 名（benchmark 为 ``None``）。
        @param dispatch 本次调用发出的 stage dispatch。
        @param audited 本次调用是否产出了审计文档（digest/reads）；计时样本不产出。
        """
        if not self.contract.instance.is_capacity:
            return
        self.launch_runs.append(
            {
                "phase": phase,
                "ordinal": int(ordinal),
                "case": case,
                "case_ordinal": int(case_ordinal),
                "trial": trial,
                "provider": self.provider,
                "resolved_shape": None if data.resolved_document is None else dict(data.resolved_document),
                "evaluation_input_digest": data.input_digest,
                "audited": bool(audited),
                "dispatch_stages": [int(record.stage) for record in dispatch],
            }
        )

    def _check_workspace_sizes(self, spec: LaunchSpec | None) -> None:
        problems = workspace_size_problems(self.abi, self.geometry, spec)
        if problems:
            raise GeometryError("; ".join(problems))

    def _bind(self, by_name: Mapping[str, Any]) -> tuple[StageBinding, ...]:
        reflection: dict[str, list[ReflectedArgument]] = {}
        for stage in self.geometry.active_stages:
            kernel = by_name.get(stage.kernel)
            if kernel is None:
                raise GeometryError(f"active stage {stage.index} kernel {stage.kernel!r} was not compiled")
            reflection[stage.kernel] = reflection_for_kernel(kernel)
        return bind_pipeline_arguments(
            self.geometry,
            contract=self.abi,
            reflection=reflection,
            limits=self.limits,
            where="metal v2",
        )

    def _check_host_plan(self) -> None:
        if self.host_plan is not None:
            abi_workspaces = {tensor.name for tensor in self.argset.contract.tensors if tensor.space == "workspace"}
            for write in self.host_plan.writes:
                if write.workspace not in self.argset.workspace_bufs:
                    if write.workspace in abi_workspaces:
                        raise GeometryError(
                            f"host tiling writes ABI workspace {write.workspace!r} that no possible stage binds; a "
                            "workspace must be declared by a stage so its fill policy and dispatch evidence are "
                            "authoritative (and so the allocation scope covers it)"
                        )
                    raise GeometryError(f"host tiling targets unknown ABI workspace {write.workspace!r}")
                if self.argset.workspace_zero.get(write.workspace, False) and not self.host_inside_timing:
                    raise GeometryError(
                        f"host tiling writes declared-zero workspace {write.workspace!r} outside timing; "
                        "the runtime would clear it before the device stage"
                    )

    def host_prepare(self) -> dict[str, Any] | None:
        if self.host_plan is None:
            return None
        return self.host_plan.execute(self.argset)

    def _check_host_program(self) -> None:
        """@brief envelope 阶段的 host/device 初始化冲突检查（V3）。

        一个 Host Program 的 workspace 由 host 自己初始化；同一块被 geometry 声明为 ``zero``
        （device_zero）就是两种互斥的所有权。这里在分配/计时之前拒绝，而不是运行时被覆盖后才
        发现。只有 ``.c`` Host Program 受这条规则约束：既有 ``.json`` linear 路径继续保持 V2
        的“fill 后 host 写”顺序与既有证据语义。

        @raises GeometryError host 写 device_zero 块、或与 linear plan 同时存在。
        """
        if self.host_program is None:
            return
        if self.host_plan is not None:
            raise GeometryError("a candidate cannot ship both host/tiling.json and host/tiling.c")
        if not self.host_inside_timing:
            raise GeometryError(
                "a C99 Host Program must run inside the timed window; its input-dependent work may not be moved "
                "outside measurement"
            )
        for name in sorted(self.host_program.written_workspaces):
            if self.argset.workspace_zero.get(name, False):
                raise GeometryError(
                    f"host program writes workspace {name!r}, but the geometry declares it device-zero; host-written "
                    "and device-zeroed initialization of one block are mutually exclusive"
                )
            if name not in self.argset.workspace_bufs:
                raise GeometryError(
                    f"host program writes workspace {name!r} that no possible stage binds; the possible-stage "
                    "union is the allocation scope"
                )

    def _stage_sets(self) -> tuple[list[Any], list[tuple[int, int, int]], list[tuple[int, int, int]], list[int]]:
        active = self.geometry.active_stages
        kernels = [self.kernels[stage.kernel] for stage in active]
        grids = [tuple(int(v) for v in stage.grid) for stage in active]
        threads = [tuple(int(v) for v in stage.threads) for stage in active]
        tg = [int(stage.threadgroup_bytes) for stage in active]
        return kernels, grids, threads, tg

    def run_once(
        self,
        *,
        benchmark: bool,
        host_outside_done: bool = False,
        record_dispatch: bool = True,
        stage_args: list[tuple[int, Any]] | None = None,
        fills: list[list[tuple]] | None = None,
        host_audit: bool | None = None,
        host_profile: bool = False,
    ) -> list[DispatchRecord]:
        """@brief 真实执行一次；可选记录 dispatch。

        ``record_dispatch=False`` is for a timed batch: the recorder is a Python callback invoked
        immediately before every ``dispatchThreadgroups`` call, so keeping it enabled inside the
        measured window charges the candidate for evidence bookkeeping that the v1 path does not
        pay.  The first warm-up launch records the same dispatch identity, outside the samples.

        ``stage_args``/``fills`` can be prebuilt by the caller (the timed Host Program path does
        that once per rotation set); rebuilding them inside the measured window is host work the
        candidate would pay for every iteration.  ``host_audit`` separates the audit run (hashes,
        evidence) from a timed run (copy + time only).
        """
        kernels, grids, threads, tg = self._stage_sets()
        if stage_args is None:
            # The timed Host Program path passes prebuilt stage args; a rebuild here is host work
            # inside the measured window and is counted so the benchmark evidence can refuse it.
            self.stage_args_builds += 1
            stage_args = [self.argset.stage_args(binding) for binding in self.bindings]
        host_written = self.host_program.written_workspaces if self.host_program is not None else frozenset()
        if fills is None:
            fills = (
                [self.argset.workspace_fills_declared_zero(exclude=host_written)] if benchmark else [[]]
            )
        if host_audit is None:
            host_audit = bool(record_dispatch)
        self.last_host_run = None
        self.last_host_timing = None
        if benchmark and self.host_plan is not None and not self.host_inside_timing and not host_outside_done:
            # Caller normally prepares before timing; this keeps a standalone invocation honest.
            self.last_host_run = self.host_prepare()
        if self.host_program is not None:
            # One native Host Program call, then one command buffer.  Host-written workspaces are excluded from
            # the device zero fills above, so no extra fill command buffer is needed and the host
            # bytes cannot be overwritten by a later GPU fill.
            outcome = self.host_program.execute(self.argset, audit=bool(host_audit), profile=bool(host_profile))
            if isinstance(outcome, HostProgramTiming):
                self.last_host_timing = outcome
            else:
                outcome["geometry_digest"] = self.selection.identity
                self.last_host_run = outcome
        # A zero-filled workspace is written by a GPU fill kernel.  For the legacy linear Host
        # Tiling that runs inside timing, encoding a CPU write into the same uncommitted command
        # buffer is too early: the GPU fill would execute later and overwrite it.  Complete the
        # fill in its own command buffer first, then write the workspace on the CPU, then submit
        # the device stages (the .c Host Program path above has no such conflict).
        if self.host_plan is not None and self.host_inside_timing:
            if benchmark and fills and fills[0]:
                self.ctx.fill_workspaces(fills[0])
                fills = [[]]
            self.last_host_run = self.host_prepare()
        records: list[DispatchRecord] = []
        sequence = 0

        def recorder(
            stage_index: int,
            stage: Any,
            grid: Any,
            threads_: Any,
            threadgroup_bytes: int,
            workspace: Any,
        ) -> None:
            nonlocal sequence
            records.append(
                DispatchRecord(
                    stage=int(self.geometry.active_stages[stage_index].index),
                    kernel=str(getattr(stage, "name", "")),
                    grid=tuple(int(v) for v in grid),
                    threads=tuple(int(v) for v in threads_),
                    threadgroup_bytes=int(threadgroup_bytes),
                    workspace=tuple(str(v) for v in workspace),
                    provider=self.provider,
                    sequence=sequence,
                )
            )
            sequence += 1

        timing = self.ctx.run_pipeline(
            kernels,
            grids,
            threads,
            tg,
            [stage_args],
            fills,
            iters=1,
            dispatch="serial",
            dispatch_recorder=recorder if record_dispatch else None,
            dispatch_workspace=[tuple(use.name for use in stage.workspace) for stage in self.geometry.active_stages],
        )
        self.last_gpu_ns = int(float(getattr(timing, "gpu_s", 0.0)) * 1_000_000_000)
        if not record_dispatch:
            return records
        from ...track import selection_parity

        problems = selection_parity(
            self.selection,
            provider=self.provider,
            observed=[
                (record.kernel, record.grid, record.threads, record.threadgroup_bytes, record.workspace)
                for record in records
            ],
            where="metal v2 runtime dispatch",
        )
        if problems:
            raise GeometryError("; ".join(problems))
        return records

    def correctness(
        self,
        *,
        cases: Sequence[str] | None,
        seed: int,
        determinism: bool = True,
    ) -> tuple[list[CaseResult], list[str], list[str], list[DispatchRecord], list[dict[str, Any]]]:
        """@brief 按 ordered case 真实执行并给出判决 + 每次调用的 HostRun 证据。

        The fifth value is a **list**: one HostRunEvidence per ordered call that produced one (L2) plus, on
        a candidate Host Program failure, that failure's document.  Returning only the last record would
        leave a continuous-batching sweep with evidence for one batch out of several, and a failed batch
        would lose the resolved shape/input digest it actually ran.

        @param cases 要跑的 case（缺省为声明的全部）；顺序就是声明的 continuous-batching 顺序。
        @param seed 输入种子。
        @param determinism 是否跑全部 correctness trial（validate 只跑第一个）。
        @return ``(results, safety, nondeterminism, dispatch, host_runs)``。
        """
        declared_cases = tuple(self.contract.instance.case_domain.cases)
        use_cases = tuple(cases or declared_cases)
        results: list[CaseResult] = []
        safety: list[str] = []
        nondet: list[str] = []
        dispatch: list[DispatchRecord] = []
        host_records: list[dict[str, Any]] = []
        call_ordinal = 0
        for index, case in enumerate(use_cases):
            data = prepare_v2_case(self.contract, case, seed + 7919 * (index + 1))
            # L2: this case's resolved shape travels with the argset, so the Host Program, the evidence
            # and the correctness comparison all read one value instead of three derivations.
            self.argset.resolved_scalars = dict(data.resolved)
            self.argset.resolved_shape_document = None if data.resolved_document is None else dict(data.resolved_document)
            self.argset.evaluation_input_digest = data.input_digest
            self.argset.load_initial(data.initial)
            outs: dict[str, dict[str, bytes]] = {}
            host_failure: dict[str, Any] | None = None
            for trial in CORRECTNESS_TRIALS:
                plan = correctness_launch_plan(self.abi, self.geometry, trial)
                self.argset.apply_fill_plan(plan)
                self.argset.load_capacity_padding(self.contract.instance, data.resolved_shape)
                host = self.host_prepare() if self.host_plan is not None and not self.host_inside_timing else None
                try:
                    records = self.run_once(benchmark=False, host_audit=True)
                except HostProgramFailed as exc:
                    # A non-zero return or native contract violation is a *candidate* failure: no
                    # span was committed and no device stage ran.  It is reported as incorrect,
                    # not as an infrastructure runtime_error.  The failure document keeps the same
                    # field set as a success, including this call's resolved shape and input digest.
                    host_failure = self.host_program.failure_document(exc.run, argset=self.argset)
                    host_failure["geometry_digest"] = self.selection.identity
                    host_failure["evaluation_case"] = case
                    host_failure["evaluation_case_ordinal"] = int(declared_cases.index(case))
                    host_failure["evaluation_phase"] = "correctness"
                    if host_run_identity(host_failure) not in {host_run_identity(item) for item in host_records}:
                        host_records.append(host_failure)
                    self._note_launch(
                        phase="correctness",
                        ordinal=call_ordinal,
                        case=case,
                        case_ordinal=int(declared_cases.index(case)),
                        data=data,
                        trial=trial,
                        dispatch=(),
                    )
                    call_ordinal += 1
                    break
                if host is None and self.last_host_run is not None:
                    host = self.last_host_run
                if host is not None:
                    host["evaluation_case"] = case
                    host["evaluation_case_ordinal"] = int(declared_cases.index(case))
                    host["evaluation_phase"] = "correctness"
                    if host_run_identity(host) not in {host_run_identity(item) for item in host_records}:
                        host_records.append(host)
                self._note_launch(
                    phase="correctness",
                    ordinal=call_ordinal,
                    case=case,
                    case_ordinal=int(declared_cases.index(case)),
                    data=data,
                    trial=trial,
                    dispatch=records,
                )
                call_ordinal += 1
                dispatch.extend(records)
                outs[trial] = self.argset.read_results()
                violations = self.argset.safety_violations()
                if violations:
                    safety.extend(f"[{case}:{trial}] {v}" for v in violations)
                if trial != "out1" and not violations and outs[trial] != outs["out1"]:
                    nondet.append(
                        f"[{case}:{trial}] output is not independent of the initial write-output contents/read_write state; "
                        "every correctness trial must restore read_write and write every pure output"
                    )
                if not determinism:
                    break
            if host_failure is not None:
                realization: dict[str, Any] | None = None
                if self.contract.instance.cu_seqlens is not None and self.contract.instance.canonical_seq_lens is not None:
                    realization = case_realization(
                        case,
                        segment_lengths(data.rounded["cu_seqlens"]),
                        tuple(int(value) for value in self.contract.instance.canonical_seq_lens),
                    )
                results.append(
                    CaseResult(
                        case=case,
                        seed=data.seed,
                        ok=False,
                        scaled_err=float("inf"),
                        allow=1.0,
                        baseline_scaled_err=None,
                        n_bad=0,
                        n_elems=0,
                        nan_count=0,
                        unwritten=0,
                        worst=None,
                        input_sha256=data.input_digest,
                        case_realization=realization,
                        resolved_shape=None if data.resolved_document is None else dict(data.resolved_document),
                        message=(
                            f"host program failed ({host_failure.get('typed_error')}): "
                            f"{host_failure.get('error_detail') or ''}"
                        ),
                    )
                )
                break
            result = _check_v2_case(self.contract, self.abi, data, outs["out1"], case)
            if safety or nondet:
                result.safety = [m for m in safety if m.startswith(f"[{case}:")][-4:]
                result.ok = False
            results.append(result)
            if not result.ok or safety or nondet:
                break
        return results, safety, nondet, dispatch, host_records

    def _record_benchmark_mixture(
        self,
        cases: Sequence[str],
        mixture_cases: Sequence[str],
        datasets: Mapping[str, "V2Case"],
    ) -> None:
        """@brief 按声明顺序记录 timed mixture 的 per-call launch 证据与 ``benchmark_batches``。

        ``case_ordinal`` is the case's index in the **declared** order, not the rotation ordinal.  With
        ``rotation > len(cases)`` the mixture repeats the first cases, and an ordinal-based value would
        claim that the repeated ``canonical`` batch is case 9..15 — the evidence would then disagree with
        the declared order it is supposed to prove.  ``ordinal`` stays the per-call sequence number.

        This is the production path (``benchmark`` calls it before timing); the capacity evidence test
        calls it directly so the ordinal mapping is covered without a Metal device.

        @param cases 声明的 case 名（声明顺序）。
        @param mixture_cases 每个 rotation set 的 case（``benchmark_mixture`` 的产物）。
        @param datasets case → 该 case 的 ``V2Case``。
        """
        declared = tuple(str(case) for case in cases)
        self.benchmark_batches = []
        for ordinal, case in enumerate(mixture_cases):
            data = datasets[case]
            self._note_launch(
                phase="benchmark",
                ordinal=int(ordinal),
                case=case,
                case_ordinal=int(declared.index(str(case))),
                data=data,
                audited=(ordinal == 0),
            )
            if self.contract.instance.is_capacity:
                self.benchmark_batches.append(
                    {
                        "ordinal": int(ordinal),
                        "case": case,
                        "provider": self.provider,
                        "resolved_shape": None if data.resolved_document is None else dict(data.resolved_document),
                        "evaluation_input_digest": data.input_digest,
                    }
                )

    def benchmark(
        self,
        *,
        seed: int,
        reps: int,
        discipline: TimingDiscipline,
        baseline_runners: Mapping[str, Any] | None = None,
    ) -> "V2Benchmark":
        """@brief 用 v1 评测同一套计时边界测候选：登记基线一起交错、同样重试、同样选择。

        ``baseline_runners`` is the task's registered baseline timers (the same ``BaselineRunner``
        objects the v1 Metal path measures).  They are measured *in the same interleaved call as
        the candidate*, which is the v1 protocol: the candidate's median is then taken under the
        same GPU load/clock state and the same attempt-selection rule.  Measuring the candidate
        alone (as this runtime did before A5 closed) is a different measurement boundary: at the
        launch-bound ``small`` instance the clock never ramps and the ratio inflates by 30-70%.

        For an L2 capacity instance the rotation mixture is the **declared continuous-batching
        sequence**: set ``i`` runs declared case ``i % len(cases)``, in declared order, including the
        full-capacity profile and the smaller ones.  Always timing ``cases[0]`` at full capacity would
        make a candidate that only handles the capacity point look like a continuous-batching winner.
        Every set is prepared (inputs, declared padding, Host Program storage) before the timed loop,
        so padding and setup stay outside the measured window and the native direct-commit path is
        unchanged.  ``benchmark_batches`` records the ordered mixture the report can be traced to.
        """
        from ...bench import measure_interleaved

        rotation = max(1, int(discipline.rotation_sets))
        cases = tuple(self.contract.instance.case_domain.cases)
        mixture_cases, datasets = benchmark_datasets(self.contract, cases, rotation, seed)
        sets: list[V2ArgSet] = []
        for case in mixture_cases:
            data = datasets[case]
            one = V2ArgSet(self.ctx, self.abi, self.geometry)
            one.resolved_scalars = dict(data.resolved)
            one.resolved_shape_document = None if data.resolved_document is None else dict(data.resolved_document)
            one.evaluation_input_digest = data.input_digest
            one.load_initial(data.initial)
            fill_plan = benchmark_launch_plan(self.abi, self.geometry, discipline=discipline)
            one.apply_fill_plan(fill_plan)
            one.load_capacity_padding(self.contract.instance, data.resolved_shape)
            if self.host_plan is not None and not self.host_inside_timing:
                self.host_plan.execute(one)
            sets.append(one)
        # Ordered launch evidence: one entry per rotation set (the mixture that is timed), appended
        # before timing so a crash mid-measurement still says which batches were prepared.
        self._record_benchmark_mixture(cases, mixture_cases, datasets)
        rotation = len(sets)
        if self.contract.instance.is_capacity:
            problems = benchmark_mixture_problems(self.contract.instance, self.benchmark_batches, cases=cases)
            if problems:
                raise GeometryError(
                    "the runtime's own continuous-batching mixture does not satisfy the declared profile order: "
                    + "; ".join(problems)
                )
            timed = {str(batch["case"]) for batch in self.benchmark_batches}
            declared_shapes = {
                (shape.total_tokens, shape.num_seqs)
                for shape in (resolved_shape_for_case(self.contract.instance, case) for case in cases)
                if shape is not None
            }
            self.benchmark_mixture_notes = {
                "declared_cases": list(cases),
                "timed_cases": list(dict.fromkeys(mixture_cases)),
                "dropped_cases": [case for case in cases if case not in timed],
                "rotation_sets": int(rotation),
                # A declaration whose profiles all resolve to one shape is measured as one shape; calling
                # it a mixture would withhold a speedup that is in fact like-for-like.
                "mixture": len(declared_shapes) > 1,
            }
        else:
            self.benchmark_mixture_notes = {
                "declared_cases": list(cases),
                "timed_cases": [cases[0]],
                "dropped_cases": list(cases[1:]) if len(cases) > 1 else [],
                "rotation_sets": int(rotation),
                "mixture": False,
            }
        original = self.argset
        dispatch: list[DispatchRecord] = []
        host_record: dict[str, Any] | None = None
        kernels, grids, threads, tg = self._stage_sets()
        dispatch_workspace = [tuple(use.name for use in stage.workspace) for stage in self.geometry.active_stages]
        # The v1 path builds its rotating arguments once before the interleaved loop.  Do the same
        # here: rebuilding them inside every candidate batch is Python work the v1 path does not
        # do, and at the launch-bound ``small`` instance that difference is visible in the wall
        # time even though it sits outside ``run_pipeline``'s timer.
        stage_sets = [[one.stage_args(binding) for binding in self.bindings] for one in sets]
        host_written = self.host_program.written_workspaces if self.host_program is not None else frozenset()
        # A workspace the Host Program writes is initialized by the host, not by a device fill;
        # the two ownerships are mutually exclusive (checked at envelope time).
        fills = [one.workspace_fills_declared_zero(exclude=host_written) for one in sets]
        # Only read_write state and candidate Host Tiling need to be re-prepared between candidate
        # batches; neither exists in the A5 migration set.  Keeping the call for every task would
        # charge the candidate for host-side setup the v1 path performs once.
        reprepare = self.host_plan is not None or self.host_program is not None or any(
            tensor.access == "read_write" for tensor in self.abi.tensors if tensor.space != "workspace"
        )

        def record_dispatch(records: list[DispatchRecord]) -> None:
            for item in records:
                if item not in dispatch:
                    dispatch.append(item)

        #: Read-only ABI tensors cannot be changed by a kernel that passed the safety stage (the
        #: runtime verifies input immutability and `access="read"` is a read-only binding).  Their
        #: initial bytes are therefore restored once per set, before timing; re-writing them on
        #: every repetition would charge the timed window for a redundant copy of the whole input
        #: set (the v1 Metal path loads the rotating sets once for the same reason).  ``read_write``
        #: tensors *are* mutated by the kernel and keep the per-repetition restore.
        read_only_restored = False
        access = {t.name: t.access for t in self.abi.tensors if t.space != "workspace"}
        # Pre-built, per-set restore plans for the timed Host Program path: only ``read_write``
        # state is re-prepared each iteration.  Read-only inputs are never re-copied inside the
        # timed window, and no fill plan is rebuilt there either.
        repeat_plans: list[Any] = []
        for _one in sets:
            _plan = benchmark_launch_plan(self.abi, self.geometry, discipline=discipline)
            repeat_plans.append(
                type(_plan)(
                    phase=_plan.phase,
                    tensors=tuple(
                        fill
                        for fill in _plan.tensors
                        if fill.action == RESTORE_INITIAL and access.get(fill.name) == "read_write"
                    ),
                    guard_pattern=_plan.guard_pattern,
                    guard_bytes=_plan.guard_bytes,
                )
            )

        def _apply_bench_plans() -> None:
            """Fresh rotation + zero-filled workspaces, applied outside the timed window."""
            nonlocal host_record, read_only_restored
            for one in sets:
                plan = benchmark_launch_plan(self.abi, self.geometry, discipline=discipline)
                if read_only_restored:
                    plan = type(plan)(
                        phase=plan.phase,
                        tensors=tuple(
                            fill
                            if not (fill.action == RESTORE_INITIAL and access.get(fill.name) == "read")
                            else type(fill)(fill.name, fill.reason, "untouched")
                            for fill in plan.tensors
                        ),
                        guard_pattern=plan.guard_pattern,
                        guard_bytes=plan.guard_bytes,
                    )
                one.apply_fill_plan(plan, reset_guards=False)
                if self.host_plan is not None and not self.host_inside_timing:
                    host_record = self.host_plan.execute(one)
            read_only_restored = True

        def _batched_run(k: int, *, audit: bool) -> float:
            """One command buffer, ``k`` back-to-back launches over the rotating sets.

            This is the shared ``tcenv.bench`` protocol (``k`` launches timed as one batch and
            divided by ``k``) that the v1 Metal path also uses: launch/sync latency is amortised
            exactly once per batch.  A declared-zero workspace is zeroed by a fill encoded inside
            this buffer, before that iteration's stages; guards are never reset.
            """
            sequence = 0

            def recorder(stage_index, stage, grid, threads_, threadgroup_bytes, workspace) -> None:
                nonlocal sequence
                # Only the first iteration is recorded: every pass over the rotating sets dispatches
                # the same canonical stages, and the evidence must stay bounded for large batches.
                if sequence >= len(kernels):
                    sequence += 1
                    return
                record_dispatch(
                    [
                        DispatchRecord(
                            stage=int(self.geometry.active_stages[stage_index].index),
                            kernel=str(getattr(stage, "name", "")),
                            grid=tuple(int(v) for v in grid),
                            threads=tuple(int(v) for v in threads_),
                            threadgroup_bytes=int(threadgroup_bytes),
                            workspace=tuple(str(v) for v in workspace),
                            provider=self.provider,
                            sequence=sequence,
                        )
                    ]
                )
                sequence += 1

            self.argset = original
            timing = self.ctx.run_pipeline(
                kernels,
                grids,
                threads,
                tg,
                stage_sets,
                fills,
                iters=k,
                dispatch="concurrent",
                dispatch_recorder=recorder if audit else None,
                dispatch_workspace=dispatch_workspace,
            )
            if not audit:
                return timing.wall_s
            from ...track import selection_parity

            problems = selection_parity(
                self.selection,
                provider=self.provider,
                observed=list(
                    dict.fromkeys(
                        (item.kernel, item.grid, item.threads, item.threadgroup_bytes, item.workspace) for item in dispatch
                    )
                ),
                where="metal v2 runtime dispatch",
            )
            if problems:
                raise GeometryError("; ".join(problems))
            return timing.wall_s

        #: The first warm-up call records the real dispatch; later calls are the timed samples and
        #: must not pay the Python recorder callback (the v1 path has no such callback).
        dispatch_recorded = False

        host_walls_ns: list[int] = []
        native_walls_ns: list[int] = []
        dataflow_samples: dict[str, list[int]] = {}
        profile_enabled = os.environ.get("SF_NATIVE_PROFILE") == "1"
        device_ns: list[int] = []
        e2e_walls_ns: list[int] = []

        def cand_run(k: int) -> float:
            nonlocal host_record, dispatch_recorded
            record = not dispatch_recorded
            dispatch_recorded = True
            if self.host_program is not None:
                # One native Host Program call and one GPU command buffer per launch.  The host call, the
                # read_write restore and the span copy are all inside the measured wall; the
                # read-only inputs are NOT re-copied and no fill plan is rebuilt here.
                wall = 0.0
                for i in range(k):
                    index = i % len(sets)
                    one = sets[index]
                    self.argset = one
                    audit_here = record
                    started = time.perf_counter()
                    cb_before = int(getattr(self.ctx, "command_buffers", 0))
                    native_before = int(getattr(getattr(self.ctx, "_native_batch", None), "calls", 0))
                    one.apply_fill_plan(repeat_plans[index], reset_guards=False)
                    self.read_only_restores += sum(
                        1
                        for fill in repeat_plans[index].tensors
                        if fill.action == RESTORE_INITIAL and access.get(fill.name) != "read_write"
                    )
                    try:
                        records = self.run_once(
                            benchmark=True,
                            host_outside_done=True,
                            record_dispatch=audit_here,
                            stage_args=stage_sets[index],
                            fills=[fills[index]],
                            host_audit=audit_here,
                            host_profile=profile_enabled,
                        )
                    except HostProgramFailed as exc:
                        raise GeometryError(
                            f"the candidate Host Program failed during benchmarking: {exc}"
                        ) from exc
                    elapsed = time.perf_counter() - started
                    wall += elapsed
                    self.candidate_command_buffers += (
                        int(getattr(self.ctx, "command_buffers", 0)) - cb_before
                    ) + (int(getattr(getattr(self.ctx, "_native_batch", None), "calls", 0)) - native_before)
                    if audit_here:
                        # The audit/warm-up launch produces the digests; it is not a timed
                        # sample, and the timed samples never hash or build JSON.
                        if self.last_host_run is not None:
                            host_record = self.last_host_run
                    else:
                        e2e_walls_ns.append(int(elapsed * 1_000_000_000))
                        if self.last_host_timing is not None:
                            host_walls_ns.append(int(self.last_host_timing.host_wall_ns))
                            native_walls_ns.append(int(self.last_host_timing.native_wall_ns))
                            if self.last_host_timing.profile_ns:
                                for name, value in self.last_host_timing.profile_ns.items():
                                    dataflow_samples.setdefault(name, []).append(int(value))
                        device_ns.append(int(self.last_gpu_ns))
                    record_dispatch(records)
                    record = False
                self.argset = original
                return wall
            if reprepare:
                _apply_bench_plans()
            if self.host_plan is not None and self.host_inside_timing:
                # Host Tiling inside timing is a real per-launch host step, so this path keeps
                # one CPU host plan and one launch per iteration (it cannot be encoded into a
                # single GPU command buffer).  No A5-migrated task ships Host Tiling.
                wall = 0.0
                for i in range(k):
                    self.argset = sets[i % len(sets)]
                    self.argset.apply_fill_plan(
                        benchmark_launch_plan(self.abi, self.geometry, discipline=discipline), reset_guards=False
                    )
                    started = time.perf_counter()
                    records = self.run_once(benchmark=True, host_outside_done=False, record_dispatch=record)
                    wall += time.perf_counter() - started
                    if self.last_host_run is not None:
                        host_record = self.last_host_run
                    record_dispatch(records)
                self.argset = original
                return wall
            return _batched_run(k, audit=record)

        host_calls_before = 0 if self.host_program is None else int(self.host_program.calls)
        host_audit_before = 0 if self.host_program is None else int(self.host_program.audit_calls)
        stage_args_before = int(self.stage_args_builds)
        read_only_before = int(self.read_only_restores)
        command_buffers_before = int(getattr(self.ctx, "command_buffers", 0))
        native_before = int(getattr(getattr(self.ctx, "_native_batch", None), "calls", 0))
        candidate_buffers_before = int(self.candidate_command_buffers)
        noise, stats_all, attempts = self._measure(cand_run, baseline_runners or {}, seed=seed, reps=reps, rotation=rotation)
        command_buffers_after = int(getattr(self.ctx, "command_buffers", 0))
        native_after = int(getattr(getattr(self.ctx, "_native_batch", None), "calls", 0))
        self.benchmark_diagnostics = {
            "dataflow_profile": bool(profile_enabled),
            "host_native_calls": 0 if self.host_program is None else int(self.host_program.calls) - host_calls_before,
            "host_audit_calls": 0 if self.host_program is None else int(self.host_program.audit_calls) - host_audit_before,
            "stage_args_rebuilds": int(self.stage_args_builds) - stage_args_before,
            "read_only_restores": int(self.read_only_restores) - read_only_before,
            # Candidate-only committed buffers for the host path (baselines interleave in the same
            # context, so the process-wide delta would count their work too).
            "command_buffers": (
                int(self.candidate_command_buffers) - candidate_buffers_before
                if self.host_program is not None
                else (command_buffers_after - command_buffers_before) + (native_after - native_before)
            ),
            "native_batch_calls": native_after - native_before,
        }
        stats = stats_all["candidate"]
        # Per the A2 timing protocol, benchmark checks are guard/hardware checks only.  We do
        # not compare output terminal values here, and we deliberately never reset guards
        # between timed launches: an out-of-bounds write on the second launch must remain
        # visible after the last rep.
        guard_violations: list[str] = []
        for one in sets:
            guard_violations.extend(one.guard_violations())
        self.workspace_probe = {name: gb.read().hex() for name, gb in sets[0].workspace_bufs.items()}
        if host_record is not None and e2e_walls_ns:
            import statistics

            # One record carries both: the digests/reads/writes are from the audit launch, the
            # median nanoseconds are from the timed launches that never hashed.
            host_record["timing_mode"] = "timed"
            host_record["digest_mode"] = "audit"
            if host_walls_ns:
                host_record["host_tiling_wall_ns"] = int(statistics.median(host_walls_ns))
            if native_walls_ns:
                host_record["native_wall_ns"] = int(statistics.median(native_walls_ns))
            if dataflow_samples:
                host_record["dataflow_ns"] = {
                    name: int(statistics.median(values)) for name, values in sorted(dataflow_samples.items())
                }
            if any(device_ns):
                host_record["device_kernel_ns"] = int(statistics.median(device_ns))
            host_record["end_to_end_wall_ns"] = int(statistics.median(e2e_walls_ns))
            # The audit launch is always rotation set 0 (the first declared case); label it so the
            # evidence says which continuous-batching batch its digests belong to.
            first = datasets[mixture_cases[0]]
            host_record["evaluation_phase"] = "benchmark"
            host_record["evaluation_case"] = mixture_cases[0]
            host_record["evaluation_case_ordinal"] = 0
            if first.input_digest:
                host_record["evaluation_input_digest"] = first.input_digest
        return V2Benchmark(
            stats=stats,
            baselines={name: value for name, value in stats_all.items() if name != "candidate"},
            rotation_sets=rotation,
            dispatch=dispatch,
            host=host_record,
            guard_violations=guard_violations,
            noise_cv=float(noise),
            attempts=int(attempts),
            batches=list(self.benchmark_batches),
        )

    def _measure(
        self,
        cand_run: Any,
        baseline_runners: Mapping[str, Any],
        *,
        seed: int,
        reps: int,
        rotation: int,
    ) -> tuple[dict[str, Any], float, int]:
        """@brief 候选与最快基线交错计时；明显更慢的 Torch MPS 基线事后单独测。

        ``torch_mps`` 的单次启动比 MLX 慢一个数量级。把它放进同一次交错会把 Apple GPU 频率留在
        对 launch-bound 候选更不利的档上，而 MLX 几乎不受影响，于是 small GEMM 的 wall ratio
        被抬过 1.10。候选和 MLX（以及其它非 Torch 基线）仍在同一次交错、同一套重试里比较。
        Torch MPS 缓冲也推迟到这次比较之后才分配。亚毫秒的启动先预热，样本前再丢 6 轮，避免
        把爬频的前几轮写进中位数。返回的 noise 只描述这次比较，不含事后单独测量的 Torch。
        """
        from ...bench import BENCH_ATTEMPTS, NOISY_CV, attempt_noise, interleaved, select_attempt

        names = {str(name) for name in baseline_runners}
        fast: dict[str, Any] = {"candidate": cand_run}
        slow: dict[str, Any] = {}
        for name, runner in baseline_runners.items():
            key = str(name)
            if "torch" in key and any(item == "mlx" or item.startswith("mlx_") for item in names):
                slow[key] = runner
            else:
                fast[key] = runner
        if int(rotation) >= 8:
            for runner in fast.values():
                t_single = runner(1)
                if t_single < 0.001:
                    runner(max(int(rotation), min(1024, int(0.015 / max(t_single, 1e-6)))))
                    if t_single < 0.0002:
                        runner(1024)
        attempts: list[tuple[float, dict[str, Any]]] = []
        target_s = 0.060 if int(rotation) >= 8 else 0.015
        for attempt in range(BENCH_ATTEMPTS):
            stats = interleaved(
                fast,
                reps=reps,
                seed=seed + attempt,
                kmin=rotation,
                warmup_reps=16,
                target_s=target_s,
            )
            attempts.append((attempt_noise(stats), stats))
            if attempts[-1][0] <= NOISY_CV:
                break
        if attempts[-1][0] > NOISY_CV and int(rotation) >= 8:
            for extra in range(BENCH_ATTEMPTS, BENCH_ATTEMPTS + 9):
                stats = interleaved(
                    fast,
                    reps=reps,
                    seed=seed + extra,
                    kmin=rotation,
                    warmup_reps=24,
                    target_s=0.090,
                )
                attempts.append((attempt_noise(stats), stats))
                if attempts[-1][0] <= NOISY_CV:
                    break
            if attempts[-1][0] > NOISY_CV:
                attempts = attempts[:BENCH_ATTEMPTS]
        noise, kept = select_attempt(attempts, noisy_cv=NOISY_CV)
        if slow:
            kept = {**kept, **interleaved(slow, reps=reps, seed=seed, kmin=rotation)}
        return float(noise), kept, min(len(attempts), BENCH_ATTEMPTS)


def _canonical_stat(value: float) -> float | str:
    """@brief 把可能为非有限值的比较统计变成可进身份文档的规范值。

    A candidate may legitimately produce NaN/Inf (poison left unwritten, a wrong recurrence, a
    missing stage).  The verdict must then be ``incorrect``, not an evaluator crash: the published
    report is part of the evaluation identity, and ``tcenv.identity`` refuses non-finite floats.
    ``scaled_err`` already had its ``"inf"`` spelling; this is the same spelling for the worst
    element's ``got``/``want`` so the whole report stays canonical.

    @param value 比较统计值。
    @return 有限浮点数，或 ``"inf"``/``"-inf"``/``"nan"``。
    """
    if value != value:
        return "nan"
    if value == float("inf"):
        return "inf"
    if value == float("-inf"):
        return "-inf"
    return float(value)


def _format_stat(value: float | str) -> str:
    """@brief 打印一个已规范化的比较统计（有限值用 ``.6g``，非有限值按规范拼写）。

    @param value 已由 :func:`_canonical_stat` 规范化的值。
    @return 诊断文本。
    """
    return f"{value:.6g}" if isinstance(value, float) else str(value)


def _check_v2_case(contract: TaskContract, abi: TensorMemoryContract, data: V2Case, out_raw: dict[str, bytes], case: str) -> CaseResult:
    """@brief 按 v2 result tensor 集合比较终值；read_write 参与差值与终值，不参与 unwritten。

    L1 还额外记录这个 case 在**本次实际分段**上到底实现了什么（exact/partial/
    not_constructible/deviates/unclassified）；它是覆盖证据，不参与候选的数值判决。
    """
    instance = contract.instance
    capacity_shape: Any = None
    if instance.is_capacity:
        # An L2 case *must* carry a resolved shape: ``resolved_shape=None`` is not "no L2 contract", it is
        # missing evidence for a capacity instance.  Gating the whole verification on ``is not None`` let a
        # caller skip the boundary/padding/digest checks and fall through to a whole-capacity comparison
        # (whose reference padding then matched the candidate's untouched bytes), so the gate is the
        # instance kind, and ``_verified_resolved_shape`` refuses the missing value itself.
        # The declared resolved shape is *not* taken on trust: it has to be the batch in the bytes, and
        # the batch has to be the declared workload profile of this case.  Otherwise a forged
        # ``resolved_shape`` (say capacity 16 while the boundaries say 1) would move the logical region
        # and hide everything else as "padding".
        capacity_shape = _verified_resolved_shape(contract, data, case)
        # Same rule as the document validator: an axis the harness cannot prove resolved-or-invariant
        # refuses the comparison instead of silently comparing the whole capacity buffer.
        require_capacity_axes(instance)
    capacity_mode = capacity_shape is not None
    realization: dict[str, Any] | None = None
    if instance.cu_seqlens is not None and instance.canonical_seq_lens is not None:
        realization = case_realization(
            case,
            segment_lengths(data.rounded["cu_seqlens"]),
            tuple(int(value) for value in instance.canonical_seq_lens),
        )
    elif capacity_mode:
        # L2 has no canonical segmentation in its identity, so the coverage evidence compares the actual
        # prefix against the workload profile's declared ``seq_lens`` for this case.
        realization = case_realization(
            case,
            _prefix_lengths(data.rounded["cu_seqlens"]),
            tuple(int(value) for value in capacity_shape.canonical_seq_lens),
        )
    worst_all = 0.0
    n_bad = 0
    n_total = 0
    nan_total = 0
    unwritten_total = 0
    padding_total = 0
    padding_detail: list[str] = []
    worst_info: dict[str, Any] | None = None
    for tensor in abi.tensors:
        if tensor.space == "workspace" or tensor.access not in ("write", "read_write"):
            continue
        d = get_dtype(tensor.dtype)
        shape = tuple(int(v) for v in tensor.shape)
        got_full = from_raw(out_raw[tensor.name], tensor.dtype, shape)
        want = data.ref[tensor.name]
        scale = data.scale[tensor.name]
        if capacity_mode:
            # L2: only the *logical* region is a result.  ``output[:resolved_total]`` and
            # ``final_state[:resolved_num_seqs]`` are compared; everything outside is padding and is
            # verified separately to still hold the declared poison byte.  A full-buffer comparison would
            # silently accept a candidate that wrote into the padding, and would fail an honest one whose
            # padding is the declared poison instead of reference values.
            logical, slabs = padding_slabs(instance, capacity_shape, shape, declared_dim_terms(instance, tensor.name))
            got = got_full[logical]
            want = want[logical]
            # The scale of an L2 result is already defined on the logical region (``prepare_v2_case``
            # computes it there, and a task hook is handed logical views only).  Slicing it here would be
            # wrong for a per-row scale, and re-deriving it from the capacity buffer would re-introduce the
            # poison denominator this fix exists to remove.
            target = tuple(int(dim) for dim in got.shape)
            if not broadcasts_scale(tuple(int(dim) for dim in scale.shape), target):
                raise TaskContractError(
                    f"{contract.reference.path}: the L2 error scale for {tensor.name!r} has shape "
                    f"{tuple(int(dim) for dim in scale.shape)}, which does not broadcast to the logical result "
                    f"shape {target}; the comparator refuses to guess a scale for the compared region"
                )
            if tensor.access == "write":
                unwritten_total += _unwritten_bytes_region(out_raw[tensor.name], shape, d.nbytes, logical)
            if slabs:
                bad_padding = _capacity_padding_bytes(
                    out_raw[tensor.name], shape, d.nbytes, slabs, int(instance.capacity.padding_poison_byte)
                )
                if bad_padding:
                    padding_total += bad_padding
                    padding_detail.append(f"{tensor.name} ({bad_padding} byte(s) outside the logical region)")
        else:
            got = got_full
            if tensor.access == "write":
                unwritten_total += _unwritten_bytes(out_raw[tensor.name], d.nbytes)
        s = scaled_error(got, want, scale, float(instance.tolerances.rtol), float(instance.tolerances.atol_rel), bool(instance.tolerances.exact), _floor(tensor.dtype))
        n_total += int(s.numel())
        nan_total += int(torch.isnan(got.to(torch.float64)).sum()) if d.is_float else 0
        bad = s > 1.0
        n_bad += int(bad.sum())
        maximum = float(s.max())
        if maximum > worst_all or worst_info is None:
            index = int(torch.argmax(s))
            import numpy as np

            position = tuple(int(v) for v in np.unravel_index(index, tuple(int(v) for v in got.shape)))
            got_value = float(got.reshape(-1)[index])
            want_value = float(want.reshape(-1)[index])
            worst_info = {
                "output": tensor.name,
                "index": list(position),
                "got": _canonical_stat(got_value),
                "want": _canonical_stat(want_value),
                "scaled_err": maximum if maximum != float("inf") else "inf",
            }
        worst_all = max(worst_all, maximum)
    ok = n_bad == 0 and padding_total == 0
    message = ""
    if padding_total:
        message = (
            f"{padding_total} padding byte(s) outside the logical region of "
            f"[total_tokens={capacity_shape.total_tokens}, num_seqs={capacity_shape.num_seqs}] were changed; "
            f"the unused region must stay at the declared poison 0x{int(instance.capacity.padding_poison_byte):02X} "
            f"({'; '.join(padding_detail)})"
        )
    if not ok:
        if message:
            message += "; "
        message += f"{n_bad}/{n_total} elements out of tolerance (worst scaled error {worst_all:.3g})"
        if unwritten_total:
            message += f"; {unwritten_total} pure-write elements still hold the poison pattern"
        if worst_info:
            message += (
                f"; worst: {worst_info['output']}{worst_info['index']} "
                f"got {_format_stat(worst_info['got'])} want {_format_stat(worst_info['want'])}"
            )
    return CaseResult(
        case=case,
        seed=data.seed,
        ok=ok,
        scaled_err=worst_all,
        allow=1.0,
        baseline_scaled_err=None,
        n_bad=n_bad,
        n_elems=n_total,
        nan_count=nan_total,
        unwritten=unwritten_total,
        worst=worst_info,
        # L1/L2：报告里记录的是**本次实际分段/resolved 形状**的输入身份，而不是声明分布；失败也能重放。
        input_sha256=data.input_digest,
        case_realization=realization,
        resolved_shape=None if not capacity_mode else dict(capacity_shape.document()),
        message=message,
    )


def _verified_resolved_shape(contract: TaskContract, data: V2Case, case: str) -> Any:
    """@brief 校验一个 L2 case 的 resolved 声明确实是**字节里的批次**和**声明的 profile**。

    ``prepare_v2_case`` already derives the shape from the declared profile and validates the produced
    bytes against it, but the comparator is the last gate before a verdict: it must not trust an object it
    was handed.  Three things have to agree, and none of them is allowed to come from the task:

    1. the ``cu_seqlens`` bytes must be a legal batch for the **declared** profile of this case
       (prefix legality, prefix length ``num_seqs + 1``, last prefix entry ``== total``, sentinel tail);
    2. ``data.resolved_shape`` / ``data.resolved`` / ``data.resolved_document`` must be exactly that
       declared shape (a forged ``total_tokens`` would move the logical region and re-label padding);
    3. the actual prefix in the bytes must be the one the declared profile names — the resolved total and
       count are read from the bytes, never from a case name or a candidate claim.

    @param contract 任务合同。
    @param data 该 case 的 ``V2Case``。
    @param case 用例名。
    @return 声明的 ``ResolvedShape``（之后所有 logical/padding/证据都用它）。
    @raises VarlenContractError resolved 声明与字节/声明 profile 不一致。
    """
    instance = contract.instance
    declared = resolved_shape_for_case(instance, case)
    if declared is None or declared.kind != "l2":  # pragma: no cover - callers gate on is_capacity
        raise VarlenContractError(f"{instance.op_name}.{instance.size}/{case}: not an L2 capacity case")
    bound = [binding for binding in instance.tensors if binding.name == "cu_seqlens"]
    if not bound:  # pragma: no cover - the varlen ABI requires it
        raise VarlenContractError("the capacity ABI declares no cu_seqlens tensor")
    expected_shape = tuple(int(dim) for dim in bound[0].shape)
    # ``correctness()`` loads ``data.initial`` (the storage bytes) into the argset, so that is the boundary
    # the device actually runs.  Rebuild it from those bytes instead of trusting ``data.rounded``: a forged
    # case could keep the rounded/ref/resolved/digest of a tiny batch and hand the device the full boundary.
    import numpy as np

    storage = data.initial.get("cu_seqlens")
    if not isinstance(storage, (bytes, bytearray)):
        raise VarlenContractError(
            f"{instance.op_name}.{instance.size}/{case}: the case carries no cu_seqlens storage bytes, so the "
            "boundary the run would load cannot be validated"
        )
    expected_bytes = int(expected_shape[0]) * 4  # the boundary ABI is i32
    if len(storage) != expected_bytes:
        raise VarlenContractError(
            f"{instance.op_name}.{instance.size}/{case}: the cu_seqlens storage carries {len(storage)} bytes, "
            f"but the declared capacity shape {expected_shape} at 4 byte(s) per offset needs {expected_bytes}"
        )
    storage_cu = torch.from_numpy(np.frombuffer(bytes(storage), dtype=np.int32).copy()).reshape(expected_shape)
    rounded = data.rounded.get("cu_seqlens")
    if not isinstance(rounded, torch.Tensor) or not torch.equal(storage_cu, rounded.reshape(expected_shape)):
        raise VarlenContractError(
            f"{instance.op_name}.{instance.size}/{case}: the boundary the run will load (`initial` storage) is "
            "not the boundary that was validated (`rounded`); the verdict and its evidence would describe a "
            "different batch than the device executes"
        )
    problems = cu_seqlens_problems(instance, declared, storage_cu, expected_shape=expected_shape)
    if problems:
        raise VarlenContractError(
            f"{instance.op_name}.{instance.size}/{case}: the cu_seqlens bytes are not the declared profile "
            f"[{declared.total_tokens} tokens / {declared.num_seqs} sequences]: {'; '.join(problems)}"
        )
    supplied = data.resolved_shape
    if supplied is None:
        raise VarlenContractError(f"{instance.op_name}.{instance.size}/{case}: an L2 case carries no resolved shape")
    mismatches: list[str] = []
    for field in (
        "kind",
        "total_tokens",
        "num_seqs",
        "capacity_total_tokens",
        "capacity_num_seqs",
        "cu_seqlens_sentinel",
        "padding_poison_byte",
        "workload_profile_id",
    ):
        if getattr(supplied, field, None) != getattr(declared, field, None):
            mismatches.append(
                f"{field}={getattr(supplied, field, None)!r} (declared {getattr(declared, field, None)!r})"
            )
    if tuple(getattr(supplied, "canonical_seq_lens", ()) or ()) != declared.canonical_seq_lens:
        mismatches.append("canonical_seq_lens do not match the declared profile")
    if dict(data.resolved or {}) != declared.resolved_scalars():
        mismatches.append(f"resolved scalars {dict(data.resolved or {})} are not {declared.resolved_scalars()}")
    # The input identity is reconstructed from the **actual** boundary bytes, not from ``data.rounded``: the
    # digest names the batch the run will execute.
    digest_inputs = dict(data.rounded)
    digest_inputs["cu_seqlens"] = storage_cu
    expected_digest = evaluation_input_digest(instance, case, int(data.seed), digest_inputs)
    if data.input_digest != expected_digest:
        mismatches.append(
            f"input digest {data.input_digest!r} is not the digest of these inputs ({expected_digest})"
        )
    if data.resolved_document is None:
        # An L2 case's evidence *is* the resolved-shape document (HostRun/launch records carry it).  A
        # missing one is refused here instead of producing a verdict/record that cannot say which batch
        # it judged; ``prepare_v2_case`` always sets it, so only a forged case can reach this.
        mismatches.append("an L2 case carries no resolved-shape document")
    elif dict(data.resolved_document) != declared.document():
        mismatches.append("the resolved-shape document is not the declared profile document")
    if mismatches:
        raise VarlenContractError(
            f"{instance.op_name}.{instance.size}/{case}: the supplied resolved shape is not the declared "
            f"workload profile of this case ({declared.workload_profile_id}): " + "; ".join(mismatches)
        )
    return declared


def host_run_identity(item: Mapping[str, Any]) -> tuple:
    """@brief 一条 HostRunEvidence 的稳定身份（去重与幂等合并共用）。

    Two correctness trials of one case feed the same read-only inputs and the same resolved shape, so the
    host program sees the same call and produces the same evidence; keeping three copies would only inflate
    the record.  Two calls with the same case name but *different* host reads/writes are different evidence
    and both survive.  Timing fields are excluded on purpose (a timed record is a different mode of the
    same audit), and the failure/success distinction is part of the key.
    """
    payload = {
        "input_reads": list(item.get("input_reads") or []),
        "workspace_writes": list(item.get("workspace_writes") or []),
        "unwritten_policy": dict(item.get("unwritten_policy") or {}),
    }
    content = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()
    return (
        str(item.get("schema_id")),
        str(item.get("candidate_sha256")),
        str(item.get("entry_path")),
        str(item.get("evaluation_phase")),
        item.get("evaluation_case_ordinal"),
        str(item.get("evaluation_case")),
        str(item.get("evaluation_input_digest")),
        int(item.get("host_status", 0)),
        str(item.get("typed_error")),
        content,
        str(item.get("resolved_scalars")),
    )


def _prefix_lengths(cu: Any) -> tuple[int, ...]:
    """@brief 从（可能带 capacity 尾部的）``cu_seqlens`` 取实际分段长度（L2 覆盖证据用）。

    Stops at the declared negative sentinel of the instance's capacity tier.  This is only used for the
    coverage record; the numeric verdict uses the resolved shape and the logical slices.

    @param cu ``cu_seqlens`` 张量。
    @return 每段 token 数（空序列为 0）。
    """
    values = [int(value) for value in cu.tolist()]
    for index, value in enumerate(values):
        if value < 0:
            values = values[:index]
            break
    return tuple(end - begin for begin, end in zip(values, values[1:]))


def _unwritten_bytes(raw: bytes, nbytes: int) -> int:
    import numpy as np

    a = np.frombuffer(raw, dtype=np.uint8).reshape(-1, nbytes)
    return int((a == POISON_PATTERN).all(axis=1).sum())


def _element_bytes(raw: bytes, shape: tuple[int, ...], nbytes: int) -> Any:
    """@brief 把裸字节看成 ``shape + (elem_nbytes,)`` 的 uint8 数组（逐元素一行）。

    @param raw 张量字节。
    @param shape 声明的 capacity 形状。
    @param nbytes 单元素字节数。
    @return numpy uint8 数组。
    """
    import numpy as np

    return np.frombuffer(raw, dtype=np.uint8).reshape(shape + (int(nbytes),))


def _unwritten_bytes_region(raw: bytes, shape: tuple[int, ...], nbytes: int, logical: tuple[slice, ...]) -> int:
    """@brief 只统计 logical 区域内仍是 poison 模式的元素数（L2 用）。

    @param raw 张量字节。
    @param shape 声明的 capacity 形状。
    @param nbytes 单元素字节数。
    @param logical logical 区域的切片元组。
    @return 元素数。
    """
    a = _element_bytes(raw, shape, nbytes)[tuple(logical)]
    return int((a == POISON_PATTERN).all(axis=-1).sum())


def _capacity_padding_bytes(
    raw: bytes,
    shape: tuple[int, ...],
    nbytes: int,
    slabs: tuple[tuple[slice, ...], ...],
    expected: int,
) -> int:
    """@brief logical 区域之外仍有几个字节不是声明的 padding poison（L2 用）。

    Byte-level on purpose: an fp16/bf16/fp32 write of ``0`` into the padding is detectable without any
    dtype-dependent NaN convention, and the same check applies to every output dtype.

    @param raw 张量字节。
    @param shape 声明的 capacity 形状。
    @param nbytes 单元素字节数。
    @param slabs padding slab 列表（``padding_slabs`` 的产物）。
    @param expected 声明的 padding 字节。
    @return 不合规的字节数（0 表示 padding 完好）。
    """
    a = _element_bytes(raw, shape, nbytes)
    total = 0
    for slab in slabs:
        region = a[tuple(slab)]
        if int(region.size) == 0:
            continue
        total += int((region != int(expected)).sum())
    return total


__all__ = [
    "DispatchRecord",
    "HOST_TILING_SCHEMA_ID",
    "HostTilingPlan",
    "HostTilingWrite",
    "MetalV2Runtime",
    "V2ArgSet",
    "V2Case",
    "host_write_span",
    "possible_workspace_uses",
    "prepare_v2_case",
    "reflection_for_kernel",
    "workspace_size_problems",
]
