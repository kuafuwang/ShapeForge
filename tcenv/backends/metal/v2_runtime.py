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
import struct
import time
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch

from ...contract_errors import GeometryError, TaskContractError
from ...dtypes import get_dtype
from ...portfolio.domain import evaluation_input_digest
from ...taskcontract.loader import TaskContract
from ...taskcontract.varlen import segment_lengths
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
    #: L1 的实际输入身份（case + seed + 实际 ``cu_seqlens`` 字节 + 输入张量身份），失败也能重放。
    #: 固定 shape 实例保持空串：它们的输入身份仍由既有实例指纹与证据字段表达，不新增键。
    input_digest: str = ""


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
    scale: dict[str, torch.Tensor] = {}
    if contract.reference.error_scale_kind not in ("none", ""):
        scale = contract.reference.error_scale(instance, rounded, ref)
    else:
        for name, value in ref.items():
            finite = torch.where(torch.isfinite(value), value, torch.zeros_like(value)) if value.is_floating_point() else value
            scale[name] = torch.sqrt(torch.mean(finite.to(torch.float64) ** 2))
    return V2Case(
        case=case,
        seed=int(seed),
        initial=initial,
        rounded=rounded,
        ref=ref,
        scale=scale,
        input_digest=(
            evaluation_input_digest(instance, case, int(seed), rounded) if instance.cu_seqlens is not None else ""
        ),
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

    def workspace_fills_declared_zero(self) -> list[tuple]:
        """@brief 计时窗内只清候选声明 ``zero`` 的 workspace。"""
        return [
            (gb.buf, gb.guard, gb.nbytes)
            for name, gb in self.workspace_bufs.items()
            if self.workspace_zero.get(name, False)
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

    def document(self) -> dict[str, Any]:
        return {
            "candidate": self.stats.to_json(),
            "baselines": {name: value.to_json() for name, value in self.baselines.items()},
            "rotation_sets": int(self.rotation_sets),
            "noise_cv": float(self.noise_cv),
            "attempts": int(self.attempts),
            "guard_violations": list(self.guard_violations),
        }


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
        self.host_inside_timing = bool(host_inside_timing)
        self.limits = limits
        by_name = {str(k.name): k for k in kernels}
        self.kernels = by_name
        self._check_workspace_sizes(launch_spec)
        self.bindings = self._bind(by_name)
        self.argset = V2ArgSet(ctx, abi, geometry)
        self.last_host_run: dict[str, Any] | None = None
        self.workspace_probe: dict[str, str] = {}
        self._check_host_plan()

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

    def _stage_sets(self) -> tuple[list[Any], list[tuple[int, int, int]], list[tuple[int, int, int]], list[int]]:
        active = self.geometry.active_stages
        kernels = [self.kernels[stage.kernel] for stage in active]
        grids = [tuple(int(v) for v in stage.grid) for stage in active]
        threads = [tuple(int(v) for v in stage.threads) for stage in active]
        tg = [int(stage.threadgroup_bytes) for stage in active]
        return kernels, grids, threads, tg

    def run_once(
        self, *, benchmark: bool, host_outside_done: bool = False, record_dispatch: bool = True
    ) -> list[DispatchRecord]:
        """@brief 真实执行一次；可选记录 dispatch。

        ``record_dispatch=False`` is for a timed batch: the recorder is a Python callback invoked
        immediately before every ``dispatchThreadgroups`` call, so keeping it enabled inside the
        measured window charges the candidate for evidence bookkeeping that the v1 path does not
        pay.  The first warm-up launch records the same dispatch identity, outside the samples.
        """
        kernels, grids, threads, tg = self._stage_sets()
        stage_args = [self.argset.stage_args(binding) for binding in self.bindings]
        fills = [self.argset.workspace_fills_declared_zero()] if benchmark else [[]]
        self.last_host_run = None
        if benchmark and self.host_plan is not None and not self.host_inside_timing and not host_outside_done:
            # Caller normally prepares before timing; this keeps a standalone invocation honest.
            self.last_host_run = self.host_prepare()
        # A zero-filled workspace is written by a GPU fill kernel.  For Host Tiling that runs
        # inside timing, encoding a CPU write into the same uncommitted command buffer is too
        # early: the GPU fill would execute later and overwrite it.  Complete the fill in its
        # own command buffer first, then write the workspace on the CPU, then submit the device
        # stages.  The whole sequence stays inside the caller's benchmark timing window.
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

        self.ctx.run_pipeline(
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
    ) -> tuple[list[CaseResult], list[str], list[str], list[DispatchRecord], dict[str, Any] | None]:
        use_cases = tuple(cases or self.contract.instance.case_domain.cases)
        results: list[CaseResult] = []
        safety: list[str] = []
        nondet: list[str] = []
        dispatch: list[DispatchRecord] = []
        host_records: list[dict[str, Any]] = []
        for index, case in enumerate(use_cases):
            data = prepare_v2_case(self.contract, case, seed + 7919 * (index + 1))
            self.argset.load_initial(data.initial)
            outs: dict[str, dict[str, bytes]] = {}
            for trial in CORRECTNESS_TRIALS:
                plan = correctness_launch_plan(self.abi, self.geometry, trial)
                self.argset.apply_fill_plan(plan)
                host = self.host_prepare() if self.host_plan is not None and not self.host_inside_timing else None
                records = self.run_once(benchmark=False)
                if host is None and self.last_host_run is not None:
                    host = self.last_host_run
                if host is not None:
                    host_records.append(host)
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
            result = _check_v2_case(self.contract, self.abi, data, outs["out1"], case)
            if safety or nondet:
                result.safety = [m for m in safety if m.startswith(f"[{case}:")][-4:]
                result.ok = False
            results.append(result)
            if not result.ok or safety or nondet:
                break
        return results, safety, nondet, dispatch, (host_records[-1] if host_records else None)

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
        """
        from ...bench import measure_interleaved

        data = prepare_v2_case(self.contract, self.contract.instance.case_domain.cases[0], seed + 1)
        rotation = max(1, int(discipline.rotation_sets))
        sets: list[V2ArgSet] = []
        for _ in range(rotation):
            one = V2ArgSet(self.ctx, self.abi, self.geometry)
            one.load_initial(data.initial)
            plan = benchmark_launch_plan(self.abi, self.geometry, discipline=discipline)
            one.apply_fill_plan(plan)
            if self.host_plan is not None and not self.host_inside_timing:
                self.host_plan.execute(one)
            sets.append(one)
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
        fills = [one.workspace_fills_declared_zero() for one in sets]
        # Only read_write state and candidate Host Tiling need to be re-prepared between candidate
        # batches; neither exists in the A5 migration set.  Keeping the call for every task would
        # charge the candidate for host-side setup the v1 path performs once.
        reprepare = self.host_plan is not None or any(
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

        def cand_run(k: int) -> float:
            nonlocal host_record, dispatch_recorded
            record = not dispatch_recorded
            dispatch_recorded = True
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

        noise, stats_all, attempts = self._measure(cand_run, baseline_runners or {}, seed=seed, reps=reps, rotation=rotation)
        stats = stats_all["candidate"]
        # Per the A2 timing protocol, benchmark checks are guard/hardware checks only.  We do
        # not compare output terminal values here, and we deliberately never reset guards
        # between timed launches: an out-of-bounds write on the second launch must remain
        # visible after the last rep.
        guard_violations: list[str] = []
        for one in sets:
            guard_violations.extend(one.guard_violations())
        self.workspace_probe = {name: gb.read().hex() for name, gb in sets[0].workspace_bufs.items()}
        return V2Benchmark(
            stats=stats,
            baselines={name: value for name, value in stats_all.items() if name != "candidate"},
            rotation_sets=rotation,
            dispatch=dispatch,
            host=host_record,
            guard_violations=guard_violations,
            noise_cv=float(noise),
            attempts=int(attempts),
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
        """@brief 候选与登记基线放进同一次可重试的交错计时。"""
        from ...bench import measure_interleaved

        runners: dict[str, Any] = {"candidate": cand_run}
        for name, runner in baseline_runners.items():
            runners[str(name)] = runner
        return measure_interleaved(runners, reps=reps, seed=seed, kmin=rotation)


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
    realization: dict[str, Any] | None = None
    if instance.cu_seqlens is not None and instance.canonical_seq_lens is not None:
        realization = case_realization(
            case,
            segment_lengths(data.rounded["cu_seqlens"]),
            tuple(int(value) for value in instance.canonical_seq_lens),
        )
    worst_all = 0.0
    n_bad = 0
    n_total = 0
    nan_total = 0
    unwritten_total = 0
    worst_info: dict[str, Any] | None = None
    for tensor in abi.tensors:
        if tensor.space == "workspace" or tensor.access not in ("write", "read_write"):
            continue
        d = get_dtype(tensor.dtype)
        got = from_raw(out_raw[tensor.name], tensor.dtype, tuple(int(v) for v in tensor.shape))
        want = data.ref[tensor.name]
        scale = data.scale[tensor.name]
        s = scaled_error(got, want, scale, float(instance.tolerances.rtol), float(instance.tolerances.atol_rel), bool(instance.tolerances.exact), _floor(tensor.dtype))
        n_total += int(s.numel())
        nan_total += int(torch.isnan(got.to(torch.float64)).sum()) if d.is_float else 0
        if tensor.access == "write":
            unwritten_total += _unwritten_bytes(out_raw[tensor.name], d.nbytes)
        bad = s > 1.0
        n_bad += int(bad.sum())
        maximum = float(s.max())
        if maximum > worst_all or worst_info is None:
            index = int(torch.argmax(s))
            import numpy as np

            position = tuple(int(v) for v in np.unravel_index(index, tuple(int(v) for v in tensor.shape)))
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
    ok = n_bad == 0
    message = ""
    if not ok:
        message = f"{n_bad}/{n_total} elements out of tolerance (worst scaled error {worst_all:.3g})"
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
        # L1：报告里记录的是**本次实际分段**的输入身份，而不是 canonical 分布；失败也能重放。
        input_sha256=data.input_digest,
        case_realization=realization,
        message=message,
    )


def _unwritten_bytes(raw: bytes, nbytes: int) -> int:
    import numpy as np

    a = np.frombuffer(raw, dtype=np.uint8).reshape(-1, nbytes)
    return int((a == POISON_PATTERN).all(axis=1).sum())


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
