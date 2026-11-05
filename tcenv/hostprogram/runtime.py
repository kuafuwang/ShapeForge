"""Execution of the compiled Host Program: one Module, one fresh Store per call (V3).

The runtime is the only place that touches live Wasm state, and it is deliberately backend
neutral: it knows a contract, a compiled module and byte ranges, not a Metal buffer or an NPU.
The Metal executor (``tcenv.backends.metal.host_program``) copies the returned valid spans into
the real workspace buffers and records ``HostRunEvidence``.

Invariants enforced here, on every call:

* the module's **own exported linear memory** is used; a module without ``memory`` is refused;
* mapped inputs, workspace spans and the ABI struct are pairwise disjoint regions of that memory
  (verified by deriving the storage base from each declared pointer and requiring one base);
* a call either returns 0 and produces validated spans, or it produces **nothing**: a non-zero
  return code, a trap, fuel exhaustion, an out-of-capacity span, a guard hit or a mutated
  read-only input all discard every span and resolved scalar.  The caller never sees a partial
  plan;
* the audit run hashes the actual read intervals and the valid span bytes; a timed run only copies
  the span bytes and counts them, never computes SHA-256 or builds JSON inside the measured window.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import struct
import time
from typing import Any, Callable, Mapping

from .contract import (
    INPUT_READ_IMPORTS,
    STORAGE_GUARD_BYTES,
    STORAGE_GUARD_PATTERN,
    HostAbiLayout,
    HostContractSpec,
    HostStorageRegion,
    storage_layout,
)
from .errors import HostProgramError
from .toolchain import HostBuild, _validate_module

ImportReader = Callable[[str, int, int], bytes]

_IMPORT_DTYPES = dict(INPUT_READ_IMPORTS)
_IMPORT_FORMATS = {"i32": "<i", "u32": "<I", "fp32": "<f"}


@dataclass(frozen=True)
class HostInputRead:
    """@brief 一次 ``sf_input_read_*`` 实际读取的区间。"""

    name: str
    tensor_id: int
    dtype: str
    elem_off: int
    count: int
    nbytes: int
    sha256: str | None

    def document(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "tensor_id": int(self.tensor_id),
            "dtype": self.dtype,
            "elem_off": int(self.elem_off),
            "count": int(self.count),
            "nbytes": int(self.nbytes),
            "sha256": self.sha256,
        }


@dataclass(frozen=True)
class HostWorkspaceSpan:
    """@brief 一个 workspace 的**单连续有效 span**（offset/count 均按元素计）。"""

    name: str
    offset: int
    count: int
    elem_nbytes: int
    data: bytes
    sha256: str | None

    @property
    def nbytes(self) -> int:
        return len(self.data)

    def document(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "offset": int(self.offset),
            "count": int(self.count),
            "elem_nbytes": int(self.elem_nbytes),
            "nbytes": int(self.nbytes),
            "sha256": self.sha256,
        }


@dataclass(frozen=True)
class HostRun:
    """@brief 一次 Wasm 调用的原子结果。"""

    status: int
    error_code: str | None
    error_detail: str
    host_wall_ns: int
    fuel_used: int
    input_reads: tuple[HostInputRead, ...]
    spans: tuple[HostWorkspaceSpan, ...]
    resolved_scalars: Mapping[str, int]
    audit: bool

    @property
    def ok(self) -> bool:
        return self.status == 0

    def document(self) -> dict[str, Any]:
        return {
            "status": int(self.status),
            "error_code": self.error_code,
            "error_detail": self.error_detail,
            "host_tiling_wall_ns": int(self.host_wall_ns),
            "fuel_used": int(self.fuel_used),
            "input_reads": [read.document() for read in self.input_reads],
            "workspace_writes": [span.document() for span in self.spans],
            "resolved_scalars": {key: int(value) for key, value in self.resolved_scalars.items()},
            "audit": bool(self.audit),
        }


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class HostRuntime:
    """@brief 复用 Module、每次调用新建 Store 的 Wasm 执行器。"""

    def __init__(self, build: HostBuild, spec: HostContractSpec) -> None:
        import wasmtime

        self.build = build
        self.spec = spec
        self.layout: HostAbiLayout = spec.layout()
        self.regions: tuple[HostStorageRegion, ...] = storage_layout(spec, build.used_workspaces)
        self._regions: dict[str, HostStorageRegion] = {region.name: region for region in self.regions}
        config = wasmtime.Config()
        config.consume_fuel = True
        self._wasmtime = wasmtime
        self.engine = wasmtime.Engine(config)
        try:
            self.module = wasmtime.Module(self.engine, build.wasm)
        except Exception as exc:  # noqa: BLE001
            raise HostProgramError(f"the stored Host Program module is not valid Wasm: {exc}") from exc
        _validate_module(self.module, where=build.entry_path)
        self._storage_bytes = max(
            (region.guard_hi + STORAGE_GUARD_BYTES for region in self.regions), default=16
        )

    def _store(self) -> Any:
        store = self._wasmtime.Store(self.engine)
        store.set_fuel(int(self.build.fuel))
        store.set_limits(memory_size=int(self.build.memory_budget_bytes), memories=1, instances=1, tables=1)
        return store

    def _linker(
        self, store: Any, reads: list[HostInputRead], read: ImportReader, audit_ref: list[bool]
    ) -> Any:
        import wasmtime

        linker = wasmtime.Linker(self.engine)
        spec = self.spec

        def make(dtype: str):
            fmt = _IMPORT_FORMATS[dtype]
            elem_nbytes = struct.calcsize(fmt)

            def callback(caller: Any, tensor_id: int, elem_off: int, dst_off: int, count: int) -> int:
                view = next((item for item in spec.inputs if item.tensor_id == int(tensor_id)), None)
                if view is None:
                    return 1
                if view.dtype != dtype:
                    return 1
                if elem_off < 0 or count < 0 or elem_off + count > view.numel:
                    return 2
                nbytes = count * elem_nbytes
                memory = caller.get("memory")
                if memory is None:
                    return 2
                if dst_off < 0 or dst_off + nbytes > memory.data_len(caller):
                    return 2
                data = read(view.name, elem_off * elem_nbytes, nbytes)
                if len(data) != nbytes:
                    return 2
                memory.write(caller, data, dst_off)
                reads.append(
                    HostInputRead(
                        name=view.name,
                        tensor_id=view.tensor_id,
                        dtype=dtype,
                        elem_off=int(elem_off),
                        count=int(count),
                        nbytes=int(nbytes),
                        sha256=_sha(data) if audit_ref[0] else None,
                    )
                )
                return 0

            return callback

        for import_name, dtype in INPUT_READ_IMPORTS:
            linker.define_func(
                "sf",
                import_name,
                wasmtime.FuncType([wasmtime.ValType.i32()] * 4, [wasmtime.ValType.i32()]),
                make(dtype),
                access_caller=True,
            )
        return linker

    def execute(self, read: ImportReader, *, values: Mapping[str, bytes] | None = None, audit: bool) -> HostRun:
        """@brief 执行一次 ``host_tiling``。

        @param read 按 ``(name, byte_off, nbytes)`` 取输入字节（不复制整块大张量）。
        @param values 可选的小输入完整快照（mapped 输入必须由它提供）。
        @param audit True 时计算读取区间/有效 span 的 sha256；计时路径传 False。
        @return 原子结果；失败时没有任何 span。
        @raises HostProgramError 模块结构/布局不安全（不是返回码问题）。
        """
        spec = self.spec
        layout = self.layout
        started = time.perf_counter_ns()
        reads: list[HostInputRead] = []
        store = self._store()
        audit_ref = [bool(audit)]
        linker = self._linker(store, reads, read, audit_ref)
        try:
            instance = linker.instantiate(store, self.module)
        except Exception as exc:  # noqa: BLE001 - a module that cannot instantiate is a candidate build defect
            raise HostProgramError(
                f"the compiled Host Program could not be instantiated within its capacity budget: {exc}"
            ) from exc
        exports = instance.exports(store)
        memory = exports.get("memory")
        if memory is None:
            raise HostProgramError("the Host Program does not export its own linear memory")
        memory_len = int(memory.data_len(store))
        base = int(exports["sf_host_abi_offset"](store))
        if base < 0 or base + layout.abi_bytes > memory_len:
            raise HostProgramError("sf_host_abi_offset points outside the module's own linear memory")

        # Derive one storage base from every allocated pointer and require them all to agree: this
        # proves the regions are the ones the generated header declared, not overlapping views.
        storage_base: int | None = None
        pointers: dict[str, int] = {}
        for region in self.regions:
            if region.kind == "input":
                view = spec.input(region.name)
                pointer_offset = base + layout.input_field(view.name).offset
            else:
                pointer_offset = base + layout.workspace_field(region.name).ptr_offset
            pointer = struct.unpack("<I", memory.read(store, pointer_offset, pointer_offset + 4))[0]
            if pointer == 0:
                raise HostProgramError(
                    f"the Host Program did not allocate {region.kind} {region.name!r} it is declared to use"
                )
            if pointer < region.data_offset:
                raise HostProgramError(f"{region.name!r} points before its declared storage region")
            candidate_base = pointer - region.data_offset
            if storage_base is None:
                storage_base = candidate_base
            elif candidate_base != storage_base:
                raise HostProgramError(
                    f"{region.name!r} does not live in the module's single declared storage array; overlapping "
                    "views are refused"
                )
            if pointer + region.nbytes > memory_len:
                raise HostProgramError(f"{region.name!r} extends past the module's linear memory")
            pointers[region.name] = pointer
        if storage_base is None:
            storage_base = 0
        if storage_base + self._storage_bytes > memory_len:
            raise HostProgramError("the declared storage array is larger than the module's linear memory")

        # Pairwise disjointness of the allocated views (input vs input, input vs workspace, ...).
        spans = [(region.name, pointers[region.name], pointers[region.name] + region.nbytes) for region in self.regions]
        ordered = sorted(spans, key=lambda item: item[1])
        for (left_name, _, left_end), (right_name, right_start, _) in zip(ordered, ordered[1:]):
            if left_end > right_start:
                raise HostProgramError(
                    f"Host Program views {left_name!r} and {right_name!r} overlap in linear memory; inputs and "
                    "workspace spans must be disjoint"
                )

        # Values: shapes and target, then the small mapped inputs.
        for field in spec.shape_fields:
            assert field.value is not None
            offset = base + layout.shape_offset(field.name)
            memory.write(store, field.packed(), offset)
        target_values = getattr(self, "_target_values", None) or {}
        for field in spec.target_fields:
            value = int(target_values.get(field.name, 0))
            memory.write(store, struct.pack("<I", value & 0xFFFFFFFF), base + self._target_offset(field.name))
        for view in spec.inputs:
            if not view.mapped:
                continue
            raw = (values or {}).get(view.name)
            if raw is None:
                raw = read(view.name, 0, view.numel * view.elem_nbytes)
            if len(raw) != view.numel * view.elem_nbytes:
                raise HostProgramError(
                    f"mapped input {view.name!r} supplied {len(raw)} bytes, declared "
                    f"{view.numel * view.elem_nbytes}"
                )
            memory.write(store, raw, pointers[view.name])
            reads.append(
                HostInputRead(
                    view.name,
                    view.tensor_id,
                    view.dtype,
                    0,
                    view.numel,
                    len(raw),
                    _sha(raw) if audit else None,
                )
            )

        arguments = [
            base + layout.inputs_offset,
            base + layout.shapes_offset,
            base + layout.target_offset,
            base + layout.workspaces_offset,
            base + layout.resolved_offset,
        ]
        status: int
        detail = ""
        try:
            status = int(exports["host_tiling"](store, *arguments))
        except Exception as exc:  # noqa: BLE001 - trap/fuel exhaustion is a candidate failure, not a crash
            remaining = 0
            try:
                remaining = int(store.get_fuel())
            except Exception:  # noqa: BLE001
                remaining = 0
            status = -1
            if remaining <= 0:
                detail = (
                    f"host_tiling exhausted its fuel budget of {self.build.fuel}; the planner did not terminate "
                    "within the declared capacity formula"
                )
            elif type(exc).__name__ == "Trap":
                detail = f"host_tiling trapped: {exc}"
            else:
                detail = f"host_tiling failed in the Wasm engine: {type(exc).__name__}: {exc}"
            wall_ns = time.perf_counter_ns() - started
            return HostRun(
                status=status,
                error_code="host_program_invalid",
                error_detail=detail,
                host_wall_ns=wall_ns,
                fuel_used=int(self.build.fuel - max(remaining, 0)),
                input_reads=tuple(reads),
                spans=(),
                resolved_scalars={},
                audit=bool(audit),
            )

        fuel_used = int(self.build.fuel - max(int(store.get_fuel()), 0))
        if status != 0:
            wall_ns = time.perf_counter_ns() - started
            return HostRun(
                status=status,
                error_code="host_program_invalid",
                error_detail=(
                    f"host_tiling returned {status}; every workspace span and resolved scalar from this call is "
                    "discarded (no partial commit)"
                ),
                host_wall_ns=wall_ns,
                fuel_used=fuel_used,
                input_reads=tuple(reads),
                spans=(),
                resolved_scalars={},
                audit=bool(audit),
            )

        collected: list[HostWorkspaceSpan] = []
        by_name = {view.name: view for view in spec.workspaces}
        for name, view in by_name.items():
            field = layout.workspace_field(name)
            offset, count = struct.unpack(
                "<II", memory.read(store, base + field.offset_offset, base + field.offset_offset + 8)
            )
            pointer = struct.unpack("<I", memory.read(store, base + field.ptr_offset, base + field.ptr_offset + 4))[0]
            if name not in self._regions:
                if offset != 0 or count != 0:
                    raise HostProgramError(
                        f"workspace {name!r} is not allocated for this program but the call recorded a span "
                        f"(offset={offset}, count={count}); a span requires storage and a declaring stage"
                    )
                continue
            if offset < 0 or count < 0 or offset + count > view.capacity:
                raise HostProgramError(
                    f"workspace {name!r} span offset={offset} count={count} exceeds its declared capacity "
                    f"{view.capacity}; the whole host run is discarded"
                )
            region = self._regions[name]
            start = pointer + offset * view.elem_nbytes
            end = start + count * view.elem_nbytes
            if end > pointer + region.nbytes or end > memory_len:
                raise HostProgramError(
                    f"workspace {name!r} span leaves its allocated region; the whole host run is discarded"
                )
            data = bytes(memory.read(store, start, end))
            collected.append(
                HostWorkspaceSpan(
                    name=name,
                    offset=int(offset),
                    count=int(count),
                    elem_nbytes=int(view.elem_nbytes),
                    data=data,
                    sha256=_sha(data) if audit else None,
                )
            )

        # Guard sentinels: an out-of-bounds planner write across regions is refused instead of
        # silently poisoning a neighbouring workspace that is copied to the device afterwards.
        guard = bytes([STORAGE_GUARD_PATTERN]) * STORAGE_GUARD_BYTES
        for region in self.regions:
            for offset in (region.guard_lo, region.guard_hi):
                actual = bytes(memory.read(store, storage_base + offset, storage_base + offset + STORAGE_GUARD_BYTES))
                if actual != guard:
                    raise HostProgramError(
                        f"workspace/input region {region.name!r} wrote into its guard sentinel; a Host Program "
                        "may only write the declared span of its own workspace"
                    )
        # Mapped read-only inputs must be unchanged: the host program has no write access to them.
        for view in spec.inputs:
            if not view.mapped:
                continue
            raw = (values or {}).get(view.name)
            if raw is None:
                raw = read(view.name, 0, view.numel * view.elem_nbytes)
            actual = bytes(memory.read(store, pointers[view.name], pointers[view.name] + len(raw)))
            if actual != raw:
                raise HostProgramError(
                    f"input {view.name!r} was modified by the Host Program; inputs are read-only"
                )
        wall_ns = time.perf_counter_ns() - started
        return HostRun(
            status=0,
            error_code=None,
            error_detail="",
            host_wall_ns=wall_ns,
            fuel_used=fuel_used,
            input_reads=tuple(reads),
            spans=tuple(collected),
            resolved_scalars={},
            audit=bool(audit),
        )

    def _target_offset(self, name: str) -> int:
        for field in self.layout.target_fields:
            if field.name == name:
                return field.offset
        raise KeyError(name)

    def set_target(self, values: Mapping[str, int]) -> None:
        """@brief 绑定 target profile 的公开整数参数（每次执行前由 backend 设置）。

        @param values target 值。
        """
        self._target_values = {str(key): int(value) for key, value in values.items()}


__all__ = [
    "HostInputRead",
    "HostRun",
    "HostRuntime",
    "HostWorkspaceSpan",
    "ImportReader",
]
