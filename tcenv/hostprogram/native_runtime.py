"""Worker-local native64 Host Program execution with the existing HostRun result semantics."""

from __future__ import annotations

import ctypes
import hashlib
import struct
import time
from typing import Any, Callable, Mapping

from .contract import INPUT_READ_IMPORTS, HostContractSpec
from .errors import HostProgramError
from .native_toolchain import NativeHostBuild
from .results import HostInputRead, HostRun, HostWorkspaceSpan

ImportReader = Callable[[str, int, int], bytes]
_IMPORT_FORMATS = {"i32": "<i", "u32": "<I", "fp32": "<f", "u16": "<H"}
_READ_CALLBACK = ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.c_int32, ctypes.c_uint32, ctypes.c_size_t, ctypes.c_uint32)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class NativeHostRuntime:
    """@brief 复用 worker-local dylib，绑定/复制 fresh inputs，并批量校验合同。

    ``audit=True`` 总是返回带 bytes/digest 的有效 spans。``audit=False`` 且提供 workspace
    地址时，C 先校验全部 span 再直接提交到调用方存储，``last_commit=True`` 且 spans 为空；
    这是避免 timed 中间对象的显式直接提交，不是空计划或失败。

    An L2 (capacity) instance additionally receives this call's *resolved* shape through
    ``SfHostResolved``.  The values are written by :meth:`execute` after the per-call reset and before
    the contract snapshot, so (a) each call sees its own total/count, (b) a candidate that rewrites them
    fails the C contract check, and (c) the values the candidate saw are returned in
    :attr:`HostRun.resolved_scalars` instead of being re-derived in Python.
    """

    def __init__(self, build: NativeHostBuild, spec: HostContractSpec) -> None:
        self.build = build
        self.spec = spec
        self._library = build.library or ctypes.CDLL(build.library_path)
        self._target_values: dict[str, int] = {}
        self._callbacks: list[object] = []
        self._reads: list[HostInputRead] = []
        self._read: ImportReader | None = None
        self._audit = False
        self._invoke = self._library.sf_native_invoke
        self._invoke.restype = ctypes.c_int32
        self._invoke.argtypes = []
        self._before = self._library.sf_native_before_call
        self._before.restype = None
        self._before.argtypes = []
        self._guard = self._library.sf_native_guard_status
        self._guard.restype = ctypes.c_int32
        self._guard.argtypes = []
        self._capture_contract = self._library.sf_native_capture_contract
        self._capture_contract.restype = None
        self._capture_contract.argtypes = []
        self._capture_inputs = self._library.sf_native_capture_inputs
        self._capture_inputs.restype = None
        self._capture_inputs.argtypes = []
        self._contract_error = self._library.sf_native_contract_error
        self._contract_error.restype = ctypes.c_char_p
        self._contract_error.argtypes = []
        self._bind_mapped = self._library.sf_native_bind_mapped_inputs
        self._bind_mapped.restype = ctypes.c_char_p
        self._bind_mapped.argtypes = [ctypes.POINTER(ctypes.c_size_t)]
        self._collect = self._library.sf_native_collect_spans
        self._collect.restype = None
        self._collect.argtypes = [
            ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_size_t)
        ]
        self._commit_spans = self._library.sf_native_commit_spans
        self._commit_spans.restype = ctypes.c_char_p
        self._commit_spans.argtypes = [ctypes.POINTER(ctypes.c_size_t)]
        self.last_commit = False
        workspace_count = len(spec.workspaces)
        self._span_offsets = (ctypes.c_uint32 * workspace_count)()
        self._span_counts = (ctypes.c_uint32 * workspace_count)()
        self._span_addresses = (ctypes.c_size_t * workspace_count)()
        self._contract_ready = False
        self.last_profile: dict[str, int] = {}
        for field in spec.shape_fields:
            function = getattr(self._library, f"sf_native_set_shape_{field.name}")
            function.argtypes = [ctypes.c_uint32]
            function.restype = None
            self._u32(f"sf_native_shape_{field.name}")
        for field in spec.target_fields:
            function = getattr(self._library, f"sf_native_set_target_{field.name}")
            function.argtypes = [ctypes.c_uint32]
            function.restype = None
            self._u32(f"sf_native_target_{field.name}")
        for field in spec.resolved_fields:
            setter = getattr(self._library, f"sf_native_set_resolved_{field.name}")
            setter.argtypes = [ctypes.c_uint32]
            setter.restype = None
            self._u32(f"sf_native_resolved_{field.name}")
        for view in spec.inputs:
            if view.mapped:
                self._address(f"sf_native_input_{view.name}_address")
                self._address(f"sf_native_input_{view.name}_abi_address")
                self._u32(f"sf_native_input_{view.name}_numel")
                binder = getattr(self._library, f"sf_native_bind_input_{view.name}")
                binder.argtypes = [ctypes.c_size_t]
                binder.restype = None
            else:
                for suffix in ("id", "numel", "elem_nbytes"):
                    self._u32(f"sf_native_input_{view.name}_{suffix}")
        for view in spec.workspaces:
            self._address(f"sf_native_workspace_{view.name}_address")
            self._address(f"sf_native_workspace_{view.name}_abi_address")
            for suffix in ("capacity", "offset", "count", "elem_nbytes"):
                self._u32(f"sf_native_workspace_{view.name}_{suffix}")
        self._install_read_callbacks()
        self._validate_views()

    def _address(self, name: str) -> None:
        function = getattr(self._library, name)
        function.restype = ctypes.c_size_t
        function.argtypes = []

    def _u32(self, name: str) -> None:
        function = getattr(self._library, name)
        function.restype = ctypes.c_uint32
        function.argtypes = []

    def _call_address(self, name: str) -> int:
        return int(getattr(self._library, name)())

    def _call_u32(self, name: str) -> int:
        return int(getattr(self._library, name)())

    def set_target(self, values: Mapping[str, int]) -> None:
        self._target_values = {str(key): int(value) for key, value in values.items()}
        self._contract_ready = False

    def _ensure_contract(self) -> None:
        if self._contract_ready:
            return
        for field in self.spec.shape_fields:
            if field.value is None:
                raise HostProgramError(f"native shape field {field.name!r} has no value")
            getattr(self._library, f"sf_native_set_shape_{field.name}")(
                self._scalar_bits(field.value, field.ctype)
            )
        for field in self.spec.target_fields:
            getattr(self._library, f"sf_native_set_target_{field.name}")(
                int(self._target_values.get(field.name, 0)) & 0xFFFFFFFF
            )
        self._capture_contract()
        self._contract_ready = True

    def _install_read_callbacks(self) -> None:
        for index, (import_name, dtype) in enumerate(INPUT_READ_IMPORTS):
            elem_nbytes = struct.calcsize(_IMPORT_FORMATS[dtype])

            def callback(tensor_id: int, elem_off: int, dst: int, count: int, *, _dtype=dtype, _elem=elem_nbytes, _import=import_name) -> int:
                try:
                    view = next((item for item in self.spec.inputs if item.tensor_id == int(tensor_id)), None)
                    if view is None:
                        return 1
                    if _dtype == "u16":
                        if view.elem_nbytes != 2:
                            return 1
                    elif view.dtype != _dtype:
                        return 1
                    if int(elem_off) + int(count) > view.numel:
                        return 2
                    if self._read is None:
                        return 2
                    nbytes = int(count) * _elem
                    data = self._read(view.name, int(elem_off) * _elem, nbytes)
                    if len(data) != nbytes:
                        return 2
                    # The destination may be candidate stack or a declared workspace.  Native
                    # pointers never leave the worker; a bad native address is contained by the
                    # external worker/watchdog just like any other candidate memory fault.
                    ctypes.memmove(int(dst), data, nbytes)
                    if self._audit:
                        self._reads.append(HostInputRead(
                            name=view.name,
                            tensor_id=view.tensor_id,
                            dtype=view.dtype,
                            elem_off=int(elem_off),
                            count=int(count),
                            nbytes=nbytes,
                            sha256=_sha(data),
                            import_name=_import,
                        ))
                    return 0
                except Exception:
                    return 2

            callback_type = _READ_CALLBACK(callback)
            self._callbacks.append(callback_type)
            setter = getattr(self._library, f"sf_native_set_read_{index}")
            setter.argtypes = [_READ_CALLBACK]
            setter.restype = None
            setter(callback_type)

    def _validate_views(self) -> None:
        for view in self.spec.inputs:
            if not view.mapped:
                for suffix in ("id", "numel", "elem_nbytes"):
                    expected = getattr(view, suffix if suffix != "id" else "tensor_id")
                    if self._call_u32(f"sf_native_input_{view.name}_{suffix}") != expected:
                        raise HostProgramError(f"native metadata input {view.name!r} {suffix} changed")
                continue
            actual = self._call_address(f"sf_native_input_{view.name}_address")
            abi = self._call_address(f"sf_native_input_{view.name}_abi_address")
            if actual == 0 or abi != actual:
                raise HostProgramError(f"native input {view.name!r} was repointed outside its declared region")
            if self._call_u32(f"sf_native_input_{view.name}_numel") != view.numel:
                raise HostProgramError(f"native input {view.name!r} numel metadata changed")
        used = set(self.build.used_workspaces)
        for view in self.spec.workspaces:
            actual = self._call_address(f"sf_native_workspace_{view.name}_address")
            abi = self._call_address(f"sf_native_workspace_{view.name}_abi_address")
            if view.name in used:
                if actual == 0 or abi != actual:
                    raise HostProgramError(f"native workspace {view.name!r} was repointed outside its declared region")
            elif actual != 0 or abi != 0:
                raise HostProgramError(f"unallocated native workspace {view.name!r} acquired a pointer")
            if self._call_u32(f"sf_native_workspace_{view.name}_capacity") != view.capacity:
                raise HostProgramError(f"native workspace {view.name!r} capacity metadata changed")
            if self._call_u32(f"sf_native_workspace_{view.name}_elem_nbytes") != view.elem_nbytes:
                raise HostProgramError(f"native workspace {view.name!r} elem_nbytes metadata changed")

    def _validate_scalars(self) -> None:
        for field in self.spec.shape_fields:
            if field.value is None:
                raise HostProgramError(f"native shape field {field.name!r} has no value")
            expected = self._scalar_bits(field.value, field.ctype)
            if self._call_u32(f"sf_native_shape_{field.name}") != expected:
                raise HostProgramError(f"native shape field {field.name!r} changed during host_tiling")
        for field in self.spec.target_fields:
            expected = int(self._target_values.get(field.name, 0)) & 0xFFFFFFFF
            if self._call_u32(f"sf_native_target_{field.name}") != expected:
                raise HostProgramError(f"native target field {field.name!r} changed during host_tiling")

    @staticmethod
    def _scalar_bits(value: int | float, ctype: str) -> int:
        if ctype == "float":
            return int.from_bytes(struct.pack("<f", float(value)), "little")
        return int(value) & 0xFFFFFFFF

    def _write_resolved(self, values: Mapping[str, int] | None) -> dict[str, int]:
        """@brief 写入本次调用的 runtime resolved scalar，并返回写入值。

        A fixed/L1 contract declares no resolved field, so a caller that supplies values is refused
        instead of being silently ignored (that would let an L2-shaped caller believe an L1 build saw a
        runtime total).  Every declared field must be present and non-negative.

        @param values resolved scalar 映射；省略时为空。
        @return 实际写入的值。
        @raises HostProgramError 字段集/取值非法。
        """
        supplied = {str(key): int(value) for key, value in dict(values or {}).items()}
        declared = {field.name for field in self.spec.resolved_fields}
        if not declared:
            if supplied:
                raise HostProgramError(
                    f"native Host Program contract declares no runtime resolved scalars, got {sorted(supplied)}"
                )
            return {}
        missing = sorted(declared - set(supplied))
        extra = sorted(set(supplied) - declared)
        if missing or extra:
            raise HostProgramError(
                f"native resolved scalars do not match the contract; missing={missing} extra={extra}"
            )
        written: dict[str, int] = {}
        for field in self.spec.resolved_fields:
            value = int(supplied[field.name])
            if value < 0:
                raise HostProgramError(f"native resolved scalar {field.name!r} must be non-negative, got {value}")
            getattr(self._library, f"sf_native_set_resolved_{field.name}")(self._scalar_bits(value, field.ctype))
            written[field.name] = value
        return written

    def _read_resolved(self, written: Mapping[str, int]) -> dict[str, int]:
        """@brief 调用后回读 runtime resolved scalar，并与写入值逐一核对。

        The C contract check already refuses a candidate that rewrote a resolved field; this read-back
        is the Python-side record of *what the candidate actually saw*, so the evidence is derived from
        the ABI rather than from the caller's expectation.

        @param written 调用前写入的值。
        @return 回读的字段值（固定/L1 契约为空字典）。
        @raises HostProgramError 回读值与写入值不一致。
        """
        observed: dict[str, int] = {}
        for field in self.spec.resolved_fields:
            observed[field.name] = int(self._call_u32(f"sf_native_resolved_{field.name}"))
        if dict(observed) != dict(written):
            raise HostProgramError(
                f"native resolved scalars are not the ones this call supplied: wrote {dict(written)}, "
                f"read back {observed}; the whole run is discarded"
            )
        return observed

    def execute(
        self,
        read: ImportReader,
        *,
        values: Mapping[str, bytes] | None = None,
        mapped_addresses: Mapping[str, int] | None = None,
        workspace_addresses: Mapping[str, int] | None = None,
        mapped_address_array: Any = None,
        workspace_address_array: Any = None,
        resolved: Mapping[str, int] | None = None,
        audit: bool,
        profile: bool = False,
    ) -> HostRun:
        started = time.perf_counter_ns()
        self._read = read
        self._audit = bool(audit)
        self._reads = []
        self.last_profile = {}
        self.last_commit = False
        try:
            self._ensure_contract()
            self._before()
            resolved_values = self._write_resolved(resolved)
            if profile:
                self.last_profile["setup"] = time.perf_counter_ns() - started
            input_started = time.perf_counter_ns()
            direct = mapped_addresses is not None or mapped_address_array is not None
            if audit and direct and mapped_addresses is None:
                raise HostProgramError("native audit requires the mapped input address dictionary")
            if direct:
                mapped_views = [view for view in self.spec.inputs if view.mapped]
                packed = mapped_address_array
                if packed is None:
                    try:
                        packed = (ctypes.c_size_t * len(mapped_views))(
                            *(int(mapped_addresses[view.name]) for view in mapped_views)
                        )
                    except (KeyError, TypeError, ValueError) as exc:
                        raise HostProgramError(f"incomplete native input address binding: {exc}") from exc
                error = self._bind_mapped(packed)
                if error:
                    raise HostProgramError(error.decode("utf-8", errors="replace"))
            for view in self.spec.inputs:
                if not view.mapped:
                    continue
                nbytes = view.numel * view.elem_nbytes
                raw = None if direct else (values or {}).get(view.name)
                if not direct and raw is None:
                    raw = read(view.name, 0, nbytes)
                if raw is not None:
                    raw = bytes(raw)
                    if len(raw) != nbytes:
                        raise HostProgramError(
                            f"mapped input {view.name!r} supplied {len(raw)} bytes, declared {nbytes}"
                        )
                if direct:
                    address = int(mapped_addresses[view.name])
                else:
                    address = self._call_address(f"sf_native_input_{view.name}_address")
                    getattr(self._library, f"sf_native_bind_input_{view.name}")(address)
                    ctypes.memmove(address, raw, len(raw))
                if audit:
                    if raw is None:
                        raw = ctypes.string_at(address, nbytes)
                    self._reads.append(HostInputRead(
                        view.name, view.tensor_id, view.dtype, 0, view.numel, nbytes,
                        _sha(raw), import_name=None,
                    ))
            if profile:
                self.last_profile["input_prepare"] = time.perf_counter_ns() - input_started
            capture_started = time.perf_counter_ns()
            self._capture_contract()
            self._capture_inputs()
            if profile:
                self.last_profile["capture"] = time.perf_counter_ns() - capture_started
            invoke_started = time.perf_counter_ns()
            status = int(self._invoke())
            if profile:
                self.last_profile["planner"] = time.perf_counter_ns() - invoke_started
            if status != 0:
                return HostRun(
                    status=status,
                    error_code="host_program_invalid",
                    error_detail=(
                        f"host_tiling returned {status}; every native workspace span and resolved scalar "
                        "from this call is discarded (no partial commit)"
                    ),
                    host_wall_ns=time.perf_counter_ns() - started,
                    input_reads=tuple(self._reads),
                    spans=(),
                    resolved_scalars={},
                    audit=bool(audit),
                )
            validate_started = time.perf_counter_ns()
            error = self._contract_error()
            if error:
                raise HostProgramError(error.decode("utf-8", errors="replace"))
            published = self._read_resolved(resolved_values)
            if profile:
                self.last_profile["validate"] = time.perf_counter_ns() - validate_started
            collect_started = time.perf_counter_ns()
            if not audit and (workspace_addresses is not None or workspace_address_array is not None):
                packed_outputs = workspace_address_array
                if packed_outputs is None:
                    packed_outputs = (ctypes.c_size_t * len(self.spec.workspaces))(
                        *(int(workspace_addresses.get(view.name, 0)) for view in self.spec.workspaces)
                    )
                error = self._commit_spans(packed_outputs)
                if error:
                    raise HostProgramError(error.decode("utf-8", errors="replace"))
                self.last_commit = True
                if profile:
                    self.last_profile["collect"] = time.perf_counter_ns() - collect_started
                return HostRun(
                    status=0,
                    error_code=None,
                    error_detail="",
                    host_wall_ns=time.perf_counter_ns() - started,
                    input_reads=tuple(self._reads),
                    spans=(),
                    resolved_scalars=published,
                    audit=False,
                )
            self._collect(self._span_offsets, self._span_counts, self._span_addresses)
            spans: list[HostWorkspaceSpan] = []
            used = set(self.build.used_workspaces)
            for index, view in enumerate(self.spec.workspaces):
                offset = int(self._span_offsets[index])
                count = int(self._span_counts[index])
                if view.name not in used:
                    if offset != 0 or count != 0:
                        raise HostProgramError(
                            f"unallocated workspace {view.name!r} recorded a non-empty native span"
                        )
                    continue
                if offset + count > view.capacity:
                    raise HostProgramError(
                        f"workspace {view.name!r} native span offset={offset} count={count} exceeds "
                        f"its declared capacity {view.capacity}; the whole run is discarded"
                    )
                data = ctypes.string_at(
                    int(self._span_addresses[index]) + offset * view.elem_nbytes,
                    count * view.elem_nbytes,
                )
                spans.append(HostWorkspaceSpan(
                    name=view.name,
                    offset=offset,
                    count=count,
                    elem_nbytes=view.elem_nbytes,
                    data=data,
                    sha256=_sha(data) if audit else None,
                ))
            if profile:
                self.last_profile["collect"] = time.perf_counter_ns() - collect_started
            return HostRun(
                status=0,
                error_code=None,
                error_detail="",
                host_wall_ns=time.perf_counter_ns() - started,
                input_reads=tuple(self._reads),
                spans=tuple(spans),
                resolved_scalars=published,
                audit=bool(audit),
            )
        finally:
            self._read = None
            self._reads = []


__all__ = ["NativeHostRuntime"]
