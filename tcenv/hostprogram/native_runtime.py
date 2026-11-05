"""Worker-local native64 Host Program execution with the existing HostRun result semantics."""

from __future__ import annotations

import ctypes
import hashlib
import struct
import time
from typing import Callable, Mapping

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
    """@brief 复用 worker-local dylib；每次调用重置 span/guard 并复制 fresh mapped inputs。"""

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
        for view in spec.inputs:
            if view.mapped:
                self._address(f"sf_native_input_{view.name}_address")
                self._address(f"sf_native_input_{view.name}_abi_address")
                self._u32(f"sf_native_input_{view.name}_numel")
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
                    self._reads.append(HostInputRead(
                        name=view.name,
                        tensor_id=view.tensor_id,
                        dtype=view.dtype,
                        elem_off=int(elem_off),
                        count=int(count),
                        nbytes=nbytes,
                        sha256=_sha(data) if self._audit else None,
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

    def execute(self, read: ImportReader, *, values: Mapping[str, bytes] | None = None, audit: bool) -> HostRun:
        started = time.perf_counter_ns()
        self._read = read
        self._audit = bool(audit)
        self._reads = []
        try:
            self._before()
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
            snapshots: dict[str, bytes] = {}
            for view in self.spec.inputs:
                if not view.mapped:
                    continue
                raw = (values or {}).get(view.name)
                if raw is None:
                    raw = read(view.name, 0, view.numel * view.elem_nbytes)
                raw = bytes(raw)
                expected = view.numel * view.elem_nbytes
                if len(raw) != expected:
                    raise HostProgramError(
                        f"mapped input {view.name!r} supplied {len(raw)} bytes, declared {expected}"
                    )
                snapshots[view.name] = raw
                address = self._call_address(f"sf_native_input_{view.name}_address")
                ctypes.memmove(address, raw, len(raw))
                self._reads.append(HostInputRead(
                    view.name, view.tensor_id, view.dtype, 0, view.numel, len(raw),
                    _sha(raw) if audit else None, import_name=None,
                ))
            self._validate_views()
            self._validate_scalars()
            status = int(self._invoke())
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
            self._validate_views()
            self._validate_scalars()
            if int(self._guard()) != 0:
                raise HostProgramError("native Host Program wrote into a workspace/input guard sentinel")
            for view in self.spec.inputs:
                if not view.mapped:
                    continue
                address = self._call_address(f"sf_native_input_{view.name}_address")
                actual = ctypes.string_at(address, len(snapshots[view.name]))
                if actual != snapshots[view.name]:
                    raise HostProgramError(f"input {view.name!r} was modified by the native Host Program")
            spans: list[HostWorkspaceSpan] = []
            used = set(self.build.used_workspaces)
            for view in self.spec.workspaces:
                offset = self._call_u32(f"sf_native_workspace_{view.name}_offset")
                count = self._call_u32(f"sf_native_workspace_{view.name}_count")
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
                base = self._call_address(f"sf_native_workspace_{view.name}_address")
                data = ctypes.string_at(base + offset * view.elem_nbytes, count * view.elem_nbytes)
                spans.append(HostWorkspaceSpan(
                    name=view.name,
                    offset=offset,
                    count=count,
                    elem_nbytes=view.elem_nbytes,
                    data=data,
                    sha256=_sha(data) if audit else None,
                ))
            return HostRun(
                status=0,
                error_code=None,
                error_detail="",
                host_wall_ns=time.perf_counter_ns() - started,
                input_reads=tuple(self._reads),
                spans=tuple(spans),
                resolved_scalars={},
                audit=bool(audit),
            )
        finally:
            self._read = None
            self._reads = []


__all__ = ["NativeHostRuntime"]
