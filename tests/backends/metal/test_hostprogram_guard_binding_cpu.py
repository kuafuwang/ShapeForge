"""CPU wiring: the Metal host binding passes each bound input's *real* guard storage to the runtime.

The native guard test in ``tests/shared/test_native_host_memory_guards.py`` proves the runtime checks
whatever descriptor it is given.  This node proves the production Metal binding actually gives it the
``GuardedBuffer`` margins of the directly bound input (not the dylib's static region), so removing the
wiring cannot leave both suites green.  No Metal context is created.
"""

from __future__ import annotations

import ctypes
from types import SimpleNamespace

from tcenv.backends.metal.host_program import MetalHostProgram
from tcenv.backends.metal.sanitize import GuardedBuffer
from tcenv.hostprogram.contract import HostContractSpec, HostField, HostInputView

SPEC = HostContractSpec(
    op_name="guard_binding_probe",
    shape_fields=(HostField("N", "uint32_t", 4),),
    inputs=(
        HostInputView("x", "fp32", 4, 4, 1, True, "float"),
        HostInputView("y", "fp32", 2, 4, 2, True, "float"),
    ),
    workspaces=(),
    target_fields=(),
)


class _FakeContext:
    """@brief GuardedBuffer 只调用 ``new_buffer``/``view``；CPU 测试用本地字节代替 Metal。"""

    def new_buffer(self, nbytes: int) -> bytearray:
        return bytearray(int(nbytes))

    def view(self, buffer: bytearray) -> memoryview:
        return memoryview(buffer)


def _guarded(payload: bytes, *, guard: int = 64, pattern: int = 0xFF) -> GuardedBuffer:
    buffer = GuardedBuffer(_FakeContext(), len(payload), guard=guard, pattern=pattern)
    buffer.write(payload)
    return buffer


class _RecordingRuntime:
    """@brief 记录 ``execute`` 收到的参数，不编译/加载任何 native 模块。"""

    last_profile: dict[str, int] = {}

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def execute(self, read, **kwargs):
        self.calls.append({"read": read, **kwargs})
        return SimpleNamespace(
            ok=True,
            spans=(),
            resolved_scalars={},
            host_wall_ns=1,
            input_reads=(),
        )


def _program() -> MetalHostProgram:
    return MetalHostProgram(
        entry=SimpleNamespace(path="host/tiling.c"),
        build=SimpleNamespace(used_workspaces=()),
        spec=SPEC,
        target_id="apple-m4-max",
        target_values={},
        native_digest=None,
        inside_timing=True,
        input_dependent=True,
        preprocesses_inputs=True,
        candidate_sha256="0" * 64,
    )


def test_storage_guards_describe_the_guarded_buffer_margins():
    program = _program()
    guarded_x = _guarded(b"\x00" * 16)
    guarded_y = _guarded(b"\x00" * 8)
    argset = SimpleNamespace(buffers={"x": guarded_x, "y": guarded_y})
    guards = program._storage_guards(argset)
    base = ctypes.addressof(ctypes.c_char.from_buffer(guarded_x.mem))
    assert guards["x"].data == base + guarded_x.guard
    assert guards["x"].lo == base
    assert guards["x"].lo_nbytes == guarded_x.guard
    assert guards["x"].hi == base + guarded_x.guard + guarded_x.nbytes
    # The tail covers the 256-byte alignment gap plus the 16 KiB margin, exactly what
    # ``GuardedBuffer.reset_guards`` fills.
    assert guards["x"].hi_nbytes == guarded_x.total - guarded_x.guard - guarded_x.nbytes
    assert guards["x"].hi_nbytes == guarded_x.guard + (guarded_x.padded - guarded_x.nbytes)
    assert guards["x"].pattern == guarded_x.pattern
    assert guards["y"].data == ctypes.addressof(ctypes.c_char.from_buffer(guarded_y.mem)) + guarded_y.guard


def test_execute_hands_the_bound_guard_storage_to_the_native_runtime():
    program = _program()
    recorder = _RecordingRuntime()
    program._runtime = recorder
    guarded_x = _guarded(b"\x00" * 16)
    guarded_y = _guarded(b"\x00" * 8)
    argset = SimpleNamespace(
        buffers={"x": guarded_x, "y": guarded_y},
        workspace_zero={},
        resolved_scalars={},
    )
    timing = program.execute(argset, audit=False)
    assert timing.native_wall_ns == 1
    assert recorder.calls
    storage = recorder.calls[-1]["storage_guards"]
    base = ctypes.addressof(ctypes.c_char.from_buffer(guarded_x.mem))
    assert storage["x"].lo == base and storage["x"].data == base + guarded_x.guard
    assert set(storage) == {"x", "y"}


def test_the_outer_guarded_buffer_negative_stays_independent_of_the_native_check():
    """@brief 外层 GuardedBuffer 的 guard 负例仍独立存在，不被 native 检查取代。"""
    guarded = _guarded(b"\x00" * 16)
    guarded.mem[guarded.guard + guarded.nbytes] = 0x00
    assert any("past the end" in item for item in guarded.guard_violations())

    guarded = _guarded(b"\x00" * 16)
    guarded.mem[guarded.guard - 1] = 0x00
    assert any("before the buffer start" in item for item in guarded.guard_violations())

    guarded = _guarded(b"\x00" * 16)
    assert guarded.guard_violations() == []
