"""Native Host Program memory checks: bitwise inputs, harness-owned snapshots, bound-buffer guards.

CPU only: each case compiles a small self-contained dylib with the system Clang and calls it in this
process.  The tests pin three contracts that the header/runtime must not relax:

* read-only inputs are compared by *storage bytes*, so an unchanged NaN is not a false positive and
  ``-0.0 -> +0.0`` is not a false negative;
* the expected ABI/input snapshots are harness-owned Python buffers passed into the native
  capture/compare entry points, not candidate-writable dylib statics;
* guard verification inspects the *actually bound* store (the caller's buffer for a directly bound
  input), and completes before any workspace span is committed.

These are not an in-process memory sandbox: arbitrary native code still runs inside the external
worker, which remains the isolation/runaway boundary.  The outer Metal ``GuardedBuffer`` negatives
(``tests/backends/metal/test_tcenv.py::test_guard_bands_detect_out_of_bounds_write`` and the
``V2ArgSet`` safety checks) stay in place and are not replaced by the checks below.
"""

from __future__ import annotations

import ctypes
from functools import lru_cache
import struct
from types import SimpleNamespace

import pytest

from tcenv.hostprogram.contract import (
    HostContractSpec,
    HostField,
    HostInputView,
    HostWorkspaceView,
    host_contract_spec,
)
from tcenv.hostprogram.errors import HostCapabilityUnavailable, HostProgramError
from tcenv.hostprogram.native_abi import render_native_support
from tcenv.hostprogram.native_runtime import HostStorageGuard, NativeHostRuntime
from tcenv.hostprogram.native_toolchain import compile_native_host_program, discover_native_toolchain

SPEC = HostContractSpec(
    op_name="native_memory_probe",
    shape_fields=(HostField("N", "uint32_t", 4),),
    inputs=(
        HostInputView("x", "fp32", 4, 4, 1, True, "float"),
        HostInputView("y", "fp32", 2, 4, 2, True, "float"),
    ),
    workspaces=(HostWorkspaceView("lengths", "i32", 4, 4, "int32_t"),),
    target_fields=(),
)

NO_WRITES = r"""
int32_t host_tiling(const SfHostInputs *inputs, const SfHostShapes *shapes,
                    const SfHostTarget *target, SfHostWorkspaces *workspaces,
                    SfHostResolved *resolved) {
    (void)inputs; (void)shapes; (void)target; (void)workspaces; (void)resolved;
    return SF_HOST_OK;
}
"""


def _toolchain():
    try:
        return discover_native_toolchain()
    except HostCapabilityUnavailable as exc:  # pragma: no cover - depends on the host
        pytest.skip(f"native Host Program toolchain unavailable ({exc}); install Clang/Xcode CLT")


@lru_cache(maxsize=16)
def _build(source: str):
    return compile_native_host_program(
        source, spec=SPEC, entry_path="host/tiling.c", toolchain=_toolchain()
    )


def _runtime(source: str = NO_WRITES, *, isolate: str) -> NativeHostRuntime:
    """@brief 为单个用例编译一份独立 dylib 并建执行器。

    生产路径一个 worker 只建一个 ``NativeHostRuntime``；这里多个用例共用进程，所以给源码加
    一个注释标记，让每个用例拿到自己的 dylib，而不是复用上一用例绑过的 ABI 静态状态。
    """
    runtime = NativeHostRuntime(_build(f"{source}\n/* isolate: {isolate} */\n"), SPEC)
    runtime.set_target({})
    return runtime


def _reader(raw: dict[str, bytes]):
    return lambda name, byte_off, nbytes: raw[name][byte_off : byte_off + nbytes]


def _raw(*, nan: bool = False) -> dict[str, bytes]:
    """@brief 一份 fp32 输入；``nan=True`` 时第一个元素是 NaN（其余用例不用 NaN）。"""
    first = float("nan") if nan else 0.5
    return {
        "x": struct.pack("<4f", first, 1.0, -0.0, 3.0),
        "y": struct.pack("<2f", 2.0, 4.0),
    }


def _guarded(name: str, payload: bytes, *, guard: int = 64, pattern: int = 0xFF):
    """@brief 手工搭一块带两侧 guard 的存储，模拟 Metal ``GuardedBuffer``。

    @param name 视图名（写进 guard 描述）。
    @param payload 逻辑载荷字节。
    @param guard 单侧 guard 字节数。
    @param pattern guard 图案。
    @return ``(buffer, data_address, HostStorageGuard)``。
    """
    buffer = ctypes.create_string_buffer(bytes([pattern]) * guard + payload + bytes([pattern]) * guard)
    base = ctypes.addressof(buffer)
    data = base + guard
    return buffer, data, HostStorageGuard(
        name=name,
        data=data,
        lo=base,
        lo_nbytes=guard,
        hi=data + len(payload),
        hi_nbytes=guard,
        pattern=pattern,
    )


# ---------------------------------------------------------------------------------------------
# B1: read-only inputs are compared by bytes, not by C values
# ---------------------------------------------------------------------------------------------


def test_an_unchanged_nan_input_passes_the_read_only_check():
    """@brief NaN 原样不动（字节未变）时不得误报为「输入被改写」。"""
    runtime = _runtime(isolate="nan-unchanged")
    run = runtime.execute(_reader(_raw(nan=True)), values=None, audit=True)
    assert run.ok, run.error_detail
    assert run.spans == ()


def test_an_unchanged_nan_shape_field_passes_the_metadata_check():
    """@brief 浮点 shape/resolved 元数据同样按位比较；未修改的 NaN 不得被 `!=` 误拒。"""
    instance = SimpleNamespace(
        op_name="native_float_shape_probe",
        symbols={},
        resolved_attrs={"scale": float("nan")},
        tensors=(),
        variable_length_kind="fixed",
    )
    spec = host_contract_spec(instance=instance)
    assert spec.shape_fields[0].ctype == "float"
    build = compile_native_host_program(
        NO_WRITES + "\n/* isolate: float-shape */\n",
        spec=spec, entry_path="host/tiling.c", toolchain=_toolchain(),
    )
    runtime = NativeHostRuntime(build, spec)
    runtime.set_target({})
    run = runtime.execute(lambda name, offset, nbytes: b"", values=None, audit=True)
    assert run.ok, run.error_detail


def test_a_nan_payload_bit_flip_is_refused():
    """@brief NaN 载荷的任何位型变化都拒绝（不是靠 ``!=`` 碰巧报错）。"""
    source = NO_WRITES.replace(
        "    return SF_HOST_OK;",
        "    ((uint32_t *)(void *)inputs->x)[0] ^= 0x1u;\n    return SF_HOST_OK;",
    )
    runtime = _runtime(source, isolate="nan-payload")
    with pytest.raises(HostProgramError, match="input x was modified"):
        runtime.execute(_reader(_raw(nan=True)), values=None, audit=True)


def test_negative_zero_rewritten_to_positive_zero_is_refused():
    """@brief ``-0.0 -> +0.0`` 的浮点比较是相等，字节比较必须拒绝。"""
    source = NO_WRITES.replace(
        "    return SF_HOST_OK;",
        "    inputs->x[2] = 0.0f;\n    return SF_HOST_OK;",
    )
    runtime = _runtime(source, isolate="negative-zero")
    with pytest.raises(HostProgramError, match="input x was modified"):
        runtime.execute(_reader(_raw()), values=None, audit=True)


def test_an_ordinary_bit_pattern_rewrite_is_refused():
    """@brief 普通位型（同值不同位型也包含在内）被改写后整个 host run 丢弃。"""
    source = NO_WRITES.replace(
        "    return SF_HOST_OK;",
        "    ((uint32_t *)(void *)inputs->y)[1] = 0x3F800001u;\n    return SF_HOST_OK;",
    )
    runtime = _runtime(source, isolate="ordinary-bits")
    with pytest.raises(HostProgramError, match="input y was modified"):
        runtime.execute(_reader(_raw()), values=None, audit=True)


# ---------------------------------------------------------------------------------------------
# B2: expected snapshots are harness-owned, not dylib statics
# ---------------------------------------------------------------------------------------------


def test_the_generated_header_has_no_candidate_writable_snapshot_storage():
    """@brief 头文件里不再有 expected ABI/input 的 static 存储。"""
    header = render_native_support(SPEC)
    assert "sf_native_expected_input_" not in header
    assert "static SfHostAbi sf_native_expected" not in header
    # The harness-owned addresses are still passed, but as call arguments.
    assert "sf_native_capture_contract(SfHostAbi *expected)" in header
    assert "sf_native_capture_inputs(const uintptr_t *expected)" in header


def test_the_read_only_snapshot_lives_in_the_harness_buffer():
    """@brief 改 Python 侧的 expected 快照就能让未变的输入判失败，证明 C 比的是这块存储。"""
    runtime = _runtime(isolate="harness-snapshot")
    raw = _raw()
    assert isinstance(runtime.expected_input_buffers["x"], ctypes.Array)
    assert isinstance(runtime.expected_contract_buffer, ctypes.Array)
    assert runtime.execute(_reader(raw), values=None, audit=True).ok
    # The native capture wrote the bound bytes into *this* harness buffer.
    assert bytes(runtime.expected_input_buffers["x"]) == raw["x"]

    # Writing the harness snapshot after capture makes an unchanged input fail, which proves the
    # native compare reads the harness-owned address rather than a dylib static.
    snapshot = runtime.expected_input_buffers["x"]
    invoke = runtime._invoke

    def tampering_invoke() -> int:
        snapshot[0] = bytes([(snapshot[0][0] + 1) % 256])
        return invoke()

    runtime._invoke = tampering_invoke
    with pytest.raises(HostProgramError, match="input x was modified"):
        runtime.execute(_reader(raw), values=None, audit=True)


# ---------------------------------------------------------------------------------------------
# B3: guards are checked on the actually bound store, before any span commit
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "statement",
    [
        "inputs->x[4] = 1.0f;",
        "((uint8_t *)(void *)inputs->x)[-1] = 0u;",
    ],
)
def test_the_bound_input_guard_is_checked_before_a_span_is_committed(statement):
    """@brief 越界写落在实际绑定缓冲的 guard 上：host run 失败，且目标 workspace 一个字节都不写。"""
    source = NO_WRITES.replace(
        "    return SF_HOST_OK;",
        f"    {statement}\n"
        "    workspaces->lengths[0] = 7;\n"
        "    workspaces->lengths_offset = 0u;\n"
        "    workspaces->lengths_count = 1u;\n"
        "    return SF_HOST_OK;",
    )
    raw = _raw()
    buffer_x, data_x, guard_x = _guarded("x", raw["x"])
    buffer_y, data_y, guard_y = _guarded("y", raw["y"])
    destination = (ctypes.c_uint32 * 4)(*([0xCDCDCDCD] * 4))
    runtime = _runtime(source, isolate=f"bound-guard-{statement}")
    with pytest.raises(HostProgramError, match="guard sentinel"):
        runtime.execute(
            _reader(raw),
            values=None,
            mapped_addresses={"x": data_x, "y": data_y},
            workspace_addresses={"lengths": ctypes.addressof(destination)},
            storage_guards={"x": guard_x, "y": guard_y},
            audit=False,
        )
    assert list(destination) == [0xCDCDCDCD] * 4
    assert not runtime.last_commit
    assert buffer_x is not None and buffer_y is not None


def test_an_intact_bound_guard_still_commits_the_span():
    """@brief 对照：guard 完好时同一条路径正常提交，说明失败不是无条件拒绝。"""
    source = NO_WRITES.replace(
        "    return SF_HOST_OK;",
        "    workspaces->lengths[0] = 7;\n"
        "    workspaces->lengths_offset = 0u;\n"
        "    workspaces->lengths_count = 1u;\n"
        "    return SF_HOST_OK;",
    )
    raw = _raw()
    buffer_x, data_x, guard_x = _guarded("x", raw["x"])
    buffer_y, data_y, guard_y = _guarded("y", raw["y"])
    destination = (ctypes.c_uint32 * 4)(*([0xCDCDCDCD] * 4))
    runtime = _runtime(source, isolate="intact-guard")
    run = runtime.execute(
        _reader(raw),
        values=None,
        mapped_addresses={"x": data_x, "y": data_y},
        workspace_addresses={"lengths": ctypes.addressof(destination)},
        storage_guards={"x": guard_x, "y": guard_y},
        audit=False,
    )
    assert run.ok and runtime.last_commit
    assert list(destination)[:1] == [7]
    assert buffer_x is not None and buffer_y is not None


def test_a_directly_bound_input_without_guard_storage_is_refused():
    """@brief 直接绑定却拿不到真实 guard 时不回退去扫 static region，而是拒绝整次调用。"""
    raw = _raw()
    buffer_x, data_x, _guard_x = _guarded("x", raw["x"])
    buffer_y, data_y, _guard_y = _guarded("y", raw["y"])
    runtime = _runtime(isolate="no-guard-storage")
    with pytest.raises(HostProgramError, match="harness-owned guard storage"):
        runtime.execute(
            _reader(raw),
            values=None,
            mapped_addresses={"x": data_x, "y": data_y},
            audit=False,
        )
    assert buffer_x is not None and buffer_y is not None


def test_a_guard_for_a_different_store_than_the_bound_pointer_is_refused():
    """@brief guard 描述的数据地址与实际绑定指针不符时拒绝，避免检查错块存储。"""
    raw = _raw()
    buffer_x, data_x, guard_x = _guarded("x", raw["x"])
    buffer_y, data_y, guard_y = _guarded("y", raw["y"])
    runtime = _runtime(isolate="wrong-store")
    with pytest.raises(HostProgramError, match="guard check would inspect the wrong store"):
        runtime.execute(
            _reader(raw),
            values=None,
            mapped_addresses={"x": data_x, "y": data_y},
            storage_guards={
                "x": HostStorageGuard("x", data_x + 8, guard_x.lo, 64, guard_x.hi, 64, 0xFF),
                "y": HostStorageGuard("y", guard_y.data, guard_y.lo, guard_y.lo_nbytes, guard_y.hi, guard_y.hi_nbytes, 0xFF),
            },
            audit=False,
        )
    assert buffer_x is not None and buffer_y is not None
