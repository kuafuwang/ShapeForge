"""Native batch encoder eligibility and real Metal parity."""

from __future__ import annotations

import numpy as np
import pytest

from tcenv.backends.metal.native_batch import eligible, prepare_native_batch
from tcenv.backends.metal.runtime import BufferArg, ScalarArg, get_context


def test_native_batch_only_accepts_single_stage_buffer_only_no_fill():
    buffer = BufferArg(object())
    assert eligible(stages=[object()], arg_sets=[[[(0, buffer)]]], fills=[[]])
    # 多阶段：参数 set 的 stage 数不匹配时拒绝（不是「多阶段一律拒绝」）。
    assert not eligible(stages=[object(), object()], arg_sets=[[[(0, buffer)]]], fills=[[]])
    assert not eligible(stages=[object()], arg_sets=[[[(0, buffer)]]], fills=[[(object(), 0, 4)]])
    assert not eligible(stages=[object()], arg_sets=[[[(0, ScalarArg(b"x"))]]], fills=[[]])
    assert not eligible(stages=[object()], arg_sets=[[[(32, buffer)]]], fills=[[]])
    assert not eligible(stages=[object()], arg_sets=[[[]]], fills=[[]])


def test_native_batch_disable_opt_out_does_not_compile(monkeypatch):
    monkeypatch.setenv("TCENV_DISABLE_NATIVE_BATCH", "1")
    context = type("Context", (), {"queue": object(), "device": object()})()
    prepare_native_batch(context)
    assert getattr(context, "_native_batch", None) is None


def test_native_batch_accepts_multi_stage_buffer_only_no_fill():
    buffer = BufferArg(object())
    # 多阶段且每个 set 都有完整的各段参数：走原生编码器。
    assert eligible(
        stages=[object(), object()],
        arg_sets=[[[(0, buffer)], [(1, buffer)]]],
        fills=[[]],
    )
    # 缺一段参数 / 带 fill / 带动态 threadgroup 时仍退回 PyObjC。
    assert not eligible(stages=[object(), object()], arg_sets=[[[(0, buffer)]]], fills=[[]])
    assert not eligible(
        stages=[object(), object()],
        arg_sets=[[[(0, buffer)], [(1, buffer)]]],
        fills=[[(object(), 0, 4)]],
    )


@pytest.mark.gpu
def test_native_multi_stage_matches_the_python_encoder():
    prepare_native_batch(get_context())
    ctx = get_context()
    k1 = ctx.compile(
        "#include <metal_stdlib>\nusing namespace metal;\n"
        "kernel void k1(device const float* a [[buffer(0)]], device float* c [[buffer(2)]], uint i [[thread_position_in_grid]]) { c[i] = a[i] * 2.0f; }",
        "k1",
    )
    k2 = ctx.compile(
        "#include <metal_stdlib>\nusing namespace metal;\n"
        "kernel void k2(device const float* b [[buffer(1)]], device float* c [[buffer(2)]], uint i [[thread_position_in_grid]]) { c[i] += b[i]; }",
        "k2",
    )
    import numpy as np

    a = np.arange(256, dtype=np.float32)
    b = np.ones(256, dtype=np.float32)
    ab = ctx.device.newBufferWithLength_options_(a.nbytes, 0)
    bb = ctx.device.newBufferWithLength_options_(b.nbytes, 0)
    cb = ctx.device.newBufferWithLength_options_(a.nbytes, 0)
    ab.contents().as_buffer(a.nbytes)[:] = a.tobytes()
    bb.contents().as_buffer(b.nbytes)[:] = b.tobytes()
    args = [[(0, BufferArg(ab)), (2, BufferArg(cb))], [(1, BufferArg(bb)), (2, BufferArg(cb))]]
    native = ctx._native_batch
    before = native.calls
    ctx.run_pipeline([k1, k2], [(1, 1, 1), (1, 1, 1)], [(256, 1, 1), (256, 1, 1)], [0, 0], [args], [[]], iters=1)
    assert native.calls == before + 1
    out = np.frombuffer(cb.contents().as_buffer(a.nbytes), dtype=np.float32)
    assert np.allclose(out, a * 2.0 + b)


@pytest.mark.gpu
def test_native_batch_matches_the_python_encoder_and_preserves_hooks():
    prepare_native_batch(get_context())
    ctx = get_context()
    assert getattr(ctx, "_native_batch", None) is not None
    kernel = ctx.compile(
        "#include <metal_stdlib>\nusing namespace metal;\n"
        "kernel void k(device float* out [[buffer(0)]], uint i [[thread_position_in_grid]]) "
        "{ out[i] = 42.0f; }",
        "k",
    )
    buf = ctx.device.newBufferWithLength_options_(4, 0)
    args = [[(0, BufferArg(buf))]]
    native = ctx._native_batch
    before = native.calls
    hooks = []
    from tcenv.backends.metal.runtime import set_launch_hooks
    import tcenv.backends.metal.runtime as runtime_module

    previous_hooks = runtime_module._LAUNCH_HOOKS
    set_launch_hooks(lambda: hooks.append("begin"), lambda: hooks.append("end"))
    try:
        native_timing = ctx.run_pipeline([kernel], [(1, 1, 1)], [(1, 1, 1)], [0], [args], [[]], iters=4)
    finally:
        runtime_module._LAUNCH_HOOKS = previous_hooks
    native_value = np.frombuffer(buf.contents().as_buffer(4), dtype=np.float32)[0]
    assert native.calls == before + 1
    assert hooks == ["begin", "end"]
    assert native_value == 42.0
    assert native_timing.wall_s > 0 and native_timing.gpu_s >= 0

    buf.contents().as_buffer(4)[:] = b"\0\0\0\0"
    ctx._native_batch = None
    python_timing = ctx.run_pipeline([kernel], [(1, 1, 1)], [(1, 1, 1)], [0], [args], [[]], iters=4)
    python_value = np.frombuffer(buf.contents().as_buffer(4), dtype=np.float32)[0]
    assert python_value == native_value
    assert python_timing.wall_s > 0 and python_timing.gpu_s >= 0
    ctx._native_batch = native
