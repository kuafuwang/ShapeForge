"""Native batch encoder eligibility and real Metal parity."""

from __future__ import annotations

import numpy as np
import pytest

from tcenv.backends.metal.native_batch import eligible, prepare_native_batch
from tcenv.backends.metal.runtime import BufferArg, ScalarArg, get_context


def test_native_batch_only_accepts_single_stage_buffer_only_no_fill():
    buffer = BufferArg(object())
    assert eligible(stages=[object()], arg_sets=[[[(0, buffer)]]], fills=[[]])
    assert not eligible(stages=[object(), object()], arg_sets=[[[(0, buffer)]]], fills=[[]])
    assert not eligible(stages=[object()], arg_sets=[[[(0, buffer)]]], fills=[[(object(), 0, 4)]])
    assert not eligible(stages=[object()], arg_sets=[[[(0, ScalarArg(b"x"))]]], fills=[[]])
    assert not eligible(stages=[object()], arg_sets=[[[(32, buffer)]]], fills=[[]])
    assert not eligible(stages=[object()], arg_sets=[[[]]], fills=[[]])


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
