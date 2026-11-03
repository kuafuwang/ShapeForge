"""Candidate dispatch mode (serial vs concurrent) on a real Metal device.

The CPU half of the old ``test_dispatch.py`` (the batch-size probe and the interleaved
timing loop) now lives in ``tests/shared/test_bench.py``.
"""

from __future__ import annotations

import pytest

from ...support.v1_tasks import V1_TASK, v1_task

# --------------------------------------------------------------------------- GPU

gpu = pytest.mark.gpu


@gpu
def test_unknown_dispatch_mode_is_rejected():
    """@brief 验证真实 Metal 设备上未知的 dispatch 模式被拒绝，异常信息提到 dispatch。
    """
    from tcenv.backends.metal.runtime import get_context

    task = v1_task()
    c = task.contract
    ctx = get_context()
    src, _ = c.assemble(task.naive_msl)
    k = ctx.compile(src, c.kernel_name)
    with pytest.raises(ValueError, match="dispatch"):
        ctx.run(k, [[]], c.grid, c.threads, dispatch="parallel")


@gpu
@pytest.mark.parametrize("task_name", ["lion", "gemv", "conv2d"])
def test_concurrent_dispatch_over_rotating_sets_gives_the_right_outputs(task_name):
    """@brief 验证并发派发跨多组轮转缓冲区、跑过不止一遍时输出仍然正确，且各组没有安全违规。

    A5 retired the migrated relu/softmax assemblies; the same three families (elementwise,
    rowwise, compute) are covered by the preserved v1 tasks lion, gemv and conv2d.

    @param task_name 参数化的任务名（lion、gemv、conv2d）。
    """
    from tcenv.backends.metal.baselines import rotation_count
    from tcenv.backends.metal.runtime import get_context
    from tcenv.backends.metal.sanitize import ArgSet
    from tcenv.backends.metal.testing import check_case, prepare_case

    task = v1_task(task_name)
    c = task.contract
    ctx = get_context()
    src, _ = c.assemble(task.naive_msl)
    k = ctx.compile(src, c.kernel_name)
    data = prepare_case(task, "normal", 4)
    R = max(3, rotation_count(task))
    sets = [ArgSet(ctx, c) for _ in range(R)]
    for s in sets:
        s.load_inputs(data.raw)
        s.poison_outputs()
    ctx.run(k, [s.args() for s in sets], c.grid, c.threads, iters=3 * R + 1, dispatch="concurrent")  # more than one pass: barriers are exercised
    for s in sets:
        assert not s.safety_violations()
        assert check_case(task, data, s.read_outputs(), None).ok


@gpu
def test_the_benchmark_reports_both_dispatch_modes_and_selects_on_the_concurrent_one():
    """@brief 验证基准同时报告串行与并发两种派发，并以并发结果作为候选耗时，基线里不含串行候选。
    """
    from tcenv.evaluate import evaluate

    task = v1_task()
    rep = evaluate(V1_TASK, "fp16", "small", task.naive_msl, mode="full", seed=2, reps=5, measure_serial=True)
    assert rep.status == "ok", rep.message
    b = rep.benchmark
    assert b["candidate_serial_us"] > 0 and "concurrent" in b["dispatch"]
    assert set(rep.baseline_us) == set(b["baselines"]) and "candidate_serial" not in rep.baseline_us
    assert rep.candidate_us == pytest.approx(b["candidate"]["median_us"])
