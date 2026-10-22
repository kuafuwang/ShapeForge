"""Candidate dispatch mode (serial vs concurrent) and the batch-size probe of the benchmark."""

from __future__ import annotations

import pytest
import torch

from tcenv.bench import choose_iters, interleaved


# --------------------------------------------------------------------------- CPU: the batch-size probe


def test_choose_iters_ignores_a_one_time_warmup_cost():
    """A compiled baseline's first call includes JIT compilation; that must not be mistaken for the time per launch."""
    calls = {"n": 0}

    def run_k(k: int) -> float:
        calls["n"] += 1
        return (1.0 if calls["n"] == 1 else 100e-6) * k  # first call: 1 s of compilation, afterwards 100 us per launch

    assert choose_iters(run_k, target_s=0.015, kmin=4) == 150


def test_choose_iters_still_uses_single_launches_for_slow_kernels():
    calls = []

    def run_k(k: int) -> float:
        calls.append(k)
        return 0.3 * k

    assert choose_iters(run_k) == 4 and max(calls) == 1  # kmin buffer sets, never a long probe batch


def test_interleaved_reports_every_runner():
    st = interleaved({"a": lambda k: 1e-3 * k, "b": lambda k: 2e-3 * k}, reps=4, warmup_reps=1)
    assert st["a"].median_s == pytest.approx(1e-3) and st["b"].median_s == pytest.approx(2e-3)


# --------------------------------------------------------------------------- GPU

gpu = pytest.mark.gpu


@gpu
def test_unknown_dispatch_mode_is_rejected():
    from tasks import get_task
    from tcenv.metal_runtime import get_context

    c = get_task("relu", "fp16", "small").contract
    ctx = get_context()
    src, _ = c.assemble(get_task("relu", "fp16", "small").naive_msl)
    k = ctx.compile(src, c.kernel_name)
    with pytest.raises(ValueError, match="dispatch"):
        ctx.run(k, [[]], c.grid, c.threads, dispatch="parallel")


@gpu
@pytest.mark.parametrize("task_name", ["relu", "gemv", "softmax"])
def test_concurrent_dispatch_over_rotating_sets_gives_the_right_outputs(task_name):
    from tasks import get_task
    from tcenv.baselines import rotation_count
    from tcenv.metal_runtime import get_context
    from tcenv.sanitize import ArgSet
    from tcenv.testing import check_case, prepare_case

    task = get_task(task_name, "fp16", "small")
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
    from tasks import get_task
    from tcenv.evaluate import evaluate

    rep = evaluate("relu", "fp16", "small", get_task("relu", "fp16", "small").naive_msl, mode="full", seed=2, reps=5, measure_serial=True)
    assert rep.status == "ok", rep.message
    b = rep.benchmark
    assert b["candidate_serial_us"] > 0 and "concurrent" in b["dispatch"]
    assert set(rep.baseline_us) == set(b["baselines"]) and "candidate_serial" not in rep.baseline_us
    assert rep.candidate_us == pytest.approx(b["candidate"]["median_us"])
