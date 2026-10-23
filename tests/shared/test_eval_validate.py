"""--timed --validate runs the instrumented pass first; plain --timed does not. No GPU, no model calls."""

from __future__ import annotations

import pytest

from tcenv import cli
from tcenv import evaluate as ev
from tcenv.evaluate import EvalReport

KERNEL = "kernel void relu_kernel(device const float* x [[buffer(0)]], device float* y [[buffer(1)]], uint i [[thread_position_in_grid]]) { y[i] = x[i]; }\n"


def _report(mode: str, status: str = "ok") -> EvalReport:
    rep = EvalReport(task="relu", dtype="fp16", size="small", mode=mode, status=status, stage="done")
    if status == "ok":
        rep.benchmark = {"candidate": {"median_us": 1.0, "cv": 0.01, "reps": 3}, "baselines": {}, "noisy": False}
    return rep


@pytest.fixture
def workers(monkeypatch):
    calls: list[tuple[str, bool]] = []
    state = {"validate_status": "ok"}

    def fake_run_worker(task, dtype, size, kp, mode, seed, reps, timeout, *, validation=False, **kw):
        calls.append((mode, validation))
        if mode == "validate":
            return _report("validate", state["validate_status"])
        return _report(mode)

    monkeypatch.setattr(ev, "_run_worker", fake_run_worker)
    return calls, state


def test_timed_alone_does_not_pay_for_the_instrumented_pass(workers):
    calls, _ = workers
    rep = ev.evaluate("relu", "fp16", "small", KERNEL, mode="timed", reps=3)
    assert calls == [("full", False)]
    assert rep.mode == "timed" and rep.status == "ok"


def test_timed_with_validate_runs_the_instrumented_pass_first(workers):
    calls, _ = workers
    rep = ev.evaluate("relu", "fp16", "small", KERNEL, mode="timed", reps=3, validate=True)
    assert calls == [("validate", True), ("full", False)]
    assert rep.mode == "timed" and rep.status == "ok"
    assert rep.safety.get("shader_validation") == "clean"


def test_a_failed_validation_stops_before_the_benchmark(workers):
    calls, state = workers
    state["validate_status"] = "unsafe"
    rep = ev.evaluate("relu", "fp16", "small", KERNEL, mode="timed", reps=3, validate=True)
    assert calls == [("validate", True)]  # 没有跑计时
    assert rep.status == "unsafe" and rep.mode == "timed"


def test_the_cli_passes_validate_through_and_refuses_it_with_quick(monkeypatch, tmp_path):
    seen = {}
    kernel = tmp_path / "kernel.metal"
    kernel.write_text(KERNEL)

    def fake_evaluate(task, dtype, size, src, **kw):
        seen.update(kw)
        return _report("timed")

    monkeypatch.setattr(ev, "evaluate", fake_evaluate)
    assert cli.main(["eval", "relu", str(kernel), "--dtype", "fp16", "--size", "small", "--timed", "--validate"]) == 0
    assert seen["mode"] == "timed" and seen["validate"] is True

    seen.clear()
    assert cli.main(["eval", "relu", str(kernel), "--dtype", "fp16", "--size", "small", "--quick", "--validate"]) == 2
    assert seen == {}
