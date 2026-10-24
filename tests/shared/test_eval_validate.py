"""--timed --validate runs the instrumented pass first; plain --timed does not. No GPU, no model calls."""

from __future__ import annotations

import pytest

from tcenv import cli
from tcenv import evaluate as ev
from tcenv.evaluate import EvalReport

KERNEL = "kernel void relu_kernel(device const float* x [[buffer(0)]], device float* y [[buffer(1)]], uint i [[thread_position_in_grid]]) { y[i] = x[i]; }\n"


def _report(mode: str, status: str = "ok") -> EvalReport:
    """@brief 造一份评测报告，``ok`` 时补上计时字段。

    @param mode 评测模式，例如 ``timed`` / ``validate`` / ``full``。
    @param status 报告状态；非 ``ok`` 时不写基准数据。
    @return 报告对象。
    """
    rep = EvalReport(task="relu", dtype="fp16", size="small", mode=mode, status=status, stage="done")
    if status == "ok":
        rep.benchmark = {"candidate": {"median_us": 1.0, "cv": 0.01, "reps": 3}, "baselines": {}, "noisy": False}
    return rep


@pytest.fixture
def workers(monkeypatch):
    """@brief 用假的 worker 替掉真实执行，记录每次调用的模式与是否插桩。

    @param monkeypatch pytest 的替换夹具。
    @return ``(calls, state)``：调用记录列表，以及可改写 ``validate_status`` 的状态字典。
    """
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
    """@brief 验证只给 ``--timed`` 时不额外跑插桩那一遍，只调用一次 full。

    @param workers 假 worker 夹具，提供调用记录与校验状态。
    """
    calls, _ = workers
    rep = ev.evaluate("relu", "fp16", "small", KERNEL, mode="timed", reps=3)
    assert calls == [("full", False)]
    assert rep.mode == "timed" and rep.status == "ok"


def test_timed_with_validate_runs_the_instrumented_pass_first(workers):
    """@brief 验证 ``--timed --validate`` 先跑插桩校验再跑 full，并把 shader 校验结论写进 safety。

    @param workers 假 worker 夹具，提供调用记录与校验状态。
    """
    calls, _ = workers
    rep = ev.evaluate("relu", "fp16", "small", KERNEL, mode="timed", reps=3, validate=True)
    assert calls == [("validate", True), ("full", False)]
    assert rep.mode == "timed" and rep.status == "ok"
    assert rep.safety.get("shader_validation") == "clean"


def test_a_failed_validation_stops_before_the_benchmark(workers):
    """@brief 验证校验不通过时在计时之前就停住，报告状态沿用 unsafe。

    @param workers 假 worker 夹具；本用例把 ``validate_status`` 改成 unsafe。
    """
    calls, state = workers
    state["validate_status"] = "unsafe"
    rep = ev.evaluate("relu", "fp16", "small", KERNEL, mode="timed", reps=3, validate=True)
    assert calls == [("validate", True)]  # 没有跑计时
    assert rep.status == "unsafe" and rep.mode == "timed"


def test_the_cli_passes_validate_through_and_refuses_it_with_quick(monkeypatch, tmp_path):
    """@brief 验证 CLI 把 ``--validate`` 透传给 evaluate，而 ``--quick`` 与它互斥并直接以退出码 2 拒绝。

    @param monkeypatch 用假 evaluate 记录关键字参数。
    @param tmp_path 临时目录，用来放待评测的内核文件。
    """
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
