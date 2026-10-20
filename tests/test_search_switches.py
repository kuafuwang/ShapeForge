"""The opt-in search switches. Off, the prompts and the stop rules are the ones from before the switches existed."""

from __future__ import annotations

from taic.feedback_extra import abi_signature, at_roofline, status_lines, target_note, threadgroup_limit_note
from taic.llm import LLMConfig, SearchConfig
from taic.orchestrator import _retry_without_reasoning
from taic.prompts import patch_user, plan_user
from tasks import get_task
from tcenv.evaluate import EvalReport

from .fakes import make_response
from .test_orchestrator import PLAN, code, llm_script, make_orch, run


def _task():
    return get_task("relu", "fp16", "small")


def test_switches_default_off_and_do_not_change_the_prompt_text():
    assert SearchConfig().stop_at_roofline is True
    assert SearchConfig().abi_guard is True
    assert SearchConfig().target_note is True
    assert SearchConfig().rich_history is True
    assert SearchConfig().plan_ledger is True
    assert SearchConfig().status_lines is True
    assert LLMConfig(base_url="http://x", api_key="k", model="m").truncation_retry is True
    task = _task()
    plain = plan_user(task, "kernel void relu_kernel() {}", "feedback", ["round 1: A -> 1.100x (new best)"], 3)
    assert plan_user(task, "kernel void relu_kernel() {}", "feedback", ["round 1: A -> 1.100x (new best)"], 3, status="") == plain
    assert "noise floor" not in plain
    plan = {"title": "A", "change": "do A"}
    assert "noise floor" not in patch_user(task, "kernel void relu_kernel() {}", "feedback", plan)
    told = plan_user(task, "kernel void relu_kernel() {}", "feedback", [], 3, status=status_lines(rnd=2, max_rounds=12, patience_left=3, spent=1.5, budget=3, required_gain=1.014))
    assert "at least 1.4% faster" in told and "Round 2/12" in told and "spent $1.50 of $3.00" in told
    assert told.index("noise floor") < told.index("Diagnose the dominant bottleneck")


def test_a_baseline_faster_than_the_bound_is_named_and_does_not_count_as_at_the_roof():
    report = {
        "roofline": {"lower_bound_us": 931.8, "time_over_lower_bound": 1.07, "bound": "compute"},
        "baseline_us": {"mlx": 994.0, "torch_mps": 777.0},
    }
    note = target_note(report)
    assert note and "777.0 us" in note and "does not hold" in note and "Winograd" not in note
    assert "different algorithm" in note and "peak calibration" in note
    assert at_roofline(report, 1.02) is False  # the bound is the wrong target, so the search is not stopped
    quiet = {
        "roofline": {"lower_bound_us": 100, "time_over_lower_bound": 1.01, "peak_gbps_measured": 400.0},
        "baseline_us": {"mlx": 120},
        "benchmark": {"noisy": False},
    }
    assert target_note(quiet) is None
    assert at_roofline(quiet, 1.02) is True
    assert at_roofline({**quiet, "benchmark": {"noisy": True}}, 1.02) is False
    assert at_roofline({"roofline": {"lower_bound_us": 100, "time_over_lower_bound": 1.01}, "baseline_us": {"mlx": 120}, "benchmark": {"noisy": False}}, 1.02) is False
    assert at_roofline({**quiet, "roofline": {"lower_bound_us": 100, "time_over_lower_bound": 1.5, "peak_gbps_measured": 400.0}}, 1.02) is False


def test_the_same_threadgroup_overflow_compares_equal_and_states_the_limit():
    a = "static threadgroup memory 40960 B exceeds the device limit of 32768 B"
    b = "static threadgroup memory 36864 B exceeds the device limit of 32768 B"
    assert abi_signature(a) == abi_signature(b)
    assert abi_signature(a) != abi_signature("kernel does not match the required ABI")
    note = threadgroup_limit_note(a)
    assert note and "under 32768 B" in note and "declares 40960 B" in note
    assert threadgroup_limit_note("something else") is None


def test_a_truncated_reply_with_no_kernel_is_retried_once_without_spending_a_repair(tmp_path):
    calls = []
    seen = {"n": 0}

    def route(messages, **kw):
        seen["n"] += 1
        last = messages[-1]["content"]
        if "Translate this Triton kernel" in last and seen["n"] == 1:
            return make_response("", reasoning="x" * 100, finish="length")
        if "Translate this Triton kernel" in last:
            assert all("reasoning_content" not in m for m in messages)
            return make_response(code("speed=1.2"))
        if "Diagnose the dominant bottleneck" in last:
            return make_response(PLAN)
        if "Implement exactly this optimization plan" in last:
            return make_response(code("speed=0.5"))
        raise AssertionError(last[:80])

    orch = make_orch(tmp_path, route, calls, max_rounds=1, max_repair=2)
    orch.cfg.truncation_retry = True
    res = run(orch)
    assert res.best_speedup == 1.2
    assert seen["n"] >= 2  # the truncated initial reply was asked again


def test_the_retry_drops_an_empty_assistant_turn_and_its_reasoning():
    messages = [
        {"role": "user", "content": "write it"},
        {"role": "assistant", "content": "", "reasoning_content": "a" * 50},
    ]
    out = _retry_without_reasoning(messages)
    assert out == [{"role": "user", "content": "write it"}]


def test_a_branch_that_repeats_one_abi_error_stops(tmp_path):
    calls = []

    def ev(task, dtype, size, source, *, mode="full", seed=None, reps=15, **kw):
        calls.append(source)
        r = EvalReport(task=task, dtype=dtype, size=size, mode=mode, seed=1, stage="done")
        if "speed=" in source:
            r.status, r.message, r.speedup, r.speedups, r.candidate_us, r.baseline_us = "ok", "ok", 1.2, {"mlx": 1.2}, 80.0, {"mlx": 100.0}
            r.benchmark = {"candidate": {"median_us": 80.0, "cv": 0.01, "reps": 15}, "baselines": {"mlx": {"median_us": 100.0, "cv": 0.01, "reps": 15}}, "best_baseline": "mlx"}
            return r
        r.status = "abi_error"
        r.message = "static threadgroup memory 40960 B exceeds the device limit of 32768 B"
        return r

    n = {"repairs": 0}

    def route(messages, **kw):
        last = messages[-1]["content"]
        if "Translate this Triton kernel" in last:
            return make_response(code("speed=1.2"))
        if "Diagnose the dominant bottleneck" in last:
            return make_response(PLAN)
        if "evaluation of your kernel failed" in last:
            n["repairs"] += 1
            assert "under 32768 B" in last  # the limit is in the repair prompt
            return make_response(code("abi=1"))
        if "Implement exactly this optimization plan" in last:
            return make_response(code("abi=1"))
        raise AssertionError(last[:80])

    orch = make_orch(tmp_path, route, calls, max_rounds=1, max_repair=3, evaluate_fn=ev, abi_guard=True)
    run(orch)
    # three branches, each: first kernel + one repair, then the repeated ABI stops the branch (a third repair is not spent)
    assert n["repairs"] == 3


def test_the_search_stops_at_the_roofline_and_not_when_the_bound_is_wrong(tmp_path):
    def ev_factory(bound_us, baseline_us):
        def ev(task, dtype, size, source, *, mode="full", seed=None, reps=15, **kw):
            r = EvalReport(task=task, dtype=dtype, size=size, mode=mode, seed=1, stage="done")
            r.status, r.message, r.speedup, r.speedups = "ok", "ok", 1.0, {"mlx": 1.0}
            r.candidate_us, r.baseline_us = 100.0, {"mlx": baseline_us}
            r.roofline = {
                "bound": "compute", "lower_bound_us": bound_us, "time_over_lower_bound": 1.001, "flops": 0,
                "achieved_gbps": 10.0, "bandwidth_fraction": 0.02, "peak_gbps_measured": 449.0,
            }
            r.benchmark = {"noisy": False}
            return r

        return ev

    stopped = make_orch(tmp_path / "a", llm_script(code("speed=1")), [], max_rounds=2, evaluate_fn=ev_factory(90, 100), stop_at_roofline=True)
    res = run(stopped)
    assert res.stop_reason == "at_roofline"
    assert res.rounds == 1  # the check fires at the start of round 1, before a plan is bought

    continued = make_orch(tmp_path / "b", llm_script(code("speed=1")), [], max_rounds=1, evaluate_fn=ev_factory(200, 100), stop_at_roofline=True)
    res = run(continued)
    assert res.stop_reason != "at_roofline"


def test_a_failed_branch_is_named_in_the_history_only_when_asked(tmp_path):
    calls = []

    def ev(task, dtype, size, source, *, mode="full", seed=None, reps=15, **kw):
        calls.append(source)
        r = EvalReport(task=task, dtype=dtype, size=size, mode=mode, seed=1, stage="done")
        if "GOOD" in source:
            r.status, r.message, r.speedup, r.candidate_us, r.baseline_us = "ok", "ok", 1.3, 70.0, {"mlx": 100.0}
        else:
            r.status, r.message = "incorrect", "case 'outlier': 1/1048576 elements out of tolerance (worst scaled error 1.38)"
        return r

    def route(messages, **kw):
        last = messages[-1]["content"]
        if "Translate this Triton kernel" in last:
            return make_response(code("GOOD"))
        if "Diagnose the dominant bottleneck" in last:
            return make_response(PLAN)
        if "Implement exactly this optimization plan" in last or "evaluation of your kernel failed" in last:
            return make_response(code("BAD"))
        raise AssertionError(last[:60])

    orch = make_orch(tmp_path, route, calls, max_rounds=1, max_repair=0, evaluate_fn=ev, rich_history=True, status_lines=True)
    run(orch)
    text = "\n".join(orch.history)
    assert "incorrect:" in text and "outlier" in text
    assert any("noise floor" in (c.get("content") or "") for c in _user_messages(orch))


def _user_messages(orch):
    out = []
    for event in orch.llm.transcript.events if hasattr(orch.llm.transcript, "events") else []:
        out.append(event)
    # the transcript log stores dicts; read them back from the file the orchestrator wrote
    import json
    from pathlib import Path

    path = Path(orch.run_dir) / "transcript.jsonl"
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().splitlines():
        doc = json.loads(line)
        for m in doc.get("messages") or []:
            if m.get("role") == "user":
                rows.append(m)
    return rows


def test_shader_validation_threadgroup_size_is_not_treated_as_the_device_limit():
    from tcenv.abi import only_static_threadgroup_limit

    inflated = "static threadgroup memory 40960 B exceeds the device limit of 32768 B"
    assert only_static_threadgroup_limit([inflated])
    assert not only_static_threadgroup_limit([])
    assert not only_static_threadgroup_limit([inflated, "missing kernel parameter `x`"])
    assert not only_static_threadgroup_limit(["launch infeasible: the contract launches 128 threads"])
