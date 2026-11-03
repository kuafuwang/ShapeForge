"""The opt-in search switches. Off, the prompts and the stop rules are the ones from before the switches existed."""

from __future__ import annotations

from taic.feedback_extra import abi_signature, at_roofline, status_lines, target_note, threadgroup_limit_note
from taic.llm import LLMConfig, SearchConfig
from taic.orchestrator import _retry_without_reasoning
from taic.prompts import patch_user, plan_user
from tcenv.evaluate import EvalReport

from ..support.fakes import make_response
from ..support.v1_tasks import V1_TASK, v1_task
from ..support.orchestrator import PLAN, code, llm_script, make_orch, run


def _task():
    """@brief 取一个保留 v1 任务对象（开关文本与题目数学无关）。

    @return 该规格的 Task。
    """
    return v1_task()


def test_switches_default_off_and_do_not_change_the_prompt_text():
    """@brief 验证各开关的默认值，以及关闭时不改变提示词文本；打开后状态行才被注入。"""
    assert SearchConfig().stop_at_roofline is True
    assert SearchConfig().abi_guard is True
    assert SearchConfig().target_note is True
    assert SearchConfig().rich_history is True
    assert SearchConfig().plan_ledger is True
    assert SearchConfig().status_lines is True
    assert SearchConfig().harness_auto is False
    assert SearchConfig().harness_shadow is True
    assert LLMConfig(base_url="http://x", api_key="k", model="m").truncation_retry is True
    task = _task()
    plain = plan_user(task, "kernel void relu_kernel() {}", "feedback", ["round 1: A -> 1.100x (new best)"], 3, source_fence="metal")
    assert plan_user(task, "kernel void relu_kernel() {}", "feedback", ["round 1: A -> 1.100x (new best)"], 3, status="", source_fence="metal") == plain
    assert "noise floor" not in plain
    plan = {"title": "A", "change": "do A"}
    assert "noise floor" not in patch_user(task, "kernel void relu_kernel() {}", "feedback", plan, source_fence="metal")
    told = plan_user(task, "kernel void relu_kernel() {}", "feedback", [], 3, status=status_lines(rnd=2, max_rounds=12, patience_left=3, spent=1.5, budget=3, required_gain=1.014), source_fence="metal")
    assert "at least 1.4% faster" in told and "Round 2/12" in told and "spent $1.50 of $3.00" in told
    assert told.index("noise floor") < told.index("Diagnose the dominant bottleneck")


def test_a_baseline_faster_than_the_bound_is_named_and_does_not_count_as_at_the_roof():
    """@brief 验证基准比理论下界更快时会在提示里点名该基准，且不把搜索当成已到屋顶停机。

    同时覆盖噪音、缺少实测峰值带宽、越过下界等不应判为到屋顶的情形。
    """
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
    assert at_roofline({**quiet, "roofline": {"lower_bound_us": 100, "time_over_lower_bound": 0.90, "peak_gbps_measured": 400.0}}, 1.02) is False
    launched = {**quiet, "compile": {"ok": True, "launch": "math=fast"}}
    assert at_roofline(launched, 1.02) is True


def test_the_same_threadgroup_overflow_compares_equal_and_states_the_limit():
    """@brief 验证同类 threadgroup 显存溢出产生相同 ABI 签名，并转成写明设备上限的中文提示。"""
    a = "static threadgroup memory 40960 B exceeds the device limit of 32768 B"
    b = "static threadgroup memory 36864 B exceeds the device limit of 32768 B"
    assert abi_signature(a) == abi_signature(b)
    assert abi_signature(a) != abi_signature("kernel does not match the required ABI")
    note = threadgroup_limit_note(a)
    assert note and "under 32768 B" in note and "declares 40960 B" in note
    assert threadgroup_limit_note("something else") is None


def test_a_truncated_reply_with_no_kernel_is_retried_once_without_spending_a_repair(tmp_path):
    """@brief 验证被截断且无内核的回复会重试一次，且不消耗修复额度。

    @param tmp_path pytest 提供的临时目录。
    """
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
    """@brief 验证重试会剔除内容为空的上轮助手消息及其 reasoning，只留下用户消息。"""
    messages = [
        {"role": "user", "content": "write it"},
        {"role": "assistant", "content": "", "reasoning_content": "a" * 50},
    ]
    out = _retry_without_reasoning(messages)
    assert out == [{"role": "user", "content": "write it"}]


def test_a_branch_that_repeats_one_abi_error_stops(tmp_path):
    """@brief 验证同一 ABI 错误反复出现时分支提前停下，不就地修复，只把限制写进下一轮反馈。

    @param tmp_path pytest 提供的临时目录。
    """
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
    # The session is not repaired in place. The limit is written into the feedback the next session would read.
    assert n["repairs"] == 0
    assert "32768" in orch._failure_feedback


def test_the_search_stops_at_the_roofline_and_not_when_the_bound_is_wrong(tmp_path):
    """@brief 验证达屋顶时在第 1 轮开头就停机，而下界估错（候选快于下界）时不停机。

    @param tmp_path pytest 提供的临时目录。
    """
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
    """@brief 验证开启富历史后失败分支的具体错误写进历史，并附上噪音下限证据；关闭时不写。

    @param tmp_path pytest 提供的临时目录。
    """
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
    assert any("noise floor" in (brief.evidence or "") for brief in orch._session_briefs)


def _user_messages(orch):
    """@brief 从运行目录的 transcript.jsonl 里取出所有 user 角色消息。

    @param orch 已写入 transcript 的 Orchestrator。
    @return user 消息字典列表；文件不存在时返回空列表。
    """
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
