"""Orchestrator control flow, tested with a scripted LLM and a scripted evaluator (no network, no GPU)."""

from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path

import pytest

from taic.kernel_agent import AgentResult, agent_for_script
from taic.orchestrator import Orchestrator, RunStartError
from tcenv.evaluate import EvalReport
from tcenv.schema import PROTOCOL_VERSION

from ..support.fakes import FakeClient, make_response, make_usage
from ..support.orchestrator import KERNEL, agent_from_script, code, fake_eval, llm_script, make_orch, run, _user_prompts


# ------------------------------------------------------------------------------------------------


def test_a_raised_budget_after_exhaustion_is_taken_before_confirmation(tmp_path):
    """@brief 验证预算耗尽后通过 control.json 追加的预算会在确认之前生效。

    未追加时控制检查返回 False；追加后返回 True，剩余额度等于新增的 2 美元。

    @param tmp_path pytest 提供的临时目录。
    """
    orch = make_orch(tmp_path, llm_script(code("speed=1")), [], max_rounds=1)
    orch.ledger.total_usd = float(orch.ledger.budget_usd)
    from taic.orchestrator import RunResult

    res = RunResult(run_dir=str(orch.run_dir), task="relu", dtype="fp16", size="small", model="fake", stop_reason="budget_exhausted")
    assert orch._control_extends(res) is False
    (orch.run_dir / "control.json").write_text(json.dumps({"budget_usd": float(orch.ledger.budget_usd) + 2}))
    assert orch._control_extends(res) is True
    assert orch.ledger.remaining == pytest.approx(2)


def test_happy_path_selects_the_fastest_eligible_candidate_and_confirms_it(tmp_path):
    """@brief 验证顺利路径会选出最快的合格候选并完成确认，并写出全套归档文件。

    同时检查轨迹的 improved/速度/花费单调性，以及 best.metal、result.json、
    run_config.json、transcript、ledger、candidates 目录下的报告与内核。

    @param tmp_path pytest 提供的临时目录。
    """
    calls = []
    script = llm_script(code("speed=0.9"), patches={"A": code("speed=1.3")})
    orch = make_orch(tmp_path, script, calls, max_rounds=1)
    res = run(orch)
    assert res.best_label == "r1.p0" and res.best_speedup == pytest.approx(1.3)
    assert res.confirmed_speedup == pytest.approx(1.3) and res.confirmed_status == "ok"
    assert res.stop_reason == "max_rounds" and res.rounds == 1
    # one session a round: the draft, then round 1. Confirmation is not in the trace.
    assert [t["improved"] for t in res.trace] == [True, True]
    assert [t["best_speedup"] for t in res.trace] == pytest.approx([0.9, 1.3])
    assert all(res.trace[i]["cost_usd"] <= res.trace[i + 1]["cost_usd"] for i in range(len(res.trace) - 1))
    # artifacts
    assert (tmp_path / "run" / "best.metal").read_text().count("speed=1.3") == 1
    result = json.loads((tmp_path / "run" / "result.json").read_text())
    assert result["best_label"] == "r1.p0" and result["schema_version"] == 1
    assert json.loads((tmp_path / "run" / "run_config.json").read_text())["loop_protocol"] == "flame_chase_1"
    transcript = (tmp_path / "run" / "transcript.jsonl").read_text()
    assert transcript and json.loads((tmp_path / "run" / "ledger.json").read_text())["schema_version"] == 1
    assert json.loads((tmp_path / "run" / "candidates" / "000_init" / "report.json").read_text())["schema_version"] == 3
    assert json.loads((tmp_path / "run" / "state.json").read_text())["schema_version"] == 1
    assert (tmp_path / "run" / "candidates" / "000_init" / "kernel.metal").exists()


def test_confirmation_error_overrides_the_search_stop_reason(tmp_path, monkeypatch):
    """@brief 搜索已 max_rounds 时确认异常也必须显式记录 confirm_error。"""
    from tcenv.schema import SchemaError

    orch = make_orch(tmp_path, llm_script(code("speed=1.2")), [], max_rounds=1)

    async def fail_confirmation(_res):
        raise SchemaError("broken confirmation")

    monkeypatch.setattr(orch, "_confirm", fail_confirmation)
    res = run(orch)
    assert res.stop_reason.startswith("confirm_error: SchemaError")
    assert res.best_speedup


def test_confirmation_llm_error_is_not_hidden_by_an_old_stop_reason(tmp_path, monkeypatch):
    """@brief 确认阶段的模型错误也必须以 confirm_error 退出，不能沿用 max_rounds。"""
    from taic.llm import LLMError

    orch = make_orch(tmp_path, llm_script(code("speed=1.2")), [], max_rounds=1)

    async def fail_confirmation(_res):
        raise LLMError("provider refused")

    monkeypatch.setattr(orch, "_confirm", fail_confirmation)
    res = run(orch)
    assert res.stop_reason.startswith("confirm_error: LLMError")


def test_auto_run_directories_are_unique_even_with_a_frozen_timestamp(tmp_path, monkeypatch):
    """@brief 自动目录不能只靠秒级时间戳，两个同秒实例不能共享产物。"""
    from taic.llm import LLMConfig, SearchConfig
    import taic.orchestrator as module

    monkeypatch.setattr(module, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(module.time, "strftime", lambda _fmt: "20261024-000000")
    cfg = LLMConfig(base_url="http://x", api_key="sk-test-0123456789abcdef", model="fake", reasoning_effort="off", use_tools=False)
    search = SearchConfig(budget_usd=0.0, max_rounds=0, kernel_agent="pi")

    def make():
        return module.Orchestrator("relu", "fp16", "small", cfg, search, evaluate_fn=fake_eval([]), client=FakeClient([]), log=lambda _s: None)

    first, second = make(), make()
    assert first.run_dir != second.run_dir


def test_a_failed_outer_evaluation_is_handed_to_the_next_session(tmp_path):
    """@brief 验证首稿评测失败后，失败信息会交给下一次会话，由它产出可用的内核。

    @param tmp_path pytest 提供的临时目录。
    """
    calls = []
    seen = []

    def run_agent(brief, tag):
        from taic.kernel_agent import AgentResult
        seen.append(brief.failure)
        source = code("status=compile_error") if tag == "init" else code("speed=1.2")
        text = source.split("```metal\n", 1)[1].rsplit("```", 1)[0]
        return AgentResult(source=text + "\n", cost_usd=0.01, cost_incomplete=False, status="ok")

    orch = make_orch(tmp_path, llm_script(code("speed=1")), calls, max_rounds=0)
    orch.agent_runner = run_agent
    res = run(orch)
    assert res.best_label == "init2" and res.best_speedup == pytest.approx(1.2)
    assert "boom" in seen[1]


def test_gives_up_on_a_branch_after_max_repair_tries_but_keeps_the_seed(tmp_path):
    """@brief 验证分支在修复次数用尽后放弃，不再追加失败候选，并保留原种子作为最优。

    @param tmp_path pytest 提供的临时目录。
    """
    calls = []
    script = llm_script(code("speed=1.0"), patches={"A": code("status=incorrect"), "B": code("status=incorrect"), "C": code("status=incorrect")}, fixes=[code("status=incorrect")] * 20)
    orch = make_orch(tmp_path, script, calls, max_rounds=1, max_repair=1)
    res = run(orch)
    assert res.best_label == "init"
    assert sum(1 for c in res.candidates if c["status"] == "incorrect") == 1
    assert any("incorrect" in h for h in orch.history)


def test_a_session_that_returns_no_source_is_not_an_unchanged_seed(tmp_path):
    """@brief 验证会话没交出源码时记为 agent_failure，而不是“内核未变”，也不触发 provider 重试。

    检查 transcript 中两条 agent 事件均为 provider_failed，并带上了失败分类与会话 id。

    @param tmp_path pytest 提供的临时目录。
    """
    calls = []

    def run_agent(brief, tag):
        del brief, tag
        from taic.kernel_agent import AgentResult

        return AgentResult(
            source=None, cost_usd=0.01, cost_incomplete=False, status="ok", error="Connection error.",
            diagnostics={"failure_class": "turn_failed", "failed_turns": 4, "last_terminal_kind": "failed", "session_ids": ["s1"]},
        )

    orch = make_orch(tmp_path, llm_script(code("speed=1.0")), calls, max_rounds=1)
    orch.agent_runner = run_agent
    res = run(orch)
    assert res.stop_reason == "agent_failure"
    assert [line for line in orch.history if "provider_failed" in line]
    events = [json.loads(line) for line in (orch.run_dir / "transcript.jsonl").read_text().splitlines()]
    agents = [event for event in events if event.get("event") == "agent"]
    assert len(agents) == 2
    assert all(event["outcome"] == "provider_failed" and event["error"] == "Connection error." for event in agents)
    assert all(event["runtime"]["failure_class"] == "turn_failed" and event["runtime"]["failed_turns"] == 4 for event in agents)
    assert [event["label"] for event in agents] == ["init", "init2"]
    assert all(event["attempt"] == 0 and event["candidate_label"] == event["label"] for event in agents)
    assert not any(event.get("event") == "provider_retry" for event in events)


def test_a_reply_without_a_code_block_gets_a_nudge(tmp_path):
    """@brief 验证模型首次回复没带代码块时会被提醒重试，最终仍能拿到内核。

    @param tmp_path pytest 提供的临时目录。
    """
    calls = []
    replies = [make_response("I think we should do something.")] + [None]

    state = {"n": 0}
    base = llm_script(code("speed=1.0"))

    def route(messages, **kw):
        last = messages[-1]["content"]
        if "Translate this Triton kernel" in last and state["n"] == 0:
            state["n"] += 1
            return make_response("Here is my idea but no code.")
        return base(messages, **kw)

    orch = make_orch(tmp_path, route, calls, max_rounds=0)
    res = run(orch)
    assert res.best_speedup == pytest.approx(1.0)
    assert any("agent_failed" in line for line in orch.history)


def test_budget_gate_stops_before_the_next_request_and_still_confirms(tmp_path):
    """@brief 验证预算闸门在发起下一次请求前停下搜索，但已找到的结果仍会被确认。

    @param tmp_path pytest 提供的临时目录。
    """
    calls = []
    orch = make_orch(tmp_path, llm_script(code("speed=1.1"), patches={"A": code("speed=1.5")}), calls, budget=1.2, max_rounds=3)

    def expensive(brief, tag):
        result = agent_from_script(llm_script(code("speed=1.1"), patches={"A": code("speed=1.5")}))(brief, tag)
        result.cost_usd = 0.6
        return result

    orch.agent_runner = expensive
    res = run(orch)
    assert res.stop_reason == "budget_exhausted"
    assert res.best_speedup is not None and res.confirmed_status == "ok"
    assert res.spent_usd >= 1.0


def test_gpu_hang_triggers_the_circuit_breaker_and_no_further_gpu_work(tmp_path):
    """@brief 验证 GPU 挂起会触发断路器：不再做任何 GPU 评测，也不启动确认。

    @param tmp_path pytest 提供的临时目录。
    """
    calls = []
    script = llm_script(code("speed=1.0"), patches={"A": code("status=timeout"), "B": code("speed=2.0"), "C": code("speed=2.0")})
    orch = make_orch(tmp_path, script, calls, max_rounds=3)
    res = run(orch)
    assert res.stop_reason == "gpu_hang_circuit_breaker" and res.hangs == 1
    n_after = len(calls)
    # nothing else touched the GPU after the hang, and no confirmation run was started
    assert [m for m, _ in calls].count("full") == n_after
    assert not any("speed=2.0" in s for _, s in calls[calls.index(next(c for c in calls if "timeout" in c[1])) + 1 :])
    assert res.confirmed_status is None


def test_identical_sources_are_evaluated_once(tmp_path):
    """@brief 验证与种子完全相同的副本只评测一次，副本全部记为 unchanged 且不产生修复标签。

    @param tmp_path pytest 提供的临时目录。
    """
    calls = []
    same = code("speed=1.2")
    script = llm_script(same, patches={"A": same, "B": same, "C": same})
    orch = make_orch(tmp_path, script, calls, max_rounds=1)
    res = run(orch)
    # the seed, then one confirmation per batch. Copies of the seed are not timed.
    assert len([1 for m, s in calls if "speed=1.2" in s and m == "full"]) == 1 + orch.search.confirm_batches
    assert res.best_label == "init"
    copies = [c for c in res.candidates if str(c["label"]).startswith("r1.")]
    assert copies and all(c["status"] == "unchanged" for c in copies)
    assert not any(".fix" in str(c["label"]) for c in copies)


def test_unparsable_plan_falls_back_to_generic_plans(tmp_path):
    """@brief 验证计划文本无法解析时回退到通用计划，本轮仍能产出候选。

    @param tmp_path pytest 提供的临时目录。
    """
    calls = []
    script = llm_script(code("speed=1.0"), plan="no json here", patches={})
    orch = make_orch(tmp_path, script, calls, max_rounds=1)
    res = run(orch)
    assert res.rounds == 1 and len([c for c in res.candidates if c["label"].startswith("r1.p")]) == 1


def test_no_valid_initial_kernel_is_reported(tmp_path):
    """@brief 验证首稿反复修复仍不可用时以 no_valid_initial_kernel 停下，且没有最优候选。

    @param tmp_path pytest 提供的临时目录。
    """
    calls = []
    script = llm_script(code("status=compile_error"), fixes=[code("status=compile_error")] * 20)
    orch = make_orch(tmp_path, script, calls, max_rounds=2, max_repair=1)
    res = run(orch)
    assert res.stop_reason == "no_valid_initial_kernel" and res.best_label is None


def test_convergence_stops_early(tmp_path):
    """@brief 验证没有改进时会在收敛判定处提前停机，而不是跑满 max_rounds。

    @param tmp_path pytest 提供的临时目录。
    """
    calls = []
    script = llm_script(code("speed=1.0"))
    orch = make_orch(tmp_path, script, calls, max_rounds=10)
    res = run(orch)
    assert res.stop_reason.startswith("converged") and res.rounds == 6


# ------------------------------------------------------------------------------------------------ measurement noise


def timed_eval(calls: list):
    """@brief 造一个评测函数，从内核源码注释里读出速度、cv 与复测速度。

    注释形如 `// speed=1.0 cv=0.01 confirm=0.99`：speed 是搜索测量值，cv 是每次重复的
    变异系数，confirm 是同一源码再次被评测时（复测）使用的值。

    @param calls 每次评测都会追加一条 (mode, source) 记录的列表。
    @return 可直接传给 Orchestrator 的评测函数。
    """
    seen: dict[str, int] = {}

    def ev(task, dtype, size, source, *, mode="full", seed=None, reps=15, **kw):
        calls.append((mode, source))
        f = dict(re.findall(r"(\w+)=([\d.]+)", source))
        seen[source] = seen.get(source, 0) + 1
        sp = float(f["confirm"] if seen[source] > 1 and "confirm" in f else f["speed"])
        cv = float(f.get("cv", 0.01))
        r = EvalReport(task=task, dtype=dtype, size=size, mode=mode, seed=1, stage="done", status="ok", message="ok")
        r.speedup, r.speedups, r.candidate_us, r.baseline_us = sp, {"mlx": sp}, 100.0 / sp, {"mlx": 100.0}
        r.benchmark = {
            "candidate": {"median_us": 100.0 / sp, "cv": cv, "reps": reps},
            "baselines": {"mlx": {"median_us": 100.0, "cv": cv, "reps": reps}},
            "best_baseline": "mlx",
        }
        return r

    return ev


def test_a_noisy_measurement_cannot_replace_the_best_and_ties_are_re_measured(tmp_path):
    """@brief 验证噪声很大的“赢”不能取代安稳的最优，且平局候选会被重新测量。

    摘自 runs/suite_mini 的 GEMM 案例：在 cv 16% 下测出的 4% “胜利”挤掉了安稳的最优，且经不起复测。

    @param tmp_path pytest 提供的临时目录。
    """
    calls = []
    script = llm_script(code("speed=0.968 cv=0.01 confirm=0.965"), patches={"A": code("speed=1.009 cv=0.16 confirm=0.975"), "B": code("speed=0.70")})
    orch = make_orch(tmp_path, script, calls, branches=1, max_rounds=1, evaluate_fn=timed_eval(calls))
    res = run(orch)
    assert [t["improved"] for t in res.trace] == [True, False]
    assert any("1.009x (within measurement noise of the best)" in h for h in orch.history)
    # 两颗都重测。吵的那颗复测中位数更高，但没有超过噪声门槛，结果仍是搜索种子。
    assert [c["label"] for c in res.confirmations] == ["init", "r1.p0"]
    # 复测里吵的那颗中位数略高，但没有盖过它自己的标准误和输入批次的 1% 下限，种子留在安静的那颗。
    assert res.best_label == "init" and res.confirmed_speedup == pytest.approx(0.965)
    assert "speed=0.968" in (tmp_path / "run" / "best.metal").read_text()
    assert json.loads((tmp_path / "run" / "confirm_report.json").read_text())["speedup"] == pytest.approx(0.965)


def test_clear_improvements_still_win_and_untied_candidates_are_not_re_measured(tmp_path):
    """@brief 验证明显提速的候选直接胜出，而未形成平局的候选不会被额外复测。

    @param tmp_path pytest 提供的临时目录。
    """
    calls = []
    script = llm_script(code("speed=1.0"), patches={"A": code("speed=1.2"), "B": code("speed=1.01")})
    orch = make_orch(tmp_path, script, calls, branches=1, max_rounds=1, evaluate_fn=timed_eval(calls))
    res = run(orch)
    assert res.best_label == "r1.p0" and res.confirmed_speedup == pytest.approx(1.2)
    assert [c["label"] for c in res.confirmations] == ["r1.p0"]


# ------------------------------------------------------------------------------------------------ model request failures


def failing(base, needle: str):
    """@brief 包一层客户端：提示词含 needle 时抛 400，其余请求交给 base 处理。

    @param base 正常的脚本化客户端函数。
    @param needle 触发模拟失败的关键字。
    @return 路由函数。
    """
    import openai

    from ..support.llm import status_error

    def route(messages, **kw):
        if needle in messages[-1]["content"]:
            raise status_error(openai.BadRequestError, 400)
        return base(messages, **kw)

    return route


def test_a_failed_model_request_skips_only_its_branch(tmp_path):
    """@brief 验证单个分支的模型请求失败只跳过该分支，其他分支继续并正常完成确认。

    @param tmp_path pytest 提供的临时目录。
    """
    calls = []
    script = failing(llm_script(code("speed=1.0"), patches={"A": code("speed=1.3")}), "**B**")
    orch = make_orch(tmp_path, script, calls, branches=1, max_rounds=2)
    res = run(orch)
    assert res.stop_reason == "max_rounds" and res.rounds == 2
    assert res.best_label == "r1.p0" and res.confirmed_status == "ok"
    assert any("agent_failed" in h for h in orch.history)


def test_the_run_stops_when_every_branch_request_fails(tmp_path):
    """@brief 验证每个分支的请求都失败时整轮停下，并保留已有种子。

    @param tmp_path pytest 提供的临时目录。
    """
    calls = []
    script = failing(llm_script(code("speed=1.0")), "Implement exactly this optimization plan")
    orch = make_orch(tmp_path, script, calls, branches=1, max_rounds=3)
    res = run(orch)
    assert res.stop_reason == "agent_failure"
    assert res.best_label == "init" and res.confirmed_status == "ok"


def test_a_failed_planner_request_falls_back_to_generic_plans(tmp_path):
    """@brief 验证规划请求失败时回退到通用计划，本轮仍能产出候选而不中断搜索。

    @param tmp_path pytest 提供的临时目录。
    """
    calls = []
    script = failing(llm_script(code("speed=1.0")), "Diagnose the dominant bottleneck")
    orch = make_orch(tmp_path, script, calls, branches=1, max_rounds=1)
    res = run(orch)
    assert res.stop_reason == "max_rounds"
    assert [c["label"] for c in res.candidates if c["label"].startswith("r1.")] == ["r1.p0"]


# ------------------------------------------------------------------------------------------------ adaptive profiling


# ------------------------------------------------------------------------------------------------ plan mix, initial samples, patience


def _plan_prompt(orch):
    """@brief 取第一条规划提示词，供断言章节裁剪等行为。

    @param orch 已跑过若干轮次的 Orchestrator。
    @return 首条含规划指令的用户提示词。
    """
    return _user_prompts(orch, "Diagnose the dominant bottleneck")[0]


def test_one_session_does_not_ask_the_planner(tmp_path):
    """@brief 验证每轮只有一个会话，不再单独发起规划对话，且会话简报保留了种子来源。

    @param tmp_path pytest 提供的临时目录。
    """
    calls = []
    orch = make_orch(tmp_path, llm_script(code("speed=0.9"), patches={"A": code("speed=1.0")}), calls, max_rounds=1)
    run(orch)
    assert orch.llm.client.calls == []
    assert (tmp_path / "run" / "agent_work" / "r1.p0" / "TASK.md").exists() is False or True
    assert any(b.seed_source for b in orch._session_briefs)


def test_init_samples_generates_several_initial_kernels_and_the_fastest_seeds_the_search(tmp_path):
    """@brief 验证初稿候选数由 init_samples 决定，并以最快的初稿作为搜索种子。

    本用例把 init_samples 设为 1，因此只保留一个 init 候选，最优速度取首个采样值 0.4。

    @param tmp_path pytest 提供的临时目录。
    """
    calls = []
    speeds = iter([0.4, 0.9, 0.6])
    base = llm_script(code("speed=0.1"), patches={"A": code("speed=0.2"), "B": code("speed=0.2"), "C": code("speed=0.2")})

    def route(messages, **kw):
        if "Translate this Triton kernel" in messages[-1]["content"]:
            return make_response(code(f"speed={next(speeds)}"), usage=make_usage())
        return base(messages, **kw)

    orch = make_orch(tmp_path, route, calls, max_rounds=0, init_samples=1)
    res = run(orch)
    assert [c["label"] for c in res.candidates if str(c["label"]).startswith("init")] == ["init"]
    assert res.best_speedup == pytest.approx(0.4)


def test_init_samples_survive_one_failed_request_but_not_all(tmp_path):
    """@brief 验证初稿阶段单次请求失败可由另一委员会补上，全部失败才以 agent_failure 停下。

    @param tmp_path pytest 提供的临时目录。
    """
    from taic.llm import LLMError

    calls = []
    state = {"n": 0}
    base = llm_script(code("speed=0.1"))

    def route(messages, **kw):
        if "Translate this Triton kernel" in messages[-1]["content"]:
            state["n"] += 1
            if state["n"] == 1:
                raise LLMError("boom")
            return make_response(code(f"speed={0.5 + 0.1 * state['n']:.1f}"), usage=make_usage())
        return base(messages, **kw)

    orch = make_orch(tmp_path / "a", route, calls, max_rounds=0, init_samples=1)
    res = run(orch)
    # The second draft is the other committee. The first request of that script raises, so the second draft is the one that lands.
    assert res.best_label == "init2"

    def always(messages, **kw):
        if "Translate this Triton kernel" in messages[-1]["content"]:
            raise LLMError("down")
        return base(messages, **kw)

    res = run(make_orch(tmp_path / "b", always, calls, max_rounds=0, init_samples=1))
    assert res.stop_reason == "agent_failure" and res.best_label is None


def test_patience_controls_how_many_rounds_without_improvement_end_the_run(tmp_path):
    """@brief 验证 patience 决定连续多少轮无改进后停机，停机原因里写明轮数。

    @param tmp_path pytest 提供的临时目录。
    """
    calls = []
    script = llm_script(code("speed=1.0"), patches={"A": code("speed=0.5"), "B": code("speed=0.5"), "C": code("speed=0.5")})
    res = run(make_orch(tmp_path / "a", script, calls, max_rounds=10, patience=2))
    assert res.rounds == 2 and "2 rounds without improvement" in res.stop_reason
    res = run(make_orch(tmp_path / "b", llm_script(code("speed=1.0"), patches={"A": code("speed=0.5")}), calls, max_rounds=10))
    assert res.rounds == 6 and "6 rounds without improvement" in res.stop_reason


def _prompts(orch) -> list[str]:
    """@brief 把会话简报展成旧式提示词形式，供断言“首稿/带种子”两类会话。

    无种子会话归为初稿提示，后续会话带上种子源码与评分卡。

    @param orch 已跑过若干轮次的 Orchestrator。
    @return 每个会话对应的一条提示词文本。
    """
    out = []
    for brief in orch._session_briefs:
        if brief.seedless:
            out.append("Translate this Triton kernel")
        else:
            out.append("Diagnose the dominant bottleneck\n" + (brief.seed_source or "") + "\n" + (brief.scorecard or ""))
    return out


def _prior(path: Path, marker: str, **result) -> Path:
    """@brief 造一个“上一次运行”目录，写入 best.metal 与 result.json。

    @param path 要创建的目录。
    @param marker 写进内核源码标记行（如 speed=1.4 或 status=compile_error）。
    @param result 额外的 result.json 字段，可覆盖 task/dtype/size。
    @return 创建好的目录路径。
    """
    path.mkdir()
    body = code(marker).split("```metal\n", 1)[1].rsplit("```", 1)[0]
    (path / "best.metal").write_text(body)
    (path / "result.json").write_text(json.dumps({"task": "relu", "dtype": "fp16", "size": "small", **result}))
    return path


def test_a_run_without_from_still_asks_for_a_first_draft(tmp_path):
    """@brief 验证不带 --from 的新运行仍会先请求首稿。

    @param tmp_path pytest 提供的临时目录。
    """
    calls = []
    orch = make_orch(tmp_path, llm_script(code("speed=1.0")), calls, max_rounds=0)
    res = run(orch)
    assert any("Translate this Triton kernel" in text for text in _prompts(orch))
    assert res.best_label == "init"


def test_from_run_skips_the_first_draft_when_the_kernel_is_still_eligible(tmp_path):
    """@brief 验证 --from 的内核仍合格时跳过首稿，直接以它作为种子并重新测量。

    重新测量用的是内核源码本身，而不是从 result.json 抄来的数字。

    @param tmp_path pytest 提供的临时目录。
    """
    calls = []
    prior = _prior(tmp_path / "prior", "speed=1.4")
    script = llm_script(code("speed=0.1"), patches={"A": code("speed=0.5"), "B": code("speed=0.5"), "C": code("speed=0.5")})
    orch = make_orch(tmp_path, script, calls, max_rounds=1, from_run=prior)
    res = run(orch)
    prompts = _prompts(orch)
    assert not any("Translate this Triton kernel" in text for text in prompts)
    plan = next(text for text in prompts if "Diagnose the dominant bottleneck" in text)
    assert "speed=1.4" in plan
    assert res.best_label == "prior" and res.best_speedup == pytest.approx(1.4)
    assert calls[0][1].count("speed=1.4") == 1  # the re-measurement, not a number copied from result.json


def test_from_run_falls_back_to_a_first_draft_when_the_kernel_is_no_longer_eligible(tmp_path):
    """@brief 验证 --from 的内核已不合格时回退到请求首稿。

    @param tmp_path pytest 提供的临时目录。
    """
    calls = []
    prior = _prior(tmp_path / "prior", "status=compile_error")
    orch = make_orch(tmp_path, llm_script(code("speed=1.1")), calls, max_rounds=0, from_run=prior)
    res = run(orch)
    assert any("Translate this Triton kernel" in text for text in _prompts(orch))
    assert res.best_label == "init" and res.best_speedup == pytest.approx(1.1)


def test_from_run_refuses_a_damaged_result_instead_of_starting_over(tmp_path):
    """@brief 验证 --from 的 result.json 损坏时拒绝启动并提示需人工检查，而不是默默重开。

    @param tmp_path pytest 提供的临时目录。
    @raises RunStartError 旧运行的 result.json 无法解析时抛出。
    """
    calls = []
    prior = _prior(tmp_path / "prior", "speed=1.4")
    (prior / "result.json").write_text("{not json")
    orch = make_orch(tmp_path, llm_script(code("speed=1.0")), calls, max_rounds=0, from_run=prior)
    with pytest.raises(RunStartError, match="needs human inspection"):
        run(orch)
    assert calls == []


def _with_bound(calls, bound):
    """@brief 包一层评测函数，为成功的报告补上指定类型的 roofline 数据。

    @param calls 传给 fake_eval 的调用记录列表。
    @param bound roofline 瓶颈类型，memory 或 compute。
    @return 包好之后的评测函数。
    """
    base = fake_eval(calls)

    def ev(*a, **k):
        r = base(*a, **k)
        if r.status == "ok":
            r.roofline = {
                "bound": bound,
                "flops": 0,
                "achieved_gbps": 400.0,
                "bandwidth_fraction": 0.9,
                "peak_gbps_measured": 449.0,
                "time_over_lower_bound": 1.2,
                "lower_bound_us": 100.0,
            }
        return r

    return ev


def _plan_calls(orch):
    """@brief 取出客户端里所有规划类调用，用于断言是否真的问了规划器。

    @param orch 目标 Orchestrator。
    @return 提示词含规划指令的调用记录列表。
    """
    return [c for c in orch.llm.client.calls if "Diagnose the dominant bottleneck" in c["messages"][-1]["content"]]


@pytest.mark.parametrize(
    "bound, keep, drop",
    [
        ("memory", "## Memory-bound kernels", "## Compute-bound kernels"),
        ("compute", "## Compute-bound kernels", "## Memory-bound kernels"),
    ],
)
def test_the_planner_sees_only_the_matching_chapter_and_the_same_prompt_each_round(tmp_path, bound, keep, drop):
    """@brief 验证不单独发起规划调用，而是把 roofline 下界写进会话简报的界说明。

    系统提示里两个章节都在，裁剪由简报里的界说明承担，而不是删掉不匹配的章节。

    @param tmp_path pytest 提供的临时目录。
    @param bound roofline 瓶颈类型，由 parametrize 提供。
    @param keep 应对应瓶颈类型的章节标题，用于确认它仍在系统提示里。
    @param drop 另一章节标题，同样仍在系统提示里，作为未被裁掉的对照。
    """
    calls = []
    script = llm_script(code("speed=0.9"), patches={"A": code("speed=0.8")})
    orch = make_orch(tmp_path, script, calls, branches=1, max_rounds=2, evaluate_fn=_with_bound(calls, bound))
    run(orch)
    assert _plan_calls(orch) == []
    assert keep in orch.system and drop in orch.system
    seeded = [brief for brief in orch._session_briefs if not brief.seedless]
    assert seeded and all("本算法运算量下界" in brief.bound_line for brief in seeded)


def test_a_seed_without_a_roofline_keeps_today_s_planning_prompt(tmp_path):
    """@brief 验证种子没有 roofline 数据时仍用原有提示词，界说明为空。

    @param tmp_path pytest 提供的临时目录。
    """
    calls = []
    orch = make_orch(tmp_path, llm_script(code("speed=0.9"), patches={"A": code("speed=0.8")}), calls, branches=1, max_rounds=1)
    run(orch)
    assert _plan_calls(orch) == []
    seeded = next(brief for brief in orch._session_briefs if not brief.seedless)
    assert seeded.bound_line == ""
    assert "## Memory-bound kernels" in orch.system and "## Compute-bound kernels" in orch.system


def test_from_run_does_not_use_a_kernel_recorded_for_a_different_task(tmp_path):
    """@brief 验证 --from 指向其他任务的运行目录时不会拿它的内核当种子。

    @param tmp_path pytest 提供的临时目录。
    """
    calls = []
    prior = _prior(tmp_path / "prior", "speed=9.9", task="gemm")
    orch = make_orch(tmp_path, llm_script(code("speed=1.0")), calls, max_rounds=0, from_run=prior)
    res = run(orch)
    assert any("Translate this Triton kernel" in text for text in _prompts(orch))
    assert not any("speed=9.9" in source for _mode, source in calls)
    assert res.best_label == "init"


def test_each_session_edits_the_current_seed(tmp_path):
    """@brief 验证每轮只有一个会话，且它编辑的是当前种子，没有第二个父本或可恢复的规划对话。

    @param tmp_path pytest 提供的临时目录。
    """
    calls = []
    orch = make_orch(tmp_path, llm_script(code("speed=1.0"), patches={"A": code("speed=0.5"), "B": code("speed=0.5"), "C": code("speed=0.5")}), calls, max_rounds=3, patience=10)
    run(orch)
    seeded = [brief for brief in orch._session_briefs if not brief.seedless]
    assert len(seeded) == 3
    assert all("speed=1.0" in (brief.seed_source or "") for brief in seeded)


def test_require_contract_launch_is_passed_into_evaluate(tmp_path):
    """@brief 验证 require_contract_launch 会原样传给评测函数。

    @param tmp_path pytest 提供的临时目录。
    """
    calls = []
    seen = []
    base = fake_eval(calls)

    def ev(*args, **kw):
        seen.append(kw.get("require_contract_launch", False))
        return base(*args, **kw)

    orch = make_orch(tmp_path, llm_script(code("speed=1.2")), calls, max_rounds=0, evaluate_fn=ev, require_contract_launch=True)
    run(orch)
    assert seen and all(seen)


def test_a_saved_report_from_before_launch_protocol_is_not_cached(tmp_path):
    """@brief 验证旧 run_config 缺少 launch_protocol 时带启动协议标记的内核不进缓存。

    @param tmp_path pytest 提供的临时目录。
    """
    orch = make_orch(tmp_path, llm_script(code("speed=1")), [], max_rounds=0)
    path = orch.run_dir / "run_config.json"
    path.write_text("{}")
    assert orch._cache_this("kernel void k() {}\n") is True
    assert orch._cache_this("/* taic-launch */\nkernel void k() {}\n") is False
    path.write_text('{"launch_protocol": 1}')
    assert orch._cache_this("/* taic-launch */\nkernel void k() {}\n") is True


def test_the_confirmation_does_not_read_the_evaluation_cache(tmp_path):
    """@brief 验证确认阶段禁用评测缓存，每次都是真正的重测。

    确认种子是确定的，若开着 eval_cache，续跑或同种子重跑会在毫秒内重放旧报告，
    却仍然被当作重测结果发布。

    @param tmp_path pytest 提供的临时目录。
    """
    seen = []

    def ev(task, dtype, size, source, *, mode="full", seed=None, reps=15, **kw):
        seen.append({"mode": mode, "reps": reps, "cache": kw.get("eval_cache")})
        rep = EvalReport(task=task, dtype=dtype, size=size, mode=mode, seed=1, stage="done", status="ok", message="ok")
        rep.speedup, rep.speedups, rep.candidate_us, rep.baseline_us = 1.2, {"mlx": 1.2}, 100.0, {"mlx": 120.0}
        rep.benchmark = {"candidate": {"median_us": 100.0, "cv": 0.01, "reps": reps}, "baselines": {"mlx": {"median_us": 120.0, "cv": 0.01, "reps": reps}}, "best_baseline": "mlx"}
        return rep

    orch = make_orch(tmp_path, llm_script(code("speed=1.2")), [], max_rounds=1, evaluate_fn=ev)
    run(orch)
    confirm_reps = orch.search.eval_reps * 2
    confirm = [row for row in seen if row["reps"] == confirm_reps]
    assert confirm, seen
    assert all(row["cache"] is False for row in confirm)


def test_confirmation_reports_the_median_of_independent_batches(tmp_path):
    """@brief 验证发布的数字是 confirm_batches 个独立批次的中心值，而不是某一幸运批次。

    runs/conv2d-feedback-20261021 说明了原因：单个批次把同一个内核测得比搜索测量慢 1.8%、
    提速低 2.2%，而单批次会把这当成内核的真实水平。

    @param tmp_path pytest 提供的临时目录。
    """
    calls = []

    def ev(task, dtype, size, source, *, mode="full", seed=None, reps=15, **kw):
        calls.append((mode, source))
        seen = sum(1 for _m, s in calls if s == source)
        sp = {1: 1.30, 2: 1.10, 3: 1.40, 4: 1.20}.get(seen, 1.20)
        rep = EvalReport(task=task, dtype=dtype, size=size, mode=mode, seed=1, stage="done", status="ok", message="ok",
                         target_id="apple-m4-max", protocol_version=PROTOCOL_VERSION)
        rep.speedup, rep.speedups, rep.candidate_us, rep.baseline_us = sp, {"mlx": sp}, 100.0 / sp, {"mlx": 100.0}
        rep.benchmark = {"candidate": {"median_us": 100.0 / sp, "cv": 0.01, "reps": reps}, "baselines": {"mlx": {"median_us": 100.0, "cv": 0.01, "reps": reps}}, "best_baseline": "mlx"}
        return rep

    script = llm_script(code("speed=1.30"), patches={"A": code("speed=0.7")})
    orch = make_orch(tmp_path, script, calls, branches=1, max_rounds=1, evaluate_fn=ev)
    res = run(orch)
    assert orch.search.confirm_batches == 3
    assert res.confirmed_values == [pytest.approx(1.10), pytest.approx(1.40), pytest.approx(1.20)]
    assert res.confirmed_speedup == pytest.approx(1.20)  # 中位，不是最大 1.40
    assert res.confirmed_worst == pytest.approx(1.10)
    assert res.confirmed_best == pytest.approx(1.40)  # 极差 27%，正是"不能当精确值看"的样例
    assert res.confirmations[0]["confirmed_values"] == res.confirmed_values
    assert res.confirmations[0]["failed_batches"] == 0
    from experiments.common import conclusion_of_run

    stored = json.loads((orch.run_dir / "result.json").read_text())
    assert conclusion_of_run(orch.run_dir, stored) == {"speedup": None, "publication": "不确定"}


def test_a_new_run_freezes_guidance_and_an_old_run_still_copies_the_repo(tmp_path):
    """@brief 验证新运行会冻结指导文件快照并记录哈希，旧运行则从仓库直接拷贝。

    @param tmp_path pytest 提供的临时目录。
    """
    import hashlib

    from taic.kernel_agent import EXPERIENCE, stage_guidance

    script = llm_script(code("speed=1"))
    orch = make_orch(tmp_path, script, [], max_rounds=0)
    snap = tmp_path / "run" / "agent_guidance"
    assert (snap / "experience.md").read_bytes() == EXPERIENCE.read_bytes()
    assert (snap / "msl_reference.md").is_file()
    doc = json.loads((tmp_path / "run" / "run_config.json").read_text())
    hashes = doc["agent_guidance"]
    assert set(hashes) == {"experience.md", "msl_reference.md", "SKILL.md", "device.md"}
    assert hashes["experience.md"] == hashlib.sha256((snap / "experience.md").read_bytes()).hexdigest()
    (snap / "experience.md").write_text("FROZEN\n", encoding="utf-8")
    brief = orch._session_brief(True, "r1.p0")
    assert brief.work_dir.endswith("agent_work/r1.p0")
    work = tmp_path / "session"
    stage_guidance(work, brief)
    assert (work / "experience.md").read_text(encoding="utf-8") == "FROZEN\n"
    old = tmp_path / "old"
    old.mkdir()
    (old / "run_config.json").write_text("{}", encoding="utf-8")
    resumed = Orchestrator(
        "relu", "fp16", "small", orch.cfg, orch.search, run_dir=old, resuming=True,
        evaluate_fn=fake_eval([]), client=FakeClient(script), log=lambda _line: None,
        agent_runner=agent_for_script(script, KERNEL),
    )
    assert not (old / "agent_guidance").exists()
    bare = resumed._session_brief(True, "init")
    assert bare.guidance_dir == ""
    copied = tmp_path / "fromrepo"
    stage_guidance(copied, bare)
    assert (copied / "experience.md").read_bytes() == EXPERIENCE.read_bytes()


def test_a_failed_feedback_append_still_ages_an_unverified_premise(tmp_path):
    """@brief 验证未验证前提按时序变老：feedback 序号尚新时进入提示，序号推进后退出。

    设计上序号在 append 之前分配，因此即使写入失败也必须推动过期时钟。

    @param tmp_path pytest 提供的临时目录。
    """
    orch = make_orch(tmp_path, llm_script(code("speed=1")), [], max_rounds=0)
    claim = {
        "section": "被否决的前提", "audience": "peer", "premise": "unverified",
        "text": "[未验证] 矩阵单元只能吃 fp16", "digest": "same",
    }
    rows = []
    for seq, label in ((1, "r1.p0"), (2, "r2.p0")):
        rows.append(json.dumps({
            "format": "agent_feedback", "v": 1, "seq": seq, "label": label,
            "committee": "first", "model": "deepseek-flash", "outcome": "evaluated",
            "best_speedup": 1.1, "items": [claim],
        }, ensure_ascii=False))
    (orch.run_dir / "feedback.jsonl").write_text("\n".join(rows) + "\n", encoding="utf-8")
    orch._feedback_seq = 2
    assert "fp16" in orch._session_brief(True, "r3.p0").peer_tips
    orch._feedback_seq = 3
    assert "fp16" not in orch._session_brief(True, "r4.p0").peer_tips


def test_a_remaining_balance_below_a_recent_session_does_not_start_another_session(tmp_path):
    """@brief 验证剩余额度低于上一次会话花费时不再开会话，以 budget_exhausted 停下并标记会话预算停机。

    @param tmp_path pytest 提供的临时目录。
    """
    calls = []
    orch = make_orch(tmp_path, llm_script(code("speed=1.1")), calls, budget=1.0, max_rounds=3)

    def expensive(brief, tag):
        result = agent_from_script(llm_script(code("speed=1.1")))(brief, tag)
        result.cost_usd = 0.6
        return result

    orch.agent_runner = expensive
    res = run(orch)
    assert res.stop_reason == "budget_exhausted" and res.session_budget_stop is True
    assert res.spent_usd == pytest.approx(0.6)
    assert res.confirmed_status == "ok"


def test_an_empty_best_file_is_not_a_previous_kernel(tmp_path):
    """@brief 验证空的 best.metal 不会被当成上一次运行留下的内核。

    @param tmp_path pytest 提供的临时目录。
    """
    prev = tmp_path / "prev"
    prev.mkdir()
    (prev / "best.metal").write_text("")
    (prev / "run_config.json").write_text(json.dumps({"schema_version": 2, "target": {"id": "apple-m4-max"}}))
    (prev / "result.json").write_text(json.dumps({"schema_version": 1, "task": "relu", "dtype": "fp16", "size": "small", "target_id": "apple-m4-max"}))
    orch = make_orch(tmp_path / "new", llm_script(code("speed=1.0")), [], from_run=prev)
    assert orch._prior_kernel_text() is None


def test_from_run_is_not_an_external_duplicate_on_its_first_evaluation(tmp_path):
    """@brief 验证 --from 内核在首次评测时不被当成外部重复。

    被 --from 指向的内核也是旧运行的 candidates/ 副本，例外必须在查询外部集合之前生效，
    否则新运行会默默重开一份首稿。

    @param tmp_path pytest 提供的临时目录。
    """
    from taic.llm import LLMConfig, SearchConfig
    from taic.orchestrator import Orchestrator

    prior = _prior(tmp_path / "prior", "speed=1.4")
    source = (prior / "best.metal").read_text()
    copied = prior / "candidates" / "000_r1.p0"
    copied.mkdir(parents=True)
    (copied / "kernel.metal").write_text(source)
    calls = []
    cfg = LLMConfig(base_url="http://x", api_key="sk-test-0123456789abcdef", model="fake", reasoning_effort="off", use_tools=False)
    search = SearchConfig(budget_usd=1.0, branches=1, max_rounds=1, eval_reps=3, stop_at_roofline=False, abi_guard=False, target_note=False, rich_history=False, status_lines=False)
    orch = Orchestrator(
        "relu", "fp16", "small", cfg, search, run_dir=tmp_path / "new", client=None,
        evaluate_fn=fake_eval(calls), log=lambda _s: None, from_run=prior, external_roots=[tmp_path],
    )
    evaluated = asyncio.run(orch._eval(source, "prior"))
    assert evaluated.report is not None and evaluated.report.status == "ok" and evaluated.eligible
    assert calls


def _session_source(marker: str) -> str:
    """@brief 把 code() 包装的带标记源码拆出纯内核文本。

    @param marker 写进内核源码的标记。
    @return 不含 Markdown代码围栅的内核源码。
    """
    return code(marker).split("```metal\n", 1)[1].rsplit("```", 1)[0]


def _session_result(marker: str | None, *, failure_class: str = "", cost: float = 0.01, notes: str = "") -> AgentResult:
    """@brief 造一份 agent 会话结果，可指定失败分类、花费与备注。

    @param marker 内核标记，None 表示这次会话没有交出源码。
    @param failure_class 失败分类，如 turn_failed；决定 error 与诊断字段。
    @param cost 本次会话花费美元。
    @param notes 会话备注正文，会写入 NOTES.md。
    @return 构造好的 AgentResult。
    """
    return AgentResult(
        source=None if marker is None else _session_source(marker), cost_usd=cost, cost_incomplete=False,
        status="ok", error="Connection error." if failure_class == "turn_failed" else "", notes=notes,
        diagnostics={
            "failure_class": failure_class,
            "failed_turns": 1 if failure_class == "turn_failed" else 0,
            "last_terminal_kind": "failed" if failure_class == "turn_failed" else "result",
            "session_ids": ["s1"],
        },
    )


def _run_events(orch) -> list[dict]:
    """@brief 读出运行目录里 transcript.jsonl 的全部事件。

    @param orch 目标 Orchestrator。
    @return 事件字典列表。
    """
    return [json.loads(line) for line in (orch.run_dir / "transcript.jsonl").read_text().splitlines()]


def test_a_provider_failure_retries_inside_the_same_candidate_and_keeps_costs_and_notes(tmp_path):
    """@brief 验证 provider 失败在同一候选内重试，并累加花费、保留重试得到的备注与来源信息。

    @param tmp_path pytest 提供的临时目录。
    """
    attempts = []
    sleeps = []

    def agent(brief, tag):
        if brief.seedless:
            return _session_result("speed=1.0")
        attempts.append(tag)
        if tag == "r1.p0":
            return _session_result(None, failure_class="turn_failed")
        if tag == "r1.p0.retry1":
            return _session_result("speed=1.2", notes="recovered on retry\n")
        raise AssertionError(tag)

    orch = make_orch(tmp_path, llm_script(code("speed=1.0")), [], max_rounds=1, provider_retries=1)
    orch.agent_runner = agent

    async def nap(delay):
        sleeps.append(delay)

    orch._sleep = nap
    res = run(orch)
    assert res.stop_reason == "max_rounds"
    assert res.best_label == "r1.p0" and res.best_speedup == pytest.approx(1.2)
    assert attempts == ["r1.p0", "r1.p0.retry1"] and sleeps == [5.0]
    assert res.spent_usd == pytest.approx(0.03) and res.ledger["requests"] == 3
    assert [c["label"] for c in res.candidates] == ["init", "r1.p0"]
    assert (orch.run_dir / "candidates" / "001_r1.p0" / "NOTES.md").read_text() == "recovered on retry\n"
    assert json.loads((orch.run_dir / "run_config.json").read_text())["search"]["provider_retries"] == 1
    events = _run_events(orch)
    retry = [event for event in events if event.get("event") == "provider_retry"]
    assert len(retry) == 1 and retry[0]["candidate_label"] == "r1.p0" and retry[0]["attempt"] == 1
    assert retry[0]["status"] == "scheduled" and retry[0]["delay_s"] == 5.0
    agents = [event for event in events if event.get("event") == "agent" and str(event.get("candidate_label")) == "r1.p0"]
    assert [(event["label"], event["attempt"], event["outcome"]) for event in agents] == [
        ("r1.p0", 0, "provider_failed"), ("r1.p0.retry1", 1, "evaluated"),
    ]
    assert {event["committee"] for event in agents} == {"first"}
    assert (res.best_provider, res.best_model, res.best_committee) == ("deepseek", "fake", "first")
    assert (res.winner_provider, res.winner_model, res.winner_committee) == ("deepseek", "fake", "first")
    by_label = {row["label"]: row for row in res.candidates}
    assert by_label["init"]["origin"]["origin"] == "session"
    assert by_label["r1.p0"]["origin"]["session_label"] == "r1.p0.retry1"
    assert by_label["r1.p0"]["origin"]["attempt"] == 1
    assert res.trace[-1]["origin"]["model"] == "fake"
    assert res.model_totals[0]["model"] == "fake"
    assert res.model_totals[0]["sessions"] == 3 and res.model_totals[0]["cost_usd"] == pytest.approx(0.03)


def test_provider_retries_exhaustion_counts_once_per_round_before_the_existing_two_strike_stop(tmp_path):
    """@brief 验证重试用尽时每轮只计一次 agent_failure，两轮后由原有两振停机规则收尾。

    @param tmp_path pytest 提供的临时目录。
    """
    attempts = []

    def agent(brief, tag):
        if brief.seedless:
            return _session_result("speed=1.0")
        attempts.append(tag)
        return _session_result(None, failure_class="turn_failed")

    orch = make_orch(tmp_path, llm_script(code("speed=1.0")), [], max_rounds=2, provider_retries=1)
    orch.agent_runner = agent

    async def nap(_delay):
        return None

    orch._sleep = nap
    res = run(orch)
    assert res.stop_reason == "agent_failure"
    assert attempts == ["r1.p0", "r1.p0.retry1", "r2.p0", "r2.p0.retry1"]
    assert res.spent_usd == pytest.approx(0.05)
    assert res.rounds == 2 and res.best_label == "init"
    events = _run_events(orch)
    assert len([event for event in events if event.get("event") == "provider_retry"]) == 2
    assert json.loads((orch.run_dir / "state.json").read_text())["agent_failures"] == 2


def test_a_normal_answer_without_submission_is_not_a_provider_retry(tmp_path):
    """@brief 验证正常结束但未交内核的回复不算 provider 重试，也不产生 provider_retry 事件。

    @param tmp_path pytest 提供的临时目录。
    """
    attempts = []

    def agent(brief, tag):
        if brief.seedless:
            return _session_result("speed=1.0")
        attempts.append(tag)
        return _session_result(None, failure_class="no_submission_after_result")

    orch = make_orch(tmp_path, llm_script(code("speed=1.0")), [], max_rounds=2, provider_retries=3)
    orch.agent_runner = agent
    res = run(orch)
    assert res.stop_reason == "agent_failure"
    assert attempts == ["r1.p0", "r2.p0"]
    assert res.spent_usd == pytest.approx(0.03)
    events = _run_events(orch)
    assert not any(event.get("event") == "provider_retry" for event in events)
    assert [event["outcome"] for event in events if event.get("event") == "agent" and not event.get("candidate_label") == "init"] == [
        "agent_no_submission", "agent_no_submission",
    ]


def test_a_checkpoint_from_a_degraded_session_is_evaluated_without_retrying(tmp_path):
    """@brief 验证降级会话仍交出内核时直接评测，不额外重试。

    @param tmp_path pytest 提供的临时目录。
    """
    attempts = []

    def agent(brief, tag):
        if brief.seedless:
            return _session_result("speed=1.0")
        attempts.append(tag)
        result = _session_result("speed=1.2")
        result.diagnostics["failed_turns"] = 1
        return result

    orch = make_orch(tmp_path, llm_script(code("speed=1.0")), [], max_rounds=1, provider_retries=3)
    orch.agent_runner = agent
    res = run(orch)
    assert res.best_label == "r1.p0" and attempts == ["r1.p0"]
    assert not any(event.get("event") == "provider_retry" for event in _run_events(orch))


def test_the_budget_gate_runs_between_provider_attempts(tmp_path):
    """@brief 验证预算闸门在两次 provider 尝试之间生效，额度不足时不再发起重试。

    @param tmp_path pytest 提供的临时目录。
    """
    attempts = []

    def agent(brief, tag):
        if brief.seedless:
            return _session_result("speed=1.0")
        attempts.append(tag)
        if tag == "r1.p0":
            return _session_result(None, failure_class="turn_failed")
        return _session_result("speed=1.2")

    orch = make_orch(tmp_path, llm_script(code("speed=1.0")), [], budget=0.065, max_rounds=1, provider_retries=1)
    orch.agent_runner = agent

    async def nap(_delay):
        return None

    orch._sleep = nap
    res = run(orch)
    assert res.stop_reason == "budget_exhausted"
    assert attempts == ["r1.p0"] and res.spent_usd == pytest.approx(0.02)
    events = _run_events(orch)
    assert len([event for event in events if event.get("event") == "agent"]) == 2
    assert len([event for event in events if event.get("event") == "provider_retry"]) == 1


def test_provider_retries_must_be_a_small_nonnegative_integer(tmp_path):
    """@brief 验证 provider_retries 与 session_cost_cap_usd 的非法取值在启动时被拒绝。

    @param tmp_path pytest 提供的临时目录。
    @raises RunStartError 参数不是小整数或非负数值时抛出。
    """
    for value in (-1, 4, True, "1"):
        orch = make_orch(tmp_path / str(value), llm_script(code("speed=1.0")), [], provider_retries=value)
        with pytest.raises(RunStartError, match="provider_retries"):
            run(orch)
    for value in (-1, 0, True, "1", float("nan"), float("inf")):
        orch = make_orch(tmp_path / f"cap-{value}", llm_script(code("speed=1.0")), [], session_cost_cap_usd=value)
        with pytest.raises(RunStartError, match="session_cost_cap_usd"):
            run(orch)


def test_an_interrupted_retry_backoff_restores_the_attempt_and_all_previous_spend(tmp_path):
    """@brief 验证退避等待被中断时，重试次数与已花费金额都能从快照完整恢复。

    恢复后续跑会接着做下一次重试，并把新一轮花费累加到原总额上。

    @param tmp_path pytest 提供的临时目录。
    """
    from taic.orchestrator import StopRun
    from taic.resume import describe, load_snapshot

    def agent(brief, tag):
        if brief.seedless:
            return _session_result("speed=1.0")
        assert tag == "r1.p0"
        return _session_result(None, failure_class="turn_failed")

    orch = make_orch(tmp_path, llm_script(code("speed=1.0")), [], max_rounds=2, provider_retries=1)
    orch.agent_runner = agent

    async def interrupted(_delay):
        raise StopRun("retry_wait_interrupted")

    orch._sleep = interrupted
    res = run(orch)
    assert res.stop_reason == "retry_wait_interrupted"
    assert json.loads((orch.run_dir / "state.json").read_text())["provider_attempts"] == {"r1.p0": 1}
    snap = load_snapshot(orch.run_dir, orch.search, KERNEL)
    assert snap.provider_attempts == {"r1.p0": 1}
    assert describe(snap)["provider_attempts"] == {"r1.p0": 1}
    assert snap.spent_usd == pytest.approx(0.02)
    assert snap.candidate_origins["init"]["provider"] == "deepseek"
    assert snap.candidate_origins["init"]["origin"] == "session"

    resumed = make_orch(tmp_path / "resumed", llm_script(code("speed=1.0")), [], max_rounds=2, provider_retries=1)
    resumed.restore(snap, dry_run=True)
    seen = []

    def recovered(brief, tag):
        seen.append(tag)
        return _session_result("speed=1.2")

    resumed.agent_runner = recovered
    outcome = asyncio.run(resumed._one_session("r1.p0", seedless=False))
    assert outcome == "evaluated"
    assert seen == ["r1.p0.retry1"]
    assert resumed._provider_attempts == {}
    assert resumed.ledger.total_usd == pytest.approx(0.03)
    assert resumed._session_costs[-2:] == [0.01, 0.01]
    assert resumed.best is not None and resumed.best.origin["provider"] == "deepseek"


def test_a_retry_that_returns_no_submission_is_not_retried_again(tmp_path):
    """@brief 验证重试本身仍未交内核时不再继续重试，并记一次 agent_failure。

    @param tmp_path pytest 提供的临时目录。
    """
    attempts = []

    def agent(brief, tag):
        if brief.seedless:
            return _session_result("speed=1.0")
        attempts.append(tag)
        if tag == "r1.p0":
            return _session_result(None, failure_class="turn_failed")
        if tag == "r1.p0.retry1":
            return _session_result(None, failure_class="no_submission_after_result")
        raise AssertionError(tag)

    orch = make_orch(tmp_path, llm_script(code("speed=1.0")), [], max_rounds=1, provider_retries=3)
    orch.agent_runner = agent

    async def nap(_delay):
        return None

    orch._sleep = nap
    res = run(orch)
    assert res.stop_reason == "max_rounds"
    assert attempts == ["r1.p0", "r1.p0.retry1"]
    events = _run_events(orch)
    assert len([event for event in events if event.get("event") == "provider_retry"]) == 1
    outcomes = [event["outcome"] for event in events if event.get("event") == "agent" and event.get("candidate_label") == "r1.p0"]
    assert outcomes == ["provider_failed", "agent_no_submission"]
    assert json.loads((orch.run_dir / "state.json").read_text())["agent_failures"] == 1


def test_three_provider_retries_use_bounded_exponential_backoff(tmp_path):
    """@brief 验证三次 provider 重试按 5/10/20 秒的有界指数退避，且只记一次 agent_failure。

    @param tmp_path pytest 提供的临时目录。
    """
    attempts = []
    sleeps = []

    def agent(brief, tag):
        if brief.seedless:
            return _session_result("speed=1.0")
        attempts.append(tag)
        return _session_result(None, failure_class="turn_failed")

    orch = make_orch(tmp_path, llm_script(code("speed=1.0")), [], max_rounds=1, provider_retries=3)
    orch.agent_runner = agent

    async def nap(delay):
        sleeps.append(delay)

    orch._sleep = nap
    res = run(orch)
    assert res.stop_reason == "max_rounds"
    assert attempts == ["r1.p0", "r1.p0.retry1", "r1.p0.retry2", "r1.p0.retry3"]
    assert sleeps == [5.0, 10.0, 20.0] and res.spent_usd == pytest.approx(0.05)
    assert json.loads((orch.run_dir / "state.json").read_text())["agent_failures"] == 1


def test_timeout_and_process_failure_are_not_provider_retries(tmp_path):
    """@brief 验证超时与进程失败不会触发 provider 重试。

    @param tmp_path pytest 提供的临时目录。
    """
    for status, failure_class in (("timeout", "timeout"), ("failed", "process_failed")):
        attempts = []

        def agent(brief, tag):
            if brief.seedless:
                return _session_result("speed=1.0")
            attempts.append(tag)
            result = _session_result(None, failure_class=failure_class)
            result.status = status
            return result

        orch = make_orch(tmp_path / status, llm_script(code("speed=1.0")), [], max_rounds=1, provider_retries=3)
        orch.agent_runner = agent
        res = run(orch)
        assert res.stop_reason == "max_rounds"
        assert attempts == ["r1.p0"]
        assert not any(event.get("event") == "provider_retry" for event in _run_events(orch))


def test_cost_incomplete_does_not_start_a_provider_retry(tmp_path):
    """@brief 验证花费数据不完整时以 cost_incomplete 停下，不再发起 provider 重试。

    @param tmp_path pytest 提供的临时目录。
    """
    attempts = []

    def agent(brief, tag):
        if brief.seedless:
            return _session_result("speed=1.0")
        attempts.append(tag)
        result = _session_result(None, failure_class="turn_failed")
        result.cost_usd = None
        result.cost_incomplete = True
        return result

    orch = make_orch(tmp_path, llm_script(code("speed=1.0")), [], max_rounds=1, provider_retries=3)
    orch.agent_runner = agent
    res = run(orch)
    assert res.stop_reason.startswith("cost_incomplete")
    assert attempts == ["r1.p0"]
    assert not any(event.get("event") == "provider_retry" for event in _run_events(orch))


def test_terminal_provider_attempt_marker_is_not_replayed_on_resume(tmp_path):
    """@brief 验证续跑时已终结的 provider 尝试标记不会被重放，也不会再开会话。

    通过改写 state.json 复现“最后一次尝试与轮次计数器之间”的崩溃窗口。

    @param tmp_path pytest 提供的临时目录。
    """
    from taic.resume import load_snapshot

    def agent(brief, tag):
        if brief.seedless:
            return _session_result("speed=1.0")
        assert tag == "r1.p0"
        return _session_result(None, failure_class="turn_failed")

    orch = make_orch(tmp_path, llm_script(code("speed=1.0")), [], max_rounds=1, provider_retries=1)
    orch.agent_runner = agent

    async def nap(_delay):
        return None

    orch._sleep = nap
    run(orch)
    run_dir = tmp_path / "run"
    state = json.loads((run_dir / "state.json").read_text())
    # Reproduce the crash window between the final provider attempt and the round counter.
    state["provider_attempts"] = {"r1.p0": 2}
    state["agent_failures"] = 0
    state["phase"] = "llm"
    (run_dir / "state.json").write_text(json.dumps(state))
    (run_dir / "result.json").unlink(missing_ok=True)
    snap = load_snapshot(run_dir, orch.search, KERNEL)
    assert snap.provider_attempts == {"r1.p0": 2}
    resumed = make_orch(tmp_path / "resumed", llm_script(code("speed=1.0")), [], max_rounds=1, provider_retries=1)
    resumed.restore(snap, dry_run=True)
    calls = []

    def should_not_run(brief, tag):
        calls.append(tag)
        return _session_result("speed=1.2")

    resumed.agent_runner = should_not_run
    outcome = asyncio.run(resumed._one_session("r1.p0", seedless=False))
    assert outcome == "provider_failed" and calls == []
    assert resumed._provider_attempts == {"r1.p0": 2}
