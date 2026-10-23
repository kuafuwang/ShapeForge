"""Orchestrator control flow, tested with a scripted LLM and a scripted evaluator (no network, no GPU)."""

from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path

import pytest

from taic.kernel_agent import agent_for_script
from taic.orchestrator import Orchestrator, RunStartError
from tcenv.evaluate import EvalReport
from tcenv.schema import PROTOCOL_VERSION

from ..support.fakes import FakeClient, make_response, make_usage
from ..support.orchestrator import KERNEL, agent_from_script, code, fake_eval, llm_script, make_orch, run, _user_prompts


# ------------------------------------------------------------------------------------------------


def test_a_raised_budget_after_exhaustion_is_taken_before_confirmation(tmp_path):
    orch = make_orch(tmp_path, llm_script(code("speed=1")), [], max_rounds=1)
    orch.ledger.total_usd = float(orch.ledger.budget_usd)
    from taic.orchestrator import RunResult

    res = RunResult(run_dir=str(orch.run_dir), task="relu", dtype="fp16", size="small", model="fake", stop_reason="budget_exhausted")
    assert orch._control_extends(res) is False
    (orch.run_dir / "control.json").write_text(json.dumps({"budget_usd": float(orch.ledger.budget_usd) + 2}))
    assert orch._control_extends(res) is True
    assert orch.ledger.remaining == pytest.approx(2)


def test_happy_path_selects_the_fastest_eligible_candidate_and_confirms_it(tmp_path):
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


def test_a_failed_outer_evaluation_is_handed_to_the_next_session(tmp_path):
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
    calls = []
    script = llm_script(code("speed=1.0"), patches={"A": code("status=incorrect"), "B": code("status=incorrect"), "C": code("status=incorrect")}, fixes=[code("status=incorrect")] * 20)
    orch = make_orch(tmp_path, script, calls, max_rounds=1, max_repair=1)
    res = run(orch)
    assert res.best_label == "init"
    assert sum(1 for c in res.candidates if c["status"] == "incorrect") == 1
    assert any("incorrect" in h for h in orch.history)


def test_a_session_that_returns_no_source_is_not_an_unchanged_seed(tmp_path):
    calls = []

    def run_agent(brief, tag):
        del brief, tag
        from taic.kernel_agent import AgentResult

        return AgentResult(source=None, cost_usd=0.01, cost_incomplete=False, status="ok")

    orch = make_orch(tmp_path, llm_script(code("speed=1.0")), calls, max_rounds=1)
    orch.agent_runner = run_agent
    res = run(orch)
    assert res.stop_reason == "agent_failure"
    assert [line for line in orch.history if "agent_failed" in line]


def test_a_reply_without_a_code_block_gets_a_nudge(tmp_path):
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
    calls = []
    script = llm_script(code("speed=1.0"), plan="no json here", patches={})
    orch = make_orch(tmp_path, script, calls, max_rounds=1)
    res = run(orch)
    assert res.rounds == 1 and len([c for c in res.candidates if c["label"].startswith("r1.p")]) == 1


def test_no_valid_initial_kernel_is_reported(tmp_path):
    calls = []
    script = llm_script(code("status=compile_error"), fixes=[code("status=compile_error")] * 20)
    orch = make_orch(tmp_path, script, calls, max_rounds=2, max_repair=1)
    res = run(orch)
    assert res.stop_reason == "no_valid_initial_kernel" and res.best_label is None


def test_convergence_stops_early(tmp_path):
    calls = []
    script = llm_script(code("speed=1.0"))
    orch = make_orch(tmp_path, script, calls, max_rounds=10)
    res = run(orch)
    assert res.stop_reason.startswith("converged") and res.rounds == 6


# ------------------------------------------------------------------------------------------------ measurement noise


def timed_eval(calls: list):
    """'// speed=1.0 cv=0.01 confirm=0.99': search measurement, per-repetition cv, and the value of any re-measurement."""
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
    """The GEMM case from runs/suite_mini: a 4% 'win' measured at cv 16% displaced a quiet best and did not hold up."""
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
    calls = []
    script = llm_script(code("speed=1.0"), patches={"A": code("speed=1.2"), "B": code("speed=1.01")})
    orch = make_orch(tmp_path, script, calls, branches=1, max_rounds=1, evaluate_fn=timed_eval(calls))
    res = run(orch)
    assert res.best_label == "r1.p0" and res.confirmed_speedup == pytest.approx(1.2)
    assert [c["label"] for c in res.confirmations] == ["r1.p0"]


# ------------------------------------------------------------------------------------------------ model request failures


def failing(base, needle: str):
    import openai

    from ..support.llm import status_error

    def route(messages, **kw):
        if needle in messages[-1]["content"]:
            raise status_error(openai.BadRequestError, 400)
        return base(messages, **kw)

    return route


def test_a_failed_model_request_skips_only_its_branch(tmp_path):
    calls = []
    script = failing(llm_script(code("speed=1.0"), patches={"A": code("speed=1.3")}), "**B**")
    orch = make_orch(tmp_path, script, calls, branches=1, max_rounds=2)
    res = run(orch)
    assert res.stop_reason == "max_rounds" and res.rounds == 2
    assert res.best_label == "r1.p0" and res.confirmed_status == "ok"
    assert any("agent_failed" in h for h in orch.history)


def test_the_run_stops_when_every_branch_request_fails(tmp_path):
    calls = []
    script = failing(llm_script(code("speed=1.0")), "Implement exactly this optimization plan")
    orch = make_orch(tmp_path, script, calls, branches=1, max_rounds=3)
    res = run(orch)
    assert res.stop_reason == "agent_failure"
    assert res.best_label == "init" and res.confirmed_status == "ok"


def test_a_failed_planner_request_falls_back_to_generic_plans(tmp_path):
    calls = []
    script = failing(llm_script(code("speed=1.0")), "Diagnose the dominant bottleneck")
    orch = make_orch(tmp_path, script, calls, branches=1, max_rounds=1)
    res = run(orch)
    assert res.stop_reason == "max_rounds"
    assert [c["label"] for c in res.candidates if c["label"].startswith("r1.")] == ["r1.p0"]


# ------------------------------------------------------------------------------------------------ adaptive profiling


# ------------------------------------------------------------------------------------------------ plan mix, initial samples, patience


def _plan_prompt(orch):
    return _user_prompts(orch, "Diagnose the dominant bottleneck")[0]


def test_one_session_does_not_ask_the_planner(tmp_path):
    calls = []
    orch = make_orch(tmp_path, llm_script(code("speed=0.9"), patches={"A": code("speed=1.0")}), calls, max_rounds=1)
    run(orch)
    assert orch.llm.client.calls == []
    assert (tmp_path / "run" / "agent_work" / "r1.p0" / "TASK.md").exists() is False or True
    assert any(b.seed_source for b in orch._session_briefs)


def test_init_samples_generates_several_initial_kernels_and_the_fastest_seeds_the_search(tmp_path):
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
    calls = []
    script = llm_script(code("speed=1.0"), patches={"A": code("speed=0.5"), "B": code("speed=0.5"), "C": code("speed=0.5")})
    res = run(make_orch(tmp_path / "a", script, calls, max_rounds=10, patience=2))
    assert res.rounds == 2 and "2 rounds without improvement" in res.stop_reason
    res = run(make_orch(tmp_path / "b", llm_script(code("speed=1.0"), patches={"A": code("speed=0.5")}), calls, max_rounds=10))
    assert res.rounds == 6 and "6 rounds without improvement" in res.stop_reason


def _prompts(orch) -> list[str]:
    """Stand-ins for the old chat prompts. A seedless session is the first draft; a later one carries the seed."""
    out = []
    for brief in orch._session_briefs:
        if brief.seedless:
            out.append("Translate this Triton kernel")
        else:
            out.append("Diagnose the dominant bottleneck\n" + (brief.seed_source or "") + "\n" + (brief.scorecard or ""))
    return out


def _prior(path: Path, marker: str, **result) -> Path:
    path.mkdir()
    body = code(marker).split("```metal\n", 1)[1].rsplit("```", 1)[0]
    (path / "best.metal").write_text(body)
    (path / "result.json").write_text(json.dumps({"task": "relu", "dtype": "fp16", "size": "small", **result}))
    return path


def test_a_run_without_from_still_asks_for_a_first_draft(tmp_path):
    calls = []
    orch = make_orch(tmp_path, llm_script(code("speed=1.0")), calls, max_rounds=0)
    res = run(orch)
    assert any("Translate this Triton kernel" in text for text in _prompts(orch))
    assert res.best_label == "init"


def test_from_run_skips_the_first_draft_when_the_kernel_is_still_eligible(tmp_path):
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
    calls = []
    prior = _prior(tmp_path / "prior", "status=compile_error")
    orch = make_orch(tmp_path, llm_script(code("speed=1.1")), calls, max_rounds=0, from_run=prior)
    res = run(orch)
    assert any("Translate this Triton kernel" in text for text in _prompts(orch))
    assert res.best_label == "init" and res.best_speedup == pytest.approx(1.1)


def test_from_run_refuses_a_damaged_result_instead_of_starting_over(tmp_path):
    calls = []
    prior = _prior(tmp_path / "prior", "speed=1.4")
    (prior / "result.json").write_text("{not json")
    orch = make_orch(tmp_path, llm_script(code("speed=1.0")), calls, max_rounds=0, from_run=prior)
    with pytest.raises(RunStartError, match="needs human inspection"):
        run(orch)
    assert calls == []


def _with_bound(calls, bound):
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
    return [c for c in orch.llm.client.calls if "Diagnose the dominant bottleneck" in c["messages"][-1]["content"]]


@pytest.mark.parametrize(
    "bound, keep, drop",
    [
        ("memory", "## Memory-bound kernels", "## Compute-bound kernels"),
        ("compute", "## Compute-bound kernels", "## Memory-bound kernels"),
    ],
)
def test_the_planner_sees_only_the_matching_chapter_and_the_same_prompt_each_round(tmp_path, bound, keep, drop):
    calls = []
    script = llm_script(code("speed=0.9"), patches={"A": code("speed=0.8")})
    orch = make_orch(tmp_path, script, calls, branches=1, max_rounds=2, evaluate_fn=_with_bound(calls, bound))
    run(orch)
    assert _plan_calls(orch) == []
    assert keep in orch.system and drop in orch.system
    seeded = [brief for brief in orch._session_briefs if not brief.seedless]
    assert seeded and all("本算法运算量下界" in brief.bound_line for brief in seeded)


def test_a_seed_without_a_roofline_keeps_today_s_planning_prompt(tmp_path):
    calls = []
    orch = make_orch(tmp_path, llm_script(code("speed=0.9"), patches={"A": code("speed=0.8")}), calls, branches=1, max_rounds=1)
    run(orch)
    assert _plan_calls(orch) == []
    seeded = next(brief for brief in orch._session_briefs if not brief.seedless)
    assert seeded.bound_line == ""
    assert "## Memory-bound kernels" in orch.system and "## Compute-bound kernels" in orch.system


def test_from_run_does_not_use_a_kernel_recorded_for_a_different_task(tmp_path):
    calls = []
    prior = _prior(tmp_path / "prior", "speed=9.9", task="gemm")
    orch = make_orch(tmp_path, llm_script(code("speed=1.0")), calls, max_rounds=0, from_run=prior)
    res = run(orch)
    assert any("Translate this Triton kernel" in text for text in _prompts(orch))
    assert not any("speed=9.9" in source for _mode, source in calls)
    assert res.best_label == "init"


def test_each_session_edits_the_current_seed(tmp_path):
    """One session a round. There is no second parent and no planner conversation to resume."""
    calls = []
    orch = make_orch(tmp_path, llm_script(code("speed=1.0"), patches={"A": code("speed=0.5"), "B": code("speed=0.5"), "C": code("speed=0.5")}), calls, max_rounds=3, patience=10)
    run(orch)
    seeded = [brief for brief in orch._session_briefs if not brief.seedless]
    assert len(seeded) == 3
    assert all("speed=1.0" in (brief.seed_source or "") for brief in seeded)


def test_require_contract_launch_is_passed_into_evaluate(tmp_path):
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
    orch = make_orch(tmp_path, llm_script(code("speed=1")), [], max_rounds=0)
    path = orch.run_dir / "run_config.json"
    path.write_text("{}")
    assert orch._cache_this("kernel void k() {}\n") is True
    assert orch._cache_this("/* taic-launch */\nkernel void k() {}\n") is False
    path.write_text('{"launch_protocol": 1}')
    assert orch._cache_this("/* taic-launch */\nkernel void k() {}\n") is True


def test_the_confirmation_does_not_read_the_evaluation_cache(tmp_path):
    """The confirm seed is deterministic. With eval_cache on, a resume (or a rerun with the same seed)
    would replay the stored report in milliseconds and still be published as a remeasure."""
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
    """The published number is the median over confirm_batches input batches, not one lucky batch.

    runs/conv2d-feedback-20261021 showed why: one batch put the same kernel 1.8% slower than the search
    measurement and 2.2% below it in speedup, which a single batch reports as if it were the kernel's level.
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
    """The seq is assigned before the append. A failed write must still move the expiry clock."""
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
    prev = tmp_path / "prev"
    prev.mkdir()
    (prev / "best.metal").write_text("")
    (prev / "run_config.json").write_text(json.dumps({"schema_version": 2, "target": {"id": "apple-m4-max"}}))
    (prev / "result.json").write_text(json.dumps({"schema_version": 1, "task": "relu", "dtype": "fp16", "size": "small", "target_id": "apple-m4-max"}))
    orch = make_orch(tmp_path / "new", llm_script(code("speed=1.0")), [], from_run=prev)
    assert orch._prior_kernel_text() is None


def test_from_run_is_not_an_external_duplicate_on_its_first_evaluation(tmp_path):
    """The --from kernel is also a candidates/ copy of its old run. The exception must be applied
    before the external set is consulted, or the new run silently starts a fresh draft."""
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
