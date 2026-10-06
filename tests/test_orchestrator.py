"""Orchestrator control flow, tested with a scripted LLM and a scripted evaluator (no network, no GPU)."""

from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path

import pytest

from taic.llm import LLMConfig, SearchConfig
from taic.orchestrator import Orchestrator
from tcenv.evaluate import EvalReport

from .fakes import FakeClient, make_response, make_usage

KERNEL = "relu_kernel"


def code(marker: str, kernel: str = KERNEL) -> str:
    return f"```metal\n// {marker}\nkernel void {kernel}(device const half* x [[buffer(0)]], device half* y [[buffer(1)]]) {{}}\n```"


def fake_eval(calls: list):
    """Parses '// speed=1.3' or '// status=compile_error' markers out of the source."""

    def ev(task, dtype, size, source, *, mode="full", seed=None, reps=15, **kw):
        calls.append((mode, source))
        m = re.search(r"// (\w+)=([\w.]+)", source)
        key, val = m.group(1), m.group(2)
        r = EvalReport(task=task, dtype=dtype, size=size, mode=mode, seed=1, stage="done")
        if key == "speed":
            sp = float(val)
            r.status, r.message, r.speedup, r.speedups, r.candidate_us, r.baseline_us = "ok", "ok", sp, {"mlx": sp}, 100.0 / sp, {"mlx": 100.0}
            r.benchmark = {"candidate": {"median_us": 100.0 / sp, "cv": 0.01}, "baselines": {"mlx": {"median_us": 100.0}}, "best_baseline": "mlx"}
        elif val == "compile_error":
            r.status, r.stage, r.message, r.compile = "compile_error", "compile", "Metal compiler rejected the kernel", {"ok": False, "errors": "kernel.metal:3:5: error: boom"}
        elif val == "timeout":
            r.status, r.stage, r.message = "timeout", "timeout", "a single GPU launch ran for more than 3s"
        else:
            r.status, r.message = val, f"{val} happened"
        return r

    return ev


PLAN = '```json\n{"bottleneck":"memory","analysis":"x","plans":[{"title":"A","change":"do A"},{"title":"B","change":"do B"},{"title":"C","change":"do C"}]}\n```'


def llm_script(init: str, patches: dict[str, str] | None = None, fixes: list[str] | None = None, plan: str = PLAN, usage=None):
    """Route on the last user message."""
    patches = patches or {}
    fixes = list(fixes or [])
    seen = {"calls": []}

    def route(messages, **kw):
        last = messages[-1]["content"]
        seen["calls"].append(last[:60])
        u = usage or make_usage()
        if "Translate this Triton kernel" in last:
            return make_response(init, usage=u)
        if "Diagnose the dominant bottleneck" in last or "Reply with the JSON object only" in last:
            return make_response(plan, usage=u)
        if "Implement exactly this optimization plan" in last:
            title = re.search(r"\*\*(.+?)\*\*", last).group(1)
            return make_response(patches.get(title, code("speed=0.5")), usage=u)
        if "evaluation of your kernel failed" in last or "did not contain" in last:
            return make_response(fixes.pop(0) if fixes else code("status=compile_error"), usage=u)
        raise AssertionError("unexpected prompt: " + last[:100])

    route.seen = seen
    return route


def make_orch(tmp_path: Path, script, calls: list, *, budget=5.0, branches=3, max_rounds=2, max_repair=2, evaluate_fn=None, max_hangs=1, **search_kw):
    llm_cfg = LLMConfig(base_url="http://x", api_key="sk-test-0123456789abcdef", model="fake", reasoning_effort="off", use_tools=False)
    search = SearchConfig(budget_usd=budget, branches=branches, max_rounds=max_rounds, max_repair_tries=max_repair, max_hangs=max_hangs, eval_reps=3, **search_kw)
    return Orchestrator("relu", "fp16", "small", llm_cfg, search, run_dir=tmp_path / "run", evaluate_fn=evaluate_fn or fake_eval(calls), client=FakeClient(script), log=lambda s: None)


def run(orch):
    return asyncio.run(orch.run())


# ------------------------------------------------------------------------------------------------


def test_happy_path_selects_the_fastest_eligible_candidate_and_confirms_it(tmp_path):
    calls = []
    script = llm_script(code("speed=0.9"), patches={"A": code("speed=1.0"), "B": code("speed=1.3"), "C": code("speed=1.1")})
    orch = make_orch(tmp_path, script, calls, max_rounds=1)
    res = run(orch)
    assert res.best_label == "r1.p1" and res.best_speedup == pytest.approx(1.3)
    assert res.confirmed_speedup == pytest.approx(1.3) and res.confirmed_status == "ok"
    assert res.stop_reason == "max_rounds" and res.rounds == 1
    # trace: init 0.9 (new best), A 1.0 (new best), B 1.3 (new best), C 1.1 (not), then the confirmation is not in the trace
    assert [t["improved"] for t in res.trace] == [True, True, True, False]
    assert [t["best_speedup"] for t in res.trace] == pytest.approx([0.9, 1.0, 1.3, 1.3])
    assert all(res.trace[i]["cost_usd"] <= res.trace[i + 1]["cost_usd"] for i in range(len(res.trace) - 1))
    # artifacts
    assert (tmp_path / "run" / "best.metal").read_text().count("speed=1.3") == 1
    assert json.loads((tmp_path / "run" / "result.json").read_text())["best_label"] == "r1.p1"
    assert (tmp_path / "run" / "transcript.jsonl").exists() and (tmp_path / "run" / "ledger.json").exists()
    assert (tmp_path / "run" / "candidates" / "000_init" / "kernel.metal").exists()


def test_compile_error_is_repaired_with_the_feedback_in_the_next_prompt(tmp_path):
    calls = []
    script = llm_script(code("status=compile_error"), fixes=[code("speed=1.2")])
    orch = make_orch(tmp_path, script, calls, max_rounds=0)
    res = run(orch)
    assert res.best_label == "init.fix1" and res.best_speedup == pytest.approx(1.2)
    repair_prompt = [m for m in orch.llm.client.calls[1]["messages"] if m["role"] == "user"][-1]["content"]
    assert "repair attempt 1/2" in repair_prompt and "kernel.metal:3:5: error: boom" in repair_prompt


def test_gives_up_on_a_branch_after_max_repair_tries_but_keeps_the_seed(tmp_path):
    calls = []
    script = llm_script(code("speed=1.0"), patches={"A": code("status=incorrect"), "B": code("status=incorrect"), "C": code("status=incorrect")}, fixes=[code("status=incorrect")] * 20)
    orch = make_orch(tmp_path, script, calls, max_rounds=1, max_repair=1)
    res = run(orch)
    assert res.best_label == "init"
    assert sum(1 for c in res.candidates if c["status"] == "incorrect") == 6  # 3 branches x (first try + 1 repair)
    assert any("no valid kernel after repair" in h for h in orch.history)


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
    assert "did not contain a ```metal code block" in orch.llm.client.calls[1]["messages"][-1]["content"]


def test_budget_gate_stops_before_the_next_request_and_still_confirms(tmp_path):
    calls = []
    big = make_usage(miss=2_000_000, out=0)  # $0.60 per call
    script = llm_script(code("speed=1.1"), patches={"A": code("speed=1.5")}, usage=big)
    orch = make_orch(tmp_path, script, calls, budget=1.0, max_rounds=3)
    res = run(orch)
    assert res.stop_reason == "budget_exhausted"
    assert res.best_speedup is not None and res.confirmed_status == "ok"
    assert res.spent_usd >= 1.0 and len(orch.llm.client.calls) <= 4  # init, plan, then patches already in flight; nothing after the gate


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
    assert len([1 for m, s in calls if "speed=1.2" in s and m == "full"]) == 2  # once during the search + once for the confirmation re-run
    assert res.best_label == "init"  # the copies are not "faster than the best"


def test_unparsable_plan_falls_back_to_generic_plans(tmp_path):
    calls = []
    script = llm_script(code("speed=1.0"), plan="no json here", patches={})
    orch = make_orch(tmp_path, script, calls, max_rounds=1)
    res = run(orch)
    assert res.rounds == 1 and len([c for c in res.candidates if c["label"].startswith("r1.p")]) >= 3


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
    assert res.stop_reason.startswith("converged") and res.rounds == 4


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
    orch = make_orch(tmp_path, script, calls, branches=2, max_rounds=1, evaluate_fn=timed_eval(calls))
    res = run(orch)
    assert [t["improved"] for t in res.trace] == [True, False, False]
    assert any("A -> 1.009x (within measurement noise of the best)" in h for h in orch.history)
    # both tied kernels are re-measured; the noisy one turns out faster on re-measurement and becomes the result
    assert [c["label"] for c in res.confirmations] == ["init", "r1.p0"]
    assert res.best_label == "r1.p0" and res.confirmed_speedup == pytest.approx(0.975)
    assert "speed=1.009" in (tmp_path / "run" / "best.metal").read_text()
    assert json.loads((tmp_path / "run" / "confirm_report.json").read_text())["speedup"] == pytest.approx(0.975)


def test_clear_improvements_still_win_and_untied_candidates_are_not_re_measured(tmp_path):
    calls = []
    script = llm_script(code("speed=1.0"), patches={"A": code("speed=1.2"), "B": code("speed=1.01")})
    orch = make_orch(tmp_path, script, calls, branches=2, max_rounds=1, evaluate_fn=timed_eval(calls))
    res = run(orch)
    assert res.best_label == "r1.p0" and res.confirmed_speedup == pytest.approx(1.2)
    assert [c["label"] for c in res.confirmations] == ["r1.p0"]


# ------------------------------------------------------------------------------------------------ model request failures


def failing(base, needle: str):
    import openai

    from .test_taic_llm import status_error

    def route(messages, **kw):
        if needle in messages[-1]["content"]:
            raise status_error(openai.BadRequestError, 400)
        return base(messages, **kw)

    return route


def test_a_failed_model_request_skips_only_its_branch(tmp_path):
    calls = []
    script = failing(llm_script(code("speed=1.0"), patches={"A": code("speed=1.3")}), "**B**")
    orch = make_orch(tmp_path, script, calls, branches=2, max_rounds=2)
    res = run(orch)
    assert res.stop_reason == "max_rounds" and res.rounds == 2
    assert res.best_label == "r1.p0" and res.confirmed_status == "ok"
    assert any(h.startswith("round 1: B -> skipped") for h in orch.history)


def test_the_run_stops_when_every_branch_request_fails(tmp_path):
    calls = []
    script = failing(llm_script(code("speed=1.0")), "Implement exactly this optimization plan")
    orch = make_orch(tmp_path, script, calls, branches=2, max_rounds=3)
    res = run(orch)
    assert res.stop_reason.startswith("llm_error") and res.rounds == 1
    assert res.best_label == "init" and res.confirmed_status == "ok"


def test_a_failed_planner_request_falls_back_to_generic_plans(tmp_path):
    calls = []
    script = failing(llm_script(code("speed=1.0")), "Diagnose the dominant bottleneck")
    orch = make_orch(tmp_path, script, calls, branches=2, max_rounds=1)
    res = run(orch)
    assert res.stop_reason == "max_rounds"
    assert [c["label"] for c in res.candidates if c["label"].startswith("r1.")] == ["r1.p0", "r1.p1"]


# ------------------------------------------------------------------------------------------------ adaptive profiling


def _trace(spilled=4096, events=3, code_bytes=6000, active=0.95):
    from tcenv.xctrace import TraceSummary

    return TraceSummary(window_s=1.0, n_dispatches=40, dispatch_us_median=50.0, dispatch_us_p10=40.0, dispatch_us_p90=60.0, gpu_active_fraction=active,
                        spilled_bytes=spilled, spill_events=events, shader_code_bytes=code_bytes, thermal_states=["Nominal"])


def _user_prompts(orch, marker):
    return [m["content"] for c in orch.llm.client.calls for m in c["messages"] if m["role"] == "user" and marker in m["content"]]


def test_a_slow_seed_is_profiled_and_the_findings_reach_the_planner_and_every_patch_author(tmp_path, monkeypatch):
    from tcenv import xctrace

    traced = []
    monkeypatch.setattr(xctrace, "profile_kernel", lambda task, dtype, size, source, **kw: traced.append(source) or _trace())
    calls = []
    script = llm_script(code("speed=0.8"), patches={"A": code("speed=0.9"), "B": code("speed=0.5"), "C": code("speed=0.5")})
    orch = make_orch(tmp_path, script, calls, max_rounds=1, profile_poor=True)
    res = run(orch)
    assert len(traced) == 1 and "speed=0.8" in traced[0]  # the seed of round 1 is the initial kernel
    plan_p, patch_p = _user_prompts(orch, "Diagnose the dominant bottleneck"), _user_prompts(orch, "Implement exactly this optimization plan")
    assert len(plan_p) == 1 and len(patch_p) == 3
    for t in plan_p + patch_p:
        assert "Profiler evidence for this kernel" in t and "register spills: 3 events, 4096 bytes" in t
        assert "The compiler spilled 4096 bytes of registers" in t and "Cut live state per thread" in t
    assert res.profiles and res.profiles[0]["round"] == 1 and res.profiles[0]["spilled_bytes"] == 4096 and res.profiles[0]["seed"] == "init"
    assert (tmp_path / "run" / "profiles" / "r1_init.txt").read_text().count("spilled") >= 1
    assert json.loads((tmp_path / "run" / "result.json").read_text())["profiles"][0]["seed_speedup"] == pytest.approx(0.8)


def test_profiling_is_off_by_default_and_skipped_for_a_seed_that_beats_the_baseline(tmp_path, monkeypatch):
    from tcenv import xctrace

    monkeypatch.setattr(xctrace, "profile_kernel", lambda *a, **kw: pytest.fail("must not be traced"))
    calls = []
    off = make_orch(tmp_path / "a", llm_script(code("speed=0.8"), patches={"A": code("speed=0.9")}), calls, max_rounds=1)
    res = run(off)
    assert res.profiles == [] and not _user_prompts(off, "Profiler evidence")
    fast = make_orch(tmp_path / "b", llm_script(code("speed=1.2"), patches={"A": code("speed=1.3")}), calls, max_rounds=1, profile_poor=True)
    assert run(fast).profiles == [] and not _user_prompts(fast, "Profiler evidence")


def test_each_source_is_traced_once_and_the_number_of_traces_is_capped(tmp_path, monkeypatch):
    from tcenv import xctrace

    traced = []
    monkeypatch.setattr(xctrace, "profile_kernel", lambda task, dtype, size, source, **kw: traced.append(source) or _trace())
    calls = []
    # every round the best stays the initial kernel (patches are worse): one trace, reused by every round
    same = make_orch(tmp_path / "a", llm_script(code("speed=0.8"), patches={"A": code("speed=0.4"), "B": code("speed=0.4"), "C": code("speed=0.4")}), calls, max_rounds=3, profile_poor=True)
    run(same)
    assert len(traced) == 1 and len(_user_prompts(same, "Profiler evidence for this kernel")) == 3 + 3 * 3  # 3 planner prompts + 9 patch prompts
    # the best improves every round (a new source each time): traced until profile_max is reached, then no more
    traced.clear()
    base = llm_script(code("speed=0.5"), patches={"B": code("speed=0.4"), "C": code("speed=0.4")})
    n = {"i": 0}

    def route(messages, **kw):
        if "Implement exactly this optimization plan" in messages[-1]["content"] and "**A**" in messages[-1]["content"]:
            n["i"] += 1
            return make_response(code(f"speed={0.5 + 0.1 * n['i']:.1f} version{n['i']}"), usage=make_usage())
        return base(messages, **kw)

    improving = make_orch(tmp_path / "b", route, calls, max_rounds=4, profile_poor=True, profile_max=2)
    res = run(improving)
    assert res.best_speedup == pytest.approx(0.9) and res.rounds == 4
    assert [p["round"] for p in res.profiles] == [1, 2] and len(traced) == 2
    assert len(_user_prompts(improving, "Profiler evidence for this kernel")) == 2 * (1 + 3)  # planner + 3 patch prompts, rounds 1 and 2 only


def test_a_failing_trace_never_stops_the_search_and_is_not_retried(tmp_path, monkeypatch):
    from tcenv import xctrace

    n = {"calls": 0}

    def boom(*a, **kw):
        n["calls"] += 1
        raise xctrace.TraceError("xctrace record failed")

    monkeypatch.setattr(xctrace, "profile_kernel", boom)
    calls = []
    orch = make_orch(tmp_path, llm_script(code("speed=0.8"), patches={"A": code("speed=0.9"), "B": code("speed=0.4"), "C": code("speed=0.4")}), calls, max_rounds=2, profile_poor=True)
    res = run(orch)
    assert res.best_label == "r1.p0" and res.best_speedup == pytest.approx(0.9)
    assert res.profiles[0]["error"].startswith("TraceError") and not _user_prompts(orch, "Profiler evidence")
    assert n["calls"] == 2  # round 1 seed (init) and round 2 seed (r1.p0): each source once


def test_xctrace_advice_only_reports_what_the_trace_supports():
    from tcenv.xctrace import advice, format_for_llm

    clean = advice(_trace(spilled=0, events=0))
    assert len(clean) == 1 and "No register spills" in clean[0]
    sick = advice(_trace(spilled=100, events=2, code_bytes=100_000, active=0.5))
    text = " ".join(sick)
    assert "spilled 100 bytes" in text and "active only 50%" in text and "compiled shader is large" in text and "No register spills" not in text
    t = _trace()
    t.thermal_states = ["Nominal", "Fair"]
    assert any("throttled" in a for a in advice(t))
    assert format_for_llm(_trace()).count("findings:") == 1


# ------------------------------------------------------------------------------------------------ plan mix, initial samples, patience


def _plan_prompt(orch):
    return _user_prompts(orch, "Diagnose the dominant bottleneck")[0]


def test_plan_mix_asks_for_a_local_low_risk_plan_and_the_default_prompt_is_unchanged(tmp_path):
    calls = []
    mixed = make_orch(tmp_path / "a", llm_script(code("speed=0.9"), patches={"A": code("speed=1.0")}), calls, max_rounds=1, plan_mix=True)
    run(mixed)
    plain = make_orch(tmp_path / "b", llm_script(code("speed=0.9"), patches={"A": code("speed=1.0")}), calls, max_rounds=1)
    run(plain)
    m, p = _plan_prompt(mixed), _plan_prompt(plain)
    assert "Plans 1 to 2 are concrete structural changes" in m and "Plan 3 is the opposite: a LOCAL, low-risk change" in m
    assert "each a concrete structural change (not a micro-tweak)" in p and "LOCAL" not in p
    # same bottleneck question and the same reply format in both
    assert m.count("Reply with JSON only") == p.count("Reply with JSON only") == 1
    assert m.split("Diagnose the dominant bottleneck")[0] == p.split("Diagnose the dominant bottleneck")[0]


def test_init_samples_generates_several_initial_kernels_and_the_fastest_seeds_the_search(tmp_path):
    calls = []
    speeds = iter([0.4, 0.9, 0.6])
    base = llm_script(code("speed=0.1"), patches={"A": code("speed=0.2"), "B": code("speed=0.2"), "C": code("speed=0.2")})

    def route(messages, **kw):
        if "Translate this Triton kernel" in messages[-1]["content"]:
            return make_response(code(f"speed={next(speeds)}"), usage=make_usage())
        return base(messages, **kw)

    orch = make_orch(tmp_path, route, calls, max_rounds=0, init_samples=3)
    res = run(orch)
    init = sorted(c["label"] for c in res.candidates if c["label"].startswith("init"))
    assert init == ["init", "init.s2", "init.s3"]
    assert len(_user_prompts(orch, "Translate this Triton kernel")) == 3
    assert res.best_speedup == pytest.approx(0.9)  # the fastest of the three samples is the result and would seed round 1
    best = next(c for c in res.candidates if c["label"] == res.best_label)
    assert best["speedup"] == pytest.approx(0.9)


def test_init_samples_survive_one_failed_request_but_not_all(tmp_path):
    from taic.llm import LLMError

    calls = []
    state = {"n": 0}
    base = llm_script(code("speed=0.1"))

    def route(messages, **kw):
        if "Translate this Triton kernel" in messages[-1]["content"]:
            state["n"] += 1
            if state["n"] == 2:
                raise LLMError("boom")
            return make_response(code(f"speed={0.5 + 0.1 * state['n']:.1f}"), usage=make_usage())
        return base(messages, **kw)

    orch = make_orch(tmp_path / "a", route, calls, max_rounds=0, init_samples=3)
    res = run(orch)
    assert res.best_speedup == pytest.approx(0.8) and sum(1 for c in res.candidates if c["label"].startswith("init")) == 2

    def always(messages, **kw):
        if "Translate this Triton kernel" in messages[-1]["content"]:
            raise LLMError("down")
        return base(messages, **kw)

    res = run(make_orch(tmp_path / "b", always, calls, max_rounds=0, init_samples=2))
    assert res.stop_reason.startswith("llm_error") and res.best_label is None


def test_patience_controls_how_many_rounds_without_improvement_end_the_run(tmp_path):
    calls = []
    script = llm_script(code("speed=1.0"), patches={"A": code("speed=0.5"), "B": code("speed=0.5"), "C": code("speed=0.5")})
    res = run(make_orch(tmp_path / "a", script, calls, max_rounds=10, patience=2))
    assert res.rounds == 2 and "2 rounds without improvement" in res.stop_reason
    res = run(make_orch(tmp_path / "b", llm_script(code("speed=1.0"), patches={"A": code("speed=0.5")}), calls, max_rounds=10))
    assert res.rounds == 4 and "4 rounds without improvement" in res.stop_reason


def test_the_profiler_skill_is_only_in_the_system_prompt_of_runs_that_can_attach_a_trace(tmp_path):
    from taic import prompts
    from tasks import get_task

    task = get_task("relu", "fp16", "small")
    base = prompts.system_prompt(task, use_tools=False)
    assert "Skill 2b" not in base and "L1 Eviction Rate" not in base  # every other run keeps the byte-identical prompt
    assert prompts.system_prompt(task, use_tools=False, profiler_evidence=False) == base
    withp = prompts.system_prompt(task, use_tools=False, profiler_evidence=True)
    assert withp.startswith(base.split("# Skill 3")[0].rstrip()) and "Skill 2b" in withp and "highest limiter is the bottleneck" in withp
    calls: list = []
    plain = make_orch(tmp_path / "a", llm_script(code("speed=0.9")), calls, max_rounds=1)
    prof = make_orch(tmp_path / "b", llm_script(code("speed=0.9")), calls, max_rounds=1, profile_poor=True)
    assert "Skill 2b" not in plain.system and "Skill 2b" in prof.system
