"""Metal profiling integration in the orchestrator (xctrace commands and prompt skill)."""

from __future__ import annotations

import json

import pytest

from tcenv.backends.metal.xctrace import TraceSummary, advice, format_for_llm
from ...support.fakes import make_response, make_usage
from ...support.orchestrator import _user_prompts, code, llm_script, make_orch, run


def _trace(spilled=4096, events=3, code_bytes=6000, active=0.95):
    """@brief 构造一个字段固定的 TraceSummary 作为 profile 结果。

    @param spilled 寄存器溢出字节数。
    @param events 溢出事件数。
    @param code_bytes shader 代码字节数。
    @param active GPU 活跃时间占比。
    @return TraceSummary。
    """
    from tcenv.backends.metal.xctrace import TraceSummary

    return TraceSummary(window_s=1.0, n_dispatches=40, dispatch_us_median=50.0, dispatch_us_p10=40.0, dispatch_us_p90=60.0, gpu_active_fraction=active,
                        spilled_bytes=spilled, spill_events=events, shader_code_bytes=code_bytes, thermal_states=["Nominal"])


def test_a_slow_seed_is_profiled_and_the_findings_reach_the_planner_and_every_patch_author(tmp_path, monkeypatch):
    """@brief 慢种子只 profile 一次，发现会进入规划者和每个补丁作者的提示，并写入 profiles/ 与 result.json。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 补丁工具，用于替掉真实的 xctrace。
    """
    from tcenv.backends.metal import xctrace

    traced = []
    monkeypatch.setattr(xctrace, "profile_kernel", lambda task, dtype, size, source, **kw: traced.append(source) or _trace())
    calls = []
    script = llm_script(code("speed=0.8"), patches={"A": code("speed=0.9"), "B": code("speed=0.5"), "C": code("speed=0.5")})
    orch = make_orch(tmp_path, script, calls, max_rounds=1, profile_poor=True)
    res = run(orch)
    assert len(traced) == 1 and "speed=0.8" in traced[0]  # the seed of round 1 is the initial kernel
    told = _user_prompts(orch, "Profiler evidence for this kernel")
    assert len(told) == 1
    assert "register spills: 3 events, 4096 bytes" in told[0]
    assert "The compiler spilled 4096 bytes of registers" in told[0] and "Cut live state per thread" in told[0]
    assert res.profiles and res.profiles[0]["round"] == 1 and res.profiles[0]["spilled_bytes"] == 4096 and res.profiles[0]["seed"] == "init"
    assert (tmp_path / "run" / "profiles" / "r1_init.txt").read_text().count("spilled") >= 1
    assert json.loads((tmp_path / "run" / "result.json").read_text())["profiles"][0]["seed_speedup"] == pytest.approx(0.8)


def test_profiling_is_off_by_default_and_skipped_for_a_seed_that_beats_the_baseline(tmp_path, monkeypatch):
    """@brief 默认不 profile；种子已经快于 baseline 时也跳过。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 补丁工具，用于断言 profile 不会被调用。
    """
    from tcenv.backends.metal import xctrace

    monkeypatch.setattr(xctrace, "profile_kernel", lambda *a, **kw: pytest.fail("must not be traced"))
    calls = []
    off = make_orch(tmp_path / "a", llm_script(code("speed=0.8"), patches={"A": code("speed=0.9")}), calls, max_rounds=1)
    res = run(off)
    assert res.profiles == [] and not _user_prompts(off, "Profiler evidence")
    fast = make_orch(tmp_path / "b", llm_script(code("speed=1.2"), patches={"A": code("speed=1.3")}), calls, max_rounds=1, profile_poor=True)
    assert run(fast).profiles == [] and not _user_prompts(fast, "Profiler evidence")


def test_each_source_is_traced_once_and_the_number_of_traces_is_capped(tmp_path, monkeypatch):
    """@brief 同一源码只 trace 一次并被多轮复用；出现新源码时继续 trace，直到达到 profile_max。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 补丁工具，用于记录被 trace 的源码。
    """
    from tcenv.backends.metal import xctrace

    traced = []
    monkeypatch.setattr(xctrace, "profile_kernel", lambda task, dtype, size, source, **kw: traced.append(source) or _trace())
    calls = []
    # every round the best stays the initial kernel (patches are worse): one trace, reused by every round
    same = make_orch(tmp_path / "a", llm_script(code("speed=0.8"), patches={"A": code("speed=0.4"), "B": code("speed=0.4"), "C": code("speed=0.4")}), calls, max_rounds=3, profile_poor=True)
    run(same)
    assert len(traced) == 1 and len(_user_prompts(same, "Profiler evidence for this kernel")) == 3
    # the best improves every round (a new source each time): traced until profile_max is reached, then no more
    traced.clear()
    base = llm_script(code("speed=0.5"), patches={"B": code("speed=0.4"), "C": code("speed=0.4")})
    n = {"i": 0}

    def route(messages, **kw):
        if "Implement exactly this optimization plan" in messages[-1]["content"]:
            n["i"] += 1
            return make_response(code(f"speed={0.5 + 0.1 * n['i']:.1f} version{n['i']}"), usage=make_usage())
        return base(messages, **kw)

    improving = make_orch(tmp_path / "b", route, calls, max_rounds=4, profile_poor=True, profile_max=2)
    res = run(improving)
    assert res.best_speedup == pytest.approx(0.9) and res.rounds == 4
    assert [p["round"] for p in res.profiles] == [1, 2] and len(traced) == 2
    assert len(_user_prompts(improving, "Profiler evidence for this kernel")) == 2


def test_a_failing_trace_never_stops_the_search_and_is_not_retried(tmp_path, monkeypatch):
    """@brief trace 失败不中断搜索、不重试，只记录错误；每个源码仍只尝试一次。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 补丁工具，用于让 profile_kernel 抛 TraceError。
    """
    from tcenv.backends.metal import xctrace

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
    """@brief advice 只报告 trace 支持的结论；无问题时明说未发现，format_for_llm 只给出一条 findings。
    """
    from tcenv.backends.metal.xctrace import advice, format_for_llm

    clean = advice(_trace(spilled=0, events=0))
    assert len(clean) == 1 and "Nothing was found in the recorded tables" in clean[0]
    sick = advice(_trace(spilled=100, events=2, code_bytes=100_000, active=0.5))
    text = " ".join(sick)
    assert "spilled 100 bytes" in text and "active only 50%" in text and "compiled shader is large" in text and "Nothing was found" not in text
    t = _trace()
    t.thermal_states = ["Nominal", "Fair"]
    assert any("throttled" in a for a in advice(t))
    assert format_for_llm(_trace()).count("findings:") == 1


def test_the_profiler_skill_is_only_in_the_system_prompt_of_runs_that_can_attach_a_trace(tmp_path):
    """@brief profiler 技能只出现在能附上 trace 的 run 的系统提示里，其他 run 的提示逐字节不变。

    @param tmp_path pytest 提供的临时目录。
    """
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
