"""Flame-chase sessions: one hmz turn a round, timed self-checks, and the outer gate."""

from __future__ import annotations

import json
from pathlib import Path

from taic.kernel_agent import AgentResult
from tcenv.selfcheck import append, earlier_hang
from tcenv.verdict import replacement_line

from .test_orchestrator import code, llm_script, make_orch, run


def _text(fenced: str) -> str:
    return fenced.split("```metal\n", 1)[1].rsplit("```", 1)[0] + "\n"


def test_two_session_failures_stop_and_one_success_clears_the_count(tmp_path):
    calls = []

    def fail(brief, tag):
        del brief, tag
        return AgentResult(source=None, cost_usd=0.01, cost_incomplete=False, status="failed", error="down")

    orch = make_orch(tmp_path / "a", llm_script(code("speed=1")), calls, max_rounds=4)
    orch.agent_runner = fail
    res = run(orch)
    assert res.stop_reason == "agent_failure"
    assert res.best_label is None

    state = {"n": 0}

    def once(brief, tag):
        del brief, tag
        state["n"] += 1
        if state["n"] == 1:
            return AgentResult(source=None, cost_usd=0.01, cost_incomplete=False, status="timeout", error="slow")
        return AgentResult(source=_text(code("speed=1.1")), cost_usd=0.01, cost_incomplete=False, status="ok")

    orch = make_orch(tmp_path / "b", llm_script(code("speed=1")), calls, max_rounds=1)
    orch.agent_runner = once
    res = run(orch)
    assert res.best_label == "init2"
    assert res.stop_reason == "max_rounds"


def test_a_finished_session_without_a_token_count_stops_the_run(tmp_path):
    calls = []

    def blind(brief, tag):
        del brief, tag
        return AgentResult(source=_text(code("speed=1.0")), cost_usd=None, cost_incomplete=True, status="ok")

    orch = make_orch(tmp_path, llm_script(code("speed=1")), calls, max_rounds=2)
    orch.agent_runner = blind
    res = run(orch)
    assert res.stop_reason.startswith("cost_incomplete")


def test_a_kernel_from_another_run_is_not_a_result(tmp_path):
    calls = []
    other = tmp_path / "runs" / "older"
    other.mkdir(parents=True)
    stolen = _text(code("speed=9.0"))
    (other / "best.metal").write_text(stolen)
    orch = make_orch(tmp_path, llm_script(code("speed=1.0")), calls, max_rounds=1)
    orch._external_roots = [tmp_path / "runs"]
    orch._external_texts = None

    def steal(brief, tag):
        source = _text(code("speed=1.0")) if brief.seedless else stolen
        return AgentResult(source=source, cost_usd=0.01, cost_incomplete=False, status="ok")

    orch.agent_runner = steal
    res = run(orch)
    assert res.best_label == "init"
    assert any(c["status"] == "external_duplicate" for c in res.candidates)


def test_a_timed_self_check_that_hangs_closes_the_workspace_and_the_search(tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    append(work, {"status": "timeout", "candidate_us": None, "cv": None, "elapsed_s": 3.0})
    assert "timeout" in (earlier_hang(work) or "")

    calls = []

    def hang(brief, tag):
        del brief
        folder = tmp_path / "run" / "agent_work" / tag
        folder.mkdir(parents=True, exist_ok=True)
        append(folder, {"status": "timeout", "elapsed_s": 2.0})
        return AgentResult(
            source=_text(code("speed=1.0")), cost_usd=0.01, cost_incomplete=False, status="timeout",
            error="slow", self_checks=1, self_check_timeouts=1, self_check_gpu_s=2.0,
        )

    orch = make_orch(tmp_path, llm_script(code("speed=1")), calls, max_rounds=3, max_hangs=1)
    orch.agent_runner = hang
    res = run(orch)
    assert res.stop_reason == "gpu_hang_circuit_breaker"
    assert res.hangs >= 1


def test_the_replacement_line_uses_the_outer_rule_and_says_the_self_check_does_not_count():
    seed = {
        "speedup": 1.0,
        "benchmark": {
            "candidate": {"cv": 0.01, "reps": 15},
            "baselines": {"mlx": {"cv": 0.01, "reps": 15}},
            "best_baseline": "mlx",
        },
    }
    fast = {"speedup": 1.2, "benchmark": seed["benchmark"]}
    close = {"speedup": 1.001, "benchmark": seed["benchmark"]}
    assert "够换种子" in replacement_line(fast, seed, min_improvement=1.005, noise_z=2.0)
    assert "不够换种子" in replacement_line(close, seed, min_improvement=1.005, noise_z=2.0)
    assert "自测不算数" in replacement_line(fast, seed, min_improvement=1.005, noise_z=2.0)
    assert "没有计时" in replacement_line({"speedup": None}, seed, min_improvement=1.005, noise_z=2.0)


def test_an_old_run_directory_cannot_be_resumed(tmp_path):
    from taic.resume import ResumeError, load_snapshot
    from taic.llm import SearchConfig

    run_dir = tmp_path / "old"
    run_dir.mkdir()
    (run_dir / "state.json").write_text(json.dumps({"task": "relu", "dtype": "fp16", "size": "small"}))
    (run_dir / "result.json").write_text(json.dumps({"task": "relu"}))
    try:
        load_snapshot(run_dir, SearchConfig(), "relu_kernel")
    except ResumeError as exc:
        assert "flame_chase_1" in str(exc)
    else:
        raise AssertionError("an old directory was resumed")
