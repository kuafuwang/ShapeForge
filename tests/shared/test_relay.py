"""Flame-chase sessions: one hmz turn a round, timed self-checks, and the outer gate."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from taic.kernel_agent import AgentResult
from tcenv.selfcheck import append, earlier_hang
from tcenv.verdict import gpu_timestamp_line, replacement_line

from ..support.orchestrator import code, llm_script, make_orch, run


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


def _records(run: Path) -> list[dict]:
    path = run / "feedback.jsonl"
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text().splitlines():
        if not line.strip().startswith("{"):
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def _kernel(speed: str) -> str:
    return _text(code(f"speed={speed}"))


_TIP = "kind: none\n\n# 给下一棒的避坑提示\n- float8 是保留字\n"
_HELP = "kind: tool, prompt\n\n# 需要帮助\n- 需要一把能读带宽的探针\n" + _TIP


def test_a_normal_session_records_feedback_once_and_copies_it(tmp_path):
    calls = []

    def write(brief, tag):
        del brief
        return AgentResult(source=_kernel("1.2"), cost_usd=0.01, cost_incomplete=False, status="ok", feedback=_HELP, notes="kept\n")

    orch = make_orch(tmp_path, llm_script(code("speed=1")), calls, max_rounds=0)
    orch.agent_runner = write
    res = run(orch)
    assert res.best_label == "init"
    rows = _records(tmp_path / "run")
    assert len(rows) == 1
    row = rows[0]
    assert row["seq"] == 1 and row["exit"] == "normal" and row["outcome"] == "evaluated"
    assert row["kinds"] == "tool,prompt" and row["label"] == "init" and row["round"] is None
    assert row["candidate_status"] == "ok" and row["work_dir"] == "agent_work/init"
    assert any(item["section"] == "需要帮助" for item in row["items"])
    copied = tmp_path / "run" / "candidates" / "000_init" / "FEEDBACK.md"
    assert copied.read_text().startswith("kind: tool")
    events = [json.loads(line) for line in (tmp_path / "run" / "transcript.jsonl").read_text().splitlines() if '"event": "agent"' in line]
    assert events[0]["feedback_kinds"] == "tool,prompt"
    assert "需要帮助" in events[0]["feedback_sections"]
    assert "float8" not in events[0]


def test_cost_incomplete_hang_exception_and_eval_error_each_record_once(tmp_path):
    calls = []

    def blind(brief, tag):
        del brief, tag
        return AgentResult(source=_kernel("1.0"), cost_usd=None, cost_incomplete=True, status="ok", feedback=_TIP)

    orch = make_orch(tmp_path / "cost", llm_script(code("speed=1")), calls, max_rounds=2)
    orch.agent_runner = blind
    res = run(orch)
    assert res.stop_reason.startswith("cost_incomplete")
    rows = _records(tmp_path / "cost" / "run")
    assert len(rows) == 1 and rows[0]["exit"] == "cost_incomplete" and rows[0]["outcome"] is None
    assert rows[0]["candidate_status"] is None
    events = [json.loads(line) for line in (tmp_path / "cost" / "run" / "transcript.jsonl").read_text().splitlines() if '"feedback_chars"' in line]
    assert len(events) == 1 and events[0]["outcome"] == "cost_incomplete"

    def hang(brief, tag):
        del brief, tag
        return AgentResult(
            source=_kernel("1.0"), cost_usd=0.01, cost_incomplete=False, status="timeout", feedback=_TIP,
            self_checks=1, self_check_timeouts=1,
        )

    orch = make_orch(tmp_path / "hang", llm_script(code("speed=1")), calls, max_rounds=3, max_hangs=1)
    orch.agent_runner = hang
    res = run(orch)
    assert res.stop_reason == "gpu_hang_circuit_breaker"
    rows = _records(tmp_path / "hang" / "run")
    assert len(rows) == 1 and rows[0]["exit"] == "hang_circuit_breaker" and rows[0]["outcome"] is None
    assert rows[0]["candidate_status"] is None

    def boom(brief, tag):
        folder = tmp_path / "boom" / "run" / "agent_work" / tag
        folder.mkdir(parents=True, exist_ok=True)
        if tag == "init":
            (folder / "FEEDBACK.md").write_text(_TIP)
        raise RuntimeError("hmz down")

    orch = make_orch(tmp_path / "boom", llm_script(code("speed=1")), calls, max_rounds=2)
    orch.agent_runner = boom
    res = run(orch)
    assert res.stop_reason == "agent_failure"
    rows = _records(tmp_path / "boom" / "run")
    assert [row["exit"] for row in rows] == ["agent_exception", "agent_exception"]
    assert rows[0]["outcome"] is None and "float8" in rows[0]["raw"]
    assert rows[1]["items"] == []

    def crash_eval(task, dtype, size, source, **kw):
        raise RuntimeError("eval broke")

    def fine(brief, tag):
        del brief, tag
        return AgentResult(source=_kernel("1.0"), cost_usd=0.01, cost_incomplete=False, status="ok", feedback=_TIP)

    orch = make_orch(tmp_path / "eval", llm_script(code("speed=1")), calls, max_rounds=1, evaluate_fn=crash_eval)
    orch.agent_runner = fine
    with pytest.raises(RuntimeError, match="eval broke"):
        run(orch)
    rows = _records(tmp_path / "eval" / "run")
    assert len(rows) == 1 and rows[0]["exit"] == "eval_error" and rows[0]["outcome"] is None


def test_a_stale_feedback_file_is_removed_before_the_session_and_not_recorded(tmp_path):
    calls = []
    seen = {}

    def look(brief, tag):
        folder = tmp_path / "run" / "agent_work" / tag
        seen[tag] = {"feedback": (folder / "FEEDBACK.md").exists(), "notes": (folder / "NOTES.md").exists()}
        raise RuntimeError("stop")

    orch = make_orch(tmp_path, llm_script(code("speed=1")), calls, max_rounds=0)
    stale = orch.run_dir / "agent_work" / "init"
    stale.mkdir(parents=True)
    (stale / "FEEDBACK.md").write_text("STALE old premise\n")
    (stale / "NOTES.md").write_text("old notes\n")
    orch.agent_runner = look
    run(orch)
    assert seen["init"] == {"feedback": False, "notes": False}
    rows = _records(tmp_path / "run")
    assert rows and all("STALE" not in row.get("raw", "") for row in rows)
    assert all(row["exit"] == "agent_exception" and row["items"] == [] for row in rows)


def test_peer_tips_pass_within_one_run_and_a_switch_turns_them_off(tmp_path):
    calls = []
    seen = {}

    def handoff(brief, tag):
        seen[tag] = brief.peer_tips
        if tag == "init":
            return AgentResult(source=None, cost_usd=0.01, cost_incomplete=False, status="failed", feedback=_TIP)
        return AgentResult(source=_kernel("1.1"), cost_usd=0.01, cost_incomplete=False, status="ok")

    orch = make_orch(tmp_path / "hand", llm_script(code("speed=1")), calls, max_rounds=0)
    orch.agent_runner = handoff
    res = run(orch)
    assert res.best_label == "init2"
    assert seen["init"] == ""
    assert "float8" in seen["init2"] and "未验证" not in seen["init2"] or "float8" in seen["init2"]

    premise = "kind: none\n\n# 被否决的前提\n- [未验证] 矩阵单元只能吃 fp16\n"
    speeds = {"n": 0}

    def restate(brief, tag):
        seen[tag] = brief.peer_tips
        if tag == "init":
            return AgentResult(source=_kernel("1.0"), cost_usd=0.01, cost_incomplete=False, status="ok")
        speeds["n"] += 1
        return AgentResult(source=_kernel(f"{1.0 + speeds['n'] * 0.01:.2f}"), cost_usd=0.01, cost_incomplete=False, status="ok", feedback=premise)

    orch = make_orch(tmp_path / "age", llm_script(code("speed=1")), calls, max_rounds=4)
    orch.agent_runner = restate
    run(orch)
    assert "fp16" in seen["r2.p0"] and "fp16" in seen["r3.p0"]
    assert "fp16" not in seen["r4.p0"]

    def quiet(brief, tag):
        seen["off-" + tag] = brief.peer_tips
        if tag == "init":
            return AgentResult(source=None, cost_usd=0.01, cost_incomplete=False, status="failed", feedback=_TIP)
        return AgentResult(source=_kernel("1.1"), cost_usd=0.01, cost_incomplete=False, status="ok")

    orch = make_orch(tmp_path / "off", llm_script(code("speed=1")), calls, max_rounds=0, peer_tips=False)
    orch.agent_runner = quiet
    run(orch)
    assert seen["off-init2"] == ""
    assert json.loads((tmp_path / "off" / "run" / "run_config.json").read_text())["search"]["peer_tips"] is False


def test_silent_sessions_count_toward_the_two_session_premise_window(tmp_path):
    calls = []
    seen = {}
    kernel = _kernel("1.0")
    premise = "kind: none\n\n# 被否决的前提\n- [未验证] 矩阵单元只能吃 fp16\n"

    def write(brief, tag):
        seen[tag] = brief.peer_tips
        return AgentResult(
            source=kernel, cost_usd=0.01, cost_incomplete=False, status="ok",
            feedback=premise if tag == "init" else "",
        )

    orch = make_orch(tmp_path, llm_script(code("speed=1")), calls, max_rounds=3)
    orch.agent_runner = write
    run(orch)
    assert "fp16" in seen["r1.p0"] and "fp16" in seen["r2.p0"]
    assert "fp16" not in seen["r3.p0"]
    rows = _records(tmp_path / "run")
    assert [row["seq"] for row in rows] == [1, 2, 3, 4]
    assert rows[1]["items"] == [] and rows[2]["items"] == []


def test_peer_tips_has_a_command_line_switch_and_resume_keeps_the_recorded_value(monkeypatch):
    from taic import cli

    got = {}
    monkeypatch.setattr(cli, "cmd_run", lambda args: got.update(run=vars(args)) or 0)
    monkeypatch.setattr(cli, "cmd_resume", lambda args: got.update(resume=vars(args)) or 0)
    assert cli.main(["run", "relu", "--no-peer-tips"]) == 0
    assert got["run"]["peer_tips"] is False
    assert cli.main(["resume", "somewhere", "--peer-tips"]) == 0
    assert got["resume"]["peer_tips"] is True
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-0123456789abcdef")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://x")
    from ..support.resume import ns

    settings = {"search": {"peer_tips": False}, "llm": {"model": "fake"}, "provenance": {"search.peer_tips": "run_config.json"}}
    _cfg, search, prov = cli._resume_config(ns(Path("somewhere"), dry_run=True), settings, {"model": None})
    assert search.peer_tips is False
    assert prov["search.peer_tips"] == "run_config.json"


def test_provider_retries_has_a_cli_switch_and_resume_preserves_or_overrides_it(monkeypatch):
    from taic import cli
    from ..support.resume import ns

    got = {}
    monkeypatch.setattr(cli, "cmd_run", lambda args: got.update(run=vars(args)) or 0)
    monkeypatch.setattr(cli, "cmd_resume", lambda args: got.update(resume=vars(args)) or 0)
    assert cli.main(["run", "relu", "--provider-retries", "2"]) == 0
    assert got["run"]["provider_retries"] == 2
    assert cli.main(["resume", "somewhere", "--provider-retries", "0"]) == 0
    assert got["resume"]["provider_retries"] == 0
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-0123456789abcdef")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://x")
    settings = {"search": {"provider_retries": 2}, "llm": {"model": "fake"},
                "provenance": {"search.provider_retries": "run_config.json"}}
    _cfg, search, prov = cli._resume_config(ns(Path("somewhere"), dry_run=True), settings, {"model": None})
    assert search.provider_retries == 2 and prov["search.provider_retries"] == "run_config.json"
    _cfg, search, prov = cli._resume_config(ns(Path("somewhere"), dry_run=True, provider_retries=0), settings, {"model": None})
    assert search.provider_retries == 0 and prov["search.provider_retries"] == "command line"


def test_feedback_seq_continues_after_a_broken_line(tmp_path):
    calls = []

    def write(brief, tag):
        del brief, tag
        return AgentResult(source=_kernel("1.0"), cost_usd=0.01, cost_incomplete=False, status="ok", feedback=_TIP)

    orch = make_orch(tmp_path, llm_script(code("speed=1")), calls, max_rounds=0)
    (orch.run_dir / "feedback.jsonl").write_text(json.dumps({"format": "agent_feedback", "v": 1, "seq": 4, "items": []}) + "\n{broken\n")
    orch.agent_runner = write
    run(orch)
    rows = _records(orch.run_dir)
    assert [row["seq"] for row in rows] == [4, 5]


def test_gpu_timestamp_line_compares_seed_and_candidate():
    seed = {"benchmark": {"candidate_gpu_us_median": 300.03}}
    candidate = {"benchmark": {"candidate_gpu_us_median": 342.94}}
    line = gpu_timestamp_line(candidate, seed)
    assert "0.875x" in line and "seed 300.03 us" in line and "candidate 342.94 us" in line
    assert gpu_timestamp_line({}, seed) == ""
