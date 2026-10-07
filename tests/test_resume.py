"""Live budget top-up, the budget history, the run lock, and continuing interrupted / budget-stopped runs (scripted LLM and evaluator: no network, no GPU)."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
from pathlib import Path

import pytest

from taic import cli, control, resume as rs
from taic.llm import BudgetExhausted, CostLedger, LLMConfig, SearchConfig
from taic.orchestrator import Orchestrator
from tcenv.evaluate import EvalReport

from .fakes import FakeClient, make_response, make_usage
from .test_orchestrator import KERNEL, code, fake_eval, make_orch, run

CFG = LLMConfig(base_url="http://x", api_key="sk-test-0123456789abcdef", model="fake", reasoning_effort="off", use_tools=False)
BIG = make_usage(miss=100_000, out=50_000)  # $0.09 per request with the default prices


def ns(run_dir: Path, **kw) -> argparse.Namespace:
    base = dict(run_dir=str(run_dir), add_budget=None, budget=None, dry_run=False, who="cli", config=None, preset=None, model=None, branches=None, max_rounds=None,
                effort=None, no_tools=False, profile_best=False, profile_poor=False, plan_mix=False, init_samples=None, patience=None)
    base.update(kw)
    return argparse.Namespace(**base)


# ------------------------------------------------------------------------------------------------ control file / ledger


def test_atomic_write_leaves_no_temporary_file(tmp_path):
    control.write_json_atomic(tmp_path / "x.json", {"a": 1})
    assert json.loads((tmp_path / "x.json").read_text()) == {"a": 1}
    assert [p.name for p in tmp_path.iterdir()] == ["x.json"]


def test_ledger_takes_a_raised_budget_before_the_gate(tmp_path):
    led = CostLedger(CFG, 1.0)
    led.watch_control(tmp_path / "control.json")
    led.add(make_usage(miss=3_000_000, out=0))  # $0.90
    led.check()
    seen = []
    led.on_budget_change = lambda old, new, doc: seen.append((old, new, doc["who"]))
    led.add(make_usage(miss=1_000_000, out=0))  # $1.20 spent: over the budget
    with pytest.raises(BudgetExhausted):
        led.check()
    control.write_control(tmp_path, 5.0, who="dashboard", amount_usd=4.0, old_budget_usd=1.0)
    led.check()  # no exception: the budget is now 5
    assert led.budget_usd == 5.0 and seen == [(1.0, 5.0, "dashboard")]
    led.check()  # the unchanged file is not applied twice
    assert len(seen) == 1


def test_a_lower_request_never_goes_below_what_is_spent(tmp_path):
    led = CostLedger(CFG, 10.0)
    led.watch_control(tmp_path / "control.json")
    led.add(make_usage(miss=2_000_000, out=0))  # $0.60
    control.write_control(tmp_path, 0.1, who="cli")
    with pytest.raises(BudgetExhausted):
        led.check()
    assert led.budget_usd == pytest.approx(led.total_usd)  # clamped to the spend, not to 0.1


@pytest.mark.parametrize("content", ["{not json", "[]", '{"budget_usd": "lots"}', '{"budget_usd": -3}', '{"budget_usd": NaN}'])
def test_a_malformed_control_file_is_ignored_with_a_note(tmp_path, content):
    led = CostLedger(CFG, 2.0)
    led.watch_control(tmp_path / "control.json")
    notes = []
    led.on_control_problem = notes.append
    (tmp_path / "control.json").write_text(content)
    led.check()
    assert led.budget_usd == 2.0 and len(notes) == 1 and "ignored" in notes[0]
    led.check()
    assert len(notes) == 1  # the same damaged file is reported once


def test_a_control_file_from_an_earlier_session_is_not_applied_at_start(tmp_path):
    control.write_control(tmp_path, 99.0, who="dashboard")
    led = CostLedger(CFG, 2.0)
    led.watch_control(tmp_path / "control.json")
    led.check()
    assert led.budget_usd == 2.0


def test_events_history_and_pending_requests(tmp_path):
    control.write_control(tmp_path, 8.0, who="dashboard", amount_usd=5.0, old_budget_usd=3.0)
    h = control.budget_history(tmp_path, 3.0)
    assert h["topped_up"] is False and [e["new_budget_usd"] for e in h["pending"]] == [8.0]
    control.record_event(tmp_path, "applied", 3.0, 8.0, who="dashboard")
    h = control.budget_history(tmp_path, 8.0)
    assert h["topped_up"] is True and h["initial_budget_usd"] == 3.0 and h["pending"] == []
    ev = control.read_events(tmp_path)
    assert [e["kind"] for e in ev] == ["requested", "applied"] and ev[1]["amount_usd"] == 5.0


def test_write_control_rejects_bad_numbers(tmp_path):
    for bad in (0, -1, float("nan"), float("inf"), None, "x", True):
        with pytest.raises(control.ControlError):
            control.write_control(tmp_path, bad, who="cli")
    assert not (tmp_path / "control.json").exists()


def test_run_lock_is_exclusive_and_stale_locks_are_replaced(tmp_path):
    control.acquire_run_lock(tmp_path)
    assert control.lock_holder(tmp_path)["pid"] == os.getpid()
    control.control_path(tmp_path)  # (no effect)
    # a second process (simulated: the lock names a live pid that is not us) is refused
    (tmp_path / "resume.lock").write_text(json.dumps({"pid": os.getppid(), "started_at": 1}))
    with pytest.raises(control.RunLockError):
        control.acquire_run_lock(tmp_path)
    # a dead pid is stale
    (tmp_path / "resume.lock").write_text(json.dumps({"pid": 2**22 + 12345, "started_at": 1}))
    assert control.lock_holder(tmp_path) is None
    control.acquire_run_lock(tmp_path)
    control.release_run_lock(tmp_path)
    assert not (tmp_path / "resume.lock").exists()


# ------------------------------------------------------------------------------------------------ a live run picks up a top-up


def test_a_running_search_continues_past_its_old_budget_after_a_top_up(tmp_path):
    calls = []
    seen = {"n": 0}
    inner = FakeClient(None)

    def script(messages, **kw):
        seen["n"] += 1
        last = messages[-1]["content"]
        if seen["n"] == 2:  # the dashboard writes the control file while the run is going
            control.write_control(tmp_path / "run", 5.0, who="dashboard", amount_usd=4.5, old_budget_usd=0.5)
        if "Translate this Triton kernel" in last:
            return make_response(code("speed=1.0"), usage=BIG)
        if "Diagnose the dominant bottleneck" in last:
            return make_response('```json\n{"bottleneck":"m","analysis":"a","plans":[{"title":"A","change":"c"}]}\n```', usage=BIG)
        return make_response(code(f"speed={1.1 + 0.1 * seen['n']:.2f}"), usage=BIG)

    orch = make_orch(tmp_path, script, calls, budget=0.5, branches=1, max_rounds=6, patience=50)
    res = run(orch)
    assert res.stop_reason == "max_rounds" and res.spent_usd > 0.5  # without the top-up it stops at 0.5
    assert res.budget_usd == 5.0 and res.initial_budget_usd == 0.5
    assert [e["kind"] for e in res.budget_events] == ["applied"] and res.budget_events[0]["who"] == "dashboard"
    st = json.loads((tmp_path / "run" / "state.json").read_text())
    assert st["control_supported"] is True and st["budget_usd"] == 5.0 and st["initial_budget_usd"] == 0.5
    assert json.loads((tmp_path / "run" / "run_config.json").read_text())["search"]["budget_usd"] == 0.5


def test_a_running_search_picks_up_a_higher_round_limit(tmp_path):
    calls = []
    seen = {"n": 0}

    def script(messages, **kw):
        seen["n"] += 1
        if seen["n"] == 2:  # during round 1 the dashboard lifts the round limit without changing the budget
            control.write_control(tmp_path / "run", 5.0, who="dashboard", max_rounds=3, patience=100)
        last = messages[-1]["content"]
        if "Translate this Triton kernel" in last:
            return make_response(code("speed=1.0"), usage=BIG)
        if "Diagnose the dominant bottleneck" in last:
            return make_response('```json\n{"bottleneck":"m","analysis":"a","plans":[{"title":"A","change":"c"}]}\n```', usage=BIG)
        return make_response(code(f"speed={1.1 + 0.01 * seen['n']:.2f}"), usage=BIG)

    orch = make_orch(tmp_path, script, calls, budget=5.0, branches=1, max_rounds=1, patience=50)
    res = run(orch)
    assert res.rounds == 3 and res.stop_reason == "max_rounds"  # the control file raised the limit from 1 while round 1 was running


def test_run_config_holds_the_settings_and_no_credentials(tmp_path):
    calls = []
    orch = make_orch(tmp_path, lambda m, **kw: make_response(code("speed=1.0")), calls, max_rounds=0)
    doc = (tmp_path / "run" / "run_config.json").read_text()
    assert "sk-test" not in doc and "api_key" not in doc and "http://x" not in doc
    cfg = json.loads(doc)
    assert cfg["search"]["branches"] == 3 and cfg["llm"]["model"] == "fake" and cfg["task"] == "relu"


# ------------------------------------------------------------------------------------------------ continuing runs


def script_for(counter: dict | None = None, usage=None):
    """Deterministic, content-based scripted model: the answer depends only on the prompt, never on the order of the calls."""
    pat = {"A": ["status=compile_error", "speed=1.4"], "B": ["speed=1.2"], "C": ["status=incorrect", "status=incorrect", "status=incorrect"]}

    def route(messages, **kw):
        if counter is not None:
            counter["n"] = counter.get("n", 0) + 1
            if counter.get("kill_at") == counter["n"]:
                raise SystemExit("killed")
        u = usage or make_usage()
        last = messages[-1]["content"]
        if "Translate this Triton kernel" in last:
            return make_response(code("speed=1.0"), usage=u)
        if "Diagnose the dominant bottleneck" in last:
            rnd = max([int(m) for m in re.findall(r"^- round (\d+):", last, re.M)] or [0]) + 1  # distinct plan titles in every round
            plans = [{"title": f"{t}{rnd}", "change": f"do {t}"} for t in "ABC"]
            return make_response("```json\n" + json.dumps({"bottleneck": "m", "analysis": "a", "plans": plans}) + "\n```", usage=u)
        if "Implement exactly this optimization plan" in last:
            title = re.search(r"\*\*(.+?)\*\*", last).group(1)
            return make_response(code(f"{pat[title[0]][0]} // {title}"), usage=u)
        if "evaluation of your kernel failed" in last:
            title = re.search(r"\*\*(\w\d+)\*\*", messages[1]["content"]).group(1)
            n = int(re.search(r"repair attempt (\d+)/", last).group(1))
            seq = pat[title[0]]
            return make_response(code(f"{seq[min(n, len(seq) - 1)]} // {title}"), usage=u)
        raise AssertionError(last[:80])

    return route


def mark_dead(run_dir: Path) -> None:
    """The test process plays the killed process: after the "kill" its pid must look dead to the next one."""
    st = json.loads((run_dir / "state.json").read_text())
    st["pid"] = 2**22 + 4242
    (run_dir / "state.json").write_text(json.dumps(st))


def snapshot_of(res, orch):
    cands = sorted((c.label, c.report.status, round(c.report.speedup or 0, 6)) for c in orch.candidates)
    return {"best": res.best_label, "best_speedup": res.best_speedup, "cands": cands, "requests": res.ledger["requests"], "spent": round(res.spent_usd, 9), "history": sorted(orch.history)}


def reference(tmp_path, **kw):
    calls = []
    orch = make_orch(tmp_path / "ref", script_for(), calls, budget=5.0, max_rounds=3, max_repair=2, patience=20, **kw)
    res = run(orch)
    return snapshot_of(res, orch), len(calls)


@pytest.mark.parametrize("kill_at", range(2, 31))
def test_a_killed_run_continues_to_the_same_result_as_an_uninterrupted_one(tmp_path, kill_at):
    want, _ = reference(tmp_path)
    counter = {"kill_at": kill_at}
    calls = []
    orch = make_orch(tmp_path, script_for(counter), calls, budget=5.0, max_rounds=3, max_repair=2, patience=20)
    try:
        run(orch)
    except SystemExit:
        pass
    else:
        pytest.skip("the run finished before the kill point")
    assert not (tmp_path / "run" / "result.json").exists()
    mark_dead(tmp_path / "run")
    spent_at_kill = rs.inspect_run(tmp_path / "run")["spent_usd"]
    try:
        o2, snap, report, info, nb = cli.prepare_resume(ns(tmp_path / "run"), client=FakeClient(script_for()), evaluate_fn=fake_eval([]), log=lambda s: None)
    except cli.ResumeRefused as e:  # killed before any eligible kernel existed
        assert e.code == 3 and "nothing to continue from" in str(e)
        return
    assert report["spent_usd"] == pytest.approx(spent_at_kill)  # the ledger carries over exactly
    res = cli.run_prepared(ns(tmp_path / "run"), o2, snap, info, nb)
    got = snapshot_of(res, o2)
    assert got == want
    assert res.resumes == 1 and not (tmp_path / "run" / "resume.lock").exists()


def test_the_interrupted_round_keeps_finished_branches_and_continues_the_rest(tmp_path):
    counter = {"kill_at": 5}
    calls = []
    orch = make_orch(tmp_path, script_for(counter), calls, budget=5.0, max_rounds=3, max_repair=2, patience=20)
    with pytest.raises(SystemExit):
        run(orch)
    mark_dead(tmp_path / "run")
    asked = []

    def spy(messages, **kw):
        asked.append(messages[-1]["content"][:40])
        return script_for()(messages, **kw)

    o2, snap, report, info, nb = cli.prepare_resume(ns(tmp_path / "run", dry_run=True), client=FakeClient(spy), evaluate_fn=fake_eval([]), log=lambda s: None)
    assert report["dry_run"] is True and report["continues_in_round"] == 1
    rnd = report["round"]
    assert rnd["plans_from"].startswith("transcript") and rnd["plans"] == ["A1", "B1", "C1"]
    assert len(rnd["finished_branches"]) + len(rnd["continuing_branches"]) + len(rnd["branches_to_start"]) == 3
    assert asked == []  # a dry run asks the model nothing and writes nothing
    assert not (tmp_path / "run" / "resume.lock").exists() and len(list((tmp_path / "run").glob("state.before_resume_*"))) == 0


def test_resuming_a_run_that_stopped_for_budget_adds_budget_and_keeps_the_old_result(tmp_path):
    calls = []
    orch = make_orch(tmp_path, script_for(usage=BIG), calls, budget=0.5, max_rounds=3, max_repair=2, patience=20)
    res = run(orch)
    assert res.stop_reason == "budget_exhausted"
    run_dir = tmp_path / "run"
    first = json.loads((run_dir / "result.json").read_text())
    assert rs.inspect_run(run_dir)["kind"] == "budget_exhausted" and rs.inspect_run(run_dir)["resumable"]
    with pytest.raises(cli.ResumeRefused) as e:  # no new money: refused with the way out
        cli.prepare_resume(ns(run_dir, dry_run=True), client=FakeClient(script_for(usage=BIG)), evaluate_fn=fake_eval([]), log=lambda s: None)
    assert e.value.code == 5 and "--add-budget" in str(e.value)

    o2, snap, report, info, nb = cli.prepare_resume(ns(run_dir, add_budget=10.0, who="dashboard"), client=FakeClient(script_for(usage=BIG)), evaluate_fn=fake_eval([]), log=lambda s: None)
    assert nb == pytest.approx(10.5) and report["spent_usd"] == pytest.approx(first["spent_usd"])
    res2 = cli.run_prepared(ns(run_dir, add_budget=10.0, who="dashboard"), o2, snap, info, nb)
    assert res2.stop_reason == "max_rounds" and res2.spent_usd > first["spent_usd"] and res2.rounds == 3
    assert res2.budget_usd == pytest.approx(10.5) and res2.initial_budget_usd == pytest.approx(0.5) and res2.resumes == 1
    # the earlier result survives, untouched
    assert json.loads((run_dir / "result.before_topup_1.json").read_text()) == first
    assert (run_dir / "confirm_report.before_topup_1.json").exists() or first.get("confirmed_speedup") is None
    # round numbering and ledger carry over: labels continue, no label is used twice, the ledger lists every request of both sessions
    labels = [t["label"] for t in res2.trace]
    assert len(labels) == len(set(labels))
    assert res2.ledger["requests"] > first["ledger"]["requests"]
    ledger = json.loads((run_dir / "ledger.json").read_text())
    assert [r["total_usd"] for r in ledger["requests"]] == sorted(r["total_usd"] for r in ledger["requests"])
    ev = control.read_events(run_dir)
    assert [(e["kind"], e["who"]) for e in ev] == [("resume", "dashboard")] and ev[0]["old_budget_usd"] == 0.5 and ev[0]["new_budget_usd"] == pytest.approx(10.5)
    assert any(json.loads(line).get("event") == "resume" for line in (run_dir / "transcript.jsonl").read_text().splitlines())


@pytest.mark.parametrize("stop, why", [("converged (4 rounds without improvement)", "not for budget"), ("max_rounds", "not for budget"), ("gpu_hang_circuit_breaker", "not for budget"), ("llm_error: x", "not for budget")])
def test_runs_that_stopped_for_another_reason_are_refused(tmp_path, stop, why):
    (tmp_path / "result.json").write_text(json.dumps({"task": "relu", "dtype": "fp16", "stop_reason": stop, "spent_usd": 1.0, "budget_usd": 2.0, "best_label": "init"}))
    (tmp_path / "state.json").write_text(json.dumps({"phase": "done", "stop_reason": stop, "best_label": "init"}))
    info = rs.inspect_run(tmp_path)
    assert info["resumable"] is False and info["kind"] == "finished" and why in info["reason"]
    with pytest.raises(cli.ResumeRefused) as e:
        cli.prepare_resume(ns(tmp_path, add_budget=5.0), log=lambda s: None)
    assert e.value.code == 3


def test_a_live_run_and_a_held_lock_cannot_be_resumed(tmp_path):
    (tmp_path / "state.json").write_text(json.dumps({"pid": os.getpid(), "phase": "llm", "task": "relu", "best_label": "init", "spent_usd": 0.1, "budget_usd": 1.0, "control_supported": True}))
    info = rs.inspect_run(tmp_path)
    assert info["kind"] == "running" and info["resumable"] is False and info["can_topup_live"] is True
    (tmp_path / "state.json").write_text(json.dumps({"pid": os.getpid(), "phase": "llm", "task": "relu", "best_label": "init", "spent_usd": 0.1, "budget_usd": 1.0}))
    info = rs.inspect_run(tmp_path)
    assert info["can_topup_live"] is False and "older version" in info["live_reason"]
    (tmp_path / "state.json").write_text(json.dumps({"pid": 2**22 + 777, "phase": "llm", "task": "relu", "best_label": "init", "spent_usd": 0.1, "budget_usd": 1.0}))
    assert rs.inspect_run(tmp_path)["kind"] == "interrupted" and rs.inspect_run(tmp_path)["resumable"]
    control.acquire_run_lock(tmp_path)
    info = rs.inspect_run(tmp_path)
    assert info["kind"] == "resuming" and info["resumable"] is False
    control.release_run_lock(tmp_path)


def test_double_resume_is_refused_while_the_first_one_runs(tmp_path):
    calls = []
    orch = make_orch(tmp_path, script_for(usage=BIG), calls, budget=0.5, max_rounds=3, max_repair=2, patience=20)
    run(orch)
    run_dir = tmp_path / "run"
    holder = {}

    def script(messages, **kw):
        if "who" not in holder:  # first model request of the resumed run: a second resume starts now
            holder["who"] = True
            with pytest.raises(cli.ResumeRefused) as e:
                cli.prepare_resume(ns(run_dir, add_budget=5.0), client=FakeClient(script_for()), evaluate_fn=fake_eval([]), log=lambda s: None)
            holder["code"], holder["msg"] = e.value.code, str(e.value)
        return script_for(usage=BIG)(messages, **kw)

    o2, snap, report, info, nb = cli.prepare_resume(ns(run_dir, add_budget=5.0), client=FakeClient(script), evaluate_fn=fake_eval([]), log=lambda s: None)
    cli.run_prepared(ns(run_dir, add_budget=5.0), o2, snap, info, nb)
    assert holder["code"] == 3 and "being resumed" in holder["msg"]


# ------------------------------------------------------------------------------------------------ rebuilding from older directories


def test_settings_of_a_run_without_run_config_are_inferred_from_its_prompts(tmp_path):
    calls = []
    orch = make_orch(tmp_path, script_for(), calls, budget=5.0, max_rounds=2, max_repair=2, patience=20, plan_mix=True, init_samples=2)
    run(orch)
    run_dir = tmp_path / "run"
    exact = rs.infer_settings(run_dir)["search"]
    assert exact["plan_mix"] is True and exact["init_samples"] == 2  # run_config.json
    (run_dir / "run_config.json").unlink()
    inferred = rs.infer_settings(run_dir)
    for k in ("plan_mix", "init_samples", "branches", "max_repair_tries", "profile_poor", "max_rounds", "budget_usd"):
        assert inferred["search"][k] == exact[k], k
    assert inferred["llm"]["use_tools"] is False and inferred["llm"]["model"] == "fake"
    assert inferred["system"] and "patience" not in inferred["search"]  # not recorded by older runs


def test_the_ledger_is_rebuilt_from_the_transcript_when_state_is_behind_and_never_under_counted(tmp_path):
    counter = {"kill_at": 12}
    calls = []
    orch = make_orch(tmp_path, script_for(counter, usage=BIG), calls, budget=50.0, max_rounds=3, max_repair=2, patience=20)
    with pytest.raises(SystemExit):
        run(orch)
    run_dir = tmp_path / "run"
    state = json.loads((run_dir / "state.json").read_text())
    ev, _ = rs.read_transcript(run_dir)
    total = [e for e in ev if e["event"] == "chat"][-1]["total_usd"]
    assert total >= state["spent_usd"] - 1e-9  # the transcript is exact, state.json can lag behind it
    # the transcript is lost: state.json says more than anything we can account for
    state["spent_usd"] = total + 0.37
    (run_dir / "state.json").write_text(json.dumps(state))
    snap = rs.load_snapshot(run_dir, SearchConfig(branches=3, max_repair_tries=2), KERNEL)
    assert snap.spent_usd == pytest.approx(total + 0.37) and snap.requests[-1]["label"] == "unaccounted"
    assert any("unaccounted" in n for n in snap.notes)


def test_a_damaged_last_transcript_line_is_skipped(tmp_path):
    counter = {"kill_at": 12}
    calls = []
    orch = make_orch(tmp_path, script_for(counter), calls, budget=5.0, max_rounds=3, max_repair=2, patience=20)
    with pytest.raises(SystemExit):
        run(orch)
    run_dir = tmp_path / "run"
    with open(run_dir / "transcript.jsonl", "a") as f:
        f.write('{"event": "chat", "label": "r1.p0", "messag')
    snap = rs.load_snapshot(run_dir, SearchConfig(branches=3, max_repair_tries=2), KERNEL)
    assert any("damaged line" in n for n in snap.notes) and snap.best_label


def test_trace_entries_missing_from_state_json_are_rebuilt_by_replaying_the_selection(tmp_path):
    counter = {"kill_at": 14}
    calls = []
    orch = make_orch(tmp_path, script_for(counter), calls, budget=5.0, max_rounds=3, max_repair=2, patience=20)
    with pytest.raises(SystemExit):
        run(orch)
    run_dir = tmp_path / "run"
    state = json.loads((run_dir / "state.json").read_text())
    full = [dict(t) for t in state["trace"]]
    state["trace"] = state["trace"][:-2]  # a candidate finished after state.json was last written
    (run_dir / "state.json").write_text(json.dumps(state))
    snap = rs.load_snapshot(run_dir, SearchConfig(branches=3, max_repair_tries=2), KERNEL)
    assert [t["label"] for t in snap.trace] == [t["label"] for t in full]
    assert [t["improved"] for t in snap.trace] == [t["improved"] for t in full] or snap.best_label == next(t["label"] for t in reversed(full) if t["improved"])
    assert any("rebuilt by replaying" in n for n in snap.notes)


def test_chat_with_tools_continues_a_conversation_at_the_given_turn(tmp_path):
    from taic.llm import LLM, Transcript

    sent = []

    def route(messages, **kw):
        sent.append([m["role"] for m in messages])
        return make_response("done", usage=make_usage())

    llm = LLM(CFG, CostLedger(CFG, 5.0), Transcript(tmp_path / "t.jsonl"), client=FakeClient(route))
    conv = [{"role": "user", "content": "q"}, {"role": "assistant", "content": "", "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "lookup_msl", "arguments": "{}"}}]}]
    handlers = {"lookup_msl": lambda args: "docs"}
    reply, msgs = asyncio.run(llm.chat_with_tools(conv, [], handlers, label="r1.p0", start_turn=1, pending=[{"id": "c1", "name": "lookup_msl", "arguments": "{}"}]))
    assert reply.text() == "done" and sent == [["user", "assistant", "tool"]] and msgs[2] == {"role": "tool", "tool_call_id": "c1", "content": "docs"}
    labels = [json.loads(line)["label"] for line in (tmp_path / "t.jsonl").read_text().splitlines()]
    assert labels == ["r1.p0#t2"]


# ------------------------------------------------------------------------------------------------ the same with tool calls (quick_check) in every conversation


def script_with_tools(counter: dict | None = None):
    """Every kernel request first calls ``quick_check`` on a stub, then answers like ``script_for``; the planner answers directly."""
    base = script_for(None)

    def route(messages, **kw):
        if counter is not None:
            counter["n"] = counter.get("n", 0) + 1
            if counter.get("kill_at") == counter["n"]:
                raise SystemExit("killed")
        last = messages[-1]
        if last["role"] == "tool":
            j = max(i for i, m in enumerate(messages) if m["role"] == "user")
            return base(messages[: j + 1], **kw)
        if last["role"] == "user" and "Diagnose the dominant bottleneck" not in last["content"]:
            return make_response("let me check", tool_calls=[("call1", "quick_check", {"kernel_source": "// speed=1.0"})], usage=make_usage())
        return base(messages, **kw)

    return route


def make_tool_orch(tmp_path, script, calls):
    llm_cfg = LLMConfig(base_url="http://x", api_key="sk-test-0123456789abcdef", model="fake", reasoning_effort="off", use_tools=True)
    search = SearchConfig(budget_usd=5.0, branches=3, max_rounds=3, max_repair_tries=2, eval_reps=3, patience=20)
    return Orchestrator("relu", "fp16", "small", llm_cfg, search, run_dir=tmp_path / "run", evaluate_fn=fake_eval(calls), client=FakeClient(script), log=lambda s: None)


@pytest.mark.parametrize("kill_at", range(2, 60, 2))
def test_a_killed_run_with_tool_calls_continues_to_the_same_result(tmp_path, kill_at):
    ref_orch = make_tool_orch(tmp_path / "ref", script_with_tools(), [])
    want = snapshot_of(run(ref_orch), ref_orch)
    counter = {"kill_at": kill_at}
    orch = make_tool_orch(tmp_path, script_with_tools(counter), [])
    try:
        run(orch)
    except SystemExit:
        pass
    else:
        pytest.skip("the run finished before the kill point")
    mark_dead(tmp_path / "run")
    try:
        o2, snap, report, info, nb = cli.prepare_resume(ns(tmp_path / "run"), client=FakeClient(script_with_tools()), evaluate_fn=fake_eval([]), log=lambda s: None)
    except cli.ResumeRefused as e:
        assert e.code == 3 and "nothing to continue from" in str(e)
        return
    res = cli.run_prepared(ns(tmp_path / "run"), o2, snap, info, nb)
    assert snapshot_of(res, o2) == want
