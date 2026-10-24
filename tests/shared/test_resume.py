"""Live budget top-up, the budget history, the run lock, and continuing interrupted / budget-stopped runs (scripted LLM and evaluator: no network, no GPU)."""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from taic import cli, control, resume as rs
from taic.llm import BudgetExhausted, CostLedger, LLMConfig, SearchConfig
from taic.orchestrator import Orchestrator, RunStartError
from tcenv.evaluate import EvalReport

from ..support.fakes import FakeClient, make_response, make_usage
from ..support.orchestrator import KERNEL, code, fake_eval, llm_script, make_orch, run
from ..support.resume import mark_dead, ns, script_for

CFG = LLMConfig(base_url="http://x", api_key="sk-test-0123456789abcdef", model="fake", reasoning_effort="off", use_tools=False)
BIG = make_usage(miss=100_000, out=50_000)  # $0.09 per request with the default prices


# ------------------------------------------------------------------------------------------------ control file / ledger


def test_atomic_write_leaves_no_temporary_file(tmp_path):
    """@brief 验证 control.write_json_atomic 写完控制文件后目录里只剩目标文件，没有残留的临时文件。

    @param tmp_path pytest 提供的临时目录。
    """
    control.write_json_atomic(tmp_path / "x.json", {"a": 1})
    assert json.loads((tmp_path / "x.json").read_text()) == {"a": 1}
    assert [p.name for p in tmp_path.iterdir()] == ["x.json"]


def test_ledger_takes_a_raised_budget_before_the_gate(tmp_path):
    """@brief 验证 CostLedger 在预算闸门之前读入调高后的 control.json：闸门不再抛 BudgetExhausted，on_budget_change 只回调一次，同一份文件不会二次生效。

    @param tmp_path pytest 提供的临时目录。
    """
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
    """@brief 验证调低预算的请求不会把上限压到已花金额以下：预算被夹到 total_usd，闸门仍然抛 BudgetExhausted。

    @param tmp_path pytest 提供的临时目录。
    """
    led = CostLedger(CFG, 10.0)
    led.watch_control(tmp_path / "control.json")
    led.add(make_usage(miss=2_000_000, out=0))  # $0.60
    control.write_control(tmp_path, 0.1, who="cli")
    with pytest.raises(BudgetExhausted):
        led.check()
    assert led.budget_usd == pytest.approx(led.total_usd)  # clamped to the spend, not to 0.1


@pytest.mark.parametrize("content", ["{not json", "[]", '{"budget_usd": "lots"}', '{"budget_usd": -3}', '{"budget_usd": NaN}'])
def test_a_malformed_control_file_is_ignored_with_a_note(tmp_path, content):
    """@brief 验证内容损坏的 control.json 被忽略并只记一条提示，预算保持原值，同一份坏文件不会重复报警。

    @param tmp_path pytest 提供的临时目录。
    @param content 参数化给出的损坏 control.json 内容。
    """
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
    """@brief 验证启动前就存在的旧 control.json 不会被新会话在开头套用，预算仍是构造时给出的 2.0。

    @param tmp_path pytest 提供的临时目录。
    """
    control.write_control(tmp_path, 99.0, who="dashboard")
    led = CostLedger(CFG, 2.0)
    led.watch_control(tmp_path / "control.json")
    led.check()
    assert led.budget_usd == 2.0


def test_events_history_and_pending_requests(tmp_path):
    """@brief 验证 budget_history 的 topped_up、initial_budget_usd 与 pending 待生效请求，以及 read_events 记下的 requested 与 applied 事件及金额。

    @param tmp_path pytest 提供的临时目录。
    """
    control.write_control(tmp_path, 8.0, who="dashboard", amount_usd=5.0, old_budget_usd=3.0)
    h = control.budget_history(tmp_path, 3.0)
    assert h["topped_up"] is False and [e["new_budget_usd"] for e in h["pending"]] == [8.0]
    control.record_event(tmp_path, "applied", 3.0, 8.0, who="dashboard")
    h = control.budget_history(tmp_path, 8.0)
    assert h["topped_up"] is True and h["initial_budget_usd"] == 3.0 and h["pending"] == []
    ev = control.read_events(tmp_path)
    assert [e["kind"] for e in ev] == ["requested", "applied"] and ev[1]["amount_usd"] == 5.0
    control.record_event(tmp_path, "requested", 8.0, 8.0, who="dashboard")
    h = control.budget_history(tmp_path, 8.0)
    assert [e["new_budget_usd"] for e in h["pending"]] == [8.0]


def test_write_control_rejects_bad_numbers(tmp_path):
    """@brief 验证 write_control 拒绝非法金额（0、负数、NaN、Inf、None、字符串、布尔），且不留下 control.json。

    @param tmp_path pytest 提供的临时目录。
    """
    for bad in (0, -1, float("nan"), float("inf"), None, "x", True):
        with pytest.raises(control.ControlError):
            control.write_control(tmp_path, bad, who="cli")
    assert not (tmp_path / "control.json").exists()


def test_run_lock_is_exclusive_and_stale_locks_are_replaced(tmp_path):
    """@brief 验证运行锁的独占性与失效判定：活着的其他 pid 被拒，已死 pid 视为过期可覆盖，空锁文件也能重新加锁并释放。

    @param tmp_path pytest 提供的临时目录。
    """
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
    (tmp_path / "resume.lock").write_text("")
    control.acquire_run_lock(tmp_path)
    assert json.loads((tmp_path / "resume.lock").read_text())["pid"] == os.getpid()
    control.release_run_lock(tmp_path)


def test_two_processes_cannot_both_hold_the_run_lock(tmp_path):
    """@brief 用两个真实子进程验证同一运行目录的运行锁只能被一个进程持有，第二个进程拿到 RunLockError。

    @param tmp_path pytest 提供的临时目录。
    """
    script = r"""
import os, sys, time
sys.path.insert(0, os.environ["REPO"])
from pathlib import Path
from taic import control
try:
    control.acquire_run_lock(Path(os.environ["RUN"]))
    print("GOT", flush=True)
    time.sleep(30)
except control.RunLockError:
    print("REFUSED", flush=True)
"""
    env = os.environ.copy()
    env["REPO"] = str(Path(__file__).resolve().parents[2])
    env["RUN"] = str(tmp_path)
    holder = subprocess.Popen([sys.executable, "-c", script], env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        assert holder.stdout.readline().strip() == "GOT"
        other = subprocess.Popen([sys.executable, "-c", script], env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        out, err = other.communicate(timeout=5)
        assert (out.strip() or err.strip()) == "REFUSED"
    finally:
        holder.kill()
        holder.wait(timeout=5)


def test_a_budget_that_clears_the_file_spend_but_not_the_transcript_is_refused(tmp_path):
    """@brief 验证续跑预算只盖得住 state.json 的已花金额、却盖不住 transcript 里实际花掉的钱时被拒绝（ResumeRefused，退出码 5）。

    @param tmp_path pytest 提供的临时目录。
    """
    run = tmp_path / "run"
    folder = run / "candidates" / "000_init"
    folder.mkdir(parents=True)
    (folder / "kernel.metal").write_text("kernel void k_init() {}\n")
    (folder / "report.json").write_text(json.dumps(_ok_report(1.1).to_json()))
    (run / "state.json").write_text(json.dumps({
        "task": "relu", "dtype": "fp16", "size": "small", "phase": "llm", "pid": 2**22 + 3,
        "best_label": "init", "spent_usd": 1.0, "budget_usd": 5.0,
        "trace": [{"idx": 0, "label": "init", "improved": True, "speedup": 1.1, "cost_usd": 0.01, "t_s": 1.0}],
    }))
    (run / "transcript.jsonl").write_text(json.dumps({"event": "chat", "label": "init", "cost_usd": 4.0, "total_usd": 4.0, "usage": {}}) + "\n")
    (run / "run_config.json").write_text(json.dumps({"loop_protocol": "flame_chase_1"}))
    with pytest.raises(cli.ResumeRefused) as caught:
        cli.prepare_resume(ns(run, dry_run=True, budget=2.0), log=lambda _s: None)
    assert caught.value.code == 5 and "4.0000" in str(caught.value)


# ------------------------------------------------------------------------------------------------ a live run picks up a top-up


def test_a_running_search_continues_past_its_old_budget_after_a_top_up(tmp_path):
    """@brief 验证运行中的搜索在别人写下 control.json 调高预算后越过原预算继续跑，结果与 state.json 都记下新预算、初始预算与 dashboard 事件。

    @param tmp_path pytest 提供的临时目录。
    """
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
    """@brief 验证运行中的搜索能读入 control.json 抬高的轮数上限与耐心值，在第 1 轮还在跑时就把 max_rounds 从 1 提到 3。

    @param tmp_path pytest 提供的临时目录。
    """
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
    """@brief 验证 run_config.json 记录搜索与模型设置以及 loop_protocol，且不含 API key、base_url 等凭据。

    @param tmp_path pytest 提供的临时目录。
    """
    calls = []
    orch = make_orch(tmp_path, lambda m, **kw: make_response(code("speed=1.0")), calls, max_rounds=0)
    doc = (tmp_path / "run" / "run_config.json").read_text()
    assert "sk-test" not in doc and "api_key" not in doc and "http://x" not in doc
    cfg = json.loads(doc)
    assert cfg["search"]["branches"] == 1 and cfg["llm"]["model"] == "fake" and cfg["task"] == "relu"
    assert cfg["loop_protocol"] == "flame_chase_1"


# ------------------------------------------------------------------------------------------------ continuing runs


def snapshot_of(res, orch):
    """@brief 把一次运行的结论压成可比较的快照，用于判定中断续跑与一次跑完是否等价。

    @param res 运行结束得到的 RunResult。
    @param orch 产生该结果的 Orchestrator。
    @return 含 best、best_speedup、cands、requests、spent 与 history 的字典。
    """
    cands = sorted((c.label, c.report.status, round(c.report.speedup or 0, 6)) for c in orch.candidates)
    return {"best": res.best_label, "best_speedup": res.best_speedup, "cands": cands, "requests": res.ledger["requests"], "spent": round(res.spent_usd, 9), "history": sorted(orch.history)}


def reference(tmp_path, **kw):
    """@brief 跑一次不被打断的参照搜索，返回它的快照与模型调用次数。

    @param tmp_path pytest 提供的临时目录，运行目录建在它的 ref 子目录下。
    @param kw 传给 make_orch 的额外搜索配置。
    @return (快照字典, 模型调用次数) 二元组。
    """
    calls = []
    orch = make_orch(tmp_path / "ref", script_for(), calls, budget=5.0, max_rounds=3, max_repair=2, patience=20, **kw)
    res = run(orch)
    return snapshot_of(res, orch), len(calls)


@pytest.mark.parametrize("kill_at", [2])
def test_a_killed_run_continues_to_the_same_result_as_an_uninterrupted_one(tmp_path, kill_at):
    """@brief 验证在第 kill_at 次模型调用处被杀的运行，续跑后得到与一次跑完完全相同的结论，花费、轮次也对得上。

    @param tmp_path pytest 提供的临时目录。
    @param kill_at 参数化的杀点：第几次模型调用时中断。
    """
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
    """@brief 验证被打断那轮里已完成的候选被保留、只续做剩下的分支，且 dry-run 不调模型、不写运行锁、不留 state 备份。

    @param tmp_path pytest 提供的临时目录。
    """
    counter = {"kill_at": 2}
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
    assert report["dry_run"] is True and report["continues_in_round"] >= 1
    assert report["round"] is None  # a new session, not a half-written branch conversation
    assert asked == []  # a dry run asks the model nothing and writes nothing
    assert not (tmp_path / "run" / "resume.lock").exists() and len(list((tmp_path / "run").glob("state.before_resume_*"))) == 0


def test_a_report_from_another_protocol_epoch_cannot_be_resumed(tmp_path):
    """@brief 验证候选 report.json 的 protocol_version 不属于当前协议时，load_snapshot 抛 NeedsInspection 而不是续跑。

    @param tmp_path pytest 提供的临时目录。
    """
    orch = make_orch(tmp_path, llm_script(code("speed=1.2")), [], max_rounds=1)
    run(orch)
    run_dir = tmp_path / "run"
    report_path = next((run_dir / "candidates").glob("*/report.json"))
    doc = json.loads(report_path.read_text())
    doc["protocol_version"] = "0"
    report_path.write_text(json.dumps(doc))
    (run_dir / "result.json").unlink()
    with pytest.raises(rs.NeedsInspection):
        rs.load_snapshot(run_dir, orch.search, KERNEL)


def test_a_damaged_confirm_report_is_refused_before_any_file_moves(tmp_path):
    """@brief 验证 confirm_report.json 损坏时 snapshot_results 在任何文件移动之前就抛 ResumeError。

    旧实现先挪走 result.json 再解析 confirm_report.json，结果坏报告让运行既没有 result.json、又只留下 traceback，
    而不是干脆地拒绝；这里断言 result.json 还在、也没有生成 result.before_topup_1.json。

    @param tmp_path pytest 提供的临时目录。
    """
    run = tmp_path / "run"
    run.mkdir()
    (run / "result.json").write_text('{"task": "relu"}')
    (run / "confirm_report.json").write_text("{not json")
    with pytest.raises(rs.ResumeError):
        rs.snapshot_results(run)
    assert (run / "result.json").is_file()
    assert not (run / "result.before_topup_1.json").exists()


def test_resume_exits_6_when_the_confirmation_failed(tmp_path, monkeypatch):
    """@brief 验证确认阶段失败（没有 confirmed_speedup、stop_reason 为 confirm_error）时 taic resume 以退出码 6 结束。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 的替换工具，用来桩掉 prepare_resume 与 run_prepared。
    """
    from types import SimpleNamespace

    monkeypatch.setattr(cli, "prepare_resume", lambda a: (SimpleNamespace(log=lambda s: None), None, {"unrecorded_settings": []}, {}, 5.0))
    monkeypatch.setattr(cli, "run_prepared", lambda a, orch, snap, info, nb: SimpleNamespace(
        task="relu", dtype="fp16", model="fake", best_speedup=1.2, confirmed_speedup=None,
        spent_usd=0.1, budget_usd=5.0, stop_reason="confirm_error: RuntimeError: boom", run_dir=str(tmp_path),
    ))
    assert cli.main(["resume", str(tmp_path)]) == 6


def test_a_rejected_candidate_is_not_replayed_as_the_resumed_seed(tmp_path):
    """@brief 验证被 differential_recheck 判定为错误的候选在重建快照后仍标着 recheck_rejected，不会再次成为续跑的种子。

    旧续跑路径强制 relation_to_best(True, ...)，会把已知有问题的内核重新交回去当种子；这里断言 best_label 为空、
    notes 里留下 differential recheck 的说明。

    @param tmp_path pytest 提供的临时目录。
    """
    orch = make_orch(tmp_path, llm_script(code("speed=1.2")), [], max_rounds=1)
    run(orch)
    run_dir = tmp_path / "run"
    label = json.loads((run_dir / "state.json").read_text())["search_seed_label"]
    assert label
    folder = next(p for p in (run_dir / "candidates").iterdir() if p.name.endswith("_" + label))
    (folder / "differential_recheck.json").write_text(json.dumps({"counts_as_kernel_error": True}))
    (run_dir / "result.json").unlink()
    snap = rs.load_snapshot(run_dir, orch.search, KERNEL)
    assert snap.candidates and snap.candidates[0]["recheck_rejected"] is True
    assert snap.best_label is None
    assert any("differential recheck" in note for note in snap.notes)


def test_a_run_killed_during_confirmation_resumes_without_a_new_session(tmp_path):
    """@brief 验证在确认阶段被杀的运行续跑时只补做确认、不再开新会话（agent_runner 一旦被调用就断言失败），最后照常写出 result.json。

    最后一轮已经跑完的搜索只差确认这一步。

    @param tmp_path pytest 提供的临时目录。
    """
    orch = make_orch(tmp_path, llm_script(code("speed=1.2")), [], max_rounds=1)
    run(orch)
    run_dir = tmp_path / "run"
    (run_dir / "result.json").unlink()
    state = json.loads((run_dir / "state.json").read_text())
    state["phase"] = "confirm"
    (run_dir / "state.json").write_text(json.dumps(state))
    mark_dead(run_dir)
    snap = rs.load_snapshot(run_dir, orch.search, KERNEL)
    assert snap.search_finished is True and snap.next_round <= orch.search.max_rounds
    assert rs.inspect_run(run_dir)["resumable"] is True

    def must_not_start(*_a, **_k):
        raise AssertionError("a run killed during confirmation must not start a search session")

    o2, snap2, report, info, nb = cli.prepare_resume(
        ns(run_dir, add_budget=5.0), client=FakeClient([]), evaluate_fn=fake_eval([]), log=lambda s: None,
    )
    o2.agent_runner = must_not_start
    res = cli.run_prepared(ns(run_dir, add_budget=5.0, who="test"), o2, snap2, info, nb)
    assert (run_dir / "result.json").is_file()
    assert res.stop_reason == "max_rounds"


def test_resuming_a_run_that_stopped_for_budget_adds_budget_and_keeps_the_old_result(tmp_path):
    """@brief 验证因预算耗尽的运行在加钱后能续跑：旧 result 与 confirm_report 备份成 before_topup_1、预算与初始预算正确、轮次与账本连续、标签不重复、transcript 记下 resume 事件。

    @param tmp_path pytest 提供的临时目录。
    """
    calls = []
    orch = make_orch(tmp_path, script_for(usage=BIG), calls, budget=0.15, max_rounds=3, max_repair=2, patience=20)
    res = run(orch)
    assert res.stop_reason == "budget_exhausted"
    run_dir = tmp_path / "run"
    first = json.loads((run_dir / "result.json").read_text())
    assert rs.inspect_run(run_dir)["kind"] == "budget_exhausted" and rs.inspect_run(run_dir)["resumable"]
    with pytest.raises(cli.ResumeRefused) as e:  # no new money: refused with the way out
        cli.prepare_resume(ns(run_dir, dry_run=True), client=FakeClient(script_for(usage=BIG)), evaluate_fn=fake_eval([]), log=lambda s: None)
    assert e.value.code == 5 and "--add-budget" in str(e.value)

    o2, snap, report, info, nb = cli.prepare_resume(ns(run_dir, add_budget=10.0, who="dashboard"), client=FakeClient(script_for(usage=BIG)), evaluate_fn=fake_eval([]), log=lambda s: None)
    assert nb == pytest.approx(10.15) and report["spent_usd"] == pytest.approx(first["spent_usd"])
    res2 = cli.run_prepared(ns(run_dir, add_budget=10.0, who="dashboard"), o2, snap, info, nb)
    assert res2.stop_reason == "max_rounds" and res2.spent_usd > first["spent_usd"] and res2.rounds == 3
    assert res2.budget_usd == pytest.approx(10.15) and res2.initial_budget_usd == pytest.approx(0.15) and res2.resumes == 1
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
    assert [(e["kind"], e["who"]) for e in ev] == [("resume", "dashboard")] and ev[0]["old_budget_usd"] == pytest.approx(0.15) and ev[0]["new_budget_usd"] == pytest.approx(10.15)
    assert any(json.loads(line).get("event") == "resume" for line in (run_dir / "transcript.jsonl").read_text().splitlines())


@pytest.mark.parametrize("stop, why", [("converged (4 rounds without improvement)", "not for budget"), ("max_rounds", "not for budget"), ("gpu_hang_circuit_breaker", "not for budget"), ("llm_error: x", "not for budget")])
def test_runs_that_stopped_for_another_reason_are_refused(tmp_path, stop, why):
    """@brief 验证因收敛、轮数上限、GPU 挂起熔断或 LLM 错误等非预算原因停止的运行不可续，prepare_resume 以退出码 3 拒绝。

    @param tmp_path pytest 提供的临时目录。
    @param stop 参数化的停止原因。
    @param why 期望出现在 info 的 reason 里的说明片段。
    """
    (tmp_path / "result.json").write_text(json.dumps({"task": "relu", "dtype": "fp16", "stop_reason": stop, "spent_usd": 1.0, "budget_usd": 2.0, "best_label": "init"}))
    (tmp_path / "state.json").write_text(json.dumps({"phase": "done", "stop_reason": stop, "best_label": "init"}))
    info = rs.inspect_run(tmp_path)
    assert info["resumable"] is False and info["kind"] == "finished" and why in info["reason"]
    with pytest.raises(cli.ResumeRefused) as e:
        cli.prepare_resume(ns(tmp_path, add_budget=5.0), log=lambda s: None)
    assert e.value.code == 3


def test_a_run_from_the_old_protocol_is_readable_but_never_resumable(tmp_path):
    """@brief 验证非 flame_chase_1 协议的旧运行可以读取但永不可续：inspect_run 报 old_protocol 且不可续（包括看板的重置路径），load_snapshot 因缺 loop_protocol 抛 ResumeError。

    这里若给出 resumable: true，看板就会渲染出一个点下去必然失败的续跑按钮。

    @param tmp_path pytest 提供的临时目录。
    """
    (tmp_path / "state.json").write_text(json.dumps({"pid": 2**22 + 778, "phase": "llm", "task": "relu", "best_label": "init", "spent_usd": 3.0, "budget_usd": 3.0}))
    (tmp_path / "result.json").write_text(json.dumps({"task": "relu", "dtype": "fp16", "stop_reason": "budget_exhausted", "spent_usd": 3.0, "budget_usd": 3.0, "best_label": "init"}))
    (tmp_path / "run_config.json").write_text(json.dumps({"note": "a chat-protocol run, no loop_protocol key"}))
    info = rs.inspect_run(tmp_path)
    assert info["kind"] == "old_protocol" and info["resumable"] is False
    assert "loop_protocol" in info["reason"]
    info = rs.inspect_run(tmp_path, allow_count_stops=True)
    assert info["resumable"] is False  # 看板的重置路径也不能把它变成可续
    with pytest.raises(rs.ResumeError, match="loop_protocol"):
        rs.load_snapshot(tmp_path, SearchConfig(), "k")


def test_a_live_run_and_a_held_lock_cannot_be_resumed(tmp_path):
    """@brief 验证正在运行的搜索（本进程 pid）不可续但可在线加预算，而锁被别的活进程持有时归类为 resuming 同样不可续。

    @param tmp_path pytest 提供的临时目录。
    """
    (tmp_path / "state.json").write_text(json.dumps({"pid": os.getpid(), "phase": "llm", "task": "relu", "best_label": "init", "spent_usd": 0.1, "budget_usd": 1.0, "control_supported": True}))
    info = rs.inspect_run(tmp_path)
    assert info["kind"] == "running" and info["resumable"] is False and info["can_topup_live"] is True
    (tmp_path / "state.json").write_text(json.dumps({"pid": os.getpid(), "phase": "llm", "task": "relu", "best_label": "init", "spent_usd": 0.1, "budget_usd": 1.0}))
    info = rs.inspect_run(tmp_path)
    assert info["can_topup_live"] is False and "older version" in info["live_reason"]
    (tmp_path / "state.json").write_text(json.dumps({"pid": 2**22 + 777, "phase": "llm", "task": "relu", "best_label": "init", "spent_usd": 0.1, "budget_usd": 1.0}))
    # 只有 flame_chase_1 的目录可以续（见下面的 old_protocol 用例）。
    (tmp_path / "run_config.json").write_text(json.dumps({"loop_protocol": "flame_chase_1"}))
    assert rs.inspect_run(tmp_path)["kind"] == "interrupted" and rs.inspect_run(tmp_path)["resumable"]
    # 锁上是本进程或它的父进程：这是看板刚拉起的续跑，不是别人占着。
    (tmp_path / "resume.lock").write_text(json.dumps({"pid": os.getppid(), "starting": True}))
    assert rs.inspect_run(tmp_path)["kind"] != "resuming"
    other = subprocess.Popen(["sleep", "30"])
    try:
        (tmp_path / "resume.lock").write_text(json.dumps({"pid": other.pid, "started_at": 1}))
        info = rs.inspect_run(tmp_path)
        assert info["kind"] == "resuming" and info["resumable"] is False
    finally:
        other.kill()
        other.wait(timeout=5)


def test_cli_resume_continues_a_scheduled_provider_retry_from_disk(tmp_path):
    """@brief 验证被杀在 provider 重试退避中的运行，续跑能从磁盘恢复 provider_attempts 并补做那次重试，花费只按 transcript 记下的算。

    @param tmp_path pytest 提供的临时目录。
    """
    from taic.kernel_agent import AgentResult

    def source(marker: str) -> str:
        return code(marker).split("```metal\n", 1)[1].rsplit("```", 1)[0]

    def result(marker: str | None, failure_class: str = "") -> AgentResult:
        return AgentResult(
            source=None if marker is None else source(marker), cost_usd=0.01, cost_incomplete=False,
            status="ok", error="Connection error." if failure_class == "turn_failed" else "",
            diagnostics={"failure_class": failure_class, "failed_turns": 1 if failure_class else 0,
                         "last_terminal_kind": "failed" if failure_class else "result"},
        )

    def agent(brief, tag):
        if brief.seedless:
            return result("speed=1.0")
        assert tag == "r1.p0"
        return result(None, "turn_failed")

    orch = make_orch(tmp_path, llm_script(code("speed=1.0")), [], max_rounds=1, provider_retries=1)
    orch.agent_runner = agent

    async def killed(_delay):
        raise SystemExit("killed during provider retry backoff")

    orch._sleep = killed
    with pytest.raises(SystemExit):
        run(orch)
    run_dir = tmp_path / "run"
    assert not (run_dir / "result.json").exists()
    assert json.loads((run_dir / "state.json").read_text())["provider_attempts"] == {"r1.p0": 1}
    mark_dead(run_dir)
    assert rs.inspect_run(run_dir)["kind"] == "interrupted"
    assert rs.inspect_run(run_dir)["resumable"] is True

    def script(_messages, **_kw):
        return make_response(code("speed=1.2"), usage=make_usage())

    resumed, snap, report, info, new_budget = cli.prepare_resume(
        ns(run_dir, add_budget=5.0), client=FakeClient(script), evaluate_fn=fake_eval([]), log=lambda _s: None,
    )
    assert snap.provider_attempts == {"r1.p0": 1}
    assert report["provider_attempts"] == {"r1.p0": 1}
    res = cli.run_prepared(ns(run_dir, add_budget=5.0), resumed, snap, info, new_budget)
    events = [json.loads(line) for line in (run_dir / "transcript.jsonl").read_text().splitlines()]
    retry = next(event for event in events if event.get("event") == "agent" and event.get("label") == "r1.p0.retry1")
    assert retry["candidate_label"] == "r1.p0" and retry["attempt"] == 1 and retry["outcome"] == "evaluated"
    assert res.best_label == "r1.p0" and res.best_speedup == pytest.approx(1.2)
    assert res.resumes == 1 and res.stop_reason == "max_rounds"
    assert res.spent_usd == pytest.approx(0.02 + retry["cost_usd"])
    assert json.loads((run_dir / "state.json").read_text())["provider_attempts"] == {}


def test_double_resume_is_refused_while_the_first_one_runs(tmp_path):
    """@brief 验证第一次续跑持锁运行期间，第二次续跑被拒绝（退出码 3，提示仍在运行）。

    @param tmp_path pytest 提供的临时目录。
    """
    calls = []
    orch = make_orch(tmp_path, script_for(usage=BIG), calls, budget=0.15, max_rounds=3, max_repair=2, patience=20)
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
    # restore() 已经把本进程 pid 写进 state.json，锁也是这个 pid，所以第二次续跑看到的是「仍在运行」，退出码仍是 3。
    assert holder["code"] == 3 and "still running" in holder["msg"]


# ------------------------------------------------------------------------------------------------ rebuilding from older directories


def test_settings_of_a_run_without_run_config_are_inferred_from_its_prompts(tmp_path):
    """@brief 验证 run_config 缺失时能从提示词推断部分设置（plan_mix、init_samples），而删掉 run_config.json 后则因缺 loop_protocol 拒绝续跑。

    @param tmp_path pytest 提供的临时目录。
    """
    calls = []
    orch = make_orch(tmp_path, script_for(), calls, budget=5.0, max_rounds=2, max_repair=2, patience=20, plan_mix=True, init_samples=1)
    run(orch)
    run_dir = tmp_path / "run"
    exact = rs.infer_settings(run_dir)["search"]
    assert exact["plan_mix"] is True and exact["init_samples"] == 1  # run_config.json
    (run_dir / "run_config.json").unlink()
    with pytest.raises(rs.ResumeError, match="loop_protocol"):
        rs.load_snapshot(run_dir, SearchConfig(branches=1), KERNEL)


def test_a_resume_lists_settings_the_old_run_did_not_record_and_still_continues(tmp_path, monkeypatch, capsys):
    """@brief 验证续跑会报告旧运行没记录下来的设置：有 run_config.json 时 unrecorded_settings 为空，删掉后 cmd_resume 以非零退出码指出 loop_protocol 问题。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 的替换工具，用来为 prepare_resume 准备环境。
    @param capsys pytest 的捕获工具，用来读命令行输出。
    """
    calls = []
    orch = make_orch(tmp_path, script_for(usage=BIG), calls, budget=0.15, max_rounds=3, max_repair=2, patience=20)
    res = run(orch)
    assert res.stop_reason == "budget_exhausted"
    run_dir = tmp_path / "run"
    recorded = cli.prepare_resume(ns(run_dir, dry_run=True, add_budget=5), client=FakeClient(script_for()), evaluate_fn=fake_eval([]), log=lambda s: None)
    assert recorded[2]["unrecorded_settings"] == []  # this run wrote run_config.json
    (run_dir / "run_config.json").unlink()
    code = cli.cmd_resume(ns(run_dir, dry_run=True, add_budget=5))
    assert code != 0 and "loop_protocol" in capsys.readouterr().err + capsys.readouterr().out


def test_the_ledger_is_rebuilt_from_the_transcript_when_state_is_behind_and_never_under_counted(tmp_path):
    """@brief 验证账本从 transcript 重建且绝不少算：state.json 落后时以 transcript 为准，transcript 不足时多出的金额记为 unaccounted。

    @param tmp_path pytest 提供的临时目录。
    """
    counter = {"kill_at": 2}
    calls = []
    orch = make_orch(tmp_path, script_for(counter, usage=BIG), calls, budget=50.0, max_rounds=3, max_repair=2, patience=20)
    with pytest.raises(SystemExit):
        run(orch)
    run_dir = tmp_path / "run"
    state = json.loads((run_dir / "state.json").read_text())
    ev, _ = rs.read_transcript(run_dir)
    billed = [e for e in ev if e.get("event") in ("chat", "agent") and isinstance(e.get("cost_usd"), (int, float))]
    total = billed[-1]["cost_usd"] if billed[-1].get("event") == "agent" else billed[-1]["total_usd"]
    # Agent lines store the session cost, not a running total. Sum them.
    if billed[-1].get("event") == "agent":
        total = sum(float(e["cost_usd"]) for e in billed)
    assert total >= state["spent_usd"] - 1e-9  # the transcript is exact, state.json can lag behind it
    # the transcript is lost: state.json says more than anything we can account for
    state["spent_usd"] = total + 0.37
    (run_dir / "state.json").write_text(json.dumps(state))
    snap = rs.load_snapshot(run_dir, SearchConfig(branches=3, max_repair_tries=2), KERNEL)
    assert snap.spent_usd == pytest.approx(total + 0.37) and snap.requests[-1]["label"] == "unaccounted"
    assert any("unaccounted" in n for n in snap.notes)


def test_a_damaged_last_transcript_line_is_skipped(tmp_path):
    """@brief 验证 transcript.jsonl 最后一行损坏时被跳过，快照仍能读出最佳候选并留下 damaged line 说明。

    @param tmp_path pytest 提供的临时目录。
    """
    counter = {"kill_at": 2}
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
    """@brief 验证 state.json 里缺失的 trace 条目通过重放选择过程重建，标签与 improved 与完整 trace 一致。

    @param tmp_path pytest 提供的临时目录。
    """
    counter = {"kill_at": 2}
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


def _ok_report(speed: float):
    """@brief 造一份 status 为 ok、带给定加速比的 EvalReport，供候选目录里的 report.json 使用。

    @param speed 该报告的加速比。
    @return 填好 speedups 的 EvalReport 实例。
    """
    from tcenv.evaluate import EvalReport

    report = EvalReport(task="relu", dtype="fp16", size="small", mode="full", seed=1, status="ok", stage="done", speedup=speed)
    report.speedups = {"mlx": speed}
    return report


def test_a_recheck_rejection_is_not_replayed_as_the_new_seed(tmp_path):
    """@brief 验证被 differential_recheck 拒绝的候选在 _replay 里不算改进，种子仍停在前一个可用内核。

    @param tmp_path pytest 提供的临时目录。
    """
    run = tmp_path / "run"
    for idx, label, speed, rejected in ((0, "init", 1.1, False), (1, "r1.p0", 2.0, True)):
        folder = run / "candidates" / f"{idx:03d}_{label}"
        folder.mkdir(parents=True)
        (folder / "kernel.metal").write_text(f"kernel void k_{label}() {{}}\n")
        (folder / "report.json").write_text(json.dumps(_ok_report(speed).to_json()))
        if rejected:
            (folder / "differential_recheck.json").write_text(json.dumps({"counts_as_kernel_error": True}))
    notes: list[str] = []
    cands = rs._read_candidates(run, notes)
    replay = rs._replay(cands, SearchConfig(branches=1, min_improvement=1.005, noise_z=0))
    assert cands[1]["recheck_rejected"] is True
    assert [row["improved"] for row in replay] == [True, False]
    assert replay[-1]["best_speedup"] == pytest.approx(1.1)


def test_a_stale_search_seed_does_not_hide_a_newer_kernel(tmp_path):
    """@brief 验证 state.json 里过期的 search_seed_label 不会遮住更晚写出的更好内核，快照取较新的候选并留下说明。

    @param tmp_path pytest 提供的临时目录。
    """
    run = tmp_path / "run"
    for idx, label, speed in ((0, "init", 1.1), (1, "r1.p0", 1.8)):
        folder = run / "candidates" / f"{idx:03d}_{label}"
        folder.mkdir(parents=True)
        (folder / "kernel.metal").write_text(f"kernel void k_{label}() {{}}\n")
        (folder / "report.json").write_text(json.dumps(_ok_report(speed).to_json()))
    (run / "state.json").write_text(json.dumps({
        "task": "relu", "dtype": "fp16", "size": "small", "phase": "llm", "search_seed_label": "init",
        "trace": [{"idx": 0, "label": "init", "improved": True, "speedup": 1.1, "cost_usd": 0.01, "t_s": 1.0}],
    }))
    (run / "run_config.json").write_text(json.dumps({"loop_protocol": "flame_chase_1"}))
    snap = rs.load_snapshot(run, SearchConfig(branches=1, max_repair_tries=1, min_improvement=1.005, noise_z=0), "k")
    assert snap.best_label == "r1.p0"
    assert any("older than r1.p0" in note for note in snap.notes)


def test_a_slower_kernel_written_after_state_does_not_become_the_seed(tmp_path):
    """@brief 验证在 state.json 之后才写出的更慢内核不会成为种子：种子仍是 init，慢候选不记为改进。

    @param tmp_path pytest 提供的临时目录。
    """
    run = tmp_path / "run"
    for idx, label, speed in ((0, "init", 1.8), (1, "r1.p0", 1.1)):
        folder = run / "candidates" / f"{idx:03d}_{label}"
        folder.mkdir(parents=True)
        (folder / "kernel.metal").write_text(f"kernel void k_{label}() {{}}\n")
        (folder / "report.json").write_text(json.dumps(_ok_report(speed).to_json()))
    (run / "state.json").write_text(json.dumps({
        "task": "relu", "dtype": "fp16", "size": "small", "phase": "llm", "search_seed_label": "init",
        "trace": [{"idx": 0, "label": "init", "improved": True, "speedup": 1.8, "cost_usd": 0.01, "t_s": 1.0}],
    }))
    (run / "run_config.json").write_text(json.dumps({"loop_protocol": "flame_chase_1"}))
    snap = rs.load_snapshot(run, SearchConfig(branches=1, min_improvement=1.005, noise_z=0), "k")
    assert snap.best_label == "init"
    assert snap.trace[-1]["improved"] is False


def test_chat_with_tools_continues_a_conversation_at_the_given_turn(tmp_path):
    """@brief 验证 chat_with_tools 能从指定回合继续带工具调用的对话：补上待处理工具的结果，并以 #t2 这样的标签写 transcript。

    @param tmp_path pytest 提供的临时目录。
    """
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
    """@brief 造一个先对桩内核调 quick_check、再按 script_for 回答的脚本化模型；规划请求直接作答。

    @param counter 可选的调用计数器，可在指定的第几次调用处抛 SystemExit 模拟被杀。
    @return 接收 (messages, **kw) 的路由函数。
    """
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
    """@brief 造一个开启工具调用（use_tools=True、kernel_agent=pi）的 Orchestrator。

    @param tmp_path pytest 提供的临时目录。
    @param script 脚本化模型的路由函数。
    @param calls 评测调用记录列表，交给 fake_eval 填充。
    @return 配置好的 Orchestrator。
    """
    llm_cfg = LLMConfig(base_url="http://x", api_key="sk-test-0123456789abcdef", model="fake", reasoning_effort="off", use_tools=True)
    search = SearchConfig(budget_usd=5.0, branches=1, max_rounds=3, max_repair_tries=2, eval_reps=3, patience=20, kernel_agent="pi")
    return Orchestrator("relu", "fp16", "small", llm_cfg, search, run_dir=tmp_path / "run", evaluate_fn=fake_eval(calls), client=FakeClient(script), log=lambda s: None)


def test_a_reply_that_only_asks_for_a_tool_does_not_become_a_kernel(tmp_path):
    """@brief 验证只索要工具、最终没交内核的会话不会被当成内核：搜索以 agent_failure 停下且没有胜者。

    搜索不再在工具调用对话里做修复；一个没留下内核的会话算失败，连续两次失败就停。

    @param tmp_path pytest 提供的临时目录。
    """
    res = run(make_tool_orch(tmp_path, script_with_tools(), []))
    assert res.stop_reason == "agent_failure"
    assert res.best_label is None


def test_resume_restores_notes_and_the_feedback_sequence(tmp_path):
    """@brief 验证续跑能恢复上一轮的 NOTES.md 摘要与对方给的提示（feedback），并把反馈序号接到 2。

    @param tmp_path pytest 提供的临时目录。
    """
    from taic.kernel_agent import AgentResult

    calls = []

    def write(brief, tag):
        del brief
        folder = tmp_path / "run" / "agent_work" / tag
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "NOTES.md").write_text("tile 32 overflowed\n")
        source = code("speed=1.2").split("```metal\n", 1)[1].rsplit("```", 1)[0] + "\n"
        feedback = "kind: none\n\n# 给下一棒的避坑提示\n- float8 是保留字\n"
        return AgentResult(source=source, cost_usd=0.01, cost_incomplete=False, status="ok", notes="tile 32 overflowed\n", feedback=feedback)

    orch = make_orch(tmp_path, llm_script(code("speed=1")), calls, max_rounds=0)
    orch.agent_runner = write
    res = run(orch)
    assert res.best_label == "init"
    snap = rs.load_snapshot(orch.run_dir, orch.search, KERNEL)
    follow = Orchestrator(
        "relu", "fp16", "small", CFG, orch.search, run_dir=orch.run_dir, resuming=True,
        evaluate_fn=fake_eval([]), client=FakeClient(lambda *_a, **_k: None), log=lambda _s: None,
    )
    follow.restore(snap)
    assert "tile 32 overflowed" in follow._notes_summary
    assert "float8" in follow._session_brief(False).peer_tips
    assert follow._next_feedback_seq() == 2


def test_agent_tokens_survive_a_resume_ledger_restore():
    """@brief 验证 transcript 里 agent 行的四类 token（input、output、cache_read、cache_write）在账本恢复后仍分别记在 hit、miss、write、out 上。
    """
    events = [{"event": "agent", "label": "init", "cost_usd": 1.0,
               "tokens": {"input": 100, "output": 200, "cache_read": 300, "cache_write": 400}}]
    reqs = rs._requests_from_transcript(events, [])
    assert reqs[0]["hit"] == 300 and reqs[0]["miss"] == 100 and reqs[0]["write"] == 400 and reqs[0]["out"] == 200
    ledger = CostLedger(CFG, 10.0)
    ledger.restore(reqs)
    assert ledger.in_tokens_hit == 300 and ledger.in_tokens_miss == 100
    assert ledger.in_tokens_write == 400 and ledger.out_tokens == 200


def test_chat_usage_write_tokens_survive_a_resume():
    """@brief 验证 chat 行 usage 里的 cache_write_tokens 能从 transcript 恢复到续跑的账本请求里。
    """
    events = [{"event": "chat", "label": "init", "cost_usd": 0.1, "total_usd": 0.1,
               "usage": {"prompt_tokens": 10, "completion_tokens": 2,
                         "prompt_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 4}}}]
    reqs = rs._requests_from_transcript(events, [])
    assert reqs[0]["write"] == 4


def test_a_resume_counts_the_spend_recorded_in_result_json(tmp_path):
    """@brief 验证续跑把 result.json 记录的已花金额也算进总花费，差额记为 unaccounted。

    result.json 是最后写下的结论；忽略它会按差额少算花费。

    @param tmp_path pytest 提供的临时目录。
    """
    orch = make_orch(tmp_path, llm_script(code("speed=1.2")), [], max_rounds=1)
    run(orch)
    run_dir = tmp_path / "run"
    result = json.loads((run_dir / "result.json").read_text())
    spent = float(result["spent_usd"])
    result["spent_usd"] = spent + 3.0
    (run_dir / "result.json").write_text(json.dumps(result))
    snap = rs.load_snapshot(run_dir, orch.search, KERNEL)
    assert snap.spent_usd == pytest.approx(spent + 3.0)
    assert any("unaccounted" in note for note in snap.notes)


def test_a_second_topup_never_overwrites_an_earlier_audit_copy(tmp_path):
    """@brief 验证第二次加预算不会覆盖第一次的审计副本：已存在的 confirm_report.before_topup_1.json 保持原样，本次写 before_topup_2。

    snapshot_results 对 result 和 confirm_report 复用同一个 generation 号；在确认阶段被杀的运行可能已经有
    confirm_report.before_topup_1.json。

    @param tmp_path pytest 提供的临时目录。
    """
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "result.json").write_text('{"generation": 2}')
    (run_dir / "confirm_report.json").write_text('{"generation": 2}')
    (run_dir / "confirm_report.before_topup_1.json").write_text('{"generation": 1}')
    moved = rs.snapshot_results(run_dir)
    assert "confirm_report.before_topup_2.json" in moved
    assert json.loads((run_dir / "confirm_report.before_topup_1.json").read_text()) == {"generation": 1}
    assert json.loads((run_dir / "confirm_report.before_topup_2.json").read_text()) == {"generation": 2}


def test_a_run_config_from_a_newer_code_version_is_refused(tmp_path):
    """@brief 验证 schema_version 高于当前代码的 run_config.json 被判定不可续，load_snapshot 抛 NeedsInspection。

    @param tmp_path pytest 提供的临时目录。
    """
    orch = make_orch(tmp_path, llm_script(code("speed=1.0")), [], max_rounds=1)
    run(orch)
    run_dir = tmp_path / "run"
    doc = json.loads((run_dir / "run_config.json").read_text())
    doc["schema_version"] = 99
    (run_dir / "run_config.json").write_text(json.dumps(doc))
    info = rs.inspect_run(run_dir)
    assert info["resumable"] is False and "newer" in info["reason"]
    with pytest.raises(rs.NeedsInspection):
        rs.load_snapshot(run_dir, orch.search, KERNEL)


def test_a_resume_that_cannot_start_puts_the_old_result_back(tmp_path):
    """@brief 验证续跑在启动阶段失败时，把 result.json 与 confirm_report.json 原样放回，并释放运行锁。

    @param tmp_path pytest 提供的临时目录。
    """
    orch = make_orch(tmp_path, script_for(usage=BIG), [], budget=0.15, max_rounds=3, max_repair=2, patience=20)
    run(orch)
    run_dir = tmp_path / "run"
    before = {p.name: p.read_bytes() for p in run_dir.glob("*.json") if p.name in ("result.json", "confirm_report.json")}
    assert "result.json" in before
    o2, snap, report, info, nb = cli.prepare_resume(
        ns(run_dir, add_budget=5.0, who="test"), client=FakeClient(script_for(usage=BIG)), evaluate_fn=fake_eval([]), log=lambda s: None,
    )

    async def boom():
        raise RunStartError("startup failed after the snapshot")

    o2.run = boom
    with pytest.raises(RunStartError, match="startup failed"):
        cli.run_prepared(ns(run_dir, add_budget=5.0, who="test"), o2, snap, info, nb)
    for name, data in before.items():
        assert (run_dir / name).read_bytes() == data, name
    assert control.lock_holder(run_dir) is None


def test_resume_dry_run_leaves_the_run_directory_byte_identical(tmp_path):
    """@brief 验证 --dry-run 续跑后运行目录逐字节不变，退出码为 0。

    @param tmp_path pytest 提供的临时目录。
    """
    orch = make_orch(tmp_path, script_for(usage=BIG), [], budget=0.15, max_rounds=3, max_repair=2, patience=20)
    run(orch)
    run_dir = tmp_path / "run"

    def snapshot():
        return {str(p.relative_to(run_dir)): p.read_bytes() for p in run_dir.rglob("*") if p.is_file()}

    before = snapshot()
    assert cli.main(["resume", str(run_dir), "--dry-run", "--add-budget", "5"]) == 0
    assert snapshot() == before


def test_a_cost_incomplete_run_continues_only_with_the_explicit_flag(tmp_path):
    """@brief 验证因费用统计不完整（cost_incomplete）停下的运行默认不可续，只有显式接受费用缺口才能续。

    @param tmp_path pytest 提供的临时目录。
    """
    orch = make_orch(tmp_path, llm_script(code("speed=1.2")), [], max_rounds=1)
    run(orch)
    run_dir = tmp_path / "run"
    for name in ("result.json", "state.json"):
        doc = json.loads((run_dir / name).read_text())
        doc["stop_reason"] = "cost_incomplete: the session finished without a token count"
        (run_dir / name).write_text(json.dumps(doc))
    assert rs.inspect_run(run_dir)["resumable"] is False
    flagged = rs.inspect_run(run_dir, allow_cost_incomplete=True)
    assert flagged["resumable"] is True and flagged["kind"] == "cost_incomplete"
    with pytest.raises(cli.ResumeRefused):
        cli.prepare_resume(ns(run_dir), client=FakeClient([]), evaluate_fn=fake_eval([]), log=lambda s: None)
    prepared = cli.prepare_resume(
        ns(run_dir, accept_cost_gap=True), client=FakeClient(llm_script(code("speed=1.2"))), evaluate_fn=fake_eval([]), log=lambda s: None,
    )
    assert prepared[2]["dry_run"] is False


def test_a_resume_carries_unbilled_hmz_sessions_as_unaccounted(tmp_path, monkeypatch):
    """@brief 验证续跑把未计费的 hmz 会话花费带进来，记成 unaccounted 请求并在 notes 里说明 hmz epics 的缺口。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 的替换工具，用来桩掉 epic_spend_by_label。
    """
    import taic.kernel_agent as kernel_agent

    orch = make_orch(tmp_path, llm_script(code("speed=1.2")), [], max_rounds=0)
    run(orch)
    run_dir = tmp_path / "run"
    monkeypatch.setattr(kernel_agent, "epic_spend_by_label", lambda run_dir, root=None: {"init": 5.0})
    snap = rs.load_snapshot(run_dir, orch.search, KERNEL)
    assert snap.spent_usd == pytest.approx(5.0)
    assert any("hmz epics" in note for note in snap.notes)
    rows = [row for row in snap.requests if row.get("label") == "unaccounted"]
    assert rows and rows[-1]["total_usd"] == pytest.approx(5.0)

