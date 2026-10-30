"""Observe request reads advance the index incrementally; explicit refresh remains exhaustive.

Also pins the §5.2 detail split: the summary carries only the first-screen fields, each panel
reads its own source files, and a matching stamp must not read a byte of the run directory.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import statistics
from contextlib import contextmanager
from pathlib import Path

import pytest

from experiments.manifest import attach, ensure
from observe import detail, query
from observe.index import Index, get_index
from taic.agent_feedback import parse_feedback
from tcenv.schema import PROTOCOL_VERSION


def _use_temp_runs(tmp_path, monkeypatch):
    runs = tmp_path / "runs"
    runs.mkdir()
    monkeypatch.setenv("AI_LOWERLING_RUNS_DIR", str(runs))
    monkeypatch.setenv("OBSERVE_DB", str(tmp_path / "observe.sqlite"))
    monkeypatch.setenv("AI_LOWERLING_BOOKS", str(tmp_path / "books.sqlite"))
    monkeypatch.setenv("OBSERVE_TICK_INTERVAL", "0")
    monkeypatch.setenv("OBSERVE_FULL_TICK_INTERVAL", "3600")
    return runs


def _write_run(root, name, *, task="relu", spent=0.25, budget=2.0):
    directory = root / name
    directory.mkdir(parents=True)
    common = {
        "task": task,
        "dtype": "fp16",
        "started_at": 10.0,
        "spent_usd": spent,
        "budget_usd": budget,
    }
    (directory / "state.json").write_text(json.dumps({**common, "phase": "done", "updated_at": 10.0}), encoding="utf-8")
    (directory / "result.json").write_text(json.dumps({**common, "stop_reason": "max_rounds"}), encoding="utf-8")
    return directory


def _forbid_sync_known(monkeypatch):
    def forbidden(self):
        raise AssertionError("request reads must not call Index.sync_known")

    monkeypatch.setattr(Index, "sync_known", forbidden)


def test_request_reads_do_not_sync_known_and_keep_their_results(tmp_path, monkeypatch):
    """@brief 禁止全量核对后，实验队列、比较、账本与总览仍返回核对前相同的内容。"""
    runs = _use_temp_runs(tmp_path, monkeypatch)
    _write_run(runs, "plain")
    queue = runs / "queue"
    queue.mkdir()
    (queue / "state.json").write_text(json.dumps({
        "script": "repeatability", "phase": "done", "started_at": 1.0, "updated_at": 2.0,
    }), encoding="utf-8")
    Index(tmp_path / "observe.sqlite").sync_known()
    monkeypatch.setattr(query.detail, "gpu_snapshot", lambda: {"holders": [], "waiters": [], "events": []})
    monkeypatch.setattr(query.detail, "live_evals", lambda: [])

    expected = {
        "jobs": query.experiment_jobs(),
        "compare": query.compare("queue"),
        "ledger": query.ledger_report(),
        "overview": query.overview(),
    }
    expected["overview"].pop("now")

    _forbid_sync_known(monkeypatch)
    actual = {
        "jobs": query.experiment_jobs(),
        "compare": query.compare("queue"),
        "ledger": query.ledger_report(),
        "overview": query.overview(),
    }
    actual["overview"].pop("now")
    assert actual == expected
    assert actual["jobs"][0]["id"] == "queue"
    assert actual["ledger"]["runs"] == 1


def test_new_experiment_and_run_are_discovered_by_tick(tmp_path, monkeypatch):
    """@brief 初始索引后新增的实验和子运行由增量 tick 发现，无需全量 sync_known。"""
    runs = _use_temp_runs(tmp_path, monkeypatch)
    query.refresh()
    _forbid_sync_known(monkeypatch)

    experiment = runs / "exp"
    frozen = {"task": "relu", "dtype": "fp16", "size": "small", "budget_usd": 2.0}
    ensure(
        experiment,
        kind="repeatability",
        driver="repeatability",
        frozen=frozen,
        arms=[{"arm_id": "relu", "delta": {}, "planned_n": 1}],
    )
    _write_run(experiment, "relu/trial_1")
    attach(experiment, "relu", "relu/trial_1")

    compared = query.compare("exp")
    assert compared["found"] is True
    assert compared["arms"][0]["arm_id"] == "relu"
    assert compared["arms"][0]["n"] == 1
    assert query.ledger_report()["runs"] == 1


# §5.2 的字段去向表，逐行搬成可执行断言。摘要在表之外多了四个派生键：``state``/``result`` 移出摘要后，
# 前端仍要任务名、种子标签、确认胜者与在飞标签，不能让页面为了三个短字符串去开 ``/raw``。
SUMMARY_FIELDS = frozenset({
    "id", "status", "target_id", "backend", "backend_version", "target_line",
    "budget", "control", "topped_up", "initial_budget_usd",
    "publication", "published_speedup", "confirmed_point", "winner", "model_totals",
    "timing", "stop_reason", "eval_progress", "pending_evaluation", "kernel_writer", "handoff",
    "task", "best_label", "confirm_winner_label", "inflight_label",
})
PANEL_FIELDS = {
    "candidate_marks": "/candidates",
    "candidates": "/candidates",
    "timeline": "/timeline",
    "budget_events": "/budget-events",
    "llm": "/llm",
    "state": "/raw?key=state",
    "result": "/raw?key=result",
    "feedback": "/feedback",
    "feature_grid": "/grid",
    "plan_outcomes": "/plan",
}
# trace 与 confirmations 与 state.trace / result.confirmations 逐字节相同，拆详情时直接删掉。
REMOVED_FIELDS = frozenset({"trace", "confirmations"})

T0 = 1_700_000_000.0
_SOURCE = "kernel void conv2d() { /* coalesced epilogue */ }\n"
_FEEDBACK_TEXT = """kind: tool

# 需要帮助
- 缺一把探针

# 给下一棒的避坑提示
- float8 是保留字
"""
_SECRET = "SECRET raw text sk-abcdefghijklmnop"


def _digest() -> str:
    """@brief 算出用例内核源码的 sha256，供确认报告绑定复测内核。

    @return ``_SOURCE`` 的 sha256 十六进制摘要。
    """
    return hashlib.sha256(_SOURCE.encode()).hexdigest()


def _feedback_record(seq: int, label: str, raw: str = _SECRET) -> dict:
    """@brief 造一条可被 ``load_records`` 接受的 feedback.jsonl 记录。

    @param seq 记录序号。
    @param label 候选标签。
    @param raw 记录全文（列表端点不应内联它）。
    @return 反馈记录字典。
    """
    parsed = parse_feedback(_FEEDBACK_TEXT)
    return {
        "format": "agent_feedback", "v": 1, "seq": seq, "t": 10.0 + seq, "task": "conv2d", "dtype": "fp16",
        "size": "paper", "label": label, "round": 1, "committee": "first", "provider": "deepseek",
        "model": "deepseek-flash", "effort": "high", "outcome": "evaluated", "exit": "normal",
        "best_speedup": 1.25, "kinds": parsed["kinds"], "work_dir": f"agent_work/{label}",
        "sections": parsed["sections"], "items": parsed["items"],
        "unverified_premises": parsed["unverified_premises"], "raw": raw,
    }


def _rich_run(root: Path, name: str = "conv2d/rich", *, candidates: int = 2) -> Path:
    """@brief 造一份带确认报告、候选报告、反馈、预算事件与转录的已结束 run。

    @param root runs 根目录。
    @param name run id（相对 runs/）。
    @param candidates 候选目录数量（至少 2 才有“改写一颗”的余地）。
    @return 新建的 run 目录。
    """
    directory = root / name
    directory.mkdir(parents=True)
    digest = _digest()
    (directory / "best.metal").write_text(_SOURCE, encoding="utf-8")
    trace = [
        {"label": "init", "status": "ok", "speedup": 1.0, "best_speedup": 1.0, "cost_usd": 0.1},
        {"label": "r1.p0", "status": "ok", "speedup": 1.25, "best_speedup": 1.25, "cost_usd": 0.3},
    ]
    labels = ["init", "r1.p0"][:candidates]
    from tcenv.evaluate import EvalReport

    from ..support.orchestrator import stamp_report

    for index, label in enumerate(labels):
        folder = directory / "candidates" / f"{index:03d}_{label}"
        folder.mkdir(parents=True)
        (folder / "kernel.metal").write_text(_SOURCE, encoding="utf-8")
        # Schema 4: the detail/review surfaces only quote a report they can validate.
        report = EvalReport(
            task="conv2d", dtype="fp16", size="paper", mode="full", seed=1, stage="bench",
            status="ok", message=f"report {label}", speedup=1.0 + index * 0.25,
            target_id="apple-m4-max",
        )
        report.candidate_us = 770.0 - index * 10
        report.benchmark = {"noisy": False, "attempts": 3, "noise_cv": 0.01}
        report.roofline = {"lower_bound_us": 414.0, "bound_kind": "reference_algorithm"}
        stamp_report(report, "conv2d", "fp16", "paper", _SOURCE)
        (folder / "report.json").write_text(json.dumps(report.to_json()), encoding="utf-8")
    (directory / "state.json").write_text(json.dumps({
        "pid": 1, "phase": "done", "started_at": T0, "updated_at": T0 + 30, "task": "conv2d",
        "dtype": "fp16", "spent_usd": 0.4, "budget_usd": 2.0, "best_label": "r1.p0",
        "inflight_label": "r1.p0", "committee": "first", "round": 1, "model": "deepseek-flash",
        "trace": trace,
    }), encoding="utf-8")
    from experiments.common import synthetic_confirmation_documents

    result, report = synthetic_confirmation_documents(
        task="conv2d", dtype="fp16", size="paper", target_id="apple-m4-max",
        speedup=1.25, label="r1.p0", kernel_sha256=digest,
    )
    result.update({
        "stop_reason": "max_rounds", "started_at": T0, "spent_usd": 0.5, "budget_usd": 2.0,
        "best_speedup": 9.9, "winner_label": "r1.p0",
        "winner_provider": "deepseek", "winner_model": "deepseek-flash", "winner_committee": "first",
        "winner_origin": "session", "model_totals": [{"provider": "deepseek", "model": "deepseek-flash",
        "committee": "first", "sessions": 1, "cost_usd": 0.3, "output_tokens": 10}],
        "trace": trace,
    })
    (directory / "confirm_report.json").write_text(json.dumps(report), encoding="utf-8")
    (directory / "result.json").write_text(json.dumps(result), encoding="utf-8")
    (directory / "run_config.json").write_text(json.dumps({
        "target": {"id": "apple-m4-max", "backend": "metal", "backend_version": "v1"},
        "search": {"kernel_agent": "pi", "plan_ledger": True},
        "committees": [
            {"name": "first", "members": [{"provider": "deepseek", "model": "deepseek-flash"}]},
            {"name": "second", "members": [{"provider": "moonshot", "model": "kimi-k3"}]},
        ],
    }), encoding="utf-8")
    with (directory / "feedback.jsonl").open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(_feedback_record(1, "r1.p0"), ensure_ascii=False) + "\n")
        handle.write(json.dumps(_feedback_record(2, "init"), ensure_ascii=False) + "\n")
    (directory / "ledger.json").write_text(json.dumps({"requests": [{"t": 4.5, "label": "init#t0"}]}), encoding="utf-8")
    (directory / "budget_events.json").write_text(json.dumps([
        {"t": T0 + 20, "time": "2023-11-14T22:13:40Z", "kind": "applied", "who": "op",
         "old_budget_usd": 1.0, "new_budget_usd": 2.0, "note": "topup"},
    ]), encoding="utf-8")
    (directory / "llm.json").write_text(json.dumps({"label": "r2.plan", "model": "deepseek-flash"}), encoding="utf-8")
    (directory / "eval_progress.json").write_text(json.dumps({
        "stage": "bench", "entered_at": T0 + 18, "updated_at": T0 + 20, "finished": True,
        "task": "conv2d", "messages": [41],
    }), encoding="utf-8")
    (directory / "transcript.jsonl").write_text(json.dumps({
        "event": "chat", "label": "init#t0", "model": "deepseek-flash", "seconds": 2.0, "t": T0 + 5,
        "messages": [{"role": "user", "content": "prompt"}],
        "reply": {"content": "answer"},
    }) + "\n", encoding="utf-8")
    return directory


@contextmanager
def _counting_opens(monkeypatch):
    """@brief 统计 ``io.open`` 打开的路径与模式；``read_text``/``Path.open`` 都走这里。

    记录的是“打开了哪些源文件”，所以调用方用“没有打开”来证明读字节为 0。

    @param monkeypatch pytest 的替换工具。
    @return 上下文管理器，产出已打开路径列表。
    """
    opened: list[tuple[str, str]] = []
    real = io.open

    def counting(file, mode="r", *args, **kwargs):
        opened.append((str(file), mode))
        return real(file, mode, *args, **kwargs)

    monkeypatch.setattr(io, "open", counting)
    try:
        yield opened
    finally:
        monkeypatch.setattr(io, "open", real)


def _reads_under(opened: list[tuple[str, str]], root: Path) -> list[str]:
    """@brief 过滤出落在某个目录下的打开记录。

    @param opened ``_counting_opens`` 记下的 (路径, 模式) 列表。
    @param root 只关心这个目录下的文件。
    @return 相对该目录的路径列表。
    """
    base = str(root)
    return [path for path, _mode in opened if str(path).startswith(base)]


def _count_loads(monkeypatch) -> list[str]:
    """@brief 给 ``observe.detail._load`` 装一个记录器，返回被读过的路径列表。

    @param monkeypatch pytest 的替换工具。
    @return 会被持续追加的路径列表。
    """
    seen: list[str] = []
    real = detail._load

    def counting(path):
        seen.append(str(path))
        return real(path)

    monkeypatch.setattr(detail, "_load", counting)
    return seen


def _open_rich(tmp_path, monkeypatch) -> tuple[Path, Path]:
    """@brief 建临时 runs、写一份富运行并同步索引，返回 runs 根与 run 目录。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 的替换工具。
    @return ``(runs 根, run 目录)``。
    """
    runs = _use_temp_runs(tmp_path, monkeypatch)
    directory = _rich_run(runs)
    Index(tmp_path / "observe.sqlite").sync_known()
    return runs, directory


def test_the_summary_keeps_every_summary_row_and_no_panel_row(tmp_path, monkeypatch):
    """@brief 验收⑤：§5.2 表里每个“摘要”行都在摘要里，面板行都不在，字段归属不漏行。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 的替换工具。
    """
    _runs, _directory = _open_rich(tmp_path, monkeypatch)
    summary = query.run_summary("conv2d/rich")
    full = query.run_detail("conv2d/rich")
    assert summary is not None and full is not None

    missing = SUMMARY_FIELDS - set(summary)
    assert missing == set(), f"摘要漏了 §5.2 的摘要字段: {sorted(missing)}"
    leaked = set(PANEL_FIELDS) & set(summary)
    assert leaked == set(), f"面板字段不该在摘要里: {sorted(leaked)}"
    for key in REMOVED_FIELDS:
        assert key not in summary and key not in full, f"{key} 与 state/result 内的字段重复，应该删掉"

    # 字段表必须把旧详情的每个键都安排到摘要、面板或删除三类，不能出现“没人认领”的键。
    landed = SUMMARY_FIELDS | set(PANEL_FIELDS)
    assert set(full) - {"detail_stamp"} - REMOVED_FIELDS == landed
    assert set(summary) == SUMMARY_FIELDS | {"detail_stamp"}, "摘要多出或少了没登记的键"


# 独立的摘要期望值：直接照着 fixture 写出来，不调用生产的组装函数。
# 它同时是 ① 的“同名字段同值”证明：除声明过的投影字段外，摘要必须与旧详情逐字段相等。
GOLDEN_SUMMARY = {
    "id": "conv2d/rich",
    "status": "done",
    "target_id": "apple-m4-max",
    "backend": "metal",
    "backend_version": "v1",
    "target_line": "目标: apple-m4-max (metal/v1)",
    "budget": {
        "budget": 2.0, "flagged": False, "flags": [], "label": "", "label_cn": "", "level": "ok",
        "over_pct": 0.0, "over_usd": 0.0, "remaining_usd": 1.5, "spent": 0.5,
        "text": "$0.500 / $2.00 · 25.0%", "used_pct": 25.0,
    },
    "control": {
        "action": None,
        "reason": "stopped for 'max_rounds', not for budget: only runs that ran out of budget "
                  "(or were interrupted) can be continued",
    },
    "topped_up": True,
    "initial_budget_usd": 1.0,
    "publication": "confirmed",
    "published_speedup": 1.25,
    "winner": {"label": "r1.p0", "provider": "deepseek", "model": "deepseek-flash",
               "committee": "first", "origin": "session"},
    "model_totals": [{"provider": "deepseek", "model": "deepseek-flash", "committee": "first",
                      "sessions": 1, "cost_usd": 0.3, "output_tokens": 10}],
    "timing": {"started_at": T0, "ended_at": T0 + 30, "elapsed_s": 30.0, "running": False,
               "ended_at_estimated": False},
    "stop_reason": "max_rounds",
    "eval_progress": {"stage": "bench", "entered_at": T0 + 18, "updated_at": T0 + 20,
                      "finished": True, "task": "conv2d", "messages": [41]},
    "pending_evaluation": {"count": 0, "requests": []},
    "kernel_writer": "pi",
    "handoff": {"recorded": True, "note": "", "rounds": [
        {"round": 0, "label": "初稿", "committee": "first", "model": "deepseek-flash"},
        {"round": 1, "label": "第 1 轮", "committee": "first", "model": "deepseek-flash"},
    ]},
    "confirmed_point": {"cost_usd": 0.5, "speedup": 1.25, "search_speedup": 9.9,
                        "batches": [1.25], "worst": 1.25, "best": 1.25, "label": "确认"},
    "task": "conv2d",
    "best_label": "r1.p0",
    "confirm_winner_label": "r1.p0",
    "inflight_label": "r1.p0",
}
def test_the_summary_matches_an_independent_golden_and_the_old_detail(tmp_path, monkeypatch):
    """@brief 验收①：摘要每个键都与旧详情逐字段同值；顶层 stop_reason 取旧推导。

    这里不做任何“投影豁免”：§5.2 的 8KB 是 (C) 校准目标，本测试只用真实语料记录实测大小。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 的替换工具。
    """
    _runs, _directory = _open_rich(tmp_path, monkeypatch)
    summary = query.run_summary("conv2d/rich")
    full = query.run_detail("conv2d/rich")
    assert summary is not None and full is not None
    got = {key: value for key, value in summary.items() if key != "detail_stamp"}
    assert got == GOLDEN_SUMMARY

    for key in got:
        assert got[key] == full[key], f"{key} 的摘要值与旧详情不一致"
    assert full["eval_progress"]["messages"] == [41]
    assert summary["stop_reason"] == full["stop_reason"] == "max_rounds"
    # 两种负载形状不同，戳也必须不同：摘要只覆盖摘要依赖，完整详情还要覆盖候选/反馈/transcript 等。
    # 共用同一个戳会让完整详情在候选报告就地改写后一直回 ``same``（见下一条回归）。
    assert summary["detail_stamp"] and full["detail_stamp"]
    assert summary["detail_stamp"] != full["detail_stamp"]
    # 固定 fixture + 固定序列化参数下的字节回归；§5.2 的 8KB 是 (C) 校准目标（plan §0.4），
    # 真实 63 场最大约 3.7KB。这里不截断任何字段，字段同值由上面的循环保证。
    size = len(json.dumps(summary, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
    assert size <= 8192, f"fixture 摘要 {size}B 超过 8KB"


def test_a_cold_summary_does_not_ingest_or_read_any_panel_file(tmp_path, monkeypatch):
    """@brief 验收②：索引里没有该运行时，摘要也不摄取、不读候选报告与反馈日志。

    真实页面先取列表页，但直接打开一个未索引的 ``?run=`` 也不能因为拿 stamp 而触发索引器。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 的替换工具。
    """
    runs = _use_temp_runs(tmp_path, monkeypatch)
    directory = _rich_run(runs)
    seen = _count_loads(monkeypatch)
    with _counting_opens(monkeypatch) as opened:
        summary = query.run_summary("conv2d/rich")
    assert summary is not None
    assert summary["id"] == "conv2d/rich" and summary["detail_stamp"] == ""
    assert [path for path in seen if path.endswith("report.json")] == []
    touched = _reads_under(opened, directory)
    assert [path for path in touched if path.endswith("feedback.jsonl")] == []
    assert [path for path in touched if "candidates" in path and path.endswith("report.json")] == []
    # 没有进索引：stamp 为空，后续同 stamp 短路不成立，页面会拿全量摘要。
    assert get_index().conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0] == 0


def test_a_dirty_index_summary_does_not_ingest_or_read_candidate_reports(tmp_path, monkeypatch):
    """@brief 验收②补：索引已有一行、但候选目录之后新增了报告时，摘要读路径也不摄取。

    旧实现 ``resolve_run`` → ``_ingest`` 会在 stamp 判断前把新候选报告读一遍。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 的替换工具。
    """
    runs, directory = _open_rich(tmp_path, monkeypatch)
    late = directory / "candidates" / "002_r2.p0"
    late.mkdir()
    (late / "kernel.metal").write_text(_SOURCE, encoding="utf-8")
    (late / "report.json").write_text(json.dumps({"status": "ok", "stage": "bench", "speedup": 1.3}),
                                       encoding="utf-8")
    seen = _count_loads(monkeypatch)
    with _counting_opens(monkeypatch) as opened:
        summary = query.run_summary("conv2d/rich")
    assert summary is not None
    assert [path for path in seen if path.endswith("report.json")] == []
    touched = _reads_under(opened, directory)
    assert [path for path in touched if "candidates" in path and path.endswith("report.json")] == []
    # 索引行没有被这次读取改写：新候选仍要等 tick/refresh 才入库。
    assert get_index().conn.execute("SELECT COUNT(*) FROM candidates WHERE run_id = 'conv2d/rich'").fetchone()[0] == 2


def test_a_feedback_append_is_read_once_when_the_panel_opens(tmp_path, monkeypatch):
    """@brief 验收③：索引已追平后追加一条反馈，面板只读 feedback.jsonl 一次且能看到新条目。

    旧实现先经 ``Index.resolve`` 摄取（读一遍反馈）再 ``entries()``（再读一遍），会打开两次。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 的替换工具。
    """
    _runs, directory = _open_rich(tmp_path, monkeypatch)
    with (directory / "feedback.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_feedback_record(3, "init", raw="SECRET third"), ensure_ascii=False) + "\n")
    with _counting_opens(monkeypatch) as opened:
        panel = query.feedback_panel("conv2d/rich")
    assert panel is not None
    log = [path for path, _mode in opened if path.endswith("feedback.jsonl")]
    assert len(log) == 1, f"feedback.jsonl 被读了 {len(log)} 次"
    records = panel["feedback"]["records"]
    assert [record["seq"] for record in records] == [1, 2, 3]


def test_a_running_run_with_the_same_stamp_reads_nothing(tmp_path, monkeypatch):
    """@brief 验收④补：运行中的 run 同 stamp 重开也不追平转录（旧实现会在 stamp 前 ``_project_turns``）。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 的替换工具。
    """
    runs = _use_temp_runs(tmp_path, monkeypatch)
    directory = runs / "live"
    directory.mkdir(parents=True)
    (directory / "state.json").write_text(json.dumps({
        "pid": os.getpid(), "phase": "plan", "started_at": T0, "updated_at": T0 + 1, "task": "conv2d",
        "dtype": "fp16",
    }), encoding="utf-8")
    (directory / "transcript.jsonl").write_text(json.dumps({
        "event": "chat", "label": "init", "model": "m", "seconds": 1.0, "t": T0 + 0.5,
    }) + "\n", encoding="utf-8")
    Index(tmp_path / "observe.sqlite").sync_known()
    first = query.run_summary("live")
    assert first["detail_stamp"] != "" and first["status"] == "running"

    with (directory / "transcript.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"event": "chat", "label": "r1", "model": "m", "seconds": 1.0, "t": T0 + 0.6}) + "\n")
    with _counting_opens(monkeypatch) as opened:
        again = query.run_summary("live", stamp=first["detail_stamp"])
    assert again is not None and again["same"] is True
    assert _reads_under(opened, directory) == []


def test_the_timeline_stamp_covers_the_transcript_and_the_global_gpu_log(tmp_path, monkeypatch):
    """@brief 验收②补：时间线戳必须覆盖它真正依赖的 transcript 与机全局 GPU 事件日志。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 的替换工具。
    """
    runs = _use_temp_runs(tmp_path, monkeypatch)
    cache = tmp_path / "observe-cache"
    monkeypatch.setenv("AI_LOWERLING_OBSERVE_DIR", str(cache))
    _rich_run(runs)
    gpu = cache / "observe" / "gpu_events.jsonl"
    gpu.parent.mkdir(parents=True)
    gpu.write_text("".join(json.dumps(event) + "\n" for event in (
        {"t": T0 + 4, "pid": 1, "kind": "wait", "mode": "ex", "what": "benchmark of conv2d fp16 paper"},
        {"t": T0 + 6, "pid": 1, "kind": "acquire", "mode": "ex", "what": "benchmark of conv2d fp16 paper"},
    )), encoding="utf-8")
    first = query.timeline_panel("conv2d/rich")
    assert first is not None
    lanes = {lane["label"]: lane["spans"] for lane in first["timeline"]["lanes"]}
    assert len(lanes["等锁"]) == 1 and len(lanes["问模型"]) == 1

    with (runs / "conv2d" / "rich" / "transcript.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"event": "chat", "label": "r1.plan", "model": "m", "seconds": 5.0,
                                 "t": T0 + 10}) + "\n")
    again = query.timeline_panel("conv2d/rich", stamp=first["stamp"])
    assert again is not None and again["same"] is False
    lanes = {lane["label"]: lane["spans"] for lane in again["timeline"]["lanes"]}
    assert len(lanes["问模型"]) == 2

    stamped = again["stamp"]
    unchanged = query.timeline_panel("conv2d/rich", stamp=stamped)
    assert unchanged["same"] is True

    gpu.write_text(gpu.read_text(encoding="utf-8") + json.dumps(
        {"t": T0 + 7, "pid": 1, "kind": "wait", "mode": "ex", "what": "benchmark of conv2d fp16 paper"}
    ) + "\n", encoding="utf-8")
    moved = query.timeline_panel("conv2d/rich", stamp=stamped)
    assert moved["same"] is False


def test_the_grid_stamp_covers_the_confirmation_and_the_run_config(tmp_path, monkeypatch):
    """@brief 验收②补：特征格子戳覆盖确认报告与 run_config（它会影响发布标记与后端签名）。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 的替换工具。
    """
    _runs, directory = _open_rich(tmp_path, monkeypatch)
    first = query.grid_panel("conv2d/rich")
    assert first is not None and first["same"] is False

    report_path = directory / "confirm_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["speedup"] = 1.30
    report_path.write_text(json.dumps(report), encoding="utf-8")
    after_confirm = query.grid_panel("conv2d/rich", stamp=first["stamp"])
    assert after_confirm["same"] is False and after_confirm["stamp"] != first["stamp"]

    config_path = directory / "run_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["target"]["backend_version"] = "v2"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    after_config = query.grid_panel("conv2d/rich", stamp=after_confirm["stamp"])
    assert after_config["same"] is False and after_config["stamp"] != after_confirm["stamp"]


def test_the_feedback_stamp_covers_the_candidate_reports_it_quotes(tmp_path, monkeypatch):
    """@brief 验收②补：反馈记录带候选报告摘要，报告改写必须让反馈面板戳变化。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 的替换工具。
    """
    _runs, directory = _open_rich(tmp_path, monkeypatch)
    first = query.feedback_panel("conv2d/rich")
    path = directory / "candidates" / "001_r1.p0" / "report.json"
    report = json.loads(path.read_text(encoding="utf-8"))
    report["candidate_us"] = 640.0
    path.write_text(json.dumps(report), encoding="utf-8")
    second = query.feedback_panel("conv2d/rich", stamp=first["stamp"])
    assert second is not None and second["same"] is False and second["stamp"] != first["stamp"]


def test_the_feedback_list_has_no_claim_text_and_the_record_endpoint_has_it(tmp_path, monkeypatch):
    """@brief 验收③：列表只留 seq/label/round/kinds/计数与状态；正文、证据、历史、原文只在单条端点。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 的替换工具。
    """
    _runs, _directory = _open_rich(tmp_path, monkeypatch)
    panel = query.feedback_panel("conv2d/rich")
    records = panel["feedback"]["records"]
    # §10.4：记录一页 10 条，“加载更多”靠 ``next``；总数在 ``total``。
    assert set(panel["feedback"]) == {"records", "total", "next", "page_index"}
    assert panel["feedback"]["total"] >= len(records)
    record = records[0]
    assert set(record) == {
        "seq", "t", "label", "round", "committee", "provider", "model", "effort",
        "outcome", "exit", "best_speedup", "seed_label", "kinds", "backend", "target_id",
        "truncated", "truncated_sections", "report_path", "report", "claims", "claim_counts",
    }
    assert record["claim_counts"]["total"] == len(record["claims"])
    for claim in record["claims"]:
        assert set(claim) == {
            "digest", "section", "audience", "premise", "status", "verification", "topic",
            "subclaims_count", "open_subclaims_count",
        }
    body = json.dumps(panel, ensure_ascii=False)
    assert '"raw":' not in body and '"text":' not in body and '"evidence":' not in body
    assert '"history":' not in body and '"subclaims":' not in body
    assert _SECRET not in body and "缺一把探针" not in body
    assert record["report"]["candidate_us"] is not None

    full = query.feedback_record("conv2d/rich", 1)
    assert full is not None
    assert full["record"]["raw"] == _SECRET
    assert full["record"]["claims"][0]["text"]
    assert "calibration" in full


def test_candidate_filters_paging_and_the_leaf_panel_stamps(tmp_path, monkeypatch):
    """@brief §5.2 端点契约：候选支持 status/stage/limit/cursor；单颗候选与单条反馈也有 stamp 短路。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 的替换工具。
    """
    _runs, directory = _open_rich(tmp_path, monkeypatch)
    page = query.candidates_panel("conv2d/rich", limit=1)
    assert page["total"] == 2 and len(page["candidates"]) == 1 and page["next"]
    rest = query.candidates_panel("conv2d/rich", limit=1, cursor=page["next"])
    assert [row["label"] for row in rest["candidates"]] == ["init"] and rest["next"] is None
    assert query.candidates_panel("conv2d/rich", stage="nope")["total"] == 0
    assert query.candidates_panel("conv2d/rich", status="ok")["total"] == 2
    assert query.candidates_panel("conv2d/rich", status="failed")["total"] == 0

    one = query.candidate_panel("conv2d/rich", "r1.p0")
    detail._target_identity_cached.cache_clear()  # 冷缓存也不许为了取内核名读配置
    with _counting_opens(monkeypatch) as opened:
        same = query.candidate_panel("conv2d/rich", "r1.p0", stamp=one["stamp"])
    assert same["same"] is True and _reads_under(opened, directory) == []
    assert [path for path, _mode in opened if path.endswith("run_config.json")] == []

    record = query.feedback_record("conv2d/rich", 1)
    detail._target_identity_cached.cache_clear()  # 冷缓存也要能只 stat 地算戳
    with _counting_opens(monkeypatch) as opened:
        same_record = query.feedback_record("conv2d/rich", 1, stamp=record["stamp"])
    assert same_record["same"] is True and _reads_under(opened, directory) == []
    # 戳的计算也不解析目标档案：目标 id 来自索引，标定路径是拼出来的。
    assert [path for path, _mode in opened if path.endswith("target.toml")] == []
    assert [path for path, _mode in opened if path.endswith("run_config.json")] == []


def test_a_state_only_run_derives_stop_reason_the_old_way(tmp_path, monkeypatch):
    """@brief 验收①：没有 result.json 时 stop_reason 仍取 state.stop_reason，与旧详情一致。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 的替换工具。
    """
    runs = _use_temp_runs(tmp_path, monkeypatch)
    directory = runs / "stopped"
    directory.mkdir(parents=True)
    (directory / "state.json").write_text(json.dumps({
        "task": "conv2d", "dtype": "fp16", "phase": "done", "started_at": 1.0, "updated_at": 2.0,
        "stop_reason": "budget_exhausted",
    }), encoding="utf-8")
    query.refresh()
    summary = query.run_summary("stopped")
    full = query.run_detail("stopped")
    assert summary["stop_reason"] == "budget_exhausted"
    assert summary["stop_reason"] == full["stop_reason"]


def test_a_cold_summary_reads_no_candidate_report_and_no_feedback_log(tmp_path, monkeypatch):
    """@brief 验收②：冷开只发摘要；未展开面板不读 report.json / feedback.jsonl。

    索引已在“先看列表页”那一步追平（生产请求路径不调 ``sync_known``），所以这里先 refresh 再清零计数。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 的替换工具。
    """
    _runs, directory = _open_rich(tmp_path, monkeypatch)
    seen = _count_loads(monkeypatch)
    with _counting_opens(monkeypatch) as opened:
        summary = query.run_summary("conv2d/rich")
    assert summary is not None

    assert [path for path in seen if path.endswith("report.json")] == []
    touched = _reads_under(opened, directory)
    assert [path for path in touched if path.endswith("feedback.jsonl")] == []
    # 发布结论要核对 best.metal 与胜者 kernel 的摘要，所以候选目录里会读内核；不许读的是候选报告。
    assert [path for path in touched if "candidates" in path and path.endswith("report.json")] == []


def test_the_pending_evaluation_table_matches_the_outbox_and_follows_an_in_place_rewrite(tmp_path, monkeypatch):
    """@brief §5.7：``run_pending_eval`` 与旧 ``pending_evaluation()`` 逐 run 一致，且跟随后到的原地改写。

    夹具含：正常 pending、unavailable（带 error）、坏 JSON、``receipts.json`` 与 ``*.result.json``
    （两者不算待评测），以及 10 条请求（表里只留最后 8 条，但计数是全量）。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 的替换工具。
    """
    runs = _use_temp_runs(tmp_path, monkeypatch)
    monkeypatch.setenv("OBSERVE_FULL_TICK_INTERVAL", "0")  # 让 tick 走生产里的周期性全量对账
    directory = _write_run(runs, "conv2d/pending")
    outbox = directory / "evaluation_outbox"
    outbox.mkdir()
    for index in range(10):
        status = "pending" if index % 2 else "unavailable"
        (outbox / f"req-{index:02d}.json").write_text(json.dumps({
            "request_id": f"req-{index}", "status": status,
            "error": {"code": "timeout", "retryable": index % 2 == 0},
        }), encoding="utf-8")
    (outbox / "broken.json").write_text("{not json", encoding="utf-8")
    (outbox / "receipts.json").write_text(json.dumps({"request_id": "r", "status": "pending"}), encoding="utf-8")
    (outbox / "done.result.json").write_text(json.dumps({"request_id": "d", "status": "pending"}), encoding="utf-8")
    index = Index(tmp_path / "observe.sqlite")
    index.sync_known()

    old = detail.pending_evaluation(directory)
    assert old["count"] == 10 and len(old["requests"]) == 8
    stored = index.stored_pending_evaluation("conv2d/pending")
    assert stored == old, "窄投影必须与旧 pending_evaluation() 逐字段一致"
    row = dict(index.conn.execute(
        "SELECT pending, unavailable, last_error_code, retryable, requests_json"
        " FROM run_pending_eval WHERE run_id = 'conv2d/pending'"
    ).fetchone())
    assert row["pending"] == 5 and row["unavailable"] == 5 and row["retryable"] == 0
    assert row["last_error_code"] == "timeout"
    assert len(json.loads(row["requests_json"])) == 8

    # 摘要读表，不去 glob outbox。
    real_pending = detail._pending_requests

    def _forbidden(_directory):
        raise AssertionError("summary read the outbox")

    monkeypatch.setattr(detail, "_pending_requests", _forbidden)
    summary = query.run_summary("conv2d/pending")
    assert summary is not None and summary["pending_evaluation"] == old
    monkeypatch.setattr(detail, "_pending_requests", real_pending)

    # 就地改写一份请求（目录 mtime 不变）：下一次摄入必须把表改过来（fp_extra 里带了 outbox 逐文件指纹）。
    before = directory.stat().st_mtime_ns
    confirmed_before = index._fingerprints(directory)[4]
    (outbox / "req-03.json").write_text(json.dumps({
        "request_id": "req-03", "status": "pending", "error": {"code": "ok", "retryable": False},
    }), encoding="utf-8")
    assert directory.stat().st_mtime_ns == before, "就地改写单个文件不该改运行目录 mtime"
    assert index._fingerprints(directory)[4] != confirmed_before, "outbox 指纹必须包含逐文件 stat"
    index.tick()
    updated = index.stored_pending_evaluation("conv2d/pending")
    assert updated == detail.pending_evaluation(directory)
    assert updated["count"] == 10
    assert next(item for item in updated["requests"] if item["request_id"] == "req-03")["error_code"] == "ok"


def test_missing_outbox_means_an_empty_pending_projection(tmp_path, monkeypatch):
    """@brief §5.7：没有 ``evaluation_outbox/`` 时旧函数与表都给空计数。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 的替换工具。
    """
    runs = _use_temp_runs(tmp_path, monkeypatch)
    directory = _write_run(runs, "conv2d/plain")
    index = Index(tmp_path / "observe.sqlite")
    index.sync_known()
    assert detail.pending_evaluation(directory) == {"count": 0, "requests": []}
    assert index.stored_pending_evaluation("conv2d/plain") == {"count": 0, "requests": []}


def test_the_feedback_panel_reads_the_log_once_and_keeps_raw_out_of_the_list(tmp_path, monkeypatch):
    """@brief 验收③：展开 feedback 只读 feedback.jsonl 一次，响应体不含 ``raw``；单条全文另有端点。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 的替换工具。
    """
    _runs, directory = _open_rich(tmp_path, monkeypatch)
    with _counting_opens(monkeypatch) as opened:
        panel = query.feedback_panel("conv2d/rich")
    assert panel is not None
    log = [path for path, _mode in opened if path.endswith("feedback.jsonl")]
    assert len(log) == 1, f"feedback.jsonl 被读了 {len(log)} 次"
    body = json.dumps(panel, ensure_ascii=False)
    assert '"raw":' not in body
    records = panel["feedback"]["records"]
    assert len(records) == 2
    assert all("raw" not in record for record in records)
    assert records[0]["claims"], "列表仍要带 claim 计数与状态"

    single = query.feedback_record("conv2d/rich", records[0]["seq"])
    assert single is not None and single["record"]["raw"] == _SECRET
    assert query.feedback_record("conv2d/rich", 99999) is None


def test_a_matching_stamp_reopens_every_panel_without_reading_the_run(tmp_path, monkeypatch):
    """@brief 验收④：同 stamp 重复打开摘要与各面板，读字节为 0（连一个源文件都不打开）。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 的替换工具。
    """
    _runs, directory = _open_rich(tmp_path, monkeypatch)
    summary = query.run_summary("conv2d/rich")
    assert summary is not None
    panels = {
        "candidates": query.candidates_panel("conv2d/rich"),
        "feedback": query.feedback_panel("conv2d/rich"),
        "timeline": query.timeline_panel("conv2d/rich"),
        "budget-events": query.budget_events_panel("conv2d/rich"),
        "llm": query.llm_panel("conv2d/rich"),
        "grid": query.grid_panel("conv2d/rich"),
        "plan": query.plan_panel("conv2d/rich"),
    }
    loaders = {
        "candidates": query.candidates_panel, "feedback": query.feedback_panel,
        "timeline": query.timeline_panel, "budget-events": query.budget_events_panel,
        "llm": query.llm_panel, "grid": query.grid_panel, "plan": query.plan_panel,
    }
    assert all(panel and panel.get("same") is False for panel in panels.values())

    with _counting_opens(monkeypatch) as opened:
        again = query.run_summary("conv2d/rich", stamp=summary["detail_stamp"])
        same_panels = {name: loader("conv2d/rich", stamp=panels[name]["stamp"]) for name, loader in loaders.items()}
    assert again is not None and again["same"] is True
    assert all(panel is not None and panel.get("same") is True for panel in same_panels.values())
    assert _reads_under(opened, directory) == []


def test_rewriting_one_candidate_report_changes_the_candidates_stamp_only(tmp_path, monkeypatch):
    """@brief 候选面板 stamp 守卫：就地改写一颗 report.json（不动目录 mtime）必须让 stamp 变化。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 的替换工具。
    """
    _runs, directory = _open_rich(tmp_path, monkeypatch)
    first = query.candidates_panel("conv2d/rich")
    assert first is not None
    root = directory / "candidates"
    folder = root / "001_r1.p0"
    before = (root.stat().st_mtime_ns, folder.stat().st_mtime_ns)

    path = folder / "report.json"
    report = json.loads(path.read_text(encoding="utf-8"))
    report["speedup"] = 2.5
    path.write_text(json.dumps(report), encoding="utf-8")
    os.utime(path, ns=(path.stat().st_atime_ns, path.stat().st_mtime_ns + 1_000_000))
    after = (root.stat().st_mtime_ns, folder.stat().st_mtime_ns)
    assert after == before, "就地改写报告不应改变目录 mtime（否则这条守卫测的是别的东西）"
    assert (folder / "report.json").stat().st_mtime_ns != 0

    second = query.candidates_panel("conv2d/rich", stamp=first["stamp"])
    assert second is not None and second["same"] is False
    assert second["stamp"] != first["stamp"]
    by_label = {row["label"]: row for row in second["candidates"]}
    assert by_label["r1.p0"]["speedup"] == pytest.approx(2.5)


def test_the_candidate_kernel_uses_the_target_file_and_the_stamp_covers_every_kernel(tmp_path, monkeypatch):
    """@brief 候选内核走 target 的权威文件名；stamp 覆盖全部 ``kernel.*``，混入 ``kernel.aaa`` 也不失明。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 的替换工具。
    """
    _runs, directory = _open_rich(tmp_path, monkeypatch)
    folder = directory / "candidates" / "001_r1.p0"
    (folder / "kernel.aaa").write_text("kernel void aaa() {}\n", encoding="utf-8")

    one = query.candidate_panel("conv2d/rich", "r1.p0")
    assert one["excerpt"] == _SOURCE, "payload 必须取权威的 kernel.metal，不能取字典序第一个"
    listed = query.candidates_panel("conv2d/rich")
    row = next(item for item in listed["candidates"] if item["label"] == "r1.p0")
    assert row["excerpt"] == _SOURCE

    # 只改权威 kernel.metal（kernel.aaa 不动）：旧戳不能命中。
    (folder / "kernel.metal").write_text("kernel void changed() {}\n", encoding="utf-8")
    after = query.candidate_panel("conv2d/rich", "r1.p0", stamp=one["stamp"])
    assert after["same"] is not True
    assert after["excerpt"] == "kernel void changed() {}\n"
    listed_after = query.candidates_panel("conv2d/rich", stamp=listed["stamp"])
    assert listed_after["same"] is not True
    row_after = next(item for item in listed_after["candidates"] if item["label"] == "r1.p0")
    assert row_after["excerpt"] == "kernel void changed() {}\n"


def test_renaming_a_candidate_dir_changes_the_single_candidate_stamp(tmp_path, monkeypatch):
    """@brief 单候选戳覆盖响应里的路径字段：目录改名（idx/report_path 变）必须让旧戳失效。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 的替换工具。
    """
    _runs, directory = _open_rich(tmp_path, monkeypatch)
    one = query.candidate_panel("conv2d/rich", "r1.p0")
    assert one["idx"] == 1
    source = directory / "candidates" / "001_r1.p0"
    source.rename(directory / "candidates" / "009_r1.p0")
    after = query.candidate_panel("conv2d/rich", "r1.p0", stamp=one["stamp"])
    assert after["same"] is not True
    assert after["idx"] == 9
    assert after["report_path"].endswith("candidates/009_r1.p0/report.json")


def test_the_grid_payload_refreshes_after_a_report_or_best_rewrite(tmp_path, monkeypatch):
    """@brief 特征格子：候选报告就地改写要刷新数值（缓存身份），只改 best.metal 要刷新 published。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 的替换工具。
    """
    _runs, directory = _open_rich(tmp_path, monkeypatch)
    first = query.grid_panel("conv2d/rich")
    assert first["same"] is False
    cell = first["feature_grid"][0]
    assert cell["label"] == "r1.p0" and cell["speedup"] == pytest.approx(1.25) and cell["published"] is True

    report_path = directory / "candidates" / "001_r1.p0" / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["speedup"] = 9.0
    report_path.write_text(json.dumps(report), encoding="utf-8")
    after_report = query.grid_panel("conv2d/rich", stamp=first["stamp"])
    assert after_report["same"] is False
    assert after_report["feature_grid"][0]["speedup"] == pytest.approx(9.0), "缓存不能返回旧的 1.25"

    (directory / "best.metal").write_text("kernel void other() {}\n", encoding="utf-8")
    after_best = query.grid_panel("conv2d/rich", stamp=after_report["stamp"])
    assert after_best["same"] is False
    assert after_best["feature_grid"][0]["published"] is False


def test_the_candidates_stamp_is_bound_to_the_page_shape(tmp_path, monkeypatch):
    """@brief 候选面板戳绑定页面形状：limit 变化、过滤响应、续页都不能用旧戳短路掉另一种形状。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 的替换工具。
    """
    _runs, directory = _open_rich(tmp_path, monkeypatch)
    first = query.candidates_panel("conv2d/rich", limit=1)
    assert len(first["candidates"]) == 1 and first["next"]

    # limit=1 的戳不能短路 limit=2：后者必须拿到两行与自己的 next。
    wider = query.candidates_panel("conv2d/rich", limit=2, stamp=first["stamp"])
    assert wider.get("same") is not True
    assert len(wider["candidates"]) == 2 and wider["next"] is None
    assert wider["stamp"] != first["stamp"]

    # 过滤响应的戳不能短路未过滤请求。
    filtered = query.candidates_panel("conv2d/rich", status="ok", limit=2)
    unfiltered = query.candidates_panel("conv2d/rich", limit=2, stamp=filtered["stamp"])
    assert unfiltered.get("same") is not True
    assert len(unfiltered["candidates"]) == 2

    # 续页请求也有自己的形状：首页的戳不能短路第二页。
    second = query.candidates_panel("conv2d/rich", limit=1, cursor=first["next"], stamp=first["stamp"])
    assert second.get("same") is not True
    assert [row["label"] for row in second["candidates"]] == ["init"]

    # 同形状 + 同戳仍然 0 读取。
    same_page = query.candidates_panel("conv2d/rich", limit=1)
    with _counting_opens(monkeypatch) as opened:
        again = query.candidates_panel("conv2d/rich", limit=1, stamp=same_page["stamp"])
    assert again.get("same") is True
    assert _reads_under(opened, directory) == []


def test_filtered_candidates_use_current_reports_and_page(tmp_path, monkeypatch):
    """@brief 过滤分页按当前报告判定：就地改状态/新增候选立刻生效，零匹配不退回全扫。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 的替换工具。
    """
    runs, _directory = _open_rich(tmp_path, monkeypatch)
    directory = runs / "conv2d" / "rich"
    # 索引已收录两颗候选；现在就地改 r1 的状态（candidates/ 目录 mtime 不变），再加一颗新候选。
    report_path = directory / "candidates" / "001_r1.p0" / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["status"] = "failed"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    fresh = directory / "candidates" / "002_r2.p0"
    fresh.mkdir()
    (fresh / "kernel.metal").write_text(_SOURCE, encoding="utf-8")
    (fresh / "report.json").write_text(json.dumps({"schema_version": 4, "status": "ok", "stage": "bench", "speedup": 1.4}),
                                        encoding="utf-8")

    ok = query.candidates_panel("conv2d/rich", status="ok")
    assert [row["label"] for row in ok["candidates"]] == ["r2.p0", "init"], "旧状态与新候选都按当前报告判定，且首页 newest-first"
    assert ok["total"] == 2
    failed = query.candidates_panel("conv2d/rich", status="failed")
    assert [row["label"] for row in failed["candidates"]] == ["r1.p0"]

    zero = query.candidates_panel("conv2d/rich", status="nope")
    assert zero["total"] == 0 and zero["candidates"] == []

    # 过滤分页：limit=1 走两页，游标是目录名。
    first = query.candidates_panel("conv2d/rich", status="ok", limit=1)
    assert [row["label"] for row in first["candidates"]] == ["r2.p0"] and first["next"]
    second = query.candidates_panel("conv2d/rich", status="ok", limit=1, cursor=first["next"])
    assert [row["label"] for row in second["candidates"]] == ["init"] and second["next"] is None


def test_candidate_paging_reads_only_the_requested_page(tmp_path, monkeypatch):
    """@brief 验收 PR3：250 颗候选、``limit=1`` 只读当前页的 report，不先物化全部。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 的替换工具。
    """
    runs, _directory = _open_rich(tmp_path, monkeypatch)
    directory = runs / "conv2d" / "rich"
    for index in range(2, 250):
        folder = directory / "candidates" / f"{index:03d}_c{index}"
        folder.mkdir()
        (folder / "kernel.metal").write_text(_SOURCE, encoding="utf-8")
        (folder / "report.json").write_text(json.dumps({
            "status": "ok", "stage": "bench", "speedup": 1.0 + index / 100, "message": f"c{index}",
        }), encoding="utf-8")

    with _counting_opens(monkeypatch) as opened:
        page = query.candidates_panel("conv2d/rich", limit=1)
    reports = [
        path for path, _mode in opened
        if "candidates" in path and path.endswith("report.json")
    ]
    assert len(page["candidates"]) == 1 and page["total"] == 250 and page["next"]
    assert len(reports) == 1, f"limit=1 读了 {len(reports)} 份候选报告"

    seen = {page["candidates"][0]["label"]}
    cursor = page["next"]
    while cursor:
        following = query.candidates_panel("conv2d/rich", limit=50, cursor=cursor)
        seen.update(row["label"] for row in following["candidates"])
        cursor = following["next"]
    assert len(seen) == 250


def test_a_non_integer_candidate_limit_is_a_400(tmp_path, monkeypatch):
    """@brief ``/candidates?limit=abc`` 返回 400，不是 500。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 的替换工具。
    """
    from starlette.testclient import TestClient

    from observe.app import app

    _open_rich(tmp_path, monkeypatch)
    with TestClient(app) as client:
        response = client.get("/api/v2/runs/conv2d/rich/candidates?limit=abc")
        assert response.status_code == 400
        assert "limit" in response.json()["error"]


def test_the_candidate_panel_route_returns_one_report_and_a_bounded_excerpt(tmp_path, monkeypatch):
    """@brief 单颗候选端点：完整 report、相对路径与有界源码片段；不存在的标签返回 None。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 的替换工具。
    """
    _runs, _directory = _open_rich(tmp_path, monkeypatch)
    one = query.candidate_panel("conv2d/rich", "r1.p0")
    assert one is not None
    assert one["idx"] == 1 and one["label"] == "r1.p0"
    assert one["report"]["message"] == "report r1.p0"
    assert one["report_path"].endswith("candidates/001_r1.p0/report.json")
    assert one["excerpt"] == _SOURCE
    assert query.candidate_panel("conv2d/rich", "nope") is None


def _confirmed_run(root: Path, name: str, *, speedup: float = 1.1) -> Path:
    """@brief 造一个已通过确认门的瘦运行：确认报告、result、best 内核与胜者该有的都在。

    @param root runs 根目录。
    @param name run id（相对 runs/）。
    @param speedup 确认加速比。
    @return 新建的 run 目录。
    """
    directory = root / name
    (directory / "candidates" / "000_r1.p0").mkdir(parents=True)
    directory.joinpath("best.metal").write_text(_SOURCE, encoding="utf-8")
    directory.joinpath("candidates", "000_r1.p0", "kernel.metal").write_text(_SOURCE, encoding="utf-8")
    directory.joinpath("state.json").write_text(json.dumps({
        "pid": 1, "phase": "done", "started_at": 10.0, "updated_at": 20.0, "task": "conv2d", "dtype": "fp16",
    }), encoding="utf-8")
    directory.joinpath("run_config.json").write_text(json.dumps({
        "target": {"id": "apple-m4-max", "backend": "metal", "backend_version": "v1"},
    }), encoding="utf-8")
    _write_confirmation(directory, speedup)
    return directory


def _write_confirmation(directory: Path, speedup: float) -> None:
    """@brief 就地写 result.json 与 confirm_report.json，让它们绑定同一份内核与同一个数字。

    两个文件都是就地改写：目录条目不增不减，所以 run 目录与 ``runs/`` 根的 mtime 都不变。

    @param directory 运行目录。
    @param speedup 确认加速比。
    """
    from experiments.common import synthetic_confirmation_documents

    # The documents must pass the one publication gate: a synthetic run may not claim a conclusion
    # the real gate would refuse (plan §8).  Schema 4 only — an older confirm report is undecidable.
    result, report = synthetic_confirmation_documents(
        task="conv2d", dtype="fp16", size="paper", target_id="apple-m4-max",
        speedup=speedup, label="r1.p0", kernel_sha256=_digest(),
    )
    result.update({"stop_reason": "max_rounds", "started_at": 10.0, "spent_usd": 0.5})
    directory.joinpath("result.json").write_text(json.dumps(result), encoding="utf-8")
    directory.joinpath("confirm_report.json").write_text(json.dumps(report), encoding="utf-8")


def test_best_reingests_only_the_confirmed_run_whose_fingerprint_changed(tmp_path, monkeypatch):
    """@brief 验收②：100 场已确认运行里只改 1 场，``_ingest`` 计数 = 1，且新值出现。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 的替换工具。
    """
    runs = _use_temp_runs(tmp_path, monkeypatch)
    for index in range(100):
        _confirmed_run(runs, f"confirmed/{index:03d}")
    Index(tmp_path / "observe.sqlite").sync_known()
    assert query.best_confirmed(limit=100)["best"][0]["published_speedup"] == pytest.approx(1.1)

    touched = runs / "confirmed" / "042"
    root_mtime = runs.stat().st_mtime_ns
    _write_confirmation(touched, 1.75)
    assert runs.stat().st_mtime_ns == root_mtime, "就地改写不应改变 runs/ 根目录 mtime"

    calls: list[str] = []
    real = Index._ingest

    def counting(self, directory, *, force=False):
        calls.append(str(directory))
        return real(self, directory, force=force)

    monkeypatch.setattr(Index, "_ingest", counting)
    best = query.best_confirmed(limit=100)
    assert len(calls) == 1, f"只想重新摄取 1 场，实际 {len(calls)} 次"
    assert calls[0].endswith("confirmed/042")
    assert best["best"][0]["run_id"] == "confirmed/042"
    assert best["best"][0]["published_speedup"] == pytest.approx(1.75)


def test_the_best_endpoint_sees_an_in_place_confirmation_rewrite(tmp_path, monkeypatch):
    """@brief 验收①回归：改确认文件、不改 runs/ 根 mtime，``/api/v2/best`` 仍要拿到新值。

    第二步只改 ``confirm_report.json``（让它与 result 不一致）：行必须从 best 里消失，
    证明确认指纹本身被核对，而不是碰巧被 result 的指纹带出来。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 的替换工具。
    """
    from starlette.testclient import TestClient

    from observe.app import app

    runs = _use_temp_runs(tmp_path, monkeypatch)
    directory = _confirmed_run(runs, "conv2d/winner")
    Index(tmp_path / "observe.sqlite").sync_known()
    root_mtime = runs.stat().st_mtime_ns

    with TestClient(app) as client:
        before = client.get("/api/v2/best").json()["best"]
        assert [row["published_speedup"] for row in before] == [pytest.approx(1.1)]

        _write_confirmation(directory, 1.4)
        assert runs.stat().st_mtime_ns == root_mtime
        after = client.get("/api/v2/best").json()["best"]
        assert [row["published_speedup"] for row in after] == [pytest.approx(1.4)]

        report = json.loads(directory.joinpath("confirm_report.json").read_text(encoding="utf-8"))
        report["speedup"] = 1.3
        report["benchmark"]["confirmation"]["speedup"] = 1.3
        directory.joinpath("confirm_report.json").write_text(json.dumps(report), encoding="utf-8")
        assert runs.stat().st_mtime_ns == root_mtime, "只改确认报告也不该动根目录 mtime"
        assert client.get("/api/v2/best").json()["best"] == []


def test_the_confirmation_stat_pass_does_not_hold_the_write_lock(tmp_path, monkeypatch):
    """@brief 验收③：确认指纹的 stat 阶段在 ``_WRITE`` 之外；没有差异时不摄取任何行。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 的替换工具。
    """
    import observe.index as index_mod

    runs = _use_temp_runs(tmp_path, monkeypatch)
    _confirmed_run(runs, "conv2d/one")
    _confirmed_run(runs, "conv2d/two")
    index_mod.Index(tmp_path / "observe.sqlite").sync_known()

    lock_states: list[bool] = []
    real = index_mod._fingerprint

    def spy(path):
        lock_states.append(index_mod._WRITE.locked())
        return real(path)

    monkeypatch.setattr(index_mod, "_fingerprint", spy)
    calls: list[str] = []
    real_ingest = index_mod.Index._ingest

    def counting(self, directory, *, force=False):
        calls.append(str(directory))
        return real_ingest(self, directory, force=force)

    monkeypatch.setattr(index_mod.Index, "_ingest", counting)
    best = query.best_confirmed()
    assert [row["run_id"] for row in best["best"]] == ["conv2d/one", "conv2d/two"]
    assert calls == []
    assert lock_states, "指纹阶段没被调用，这条测试就什么也没证明"
    assert not any(lock_states), "stat 阶段不该持全局写锁"


def _result_only_confirmed_run(root: Path, name: str, *, mtime_ns: int, speedup: float = 1.1) -> Path:
    """@brief 造一个没有 state.json 的已确认 run；``started_at`` 只能退回目录 mtime。

    @param root runs 根目录。
    @param name run id（相对 runs/）。
    @param mtime_ns 目录 mtime，用来控制排序。
    @param speedup 确认加速比。
    @return 新建的 run 目录。
    """
    directory = root / name
    (directory / "candidates" / "000_r1.p0").mkdir(parents=True)
    directory.joinpath("best.metal").write_text(_SOURCE, encoding="utf-8")
    directory.joinpath("candidates", "000_r1.p0", "kernel.metal").write_text(_SOURCE, encoding="utf-8")
    directory.joinpath("run_config.json").write_text(json.dumps({
        "target": {"id": "apple-m4-max", "backend": "metal", "backend_version": "v1"},
    }), encoding="utf-8")
    _write_confirmation(directory, speedup)
    os.utime(directory, ns=(mtime_ns, mtime_ns))
    return directory


def test_best_orders_result_only_runs_by_the_directory_mtime(tmp_path, monkeypatch):
    """@brief 没有 started_at 的 run 用目录 mtime 排序；目录 mtime 变了，/best 的排序也要跟着变。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 的替换工具。
    """
    from starlette.testclient import TestClient

    from observe.app import app

    runs = _use_temp_runs(tmp_path, monkeypatch)
    _result_only_confirmed_run(runs, "conv2d/older", mtime_ns=1_000_000_000_000_000_000)
    _result_only_confirmed_run(runs, "conv2d/newer", mtime_ns=1_000_000_100_000_000_000)
    Index(tmp_path / "observe.sqlite").sync_known()
    with TestClient(app) as client:
        order = [row["run_id"] for row in client.get("/api/v2/best").json()["best"]]
        assert order == ["conv2d/newer", "conv2d/older"]
        # 只改目录 mtime（等价于顶层增删文件），started_at 的回退值随之改变。
        old = runs / "conv2d" / "older"
        os.utime(old, ns=(1_000_000_200_000_000_000, 1_000_000_200_000_000_000))
        order = [row["run_id"] for row in client.get("/api/v2/best").json()["best"]]
        assert order == ["conv2d/older", "conv2d/newer"]


def test_an_unchanged_result_only_run_is_not_reingested(tmp_path, monkeypatch):
    """@brief 未修改的 result-only confirmed run 第二次 ``/best`` 不能再摄取；只有目录 mtime 变了才摄取。

    摄取与定向核对必须用同一条 fp_state 规则（有 state.json 用文件，没有就用目录），
    否则已存的目录指纹永远不等于现算的空文件指纹。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 的替换工具。
    """
    runs = _use_temp_runs(tmp_path, monkeypatch)
    directory = _result_only_confirmed_run(runs, "conv2d/only", mtime_ns=1_000_000_000_000_000_000)
    Index(tmp_path / "observe.sqlite").sync_known()
    assert [row["run_id"] for row in query.best_confirmed()["best"]] == ["conv2d/only"]

    calls: list[str] = []
    real = Index._ingest

    def counting(self, path, *, force=False):
        calls.append(str(path))
        return real(self, path, force=force)

    monkeypatch.setattr(Index, "_ingest", counting)
    query.best_confirmed()
    assert query.best_confirmed()["best"][0]["run_id"] == "conv2d/only"
    assert calls == [], f"未修改的 result-only run 不该重摄取，实际 {calls}"

    os.utime(directory, ns=(1_000_000_100_000_000_000, 1_000_000_100_000_000_000))
    query.best_confirmed()
    assert len(calls) == 1, "目录 mtime 变了才该重摄取（started_at 会跟着变）"
    assert query.best_confirmed()["best"][0]["run_id"] == "conv2d/only"


def test_schema_11_discovery_survives_deleting_the_last_run_in_a_parent(tmp_path, monkeypatch):
    """@brief schema-11 结构下，某父目录的最后一场被删后，同目录新建的 sibling 仍要被发现。

    ``run_parents`` 缺失时，父目录从**历史** run_id（含 missing）与顶层 watched 目录恢复；
    只看 ``status != 'missing'`` 会在“最后一场已删除”时丢掉这个父目录。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 的替换工具。
    """
    import observe.index as index_mod

    runs = _use_temp_runs(tmp_path, monkeypatch)
    _confirmed_run(runs, "conv2d/old")
    index = Index(tmp_path / "observe.sqlite")
    index.sync_known()
    _shape_like_schema_11(index)
    index_mod._SINGLE = index

    shutil.rmtree(runs / "conv2d" / "old")
    assert query.best_confirmed()["best"] == []
    row = index.conn.execute("SELECT status FROM runs WHERE run_id = 'conv2d/old'").fetchone()
    assert row["status"] == "missing"

    _confirmed_run(runs, "conv2d/new")
    assert [row["run_id"] for row in query.best_confirmed()["best"]] == ["conv2d/new"]


def test_best_deletion_works_without_the_confirmed_best_table(tmp_path, monkeypatch):
    """@brief PR4 不依赖 schema 12：删掉 ``confirmed_best`` 后，删除回归仍然成立。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 的替换工具。
    """
    from starlette.testclient import TestClient

    from observe.app import app

    runs = _use_temp_runs(tmp_path, monkeypatch)
    _confirmed_run(runs, "conv2d/gone")
    _confirmed_run(runs, "conv2d/kept")
    index = Index(tmp_path / "observe.sqlite")
    index.sync_known()
    index.conn.execute("DROP TABLE confirmed_best")  # 模拟尚未落地 schema 12 的环境
    index.conn.commit()
    with TestClient(app) as client:
        assert len(client.get("/api/v2/best").json()["best"]) == 2
        shutil.rmtree(runs / "conv2d" / "gone")
        remaining = [row["run_id"] for row in client.get("/api/v2/best").json()["best"]]
        assert remaining == ["conv2d/kept"]
    row = get_index().conn.execute(
        "SELECT status, publication, published_speedup FROM runs WHERE run_id = 'conv2d/gone'"
    ).fetchone()
    assert dict(row) == {"status": "missing", "publication": None, "published_speedup": None}


def test_best_discovers_a_new_confirmed_run_without_another_entry_point(tmp_path, monkeypatch):
    """@brief 验收 PR4 等价：索引里还没有的新 confirmed run，下一次 ``/best`` 自己要能发现。

    新 run 一定先改它父目录的 mtime；发现路径只 stat “历史上装过 run 的父目录”，不扫全部 watched。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 的替换工具。
    """
    runs = _use_temp_runs(tmp_path, monkeypatch)
    Index(tmp_path / "observe.sqlite").sync_known()
    assert query.best_confirmed()["best"] == []

    _confirmed_run(runs, "conv2d/one")
    assert [row["run_id"] for row in query.best_confirmed()["best"]] == ["conv2d/one"]

    # 父目录已经收录过：新 run 只改父目录 mtime，也要被发现。
    _confirmed_run(runs, "conv2d/two")
    assert sorted(row["run_id"] for row in query.best_confirmed()["best"]) == ["conv2d/one", "conv2d/two"]

    # 更深一层：先有 conv2d/exp/old，再有 conv2d/exp/new。
    _confirmed_run(runs, "conv2d/exp/old")
    query.best_confirmed()
    _confirmed_run(runs, "conv2d/exp/new")
    assert "conv2d/exp/new" in {row["run_id"] for row in query.best_confirmed()["best"]}


def test_best_discovers_a_new_run_next_to_an_existing_unconfirmed_run(tmp_path, monkeypatch):
    """@brief 验收 PR4：同父目录里已有一场“非确认”run 时，新 confirmed run 仍要被 /best 发现。

    父目录由**每个**有效 run 记录；如果只在 confirmed 时记录，这场新 run 的父目录就不在发现集合里。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 的替换工具。
    """
    runs = _use_temp_runs(tmp_path, monkeypatch)
    _confirmed_run(runs, "conv2d/existing-unconfirmed")
    unconfirmed = runs / "conv2d" / "existing-unconfirmed"
    # 把它改成未通过发布门的运行：result/confirm 不一致。
    report = json.loads((unconfirmed / "confirm_report.json").read_text(encoding="utf-8"))
    report["speedup"] = 1.30
    (unconfirmed / "confirm_report.json").write_text(json.dumps(report), encoding="utf-8")
    Index(tmp_path / "observe.sqlite").sync_known()
    assert query.best_confirmed()["best"] == []
    assert get_index().conn.execute("SELECT COUNT(*) FROM run_parents").fetchone()[0] >= 1

    _confirmed_run(runs, "conv2d/new-confirmed")
    assert [row["run_id"] for row in query.best_confirmed()["best"]] == ["conv2d/new-confirmed"]


def test_switching_the_runs_root_clears_run_parents(tmp_path, monkeypatch):
    """@brief 验收 PR4：``runs/`` 根切换时，父目录发现集合必须跟着清掉，不能拿旧根的目录去 stat。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 的替换工具。
    """
    first = _use_temp_runs(tmp_path, monkeypatch)
    _confirmed_run(first, "conv2d/one")
    query.best_confirmed()
    assert get_index().conn.execute("SELECT COUNT(*) FROM run_parents").fetchone()[0] >= 1

    second = tmp_path / "runs2"
    second.mkdir()
    monkeypatch.setenv("AI_LOWERLING_RUNS_DIR", str(second))
    _confirmed_run(second, "other/two")
    assert [row["run_id"] for row in query.best_confirmed()["best"]] == ["other/two"]
    parents = {row[0] for row in get_index().conn.execute("SELECT path FROM run_parents")}
    assert parents == {"other"}


def test_best_sees_a_running_run_that_finishes_and_confirms(tmp_path, monkeypatch):
    """@brief 验收 PR4 等价：索引里还是 running 的 run 在 /best 里完成确认，下一次 /best 要能看到。

    旧 ``tick()`` 每帧复查 running 行；``discover_new`` 保留这一步（数量很小）。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 的替换工具。
    """
    runs = _use_temp_runs(tmp_path, monkeypatch)
    directory = runs / "conv2d" / "live"
    directory.mkdir(parents=True)
    (directory / "state.json").write_text(json.dumps({
        "pid": os.getpid(), "phase": "plan", "started_at": 10.0, "updated_at": 20.0,
        "task": "conv2d", "dtype": "fp16",
    }), encoding="utf-8")
    (directory / "run_config.json").write_text(json.dumps({
        "target": {"id": "apple-m4-max", "backend": "metal", "backend_version": "v1"},
    }), encoding="utf-8")
    Index(tmp_path / "observe.sqlite").sync_known()
    assert query.best_confirmed()["best"] == []

    (directory / "candidates" / "000_r1.p0").mkdir(parents=True)
    (directory / "best.metal").write_text(_SOURCE, encoding="utf-8")
    (directory / "candidates" / "000_r1.p0" / "kernel.metal").write_text(_SOURCE, encoding="utf-8")
    _write_confirmation(directory, 1.25)
    assert [row["run_id"] for row in query.best_confirmed()["best"]] == ["conv2d/live"]


def _shape_like_schema_11(index) -> None:
    """@brief 把一个 schema-12 库削回 HEAD schema-11 的结构（删掉 PR4 用不到的新表/触发器）。

    不删 ``runs`` 上的 schema-12 窄列（``control_action``/``topped_up``/``initial_budget_usd``…）：
    PR7 的摄入路径会写这些列，而真实的 schema-11 库在打开时就会因版本号不符整库重建。
    就地删列只会造出一个“版本 12 但缺列”的库，它不对应任何真实迁移路径，也不属于 PR4 的验收面。

    @param index 已打开的 ``Index``（之后继续用同一个连接，避免 ``_create()`` 把结构补回来）。
    """
    for name in (
        "run_parents", "confirmed_best", "lock_spans", "gpu_waits", "gpu_events",
        "failure_members", "kernel_history", "run_rollup", "run_pending_eval", "blob_refs",
        "runs_fts", "cands_fts",
    ):
        index.conn.execute(f"DROP TABLE IF EXISTS {name}")
    for name in (
        "runs_fts_ai", "runs_fts_ad", "runs_fts_au",
        "cands_fts_ai", "cands_fts_ad", "cands_fts_au",
    ):
        index.conn.execute(f"DROP TRIGGER IF EXISTS {name}")
    index.conn.commit()
    # 结构在进程内被改过：让表存在性/父目录发现重新判定，而不是复用旧缓存。
    index._tables.clear()


def test_best_runs_on_a_schema_11_shaped_database(tmp_path, monkeypatch):
    """@brief PR4 独立于 schema 12：删掉新表/新列后，定向删除与“新 run 发现”都仍然成立。

    ``run_parents`` 缺失时用 ``runs.run_id`` 现推父目录（不是靠吞异常）；``confirmed_best`` 缺失时
    只跳过物化维护，``/best`` 仍读 ``runs``。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 的替换工具。
    """
    runs = _use_temp_runs(tmp_path, monkeypatch)
    _confirmed_run(runs, "conv2d/gone")
    _confirmed_run(runs, "conv2d/kept")
    index = Index(tmp_path / "observe.sqlite")
    index.sync_known()
    _shape_like_schema_11(index)
    # 之后 ``query.best_confirmed`` 必须复用这个被削过的连接，而不是 get_index() 新建一个把表补回来。
    import observe.index as index_mod

    index_mod._SINGLE = index
    assert "dir_mtime_ns" not in {row[1] for row in index.conn.execute("PRAGMA table_info(runs)")}

    # 删除回归：没有 confirmed_best 时也把删掉的已确认行从 /best 移除。
    shutil.rmtree(runs / "conv2d" / "gone")
    assert [row["run_id"] for row in query.best_confirmed()["best"]] == ["conv2d/kept"]

    # 发现回归：没有 run_parents 时，从 runs.run_id 现推父目录发现同父目录的新 run。
    _confirmed_run(runs, "conv2d/new")
    assert "conv2d/new" in {row["run_id"] for row in query.best_confirmed()["best"]}


def _two_candidate_confirmed_run(root: Path, name: str, *, winner: str = "b") -> Path:
    """@brief 造一颗两候选的已确认 run，确认绑定到指定候选的内核。

    @param root runs 根目录。
    @param name run id。
    @param winner 胜者候选标签（``a`` 或 ``b``）。
    @return run 目录。
    """
    import hashlib

    directory = root / name
    directory.mkdir(parents=True)
    kernels = {"a": "kernel void ka() {}\n", "b": "kernel void kb() {}\n"}
    directory.joinpath("state.json").write_text(json.dumps({
        "task": "conv2d", "dtype": "fp16", "phase": "done", "started_at": T0, "updated_at": T0 + 10,
    }), encoding="utf-8")
    directory.joinpath("run_config.json").write_text(json.dumps({
        "target": {"id": "apple-m4-max", "backend": "metal", "backend_version": "v1"},
    }), encoding="utf-8")
    for index, label in enumerate(("a", "b")):
        folder = directory / "candidates" / f"{index:03d}_{label}"
        folder.mkdir(parents=True)
        (folder / "kernel.metal").write_text(kernels[label], encoding="utf-8")
        (folder / "report.json").write_text(json.dumps({"status": "ok", "stage": "bench", "speedup": 1.0}),
                                            encoding="utf-8")
    digest = hashlib.sha256(kernels[winner].encode()).hexdigest()
    directory.joinpath("best.metal").write_text(kernels[winner], encoding="utf-8")
    from experiments.common import synthetic_confirmation_documents

    result, report = synthetic_confirmation_documents(
        task="conv2d", dtype="fp16", size="paper", target_id="apple-m4-max",
        speedup=1.25, label=winner, kernel_sha256=digest,
    )
    result.update({"stop_reason": "max_rounds", "started_at": T0, "spent_usd": 0.5})
    directory.joinpath("result.json").write_text(json.dumps(result), encoding="utf-8")
    directory.joinpath("confirm_report.json").write_text(json.dumps(report), encoding="utf-8")
    return directory


def test_best_tracks_the_bound_winner_kernel_in_place(tmp_path, monkeypatch):
    """@brief 定向核对要覆盖胜者内核：只改获胜候选的 kernel.metal、不动根 mtime，/best 必须变。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 的替换工具。
    """
    runs = _use_temp_runs(tmp_path, monkeypatch)
    directory = _two_candidate_confirmed_run(runs, "conv2d/winner")
    Index(tmp_path / "observe.sqlite").sync_known()
    root_mtime = runs.stat().st_mtime_ns
    assert [row["run_id"] for row in query.best_confirmed()["best"]] == ["conv2d/winner"]

    (directory / "candidates" / "001_b" / "kernel.metal").write_text("kernel void changed() {}\n", encoding="utf-8")
    assert runs.stat().st_mtime_ns == root_mtime
    assert query.best_confirmed()["best"] == [], "胜者内核被改写后发布绑定必须失效"


def test_ingest_retries_when_files_change_during_the_read(tmp_path, monkeypatch):
    """@brief 读快照期间文件被改写时，不能把“旧内容 + 新指纹”写进库（否则定向路径永久陈旧）。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 的替换工具。
    """
    runs = _use_temp_runs(tmp_path, monkeypatch)
    directory = _confirmed_run(runs, "conv2d/one")
    state_path = directory / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["started_at"] = 20.0
    state_path.write_text(json.dumps(state), encoding="utf-8")
    index = Index(tmp_path / "observe.sqlite")
    index.sync_known()
    row = index.conn.execute("SELECT started_at FROM runs WHERE run_id = 'conv2d/one'").fetchone()
    assert row["started_at"] == 20.0

    fired = {"n": 0}
    real = Index._read_json

    def racing(self, path):
        doc = real(self, path)
        if path.name == "result.json" and fired["n"] == 0:
            fired["n"] = 1
            newest = json.loads(state_path.read_text(encoding="utf-8"))
            newest["started_at"] = 30.0
            state_path.write_text(json.dumps(newest), encoding="utf-8")
        return doc

    monkeypatch.setattr(Index, "_read_json", racing)
    index._ingest(directory, force=True)
    index.conn.commit()
    row = index.conn.execute("SELECT started_at FROM runs WHERE run_id = 'conv2d/one'").fetchone()
    assert row["started_at"] == 30.0, "读取窗口内被改写时必须整份重读"

    calls: list[str] = []
    monkeypatch.setattr(Index, "_ingest", lambda self, path, *, force=False: calls.append(str(path)))
    query.best_confirmed()
    assert calls == [], "重读后的内容与指纹一致，定向路径不应再摄取"


def test_discovery_survives_a_removed_and_recreated_nested_parent(tmp_path, monkeypatch):
    """@brief 嵌套父目录被删后再重建，新 run 仍要被 /best 发现（祖先目录在候选集里）。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 的替换工具。
    """
    runs = _use_temp_runs(tmp_path, monkeypatch)
    _confirmed_run(runs, "conv2d/exp/old")
    Index(tmp_path / "observe.sqlite").sync_known()
    assert [row["run_id"] for row in query.best_confirmed()["best"]] == ["conv2d/exp/old"]

    shutil.rmtree(runs / "conv2d" / "exp")
    assert query.best_confirmed()["best"] == []

    _confirmed_run(runs, "conv2d/exp/new")
    assert [row["run_id"] for row in query.best_confirmed()["best"]] == ["conv2d/exp/new"]


def test_ingest_aborts_when_the_snapshot_never_settles(tmp_path, monkeypatch):
    """@brief 连续三次都不稳定时不写行：旧内容不能绑定新指纹，下一次仍可重试。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 的替换工具。
    """
    runs = _use_temp_runs(tmp_path, monkeypatch)
    directory = _confirmed_run(runs, "conv2d/one")
    state_path = directory / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["started_at"] = 20.0
    state_path.write_text(json.dumps(state), encoding="utf-8")
    index = Index(tmp_path / "observe.sqlite")
    index.sync_known()
    stored = index.conn.execute("SELECT fp_state FROM runs WHERE run_id = 'conv2d/one'").fetchone()[0]

    def unstable(self, path):
        from observe.index import _fingerprint

        doc = json.loads(path.read_text(encoding="utf-8")) if path.name == "state.json" else None
        if path.name == "state.json":
            doc["started_at"] = doc.get("started_at", 0) + 1
            path.write_text(json.dumps(doc), encoding="utf-8")
        return doc

    monkeypatch.setattr(Index, "_read_json", unstable)
    index._ingest(directory, force=True)
    index.conn.commit()
    row = index.conn.execute("SELECT started_at, fp_state FROM runs WHERE run_id = 'conv2d/one'").fetchone()
    assert row["started_at"] == 20.0, "不稳定时不能把半读到的内容写进库"
    assert row["fp_state"] == stored
    from observe.index import _fingerprint

    assert _fingerprint(state_path) != stored, "库里留下的旧指纹必须与当前文件不同，下一次才会重试"


def test_best_drops_a_duplicate_winner_label(tmp_path, monkeypatch):
    """@brief 新增一颗同名 winner 目录后，定向指纹必须发现“winner 不唯一”并让发布失效。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 的替换工具。
    """
    runs = _use_temp_runs(tmp_path, monkeypatch)
    directory = _two_candidate_confirmed_run(runs, "conv2d/winner")
    Index(tmp_path / "observe.sqlite").sync_known()
    assert [row["run_id"] for row in query.best_confirmed()["best"]] == ["conv2d/winner"]

    duplicate = directory / "candidates" / "002_b"
    duplicate.mkdir()
    (duplicate / "kernel.metal").write_text("kernel void kb2() {}\n", encoding="utf-8")
    (duplicate / "report.json").write_text(json.dumps({"status": "ok", "stage": "bench", "speedup": 1.0}),
                                          encoding="utf-8")
    assert query.best_confirmed()["best"] == [], "winner 不再唯一时必须停止发布"


def test_best_sees_a_run_created_in_a_pre_existing_empty_container(tmp_path, monkeypatch):
    """@brief 预先存在的空容器（runs/conv2d/exp/）要作为 sentinel，之后在里面建 run 能被发现。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 的替换工具。
    """
    runs = _use_temp_runs(tmp_path, monkeypatch)
    (runs / "conv2d" / "exp").mkdir(parents=True)
    Index(tmp_path / "observe.sqlite").sync_known()
    assert query.best_confirmed()["best"] == []

    _confirmed_run(runs, "conv2d/exp/new")
    assert [row["run_id"] for row in query.best_confirmed()["best"]] == ["conv2d/exp/new"]


def test_best_sees_a_pre_created_empty_run_dir_filled_later(tmp_path, monkeypatch):
    """@brief 预建空 run 目录，之后再补 state/result/确认文件：容器 sentinel 的变化要能触发摄取。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 的替换工具。
    """
    runs = _use_temp_runs(tmp_path, monkeypatch)
    (runs / "conv2d" / "pending").mkdir(parents=True)
    Index(tmp_path / "observe.sqlite").sync_known()
    assert query.best_confirmed()["best"] == []

    target = runs / "conv2d" / "pending"
    _confirmed_run(runs, "conv2d/seed")
    for name in ("state.json", "result.json", "run_config.json", "confirm_report.json", "best.metal"):
        (target / name).write_text((runs / "conv2d" / "seed" / name).read_text(encoding="utf-8"), encoding="utf-8")
    (target / "candidates").mkdir()
    (target / "candidates" / "000_r1.p0").mkdir()
    (target / "candidates" / "000_r1.p0" / "kernel.metal").write_text(_SOURCE, encoding="utf-8")
    (target / "candidates" / "000_r1.p0" / "report.json").write_text(
        json.dumps({"status": "ok", "stage": "bench", "speedup": 1.25}), encoding="utf-8")
    found = {row["run_id"] for row in query.best_confirmed()["best"]}
    assert "conv2d/pending" in found


def test_schema_11_cleanup_paths_skip_missing_schema_12_tables(tmp_path, monkeypatch):
    """@brief schema-11 结构下切 root、归档删除、迁移都不碰 schema-12 表，也不半途中断。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 的替换工具。
    """
    import observe.index as index_mod

    runs = _use_temp_runs(tmp_path, monkeypatch)
    _confirmed_run(runs, "conv2d/one")
    _confirmed_run(runs, "conv2d/two")
    index = Index(tmp_path / "observe.sqlite")
    index.sync_known()
    _shape_like_schema_11(index)
    index_mod._SINGLE = index

    second = tmp_path / "runs2"
    second.mkdir()
    monkeypatch.setenv("AI_LOWERLING_RUNS_DIR", str(second))
    _confirmed_run(second, "other/one")
    assert [row["run_id"] for row in query.best_confirmed()["best"]] == ["other/one"], "切 root 不能因缺表失败"

    index.drop_archived(["other/one"], [])
    assert index.conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0] == 0
    index.forget_moved("other/one")


def test_best_watches_every_best_file_not_just_the_first(tmp_path, monkeypatch):
    """@brief 验收 PR4：混入 ``best.aaa`` 后只改权威的 ``best.metal`` 也必须被发现；改名同样要发现。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 的替换工具。
    """
    from starlette.testclient import TestClient

    from observe.app import app

    runs = _use_temp_runs(tmp_path, monkeypatch)
    directory = _confirmed_run(runs, "conv2d/one")
    (directory / "best.aaa").write_text("kernel void other() {}\n", encoding="utf-8")
    Index(tmp_path / "observe.sqlite").sync_known()
    root_mtime = runs.stat().st_mtime_ns

    with TestClient(app) as client:
        assert len(client.get("/api/v2/best").json()["best"]) == 1
        # 只改 best.metal：发布门读的就是它，行必须变成未确认。
        (directory / "best.metal").write_text("kernel void changed() {}\n", encoding="utf-8")
        assert runs.stat().st_mtime_ns == root_mtime
        assert client.get("/api/v2/best").json()["best"] == []

    # 改名：best.metal 不在了，即使 best.aaa 还在也必须从 /best 消失。
    other = _confirmed_run(runs, "conv2d/two")
    (other / "best.aaa").write_text("kernel void other() {}\n", encoding="utf-8")
    Index(tmp_path / "observe.sqlite").sync_known()
    with TestClient(app) as client:
        assert [row["run_id"] for row in client.get("/api/v2/best").json()["best"]] == ["conv2d/two"]
        (other / "best.metal").rename(other / "best.zzz")
        assert [row["run_id"] for row in client.get("/api/v2/best").json()["best"]] == []


def test_the_confirmation_stat_pass_opens_no_file_even_with_a_cold_cache(tmp_path, monkeypatch):
    """@brief 验收 PR4：清空目标身份缓存模拟新进程后，确认核对仍然只 stat，不打开配置或目标档案。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 的替换工具。
    """
    runs = _use_temp_runs(tmp_path, monkeypatch)
    _confirmed_run(runs, "conv2d/one")
    _confirmed_run(runs, "conv2d/two")
    Index(tmp_path / "observe.sqlite").sync_known()
    detail._target_identity_cached.cache_clear()

    with _counting_opens(monkeypatch) as opened:
        best = query.best_confirmed()
    assert [row["run_id"] for row in best["best"]] == ["conv2d/one", "conv2d/two"]
    names = [path for path, _mode in opened]
    assert [path for path in names if path.endswith("run_config.json")] == []
    assert [path for path in names if path.endswith("target.toml")] == []
    assert _reads_under(opened, runs) == []


def test_best_refreshes_state_and_config_changes_and_drops_a_deleted_run(tmp_path, monkeypatch):
    """@brief 验收 PR4：state/run_config 原地改写要反映到 /best 的整行与排序；已确认目录删除后该行必须消失。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 的替换工具。
    """
    from starlette.testclient import TestClient

    from observe.app import app

    runs = _use_temp_runs(tmp_path, monkeypatch)
    _confirmed_run(runs, "conv2d/old")
    _confirmed_run(runs, "conv2d/new")
    Index(tmp_path / "observe.sqlite").sync_known()
    root_mtime = runs.stat().st_mtime_ns

    with TestClient(app) as client:
        # 加速比相同时按 started_at 倒序；把 old 的开始时间改到最晚。
        state_path = runs / "conv2d" / "old" / "state.json"
        state = json.loads(state_path.read_text(encoding="utf-8"))
        state["started_at"] = 99.0
        state_path.write_text(json.dumps(state), encoding="utf-8")
        assert runs.stat().st_mtime_ns == root_mtime
        order = [row["run_id"] for row in client.get("/api/v2/best").json()["best"]]
        assert order == ["conv2d/old", "conv2d/new"]

        # run_config 的 target 变了，行的 target_id 与按 target 过滤都要跟着变。
        config_path = runs / "conv2d" / "old" / "run_config.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        config["target"]["id"] = "apple-m5-test"
        config_path.write_text(json.dumps(config), encoding="utf-8")
        rows = client.get("/api/v2/best").json()["best"]
        assert {row["run_id"]: row["target_id"] for row in rows}["conv2d/old"] == "apple-m5-test"
        assert client.get("/api/v2/best?target=apple-m4-max").json()["best"][0]["run_id"] == "conv2d/new"

        shutil.rmtree(runs / "conv2d" / "old")
        remaining = client.get("/api/v2/best").json()["best"]
        assert [row["run_id"] for row in remaining] == ["conv2d/new"]

    row = get_index().conn.execute(
        "SELECT status, publication, published_speedup FROM runs WHERE run_id = 'conv2d/old'"
    ).fetchone()
    assert dict(row) == {"status": "missing", "publication": None, "published_speedup": None}


def test_the_panel_routes_serve_the_summary_and_each_panel(tmp_path, monkeypatch):
    """@brief 端点级验收：摘要与九个面板各自可达，非法候选/raw key 返回 404。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 的替换工具。
    """
    from starlette.testclient import TestClient

    from observe.app import app

    _open_rich(tmp_path, monkeypatch)
    with TestClient(app) as client:
        summary = client.get("/api/v2/runs/conv2d/rich")
        assert summary.status_code == 200
        assert summary.json()["stop_reason"] == "max_rounds"
        detail = client.get("/api/v2/runs/conv2d/rich/detail").json()
        assert set(summary.json()) <= set(detail)

        listed = client.get("/api/v2/runs/conv2d/rich/candidates").json()
        assert [row["label"] for row in listed["candidates"]] == ["r1.p0", "init"] and listed["total"] == 2
        first_page = client.get("/api/v2/runs/conv2d/rich/candidates?status=ok&stage=bench&limit=1").json()
        assert len(first_page["candidates"]) == 1 and first_page["next"]
        second_page = client.get(
            "/api/v2/runs/conv2d/rich/candidates?limit=1&cursor=" + first_page["next"]
        ).json()
        assert [row["label"] for row in second_page["candidates"]] == ["init"]
        assert client.get("/api/v2/runs/conv2d/rich/candidates?status=failed").json()["total"] == 0

        candidate = client.get("/api/v2/runs/conv2d/rich/candidate/r1.p0").json()
        assert candidate["report"]["speedup"] == pytest.approx(1.25)
        same_candidate = client.get(
            "/api/v2/runs/conv2d/rich/candidate/r1.p0?stamp=" + candidate["stamp"]
        ).json()
        assert same_candidate["same"] is True
        assert client.get("/api/v2/runs/conv2d/rich/candidate/nope").status_code == 404

        feedback = client.get("/api/v2/runs/conv2d/rich/feedback").json()
        assert len(feedback["feedback"]["records"]) == 2
        assert "raw" not in feedback["feedback"]["records"][0]
        record = client.get("/api/v2/runs/conv2d/rich/feedback/1").json()
        assert record["record"]["raw"] == _SECRET
        same_record = client.get("/api/v2/runs/conv2d/rich/feedback/1?stamp=" + record["stamp"]).json()
        assert same_record["same"] is True
        assert client.get("/api/v2/runs/conv2d/rich/feedback/99999").status_code == 404

        assert client.get("/api/v2/runs/conv2d/rich/timeline").json()["timeline"]
        assert client.get("/api/v2/runs/conv2d/rich/budget-events").json()["budget_events"]
        assert client.get("/api/v2/runs/conv2d/rich/llm").json()["llm"]["label"] == "r2.plan"
        assert client.get("/api/v2/runs/conv2d/rich/grid").json()["feature_grid"]
        assert client.get("/api/v2/runs/conv2d/rich/plan").json()["plan_outcomes"] is not None
        raw = client.get("/api/v2/runs/conv2d/rich/raw?key=state").json()
        assert raw["doc"]["task"] == "conv2d" and raw["path"].endswith("state.json")
        assert client.get("/api/v2/runs/conv2d/rich/raw?key=nope").status_code == 404
        assert client.get("/api/v2/runs/nope").status_code == 404
        assert client.get("/api/v2/runs/nope/candidates").status_code == 404


def test_explicit_refresh_still_marks_deleted_runs_missing(tmp_path, monkeypatch):
    """@brief 显式 refresh 保留全量核对语义，删除运行后会把索引行标为 missing。"""
    runs = _use_temp_runs(tmp_path, monkeypatch)
    directory = _write_run(runs, "gone")
    query.refresh()
    assert query.history()[0]["id"] == "gone"

    shutil.rmtree(directory)
    query.refresh()

    row = get_index().conn.execute("SELECT status, publication, published_speedup FROM runs WHERE run_id = 'gone'").fetchone()
    assert dict(row) == {"status": "missing", "publication": None, "published_speedup": None}
    assert query.history() == []


def test_an_in_place_eval_progress_rewrite_invalidates_the_summary_stamp(tmp_path, monkeypatch):
    """@brief PR3 回归：``eval_progress.json`` 是摘要字段，必须在 detail stamp 里。

    只改这份文件（不改 state/result，也不改目录 mtime）后 refresh，旧 stamp 不得再命中 ``same``；
    新摘要必须带新的阶段与消息。旧实现漏了这条指纹，stale stamp 会一直返回旧内容。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 的替换工具。
    """
    runs = _use_temp_runs(tmp_path, monkeypatch)
    directory = _write_run(runs, "conv2d/live")
    (directory / "eval_progress.json").write_text(json.dumps({
        "stage": "bench", "entered_at": 20.0, "updated_at": 21.0, "finished": False,
        "task": "conv2d", "messages": [1],
    }), encoding="utf-8")
    index = Index(tmp_path / "observe.sqlite")
    index.sync_known()
    first = query.run_summary("conv2d/live")
    assert first is not None and first["eval_progress"]["stage"] == "bench"
    stamp = first["detail_stamp"]
    assert stamp

    (directory / "eval_progress.json").write_text(json.dumps({
        "stage": "measure", "entered_at": 22.0, "updated_at": 23.0, "finished": False,
        "task": "conv2d", "messages": [41, 42],
    }), encoding="utf-8")
    index.sync_known()
    again = query.run_summary("conv2d/live", stamp=stamp)
    assert again is not None and not again.get("same"), "eval_progress 改写后旧 stamp 必须失效"
    fresh = query.run_summary("conv2d/live")
    assert fresh["eval_progress"]["stage"] == "measure" and fresh["eval_progress"]["messages"] == [41, 42]
    assert fresh["detail_stamp"] != stamp


def _many_candidates(directory: Path, total: int = 250) -> None:
    """@brief 给一份富运行补足候选目录，每颗都有 ``report.json`` 与 ``kernel.metal``。

    旧摘要戳把每颗候选的 ``(report, kernel)`` 指纹拼进 ``fp_extra``：候选越多戳越长（实测 250 颗时
    17,730 字符），而且改写一颗非 winner 的报告会让首屏摘要换戳。这里造出那个规模。

    @param directory 运行目录。
    @param total 补足到的候选数。
    """
    for index in range(2, total):
        label = f"r{index}.p0"
        folder = directory / "candidates" / f"{index:03d}_{label}"
        folder.mkdir(parents=True)
        (folder / "kernel.metal").write_text(_SOURCE, encoding="utf-8")
        (folder / "report.json").write_text(json.dumps({
            "status": "ok", "stage": "bench", "speedup": 1.0, "candidate_us": 770.0,
            "message": f"report {label}",
        }), encoding="utf-8")


def test_a_250_candidate_run_keeps_the_summary_bounded_and_its_stamp_off_the_candidates(tmp_path, monkeypatch):
    """@brief 评审回归 ①：摘要戳与候选指纹分开——有界，且非 winner 改写不再串扰摘要。

    250 颗候选时摘要仍必须 ≤8KB、戳有界（旧实现 18,845B 摘要，其中戳 17,730 字符）；就地改写
    一颗**非 winner** 的 ``report.json`` 并跑 ``sync_known`` 之后，旧摘要 stamp 必须继续命中
    ``same``，只有候选面板换戳。只把整串 ``fp_extra`` 做 hash 能满足长度，但满足不了这条。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 的替换工具。
    """
    runs = _use_temp_runs(tmp_path, monkeypatch)
    directory = _rich_run(runs, "conv2d/many")
    _many_candidates(directory, 250)
    Index(tmp_path / "observe.sqlite").sync_known()

    summary = query.run_summary("conv2d/many")
    assert summary is not None
    size = len(json.dumps(summary, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
    assert size <= 8192, f"250 颗候选的摘要 {size}B 超过 8KB"
    stamp = summary["detail_stamp"]
    assert 0 < len(stamp) <= 128, f"摘要戳必须有界，实测 {len(stamp)} 字符"
    panel = query.candidates_panel("conv2d/many")
    assert panel is not None and panel["total"] == 250

    # winner 是 candidates/001_r1.p0；改写 000_init 这颗非 winner。
    report = directory / "candidates" / "000_init" / "report.json"
    report.write_text(json.dumps({
        "schema_version": 4, "status": "ok", "stage": "bench", "speedup": 9.5, "message": "非 winner 就地改写",
    }), encoding="utf-8")
    os.utime(report, ns=(report.stat().st_atime_ns, report.stat().st_mtime_ns + 1_000_000))
    Index(tmp_path / "observe.sqlite").sync_known()

    again = query.run_summary("conv2d/many", stamp=stamp)
    assert again is not None and again.get("same") is True, "改写非 winner 候选不得让摘要换戳"
    changed = query.candidates_panel("conv2d/many", stamp=panel["stamp"])
    assert changed is not None and changed.get("same") is False, "候选面板必须换戳"
    located = query.candidates_panel("conv2d/many", label="init")
    assert located["page_index"] is not None
    by_label = {row["label"]: row for row in located["candidates"]}
    assert by_label["init"]["speedup"] == pytest.approx(9.5)


def test_the_full_detail_stamp_covers_candidates_and_feedback(tmp_path, monkeypatch):
    """@brief 评审回归：完整详情的戳要覆盖它真正读的面板，不能复用摘要戳。

    旧实现让 ``run_detail`` 与 ``run_summary`` 共用摘要戳，于是就地改写一颗候选报告、或改写
    ``feedback.jsonl`` 之后，``run_detail(old_stamp)`` 仍然返回 ``same``，而不用戳重新读已是新值。
    这里两种改写都必须让完整详情的旧戳失效，同时**不得**串扰摘要戳（摘要里没有这两块）。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 的替换工具。
    """
    runs, directory = _open_rich(tmp_path, monkeypatch)
    summary = query.run_summary("conv2d/rich")
    full = query.run_detail("conv2d/rich")
    summary_stamp, full_stamp = summary["detail_stamp"], full["detail_stamp"]

    report = directory / "candidates" / "001_r1.p0" / "report.json"
    body = json.loads(report.read_text(encoding="utf-8"))
    body["speedup"] = 2.5
    report.write_text(json.dumps(body), encoding="utf-8")
    os.utime(report, ns=(report.stat().st_atime_ns, report.stat().st_mtime_ns + 1_000_000))
    Index(tmp_path / "observe.sqlite").sync_known()

    again = query.run_detail("conv2d/rich", stamp=full_stamp)
    assert again is not None and not again.get("same"), "候选改写后完整详情的旧戳必须失效"
    by_label = {row["label"]: row for row in again["candidates"]}
    assert by_label["r1.p0"]["speedup"] == pytest.approx(2.5)
    assert query.run_summary("conv2d/rich", stamp=summary_stamp).get("same") is True, "摘要不该被候选串扰"
    full_stamp = again["detail_stamp"]

    feedback = directory / "feedback.jsonl"
    feedback.write_text(feedback.read_text(encoding="utf-8") + json.dumps(
        _feedback_record(3, "init", raw="新增的一条反馈")
    ) + "\n", encoding="utf-8")
    Index(tmp_path / "observe.sqlite").sync_known()
    after = query.run_detail("conv2d/rich", stamp=full_stamp)
    assert after is not None and not after.get("same"), "反馈改写后完整详情的旧戳必须失效"
    assert len(after["feedback"]["records"]) == 3
    assert query.run_summary("conv2d/rich", stamp=summary_stamp).get("same") is True, "摘要不该被反馈串扰"


def test_the_summary_stamp_covers_the_derived_kernel_writer_on_legacy_runs(tmp_path, monkeypatch):
    """@brief 评审回归：没有 run_config 的旧 run，``kernel_writer`` 由 transcript/agent_work 现推。

    摘要里带 ``kernel_writer``，所以它的真实依赖也必须在摘要戳里。旧实现只把 config/state 等放进
    指纹，于是空 transcript 的旧 run 是 ``chat``，追加一条 ``event=agent`` 后不用戳读到 ``pi``，
    旧戳却仍然 ``same``。现在它进摄入窄投影 ``runs.kernel_writer``，兜底来源（transcript/
    agent_work）由 ``_refresh_kernel_writer`` 增量复核。这里覆盖 chat→pi→chat 两个方向。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 的替换工具。
    """
    runs = _use_temp_runs(tmp_path, monkeypatch)
    directory = _write_run(runs, "legacy/old")
    (directory / "transcript.jsonl").write_text("", encoding="utf-8")
    index = Index(tmp_path / "observe.sqlite")
    index.sync_known()
    first = query.run_summary("legacy/old")
    assert first is not None and first["kernel_writer"] == "chat"
    stamp = first["detail_stamp"]
    assert index.conn.execute("SELECT kernel_writer FROM runs WHERE run_id = 'legacy/old'").fetchone()[0] == "chat"

    with (directory / "transcript.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"event": "agent", "label": "r0", "outcome": "evaluated"}) + "\n")
    index.sync_known()

    assert index.conn.execute("SELECT kernel_writer FROM runs WHERE run_id = 'legacy/old'").fetchone()[0] == "pi"
    again = query.run_summary("legacy/old", stamp=stamp)
    assert again is not None and not again.get("same"), "kernel_writer 变了旧摘要戳必须失效"
    fresh = query.run_summary("legacy/old")
    assert fresh["kernel_writer"] == "pi" and fresh["detail_stamp"] != stamp

    # 反向：transcript 被就地改短（agent 事件消失）时，窄投影要降回 chat，戳也要跟着动。
    pi_stamp = fresh["detail_stamp"]
    (directory / "transcript.jsonl").write_text("", encoding="utf-8")
    index.sync_known()
    assert index.conn.execute("SELECT kernel_writer FROM runs WHERE run_id = 'legacy/old'").fetchone()[0] == "chat"
    back = query.run_summary("legacy/old", stamp=pi_stamp)
    assert back is not None and not back.get("same"), "kernel_writer 降回 chat 时旧戳也必须失效"
    assert query.run_summary("legacy/old")["kernel_writer"] == "chat"


def test_the_recorded_kernel_agent_wins_over_the_fallback_sources(tmp_path, monkeypatch):
    """@brief 评审回归：配置里记下的 ``kernel_agent`` 优先于 agent_work/transcript，空串也算已记录。

    ``detail.kernel_writer`` 的语义是：任何字符串（包括 ``''``）都表示“已记录”，只有 ``pi`` 是 pi、
    其余都是 chat；只有键缺失/不是字符串时才回退。窄投影必须逐字同义，否则 ``chat`` 配置 +
    非空 ``agent_work`` 会被兜底来源覆盖成 pi，摘要和详情互相矛盾。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 的替换工具。
    """
    runs = _use_temp_runs(tmp_path, monkeypatch)
    for name, recorded in (("cfg/chat", "chat"), ("cfg/empty", ""), ("cfg/pi", "pi")):
        directory = _write_run(runs, name)
        (directory / "run_config.json").write_text(
            json.dumps({"search": {"kernel_agent": recorded}}), encoding="utf-8")
        work = directory / "agent_work" / "s1"
        work.mkdir(parents=True)
        (work / "notes.md").write_text("x", encoding="utf-8")
    index = Index(tmp_path / "observe.sqlite")
    index.sync_known()
    # 第二遍走短路刷新路径：旧实现只在列为空时才读配置，这里会退回 agent_work。
    index.sync_known()

    for name, expected in (("cfg/chat", "chat"), ("cfg/empty", "chat"), ("cfg/pi", "pi")):
        directory = runs / name
        stored = index.conn.execute(
            "SELECT kernel_writer FROM runs WHERE run_id = ?", (name,)
        ).fetchone()[0]
        assert stored == expected == detail.kernel_writer(directory), f"{name} 的窄投影与权威语义不一致"
        summary = query.run_summary(name)
        assert summary["kernel_writer"] == expected
        stamp = summary["detail_stamp"]
        assert query.run_summary(name, stamp=stamp).get("same") is True


def test_the_kernel_writer_projection_tracks_rewrites_and_agent_work_removal(tmp_path, monkeypatch):
    """@brief 评审回归：初始就是 pi 的 run 也要维护完整来源指纹，不能在等长改写/删除后停在 pi。

    旧实现只在 ``current != 'pi'`` 时才看来源，且只记追加偏移：初始 transcript 含 ``event=agent``
    的旧 run 从不写偏移，随后把 transcript 清空、把内容改成等长的 chat 行、或删掉唯一的
    ``agent_work``，索引都会一直是 pi，而 ``detail.kernel_writer`` 已经是 chat。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 的替换工具。
    """
    runs = _use_temp_runs(tmp_path, monkeypatch)
    agent_line = json.dumps({"event": "agent", "label": "r0", "outcome": "evaluated"}) + "\n"
    # 与 agent_line 等长、但不含 agent 事件的一行。
    chat_line = json.dumps({"event": "chat!", "label": "r0", "outcome": "evaluated"}) + "\n"
    assert len(chat_line) == len(agent_line)

    for name in ("legacy/init", "legacy/equal", "legacy/work"):
        _write_run(runs, name)
    (runs / "legacy" / "init" / "transcript.jsonl").write_text(agent_line, encoding="utf-8")
    (runs / "legacy" / "equal" / "transcript.jsonl").write_text(agent_line, encoding="utf-8")
    work = runs / "legacy" / "work" / "agent_work" / "s1"
    work.mkdir(parents=True)
    (work / "notes.md").write_text("x", encoding="utf-8")

    index = Index(tmp_path / "observe.sqlite")
    index.sync_known()
    index.sync_known()  # 短路刷新：旧实现不会为初始 pi 写偏移
    for name in ("legacy/init", "legacy/equal", "legacy/work"):
        assert index.conn.execute(
            "SELECT kernel_writer FROM runs WHERE run_id = ?", (name,)
        ).fetchone()[0] == "pi"
        assert index.conn.execute(
            "SELECT COUNT(*) FROM meta WHERE key = ?", (f"kw_src:{name}",)
        ).fetchone()[0] == 1, "来源指纹必须落盘，否则后续改写无从判断"

    cases = {
        # 清空 transcript（变短）
        "legacy/init": lambda directory: (directory / "transcript.jsonl").write_text("", encoding="utf-8"),
        # 等长就地改写，agent 事件消失
        "legacy/equal": lambda directory: (directory / "transcript.jsonl").write_text(chat_line, encoding="utf-8"),
        # 删掉唯一的 agent_work
        "legacy/work": lambda directory: shutil.rmtree(directory / "agent_work"),
    }
    for name, mutate in cases.items():
        directory = runs / name
        stamp = query.run_summary(name)["detail_stamp"]
        assert query.run_summary(name)["kernel_writer"] == "pi"
        mutate(directory)
        index.sync_known()
        assert detail.kernel_writer(directory) == "chat"
        assert index.conn.execute(
            "SELECT kernel_writer FROM runs WHERE run_id = ?", (name,)
        ).fetchone()[0] == "chat", f"{name} 的窄投影没有跟上来源改写"
        again = query.run_summary(name, stamp=stamp)
        assert again is not None and not again.get("same"), f"{name} 旧摘要戳必须失效"
        fresh = query.run_summary(name)
        assert fresh["kernel_writer"] == "chat" and fresh["detail_stamp"] != stamp


def test_the_kernel_writer_projection_matches_detail_on_an_unterminated_last_line(tmp_path, monkeypatch):
    """@brief 评审回归：没有换行结尾的 agent 记录也要算进 kernel_writer。

    ``detail.kernel_writer`` 逐行检查 transcript，最后一条没有换行的记录同样会被看到；旧扫描只取
    “最后一个换行之前”，于是这种 run 的窄投影是 chat、权威推导是 pi，摘要与详情互相矛盾。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 的替换工具。
    """
    runs = _use_temp_runs(tmp_path, monkeypatch)
    agent_line = json.dumps({"event": "agent", "label": "r0", "outcome": "evaluated"})
    chat_line = json.dumps({"event": "chat", "label": "r0", "reply": "hello"})
    for name, text in (("tail/agent", agent_line), ("tail/chat", chat_line)):
        directory = _write_run(runs, name)
        # 故意不写结尾换行：它就是“未终止的尾行”。
        (directory / "transcript.jsonl").write_text(text, encoding="utf-8")
    index = Index(tmp_path / "observe.sqlite")
    index.sync_known()
    index.sync_known()

    for name, expected in (("tail/agent", "pi"), ("tail/chat", "chat")):
        directory = runs / name
        assert detail.kernel_writer(directory) == expected
        stored = index.conn.execute(
            "SELECT kernel_writer FROM runs WHERE run_id = ?", (name,)
        ).fetchone()[0]
        assert stored == expected, f"{name} 的窄投影与 detail.kernel_writer 不一致"
        assert query.run_summary(name)["kernel_writer"] == expected


def test_the_kernel_writer_projection_rescans_a_replaced_ended_transcript(tmp_path, monkeypatch):
    """@brief 评审回归：已结束 run 的 transcript 被换成**更长**的纯 chat 内容也要从头复核。

    旧实现只在“等长或变短”时丢弃扫描状态，于是先摄取 agent 记录、再换成更长的 chat 转录后，
    索引一直是 pi、权威推导已是 chat，旧摘要戳还命中 same。现在已结束 run 的任意
    transcript 指纹变化都从头分块复核。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 的替换工具。
    """
    runs = _use_temp_runs(tmp_path, monkeypatch)
    directory = _write_run(runs, "legacy/grow")
    transcript = directory / "transcript.jsonl"
    transcript.write_text(json.dumps({"event": "agent", "label": "r0", "outcome": "evaluated"}) + "\n",
                          encoding="utf-8")
    index = Index(tmp_path / "observe.sqlite")
    index.sync_known()
    index.sync_known()
    first = query.run_summary("legacy/grow")
    assert first["kernel_writer"] == "pi"
    stamp = first["detail_stamp"]

    # 更长的纯 chat 转录：大小变大，但前缀完全换了。
    transcript.write_text(json.dumps({"event": "chat", "label": "r1", "reply": "y" * 500}) + "\n",
                          encoding="utf-8")
    index.sync_known()
    assert detail.kernel_writer(directory) == "chat"
    assert index.conn.execute(
        "SELECT kernel_writer FROM runs WHERE run_id = 'legacy/grow'"
    ).fetchone()[0] == "chat"
    again = query.run_summary("legacy/grow", stamp=stamp)
    assert again is not None and not again.get("same"), "变长替换后旧摘要戳也必须失效"
    fresh = query.run_summary("legacy/grow")
    assert fresh["kernel_writer"] == "chat" and fresh["detail_stamp"] != stamp


def test_the_kernel_writer_meta_is_cleaned_on_delete_move_and_rebuild(tmp_path, monkeypatch):
    """@brief ``kw_src:<run>`` 是 per-run 派生状态：删除、改名与同 ID 重建都不能留着旧记录。

    留着旧记录本身的后果是“同 ID 重建后第一次复核可能沿用旧来源状态”，所以 ``_clear_sessions_projection``
    统一清掉它；这里覆盖 ``drop_archived``、``forget_moved`` 与同 ID 重建三条路径。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 的替换工具。
    """
    runs = _use_temp_runs(tmp_path, monkeypatch)
    index = Index(tmp_path / "observe.sqlite")
    agent_line = json.dumps({"event": "agent", "label": "r0", "outcome": "evaluated"}) + "\n"
    chat_line = json.dumps({"event": "chat", "label": "r0", "reply": "hello"}) + "\n"

    def meta_rows(run_id: str) -> int:
        return index.conn.execute(
            "SELECT COUNT(*) FROM meta WHERE key = ?", (f"kw_src:{run_id}",)
        ).fetchone()[0]

    # drop_archived
    directory = _write_run(runs, "gone/x")
    (directory / "transcript.jsonl").write_text(agent_line, encoding="utf-8")
    index.sync_known()
    # 冷建第一行不读 transcript；第二次对账才补 kw_src（§11.4 的懒投影）。
    index.sync_known()
    assert meta_rows("gone/x") == 1
    index.drop_archived(["gone/x"], [])
    assert meta_rows("gone/x") == 0, "drop_archived 必须清掉 kw_src"

    # 改名（move）：旧 id 的记录不能留；新目录摄取后有自己的记录。
    old = _write_run(runs, "moved/old")
    (old / "transcript.jsonl").write_text(agent_line, encoding="utf-8")
    index.sync_known()
    index.sync_known()
    assert meta_rows("moved/old") == 1
    new = runs / "moved" / "new"
    old.rename(new)
    index.forget_moved("moved/old")
    assert meta_rows("moved/old") == 0, "forget_moved 必须清掉 kw_src"
    index.sync_known()
    index.sync_known()
    assert meta_rows("moved/new") == 1

    # 同 ID 重建：先 agent 记录（pi），删掉后在同一路径写纯 chat，不能沿用旧的 pi。
    rebuilt = _write_run(runs, "rebuilt/x")
    (rebuilt / "transcript.jsonl").write_text(agent_line, encoding="utf-8")
    index.sync_known()
    index.sync_known()
    assert index.conn.execute(
        "SELECT kernel_writer FROM runs WHERE run_id = 'rebuilt/x'"
    ).fetchone()[0] == "pi"
    index.drop_archived(["rebuilt/x"], [])
    shutil.rmtree(rebuilt)
    rebuilt = _write_run(runs, "rebuilt/x")
    (rebuilt / "transcript.jsonl").write_text(chat_line, encoding="utf-8")
    index.sync_known()
    index.sync_known()
    assert index.conn.execute(
        "SELECT kernel_writer FROM runs WHERE run_id = 'rebuilt/x'"
    ).fetchone()[0] == "chat"


def test_every_cli_filter_reaches_the_same_where_clause(tmp_path, monkeypatch, capsys):
    """@brief 验收（plan §6.3）：``runs --dtype/--model/--target``、``failures --run/--label/--sig``、
    ``best --target`` 的 CLI → kwargs → WHERE 三层等价。

    每层都问同一个问题：带过滤时只剩目标行，不带过滤时两行都在。只断言“接线存在”不够——这里对
    每个过滤参数都跑命令行的 ``--jsonl`` 输出、``taic.retrieval`` 与 ``observe.query`` 三层，
    三层的 run 集合必须一致，而且必须真的把另一行滤掉。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 的替换工具。
    @param capsys pytest 的输出捕获器。
    """
    from taic import cli, retrieval

    runs = _use_temp_runs(tmp_path, monkeypatch)
    first = _rich_run(runs, "conv2d/first")
    second = _rich_run(runs, "conv2d/second")
    # 第二行换 target/dtype/model，用来证明每层 WHERE 真的在过滤。
    config = json.loads((second / "run_config.json").read_text())
    config["target"] = {"id": "other-gpu", "backend": "metal", "backend_version": "v1"}
    config["llm"] = {"model": "kimi-k3"}
    (second / "run_config.json").write_text(json.dumps(config), encoding="utf-8")
    for name in ("state.json", "result.json"):
        doc = json.loads((second / name).read_text())
        doc["dtype"] = "bf16"
        (second / name).write_text(json.dumps(doc), encoding="utf-8")
    # 两行各放一颗可复现的失败候选（不同 message → 不同 sig）。
    for directory, label, message in (
        (first, "r1.abi", "kernel does not match the required ABI"),
        (second, "r2.abi", "Metal compiler rejected the kernel"),
    ):
        folder = directory / "candidates" / f"900_{label}"
        folder.mkdir(parents=True)
        (folder / "kernel.metal").write_text(_SOURCE, encoding="utf-8")
        (folder / "report.json").write_text(json.dumps({
            "schema_version": 4, "status": "abi_error", "stage": "abi", "message": message,
        }), encoding="utf-8")
    Index(tmp_path / "observe.sqlite").sync_known()

    def cli_rows(*argv):
        assert cli.main(["query", *argv, "--jsonl"]) == 0
        out = capsys.readouterr().out
        return [json.loads(line) for line in out.splitlines() if line.strip()]

    run_filters = {
        "dtype": ("--dtype", "bf16"),
        "model": ("--model", "kimi-k3"),
        "target": ("--target", "other-gpu"),
    }
    for name, (flag, value) in run_filters.items():
        kwargs = {name: value}
        assert [row["run_id"] for row in cli_rows("runs", flag, value)] == ["conv2d/second"]
        assert [row["run_id"] for row in retrieval.runs(**kwargs, limit=20)["runs"]] == ["conv2d/second"]
        assert [row["run_id"] for row in query.runs_page(**kwargs, limit=20)["runs"]] == ["conv2d/second"]
        # 不带过滤时两行都在：证明上面不是“本来就只剩一行”。
        assert len(query.runs_page(limit=20)["runs"]) == 2
    assert len(cli_rows("runs")) == 2

    index = get_index()
    sigs = {
        row["run_id"]: row["failure_sig"]
        for row in index.conn.execute("SELECT run_id, failure_sig FROM candidates WHERE failure_sig IS NOT NULL")
    }
    assert set(sigs) == {"conv2d/first", "conv2d/second"}
    failure_filters = {
        "run": ("--run", "conv2d/first", "conv2d/first"),
        "label": ("--label", "r2.abi", "conv2d/second"),
        "sig": ("--sig", sigs["conv2d/second"], "conv2d/second"),
    }
    for name, (flag, value, expected) in failure_filters.items():
        cli_failures = cli_rows("failures", flag, value)
        assert [row["run_id"] for row in cli_failures] == [expected], f"CLI {name} 过滤不对"
        assert [row["run_id"] for row in retrieval.failures(**{name: value}, limit=20)["failures"]] == [expected]
        assert [row["run_id"] for row in query.failures(**{name: value}, limit=20)["failures"]] == [expected]
    assert len(query.failures(limit=20)["failures"]) == 2

    assert [row["run_id"] for row in cli_rows("best", "--target", "other-gpu")] == ["conv2d/second"]
    assert [row["run_id"] for row in retrieval.best(target="other-gpu", limit=20)["best"]] == ["conv2d/second"]
    assert [row["run_id"] for row in query.best_confirmed(target="other-gpu", limit=20)["best"]] == ["conv2d/second"]
    assert {row["run_id"] for row in query.best_confirmed(limit=20)["best"]} == {"conv2d/first", "conv2d/second"}


def test_the_full_detail_stamp_covers_the_calibration_source(tmp_path, monkeypatch):
    """@brief 评审回归：完整详情的 feedback.calibration 也要进戳。

    ``query.run_detail`` 调 ``feedback.entries()``，默认附带 ``_calibration()`` 读到的本机标定；
    旧完整戳只含 feedback.jsonl/review 等，改写标定文件后 ``run_detail(old_stamp)`` 仍回 ``same``。
    这里用 ``TCENV_CALIBRATION`` 指向临时文件，验证戳盯的是 entries 实际读取的那份。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 的替换工具。
    """
    runs = _use_temp_runs(tmp_path, monkeypatch)
    directory = _write_run(runs, "conv2d/calib")
    (directory / "feedback.jsonl").write_text(
        json.dumps(_feedback_record(1, "init"), ensure_ascii=False) + "\n", encoding="utf-8")
    calibration = tmp_path / "calibration.json"
    calibration.write_text(json.dumps({
        "peak_tflops": {"fp16": 1, "bf16": 1, "fp32": 1}, "read_gbps": 1, "copy_gbps": 1,
    }), encoding="utf-8")
    monkeypatch.setenv("TCENV_CALIBRATION", str(calibration))
    Index(tmp_path / "observe.sqlite").sync_known()

    first = query.run_detail("conv2d/calib")
    assert first["feedback"]["calibration"]["fp16"] == 1
    stamp = first["detail_stamp"]
    summary_stamp = query.run_summary("conv2d/calib")["detail_stamp"]

    calibration.write_text(json.dumps({
        "peak_tflops": {"fp16": 999, "bf16": 1, "fp32": 1}, "read_gbps": 1, "copy_gbps": 1,
    }), encoding="utf-8")
    again = query.run_detail("conv2d/calib", stamp=stamp)
    assert again is not None and not again.get("same"), "标定改写后完整详情的旧戳必须失效"
    assert again["feedback"]["calibration"]["fp16"] == 999
    assert again["detail_stamp"] != stamp
    # 摘要不含 calibration，不该被它串扰。
    assert query.run_summary("conv2d/calib", stamp=summary_stamp).get("same") is True


def test_a_tick_only_eval_progress_rewrite_still_invalidates_the_summary_stamp(tmp_path, monkeypatch):
    """@brief 评审回归 ②：生产 ``tick()`` 路径（不调 ``sync_known``/``resolve``）也要追上 eval_progress。

    已结束 run 的周期对账键必须含 ``eval_progress.json``：只调 ``index.tick()`` 就改写它，
    旧 stamp 必须不再命中 ``same``，新摘要要带新阶段。上面那条 ``sync_known`` 版本掩盖了这条路径。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 的替换工具。
    """
    runs = _use_temp_runs(tmp_path, monkeypatch)
    # 让下一次 tick() 走全量对账分支（``_use_temp_runs`` 把它推到 3600 秒之后）。
    monkeypatch.setenv("OBSERVE_FULL_TICK_INTERVAL", "0")
    directory = _write_run(runs, "conv2d/ticked")
    (directory / "eval_progress.json").write_text(json.dumps({
        "stage": "bench", "entered_at": 20.0, "updated_at": 21.0, "finished": False,
        "task": "conv2d", "messages": [1],
    }), encoding="utf-8")
    index = Index(tmp_path / "observe.sqlite")
    index.sync_known()
    first = query.run_summary("conv2d/ticked")
    assert first is not None and first["eval_progress"]["stage"] == "bench"
    stamp = first["detail_stamp"]
    assert stamp

    (directory / "eval_progress.json").write_text(json.dumps({
        "stage": "measure", "entered_at": 22.0, "updated_at": 23.0, "finished": False,
        "task": "conv2d", "messages": [41, 42],
    }), encoding="utf-8")
    _forbid_sync_known(monkeypatch)

    def forbidden_resolve(self, run_id):
        raise AssertionError("request reads must not call Index.resolve")

    monkeypatch.setattr(Index, "resolve", forbidden_resolve)
    index.tick()

    again = query.run_summary("conv2d/ticked", stamp=stamp)
    assert again is not None and not again.get("same"), "只 tick 之后旧 stamp 也必须失效"
    fresh = query.run_summary("conv2d/ticked")
    assert fresh["eval_progress"]["stage"] == "measure" and fresh["eval_progress"]["messages"] == [41, 42]
    assert fresh["detail_stamp"] != stamp
