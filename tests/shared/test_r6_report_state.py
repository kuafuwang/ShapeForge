"""R6: only a schema-4 candidate report is current evidence (plan §8).

CPU only.  The index, the detail panel and the candidate queries must classify a report instead of
guessing: ``current`` for schema 4, ``old_format`` for anything older, ``unreadable`` for a file that
exists but does not parse, and ``missing`` for a candidate directory without one.  An old-format
report keeps only the fields it literally carries; the build/evaluation identity columns stay empty
and are never inferred from ``status``, the kernel or ``result.json``.
"""

from __future__ import annotations

import json

import pytest

from observe import detail, query
from observe.index import Index
from tcenv.evaluate import EvalReport
from tcenv.schema import REPORT_SCHEMA_VERSION

from ..support.orchestrator import stamp_report
from tests.support.v1_tasks import V1_TASK

KERNEL = "kernel void k() {}\n"
SOURCE = b"kernel void k() {}\n"


def _env(tmp_path, monkeypatch):
    runs = tmp_path / "runs"
    runs.mkdir(parents=True)
    monkeypatch.setenv("AI_LOWERLING_RUNS_DIR", str(runs))
    monkeypatch.setenv("OBSERVE_DB", str(tmp_path / "observe.sqlite"))
    monkeypatch.setenv("AI_LOWERLING_BOOKS", str(tmp_path / "none.sqlite"))
    monkeypatch.setenv("OBSERVE_REFRESH_ON_READ", "0")
    return runs


def _schema4_report(task: str = V1_TASK) -> dict:
    report = EvalReport(
        task=task, dtype="fp16", size="small", mode="full", seed=1, stage="done",
        status="ok", message="ok", speedup=1.2, target_id="apple-m4-max",
    )
    report.speedup, report.speedups, report.candidate_us, report.baseline_us = 1.2, {"mlx": 1.2}, 100.0, {"mlx": 120.0}
    stamp_report(report, task, "fp16", "small", KERNEL, build_kind="native")
    return report.to_json()


def _run(runs, name: str, *, report: dict | None, write_report: bool = True):
    directory = runs / name
    folder = directory / "candidates" / "000_r1.p0"
    folder.mkdir(parents=True)
    (folder / "kernel.metal").write_text(KERNEL, encoding="utf-8")
    if write_report:
        (folder / "report.json").write_text(json.dumps(report if report is not None else {}), encoding="utf-8")
    (directory / "state.json").write_text(json.dumps({
        "task": V1_TASK, "dtype": "fp16", "phase": "done", "started_at": 1.0, "updated_at": 2.0,
    }), encoding="utf-8")
    (directory / "result.json").write_text(json.dumps({
        "task": V1_TASK, "dtype": "fp16", "size": "small", "stop_reason": "max_rounds",
        "started_at": 1.0, "spent_usd": 0.1, "best_speedup": 1.2,
    }), encoding="utf-8")
    return directory


def _row(index, run_id: str) -> dict:
    return dict(index.conn.execute(
        "SELECT * FROM candidates WHERE run_id = ?", (run_id,)
    ).fetchone())


def test_a_schema4_report_is_current_and_carries_its_build_identity(tmp_path, monkeypatch):
    runs = _env(tmp_path, monkeypatch)
    _run(runs, "conv2d/current", report=_schema4_report())
    index = Index(tmp_path / "observe.sqlite")
    index.sync_known()
    row = _row(index, "conv2d/current")
    assert row["report_state"] == "current"
    assert row["build_shape"] == "full"
    assert row["build_kind"] == "native"
    assert row["evaluation_fingerprint"]
    panel = detail.candidate_row(runs / "conv2d/current", runs / "conv2d/current" / "candidates" / "000_r1.p0")
    assert panel["report_state"] == "current"
    assert panel["build"]["shape"] == "full" and panel["build"]["build_kind"] == "native"
    assert panel["evaluation_fingerprint"] == row["evaluation_fingerprint"]


def test_an_old_format_report_is_marked_undecidable_and_gets_no_identity(tmp_path, monkeypatch):
    runs = _env(tmp_path, monkeypatch)
    old = _schema4_report()
    old["schema_version"] = 3
    old.pop("build")
    old.pop("evaluation_fingerprint")
    _run(runs, "conv2d/old", report=old)
    index = Index(tmp_path / "observe.sqlite")
    index.sync_known()
    row = _row(index, "conv2d/old")
    assert row["report_state"] == "old_format"
    # No measurement conclusion and no identity: the row says *which* state it is, nothing else.
    assert row["status"] == "old_format" and row["speedup"] is None
    assert row["build_shape"] is None and row["build_kind"] is None
    assert row["evaluation_fingerprint"] is None
    panel = detail.candidate_row(runs / "conv2d/old", runs / "conv2d/old" / "candidates" / "000_r1.p0")
    assert panel["report_state"] == "old_format"
    assert panel["status"] == "old_format" and panel["speedup"] is None
    assert panel["cases"] == [] and panel["roofline"]["bound"] is None
    assert panel["build"] == {} and panel["evaluation_fingerprint"] is None


def test_a_damaged_report_is_unreadable_not_missing(tmp_path, monkeypatch):
    runs = _env(tmp_path, monkeypatch)
    directory = _run(runs, "conv2d/broken", report=None, write_report=False)
    (directory / "candidates" / "000_r1.p0" / "report.json").write_text("{not json", encoding="utf-8")
    index = Index(tmp_path / "observe.sqlite")
    index.sync_known()
    row = _row(index, "conv2d/broken")
    assert row["report_state"] == "unreadable"
    assert row["build_shape"] is None and row["evaluation_fingerprint"] is None


def test_a_candidate_directory_without_a_report_is_missing(tmp_path, monkeypatch):
    runs = _env(tmp_path, monkeypatch)
    _run(runs, "conv2d/absent", report=None, write_report=False)
    index = Index(tmp_path / "observe.sqlite")
    index.sync_known()
    row = _row(index, "conv2d/absent")
    assert row["report_state"] == "missing"
    assert row["build_shape"] is None and row["build_kind"] is None


def test_a_schema4_report_without_a_build_record_does_not_get_one(tmp_path, monkeypatch):
    """Current schema but no ``build``: the columns stay empty rather than being filled from the kernel."""
    runs = _env(tmp_path, monkeypatch)
    report = _schema4_report()
    report.pop("build")
    _run(runs, "conv2d/nobuild", report=report)
    index = Index(tmp_path / "observe.sqlite")
    index.sync_known()
    row = _row(index, "conv2d/nobuild")
    assert row["report_state"] == "current"
    assert row["build_shape"] is None and row["build_kind"] is None
    assert row["evaluation_fingerprint"]


def test_the_report_schema_constant_is_the_current_one():
    assert REPORT_SCHEMA_VERSION == 4


def test_candidate_queries_expose_the_report_state(tmp_path, monkeypatch):
    runs = _env(tmp_path, monkeypatch)
    _run(runs, "conv2d/current", report=_schema4_report())
    old = _schema4_report()
    old["schema_version"] = 3
    _run(runs, "conv2d/old", report=old)
    index = Index(tmp_path / "observe.sqlite")
    index.sync_known()
    listed = {row["run_id"]: row for row in query.candidates_page("conv2d/current", limit=10)["candidates"]}
    assert listed["conv2d/current"]["report_state"] == "current"
    old_rows = query.candidates_page("conv2d/old", limit=10)["candidates"]
    assert [row["report_state"] for row in old_rows] == ["old_format"]
    assert old_rows[0]["build_shape"] is None and old_rows[0]["evaluation_fingerprint"] is None
    # An old report is undecidable, so it is neither a passed row nor a failure row.
    assert old_rows[0]["status"] == "old_format" and old_rows[0]["speedup"] is None
    assert all(
        row["status"] != "old_format" for row in query.failures(limit=10)["failures"]
    ), "旧格式不是一次失败：它不进失败簇"
    # The failure join carries the same column; neither report failed, so it has no rows here.
    assert query.failures(limit=10)["failures"] == []


def test_the_candidate_panel_returns_no_body_for_an_old_or_damaged_report(tmp_path, monkeypatch):
    """候选面板只给 schema-4 正文：schema-3/损坏/缺失时标状态、正文为空，不展示旧测量结论。

    The reviewer's repro: a schema-3 ``status=ok, speedup=9.9`` report came back as
    ``{'report_state': None, 'report': {...9.9}}``.  The panel is a reading surface (the raw bytes
    have their own explicit route), so an old report must be marked and must not hand over its
    measurement conclusion as if it were current evidence.
    """
    runs = _env(tmp_path, monkeypatch)
    old = {"schema_version": 3, "status": "ok", "speedup": 9.9, "candidate_us": 1.0}
    _run(runs, f"{V1_TASK}/old", report=old)
    broken = _run(runs, f"{V1_TASK}/broken", report=None)
    (broken / "candidates" / "000_r1.p0" / "report.json").write_text("{not json", encoding="utf-8")
    _run(runs, f"{V1_TASK}/missing", report=None, write_report=False)
    _run(runs, f"{V1_TASK}/current", report=_schema4_report())

    panel = query.candidate_panel(f"{V1_TASK}/old", "r1.p0")
    assert panel["report_state"] == "old_format"
    assert panel["report"] == {}, "旧格式报告的正文不得作为当前候选证据返回"
    assert panel["report_path"].endswith("candidates/000_r1.p0/report.json")
    assert panel["excerpt"] == KERNEL, "源码片段与报告当前性无关，仍然给出"
    assert "speedup" not in json.dumps(panel), "旧报告的测量结论不能从面板漏出"

    broken_panel = query.candidate_panel(f"{V1_TASK}/broken", "r1.p0")
    assert broken_panel["report_state"] == "unreadable" and broken_panel["report"] == {}
    missing_panel = query.candidate_panel(f"{V1_TASK}/missing", "r1.p0")
    assert missing_panel["report_state"] == "missing" and missing_panel["report"] == {}

    current = query.candidate_panel(f"{V1_TASK}/current", "r1.p0")
    assert current["report_state"] == "current"
    assert current["report"]["schema_version"] == REPORT_SCHEMA_VERSION
    assert current["report"]["speedup"] == pytest.approx(1.2)


def test_the_feature_grid_takes_no_speedup_from_an_old_report(tmp_path, monkeypatch):
    """网格格子只记当前报告的速度：旧格式候选不能靠 9.9 抢到它那一格。"""
    runs = _env(tmp_path, monkeypatch)
    run = _run(runs, f"{V1_TASK}/grid", report=_schema4_report())
    old = run / "candidates" / "001_r2.p0"
    old.mkdir(parents=True)
    (old / "kernel.metal").write_text("kernel void k() {} // different\n", encoding="utf-8")
    (old / "report.json").write_text(json.dumps({"schema_version": 3, "status": "ok", "speedup": 9.9}), encoding="utf-8")
    result = json.loads((run / "result.json").read_text())
    rows = query.feature_grid(run, result)
    assert rows, "两格都要出现：一格是当前报告，一格没有速度"
    speeds = [row["speedup"] for row in rows]
    assert 9.9 not in speeds, "旧格式报告的速度不得进入网格"
    assert any(speed == pytest.approx(1.2) for speed in speeds)
