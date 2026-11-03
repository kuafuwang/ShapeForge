"""Versioned worker reports and run checkpoints. No model calls and no GPU."""

from __future__ import annotations

import json
from dataclasses import fields
from pathlib import Path

import pytest

from experiments.common import read_result
from taic.artifacts import NeedsInspection
from taic.llm import SearchConfig
from taic.resume import ResumeError, inspect_run, load_snapshot
from tcenv.bench import BenchStats
from tcenv.evaluate import EvalReport
from tcenv.schema import REPORT_SCHEMA_VERSION, SchemaError, read_request, write_json_atomic
from tests.support.report_build import attach_identity, ok_build, ok_report_dict
from tests.support.v1_tasks import V1_TASK

KERNEL = "relu_kernel"


def _legacy_report(**extra) -> dict:
    """@brief 造一份 R4 之前的报告 JSON（无 ``schema_version``、无 ``build``），可按需补字段。

    @param extra 追加或覆盖的顶层字段。
    @return 报告字典。
    """
    doc = {"task": V1_TASK, "dtype": "fp16", "size": "small", "mode": "quick", "status": "ok", "eligible": True}
    doc.update(extra)
    return doc


def test_an_old_report_is_refused_instead_of_migrated():
    """@brief 验证 v0/v3 报告被拒绝：R4 只读 schema 4，没有迁移、没有补状态。"""
    with pytest.raises(SchemaError, match="not the schema this process reads"):
        EvalReport.from_json(_legacy_report(speedup=1.2))
    with pytest.raises(SchemaError, match="not the schema this process reads"):
        EvalReport.from_json(_legacy_report(schema_version=3, build=None))
    with pytest.raises(SchemaError, match="only version this process reads|not the schema"):
        EvalReport.from_json(_legacy_report(schema_version=2, build=None))


def test_a_new_report_carries_backend_version_and_a_stage_timeline():
    """@brief 验证 schema 4 报告带上后端版本与分阶段时间线，往返一次内容不变。"""
    from tasks import get_task

    task_obj = get_task(V1_TASK, "fp16", "small")
    report = EvalReport(
        task=V1_TASK, dtype="fp16", size="small", mode="full", status="ok",
        backend="sim-npu", backend_version="1", target_id="apple-m4-max",
        timeline=[{"stage": "compile", "duration_s": 0.1}, {"stage": "done", "status": "ok", "duration_s": 0.2}],
    )
    report.build = ok_build(V1_TASK, "fp16", "small")
    attach_identity(report, task_obj)
    doc = report.to_json()
    assert doc["schema_version"] == REPORT_SCHEMA_VERSION == 4
    assert doc["backend_version"] == "1"
    assert doc["timeline"][0]["stage"] == "compile"
    again = EvalReport.from_json(doc)
    assert again.backend_version == "1" and again.timeline == report.timeline


def test_eligible_alone_is_not_a_status():
    """@brief 验证只有 ``eligible: true`` 没有 status 的旧报告被拒绝，不再补成 ok。"""
    with pytest.raises(SchemaError, match="not the schema this process reads"):
        EvalReport.from_json({"task": V1_TASK, "dtype": "fp16", "size": "small", "mode": "full", "eligible": True, "speedup": 1.1})


def test_a_report_with_an_unknown_field_or_a_newer_version_needs_inspection():
    """@brief 验证未知字段、更新的版本号以及把版本写成 bool 都被判为需要人工检查。"""
    good = ok_report_dict(V1_TASK, "fp16", "small")
    with pytest.raises(SchemaError, match="needs human inspection"):
        EvalReport.from_json({**good, "future_field": 1})
    with pytest.raises(SchemaError, match="needs human inspection"):
        EvalReport.from_json({**good, "schema_version": REPORT_SCHEMA_VERSION + 1})
    with pytest.raises(SchemaError, match="needs human inspection"):
        EvalReport.from_json({**good, "schema_version": True})


def test_bench_stats_does_not_grow_a_per_rep_sample_list():
    """@brief 验证 ``BenchStats`` 不新增逐次样本字段，且差分复检默认关闭。"""
    assert "samples" not in {item.name for item in fields(BenchStats)}
    assert SearchConfig().differential_recheck is False


def test_old_result_json_loads_and_a_newer_or_damaged_file_is_not_treated_as_missing(tmp_path):
    """@brief 验证缺文件返回 None，旧 result.json 照常加载，而更新版本或损坏的文件抛 ``NeedsInspection`` 而不是当成缺失。

    @param tmp_path 临时目录，四种运行目录都在这里造。
    """
    missing = tmp_path / "missing"
    missing.mkdir()
    assert read_result(missing) is None

    old = tmp_path / "old"
    old.mkdir()
    (old / "result.json").write_text(json.dumps({"task": V1_TASK, "spent_usd": 1.5, "stop_reason": "budget_exhausted"}))
    loaded = read_result(old)
    assert loaded is not None and loaded["task"] == V1_TASK and loaded["spent_usd"] == 1.5

    newer = tmp_path / "newer"
    newer.mkdir()
    (newer / "result.json").write_text(json.dumps({"schema_version": 99, "task": V1_TASK, "spent_usd": 1}))
    with pytest.raises(NeedsInspection, match="needs human inspection"):
        read_result(newer)

    damaged = tmp_path / "damaged"
    damaged.mkdir()
    (damaged / "result.json").write_text("{not json")
    with pytest.raises(NeedsInspection, match="needs human inspection"):
        read_result(damaged)


def _candidate(run: Path, report: dict | str) -> None:
    """@brief 造一个只含单候选（``000_init``）的运行目录，并写好报告、state 与运行配置。

    @param run 运行目录根。
    @param report 候选报告；dict 会转成 JSON，字符串原样写入（用来造损坏文件）。
    """
    folder = run / "candidates" / "000_init"
    folder.mkdir(parents=True)
    (folder / "kernel.metal").write_text(f"kernel void {KERNEL}() {{}}\n")
    text = report if isinstance(report, str) else json.dumps(report)
    (folder / "report.json").write_text(text)
    (run / "run_config.json").write_text(json.dumps({"loop_protocol": "flame_chase_1", "report_schema_epoch": REPORT_SCHEMA_VERSION}))
    (run / "state.json").write_text(json.dumps({
        "task": V1_TASK, "dtype": "fp16", "size": "small", "model": "fake", "spent_usd": 0.2, "round": 0, "best_label": "init",
        "requests": [{"label": "forged", "cost_usd": 9, "total_usd": 9, "hit": 0, "miss": 0, "out": 0}],
    }))


def test_a_damaged_candidate_report_needs_inspection_and_a_missing_one_does_not(tmp_path):
    """@brief 验证候选报告损坏时抛 ``NeedsInspection``，空目录判为不可续跑，缺候选报告则报 no candidates。

    @param tmp_path 临时目录，几种运行目录都在这里造。
    """
    damaged = tmp_path / "damaged"
    _candidate(damaged, "{not json")
    (damaged / "transcript.jsonl").write_text(json.dumps({"event": "chat", "label": "init", "usage": {}, "cost_usd": 0.2, "total_usd": 0.2}) + "\n")
    with pytest.raises(NeedsInspection, match="needs human inspection"):
        load_snapshot(damaged, SearchConfig(), KERNEL)
    (tmp_path / "empty").mkdir()
    info = inspect_run(tmp_path / "empty")
    assert info["resumable"] is False and "not a search run" in info["reason"]
    broken = tmp_path / "broken"
    broken.mkdir()
    (broken / "result.json").write_text("{")
    seen = inspect_run(broken)
    assert seen["resumable"] is False and "needs human inspection" in seen["reason"] and "not a search run" not in seen["reason"]

    missing_report = tmp_path / "missing-report"
    folder = missing_report / "candidates" / "000_init"
    folder.mkdir(parents=True)
    (folder / "kernel.metal").write_text("kernel void relu_kernel() {}\n")
    (missing_report / "state.json").write_text(json.dumps({"task": V1_TASK, "dtype": "fp16", "spent_usd": 0.1}))
    (missing_report / "run_config.json").write_text(json.dumps({"loop_protocol": "flame_chase_1", "report_schema_epoch": REPORT_SCHEMA_VERSION}))
    with pytest.raises(ResumeError, match="no candidates"):
        load_snapshot(missing_report, SearchConfig(), KERNEL)


def test_a_checkpoint_request_list_does_not_replace_the_transcript(tmp_path):
    """@brief 验证账单请求以 ``transcript.jsonl`` 为准，checkpoint 里的请求列表不会覆盖它，且 transcript 不被改写。

    @param tmp_path 临时目录，运行目录在这里造。
    """
    run = tmp_path / "run"
    _candidate(run, ok_report_dict(V1_TASK, "fp16", "small", speedup=1.1))
    transcript = run / "transcript.jsonl"
    transcript.write_text(json.dumps({"event": "chat", "label": "init", "usage": {}, "cost_usd": 0.2, "total_usd": 0.2}) + "\n")
    (run / "ledger.json").write_text(json.dumps({
        "requests": [{"label": "forged-ledger", "cost_usd": 4, "total_usd": 4, "hit": 0, "miss": 0, "out": 0}],
        "summary": {"total_usd": 0.2},
    }))
    before = transcript.read_bytes()
    snap = load_snapshot(run, SearchConfig(), KERNEL)
    assert transcript.read_bytes() == before
    assert [row["label"] for row in snap.requests] == ["init"]
    assert snap.requests[0]["cost_usd"] == pytest.approx(0.2)
    assert any("transcript.jsonl is the billed-request record" in note for note in snap.notes)
    transcript.unlink()
    with pytest.raises(NeedsInspection, match="cannot replace the billed transcript"):
        load_snapshot(run, SearchConfig(), KERNEL)


def test_worker_request_version_0_loads_and_a_newer_request_is_refused():
    """@brief 验证 worker 请求缺版本号时按 v0 接受，未知字段与更新的版本被拒。"""
    from tcenv.schema import REQUEST_SCHEMA_VERSION

    base = {"task": V1_TASK, "dtype": "fp16", "size": "small", "kernel_path": "k.metal", "mode": "quick", "seed": 1, "reps": 3, "out": "out.json", "target_id": "apple-m4-max"}
    assert read_request(base)["seed"] == 1
    assert read_request({**base, "schema_version": 4})["target_id"] == "apple-m4-max"
    with pytest.raises(SchemaError, match="needs human inspection"):
        read_request({**base, "skip_safety": True})
    with pytest.raises(SchemaError, match="needs human inspection"):
        read_request({**base, "schema_version": REQUEST_SCHEMA_VERSION + 1})


def test_atomic_json_replace_leaves_only_the_finished_file(tmp_path):
    """@brief 验证原子写只留下最终文件，目录里不残留临时文件。

    @param tmp_path 临时目录，目标路径的父目录会被自动创建。
    """
    path = tmp_path / "nested" / "report.json"
    write_json_atomic(path, {"schema_version": 1, "status": "ok"})
    assert json.loads(path.read_text())["status"] == "ok"
    assert [item.name for item in path.parent.iterdir()] == ["report.json"]


def test_a_damaged_worker_report_is_not_treated_as_a_missing_report(monkeypatch):
    """@brief 验证 worker 写出损坏报告时归为 crash 并提示人工检查，而不是当成没写报告。

    @param monkeypatch 把评测后的冷却时间置零，避免额外等待。
    """
    from tcenv import evaluate as ev

    monkeypatch.setattr(ev, "POST_KILL_COOLDOWN_S", 0.0)
    rep = ev._run_worker(
        V1_TASK, "fp16", "small", Path("/dev/null"), "quick", 1, 3, 30.0,
        validation=False, cases=None, worker_module="tests.support.fake_worker_bad_report",
    )
    assert rep.status == "crash"
    assert "needs human inspection" in rep.message
    assert "without a report" not in rep.message


def test_a_report_records_the_measurement_protocol():
    """@brief 验证报告带上测量协议版本，并能原样往返。"""
    from tasks import get_task

    from tcenv.schema import PROTOCOL_VERSION

    task_obj = get_task(V1_TASK, "fp16", "small")
    report = EvalReport(task=V1_TASK, dtype="fp16", size="small", mode="full", status="ok", target_id="apple-m4-max")
    report.build = ok_build(V1_TASK, "fp16", "small")
    attach_identity(report, task_obj)
    report.protocol_version = PROTOCOL_VERSION
    doc = report.to_json()
    assert doc["protocol_version"] == PROTOCOL_VERSION
    assert EvalReport.from_json(doc).protocol_version == PROTOCOL_VERSION
