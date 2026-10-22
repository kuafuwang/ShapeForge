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

KERNEL = "relu_kernel"


def _legacy_report(**extra) -> dict:
    doc = {"task": "relu", "dtype": "fp16", "size": "small", "mode": "quick", "status": "ok", "eligible": True}
    doc.update(extra)
    return doc


def test_a_legacy_report_loads_and_a_round_trip_records_the_version():
    rep = EvalReport.from_json(_legacy_report(speedup=1.2))
    assert rep.status == "ok" and rep.eligible and rep.speedup == pytest.approx(1.2)
    again = EvalReport.from_json(rep.to_json())
    assert again.to_json()["schema_version"] == REPORT_SCHEMA_VERSION == 2
    assert again.status == "ok" and again.speedup == pytest.approx(1.2)


def test_eligible_true_without_a_status_loads_as_ok():
    rep = EvalReport.from_json({"task": "relu", "dtype": "fp16", "size": "small", "mode": "full", "eligible": True, "speedup": 1.1})
    assert rep.status == "ok" and rep.eligible and rep.speedup == pytest.approx(1.1)


def test_a_report_with_an_unknown_field_or_a_newer_version_needs_inspection():
    with pytest.raises(SchemaError, match="needs human inspection"):
        EvalReport.from_json(_legacy_report(future_field=1))
    with pytest.raises(SchemaError, match="needs human inspection"):
        EvalReport.from_json(_legacy_report(schema_version=3))
    with pytest.raises(SchemaError, match="needs human inspection"):
        EvalReport.from_json(_legacy_report(schema_version=True))


def test_bench_stats_does_not_grow_a_per_rep_sample_list():
    assert "samples" not in {item.name for item in fields(BenchStats)}
    assert SearchConfig().differential_recheck is False


def test_old_result_json_loads_and_a_newer_or_damaged_file_is_not_treated_as_missing(tmp_path):
    missing = tmp_path / "missing"
    missing.mkdir()
    assert read_result(missing) is None

    old = tmp_path / "old"
    old.mkdir()
    (old / "result.json").write_text(json.dumps({"task": "relu", "spent_usd": 1.5, "stop_reason": "budget_exhausted"}))
    loaded = read_result(old)
    assert loaded is not None and loaded["task"] == "relu" and loaded["spent_usd"] == 1.5

    newer = tmp_path / "newer"
    newer.mkdir()
    (newer / "result.json").write_text(json.dumps({"schema_version": 99, "task": "relu", "spent_usd": 1}))
    with pytest.raises(NeedsInspection, match="needs human inspection"):
        read_result(newer)

    damaged = tmp_path / "damaged"
    damaged.mkdir()
    (damaged / "result.json").write_text("{not json")
    with pytest.raises(NeedsInspection, match="needs human inspection"):
        read_result(damaged)


def _candidate(run: Path, report: dict | str) -> None:
    folder = run / "candidates" / "000_init"
    folder.mkdir(parents=True)
    (folder / "kernel.metal").write_text(f"kernel void {KERNEL}() {{}}\n")
    text = report if isinstance(report, str) else json.dumps(report)
    (folder / "report.json").write_text(text)
    (run / "run_config.json").write_text(json.dumps({"loop_protocol": "flame_chase_1"}))
    (run / "state.json").write_text(json.dumps({
        "task": "relu", "dtype": "fp16", "size": "small", "model": "fake", "spent_usd": 0.2, "round": 0, "best_label": "init",
        "requests": [{"label": "forged", "cost_usd": 9, "total_usd": 9, "hit": 0, "miss": 0, "out": 0}],
    }))


def test_a_damaged_candidate_report_needs_inspection_and_a_missing_one_does_not(tmp_path):
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
    (missing_report / "state.json").write_text(json.dumps({"task": "relu", "dtype": "fp16", "spent_usd": 0.1}))
    (missing_report / "run_config.json").write_text(json.dumps({"loop_protocol": "flame_chase_1"}))
    with pytest.raises(ResumeError, match="no candidates"):
        load_snapshot(missing_report, SearchConfig(), KERNEL)


def test_a_checkpoint_request_list_does_not_replace_the_transcript(tmp_path):
    run = tmp_path / "run"
    _candidate(run, _legacy_report(speedup=1.1))
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
    from tcenv.schema import REQUEST_SCHEMA_VERSION

    base = {"task": "relu", "dtype": "fp16", "size": "small", "kernel_path": "k.metal", "mode": "quick", "seed": 1, "reps": 3, "out": "out.json", "target_id": "apple-m4-max"}
    assert read_request(base)["seed"] == 1
    assert read_request({**base, "schema_version": 4})["target_id"] == "apple-m4-max"
    with pytest.raises(SchemaError, match="needs human inspection"):
        read_request({**base, "skip_safety": True})
    with pytest.raises(SchemaError, match="needs human inspection"):
        read_request({**base, "schema_version": REQUEST_SCHEMA_VERSION + 1})


def test_atomic_json_replace_leaves_only_the_finished_file(tmp_path):
    path = tmp_path / "nested" / "report.json"
    write_json_atomic(path, {"schema_version": 1, "status": "ok"})
    assert json.loads(path.read_text())["status"] == "ok"
    assert [item.name for item in path.parent.iterdir()] == ["report.json"]


def test_a_damaged_worker_report_is_not_treated_as_a_missing_report(monkeypatch):
    from tcenv import evaluate as ev

    monkeypatch.setattr(ev, "POST_KILL_COOLDOWN_S", 0.0)
    rep = ev._run_worker(
        "relu", "fp16", "small", Path("/dev/null"), "quick", 1, 3, 30.0,
        validation=False, cases=None, worker_module="tests.fake_worker_bad_report",
    )
    assert rep.status == "crash"
    assert "needs human inspection" in rep.message
    assert "without a report" not in rep.message


def test_a_report_records_the_measurement_protocol():
    from tcenv.schema import PROTOCOL_VERSION

    report = EvalReport(task="relu", dtype="fp16", size="small", mode="full")
    report.protocol_version = PROTOCOL_VERSION
    doc = report.to_json()
    assert doc["protocol_version"] == PROTOCOL_VERSION
    assert EvalReport.from_json(doc).protocol_version == PROTOCOL_VERSION
