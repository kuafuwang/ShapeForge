"""Omission/outcome analysis is stratified by round and never claims causality."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from taic.omission_impact_audit import SCHEMA, audit, render_summary


def _session(run: Path, label: str, *, omitted: int, improved: bool, status: str = "ok") -> None:
    run.mkdir(parents=True, exist_ok=True)
    result_path = run / "result.json"
    result = json.loads(result_path.read_text()) if result_path.exists() else {"trace": []}
    result["trace"].append({"label": label, "improved": improved, "status": status, "within_noise": False})
    result_path.write_text(json.dumps(result), encoding="utf-8")
    folder = run / "sessions" / label
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "task_delivery.json").write_text(json.dumps({
        "dynamic": {"delivered_bytes": 100, "omitted_bytes": omitted, "over_budget": False},
        "situation": {"status": "delivered", "layers": []}, "blocks": [],
    }), encoding="utf-8")


def test_round_stratification_prevents_init_and_late_round_confounding(tmp_path):
    runs = tmp_path / "runs"
    run = runs / "target" / "run-a"
    _session(run, "init", omitted=0, improved=True)
    _session(run, "r1.p0", omitted=0, improved=True)
    _session(run, "r2.p0", omitted=100, improved=False)
    _session(run, "r3.p0", omitted=200, improved=False)
    _session(run, "r4.p0", omitted=300, improved=True)
    document = audit(runs)
    assert document["schema"] == SCHEMA
    groups = {(row["stage"], row["arm"]): row for row in document["groups"]}
    assert groups[("init", "clean")]["new_best_rate"] == 1.0
    assert groups[("r1-2", "clean")]["new_best_rate"] == 1.0
    assert groups[("r1-2", "omitted")]["new_best_rate"] == 0.0
    assert groups[("r3+", "omitted")]["new_best_rate"] == 0.5
    assert document["early_new_best_rate_difference_omitted_minus_clean"] == -1.0
    assert document["verdict"] == "association_confounded_by_round"
    assert document["causal_claim_allowed"] is False
    assert "r1-2" in render_summary(document)


def test_missing_result_or_trace_is_skipped_not_scored_as_failure(tmp_path):
    runs = tmp_path / "runs"
    run = runs / "run-missing"
    _session(run, "r1.p0", omitted=100, improved=False)
    result = json.loads((run / "result.json").read_text())
    result["trace"] = []
    (run / "result.json").write_text(json.dumps(result), encoding="utf-8")
    document = audit(runs)
    assert document["groups"] == []
    assert document["skipped"]["missing_trace"] == 1
    assert document["causal_claim_allowed"] is False


def test_real_association_is_round_stratified_and_read_only():
    runs = Path(__file__).resolve().parents[2] / "runs"
    if not runs.is_dir():
        pytest.skip("requires the repository's real runs/ corpus")
    files = list(runs.rglob("task_delivery.json"))
    before = {path: path.stat().st_mtime_ns for path in files}
    document = audit(runs)
    after = {path: path.stat().st_mtime_ns for path in files}
    assert after == before
    assert document["causal_claim_allowed"] is False
    assert document["verdict"] == "association_confounded_by_round"
    groups = {(row["stage"], row["arm"]): row for row in document["groups"]}
    assert groups[("init", "clean")]["new_best_rate"] == 1.0
    assert groups[("r1-2", "omitted")]["sessions"] > 0
    assert groups[("r3+", "omitted")]["sessions"] > groups[("r3+", "clean")]["sessions"]
    for row in groups.values():
        assert row["sessions"] >= row["new_best"] >= 0
        assert row["sessions"] >= row["ok"] >= 0
        if row["sessions"]:
            assert row["new_best_rate"] == pytest.approx(row["new_best"] / row["sessions"])
            assert row["ok_rate"] == pytest.approx(row["ok"] / row["sessions"])
        else:
            assert row["new_best_rate"] is None and row["ok_rate"] is None
