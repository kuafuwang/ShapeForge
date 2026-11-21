"""H0 inventory freezes run roots, cohorts and missing-data contracts."""

from __future__ import annotations

import json
from pathlib import Path

from taic.scorecard_contract import (
    IDENTITY_FIELDS,
    SUFFICIENCY_THRESHOLDS,
    canonical_session_label,
    compare_identity,
    cost_metrics,
    evidence_kind,
    metric_status,
    resolve_candidate_status,
)
from taic.scorecard_inventory import inventory


def _write(path: Path, value: dict | str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) if isinstance(value, dict) else value, encoding="utf-8")


def _result(stop: str = "max_rounds") -> dict:
    return {"task": "conv2d", "trace": [], "stop_reason": stop, "spent_usd": 0.25}


def test_run_roots_nested_cells_and_dry_runs_are_distinct(tmp_path: Path):
    runs = tmp_path / "runs"
    natural = runs / "apple-m4-max" / "20261121-010101_conv2d_fp16_deepseek-flash_x"
    _write(natural / "result.json", _result())
    _write(natural / "run_config.json", {"task": "conv2d"})
    custom = runs / "apple-m4-max" / "custom-paid-smoke"
    _write(custom / "result.json", _result())
    experiment = runs / "experiment"
    _write(experiment / "plan.json", {"schema_id": "experiments.plan.1", "cells": []})
    _write(experiment / "approval.json", {"approved_by": "kuafu", "experiment_id": "exp-1"})
    cell = experiment / "control" / "cell-1"
    _write(cell / "result.json", _result())
    _write(cell / "run_config.json", {"task": "conv2d"})
    # A real paid cell's workdir overlay is not another dry run.
    _write(cell / "agent_work" / "r1.p0" / "experience_overlay.json", {"overlay": True})
    dry = runs / "apple-m4-max" / "render-only"
    _write(dry / "run_config.json", {"task": "conv2d"})
    _write(dry / "experience_overlay.json", {"overlay": True})
    broken = runs / "apple-m4-max" / "broken"
    _write(broken / "state.json", {"task": "conv2d", "spent_usd": 0.0})
    unclassified = runs / "apple-m4-max" / "bad-result"
    _write(unclassified / "result.json", {})
    _write(unclassified / "run_config.json", {"task": "conv2d"})

    document = inventory(runs)
    by_root = {row["run_root"]: row for row in document["runs"]}
    assert len(by_root) == 6
    assert not any("agent_work" in root for root in by_root)
    assert by_root["apple-m4-max/20261121-010101_conv2d_fp16_deepseek-flash_x"]["cohort"] == "natural"
    assert by_root["apple-m4-max/custom-paid-smoke"]["cohort"] == "unknown"
    cell_row = by_root["experiment/control/cell-1"]
    assert cell_row["cohort"] == "controlled"
    assert cell_row["experiment_kind"] == "paid"
    assert cell_row["experiment_id"] == "exp-1"
    assert cell_row["cell_id"] == "control/cell-1"
    assert by_root["apple-m4-max/render-only"]["lifecycle"] == "dry_run"
    assert by_root["apple-m4-max/broken"]["lifecycle"] == "incomplete_or_aborted"
    assert by_root["apple-m4-max/bad-result"]["lifecycle"] == "unclassified"
    assert document["experiment_containers"] == [{"path": "experiment", "child_runs": 1, "counted_as_run": False}]
    assert document["h1_implemented"] is False


def test_old_plan_without_id_gets_a_fingerprint_not_a_fabricated_id(tmp_path: Path):
    runs = tmp_path / "runs"
    _write(runs / "exp" / "plan.json", {"schema_id": "experiments.plan.1"})
    _write(runs / "exp" / "control" / "result.json", _result())
    row = inventory(runs)["runs"][0]
    assert row["experiment_id"] is None
    assert len(row["experiment_fingerprint"]) == 64
    assert row["cell_id"] == "control"


def test_recovery_requires_an_explicit_link(tmp_path: Path):
    runs = tmp_path / "runs"
    _write(runs / "exp" / "plan.json", {"schema_id": "experiments.plan.1"})
    _write(runs / "exp" / "recovery" / "result.json", {**_result("agent_failure"), "recovery_of": "control/cell-1"})
    row = inventory(runs)["runs"][0]
    assert row["cohort"] == "recovery"
    assert row["recovery_of"] == "control/cell-1"
    assert row["lifecycle"] == "finalized"
    assert row["stop_reason"] == "agent_failure"
    assert row["confirmed_status"] is None


def test_identity_cost_retry_and_evidence_contracts():
    left = {name: name for name in IDENTITY_FIELDS}
    assert compare_identity(left, left) == {"status": "match", "identity_scope": "full", "conflicts": []}
    partial = compare_identity(left, {"candidate": "candidate"})
    assert partial["status"] == "partial" and partial["identity_scope"] == "partial"
    mismatch = compare_identity(left, {**left, "candidate": "other"})
    assert mismatch["status"] == "mismatch" and mismatch["conflicts"] == ["candidate"]
    assert compare_identity({}, {})["status"] == "unavailable"

    cost = cost_metrics(1.0, 2, incomplete=True)
    assert cost["ledger_cost_per_ok"] == 0.5
    assert cost["exact_cost_per_ok"] is None
    zero = cost_metrics(1.0, 0, incomplete=False)
    assert zero["ledger_cost_per_ok"] is None and zero["ledger_spent_without_ok_usd"] == 1.0
    assert cost_metrics(None, 2, incomplete=None)["cost_complete"] is None

    assert canonical_session_label("r1.p0.retry2") == "r1.p0"
    assert canonical_session_label("r1.p0") == "r1.p0"
    resolved = resolve_candidate_status("ok", "incorrect")
    assert resolved["status"] == "incorrect" and resolved["source"] == "outer_report" and resolved["conflict"] is True
    assert resolve_candidate_status(None, None)["status"] is None
    assert evidence_kind({"measurement_kind": "measured"}) == "real"
    assert evidence_kind({"measurement_kind": "measured"}, {"measurement_type": "simulated"}) == "simulated"
    assert evidence_kind({}) is None


def test_sufficiency_is_per_unit_and_coverage_not_post_hoc():
    threshold = SUFFICIENCY_THRESHOLDS["run_level_min_units"]
    assert metric_status("run_level_min_units", units=threshold, known_units=threshold) == "measured"
    assert metric_status("run_level_min_units", units=threshold, known_units=threshold - 1) == "insufficient"
    assert metric_status("run_level_min_units", units=10, known_units=7) == "insufficient"
    assert metric_status("run_level_min_units", units=0, known_units=0) == "unavailable"


def test_real_inventory_is_read_only_and_never_promotes_custom_runs_to_natural():
    runs = Path(__file__).resolve().parents[2] / "runs"
    if not runs.is_dir():
        return
    before = {path: path.stat().st_mtime_ns for path in runs.rglob("*.json")}
    document = inventory(runs)
    after = {path: path.stat().st_mtime_ns for path in runs.rglob("*.json")}
    assert after == before
    assert document["run_count"] == len(document["runs"])
    assert sum(document["counts"]["lifecycle"].values()) == document["run_count"]
    for row in document["runs"]:
        assert not any(part in row["run_root"].split("/") for part in ("agent_work", "sessions", "candidates", ".selfcheck"))
        if row["cohort"] == "natural":
            import re
            assert re.match(r"^[0-9]{8}-[0-9]{6}_", row["run_root"].rsplit("/", 1)[-1])
        if row["lifecycle"] == "finalized":
            assert row["lifecycle"] != "completed_ok"
    assert document["h1_implemented"] is False
    assert document["publication_allowed"] is False
