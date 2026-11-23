"""N0/N1 audit keeps natural runs, rechecks, duplicates and missing units apart."""

from __future__ import annotations

import json
from pathlib import Path

from taic.n1_metrics_audit import SCHEMA, audit, render_summary


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _selection(path: Path) -> None:
    _write(path, {
        "session_label": "r1.p0", "nominations": 2, "unique_eligible": 2,
        "reviewed": 1, "uncovered_count": 1, "ran_gpu": True, "gpu_seconds": 0.125,
        "measurements": [{"conclusion": "too_noisy"}],
        "uncovered": [{"omit_reason": "over_k", "conclusion": "no_diff"}],
    })


def test_selection_copies_are_deduped_and_plain_pair_time_is_not_gpu_seconds(tmp_path: Path):
    runs = tmp_path / "runs"
    run = runs / "target" / "run-a"
    _write(run / "run_config.json", {"task": "conv2d", "target": {"id": "apple-m4-max", "backend": "metal"}, "llm": {"model": "deepseek-flash"}, "search": {"paired_seed_observation": True}})
    _write(run / "result.json", {
        "task": "conv2d", "dtype": "fp16", "size": "small", "model": "deepseek-flash",
        "best_speedup": 1.2, "confirmed_speedup": 1.1, "confirmed_status": "ok", "spent_usd": 0.25,
        "trace": [{"idx": 0, "label": "prior", "status": "ok"}, {"idx": 1, "label": "r1.p0", "status": "ok"}],
    })
    _write(run / "candidates/001_r1.p0/paired_comparison.json", {
        "applied": True, "conclusion": "too_noisy", "elapsed_s": 0.5,
    })
    _selection(run / "sessions/r1.p0/paired_host_selection.json")
    _selection(run / "candidates/001_r1.p0/paired_host_selection.json")
    document = audit(runs)
    assert document["schema"] == SCHEMA
    selection = document["nomination_selection"]
    assert selection["counts"]["sessions"] == 1
    assert selection["candidate_duplicate_copies_ignored"] == 1
    assert selection["counts"]["unique_eligible"] == 2
    assert selection["counts"]["reviewed"] == 1
    assert selection["counts"]["over_k"] == 1
    assert selection["counts"]["uncovered_no_diff_or_regressed"] == 1
    assert selection["review_coverage"] == 0.5
    assert selection["gpu_seconds"] == 0.125
    pairs = document["natural_paired_comparisons"]
    assert pairs["elapsed_s_sum"] == 0.5
    assert pairs["gpu_seconds"] is None
    row = document["n0"]["runs"][0]
    assert row["first_ok_label"] == "prior"
    assert row["first_model_ok_label"] == "r1.p0"
    assert row["first_model_ok_round"] == 1
    assert document["nomination_selection"]["k_change_recommended"] is False
    assert "pure GPU seconds: unavailable" in render_summary(document)


def test_missing_selection_is_not_zero_nominations(tmp_path: Path):
    runs = tmp_path / "runs"
    run = runs / "run-b"
    _write(run / "result.json", {"task": "gemm", "trace": [], "spent_usd": None})
    document = audit(runs)
    selection = document["nomination_selection"]
    assert selection["counts"].get("sessions", 0) == 0
    assert selection["counts"].get("nominations", 0) == 0
    assert selection["review_coverage"] is None
    assert selection["coverage"] == "unavailable"
    assert selection["gpu_seconds"] is None
    assert document["n0"]["runs"][0]["spent_usd"] is None


def test_real_audit_is_read_only_and_never_mixes_rechecks_or_speedups():
    runs = Path(__file__).resolve().parents[2] / "runs"
    if not runs.is_dir():
        pytest.skip("requires the repository's real runs/ corpus")
    before = {path: path.stat().st_mtime_ns for path in runs.rglob("paired*json")}
    document = audit(runs)
    after = {path: path.stat().st_mtime_ns for path in runs.rglob("paired*json")}
    assert after == before
    selection = document["nomination_selection"]
    if selection["counts"].get("unique_eligible"):
        assert 0.0 <= selection["review_coverage"] <= 1.0
    assert selection["k_change_recommended"] is False
    assert document["natural_paired_comparisons"]["gpu_seconds"] is None
    assert document["rechecks"]["files"] >= 0
    for group in document["n0"]["groups"]:
        assert "speedup" not in group
