"""Paired observations are compared with the actual trace decision, never with changes_seed."""

from __future__ import annotations

import json
from pathlib import Path

from taic.seed_decision_audit import SCHEMA, audit, render_summary


def _pair(run: Path, label: str, **fields) -> None:
    folder = run / "candidates" / f"001_{label}"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "paired_comparison.json").write_text(json.dumps({
        "schema": "taic.paired_comparison.1", "applied": True, "changes_seed": False,
        "conclusion": "too_noisy", "session_label": label,
        "median_a_s": 1.0, "median_b_s": 1.0, **fields,
    }), encoding="utf-8")


def test_decisive_pairs_are_compared_with_the_trace_not_changes_seed(tmp_path):
    runs = tmp_path / "runs"
    run = runs / "target" / "run-a"
    run.mkdir(parents=True)
    trace = [
        {"label": "r1", "improved": True},
        {"label": "r2", "within_noise": True},
        {"label": "r3", "improved": True},
        {"label": "r4", "improved": False, "within_noise": False},
        {"label": "r5", "improved": False},
    ]
    (run / "result.json").write_text(json.dumps({"trace": trace}), encoding="utf-8")
    (run / "candidates" / "000_init").mkdir(parents=True)
    (run / "candidates" / "000_init" / "paired_comparison.json").write_text(json.dumps({
        "applied": False, "reason": "no current seed to pair with",
    }), encoding="utf-8")
    _pair(run, "r1", conclusion="diff", direction="a_faster", median_a_s=1.0, median_b_s=1.1)
    _pair(run, "r2", conclusion="diff", direction="b_faster", median_a_s=1.1, median_b_s=1.0)
    _pair(run, "r3", conclusion="too_noisy")
    # Even if the observation field were hand-edited to true, the trace still says keep.
    _pair(run, "r4", conclusion="diff", direction="a_faster", changes_seed=True)
    _pair(run, "r5", conclusion="diff", direction=None)

    document = audit(runs)
    counts = document["counts"]
    assert document["schema"] == SCHEMA
    assert counts["pair_files"] == 6 and counts["not_applied"] == 1
    assert counts["applied"] == 5
    assert counts["decisive"] == 3
    assert counts["agree"] == 2 and counts["disagree"] == 1
    assert counts["unresolved"] == 2 and counts["unresolved_replace"] == 1
    assert counts["too_noisy"] == 1
    assert document["verdict"] == "review_decisive_mismatches"
    assert document["decision_change_recommended"] is False
    assert [row["verdict"] for row in document["decisive_pairs"]] == ["agree", "agree", "disagree"]
    assert document["decisive_pairs"][0]["ratio_candidate_over_seed"] == 1.1
    assert "too_noisy is not evidence" in render_summary(document)


def test_missing_trace_is_not_counted_twice(tmp_path):
    runs = tmp_path / "runs"
    run = runs / "run-missing"
    run.mkdir(parents=True)
    (run / "result.json").write_text(json.dumps({"trace": []}), encoding="utf-8")
    _pair(run, "r-unknown", conclusion="too_noisy")
    document = audit(runs)
    assert document["counts"]["missing_trace"] == 1
    assert document["counts"]["unresolved"] == 1
    assert document["counts"]["decisive"] == 0
    assert document["verdict"] == "insufficient_decisive_evidence"


def test_real_runs_are_read_only_and_internally_consistent():
    runs = Path(__file__).resolve().parents[2] / "runs"
    if not runs.is_dir():
        pytest.skip("requires the repository's real runs/ corpus")
    result_files = list(runs.rglob("result.json"))
    before = {path: path.stat().st_mtime_ns for path in result_files}
    document = audit(runs)
    after = {path: path.stat().st_mtime_ns for path in result_files}
    assert after == before
    counts = document["counts"]
    assert counts["pair_files"] == counts["applied"] + counts["not_applied"]
    assert counts["applied"] == counts["agree"] + counts["disagree"] + counts["unresolved"]
    assert counts["decisive"] == counts["agree"] + counts["disagree"]
    assert counts["too_noisy"] <= counts["unresolved"]
    assert counts["unresolved_replace"] + counts["unresolved_keep"] + counts["unresolved_tie"] + counts["missing_trace"] <= counts["applied"]
    assert document["decision_change_recommended"] is False
    # A disagreement changes the review target, never the production rule automatically.
    if counts["disagree"]:
        assert document["verdict"] == "review_decisive_mismatches"
    elif counts["agree"]:
        assert document["verdict"] == "no_observed_misdecision"
    else:
        assert document["verdict"] == "insufficient_decisive_evidence"
