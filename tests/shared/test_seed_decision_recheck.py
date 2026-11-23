"""Exact-SHA rechecks distinguish disagreement from a better shadow decision."""

from __future__ import annotations

import json
from pathlib import Path

from taic.seed_decision_recheck import SCHEMA, audit, render_summary


def _recheck(run: Path, label: str, conclusion: str, direction: str | None) -> None:
    folder = run / "candidates" / f"001_{label}"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "paired_comparison_recheck.json").write_text(json.dumps({
        "document": {"conclusion": conclusion, "direction": direction, "timed_by": "gpu_timestamp"},
    }), encoding="utf-8")


def test_disagreement_does_not_imply_a_better_alternative(tmp_path):
    runs = tmp_path / "runs"
    run = runs / "target" / "run-a"
    run.mkdir(parents=True)
    (run / "result.json").write_text(json.dumps({
        "best_label": "r2.p0",
        "trace": [
            {"idx": 0, "label": "init", "improved": True, "speedup": 1.0},
            {"idx": 1, "label": "r1.p0", "improved": True, "speedup": 1.2},
            {"idx": 2, "label": "r2.p0", "improved": True, "speedup": 1.3},
        ],
    }), encoding="utf-8")
    _recheck(run, "r1.p0", "diff", "a_faster")
    _recheck(run, "r2.p0", "diff", "b_faster")
    document = audit(runs)
    assert document["schema"] == SCHEMA
    assert document["counts"]["agree"] == 1 and document["counts"]["disagree"] == 1
    assert document["verdict"] == "review_decisive_mismatches"
    replay = document["shadow_replays"][0]
    assert replay["actual_best"] == "r2.p0"
    assert replay["alternative_best"] == "r1.p0"
    assert replay["alternative_search_speedup"] < replay["actual_search_speedup"]
    assert replay["outcome"] == "alternative_worse"
    assert document["decision_change_recommended"] is False
    assert "不是同一窗口" in document["note"]


def test_no_diff_stays_unresolved(tmp_path):
    runs = tmp_path / "runs"
    run = runs / "run-b"
    run.mkdir(parents=True)
    (run / "result.json").write_text(json.dumps({"best_label": "init", "trace": [{"idx": 0, "label": "init", "speedup": 1.0}]}), encoding="utf-8")
    _recheck(run, "r1.p0", "no_diff", None)
    document = audit(runs)
    assert document["counts"]["no_diff"] == 1
    assert document["counts"]["unresolved"] == 1
    assert document["counts"].get("decisive", 0) == 0
    assert document["verdict"] == "no_observed_misdecision"


def test_real_rechecks_are_read_only_and_balanced():
    runs = Path(__file__).resolve().parents[2] / "runs"
    if not runs.is_dir():
        pytest.skip("requires the repository's real runs/ corpus")
    files = list(runs.rglob("paired_comparison_recheck.json"))
    before = {path: path.stat().st_mtime_ns for path in files}
    document = audit(runs)
    after = {path: path.stat().st_mtime_ns for path in files}
    assert after == before
    counts = document["counts"]
    assert counts["decisive"] == counts["agree"] + counts["disagree"]
    assert counts["decisive"] + counts["unresolved"] == len(files)
    assert len(document["decisive_disagreements"]) == counts["disagree"]
    assert document["decision_change_recommended"] is False
    outcomes = [row["outcome"] for row in document["shadow_replays"]]
    assert sum(counts[f"replay_{outcome}"] for outcome in set(outcomes)) == document["runs_with_rechecks"]
    assert "不是同一窗口" in document["note"]
