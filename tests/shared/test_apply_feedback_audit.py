"""The offline audit write-back is explicit, idempotent in shape, and never merges premises."""

from __future__ import annotations

import json
from pathlib import Path

from experiments.apply_feedback_audit import actions, apply_actions


def _row(seq, digest, *, section, verdict, topic, measured=0):
    return {
        "id": f"run#{seq}:{digest}", "run_id": "run", "seq": seq, "digest": digest,
        "section": section, "verdict": verdict, "topic": topic,
        "measured_valid": measured,
    }


def test_statuses_cover_all_verdicts_and_merge_only_system_suggestions(tmp_path):
    audit = {"rows": [
        _row(1, "a", section="被否决的前提", verdict="valid_measured", topic="algorithm", measured=1),
        _row(2, "b", section="被否决的前提", verdict="open_unverified", topic="algorithm"),
        _row(3, "c", section="需要帮助", verdict="drop_empty", topic="other"),
        _row(4, "d", section="对系统的建议", verdict="fixed_in_code", topic="self_check_dir"),
        _row(5, "e", section="对系统的建议", verdict="actionable_open", topic="bench2"),
        _row(6, "f", section="对系统的建议", verdict="actionable_open", topic="bench2"),
        _row(7, "g", section="对系统的建议", verdict="policy_decision", topic="budget"),
    ]}
    rows = actions(audit)
    by = {row["digest"]: row for row in rows}
    assert by["a"]["body"]["status"] == "adopted"
    assert by["b"]["body"]["status"] == "info"
    assert by["c"]["body"]["status"] == "obsolete"
    assert by["d"]["body"]["status"] == "done" and by["d"]["body"]["change_ref"].startswith("commit:")
    assert by["e"]["body"]["status"] == by["f"]["body"]["status"] == "pending"
    merged = {row["digest"]: row for row in actions(audit, merge_topic=True)}
    assert merged["e"]["body"]["status"] == "pending"
    assert merged["f"]["body"]["status"] == "obsolete"
    assert "duplicate of run#5:e" in merged["f"]["body"]["note"]
    # A premise is never merged by topic.
    assert merged["a"]["body"]["status"] == "adopted"
    assert merged["b"]["body"]["status"] == "info"


def test_apply_writes_review_files_and_a_missing_run_is_an_error(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "feedback.jsonl").write_text("{}\n")
    rows = actions({"rows": [_row(1, "a", section="被否决的前提", verdict="valid_measured", topic="algorithm", measured=1)]})
    result = apply_actions(tmp_path, rows, apply=True)
    assert result["written"] == 1 and result["errors"] == 0
    doc = json.loads((run / "feedback_review.json").read_text())
    assert doc["claims"]["1:a"]["status"] == "adopted"
    assert doc["claims"]["1:a"]["evidence"] == "1 measured lines"
    bad = _row(9, "z", section="需要帮助", verdict="drop_empty", topic="other")
    bad.update({"id": "missing#9:z", "run_id": "missing"})
    missing = actions({"rows": [bad]})
    assert apply_actions(tmp_path, missing, apply=True)["errors"] == 1
