"""The pilot report reads files only; no model call and no GPU."""

from __future__ import annotations

import json

from experiments.harness_pilot import summarize


def test_a_pilot_summary_counts_placeholders_shadow_and_unchanged(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "feedback.jsonl").write_text(
        json.dumps({"format": "agent_feedback", "v": 1, "seq": 1, "label": "r1", "outcome": "unchanged", "items": [], "placeholder_items": 2}) + "\n"
    )
    (run / "harness_proposals.jsonl").write_text(
        json.dumps({"status": "valid", "proposal_key": "k", "proposal": {"surface": "tip"}}) + "\n"
    )
    (run / "harness_decisions.jsonl").write_text(json.dumps({"decision": "shadow", "proposal_key": "k"}) + "\n")
    (run / "harness_overlay.jsonl").write_text(json.dumps({"event": "overlay", "shadow": True, "proposal_key": "k"}) + "\n")
    (run / "result.json").write_text(json.dumps({"spent_usd": 1.25, "best_speedup": 1.1}))
    row = summarize(run)
    assert row["sessions"] == 1 and row["unchanged_rate"] == 1.0
    assert row["placeholder_items"] == 2
    assert row["proposal_surfaces"] == {"tip": 1}
    assert row["overlay"] == {"shadow": 1}
    assert row["spent_usd"] == 1.25
