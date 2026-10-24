"""Adoption stages and costs are separate; no model or GPU."""

from __future__ import annotations

import json

from taic.adoption import adoption_summary, shortlist


def test_cost_unknown_is_not_zero_and_legacy_adoption_is_distinct(tmp_path):
    (tmp_path / "harness_proposals.jsonl").write_text(json.dumps({"status": "valid", "proposal_key": "k", "t": 1, "proposal": {"kind": "tool", "action": "add a tool"}}) + "\n")
    (tmp_path / "harness_reviews.jsonl").write_text(json.dumps({"event": "review", "proposal_key": "k", "useful": True, "cost_usd": 0.1, "t": 2}) + "\n")
    (tmp_path / "code_promotions").mkdir()
    (tmp_path / "code_promotions" / "k.json").write_text(json.dumps({"proposal_key": "k", "cost_usd": None}))
    (tmp_path / "harness_decisions.jsonl").write_text(json.dumps({"event": "promotion_commit", "proposal_key": "k", "t": 3}) + "\n")
    summary = adoption_summary(tmp_path)
    assert summary["proposals"] == 1 and summary["stages"]["committed"] == 1
    assert summary["cost_unknown_proposals"] == 1
    assert summary["items"][0]["review_cost_usd"] == 0.1
    assert summary["items"][0]["implementation_cost_usd"] is None
    assert summary["items"][0]["elapsed_s"] == 2
    assert len(shortlist([tmp_path])) == 1


def test_duplicate_proposals_do_not_inflate_the_shortlist(tmp_path):
    (tmp_path / "harness_proposals.jsonl").write_text(json.dumps({"status": "valid", "proposal_key": "k", "proposal": {"kind": "tool", "action": "same tool"}}) + "\n")
    other = tmp_path / "other"
    other.mkdir()
    (other / "harness_proposals.jsonl").write_text(json.dumps({"status": "valid", "proposal_key": "k2", "proposal": {"kind": "tool", "action": "same tool"}}) + "\n")
    rows = shortlist([tmp_path, other], limit=10)
    assert len(rows) == 1 and rows[0]["duplicate_count"] == 2


def test_experiment_result_cost_is_not_zero_or_counted_twice(tmp_path):
    """@brief 已执行实验按 experiment_id 去重记账，缺 validation 也不能报 0。"""
    (tmp_path / "harness_proposals.jsonl").write_text(json.dumps({"status": "valid", "proposal_key": "k", "t": 1, "proposal": {"kind": "tool", "action": "add a tool"}}) + "\n")
    (tmp_path / "harness_reviews.jsonl").write_text(json.dumps({"event": "review", "proposal_key": "k", "useful": True, "cost_usd": 0.1, "t": 2}) + "\n")
    decisions = [
        {"event": "experiment_result", "proposal_key": "k", "experiment_id": "exp1", "spent_usd": 2.0, "t": 3},
        {"event": "experiment_result", "proposal_key": "k", "experiment_id": "exp1", "spent_usd": 2.0, "t": 4},
    ]
    (tmp_path / "harness_decisions.jsonl").write_text("\n".join(json.dumps(row) for row in decisions) + "\n")
    item = adoption_summary(tmp_path)["items"][0]
    assert item["experiment_cost_usd"] == 2.0


def test_experiment_cost_and_stage_times_are_attributed_to_the_proposal(tmp_path):
    proposal = {"status": "valid", "proposal_key": "k", "t": 1, "proposal": {"kind": "tool", "action": "add a tool"}}
    review = {"event": "review", "proposal_key": "k", "useful": True, "cost_usd": 0.1, "t": 2}
    decisions = [
        {"event": "implementation", "proposal_key": "k", "accepted": True, "host_verified": True, "t": 3},
        {"event": "effect_validation", "proposal_key": "k", "validated": True, "approved_by": "bob", "spent_usd": 2.0, "t": 5},
        {"event": "promotion_commit", "proposal_key": "k", "t": 6},
    ]
    (tmp_path / "harness_proposals.jsonl").write_text(json.dumps(proposal) + "\n")
    (tmp_path / "harness_reviews.jsonl").write_text(json.dumps(review) + "\n")
    (tmp_path / "harness_decisions.jsonl").write_text("\n".join(json.dumps(row) for row in decisions) + "\n")
    item = adoption_summary(tmp_path)["items"][0]
    assert item["experiment_cost_usd"] == 2.0
    assert item["known_adoption_cost_usd"] == 2.1 and item["cost_unknown"] is False
    assert item["stage_times_s"] == {"reviewed": 1.0, "implemented": 2.0, "experiment_validated": 4.0, "local": None, "committed": 5.0}
