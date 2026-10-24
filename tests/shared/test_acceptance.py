"""Proposal-specific evidence and acceptance-to-implementation contracts."""

from __future__ import annotations

import json
import time

import pytest

from taic import acceptance, code_agent, harness


def proposal(kind="tool"):
    evidence = {"probe": "", "measurement": "", "reproduction": "The report omits the needed field.", "expected": "A host-written result contains that field.", "acceptance_checks": ["test result contains the field"], "allowed_paths": ["tcenv/report.py"], "effect_metric_id": "first_eligible_cost_usd"}
    return {"proposal_key": "0123456789abcdef", "run_id": "run", "seq": 1, "claim_digest": "d", "proposal": {"surface": "tip", "kind": kind, "scope": {"target_id": "apple-m4-max", "task": "conv2d", "dtype": "fp16"}, "action": "Add a diagnostic tool.", "evidence": evidence, "rationale": "Make the limitation measurable."}}


def test_a_tool_tip_needs_reproduction_and_expected_result_not_a_probe(tmp_path):
    doc = proposal()["proposal"]
    doc["scope"] = {"target_id": "apple-m4-max", "task": "conv2d", "dtype": "fp16"}
    normalized = harness.validate_proposal(doc, target_id="apple-m4-max", task="conv2d", dtype="fp16", work=tmp_path, started=time.time())
    assert normalized["kind"] == "tool" and not normalized["probe_path"]
    doc["evidence"] = {**doc["evidence"], "reproduction": ""}
    with pytest.raises(harness.ProposalError, match="reproduction"):
        harness.validate_proposal(doc, target_id="apple-m4-max", task="conv2d", dtype="fp16", work=tmp_path, started=time.time())


def test_a_premise_still_needs_a_probe_and_measurement(tmp_path):
    doc = {"surface": "premise", "scope": {"target_id": "apple-m4-max", "task": "conv2d", "dtype": "fp16"}, "action": "先验证再选择。", "evidence": {"reproduction": "a", "expected": "b"}}
    with pytest.raises(harness.ProposalError, match="probe"):
        harness.validate_proposal(doc, target_id="apple-m4-max", task="conv2d", dtype="fp16", work=tmp_path, started=time.time())


def test_a_tip_cannot_change_evaluation_or_budget(tmp_path):
    doc = proposal()["proposal"]
    doc["evidence"]["expected"] = "Change correctness tolerance so the candidate passes."
    with pytest.raises(harness.ProposalError, match="evaluation"):
        harness.validate_proposal(doc, target_id="apple-m4-max", task="conv2d", dtype="fp16", work=tmp_path, started=time.time())


def test_review_recommended_test_reaches_the_implementation_prompt():
    review = {"useful": True, "confidence": 0.8, "risk": "wrong metric", "recommended_test": "compare the paired metric", "review_sha256": "a" * 64}
    spec = acceptance.make_spec(proposal(), review)
    assert acceptance.read_spec(spec) == spec
    prompt = code_agent.build_prompt(proposal()["proposal"], run_id="run", acceptance_spec=spec)
    assert "compare the paired metric" in prompt and "acceptance" in prompt
    report = {"acceptance": [{"check": "test result contains the field", "passed": True}, {"check": "compare the paired metric", "passed": True}]}
    implementation = acceptance.implementation_report(report, spec, ["tcenv/report.py"], "b" * 64)
    assert implementation["accepted"] is True
    implementation = acceptance.implementation_report(report, spec, ["tcenv/evaluate.py"], "b" * 64)
    assert implementation["accepted"] is False and implementation["outside_paths"] == ["tcenv/evaluate.py"]


def test_acceptance_digest_catches_changed_checks():
    spec = acceptance.make_spec(proposal())
    changed = {**spec, "acceptance_checks": ["a much weaker test"]}
    with pytest.raises(acceptance.AcceptanceError, match="digest"):
        acceptance.read_spec(changed)


def test_a_board_keeps_review_implementation_and_effect_separate(tmp_path):
    row = {"schema_version": 1, "event": "proposal", "status": "valid", **proposal()}
    (tmp_path / "harness_proposals.jsonl").write_text(json.dumps(row) + "\n")
    (tmp_path / "harness_reviews.jsonl").write_text(json.dumps({"event": "review", "proposal_key": row["proposal_key"], "useful": True, "cost_usd": 0.1}) + "\n")
    (tmp_path / "harness_decisions.jsonl").write_text(json.dumps({"event": "code_promotion", "proposal_key": row["proposal_key"], "status": "local"}) + "\n")
    milestones = harness.board(tmp_path)["acceptance_specs"][0]["milestones"]
    assert milestones == {"reviewed": True, "implemented": False, "experiment_validated": False, "local": True, "committed": False}


def test_verify_code_records_host_acceptance_before_apply(tmp_path, monkeypatch):
    key = "0123456789abcdef"
    row = proposal()
    spec = acceptance.make_spec(row, {"useful": True, "cost_usd": 0.1})
    spec["host_checks"] = [{"check_id": "reporting.offline_contract", "description": "offline report check"}]
    spec["spec_sha256"] = acceptance.digest({key: value for key, value in spec.items() if key != "spec_sha256"})
    (tmp_path / "harness_proposals.jsonl").write_text(json.dumps({"status": "valid", **row}) + "\n")
    (tmp_path / "harness_reviews.jsonl").write_text(json.dumps({"event": "review", "proposal_key": key, "useful": True, "cost_usd": 0.1}) + "\n")
    patch = "diff --git a/tcenv/report.py b/tcenv/report.py\nnew file mode 100644\n--- /dev/null\n+++ b/tcenv/report.py\n@@ -0,0 +1 @@\n+x\n"
    code_agent.write_patch(tmp_path, key, {"patch": patch, "report": {"acceptance": [{"check": "test result contains the field", "passed": True}]}}, acceptance_spec=spec)
    monkeypatch.setattr(code_agent, "_test_patch_in_worktree", lambda *args, **kwargs: [{"check_id": item["check_id"], "passed": True, "exit_code": 0, "output_excerpt": "offline fake", "reason": ""} for item in kwargs["check_items"]])
    monkeypatch.setattr("taic.acceptance_checks.run_checks", lambda items, **kwargs: [{"check_id": item["check_id"], "passed": True, "exit_code": 0, "output_excerpt": "offline fake", "reason": ""} for item in items])
    verified = code_agent.verify_code_patch(tmp_path, key, execute=True, approved_by="alice")
    assert verified["host_verified"] is True
    with pytest.raises(code_agent.CodeAgentError, match="no effect validation"):
        code_agent.apply_code_patch(tmp_path, proposal_key=key, require_effect_validation=True)


def test_static_verification_does_not_execute_patch_code(tmp_path, monkeypatch):
    """@brief 默认静态验收只检查证据，不运行补丁中的 import/pytest。"""
    key = "0123456789abcdef"
    row = proposal()
    spec = acceptance.make_spec(row, {"useful": True, "cost_usd": 0.1})
    spec["host_checks"] = [{"check_id": "reporting.offline_contract", "description": "offline report check"}]
    spec["spec_sha256"] = acceptance.digest({key: value for key, value in spec.items() if key != "spec_sha256"})
    (tmp_path / "harness_proposals.jsonl").write_text(json.dumps({"status": "valid", **row}) + "\n")
    (tmp_path / "harness_reviews.jsonl").write_text(json.dumps({"event": "review", "proposal_key": key, "useful": True, "cost_usd": 0.1}) + "\n")
    patch = "diff --git a/tcenv/report.py b/tcenv/report.py\nnew file mode 100644\n--- /dev/null\n+++ b/tcenv/report.py\n@@ -0,0 +1 @@\n+x\n"
    code_agent.write_patch(tmp_path, key, {"patch": patch, "report": {"acceptance": [{"check": "test result contains the field", "passed": True}]}}, acceptance_spec=spec)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("static verification executed patch code")

    monkeypatch.setattr(code_agent, "_test_patch_in_worktree", forbidden)
    verified = code_agent.verify_code_patch(tmp_path, key)
    assert verified["static_verified"] is True and verified["host_verified"] is False


def test_legacy_text_requires_named_approval_and_real_registered_checks(tmp_path, monkeypatch):
    key = "0123456789abcdef"
    row = proposal()
    spec = acceptance.make_spec(row, {"useful": True, "cost_usd": 0.1})
    (tmp_path / "harness_proposals.jsonl").write_text(json.dumps({"status": "valid", **row}) + "\n")
    (tmp_path / "harness_reviews.jsonl").write_text(json.dumps({"event": "review", "proposal_key": key, "useful": True, "cost_usd": 0.1}) + "\n")
    patch = "diff --git a/tcenv/report.py b/tcenv/report.py\nnew file mode 100644\n--- /dev/null\n+++ b/tcenv/report.py\n@@ -0,0 +1 @@\n+x\n"
    code_agent.write_patch(tmp_path, key, {"patch": patch, "acceptance_spec": spec, "report": {}})
    with pytest.raises(code_agent.CodeAgentError, match="legacy acceptance"):
        code_agent.verify_code_patch(tmp_path, key, legacy_text="old human check", check_ids=["reporting.offline_contract"])
    monkeypatch.setattr(code_agent, "_test_patch_in_worktree", lambda *args, **kwargs: [{"check_id": item["check_id"], "passed": True, "exit_code": 0, "output_excerpt": "ran", "reason": ""} for item in kwargs["check_items"]])
    verified = code_agent.verify_code_patch(tmp_path, key, approved_by="alice", legacy_text="old human check", check_ids=["reporting.offline_contract"], execute=True)
    assert verified["legacy_upgrade"]["approved_by"] == "alice"
    assert verified["legacy_upgrade"]["original_text"] == "old human check"
    assert verified["host_checks"][0]["output_excerpt"] == "ran"
    assert verified["risk_tier"] == "paired_required"


def test_risk_is_recomputed_from_the_patch_not_the_model_label(tmp_path):
    spec = acceptance.make_spec({"proposal_key": "0123456789abcdef", "proposal": {"surface": "tip", "kind": "tool", "action": "offline tool", "scope": {}, "evidence": {"reproduction": "r", "expected": "e", "allowed_paths": ["tcenv/report.py"], "risk_tier": "offline_only", "host_checks": [{"check_id": "reporting.offline_contract"}]}}})
    patch = "diff --git a/tcenv/evaluate.py b/tcenv/evaluate.py\n--- a/tcenv/evaluate.py\n+++ b/tcenv/evaluate.py\n@@ -0,0 +1 @@\n+change\n"
    assert acceptance.classify_change(spec, patch)["risk_tier"] == "human_gated"
