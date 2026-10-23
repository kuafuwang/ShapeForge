"""Free-text suggestions stay human-reviewed until a person confirms a structured conversion."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from taic import harness


def _run(tmp_path: Path) -> tuple[Path, str]:
    run = tmp_path / "run"
    work = run / "agent_work" / "init"
    work.mkdir(parents=True)
    digest = "fb-digest"
    record = {
        "seq": 1, "t": time.time() - 1, "task": "flash_attention", "dtype": "fp16",
        "label": "init", "committee": "first", "provider": "deepseek", "model": "deepseek-flash",
        "work_dir": "agent_work/init", "run_id": "apple-m4-max/flash-attention-pilot",
        "items": [{"section": "对系统的建议", "digest": digest, "text": "成绩单加印 GPU 时间戳加速比。"}],
    }
    (run / "state.json").write_text(json.dumps({"task": "flash_attention", "phase": "done"}) + "\n")
    (run / "feedback.jsonl").write_text(json.dumps(record, ensure_ascii=False) + "\n")
    return run, digest


def test_a_free_text_suggestion_is_visible_but_not_a_proposal_until_confirmed(tmp_path):
    run, digest = _run(tmp_path)
    suggestions = harness.free_text_suggestions(run)
    assert suggestions and suggestions[0]["digest"] == digest
    assert suggestions[0]["converted_key"] == ""
    with pytest.raises(harness.ProposalError, match="human confirmation"):
        harness.convert_feedback(
            run, seq=1, digest=digest, target_id="apple-m4-max", task="flash_attention",
            dtype="fp16", surface="skill", action="加一行 GPU 时间戳加速比", by="alice",
        )
    assert not (run / "harness_proposals.jsonl").exists()


def test_a_person_can_convert_free_text_to_a_p2_proposal_with_provenance(tmp_path):
    run, digest = _run(tmp_path)
    row = harness.convert_feedback(
        run, seq=1, digest=digest, target_id="apple-m4-max", task="flash_attention",
        dtype="fp16", surface="skill", action="在成绩单加印按 GPU 时间戳算的加速比",
        rationale="避免墙钟 launch 开销误导", by="alice", confirm=True, mode="manual",
    )
    assert row["origin"] == "conversion" and row["status"] == "valid"
    assert row["conversion"]["confirmed_by"] == "alice"
    assert row["conversion"]["source_text"].startswith("成绩单")
    assert row["claim_digest"] == digest
    doc = harness.board(run)
    assert doc["suggestions"][0]["converted_key"] == row["proposal_key"]
    assert doc["proposals"][0]["proposal"]["surface"] == "skill"
    with pytest.raises(harness.ProposalError):
        harness.convert_feedback(
            run, seq=1, digest=digest, target_id="apple-m4-max", task="flash_attention",
            dtype="fp16", surface="skill", action="重复", by="alice", confirm=True,
        )


def test_a_converted_tip_still_needs_its_own_probe(tmp_path):
    run, digest = _run(tmp_path)
    with pytest.raises(harness.ProposalError, match="probe"):
        harness.convert_feedback(
            run, seq=1, digest=digest, target_id="apple-m4-max", task="flash_attention",
            dtype="fp16", surface="tip", action="先读 device.md 再选 tile", by="alice", confirm=True,
        )


def test_structure_already_in_feedback_is_not_offered_for_conversion(tmp_path):
    run, digest = _run(tmp_path)
    row = json.loads((run / "feedback.jsonl").read_text())
    row["items"][0]["text"] += '\n```harness-proposal\n{"surface":"tip"}\n```'
    (run / "feedback.jsonl").write_text(json.dumps(row, ensure_ascii=False) + "\n")
    assert harness.free_text_suggestions(run) == []


def test_the_dashboard_conversion_route_requires_confirmation_and_writes_locally(tmp_path, monkeypatch):
    from observe import actions

    runs = tmp_path / "runs"
    runs.mkdir()
    monkeypatch.setenv("AI_LOWERLING_RUNS_DIR", str(runs))
    run, digest = _run(runs)
    body = {
        "seq": 1, "digest": digest, "surface": "skill", "action": "在成绩单加印 GPU 时间戳加速比",
        "task": "flash_attention", "dtype": "fp16", "by": "alice", "confirm": True, "mode": "manual",
    }
    with pytest.raises(actions.Refused, match="confirmation"):
        actions.convert_harness("run", 1, digest, {**body, "confirm": False})
    row = actions.convert_harness("run", 1, digest, body)
    assert row["origin"] == "conversion" and row["conversion"]["confirmed_by"] == "alice"


def test_a_converted_proposal_can_be_sent_to_one_explicit_review(tmp_path, monkeypatch):
    from observe import actions
    from taic import review as review_mod

    runs = tmp_path / "runs"
    runs.mkdir()
    monkeypatch.setenv("AI_LOWERLING_RUNS_DIR", str(runs))
    run, digest = _run(runs)
    actions.convert_harness("run", 1, digest, {
        "seq": 1, "digest": digest, "surface": "skill", "action": "在成绩单加印 GPU 时间戳加速比",
        "task": "flash_attention", "dtype": "fp16", "by": "alice", "confirm": True, "mode": "manual",
    })
    calls = []

    def fake_review(*args, **kwargs):
        calls.append(kwargs)
        return {
            "useful": True, "confidence": 0.8, "reason": "concrete", "evidence": "probe",
            "risk": "", "recommended_test": "try one run", "provider": kwargs["provider"],
            "model": kwargs["model"], "effort": kwargs.get("effort") or "", "budget_usd": 0.1,
            "expected_usd": 0.02, "cost_usd": 0.01, "elapsed_s": 1.0, "tokens": {},
        }

    monkeypatch.setattr(review_mod, "run_review", fake_review)
    result = actions.review_harness("run", json.loads((run / "harness_proposals.jsonl").read_text())["proposal_key"], {})
    assert result["review"]["useful"] is True and calls
    assert (run / "harness_reviews.jsonl").is_file()
