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
    proposal_key = json.loads((run / "harness_proposals.jsonl").read_text())["proposal_key"]
    with pytest.raises(actions.Refused, match="author"):
        actions.review_harness("run", proposal_key, {"provider": "deepseek", "model": "deepseek-flash"})
    result = actions.review_harness("run", proposal_key, {})
    assert result["review"]["useful"] is True and calls
    assert (run / "harness_reviews.jsonl").is_file()


def _rewrite_record(run: Path, **fields) -> dict:
    path = run / "feedback.jsonl"
    record = json.loads(path.read_text().splitlines()[0])
    record.update(fields)
    path.write_text(json.dumps(record, ensure_ascii=False) + "\n")
    return record


def _probe(run: Path, mtime: float, name: str = "probe.py") -> Path:
    import os

    path = run / "agent_work" / "init" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("print(1)\n", encoding="utf-8")
    os.utime(path, (mtime, mtime))
    return path


def test_conversion_uses_the_recorded_session_start_not_the_feedback_time(tmp_path):
    import time as _time

    run, digest = _run(tmp_path)
    now = _time.time()
    _rewrite_record(run, t=now, session_started=now - 600)
    _probe(run, now - 300)
    row = harness.convert_feedback(
        run, seq=1, digest=digest, target_id="apple-m4-max", task="flash_attention",
        dtype="fp16", surface="tip", action="先试这个 tile", probe="probe.py",
        measurement="1.2x", by="alice", confirm=True,
    )
    assert row["probe_time_verified"] is True
    assert row["conversion"]["source_session_started"] == now - 600


def test_an_old_record_marks_probe_time_unverifiable_instead_of_using_feedback_time(tmp_path):
    import time as _time

    run, digest = _run(tmp_path)
    now = _time.time()
    _rewrite_record(run, t=now)
    record = json.loads((run / "feedback.jsonl").read_text().splitlines()[0])
    record.pop("session_started", None)
    (run / "feedback.jsonl").write_text(json.dumps(record, ensure_ascii=False) + "\n")
    _probe(run, now - 300)
    row = harness.convert_feedback(
        run, seq=1, digest=digest, target_id="apple-m4-max", task="flash_attention",
        dtype="fp16", surface="tip", action="先试这个 tile", probe="probe.py",
        measurement="1.2x", by="alice", confirm=True,
    )
    assert row["probe_time_verified"] is False
    assert "session_started" in row["probe_time_reason"]


def test_a_probe_before_the_recorded_session_start_is_still_rejected(tmp_path):
    import time as _time

    run, digest = _run(tmp_path)
    now = _time.time()
    _rewrite_record(run, t=now, session_started=now - 100)
    _probe(run, now - 400)
    with pytest.raises(harness.ProposalError, match="predates this session"):
        harness.convert_feedback(
            run, seq=1, digest=digest, target_id="apple-m4-max", task="flash_attention",
            dtype="fp16", surface="tip", action="先试这个 tile", probe="probe.py",
            measurement="1.2x", by="alice", confirm=True,
        )


def test_a_conversion_is_not_model_authored_and_its_source_model_cannot_review_it(tmp_path):
    import time as _time

    from taic import trust

    run, digest = _run(tmp_path)
    now = _time.time()
    _rewrite_record(run, t=now, session_started=now - 600)
    _probe(run, now - 300)
    row = harness.convert_feedback(
        run, seq=1, digest=digest, target_id="apple-m4-max", task="flash_attention",
        dtype="fp16", surface="tip", action="先试这个 tile", probe="probe.py",
        measurement="1.2x", by="alice", confirm=True,
    )
    assert row["provider"] == "" and row["model"] == ""
    assert row["source_author"]["provider"] == "deepseek"
    assert row["structured_by"]["confirmed_by"] == "alice"
    assert harness.reviewer_conflict(row, "deepseek", "deepseek-flash") is True
    assert harness.reviewer_conflict(row, "moonshot", "kimi-k3") is False

    root = tmp_path / "targets"
    trust.grant("apple-m4-max", "flash_attention", "deepseek", "deepseek-flash", scopes=["overlay:tip"], root=root)
    first = harness.settle(
        run, target_id="apple-m4-max", task="flash_attention",
        protocol_version=trust.current_protocol_version(), enabled=True, shadow=False, trust_root=root,
    )
    assert first["applied"] == []

    reviews = run / "harness_reviews.jsonl"
    reviews.write_text(json.dumps({
        "event": "review", "proposal_key": row["proposal_key"], "source_seq": 1, "useful": True,
        "reviewer": {"provider": "deepseek", "model": "deepseek-flash", "effort": ""},
    }) + "\n")
    same = harness.settle(
        run, target_id="apple-m4-max", task="flash_attention",
        protocol_version=trust.current_protocol_version(), enabled=True, shadow=False, trust_root=root,
    )
    assert same["applied"] == []

    trust.grant("apple-m4-max", "flash_attention", "moonshot", "kimi-k3", scopes=["overlay:tip"], root=root)
    reviews.write_text(json.dumps({
        "event": "review", "proposal_key": row["proposal_key"], "source_seq": 1, "useful": True,
        "reviewer": {"provider": "moonshot", "model": "kimi-k3", "effort": ""},
    }) + "\n")
    independent = harness.settle(
        run, target_id="apple-m4-max", task="flash_attention",
        protocol_version=trust.current_protocol_version(), enabled=True, shadow=False, trust_root=root,
    )
    assert [event["proposal_key"] for event in independent["applied"]] == [row["proposal_key"]]


def test_conversion_refuses_a_work_dir_outside_the_run(tmp_path):
    run, digest = _run(tmp_path)
    _rewrite_record(run, work_dir="../outside")
    with pytest.raises(harness.ProposalError, match="escapes"):
        harness.convert_feedback(
            run, seq=1, digest=digest, target_id="apple-m4-max", task="flash_attention",
            dtype="fp16", surface="skill", action="在成绩单加印 GPU 时间戳加速比",
            by="alice", confirm=True,
        )


def test_an_unverified_old_tip_cannot_become_reviewed_active(tmp_path):
    from taic import trust

    run, digest = _run(tmp_path)
    _probe(run, time.time() - 300)
    proposal = harness.convert_feedback(
        run, seq=1, digest=digest, target_id="apple-m4-max", task="flash_attention",
        dtype="fp16", surface="tip", action="先试这个 tile", probe="probe.py",
        by="alice", confirm=True,
    )
    assert proposal["probe_time_verified"] is False
    (run / "harness_reviews.jsonl").write_text(json.dumps({
        "event": "review", "proposal_key": proposal["proposal_key"], "source_seq": 1,
        "useful": True, "reviewer": {"provider": "moonshot", "model": "kimi-k3", "effort": ""},
    }) + "\n")
    root = tmp_path / "targets"
    trust.grant("apple-m4-max", "flash_attention", "moonshot", "kimi-k3", scopes=["overlay:tip"], root=root)
    result = harness.settle(
        run, target_id="apple-m4-max", task="flash_attention",
        protocol_version=trust.current_protocol_version(), enabled=True, shadow=False, trust_root=root,
    )
    assert result["applied"] == []


def test_a_reviewed_p2_proposal_never_becomes_a_tip_overlay(tmp_path):
    from taic import trust

    run, digest = _run(tmp_path)
    proposal = harness.convert_feedback(
        run, seq=1, digest=digest, target_id="apple-m4-max", task="flash_attention",
        dtype="fp16", surface="skill", action="在成绩单加印 GPU 时间戳加速比",
        by="alice", confirm=True,
    )
    (run / "harness_reviews.jsonl").write_text(json.dumps({
        "event": "review", "proposal_key": proposal["proposal_key"], "source_seq": 1,
        "useful": True, "reviewer": {"provider": "moonshot", "model": "kimi-k3", "effort": ""},
    }) + "\n")
    root = tmp_path / "targets"
    trust.grant("apple-m4-max", "flash_attention", "moonshot", "kimi-k3", scopes=["overlay:tip"], root=root)
    result = harness.settle(
        run, target_id="apple-m4-max", task="flash_attention",
        protocol_version=trust.current_protocol_version(), enabled=True, shadow=False, trust_root=root,
    )
    assert result["applied"] == []
    assert not (run / "harness_overlay.jsonl").exists()


def test_an_old_style_conversion_row_never_counts_toward_tip_quorum(tmp_path):
    from taic import trust

    run = tmp_path / "run"
    run.mkdir()
    action = "先读 device.md 再选 tile"
    base = {
        "schema_version": 1, "event": "proposal", "status": "valid", "proposal_key": "same-key",
        "target_id": "apple-m4-max", "task": "flash_attention", "dtype": "fp16",
        "proposal": {
            "surface": "tip", "scope": {"target_id": "apple-m4-max", "task": "flash_attention", "dtype": "fp16"},
            "action": action, "evidence": {"probe": "probe.py", "measurement": "1.2x"}, "rationale": "test",
        },
        "probe_time_verified": True, "trusted_record": False,
    }
    rows = [
        {**base, "seq": 1, "origin": "conversion", "provider": "deepseek", "model": "deepseek-flash",
         "committee": "first", "probe_sha256": "a" * 64},
        {**base, "seq": 2, "origin": "model", "provider": "moonshot", "model": "kimi-k3",
         "committee": "second", "probe_sha256": "b" * 64},
    ]
    (run / "harness_proposals.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n"
    )
    root = tmp_path / "targets"
    trust.grant("apple-m4-max", "flash_attention", "deepseek", "deepseek-flash", scopes=["overlay:tip"], root=root)
    trust.grant("apple-m4-max", "flash_attention", "moonshot", "kimi-k3", scopes=["overlay:tip"], root=root)
    result = harness.settle(
        run, target_id="apple-m4-max", task="flash_attention",
        protocol_version=trust.current_protocol_version(), enabled=True, shadow=False, trust_root=root,
    )
    assert result["applied"] == []
