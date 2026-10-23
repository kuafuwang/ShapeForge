"""The review ledger is append-only, hash-chained and stores only evidence hashes."""

from __future__ import annotations

import json

from taic import review_ledger


def _event(seq: int, status: str) -> dict:
    return {
        "seq": seq,
        "digest": f"d{seq}",
        "t": 100.0 + seq,
        "status": status,
        "verification": "unchecked",
        "evidence_source": "other",
        "evidence": "some evidence",
        "note": "why",
        "reviewer": "Reviewer <r@example.test>",
        "method": "agent",
        "batch_id": "batch-1",
        "git_commit": "abc",
        "git_dirty": True,
        "backend": "metal",
        "target_id": "apple-m4-max",
    }


def test_events_chain_and_verify(tmp_path):
    path = tmp_path / "review_events.jsonl"
    hashes = review_ledger.append_events([_event(1, "info"), _event(2, "rejected")], run_id="run-a", path=path)
    assert len(hashes) == 2 and len(set(hashes)) == 2
    result = review_ledger.verify(path)
    assert result == {"ok": True, "events": 2, "broken_at": None, "last_hash": hashes[-1]}
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert rows[0]["prev_hash"] == ""
    assert rows[1]["prev_hash"] == rows[0]["event_hash"]
    assert "why" not in rows[0] and "some evidence" not in rows[0]
    assert rows[0]["note_sha256"]


def test_a_tampered_row_breaks_verification(tmp_path):
    path = tmp_path / "review_events.jsonl"
    review_ledger.append_events([_event(1, "info")], run_id="run-a", path=path)
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0]["status"] = "adopted"
    path.write_text(json.dumps(rows[0], ensure_ascii=False) + "\n")
    result = review_ledger.verify(path)
    assert result["ok"] is False and result["broken_at"] == 1
