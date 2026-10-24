"""The review ledger is append-only, hash-chained and stores only evidence hashes."""

from __future__ import annotations

import json

from taic import review_ledger


def _event(seq: int, status: str) -> dict:
    """@brief 构造一条内容固定的评审账本事件，供哈希链测试使用。

    @param seq 事件序号，决定时间戳与摘要值。
    @param status 评审状态。
    @return 待写入账本的事件字典。
    """
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
    """@brief 验证追加的事件按 prev_hash 串成哈希链、verify 返回一致结果，且正文只落哈希不落原文。

    @param tmp_path pytest 提供的临时目录。
    """
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
    """@brief 验证被篡改状态字段的行会被 verify 判定为断链，并定位到该行序号。

    @param tmp_path pytest 提供的临时目录。
    """
    path = tmp_path / "review_events.jsonl"
    review_ledger.append_events([_event(1, "info")], run_id="run-a", path=path)
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0]["status"] = "adopted"
    path.write_text(json.dumps(rows[0], ensure_ascii=False) + "\n")
    result = review_ledger.verify(path)
    assert result["ok"] is False and result["broken_at"] == 1
