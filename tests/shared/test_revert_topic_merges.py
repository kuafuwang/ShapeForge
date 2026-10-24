"""Only duplicate-topic obsolete entries are reverted; empty placeholders stay obsolete."""

from __future__ import annotations

import json
from pathlib import Path

from experiments.revert_topic_merges import find, revert


def _run(tmp_path: Path, name: str, claims: dict) -> Path:
    """@brief 写一个带反馈记录和 review claims 的 run 目录。

    @param tmp_path pytest 提供的临时目录。
    @param name run 目录名。
    @param claims feedback_review.json 里的 claims 字典。
    @return 创建出的 run 目录。
    """
    run = tmp_path / name
    run.mkdir()
    (run / "feedback.jsonl").write_text("{}\n")
    (run / "feedback_review.json").write_text(json.dumps({"v": 1, "claims": claims}))
    return run


def test_find_only_selects_duplicate_notes(tmp_path):
    """@brief find 只挑出 obsolete 且备注写明 duplicate 的条目，忽略其他 obsolete 和已完成条目。

    @param tmp_path pytest 提供的临时目录。
    """
    _run(tmp_path, "run", {
        "1:a": {"status": "obsolete", "note": "audit 2026-10-22: duplicate of run#2:b", "topic": "budget"},
        "2:b": {"status": "obsolete", "note": "audit 2026-10-22: empty placeholder", "topic": "other"},
        "3:c": {"status": "done", "note": "duplicate of x", "topic": "x"},
    })
    rows = find(tmp_path)
    assert len(rows) == 1 and rows[0]["seq"] == 1 and rows[0]["digest"] == "a"


def test_revert_writes_pending_and_keeps_history(tmp_path):
    """@brief revert 把重复条目改回 pending/unchecked，追加 history，并备份原文件。

    @param tmp_path pytest 提供的临时目录。
    """
    run = _run(tmp_path, "run", {
        "1:a": {"status": "obsolete", "note": "audit 2026-10-22: duplicate of run#2:b", "topic": "budget"},
    })
    rows = find(tmp_path)
    result = revert(rows, apply=True, backup_dir=tmp_path / "backup")
    assert result["written"] == 1 and result["errors"] == 0
    doc = json.loads((run / "feedback_review.json").read_text())
    claim = doc["claims"]["1:a"]
    assert claim["status"] == "pending" and claim["verification"] == "unchecked"
    assert "possible same topic" in claim["note"]
    assert len(claim["history"]) == 1
    assert (tmp_path / "backup" / "run" / "feedback_review.json").is_file()
