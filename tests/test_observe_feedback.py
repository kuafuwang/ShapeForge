"""Feedback index, per-claim review, and escaped dashboard text. No GPU and no model calls."""

from __future__ import annotations

import json
import subprocess
import threading
from pathlib import Path

import pytest

from observe.actions import Refused, review_feedback
from observe.feedback import entries, rejection, summary_from_index
from taic.agent_feedback import parse_feedback

_TEXT = """kind: tool

# 需要帮助
- 缺一把探针

# 卡住的原因
- 时间不够

# 被否决的前提
- [未验证] 矩阵单元只能吃 fp16

# 给下一棒的避坑提示
- float8 是保留字
<img onerror=alert(1)>
"""


def _parsed():
    return parse_feedback(_TEXT)


def _record(parsed: dict, *, seq: int = 1, t: float = 10, label: str = "r1.p0") -> dict:
    return {
        "format": "agent_feedback", "v": 1, "seq": seq, "t": t, "task": "conv2d", "dtype": "fp16", "size": "paper",
        "label": label, "round": 1, "committee": "first", "provider": "deepseek", "model": "deepseek-flash",
        "effort": "high", "outcome": "evaluated", "exit": "normal", "best_speedup": 1.01, "kinds": parsed["kinds"],
        "work_dir": f"agent_work/{label}", "sections": parsed["sections"], "items": parsed["items"],
        "unverified_premises": parsed["unverified_premises"], "raw": parsed["raw"],
    }


def _run(root: Path, name: str, record: dict, *, started_at: float) -> Path:
    directory = root / name
    directory.mkdir(parents=True)
    (directory / "state.json").write_text(json.dumps({
        "task": "conv2d", "dtype": "fp16", "phase": "done", "started_at": started_at, "updated_at": started_at,
    }))
    (directory / "feedback.jsonl").write_text(json.dumps(record, ensure_ascii=False) + "\n")
    folder = directory / "candidates" / f"000_{record['label']}"
    folder.mkdir(parents=True)
    (folder / "report.json").write_text(json.dumps({
        "status": "ok", "candidate_us": 770, "benchmark": {"candidate": {"median_us": 770, "cv": 0.01}},
        "roofline": {"lower_bound_us": 414, "bound_kind": "reference_algorithm"},
    }))
    return directory


def _open(tmp_path, monkeypatch):
    runs = tmp_path / "runs"
    runs.mkdir()
    monkeypatch.setenv("AI_LOWERLING_RUNS_DIR", str(runs))
    monkeypatch.setenv("OBSERVE_DB", str(tmp_path / "observe.sqlite"))
    monkeypatch.setenv("AI_LOWERLING_BOOKS", str(tmp_path / "none.sqlite"))
    import observe.index as index_mod

    index_mod._SINGLE = None
    return runs, index_mod.get_index()


def _digest(section: str) -> str:
    for item in _parsed()["items"]:
        if item["section"] == section:
            return item["digest"]
    raise AssertionError(section)


def test_pending_counts_system_and_premise_claims_and_review_changes_them(tmp_path, monkeypatch):
    runs, index = _open(tmp_path, monkeypatch)
    parsed = _parsed()
    _run(runs, "conv", _record(parsed), started_at=50)
    index.sync_known()
    summary = summary_from_index(index.conn)
    assert summary["pending_total"] == 2
    kinds = summary_from_index(index.conn, kind="tool")
    assert len(kinds["items"]) == 4
    assert summary_from_index(index.conn, kind="prompt")["items"] == []
    shown = entries(runs / "conv")
    assert shown["records"][0]["report"]["source"] == "candidate"
    assert shown["records"][0]["report"]["candidate_us"] == 770
    assert "<img" in shown["records"][0]["raw"]
    premise = _digest("被否决的前提")
    help_item = _digest("需要帮助")
    with pytest.raises(Refused, match="evidence"):
        review_feedback("conv", 1, help_item, {"status": "adopted", "verification": "confirmed", "evidence_source": "", "evidence": ""})
    with pytest.raises(Refused, match="change_ref"):
        review_feedback("conv", 1, help_item, {"status": "done", "verification": "confirmed", "change_ref": ""})
    with pytest.raises(Refused, match="note"):
        review_feedback("conv", 1, premise, {"status": "rejected", "verification": "refuted", "note": ""})
    with pytest.raises(Refused, match="unchecked"):
        review_feedback("conv", 1, premise, {"status": "pending", "verification": "confirmed"})
    reviewed = review_feedback("conv", 1, premise, {"status": "rejected", "verification": "refuted", "note": "探针测的是另一件事"})
    assert reviewed["status"] == "rejected"
    assert summary_from_index(index.conn)["pending_total"] == 1
    again = review_feedback("conv", 1, premise, {"status": "rejected", "verification": "confirmed", "note": "属实但不改"})
    assert len(again["history"]) == 2
    index.drop_archived(["conv"], [])
    assert summary_from_index(index.conn)["pending_total"] == 0


def test_two_reviews_of_one_run_both_land(tmp_path, monkeypatch):
    runs, index = _open(tmp_path, monkeypatch)
    parsed = _parsed()
    _run(runs, "conv", _record(parsed), started_at=50)
    index.sync_known()
    errors = []

    def one(section, body):
        try:
            review_feedback("conv", 1, _digest(section), body)
        except Exception as exc:  # noqa: BLE001 - the test records either failure
            errors.append(exc)

    first = threading.Thread(target=one, args=("需要帮助", {"status": "rejected", "verification": "refuted", "note": "已经有了"}))
    second = threading.Thread(target=one, args=("被否决的前提", {"status": "rejected", "verification": "refuted", "note": "未验证"}))
    first.start()
    second.start()
    first.join()
    second.join()
    assert errors == []
    saved = json.loads((runs / "conv" / "feedback_review.json").read_text())
    assert len(saved["claims"]) == 2


def test_a_claim_restated_by_a_later_run_is_flagged(tmp_path, monkeypatch):
    runs, index = _open(tmp_path, monkeypatch)
    parsed = _parsed()
    digest = _digest("被否决的前提")
    _run(runs, "first", _record(parsed, t=80), started_at=50)
    _run(runs, "later", _record(parsed, t=160), started_at=150)
    (runs / "first" / "feedback_review.json").write_text(json.dumps({
        "v": 1,
        "claims": {f"1:{digest}": {"status": "done", "verification": "confirmed", "change_ref": "abc123", "t": 100, "reviewed_at": 100, "topic": "fp32-mma", "history": []}},
    }))
    index.sync_known()
    summary = summary_from_index(index.conn)
    assert any(item["run_id"] == "later" for item in summary["recurrence"])
    assert all(item["run_id"] != "first" for item in summary["recurrence"])


def test_the_dashboard_escapes_feedback_text():
    page = Path("observe/static/index.html").read_text()
    assert "esc(claim.text)" in page and "esc(record.raw)" in page
    start = page.index("function esc(value)")
    end = page.index("\n}", start) + 2
    script = page[start:end] + "\nprocess.stdout.write(esc('<img onerror=\"alert(1)\">'))"
    rendered = subprocess.run(["node", "-e", script], check=True, capture_output=True, text=True).stdout
    assert rendered == "&lt;img onerror=&quot;alert(1)&quot;&gt;"
    assert rejection({"status": "adopted", "verification": "confirmed", "evidence_source": "probe", "evidence": "15.09"}) is None
