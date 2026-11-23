"""Omitted NOTES content is analyzed only when reconstruction is byte-exact."""

from __future__ import annotations

import json
from pathlib import Path

from taic.omission_content_audit import SCHEMA, audit


def _case(run: Path, *, notes: str, task: str, source_bytes_delta: int = 0) -> None:
    run.mkdir(parents=True, exist_ok=True)
    (run / "result.json").write_text(json.dumps({"trace": [
        {"idx": 1, "label": "r1.p0", "improved": True},
        {"idx": 2, "label": "r2.p0", "improved": False},
    ]}), encoding="utf-8")
    folder = run / "candidates" / "001_r1.p0"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "report.json").write_text(json.dumps({"candidate_us": 1.0, "status": "ok"}), encoding="utf-8")
    (folder / "NOTES.md").write_text(notes, encoding="utf-8")
    source = f"外层实测：1 µs，状态 ok，换了种子\n{notes}".strip()
    (run / "sessions" / "r2.p0").mkdir(parents=True, exist_ok=True)
    (run / "sessions" / "r2.p0" / "task_delivery.json").write_text(json.dumps({
        "blocks": [{"name": "上一轮 NOTES", "source_bytes": len(source.encode()) + source_bytes_delta,
                    "expected_bytes": 100, "delivered_bytes": 0, "omitted_bytes": 100}],
        "situation": {"layers": []},
    }), encoding="utf-8")
    (run / "agent_work" / "r2.p0").mkdir(parents=True, exist_ok=True)
    (run / "agent_work" / "r2.p0" / "TASK.md").write_text(task, encoding="utf-8")


def test_repeated_notes_are_measured_only_after_exact_reconstruction(tmp_path):
    runs = tmp_path / "runs"
    notes = "上一轮有一个独特结论：Winograd 需要直接回退。\n其他说明。"
    _case(runs / "target" / "run-a", notes=notes, task=f"# 其他块\n{notes}\n")
    document = audit(runs)
    assert document["schema"] == SCHEMA
    assert document["counts"]["reconstructed_exact"] == 1
    assert document["counts"].get("reconstruction_mismatch", 0) == 0
    row = document["sessions"][0]
    assert row["literal_shared_ratio"] > 0.8
    assert row["classification"] == "mostly_duplicate"
    assert document["causal_claim_allowed"] is False


def test_byte_mismatch_is_not_analyzed_as_if_it_were_the_original(tmp_path):
    runs = tmp_path / "runs"
    _case(runs / "run-mismatch", notes="unique", task="different", source_bytes_delta=1)
    document = audit(runs)
    assert document["counts"]["reconstruction_mismatch"] == 1
    assert document["counts"].get("reconstructed_exact", 0) == 0
    assert document["sessions"] == []


def test_real_omitted_notes_reconstruction_is_read_only_and_bounded():
    runs = Path(__file__).resolve().parents[2] / "runs"
    if not runs.is_dir():
        pytest.skip("requires the repository's real runs/ corpus")
    files = list(runs.rglob("task_delivery.json"))
    before = {path: path.stat().st_mtime_ns for path in files}
    document = audit(runs)
    after = {path: path.stat().st_mtime_ns for path in files}
    assert after == before
    assert document["counts"]["reconstructed_exact"] > 0
    assert document["counts"].get("reconstruction_mismatch", 0) == 0
    assert document["counts"]["omitted_notes_receipts"] >= document["counts"]["reconstructed_exact"]
    assert document["counts"].get("mostly_not_duplicated", 0) + document["counts"].get("mixed", 0) + document["counts"].get("mostly_duplicate", 0) == document["counts"]["reconstructed_exact"]
    for row in document["sessions"]:
        assert 0.0 <= row["literal_shared_ratio"] <= 1.0
        assert row["source_bytes"] >= row["omitted_bytes"] >= 0
    assert document["causal_claim_allowed"] is False
