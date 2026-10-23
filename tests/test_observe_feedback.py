"""Feedback index, per-claim review, and escaped dashboard text. No GPU and no model calls."""

from __future__ import annotations

import json
import subprocess
import threading
from pathlib import Path

import pytest

from observe.actions import Refused, review_feedback
from observe.feedback import entries, rejection, review_board, review_feed, summary_from_index
from taic.agent_feedback import read_appended_records
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


def test_review_board_lists_every_run_with_the_claim_text(tmp_path, monkeypatch):
    runs, index = _open(tmp_path, monkeypatch)
    parsed = _parsed()
    _run(runs, "conv", _record(parsed, t=10), started_at=50)
    _run(runs, "fa", _record(parsed, t=20, label="r2.p0"), started_at=80)
    index.sync_known()
    board = review_board(index.conn, status="pending")
    assert board["pending_total"] == 4
    assert {item["run_id"] for item in board["items"]} == {"conv", "fa"}
    assert all(item["text"] for item in board["items"])
    feed = review_feed(index.conn, since=None)
    assert {item["target_id"] for item in feed["upserts"]} == {"apple-m4-max"}
    assert {item["task"] for item in feed["upserts"]} == {"conv2d"}
    page = Path(__file__).resolve().parents[1] / "observe" / "static" / "index.html"
    html = page.read_text(encoding="utf-8")
    assert "reviewHardware" in html
    assert "展开当前层" in html and "折叠当前层" in html and "reviewMenuActions" in html
    assert '<button type="button" id="feedback-badge"' in html
    assert 'window.open("/reviews"' in html
    assert 'params.set("since"' in html
    assert "patchReviewDom" in html


def test_a_grown_log_is_read_from_the_cursor_and_a_review_does_not_reread_it(tmp_path, monkeypatch):
    runs, index = _open(tmp_path, monkeypatch)
    parsed = _parsed()
    directory = _run(runs, "conv", _record(parsed, t=10), started_at=50)
    index.sync_known()
    path = directory / "feedback.jsonl"
    offset = index.conn.execute("SELECT feedback_bytes FROM runs WHERE run_id = 'conv'").fetchone()["feedback_bytes"]
    assert offset == path.stat().st_size
    rev = index.feedback_revision()
    seen = []
    real = read_appended_records

    def wrapped(log, start):
        seen.append(start)
        return real(log, start)

    def boom(*_args, **_kwargs):
        raise AssertionError("the whole feedback log was parsed")

    monkeypatch.setattr("observe.feedback.read_appended_records", wrapped)
    monkeypatch.setattr("observe.feedback.load_records", boom)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_record(parsed, seq=2, t=30, label="r2.p0"), ensure_ascii=False) + "\n")
    index.refresh_feedback("conv", directory)
    assert seen == [offset]
    later = review_feed(index.conn, since=rev)
    assert later["upserts"]
    assert {item["seq"] for item in later["upserts"]} == {2}
    seen.clear()
    review_feedback("conv", 1, _digest("被否决的前提"), {"status": "rejected", "verification": "refuted", "note": "不是这回事"})
    assert seen == []


def test_an_unchanged_reread_does_not_move_the_feedback_revision_and_a_review_is_one_row(tmp_path, monkeypatch):
    runs, index = _open(tmp_path, monkeypatch)
    parsed = _parsed()
    directory = _run(runs, "conv", _record(parsed), started_at=50)
    index.sync_known()
    opened = index.feedback_revision()
    assert opened >= 1
    index.refresh_feedback("conv", directory)
    assert index.feedback_revision() == opened
    premise = _digest("被否决的前提")
    review_feedback("conv", 1, premise, {"status": "rejected", "verification": "refuted", "note": "不是这回事"})
    delta = review_feed(index.conn, since=opened)
    assert delta["same"] is False
    assert [item["digest"] for item in delta["upserts"]] == [premise]
    assert delta["upserts"][0]["status"] == "rejected"
    assert delta["upserts"][0]["text"]
    assert delta["removed"] == []
    assert review_feed(index.conn, since=delta["rev"])["same"] is True


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


def test_info_and_obsolete_are_legal_review_states_with_a_reason():
    assert rejection({"status": "info", "verification": "unchecked"}) is None
    assert rejection({"status": "info", "verification": "confirmed"}) is not None
    assert rejection({"status": "obsolete", "verification": "unchecked", "note": "duplicate of a"}) is None
    assert rejection({"status": "obsolete", "verification": "unchecked"}) is not None


def test_a_parser_upgrade_retires_a_removed_claim_with_its_review_status(tmp_path, monkeypatch):
    runs, index = _open(tmp_path, monkeypatch)
    parsed = _parsed()
    directory = _run(runs, "conv", _record(parsed), started_at=50)
    index.sync_known()
    index.conn.execute(
        "INSERT INTO feedback_claims (run_id, seq, digest, section, audience, premise, status, verification, topic, change_ref, reviewed_at, rev) "
        "VALUES ('conv', 1, 'gone', '需要帮助', 'system', '', 'obsolete', 'unchecked', '', '', NULL, 0)"
    )
    index.conn.execute("UPDATE runs SET feedback_parser = 'fbv2' WHERE run_id = 'conv'")
    index.conn.commit()
    index.refresh_feedback("conv", directory)
    row = index.conn.execute("SELECT status, reason FROM feedback_claim_removed WHERE run_id = 'conv' AND digest = 'gone'").fetchone()
    assert row is not None
    assert row["status"] == "obsolete" and row["reason"] == "parser changed"


def test_fbv4_rebuild_rechecks_a_restored_probe_and_keeps_the_review_key(tmp_path, monkeypatch):
    runs, index = _open(tmp_path, monkeypatch)
    parsed = _parsed()
    directory = _run(runs, "conv", _record(parsed), started_at=50)
    work = directory / "agent_work" / "r1.p0"
    work.mkdir(parents=True)
    (work / "probe.py").write_text("print(1)\n")
    (work / "FEEDBACK.md").write_text(
        "# 被否决的前提\n- [已实测: probe.py | 矩阵单元只能吃 fp16 | 15.09]\n", encoding="utf-8"
    )
    log = directory / "feedback.jsonl"
    record = _record(parsed)
    record["work_dir"] = "agent_work/r1.p0"
    record["items"] = [
        {"section": "被否决的前提", "audience": "peer", "premise": "downgraded",
         "text": "[未验证: 探针文件不存在 | 矩阵单元只能吃 fp16 | 15.09]", "digest": _digest("被否决的前提")},
    ]
    log.write_text(json.dumps(record, ensure_ascii=False) + "\n")
    index.sync_known()
    index.conn.execute("UPDATE runs SET feedback_parser = 'fbv2' WHERE run_id = 'conv'")
    index.conn.commit()
    index.refresh_feedback("conv", directory)
    row = index.conn.execute("SELECT premise, digest FROM feedback_claims WHERE run_id = 'conv'").fetchone()
    assert row["premise"] == "verified"
    assert row["digest"] == _digest("被否决的前提")


def test_a_fbv4_placeholder_gets_a_tombstone_even_on_a_fresh_index(tmp_path, monkeypatch):
    runs, index = _open(tmp_path, monkeypatch)
    parsed = _parsed()
    directory = _run(runs, "conv", _record(parsed), started_at=50)
    record = _record(parsed)
    record["items"] = [
        {"section": "需要帮助", "audience": "system", "premise": "", "text": "无。", "digest": "empty"},
        {"section": "对系统的建议", "audience": "system", "premise": "", "text": "请加一个配对报告。", "digest": "keep"},
    ]
    (directory / "feedback.jsonl").write_text(json.dumps(record, ensure_ascii=False) + "\n")
    index.sync_known()
    row = index.conn.execute("SELECT status, reason FROM feedback_claim_removed WHERE run_id = 'conv' AND digest = 'empty'").fetchone()
    assert row is not None
    assert row["status"] == "obsolete" and row["reason"] == "fbv4 empty placeholder"


def test_the_harness_panel_distinguishes_reviewed_from_unreviewed():
    page = Path("observe/static/index.html").read_text()
    for needle in ("未 review", "review 有用", "review 不建议", "h-yes", "h-no", "reviewed_active", "reviewer", "P2 固化", "P2 候选", "change_ref", "固化到本地", "本地已固化", "promotion_commit", "自由文本系统建议", "人确认后转结构化", "送 LLM review", "convert-modal", "已实测 / 探针可读", "未验证 / 探针不可读"):
        assert needle in page


def test_compact_suggestions_group_by_digest_and_fetch_one_full_text(tmp_path, monkeypatch):
    from observe.suggestions import compact, detail

    runs, index = _open(tmp_path, monkeypatch)
    parsed = parse_feedback("# 对系统的建议\n- 请加一个配对报告。\n")
    _run(runs, "conv", _record(parsed), started_at=50)
    index.sync_known()
    digest = parsed["items"][0]["digest"]
    compact_doc = compact(task="conv2d", status="pending", limit=10, preview=5)
    assert compact_doc["total_claims"] == 1
    group = compact_doc["groups"][0]
    assert group["digest"] == digest and group["runs"] == 1
    assert group["preview"].endswith("…") and len(group["preview"]) <= 6
    full = detail("conv", 1, digest)
    assert full["text"] == "请加一个配对报告。"
    assert full["converted_key"] == "" and full["reviews"] == []


def test_the_suggestions_http_endpoints_are_compact_and_detailed(tmp_path, monkeypatch):
    from starlette.testclient import TestClient

    from observe.app import app

    runs, index = _open(tmp_path, monkeypatch)
    parsed = parse_feedback("# 对系统的建议\n- 请加一个配对报告。\n")
    _run(runs, "conv", _record(parsed), started_at=50)
    index.sync_known()
    digest = parsed["items"][0]["digest"]
    with TestClient(app) as client:
        compact_response = client.get("/api/v2/suggestions", params={"task": "conv2d", "preview": 8, "limit": 5})
        assert compact_response.status_code == 200
        doc = compact_response.json()
        assert doc["groups"][0]["digest"] == digest
        assert len(doc["groups"][0]["preview"]) <= 9
        full = client.get("/api/v2/suggestion", params={"run": "conv", "seq": 1, "digest": digest})
        assert full.status_code == 200 and full.json()["text"] == "请加一个配对报告。"


def test_compact_suggestions_page_over_the_whole_filtered_set(tmp_path, monkeypatch):
    from observe.suggestions import compact

    runs, index = _open(tmp_path, monkeypatch)
    parsed = parse_feedback("# 对系统的建议\n- 第一条建议。\n- 第二条建议。\n- 第三条建议。\n")
    _run(runs, "conv", _record(parsed), started_at=50)
    index.sync_known()
    first = compact(task="conv2d", limit=2, preview=20)
    assert first["total_claims"] == 3 and first["total_groups"] == 3
    assert len(first["groups"]) == 2 and first["truncated"] is True
    assert first["next_offset"] == 2
    second = compact(task="conv2d", limit=2, offset=first["next_offset"], preview=20)
    assert len(second["groups"]) == 1 and second["truncated"] is False
    assert second["next_offset"] is None
    assert {row["digest"] for row in first["groups"] + second["groups"]} == {
        item["digest"] for item in parsed["items"] if item["section"] == "对系统的建议"
    }


def test_compact_does_not_attribute_an_older_conversion_to_the_newest_claim(tmp_path, monkeypatch):
    from observe.suggestions import compact, detail

    runs, index = _open(tmp_path, monkeypatch)
    parsed = parse_feedback("# 对系统的建议\n- 请加一个配对报告。\n")
    _run(runs, "conv", _record(parsed), started_at=50)
    with (runs / "conv" / "feedback.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_record(parsed, seq=2, t=11), ensure_ascii=False) + "\n")
    digest = parsed["items"][0]["digest"]
    (runs / "conv" / "harness_proposals.jsonl").write_text(json.dumps({
        "event": "proposal", "status": "valid", "seq": 1,
        "proposal_key": "older-key", "claim_digest": digest, "proposal": {"surface": "skill"},
    }) + "\n")
    index.sync_known()
    group = compact(task="conv2d", limit=5)["groups"][0]
    assert group["first"]["seq"] == 2
    assert group["link_scope"] == "first (representative claim only)"
    assert group["converted_key"] == ""
    assert detail("conv", 1, digest)["converted_key"] == "older-key"


def test_compact_links_the_exact_seq_not_only_the_first_digest_match(tmp_path, monkeypatch):
    from observe.suggestions import compact, detail

    runs, index = _open(tmp_path, monkeypatch)
    parsed = parse_feedback("# 对系统的建议\n- 请加一个配对报告。\n")
    _run(runs, "conv", _record(parsed), started_at=50)
    with (runs / "conv" / "feedback.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_record(parsed, seq=2, t=11), ensure_ascii=False) + "\n")
    digest = parsed["items"][0]["digest"]
    rows = [
        {"schema_version": 1, "event": "proposal", "status": "valid", "seq": 1,
         "proposal_key": "first-key", "claim_digest": digest, "proposal": {"surface": "skill", "action": "a"}},
        {"schema_version": 1, "event": "proposal", "status": "valid", "seq": 2,
         "proposal_key": "second-key", "claim_digest": digest, "proposal": {"surface": "skill", "action": "b"}},
    ]
    (runs / "conv" / "harness_proposals.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n"
    )
    index.sync_known()
    group = compact(task="conv2d", limit=5)["groups"][0]
    assert group["converted_key"] == "second-key"
    assert group["first"]["seq"] == 2
    assert detail("conv", 1, digest)["converted_key"] == "first-key"
    assert detail("conv", 2, digest)["converted_key"] == "second-key"
