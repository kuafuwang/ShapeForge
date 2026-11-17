"""taic.note replies and dismisses duplicates. No GPU and no model calls."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from observe.feedback import _claim_rows
from taic.agent_feedback import _digest, parse_feedback
from taic.note import NoteError, annotate, main


def _suggestion(text: str) -> dict:
    parsed = parse_feedback(f"# 对系统的建议\n- {text}\n")
    return next(item for item in parsed["items"] if item["section"] == "对系统的建议")


def test_already_measured_returns_the_stored_number_and_invents_nothing(tmp_path: Path):
    """@brief 点名已有 SHA 时回那一行的数字，不补没有测过的字段。"""
    work = tmp_path / "work"
    (work / ".selfcheck").mkdir(parents=True)
    sha = "ab" * 32
    (work / ".selfcheck" / "self_checks.jsonl").write_text(json.dumps({
        "kernel_sha256": sha, "candidate_us": 12.5, "cv": 0.02,
    }) + "\n", encoding="utf-8")
    reply = annotate(work, text=f"remeasure {sha}", section="note")
    assert reply["reply"] == "already_measured"
    assert reply["numbers"]["candidate_us"] == 12.5
    assert reply["numbers"]["cv"] == 0.02
    assert "speedup" not in reply["numbers"]
    assert reply["numbers"]["counts_as_hardware"] is False
    assert reply["dismissed"] is False


def test_simulated_measurement_is_not_hardware(tmp_path: Path):
    """@brief 模拟态测量可以回数字，但不当硬件证据。"""
    work = tmp_path / "work"
    (work / ".selfcheck").mkdir(parents=True)
    sha = "cd" * 32
    (work / ".selfcheck" / "self_checks.jsonl").write_text(json.dumps({
        "kernel_sha256": sha, "candidate_us": 3.0, "simulated": True,
    }) + "\n", encoding="utf-8")
    reply = annotate(work, text=sha, section="note")
    assert reply["numbers"]["simulated"] is True
    assert reply["numbers"]["counts_as_hardware"] is False


def test_authority_self_check_survives_a_deleted_mirror(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """@brief 删掉工作区镜像后，宿主自测权威日志仍能回数字，但不算外层硬件证据。

    围栏写不了账户缓存，所以权威目录改到临时目录。写入仍走 ``append``，查询仍走 ``authority_log``。
    行上自报的 ``host_evidence`` 也不能把这条自测升成可发布的硬件证据。
    """
    from tcenv.selfcheck import append, authority_log

    monkeypatch.setattr("tcenv.selfcheck.authority_root", lambda: tmp_path / "host-selfcheck-state")
    work = tmp_path / "work"
    directory = work / ".selfcheck"
    append(directory, {
        "status": "ok", "name": "a.metal", "candidate_us": 6.0,
        "host_evidence": True, "evidence_level": "host_process",
    })
    authority = authority_log(directory)
    mirror = directory / "self_checks.jsonl"
    assert authority.is_file() and mirror.is_file() and authority.resolve() != mirror.resolve()
    mirror.unlink()
    reply = annotate(work, text="remeasure a.metal")
    assert reply["reply"] == "already_measured"
    assert reply["numbers"]["candidate_us"] == 6.0
    assert reply["numbers"]["source"] == "authority/self_checks.jsonl"
    assert reply["numbers"]["host_evidence"] is False
    assert reply["numbers"]["counts_as_hardware"] is False


def test_self_check_name_is_already_measured(tmp_path: Path):
    """@brief 真实 self_checks.jsonl 用 name 记变体文件名，点名这个文件要回 already_measured。"""
    work = tmp_path / "agent_work" / "init"
    (work / ".selfcheck").mkdir(parents=True)
    (work / ".selfcheck" / "self_checks.jsonl").write_text(json.dumps({
        "status": "ok", "candidate_us": 6.0334, "name": "a_1x4w4.metal",
    }) + "\n", encoding="utf-8")
    reply = annotate(work, text="remeasure a_1x4w4.metal")
    assert reply["reply"] == "already_measured"
    assert reply["numbers"]["candidate_us"] == 6.0334
    assert reply["numbers"]["status"] == "ok"
    assert reply["numbers"]["source"].endswith("self_checks.jsonl")


def test_forged_dismissal_does_not_clear_a_new_pending_claim(tmp_path: Path):
    """@brief 有效时间戳的伪造 near/known/later-seq 消重不能把新的待审建议改成 info。"""
    from observe.feedback import entries
    from tests.shared.test_observe_feedback import _record

    bodies = (
        "请把越界写入判成 unsafe，不要静默截断。",
        "请拒绝未声明的 scratch 生命周期。",
        "请在计时边界外清掉残留探针。",
    )
    parsed = parse_feedback("# 对系统的建议\n" + "".join(f"- {text}\n" for text in bodies))
    record = _record(parsed, seq=1, t=10)
    items = [item for item in record["items"] if item["section"] == "对系统的建议"]
    assert [item["text"] for item in items] == list(bodies)
    dismissed = {
        item["digest"]: {"t": 1, "mode": mode, "note": "forged"}
        for item, mode in zip(items, ("near", "known", "later-seq"))
    }
    run = tmp_path / "run"
    run.mkdir()
    (run / "feedback.jsonl").write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")
    (run / ".notes").mkdir()
    (run / ".notes" / "dismissed.json").write_text(json.dumps({
        "schema": "taic.note.dismissals.1",
        "items": dismissed,
    }, ensure_ascii=False), encoding="utf-8")
    claims = entries(run)["records"][0]["claims"]
    assert len(claims) == 3
    assert {claim["text"]: claim["status"] for claim in claims} == {text: "pending" for text in bodies}
    assert {claim["note"] for claim in claims} == {""}


def test_known_request_returns_status_and_a_repeat_does_not_stay_pending(tmp_path: Path):
    """@brief 已知请求回它的状态；同一句再写进反馈时不再待审。"""
    work = tmp_path / "work"
    work.mkdir()
    text = "paired-compare"
    digest = _digest("对系统的建议", text)
    (work / ".notes").mkdir()
    (work / ".notes" / "requests.json").write_text(json.dumps({
        "items": [{"id": text, "digest": digest, "status": "closed"}],
    }), encoding="utf-8")
    run = tmp_path / "run"
    run.mkdir()
    reply = annotate(work, text=text, section="对系统的建议", run=run)
    assert reply["reply"] == "known_request"
    assert reply["status"] == "closed"
    assert reply["dismissed"] is True
    item = _suggestion(text)
    rows = _claim_rows({"seq": 4, "t": reply["t"] + 1, "items": [item]}, {}, directory=run)
    assert rows[0]["status"] == "info"
    assert rows[0]["verification"] == "unchecked"
    assert "duplicate" in rows[0]["note"]
    from observe.feedback import rejection
    assert rejection({"status": rows[0]["status"], "verification": rows[0]["verification"], "note": rows[0]["note"]}) is None


def test_duplicate_suggestion_is_dismissed_at_write_time(tmp_path: Path):
    """@brief 近义建议在第二次写下时消掉；原来的那条 digest 仍可待审。"""
    work = tmp_path / "agent_work" / "r1"
    work.mkdir(parents=True)
    run = tmp_path
    first = "add paired compare subcommand to tcenv now"
    second = "add paired compare subcommand to tcenv today"
    opened = annotate(work, text=first, section="对系统的建议", run=run, now=10)
    assert opened["reply"] == "recorded" and opened["dismissed"] is False
    repeated = annotate(work, text=second, section="对系统的建议", run=run, now=20)
    assert repeated["reply"] == "duplicate" and repeated["dismissed"] is True
    assert repeated["prior"]["digest"] == opened["digest"]
    original = _suggestion(first)
    near = _suggestion(second)
    kept = _claim_rows({"seq": 1, "t": 30, "items": [original]}, {}, directory=run)
    dropped = _claim_rows({"seq": 1, "t": 30, "items": [near]}, {}, directory=run)
    assert kept[0]["status"] == "pending"
    assert dropped[0]["status"] == "info"
    protected = _claim_rows(
        {"seq": 1, "t": 30, "items": [near]},
        {f"1:{near['digest']}": {"status": "pending"}},
        directory=run,
    )
    assert protected[0]["status"] == "pending"


def test_a_later_seq_of_the_same_sentence_does_not_raise_pending(tmp_path: Path):
    """@brief 重复 note 留下锚定的 seq=1；更晚的同文 seq=2 才是 info。人工状态优先。"""
    work = tmp_path / "work"
    work.mkdir()
    run = tmp_path / "run"
    run.mkdir()
    text = "add paired compare subcommand to tcenv now"
    item = _suggestion(text)
    (run / "feedback.jsonl").write_text(json.dumps({
        "seq": 1, "t": 5, "run_id": "run", "items": [item],
    }) + "\n", encoding="utf-8")
    held = (run / "feedback.jsonl").read_bytes()
    reply = annotate(work, text=text, section="对系统的建议", run=run, now=15)
    assert reply["reply"] == "duplicate" and reply["dismissed"] is True
    assert reply["prior"]["seq"] == 1
    assert (run / "feedback.jsonl").read_bytes() == held
    stored = json.loads((run / ".notes" / "dismissed.json").read_text(encoding="utf-8"))
    assert stored["items"][item["digest"]]["mode"] == "later-seq"
    assert stored["items"][item["digest"]]["anchor_seq"] == 1
    assert stored["items"][item["digest"]]["anchor_t"] == 5
    original = _claim_rows({"seq": 1, "t": 5, "items": [item]}, {}, directory=run)
    later = _claim_rows({"seq": 2, "t": 16, "items": [item]}, {}, directory=run)
    assert original[0]["status"] == "pending"
    assert original[0]["note"] == ""
    assert later[0]["status"] == "info"
    assert later[0]["verification"] == "unchecked"
    assert "duplicate" in later[0]["note"]
    kept = _claim_rows(
        {"seq": 2, "t": 16, "items": [item]},
        {f"2:{item['digest']}": {"status": "pending"}},
        directory=run,
    )
    assert kept[0]["status"] == "pending"
    rejected = _claim_rows(
        {"seq": 1, "t": 5, "items": [item]},
        {f"1:{item['digest']}": {"status": "rejected", "verification": "refuted"}},
        directory=run,
    )
    assert rejected[0]["status"] == "rejected"


def test_file_fallback_still_writes_notes_when_the_note_directory_is_blocked(tmp_path: Path):
    """@brief `.notes` 建不出来时改写 NOTES.md，不假装结构化写入成功。"""
    work = tmp_path / "work"
    work.mkdir()
    (work / ".notes").write_text("not a directory\n", encoding="utf-8")
    reply = annotate(work, text="keep this annotation", section="note")
    assert reply["fallback"] is True
    assert reply["wrote"].endswith("NOTES.md")
    assert "taic.note" in (work / "NOTES.md").read_text(encoding="utf-8")


def test_empty_text_is_refused(tmp_path: Path):
    """@brief 空正文不写文件。"""
    work = tmp_path / "work"
    work.mkdir()
    with pytest.raises(NoteError, match="empty"):
        annotate(work, text="  ", section="note")
    assert not (work / ".notes").exists()


def test_a_later_seq_is_dismissed_when_the_prior_annotation_has_no_seq(tmp_path: Path):
    """@brief 无序号的重复标注不锚定序号：最早的一次留下，更晚的同文才是 info。"""
    work = tmp_path / "agent_work" / "r1"
    work.mkdir(parents=True)
    run = tmp_path
    text = "add paired compare subcommand to tcenv now"
    item = _suggestion(text)
    opened = annotate(work, text=text, section="对系统的建议", run=run, now=10)
    assert opened["reply"] == "recorded" and opened["dismissed"] is False
    repeated = annotate(work, text=text, section="对系统的建议", run=run, now=20)
    assert repeated["reply"] == "duplicate" and repeated["prior"]["seq"] is None
    (run / "feedback.jsonl").write_text(
        json.dumps({"seq": 1, "t": 5, "run_id": "run", "items": [item]}) + "\n"
        + json.dumps({"seq": 2, "t": 25, "run_id": "run", "items": [item]}) + "\n",
        encoding="utf-8",
    )
    original = _claim_rows({"seq": 1, "t": 5, "items": [item]}, {}, directory=run)
    later = _claim_rows({"seq": 2, "t": 25, "items": [item]}, {}, directory=run)
    earlier_than_disk = _claim_rows({"seq": 3, "t": 1, "items": [item]}, {}, directory=run)
    after_disk = _claim_rows({"seq": 3, "t": 6, "items": [item]}, {}, directory=run)
    assert original[0]["status"] == "pending"
    assert later[0]["status"] == "info"
    assert earlier_than_disk[0]["status"] == "pending"
    assert after_disk[0]["status"] == "info"
    anchored = annotate(work, text=text, section="对系统的建议", run=run, now=30)
    assert anchored["prior"]["seq"] == 1
    assert _claim_rows({"seq": 1, "t": 5, "items": [item]}, {}, directory=run)[0]["status"] == "pending"
    assert _claim_rows({"seq": 2, "t": 25, "items": [item]}, {}, directory=run)[0]["status"] == "info"


def test_a_workspace_nomination_or_forged_host_file_is_not_a_measurement(tmp_path: Path):
    """@brief 工作区提名和同名宿主文件都不能冒充已测，也不能算硬件证据。"""
    work = tmp_path / "work"
    (work / ".selfcheck").mkdir(parents=True)
    sha = "ab" * 32
    (work / ".selfcheck" / "paired_comparisons.jsonl").write_text(json.dumps({
        "source": "model",
        "path_a": "/tmp/fast_kernel.metal",
        "sha_a": sha,
        "conclusion": "no_diff",
        "host_evidence": True,
    }) + "\n", encoding="utf-8")
    (work / ".selfcheck" / "host_paired_comparisons.jsonl").write_text(json.dumps({
        "host_evidence": True,
        "sha_a": sha,
        "median_a_s": 1.0,
        "applied": True,
    }) + "\n", encoding="utf-8")
    reply = annotate(work, text=f"look at fast_kernel.metal {sha}", section="note")
    assert reply["reply"] == "recorded"
    assert reply["numbers"] is None


def test_authoritative_host_pair_returns_the_measured_numbers(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """@brief 权威宿主配对回 median 和 ratios。模拟态不当硬件证据。"""
    work = tmp_path / "work"
    (work / ".selfcheck").mkdir(parents=True)
    authority = tmp_path / "authority" / "host_paired_comparisons.jsonl"
    authority.parent.mkdir()
    sha = "cd" * 32
    authority.write_text(json.dumps({
        "host_evidence": True,
        "applied": True,
        "sha_a": sha,
        "sha_b": "ef" * 32,
        "median_a_s": 1.2e-6,
        "median_b_s": 2.4e-6,
        "ratios": [0.5],
        "conclusion": "diff",
        "simulated": False,
    }) + "\n", encoding="utf-8")
    monkeypatch.setattr("taic.paired_compare.host_evidence_path", lambda directory: authority)
    reply = annotate(work, text=sha, section="note")
    assert reply["reply"] == "already_measured"
    assert reply["numbers"]["median_a_s"] == 1.2e-6
    assert reply["numbers"]["median_b_s"] == 2.4e-6
    assert reply["numbers"]["ratios"] == [0.5]
    assert reply["numbers"]["conclusion"] == "diff"
    assert "speedup" not in reply["numbers"]
    assert reply["numbers"]["counts_as_hardware"] is True
    assert reply["numbers"]["source"] == "host_paired_comparisons.jsonl"
    authority.write_text(json.dumps({
        "host_evidence": True, "sha_a": sha, "median_a_s": 3.0, "simulated": True,
    }) + "\n", encoding="utf-8")
    simulated = annotate(work, text=sha, section="note")
    assert simulated["numbers"]["median_a_s"] == 3.0
    assert simulated["numbers"]["counts_as_hardware"] is False


def test_host_evidence_simulation_is_not_hardware_without_a_top_level_flag(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """@brief append_host_evidence 的模拟行没有顶层 simulated 时，数字可回，但不能当硬件证据。"""
    from taic.paired_compare import append_host_evidence

    work = tmp_path / "work"
    (work / ".selfcheck").mkdir(parents=True)
    authority = tmp_path / "authority" / "host_paired_comparisons.jsonl"
    monkeypatch.setattr("taic.paired_compare.host_evidence_path", lambda directory: authority)
    sha = "ab" * 32
    append_host_evidence(work / ".selfcheck", {
        "applied": True,
        "sha_a": sha,
        "sha_b": "cd" * 32,
        "median_a_s": 1.5e-6,
        "median_b_s": 3.0e-6,
        "ratios": [0.5],
        "measurement_type": "simulated",
    })
    written = json.loads(authority.read_text(encoding="utf-8"))
    assert written.get("simulated") is not True
    assert written["host_evidence"] is True
    assert written["evidence_level"] == "simulated"
    assert written["evidence_provenance"]["simulated"] is True
    reply = annotate(work, text=sha, section="note")
    assert reply["reply"] == "already_measured"
    assert reply["numbers"]["median_a_s"] == 1.5e-6
    assert reply["numbers"]["simulated"] is True
    assert reply["numbers"]["counts_as_hardware"] is False
    stripped = {
        key: value for key, value in written.items()
        if key not in {"measurement_kind", "measurement_type", "simulated", "provenance"}
    }
    authority.write_text(json.dumps(stripped) + "\n", encoding="utf-8")
    again = annotate(work, text=sha, section="note")
    assert again["numbers"]["median_a_s"] == 1.5e-6
    assert again["numbers"]["simulated"] is True
    assert again["numbers"]["counts_as_hardware"] is False


def test_a_known_claim_outside_suggestion_sections_returns_its_status(tmp_path: Path):
    """@brief 其他小节里的精确 claim 也回已有状态，并且不把这次标注消掉。"""
    work = tmp_path / "work"
    work.mkdir()
    run = tmp_path / "run"
    run.mkdir()
    text = "the tile size was already checked"
    parsed = parse_feedback(f"# 被否决的前提\n- {text}\n")
    item = next(row for row in parsed["items"] if row["section"] == "被否决的前提")
    (run / "feedback.jsonl").write_text(json.dumps({
        "seq": 3, "t": 1, "run_id": "run", "items": [item],
    }) + "\n", encoding="utf-8")
    (run / "feedback_review.json").write_text(json.dumps({
        "claims": {f"3:{item['digest']}": {"status": "rejected"}},
    }), encoding="utf-8")
    reply = annotate(work, text=text, section="被否决的前提", run=run)
    assert reply["reply"] == "known_request"
    assert reply["status"] == "rejected"
    assert reply["dismissed"] is False


def test_requests_json_matches_the_digest_string_itself(tmp_path: Path):
    """@brief 请求表里的 digest 可以被整段正文点名，不必再算一遍正文哈希。"""
    work = tmp_path / "work"
    work.mkdir()
    text = "paired-compare"
    digest = _digest("对系统的建议", text)
    (work / ".notes").mkdir()
    (work / ".notes" / "requests.json").write_text(json.dumps({
        "items": [{"id": "req-1", "digest": digest, "text": text, "status": "closed"}],
    }), encoding="utf-8")
    run = tmp_path / "run"
    run.mkdir()
    item = _suggestion(text)
    (run / "feedback.jsonl").write_text(json.dumps({
        "seq": 2, "t": 1, "run_id": "run", "items": [item],
    }) + "\n", encoding="utf-8")
    reply = annotate(work, text=digest, section="对系统的建议", run=run)
    assert reply["reply"] == "known_request"
    assert reply["status"] == "closed"
    queued = _claim_rows({"seq": 2, "t": reply["t"] + 1, "items": [item]}, {}, directory=run)
    assert queued[0]["status"] == "info"
    missed = annotate(
        work, text=f"see {digest} later", section="对系统的建议", run=run,
        scorer=lambda left, right: 0.0, method="lexical_overlap",
    )
    assert missed["reply"] == "recorded"


def test_cli_requests_json_matches_the_digest_string(tmp_path: Path, capsys):
    """@brief 命令行对请求表 digest 同样回已知状态。"""
    work = tmp_path / "work"
    work.mkdir()
    digest = _digest("对系统的建议", "paired-compare")
    (work / ".notes").mkdir()
    (work / ".notes" / "requests.json").write_text(json.dumps({
        "items": [{"id": "req-1", "digest": digest, "status": "closed"}],
    }), encoding="utf-8")
    code = main(["--work", str(work), "--text", digest, "--section", "对系统的建议"])
    assert code == 0
    document = json.loads(capsys.readouterr().out)
    assert document["reply"] == "known_request"
    assert document["status"] == "closed"


def test_feedback_claim_matches_the_digest_string_itself(tmp_path: Path):
    """@brief feedback.jsonl 里的 digest 被整段点名时回该 claim 的状态。"""
    work = tmp_path / "work"
    work.mkdir()
    run = tmp_path / "run"
    run.mkdir()
    text = "paired-compare"
    item = _suggestion(text)
    (run / "feedback.jsonl").write_text(json.dumps({
        "seq": 4, "t": 1, "run_id": "run", "items": [item],
    }) + "\n", encoding="utf-8")
    (run / "feedback_review.json").write_text(json.dumps({
        "claims": {f"4:{item['digest']}": {"status": "rejected"}},
    }), encoding="utf-8")
    reply = annotate(work, text=item["digest"], section="对系统的建议", run=run)
    assert reply["reply"] == "known_request"
    assert reply["status"] == "rejected"


def test_cli_feedback_claim_matches_the_digest_string(tmp_path: Path, capsys):
    """@brief 命令行对 feedback.jsonl 的 digest 回该 claim 的状态。"""
    work = tmp_path / "work"
    work.mkdir()
    run = tmp_path / "run"
    run.mkdir()
    item = _suggestion("paired-compare")
    (run / "feedback.jsonl").write_text(json.dumps({
        "seq": 4, "t": 1, "run_id": "run", "items": [item],
    }) + "\n", encoding="utf-8")
    code = main([
        "--work", str(work), "--run", str(run), "--text", item["digest"], "--section", "对系统的建议",
    ])
    assert code == 0
    document = json.loads(capsys.readouterr().out)
    assert document["reply"] == "known_request"
    assert document["status"] == "pending"


def test_review_resync_keeps_note_dismissal_in_the_compact_view(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """@brief 无关评审触发索引重算后，compact 仍显示 note 消重；人保存的状态优先。"""
    from observe.query import suggestions
    from tests.shared.test_observe_feedback import _open, _record, _run

    monkeypatch.setenv("OBSERVE_FULL_TICK_INTERVAL", "0")
    runs, _index = _open(tmp_path, monkeypatch)
    dismissed_text = "paired-compare"
    other_text = "change the tolerance threshold for evaluation"
    parsed = parse_feedback(
        "# 对系统的建议\n"
        f"- {dismissed_text}\n"
        f"- {other_text}\n"
    )
    items = [item for item in parsed["items"] if item["section"] == "对系统的建议"]
    dismissed = next(item for item in items if item["text"] == dismissed_text)
    other = next(item for item in items if item["text"] == other_text)
    record = _record(parsed, t=20)
    directory = _run(runs, "metal-one", record, started_at=20)
    (directory / "result.json").write_text(json.dumps({"task": record["task"]}), encoding="utf-8")
    work = tmp_path / "work"
    work.mkdir()
    (work / ".notes").mkdir()
    (work / ".notes" / "requests.json").write_text(json.dumps({
        "items": [{"id": dismissed_text, "digest": dismissed["digest"], "status": "closed"}],
    }), encoding="utf-8")
    held = (directory / "feedback.jsonl").read_text(encoding="utf-8")
    (directory / "feedback.jsonl").unlink()
    reply = annotate(work, text=dismissed_text, section="对系统的建议", run=directory, now=30)
    (directory / "feedback.jsonl").write_text(held, encoding="utf-8")
    assert reply["reply"] == "known_request" and reply["dismissed"] is True
    assert json.loads((directory / ".notes" / "dismissed.json").read_text(encoding="utf-8"))["items"][dismissed["digest"]]["mode"] == "known"

    def digests(status: str) -> set[str]:
        document = suggestions(status=status, limit=20)
        return {group["digest"] for group in document["groups"]}

    assert dismissed["digest"] in digests("info")
    assert dismissed["digest"] not in digests("pending")
    (directory / "feedback_review.json").write_text(json.dumps({
        "v": 1,
        "claims": {f"1:{other['digest']}": {"status": "rejected", "verification": "refuted", "t": 40}},
    }), encoding="utf-8")
    assert other["digest"] in digests("rejected")
    assert dismissed["digest"] in digests("info")
    assert dismissed["digest"] not in digests("pending")
    saved = json.loads((directory / "feedback_review.json").read_text(encoding="utf-8"))
    saved["claims"][f"1:{dismissed['digest']}"] = {"status": "pending", "verification": "unchecked", "t": 50}
    (directory / "feedback_review.json").write_text(json.dumps(saved), encoding="utf-8")
    assert dismissed["digest"] in digests("pending")
    assert dismissed["digest"] not in digests("info")


def test_seqless_exact_duplicate_keeps_the_earliest_seq_after_resync(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """@brief 无序号精确重复留下最早的 seq；更晚的同文在无关评审重算后仍是 info。"""
    from observe.query import suggestions
    from tests.shared.test_observe_feedback import _open, _record, _run

    monkeypatch.setenv("OBSERVE_FULL_TICK_INTERVAL", "0")
    runs, index = _open(tmp_path, monkeypatch)
    repeated = "add paired compare subcommand to tcenv now"
    other_text = "change the tolerance threshold for evaluation"
    parsed = parse_feedback(
        "# 对系统的建议\n"
        f"- {repeated}\n"
        f"- {other_text}\n"
    )
    items = [item for item in parsed["items"] if item["section"] == "对系统的建议"]
    claim = next(item for item in items if item["text"] == repeated)
    other = next(item for item in items if item["text"] == other_text)
    record = _record(parsed, seq=2, t=25)
    directory = _run(runs, "metal-one", record, started_at=25)
    (directory / "result.json").write_text(json.dumps({"task": record["task"]}), encoding="utf-8")
    work = tmp_path / "work"
    work.mkdir()
    held = (directory / "feedback.jsonl").read_text(encoding="utf-8")
    (directory / "feedback.jsonl").unlink()
    opened = annotate(work, text=repeated, section="对系统的建议", run=directory, now=10)
    assert opened["reply"] == "recorded" and opened["dismissed"] is False
    second = annotate(work, text=repeated, section="对系统的建议", run=directory, now=20)
    (directory / "feedback.jsonl").write_text(held, encoding="utf-8")
    assert second["reply"] == "duplicate" and second["prior"]["seq"] is None
    stored = json.loads((directory / ".notes" / "dismissed.json").read_text(encoding="utf-8"))
    assert stored["items"][claim["digest"]]["mode"] == "later-seq"
    assert stored["items"][claim["digest"]]["anchor_seq"] is None
    assert stored["items"][claim["digest"]]["prior_seq"] is None
    with (directory / "feedback.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_record(parsed, seq=3, t=30), ensure_ascii=False) + "\n")

    def digests(status: str) -> set[str]:
        document = suggestions(status=status, limit=20)
        return {group["digest"] for group in document["groups"]}

    def statuses() -> dict[int, str]:
        index.refresh_feedback(index._rel(directory), directory)
        rows = index.conn.execute(
            "SELECT seq, status FROM feedback_claims WHERE run_id = ? AND digest = ?",
            (index._rel(directory), claim["digest"]),
        ).fetchall()
        return {int(row["seq"]): str(row["status"]) for row in rows}

    assert statuses() == {2: "pending", 3: "info"}
    assert claim["digest"] in digests("pending")
    assert claim["digest"] in digests("info")
    (directory / "feedback_review.json").write_text(json.dumps({
        "v": 1,
        "claims": {f"2:{other['digest']}": {"status": "rejected", "verification": "refuted", "t": 40}},
    }), encoding="utf-8")
    assert other["digest"] in digests("rejected")
    assert statuses() == {2: "pending", 3: "info"}


def _indexed_suggestion(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str):
    from tests.shared.test_observe_feedback import _open, _record, _run

    monkeypatch.setenv("OBSERVE_FULL_TICK_INTERVAL", "0")
    runs, index = _open(tmp_path, monkeypatch)
    repeated = "add paired compare subcommand to tcenv now"
    other_text = "change the tolerance threshold for evaluation"
    parsed = parse_feedback(
        "# 对系统的建议\n"
        f"- {repeated}\n"
        f"- {other_text}\n"
    )
    items = [item for item in parsed["items"] if item["section"] == "对系统的建议"]
    claim = next(item for item in items if item["text"] == repeated)
    other = next(item for item in items if item["text"] == other_text)
    record = _record(parsed, seq=7, t=25)
    directory = _run(runs, name, record, started_at=25)
    (directory / "result.json").write_text(json.dumps({"task": record["task"]}), encoding="utf-8")
    log = directory / "feedback.jsonl"
    original = log.read_bytes()
    index.sync_known()
    return directory, log, original, claim, other, repeated


def _suggestion_digests(status: str) -> set[str]:
    from observe.query import suggestions

    document = suggestions(status=status, limit=20)
    return {group["digest"] for group in document["groups"]}


def test_exact_repeat_keeps_seq1_pending_and_marks_only_seq2_info(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """@brief 真实索引：重复 note 不改日志。seq=1 仍 pending，同文 seq=2 为 info，人工状态优先。"""
    from observe.feedback import entries
    from observe.query import suggestion
    from tests.shared.test_observe_feedback import _open, _record, _run

    monkeypatch.setenv("OBSERVE_FULL_TICK_INTERVAL", "0")
    runs, index = _open(tmp_path, monkeypatch)
    text = "add paired compare subcommand to tcenv now"
    parsed = parse_feedback(f"# 对系统的建议\n- {text}\n")
    claim = next(item for item in parsed["items"] if item["section"] == "对系统的建议")
    digest = claim["digest"]
    record = _record(parsed, seq=1, t=10)
    directory = _run(runs, "metal-keep", record, started_at=10)
    (directory / "result.json").write_text(json.dumps({"task": record["task"]}), encoding="utf-8")
    log = directory / "feedback.jsonl"
    original = log.read_bytes()
    index.sync_known()
    work = tmp_path / "work"
    work.mkdir()
    reply = annotate(work, text=text, section="对系统的建议", run=directory, now=20)
    assert reply["reply"] == "duplicate" and reply["dismissed"] is True
    assert reply["prior"]["seq"] == 1
    assert log.read_bytes() == original
    run_id = index._rel(directory)

    def statuses() -> dict[int, tuple[str, str]]:
        index.refresh_feedback(run_id, directory)
        rows = index.conn.execute(
            "SELECT seq, status, verification FROM feedback_claims WHERE run_id = ? AND digest = ?",
            (run_id, digest),
        ).fetchall()
        return {int(row["seq"]): (str(row["status"]), str(row["verification"])) for row in rows}

    assert statuses() == {1: ("pending", "unchecked")}
    assert suggestion(run_id, 1, digest)["status"] == "pending"
    with log.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_record(parsed, seq=2, t=30), ensure_ascii=False) + "\n")
    assert statuses() == {1: ("pending", "unchecked"), 2: ("info", "unchecked")}
    assert suggestion(run_id, 1, digest)["status"] == "pending"
    assert suggestion(run_id, 2, digest)["status"] == "info"
    (directory / "feedback_review.json").write_text(json.dumps({
        "v": 1,
        "claims": {
            f"1:{digest}": {"status": "rejected", "verification": "refuted", "t": 40},
            f"2:{digest}": {"status": "pending", "verification": "unchecked", "t": 41},
        },
    }), encoding="utf-8")
    assert statuses() == {1: ("rejected", "refuted"), 2: ("pending", "unchecked")}
    by_seq = {
        item["seq"]: item["status"]
        for row in entries(directory)["records"]
        for item in row["claims"]
        if item["digest"] == digest
    }
    assert by_seq == {1: "rejected", 2: "pending"}


def test_an_already_indexed_claim_stays_pending_when_the_note_repeats_it(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """@brief 已索引的第一条在 note 重复后仍待审。日志不改。更晚的同文 seq 才是 info。"""
    from observe.index import get_index
    from tests.shared.test_observe_feedback import _record

    directory, log, original, claim, other, repeated = _indexed_suggestion(tmp_path, monkeypatch, "metal-one")
    work = tmp_path / "work"
    work.mkdir()
    assert claim["digest"] in _suggestion_digests("pending")
    first = annotate(work, text=repeated, section="对系统的建议", run=directory, now=30)
    second = annotate(work, text=repeated, section="对系统的建议", run=directory, now=40)
    assert first["reply"] == "duplicate" and first["dismissed"] is True
    assert first["prior"]["seq"] == 7
    assert second["reply"] == "duplicate" and second["dismissed"] is True
    assert log.read_bytes() == original
    assert not (directory / "feedback_review.json").exists()
    assert claim["digest"] in _suggestion_digests("pending")
    assert claim["digest"] not in _suggestion_digests("info")
    assert other["digest"] in _suggestion_digests("pending")
    parsed = parse_feedback(f"# 对系统的建议\n- {repeated}\n")
    with log.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_record(parsed, seq=8, t=45), ensure_ascii=False) + "\n")
    index = get_index()
    index.refresh_feedback(index._rel(directory), directory)
    rows = {
        int(row["seq"]): str(row["status"])
        for row in index.conn.execute(
            "SELECT seq, status FROM feedback_claims WHERE run_id = ? AND digest = ?",
            (index._rel(directory), claim["digest"]),
        )
    }
    assert rows[7] == "pending"
    assert rows[8] == "info"
    assert log.read_bytes().startswith(original)


def test_an_agent_work_note_keeps_the_indexed_claim_and_dismisses_a_later_seq(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """@brief agent_work 里的无序号重复不改日志，也不把已索引的第一条赶出待审。"""
    from observe.index import get_index
    from tests.shared.test_observe_feedback import _record

    directory, log, original, claim, other, repeated = _indexed_suggestion(tmp_path, monkeypatch, "metal-two")
    work = directory / "agent_work" / "s1"
    work.mkdir(parents=True)
    assert claim["digest"] in _suggestion_digests("pending")
    opened = annotate(work, text=repeated, section="对系统的建议", now=30)
    second = annotate(work, text=repeated, section="对系统的建议", now=40)
    assert opened["reply"] == "recorded" and opened["dismissed"] is False
    assert second["reply"] == "duplicate" and second["dismissed"] is True
    assert second["prior"]["seq"] is None
    assert not (directory / ".notes" / "dismissed.json").exists()
    stored = json.loads((work / ".notes" / "dismissed.json").read_text(encoding="utf-8"))
    assert stored["items"][claim["digest"]]["mode"] == "later-seq"
    assert stored["items"][claim["digest"]]["anchor_seq"] is None
    assert stored["items"][claim["digest"]]["prior_seq"] is None
    assert log.read_bytes() == original
    assert not (directory / "feedback_review.json").exists()
    assert claim["digest"] in _suggestion_digests("pending")
    assert claim["digest"] not in _suggestion_digests("info")
    assert other["digest"] in _suggestion_digests("pending")
    parsed = parse_feedback(f"# 对系统的建议\n- {repeated}\n")
    with log.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_record(parsed, seq=8, t=45), ensure_ascii=False) + "\n")
    index = get_index()
    index.refresh_feedback(index._rel(directory), directory)
    rows = {
        int(row["seq"]): str(row["status"])
        for row in index.conn.execute(
            "SELECT seq, status FROM feedback_claims WHERE run_id = ? AND digest = ?",
            (index._rel(directory), claim["digest"]),
        )
    }
    assert rows[7] == "pending"
    assert rows[8] == "info"
    assert log.read_bytes().startswith(original)


def test_cli_replies_without_a_model(tmp_path: Path, capsys):
    """@brief 命令行入口只打印回话。"""
    work = tmp_path / "work"
    work.mkdir()
    code = main(["--work", str(work), "--text", "hello note", "--section", "note"])
    assert code == 0
    document = json.loads(capsys.readouterr().out)
    assert document["reply"] == "recorded"
    assert document["schema"] == "taic.note.1"


def test_a_known_request_in_another_run_returns_the_latest_human_status(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys):
    """@brief 另一场已标 done 的同一 digest，在这场 note 里回 known_request，并把这场的新 claim 消掉。"""
    from observe.feedback import entries, rejection
    from observe.query import suggestions
    from taic.cli import main as taic_main
    from tests.shared.test_observe_feedback import _open, _record, _run

    monkeypatch.setenv("OBSERVE_FULL_TICK_INTERVAL", "0")
    runs, index = _open(tmp_path, monkeypatch)
    text = "add paired compare subcommand to tcenv now"
    parsed = parse_feedback(f"# 对系统的建议\n- {text}\n")
    claim = next(item for item in parsed["items"] if item["section"] == "对系统的建议")
    digest = claim["digest"]
    earlier = _record(parsed, seq=1, t=20)
    run_a = _run(runs, "metal-a", earlier, started_at=20)
    (run_a / "result.json").write_text(json.dumps({"task": earlier["task"]}), encoding="utf-8")
    (run_a / "feedback_review.json").write_text(json.dumps({
        "v": 1,
        "claims": {f"1:{digest}": {"status": "done", "verification": "confirmed", "change_ref": "docs/代理反馈.md", "t": 30}},
    }), encoding="utf-8")
    later = _record(parsed, seq=1, t=40)
    run_b = _run(runs, "metal-b", later, started_at=40)
    (run_b / "result.json").write_text(json.dumps({"task": later["task"]}), encoding="utf-8")
    log_a = (run_a / "feedback.jsonl").read_bytes()
    log_b = (run_b / "feedback.jsonl").read_bytes()
    index.sync_known()
    work = tmp_path / "work"
    work.mkdir()
    code = taic_main(["note", "--work", str(work), "--run", str(run_b), "--section", "对系统的建议", "--text", text])
    assert code == 0
    reply = json.loads(capsys.readouterr().out)
    assert reply["reply"] == "known_request"
    assert reply["status"] == "done"
    assert reply["dismissed"] is True
    assert reply["prior"]["run_id"] == "metal-a"
    assert log_a == (run_a / "feedback.jsonl").read_bytes()
    assert log_b == (run_b / "feedback.jsonl").read_bytes()
    assert not (run_a / ".notes" / "dismissed.json").exists()

    def digests(status: str) -> set[str]:
        return {group["digest"] for group in suggestions(status=status, limit=20)["groups"]}

    assert digest in digests("done")
    assert digest in digests("info")
    assert digest not in digests("pending")
    kept = next(item for record in entries(run_a)["records"] for item in record["claims"] if item["digest"] == digest)
    dropped = next(item for record in entries(run_b)["records"] for item in record["claims"] if item["digest"] == digest)
    assert kept["status"] == "done"
    assert dropped["status"] == "info"
    assert dropped["verification"] == "unchecked"
    assert "duplicate" in dropped["note"]
    assert rejection({"status": dropped["status"], "verification": dropped["verification"], "note": dropped["note"]}) is None
    stored = index.conn.execute(
        "SELECT status, verification FROM feedback_claims WHERE run_id = ? AND digest = ?",
        ("metal-b", digest),
    ).fetchone()
    assert stored["status"] == "info" and stored["verification"] == "unchecked"
    other = tmp_path / "digest-work"
    other.mkdir()
    code = taic_main(["note", "--work", str(other), "--run", str(run_b), "--section", "对系统的建议", "--text", digest])
    assert code == 0
    named = json.loads(capsys.readouterr().out)
    assert named["reply"] == "known_request"
    assert named["status"] == "done"
    plain = tmp_path / "text-work"
    plain.mkdir()
    code = taic_main(["note", "--work", str(plain), "--run", str(run_b), "--section", "note", "--text", text])
    assert code == 0
    body_hit = json.loads(capsys.readouterr().out)
    assert body_hit["reply"] == "known_request"
    assert body_hit["status"] == "done"
    assert body_hit["dismissed"] is False
    assert log_a == (run_a / "feedback.jsonl").read_bytes()
    assert log_b == (run_b / "feedback.jsonl").read_bytes()


def test_prune_keeps_notes_and_the_dismissal_survives_resync(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """@brief prune 留下 .notes。重同步后第一条仍待审，更晚的同文仍是 info。"""
    from observe.index import get_index
    from taic.kernel_agent import prune_work
    from tests.shared.test_observe_feedback import _record

    directory, log, _original, claim, _other, repeated = _indexed_suggestion(tmp_path, monkeypatch, "metal-prune")
    work = directory / "agent_work" / "s1"
    work.mkdir(parents=True)
    (work / "scratch.txt").write_text("temporary\n", encoding="utf-8")
    annotate(work, text=repeated, section="对系统的建议", now=30)
    second = annotate(work, text=repeated, section="对系统的建议", now=40)
    assert second["reply"] == "duplicate" and second["dismissed"] is True
    assert claim["digest"] in _suggestion_digests("pending")
    assert claim["digest"] not in _suggestion_digests("info")
    parsed = parse_feedback(f"# 对系统的建议\n- {repeated}\n")
    with log.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_record(parsed, seq=8, t=45), ensure_ascii=False) + "\n")
    pruned = prune_work(work, label="s1")
    assert ".notes" not in pruned["archived"]
    assert "scratch.txt" in pruned["archived"]
    assert (work / ".notes" / "dismissed.json").is_file()
    index = get_index()
    index.refresh_feedback(index._rel(directory), directory)
    rows = {
        int(row["seq"]): str(row["status"])
        for row in index.conn.execute(
            "SELECT seq, status FROM feedback_claims WHERE run_id = ? AND digest = ?",
            (index._rel(directory), claim["digest"]),
        )
    }
    assert rows[7] == "pending"
    assert rows[8] == "info"


def test_near_duplicate_compares_the_action_not_the_json_shell(tmp_path: Path):
    """@brief 相同外壳、不同 action 不消重；近似 action 才消重。"""
    work = tmp_path / "work"
    work.mkdir()
    shell = {"surface": "skill", "scope": {"task": "gemm", "target_id": "apple-m4-max"}, "evidence": {"probe": "probe.py"}}
    first = annotate(work, text=json.dumps({**shell, "action": "add paired compare subcommand to tcenv now"}), section="对系统的建议", now=10)
    different = annotate(work, text=json.dumps({**shell, "action": "remove unsafe optimization from the kernel"}), section="对系统的建议", now=20)
    similar = annotate(work, text=json.dumps({**shell, "action": "add paired compare subcommand to tcenv today"}), section="对系统的建议", now=30)
    assert first["reply"] == "recorded" and first["dismissed"] is False
    assert different["reply"] == "recorded" and different["dismissed"] is False
    assert similar["reply"] == "duplicate" and similar["dismissed"] is True


def test_timeout_is_not_measured_and_a_later_success_wins(tmp_path: Path):
    """@brief timeout 且没有数字不是已测；后面的成功行盖过前面的失败行。"""
    work = tmp_path / "work"
    (work / ".selfcheck").mkdir(parents=True)
    sha = "ab" * 32
    path = work / ".selfcheck" / "self_checks.jsonl"
    path.write_text(json.dumps({
        "kernel_sha256": sha, "status": "timeout", "candidate_us": None,
    }) + "\n", encoding="utf-8")
    missed = annotate(work, text=f"remeasure {sha}", section="note")
    assert missed["reply"] == "recorded"
    assert missed["numbers"] is None
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({
            "kernel_sha256": sha, "status": "ok", "candidate_us": 4.5, "applied": True,
        }) + "\n")
    measured = annotate(work, text=f"remeasure {sha}", section="note")
    assert measured["reply"] == "already_measured"
    assert measured["numbers"]["candidate_us"] == 4.5
    assert measured["numbers"]["status"] == "ok"
    assert measured["numbers"]["applied"] is True


def test_notes_fallback_is_visible_to_later_duplicates_and_a_failed_dismissal_is_an_error(tmp_path: Path, capsys):
    """@brief NOTES.md 里的 fallback 参与消重；消重写不下去时是 NoteError，CLI 退出 5。"""
    work = tmp_path / "work"
    work.mkdir()
    (work / ".notes").write_text("not a directory\n", encoding="utf-8")
    text = "add paired compare subcommand to tcenv now"
    first = annotate(work, text=text, section="对系统的建议", now=10)
    assert first["fallback"] is True and first["reply"] == "recorded"
    assert "taic.note " in (work / "NOTES.md").read_text(encoding="utf-8")
    with pytest.raises(NoteError, match="persist the dismissal"):
        annotate(work, text=text, section="对系统的建议", now=20)
    assert not (work / ".notes" / "dismissed.json").exists()
    code = main(["--work", str(work), "--text", text, "--section", "对系统的建议"])
    assert code == 5
    assert "persist the dismissal" in capsys.readouterr().err


def test_concurrent_dismissals_keep_every_digest(tmp_path: Path):
    """@brief 两个进程写不同 digest 时，后写的不能盖掉先写的。"""
    import subprocess
    import sys

    path = tmp_path / "notes" / "dismissed.json"
    path.parent.mkdir()
    script = (
        "import sys\n"
        "from pathlib import Path\n"
        "from taic.note import _record_dismissal\n"
        "_record_dismissal(Path(sys.argv[1]), {"
        "'digest': sys.argv[2], 't': 1, 'mode': 'near', 'note': 'x', "
        "'prior_digest': '', 'prior_seq': None, 'prior_run': ''})\n"
    )
    processes = [
        subprocess.Popen([sys.executable, "-c", script, str(path), f"d{i:02d}"])
        for i in range(8)
    ]
    assert [process.wait(timeout=30) for process in processes] == [0] * 8
    stored = json.loads(path.read_text(encoding="utf-8"))
    assert set(stored["items"]) == {f"d{i:02d}" for i in range(8)}


def test_bad_dismissal_timestamp_does_not_crash_or_dismiss(tmp_path: Path):
    """@brief 模型可写的坏 t 不能让反馈读取崩溃，也不能当成一次消重。"""
    from taic.note import dismissal_for

    run = tmp_path / "run"
    (run / ".notes").mkdir(parents=True)
    digest = "abcdabcdabcdabcd"
    (run / ".notes" / "dismissed.json").write_text(json.dumps({
        "schema": "taic.note.dismissals.1",
        "items": {digest: {"t": "bad", "status": "info", "note": "nope"}},
    }), encoding="utf-8")
    assert dismissal_for(run, digest, seq=1) is None
    rows = _claim_rows(
        {"seq": 1, "t": 10, "items": [{
            "section": "对系统的建议", "digest": digest, "text": "hello", "audience": "system",
        }]},
        {},
        directory=run,
    )
    assert rows[0]["status"] == "pending"
    assert rows[0]["note"] == ""


def test_relative_nested_run_refreshes_the_indexed_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys):
    """@brief 相对 --run runs/exp/r1 按当前目录展开，立即刷新的是这场，不是 runs/runs/exp/r1。"""
    from taic.cli import main as taic_main

    directory, _log, _original, claim, _other, repeated = _indexed_suggestion(tmp_path, monkeypatch, "exp/r1")
    work = directory / "agent_work" / "s1"
    work.mkdir(parents=True)
    monkeypatch.chdir(tmp_path)
    code = taic_main([
        "note", "--work", str(work), "--run", "runs/exp/r1",
        "--section", "对系统的建议", "--text", repeated,
    ])
    assert code == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["reply"] == "duplicate" and printed["dismissed"] is True
    assert (directory / ".notes" / "dismissed.json").is_file()
    assert not (tmp_path / "runs" / "runs").exists()
    assert claim["digest"] in _suggestion_digests("pending")
    assert claim["digest"] not in _suggestion_digests("info")
    from observe.index import get_index
    from tests.shared.test_observe_feedback import _record

    with (directory / "feedback.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_record(parse_feedback(f"# 对系统的建议\n- {repeated}\n"), seq=8, t=45), ensure_ascii=False) + "\n")
    index = get_index()
    index.refresh_feedback(index._rel(directory), directory)
    rows = {
        int(row["seq"]): str(row["status"])
        for row in index.conn.execute(
            "SELECT seq, status FROM feedback_claims WHERE run_id = ? AND digest = ?",
            (index._rel(directory), claim["digest"]),
        )
    }
    assert rows[7] == "pending"
    assert rows[8] == "info"
