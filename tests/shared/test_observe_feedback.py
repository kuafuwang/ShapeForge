"""Feedback index, per-claim review, and escaped dashboard text. No GPU and no model calls."""

from __future__ import annotations

import json
import subprocess
import threading
from pathlib import Path

import pytest

from observe.actions import Refused, review_feedback
from observe.feedback import apply_review, apply_reviews, entries, preview_reviews, rejection, review_board, review_feed, review_similar, summary_from_index
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
    """@brief 解析模块级样例反馈文本 ``_TEXT``，供各用例复用同一份结构化结果。

    @return ``parse_feedback`` 的解析结果（含 kinds、sections、items、unverified_premises、raw）。
    """
    return parse_feedback(_TEXT)


def _record(parsed: dict, *, seq: int = 1, t: float = 10, label: str = "r1.p0") -> dict:
    """@brief 把解析结果包装成一条 feedback.jsonl 记录，并补齐顺序号、时间与任务元数据。

    @param parsed ``parse_feedback`` 的解析结果。
    @param seq 反馈顺序号。
    @param t 记录时间戳，用于制造新旧关系。
    @param label 候选标签，同时决定 work_dir 与候选目录名。
    @return 可直接写入 feedback.jsonl 的字典。
    """
    return {
        "format": "agent_feedback", "v": 1, "seq": seq, "t": t, "task": "sem_attention_odd_scale", "dtype": "fp16", "size": "paper",
        "label": label, "round": 1, "committee": "first", "provider": "deepseek", "model": "deepseek-flash",
        "effort": "high", "outcome": "evaluated", "exit": "normal", "best_speedup": 1.01, "kinds": parsed["kinds"],
        "work_dir": f"agent_work/{label}", "sections": parsed["sections"], "items": parsed["items"],
        "unverified_premises": parsed["unverified_premises"], "raw": parsed["raw"],
    }


def _run(root: Path, name: str, record: dict, *, started_at: float) -> Path:
    """@brief 在根目录下造一个已结束的 run：state.json、feedback.jsonl 与一份 candidate report。

    @param root runs 根目录。
    @param name run 目录名（即 run_id）。
    @param record 写进 feedback.jsonl 的记录。
    @param started_at run 的 started_at／updated_at 时间戳。
    @return 新建的 run 目录路径。
    """
    directory = root / name
    directory.mkdir(parents=True)
    (directory / "state.json").write_text(json.dumps({
        "task": "sem_attention_odd_scale", "dtype": "fp16", "phase": "done", "started_at": started_at, "updated_at": started_at,
    }))
    (directory / "feedback.jsonl").write_text(json.dumps(record, ensure_ascii=False) + "\n")
    folder = directory / "candidates" / f"000_{record['label']}"
    folder.mkdir(parents=True)
    # Schema 4 only: the review panel refuses to quote an old-format report's numbers as current
    # evidence, so the fixture builds the document the current evaluator would write.
    from tcenv.evaluate import EvalReport

    from ..support.orchestrator import stamp_report

    report = EvalReport(
        task="sem_attention_odd_scale", dtype="fp16", size="small", mode="full", seed=1, stage="done",
        status="ok", message="ok", speedup=1.2, target_id="apple-m4-max",
    )
    report.candidate_us = 770
    report.benchmark = {"candidate": {"median_us": 770, "cv": 0.01}}
    report.roofline = {"lower_bound_us": 414, "bound_kind": "reference_algorithm"}
    stamp_report(report, "sem_attention_odd_scale", "fp16", "small", "kernel void hit() {}\n")
    (folder / "kernel.metal").write_text("kernel void hit() {}\n", encoding="utf-8")
    (folder / "report.json").write_text(json.dumps(report.to_json()))
    return directory


def _open(tmp_path, monkeypatch):
    """@brief 建临时 runs 目录与环境变量，重置 observe 索引单例并返回目录与索引实例。

    @param tmp_path pytest 提供的临时目录，用作 runs 与各数据库的位置。
    @param monkeypatch pytest 的环境变量夹具，负责在用例结束后恢复原值。
    @return ``(runs 目录, 索引实例)`` 二元组。
    """
    runs = tmp_path / "runs"
    runs.mkdir()
    monkeypatch.setenv("AI_LOWERLING_RUNS_DIR", str(runs))
    monkeypatch.setenv("OBSERVE_DB", str(tmp_path / "observe.sqlite"))
    monkeypatch.setenv("AI_LOWERLING_BOOKS", str(tmp_path / "none.sqlite"))
    monkeypatch.setenv("TAIC_REVIEW_EVENTS", str(tmp_path / "review_events.jsonl"))
    import observe.index as index_mod

    index_mod._SINGLE = None
    return runs, index_mod.get_index()


def _digest(section: str) -> str:
    """@brief 取样例文本中指定章节第一条 claim 的 digest。

    @param section 章节名（如“需要帮助”）。
    @return 该章节第一条 claim 的 digest。
    @raises AssertionError 样例里找不到该章节时由用例自身抛出。
    """
    for item in _parsed()["items"]:
        if item["section"] == section:
            return item["digest"]
    raise AssertionError(section)


def test_pending_counts_system_and_premise_claims_and_review_changes_them(tmp_path, monkeypatch):
    """@brief 验证待处理数只统计系统建议与前提两类 claim，并按 kind 过滤；review 后待处理数随之变化。

    同时校验 entries 里候选 report 的来源与数值、未转义的原始文本保留，以及四类缺字段的 review
    请求（缺证据／缺 change_ref／缺 note／仍是 pending 却声称已验证）都会被拒。

    @param tmp_path pytest 提供的临时目录，用来放 runs 与数据库。
    @param monkeypatch pytest 的环境变量夹具，用于把路径指向临时目录。
    @raises Refused 缺 evidence、缺 change_ref、缺 note 或仍为 pending 的 review 请求被拒。
    """
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
    """@brief 验证 review board 列出每个 run 的 claim 正文，feed 带 target/task，且 dashboard 页面含 review 交互钩子。

    @param tmp_path pytest 提供的临时目录，用来放 runs 与数据库。
    @param monkeypatch pytest 的环境变量夹具，用于把路径指向临时目录。
    """
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
    assert {item["task"] for item in feed["upserts"]} == {"sem_attention_odd_scale"}
    page = Path(__file__).resolve().parents[2] / "observe" / "static" / "index.html"
    html = page.read_text(encoding="utf-8")
    assert "reviewHardware" in html
    assert "展开当前层" in html and "折叠当前层" in html and "reviewMenuActions" in html
    assert '<button type="button" id="feedback-badge"' in html
    assert 'window.open("/reviews"' in html
    assert 'params.set("since"' in html
    assert "patchReviewDom" in html
    assert "priorMarkup" in html and "此前结论" in html
    assert "similarMarkup" in html and "相似结论" in html


def test_a_review_records_the_git_style_actor_and_rejects_a_stale_expected_status(tmp_path, monkeypatch):
    """@brief 验证 review 记下 git 风格 actor 与来源：显式给 reviewer 时 source 为 explicit，未给时回退到 git 身份。

    @param tmp_path pytest 提供的临时目录，用来放 runs 与数据库。
    @param monkeypatch pytest 的环境变量与替换夹具，用于把 git 作者换成本地默认身份。
    @raises ValueError 调用时给的 expected_status 与当前状态不一致时抛出。
    """
    runs, index = _open(tmp_path, monkeypatch)
    parsed = _parsed()
    directory = _run(runs, "conv", _record(parsed), started_at=50)
    index.sync_known()
    digest = _digest("需要帮助")
    with pytest.raises(ValueError, match="expected"):
        apply_review(directory, 1, digest, {
            "status": "info", "verification": "unchecked", "expected_status": "adopted",
        })
    stored = apply_review(
        directory, 1, digest,
        {
            "status": "info", "verification": "unchecked", "expected_status": "pending",
            "batch_id": "batch-1",
        },
        reviewer="Reviewer <reviewer@example.test>", method="agent",
    )
    assert stored["reviewer"] == "Reviewer <reviewer@example.test>"
    assert stored["reviewer_source"] == "explicit"
    assert stored["method"] == "agent"
    assert stored["batch_id"] == "batch-1"
    assert stored["git_commit"]
    assert stored["history"][0]["reviewer"] == "Reviewer <reviewer@example.test>"
    row = next(c for c in entries(directory)["records"][0]["claims"] if c["digest"] == digest)
    assert row["reviewer"] == "Reviewer <reviewer@example.test>"
    from taic.git_identity import GitAuthor
    monkeypatch.setattr("taic.git_identity.author", lambda root=None: GitAuthor("Auto Reviewer", "auto@example.test", "git"))
    automatic = apply_review(
        directory, 1, digest,
        {"status": "info", "verification": "unchecked", "expected_status": "info"},
        method="script",
    )
    assert automatic["reviewer"] == "Auto Reviewer <auto@example.test>"
    assert automatic["reviewer_source"] == "git"
    assert automatic["method"] == "script"


def test_a_review_batch_is_previewed_and_written_atomically(tmp_path, monkeypatch):
    """@brief 验证批量 review 先预览不落盘，写入是原子的：行内状态不匹配时整批不写。

    成功后两行一次写入 feedback_review.json，均带同一个 batch_id 与 reviewer，
    并在 review 账本里产生两条可通过校验的事件。

    @param tmp_path pytest 提供的临时目录，用来放 runs、数据库与 review 账本。
    @param monkeypatch pytest 的环境变量夹具，用于把路径指向临时目录。
    @raises ValueError 批里第 2 行的 expected_status 已过期时不写入并抛出（提示 ``item 1``）。
    """
    runs, index = _open(tmp_path, monkeypatch)
    parsed = _parsed()
    directory = _run(runs, "conv", _record(parsed), started_at=50)
    index.sync_known()
    help_digest = _digest("需要帮助")
    premise = _digest("被否决的前提")
    review_path = directory / "feedback_review.json"
    before = review_path.read_text() if review_path.exists() else ""
    rows = [
        {"seq": 1, "digest": help_digest, "status": "info", "verification": "unchecked", "expected_status": "pending"},
        {"seq": 1, "digest": premise, "status": "rejected", "verification": "refuted", "note": "wrong probe", "expected_status": "pending"},
    ]
    preview = preview_reviews(directory, rows)
    assert [row["to_status"] for row in preview] == ["info", "rejected"]
    assert (review_path.read_text() if review_path.exists() else "") == before
    with pytest.raises(ValueError, match="item 1"):
        apply_reviews(directory, rows[:1] + [{**rows[1], "expected_status": "adopted"}], reviewer="Batch <b@example.test>")
    assert (review_path.read_text() if review_path.exists() else "") == before
    stored = apply_reviews(directory, rows, reviewer="Batch <b@example.test>", batch_id="batch-test")
    assert [row["status"] for row in stored] == ["info", "rejected"]
    assert all(row["batch_id"] == "batch-test" and row["reviewer"] == "Batch <b@example.test>" for row in stored)
    doc = json.loads(review_path.read_text())["claims"]
    assert doc[f"1:{help_digest}"]["status"] == "info"
    assert doc[f"1:{premise}"]["status"] == "rejected"
    from taic.review_ledger import verify
    ledger = verify(tmp_path / "review_events.jsonl")
    assert ledger["ok"] and ledger["events"] == 2


def test_the_feedback_cli_previews_then_writes_a_batch(tmp_path, monkeypatch, capsys):
    """@brief 验证 ``feedback`` 子命令：``--dry-run`` 不改文件，``--confirm`` 写入并带上 batch_id 与 reviewer 信息。

    @param tmp_path pytest 提供的临时目录，用来放 runs、数据库与批文件。
    @param monkeypatch pytest 的环境变量夹具，用于把路径指向临时目录。
    @param capsys pytest 的输出捕获夹具，用于断言命令行输出了 reviewer 相关说明。
    """
    from taic.cli import main

    runs, index = _open(tmp_path, monkeypatch)
    parsed = _parsed()
    _run(runs, "conv", _record(parsed), started_at=50)
    index.sync_known()
    batch = tmp_path / "batch.json"
    batch.write_text(json.dumps([{
        "seq": 1, "digest": _digest("需要帮助"), "status": "info",
        "verification": "unchecked", "expected_status": "pending",
    }]), encoding="utf-8")
    assert main(["feedback", "conv", "--file", str(batch), "--dry-run"]) == 0
    assert "expected_status" not in (runs / "conv" / "feedback_review.json").read_text() if (runs / "conv" / "feedback_review.json").exists() else True
    assert main(["feedback", "conv", "--file", str(batch), "--confirm", "--batch-id", "cli-batch"]) == 0
    doc = json.loads((runs / "conv" / "feedback_review.json").read_text())
    assert doc["claims"][f"1:{_digest('需要帮助')}"]["status"] == "info"
    assert doc["claims"][f"1:{_digest('需要帮助')}"]["batch_id"] == "cli-batch"
    output = capsys.readouterr().out
    assert "reviewer" in output


def test_open_subclaims_are_counted_without_changing_the_parent_pending_status(tmp_path, monkeypatch):
    """@brief 验证拆分出的子结论单独计数：父 claim 保持 info，未决子结论计入总数且按序入库。

    @param tmp_path pytest 提供的临时目录，用来放 runs 与数据库。
    @param monkeypatch pytest 的环境变量夹具，用于把路径指向临时目录。
    """
    runs, index = _open(tmp_path, monkeypatch)
    parsed = parse_feedback("# 对系统的建议\n- 请加一个配对报告。\n")
    _run(runs, "conv", _record(parsed), started_at=50)
    digest = parsed["items"][0]["digest"]
    apply_review(runs / "conv", 1, digest, {
        "status": "info", "verification": "unchecked", "note": "split",
        "subclaims": [
            {"text": "measured", "status": "adopted", "verification": "confirmed",
             "evidence_source": "probe", "evidence": "probe.py 1.0"},
            {"text": "still open", "status": "info", "verification": "unchecked", "note": "needs a report"},
        ],
    })
    index.sync_known()
    row = next(item for item in review_board(index.conn)["items"] if item["digest"] == digest)
    assert row["status"] == "info"
    assert row["open_subclaims"] and row["open_subclaims"][0]["text"] == "still open"
    subs = index.conn.execute(
        "SELECT status, text FROM feedback_subclaims WHERE run_id = 'conv' AND parent_digest = ? ORDER BY sub_index",
        (digest,),
    ).fetchall()
    assert [(row["status"], row["text"]) for row in subs] == [("adopted", "measured"), ("info", "still open")]
    assert summary_from_index(index.conn)["open_subclaims_total"] == 1
    assert summary_from_index(index.conn)["open_work_total"] == 1
    assert review_board(index.conn)["open_subclaims_total"] == 1
    assert review_feed(index.conn)["open_subclaims_total"] == 1


def test_a_subclaim_text_can_match_an_earlier_subclaim(tmp_path, monkeypatch):
    """@brief 验证子结论文本相同就能跨 run 匹配到更早的子结论，board 与 detail 都给出 prior。

    @param tmp_path pytest 提供的临时目录，用来放两个 run 与数据库。
    @param monkeypatch pytest 的环境变量夹具，用于把路径指向临时目录。
    """
    runs, index = _open(tmp_path, monkeypatch)
    first_text = parse_feedback("# 对系统的建议\n- first parent\n")
    second_text = parse_feedback("# 对系统的建议\n- second parent\n")
    _run(runs, "conv", _record(first_text, t=10), started_at=50)
    _run(runs, "conv2", _record(second_text, t=20), started_at=80)
    first_digest = first_text["items"][0]["digest"]
    second_digest = second_text["items"][0]["digest"]
    apply_review(runs / "conv", 1, first_digest, {
        "status": "info", "verification": "unchecked", "note": "split",
        "subclaims": [{"text": "shared open subclaim", "status": "info", "verification": "unchecked", "note": "first"}],
    })
    apply_review(runs / "conv2", 1, second_digest, {
        "status": "info", "verification": "unchecked", "note": "split",
        "subclaims": [{"text": "shared open subclaim", "status": "info", "verification": "unchecked", "note": "second"}],
    })
    index.sync_known()
    row = next(item for item in review_board(index.conn)["items"] if item["run_id"] == "conv2" and item["digest"] == second_digest)
    assert any(prior["run_id"] == "conv" and prior["text"] == "shared open subclaim" for prior in row["prior"])
    from observe.suggestions import detail
    full = detail("conv2", 1, second_digest)
    assert any(prior["run_id"] == "conv" and prior["text"] == "shared open subclaim" for prior in full["prior"])


def test_a_review_shows_prior_exact_digest_verdicts_scoped_to_task_and_target(tmp_path, monkeypatch):
    """@brief 验证同一 claim 在本 run 之前的历史结论会作为 prior 展示，且当前行不会被当自己的 prior。

    @param tmp_path pytest 提供的临时目录，用来放两个 run 与数据库。
    @param monkeypatch pytest 的环境变量夹具，用于把路径指向临时目录。
    """
    runs, index = _open(tmp_path, monkeypatch)
    parsed = _parsed()
    first = _run(runs, "conv", _record(parsed, t=10), started_at=50)
    second = _run(runs, "conv2", _record(parsed, t=20), started_at=80)
    index.sync_known()
    premise = _digest("被否决的前提")
    review_feedback("conv", 1, premise, {
        "status": "rejected", "verification": "refuted", "note": "same probe measured the other question",
    })
    index.refresh_feedback("conv", first)
    board = review_board(index.conn, status="pending")
    row = next(item for item in board["items"] if item["run_id"] == "conv2" and item["digest"] == premise)
    assert row["prior_summary"] == "此前 rejected"
    assert row["prior"][0]["run_id"] == "conv" and row["prior"][0]["status"] == "rejected"
    assert "other question" in row["prior"][0]["note"]
    feed = review_feed(index.conn, since=None)
    fed = next(item for item in feed["upserts"] if item["run_id"] == "conv2" and item["digest"] == premise)
    assert fed["prior"][0]["run_id"] == "conv"
    from observe.suggestions import detail
    full = detail("conv2", 1, premise)
    assert full["prior"][0]["run_id"] == "conv" and full["prior"][0]["status"] == "rejected"
    # The current row is not its own prior.
    current = next(item for item in feed["upserts"] if item["run_id"] == "conv" and item["digest"] == premise)
    assert current["prior"] == []


def test_the_first_feedback_page_defers_similar_and_a_run_query_computes_only_that_run(tmp_path, monkeypatch):
    """@brief 首屏快照默认不算语义近邻；按 run 查询只算那一场的待审 claim，显式 similar=True 保持旧行为。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch 用假的 similar_claims 记录调用，避免依赖 Apple 句向量。
    """
    runs, index = _open(tmp_path, monkeypatch)
    first = parse_feedback("# 对系统的建议\n- 矩阵单元和 SIMT ALU 吞吐相近。\n")
    second = parse_feedback("# 对系统的建议\n- 矩阵单元和 SIMT ALU 吞吐差不多。\n")
    _run(runs, "conv", _record(first, t=10), started_at=50)
    _run(runs, "conv2", _record(second, t=20), started_at=80)
    index.sync_known()
    calls = []

    def fake_similar(conn, *, digest, task, target_id, text, before_t=None, exclude_run="", exclude_seq=None, limit=3):
        calls.append(exclude_run)
        return [{"run_id": "conv", "seq": 1, "digest": "old", "text": "旧建议", "score": 0.99}]

    monkeypatch.setattr("observe.feedback.similar_claims", fake_similar)
    feed = review_feed(index.conn, since=None)
    assert calls == [], "首屏不应为全部待审 claim 算近邻"
    assert feed["similar_attached"] is False
    assert all("similar" not in row for row in feed["upserts"])
    review_feed(index.conn, since=None, similar=True)
    assert calls, "显式 similar=True 必须保留旧的 eager 行为"
    calls.clear()
    doc = review_similar(index.conn, "conv2")
    assert calls == ["conv2"], calls
    assert doc["run_id"] == "conv2"
    assert doc["items"][0]["similar"][0]["run_id"] == "conv"
    calls.clear()
    review_similar(index.conn, "conv")
    assert calls == ["conv"], "只能算被请求那一场，不能顺带算其他 run"


def test_one_run_similar_caps_at_two_hundred_claims_and_says_so(tmp_path, monkeypatch):
    """@brief 单场 ``/similar`` 也有界：250 条待审 claim 只算 200 条，并带 ``truncated``。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 的替换工具。
    """
    from observe import feedback, similarity

    # 本机有句向量时 200 条会真的去算（实测 2.9s）；这条只锁“单场有界 + truncated”，
    # 把检索关掉，别让性能测试拖慢快套件。
    monkeypatch.setattr(similarity, "available", lambda: False)
    runs = tmp_path / "runs"
    runs.mkdir()
    monkeypatch.setenv("AI_LOWERLING_RUNS_DIR", str(runs))
    monkeypatch.setenv("OBSERVE_DB", str(tmp_path / "observe.sqlite"))
    monkeypatch.setenv("AI_LOWERLING_BOOKS", str(tmp_path / "none.sqlite"))
    import observe.index as index_mod

    index_mod._SINGLE = None
    index = index_mod.get_index()
    directory = runs / "many"
    directory.mkdir(parents=True)
    (directory / "state.json").write_text(json.dumps({
        "task": "sem_attention_odd_scale", "dtype": "fp16", "phase": "done", "started_at": 10, "updated_at": 10,
    }), encoding="utf-8")
    # 一条记录最多 50 项，还按“去数字后”的文本去重（digest 不含数字）；250 条 claim 要用 5 条
    # 记录，且每条正文得带不同的字母尾巴，否则 50 项会被折叠成十几项。
    def tag(number: int) -> str:
        letters = ""
        while True:
            letters = chr(ord("a") + number % 26) + letters
            number //= 26
            if not number:
                return letters

    lines = []
    for record_seq in range(5):
        bullets = "".join(
            f"- 记录 {tag(record_seq * 50 + number)} 的建议：把 {tag(record_seq * 50 + number)} 步做窄。\n"
            for number in range(50)
        )
        parsed = parse_feedback("# 对系统的建议\n" + bullets)
        assert len(parsed["items"]) == 50
        lines.append(json.dumps(_record(parsed, seq=record_seq + 1, t=10.0 + record_seq), ensure_ascii=False))
    (directory / "feedback.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    index.sync_known()
    claims = index.conn.execute("SELECT COUNT(*) FROM feedback_claims WHERE run_id = 'many'").fetchone()[0]
    assert claims == 250, claims
    doc = feedback.review_similar(index.conn, "many")
    assert doc["truncated"] is True
    assert len(doc["items"]) == feedback.REVIEW_SIMILAR_MAX_CLAIMS == 200
    assert all(set(item) == {"run_id", "seq", "digest", "similar"} for item in doc["items"])


def test_a_record_with_two_claims_pages_by_claim_without_losing_one(tmp_path, monkeypatch):
    """@brief 回归：快照分页单位是 claim。一条 record 下两条 claim 时，limit=1 必须能翻到两条。

    过去 ``total`` 取 ``feedback_items`` 的行数、分页却按 ``feedback_claims`` 走，于是
    total=1、next_offset=None，前端翻页循环停在第 1 页，第二条 claim 永远看不到。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 的替换工具。
    """
    import observe.index as index_mod

    runs = tmp_path / "runs"
    runs.mkdir()
    monkeypatch.setenv("AI_LOWERLING_RUNS_DIR", str(runs))
    monkeypatch.setenv("OBSERVE_DB", str(tmp_path / "observe.sqlite"))
    monkeypatch.setenv("AI_LOWERLING_BOOKS", str(tmp_path / "none.sqlite"))
    index_mod._SINGLE = None
    index = index_mod.get_index()
    # 一条 feedback 记录（run_id=r, seq=1）下两条 claim（d1/d2）。
    index.conn.execute(
        "INSERT INTO feedback_items (run_id, seq, t, task, dtype, size, label)"
        " VALUES ('r', 1, 2.0, 'relu', 'fp16', 'paper', 'r1')",
    )
    index.conn.executemany(
        "INSERT INTO feedback_claims (run_id, seq, digest, section, audience, premise, status, rev)"
        " VALUES ('r', 1, ?, '对系统的建议', 'system', 'p', 'pending', 1)",
        [("d1",), ("d2",)],
    )
    index.conn.commit()

    first = review_feed(index.conn, limit=1)
    assert first["total"] == 2, first["total"]
    assert first["next_offset"] == 1, first["next_offset"]
    second = review_feed(index.conn, limit=1, offset=first["next_offset"])
    assert second["total"] == 2 and second["next_offset"] is None, second
    # 逐页取完不重不漏：前端就是按 next_offset 循环的。
    seen = [row["digest"] for row in first["upserts"] + second["upserts"]]
    assert sorted(seen) == ["d1", "d2"], seen


def test_semantic_similarity_finds_a_rephrased_claim_in_the_same_task_and_target(tmp_path, monkeypatch):
    """@brief 验证同一 task/target 下改写过的 claim 能被语义相似度匹配（score 不低于 0.975）。

    若本机不可用 Apple NaturalLanguage 句向量，用例直接 skip。

    @param tmp_path pytest 提供的临时目录，用来放两个 run 与数据库。
    @param monkeypatch pytest 的环境变量夹具，用于把路径指向临时目录。
    """
    from observe.similarity import available

    if not available():
        pytest.skip("Apple NaturalLanguage sentence embeddings are unavailable")
    runs, index = _open(tmp_path, monkeypatch)
    first_text = "矩阵单元和 SIMT ALU 吞吐相近，而且工作相加。"
    second_text = "矩阵单元和 SIMT ALU 吞吐差不多，而且两者工作相加。"
    first = parse_feedback("# 对系统的建议\n- " + first_text + "\n")
    second = parse_feedback("# 对系统的建议\n- " + second_text + "\n")
    _run(runs, "conv", _record(first, t=10), started_at=50)
    _run(runs, "conv2", _record(second, t=20), started_at=80)
    index.sync_known()
    board = review_board(index.conn, status="pending")
    row = next(item for item in board["items"] if item["run_id"] == "conv2")
    assert row["similar"][0]["run_id"] == "conv"
    assert row["similar"][0]["score"] >= 0.975
    from observe.suggestions import detail
    full = detail("conv2", 1, second["items"][0]["digest"])
    assert full["similar"][0]["run_id"] == "conv"


def test_a_grown_log_is_read_from_the_cursor_and_a_review_does_not_reread_it(tmp_path, monkeypatch):
    """@brief 验证增长的 feedback 日志从记录的字节游标续读，而不是解析整份；review 之后也不再重读日志。

    用例把 ``read_appended_records`` 包起来记录起始偏移，并把整份读取入口换成会失败的桩。

    @param tmp_path pytest 提供的临时目录，用来放 runs 与数据库。
    @param monkeypatch pytest 的环境变量与替换夹具，用于包裹读游标函数并封死整份读取入口。
    """
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
    """@brief 验证内容未变的重读不会推高 feedback revision，而一次 review 只产生一行增量。

    @param tmp_path pytest 提供的临时目录，用来放 runs 与数据库。
    @param monkeypatch pytest 的环境变量夹具，用于把路径指向临时目录。
    """
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
    """@brief 验证两个线程并发 review 同一 run 的不同 claim 都不报错，两份结论都写入。

    @param tmp_path pytest 提供的临时目录，用来放 runs 与数据库。
    @param monkeypatch pytest 的环境变量夹具，用于把路径指向临时目录。
    """
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


def test_the_resolve_existence_check_runs_inside_the_write_lock(tmp_path, monkeypatch):
    """@brief 回归：``Index.resolve`` 的行存在性检查必须在写锁内。

    锁外读共享写连接时，并行请求的写事务会让这条 SELECT 瞬时看不到已提交的行：
    ``test_two_reviews_of_one_run_both_land`` 因此偶发 404 ``no such run``（六进程压力下实测
    约 1% 失效）。用 SQL trace 断言检查语句执行时当前线程持有 ``_WRITE``。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 的环境变量夹具。
    """
    import observe.index as index_mod

    runs, index = _open(tmp_path, monkeypatch)
    _run(runs, "conv", _record(_parsed()), started_at=50)
    index.sync_known()
    seen: list[tuple[str, bool]] = []
    index.conn.set_trace_callback(lambda sql: seen.append((sql, index_mod._WRITE.locked())))
    try:
        resolved = index.resolve("conv")
    finally:
        index.conn.set_trace_callback(None)
    assert resolved is not None and resolved.name == "conv", resolved
    checks = [item for item in seen if item[0].lstrip().lower().startswith("select 1 from runs where run_id")]
    assert checks, seen
    assert all(held for _sql, held in checks), checks


def test_a_claim_restated_by_a_later_run_is_flagged(tmp_path, monkeypatch):
    """@brief 验证已被处理过的 claim 被后来的 run 重述时标记为 recurrence，原 run 不算重复。

    @param tmp_path pytest 提供的临时目录，用来放先后两个 run 与数据库。
    @param monkeypatch pytest 的环境变量夹具，用于把路径指向临时目录。
    """
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
    """@brief 验证 dashboard 的 ``esc`` 会把反馈文本转义：用 node 执行页面里的函数，HTML 被转成实体。

    同时确认合法的 review（带证据的 adopted）不会被拒。
    """
    page = Path("observe/static/index.html").read_text()
    assert "esc(claim.text)" in page and "esc(rawWindow.text)" in page and "tailWindow(record.raw)" in page
    start = page.index("function esc(value)")
    end = page.index("\n}", start) + 2
    script = page[start:end] + "\nprocess.stdout.write(esc('<img onerror=\"alert(1)\">'))"
    rendered = subprocess.run(["node", "-e", script], check=True, capture_output=True, text=True).stdout
    assert rendered == "&lt;img onerror=&quot;alert(1)&quot;&gt;"
    assert rejection({"status": "adopted", "verification": "confirmed", "evidence_source": "probe", "evidence": "15.09"}) is None


def test_info_and_obsolete_are_legal_review_states_with_a_reason():
    """@brief 验证 info 与 obsolete 两种状态的合法性规则：info 只能 unchecked，obsolete 必须带 note。"""
    assert rejection({"status": "info", "verification": "unchecked"}) is None
    assert rejection({"status": "info", "verification": "confirmed"}) is not None
    assert rejection({"status": "obsolete", "verification": "unchecked", "note": "duplicate of a"}) is None
    assert rejection({"status": "obsolete", "verification": "unchecked"}) is not None


def test_a_parser_upgrade_retires_a_removed_claim_with_its_review_status(tmp_path, monkeypatch):
    """@brief 验证解析器版本变化后消失的 claim 会留下 tombstone，保留它原本的 review 状态并记下原因。

    @param tmp_path pytest 提供的临时目录，用来放 runs 与数据库。
    @param monkeypatch pytest 的环境变量夹具，用于把路径指向临时目录。
    """
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
    """@brief 验证 fbv4 重算时探针文件已恢复就把前提从降级状态重判为 verified，同时保留 review 键（digest）。

    @param tmp_path pytest 提供的临时目录，用来放 runs 与数据库。
    @param monkeypatch pytest 的环境变量夹具，用于把路径指向临时目录。
    """
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
    """@brief 验证新索引上 fbv4 的空占位 claim（“无。”）也会留下 obsolete tombstone，而正常 claim 不受影响。

    @param tmp_path pytest 提供的临时目录，用来放 runs 与数据库。
    @param monkeypatch pytest 的环境变量夹具，用于把路径指向临时目录。
    """
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
    """@brief 验证 harness 面板同时区分已 review／未 review 及子结论、固化状态等文案与样式钩子。"""
    page = Path("observe/static/index.html").read_text()
    for needle in ("未 review", "review 有用", "review 不建议", "h-yes", "h-no", "reviewed_active", "reviewer", "P2 固化", "P2 候选", "change_ref", "固化到本地", "本地已固化", "promotion_commit", "自由文本系统建议", "人确认后转结构化", "送 LLM review", "convert-modal", "已实测 / 探针可读", "未验证 / 探针不可读", "fb-subclaim"):
        assert needle in page


def test_compact_suggestions_group_by_digest_and_fetch_one_full_text(tmp_path, monkeypatch):
    """@brief 验证紧凑列表按 digest 分组、保留总数与截断预览，详情接口返回完整正文。

    @param tmp_path pytest 提供的临时目录，用来放 runs 与数据库。
    @param monkeypatch pytest 的环境变量夹具，用于把路径指向临时目录。
    """
    from observe.suggestions import compact, detail

    runs, index = _open(tmp_path, monkeypatch)
    parsed = parse_feedback("# 对系统的建议\n- 请加一个配对报告。\n")
    _run(runs, "conv", _record(parsed), started_at=50)
    index.sync_known()
    digest = parsed["items"][0]["digest"]
    compact_doc = compact(task="sem_attention_odd_scale", status="pending", limit=10, preview=5)
    assert compact_doc["total_claims"] == 1
    group = compact_doc["groups"][0]
    assert group["digest"] == digest and group["runs"] == 1
    assert group["preview"].endswith("…") and len(group["preview"]) <= 6
    full = detail("conv", 1, digest)
    assert full["text"] == "请加一个配对报告。"
    assert full["converted_key"] == "" and full["reviews"] == []


def test_the_suggestions_http_endpoints_are_compact_and_detailed(tmp_path, monkeypatch):
    """@brief 验证 ``/api/v2/suggestions`` 返回紧凑分组、``/api/v2/suggestion`` 返回单条完整正文。

    @param tmp_path pytest 提供的临时目录，用来放 runs 与数据库。
    @param monkeypatch pytest 的环境变量夹具，用于把路径指向临时目录。
    """
    from starlette.testclient import TestClient

    from observe.app import app

    runs, index = _open(tmp_path, monkeypatch)
    parsed = parse_feedback("# 对系统的建议\n- 请加一个配对报告。\n")
    _run(runs, "conv", _record(parsed), started_at=50)
    index.sync_known()
    digest = parsed["items"][0]["digest"]
    with TestClient(app) as client:
        compact_response = client.get("/api/v2/suggestions", params={"task": "sem_attention_odd_scale", "preview": 8, "limit": 5})
        assert compact_response.status_code == 200
        doc = compact_response.json()
        assert doc["groups"][0]["digest"] == digest
        assert len(doc["groups"][0]["preview"]) <= 9
        full = client.get("/api/v2/suggestion", params={"run": "conv", "seq": 1, "digest": digest})
        assert full.status_code == 200 and full.json()["text"] == "请加一个配对报告。"


def test_compact_suggestions_page_over_the_whole_filtered_set(tmp_path, monkeypatch):
    """@brief 验证紧凑列表分页覆盖整个过滤集合：两页拼接后的 digest 集正好等于全部建议。

    @param tmp_path pytest 提供的临时目录，用来放 runs 与数据库。
    @param monkeypatch pytest 的环境变量夹具，用于把路径指向临时目录。
    """
    from observe.suggestions import compact

    runs, index = _open(tmp_path, monkeypatch)
    parsed = parse_feedback("# 对系统的建议\n- 第一条建议。\n- 第二条建议。\n- 第三条建议。\n")
    _run(runs, "conv", _record(parsed), started_at=50)
    index.sync_known()
    first = compact(task="sem_attention_odd_scale", limit=2, preview=20)
    assert first["total_claims"] == 3 and first["total_groups"] == 3
    assert len(first["groups"]) == 2 and first["truncated"] is True
    assert first["next_offset"] == 2
    second = compact(task="sem_attention_odd_scale", limit=2, offset=first["next_offset"], preview=20)
    assert len(second["groups"]) == 1 and second["truncated"] is False
    assert second["next_offset"] is None
    assert {row["digest"] for row in first["groups"] + second["groups"]} == {
        item["digest"] for item in parsed["items"] if item["section"] == "对系统的建议"
    }


def test_compact_does_not_attribute_an_older_conversion_to_the_newest_claim(tmp_path, monkeypatch):
    """@brief 验证更早的转换不会被算到最新那条 claim 头上：分组取首条代表并标明链接范围，而 detail 仍能查到旧 key。

    @param tmp_path pytest 提供的临时目录，用来放 runs 与数据库。
    @param monkeypatch pytest 的环境变量夹具，用于把路径指向临时目录。
    """
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
    group = compact(task="sem_attention_odd_scale", limit=5)["groups"][0]
    assert group["first"]["seq"] == 2
    assert group["link_scope"] == "first (representative claim only)"
    assert group["converted_key"] == ""
    assert detail("conv", 1, digest)["converted_key"] == "older-key"


def test_compact_links_the_exact_seq_not_only_the_first_digest_match(tmp_path, monkeypatch):
    """@brief 验证转换链接按精确的 seq 匹配而不是只取第一个同 digest 的提案。

    @param tmp_path pytest 提供的临时目录，用来放 runs 与数据库。
    @param monkeypatch pytest 的环境变量夹具，用于把路径指向临时目录。
    """
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
    group = compact(task="sem_attention_odd_scale", limit=5)["groups"][0]
    assert group["converted_key"] == "second-key"
    assert group["first"]["seq"] == 2
    assert detail("conv", 1, digest)["converted_key"] == "first-key"
    assert detail("conv", 2, digest)["converted_key"] == "second-key"


def test_a_split_claim_keeps_subclaim_statuses_and_a_parked_parent(tmp_path, monkeypatch):
    """@brief 验证拆分后的 claim 保留各子结论的状态，父 claim 停在 info，子结论继承 backend/target_id。

    并确认非法组合会被拒：子结论全为已证实却把父状态改成 adopted，或子结论状态名不存在。

    @param tmp_path pytest 提供的临时目录，用来放 runs 与数据库。
    @param monkeypatch pytest 的环境变量夹具，用于把路径指向临时目录。
    """
    from observe.feedback import apply_review

    runs, index = _open(tmp_path, monkeypatch)
    parsed = parse_feedback("# 对系统的建议\n- 请加一个配对报告。\n")
    _run(runs, "conv", _record(parsed), started_at=50)
    (runs / "conv" / "run_config.json").write_text(json.dumps({
        "target": {"id": "apple-m4-max", "backend": "metal"},
    }))
    digest = parsed["items"][0]["digest"]
    body = {
        "status": "info", "verification": "unchecked", "note": "split into subclaims",
        "subclaims": [
            {"text": "measured fact", "status": "adopted", "verification": "confirmed",
             "evidence_source": "probe", "evidence": "probe.py 1.0"},
            {"text": "refuted claim", "status": "rejected", "verification": "refuted",
             "note": "measurement says the opposite"},
        ],
    }
    assert rejection(body) is None
    stored = apply_review(runs / "conv", 1, digest, body)
    assert stored["status"] == "info" and len(stored["subclaims"]) == 2
    assert stored["backend"] == "metal" and stored["target_id"] == "apple-m4-max"
    assert all(s["backend"] == "metal" and s["target_id"] == "apple-m4-max" for s in stored["subclaims"])
    index.sync_known()
    claim = next(c for c in entries(runs / "conv")["records"][0]["claims"] if c["digest"] == digest)
    assert [s["status"] for s in claim["subclaims"]] == ["adopted", "rejected"]
    assert claim["status"] == "info"

    assert rejection({**body, "status": "adopted", "verification": "confirmed"}) is not None
    assert rejection({**body, "subclaims": [{"text": "x", "status": "bogus", "verification": "unchecked"}]}) is not None
