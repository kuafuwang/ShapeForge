"""CPU tests for the feedback-evolution milestones. No GPU and no model calls."""

from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path

import pytest

from taic.feedback_topics import TopicError, announce_open_tool_topics, close_topic, record_tool_topic, refresh_stale_hint
from taic.paired_compare import (
    append_nomination,
    compare_callables,
    conclude_samples,
    quota_debit,
    record_nomination_and_debit,
    run_selected,
    select_nominations,
)
from taic.theory_record import validate_record


def _sha() -> str:
    return "a" * 64


def _other() -> str:
    return "b" * 64


def test_the_same_kernel_interval_contains_one():
    """@brief 同一颗假 runner 比自己时，比值区间包含 1，结论是没差。"""
    left = "ab" * 32
    document = compare_callables(lambda k: 1e-4 * k, lambda k: 1e-4 * k, sha_a=left, sha_b=left, reps=6, seed=7)
    lo, hi = document["interval"]
    assert lo <= 1.0 <= hi
    assert document["conclusion"] == "no_diff"
    assert document["changes_seed"] is False
    assert document["rounds"] == 6


def test_a_faster_runner_is_reported_in_the_right_direction():
    """@brief 一快一慢时，更快的一侧方向正确。"""
    document = compare_callables(
        lambda k: 1e-4 * k, lambda k: 2e-4 * k, sha_a="ab" * 32, sha_b="cd" * 32, reps=6, seed=7,
    )
    assert document["conclusion"] == "diff"
    assert document["direction"] == "a_faster"
    assert document["interval"][1] < 0.98


def test_a_noisy_pair_is_too_noisy():
    """@brief 比值跨过没差区间时，结论是噪声太大。"""
    flips = iter([1e-4, 4e-4] * 8)

    def noisy(k: int) -> float:
        return next(flips) * k

    document = compare_callables(noisy, lambda k: 2e-4 * k, sha_a="ab" * 32, sha_b="cd" * 32, reps=8, seed=7)
    assert document["conclusion"] == "too_noisy"
    lo, hi = document["interval"]
    assert lo < 0.98 < hi or hi > 1.02 and lo < 1.02


def test_too_few_rounds_are_too_noisy():
    """@brief 不足两轮不能给出有差或没差。"""
    document = conclude_samples([1.0], [1.0], sha_a="ab" * 32, sha_b="cd" * 32)
    assert document["conclusion"] == "too_noisy"
    assert document["interval"] is None


def _kernel(directory: Path, name: str, body: bytes) -> tuple[str, str]:
    path = directory / name
    path.write_bytes(body)
    return hashlib.sha256(body).hexdigest(), str(path)


def test_nomination_order_does_not_change_the_host_choice(tmp_path: Path):
    """@brief 模型打乱提名顺序、重复 SHA 对，宿主选中的那一对不变。自报 flag 不算文件。"""
    slow, slow_path = _kernel(tmp_path, "slow.metal", b"slow")
    fast, fast_path = _kernel(tmp_path, "fast.metal", b"fast")
    other, other_path = _kernel(tmp_path, "other.metal", b"other")
    preferred = {
        "sha_a": fast, "sha_b": slow, "path_a": fast_path, "path_b": slow_path,
        "conclusion": "no_diff", "cited_by_rejected_premise": True,
        "files_present": False, "identity_ok": False,
    }
    plain = {
        "sha_a": other, "sha_b": slow, "path_a": other_path, "path_b": slow_path,
        "conclusion": "diff", "cited_by_rejected_premise": False,
        "files_present": True, "identity_ok": True,
    }
    duplicate = dict(preferred)
    forward = select_nominations([plain, preferred, duplicate])
    backward = select_nominations([duplicate, preferred, plain])
    assert forward["selected"] == backward["selected"]
    assert forward["selected"][0]["sha_pair"] == sorted([fast, slow])
    assert forward["nominations"] == 3
    assert forward["reviewed"] == 0
    assert forward["uncovered_count"] >= 1
    assert any(item["omit_reason"] == "duplicate_sha_pair" for item in forward["uncovered"])


def test_missing_identity_and_existing_evidence_do_not_spend_the_one_slot(tmp_path: Path):
    """@brief 缺文件、身份错误、已有宿主证据都不占用 K=1。自报 flag 不能补上文件。"""
    good_a, good_a_path = _kernel(tmp_path, "good-a.metal", b"good-a")
    good_b, good_b_path = _kernel(tmp_path, "good-b.metal", b"good-b")
    flag_a, flag_a_path = _kernel(tmp_path, "flag-a.metal", b"flag-a")
    flag_b, flag_b_path = _kernel(tmp_path, "flag-b.metal", b"flag-b")
    present = tmp_path / "present.metal"
    present.write_bytes(b"present")
    good = {
        "sha_a": good_a, "sha_b": good_b, "path_a": good_a_path, "path_b": good_b_path,
        "conclusion": "diff", "files_present": False, "identity_ok": False,
    }
    flagged = {
        "sha_a": flag_a, "sha_b": flag_b, "path_a": flag_a_path, "path_b": flag_b_path,
        "has_host_evidence": True, "conclusion": "no_diff", "cited_by_rejected_premise": True,
        "files_present": False, "identity_ok": False,
    }
    rows = [
        {"sha_a": "66" * 32, "sha_b": "77" * 32, "path_a": "", "path_b": "", "files_present": True, "identity_ok": True, "conclusion": "no_diff", "cited_by_rejected_premise": True},
        {"sha_a": "88" * 32, "sha_b": "99" * 32, "path_a": str(present), "path_b": str(present), "files_present": True, "identity_ok": True, "conclusion": "regressed", "cited_by_rejected_premise": True},
        flagged,
        good,
    ]
    ignored = select_nominations(rows)
    assert ignored["reviewed"] == 0
    assert ignored["selected"][0]["sha_pair"] == sorted([flag_a, flag_b])
    chosen = select_nominations(rows, host_pairs={tuple(sorted([flag_a, flag_b]))})
    assert chosen["reviewed"] == 0
    assert chosen["selected"][0]["sha_pair"] == sorted([good_a, good_b])
    reasons = {item["omit_reason"] for item in chosen["uncovered"]}
    assert {"missing_file", "identity_mismatch", "equivalent_host_evidence"} <= reasons


def test_no_eligible_nomination_does_not_call_the_timer():
    """@brief 没有合格提名时不跑 GPU。空路径配上自报 flag 仍然不合格。"""
    chosen = select_nominations([
        {"sha_a": "11" * 32, "sha_b": "22" * 32, "path_a": "", "path_b": "", "files_present": True, "identity_ok": True},
    ])

    def boom() -> float:
        raise AssertionError("timer must not run")

    result = run_selected(chosen, boom)
    assert result["ran_gpu"] is False
    assert result["gpu_seconds"] == 0.0
    assert result["selected"] == []


def test_a_preferred_regressed_branch_without_host_evidence_wins_the_slot(tmp_path: Path):
    """@brief 尚无宿主证据的 no_diff/regressed 前提分支优先于按 SHA 更小的其它提名。"""
    later, later_path = _kernel(tmp_path, "later.metal", b"later-kernel")
    early, early_path = _kernel(tmp_path, "early.metal", b"early-kernel")
    mid, mid_path = _kernel(tmp_path, "mid.metal", b"mid-kernel")
    chosen = select_nominations([
        {"sha_a": early, "sha_b": mid, "path_a": early_path, "path_b": mid_path, "conclusion": "diff", "files_present": False, "identity_ok": False},
        {
            "sha_a": later, "sha_b": mid, "path_a": later_path, "path_b": mid_path, "conclusion": "regressed",
            "cited_by_rejected_premise": True, "files_present": False, "identity_ok": False,
        },
    ])
    assert chosen["selected"][0]["sha_pair"] == sorted([later, mid])
    assert chosen["nominations"] == 2
    assert any(item["omit_reason"] == "over_k" for item in chosen["uncovered"])


def test_file_bytes_must_match_the_nominated_sha(tmp_path: Path):
    """@brief 磁盘上的文件摘要和提名不一致时，这一对不占名额。"""
    good = tmp_path / "good.metal"
    bad = tmp_path / "bad.metal"
    good.write_bytes(b"kernel-a")
    bad.write_bytes(b"kernel-b")
    sha_good = hashlib.sha256(b"kernel-a").hexdigest()
    chosen = select_nominations([
        {"sha_a": sha_good, "sha_b": "0" * 64, "path_a": str(good), "path_b": str(bad), "conclusion": "no_diff", "cited_by_rejected_premise": True},
    ])
    assert chosen["selected"] == []
    assert chosen["uncovered"][0]["omit_reason"] == "identity_mismatch"


def test_a_tampered_nomination_line_with_empty_paths_is_not_selected(tmp_path: Path):
    """@brief 直接改写提名文件、空路径配上自报 flag，不占 K=1，也不启动重跑。"""
    from taic.paired_compare import NOMINATION_NAME, host_select

    directory = tmp_path / ".selfcheck"
    directory.mkdir()
    forged = {
        "sha_a": "ab" * 32,
        "sha_b": "cd" * 32,
        "path_a": "",
        "path_b": "",
        "files_present": True,
        "identity_ok": True,
        "conclusion": "no_diff",
        "cited_by_rejected_premise": True,
    }
    (directory / NOMINATION_NAME).write_text(json.dumps(forged) + "\n", encoding="utf-8")

    def boom(_selected):
        raise AssertionError("a nomination without files must not start a worker")

    saved = host_select(directory, rerun=boom)
    assert saved["selected"] == []
    assert saved["ran_gpu"] is False
    assert saved["reviewed"] == 0
    assert saved["uncovered"][0]["omit_reason"] == "missing_file"


def test_nomination_file_keeps_both_shas_and_the_quota_debit_does_not(tmp_path: Path):
    """@brief 自测行写在工作目录的独立文件里；配额记账行不含两个 SHA。"""
    directory = tmp_path / ".selfcheck"
    left, right = "ab" * 32, "cd" * 32
    path = append_nomination(directory, {"sha_a": left, "sha_b": right, "conclusion": "no_diff"})
    stored = json.loads(path.read_text(encoding="utf-8"))
    assert stored["sha_a"] == left and stored["sha_b"] == right
    assert stored["host_evidence"] is False
    debit = quota_debit(0.25)
    assert "sha_a" not in debit and "sha_b" not in debit
    assert debit["kind"] == "paired_compare_quota"


def test_a_full_quota_refuses_the_nomination_before_writing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """@brief 配额用尽时不写提名。"""
    directory = tmp_path / ".selfcheck"
    directory.mkdir()
    monkeypatch.setattr("tcenv.selfcheck.budget_reason", lambda _directory: "quota spent")
    with pytest.raises(Exception, match="quota spent"):
        record_nomination_and_debit(directory, {"sha_a": "ab" * 32, "sha_b": "cd" * 32}, elapsed_s=0.1)
    assert not (directory / "paired_comparisons.jsonl").exists()


def test_an_unimplemented_backend_is_unsupported_and_does_not_call_runners():
    """@brief 没实现的后端返回不支持。shared 测试不导入 Metal。"""
    from taic.paired_compare import backend_capability

    refused = backend_capability("qingwei")
    assert refused["supported"] is False
    assert "qingwei" in refused["reason"]


def test_close_topic_writes_nothing_when_one_claim_does_not_match(tmp_path: Path):
    """@brief 整批校验，一条对不上就整份不写。"""
    ledger = tmp_path / "topics.json"
    record_tool_topic(
        ledger, "eval-validate", head="abc", dirty=False, modules={"tcenv/cli.py": "1" * 64},
        claims=[
            {"run_id": "r", "seq": 1, "digest": "d1", "status": "pending"},
            {"run_id": "r", "seq": 2, "digest": "d2", "status": "info"},
        ],
    )
    before = ledger.read_text(encoding="utf-8")
    with pytest.raises(TopicError, match="item 1"):
        close_topic(
            ledger, "eval-validate", "commit:" + "a" * 12, confirm=True,
            live_status={("r", 1, "d1"): "pending", ("r", 2, "d2"): "rejected"},
        )
    assert ledger.read_text(encoding="utf-8") == before
    with pytest.raises(TopicError, match="missing from the live status"):
        close_topic(
            ledger, "eval-validate", "commit:" + "a" * 12, confirm=True,
            live_status={("r", 1, "d1"): "pending"},
        )
    assert ledger.read_text(encoding="utf-8") == before
    preview = close_topic(ledger, "eval-validate", "commit:" + "b" * 12, dry_run=True)
    assert preview["closed"] is False
    assert ledger.read_text(encoding="utf-8") == before
    done = close_topic(ledger, "eval-validate", "commit:" + "c" * 12, confirm=True)
    assert done["closed"] is True
    assert json.loads(ledger.read_text(encoding="utf-8"))["topics"]["eval-validate"]["status"] == "closed"


def test_a_hash_change_without_coverage_is_only_a_stale_hint(tmp_path: Path):
    """@brief 模块摘要变了但没有覆盖证据时，只产生 possibly_stale，不关单。"""
    ledger = tmp_path / "topics.json"
    record_tool_topic(
        ledger, "probe", head="abc", dirty=True, modules={"taic/cli.py": "a" * 64},
        claims=[{"run_id": "r", "seq": 1, "digest": "d", "status": "pending"}],
    )
    hinted = refresh_stale_hint(
        ledger, "probe", modules_now={"taic/cli.py": "b" * 64},
        coverage={"exercised": False, "new_reports": 0}, observed_at="2026-11-16T00:00:00Z",
    )
    assert hinted["stale_hint"] == "possibly_stale"
    assert hinted["status"] == "open"
    assert hinted["coverage"] == "not_exercised"
    assert hinted["zero_reproduction"] is False
    assert hinted["fixed"] is False
    assert hinted["change_ref"] == ""
    assert hinted["before"]["taic/cli.py"] != hinted["after"]["taic/cli.py"]


def test_zero_reports_without_exercising_the_path_are_not_a_fix(tmp_path: Path):
    """@brief 没有调用该工具时，零上报不能写成已修复。"""
    ledger = tmp_path / "topics.json"
    record_tool_topic(
        ledger, "probe", head="abc", dirty=False, modules={"taic/cli.py": "a" * 64},
        claims=[{"run_id": "r", "seq": 1, "digest": "d", "status": "info"}],
    )
    hinted = refresh_stale_hint(
        ledger, "probe", modules_now={"taic/cli.py": "a" * 64},
        coverage={"exercised": False, "new_reports": 0}, observed_at="t",
    )
    assert hinted["coverage"] == "not_exercised"
    assert hinted["fixed"] is False
    assert "fixed" not in json.dumps({key: hinted[key] for key in ("coverage", "zero_reproduction", "stale_hint")}) or hinted["fixed"] is False
    covered = refresh_stale_hint(
        ledger, "probe", modules_now={"taic/cli.py": "a" * 64},
        coverage={"exercised": True, "new_reports": 0}, observed_at="t2",
    )
    assert covered["zero_reproduction"] is True
    assert covered["status"] == "open"
    assert covered["change_ref"] == ""
    with pytest.raises(TopicError, match="commit:"):
        close_topic(ledger, "probe", "possibly_stale", confirm=True)


def test_open_topics_are_announced_and_do_not_block(tmp_path: Path):
    """@brief 实验开始前只打印未关闭的工具 topic，不因此拒绝启动。"""
    ledger = tmp_path / "topics.json"
    record_tool_topic(
        ledger, "still-open", head="abc", dirty=False, modules={},
        claims=[{"run_id": "r", "seq": 1, "digest": "d", "status": "pending"}],
    )
    lines: list[str] = []
    announce_open_tool_topics(ledger, lines.append)
    assert any("still-open" in line for line in lines)
    assert close_topic(ledger, "still-open", "commit:" + "d" * 12, confirm=True)["closed"] is True
    assert announce_open_tool_topics(ledger, lines.append) == []


def _reviewed(**extra) -> dict:
    body = {
        "reviewer": "self-review，会话 7670eeb0-dee4-4b02-b98b-1951879533e8，非独立复核",
        "reviewed_on": "2026-11-16",
        "counterexample": "这只覆盖写下的范围",
    }
    body.update(extra)
    return body


def test_empty_evidence_cannot_be_supported_and_kinds_stay_distinct():
    """@brief 没有证据不能写成 supported；每条都要有复核人和反例。"""
    assert validate_record(_reviewed(
        kind="理论解释", status="proposed", statement="注入可能无效", scope="",
    )) == []
    assert any("empty evidence" in item for item in validate_record(_reviewed(
        kind="理论解释", status="supported", evidence="", scope="one task",
    )))
    assert any("reviewer" in item for item in validate_record({
        "kind": "理论解释", "status": "proposed", "counterexample": "有替代解释", "reviewed_on": "2026-11-16",
        "reviewer": "待本阶段独立复核",
    }))
    assert any("traceable reviewer" in item for item in validate_record(_reviewed(
        kind="理论解释", status="proposed", reviewer="待复核",
    )))
    assert any("review date" in item for item in validate_record(_reviewed(
        kind="理论解释", status="proposed", reviewed_on="无",
    )))
    assert any("review date" in item for item in validate_record(_reviewed(
        kind="理论解释", status="proposed", reviewed_on="2026-02-31",
    )))
    assert any("人工复核" in item for item in validate_record(_reviewed(reviewer="人工复核")))
    assert any("traceable reviewer" in item for item in validate_record(_reviewed(
        kind="观察事实", status="supported", evidence="docs/架构/experience-overlay-paid-result.md",
        scope="one task", reviewer="待复核",
    )))
    assert any("counterexample" in item for item in validate_record({
        "kind": "理论解释", "status": "proposed", "reviewer": "人工复核", "reviewed_on": "2026-11-16",
    }))
    assert any("observed fact" in item for item in validate_record(_reviewed(
        kind="理论解释", status="proposed", as_fact=True,
    )))
    assert any("effect" in item for item in validate_record(_reviewed(
        kind="人为决策", status="proposed", as_effect=True,
    )))
    assert any("evidence file" in item for item in validate_record(_reviewed(
        kind="观察事实", status="refuted", evidence="docs/架构/experience-overlay-paid-result.md",
        scope="gemm_silu/fp16/small, two pairs, frozen protocol",
        reviewer="docs/架构/experience-overlay-paid-result.md", reviewed_on="2026-11-15",
    )))
    assert any("evidence file" in item for item in validate_record(_reviewed(
        reviewer="[experience-overlay-paid-result.md](../架构/experience-overlay-paid-result.md)",
    )))
    assert validate_record(_reviewed(
        kind="观察事实", status="refuted", evidence="docs/架构/experience-overlay-paid-result.md",
        scope="gemm_silu/fp16/small, two pairs, frozen protocol",
        reviewer="self-review，会话 7670eeb0-dee4-4b02-b98b-1951879533e8，非独立复核",
        reviewed_on="2026-11-16",
    )) == []


def test_theory_note_records_are_complete():
    """@brief 对照表的每一条都有可追踪复核人和真实日期；空证据仍是 proposed。"""
    import re

    text = Path("docs/可信harness/架构与理论.md").read_text(encoding="utf-8")
    note = Path("docs/可信harness/架构与理论.md")
    assert "待本阶段" not in text
    section = text.split("## 理论与实践对照", 1)[1]
    chunks = [chunk for chunk in section.split("\n### ")[1:] if chunk.strip()]
    assert len(chunks) >= 7
    link = re.compile(r"\[[^\]]*\]\(([^)]+)\)")
    by_title = {}
    for chunk in chunks:
        fields = {}
        for line in chunk.splitlines():
            if line.startswith("- ") and "：" in line:
                key, value = line[2:].split("：", 1)
                fields[key] = value.strip()
        title = chunk.splitlines()[0].strip()
        by_title[title] = fields
        evidence = "" if fields["证据"].startswith("无") else fields["证据"]
        row = {
            "kind": fields["类别"],
            "status": fields["状态"],
            "evidence": evidence,
            "scope": fields["范围"],
            "counterexample": fields["反例或替代解释"],
            "reviewer": fields["复核人"],
            "reviewed_on": fields["复核日期"],
        }
        assert validate_record(row) == [], (fields.get("命题"), validate_record(row))
        assert fields["复核人"] != "待复核"
        assert not fields["复核人"].startswith("待")
        assert "人工复核" not in chunk
        assert fields["复核日期"] != "无"
        assert fields["反例或替代解释"]
        assert "self-review" in fields["复核人"]
        assert "非独立复核" in fields["复核人"]
        assert "7670eeb0-dee4-4b02-b98b-1951879533e8" in fields["复核人"]
        assert "](" not in fields["复核人"]
        assert not fields["复核人"].endswith(".md")
        if not evidence:
            assert row["status"] == "proposed"
        else:
            # validate_record explicitly allows a scoped, evidence-backed refutation.
            assert row["status"] in {"supported", "refuted"}
            evidence_links = link.findall(fields["证据"])
            assert evidence_links, title
            assert (note.parent / evidence_links[0]).resolve().is_file()
    narrow = by_title["完整 overlay 的主指标"]
    assert narrow["状态"] == "supported"
    assert "no_gain" in narrow["观察"] and "adopt_overlay=false" in narrow["观察"]
    assert "gemm_silu/fp16/small" in narrow["范围"]
    for title in (
        "所有经验注入都无效",
        "退化来自上下文长度",
        "宿主选择减少幸存者偏差",
        "假设决策改善种子选择",
        "分槽预算降低负担",
        "施工顺序",
    ):
        assert by_title[title]["状态"] == "proposed"
        assert by_title[title]["证据"].startswith("无")


def _developer(action: str, **extra) -> dict:
    body = {
        "action": action,
        "audience": "developer",
        "needs": "repo",
        "rationale": "这要改仓库",
        "evidence": {"reproduction": "看当前代码", "expected": "命令可用", "measurement": ""},
    }
    body.update(extra)
    return body


def test_developer_lines_stay_in_the_queue_and_invalid_lines_keep_their_body(tmp_path: Path):
    """@brief 11–15 的 20 条无效行：自报 developer 的 16 条进队列，其余保留正文和拒绝原因。"""
    from taic.agent_feedback import parse_feedback
    from taic.harness import ProposalError, candidate_patch, inherit_prompt_text, parse_proposals, record_proposals, settle

    rows = json.loads(Path("tests/support/feedback_invalid_lines.json").read_text(encoding="utf-8"))
    assert len(rows) == 20
    queue = []
    declared = []
    extras = []
    rejected = []
    for item in rows:
        text = "# 对系统的建议\n```harness-proposal\n" + item["raw"] + "\n```\n"
        parsed = parse_feedback(text)
        found = parse_proposals(
            parsed, target_id=item["target_id"] or "apple-m4-max", task=item["task"] or "gemm",
            dtype=item["dtype"] or "fp16", work=tmp_path, started=None,
        )
        assert len(found) == 1
        outcome = found[0]
        try:
            document = json.loads(item["raw"])
        except json.JSONDecodeError:
            document = {}
        if outcome["ok"] and outcome["routing"]["disposition"] == "developer_queue":
            queue.append(outcome)
            if document.get("audience") == "developer":
                declared.append(item)
            else:
                extras.append(item)
        else:
            assert outcome["ok"] is False
            rejected.append((item, outcome, document))
    assert len(declared) == 16
    assert len(queue) == 17
    assert len(rejected) == 3
    assert len(extras) == 1 and "tolerance" in extras[0]["raw"].lower()
    run = tmp_path / "invalid-run"
    run.mkdir()
    for item, outcome, document in rejected:
        parsed = {
            "proposal_blocks": [{"raw": item["raw"], "section": "对系统的建议"}],
            "sections": {"对系统的建议": item["raw"]},
            "items": [{"text": "x", "digest": "d"}],
        }
        written = record_proposals(
            run, record={"seq": item["seq"] or 1}, parsed=parsed, target_id=item["target_id"] or "apple-m4-max",
            task=item["task"] or "gemm", dtype=item["dtype"] or "fp16", work=tmp_path, started=0,
        )
        assert written[0]["status"] == "invalid"
        assert written[0]["proposal"] != {}
        assert written[0]["reason"]
        if item["original_reason"].startswith("invalid JSON"):
            assert "invalid JSON" in written[0]["reason"]
            assert item["raw"].strip()[:40] in written[0]["proposal"]["raw"]
        else:
            assert written[0]["proposal"]["action"] == document.get("action")
    kept = next(item for item in queue if any(token in item["action"] for token in ("ABI", "tasks/", "tolerance")))
    run = tmp_path / "run"
    run.mkdir()
    row = {
        "status": "valid", "proposal_key": kept["proposal_key"], "target_id": "apple-m4-max", "task": "gemm",
        "probe_time_verified": True, "probe_sha256": "abc", "proposal": {**kept, "surface": "tip"},
    }
    (run / "harness_proposals.jsonl").write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
    settled = settle(run, target_id="apple-m4-max", task="gemm", protocol_version="v", enabled=True, shadow=False)
    assert kept["proposal_key"] not in settled["applied"] and kept["proposal_key"] not in settled["shadow"]
    assert "ABI" not in inherit_prompt_text(run, target_id="apple-m4-max", task="gemm")
    with pytest.raises(ProposalError, match="cannot be promoted"):
        candidate_patch({**kept, "surface": "tip"}, target_id="apple-m4-max", proposal_key=kept["proposal_key"])


def test_a_fence_above_the_prose_cap_is_still_parsed(tmp_path: Path):
    """@brief 围栏在 1200 字散文上限之外时，仍从原文取回。"""
    from taic.agent_feedback import parse_feedback
    from taic.harness import parse_proposals

    document = _developer("add a paired-compare subcommand to tcenv", rationale="甲" * 1500)
    fence = "```harness-proposal\n" + json.dumps(document, ensure_ascii=False) + "\n```"
    text = "# 对系统的建议\n" + fence + "\n" + ("乙" * 2000)
    parsed = parse_feedback(text)
    assert "harness-proposal" not in parsed["sections"]["对系统的建议"]
    assert len(parsed["sections"]["对系统的建议"]) == 1200
    assert parsed["proposal_blocks"][0]["raw"].lstrip().startswith("{")
    source = next(item for item in parsed["items"] if item.get("proposal_block") is True)
    assert len(source["text"]) > 1200
    assert source["digest"] == parsed["proposal_blocks"][0]["claim_digest"]
    found = parse_proposals(parsed, target_id="apple-m4-max", task="gemm", work=tmp_path, started=None)
    assert found[0]["ok"] is True
    assert found[0]["claim_digest"] == source["digest"]
    assert found[0]["routing"]["disposition"] == "developer_queue"


def test_fenced_suggestions_stay_indexable_through_review(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """@brief 纯围栏也留下源 claim，并穿过记录、索引、建议查询和评审。"""
    from observe.actions import review_feedback
    from observe.feedback import summary_from_index
    from observe.suggestions import compact
    from taic.agent_feedback import parse_feedback
    from tests.shared.test_observe_feedback import _open
    from tests.support.orchestrator import llm_script, make_orch

    runs, index = _open(tmp_path, monkeypatch)
    notes: list[str] = []
    orch = make_orch(runs, llm_script(""), [], max_rounds=0, log=notes.append)
    member = orch.committees[0].members[0]
    work = orch.run_dir / "agent_work" / "src"
    work.mkdir(parents=True)
    overlay = {
        "surface": "tip",
        "scope": {"target_id": orch.target_id, "task": orch.task_name, "dtype": orch.dtype},
        "action": "Keep the fp16 accumulator in registers while measuring the kernel.",
        "audience": "peer",
        "needs": "none",
        "peer_action": "Keep the fp16 accumulator in registers.",
        "evidence": {"reproduction": "read the kernel accumulator", "expected": "the value stays in registers"},
        "rationale": "the next kernel session can do this itself",
    }
    bodies = {
        "developer_queue": json.dumps(_developer("add a paired-compare subcommand to tcenv"), ensure_ascii=False),
        "overlay_candidate": json.dumps(overlay, ensure_ascii=False),
        "invalid": "{ this is not json",
    }
    for label, body in bodies.items():
        parsed = parse_feedback("# 对系统的建议\n```harness-proposal\n" + body + "\n```\n")
        assert parsed["sections"] == {}
        assert len(parsed["items"]) == 1
        assert parsed["items"][0]["proposal_block"] is True
        assert parsed["items"][0]["digest"]
        assert parsed["proposal_blocks"][0]["claim_digest"] == parsed["items"][0]["digest"]
        orch._record_feedback(label, member, work, parsed, "evaluated", None, "normal", None, started=1.0)
    assert not any("harness proposals skipped" in line for line in notes), notes
    (orch.run_dir / "state.json").write_text(json.dumps({
        "task": orch.task_name, "dtype": orch.dtype, "phase": "done", "started_at": 50, "updated_at": 50,
    }), encoding="utf-8")
    index.refresh_feedback(orch.run_dir.name, orch.run_dir)
    claims = [dict(row) for row in index.conn.execute(
        "SELECT run_id, seq, digest, status, section FROM feedback_claims ORDER BY seq"
    )]
    assert len(claims) == 3
    assert {row["section"] for row in claims} == {"对系统的建议"}
    assert summary_from_index(index.conn)["pending_total"] == 3
    proposals = [
        json.loads(line)
        for line in (orch.run_dir / "harness_proposals.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(proposals) == 3
    by_seq = {int(row["seq"]): row for row in proposals}
    for claim in claims:
        row = by_seq[int(claim["seq"])]
        assert row["claim_digest"] == claim["digest"]
        assert row["claim_digest"]
    developer = next(row for row in proposals if row.get("status") == "valid" and row["proposal"]["routing"]["disposition"] == "developer_queue")
    overlaid = next(row for row in proposals if row.get("status") == "valid" and row["proposal"]["routing"]["disposition"] == "overlay_candidate")
    invalid = next(row for row in proposals if row.get("status") == "invalid")
    assert "invalid JSON" in invalid["reason"]
    found = {item["digest"]: item for item in compact(limit=20)["groups"]}
    assert found[developer["claim_digest"]]["converted_key"] == developer["proposal_key"]
    assert found[overlaid["claim_digest"]]["converted_key"] == overlaid["proposal_key"]
    assert invalid["claim_digest"] in found
    assert found[invalid["claim_digest"]]["converted_key"] == ""
    rejected = review_feedback(claims[0]["run_id"], invalid["seq"], invalid["claim_digest"], {
        "status": "rejected", "verification": "refuted", "note": "无效围栏", "expected_status": "pending",
    })
    assert rejected["status"] == "rejected"
    assert summary_from_index(index.conn)["pending_total"] == 2


def test_harness_cell_announces_open_topics_before_paying(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys):
    """@brief harness cell 在 orch.run 之前打印未关闭 topic，dry-run 不进入这条路径。"""
    import sys
    from types import SimpleNamespace

    from experiments import harness_cell
    from experiments.harness_runner import dry_run
    from taic.harness_experiment import ExperimentError
    from tests.support.v2_specimen import V2_TASK

    ledger = tmp_path / "topics.json"
    record_tool_topic(
        ledger, "still-open", head="abc", dirty=False, modules={},
        claims=[{"run_id": "r", "seq": 1, "digest": "d", "status": "info"}],
    )
    monkeypatch.setenv("TAIC_TOOL_TOPICS", str(ledger))
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://127.0.0.1:9")
    toml = tmp_path / "taic.toml"
    toml.write_text(
        '[llm]\ncredentials_file = "/no/such/models.json"\nprovider = "deepseek"\nmodel = "deepseek-flash"\n'
        'use_tools = false\nreasoning_effort = "off"\n'
        "[search]\nbudget_usd = 5\nbranches = 1\nmax_rounds = 0\neval_reps = 3\n"
        "stop_at_roofline = false\nabi_guard = false\ntarget_note = false\nrich_history = false\nstatus_lines = false\n"
        '[[committees]]\nname = "first"\n[[committees.members]]\nprovider = "deepseek"\nmodel = "model-a"\n'
        '[[committees]]\nname = "second"\n[[committees.members]]\nprovider = "deepseek"\nmodel = "model-b"\n',
        encoding="utf-8",
    )
    request = tmp_path / "request.json"
    (tmp_path / "cell").mkdir()
    request.write_text(json.dumps({
        "config": str(toml), "task": V2_TASK, "dtype": "fp16", "size": "small",
        "run_dir": str(tmp_path / "cell"), "target_id": "apple-m4-max", "seed": 1,
    }), encoding="utf-8")
    seen = {}

    class Stub:
        def __init__(self, *args, **kwargs):
            return None

        async def run(self):
            seen["during_run"] = capsys.readouterr().out
            return SimpleNamespace(spent_usd=0.0, stop_reason="max_rounds", run_seed=1)

    monkeypatch.setattr("taic.orchestrator.Orchestrator", Stub)
    monkeypatch.setattr(sys, "argv", ["harness_cell", str(request)])
    assert harness_cell.main() == 0
    assert "still-open" in seen["during_run"]

    def refuse(*_args, **_kwargs):
        raise AssertionError("dry-run entered the paid cell")

    monkeypatch.setattr("taic.orchestrator.Orchestrator", refuse)
    monkeypatch.setattr("experiments.harness_cell.main", refuse)
    monkeypatch.setattr("taic.feedback_topics.hint_open_tool_topics", refuse)
    with pytest.raises(ExperimentError):
        dry_run({}, out_dir=tmp_path / "out", repo=tmp_path)


def test_portfolio_shard_announces_open_topics_before_paying(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys):
    """@brief 组合搜索的真实分片在 orch.run 之前提示；dry-run 不构造编排器。"""
    from types import SimpleNamespace

    from taic.cli import main
    from taic.portfolio_search import _Runners, _default_shard_runner
    from tests.support.v2_specimen import V2_TASK

    ledger = tmp_path / "topics.json"
    record_tool_topic(
        ledger, "still-open", head="abc", dirty=False, modules={},
        claims=[{"run_id": "r", "seq": 1, "digest": "d", "status": "info"}],
    )
    monkeypatch.setenv("TAIC_TOOL_TOPICS", str(ledger))
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://127.0.0.1:9")
    toml = tmp_path / "taic.toml"
    toml.write_text(
        '[llm]\ncredentials_file = "/no/such/models.json"\nprovider = "deepseek"\nmodel = "deepseek-flash"\n'
        'use_tools = false\nreasoning_effort = "off"\n'
        "[search]\nbudget_usd = 5\nbranches = 1\nmax_rounds = 0\neval_reps = 3\n"
        "stop_at_roofline = false\nabi_guard = false\ntarget_note = false\nrich_history = false\nstatus_lines = false\n"
        '[[committees]]\nname = "first"\n[[committees.members]]\nprovider = "deepseek"\nmodel = "model-a"\n'
        '[[committees]]\nname = "second"\n[[committees.members]]\nprovider = "deepseek"\nmodel = "model-b"\n',
        encoding="utf-8",
    )
    seen = {}

    class Stub:
        def __init__(self, *args, **kwargs):
            return None

        async def run(self):
            seen["during_run"] = capsys.readouterr().out
            return SimpleNamespace(spent_usd=0.0, stop_reason="max_rounds", run_seed=1)

    monkeypatch.setattr("taic.orchestrator.Orchestrator", Stub)
    opened = SimpleNamespace(
        locator=SimpleNamespace(op_name=V2_TASK, dtype="fp16"),
        budget=1.0, target=SimpleNamespace(id="apple-m4-max"), track="msl-single",
    )
    runners = _Runners(None, None, None, None, None, str(toml), None)
    result = _default_shard_runner(opened, runners, 7)("small", tmp_path / "shard")
    assert result.stop_reason == "max_rounds"
    assert "still-open" in seen["during_run"]

    def refuse(*_args, **_kwargs):
        raise AssertionError("dry-run started a search")

    monkeypatch.setattr("taic.orchestrator.Orchestrator", refuse)
    monkeypatch.setattr("taic.feedback_topics.hint_open_tool_topics", refuse)
    assert main([
        "portfolio-run", "gemm", "--dtype", "fp16", "--run-dir", str(tmp_path / "dry"),
        "--dry-run", "--budget", "1", "--assemble-budget", "1",
    ]) == 0
    assert not (tmp_path / "dry" / "shards").exists()
    """@brief 宿主只认被否决前提正文里的 SHA，不认模型自己打的旗标。"""
    from taic.paired_compare import apply_premise_citations

    first, first_path = _kernel(tmp_path, "early.metal", b"early-body")
    second, second_path = _kernel(tmp_path, "later.metal", b"later-body")
    mid, mid_path = _kernel(tmp_path, "mid.metal", b"mid-body")
    low_pair = sorted([first, mid])
    high_pair = sorted([second, mid])
    if high_pair < low_pair:
        first, second = second, first
        first_path, second_path = second_path, first_path
        low_pair, high_pair = high_pair, low_pair
    early, later = first, second
    early_path, later_path = first_path, second_path
    low, high = low_pair, high_pair
    rows = [
        {"sha_a": early, "sha_b": mid, "path_a": early_path, "path_b": mid_path, "conclusion": "diff", "cited_by_rejected_premise": False},
        {"sha_a": later, "sha_b": mid, "path_a": later_path, "path_b": mid_path, "conclusion": "no_diff", "cited_by_rejected_premise": True},
    ]
    chosen = select_nominations(apply_premise_citations(rows, "# 被否决的前提\n- 这里没有 SHA\n"))
    assert chosen["selected"][0]["sha_pair"] == low
    cited = apply_premise_citations(rows, f"# 被否决的前提\n- {later} {mid}\n")
    assert select_nominations(cited)["selected"][0]["sha_pair"] == high


def test_the_observation_switch_records_a_tie_without_replacing_the_seed(tmp_path: Path):
    """@brief 开关打开时，噪声内的 tie 仍不换种子；关闭时不写配对文件。"""
    from tests.support.orchestrator import code, llm_script, make_orch, run

    off = make_orch(
        tmp_path / "off", llm_script(code("speed=1.0"), patches={"A": code("speed=1.002")}), [],
        max_rounds=1, correctness_only=True,
    )
    run(off)
    assert off.best.label == "init"
    assert list(off.run_dir.glob("candidates/*/paired_comparison.json")) == []

    on = make_orch(
        tmp_path / "on", llm_script(code("speed=1.0"), patches={"A": code("speed=1.002")}), [],
        max_rounds=1, correctness_only=True, paired_seed_observation=True,
    )
    on._paired_runners = lambda cand, best: (lambda k: 1e-4 * k, lambda k: 2e-4 * k)
    run(on)
    assert on.best.label == "init"
    assert on.best.within_noise is False or on.candidates[-1].within_noise is True
    documents = [json.loads(path.read_text(encoding="utf-8")) for path in on.run_dir.glob("candidates/*/paired_comparison.json")]
    assert documents
    assert all(item["changes_seed"] is False for item in documents)
    assert any(item.get("applied") and item.get("conclusion") == "diff" for item in documents), documents
    on._local_evaluator = False

    class Box:
        source = "kernel void x() {}\n"
        label = "a"
        report = object()

    remote = on._paired_seed_observation(Box(), Box())
    assert remote["reason"] == "remote evaluator; paired seed observation is local-only"
    assert remote["changes_seed"] is False


def test_a_session_without_nominations_does_not_write_a_host_selection(tmp_path: Path):
    """@brief 没有提名文件时，候选目录不出现宿主选择记录。"""
    from tests.support.orchestrator import code, llm_script, make_orch

    orch = make_orch(tmp_path, llm_script(code("speed=1.0")), [], max_rounds=1)
    folder = orch.run_dir / "candidates" / "000_init"
    folder.mkdir(parents=True)
    work = orch.run_dir / "agent_work" / "init"
    work.mkdir(parents=True)

    class Cand:
        idx = 0
        label = "init"

    class Result:
        notes = ""
        feedback = ""

    asyncio.run(orch._file_session_notes(Cand(), Result()))
    assert not (folder / "paired_host_selection.json").exists()
    called = {"n": 0}

    def boom() -> float:
        called["n"] += 1
        return 1.0

    orch._host_pair_timer = boom
    append_nomination(work / ".selfcheck", {"sha_a": "ab" * 32, "sha_b": "cd" * 32, "conclusion": "no_diff"})
    asyncio.run(orch._file_session_notes(Cand(), Result()))
    saved = json.loads((folder / "paired_host_selection.json").read_text(encoding="utf-8"))
    assert saved["ran_gpu"] is False
    assert called["n"] == 0
    assert saved["nominations"] == 1


def test_a_failed_session_still_settles_its_nominations(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """@brief agent 失败、没有候选时，本场提名仍写入 sessions/ 选择记录。"""
    from taic.kernel_agent import AgentError
    from tests.support.orchestrator import code, llm_script, make_orch

    orch = make_orch(tmp_path, llm_script(code("speed=1.0")), [], max_rounds=1)
    left = tmp_path / "left.metal"
    right = tmp_path / "right.metal"
    left.write_text("kernel void left() {}\n", encoding="utf-8")
    right.write_text("kernel void right() {}\n", encoding="utf-8")
    sha_a = hashlib.sha256(left.read_bytes()).hexdigest()
    sha_b = hashlib.sha256(right.read_bytes()).hexdigest()
    host_dir = tmp_path / "host-authority"
    monkeypatch.setattr(
        "taic.paired_compare.host_evidence_path",
        lambda _directory: host_dir / "host_paired_comparisons.jsonl",
    )

    def agent(brief, _tag):
        append_nomination(Path(brief.work_dir) / ".selfcheck", {
            "sha_a": sha_a, "sha_b": sha_b, "path_a": str(left), "path_b": str(right), "conclusion": "no_diff",
        })
        raise AgentError("no kernel")

    orch.agent_runner = agent
    orch._paired_runner_binder = lambda _path_a, _path_b: (lambda k: 1e-4 * k, lambda k: 2e-4 * k)
    outcome = asyncio.run(orch._one_session_attempt("init", seedless=True, candidate_label="init", attempt=0))
    assert outcome == "agent_failed"
    saved = json.loads((orch.run_dir / "sessions" / "init" / "paired_host_selection.json").read_text(encoding="utf-8"))
    assert saved["nominations"] == 1
    assert saved["reviewed"] == 1
    assert not list((orch.run_dir / "candidates").glob("*/paired_host_selection.json"))
    assert not (orch.run_dir / "harness_route_m.jsonl").exists()


def test_a_resumed_session_drops_the_previous_nominations(tmp_path: Path):
    """@brief 场次开始时清掉上一场提名。续跑不会把旧 SHA 对当成这一场。"""
    from taic.kernel_agent import AgentError
    from tests.support.orchestrator import code, llm_script, make_orch

    orch = make_orch(tmp_path, llm_script(code("speed=1.0")), [], max_rounds=1)
    work = orch.run_dir / "agent_work" / "init"
    left = tmp_path / "left.metal"
    right = tmp_path / "right.metal"
    left.write_text("kernel void left() {}\n", encoding="utf-8")
    right.write_text("kernel void right() {}\n", encoding="utf-8")
    sha_a = hashlib.sha256(left.read_bytes()).hexdigest()
    sha_b = hashlib.sha256(right.read_bytes()).hexdigest()
    append_nomination(work / ".selfcheck", {
        "sha_a": sha_a, "sha_b": sha_b, "path_a": str(left), "path_b": str(right), "conclusion": "no_diff",
    })
    assert (work / ".selfcheck" / "paired_comparisons.jsonl").is_file()
    orch._prepare_session_dir(work)
    assert not (work / ".selfcheck" / "paired_comparisons.jsonl").exists()

    def agent(_brief, _tag):
        raise AgentError("stopped before a new nomination")

    orch.agent_runner = agent
    outcome = asyncio.run(orch._one_session_attempt("init", seedless=True, candidate_label="init", attempt=0))
    assert outcome == "agent_failed"
    assert not (work / ".selfcheck" / "paired_comparisons.jsonl").exists()
    assert not (orch.run_dir / "sessions" / "init" / "paired_host_selection.json").exists()


def test_the_paired_command_keeps_shas_out_of_the_quota_line(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys):
    """@brief 配额在计时前检查。真正计时后写一条不含 SHA 的配额行，提名保留两个 SHA。"""
    from tcenv.cli import main
    from tcenv.selfcheck import budget_reason

    left = tmp_path / "a.metal"
    right = tmp_path / "b.metal"
    left.write_text("kernel void a() {}\n", encoding="utf-8")
    right.write_text("kernel void b() {}\n", encoding="utf-8")
    directory = tmp_path / ".selfcheck"
    code = main(["paired-compare", str(left), str(right), "--self-check-dir", str(directory), "--backend", "qingwei"])
    assert code == 2
    stored = json.loads((directory / "paired_comparisons.jsonl").read_text(encoding="utf-8"))
    assert len(stored["sha_a"]) == 64 and len(stored["sha_b"]) == 64
    assert not (directory / "self_checks.jsonl").exists()

    calls = {"n": 0}

    def fake_bind(path_a, path_b, **kwargs):
        calls["n"] += 1
        calls["paths"] = (str(path_a), str(path_b))
        return (lambda k: 1e-4 * k, lambda k: 2e-4 * k)

    monkeypatch.setattr("tcenv.backends.metal.paired_compare.bind_metal_runners", fake_bind)
    monkeypatch.setattr("tcenv.selfcheck.budget_reason", lambda _directory: "quota spent")
    blocked = tmp_path / "blocked"
    assert main(["paired-compare", str(left), str(right), "--self-check-dir", str(blocked)]) == 2
    assert calls["n"] == 0
    assert not (blocked / "paired_comparisons.jsonl").exists()
    assert not (blocked / "self_checks.jsonl").exists()

    monkeypatch.setattr("tcenv.selfcheck.budget_reason", budget_reason)
    measured = tmp_path / "measured"
    assert main(["paired-compare", str(left), str(right), "--self-check-dir", str(measured)]) == 0
    assert calls["paths"] == (str(left), str(right))
    document = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    for key in ("median_a_s", "median_b_s", "ratios", "interval", "rounds", "cv", "elapsed_s"):
        assert document[key] is not None, key
    assert document["applied"] is True and document["gpu"] is True
    assert document["rounds"] >= 2 and document["elapsed_s"] > 0
    nomination = json.loads((measured / "paired_comparisons.jsonl").read_text(encoding="utf-8"))
    assert len(nomination["sha_a"]) == 64 and len(nomination["sha_b"]) == 64
    debit = json.loads((measured / "self_checks.jsonl").read_text(encoding="utf-8").splitlines()[-1])
    assert "sha_a" not in debit and "sha_b" not in debit
    assert debit["kind"] == "paired_compare_quota"
    assert debit["elapsed_s"] == document["elapsed_s"]


def test_a_supported_failure_debits_the_timed_quota(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """@brief 已进入支持后端的失败也占一次次数。超时进入挂死记账；未知耗时不写成秒数。"""
    from tcenv.cli import main
    from tcenv.selfcheck import earlier_hang

    left = tmp_path / "a.metal"
    right = tmp_path / "b.metal"
    left.write_text("kernel void a() {}\n", encoding="utf-8")
    right.write_text("kernel void b() {}\n", encoding="utf-8")

    def failed(_backend, _path_a, _path_b, **_kwargs):
        return {"applied": False, "supported": True, "backend": "metal", "elapsed_s": 420, "changes_seed": False}

    monkeypatch.setattr("tcenv.cli_metal._compare_pair", failed)
    spent = tmp_path / "spent"
    assert main(["paired-compare", str(left), str(right), "--self-check-dir", str(spent)]) == 2
    nomination = json.loads((spent / "paired_comparisons.jsonl").read_text(encoding="utf-8"))
    assert len(nomination["sha_a"]) == 64
    debit = json.loads((spent / "self_checks.jsonl").read_text(encoding="utf-8"))
    assert debit["elapsed_s"] == 420.0
    assert debit["kind"] == "paired_compare_quota"
    assert "sha_a" not in debit

    def unknown(_backend, _path_a, _path_b, **_kwargs):
        return {"applied": False, "supported": True, "backend": "metal", "status": "error"}

    monkeypatch.setattr("tcenv.cli_metal._compare_pair", unknown)
    blind = tmp_path / "blind"
    assert main(["paired-compare", str(left), str(right), "--self-check-dir", str(blind)]) == 2
    unknown_row = json.loads((blind / "self_checks.jsonl").read_text(encoding="utf-8"))
    assert unknown_row["elapsed"] == "unknown"
    assert "elapsed_s" not in unknown_row
    assert unknown_row["status"] == "error"

    def timed_out(_backend, _path_a, _path_b, **_kwargs):
        return {"applied": False, "supported": True, "backend": "metal", "status": "timeout"}

    monkeypatch.setattr("tcenv.cli_metal._compare_pair", timed_out)
    hung = tmp_path / "hung"
    assert main(["paired-compare", str(left), str(right), "--self-check-dir", str(hung)]) == 2
    hang_row = json.loads((hung / "self_checks.jsonl").read_text(encoding="utf-8"))
    assert hang_row["status"] == "timeout"
    assert hang_row["elapsed"] == "unknown"
    assert earlier_hang(hung)
    before = (hung / "paired_comparisons.jsonl").read_bytes()
    assert main(["paired-compare", str(left), str(right), "--self-check-dir", str(hung)]) == 2
    assert (hung / "paired_comparisons.jsonl").read_bytes() == before


def test_a_worker_crash_debits_once_and_a_missing_file_does_not(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """@brief 进入支持后端后的 RuntimeError 记一次 crash；读文件失败不占配额。"""
    from tcenv.cli import main
    from tcenv.selfcheck import earlier_hang

    left = tmp_path / "a.metal"
    right = tmp_path / "b.metal"
    left.write_text("kernel void a() {}\n", encoding="utf-8")
    right.write_text("kernel void b() {}\n", encoding="utf-8")
    absent = tmp_path / "absent"
    assert main(["paired-compare", str(left), str(tmp_path / "missing.metal"), "--self-check-dir", str(absent)]) == 2
    assert not (absent / "paired_comparisons.jsonl").exists()
    assert not (absent / "self_checks.jsonl").exists()

    def crash(_backend, _path_a, _path_b, **_kwargs):
        raise RuntimeError("worker launch crashed")

    monkeypatch.setattr("tcenv.cli_metal._compare_pair", crash)
    crashed = tmp_path / "crashed"
    assert main(["paired-compare", str(left), str(right), "--self-check-dir", str(crashed)]) == 2
    nomination = (crashed / "paired_comparisons.jsonl").read_text(encoding="utf-8")
    assert nomination.count("\n") == 1
    debit = json.loads((crashed / "self_checks.jsonl").read_text(encoding="utf-8"))
    assert debit["status"] == "crash"
    assert debit["elapsed"] == "unknown"
    assert "elapsed_s" not in debit
    assert "worker launch crashed" in debit.get("reason", "") or earlier_hang(crashed)
    assert earlier_hang(crashed)
    before = (crashed / "paired_comparisons.jsonl").read_bytes()
    assert main(["paired-compare", str(left), str(right), "--self-check-dir", str(crashed)]) == 2
    assert (crashed / "paired_comparisons.jsonl").read_bytes() == before
    assert (crashed / "self_checks.jsonl").read_text(encoding="utf-8").count("\n") == 1


def test_the_observation_switch_binds_both_sources_without_a_test_runner(tmp_path: Path):
    """@brief 开关打开且没有测试 runner 时，默认路径会把两份源码交给绑定器。"""
    from tests.support.orchestrator import code, llm_script, make_orch, run

    on = make_orch(
        tmp_path, llm_script(code("speed=1.0"), patches={"A": code("speed=1.002")}), [],
        max_rounds=1, correctness_only=True, paired_seed_observation=True,
    )
    seen = {}

    def binder(path_a, path_b):
        seen["a"] = Path(path_a).read_text(encoding="utf-8")
        seen["b"] = Path(path_b).read_text(encoding="utf-8")
        return (lambda k: 1e-4 * k, lambda k: 2e-4 * k)

    on._paired_runner_binder = binder
    run(on)
    assert seen["a"] and seen["b"] and seen["a"] != seen["b"]
    documents = [json.loads(path.read_text(encoding="utf-8")) for path in on.run_dir.glob("candidates/*/paired_comparison.json")]
    assert any(item.get("applied") and item.get("median_a_s") and item.get("interval") and item["changes_seed"] is False for item in documents)


def test_the_host_rerun_saves_a_measurement_and_looks_up_its_own_evidence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """@brief 宿主重跑保存测量文档。已有证据按文件查找，坏 JSON 留在未覆盖项里。"""
    from tcenv.cli import main
    from tests.support.orchestrator import code, llm_script, make_orch

    orch = make_orch(tmp_path, llm_script(code("speed=1.0")), [], max_rounds=1)
    folder = orch.run_dir / "candidates" / "000_init"
    folder.mkdir(parents=True)
    work = orch.run_dir / "agent_work" / "init"
    work.mkdir(parents=True)
    left = tmp_path / "left.metal"
    right = tmp_path / "right.metal"
    left.write_text("kernel void left() {}\n", encoding="utf-8")
    right.write_text("kernel void right() {}\n", encoding="utf-8")
    directory = work / ".selfcheck"

    def fake_bind(path_a, path_b, **kwargs):
        return (lambda k: 1e-4 * k, lambda k: 2e-4 * k)

    monkeypatch.setattr("tcenv.backends.metal.paired_compare.bind_metal_runners", fake_bind)
    assert main(["paired-compare", str(left), str(right), "--self-check-dir", str(directory)]) == 0
    with (directory / "paired_comparisons.jsonl").open("a", encoding="utf-8") as handle:
        handle.write("{not json\n")
    calls = {"n": 0}

    def binder(path_a, path_b):
        calls["n"] += 1
        assert Path(path_a).read_text(encoding="utf-8") == left.read_text(encoding="utf-8")
        return (lambda k: 1e-4 * k, lambda k: 2e-4 * k)

    orch._paired_runner_binder = binder
    host_dir = tmp_path / "host-authority"
    monkeypatch.setattr(
        "taic.paired_compare.host_evidence_path",
        lambda _directory: host_dir / "host_paired_comparisons.jsonl",
    )

    class Cand:
        idx = 0
        label = "init"

    class Result:
        notes = ""
        feedback = ""

    asyncio.run(orch._file_session_notes(Cand(), Result()))
    saved = json.loads((folder / "paired_host_selection.json").read_text(encoding="utf-8"))
    assert calls["n"] == 1
    assert saved["reviewed"] == 1
    assert saved["gpu_seconds"] > 0
    measured = saved["measurements"][0]
    for key in ("median_a_s", "median_b_s", "ratios", "interval", "rounds", "cv", "elapsed_s", "sha_a", "sha_b"):
        assert measured[key] is not None
    assert any(item["omit_reason"] == "unreadable_nomination" for item in saved["uncovered"])
    from taic.paired_compare import host_evidence_path

    evidence_path = host_evidence_path(directory)
    assert evidence_path != directory / "host_paired_comparisons.jsonl"
    assert not (directory / "host_paired_comparisons.jsonl").is_file()
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    assert evidence["host_evidence"] is True and evidence["applied"] is True
    nomination = json.loads((directory / "paired_comparisons.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert nomination["host_evidence"] is False

    def refuse(path_a, path_b):
        raise AssertionError("an already measured pair must not be rerun")

    orch._paired_runner_binder = refuse
    asyncio.run(orch._file_session_notes(Cand(), Result()))
    again = json.loads((folder / "paired_host_selection.json").read_text(encoding="utf-8"))
    assert again["reviewed"] == 0
    assert any(item["omit_reason"] == "equivalent_host_evidence" for item in again["uncovered"])


def test_a_non_metal_nomination_does_not_start_a_worker(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """@brief 不支持 paired 的场次只记录原因，绑定器和 worker 都不会被调用。"""
    from tcenv.cli import main
    from tests.support.orchestrator import code, llm_script, make_orch

    orch = make_orch(tmp_path, llm_script(code("speed=1.0")), [], max_rounds=1)

    class _Backend:
        backend = "qingwei"

    orch.target = _Backend()
    folder = orch.run_dir / "candidates" / "000_init"
    folder.mkdir(parents=True)
    work = orch.run_dir / "agent_work" / "init"
    work.mkdir(parents=True)
    left = tmp_path / "left.metal"
    right = tmp_path / "right.metal"
    left.write_text("kernel void left() {}\n", encoding="utf-8")
    right.write_text("kernel void right() {}\n", encoding="utf-8")
    directory = work / ".selfcheck"
    assert main(["paired-compare", str(left), str(right), "--self-check-dir", str(directory), "--backend", "qingwei"]) == 2
    assert (directory / "paired_comparisons.jsonl").is_file()
    calls = {"binder": 0, "worker": 0, "compare": 0}

    def binder(_path_a, _path_b):
        calls["binder"] += 1
        raise AssertionError("binder")

    def worker(*_args, **_kwargs):
        calls["worker"] += 1
        raise AssertionError("worker")

    def compare(*_args, **_kwargs):
        calls["compare"] += 1
        raise AssertionError("compare")

    monkeypatch.setattr("tcenv.v2_worker.run_bounded_v2", worker)
    monkeypatch.setattr("tcenv.backends.metal.paired_compare.compare_paths", compare)
    monkeypatch.setattr("tcenv.backends.metal.paired_compare.compare_for_backend", compare)
    orch._paired_runner_binder = binder

    class Cand:
        source = left.read_text(encoding="utf-8")
        label = "init"

    class Best:
        source = right.read_text(encoding="utf-8")
        label = "seed"

    observed = orch._paired_seed_observation(Cand(), Best())
    assert observed["supported"] is False
    assert observed["gpu"] is False
    assert observed["applied"] is False
    assert "does not support" in observed["reason"]
    asyncio.run(orch._select_paired_nominations(folder, work, session_label="init"))
    saved = json.loads((orch.run_dir / "sessions" / "init" / "paired_host_selection.json").read_text(encoding="utf-8"))
    assert saved["reviewed"] == 0
    assert saved["ran_gpu"] is False
    assert saved["measurements"]
    assert saved["measurements"][0]["supported"] is False
    assert "does not support" in saved["measurements"][0]["reason"]
    assert calls == {"binder": 0, "worker": 0, "compare": 0}


def test_a_remote_metal_session_does_not_rerun_locally(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """@brief 远端 Metal 场次的提名收尾不启动本地 paired worker。"""
    from taic.paired_compare import append_nomination
    from tests.support.orchestrator import code, llm_script, make_orch

    orch = make_orch(tmp_path, llm_script(code("speed=1.0")), [], max_rounds=1)
    orch._local_evaluator = False
    calls = {"binder": 0, "worker": 0, "compare": 0}

    def binder(_path_a, _path_b):
        calls["binder"] += 1
        raise AssertionError("binder")

    def worker(*_args, **_kwargs):
        calls["worker"] += 1
        raise AssertionError("worker")

    def compare(*_args, **_kwargs):
        calls["compare"] += 1
        raise AssertionError("compare")

    monkeypatch.setattr("tcenv.v2_worker.run_bounded_v2", worker)
    monkeypatch.setattr("tcenv.backends.metal.paired_compare.compare_paths", compare)
    orch._paired_runner_binder = binder
    orch._host_pair_timer = compare
    folder = orch.run_dir / "candidates" / "000_init"
    folder.mkdir(parents=True)
    work = orch.run_dir / "agent_work" / "init"
    work.mkdir(parents=True)
    left = tmp_path / "left.metal"
    right = tmp_path / "right.metal"
    left.write_text("kernel void left() {}\n", encoding="utf-8")
    right.write_text("kernel void right() {}\n", encoding="utf-8")
    sha_a = hashlib.sha256(left.read_bytes()).hexdigest()
    sha_b = hashlib.sha256(right.read_bytes()).hexdigest()
    append_nomination(work / ".selfcheck", {
        "sha_a": sha_a, "sha_b": sha_b, "path_a": str(left), "path_b": str(right),
    })
    direct = orch._host_rerun([{
        "path_a": str(left), "path_b": str(right), "sha_a": sha_a, "sha_b": sha_b,
        "sha_pair": [sha_a, sha_b],
    }])
    assert direct[0]["applied"] is False
    assert "local-only" in direct[0]["reason"]
    asyncio.run(orch._select_paired_nominations(folder, work, session_label="init"))
    saved = json.loads((orch.run_dir / "sessions" / "init" / "paired_host_selection.json").read_text(encoding="utf-8"))
    assert saved["reviewed"] == 0
    assert saved["ran_gpu"] is False
    assert "local-only" in saved["measurements"][0]["reason"]
    assert calls == {"binder": 0, "worker": 0, "compare": 0}


def test_close_topic_and_suggestion_query_follow_the_queue_rules(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """@brief close-topic 整批校验；前提默认不进待审；建议查询同时覆盖两个小节和后端。"""
    from observe.actions import Refused, close_feedback_topic, review_feedback
    from observe.feedback import summary_from_index
    from observe.suggestions import compact
    from taic.agent_feedback import parse_feedback
    from taic.cli import main

    from tests.shared.test_observe_feedback import _digest, _open, _parsed, _record, _run

    monkeypatch.setenv("TAIC_TOOL_TOPICS", str(tmp_path / "topics.json"))
    runs, index = _open(tmp_path, monkeypatch)
    directory = _run(runs, "conv", _record(_parsed()), started_at=50)
    (directory / "run_config.json").write_text(json.dumps({"target": {"id": "apple-m4-max", "backend": "metal"}}), encoding="utf-8")
    help_text = parse_feedback("# 需要帮助\n- 另一把探针\n")
    other = runs / "other"
    _run(runs, "other", _record(help_text), started_at=80)
    (other / "result.json").write_text(json.dumps({"simulated": True, "measurement_kind": "simulated"}), encoding="utf-8")
    index.sync_known()
    help_digest = _digest("需要帮助")
    premise = _digest("被否决的前提")
    review_feedback("conv", 1, help_digest, {"status": "pending", "verification": "unchecked", "topic": "probe", "expected_status": "pending"})
    review_feedback("conv", 1, premise, {"status": "info", "verification": "unchecked", "topic": "probe", "expected_status": "info"})
    assert summary_from_index(index.conn)["pending_total"] == 2
    review_feedback("conv", 1, premise, {"status": "info", "verification": "unchecked", "topic": "probe", "conflict": True, "expected_status": "info"})
    assert summary_from_index(index.conn)["pending_total"] == 3
    review_feedback("conv", 1, premise, {"status": "info", "verification": "unchecked", "topic": "probe", "expected_status": "pending"})
    assert summary_from_index(index.conn)["pending_total"] == 2
    batch = tmp_path / "batch.json"
    batch.write_text(json.dumps([
        {"seq": 1, "digest": help_digest, "status": "pending", "verification": "unchecked", "topic": "probe", "expected_status": "pending"},
    ]), encoding="utf-8")
    assert main(["feedback", str(directory), "--file", str(batch), "--dry-run"]) == 0
    preview = close_feedback_topic("probe", "commit:" + "a" * 12, dry_run=True)
    assert preview["closed"] is False
    before = (directory / "feedback_review.json").read_text(encoding="utf-8")
    doc = json.loads(before)
    key = f"1:{premise}"
    doc["claims"][key]["status"] = "rejected"
    (directory / "feedback_review.json").write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(Refused, match="item"):
        close_feedback_topic("probe", "commit:" + "b" * 12, confirm=True)
    assert json.loads((directory / "feedback_review.json").read_text(encoding="utf-8"))["claims"][f"1:{help_digest}"]["status"] == "pending"
    (directory / "feedback_review.json").write_text(before, encoding="utf-8")
    closed = close_feedback_topic("probe", "commit:" + "c" * 12, confirm=True)
    assert closed["closed"] is True
    assert main(["feedback", str(directory), "--file", str(tmp_path / "missing.json"), "--dry-run"]) == 5
    groups = compact(status=None, limit=20)
    sections = set()
    for group in groups["groups"]:
        if group["digest"] == help_text["items"][0]["digest"]:
            assert group["simulated"] is True
        if group["digest"] == help_digest:
            assert group["backend"] == "metal"
            assert group["simulated"] is False
    filtered = compact(backend="metal", limit=20)
    assert help_digest in {item["digest"] for item in filtered["groups"]}
    assert help_text["items"][0]["digest"] not in {item["digest"] for item in filtered["groups"]}


def test_prompt_fingerprint_and_pack_closure_stay_on_the_anchor():
    """@brief 提示指纹与当前 78 文件闭包都以 live anchor 为准。"""
    from taic.trust import current_prompt_sha1
    from tests.support.live_anchors import load_live_anchors, measure_pack_closure

    anchors = load_live_anchors()
    files, digest = measure_pack_closure()
    assert current_prompt_sha1() == anchors["prompt_sha1"]
    assert digest == anchors["pack_closure"]["sha256"]
    assert len(files) == len(anchors["pack_closure"]["files"]) == 78
    assert "simulated=true" in Path("tests/support/npu_sim/varlen_worklist.py").read_text(encoding="utf-8")
    assert "publication_allowed=false" in Path("tests/support/npu_sim/varlen_worklist.py").read_text(encoding="utf-8")


def test_a_paid_search_entry_prints_open_topics_and_still_returns(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """@brief 付费入口只打印未关闭的工具 topic。"""
    from experiments.common import _hint_open_tool_topics

    ledger = tmp_path / "topics.json"
    record_tool_topic(
        ledger, "still-open", head="abc", dirty=False, modules={},
        claims=[{"run_id": "r", "seq": 1, "digest": "d", "status": "info"}],
    )
    monkeypatch.setenv("TAIC_TOOL_TOPICS", str(ledger))
    lines: list[str] = []
    _hint_open_tool_topics(lines.append)
    assert any("still-open" in line for line in lines)


def test_run_and_resume_announce_open_topics_before_paying(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys):
    """@brief taic run 和 resume 的真实入口在进入 orch.run 之前打印未关闭 topic。"""
    from types import SimpleNamespace

    from taic import cli
    from tests.support.v2_specimen import V2_TASK

    ledger = tmp_path / "topics.json"
    record_tool_topic(
        ledger, "still-open", head="abc", dirty=False, modules={},
        claims=[{"run_id": "r", "seq": 1, "digest": "d", "status": "info"}],
    )
    monkeypatch.setenv("TAIC_TOOL_TOPICS", str(ledger))
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://127.0.0.1:9")
    toml = tmp_path / "taic.toml"
    toml.write_text(
        '[llm]\ncredentials_file = "/no/such/models.json"\nprovider = "deepseek"\nmodel = "deepseek-flash"\n'
        'use_tools = false\nreasoning_effort = "off"\n'
        "[search]\nbudget_usd = 5\nbranches = 1\nmax_rounds = 0\neval_reps = 3\n"
        "stop_at_roofline = false\nabi_guard = false\ntarget_note = false\nrich_history = false\nstatus_lines = false\n"
        '[[committees]]\nname = "first"\n[[committees.members]]\nprovider = "deepseek"\nmodel = "model-a"\n'
        '[[committees]]\nname = "second"\n[[committees.members]]\nprovider = "deepseek"\nmodel = "model-b"\n',
        encoding="utf-8",
    )
    called = {"run": 0, "resume": 0}

    class Stub:
        def __init__(self, *args, **kwargs):
            return None

        async def run(self):
            called["run"] += 1
            return SimpleNamespace(
                task=V2_TASK, dtype="fp16", model="model-a", best_speedup=1.0,
                confirmed_speedup=None, spent_usd=0.0, stop_reason="max_rounds", run_dir=str(tmp_path),
            )

    monkeypatch.setattr("taic.orchestrator.Orchestrator", Stub)
    assert cli.main(["run", V2_TASK, "--size", "small", "--config", str(toml), "--run-dir", str(tmp_path / "stub")]) == 0
    assert called["run"] == 1
    assert "still-open" in capsys.readouterr().out

    class Resume:
        agent_runner = object()
        _client_injected = True

        def _guard_shape(self):
            return None

        def log(self, _message):
            return None

        def restore(self, _snap, dry_run=False):
            return None

        async def run(self):
            called["resume"] += 1
            return SimpleNamespace()

    run_dir = tmp_path / "resume-run"
    run_dir.mkdir()
    cli.run_prepared(
        SimpleNamespace(run_dir=str(run_dir), who="test"),
        Resume(),
        SimpleNamespace(partial=None, next_round=1, spent_usd=0.0),
        {"budget_usd": 1.0, "kind": "interrupted"},
        1.0,
    )
    assert called["resume"] == 1
    assert "still-open" in capsys.readouterr().out


def test_the_paid_hint_refreshes_a_changed_module_before_printing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """@brief 付费前通知按当前文件重算。变化记成 possibly_stale，不关单，覆盖仍是 not_exercised。"""
    from experiments.common import _hint_open_tool_topics

    module = tmp_path / "watched.py"
    module.write_text("value = 1\n", encoding="utf-8")
    original = hashlib.sha256(module.read_bytes()).hexdigest()
    ledger = tmp_path / "topics.json"
    record_tool_topic(
        ledger, "watched-tool", head="abc", dirty=False, modules={str(module): original},
        claims=[{"run_id": "r", "seq": 1, "digest": "d", "status": "pending"}],
    )
    module.write_text("value = 2\n", encoding="utf-8")
    monkeypatch.setenv("TAIC_TOOL_TOPICS", str(ledger))
    lines: list[str] = []
    _hint_open_tool_topics(lines.append)
    assert any("watched-tool" in line and "possibly_stale" in line for line in lines)
    topic = json.loads(ledger.read_text(encoding="utf-8"))["topics"]["watched-tool"]
    assert topic["stale_hint"] == "possibly_stale"
    assert topic["coverage"] == "not_exercised"
    assert topic["status"] == "open" and topic["change_ref"] == "" and topic["fixed"] is False
    assert topic["before"][str(module)] == original
    assert topic["after"][str(module)] == hashlib.sha256(module.read_bytes()).hexdigest()
    assert topic["observed_at"]

    def boom(_line: str) -> None:
        raise RuntimeError("log failed")

    _hint_open_tool_topics(boom)


def test_a_claim_review_records_the_topic_and_close_removes_the_paid_hint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """@brief 评审建档、代码变化只产生 possibly_stale、确认关闭后付费提示消失。"""
    import hashlib

    from experiments.common import _hint_open_tool_topics
    from taic.agent_feedback import parse_feedback
    from taic.cli import main

    from tests.shared.test_observe_feedback import _digest, _open, _record, _run

    monkeypatch.setenv("TAIC_TOOL_TOPICS", str(tmp_path / "topics.json"))
    runs, index = _open(tmp_path, monkeypatch)
    text = parse_feedback("# 需要帮助\n- 缺一把探针\n")
    directory = _run(runs, "conv", _record(text), started_at=50)
    index.sync_known()
    module = tmp_path / "watched.py"
    module.write_text("value = 1\n", encoding="utf-8")
    original = hashlib.sha256(module.read_bytes()).hexdigest()
    batch = tmp_path / "review.json"
    batch.write_text(json.dumps([{
        "seq": 1,
        "digest": _digest("需要帮助"),
        "status": "pending",
        "verification": "unchecked",
        "topic": "watched-tool",
        "expected_status": "pending",
        "note": str(module),
    }]), encoding="utf-8")
    assert main(["feedback", str(directory), "--file", str(batch), "--confirm"]) == 0
    ledger = json.loads((tmp_path / "topics.json").read_text(encoding="utf-8"))
    topic = ledger["topics"]["watched-tool"]
    assert topic["status"] == "open"
    assert topic["head"] and topic["dirty"] in (True, False)
    assert topic["modules"][str(module)] == original
    assert topic["change_ref"] == ""
    module.write_text("value = 2\n", encoding="utf-8")
    assert main(["feedback", "close-topic", "--topic", "watched-tool", "--change-ref", "commit:" + "a" * 12, "--dry-run"]) == 0
    hinted = json.loads((tmp_path / "topics.json").read_text(encoding="utf-8"))["topics"]["watched-tool"]
    assert hinted["stale_hint"] == "possibly_stale"
    assert hinted["coverage"] == "not_exercised"
    assert hinted["status"] == "open" and hinted["change_ref"] == "" and hinted["fixed"] is False
    assert hinted["after"][str(module)] != original
    assert json.loads((directory / "feedback_review.json").read_text(encoding="utf-8"))["claims"][f"1:{_digest('需要帮助')}"]["status"] == "pending"
    before = (tmp_path / "topics.json").read_bytes()
    review_before = (directory / "feedback_review.json").read_bytes()
    assert main(["feedback", "close-topic", "--topic", "watched-tool", "--change-ref", "possibly_stale", "--confirm"]) == 5
    assert (tmp_path / "topics.json").read_bytes() == before
    assert (directory / "feedback_review.json").read_bytes() == review_before
    assert main(["feedback", "close-topic", "--topic", "watched-tool", "--change-ref", "commit:" + "c" * 12, "--confirm"]) == 0
    closed = json.loads((tmp_path / "topics.json").read_text(encoding="utf-8"))["topics"]["watched-tool"]
    assert closed["status"] == "closed"
    assert closed["change_ref"] == "commit:" + "c" * 12
    assert closed["stale_hint"] == "possibly_stale"
    lines: list[str] = []
    _hint_open_tool_topics(lines.append)
    assert all("watched-tool" not in line for line in lines)


def test_a_fence_ahead_of_a_long_tail_is_still_extracted(tmp_path: Path):
    """@brief 围栏在文件上限之前提取。17000 个尾随字符不能把它切掉。"""
    from taic.agent_feedback import parse_feedback, read_feedback_text

    text = (
        "# 对系统的建议\n"
        "```harness-proposal\n"
        '{"action":"rename the skill","audience":"developer","needs":"repo"}\n'
        "```\n"
        + ("尾" * 17000)
    )
    parsed = parse_feedback(text)
    assert parsed.get("proposal_blocks")
    block = parsed["proposal_blocks"][0]
    assert block["section"] == "对系统的建议"
    assert "rename the skill" in block["raw"]
    assert len(block["raw"]) <= 4000
    assert len(parsed["sections"].get("对系统的建议", "")) <= 1200
    path = tmp_path / "FEEDBACK.md"
    path.write_text(text, encoding="utf-8")
    full, over = read_feedback_text(path)
    assert over is True and full.startswith("# 对系统的建议")
    assert parse_feedback(full, file_over=over).get("proposal_blocks")


def test_a_capped_session_result_still_parses_a_leading_fence(tmp_path: Path):
    """@brief 会话结果只留尾部时，超长文件开头的围栏仍从磁盘全文提取。"""
    from types import SimpleNamespace

    from taic.agent_feedback import FILE_CAP
    from taic.orchestrator import Orchestrator

    text = (
        "# 对系统的建议\n"
        "```harness-proposal\n"
        '{"action":"rename the skill","audience":"developer","needs":"repo"}\n'
        "```\n"
        + ("尾" * 17000)
    )
    path = tmp_path / "FEEDBACK.md"
    path.write_text(text, encoding="utf-8")
    stored = text[-FILE_CAP:]
    assert not stored.startswith("# 对系统的建议")
    parsed = Orchestrator._parsed_feedback(
        SimpleNamespace(), tmp_path, SimpleNamespace(feedback=stored), 0.0,
    )
    assert parsed.get("proposal_blocks")
    assert "rename the skill" in parsed["proposal_blocks"][0]["raw"]


def test_paired_observation_holds_the_same_gpu_lock(tmp_path: Path):
    """@brief 配对观测持有 differential_recheck 用的那把 gpu_lock，并且不堵在事件循环线程上。"""
    import asyncio
    import threading

    from tests.support.orchestrator import code, llm_script, make_orch

    orch = make_orch(tmp_path, llm_script(code("speed=1.0")), [], max_rounds=1)
    started = threading.Event()
    release = threading.Event()
    seen = {}

    def observation(_cand, _best):
        seen["locked"] = orch.gpu_lock.locked()
        seen["thread"] = threading.get_ident()
        started.set()
        assert release.wait(2)
        return {"applied": False, "changes_seed": False, "gpu": False, "reason": "probe"}

    orch._paired_seed_observation = observation

    class Cand:
        source = "kernel void a() {}\n"
        label = "a"

    async def other(flag: dict):
        async with orch.gpu_lock:
            flag["entered"] = True

    async def main():
        task = asyncio.create_task(orch._paired_under_lock(Cand(), Cand()))
        while not started.is_set():
            await asyncio.sleep(0.01)
        flag = {"entered": False}
        waiter = asyncio.create_task(other(flag))
        await asyncio.sleep(0.05)
        assert flag["entered"] is False
        assert seen["locked"] is True
        release.set()
        await task
        await waiter
        assert flag["entered"] is True

    caller = threading.get_ident()
    asyncio.run(main())
    assert seen["thread"] != caller


def test_the_host_rerun_holds_the_gpu_lock_off_the_event_loop(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """@brief 提名重跑持有同一把 asyncio GPU 锁，并且不在事件循环线程上计时。"""
    import threading

    from taic.paired_compare import append_nomination
    from tests.support.orchestrator import code, llm_script, make_orch

    orch = make_orch(tmp_path, llm_script(code("speed=1.0")), [], max_rounds=1)
    work = orch.run_dir / "agent_work" / "init"
    folder = orch.run_dir / "candidates" / "000_init"
    work.mkdir(parents=True)
    folder.mkdir(parents=True)
    left = tmp_path / "left.metal"
    right = tmp_path / "right.metal"
    left.write_text("kernel void left() {}\n", encoding="utf-8")
    right.write_text("kernel void right() {}\n", encoding="utf-8")
    sha_a = hashlib.sha256(left.read_bytes()).hexdigest()
    sha_b = hashlib.sha256(right.read_bytes()).hexdigest()
    append_nomination(work / ".selfcheck", {
        "sha_a": sha_a, "sha_b": sha_b, "path_a": str(left), "path_b": str(right), "conclusion": "no_diff",
    })
    host_dir = tmp_path / "host-authority"
    monkeypatch.setattr(
        "taic.paired_compare.host_evidence_path",
        lambda _directory: host_dir / "host_paired_comparisons.jsonl",
    )
    seen = {}
    started = threading.Event()
    release = threading.Event()

    def fake_run(payload, **kwargs):
        seen["thread"] = threading.get_ident()
        seen["locked"] = orch.gpu_lock.locked()
        seen["deadline"] = kwargs.get("launch_deadline_s")
        seen["timeout"] = kwargs.get("timeout")
        seen["module"] = kwargs.get("worker_module")
        started.set()
        assert release.wait(2)
        return {"paired": {
            "applied": True, "elapsed_s": 0.01, "sha_a": sha_a, "sha_b": sha_b,
            "changes_seed": False, "gpu": True,
        }}

    monkeypatch.setattr("tcenv.v2_worker.run_bounded_v2", fake_run)

    async def other(flag: dict):
        async with orch.gpu_lock:
            flag["entered"] = True

    async def main():
        task = asyncio.create_task(orch._select_paired_nominations(folder, work))
        while not started.is_set():
            await asyncio.sleep(0.01)
        flag = {"entered": False}
        waiter = asyncio.create_task(other(flag))
        await asyncio.sleep(0.05)
        assert flag["entered"] is False
        assert seen["locked"] is True
        release.set()
        await task
        await waiter
        assert flag["entered"] is True

    caller = threading.get_ident()
    asyncio.run(main())
    assert seen["thread"] != caller
    assert seen["deadline"] == 8.0
    assert seen["timeout"] == 420.0
    assert seen["module"] == "tcenv.backends.metal.paired_compare"


def test_a_forged_host_line_in_the_model_directory_does_not_count(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """@brief 模型目录里预先写好的 host evidence 不能跳过宿主重跑。"""
    from tcenv.selfcheck import authority_root
    from taic.paired_compare import append_nomination, host_evidence_path, host_select

    left = tmp_path / "left.metal"
    right = tmp_path / "right.metal"
    left.write_text("kernel void left() {}\n", encoding="utf-8")
    right.write_text("kernel void right() {}\n", encoding="utf-8")
    directory = tmp_path / ".selfcheck"
    sha_a = hashlib.sha256(left.read_bytes()).hexdigest()
    sha_b = hashlib.sha256(right.read_bytes()).hexdigest()
    directory.mkdir()
    forged = {"applied": True, "host_evidence": True, "sha_a": sha_a, "sha_b": sha_b}
    (directory / "host_paired_comparisons.jsonl").write_text(json.dumps(forged) + "\n", encoding="utf-8")
    append_nomination(directory, {
        "sha_a": sha_a, "sha_b": sha_b, "path_a": str(left), "path_b": str(right), "conclusion": "no_diff",
    })
    real = host_evidence_path(directory)
    assert real.parent.parent == authority_root()
    assert directory not in real.parents
    host_dir = tmp_path / "host-authority"
    monkeypatch.setattr(
        "taic.paired_compare.host_evidence_path",
        lambda _directory: host_dir / "host_paired_comparisons.jsonl",
    )
    calls = {"n": 0}

    def rerun(_selected):
        calls["n"] += 1
        return [{
            "applied": True, "elapsed_s": 0.01, "sha_a": sha_a, "sha_b": sha_b,
            "median_a_s": 1.0, "median_b_s": 1.0, "ratios": [1.0], "interval": [0.9, 1.1],
            "rounds": 2, "cv": 0.0, "changes_seed": False,
        }]

    saved = host_select(directory, rerun=rerun)
    assert calls["n"] == 1
    assert saved["reviewed"] == 1
    assert all(item.get("omit_reason") != "equivalent_host_evidence" for item in saved["uncovered"])
    assert (host_dir / "host_paired_comparisons.jsonl").is_file()
    assert (directory / "host_paired_comparisons.jsonl").read_text(encoding="utf-8") == json.dumps(forged) + "\n"


def test_a_rejected_claim_blocks_the_whole_topic_close(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """@brief 账本里另一条已变成 rejected 且索引已同步时，整批评审、账本和事件都不写。"""
    from observe.actions import Refused, close_feedback_topic, review_feedback
    from taic.agent_feedback import parse_feedback
    from taic.feedback_topics import ledger_path
    from taic.review_ledger import ledger_path as review_events_path

    from tests.shared.test_observe_feedback import _open, _record, _run

    monkeypatch.setenv("TAIC_TOOL_TOPICS", str(tmp_path / "topics.json"))
    runs, index = _open(tmp_path, monkeypatch)
    text = parse_feedback("# 需要帮助\n- 第一条探针\n- 第二条探针\n")
    run = _run(runs, "split", _record(text), started_at=10)
    index.sync_known()
    help_items = [item for item in text["items"] if item["section"] == "需要帮助"]
    assert len(help_items) == 2
    first, second = help_items[0]["digest"], help_items[1]["digest"]
    pending = {"status": "pending", "verification": "unchecked", "topic": "split-topic", "expected_status": "pending"}
    review_feedback("split", 1, first, dict(pending))
    review_feedback("split", 1, second, dict(pending))
    review_feedback("split", 1, second, {
        "status": "rejected", "verification": "refuted", "topic": "split-topic",
        "expected_status": "pending", "note": "索引已同步",
    })
    review_file = run / "feedback_review.json"
    events = review_events_path()
    ledger = ledger_path()
    before_review = review_file.read_bytes()
    before_ledger = ledger.read_bytes()
    before_events = events.read_bytes() if events.is_file() else None
    with pytest.raises(Refused, match="item 1"):
        close_feedback_topic("split-topic", "commit:" + "f" * 12, confirm=True)
    assert review_file.read_bytes() == before_review
    assert ledger.read_bytes() == before_ledger
    after_events = events.read_bytes() if events.is_file() else None
    assert after_events == before_events
    statuses = {
        row["digest"]: row["status"]
        for row in index.conn.execute(
            "SELECT digest, status FROM feedback_claims WHERE run_id = ?", ("split",),
        )
    }
    assert statuses[first] == "pending"
    assert statuses[second] == "rejected"


def test_a_second_run_mismatch_at_apply_leaves_the_first_run_unchanged(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """@brief 第二个 run 在写入时失配，第一个 run 的评审文件逐字节不变。"""
    from observe.actions import Refused, close_feedback_topic, review_feedback
    from observe.feedback import apply_reviews

    from tests.shared.test_observe_feedback import _digest, _open, _parsed, _record, _run

    events = tmp_path / "review-events.jsonl"
    monkeypatch.setenv("TAIC_REVIEW_EVENTS", str(events))
    monkeypatch.setenv("TAIC_TOOL_TOPICS", str(tmp_path / "topics.json"))
    runs, index = _open(tmp_path, monkeypatch)
    text = _parsed()
    first = _run(runs, "alpha", _record(text), started_at=10)
    second = _run(runs, "beta", _record(text), started_at=20)
    index.sync_known()
    digest = _digest("需要帮助")
    body = {"status": "pending", "verification": "unchecked", "topic": "atomic", "expected_status": "pending"}
    review_feedback("alpha", 1, digest, dict(body))
    review_feedback("beta", 1, digest, dict(body))
    before_first = (first / "feedback_review.json").read_bytes()
    before_second = (second / "feedback_review.json").read_bytes()
    before_ledger = (tmp_path / "topics.json").read_bytes()
    before_events = events.read_bytes() if events.is_file() else None
    real = apply_reviews

    def fail_second(run_dir, rows, **kwargs):
        if Path(run_dir).name == "beta":
            raise ValueError("expected_status does not match")
        return real(run_dir, rows, **kwargs)

    monkeypatch.setattr("observe.feedback.apply_reviews", fail_second)
    with pytest.raises(Refused, match="expected_status"):
        close_feedback_topic("atomic", "commit:" + "d" * 12, confirm=True)
    assert (first / "feedback_review.json").read_bytes() == before_first
    assert (second / "feedback_review.json").read_bytes() == before_second
    assert (tmp_path / "topics.json").read_bytes() == before_ledger
    after_events = events.read_bytes() if events.is_file() else None
    assert after_events == before_events
    status = index.conn.execute(
        "SELECT status FROM feedback_claims WHERE run_id = ? AND digest = ?", ("alpha", digest),
    ).fetchone()
    assert status["status"] == "pending"


def test_a_refresh_failure_rolls_back_review_events_and_reindexes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """@brief 后置索引刷新失败时，评审事件回到原字节，第一个 run 重新读回 pending。"""
    from observe.actions import Refused, close_feedback_topic, review_feedback

    from tests.shared.test_observe_feedback import _digest, _open, _parsed, _record, _run

    events = tmp_path / "review-events.jsonl"
    monkeypatch.setenv("TAIC_REVIEW_EVENTS", str(events))
    monkeypatch.setenv("TAIC_TOOL_TOPICS", str(tmp_path / "topics.json"))
    runs, index = _open(tmp_path, monkeypatch)
    text = _parsed()
    first = _run(runs, "alpha", _record(text), started_at=10)
    second = _run(runs, "beta", _record(text), started_at=20)
    index.sync_known()
    digest = _digest("需要帮助")
    body = {"status": "pending", "verification": "unchecked", "topic": "atomic", "expected_status": "pending"}
    review_feedback("alpha", 1, digest, dict(body))
    review_feedback("beta", 1, digest, dict(body))
    before_first = (first / "feedback_review.json").read_bytes()
    before_events = events.read_bytes() if events.is_file() else None
    calls = {"beta": 0}
    real = index.refresh_feedback

    def fail_beta(run_id, directory):
        if run_id == "beta" and calls["beta"] == 0:
            calls["beta"] += 1
            raise RuntimeError("refresh failed")
        return real(run_id, directory)

    monkeypatch.setattr(index, "refresh_feedback", fail_beta)
    with pytest.raises(RuntimeError, match="refresh failed"):
        close_feedback_topic("atomic", "commit:" + "e" * 12, confirm=True)
    assert (first / "feedback_review.json").read_bytes() == before_first
    after_events = events.read_bytes() if events.is_file() else None
    assert after_events == before_events
    assert calls["beta"] >= 1
    status = index.conn.execute(
        "SELECT status FROM feedback_claims WHERE run_id = ? AND digest = ?", ("alpha", digest),
    ).fetchone()
    assert status["status"] == "pending"


def test_mixed_real_and_simulated_runs_keep_separate_claim_labels(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """@brief 同一 digest 在真实 run 和模拟 run 上各有自己的 simulated 标签。"""
    from observe.feedback import entries
    from observe.suggestions import compact
    from taic.agent_feedback import parse_feedback

    from tests.shared.test_observe_feedback import _open, _record, _run

    runs, index = _open(tmp_path, monkeypatch)
    text = parse_feedback("# 对系统的建议\n- 同一句话\n")
    real = _run(runs, "real-run", _record(text), started_at=10)
    (real / "result.json").write_text(json.dumps({"simulated": False, "measurement_kind": "measured"}), encoding="utf-8")
    simulated = _run(runs, "sim-run", _record(text), started_at=20)
    (simulated / "result.json").write_text(json.dumps({"simulated": True, "measurement_kind": "simulated"}), encoding="utf-8")
    index.sync_known()
    real_claims = entries(real)["records"][0]["claims"]
    simulated_claims = entries(simulated)["records"][0]["claims"]
    assert real_claims[0]["digest"] == simulated_claims[0]["digest"]
    assert real_claims[0]["simulated"] is False
    assert simulated_claims[0]["simulated"] is True
    group = next(item for item in compact(limit=20)["groups"] if item["digest"] == real_claims[0]["digest"])
    flags = {item["run_id"]: item["simulated"] for item in group["claim_simulated"]}
    assert flags == {"real-run": False, "sim-run": True}
    assert group["simulated"] is None
