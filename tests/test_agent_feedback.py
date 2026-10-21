"""Parsing FEEDBACK.md and rendering peer tips. No model calls and no GPU."""

import json
from pathlib import Path

from taic.agent_feedback import _kinds, parse_feedback, peer_tips_for


def _write(path: Path, rows: list[dict], *, broken_at: int | None = None) -> None:
    lines = []
    for index, row in enumerate(rows, start=1):
        if broken_at == index:
            lines.append("{not json")
        lines.append(json.dumps(row, ensure_ascii=False))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _row(seq: int, items: list[dict], *, label: str = "r1.p0") -> dict:
    return {
        "format": "agent_feedback",
        "v": 1,
        "seq": seq,
        "label": label,
        "round": seq,
        "committee": "first",
        "provider": "deepseek",
        "model": "deepseek-flash",
        "outcome": "evaluated",
        "exit": "normal",
        "best_speedup": 1.1,
        "items": items,
    }


def test_kinds_sections_and_digest_ignore_numbers_but_keep_dtype_names():
    slow = parse_feedback(
        "kind: tool, prompt, bogus\n\n"
        "# 对系统的建议\n- 成绩单在约790µs处截断\n"
        "# 给下一棒的避坑提示\n- float8 是保留字\n"
    )
    again = parse_feedback("# 对系统的建议\n- 成绩单在约1100µs处截断\n")
    assert slow["kinds"] == "tool,prompt"
    assert slow["section_names"] == ["对系统的建议", "给下一棒的避坑提示"]
    assert slow["items"][0]["digest"]
    assert slow["items"][0]["digest"] == again["items"][0]["digest"]
    assert "fp16" in parse_feedback("# 被否决的前提\n- [未验证] 矩阵单元只能吃 fp16\n")["items"][0]["text"]
    measured = parse_feedback("# 被否决的前提\n- [已实测: probe.py | 速率 | 15.09] 矩阵单元只能吃 fp16\n")
    assert measured["items"][0]["digest"] == parse_feedback("# 被否决的前提\n- [未验证] 矩阵单元只能吃 fp16\n")["items"][0]["digest"]
    other = parse_feedback("# 给下一棒的避坑提示\n- [未验证] 矩阵单元只能吃 fp16\n")
    assert other["items"][0]["digest"] != measured["items"][0]["digest"]
    assert parse_feedback("kind: nope\n")["kinds"] == "none"
    assert parse_feedback("# 给下一棒的避坑提示\n- 只是一个坑\n")["items"][0]["digest"]


def test_a_long_system_section_does_not_drop_the_peer_sections(tmp_path):
    text = "kind: prompt\n\n# 需要帮助\n" + ("甲" * 5000) + "\n\n# 给下一棒的避坑提示\n- 分支里不能 simd_any\n# 被否决的前提\n- [未验证] 没有探针\n"
    parsed = parse_feedback(text, file_over=True)
    assert parsed["truncated"] is True
    assert len(parsed["sections"]["需要帮助"]) == 1200
    assert any(item["section"] == "给下一棒的避坑提示" for item in parsed["items"])
    assert any(item["section"] == "被否决的前提" for item in parsed["items"])
    unknown = parse_feedback("v: 99\n# 需要帮助\nhello\n")
    assert unknown["unrecognized_version"] is True and unknown["items"] == [] and "hello" in unknown["raw"]


def test_a_missing_probe_or_an_escaping_path_is_downgraded(tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    (work / "probe_fmma.py").write_text("print(1)\n")
    ok = parse_feedback(
        "# 被否决的前提\n- [已实测: probe_fmma.py | fp32 乘加速率 | 15.09] fp32 不慢\n",
        work=work, repo=tmp_path,
    )
    assert ok["items"][0]["premise"] == "verified"
    missing = parse_feedback(
        "# 被否决的前提\n- [已实测: missing.py | 速率 | 1] fp32 不慢\n",
        work=work, repo=tmp_path,
    )
    assert missing["items"][0]["premise"] == "downgraded"
    assert "探针文件不存在" in missing["items"][0]["text"]
    assert "missing.py" in missing["raw"]
    escaped = parse_feedback(
        "# 被否决的前提\n- [已实测: ../../etc/passwd | 速率 | 1] 不该读到\n",
        work=work, repo=tmp_path,
    )
    assert escaped["items"][0]["premise"] == "downgraded"


def test_peer_tips_expire_unverified_claims_and_stay_inside_one_run(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    claim = {"section": "被否决的前提", "audience": "peer", "text": "[未验证] 矩阵单元只能吃 fp16", "digest": "same", "premise": "unverified"}
    rows = [
        _row(1, [claim], label="r1.p0"),
        _row(2, [claim], label="r2.p0"),
        _row(3, [claim], label="r3.p0"),
    ]
    _write(run / "feedback.jsonl", rows[:2], broken_at=2)
    assert "fp16" in peer_tips_for(run)
    _write(run / "feedback.jsonl", rows, broken_at=2)
    assert peer_tips_for(run) == ""
    upgraded = dict(claim, text="[已实测: probe.py | 速率 | 15] 矩阵单元只能吃 fp16", premise="verified")
    rows.append(_row(5, [upgraded], label="r5.p0"))
    _write(run / "feedback.jsonl", rows)
    assert "r5.p0" in peer_tips_for(run)
    other = tmp_path / "other"
    other.mkdir()
    assert peer_tips_for(other) == ""


def test_silent_sessions_still_age_an_unverified_premise(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    claim = {"section": "被否决的前提", "audience": "peer", "text": "[未验证] 矩阵单元只能吃 fp16", "digest": "same", "premise": "unverified"}
    _write(run / "feedback.jsonl", [_row(1, [claim], label="r1.p0"), _row(2, [], label="r2.p0")])
    assert "fp16" in peer_tips_for(run)
    _write(run / "feedback.jsonl", [_row(1, [claim], label="r1.p0"), _row(2, [], label="r2.p0"), _row(3, [], label="r3.p0")])
    assert peer_tips_for(run) == ""


def test_peer_tips_keep_eight_and_skip_a_broken_middle_line(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    rows = []
    for seq in range(1, 10):
        rows.append(_row(seq, [{
            "section": "给下一棒的避坑提示",
            "audience": "peer",
            "text": f"坑 {seq} 号，编译器拒绝这个写法",
            "digest": f"d{seq}",
            "premise": "",
        }], label=f"r{seq}.p0"))
    _write(run / "feedback.jsonl", rows)
    text = peer_tips_for(run)
    assert "r1.p0" not in text and "r2.p0" in text and "r9.p0" in text
    assert text.count("\n") + 1 == 8


def test_the_contract_tells_the_agent_to_write_nothing_rather_than_a_placeholder():
    """An empty section must cost zero tokens: the brief says omit it, not "无" or "（空）".

    Output is billed, and a placeholder also becomes a reviewable claim on the board (7 of 44 pending
    claims in runs/conv2d-feedback-20261021 were literally "无。"/"（空）").
    """
    from pathlib import Path

    from tasks import get_task

    from taic.kernel_agent import SessionBrief, render_contract

    task = get_task("conv2d", "fp16", "small")
    contract = task.contract
    brief = SessionBrief(
        task_name="conv2d", dtype="fp16", size="small", repo=Path("/repo"), label="r1.p0",
        kernel_name=contract.kernel_name, triton_source=task.triton_source, description=task.description,
        contract_text=contract.describe(), abi_stub="", rtol=task.rtol, atol_rel=task.atol_rel,
        min_improvement=1.005, noise_z=2.0,
    )
    text = render_contract(brief)
    assert "没有内容就不要写" in text
    assert "不要写「无」「（空）」「none」这类占位符" in text
    assert "整份 `FEEDBACK.md` 都没有内容时，不要创建这个文件" in text
    # 没有系统建议时连 kind 行都不写；解析侧仍然把缺失视为 none。
    assert "没有系统层面的建议或求助时，这一行也不要写" in text
    assert _kinds("## 给下一棒的避坑提示\n- x\n") == "none"


def test_a_file_with_only_sections_left_blank_is_not_a_record():
    """The parser already treats an empty file as nothing; the brief change keeps it from happening at all."""
    parsed = parse_feedback("", work=None, repo=None)
    assert parsed["raw"] == "" and parsed["items"] == []
