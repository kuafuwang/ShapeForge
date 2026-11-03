"""Parsing FEEDBACK.md and rendering peer tips. No model calls and no GPU."""

import json
from pathlib import Path

from tests.support.v1_specimen import specimen_task  # B2：conv2d/lion 已迁 v2 的样本夹具

from taic.agent_feedback import VERIFIED_TIPS_CHARS, _kinds, load_records, main, parse_feedback, peer_tips_for


def _write(path: Path, rows: list[dict], *, broken_at: int | None = None) -> None:
    """@brief 把若干记录写成 jsonl，可在指定序号处插一行坏 JSON。

    @param path 目标文件。
    @param rows 记录列表。
    @param broken_at 从 1 开始的行号，在该行前插入一行坏 JSON。
    """
    lines = []
    for index, row in enumerate(rows, start=1):
        if broken_at == index:
            lines.append("{not json")
        lines.append(json.dumps(row, ensure_ascii=False))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _row(seq: int, items: list[dict], *, label: str = "r1.p0") -> dict:
    """@brief 造一条形状完整的 ``agent_feedback`` 记录。

    @param seq 记录序号。
    @param items 该记录的候选项列表。
    @param label 候选标签。
    @return 记录字典。
    """
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
    """@brief 验证 kind 行只留已知取值、章节名按顺序记录，摘要忽略数字但保留 fp16 这类 dtype 名。

    同时核对同一命题在不同前提状态、不同章节下的摘要取合与区分。
    """
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
    """@brief 验证超长的系统建议被截断成 1200 字，但后面的同伴提示与被否决前提仍被解析；未知版本号被标记并保留原文。

    @param tmp_path pytest 临时目录（本用例不落盘）。
    """
    text = "kind: prompt\n\n# 需要帮助\n" + ("甲" * 5000) + "\n\n# 给下一棒的避坑提示\n- 分支里不能 simd_any\n# 被否决的前提\n- [未验证] 没有探针\n"
    parsed = parse_feedback(text, file_over=True)
    assert parsed["truncated"] is True
    assert len(parsed["sections"]["需要帮助"]) == 1200
    assert any(item["section"] == "给下一棒的避坑提示" for item in parsed["items"])
    assert any(item["section"] == "被否决的前提" for item in parsed["items"])
    unknown = parse_feedback("v: 99\n# 需要帮助\nhello\n")
    assert unknown["unrecognized_version"] is True and unknown["items"] == [] and "hello" in unknown["raw"]


def test_a_missing_probe_or_an_escaping_path_is_downgraded(tmp_path):
    """@brief 验证探针文件不存在、或路径逃出工作目录时，实测前提被降级为未验证并写明原因。

    @param tmp_path 临时目录，工作目录与探针文件造在这里。
    """
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
    """@brief 验证同伴提示里未验证的命题两次会话后就过期，已验证的保留，并且不跟到其他运行。

    @param tmp_path 临时目录，两个运行目录造在这里。
    """
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
    """@brief 验证即使后续会话没写任何内容，未验证的命题也会按时老化过期。

    @param tmp_path 临时目录，运行目录造在这里。
    """
    run = tmp_path / "run"
    run.mkdir()
    claim = {"section": "被否决的前提", "audience": "peer", "text": "[未验证] 矩阵单元只能吃 fp16", "digest": "same", "premise": "unverified"}
    _write(run / "feedback.jsonl", [_row(1, [claim], label="r1.p0"), _row(2, [], label="r2.p0")])
    assert "fp16" in peer_tips_for(run)
    _write(run / "feedback.jsonl", [_row(1, [claim], label="r1.p0"), _row(2, [], label="r2.p0"), _row(3, [], label="r3.p0")])
    assert peer_tips_for(run) == ""


def test_peer_tips_keep_eight_and_skip_a_broken_middle_line(tmp_path):
    """@brief 验证同伴提示最多保留 8 行、丢掉最早的一条，并跳过中间一行坏 JSON。

    @param tmp_path 临时目录，运行目录造在这里。
    """
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
    """@brief 验证契约要求空章节直接省略，而不是写「无」或「（空）」这类占位符。

    空章节必须零 token：简报说整节省略。输出要计费，占位符还会变成看板上可审阅的主张
    （runs/conv2d-feedback-20261021 里 44 条待审主张中有 7 条就是「无。」「（空）」）。
    """
    from pathlib import Path

    from tasks import get_task

    from taic.kernel_agent import SessionBrief, render_contract

    task = specimen_task("conv2d", "fp16", "small")
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
    assert "缺权限、sudo、许可证、凭据：当场写 `# 需要帮助`" in text
    assert "能绕开也写" in text
    assert _kinds("## 给下一棒的避坑提示\n- x\n") == "none"


def test_a_file_with_only_sections_left_blank_is_not_a_record():
    """@brief 验证只有空章节、没有任何内容的文件不算一条记录。

    解析器本来就把空文件当成空；简报的改动让这种文件根本不会被创建。
    """
    parsed = parse_feedback("", work=None, repo=None)
    assert parsed["raw"] == "" and parsed["items"] == []


def test_a_bracket_only_claim_keeps_its_proposition_and_ignores_the_probe(tmp_path):
    """@brief 验证只有方括号、没有命题正文时仍保留该命题，且摘要不随探针与结论来源（分析否决、推理否决）变化。

    @param tmp_path 临时目录，工作目录与探针文件造在这里。
    """
    work = tmp_path / "work"
    work.mkdir()
    (work / "probe.py").write_text("print(1)\n")
    measured = parse_feedback(
        "# 被否决的前提\n- [已实测: probe.py | 同一命题 | 15.09]\n",
        work=work, repo=tmp_path,
    )
    other_probe = parse_feedback("# 被否决的前提\n- [已实测: other.py | 同一命题 | 99]\n", work=work, repo=tmp_path)
    piped = parse_feedback("# 被否决的前提\n- [未验证 | 同一命题 | 这一轮没测]\n")
    assert measured["items"][0]["digest"]
    assert measured["items"][0]["premise"] == "verified"
    assert measured["items"][0]["digest"] == other_probe["items"][0]["digest"] == piped["items"][0]["digest"]
    assert other_probe["items"][0]["premise"] == "downgraded"
    assert other_probe["items"][0]["text"].startswith("[未验证: 探针文件不存在 | 同一命题 | 99]")
    assert "other.py" not in other_probe["items"][0]["text"]
    analysed = parse_feedback("# 被否决的前提\n- [分析否决: 旧笔记 | 同一命题 | 没测过]\n")
    reasoned = parse_feedback("# 被否决的前提\n- [推理否决: 上一场 | 同一命题 | 理由不同]\n")
    assert analysed["items"][0]["digest"] == reasoned["items"][0]["digest"] == piped["items"][0]["digest"]


def test_a_probe_with_arguments_or_a_sibling_session_still_counts(tmp_path):
    """@brief 验证探针带参数、写成兄弟会话的路径、或一行提两个文件时，都仍算实测。

    @param tmp_path 临时目录，两个会话目录与探针文件造在这里。
    """
    agent = tmp_path / "agent_work"
    here = agent / "r3.p0"
    here.mkdir(parents=True)
    (here / "sim_f16.py").write_text("print(1)\n")
    sibling = agent / "r9.p0"
    sibling.mkdir()
    (sibling / "sim_w63.py").write_text("print(1)\n")
    (here / "gen.py").write_text("print(1)\n")
    args = parse_feedback(
        "# 被否决的前提\n- [已实测: sim_f16.py --half | 操作数 | 1]\n",
        work=here, repo=tmp_path,
    )
    sibling_claim = parse_feedback(
        "# 被否决的前提\n- [已实测: r9.p0/sim_w63.py | 宽度 | 2]\n",
        work=here, repo=tmp_path,
    )
    either = parse_feedback(
        "# 被否决的前提\n- [已实测: bench3.py + gen.py 扫描 | 口径 | 3]\n",
        work=here, repo=tmp_path,
    )
    assert args["items"][0]["premise"] == "verified"
    assert sibling_claim["items"][0]["premise"] == "verified"
    assert either["items"][0]["premise"] == "verified"


def test_a_placeholder_section_is_not_an_item_and_a_sentence_is():
    """@brief 验证整节只有占位符时不产生条目、并把这几个章节记为占位，同时保留原文。"""
    parsed = parse_feedback("# 需要帮助\n无。\n# 卡住的原因\n没有阻塞性的事\n# 对系统的建议\n（空）\n")
    assert parsed["placeholder_sections"] == ["需要帮助", "对系统的建议", "卡住的原因"]
    assert parsed["placeholder_items"] == 3
    assert parsed["items"] == []
    assert "无" in parsed["raw"]


def test_the_seven_audited_absence_notes_are_placeholders_and_a_request_is_kept():
    """@brief 验证审计出的七条“无事可报”说法都按占位符处理，而真正的请求会被保留。

    用例里的句子取自 docs/feedback_audit_20261022.json 的 drop_empty 集合。
    """
    notes = [
        "无（预算 $14.79/$15.00 已尽，后续各棒基本只能复测种子）。",
        "无。本轮不需要外部改动就能推进，卡点是硬件预算是死的（见 NOTES §3）。",
        "没有阻塞性的事。这一轮不需要人介入；下面两条只是想让人看一眼。",
        "没有。本轮不需要人介入，交回的核已验证（correct / safe / validated）。",
        "无。核已合格、比种子快（交错 A/B 一致约 1.05×），`--timed --validate` 三次里有两次印「够换种子」。",
        "无。本目录的核已经能编译、过六用例、`--timed --validate` 三次都印「够换种子」。",
        "无（这轮能拿到的东西都拿到了；真正值得系统层面改的是下面「被否决的前提」里那条\n"
        "——`experience.md` 里「32 个累加器片段就溢出」的说法在这一题上被过度推广了，见下）。",
    ]
    for note in notes:
        parsed = parse_feedback("# 需要帮助\n" + note + "\n")
        assert parsed["items"] == [], note
        assert parsed["placeholder_sections"] == ["需要帮助"], note
    request = parse_feedback("# 需要帮助\n无。请修一下 `taic.agent_feedback check` 的退出码。\n")
    assert len(request["items"]) == 1
    assert request["placeholder_sections"] == []
    mixed = parse_feedback("# 对系统的建议\n- 无。\n- 请加一个配对 A/B 报告。\n")
    assert [item["text"] for item in mixed["items"]] == ["请加一个配对 A/B 报告。"]
    assert mixed["placeholder_items"] == 1


def test_a_capped_section_names_itself():
    """@brief 验证被截断的章节如实报出自己的名字，并截到 500 字。"""
    parsed = parse_feedback("# 给下一棒的避坑提示\n" + ("坑" * 600) + "\n")
    assert parsed["truncated"] is True
    assert parsed["truncated_sections"] == ["给下一棒的避坑提示"]
    assert len(parsed["sections"]["给下一棒的避坑提示"]) == 500


def test_verified_tips_have_their_own_budget_and_a_downgrade_does_not_erase_them(tmp_path):
    """@brief 验证已验证的提示有独立预算（按字符裁剪并保留较新的条目），降级记录不会抹掉同命题的已验证条目。

    @param tmp_path 临时目录，运行目录造在这里。
    """
    run = tmp_path / "run"
    run.mkdir()
    rows = []
    for seq in range(1, 10):
        rows.append(_row(seq, [{
            "section": "被否决的前提", "audience": "peer", "premise": "verified",
            "text": f"命题 {seq}", "digest": f"v{seq}",
        }], label=f"r{seq}.p0"))
    _write(run / "feedback.jsonl", rows)
    kept = peer_tips_for(run)
    assert kept.count("\n") + 1 == 9
    assert "r1.p0" in kept and "r9.p0" in kept
    assert kept.index("r1.p0") < kept.index("r9.p0")
    long_rows = []
    for seq in range(1, 18):
        long_rows.append(_row(seq, [{
            "section": "被否决的前提", "audience": "peer", "premise": "verified",
            "text": f"命题{seq} " + ("甲" * 200), "digest": f"long{seq}",
        }], label=f"r{seq}.p0"))
    _write(run / "feedback.jsonl", long_rows)
    trimmed = peer_tips_for(run)
    assert "r1.p0" not in trimmed and "r17.p0" in trimmed
    assert sum(len(line) + 1 for line in trimmed.splitlines()) <= VERIFIED_TIPS_CHARS
    downgraded = {
        "section": "被否决的前提", "audience": "peer", "premise": "downgraded",
        "text": "降级后的句子", "digest": "same",
    }
    verified = dict(downgraded, premise="verified", text="实测命题")
    _write(run / "feedback.jsonl", [
        _row(1, [verified], label="r1.p0"),
        _row(2, [downgraded], label="r2.p0"),
    ])
    again = peer_tips_for(run)
    assert "实测命题" in again and "降级后的句子" not in again and "r1.p0" in again
    expired = {
        "section": "被否决的前提", "audience": "peer", "premise": "downgraded",
        "text": "探针没了", "digest": "gone",
    }
    _write(run / "feedback.jsonl", [_row(1, [expired], label="r1.p0"), _row(2, [], label="r2.p0"), _row(3, [], label="r3.p0")])
    assert peer_tips_for(run) == ""


def test_unverified_and_ordinary_tips_share_the_eight_lines(tmp_path):
    """@brief 验证未验证提示与普通提示共用 8 行额度，并优先保留较新的会话。

    @param tmp_path 临时目录，运行目录造在这里。
    """
    run = tmp_path / "run"
    run.mkdir()
    rows = []
    for seq in range(1, 5):
        rows.append(_row(seq, [{
            "section": "给下一棒的避坑提示", "audience": "peer", "premise": "",
            "text": f"普通坑 {seq}", "digest": f"o{seq}",
        }], label=f"r{seq}.p0"))
    rows.append(_row(9, [{
        "section": "被否决的前提", "audience": "peer", "premise": "unverified",
        "text": f"未验证 {index}", "digest": f"u{index}",
    } for index in range(6)], label="r9.p0"))
    _write(run / "feedback.jsonl", rows)
    text = peer_tips_for(run)
    assert text.count("\n") + 1 == 8
    assert text.count("未验证") == 6
    assert "r4.p0" in text and "r3.p0" in text
    assert "r1.p0" not in text and "r2.p0" not in text


def test_old_records_regain_a_digest_without_rewriting_the_file(tmp_path):
    """@brief 验证缺摘要的旧记录在读取时补齐摘要与被截断章节，不重写文件，且重复命题只提示一次。

    @param tmp_path 临时目录，运行目录造在这里。
    """
    run = tmp_path / "run"
    run.mkdir()
    piece = "[已实测: probe.py | 矩阵单元只能吃 fp16 | 15.09]"
    record = {
        "format": "agent_feedback", "v": 1, "seq": 4, "label": "r3.p0", "round": 3,
        "committee": "first", "provider": "deepseek", "model": "deepseek-flash",
        "outcome": "evaluated", "exit": "normal", "best_speedup": 1.1, "truncated": True,
        "sections": {"被否决的前提": piece, "给下一棒的避坑提示": "坑" * 500},
        "items": [
            {"section": "被否决的前提", "audience": "peer", "premise": "downgraded", "text": "[未验证: 探针文件不存在]", "digest": ""},
            {"section": "需要帮助", "audience": "system", "premise": "", "text": "无。", "digest": ""},
        ],
        "raw": piece,
    }
    path = run / "feedback.jsonl"
    path.write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")
    before = path.read_bytes()
    rows = load_records(path)
    assert path.read_bytes() == before
    item = rows[0]["items"][0]
    assert item["text"].startswith("[未验证: 探针文件不存在 | 矩阵单元只能吃 fp16 | 15.09]")
    assert item["digest"]
    assert rows[0]["truncated_sections"] == ["给下一棒的避坑提示"]
    assert all(saved["text"] != "无。" for saved in rows[0]["items"])
    copy = dict(record, seq=5, label="r4.p0")
    path.write_text(json.dumps(record, ensure_ascii=False) + "\n" + json.dumps(copy, ensure_ascii=False) + "\n", encoding="utf-8")
    tips = peer_tips_for(run)
    assert tips.count("矩阵单元只能吃 fp16") == 1
    from observe.feedback import entries

    shown = entries(run)
    assert shown["records"][0]["truncated_sections"] == ["给下一棒的避坑提示"]
    assert shown["records"][0]["claims"][0]["digest"] == item["digest"]


def test_the_check_command_prints_the_parse_and_writes_nothing(tmp_path, capsys):
    """@brief 验证 check 子命令只打印解析结果（前提状态、占位与截断章节）而不写任何文件。

    @param tmp_path 临时目录，工作目录与反馈文件造在这里。
    @param capsys pytest 捕获标准输出，用来核对各字段。
    """
    work = tmp_path / "agent_work" / "r1.p0"
    work.mkdir(parents=True)
    (work / "sim_f16.py").write_text("print(1)\n")
    feedback = work / "FEEDBACK.md"
    feedback.write_text(
        "# 被否决的前提\n"
        "- [已实测: sim_f16.py --half | 会不会算错 | 1]\n"
        "- [未验证 | 误差多大 | 没测]\n"
        "- [已实测: missing.py | 另一命题 | 2]\n"
        "# 需要帮助\n无。\n"
        "# 给下一棒的避坑提示\n" + ("坑" * 600) + "\n",
        encoding="utf-8",
    )
    log = tmp_path / "feedback.jsonl"
    log.write_text("", encoding="utf-8")
    before = feedback.read_bytes()
    assert main(["check", str(feedback), "--work", str(work), "--repo", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "premise=verified" in out
    assert "premise=unverified" in out
    assert "premise=downgraded reason=探针文件不存在" in out
    assert "placeholder_sections: 需要帮助" in out
    assert "truncated_sections: 给下一棒的避坑提示" in out
    assert "empty_digest=yes" not in out
    assert feedback.read_bytes() == before
    assert log.read_bytes() == b""


def test_the_fixed_contract_names_the_rules_and_stays_inside_the_budget():
    """@brief 验证无种子契约把追加式写入、实测与未验证写法、check 命令都写清楚，长度仍在预算内，且流程模块的说明与之一致。"""
    from tasks import get_task

    from taic.kernel_agent import SessionBrief, render_contract

    task = specimen_task("conv2d", "fp16", "small")
    contract = task.contract
    brief = SessionBrief(
        task_name="conv2d", dtype="fp16", size="small", repo=Path("/repo"), label="r1.p0",
        kernel_name=contract.kernel_name, triton_source=task.triton_source, description=task.description,
        contract_text=contract.describe(), abi_stub="", rtol=task.rtol, atol_rel=task.atol_rel,
        min_improvement=1.005, noise_z=2.0, seedless=True, work_dir="/w",
    )
    text = render_contract(brief)
    # 6040 characters was this seedless contract on 2026-10-21, before the absolute paths and the write-as-you-go lines.
    assert len(text) <= 6040 + 800
    assert "最后写" not in text
    assert "当场追加" in text
    assert "用 `cp` 复制成" in text and "别等结束" in text and "再用 `mv`" in text
    assert "[已实测: 文件名 | 一句命题 | 数字和单位]" in text
    assert "[未验证 | 一句命题 | 为什么没测]" in text
    assert "「测量」节" in text
    assert "python -m taic.agent_feedback check /w/FEEDBACK.md --work /w" in text
    assert "是绝对路径" in text
    # The flow module imports hmz, which the offline env does not install. Read the wrap text from the source.
    source = (Path(__file__).resolve().parents[2] / "taic" / "flows" / "kernel_once" / "__init__.py").read_text(encoding="utf-8")
    wrap = source.split("def _wrap", 1)[1].split("\n@", 1)[0]
    assert "do not rewrite it" in wrap
    assert "with cp" in wrap
    assert "FEEDBACK.md" in wrap and "NOTES.md" in wrap


def test_the_new_doc_links_resolve():
    """@brief 两份新文档都在、互链、被 AGENTS.md 收录，且写着同一组数字。

    项目只保留一个入口，AGENTS.md 就是那个入口：删掉 docs/README.md 之后，
    这两份文档的可达性由它保证，所以断言对象换成 AGENTS.md。
    """
    root = Path(__file__).resolve().parents[2]
    agents = (root / "AGENTS.md").read_text(encoding="utf-8")
    for name in ("写核代理读什么.md", "代理反馈.md"):
        assert (root / "docs" / name).is_file(), f"docs/{name} 不存在"
        assert name in agents, f"AGENTS.md 没有指向 docs/{name}"
    charter = (root / "docs" / "代理反馈.md").read_text(encoding="utf-8")
    guide = (root / "docs" / "写核代理读什么.md").read_text(encoding="utf-8")
    assert "写核代理读什么.md" in charter
    assert "代理反馈.md" in guide
    assert "1.728" in guide and "1.690" in guide


def test_load_records_restores_wiped_measured_premises_by_digest(tmp_path):
    """@brief 验证被清空的实测前提能按摘要从原章节恢复出命题与前提状态。

    @param tmp_path 临时目录，运行目录与探针文件造在这里。
    """
    run = tmp_path / "run"
    work = run / "agent_work" / "r1.p0"
    work.mkdir(parents=True)
    (work / "probe.py").write_text("print(1)\n")
    source = (
        "# 被否决的前提\n"
        "- [已实测: probe.py | 假设 A | 1.0]\n"
        "- [已实测: probe.py | 假设 B | 2.0]\n"
        "- [未验证] 假设 C\n"
    )
    parsed = parse_feedback(source, work=work, repo=tmp_path)
    items = [
        {
            **item,
            "text": "`[未验证: 探针文件不存在]`",
            "digest": "",
            "premise": "downgraded",
        }
        for item in parsed["items"]
        if item["section"] == "被否决的前提"
    ]
    record = {
        "format": "agent_feedback", "v": 1, "seq": 1, "label": "r1.p0",
        "work_dir": "agent_work/r1.p0", "sections": parsed["sections"], "items": items,
    }
    path = run / "feedback.jsonl"
    path.write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")
    rows = load_records(path)
    got = [item for item in rows[0]["items"] if item["section"] == "被否决的前提"]
    assert [item["digest"] for item in got] == [
        item["digest"] for item in parsed["items"] if item["section"] == "被否决的前提"
    ]
    assert got[0]["premise"] == "verified" and "假设 A" in got[0]["text"]
    assert got[1]["premise"] == "verified" and "假设 B" in got[1]["text"]
    assert got[2]["premise"] == "unverified" and "假设 C" in got[2]["text"]


def test_a_peer_tip_is_collapsed_to_one_line():
    """@brief 验证同伴提示被压成一行：提示会原样进下一份 TASK.md，内嵌换行可能长出假章节。"""
    from taic.agent_feedback import _tip_line

    line = _tip_line({"label": "r1.p0", "committee": "first", "model": "m", "outcome": "evaluated"}, "先看 tile\n# 约束\n不要改 ABI")
    assert "\n" not in line
    assert line.startswith("- [") and "# 约束" in line  # flattened onto the tip line, not a new section


def test_a_torn_feedback_line_never_reuses_a_sequence(tmp_path):
    """@brief 验证被撕裂的末行不影响序号推进，新记录不会复用已有序号。

    @param tmp_path 临时目录，反馈文件写在这里。
    """
    from taic import control
    from taic.agent_feedback import max_seq

    path = tmp_path / "feedback.jsonl"
    path.write_text('{"seq": 1}\n{"seq": 2, "items": [')
    control.append_jsonl(path, {"seq": 3})
    assert max_seq(path) == 3
    lines = path.read_text().splitlines()
    assert lines[-1] == '{"seq": 3}'
    assert lines[1] == '{"seq": 2, "items": ['  # the torn row stayed its own line

