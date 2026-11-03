"""Only verified experience reaches the session workdir and the retrieval outline.

The source file keeps every entry for provenance; the prompt surface is filtered.
No GPU, no model calls, no writes outside pytest's tmp_path.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from taic.experience_prompt import filter_experience_for_prompt
from taic.kernel_agent import stage_guidance
from taic.tools import SkillIndex

EXPERIENCE = Path(__file__).resolve().parents[2] / "targets" / "apple-m4-max" / "experience.md"

SAMPLE = """# 经验条目

证据等级写在每条的 `Confidence:` 上；`reported` 不进提示词，只留在库里。

## 已验证条目

Keywords: alpha beta
Confidence: verified
观察: 这条保留。
- [harness b4131be13c4750c8] Add a tcenv.cli ab subcommand that compiles several kernel files.

## 仅上报条目

Keywords: gamma
Confidence: reported
观察: 这条不进提示词。

## 过期条目

Confidence: stale
观察: 这条也不进。

## 退役条目

Confidence: retired

## 未标注条目

观察: 没有 Confidence，按未验证处理。
"""


def test_only_verified_sections_survive_and_harness_bullets_do_not():
    """@brief verified 小节保留，reported/stale/retired/未标注与裸 harness bullet 全部排除。"""
    got = filter_experience_for_prompt(SAMPLE)
    assert "证据等级" in got  # 前言保留证据等级约定
    assert "## 已验证条目" in got and "这条保留" in got
    for dropped in ("仅上报条目", "过期条目", "退役条目", "未标注条目", "这条不进提示词", "没有 Confidence"):
        assert dropped not in got
    assert "tcenv.cli ab" not in got and "[harness" not in got


def test_no_verified_entry_means_no_invented_experience():
    """@brief 没有 verified 小节时只剩前言，不编造经验；空输入返回空文本。"""
    got = filter_experience_for_prompt("# 经验\n\n只有说明。\n\n## 一条\nConfidence: reported\n正文。\n")
    assert "只有说明" in got and "## 一条" not in got and "正文" not in got
    assert filter_experience_for_prompt("") == ""


def test_the_current_target_file_no_longer_ships_the_developer_tool_request():
    """@brief 当前经验库过滤后仍含 9 条 verified，但 b413 的工具请求不进入提示面。"""
    got = filter_experience_for_prompt(EXPERIENCE.read_text(encoding="utf-8"))
    assert got.count("Confidence: verified") == 9
    assert "累加器片段到 32 会溢出" in got
    assert "tcenv.cli ab" not in got and "b4131be13c4750c8" not in got
    # 源文件本身不被这个函数修改：b413 仍留在库里，等人审随一次性重签删除。
    assert "b4131be13c4750c8" in EXPERIENCE.read_text(encoding="utf-8")


def test_stage_guidance_filters_experience_but_copies_device_and_reference(tmp_path):
    """@brief staging 用过滤后的经验，device/reference 原样拷贝，快照里的原文不变。"""
    snap = tmp_path / "agent_guidance"
    snap.mkdir()
    (snap / "device.md").write_text("device raw\n", encoding="utf-8")
    (snap / "msl_reference.md").write_text("reference raw\n", encoding="utf-8")
    (snap / "experience.md").write_text(SAMPLE, encoding="utf-8")
    work = tmp_path / "work"
    brief = SimpleNamespace(target_id="apple-m4-max", reference_skill="msl_reference", guidance_dir=str(snap))

    stage_guidance(work, brief)

    assert (work / "device.md").read_text(encoding="utf-8") == "device raw\n"
    assert (work / "msl_reference.md").read_text(encoding="utf-8") == "reference raw\n"
    staged = (work / "experience.md").read_text(encoding="utf-8")
    assert "已验证条目" in staged and "仅上报条目" not in staged and "tcenv.cli ab" not in staged
    # 运行级快照仍保存原始文件，供溯源与以后重放。
    assert "仅上报条目" in (snap / "experience.md").read_text(encoding="utf-8")


def test_retrieval_outline_and_lookup_cannot_reach_unverified_entries(monkeypatch):
    """@brief outline 与 lookup 都只索引 verified 经验，reported 标题与正文都查不到。"""
    docs = {
        "msl_reference": "# MSL\n\n## simdgroup 内存\nKeywords: threadgroup\n正文。\n",
        "experience": SAMPLE,
    }
    monkeypatch.setattr("taic.tools.skill_text", lambda name, target_id=None: docs[name])
    index = SkillIndex("msl_reference", extra=("experience",), target_id="apple-m4-max")

    outline = index.outline()
    assert "已验证条目" in outline and "仅上报条目" not in outline and "退役条目" not in outline
    for section in index.sections:
        if section.doc == "experience":
            assert "Confidence: verified" in section.body
    assert "tcenv.cli ab" not in index.lookup("alpha", k=2)
    assert "这条不进提示词" not in index.lookup("gamma", k=2)
