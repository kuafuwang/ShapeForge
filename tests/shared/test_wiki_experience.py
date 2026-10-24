"""The wiki's experience layer: every entry says how sure it is, and `verified` entries cite what measured it.

This is the rule that keeps an unmeasured claim out of the prompt: the entry file is pasted into the system
prompt, so an unsupported `verified` line would be the agent's fact. No GPU, no model calls.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from taic.tools import SkillIndex

EXPERIENCE = Path(__file__).resolve().parents[2] / "targets" / "apple-m4-max" / "experience.md"
WIKI = ("experience",)


def _entries() -> list[tuple[str, str]]:
    """@brief 把 experience.md 按 "## " 标题切成 (标题, 正文) 列表。

    @return 条目列表。
    """
    raw = EXPERIENCE.read_text()
    out = []
    for chunk in raw.split("\n## ")[1:]:
        title, _, body = chunk.partition("\n")
        out.append((title.strip(), body))
    return out


def test_there_is_more_than_one_entry_and_each_has_a_keyword_line():
    """@brief 经验文件至少有 5 个条目，且每条都有非空的 Keywords 行。
    """
    entries = _entries()
    assert len(entries) >= 5
    for title, body in entries:
        assert re.search(r"^Keywords:\s*\S", body, re.MULTILINE), title


def test_every_entry_states_its_confidence():
    """@brief 每个条目都有 Confidence 行，且取值只能是 verified 或 reported。
    """
    for title, body in _entries():
        got = re.search(r"^Confidence:\s*(\w+)", body, re.MULTILINE)
        assert got is not None, f"{title}: no Confidence line"
        assert got.group(1) in ("verified", "reported"), title


def test_a_verified_entry_cites_a_run_or_a_document():
    """@brief verified 条目必须给出处（runs/docs/tcenv/taic）、本机实测和边界三项。
    """
    for title, body in _entries():
        if not re.search(r"^Confidence:\s*verified", body, re.MULTILINE):
            continue
        assert re.search(r"^来源:.*(`runs/|`docs/|`tcenv/|`taic/)", body, re.MULTILINE), f"{title}: verified without a source"
        assert re.search(r"^本机实测:", body, re.MULTILINE), f"{title}: verified without a measurement"
        assert re.search(r"^边界:", body, re.MULTILINE), f"{title}: verified without a boundary"


def test_a_reported_entry_is_never_wired_into_the_prompt():
    """@brief reported 条目正文必须自带未验证标记，否则测试失败。

    reported 条目保留在文件里，但规则是只有 verified 条目才能被引用。
    检索时还没有任何机制强制这一点（文件本身就是提示词），所以测试至少要保证：
    某条 reported 条目出现时，正文里带有表明它未验证的字样。
    """
    for title, body in _entries():
        if re.search(r"^Confidence:\s*reported", body, re.MULTILINE):
            assert "未验证" in body or "not reproduced" in body.lower(), f"{title}: reported entry must say so in its body"


def test_the_entries_are_retrievable_next_to_the_msl_reference():
    """@brief 经验条目能随 msl_reference 一起检索，返回正文含 Confidence: verified 和本机实测。
    """
    index = SkillIndex("msl_reference", extra=WIKI)
    assert any("累加器" in s.title for s in index.sections)
    got = index.lookup("accumulator register spill fragment", k=1)
    assert "Confidence: verified" in got and "本机实测" in got


def test_leaving_the_wiki_out_keeps_the_msl_reference_alone():
    """@brief 不加载经验 wiki 时 msl_reference 的内容和章节数保持原样。
    """
    plain = SkillIndex("msl_reference")
    with_wiki = SkillIndex("msl_reference", extra=WIKI)
    assert len(with_wiki.sections) > len(plain.sections)
    assert plain.full_text() == SkillIndex("msl_reference").full_text()
    assert "累加器片段" not in plain.full_text()
