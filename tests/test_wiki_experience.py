"""The wiki's experience layer: every entry says how sure it is, and `verified` entries cite what measured it.

This is the rule that keeps an unmeasured claim out of the prompt: the entry file is pasted into the system
prompt, so an unsupported `verified` line would be the agent's fact. No GPU, no model calls.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from taic.tools import SkillIndex

EXPERIENCE = Path(__file__).resolve().parent.parent / "taic" / "skills" / "wiki" / "experience.md"
WIKI = ("wiki/experience",)


def _entries() -> list[tuple[str, str]]:
    raw = EXPERIENCE.read_text()
    out = []
    for chunk in raw.split("\n## ")[1:]:
        title, _, body = chunk.partition("\n")
        out.append((title.strip(), body))
    return out


def test_there_is_more_than_one_entry_and_each_has_a_keyword_line():
    entries = _entries()
    assert len(entries) >= 5
    for title, body in entries:
        assert re.search(r"^Keywords:\s*\S", body, re.MULTILINE), title


def test_every_entry_states_its_confidence():
    for title, body in _entries():
        got = re.search(r"^Confidence:\s*(\w+)", body, re.MULTILINE)
        assert got is not None, f"{title}: no Confidence line"
        assert got.group(1) in ("verified", "reported"), title


def test_a_verified_entry_cites_a_run_or_a_document():
    for title, body in _entries():
        if not re.search(r"^Confidence:\s*verified", body, re.MULTILINE):
            continue
        assert re.search(r"^来源:.*(`runs/|`docs/|`tcenv/|`taic/)", body, re.MULTILINE), f"{title}: verified without a source"
        assert re.search(r"^本机实测:", body, re.MULTILINE), f"{title}: verified without a measurement"
        assert re.search(r"^边界:", body, re.MULTILINE), f"{title}: verified without a boundary"


def test_a_reported_entry_is_never_wired_into_the_prompt():
    """`reported` entries stay in the file but the rule is that only verified ones may be quoted.

    Nothing enforces it at retrieval time yet (the file is the prompt), so the test at least fails loudly if a
    `reported` entry appears without the words that mark it as unverified in the text itself.
    """
    for title, body in _entries():
        if re.search(r"^Confidence:\s*reported", body, re.MULTILINE):
            assert "未验证" in body or "not reproduced" in body.lower(), f"{title}: reported entry must say so in its body"


def test_the_entries_are_retrievable_next_to_the_msl_reference():
    index = SkillIndex("msl_reference", extra=WIKI)
    assert any("累加器" in s.title for s in index.sections)
    got = index.lookup("accumulator register spill fragment", k=1)
    assert "Confidence: verified" in got and "本机实测" in got


def test_leaving_the_wiki_out_keeps_the_msl_reference_alone():
    plain = SkillIndex("msl_reference")
    with_wiki = SkillIndex("msl_reference", extra=WIKI)
    assert len(with_wiki.sections) > len(plain.sections)
    assert plain.full_text() == SkillIndex("msl_reference").full_text()
    assert "累加器片段" not in plain.full_text()
