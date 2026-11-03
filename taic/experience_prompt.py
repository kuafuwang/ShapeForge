"""Filter the target experience file before it reaches a prompt surface.

The experience file is also a prompt: ``stage_guidance`` copies it into the session
workdir and ``SkillIndex`` lists its sections in the retrieval outline.  Its header
promises that ``reported`` entries stay out of the prompt; this module is the code that
keeps that promise.  Only ``Confidence: verified`` sections are prompt-eligible, and a
raw ``- [harness <key>] …`` promotion bullet is not a verified entry.

Nothing here edits the source file, writes the experience store, or calls a model.
"""

from __future__ import annotations

import re

_CONFIDENCE = re.compile(r"^Confidence:\s*([A-Za-z_]+)\s*$", re.MULTILINE)
_HARNESS_BULLET = re.compile(r"^[ \t]*-[ \t]*\[harness\s+[^\]]+\].*$", re.MULTILINE)


def _drop_harness_bullets(text: str) -> str:
    """@brief 去掉 harness 追加的裸 bullet 行。

    这种行只是人工 promotion 的原始记录，没有 ``Confidence``，不是一条已验证经验。
    要进入提示面，必须先由人整理成带 ``Confidence: verified`` 的小节。
    """
    return _HARNESS_BULLET.sub("", text)


def filter_experience_for_prompt(raw: str) -> str:
    """@brief 只保留可进入提示面的经验内容。

    前言（第一个 ``## `` 之前的说明与证据等级约定）保留；小节只保留
    ``Confidence: verified`` 的，``reported`` / ``stale`` / ``retired`` / 未标注的一律排除。
    保留下来的小节里，裸 ``[harness …]`` bullet 也会被删掉。

    @param raw ``experience.md`` 全文。
    @return 过滤后的 Markdown；没有任何 verified 小节时只剩前言，不编造经验。
    """
    parts = ("\n" + raw.lstrip("\n")).split("\n## ")
    preamble = _drop_harness_bullets(parts[0]).strip()
    kept: list[str] = []
    for chunk in parts[1:]:
        title, _, body = chunk.partition("\n")
        match = _CONFIDENCE.search(body)
        if match is None or match.group(1) != "verified":
            continue
        body = _drop_harness_bullets(body).strip()
        kept.append(f"## {title.strip()}\n{body}")
    if not kept:
        return preamble + "\n" if preamble else ""
    head = preamble + "\n\n" if preamble else ""
    return head + "\n\n".join(kept) + "\n"


__all__ = ["filter_experience_for_prompt"]
