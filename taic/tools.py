"""Function-calling tools: documentation retrieval and a no-benchmark quick check (two of the paper's three skills)."""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Awaitable, Callable

SKILLS_DIR = Path(__file__).resolve().parent / "skills"
_WORD = re.compile(r"[a-z0-9_]+")


@dataclass
class Section:
    doc: str
    title: str
    keywords: str
    body: str

    @property
    def text(self) -> str:
        return f"## {self.title}\n{self.body}".strip()


class SkillIndex:
    """Keyword retrieval over the sections of a skill document (the 'PTX ISA retrieval' analogue)."""

    def __init__(self, doc_name: str = "msl_reference") -> None:
        self.doc = doc_name
        raw = (SKILLS_DIR / f"{doc_name}.md").read_text()
        self.preamble = raw.split("\n## ", 1)[0]
        self.sections: list[Section] = []
        for chunk in raw.split("\n## ")[1:]:
            title, _, body = chunk.partition("\n")
            kw = re.search(r"^Keywords:\s*(.*)$", body, re.MULTILINE)
            self.sections.append(Section(doc_name, title.strip(), kw.group(1) if kw else "", body.strip()))

    def outline(self) -> str:
        return "\n".join(f"- {s.title}  (keywords: {s.keywords})" for s in self.sections)

    def full_text(self) -> str:
        return self.preamble + "\n\n" + "\n\n".join(s.text for s in self.sections)

    def lookup(self, topic: str, k: int = 2, max_chars: int = 7000) -> str:
        q = set(_WORD.findall(topic.lower()))
        if not q:
            return "error: empty topic. Section titles:\n" + self.outline()
        scored = []
        for s in self.sections:
            title_w = set(_WORD.findall(s.title.lower()))
            kw_w = set(_WORD.findall(s.keywords.lower()))
            body_w = set(_WORD.findall(s.body.lower()))
            score = 4 * len(q & title_w) + 3 * len(q & kw_w) + len(q & body_w)
            # substring hit on the raw topic (e.g. 'simd_sum') is a strong signal
            if topic.lower().strip() in (s.title + " " + s.keywords + " " + s.body).lower():
                score += 6
            scored.append((score, s))
        scored.sort(key=lambda t: -t[0])
        hits = [s for sc, s in scored[:k] if sc > 0]
        if not hits:
            return f"no section matches '{topic}'. Section titles:\n" + self.outline()
        out = "\n\n".join(s.text for s in hits)
        return out[:max_chars]


LOOKUP_TOOL = {
    "type": "function",
    "function": {
        "name": "lookup_msl",
        "description": (
            "Look up the Metal Shading Language reference (attributes, address spaces, types, math precision, SIMD-group ops, "
            "barriers, simdgroup_matrix, atomics, compiler-error cookbook). Returns the best matching sections."
        ),
        "parameters": {"type": "object", "properties": {"topic": {"type": "string", "description": "e.g. 'simdgroup_matrix thread_elements' or 'simd_shuffle_xor'"}}, "required": ["topic"]},
    },
}

QUICK_CHECK_TOOL = {
    "type": "function",
    "function": {
        "name": "quick_check",
        "description": (
            "Compile and test a candidate kernel without benchmarking: Metal compiler errors, ABI check, safety checks and the "
            "randomized correctness tests. Returns the same feedback you would get from the real evaluation (minus performance). "
            "Use it to debug before answering; the number of calls is limited."
        ),
        "parameters": {
            "type": "object",
            "properties": {"kernel_source": {"type": "string", "description": "complete Metal source (helpers + kernel), no markdown fences"}},
            "required": ["kernel_source"],
        },
    },
}


def make_handlers(
    index: SkillIndex,
    quick_eval: Callable[[str], Awaitable[str]] | None,
    max_quick_checks: int,
) -> dict:
    """Handlers for one conversation. ``quick_eval`` is expected to serialize GPU access itself."""
    state = {"quick": 0}

    def lookup(args: dict) -> str:
        return index.lookup(str(args.get("topic", "")))

    async def quick(args: dict) -> str:
        if quick_eval is None:
            return "error: quick_check is not available"
        if state["quick"] >= max_quick_checks:
            return f"error: quick_check limit ({max_quick_checks}) reached for this request; answer with your best kernel"
        state["quick"] += 1
        src = str(args.get("kernel_source", ""))
        if not src.strip():
            return "error: empty kernel_source"
        return await quick_eval(src)

    h = {"lookup_msl": lookup}
    if quick_eval is not None:
        h["quick_check"] = quick
    return h
