"""Function-calling tools: documentation retrieval and a no-benchmark quick check (two of the paper's three skills)."""

# 模型在一个回合里只能用这两个工具：lookup_msl 查 MSL 资料，quick_check 做编译、ABI、安全与正确性检查。
# 完整评测（计时、换种子、噪声门槛、结束时的复测）留在编排器的 _eval，不交给模型。否则它会在测量噪声里自己挑最快的一版，也会在一次对话里把 GPU 占满。阅读时先看两份工具模式，再看 make_handlers 怎样按「这一次请求」封顶 quick_check。

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Awaitable, Callable

from tcenv.target import DEFAULT_TARGET_ID

from .target import skill_text

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


# 技能文档按二级标题切段，供 lookup_msl 检索。文档读不出来时，组系统提示和查资料都会失败。
def _sections(raw: str, doc_name: str) -> list["Section"]:
    """Split one skill document on ``## `` headings, keeping a ``Keywords:`` line for retrieval."""
    out: list[Section] = []
    for chunk in raw.split("\n## ")[1:]:
        title, _, body = chunk.partition("\n")
        kw = re.search(r"^Keywords:\s*(.*)$", body, re.MULTILINE)
        out.append(Section(doc_name, title.strip(), kw.group(1) if kw else "", body.strip()))
    return out


class SkillIndex:
    """Keyword retrieval over the sections of a skill document (the 'PTX ISA retrieval' analogue)."""

    def __init__(self, doc_name: str = "msl_reference", extra: tuple[str, ...] = (), target_id: str | None = None) -> None:
        """``extra`` are further documents (``experience`` lives on the target; other names are backend skills)."""
        self.doc = doc_name
        self.target_id = target_id or DEFAULT_TARGET_ID
        raw = skill_text(doc_name, self.target_id)
        self.preamble = raw.split("\n## ", 1)[0]
        self.sections: list[Section] = _sections(raw, doc_name)
        for name in extra:
            self.sections.extend(_sections(skill_text(name, self.target_id), name))

    # 开工具时系统提示里只有这份目录，逼模型按主题来查，而不是凭记忆编 API。
    def outline(self) -> str:
        return "\n".join(f"- {s.title}  (keywords: {s.keywords})" for s in self.sections)

    # 关工具时没有 lookup_msl，参考必须事先全部可见。
    def full_text(self) -> str:
        return self.preamble + "\n\n" + "\n\n".join(s.text for s in self.sections)

    # 用词重叠挑最相关的几段，挡住模型编造地址空间、屏障和 simdgroup_matrix。问空或没有命中时只给目录，不编一段假文档。
    def lookup(self, topic: str, k: int = 2, max_chars: int = 7000) -> str:
        q = set(_WORD.findall(topic.lower()))
        if not q:
            return "error: empty topic. Section titles:\n" + self.outline()
        scored = []
        for s in self.sections:
            title_w = set(_WORD.findall(s.title.lower()))
            kw_w = set(_WORD.findall(s.keywords.lower()))
            body_w = set(_WORD.findall(s.body.lower()))
            # 标题权重大于关键词，关键词大于正文。标识符被切词拆开后，仍可能靠下面的整段子串加分。
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


# 查资料。防的是模型把 MSL 属性、地址空间、矩阵指令写错；不碰 GPU，也不产生候选。
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

# 快速检查。只做到编译、ABI、安全、正确性，故意不含基准：是否快过测量噪声由搜索的 _eval 判定。次数在 make_handlers 里按这一次模型请求封顶。
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


# 绑到一次对话。quick_eval 为空就不注册 quick_check。次数用尽或源码为空只回错误文本，模型仍可交出最终内核，整次搜索不因此停下。GPU 串行由 quick_eval 自己保证：编排器的 _quick 与完整评测共用同一把 gpu_lock。
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
        # 计数只在这一次请求内。续跑时编排器会把已经用掉的次数从上限里扣掉再传进来。用尽后必须直接交内核，完整评测仍只发生在 _eval。
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
