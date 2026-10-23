"""Map a roofline class onto one existing section of the Apple GPU skill."""

# 按二级标题从 apple_gpu_optimization.md 切一节，不另写优化建议，也不把正文抄进新文件。
# metal_profiling.md 里 Roofline 与诊断表同时解释 memory 和 compute，没有只讲其中一类的章节，所以不收。

from __future__ import annotations

from functools import lru_cache
from typing import Any

from .tools import SkillIndex

# 标题前缀。括号里的任务名可以改，前缀对上才算这一节。
_TITLE = {
    "memory": "Memory-bound kernels",
    "compute": "Compute-bound kernels",
}


def _is_chapter(title: str, prefix: str) -> bool:
    return title == prefix or title.startswith(prefix + " ") or title.startswith(prefix + "(")


class SymptomIndex:
    """memory / compute 到已有技能章节的固定对照。问法变了结果也不变。"""

    def __init__(self) -> None:
        sections = SkillIndex("apple_gpu_optimization").sections
        found: dict[str, str] = {}
        for bound, prefix in _TITLE.items():
            hits = [s for s in sections if _is_chapter(s.title, prefix)]
            if len(hits) != 1:
                raise RuntimeError(f"expected one section for {prefix!r}, found {[s.title for s in sections]}")
            found[bound] = hits[0].text
        self.by_bound = found

    def section(self, bound: str | None) -> str | None:
        if bound not in self.by_bound:
            return None
        return self.by_bound[bound]


@lru_cache(maxsize=1)
def _index() -> SymptomIndex:
    return SymptomIndex()


def section_for(bound: str | None) -> str | None:
    """对应章节的原文（含二级标题）。bound 不是 memory 或 compute 时返回 None，不另编一类。"""
    return _index().section(bound)


def bound_of(report: Any) -> str | None:
    """读 tcenv.backends.metal.profile.roofline 写进报告的 roofline['bound']。没有这个字段，或值不是 memory/compute，就返回 None。"""
    if report is None:
        return None
    if isinstance(report, dict):
        roof = report.get("roofline")
    else:
        roof = getattr(report, "roofline", None)
    if not isinstance(roof, dict):
        return None
    bound = roof.get("bound")
    if bound in _TITLE:
        return bound
    return None
