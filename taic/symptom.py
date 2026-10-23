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
    """@brief 判断某个二级标题是否属于以 ``prefix`` 开头的那一节。

    标题本身就是前缀，或前缀后跟空格/左括号，都算同一节。

    @param title 二级标题文本。
    @param prefix 期望的标题前缀。
    @return 属于该节时为 ``True``。
    """
    return title == prefix or title.startswith(prefix + " ") or title.startswith(prefix + "(")


class SymptomIndex:
    """@brief memory / compute 到已有技能章节的固定对照，问法变了结果也不变。"""

    def __init__(self) -> None:
        """@brief 读技能文档，为每类 roofline 绑定唯一一节原文。

        @raises RuntimeError 某一类对应不到恰好一节时，说明技能文档被改过。
        """
        sections = SkillIndex("apple_gpu_optimization").sections
        found: dict[str, str] = {}
        for bound, prefix in _TITLE.items():
            hits = [s for s in sections if _is_chapter(s.title, prefix)]
            if len(hits) != 1:
                raise RuntimeError(f"expected one section for {prefix!r}, found {[s.title for s in sections]}")
            found[bound] = hits[0].text
        self.by_bound = found

    def section(self, bound: str | None) -> str | None:
        """@brief 取某一类 roofline 对应的章节原文。

        @param bound ``memory`` 或 ``compute``；其它值不另编一类。
        @return 章节原文（含二级标题）；不认识的类别返回 ``None``。
        """
        if bound not in self.by_bound:
            return None
        return self.by_bound[bound]


@lru_cache(maxsize=1)
def _index() -> SymptomIndex:
    """@brief 进程内只建一次的对照表。

    @return 缓存的 ``SymptomIndex``。
    """
    return SymptomIndex()


def section_for(bound: str | None) -> str | None:
    """@brief 取对应章节的原文（含二级标题）。

    ``bound`` 不是 memory 或 compute 时返回 ``None``，不另编一类建议。

    @param bound roofline 类别。
    @return 章节原文；不认识的类别返回 ``None``。
    """
    return _index().section(bound)


def bound_of(report: Any) -> str | None:
    """@brief 读报告里的 ``roofline['bound']``（由 tcenv.backends.metal.profile.roofline 写入）。

    @param report 评测报告；可以是字典，也可以是带 ``roofline`` 属性的对象。
    @return ``memory`` 或 ``compute``；字段缺失或取值不认识时返回 ``None``。
    """
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
