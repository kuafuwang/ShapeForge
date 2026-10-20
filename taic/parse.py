"""Extract the kernel source or the optimization plan from a model reply."""

# 模型回复是自由文本。本文件做两件抽取：extract_metal 找要评测的 Metal 源码，parse_plans 找一轮里的多份方案。
# 抽失败的判定很窄：没有带约定内核名的 Metal 块，结果是 None；规划 JSON 缺失，或每条方案的 title 与 change 都空，方案列表就是空的。编排器据此追问或改用 FALLBACK_PLANS，不会把残缺文本送去编译。

from __future__ import annotations

import json
import re

# 围栏可以跨行，语言标签允许为空。json 标签不会被当成内核源码。
_FENCE = re.compile(r"```[ \t]*([A-Za-z0-9_+\-]*)[ \t]*\n(.*?)```", re.DOTALL)
# 这些标签都可能包着 MSL。空标签也收下：模型经常忘了写 metal。
_METAL_TAGS = {"metal", "msl", "cpp", "c++", "c", "cuda", "objc", ""}


# 认定回复里要评测的那份源码。多块代码时，最后一块「标签像 MSL，且声明了 kernel void <kernel_name>」才算，前面的示例片段忽略。没有这样的块就是抽失败，编排器只追问，不把其它文本送去编译。
def extract_metal(text: str, kernel_name: str) -> str | None:
    """Return the Metal source in ``text``.

    The request asks for exactly one ```metal block. Models sometimes add illustrative snippets first, so when
    several blocks are present the *last* one that declares ``kernel void <kernel_name>`` wins.
    """
    blocks = [(tag.lower(), body) for tag, body in _FENCE.findall(text or "")]
    # 说明文字和 ```json 规划块不在候选里。
    code = [b for tag, b in blocks if tag in _METAL_TAGS]
    # 带启动说明的块优先。说明不合法时也把这块送去评测，由评测拒绝，不退回旁边那颗契约核。
    marked = [b for b in code if "taic-launch" in b]
    if marked:
        return marked[-1].strip("\n") + "\n"
    # 必须出现契约里的内核名。同名的多块取最后一块：模型常先贴片段，再贴完整内核。
    named = [b for b in code if re.search(r"\bkernel\s+void\s+" + re.escape(kernel_name) + r"\b", b)]
    if named:
        return named[-1].strip("\n") + "\n"
    # 有围栏但没有约定的 kernel void 名字，与完全没有围栏相同，都是抽失败。
    return None


# 规划回复里的第一个 JSON 对象，围栏（json 或空标签）或裸文本都行。没有任何 dict 就是抽失败，方案列表会是空的。
def extract_json(text: str) -> dict | None:
    """Parse the first JSON object in ``text`` (fenced or bare)."""
    candidates = [b for tag, b in _FENCE.findall(text or "") if tag.lower() in ("json", "")]
    candidates.append(text or "")
    for c in candidates:
        c = c.strip()
        start = c.find("{")
        while start != -1:
            # 花括号深度只在字符串之外增减，方案描述里的 { } 不能把对象截断；转义引号也不算字符串结束。
            depth, in_str, esc = 0, False, False
            for i in range(start, len(c)):
                ch = c[i]
                if in_str:
                    if esc:
                        esc = False
                    elif ch == "\\":
                        esc = True
                    elif ch == '"':
                        in_str = False
                elif ch == '"':
                    in_str = True
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        try:
                            obj = json.loads(c[start : i + 1])
                            if isinstance(obj, dict):
                                return obj
                        except json.JSONDecodeError:
                            # 括号配平了但不是合法 JSON，这次对象作废，继续找下一个 {。
                            pass
                        break
            start = c.find("{", start + 1)
    return None


# 规划文本变成（瓶颈分析, 最多 k 份方案）。没有 JSON、plans 缺失、或每条的 title 与 change 都空，方案列表为空，编排器视为抽失败：先重试，两次都不行就用 FALLBACK_PLANS。
def parse_plans(text: str, k: int) -> tuple[str, list[dict]]:
    """Return (analysis, up to ``k`` plans) from the planner's JSON reply; plans are dicts with ``title`` and ``change``."""
    obj = extract_json(text)
    if not obj:
        # 连对象都没有：分析和方案都空。
        return "", []
    plans = []
    for p in obj.get("plans", []) or []:
        if isinstance(p, str):
            # 模型把方案写成一句话而不是对象时，整句同时充当标题和改动说明，不算抽失败。
            p = {"title": p[:60], "change": p}
        # 只有标题或只有改动说明仍可用。两个都空的条目丢掉；一条都不剩就是抽失败。
        if isinstance(p, dict) and (p.get("change") or p.get("title")):
            plans.append(
                {
                    "title": str(p.get("title") or p.get("change"))[:120],
                    "change": str(p.get("change") or p.get("title")),
                    "expected_gain": str(p.get("expected_gain", "")),
                }
            )
    analysis = " ".join(str(obj.get(k_, "")) for k_ in ("bottleneck", "analysis") if obj.get(k_)).strip()
    # 截到本轮分支数 k。空列表表示这次规划没有可用方案。
    return analysis, plans[:k]
