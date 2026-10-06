"""Extract the kernel source or the optimization plan from a model reply."""

from __future__ import annotations

import json
import re

_FENCE = re.compile(r"```[ \t]*([A-Za-z0-9_+\-]*)[ \t]*\n(.*?)```", re.DOTALL)
_METAL_TAGS = {"metal", "msl", "cpp", "c++", "c", "cuda", "objc", ""}


def extract_metal(text: str, kernel_name: str) -> str | None:
    """Return the Metal source in ``text``.

    The request asks for exactly one ```metal block. Models sometimes add illustrative snippets first, so when
    several blocks are present the *last* one that declares ``kernel void <kernel_name>`` wins.
    """
    blocks = [(tag.lower(), body) for tag, body in _FENCE.findall(text or "")]
    code = [b for tag, b in blocks if tag in _METAL_TAGS]
    named = [b for b in code if re.search(r"\bkernel\s+void\s+" + re.escape(kernel_name) + r"\b", b)]
    if named:
        return named[-1].strip("\n") + "\n"
    return None


def extract_json(text: str) -> dict | None:
    """Parse the first JSON object in ``text`` (fenced or bare)."""
    candidates = [b for tag, b in _FENCE.findall(text or "") if tag.lower() in ("json", "")]
    candidates.append(text or "")
    for c in candidates:
        c = c.strip()
        start = c.find("{")
        while start != -1:
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
                            pass
                        break
            start = c.find("{", start + 1)
    return None


def parse_plans(text: str, k: int) -> tuple[str, list[dict]]:
    """Return (analysis, up to ``k`` plans) from the planner's JSON reply; plans are dicts with ``title`` and ``change``."""
    obj = extract_json(text)
    if not obj:
        return "", []
    plans = []
    for p in obj.get("plans", []) or []:
        if isinstance(p, str):
            p = {"title": p[:60], "change": p}
        if isinstance(p, dict) and (p.get("change") or p.get("title")):
            plans.append(
                {
                    "title": str(p.get("title") or p.get("change"))[:120],
                    "change": str(p.get("change") or p.get("title")),
                    "expected_gain": str(p.get("expected_gain", "")),
                }
            )
    analysis = " ".join(str(obj.get(k_, "")) for k_ in ("bottleneck", "analysis") if obj.get(k_)).strip()
    return analysis, plans[:k]
