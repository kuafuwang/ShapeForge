"""Statistical helpers for historical contract-menu experiment reports.

Current v2 candidates own their geometry; this module no longer assembles task variants or launches
a search.  ``observe`` keeps using ``judge`` to read existing experiment records.
"""

from __future__ import annotations

import statistics

def resolve_menu(base: str, tokens: list[str]) -> list[tuple[str, str]]:
    """@brief 把命令行里的契约列表解成菜单项。

    ``(directory slug, task name)`` in menu order. ``default`` is the bare task name. A missing default is an error.

    @param base 基任务名，不带 ``@变体``。
    @param tokens 契约名列表，可带或不带基任务前缀。
    @return 按菜单顺序的 ``(目录名, 任务名)`` 列表。
    @raises ValueError 菜单为空、契约重复或不是该任务的变体、或缺少 ``default`` 时。
    """
    if not tokens:
        raise ValueError("contract menu is empty")
    seen: list[str] = []
    out: list[tuple[str, str]] = []
    for raw in tokens:
        token = raw.strip()
        if not token:
            continue
        if token in ("default", base):
            slug, task = "default", base
        elif token.startswith(f"{base}@"):
            variant = token.split("@", 1)[1]
            slug, task = variant, token
        elif "@" in token:
            raise ValueError(f"contract {token!r} is not a variant of {base}")
        else:
            slug, task = token, f"{base}@{token}"
        if not slug or slug in seen:
            raise ValueError(f"duplicate or empty contract {token!r}")
        seen.append(slug)
        out.append((slug, task))
    if "default" not in seen:
        raise ValueError("contract menu must include default (the task's built-in launch); refusing to start without it")
    return out

def schedule(items: list[tuple[str, str]], trials: int) -> list[tuple[str, str, int]]:
    """@brief 把菜单交错排成执行顺序：先每套契约各一轮，再下一轮。

    Interleave trials: round 1 of every contract, then round 2, and so on.

    @param items ``resolve_menu`` 返回的菜单项。
    @param trials 每套契约重复几轮。
    @return ``(目录名, 任务名, 轮次)`` 序列。
    @raises ValueError 轮数小于 1 时。
    """
    if trials < 1:
        raise ValueError(f"trials must be >= 1, got {trials}")
    return [(slug, task, trial) for trial in range(1, trials + 1) for slug, task in items]

def _num(value) -> float | None:
    """@brief 取一个可用于统计的浮点数。

    @param value 任意值。
    @return 浮点数；布尔、非数值和 NaN 都返回 ``None``。
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if value == value else None

def judge(rows: list[dict], slugs: list[str]) -> dict:
    """@brief 判定哪套契约胜出，证据不足时返回无结论。

    Name a winner only when every contract has at least two published speedups and one range sits strictly above the others.

    ``best_speedup`` is not read. A contract with a single published number is an anecdote (``个例``) and blocks a winner.

    @param rows 已完成的试验行。
    @param slugs 菜单里的目录名列表。
    @return 含 ``verdict``/``winner``/``reason`` 与逐契约统计的字典。
    """
    by: dict[str, list[dict]] = {slug: [] for slug in slugs}
    for row in rows:
        if row.get("skipped") or row["slug"] not in by:
            continue
        by[row["slug"]].append(row)
    published: dict[str, list[float]] = {}
    for slug, items in by.items():
        published[slug] = [v for r in items if (v := _num(r.get("published_speedup"))) is not None]
    anecdotes = [slug for slug, nums in published.items() if len(nums) == 1]
    enough = bool(published) and len(published) >= 2 and all(len(nums) >= 2 for nums in published.values())
    leaders: list[str] = []
    if enough:
        leaders = [
            slug for slug, nums in published.items()
            if all(min(nums) > max(other) for other_slug, other in published.items() if other_slug != slug)
        ]
    winner = leaders[0] if len(leaders) == 1 else None
    if winner:
        verdict, reason = "胜出", f"{winner} 的发布数字范围高于其余契约"
    else:
        bits = []
        if len(published) < 2:
            bits.append("只有一套契约，没有对照")
        elif not all(len(nums) >= 2 for nums in published.values()):
            bits.append("有契约的发布数字少于 2 个")
        else:
            bits.append("发布数字的范围重叠")
        if anecdotes:
            bits.append("个例：" + ", ".join(anecdotes))
        verdict, reason = "无结论", "；".join(bits)
    per = {}
    for slug, nums in published.items():
        per[slug] = {
            "published": nums,
            "published_n": len(nums),
            "median": statistics.median(nums) if nums else None,
            "min": min(nums) if nums else None,
            "max": max(nums) if nums else None,
        }
    return {"verdict": verdict, "winner": winner, "reason": reason, "contracts": per}
