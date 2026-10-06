"""Budget alerts of the dashboard: how a run's spend compares with its USD budget.

Pure functions, no I/O, so that they can be tested without a run directory. Three alert levels, worst first:

* ``over``       spent more than the budget. The budget gate is checked *before* each request, so requests that are in flight
                 when the cap is reached finish above it (paper Section 4.3 says the same); a run overshoots by a few requests.
* ``exhausted``  the run stopped with ``stop_reason == "budget_exhausted"``: the search was cut off by money, not by convergence,
                 even when the last request left the spend slightly *under* the cap.
* ``near``       a run that is still running has used at least ``NEAR_FRACTION`` of its budget.

A run can be both over and exhausted (``flags`` lists every level that applies, ``level`` is the worst one).
"""

# 看板顶部的预算告警。三档都在纯函数里判定：只看花费、预算、停止原因和是否仍在跑，不读 runs/。
# OVER BUDGET 是花费超过预算；BUDGET EXHAUSTED 是停止原因恰好为 budget_exhausted；NEAR BUDGET 只打给仍在跑且已用到 80% 的运行。

from __future__ import annotations

import html
import math
from typing import Any
from urllib.parse import quote

NEAR_FRACTION = 0.8
EXHAUSTED_STOP_REASON = "budget_exhausted"
_EPS = 1e-9  # float noise: a run that ends at exactly its budget is not "over"

LABELS = {"over": "OVER BUDGET", "exhausted": "BUDGET EXHAUSTED", "near": "NEAR BUDGET"}
LABELS_CN = {"over": "超预算", "exhausted": "预算耗尽", "near": "接近预算"}
SEVERITY = {"over": 0, "exhausted": 1, "near": 2, "ok": 3, "unknown": 4}


# 旧结果缺字段，或写成布尔、NaN 时不能拿去比大小。布尔是 int 的子类，不先排除的话 True 会变成 1 美元。
def _num(value: Any) -> float | None:
    """A finite number, or ``None`` for a missing, non-numeric, boolean or NaN/inf value (older runs lack fields)."""
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


# 三档可以同时成立，level 取 flags 里最重的一档。缺预算或缺花费不会抛错。
# 刚好花到预算不算超（要多出 1e-9 以上）。耗尽只认停止原因，哪怕花费仍略低于上限。接近预算要求仍在跑，已结束的高花费不挂这一档。
def budget_status(spent: Any, budget: Any, stop_reason: Any = None, running: bool = False) -> dict:
    """Classify one run. ``spent`` / ``budget`` are USD (``None`` when unknown), ``stop_reason`` as in ``result.json``,
    ``running`` is true while the search is still going.

    Returns ``level`` (``over`` / ``exhausted`` / ``near`` / ``ok`` / ``unknown``), ``flags`` (every alert that applies, worst first),
    ``flagged``, ``label`` (English text of the level, ``""`` when not flagged), ``spent`` / ``budget`` (floats or ``None``),
    ``used_pct``, ``over_usd`` / ``over_pct`` (the overrun; ``0.0`` when not over, ``None`` when it cannot be computed) and ``text``.
    A missing or non-positive budget, or a missing spend, never raises: the run is ``unknown`` unless ``stop_reason`` alone flags it."""
    s = _num(spent)
    b = _num(budget)
    if b is not None and b <= 0:
        # 非正预算无法算百分比。改成缺失后走 unknown，除非停止原因单独构成「耗尽」。
        b = None
    exhausted = str(stop_reason or "").strip() == EXHAUSTED_STOP_REASON
    known = s is not None and b is not None
    # 超预算：两边都是有限数，且花费比预算至少大过浮点噪声。刚好等于预算不是 OVER BUDGET。
    over = known and s - b > _EPS
    # 接近预算：必须仍在跑，并且还没超、也还没因 budget_exhausted 停下。门槛是预算的 80%。
    near = known and bool(running) and not over and not exhausted and s >= NEAR_FRACTION * b - _EPS
    # 书写顺序就是严重程度。又超又耗尽时 level 是 over，两档都留在 flags 里。
    flags = [name for name, on in (("over", over), ("exhausted", exhausted), ("near", near)) if on]
    # 没有告警时，数齐了是 ok，缺花费或缺预算是 unknown。唯独停止原因是 budget_exhausted 时，缺数字也算 exhausted。
    level = flags[0] if flags else ("ok" if known else "unknown")
    used_pct = round(s / b * 100.0, 1) if known else None
    over_usd = (max(s - b, 0.0) if over else 0.0) if known else None
    over_pct = (round((s - b) / b * 100.0, 1) if over else 0.0) if known else None
    parts = []
    if s is not None:
        parts.append(f"${s:.3f}" + (f" / ${b:.2f}" if b is not None else ""))
    elif b is not None:
        parts.append(f"budget ${b:.2f}")
    if used_pct is not None:
        parts.append(f"{used_pct:.1f}%")
    text = " · ".join(parts) if parts else "no budget recorded"
    if over:
        text += f" · over by ${over_usd:.3f} (+{over_pct:.1f}%)"
    return {
        "level": level,
        "flags": flags,
        "flagged": bool(flags),
        "label": LABELS.get(level, ""),
        "label_cn": LABELS_CN.get(level, ""),
        "spent": s,
        "budget": b,
        "used_pct": used_pct,
        "over_usd": over_usd,
        "over_pct": over_pct,
        "text": text,
    }


# 历史行可能已经带好 budget 字典。有 level 就沿用，避免横幅和表格各算一次、边界上档位不一致。
def status_of(row: dict) -> dict:
    """The ``budget`` status of a history row (computed from its fields when the row carries none)."""
    existing = row.get("budget")
    if isinstance(existing, dict) and "level" in existing:
        return existing
    return budget_status(row.get("spent_usd"), row.get("budget_usd"), row.get("stop_reason"), row.get("status") == "running")


# 横幅上的个数按 flags 累加，所以一条又超又耗尽的运行会同时计入 OVER 和 EXHAUSTED。排序只为了把最严重的放在最前。
def summarize(rows: list[dict]) -> dict:
    """The banner: how many runs are over budget / exhausted / near, which ones (worst first), and the total spend of ``rows``."""
    flagged = []
    spent_total = budget_total = over_total = 0.0
    counts = {"over": 0, "exhausted": 0, "near": 0}
    for row in rows:
        st = status_of(row)
        if st["spent"] is not None:
            spent_total += st["spent"]
        if st["budget"] is not None:
            budget_total += st["budget"]
        if st["over_usd"]:
            over_total += st["over_usd"]
        for flag in st["flags"]:
            counts[flag] += 1
        if st["flagged"]:
            flagged.append({
                "id": row.get("id"),
                "task": row.get("task") or "",
                "dtype": row.get("dtype") or "",
                "status": row.get("status") or "",
                "level": st["level"],
                "flags": st["flags"],
                "label": st["label"],
                "spent_usd": st["spent"],
                "budget_usd": st["budget"],
                "used_pct": st["used_pct"],
                "over_usd": st["over_usd"],
                "over_pct": st["over_pct"],
                "text": st["text"],
            })
    # 先按档位，再按超出的百分比、已用百分比。同一档里超出越多越靠前。
    flagged.sort(key=lambda item: (SEVERITY[item["level"]], -(item["over_pct"] or 0.0), -(item["used_pct"] or 0.0), str(item["id"])))
    return {
        "n_runs": len(rows),
        "n_flagged": len(flagged),
        "n_over": counts["over"],
        "n_exhausted": counts["exhausted"],
        "n_near": counts["near"],
        "level": flagged[0]["level"] if flagged else "ok",
        "total_spent_usd": round(spent_total, 6),
        "total_budget_usd": round(budget_total, 6),
        "total_over_usd": round(over_total, 6),
        "flagged": flagged,
    }


# 首屏在服务端把告警画进 HTML，没开 JavaScript 也能看见。页面脚本随后用同一份报告重画。
def render_banner_html(report: dict) -> str:
    """The banner as HTML, for the first paint and for readers without JavaScript. The page script redraws it from the same report on every frame."""
    esc = html.escape
    n_runs = report.get("n_runs", 0)
    total = f"{n_runs} 个运行 · 总花费 ${report.get('total_spent_usd', 0.0):.2f} / 预算合计 ${report.get('total_budget_usd', 0.0):.2f}"
    if not report.get("n_flagged"):
        return f'<div class="bb ok" data-level="ok"><b>预算正常</b> <span class="bb-total">没有超预算、预算耗尽或接近预算的运行 · {esc(total)}</span></div>'
    head = ['<b class="bb-title">⚠ 预算告警 BUDGET ALERT</b>']
    for flag, key in (("over", "n_over"), ("exhausted", "n_exhausted"), ("near", "n_near")):
        if report.get(key):
            head.append(f'<span class="bb-count {flag}">{report[key]} {LABELS[flag]} {LABELS_CN[flag]}</span>')
    head.append(f'<span class="bb-total">{esc(total)}' + (f" · 超出合计 +${report['total_over_usd']:.3f}" if report.get("total_over_usd") else "") + "</span>")
    chips = []
    for item in report.get("flagged", []):
        run_id = str(item.get("id") or "")
        names = " + ".join(LABELS[f] for f in item["flags"])
        title = (item.get("task") or run_id) + (f" {item['dtype']}" if item.get("dtype") else "")
        chips.append(
            f'<a class="chip {item["level"]}" href="/?run={quote(run_id, safe="")}" data-run="{esc(run_id, quote=True)}">'
            f"<b>{esc(run_id)}</b> <span>{esc(title)}</span> <i>{esc(names)}</i> <span>{esc(item['text'])}</span></a>"
        )
    return f'<div class="bb alert {report["level"]}" data-level="{report["level"]}"><div class="bb-head">{" ".join(head)}</div><div class="bb-list">{"".join(chips)}</div></div>'
