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

BANNER_CHIPS = 10

NEAR_FRACTION = 0.8
EXHAUSTED_STOP_REASON = "budget_exhausted"
_EPS = 1e-9  # float noise: a run that ends at exactly its budget is not "over"

LABELS = {"over": "OVER BUDGET", "exhausted": "BUDGET EXHAUSTED", "near": "NEAR BUDGET"}
LABELS_CN = {"over": "超预算", "exhausted": "预算耗尽", "near": "接近预算"}
SEVERITY = {"over": 0, "exhausted": 1, "near": 2, "ok": 3, "unknown": 4}


# 旧结果缺字段，或写成布尔、NaN 时不能拿去比大小。布尔是 int 的子类，不先排除的话 True 会变成 1 美元。
def _num(value: Any) -> float | None:
    """@brief 把值转成有限浮点数；缺失、非数值、布尔或 NaN/inf 返回 ``None``（旧运行缺字段）。

    原始英文说明：A finite number, or ``None`` for a missing, non-numeric, boolean or NaN/inf value (older runs lack fields).

    @param value 待转换的值，可能来自旧版 ``result.json``。
    @return 有限浮点数，或 ``None``。
    """
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


# 三档可以同时成立，level 取 flags 里最重的一档。缺预算或缺花费不会抛错。
# 刚好花到预算不算超（要多出 1e-9 以上）。耗尽只认停止原因，哪怕花费仍略低于上限。接近预算要求仍在跑，已结束的高花费不挂这一档。
def _grade(spent: Any, budget: Any, stop_reason: Any, running: bool):
    """@brief 预算分级的标量实现；``budget_status`` 与 ``summarize_columns`` 共用同一套判定。

    @param spent 已花费美元数，可为 None。
    @param budget 预算美元数，可为 None；非正值视为未知。
    @param stop_reason 停止原因，等于 ``budget_exhausted`` 时计入“耗尽”。
    @param running 搜索是否仍在运行。
    @return ``(spent, budget, flags, level, known, over, used_pct, over_usd, over_pct)``。
    """
    s = _num(spent)
    b = _num(budget)
    if b is not None and b <= 0:
        b = None
    exhausted = str(stop_reason or "").strip() == EXHAUSTED_STOP_REASON
    known = s is not None and b is not None
    over = bool(known and s - b > _EPS)
    near = bool(known and running and not over and not exhausted and s >= NEAR_FRACTION * b - _EPS)
    flags = [name for name, on in (("over", over), ("exhausted", exhausted), ("near", near)) if on]
    level = flags[0] if flags else ("ok" if known else "unknown")
    used_pct = round(s / b * 100.0, 1) if known else None
    over_usd = (max(s - b, 0.0) if over else 0.0) if known else None
    over_pct = (round((s - b) / b * 100.0, 1) if over else 0.0) if known else None
    return s, b, flags, level, known, over, used_pct, over_usd, over_pct


def _status_text(spent: Any, budget: Any, used_pct: Any, over: bool, over_usd: Any, over_pct: Any) -> str:
    """@brief 横幅上的一段文字；只有 ``budget_status`` 与 flagged 行需要。

    @param spent 已花费。
    @param budget 告警预算。
    @param used_pct 已用百分比。
    @param over 是否超预算。
    @param over_usd 超出金额。
    @param over_pct 超出百分比。
    @return 展示文本。
    """
    parts = []
    if spent is not None:
        parts.append(f"${spent:.3f}" + (f" / ${budget:.2f}" if budget is not None else ""))
    elif budget is not None:
        parts.append(f"budget ${budget:.2f}")
    if used_pct is not None:
        parts.append(f"{used_pct:.1f}%")
    text = " · ".join(parts) if parts else "no budget recorded"
    if over:
        text += f" · over by ${over_usd:.3f} (+{over_pct:.1f}%)"
    return text


def budget_status(spent: Any, budget: Any, stop_reason: Any = None, running: bool = False) -> dict:
    """@brief 对一个运行做预算分级。

    ``spent`` / ``budget`` 为美元（未知时为 ``None``），``stop_reason`` 取自 ``result.json``，
    ``running`` 表示搜索仍在进行。缺预算或缺花费不会抛错：除了停止原因单独构成“耗尽”的情况外，
    该运行记为 ``unknown``。刚好花到预算不算超（需多出 1e-9 以上）；“接近预算”只发给仍在跑且已用 80% 的运行。

    @param spent 已花费美元数，可为 None。
    @param budget 预算美元数，可为 None；非正值视为未知。
    @param stop_reason 停止原因，等于 ``budget_exhausted`` 时计入“耗尽”。
    @param running 搜索是否仍在运行。
    @return 字典，含 ``level``（over/exhausted/near/ok/unknown）、``flags``（所有适用的告警，最重的在前）、
        ``flagged``、``label``（未告警时为空串）、``spent`` / ``budget``（浮点或 None）、``used_pct``、
        ``over_usd`` / ``over_pct``（未超时为 0.0，无法计算时为 None）以及 ``text``。
    """
    s, b, flags, level, known, over, used_pct, over_usd, over_pct = _grade(spent, budget, stop_reason, running)
    return {
        "level": level,
        "flags": flags,
        "flagged": bool(flags),
        "label": LABELS.get(level, ""),
        "label_cn": LABELS_CN.get(level, ""),
        "spent": s,
        "budget": b,
        "remaining_usd": (b - s) if known else None,
        "used_pct": used_pct,
        "over_usd": over_usd,
        "over_pct": over_pct,
        "text": _status_text(s, b, used_pct, over, over_usd, over_pct),
    }


# 历史行可能已经带好 budget 字典。有 level 就沿用，避免横幅和表格各算一次、边界上档位不一致。
def status_of(row: dict) -> dict:
    """@brief 取历史行的 ``budget`` 状态；行内未带上时按字段现算。

    原始英文说明：The ``budget`` status of a history row (computed from its fields when the row carries none).

    @param row 历史行字典。
    @return 已有的 budget 状态字典，或按 ``spent_usd`` / ``budget_usd`` / ``stop_reason`` / ``status`` 现算的结果。
    """
    existing = row.get("budget")
    if isinstance(existing, dict) and "level" in existing:
        return existing
    return budget_status(row.get("spent_usd"), row.get("budget_usd"), row.get("stop_reason"), row.get("status") == "running")


# 横幅上的个数按 flags 累加，所以一条又超又耗尽的运行会同时计入 OVER 和 EXHAUSTED。排序只为了把最严重的放在最前。
def summarize(rows: list[dict]) -> dict:
    """@brief 汇总横幅数据：超预算/耗尽/接近预算的运行数与名单（最重的在前），以及 ``rows`` 的总花费。

    原始英文说明：The banner: how many runs are over budget / exhausted / near, which ones (worst first), and the total spend of ``rows``.

    一条运行可同时计入多档（按 flags 累加）。

    @param rows 历史行字典列表。
    @return 含 ``n_runs``、``n_flagged``、``n_over``、``n_exhausted``、``n_near``、``level``、
        总花费/总预算/总超出额与 ``flagged`` 名单的报告字典。
    """
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
                "control": row.get("control"),  # what 'add budget' would do; observe.detail.control_hint fills this on history rows
                "topped_up": bool(row.get("topped_up")),
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


def summarize_columns(rows, *, totals=None) -> dict:
    """@brief 窄投影行的预算汇总：不构造每行的状态/历史字典，只给 flagged 行建对象。

    ``rows`` 可以是全量行（自己累加合计），也可以只是“可能 flagged”的候选集：给 ``totals``
    时合计与 ``n_runs`` 由 SQL 聚合给出，本函数只负责逐行判定（调用方必须保证候选集覆盖所有
    flagged 行）。字段与 :func:`summarize` 完全相同；判定复用 :func:`_grade`，不另写阈值。

    @param rows 可迭代的窄行：``(id, task, dtype, status, stop_reason, spent, alert_budget,
        disk_budget, control_action, control_reason, topped_up)``。
    @param totals 可选的 ``(n_runs, total_spent, total_budget, total_over)``（已由 SQL 算好）。
    @return 与 :func:`summarize` 相同形状的报告字典。
    """
    flagged = []
    spent_total = budget_total = over_total = 0.0
    counts = {"over": 0, "exhausted": 0, "near": 0}
    total_rows = 0
    for item in rows:
        total_rows += 1
        (run_id, task, dtype, status, raw_stop, spent, alert, disk,
         control_action, control_reason, topped) = item
        status = status or ""
        stop = raw_stop or ""
        shown = _num(alert)
        on_disk = _num(disk)
        spent_n = _num(spent)
        # 与 query._budget_row 一致：尚未被运行采纳的加预算会撤销旧的 budget_exhausted 标记。
        if (
            shown is not None and on_disk is not None and shown > on_disk + 1e-9
            and stop == EXHAUSTED_STOP_REASON and spent_n is not None and shown > spent_n
        ):
            stop = ""
        s, b, flags, level, known, over, used_pct, over_usd, over_pct = _grade(spent, alert, stop, status == "running")
        if s is not None:
            spent_total += s
        if b is not None:
            budget_total += b
        if over_usd:
            over_total += over_usd
        for flag in flags:
            counts[flag] += 1
        if flags:
            flagged.append({
                "id": run_id,
                "task": task or "",
                "dtype": dtype or "",
                "status": status,
                "level": level,
                "flags": flags,
                "label": LABELS.get(level, ""),
                "spent_usd": s,
                "budget_usd": b,
                "used_pct": used_pct,
                "over_usd": over_usd,
                "over_pct": over_pct,
                "text": _status_text(s, b, used_pct, over, over_usd, over_pct),
                "control": {"action": control_action, "reason": control_reason or ""},
                "topped_up": bool(topped),
            })
    flagged.sort(key=lambda item: (SEVERITY[item["level"]], -(item["over_pct"] or 0.0), -(item["used_pct"] or 0.0), str(item["id"])))
    if totals is None:
        n_runs, spent_total, budget_total, over_total = total_rows, spent_total, budget_total, over_total
    else:
        n_runs, spent_total, budget_total, over_total = totals
    return {
        "n_runs": n_runs,
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
    """@brief 把预算报告渲染成 HTML 横幅，供首屏与无 JavaScript 的读者使用。

    原始英文说明：The banner as HTML, for the first paint and for readers without JavaScript. The page script redraws it from the same report on every frame.

    @param report ``summarize`` 返回的报告字典。
    @return 横幅 HTML 字符串；无告警时返回正常状态的横幅。
    """
    esc = html.escape
    n_runs = report.get("n_runs", 0)
    total = f"{n_runs} 个运行 · 总花费 ${report.get('total_spent_usd', 0.0):.2f} / 预算合计 ${report.get('total_budget_usd', 0.0):.2f}"
    if not report.get("n_flagged"):
        return f'<div class="bb ok" data-level="ok"><b>预算正常</b> <span class="bb-total">没有超预算、预算耗尽或接近预算的运行 · {esc(total)}</span></div>'
    head = ['<b class="bb-title">⚠ 预算告警 BUDGET ALERT</b>']
    for flag, key in (("over", "n_over"), ("exhausted", "n_exhausted"), ("near", "n_near")):
        if report.get(key):
            head.append(f'<span class="bb-count {flag}">{report[key]} {LABELS[flag]} {LABELS_CN[flag]}</span>')
    head.append(f'<span class="bb-count total">共 {report.get("n_flagged", 0)} 条</span>')
    head.append(f'<span class="bb-total">{esc(total)}' + (f" · 超出合计 +${report['total_over_usd']:.3f}" if report.get("total_over_usd") else "") + "</span>")
    chips = []
    flagged = report.get("flagged", [])
    for item in flagged[:BANNER_CHIPS]:
        run_id = str(item.get("id") or "")
        names = " + ".join(LABELS[f] for f in item["flags"])
        title = (item.get("task") or run_id) + (f" {item['dtype']}" if item.get("dtype") else "")
        chips.append(
            f'<a class="chip {item["level"]}" href="/?run={quote(run_id, safe="")}" data-run="{esc(run_id, quote=True)}">'
            f'<b class="id-text" title="{esc(run_id, quote=True)}">{esc(run_id)}</b> '
            f"<span>{esc(title)}</span> <i>{esc(names)}</i> <span>{esc(item['text'])}</span></a>"
        )
    if len(flagged) > BANNER_CHIPS:
        # 总条数在标题里，列表只画前 N 条；否则 500 条告警的 HTML 会线性膨胀。
        chips.append(f'<span class="bb-more">还有 {len(flagged) - BANNER_CHIPS} 条</span>')
    head.append('<button type="button" id="budget-toggle" class="budget-toggle" aria-expanded="true" title="收起预算告警">收起</button>')
    return f'<div class="bb alert {report["level"]}" data-level="{report["level"]}"><div class="bb-head">{" ".join(head)}</div><div class="bb-list">{"".join(chips)}</div></div>'
