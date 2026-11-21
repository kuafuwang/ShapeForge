"""Read-only analysis of natural-run TASK dynamic-budget delivery receipts.

There are two different things to measure: the unified situation block itself,
and every other dynamic block competing for the same TASK budget. A delivered
situation block does not prove that NOTES, evidence or history survived.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

SCHEMA = "shapeforge.dynamic-budget-audit.1"
LAYER_NAMES = ("overlay", "host_ledger", "peer_unverified")


def _int(value: Any) -> int:
    return int(value) if isinstance(value, int) and not isinstance(value, bool) else 0


def _ratio(part: int, whole: int) -> float | None:
    return part / whole if whole else None


def _block_totals() -> dict[str, int]:
    return {"expected_bytes": 0, "delivered_bytes": 0, "omitted_bytes": 0, "receipts_with_omission": 0}


def audit(runs_root: Path) -> dict[str, Any]:
    """@brief 汇总自然 run 的 `task_delivery.json`，不写任何文件。

    @param runs_root 包含 run 目录的根目录。
    @return 静态/动态/态势块/其他动态块的送达与省略，以及热点 run。
    """
    runs_root = Path(runs_root)
    receipts = list(sorted(runs_root.rglob("task_delivery.json")))
    total = Counter()
    statuses = Counter()
    budgets = Counter()
    situation_budgets = Counter()
    layers = {name: Counter() for name in LAYER_NAMES}
    blocks: dict[str, Counter] = {}
    reasons = Counter()
    per_run: dict[str, Counter] = {}
    for path in receipts:
        document = json.loads(path.read_text(encoding="utf-8"))
        run_dir = path.parent.parent.parent
        run_id = str(run_dir.relative_to(runs_root))
        run = per_run.setdefault(run_id, Counter())
        run["receipts"] += 1
        static = document.get("static") or {}
        dynamic = document.get("dynamic") or {}
        situation = document.get("situation") or {}
        for prefix, source in (("static", static), ("dynamic_receipt", dynamic)):
            total[f"{prefix}_delivered"] += _int(source.get("delivered_bytes"))
            total[f"{prefix}_omitted"] += _int(source.get("omitted_bytes"))
        if _int(dynamic.get("omitted_bytes")):
            total["receipts_with_dynamic_omission"] += 1
            run["dynamic_omitted"] += _int(dynamic.get("omitted_bytes"))
        else:
            total["receipts_without_dynamic_omission"] += 1
        if dynamic.get("over_budget") is True:
            total["receipts_over_budget"] += 1
        budgets[_int(dynamic.get("budget_bytes"))] += 1
        situation_budgets[_int(situation.get("budget_bytes"))] += 1
        statuses[str(situation.get("status") or "missing")] += 1
        for field in ("source_bytes", "sanitized_bytes", "output_bytes", "delivered_bytes", "omitted_bytes"):
            value = _int(situation.get(field))
            total[f"situation_{field}"] += value
            run[f"situation_{field}"] += value
        for row in situation.get("layers") or []:
            name = str(row.get("name") or "")
            if name not in layers:
                continue
            for field in ("source_bytes", "sanitized_bytes", "rendered_bytes", "output_bytes", "delivered_bytes", "omitted_bytes"):
                layers[name][field] += _int(row.get(field))
            if _int(row.get("omitted_bytes")):
                layers[name]["receipts_with_omission"] += 1
        for block in document.get("blocks") or []:
            name = str(block.get("name") or "unknown")
            bucket = blocks.setdefault(name, Counter())
            expected = _int(block.get("expected_bytes"))
            delivered = _int(block.get("delivered_bytes"))
            omitted = _int(block.get("omitted_bytes"))
            bucket["expected_bytes"] += expected
            bucket["delivered_bytes"] += delivered
            bucket["omitted_bytes"] += omitted
            if omitted:
                bucket["receipts_with_omission"] += 1
                reasons[str(block.get("omission_reason") or "unknown")] += omitted
            if expected != delivered + omitted:
                total["block_accounting_mismatches"] += 1
    block_document = {}
    for name, bucket in sorted(blocks.items(), key=lambda item: item[1]["omitted_bytes"], reverse=True):
        block_document[name] = dict(bucket)
        block_document[name]["omitted_ratio"] = _ratio(bucket["omitted_bytes"], bucket["expected_bytes"])
    dynamic_expected = total["dynamic_receipt_delivered"] + total["dynamic_receipt_omitted"]
    other_omitted = sum(
        bucket["omitted_bytes"] for name, bucket in blocks.items() if name != "本轮态势"
    )
    total["other_block_omitted"] = other_omitted
    if total["static_omitted"]:
        verdict = "static_contract_was_cut"
    elif other_omitted:
        verdict = "situation_delivered_but_other_blocks_were_cut"
    else:
        verdict = "all_dynamic_content_delivered"
    hot_runs = []
    for run_id, bucket in sorted(per_run.items(), key=lambda item: item[1]["dynamic_omitted"], reverse=True):
        if not bucket["dynamic_omitted"]:
            continue
        hot_runs.append({"run_id": run_id, **dict(bucket)})
    return {
        "schema": SCHEMA,
        "runs_root": str(runs_root),
        "receipts": len(receipts),
        "runs": len(per_run),
        "counts": dict(total),
        "dynamic_omission_ratio": _ratio(total["dynamic_receipt_omitted"], dynamic_expected),
        "situation_status": dict(statuses),
        "dynamic_budgets": {str(key): value for key, value in sorted(budgets.items())},
        "situation_budgets": {str(key): value for key, value in sorted(situation_budgets.items())},
        "situation_layers": {name: dict(layers[name]) for name in LAYER_NAMES},
        "blocks": block_document,
        "omission_reasons_bytes": dict(reasons),
        "hot_runs": hot_runs[:12],
        "verdict": verdict,
        "note": (
            "situation 的 delivered 只说明统一态势块自身送达；上一轮 NOTES、现场证据、自测对照、"
            "最近的历史等其他块可能仍被同一 TASK 预算挤掉。静态 omitted=0 也不能证明动态负担下降。"
        ),
    }


def render_summary(document: dict[str, Any]) -> str:
    counts = document["counts"]
    ratio = document["dynamic_omission_ratio"]
    lines = [
        "Dynamic budget audit (read-only)",
        f"  receipts: {document['receipts']} over {document['runs']} runs",
        f"  static omitted: {counts.get('static_omitted', 0)} B",
        f"  dynamic delivered/omitted: {counts['dynamic_receipt_delivered']} / {counts['dynamic_receipt_omitted']} B"
        + (f" ({ratio:.2%} omitted)" if ratio is not None else ""),
        f"  receipts with omissions: {counts.get('receipts_with_dynamic_omission', 0)}; over budget: {counts.get('receipts_over_budget', 0)}",
        f"  situation delivered/omitted: {counts['situation_delivered_bytes']} / {counts['situation_omitted_bytes']} B",
        f"  other-block omitted: {counts.get('other_block_omitted', 0)} B; accounting mismatches: {counts.get('block_accounting_mismatches', 0)}",
        f"  verdict: {document['verdict']}",
        "  block omissions:",
    ]
    for name, row in document["blocks"].items():
        if row["omitted_bytes"]:
            lines.append(f"    {name}: {row['omitted_bytes']} B across {row['receipts_with_omission']} receipts")
    lines.append("  hottest runs:")
    for row in document["hot_runs"][:5]:
        lines.append(f"    {row['run_id']}: {row['dynamic_omitted']} B")
    return "\n".join(lines) + "\n"


__all__ = ["SCHEMA", "audit", "render_summary"]
