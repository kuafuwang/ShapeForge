"""Read-only, round-stratified association between TASK omissions and outcomes.

This does not estimate a causal effect. Later rounds both omit more content and
naturally produce fewer new bests. The analysis therefore separates r1-r2 from
r3+ and reports only descriptive associations.
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

SCHEMA = "shapeforge.dynamic-omission-impact.1"


def _stage(label: str) -> str:
    if label == "init":
        return "init"
    match = re.match(r"r(\d+)\.", label)
    return "r1-2" if match and int(match.group(1)) <= 2 else "r3+"


def _bucket(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "sessions": len(rows),
        "new_best": sum(row.get("improved") is True for row in rows),
        "ok": sum(row.get("status") == "ok" for row in rows),
        "within_noise": sum(row.get("within_noise") is True for row in rows),
        "new_best_rate": (
            sum(row.get("improved") is True for row in rows) / len(rows) if rows else None
        ),
        "ok_rate": (
            sum(row.get("status") == "ok" for row in rows) / len(rows) if rows else None
        ),
    }


def audit(runs_root: Path) -> dict[str, Any]:
    """@brief 按轮次分层比较省略/未省略会话，不写文件。

    @param runs_root 包含 run 目录的根目录。
    @return 描述性关联、样本数与不可因果判定的说明。
    """
    runs_root = Path(runs_root)
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    skipped = Counter()
    for path in sorted(runs_root.rglob("task_delivery.json")):
        run_dir = path.parent.parent.parent
        result_path = run_dir / "result.json"
        if not result_path.is_file():
            skipped["missing_result"] += 1
            continue
        result = json.loads(result_path.read_text(encoding="utf-8"))
        trace = result.get("trace") if isinstance(result.get("trace"), list) else []
        label = path.parent.name
        row = next((item for item in trace if isinstance(item, dict) and item.get("label") == label), None)
        if row is None:
            skipped["missing_trace"] += 1
            continue
        delivery = json.loads(path.read_text(encoding="utf-8"))
        omitted = (delivery.get("dynamic") or {}).get("omitted_bytes") or 0
        stage = _stage(label)
        arm = "omitted" if omitted else "clean"
        enriched = dict(row)
        enriched["dynamic_omitted_bytes"] = omitted
        enriched["run_id"] = str(run_dir.relative_to(runs_root))
        enriched["session_label"] = label
        grouped[(stage, arm)].append(enriched)
    groups = []
    for (stage, arm), rows in sorted(grouped.items()):
        groups.append({"stage": stage, "arm": arm, **_bucket(rows)})
    early = {(row["stage"], row["arm"]): row for row in groups}
    omitted_early = early.get(("r1-2", "omitted"), {"new_best_rate": None, "sessions": 0})
    clean_early = early.get(("r1-2", "clean"), {"new_best_rate": None, "sessions": 0})
    late_omitted = early.get(("r3+", "omitted"), {"sessions": 0})
    late_clean = early.get(("r3+", "clean"), {"sessions": 0})
    return {
        "schema": SCHEMA,
        "runs_root": str(runs_root),
        "groups": groups,
        "skipped": dict(skipped),
        "early_late_imbalance": {
            "r1-2_omitted": early.get(("r1-2", "omitted"), {"sessions": 0})["sessions"],
            "r1-2_clean": early.get(("r1-2", "clean"), {"sessions": 0})["sessions"],
            "r3+_omitted": late_omitted["sessions"],
            "r3+_clean": late_clean["sessions"],
        },
        "early_new_best_rate_difference_omitted_minus_clean": (
            omitted_early["new_best_rate"] - clean_early["new_best_rate"]
            if omitted_early["new_best_rate"] is not None and clean_early["new_best_rate"] is not None
            else None
        ),
        "verdict": "association_confounded_by_round",
        "causal_claim_allowed": False,
        "note": (
            "后续轮次既更容易省略，也更难出现新 best。即使省略组的新 best 比例更低，"
            "也不能据此推断省略导致下降。样本只支持定位下一轮该评估的信息价值，"
            "不支持直接调大或调小预算。"
        ),
    }


def render_summary(document: dict[str, Any]) -> str:
    lines = ["Dynamic omission impact (read-only, round-stratified)"]
    for row in document["groups"]:
        best_rate = row["new_best_rate"]
        ok_rate = row["ok_rate"]
        lines.append(
            f"  {row['stage']:5} {row['arm']:7} sessions={row['sessions']:3} "
            f"new_best={row['new_best']:3} ({'—' if best_rate is None else f'{best_rate:.1%}'}) "
            f"ok={row['ok']:3} ({'—' if ok_rate is None else f'{ok_rate:.1%}'})"
        )
    difference = document["early_new_best_rate_difference_omitted_minus_clean"]
    lines.append(
        "  early (r1-2) omitted-minus-clean new-best rate: "
        + ("—" if difference is None else f"{difference:+.2%}")
    )
    lines.append(f"  verdict: {document['verdict']}; causal claim allowed: {document['causal_claim_allowed']}")
    return "\n".join(lines) + "\n"


__all__ = ["SCHEMA", "audit", "render_summary"]
