"""Read-only analysis of exact-SHA paired rechecks and shadow seed replay.

A recheck is a different window from the original evaluation. Disagreement is a
review signal about window sensitivity, not proof that relation_to_best is wrong.
The replay asks whether following decisive pair directions would have changed
the final winner, using search-phase speedup only (never calling it confirmed).
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .seed_decision_audit import _evaluate

SCHEMA = "shapeforge.seed-decision-recheck.1"


def _speedup(trace: list[dict[str, Any]], label: str) -> float | None:
    row = next((item for item in trace if item.get("label") == label), None)
    value = row.get("speedup") if row else None
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def audit(runs_root: Path) -> dict[str, Any]:
    """@brief 分析 ``paired_comparison_recheck.json`` 与影子换种子的最终胜者。

    @param runs_root 包含 run 目录的根目录。
    @return 重测覆盖、冲突和逐 run 的影子胜者对比。
    """
    runs_root = Path(runs_root)
    by_run: dict[Path, list[tuple[Path, dict[str, Any]]]] = {}
    for path in sorted(runs_root.rglob("paired_comparison_recheck.json")):
        wrapper = json.loads(path.read_text(encoding="utf-8"))
        document = wrapper.get("document") if isinstance(wrapper.get("document"), dict) else {}
        run_dir = path.parent.parent.parent
        by_run.setdefault(run_dir, []).append((path, document))
    counts = Counter()
    disagreements = []
    replays = []
    for run_dir, entries in sorted(by_run.items(), key=lambda item: str(item[0])):
        result = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
        trace = result.get("trace") or []
        rows = []
        for path, document in entries:
            label = document.get("session_label") or path.parent.name.split("_", 1)[-1]
            trace_row = next((row for row in trace if row.get("label") == label), {})
            direction, actual, verdict = _evaluate(document, trace_row)
            if document.get("conclusion") == "too_noisy":
                counts["too_noisy"] += 1
            elif document.get("conclusion") == "no_diff":
                counts["no_diff"] += 1
            if verdict == "unresolved":
                counts["unresolved"] += 1
            else:
                counts["decisive"] += 1
                counts[verdict] += 1
                if verdict == "disagree":
                    disagreements.append({
                        "run_id": str(run_dir.relative_to(runs_root)),
                        "session_label": label,
                        "direction": direction,
                        "actual_decision": actual,
                        "timed_by": document.get("timed_by"),
                        "note": "重测与原评测是不同窗口，不据此直接判规则错误",
                    })
            rows.append((label, direction))
        # 影子规则：只有 diff+a_faster 才换种子；其余结局不换。
        alternative = trace[0].get("label") if trace else None
        for label, direction in sorted(rows, key=lambda item: next((row.get("idx", 0) for row in trace if row.get("label") == item[0]), 0)):
            if direction == "a_faster":
                alternative = label
        actual = result.get("best_label")
        actual_speed = _speedup(trace, actual)
        alternative_speed = _speedup(trace, alternative)
        outcome = "same"
        if alternative != actual:
            if actual_speed is not None and alternative_speed is not None:
                outcome = "alternative_better" if alternative_speed > actual_speed else "alternative_worse"
            else:
                outcome = "different_unmeasured"
        replays.append({
            "run_id": str(run_dir.relative_to(runs_root)),
            "actual_best": actual,
            "actual_search_speedup": actual_speed,
            "alternative_best": alternative,
            "alternative_search_speedup": alternative_speed,
            "outcome": outcome,
            "decisions": [{"label": label, "direction": direction} for label, direction in rows],
        })
        counts[f"replay_{outcome}"] += 1
    return {
        "schema": SCHEMA,
        "runs_root": str(runs_root),
        "runs_with_rechecks": len(by_run),
        "counts": dict(counts),
        "decisive_disagreements": disagreements,
        "shadow_replays": replays,
        "verdict": "review_decisive_mismatches" if disagreements else "no_observed_misdecision",
        "decision_change_recommended": False,
        "note": (
            "重测使用精确 SHA 与 GPU 时间戳，但与原评测不是同一窗口。冲突提示窗口敏感性，"
            "不是 relation_to_best 的错误证明。影子回放只比较 search-phase speedup，"
            "不是 fresh confirmation，也不构成换规则授权。"
        ),
    }


def render_summary(document: dict[str, Any]) -> str:
    counts = document["counts"]
    lines = [
        "Seed decision recheck (exact SHA, read-only)",
        f"  runs: {document['runs_with_rechecks']}; decisive: {counts.get('decisive', 0)}; agree: {counts.get('agree', 0)}; disagree: {counts.get('disagree', 0)}; unresolved: {counts.get('unresolved', 0)}",
        f"  too_noisy: {counts.get('too_noisy', 0)}; no_diff: {counts.get('no_diff', 0)}",
        f"  replay better/worse/same/different_unmeasured: {counts.get('replay_alternative_better', 0)}/{counts.get('replay_alternative_worse', 0)}/{counts.get('replay_same', 0)}/{counts.get('replay_different_unmeasured', 0)}",
        f"  verdict: {document['verdict']}; change decision rule: {document['decision_change_recommended']}",
    ]
    for row in document["decisive_disagreements"]:
        lines.append(f"    {row['run_id']} {row['session_label']}: {row['direction']} vs {row['actual_decision']}")
    return "\n".join(lines) + "\n"


__all__ = ["SCHEMA", "audit", "render_summary"]
