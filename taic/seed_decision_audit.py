"""Read-only audit of candidate-vs-seed paired observations.

The pair's ``changes_seed`` field is deliberately constant false: it records the
observation boundary, not the decision. The actual decision comes from the run's
``result.json`` trace (``improved`` / ``within_noise``). A decisive pair can
support or contradict that decision; ``too_noisy`` is never a disagreement.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

SCHEMA = "shapeforge.seed-decision-audit.1"


def _json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def _decision(row: dict[str, Any]) -> str:
    if row.get("improved") is True:
        return "replace"
    if row.get("within_noise") is True:
        return "tie"
    return "keep"


def _evaluate(pair: dict[str, Any], trace: dict[str, Any]) -> tuple[str, str, str]:
    """@brief 返回 (配对方向, 实际决策, 是否一致)。

    ``a_faster`` 表示候选比旧种子快，只有它才要求 replace；b_faster 表示旧种子
    更快，replace 才与配对矛盾。其余任何结局都不能用来指认误判。
    """
    actual = _decision(trace)
    direction = str(pair.get("direction") or "")
    if pair.get("conclusion") != "diff" or direction not in ("a_faster", "b_faster"):
        return "undecidable", actual, "unresolved"
    expected_replace = direction == "a_faster"
    replaced = actual == "replace"
    return direction, actual, "agree" if expected_replace == replaced else "disagree"


def audit(runs_root: Path) -> dict[str, Any]:
    """@brief 审计 ``runs_root`` 下所有 ``paired_comparison.json``，不写文件。

    @param runs_root 包含各 run 目录的根目录。
    @return 机器可读的配对/决策一致性与覆盖摘要。
    """
    runs_root = Path(runs_root)
    counts = {
        "pair_files": 0,
        "not_applied": 0,
        "applied": 0,
        "too_noisy": 0,
        "decisive": 0,
        "agree": 0,
        "disagree": 0,
        "unresolved": 0,
        "unresolved_replace": 0,
        "unresolved_keep": 0,
        "unresolved_tie": 0,
        "missing_trace": 0,
        "legacy_clock_unspecified": 0,
    }
    decisive = []
    runs_with_pairs = set()
    for path in sorted(runs_root.rglob("paired_comparison.json")):
        counts["pair_files"] += 1
        pair = _json(path)
        if pair.get("applied") is not True:
            counts["not_applied"] += 1
            continue
        counts["applied"] += 1
        run_dir = path.parent.parent.parent
        runs_with_pairs.add(str(run_dir))
        if not pair.get("timed_by"):
            # schema-1 的 gpu=true 只表示使用了 GPU，不保证读的是 GPU 时间戳。
            counts["legacy_clock_unspecified"] += 1
        result = _json(run_dir / "result.json")
        trace_rows = result.get("trace") if isinstance(result.get("trace"), list) else []
        by_label = {str(row.get("label") or ""): row for row in trace_rows if isinstance(row, dict)}
        label = str(pair.get("session_label") or path.parent.name.split("_", 1)[-1])
        trace = by_label.get(label)
        if trace is None:
            counts["missing_trace"] += 1
            trace = {}
        direction, actual, verdict = _evaluate(pair, trace)
        if not trace:
            actual = "unknown"
        if pair.get("conclusion") == "too_noisy":
            counts["too_noisy"] += 1
        if verdict == "unresolved":
            counts["unresolved"] += 1
            if actual in ("replace", "keep", "tie"):
                counts[f"unresolved_{actual}"] += 1
            continue
        counts["decisive"] += 1
        counts[verdict] += 1
        item = {
            "run_id": str(run_dir.relative_to(runs_root)),
            "session_label": label,
            "candidate_dir": path.parent.name,
            "direction": direction,
            "actual_decision": actual,
            "verdict": verdict,
            "ratio_candidate_over_seed": (
                float(pair["median_b_s"]) / float(pair["median_a_s"])
                if isinstance(pair.get("median_a_s"), (int, float)) and pair["median_a_s"] > 0
                and isinstance(pair.get("median_b_s"), (int, float))
                else None
            ),
            "changes_seed_field": pair.get("changes_seed"),
            "timed_by": pair.get("timed_by"),
        }
        decisive.append(item)
    if counts["disagree"]:
        verdict = "review_decisive_mismatches"
    elif counts["agree"]:
        verdict = "no_observed_misdecision"
    else:
        verdict = "insufficient_decisive_evidence"
    return {
        "schema": SCHEMA,
        "runs_root": str(runs_root),
        "runs_with_applied_pairs": len(runs_with_pairs),
        "counts": counts,
        "decisive_pairs": decisive,
        "verdict": verdict,
        "decision_change_recommended": False,
        "note": (
            "changes_seed 是观测边界字段，不是实际决策；实际决策来自 result.json trace。"
            "diff+a_faster 才支持 replace，diff+b_faster 才反对 replace；too_noisy 一律不可判。"
            "schema-1 的 gpu=true 不代表 GPU 时间戳，未标 timed_by 的旧记录只作方向参考。"
        ),
    }


def render_summary(document: dict[str, Any]) -> str:
    counts = document["counts"]
    lines = [
        "Seed decision audit (read-only)",
        f"  pair files: {counts['pair_files']}; applied: {counts['applied']}; not applied: {counts['not_applied']}",
        f"  decisive: {counts['decisive']}; agree: {counts['agree']}; disagree: {counts['disagree']}; unresolved: {counts['unresolved']}",
        f"  too_noisy: {counts['too_noisy']}; unresolved replace: {counts['unresolved_replace']}; legacy clock unspecified: {counts['legacy_clock_unspecified']}",
        f"  verdict: {document['verdict']}; change decision rule: {document['decision_change_recommended']}",
        "  too_noisy is not evidence of a wrong seed decision.",
    ]
    for row in document["decisive_pairs"]:
        ratio = row["ratio_candidate_over_seed"]
        lines.append(
            f"  {row['run_id']} {row['session_label']}: {row['direction']} vs {row['actual_decision']} "
            f"-> {row['verdict']}" + (f" (candidate/seed {ratio:.4f})" if ratio is not None else "")
        )
    return "\n".join(lines) + "\n"


__all__ = ["SCHEMA", "audit", "render_summary"]
