"""Content-level audit of omitted previous-NOTES blocks.

The omitted source is reconstructed from the previous candidate's report, NOTES
and trace decision, and accepted only when its UTF-8 length equals the receipt's
source_bytes. SequenceMatcher then measures literal sharing with the delivered
TASK after removing the NOTES block itself. This is not a semantic-importance or
causal-harm metric.
"""

from __future__ import annotations

import difflib
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

from taic.orchestrator import _notes_for_next

SCHEMA = "shapeforge.omission-content-audit.1"


def _outer(row: dict[str, Any], report: dict[str, Any]) -> str:
    us = f"{report['candidate_us']:.0f} µs" if isinstance(report.get("candidate_us"), (int, float)) else "没有微秒数"
    replaced = "换了种子" if row.get("improved") is True else "没换种子"
    return f"{us}，状态 {report.get('status')}，{replaced}"


def _without_notes(task: str) -> str:
    start = task.find("上一轮 NOTES")
    if start < 0:
        return task
    end = task.find("\n# ", start + 1)
    return task if end < 0 else task[:start] + task[end + 1:]


def _shared_ratio(source: str, task: str) -> float:
    body = "\n".join(source.splitlines()[1:])
    if not body:
        return 0.0
    matcher = difflib.SequenceMatcher(None, body, _without_notes(task), autojunk=False)
    return min(1.0, sum(block.size for block in matcher.get_matching_blocks()) / len(body))


def audit(runs_root: Path) -> dict[str, Any]:
    """@brief 对可精确重建的省略 NOTES 做字面重复度分析，不写文件。

    @param runs_root 包含 run 目录的根目录。
    @return 重建覆盖、字面共享比例分布与不可因果判定的结论。
    """
    runs_root = Path(runs_root)
    counts = Counter()
    rows = []
    for path in sorted(runs_root.rglob("task_delivery.json")):
        document = json.loads(path.read_text(encoding="utf-8"))
        block = next((item for item in document.get("blocks") or [] if item.get("name") == "上一轮 NOTES"), None)
        if not block or not block.get("omitted_bytes"):
            continue
        counts["omitted_notes_receipts"] += 1
        run_dir = path.parent.parent.parent
        result_path = run_dir / "result.json"
        if not result_path.is_file():
            counts["missing_result"] += 1
            continue
        result = json.loads(result_path.read_text(encoding="utf-8"))
        trace = result.get("trace") or []
        label = path.parent.name
        index = next((i for i, row in enumerate(trace) if row.get("label") == label), None)
        if not index:
            counts["missing_previous_candidate"] += 1
            continue
        previous = trace[index - 1]
        folder = run_dir / "candidates" / f"{previous['idx']:03d}_{previous['label']}"
        report_path = folder / "report.json"
        notes_path = folder / "NOTES.md"
        task_path = run_dir / "agent_work" / label / "TASK.md"
        if not report_path.is_file() or not notes_path.is_file() or not task_path.is_file():
            counts["missing_artifacts"] += 1
            continue
        report = json.loads(report_path.read_text(encoding="utf-8"))
        source = _notes_for_next(_outer(previous, report), notes_path.read_text(encoding="utf-8", errors="replace"))
        if len(source.encode("utf-8")) != block["source_bytes"]:
            counts["reconstruction_mismatch"] += 1
            continue
        counts["reconstructed_exact"] += 1
        ratio = _shared_ratio(source, task_path.read_text(encoding="utf-8", errors="replace"))
        classification = "mostly_duplicate" if ratio >= 0.75 else "mixed" if ratio >= 0.25 else "mostly_not_duplicated"
        counts[classification] += 1
        rows.append({
            "run_id": str(run_dir.relative_to(runs_root)),
            "session_label": label,
            "literal_shared_ratio": ratio,
            "classification": classification,
            "source_bytes": block["source_bytes"],
            "omitted_bytes": block["omitted_bytes"],
        })
    ratios = [row["literal_shared_ratio"] for row in rows]
    return {
        "schema": SCHEMA,
        "runs_root": str(runs_root),
        "counts": dict(counts),
        "literal_shared_ratio": {
            "median": statistics.median(ratios) if ratios else None,
            "mean": statistics.fmean(ratios) if ratios else None,
            "max": max(ratios) if ratios else None,
        },
        "sessions": sorted(rows, key=lambda row: row["literal_shared_ratio"], reverse=True),
        "verdict": (
            "omitted_notes_are_not_literal_duplicates"
            if ratios and statistics.median(ratios) < 0.5 and max(ratios) < 0.75
            else "mixed_literal_duplication"
        ),
        "causal_claim_allowed": False,
        "note": (
            "source_bytes 与回执完全一致才纳入。SequenceMatcher 只衡量与当前 TASK 其他部分的"
            "字面共享，不判断语义重要性。低重复不代表内容有用，高重复也不代表可安全省略。"
        ),
    }


def render_summary(document: dict[str, Any]) -> str:
    counts = document["counts"]
    ratio = document["literal_shared_ratio"]
    lines = [
        "Omitted NOTES content audit (read-only)",
        f"  omitted receipts: {counts.get('omitted_notes_receipts', 0)}; exact reconstructions: {counts.get('reconstructed_exact', 0)}",
        f"  mismatches: {counts.get('reconstruction_mismatch', 0)}; missing artifacts: {counts.get('missing_artifacts', 0)}",
        f"  literal shared ratio median/mean/max: {ratio['median']:.4f} / {ratio['mean']:.4f} / {ratio['max']:.4f}" if ratio["median"] is not None else "  no exact sources",
        f"  mostly not duplicated: {counts.get('mostly_not_duplicated', 0)}; mixed: {counts.get('mixed', 0)}; mostly duplicate: {counts.get('mostly_duplicate', 0)}",
        f"  verdict: {document['verdict']}; causal claim allowed: {document['causal_claim_allowed']}",
    ]
    for row in document["sessions"][:5]:
        lines.append(f"    {row['run_id']} {row['session_label']}: shared {row['literal_shared_ratio']:.2%}")
    return "\n".join(lines) + "\n"


__all__ = ["SCHEMA", "audit", "render_summary"]
