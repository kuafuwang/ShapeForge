"""Read-only N0/N1 metrics over existing runs.

Candidate copies of ``paired_host_selection.json`` are not extra sessions.
``gpu_seconds`` comes only from that selection record; a plain paired comparison
has only ``elapsed_s`` and must not relabel it as GPU seconds. Rechecks are
reported separately from natural run observations. Missing coverage is
unavailable, never zero.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

SCHEMA = "shapeforge.n1-metrics-audit.1"


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(float(value)) else None


def _load(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _run_id(run: Path, root: Path) -> str:
    try:
        return str(run.relative_to(root))
    except ValueError:
        return str(run)


def _selection_documents(runs_root: Path) -> tuple[list[tuple[Path, Path, dict[str, Any]]], int]:
    """@brief 每个 (run, session) 只取 session 正本；candidate 副本只作缺失时的退路。"""
    grouped: dict[tuple[Path, str], dict[str, list[Path]]] = {}
    for path in sorted(runs_root.rglob("paired_host_selection.json")):
        document = _load(path)
        label = str(document.get("session_label") or path.parent.name)
        run = path.parent.parent.parent if path.parent.parent.name in {"sessions", "candidates"} else path.parent.parent
        grouped.setdefault((run, label), {"session": [], "candidate": []})
        kind = "session" if path.parent.parent.name == "sessions" else "candidate"
        grouped[(run, label)][kind].append(path)
    selected = []
    duplicates = 0
    for (run, _label), paths in sorted(grouped.items(), key=lambda item: str(item[0][0])):
        canonical = paths["session"][0] if paths["session"] else paths["candidate"][0]
        selected.append((run, canonical, _load(canonical)))
        duplicates += len(paths["session"]) + len(paths["candidate"]) - 1
    return selected, duplicates


def _n0_rows(runs_root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = []
    groups: dict[str, Counter] = {}
    known_cost = 0.0
    known_cost_runs = 0
    for path in sorted(runs_root.rglob("result.json")):
        result = _load(path)
        if not result.get("trace") and not result.get("task"):
            continue
        run = path.parent
        config = _load(run / "run_config.json")
        target = result.get("target_id") or (config.get("target") or {}).get("id")
        backend = result.get("backend") or (config.get("target") or {}).get("backend")
        model = result.get("model") or (config.get("llm") or {}).get("model")
        trace = [row for row in (result.get("trace") or []) if isinstance(row, dict)]
        first_ok = next((row for row in trace if row.get("status") == "ok"), None)
        first_model_ok = next((row for row in trace if row.get("status") == "ok" and row.get("label") != "prior"), None)
        confirmed = _number(result.get("confirmed_speedup"))
        best = _number(result.get("best_speedup"))
        spent = _number(result.get("spent_usd"))
        if spent is not None:
            known_cost += spent
            known_cost_runs += 1
        row = {
            "run_id": _run_id(run, runs_root),
            "target": target,
            "backend": backend,
            "task": result.get("task") or config.get("task"),
            "dtype": result.get("dtype") or config.get("dtype"),
            "size": result.get("size") or config.get("size"),
            "model": model,
            "first_ok_label": first_ok.get("label") if first_ok else None,
            "first_model_ok_label": first_model_ok.get("label") if first_model_ok else None,
            "first_model_ok_round": first_model_ok.get("idx") if first_model_ok else None,
            "search_best_speedup": best,
            "confirmed_speedup": confirmed,
            "confirmed_status": result.get("confirmed_status"),
            "spent_usd": spent,
        }
        rows.append(row)
        key = "/".join(str(row.get(name) or "unknown") for name in ("target", "backend", "task", "dtype", "size", "model"))
        counts = groups.setdefault(key, Counter())
        counts["runs"] += 1
        counts["with_first_model_ok"] += first_model_ok is not None
        counts["with_confirmed_ok"] += confirmed is not None and result.get("confirmed_status") == "ok"
        counts["with_known_cost"] += spent is not None
    group_docs = [
        {"group": key, **dict(counts), "note": "不同 task/shape/model 不合并 speedup；这里只给覆盖计数"}
        for key, counts in sorted(groups.items())
    ]
    summary = {
        "runs": len(rows),
        "with_first_model_ok": sum(row["first_model_ok_label"] is not None for row in rows),
        "with_confirmed_ok": sum(row["confirmed_speedup"] is not None and row["confirmed_status"] == "ok" for row in rows),
        "with_known_cost": known_cost_runs,
        "known_ledger_cost_usd": known_cost if known_cost_runs else None,
        "cost_note": "仅是 result.json 已记录的账本成本；缺失不补 0，也不是 provider 对账单",
    }
    return rows, {"summary": summary, "groups": group_docs}


def audit(runs_root: Path) -> dict[str, Any]:
    """@brief 计算 N0/N1 的现值与覆盖，不修改任何 run。

    @param runs_root run 根目录。
    @return 配对、K=1 提名覆盖、重测分层和 N0 分组指标。
    """
    runs_root = Path(runs_root)
    pair_counts = Counter()
    pair_elapsed = 0.0
    pair_runs = set()
    for path in sorted(runs_root.rglob("paired_comparison.json")):
        document = _load(path)
        run = path.parent.parent.parent
        config = _load(run / "run_config.json")
        applied = document.get("applied") is True
        enabled = (config.get("search") or {}).get("paired_seed_observation") is True
        pair_counts["files"] += 1
        pair_counts["applied" if applied else "not_applied"] += 1
        pair_counts["paired_observation_enabled" if enabled else "paired_observation_disabled_but_pair_present"] += 1
        pair_counts[f"conclusion:{document.get('conclusion') or 'unavailable'}"] += 1
        pair_counts[f"timed_by:{document.get('timed_by') or 'unavailable'}"] += 1
        if applied:
            pair_runs.add(run)
            elapsed = _number(document.get("elapsed_s"))
            if elapsed is not None:
                pair_counts["with_elapsed_s"] += 1
                pair_elapsed += elapsed
    selections, duplicate_copies = _selection_documents(runs_root)
    selection_counts = Counter()
    gpu_seconds = 0.0
    selection_rows = []
    for run, path, document in selections:
        nominations = int(document.get("nominations") or 0)
        eligible = int(document.get("unique_eligible") or 0)
        reviewed = int(document.get("reviewed") or 0)
        uncovered = int(document.get("uncovered_count") or 0)
        selection_counts["sessions"] += 1
        selection_counts["nominations"] += nominations
        selection_counts["unique_eligible"] += eligible
        selection_counts["reviewed"] += reviewed
        selection_counts["uncovered"] += uncovered
        for item in document.get("uncovered") or []:
            conclusion = str(item.get("conclusion") or "")
            if conclusion in {"no_diff", "regressed"}:
                selection_counts["uncovered_no_diff_or_regressed"] += 1
            if item.get("omit_reason") == "over_k":
                selection_counts["over_k"] += 1
        ran_gpu = document.get("ran_gpu") is True
        selection_counts["ran_gpu_sessions"] += ran_gpu
        gpu_value = _number(document.get("gpu_seconds"))
        if ran_gpu and gpu_value is not None:
            selection_counts["gpu_seconds_known_sessions"] += 1
            gpu_seconds += gpu_value
        elif ran_gpu:
            selection_counts["gpu_seconds_unavailable_sessions"] += 1
        conclusions = Counter(str(item.get("conclusion") or "unavailable") for item in document.get("measurements") or [])
        for conclusion, count in conclusions.items():
            selection_counts[f"conclusion:{conclusion}"] += count
        selection_rows.append({
            "run_id": _run_id(run, runs_root),
            "session_label": document.get("session_label") or path.parent.name,
            "nominations": nominations,
            "unique_eligible": eligible,
            "reviewed": reviewed,
            "uncovered_count": uncovered,
            "ran_gpu": ran_gpu,
            "gpu_seconds": gpu_value if ran_gpu else None,
            "conclusions": dict(conclusions),
        })
    rechecks = list(runs_root.rglob("paired_comparison_recheck.json"))
    n0_rows, n0 = _n0_rows(runs_root)
    eligible = selection_counts["unique_eligible"]
    reviewed = selection_counts["reviewed"]
    return {
        "schema": SCHEMA,
        "runs_root": str(runs_root),
        "natural_paired_comparisons": {
            "counts": dict(pair_counts),
            "runs_with_applied": len(pair_runs),
            "elapsed_s_sum": pair_elapsed if pair_counts["with_elapsed_s"] else None,
            "elapsed_note": "elapsed_s 是配对命令总耗时，不是纯 GPU 秒；普通 paired_comparison 没有独立 gpu_seconds 字段",
            "gpu_seconds": None,
            "coverage_note": "没有 paired_comparison 的 run 是覆盖不可用，不按 0 次自然配对解释；paired_seed_observation=false 但存在文件时单列为 follow-up/legacy",
        },
        "nomination_selection": {
            "counts": dict(selection_counts),
            "candidate_duplicate_copies_ignored": duplicate_copies,
            "coverage": "available" if selection_counts["sessions"] else "unavailable",
            "review_coverage": reviewed / eligible if eligible else None,
            "gpu_seconds": (
                None
                if not selection_counts["sessions"]
                or (selection_counts["ran_gpu_sessions"] and not selection_counts["gpu_seconds_known_sessions"])
                else gpu_seconds
            ),
            "gpu_seconds_complete": selection_counts["gpu_seconds_unavailable_sessions"] == 0,
            "gpu_seconds_note": "仅汇总 ran_gpu=true 且带 gpu_seconds 的选择正本；没有选择的 run 不补 0",
            "sessions": selection_rows,
            "k_change_recommended": False,
            "k_note": "K=1 的 over_k 与未覆盖结论需要更多自然 run；当前数据不支持升 K，也不证明 K=1 永远足够",
        },
        "rechecks": {
            "files": len(rechecks),
            "note": "精确 SHA 重测是后续实验，不与自然 run 的配对次数混合",
        },
        "n0": {"runs": n0_rows, **n0},
        "verdict": "collect_more_natural_runs",
        "publication_allowed": False,
    }


def render_summary(document: dict[str, Any]) -> str:
    pairs = document["natural_paired_comparisons"]
    selection = document["nomination_selection"]
    counts = selection["counts"]
    n0 = document["n0"]["summary"]
    return "\n".join([
        "N0/N1 metrics audit (read-only)",
        f"  paired files: {pairs['counts'].get('files', 0)}; applied: {pairs['counts'].get('applied', 0)}; runs with applied: {pairs['runs_with_applied']}",
        f"  paired elapsed_s: {pairs['elapsed_s_sum']}; pure GPU seconds: unavailable",
        f"  K=1 sessions: {counts.get('sessions', 0)}; nominations/eligible/reviewed/uncovered: {counts.get('nominations', 0)}/{counts.get('unique_eligible', 0)}/{counts.get('reviewed', 0)}/{counts.get('uncovered', 0)}",
        f"  selection GPU seconds: {selection['gpu_seconds']}; review coverage: {selection['review_coverage']}",
        f"  duplicate candidate copies ignored: {selection['candidate_duplicate_copies_ignored']}; K change recommended: {selection['k_change_recommended']}",
        f"  N0 runs: {n0['runs']}; first model ok: {n0['with_first_model_ok']}; confirmed ok: {n0['with_confirmed_ok']}; known cost runs: {n0['with_known_cost']}",
        f"  verdict: {document['verdict']}",
    ]) + "\n"


__all__ = ["SCHEMA", "audit", "render_summary"]
