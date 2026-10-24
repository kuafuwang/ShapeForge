"""Read-only adoption stages, cost accounting and shortlisting for harness proposals."""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path

from .harness_experiment import milestones


def _rows(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    out = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            out.append(row)
    return out


def _number(value) -> float | None:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0 else None


def _sum(rows: list[dict], key: str = "cost_usd") -> float | None:
    if not rows:
        return 0.0
    values = [_number(row.get(key)) for row in rows]
    return sum(values) if all(value is not None for value in values) else None


def _stage_times(proposal: dict, reviews: list[dict], decisions: list[dict], first_t: float) -> dict:
    """First transition time for each adoption stage, relative to the first useful proposal."""
    checks = {
        "reviewed": lambda row: row.get("event") == "review" and row.get("useful") is True,
        "implemented": lambda row: row.get("event") == "implementation" and row.get("accepted") is True and row.get("host_verified") is True,
        "experiment_validated": lambda row: row.get("event") == "effect_validation" and row.get("validated") is True and bool(row.get("approved_by")),
        "local": lambda row: row.get("event") in {"promotion", "code_promotion"} and row.get("status") == "local",
        "committed": lambda row: row.get("event") == "promotion_commit",
    }
    rows = [proposal, *reviews, *decisions]
    out = {}
    for stage, predicate in checks.items():
        times = [float(row["t"]) for row in rows if predicate(row) and isinstance(row.get("t"), (int, float))]
        out[stage] = max(0.0, min(times) - first_t) if times else None
    return out


def proposal_costs(run_dir: Path) -> list[dict]:
    """Costs are attributed to a proposal; unknown implementation bills are not zero."""
    run_dir = Path(run_dir)
    proposals = [row for row in _rows(run_dir / "harness_proposals.jsonl") if row.get("status") == "valid"]
    reviews = _rows(run_dir / "harness_reviews.jsonl")
    decisions = _rows(run_dir / "harness_decisions.jsonl")
    implementations = []
    for path in (run_dir / "code_promotions").glob("*.json"):
        if path.name.endswith(".implementation.json"):
            continue
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(row, dict):
                implementations.append(row)
        except (OSError, json.JSONDecodeError):
            continue
    experiments = [row for row in _rows(run_dir / "harness_decisions.jsonl") if row.get("event") == "effect_validation"]
    out = []
    for proposal in proposals:
        key = str(proposal.get("proposal_key") or "")
        related_reviews = [row for row in reviews if row.get("proposal_key") == key]
        related_impl = [row for row in implementations if row.get("proposal_key") == key]
        related_experiments = [row for row in experiments if row.get("proposal_key") == key]
        review_cost = _sum(related_reviews)
        implementation_cost = _sum(related_impl)
        experiment_cost = _sum(related_experiments, "spent_usd")
        if experiment_cost is None:
            fallback = [row for row in decisions if row.get("proposal_key") == key and row.get("event") == "experiment_result"]
            experiment_cost = _sum(fallback, "spent_usd") if fallback else None
        if related_impl and any(row.get("cost_usd") is None for row in related_impl):
            implementation_cost = None
        known = [value for value in (review_cost, implementation_cost, experiment_cost) if value is not None]
        stage = milestones(decisions + reviews, key)
        first_t = min([float(row["t"]) for row in [proposal, *related_reviews] if isinstance(row.get("t"), (int, float))] or [0.0])
        latest_t = max([float(row["t"]) for row in [proposal, *related_reviews, *decisions] if isinstance(row.get("t"), (int, float))] or [first_t])
        stage_times = _stage_times(proposal, reviews, decisions, first_t)
        evidence = (proposal.get("proposal") or {}).get("evidence") if isinstance((proposal.get("proposal") or {}).get("evidence"), dict) else {}
        out.append({
            "proposal_key": key, "kind": (proposal.get("proposal") or {}).get("kind") or (proposal.get("proposal") or {}).get("surface"),
            "has_reproduction": bool(evidence.get("reproduction")), "has_expected": bool(evidence.get("expected")),
            "action": (proposal.get("proposal") or {}).get("action") or "", "milestones": stage,
            "review_cost_usd": review_cost, "implementation_cost_usd": implementation_cost, "experiment_cost_usd": experiment_cost,
            "known_adoption_cost_usd": sum(known) if known else 0.0,
            "cost_unknown": (bool(related_impl) and implementation_cost is None)
            or (bool(related_reviews) and review_cost is None)
            or (bool(related_experiments) and experiment_cost is None),
            "elapsed_s": max(0.0, latest_t - first_t),
            "stage_times_s": stage_times,
        })
    return out


def adoption_summary(run_dir: Path) -> dict:
    rows = proposal_costs(run_dir)
    stages = Counter()
    for row in rows:
        for stage, passed in row["milestones"].items():
            if passed:
                stages[stage] += 1
    total = len(rows)
    return {
        "run_dir": str(run_dir), "proposals": total, "stages": dict(stages),
        "useful_rate": stages["reviewed"] / total if total else None,
        "implementation_rate": stages["implemented"] / total if total else None,
        "validation_rate": stages["experiment_validated"] / total if total else None,
        "adoption_rate": stages["committed"] / total if total else None,
        "known_adoption_cost_usd": sum(row["known_adoption_cost_usd"] for row in rows),
        "cost_unknown_proposals": sum(row["cost_unknown"] for row in rows),
        "items": rows,
    }


def shortlist(runs: list[Path], *, limit: int = 10) -> list[dict]:
    """Select complete, distinct tool/reporting/wording suggestions; never implement them."""
    groups = {}
    for run in runs:
        for row in proposal_costs(run):
            key = row["action"].strip().lower()
            if not key or row["kind"] == "hardware_premise":
                continue
            evidence_score = int(bool(row["action"])) + int(row["milestones"]["reviewed"]) + int(row["milestones"]["implemented"]) + int(row["milestones"]["experiment_validated"])
            item = {**row, "run_dir": str(run), "evidence_score": evidence_score, "evidence_gaps": [stage for stage, passed in row["milestones"].items() if not passed] + ([] if row["has_reproduction"] else ["reproduction"]) + ([] if row["has_expected"] else ["expected"])}
            groups.setdefault(key, []).append(item)
    candidates = []
    for rows in groups.values():
        rows.sort(key=lambda row: (-row["evidence_score"], row["known_adoption_cost_usd"], row["elapsed_s"]))
        best = dict(rows[0])
        best["duplicate_count"] = len(rows)
        candidates.append(best)
    candidates.sort(key=lambda row: (-row["evidence_score"], row["known_adoption_cost_usd"], row["proposal_key"]))
    return candidates[:limit]


def append_adoption_record(run_dir: Path) -> dict:
    summary = adoption_summary(run_dir)
    row = {"schema_version": 1, "event": "adoption_review", "run_dir": str(run_dir), "summary": summary}
    path = run_dir / "harness_adoption.jsonl"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return row
