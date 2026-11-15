"""P2-W approved execution for the frozen P2-V compact-overlay plan.

P2-V deliberately refuses execution and remains immutable.  This separate entry point
requires a new, explicit approval record and then applies the same matrix, overlay,
commitment-cap and runner bindings as search_yield_ab.execute_plan.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Callable

from experiments.search_yield_ab import (
    _experience_run_seed,
    _occupied,
    _require_experience_plan,
    write_plan,
)

REPO = Path(__file__).resolve().parents[1]
FROZEN = REPO / "tests/support/golden/shared/acceptance-epochs/fixtures/p2v/evidence/dry-run-plan.json"
FROZEN_SHA256 = "d3781f05f0a5647cdb58b57582c83cbd1a6e567275e562f36bb24325c9e5525e"
OVERLAY_SHA256 = "fd72abf078d9d5a7cd2118ee5263a3c83e5695c044c0bc0cc6ca86758a0aa3d1"
EXPECTED_HEAD = "d3388497aabaf2a687f3f1d65801c87edd9fdd9c"
APPROVAL_SCHEMA = "shapeforge.p2w-compact-search-approval.1"


class CompactExecutionError(ValueError):
    pass


def approved_plan(*, approved_by: str, delta: float) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return the frozen inner plan and its separate P2-W approval, without executing."""
    raw = FROZEN.read_bytes()
    if hashlib.sha256(raw).hexdigest() != FROZEN_SHA256:
        raise CompactExecutionError("P2-V frozen plan digest changed")
    outer = json.loads(raw)
    if outer.get("approved_by") is not None or outer.get("delta") is not None or outer.get("executed") is not False:
        raise CompactExecutionError("P2-V plan is no longer the unapproved dry-run")
    if outer.get("inherits_p2u_approval") is not False or outer.get("compact_overlay_sha256") != OVERLAY_SHA256:
        raise CompactExecutionError("P2-V approval boundary or compact overlay changed")
    name = str(approved_by).strip()
    if not name or name.casefold() in {"cli", "agent", "auto"}:
        raise CompactExecutionError("a real approved-by name is required")
    if isinstance(delta, bool) or not isinstance(delta, (int, float)) or not math.isfinite(delta) or delta <= 0:
        raise CompactExecutionError("delta must be positive and finite")
    plan = copy.deepcopy(outer["experience_plan"])
    from experiments.experience_compact import COMPACT_REL
    plan["overlay_path"] = COMPACT_REL
    if plan.get("git_head") != EXPECTED_HEAD or plan.get("max_total_usd") != 12.0 or plan.get("budget_usd") != 3.0:
        raise CompactExecutionError("P2-W no longer matches the approved HEAD or budget")
    approval = {
        "schema_id": APPROVAL_SCHEMA,
        "experiment_id": "p2w-compact-overlay-paid-20261115",
        "approved_by": name,
        "approval_source": "user approved the separate compact-overlay experiment after P2-V review",
        "inherits_p2u_approval": False,
        "delta": float(delta),
        "allowed_regression": 0.0,
        "per_cell_soft_budget_usd": 3.0,
        "start_next_cell_commitment_cap_usd": 12.0,
        "tolerable_overrun": "in-flight requests only; no additional cells and no previous-cell top-ups",
        "base_plan_sha256": FROZEN_SHA256,
        "compact_overlay_sha256": OVERLAY_SHA256,
        "parent_head": EXPECTED_HEAD,
        "publication_allowed": False,
    }
    plan["compact_execution_approval"] = approval
    _require_experience_plan(plan)
    return plan, approval


def execute_approved(
    out: Path,
    plan: dict[str, Any],
    *,
    allow_spend: bool,
    allow_gpu: bool,
    runner: Callable[..., Any] | None = None,
) -> list[dict[str, Any]]:
    """Execute the four frozen cells, preserving the original cap and overlay bindings."""
    if not allow_spend or not allow_gpu:
        raise CompactExecutionError("both allow-spend and allow-gpu are required")
    approval = plan.get("compact_execution_approval")
    if not isinstance(approval, dict) or approval.get("schema_id") != APPROVAL_SCHEMA:
        raise CompactExecutionError("the separate P2-W approval is missing")
    if approval.get("inherits_p2u_approval") is not False or approval.get("compact_overlay_sha256") != OVERLAY_SHA256:
        raise CompactExecutionError("invalid compact approval boundary")
    _require_experience_plan(plan)
    from experiments.common import experiment_commitment, fits_cap, run_one, write_cap

    destination = out.resolve()
    if not destination.is_relative_to((REPO / "runs").resolve()):
        raise CompactExecutionError("output must be under runs/")
    write_plan(destination, plan)
    (destination / "approval.json").write_text(json.dumps(approval, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    budget = float(plan["budget_usd"])
    cap = float(plan["max_total_usd"])
    if budget != 3.0 or cap != 12.0:
        raise CompactExecutionError("P2-W budget is frozen at $3 per cell and a $12 commitment cap")
    write_cap(destination, cap)
    go = runner or run_one
    notes: list[dict[str, Any]] = []
    for cell in plan["cells"]:
        cell_dir = destination / cell["directory"]
        if _occupied(cell_dir):
            notes.append({"directory": cell["directory"], "skipped": "occupied"})
            continue
        if not fits_cap(experiment_commitment(destination), budget, cap):
            notes.append({"directory": cell["directory"], "skipped": "over_cap"})
            break
        extra = dict(cell["overrides"])
        extra["seed"] = _experience_run_seed(plan)
        if cell["arm"] == "treatment":
            extra["experience_overlay_path"] = plan["overlay_path"]
            extra["experience_overlay_sha256"] = plan["overlay_sha256"]
        go(cell["task"], cell["dtype"], cell["size"], cell_dir, budget=budget, **extra)
        notes.append({"directory": cell["directory"], "skipped": None})
        if experiment_commitment(destination) + 1e-9 >= cap:
            break
    return notes


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Execute the separately approved P2-W compact-overlay experiment")
    parser.add_argument("--out", required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--allow-spend", action="store_true")
    parser.add_argument("--allow-gpu", action="store_true")
    parser.add_argument("--approved-by", default="")
    parser.add_argument("--delta", type=float, default=None)
    args = parser.parse_args(argv)
    try:
        plan, approval = approved_plan(approved_by=args.approved_by, delta=args.delta)
        if not args.execute:
            print(json.dumps({"dry_run": True, "cells": len(plan["cells"]), "approval": approval}, indent=2))
            return 0
        notes = execute_approved(Path(args.out), plan, allow_spend=args.allow_spend, allow_gpu=args.allow_gpu)
        print(json.dumps({"executed": notes}, indent=2))
        return 0
    except (CompactExecutionError, ValueError) as exc:
        print(f"compact_execution: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
