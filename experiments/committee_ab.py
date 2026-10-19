"""Committee comparison matrix. The default is a dry run: it prints the arms and the nominal dollars and starts nothing.

    uv run python -m experiments.committee_ab --dry-run --n 3 --budget 3
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .manifest import ensure


ARMS = (
    {"id": "A", "committees": "first and second both deepseek-flash", "plan_effort": None, "patch_effort": None},
    {"id": "B", "committees": "first and second both deepseek-flash", "plan_effort": "high", "patch_effort": "low"},
    {"id": "C", "committees": "first high, second low", "plan_effort": None, "patch_effort": None},
)


def nominal_usd(n: int, budget: float) -> float:
    return n * len(ARMS) * budget


def declare(out: Path, *, n: int, budget: float, max_total_usd: float | None) -> dict:
    """Write the manifest for a committee comparison. This does not start a search."""
    return ensure(
        Path(out),
        kind="committee_ab",
        driver="committee_ab",
        frozen={
            "trials_per_arm": n,
            "budget_usd": budget,
            "max_total_usd": max_total_usd,
            "task": None,
            "dtype": None,
            "size": None,
            "preset": None,
            "model": None,
            "committees": None,
        },
        arms=[
            {
                "arm_id": arm["id"],
                "planned_n": n,
                "delta": {
                    "committees": arm["committees"],
                    "plan_effort": arm["plan_effort"],
                    "patch_effort": arm["patch_effort"],
                },
            }
            for arm in ARMS
        ],
    )


def matrix(*, n: int, budget: float, max_total_usd: float | None) -> dict:
    total = nominal_usd(n, budget)
    return {
        "dry_run": True,
        "arms": list(ARMS),
        "trials_per_arm": n,
        "budget_usd": budget,
        "nominal_usd": total,
        "fits": max_total_usd is None or total <= max_total_usd,
        "note": "This driver does not call a model. A paid run needs a separate approval.",
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", default=True)
    ap.add_argument("--execute", action="store_true", help="refused: a paid committee comparison needs a separate approval")
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--budget", type=float, default=3.0)
    ap.add_argument("--max-total-usd", type=float, default=None)
    ap.add_argument("--out", default=None, help="write experiment.json here and still do not start a search")
    a = ap.parse_args(argv)
    if a.execute:
        print("refused: experiments.committee_ab does not start a paid search")
        return 2
    if a.out:
        declare(Path(a.out), n=a.n, budget=a.budget, max_total_usd=a.max_total_usd)
    print(json.dumps(matrix(n=a.n, budget=a.budget, max_total_usd=a.max_total_usd), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
