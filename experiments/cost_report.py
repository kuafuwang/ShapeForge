"""Total LLM spend of finished runs: ``result.json`` files grouped by directory and model.

    uv run python -m experiments.cost_report runs/apple-m4-max/suite_full runs/apple-m4-max/p1 runs/repeatability
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path


def collect(dirs: list[Path]) -> tuple[list[dict], list[str]]:
    """Spend rows, and paths that are present but unreadable. A damaged result is not omitted as if the run never happened."""
    from taic.artifacts import RESULT_SCHEMA_VERSION, NeedsInspection, read_versioned

    rows: list[dict] = []
    inspections: list[str] = []
    for d in dirs:
        for p in sorted(Path(d).rglob("result.json")):
            try:
                r = read_versioned(p, RESULT_SCHEMA_VERSION)
            except NeedsInspection as e:
                inspections.append(str(e))
                continue
            if not isinstance(r, dict) or "spent_usd" not in r:
                continue
            rows.append({"group": str(d), "run": str(p.parent), "task": r.get("task"), "dtype": r.get("dtype"), "model": r.get("model"),
                         "spent_usd": float(r["spent_usd"]), "budget_usd": r.get("budget_usd"), "evaluations": r.get("evaluations", 0)})
    return rows, inspections


def markdown(rows: list[dict]) -> str:
    by = defaultdict(lambda: [0, 0.0, 0])
    for r in rows:
        k = (r["group"], r["model"])
        by[k][0] += 1
        by[k][1] += r["spent_usd"]
        by[k][2] += r["evaluations"] or 0
    L = ["| experiment | model | runs | evaluations | spent USD |", "| --- | --- | ---: | ---: | ---: |"]
    for (g, m), (n, usd, ev) in sorted(by.items()):
        L.append(f"| {g} | {m} | {n} | {ev} | {usd:.3f} |")
    L.append(f"| **total** | | {sum(v[0] for v in by.values())} | {sum(v[2] for v in by.values())} | **{sum(v[1] for v in by.values()):.3f}** |")
    return "\n".join(L)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("dirs", nargs="+")
    a = ap.parse_args(argv)
    rows, inspections = collect([Path(d) for d in a.dirs])
    for line in inspections:
        print(line)
    print(markdown(rows))
    return 2 if inspections else 0


if __name__ == "__main__":
    raise SystemExit(main())
