"""Total LLM spend of finished runs: ``result.json`` files grouped by directory and model.

    uv run python -m experiments.cost_report runs/suite_full runs/p1 runs/repeatability
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def collect(dirs: list[Path]) -> list[dict]:
    rows = []
    for d in dirs:
        for p in sorted(Path(d).rglob("result.json")):
            try:
                r = json.loads(p.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            if "spent_usd" not in r:
                continue
            rows.append({"group": str(d), "run": str(p.parent), "task": r.get("task"), "dtype": r.get("dtype"), "model": r.get("model"),
                         "spent_usd": float(r["spent_usd"]), "budget_usd": r.get("budget_usd"), "evaluations": r.get("evaluations", 0)})
    return rows


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
    rows = collect([Path(d) for d in a.dirs])
    print(markdown(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
