"""Archive the scratch of finished sessions so ``agent_work`` stops growing without bound.

The search reads only NOTES.md, FEEDBACK.md, self_checks.jsonl, TASK.md, kernel.metal, the
submission file and the guidance copies. Everything else (probe scripts, ``v/`` ablation
sources) goes to ``<run>/scratch/<label>.tar.gz`` and is removed from the work directory.
This script applies the same rule to runs that already exist.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from taic.kernel_agent import _WORK_KEEP, prune_work


def _entries(work: Path) -> list[Path]:
    return [path for path in work.iterdir() if path.name not in _WORK_KEEP]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m experiments.prune_agent_work")
    parser.add_argument("--runs", default="runs", help="root that contains run directories")
    parser.add_argument("--apply", action="store_true", help="archive and delete; without it, only report")
    args = parser.parse_args(argv)
    root = Path(args.runs)
    rows = []
    for work in sorted(root.rglob("*/agent_work/*")):
        if not work.is_dir():
            continue
        entries = _entries(work)
        if not entries:
            continue
        size = sum(path.stat().st_size if path.is_file() else sum(p.stat().st_size for p in path.rglob("*") if p.is_file()) for path in entries)
        row = {"work": str(work), "bytes": size, "entries": len(entries)}
        if args.apply:
            row.update(prune_work(work, label=work.name))
        rows.append(row)
    total = sum(row["bytes"] for row in rows)
    print(json.dumps({"mode": "apply" if args.apply else "dry-run", "sessions": len(rows), "bytes": total, "rows": rows}, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
