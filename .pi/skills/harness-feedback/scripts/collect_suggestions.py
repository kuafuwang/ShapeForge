#!/usr/bin/env python3
"""Read-only inventory of free-text system suggestions across TAIC runs.

Use from the repository root:

    uv run python .pi/skills/harness-feedback/scripts/collect_suggestions.py --format md

It reads feedback.jsonl and harness_* files only. It never calls a model, a GPU, or a
write path. Suggestions are grouped by their parsed digest so repeated wording across
runs is visible, while every original run/seq is retained for provenance.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from taic import harness  # noqa: E402


def _run_id(runs: Path, run: Path) -> str:
    try:
        return run.relative_to(runs).as_posix()
    except ValueError:
        return str(run)


def _reviews(run: Path, key: str) -> list[dict]:
    return [
        row for row in harness._read_jsonl(run / "harness_reviews.jsonl")
        if str(row.get("proposal_key") or "") == key
    ]


def collect(runs: Path, *, task: str = "") -> tuple[list[dict], list[dict]]:
    rows: list[dict] = []
    for feedback in sorted(runs.rglob("feedback.jsonl")):
        run = feedback.parent
        for item in harness.free_text_suggestions(run):
            if task and str(item.get("task") or "") != task:
                continue
            row = {
                **item,
                "run": _run_id(runs, run),
                "reviews": [],
            }
            if item.get("converted_key"):
                proposal = next(
                    (p for p in harness._read_jsonl(run / "harness_proposals.jsonl")
                     if p.get("proposal_key") == item["converted_key"]),
                    {},
                )
                row["proposal"] = proposal
                row["reviews"] = _reviews(run, item["converted_key"])
            rows.append(row)
    groups: dict[str, dict] = {}
    for row in rows:
        key = str(row.get("digest") or "")
        group = groups.setdefault(key, {
            "digest": key, "text": row.get("text") or "", "count": 0,
            "runs": [], "tasks": [], "models": [], "converted": set(),
            "reviewed": set(),
        })
        group["count"] += 1
        group["runs"].append(row["run"])
        if row.get("task") and row["task"] not in group["tasks"]:
            group["tasks"].append(row["task"])
        who = f"{row.get('provider')}/{row.get('model')}"
        if who not in group["models"]:
            group["models"].append(who)
        if row.get("converted_key"):
            group["converted"].add(row["converted_key"])
        for review in row.get("reviews") or []:
            group["reviewed"].add("useful" if review.get("useful") is True else "not-useful")
    ordered = sorted(groups.values(), key=lambda g: (-g["count"], g["digest"]))
    return rows, ordered


def markdown(rows: list[dict], groups: list[dict]) -> str:
    lines = [
        "# Free-text system suggestions",
        "",
        f"{len(rows)} suggestion row(s), {len(groups)} digest group(s). Read-only inventory; no review or conversion was triggered.",
        "",
        "## Groups",
        "",
        "| count | digest | tasks | converted | review | suggestion |",
        "|---:|---|---|---|---|---|",
    ]
    for group in groups:
        text = " ".join(str(group["text"]).split())
        if len(text) > 220:
            text = text[:217] + "..."
        converted = ", ".join(sorted(group["converted"])) or "—"
        reviewed = ", ".join(sorted(group["reviewed"])) or "—"
        lines.append(
            f"| {group['count']} | `{group['digest']}` | {', '.join(group['tasks']) or '—'} | "
            f"{converted} | {reviewed} | {text.replace('|', '\\|')} |"
        )
    lines += ["", "## Provenance", "", "| run | seq | task | model | digest | converted key |", "|---|---:|---|---|---|---|"]
    for row in rows:
        lines.append(
            f"| {row['run']} | {row['seq']} | {row.get('task') or '—'} | "
            f"{row.get('provider')}/{row.get('model')} | `{row.get('digest')}` | {row.get('converted_key') or '—'} |"
        )
    lines += ["", "Next: read the group text, decide whether it is a tip/premise (needs a fresh probe) or a P2 skill/task wording, then ask the human before `taic harness convert` or the dashboard conversion form."]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="runs", help="runs/ directory (default: runs)")
    parser.add_argument("--task", default="", help="only rows whose feedback task matches")
    parser.add_argument("--format", choices=["md", "json", "jsonl"], default="md")
    parser.add_argument("--out", default="", help="write to this file instead of stdout")
    args = parser.parse_args()
    runs = Path(args.root)
    if not runs.is_dir():
        print(f"not a directory: {runs}", file=sys.stderr)
        return 2
    rows, groups = collect(runs, task=args.task)
    if args.format == "md":
        output = markdown(rows, groups)
    elif args.format == "json":
        output = json.dumps({"rows": rows, "groups": groups}, ensure_ascii=False, indent=2, default=list) + "\n"
    else:
        output = "".join(json.dumps(row, ensure_ascii=False, default=list) + "\n" for row in rows)
    if args.out:
        Path(args.out).write_text(output, encoding="utf-8")
    else:
        sys.stdout.write(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
