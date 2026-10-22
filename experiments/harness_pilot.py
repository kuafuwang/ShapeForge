"""Summarize a shadow/active harness pilot without calling a model or a GPU.

The pilot itself is a person's decision. This script only reads the run files and reports
what happened: placeholder items, structured proposals, quorum decisions, shadow/active
overlay entries, unchanged sessions, and spend. Use it before deciding to turn off shadow.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def _jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def summarize(run_dir: Path) -> dict:
    from taic.agent_feedback import load_records

    run_dir = Path(run_dir)
    records = load_records(run_dir / "feedback.jsonl")
    proposals = _jsonl(run_dir / "harness_proposals.jsonl")
    decisions = _jsonl(run_dir / "harness_decisions.jsonl")
    overlay = _jsonl(run_dir / "harness_overlay.jsonl")
    result = {}
    try:
        result = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        pass
    outcomes = Counter(str(row.get("outcome") or row.get("exit") or "unknown") for row in records)
    proposal_status = Counter(str(row.get("status") or "unknown") for row in proposals)
    surfaces = Counter(str((row.get("proposal") or {}).get("surface") or "unknown") for row in proposals if row.get("status") == "valid")
    decision_kind = Counter(str(row.get("decision") or row.get("event") or "unknown") for row in decisions)
    overlay_kind = Counter("shadow" if row.get("shadow") else "active" for row in overlay if row.get("event") == "overlay")
    valid = [row for row in proposals if row.get("status") == "valid"]
    keys = {str(row.get("proposal_key")) for row in valid}
    quorum_keys = {str(row.get("proposal_key")) for row in decisions if row.get("decision") in ("shadow", "apply")}
    placeholders = sum(int(row.get("placeholder_items") or 0) for row in records)
    return {
        "run": str(run_dir),
        "sessions": len(records),
        "outcomes": dict(outcomes),
        "unchanged_rate": (outcomes.get("unchanged", 0) / len(records)) if records else None,
        "placeholder_items": placeholders,
        "proposals": len(proposals),
        "proposal_status": dict(proposal_status),
        "proposal_surfaces": dict(surfaces),
        "distinct_proposal_keys": len(keys),
        "quorum_or_decision_keys": len(quorum_keys),
        "decisions": dict(decision_kind),
        "overlay": dict(overlay_kind),
        "spent_usd": result.get("spent_usd"),
        "best_speedup": result.get("best_speedup"),
        "confirmed_speedup": result.get("confirmed_speedup"),
        "stop_reason": result.get("stop_reason"),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="experiments.harness_pilot")
    parser.add_argument("runs", nargs="+", help="run directories; no model call and no GPU")
    parser.add_argument("--out", default=None, help="write the JSON report here")
    parser.add_argument("--markdown", default=None, help="also write a short Markdown table")
    args = parser.parse_args(argv)
    rows = [summarize(Path(path)) for path in args.runs]
    doc = {"schema_version": 1, "runs": rows}
    text = json.dumps(doc, ensure_ascii=False, indent=2, default=str)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    print(text)
    if args.markdown:
        lines = [
            "# Harness pilot",
            "",
            "| run | sessions | placeholder | proposals | quorum keys | shadow | active | unchanged | spent |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for row in rows:
            lines.append(
                f"| `{row['run']}` | {row['sessions']} | {row['placeholder_items']} | {row['proposals']} | "
                f"{row['quorum_or_decision_keys']} | {row['overlay'].get('shadow', 0)} | {row['overlay'].get('active', 0)} | "
                f"{row['unchanged_rate'] if row['unchanged_rate'] is not None else '-'} | {row['spent_usd'] if row['spent_usd'] is not None else '-'} |"
            )
        lines += [
            "",
            "A shadow entry is not injected. Only `active` entries reached TASK.md. A person must review every active entry before turning shadow off.",
        ]
        Path(args.markdown).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0 if rows else 2


if __name__ == "__main__":
    sys.exit(main())
