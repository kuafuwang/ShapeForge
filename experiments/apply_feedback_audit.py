"""Apply the 2026-10-22 feedback audit to run-local review files, or show what would change.

The audit JSON is the source of verdicts. This script never deletes a claim row and never
runs a model or a GPU. ``drop_empty`` becomes ``obsolete`` with a note; the parser's fbv3
rebuild may later retire the row and leave a tombstone. The 23 same-topic rows are *not*
merged by default because a topic is not an equivalence relation (budget contains five
different actions); pass ``--merge-topic`` only after a person has checked the canonical
row for each topic.

    uv run python -m experiments.apply_feedback_audit --dry-run
    uv run python -m experiments.apply_feedback_audit --apply
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

from observe.feedback import apply_review
from taic import control

REPO = Path(__file__).resolve().parent.parent
DEFAULT_AUDIT = REPO / "docs" / "feedback_audit_20261022.json"
DEFAULT_RUNS = REPO / "runs"
# The two repairs named by the audit. A ref is a commit, not a free-form note.
FIXED_REFS = {
    "self_check_dir": "commit:1fd881d",
    "relative_path": "commit:f06f019",
    "seed_report": "commit:f06f019",
}


def _body(row: dict, *, duplicate_of: str = "") -> dict:
    verdict = row.get("verdict") or ""
    topic = row.get("topic") or ""
    evidence = f"{int(row.get('measured_valid') or 0)} measured lines"
    common = {"topic": topic, "note": f"audit 2026-10-22: {verdict}"}
    if duplicate_of:
        return {**common, "status": "obsolete", "verification": "unchecked", "note": f"audit 2026-10-22: duplicate of {duplicate_of}"}
    if verdict == "valid_measured":
        return {**common, "status": "adopted", "verification": "confirmed", "evidence_source": "probe", "evidence": evidence}
    if verdict == "fixed_in_code":
        return {
            **common, "status": "done", "verification": "confirmed",
            "change_ref": FIXED_REFS.get(topic, "commit:unknown"),
            "evidence_source": "code", "evidence": "fixed in the current tree",
        }
    if verdict == "drop_empty":
        return {**common, "status": "obsolete", "verification": "unchecked", "note": "audit 2026-10-22: empty placeholder"}
    if verdict == "open_unverified":
        return {**common, "status": "info", "verification": "unchecked", "note": "audit 2026-10-22: explicitly unverified; open question, not evidence"}
    if verdict == "missing_probe_at_capture":
        # Keep these pending: fbv3 + the restored probes may upgrade some to verified.
        return {**common, "status": "pending", "verification": "unchecked", "note": "audit 2026-10-22: probe missing at capture; reparse after restore"}
    if verdict == "policy_decision":
        return {**common, "status": "pending", "verification": "unchecked", "note": "audit 2026-10-22: policy decision for a person; not for model voting"}
    if verdict == "actionable_open":
        return {**common, "status": "pending", "verification": "unchecked", "note": "audit 2026-10-22: actionable open review"}
    return {**common, "status": "pending", "verification": "unchecked"}


def actions(audit: dict, *, merge_topic: bool = False) -> list[dict]:
    rows = audit.get("rows") if isinstance(audit, dict) else None
    if not isinstance(rows, list):
        raise ValueError("audit JSON has no rows list")
    by_topic: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        if not isinstance(row, dict) or not row.get("id"):
            continue
        by_topic[(str(row.get("task") or ""), str(row.get("topic") or ""))].append(row)
    duplicate_of: dict[str, str] = {}
    if merge_topic:
        # Only topics with more than one non-empty row. The first row is the canonical one;
        # this is intentionally opt-in, not the default audit write-back.
        for (_task, _topic), group in by_topic.items():
            live = [
                row for row in group
                if row.get("section") in ("对系统的建议", "需要帮助")
                and row.get("verdict") not in ("drop_empty", "fixed_in_code")
            ]
            if len(live) <= 1:
                continue
            canonical = live[0]["id"]
            for row in live[1:]:
                duplicate_of[str(row["id"])] = canonical
    out = []
    for row in rows:
        if not isinstance(row, dict) or not row.get("id"):
            continue
        body = _body(row, duplicate_of=duplicate_of.get(str(row["id"]), ""))
        run_id, _, claim = str(row["id"]).partition("#")
        seq_text, _, digest = claim.partition(":")
        try:
            seq = int(seq_text)
        except ValueError:
            continue
        out.append({"run_id": run_id, "seq": seq, "digest": digest, "body": body, "row": row})
    return out


def _run_dir(root: Path, run_id: str) -> Path:
    return Path(root) / run_id


def apply_actions(root: Path, rows: list[dict], *, apply: bool) -> dict:
    written = errors = 0
    touched_runs: set[str] = set()
    details = []
    for action in rows:
        directory = _run_dir(root, action["run_id"])
        if not (directory / "feedback.jsonl").is_file():
            errors += 1
            details.append({"id": action["row"]["id"], "error": "run directory or feedback.jsonl is missing"})
            continue
        if not apply:
            details.append({"id": action["row"]["id"], "body": action["body"]})
            continue
        try:
            control.acquire_run_lock(directory, who="feedback-audit")
            try:
                apply_review(directory, action["seq"], action["digest"], action["body"])
            finally:
                control.release_run_lock(directory)
        except (OSError, ValueError, control.RunLockError) as exc:
            errors += 1
            details.append({"id": action["row"]["id"], "error": str(exc)})
            continue
        written += 1
        touched_runs.add(action["run_id"])
    return {"mode": "apply" if apply else "dry-run", "rows": len(rows), "written": written, "errors": errors, "runs": sorted(touched_runs), "details": details}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="experiments.apply_feedback_audit")
    parser.add_argument("--audit", default=str(DEFAULT_AUDIT))
    parser.add_argument("--runs", default=str(DEFAULT_RUNS))
    parser.add_argument("--apply", action="store_true", help="write feedback_review.json; without it only print the plan")
    parser.add_argument("--dry-run", action="store_true", help="explicitly print the plan (the default)")
    parser.add_argument("--merge-topic", action="store_true", help="also mark every non-first live row in a topic obsolete (review the canonical row first)")
    args = parser.parse_args(argv)
    audit = json.loads(Path(args.audit).read_text(encoding="utf-8"))
    rows = actions(audit, merge_topic=args.merge_topic)
    result = apply_actions(Path(args.runs), rows, apply=args.apply)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["errors"] == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
