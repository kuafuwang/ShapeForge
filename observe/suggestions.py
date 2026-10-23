"""Compact, token-conscious retrieval of free-text system suggestions.

The SQLite index names claims but does not store their text. This module first groups the
index rows by digest and reads only one representative feedback line per group, returning a
short preview. The full sentence is fetched only for the selected (run, seq, digest).
"""

from __future__ import annotations

import json
from pathlib import Path

from observe import detail as detail_mod
from observe import feedback
from observe.index import borrow, get_index

DEFAULT_PREVIEW = 120
MAX_PREVIEW = 500
MAX_GROUPS = 200


def _record(directory: Path, line_at: int | None) -> dict | None:
    if line_at is not None:
        row = feedback._record_at(directory / "feedback.jsonl", int(line_at))
        if isinstance(row, dict):
            return row
    for record in feedback.entries(directory).get("records", []):
        for item in record.get("claims", []):
            if item.get("section") == "对系统的建议":
                return record
    return None


def _item_text(record: dict, digest: str) -> str:
    for item in record.get("items") or []:
        if isinstance(item, dict) and str(item.get("digest") or "") == str(digest):
            return str(item.get("text") or "")
    for item in record.get("claims") or []:
        if isinstance(item, dict) and str(item.get("digest") or "") == str(digest):
            return str(item.get("text") or "")
    return ""


def _preview(text: str, chars: int) -> str:
    text = " ".join(str(text or "").split())
    if len(text) <= chars:
        return text
    return text[: max(0, chars - 1)].rstrip() + "…"


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    try:
        handle = path.open(encoding="utf-8", errors="replace")
    except OSError:
        return rows
    with handle:
        for line in handle:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                rows.append(row)
    return rows


def _linked(directory: Path, digest: str) -> tuple[str, list[str]]:
    key = ""
    for row in _read_jsonl(directory / "harness_proposals.jsonl"):
        if str(row.get("claim_digest") or "") == str(digest) and row.get("status") == "valid":
            key = str(row.get("proposal_key") or "")
            break
    reviews = []
    if key:
        reviews = [
            "useful" if row.get("useful") is True else "not-useful"
            for row in _read_jsonl(directory / "harness_reviews.jsonl")
            if str(row.get("proposal_key") or "") == key
        ]
    return key, reviews


def compact(
    *,
    task: str | None = None,
    target: str | None = None,
    model: str | None = None,
    status: str | None = None,
    limit: int = 20,
    preview: int = DEFAULT_PREVIEW,
) -> dict:
    """Group system suggestions by digest and return only a short preview per group.

    This is the AI-facing first stage. It does not return full text and does not call a model.
    One representative feedback line is read per digest; the rest stay on disk until detail().
    """
    try:
        get_index().tick()
    except OSError:
        pass
    limit = max(1, min(int(limit or 20), MAX_GROUPS))
    preview = max(0, min(int(preview if preview is not None else DEFAULT_PREVIEW), MAX_PREVIEW))
    clauses = ["c.section = '对系统的建议'"]
    args: list[object] = []
    if task:
        clauses.append("i.task = ?")
        args.append(task)
    if target:
        clauses.append("r.target_id = ?")
        args.append(target)
    if model:
        clauses.append("i.model = ?")
        args.append(model)
    if status:
        clauses.append("c.status = ?")
        args.append(status)
    sql = f"""
        SELECT c.run_id, c.seq, c.digest, c.status, c.verification,
               i.task, i.dtype, i.size, i.model, i.label, i.t, i.line_at,
               r.target_id
        FROM feedback_claims c
        JOIN feedback_items i ON i.run_id = c.run_id AND i.seq = c.seq
        LEFT JOIN runs r ON r.run_id = c.run_id
        WHERE {' AND '.join(clauses)}
        ORDER BY i.t DESC, c.seq DESC
        LIMIT 5000
    """
    groups: dict[str, dict] = {}
    with borrow() as index:
        rows = [dict(row) for row in index.conn.execute(sql, args)]
    total_claims = len(rows)
    for row in rows:
        digest = str(row.get("digest") or "")
        if not digest:
            continue
        group = groups.get(digest)
        if group is None:
            directory = detail_mod.resolve_run(str(row.get("run_id") or ""))
            text = ""
            converted_key = ""
            reviews: list[str] = []
            if directory is not None:
                record = _record(directory, row.get("line_at"))
                if record is not None:
                    text = _item_text(record, digest)
                converted_key, reviews = _linked(directory, digest)
            group = groups[digest] = {
                "digest": digest,
                "claims": 0,
                "runs": set(),
                "tasks": set(),
                "models": set(),
                "targets": set(),
                "statuses": set(),
                "first": {
                    "run_id": str(row.get("run_id") or ""),
                    "seq": row.get("seq"),
                    "label": row.get("label") or "",
                    "t": row.get("t"),
                },
                "preview": _preview(text, preview),
                "converted_key": converted_key,
                "reviews": reviews,
            }
        group["claims"] += 1
        group["runs"].add(str(row.get("run_id") or ""))
        if row.get("task"):
            group["tasks"].add(str(row["task"]))
        if row.get("model"):
            group["models"].add(str(row["model"]))
        if row.get("target_id"):
            group["targets"].add(str(row["target_id"]))
        if row.get("status"):
            group["statuses"].add(str(row["status"]))
    ordered = sorted(groups.values(), key=lambda item: (-len(item["runs"]), -float(item["first"].get("t") or 0), item["digest"]))[:limit]
    for group in ordered:
        group["runs"] = len(group["runs"])
        group["tasks"] = sorted(group["tasks"])
        group["models"] = sorted(group["models"])
        group["targets"] = sorted(group["targets"])
        group["statuses"] = sorted(group["statuses"])
        group["reviewed"] = sorted(set(group.pop("reviews", [])))
    return {
        "total_claims": total_claims,
        "total_groups": len(groups),
        "limit": limit,
        "preview_chars": preview,
        "groups": ordered,
        "hint": "Use `taic query suggestion --run RUN --seq N --digest D` only for selected groups; the full text is not included here.",
    }


def detail(run_id: str, seq: int, digest: str) -> dict:
    """Fetch one exact suggestion and its conversion/review links, without reading every run."""
    directory = detail_mod.resolve_run(run_id)
    if directory is None:
        raise FileNotFoundError(run_id)
    records = feedback.entries(directory).get("records", [])
    source = None
    for record in records:
        for item in record.get("claims", []):
            if int(item.get("seq") or 0) == int(seq) and str(item.get("digest") or "") == str(digest):
                source = item
                break
        if source:
            break
    if source is None:
        # Old records may not have a claim view for a section that is still in feedback.jsonl.
        from taic import harness

        for item in harness.free_text_suggestions(directory):
            if int(item.get("seq") or 0) == int(seq) and str(item.get("digest") or "") == str(digest):
                source = item
                break
    if source is None:
        raise KeyError(f"{run_id}#{seq}#{digest}")
    from taic import harness

    board = harness.board(directory)
    proposal = next((row for row in board["proposals"] if str(row.get("claim_digest") or "") == str(digest)), None)
    reviews = []
    if proposal:
        reviews = [row for row in board["reviews"] if row.get("proposal_key") == proposal.get("proposal_key")]
    return {
        "run_id": run_id,
        "seq": int(seq),
        "digest": str(digest),
        "text": str(source.get("text") or ""),
        "section": source.get("section") or "对系统的建议",
        "status": source.get("status") or "",
        "converted_key": proposal.get("proposal_key") if proposal else "",
        "proposal": proposal,
        "reviews": reviews,
    }
