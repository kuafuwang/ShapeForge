"""Compact, token-conscious retrieval of free-text system suggestions.

The SQLite index names claims but does not store their text. This module first groups the
index rows by digest and reads only one representative feedback line per group, returning a
short preview. The full sentence is fetched only for the selected (run, seq, digest).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from observe import detail as detail_mod
from observe import feedback
from observe.index import borrow, get_index

DEFAULT_PREVIEW = 120
_CONTROL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")
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
    text = " ".join(_CONTROL.sub(" ", str(text or "")).split())
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


def _linked(directory: Path, *, seq: int, digest: str) -> tuple[str, list[str]]:
    """The exact proposal for this claim, not merely the first proposal with the same text."""
    key = ""
    for row in _read_jsonl(directory / "harness_proposals.jsonl"):
        if (
            str(row.get("claim_digest") or "") == str(digest)
            and int(row.get("seq") or 0) == int(seq)
            and row.get("status") == "valid"
        ):
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
    offset: int = 0,
) -> dict:
    """Group system suggestions by digest and return only a short preview per group.

    This is the AI-facing first stage. It does not return full text and does not call a model.
    Totals and the page are computed by SQL over the whole filtered set, so ``truncated``
    describes the returned page rather than a silent scan cap. One representative feedback
    line is read per returned digest; the rest stay on disk until detail().
    """
    try:
        get_index().tick()
    except OSError:
        pass
    limit = max(1, min(int(limit or 20), MAX_GROUPS))
    offset = max(0, int(offset or 0))
    preview = max(0, min(int(preview if preview is not None else DEFAULT_PREVIEW), MAX_PREVIEW))
    clauses = ["c.section = '对系统的建议'", "c.digest <> ''"]
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
    base = f"""
        FROM feedback_claims c
        JOIN feedback_items i ON i.run_id = c.run_id AND i.seq = c.seq
        LEFT JOIN runs r ON r.run_id = c.run_id
        WHERE {' AND '.join(clauses)}
    """
    groups: dict[str, dict] = {}
    with borrow() as index:
        total_claims = int(index.conn.execute(f"SELECT COUNT(*) {base}", args).fetchone()[0])
        total_groups = int(index.conn.execute(f"SELECT COUNT(DISTINCT c.digest) {base}", args).fetchone()[0])
        page = [dict(row) for row in index.conn.execute(
            f"""
            SELECT c.digest AS digest, COUNT(*) AS claims, COUNT(DISTINCT c.run_id) AS runs,
                   COALESCE(MAX(i.t), 0) AS latest_t
            {base}
            GROUP BY c.digest
            ORDER BY runs DESC, latest_t DESC, c.digest
            LIMIT ? OFFSET ?
            """,
            [*args, limit, offset],
        )]
        digests = [str(row["digest"]) for row in page]
        representative: dict[str, dict] = {}
        dims: dict[str, dict[str, set[str]]] = {}
        if digests:
            marks = ",".join("?" for _ in digests)
            rows = index.conn.execute(
                f"""
                WITH filtered AS (
                    SELECT c.run_id, c.seq, c.digest, c.status,
                           i.task, i.dtype, i.size, i.model, i.label, i.t, i.line_at,
                           r.target_id
                    {base} AND c.digest IN ({marks})
                ), ranked AS (
                    SELECT *, ROW_NUMBER() OVER (PARTITION BY digest ORDER BY t DESC, seq DESC) AS rn
                    FROM filtered
                )
                SELECT * FROM ranked WHERE rn = 1
                """,
                [*args, *digests],
            )
            representative = {str(row["digest"]): dict(row) for row in rows}
            for row in index.conn.execute(
                f"""
                SELECT DISTINCT c.digest, i.task, i.model, r.target_id, c.status
                {base} AND c.digest IN ({marks})
                """,
                [*args, *digests],
            ):
                bucket = dims.setdefault(str(row["digest"]), {"tasks": set(), "models": set(), "targets": set(), "statuses": set()})
                if row["task"]:
                    bucket["tasks"].add(str(row["task"]))
                if row["model"]:
                    bucket["models"].add(str(row["model"]))
                if row["target_id"]:
                    bucket["targets"].add(str(row["target_id"]))
                if row["status"]:
                    bucket["statuses"].add(str(row["status"]))
    for page_row in page:
        digest = str(page_row["digest"])
        row = representative.get(digest)
        if row is None:
            continue
        directory = detail_mod.resolve_run(str(row.get("run_id") or ""))
        text = ""
        converted_key = ""
        reviews: list[str] = []
        if directory is not None:
            record = _record(directory, row.get("line_at"))
            if record is not None:
                text = _item_text(record, digest)
            converted_key, reviews = _linked(directory, seq=int(row.get("seq") or 0), digest=digest)
        bucket = dims.get(digest) or {}
        groups[digest] = {
            "digest": digest,
            "claims": int(page_row["claims"]),
            "runs": int(page_row["runs"]),
            "tasks": sorted(bucket.get("tasks") or []),
            "models": sorted(bucket.get("models") or []),
            "targets": sorted(bucket.get("targets") or []),
            "statuses": sorted(bucket.get("statuses") or []),
            "first": {
                "run_id": str(row.get("run_id") or ""),
                "seq": row.get("seq"),
                "label": row.get("label") or "",
                "t": row.get("t"),
            },
            "preview": _preview(text, preview),
            "converted_key": converted_key,
            "link_scope": "first (representative claim only)",
            "reviews": reviews,
        }
    ordered = list(groups.values())
    for group in ordered:
        group["reviewed"] = sorted(set(group.pop("reviews", [])))
    returned = len(ordered)
    # Advance by the page the SQL returned, even if a representative row could not be read,
    # so a damaged run cannot make the caller fetch the same page forever.
    next_offset = offset + len(page)
    truncated = next_offset < total_groups
    return {
        "total_claims": total_claims,
        "total_groups": total_groups,
        "returned_groups": returned,
        "offset": offset,
        "next_offset": next_offset if truncated else None,
        "truncated": truncated,
        "limit": limit,
        "preview_chars": preview,
        "groups": ordered,
        "hint": "Links describe the representative claim only; another claim with the same digest may have a different conversion. Fetch selected run/seq/digest details and follow next_offset for more groups.",
    }


def detail(run_id: str, seq: int, digest: str) -> dict:
    """Fetch one exact suggestion and its conversion/review links, without reading every run."""
    directory = detail_mod.resolve_run(run_id)
    if directory is None:
        raise FileNotFoundError(run_id)
    records = feedback.entries(directory).get("records", [])
    source = None
    source_record = None
    for record in records:
        for item in record.get("claims", []):
            if int(item.get("seq") or 0) == int(seq) and str(item.get("digest") or "") == str(digest):
                source = item
                source_record = record
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
    proposal = next((
        row for row in board["proposals"]
        if str(row.get("claim_digest") or "") == str(digest)
        and int(row.get("seq") or 0) == int(seq)
        and row.get("status") == "valid"
    ), None)
    reviews = []
    if proposal:
        reviews = [row for row in board["reviews"] if row.get("proposal_key") == proposal.get("proposal_key")]
    started = next((
        row.get("session_started") for row in board.get("suggestions") or []
        if int(row.get("seq") or 0) == int(seq) and str(row.get("digest") or "") == str(digest)
    ), None)
    if started is None:
        started = source.get("session_started")
    task = str((source_record or {}).get("task") or "")
    target_id = str((source_record or {}).get("target_id") or "")
    item_t = (source_record or {}).get("t")
    prior: list[dict] = []
    if not task or not target_id:
        with borrow() as index:
            indexed = index.conn.execute(
                """
                SELECT i.task, i.t, r.target_id
                FROM feedback_claims c
                JOIN feedback_items i ON i.run_id = c.run_id AND i.seq = c.seq
                LEFT JOIN runs r ON r.run_id = c.run_id
                WHERE c.run_id = ? AND c.seq = ? AND c.digest = ?
                """,
                (run_id, int(seq), str(digest)),
            ).fetchone()
        if indexed is not None:
            task = task or str(indexed["task"] or "")
            target_id = target_id or str(indexed["target_id"] or "")
            item_t = item_t if item_t is not None else indexed["t"]
    if task and target_id:
        with borrow() as index:
            prior = feedback.prior_claims(
                index.conn, digest=str(digest), task=task, target_id=target_id,
                before_t=item_t, exclude_run=run_id, exclude_seq=int(seq),
            )
    return {
        "run_id": run_id,
        "seq": int(seq),
        "digest": str(digest),
        "text": str(source.get("text") or ""),
        "section": source.get("section") or "对系统的建议",
        "status": source.get("status") or "",
        "session_started": started,
        "probe_time_verified": proposal.get("probe_time_verified") if proposal else None,
        "converted_key": proposal.get("proposal_key") if proposal else "",
        "proposal": proposal,
        "reviews": reviews,
        "prior": prior,
    }
