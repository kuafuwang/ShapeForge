"""Read agent feedback for the dashboard. Parsing lives in ``taic.agent_feedback``."""

from __future__ import annotations

import json
import time
from pathlib import Path

from taic.agent_feedback import PARSER_VERSION, load_records, read_appended_records

_REVIEWABLE = {"system"}
_PREMISE = {"verified", "unverified", "downgraded"}
_STATUSES = {"pending", "adopted", "rejected", "done", "info", "obsolete"}
_VERIFICATIONS = {"unchecked", "confirmed", "refuted"}
_SOURCES = {"report", "calibration", "probe", "code", "other"}


def _review_key(seq: int, digest: str) -> str:
    return f"{seq}:{digest}"


def _load_review(path: Path) -> dict:
    if not path.is_file():
        return {"v": 1, "claims": {}}
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"v": 1, "claims": {}}
    if not isinstance(doc, dict):
        return {"v": 1, "claims": {}}
    claims = doc.get("claims")
    if not isinstance(claims, dict):
        claims = {}
    return {"v": 1, "claims": claims}


def _has_system(record: dict) -> bool:
    kinds = [part for part in str(record.get("kinds") or "").split(",") if part and part != "none"]
    sections = record.get("sections") if isinstance(record.get("sections"), dict) else {}
    return bool(kinds) or bool(str(sections.get("需要帮助") or "").strip()) or bool(str(sections.get("对系统的建议") or "").strip())


def _default_status(item: dict) -> str:
    audience = item.get("audience") or ""
    premise = item.get("premise") or ""
    if audience in _REVIEWABLE or premise in _PREMISE:
        return "pending"
    return "info"


def _claim_rows(record: dict, review: dict) -> list[dict]:
    seq = record.get("seq")
    if not isinstance(seq, int):
        return []
    rows = []
    for item in record.get("items") or []:
        if not isinstance(item, dict):
            continue
        digest = str(item.get("digest") or "")
        if not digest:
            continue
        saved = review.get(_review_key(seq, digest))
        if not isinstance(saved, dict):
            saved = {}
        status = saved.get("status") or _default_status(item)
        if status not in _STATUSES:
            status = _default_status(item)
        rows.append({
            "seq": seq,
            "digest": digest,
            "section": item.get("section") or "",
            "audience": item.get("audience") or "",
            "premise": item.get("premise") or "",
            "text": item.get("text") or "",
            "status": status,
            "verification": saved.get("verification") or "unchecked",
            "topic": saved.get("topic") or "",
            "change_ref": saved.get("change_ref") or "",
            "reviewed_at": saved.get("t") or saved.get("reviewed_at"),
            "evidence_source": saved.get("evidence_source") or "",
            "evidence": saved.get("evidence") or "",
            "note": saved.get("note") or "",
            "history": list(saved.get("history") or []),
        })
    return rows


def _feedback_revision(conn) -> int:
    row = conn.execute("SELECT value FROM meta WHERE key = 'feedback_revision'").fetchone()
    return int(row[0]) if row else 0


def _take_feedback_rev(conn) -> int:
    """One step for a batch of claim edits. Unchanged claims keep the revision they already have."""
    nxt = _feedback_revision(conn) + 1
    conn.execute(
        "INSERT INTO meta (key, value) VALUES ('feedback_revision', ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        (str(nxt),),
    )
    return nxt


def _reviewed_key(value) -> str | None:
    if value is None or value == "":
        return None
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return str(value)


def _claim_signature(row) -> tuple:
    return (
        row["section"] or "",
        row["audience"] or "",
        row["premise"] or "",
        row["status"] or "",
        row["verification"] or "",
        row["topic"] or "",
        row["change_ref"] or "",
        _reviewed_key(row["reviewed_at"]),
    )


def _item_signature(row) -> tuple:
    return (
        row["seq"],
        _reviewed_key(row["t"]),
        row["task"] or "",
        row["dtype"] or "",
        row["size"] or "",
        row["label"] or "",
        row["committee"] or "",
        row["provider"] or "",
        row["model"] or "",
        row["outcome"] or "",
        row["exit"] or "",
        row["kinds"] or "",
        int(row["has_system"] or 0),
        int(row["pending_claims"] or 0),
        int(row["unverified_premises"] or 0),
    )


def retire_run_claims(conn, run_id: str) -> bool:
    """Forget one run's claims and leave a tombstone for each, so an open review page can drop those cards."""
    rows = list(conn.execute(
        "SELECT seq, digest, section, audience, premise, status FROM feedback_claims WHERE run_id = ?",
        (run_id,),
    ))
    if not rows:
        conn.execute("DELETE FROM feedback_items WHERE run_id = ?", (run_id,))
        return False
    rev = _take_feedback_rev(conn)
    for row in rows:
        conn.execute(
            """
            INSERT INTO feedback_claim_removed (run_id, seq, digest, rev, section, audience, premise, status, reason)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(run_id, seq, digest) DO UPDATE SET
                rev = excluded.rev, section = excluded.section, audience = excluded.audience,
                premise = excluded.premise, status = excluded.status, reason = excluded.reason
            """,
            (run_id, row["seq"], row["digest"], rev, row["section"], row["audience"], row["premise"], row["status"], "run retired"),
        )
    conn.execute("DELETE FROM feedback_claims WHERE run_id = ?", (run_id,))
    conn.execute("DELETE FROM feedback_items WHERE run_id = ?", (run_id,))
    return True


def _file_fp(path: Path) -> str:
    try:
        stat = path.stat()
    except OSError:
        return ""
    return f"{stat.st_mtime_ns}:{stat.st_size}"


def _feedback_cursor(conn, run_id: str) -> tuple[int, str, str] | None:
    row = conn.execute(
        "SELECT feedback_bytes, feedback_review_fp, feedback_parser FROM runs WHERE run_id = ?",
        (run_id,),
    ).fetchone()
    if row is None:
        return None
    return int(row["feedback_bytes"] or 0), str(row["feedback_review_fp"] or ""), str(row["feedback_parser"] or "")


def _store_feedback_cursor(conn, run_id: str, offset: int, review_fp: str) -> None:
    conn.execute(
        "UPDATE runs SET feedback_bytes = ?, feedback_review_fp = ?, feedback_parser = ? WHERE run_id = ?",
        (offset, review_fp, PARSER_VERSION, run_id),
    )


def _rows_for_record(record: dict, review: dict, line_at: int | None = None) -> tuple[dict, list[dict]] | None:
    seq = record.get("seq")
    if not isinstance(seq, int) or isinstance(seq, bool) or not _has_text(record):
        return None
    built = _claim_rows(record, review)
    pending = sum(1 for claim in built if claim["status"] == "pending")
    unverified = sum(1 for claim in built if claim["status"] == "pending" and claim["premise"] in ("unverified", "downgraded"))
    item = {
        "seq": seq,
        "t": record.get("t"),
        "task": record.get("task") or "",
        "dtype": record.get("dtype") or "",
        "size": record.get("size") or "",
        "label": record.get("label") or "",
        "committee": record.get("committee") or "",
        "provider": record.get("provider") or "",
        "model": record.get("model") or "",
        "outcome": record.get("outcome") or "",
        "exit": record.get("exit") or "",
        "kinds": record.get("kinds") or "",
        "line_at": line_at,
        "has_system": 1 if _has_system(record) else 0,
        "pending_claims": pending,
        "unverified_premises": unverified,
    }
    return item, built


def _write_claim(conn, run_id: str, claim: dict, rev: int) -> None:
    conn.execute("DELETE FROM feedback_claim_removed WHERE run_id = ? AND seq = ? AND digest = ?", (run_id, claim["seq"], claim["digest"]))
    conn.execute(
        """
        INSERT INTO feedback_claims (
            run_id, seq, digest, section, audience, premise, status, verification, topic, change_ref, reviewed_at, rev
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
        ON CONFLICT(run_id, seq, digest) DO UPDATE SET
            section = excluded.section, audience = excluded.audience, premise = excluded.premise,
            status = excluded.status, verification = excluded.verification, topic = excluded.topic,
            change_ref = excluded.change_ref, reviewed_at = excluded.reviewed_at, rev = excluded.rev
        """,
        (
            run_id, claim["seq"], claim["digest"], claim["section"], claim["audience"], claim["premise"],
            claim["status"], claim["verification"], claim["topic"], claim["change_ref"], claim["reviewed_at"], rev,
        ),
    )


def _write_item(conn, run_id: str, row: dict) -> None:
    conn.execute(
        """
        INSERT INTO feedback_items (
            run_id, seq, t, task, dtype, size, label, committee, provider, model, outcome, exit, kinds,
            line_at, has_system, pending_claims, unverified_premises
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """,
        (
            run_id, row["seq"], row["t"], row["task"], row["dtype"], row["size"], row["label"], row["committee"],
            row["provider"], row["model"], row["outcome"], row["exit"], row["kinds"], row.get("line_at"),
            row["has_system"], row["pending_claims"], row["unverified_premises"],
        ),
    )


def _proposition(text: str) -> str:
    """The proposition slot of a bracket, normalized for a source/old-item match."""
    import re

    match = re.search(r"\[[^\]]*\|\s*([^|\]]+?)\s*\|", str(text or ""))
    return re.sub(r"\s+", " ", match.group(1).strip().lower()) if match else ""


def _reparsed_record(record: dict, directory: Path) -> dict:
    """Upgrade a restored measured premise without changing any old claim digest.

    ``feedback.jsonl`` is append-only and stores items. A parser bump alone cannot know that a
    probe was restored after the session, and replacing every item with a fresh split would
    change digests and orphan ``feedback_review.json``. On a rebuild, read the session's
    ``FEEDBACK.md`` only to find a verified copy of an old downgraded/unverified proposition;
    copy its text and premise but keep the old digest. The log itself is not rewritten.
    """
    from taic.agent_feedback import FILE_CAP, parse_feedback, read_feedback_text

    items = [dict(item) for item in (record.get("items") or []) if isinstance(item, dict)]
    work_rel = str(record.get("work_dir") or "")
    work = (directory / work_rel).resolve()
    try:
        work.relative_to(directory.resolve())
    except ValueError:
        return record
    path = work / "FEEDBACK.md"
    text = str(record.get("raw") or "")
    file_over = False
    if path.is_file():
        try:
            text, file_over = read_feedback_text(path)
        except OSError:
            text = ""
    if not text.strip():
        return record
    source = parse_feedback(text, work=work, repo=Path(__file__).resolve().parents[1], file_over=file_over or len(text) > FILE_CAP)
    if source.get("unrecognized_version"):
        return record
    verified = [
        item for item in source.get("items") or []
        if isinstance(item, dict) and item.get("premise") == "verified"
    ]
    old_premises = [item for item in items if item.get("premise") in ("downgraded", "unverified")]
    positions = {id(item): index for index, item in enumerate(items)}
    changed = False
    for item in old_premises:
        position = positions[id(item)]
        section_position = sum(
            1 for earlier in items[:position]
            if earlier.get("section") == item.get("section")
        )
        if item.get("premise") not in ("downgraded", "unverified"):
            continue
        prop = _proposition(str(item.get("text") or ""))
        same_section = [candidate for candidate in verified if candidate.get("section") == item.get("section")]
        match = None
        if prop:
            match = next((
                candidate for candidate in same_section
                if prop in _proposition(str(candidate.get("text") or ""))
            ), None)
        if match is None and not prop and section_position < len(same_section):
            # Older lines sometimes stored the wiped tag without the proposition, but kept the
            # explanation. The source's verified lines are in the same section order.
            match = same_section[section_position]
        if match is None:
            continue
        item["text"] = str(match.get("text") or item.get("text") or "")
        item["premise"] = "verified"
        changed = True
    if not changed:
        return record
    record = dict(record)
    record["items"] = items
    record["unverified_premises"] = sum(1 for item in items if item.get("premise") in ("unverified", "downgraded"))
    return record


def _rebuild_feedback(conn, run_id: str, directory: Path, *, reason: str = "feedback log changed") -> bool:
    """The log was replaced or the parser changed. Read it once, then the cursor takes over."""
    review = _load_review(directory / "feedback_review.json")["claims"]
    items = []
    claims = []
    located, _end = read_appended_records(directory / "feedback.jsonl", 0)
    for line_at, record in located:
        record = _reparsed_record(record, directory)
        built = _rows_for_record(record, review, line_at)
        if built is None:
            continue
        item, claim_rows = built
        items.append(item)
        claims.extend(claim_rows)
    previous = {
        (int(row["seq"]), str(row["digest"])): dict(row)
        for row in conn.execute("SELECT * FROM feedback_claims WHERE run_id = ?", (run_id,))
    }
    current = {(int(claim["seq"]), str(claim["digest"])): claim for claim in claims}
    removed = [key for key in previous if key not in current]
    changed = [claim for key, claim in current.items() if key not in previous or _claim_signature(previous[key]) != _claim_signature(claim)]
    stored_items = {
        _item_signature(row)
        for row in conn.execute("SELECT * FROM feedback_items WHERE run_id = ?", (run_id,))
    }
    items_differ = stored_items != {_item_signature(row) for row in items}
    if not removed and not changed and not items_differ:
        return False
    rev = _take_feedback_rev(conn) if (removed or changed) else None
    if items_differ or removed or changed:
        conn.execute("DELETE FROM feedback_items WHERE run_id = ?", (run_id,))
        for row in items:
            _write_item(conn, run_id, row)
    for seq, digest in removed:
        old = previous.get((seq, digest), {})
        saved = review.get(_review_key(seq, digest)) if isinstance(review.get(_review_key(seq, digest)), dict) else {}
        status = saved.get("status") or old.get("status") or "removed"
        conn.execute("DELETE FROM feedback_claims WHERE run_id = ? AND seq = ? AND digest = ?", (run_id, seq, digest))
        conn.execute(
            """
            INSERT INTO feedback_claim_removed (run_id, seq, digest, rev, section, audience, premise, status, reason)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(run_id, seq, digest) DO UPDATE SET
                rev = excluded.rev, section = excluded.section, audience = excluded.audience,
                premise = excluded.premise, status = excluded.status, reason = excluded.reason
            """,
            (run_id, seq, digest, rev, old.get("section"), old.get("audience"), old.get("premise"), status, reason),
        )
    for claim in changed:
        _write_claim(conn, run_id, claim, rev)
    return True


def _append_feedback(conn, run_id: str, records: list[tuple[int, dict]], directory: Path) -> bool:
    """Insert sessions that are not in the index yet. Existing sequence numbers stay untouched."""
    have = {
        int(row["seq"])
        for row in conn.execute("SELECT seq FROM feedback_items WHERE run_id = ?", (run_id,))
    }
    review = _load_review(directory / "feedback_review.json")["claims"]
    fresh_items = []
    fresh_claims = []
    for line_at, record in records:
        built = _rows_for_record(record, review, line_at)
        if built is None or built[0]["seq"] in have:
            continue
        item, claim_rows = built
        fresh_items.append(item)
        fresh_claims.extend(claim_rows)
        have.add(item["seq"])
    if not fresh_claims and not fresh_items:
        return False
    rev = _take_feedback_rev(conn)
    for row in fresh_items:
        _write_item(conn, run_id, row)
    for claim in fresh_claims:
        _write_claim(conn, run_id, claim, rev)
    return True


def _desired_review(row, saved: dict | None) -> dict:
    base = {
        "seq": int(row["seq"]),
        "digest": str(row["digest"]),
        "section": row["section"] or "",
        "audience": row["audience"] or "",
        "premise": row["premise"] or "",
    }
    if not saved:
        base["status"] = _default_status(base)
        base["verification"] = "unchecked"
        base["topic"] = ""
        base["change_ref"] = ""
        base["reviewed_at"] = None
        return base
    status = saved.get("status") or _default_status(base)
    if status not in _STATUSES:
        status = _default_status(base)
    base["status"] = status
    base["verification"] = saved.get("verification") or "unchecked"
    base["topic"] = saved.get("topic") or ""
    base["change_ref"] = saved.get("change_ref") or ""
    base["reviewed_at"] = saved.get("t") or saved.get("reviewed_at")
    return base


def _apply_review_file(conn, run_id: str, directory: Path) -> bool:
    """Update stored claims from ``feedback_review.json`` without reading the log."""
    saved = _load_review(directory / "feedback_review.json")["claims"]
    rows = list(conn.execute("SELECT * FROM feedback_claims WHERE run_id = ?", (run_id,)))
    changed = []
    for row in rows:
        key = f"{int(row['seq'])}:{row['digest']}"
        review = saved.get(key)
        desired = _desired_review(row, review if isinstance(review, dict) else None)
        if _claim_signature(row) != _claim_signature(desired):
            changed.append(desired)
    if not changed:
        return False
    rev = _take_feedback_rev(conn)
    for claim in changed:
        _write_claim(conn, run_id, claim, rev)
    return True


def sync_run_index(conn, run_id: str, directory: Path) -> bool:
    """Catch up one run from the append-only log and the review file.

    A log that only grew is read from the stored byte offset. The review file is applied on its own,
    so a saved review does not parse the log again. A shorter log or a new parser reads the file once,
    then the offset takes over. Unchanged claims keep their revision.
    """
    cursor = _feedback_cursor(conn, run_id)
    if cursor is None:
        return False
    offset, stored_review, stored_parser = cursor
    path = directory / "feedback.jsonl"
    review_path = directory / "feedback_review.json"
    try:
        size = path.stat().st_size if path.is_file() else 0
    except OSError:
        size = 0
    review_fp = _file_fp(review_path)
    if stored_parser != PARSER_VERSION or size < offset:
        reason = "parser changed" if stored_parser != PARSER_VERSION else "feedback log shortened"
        changed = _rebuild_feedback(conn, run_id, directory, reason=reason)
        _store_feedback_cursor(conn, run_id, size, review_fp)
        return changed
    changed = False
    if size > offset:
        records, new_offset = read_appended_records(path, offset)
        if _append_feedback(conn, run_id, records, directory):
            changed = True
        offset = new_offset
    if review_fp != stored_review:
        if _apply_review_file(conn, run_id, directory):
            changed = True
    if changed or offset != cursor[0] or review_fp != stored_review:
        _store_feedback_cursor(conn, run_id, offset, review_fp)
    return changed


def _candidate_report(directory: Path, label: str) -> tuple[dict | None, str]:
    root = directory / "candidates"
    if not label or not root.is_dir():
        return None, ""
    matches = [path for path in root.iterdir() if path.is_dir() and path.name.endswith("_" + label)]
    if not matches:
        return None, ""
    path = sorted(matches)[-1] / "report.json"
    if not path.is_file():
        return None, ""
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None, ""
    if not isinstance(doc, dict):
        return None, ""
    return doc, str(path)


def _report_brief(doc: dict | None, source: str) -> dict | None:
    if not doc:
        return None
    benchmark = doc.get("benchmark") if isinstance(doc.get("benchmark"), dict) else {}
    candidate = benchmark.get("candidate") if isinstance(benchmark.get("candidate"), dict) else {}
    roof = doc.get("roofline") if isinstance(doc.get("roofline"), dict) else {}
    return {
        "source": source,
        "status": doc.get("status"),
        "candidate_us": candidate.get("median_us", doc.get("candidate_us")),
        "cv": candidate.get("cv"),
        "roofline": {key: roof.get(key) for key in ("lower_bound_us", "bound_kind", "time_over_lower_bound") if key in roof},
    }


def _calibration() -> dict | None:
    from tcenv.target import TargetError, read_calibration

    try:
        doc, _path, _source = read_calibration()
    except (OSError, json.JSONDecodeError, TargetError):
        return None
    if not isinstance(doc, dict):
        return None
    peaks = doc.get("peak_tflops") if isinstance(doc.get("peak_tflops"), dict) else {}
    return {
        "fp16": peaks.get("fp16"),
        "bf16": peaks.get("bf16"),
        "fp32": peaks.get("fp32"),
        "read_gbps": doc.get("read_gbps"),
        "copy_gbps": doc.get("copy_gbps"),
    }


def _has_text(record: dict) -> bool:
    """A session with no feedback still occupies a sequence number. That row is a clock tick, not a review item."""
    if str(record.get("raw") or "").strip():
        return True
    return bool(record.get("items"))


def entries(directory: Path) -> dict:
    """One run's feedback, with the review state and the numbers a reviewer can paste."""
    review = _load_review(directory / "feedback_review.json")["claims"]
    records = []
    for record in load_records(directory / "feedback.jsonl"):
        if not _has_text(record):
            continue
        report, report_path = _candidate_report(directory, str(record.get("label") or ""))
        source = "candidate"
        if report is None:
            report, report_path = _candidate_report(directory, str(record.get("seed_label") or ""))
            source = "seed" if report is not None else ""
        records.append({
            "seq": record.get("seq"),
            "t": record.get("t"),
            "label": record.get("label"),
            "round": record.get("round"),
            "committee": record.get("committee"),
            "provider": record.get("provider"),
            "model": record.get("model"),
            "effort": record.get("effort"),
            "outcome": record.get("outcome"),
            "exit": record.get("exit"),
            "best_speedup": record.get("best_speedup"),
            "seed_label": record.get("seed_label"),
            "kinds": record.get("kinds"),
            "work_dir": record.get("work_dir"),
            "truncated": record.get("truncated"),
            "truncated_sections": list(record.get("truncated_sections") or []),
            "raw": record.get("raw") or "",
            "claims": _claim_rows(record, review),
            "report": _report_brief(report, source),
            "report_path": report_path,
        })
    return {"records": records, "calibration": _calibration()}


def summary_from_index(conn, *, task: str | None = None, model: str | None = None, kind: str | None = None, status: str | None = None) -> dict:
    """Counts and the filtered claim index. ``pending_total`` is every pending claim, not the filtered slice."""
    pending_total = int(conn.execute("SELECT COUNT(*) FROM feedback_claims WHERE status = 'pending'").fetchone()[0])
    where = ["1 = 1"]
    args: list[object] = []
    if task:
        where.append("i.task = ?")
        args.append(task)
    if model:
        where.append("i.model = ?")
        args.append(model)
    if kind:
        where.append("(',' || i.kinds || ',') LIKE ?")
        args.append(f"%,{kind},%")
    if status:
        where.append("c.status = ?")
        args.append(status)
    items = [
        dict(row)
        for row in conn.execute(
            f"""
            SELECT c.run_id, c.seq, c.digest, c.section, c.audience, c.premise, c.status, c.verification, c.topic,
                   c.change_ref, c.reviewed_at, i.task, i.model, i.label, i.kinds, i.t
            FROM feedback_claims c
            JOIN feedback_items i ON i.run_id = c.run_id AND i.seq = c.seq
            WHERE {' AND '.join(where)}
            ORDER BY i.t DESC, c.seq DESC
            """,
            args,
        )
    ]
    frequent_rows = conn.execute(
        """
        SELECT COALESCE(NULLIF(c.topic, ''), c.digest) AS claim_key,
               COUNT(DISTINCT c.run_id) AS runs,
               GROUP_CONCAT(DISTINCT i.model) AS models
        FROM feedback_claims c
        JOIN feedback_items i ON i.run_id = c.run_id AND i.seq = c.seq
        GROUP BY claim_key
        HAVING runs >= 1
        ORDER BY runs DESC, claim_key
        LIMIT 20
        """
    )
    frequent = [{"key": row["claim_key"], "runs": row["runs"], "models": row["models"] or ""} for row in frequent_rows]
    done = list(conn.execute(
        """
        SELECT c.run_id, c.digest, c.topic, c.reviewed_at, r.started_at
        FROM feedback_claims c
        JOIN runs r ON r.run_id = c.run_id
        WHERE c.status = 'done' AND c.reviewed_at IS NOT NULL
        """
    ))
    later = list(conn.execute(
        """
        SELECT c.run_id, c.digest, c.topic, i.t, r.started_at
        FROM feedback_claims c
        JOIN feedback_items i ON i.run_id = c.run_id AND i.seq = c.seq
        JOIN runs r ON r.run_id = c.run_id
        """
    ))
    recurrence = []
    for finished in done:
        key = finished["topic"] or finished["digest"]
        if not key:
            continue
        for other in later:
            if other["run_id"] == finished["run_id"]:
                continue
            same = (finished["topic"] and other["topic"] == finished["topic"]) or other["digest"] == finished["digest"]
            if not same:
                continue
            if other["started_at"] is None or finished["reviewed_at"] is None:
                continue
            if float(other["started_at"]) <= float(finished["reviewed_at"]):
                continue
            if other["t"] is None or float(other["t"]) <= float(finished["reviewed_at"]):
                continue
            recurrence.append({"key": key, "run_id": other["run_id"], "after": finished["run_id"]})
    return {"pending_total": pending_total, "frequent": frequent, "recurrence": recurrence, "items": items}


def review_board(conn, *, task: str | None = None, model: str | None = None, kind: str | None = None, status: str | None = None) -> dict:
    """Every review claim across runs, with the text that lives in the run directory.

    The index row names the claim. The sentence, the saved evidence and the candidate report are read from that run.
    """
    from observe.detail import resolve_run

    summary = summary_from_index(conn, task=task, model=model, kind=kind, status=status)
    grouped: dict[str, list[dict]] = {}
    for item in summary["items"]:
        grouped.setdefault(str(item["run_id"]), []).append(item)
    calibration = None
    for run_id, rows in grouped.items():
        directory = resolve_run(run_id)
        found: dict[tuple[int, str], tuple[dict, dict]] = {}
        if directory is not None:
            doc = entries(directory)
            calibration = doc.get("calibration") or calibration
            for record in doc["records"]:
                for claim in record["claims"]:
                    seq = claim.get("seq")
                    digest = str(claim.get("digest") or "")
                    if isinstance(seq, int) and digest:
                        found[(seq, digest)] = (claim, record)
        for row in rows:
            try:
                seq = int(row["seq"])
            except (TypeError, ValueError):
                seq = -1
            pair = found.get((seq, str(row.get("digest") or "")))
            claim, record = pair if pair is not None else ({}, {})
            row["text"] = claim.get("text") or ""
            row["verification"] = claim.get("verification") or row.get("verification") or "unchecked"
            row["evidence_source"] = claim.get("evidence_source") or ""
            row["evidence"] = claim.get("evidence") or ""
            row["note"] = claim.get("note") or ""
            row["topic"] = claim.get("topic") or row.get("topic") or ""
            row["change_ref"] = claim.get("change_ref") or row.get("change_ref") or ""
            row["status"] = claim.get("status") or row.get("status") or ""
            row["round"] = record.get("round")
            row["committee"] = record.get("committee") or ""
            row["report"] = record.get("report")
            row["report_path"] = record.get("report_path") or ""
    return {"pending_total": summary["pending_total"], "calibration": calibration, "items": summary["items"]}


def _record_at(path: Path, line_at: int) -> dict | None:
    """One log line. The offset was stored when that line was appended."""
    try:
        with path.open("rb") as handle:
            handle.seek(line_at)
            raw = handle.readline()
    except OSError:
        return None
    try:
        line = raw.decode("utf-8")
    except UnicodeDecodeError:
        line = raw.decode("utf-8", errors="replace")
    from taic.agent_feedback import FORMAT, _load_line, _repair_record

    row = _load_line(line)
    if row is None or row.get("format") != FORMAT:
        return None
    return _repair_record(row)


def _attach_claim_text(rows: list[dict]) -> dict | None:
    """Fill claim text from the stored line offset. A row without one falls back to reading that run's log."""
    from observe.detail import resolve_run

    grouped: dict[str, list[dict]] = {}
    for item in rows:
        grouped.setdefault(str(item["run_id"]), []).append(item)
    calibration = _calibration()
    for run_id, group in grouped.items():
        directory = resolve_run(run_id)
        found: dict[tuple[int, str], tuple[dict, dict]] = {}
        if directory is not None and any(row.get("line_at") is None for row in group):
            for record in entries(directory)["records"]:
                for claim in record["claims"]:
                    seq = claim.get("seq")
                    digest = str(claim.get("digest") or "")
                    if isinstance(seq, int) and digest:
                        found[(seq, digest)] = (claim, record)
        elif directory is not None:
            path = directory / "feedback.jsonl"
            review = _load_review(directory / "feedback_review.json")["claims"]
            by_seq: dict[int, dict] = {}
            for row in group:
                try:
                    seq = int(row["seq"])
                except (TypeError, ValueError):
                    continue
                line_at = row.get("line_at")
                if not isinstance(line_at, int) or seq in by_seq:
                    continue
                record = _record_at(path, line_at)
                if record is not None:
                    by_seq[seq] = record
            for record in by_seq.values():
                for claim in _claim_rows(record, review):
                    digest = str(claim.get("digest") or "")
                    seq = claim.get("seq")
                    if isinstance(seq, int) and digest:
                        found[(seq, digest)] = (claim, record)
        for row in group:
            try:
                seq = int(row["seq"])
            except (TypeError, ValueError):
                seq = -1
            pair = found.get((seq, str(row.get("digest") or "")))
            claim, record = pair if pair is not None else ({}, {})
            row["text"] = claim.get("text") or ""
            row["verification"] = claim.get("verification") or row.get("verification") or "unchecked"
            row["evidence_source"] = claim.get("evidence_source") or ""
            row["evidence"] = claim.get("evidence") or ""
            row["note"] = claim.get("note") or ""
            row["topic"] = claim.get("topic") or row.get("topic") or ""
            row["change_ref"] = claim.get("change_ref") or row.get("change_ref") or ""
            row["status"] = claim.get("status") or row.get("status") or ""
            row["round"] = record.get("round") if record else row.get("round")
            row["committee"] = (record.get("committee") if record else None) or row.get("committee") or ""
            if record:
                report, report_path = _candidate_report(directory, str(record.get("label") or "")) if directory is not None else (None, "")
                source = "candidate"
                if report is None and directory is not None:
                    report, report_path = _candidate_report(directory, str(record.get("seed_label") or ""))
                    source = "seed" if report is not None else ""
                row["report"] = _report_brief(report, source)
                row["report_path"] = report_path
            else:
                row["report"] = None
                row["report_path"] = ""
    return calibration


_CLAIM_LIST = """
    SELECT c.run_id, c.seq, c.digest, c.section, c.audience, c.premise, c.status, c.verification, c.topic,
           c.change_ref, c.reviewed_at, c.rev, i.task, i.dtype, i.size, i.model, i.label, i.t, i.line_at,
           r.target_id
    FROM feedback_claims c
    JOIN feedback_items i ON i.run_id = c.run_id AND i.seq = c.seq
    LEFT JOIN runs r ON r.run_id = c.run_id
"""


def review_feed(conn, *, since: int | None = None) -> dict:
    """A snapshot, or only the claims that changed after ``since``.

    ``since`` equal to the current feedback revision returns no rows. Text is read only for runs
    present in this answer.
    """
    rev = _feedback_revision(conn)
    pending_total = int(conn.execute("SELECT COUNT(*) FROM feedback_claims WHERE status = 'pending'").fetchone()[0])
    if since is not None and int(since) == rev:
        return {"same": True, "snapshot": False, "rev": rev, "pending_total": pending_total, "upserts": [], "removed": []}
    if since is None:
        rows = [dict(row) for row in conn.execute(_CLAIM_LIST + " ORDER BY i.t DESC, c.seq DESC")]
        removed: list[dict] = []
        snapshot = True
    else:
        rows = [dict(row) for row in conn.execute(_CLAIM_LIST + " WHERE c.rev > ? ORDER BY i.t DESC, c.seq DESC", (int(since),))]
        removed = [
            {"run_id": row["run_id"], "seq": row["seq"], "digest": row["digest"], "status": row["status"], "reason": row["reason"]}
            for row in conn.execute(
                "SELECT run_id, seq, digest, status, reason FROM feedback_claim_removed WHERE rev > ?",
                (int(since),),
            )
        ]
        snapshot = False
    calibration = _attach_claim_text(rows)
    return {
        "same": False,
        "snapshot": snapshot,
        "rev": rev,
        "pending_total": pending_total,
        "upserts": rows,
        "removed": removed,
        "calibration": calibration,
    }


def rejection(body: dict) -> str | None:
    """None when the review body is a legal status combination."""
    status = body.get("status")
    verification = body.get("verification")
    if status not in _STATUSES or verification not in _VERIFICATIONS:
        return "status or verification is not recognized"
    evidence_source = str(body.get("evidence_source") or "")
    evidence = str(body.get("evidence") or "").strip()
    note = str(body.get("note") or "").strip()
    change_ref = str(body.get("change_ref") or "").strip()
    if evidence_source and evidence_source not in _SOURCES:
        return "evidence_source is not recognized"
    if status == "pending":
        if verification != "unchecked":
            return "pending requires verification unchecked"
        return None
    if status == "info":
        if verification != "unchecked":
            return "info requires verification unchecked"
        return None
    if status == "obsolete":
        if not note:
            return "obsolete requires a note"
        return None
    if status == "adopted":
        if verification != "confirmed" or not evidence_source or not evidence:
            return "adopted requires a confirmed verification and evidence"
        return None
    if status == "rejected":
        if verification not in ("refuted", "confirmed"):
            return "rejected requires verification confirmed or refuted"
        if not note:
            return "rejected requires a note"
        return None
    if verification != "confirmed" or not change_ref:
        return "done requires a confirmed verification and change_ref"
    return None


def apply_review(run_dir: Path, seq: int, digest: str, body: dict) -> dict:
    """Append one review and return the stored claim. The caller holds the run lock."""
    from taic import control

    reason = rejection(body)
    if reason:
        raise ValueError(reason)
    path = run_dir / "feedback_review.json"
    doc = _load_review(path)
    key = _review_key(seq, digest)
    previous = doc["claims"].get(key)
    history = list(previous.get("history") or []) if isinstance(previous, dict) else []
    event = {
        "t": time.time(),
        "status": body.get("status"),
        "verification": body.get("verification"),
        "evidence_source": str(body.get("evidence_source") or ""),
        "evidence": str(body.get("evidence") or ""),
        "note": str(body.get("note") or ""),
        "topic": str(body.get("topic") or ""),
        "change_ref": str(body.get("change_ref") or ""),
    }
    history.append(event)
    stored = dict(event)
    stored["history"] = history
    stored["reviewed_at"] = event["t"]
    doc["claims"][key] = stored
    control.write_json_atomic(path, doc)
    return stored
