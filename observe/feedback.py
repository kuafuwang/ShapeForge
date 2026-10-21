"""Read agent feedback for the dashboard. Parsing lives in ``taic.agent_feedback``."""

from __future__ import annotations

import json
import time
from pathlib import Path

from taic.agent_feedback import load_records

_REVIEWABLE = {"system"}
_PREMISE = {"verified", "unverified", "downgraded"}
_STATUSES = {"pending", "adopted", "rejected", "done"}
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
        if status not in _STATUSES and status != "info":
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


def sync_run_index(conn, run_id: str, directory: Path) -> None:
    """Replace this run's feedback rows from the files. Large text stays on disk."""
    conn.execute("DELETE FROM feedback_claims WHERE run_id = ?", (run_id,))
    conn.execute("DELETE FROM feedback_items WHERE run_id = ?", (run_id,))
    review = _load_review(directory / "feedback_review.json")["claims"]
    for record in load_records(directory / "feedback.jsonl"):
        if not _has_text(record):
            continue
        seq = record.get("seq")
        if not isinstance(seq, int):
            continue
        claims = _claim_rows(record, review)
        pending = sum(1 for claim in claims if claim["status"] == "pending")
        unverified = sum(1 for claim in claims if claim["status"] == "pending" and claim["premise"] in ("unverified", "downgraded"))
        conn.execute(
            """
            INSERT INTO feedback_items (
                run_id, seq, t, task, dtype, size, label, committee, provider, model, outcome, exit, kinds,
                has_system, pending_claims, unverified_premises
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                run_id, seq, record.get("t"), record.get("task") or "", record.get("dtype") or "", record.get("size") or "",
                record.get("label") or "", record.get("committee") or "", record.get("provider") or "", record.get("model") or "",
                record.get("outcome") or "", record.get("exit") or "", record.get("kinds") or "",
                1 if _has_system(record) else 0, pending, unverified,
            ),
        )
        for claim in claims:
            conn.execute(
                """
                INSERT INTO feedback_claims (
                    run_id, seq, digest, section, audience, premise, status, verification, topic, change_ref, reviewed_at
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    run_id, claim["seq"], claim["digest"], claim["section"], claim["audience"], claim["premise"],
                    claim["status"], claim["verification"], claim["topic"], claim["change_ref"], claim["reviewed_at"],
                ),
            )


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
    path = Path(__file__).resolve().parent.parent / "runs" / "calibration.json"
    if not path.is_file():
        return None
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(doc, dict):
        return None
    peaks = doc.get("peak_tflops") if isinstance(doc.get("peak_tflops"), dict) else {}
    return {
        "fp16": peaks.get("fp16"),
        "bf16": peaks.get("bf16"),
        "fp32": peaks.get("fp32"),
        "read_gbps": (doc.get("bandwidth_gbps") or {}).get("read") if isinstance(doc.get("bandwidth_gbps"), dict) else None,
        "copy_gbps": (doc.get("bandwidth_gbps") or {}).get("copy") if isinstance(doc.get("bandwidth_gbps"), dict) else None,
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
            "kinds": record.get("kinds"),
            "work_dir": record.get("work_dir"),
            "truncated": record.get("truncated"),
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
