"""Append-only audit ledger for feedback review decisions.

The per-run ``feedback_review.json`` remains the source of the case. This ledger
is a small, global, hash-chained index of decisions: who reviewed, from which
state to which state, with which evidence/commit, and under which batch. It
stores hashes of free text rather than a second copy of long untrusted text.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import threading
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PATH = REPO_ROOT / "data" / "review_events.jsonl"
_LOCK = threading.Lock()
_VERSION = 1


def ledger_path(path: Path | str | None = None) -> Path:
    if path is not None:
        return Path(path)
    override = os.environ.get("TAIC_REVIEW_EVENTS")
    return Path(override) if override else DEFAULT_PATH


def _digest(text: object) -> str:
    return hashlib.sha256(str(text or "").encode("utf-8")).hexdigest()


def _canonical(payload: dict) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _project(event: dict, *, run_id: str, seq: int, digest: str) -> dict:
    subclaims = event.get("subclaims") if isinstance(event.get("subclaims"), list) else []
    return {
        "version": _VERSION,
        "reviewed_at": event.get("t") or event.get("reviewed_at"),
        "run_id": run_id,
        "seq": int(seq),
        "digest": str(digest),
        "status": str(event.get("status") or ""),
        "verification": str(event.get("verification") or ""),
        "evidence_source": str(event.get("evidence_source") or ""),
        "evidence_sha256": _digest(event.get("evidence")),
        "note_sha256": _digest(event.get("note")),
        "change_ref": str(event.get("change_ref") or ""),
        "reviewer": str(event.get("reviewer") or ""),
        "reviewer_source": str(event.get("reviewer_source") or ""),
        "method": str(event.get("method") or ""),
        "batch_id": str(event.get("batch_id") or ""),
        "expected_status": str(event.get("expected_status") or ""),
        "git_commit": str(event.get("git_commit") or ""),
        "git_dirty": event.get("git_dirty"),
        "target_id": str(event.get("target_id") or ""),
        "backend": str(event.get("backend") or ""),
        "subclaims": [
            {
                "status": str(sub.get("status") or ""),
                "verification": str(sub.get("verification") or ""),
                "text_sha256": _digest(sub.get("text")),
                "change_ref": str(sub.get("change_ref") or ""),
            }
            for sub in subclaims
            if isinstance(sub, dict)
        ],
    }


def append_events(events: list[dict], *, run_id: str, path: Path | str | None = None) -> list[str]:
    """Append one hash-chained row per event. Returns the new event hashes."""
    if not events:
        return []
    target = ledger_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    hashes: list[str] = []
    with _LOCK:
        with target.open("a+", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                handle.seek(0)
                last = {}
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        raise ValueError(f"review ledger has a damaged line: {target}") from None
                    if isinstance(row, dict):
                        last = row
                previous = str(last.get("event_hash") or "")
                for event in events:
                    projected = _project(event, run_id=run_id, seq=int(event["seq"]), digest=str(event["digest"]))
                    projected["prev_hash"] = previous
                    event_hash = hashlib.sha256(_canonical(projected).encode("utf-8")).hexdigest()
                    row = {**projected, "event_hash": event_hash}
                    handle.seek(0, os.SEEK_END)
                    handle.write(_canonical(row) + "\n")
                    handle.flush()
                    previous = event_hash
                    hashes.append(event_hash)
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    return hashes


def verify(path: Path | str | None = None) -> dict:
    """Check the chain and return counts plus the first broken row, if any."""
    target = ledger_path(path)
    if not target.is_file():
        return {"ok": True, "events": 0, "broken_at": None, "last_hash": ""}
    previous = ""
    count = 0
    with target.open(encoding="utf-8") as handle:
        for index, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                return {"ok": False, "events": count, "broken_at": index, "last_hash": previous}
            if not isinstance(row, dict):
                return {"ok": False, "events": count, "broken_at": index, "last_hash": previous}
            stored = str(row.pop("event_hash", ""))
            if row.get("prev_hash") != previous or hashlib.sha256(_canonical(row).encode("utf-8")).hexdigest() != stored:
                return {"ok": False, "events": count, "broken_at": index, "last_hash": previous}
            previous = stored
            count += 1
    return {"ok": True, "events": count, "broken_at": None, "last_hash": previous}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Verify the local feedback-review hash chain")
    parser.add_argument("--path", default=None)
    args = parser.parse_args()
    print(json.dumps(verify(args.path), ensure_ascii=False, indent=2))
