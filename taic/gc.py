"""Retention for ``data/deleted``. The default is a report; ``--apply`` is the only deleting mode.

The dashboard deletion path moves a run or experiment under ``data/deleted``. Nothing has ever
removed those directories. This module adds a TTL policy without changing that default: a caller
asks for a cutoff, sees what would go, and only passes ``--apply`` deliberately.
"""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path


def _size(path: Path) -> tuple[int, int]:
    files = 0
    total = 0
    for item in path.rglob("*"):
        if item.is_file():
            files += 1
            try:
                total += item.stat().st_size
            except OSError:
                pass
    return files, total


def scan(root: Path, *, older_than_days: float, now: float | None = None) -> list[dict]:
    """One row per child directory. ``eligible`` is age >= the cutoff."""
    root = Path(root)
    if not root.is_dir():
        return []
    moment = time.time() if now is None else now
    rows = []
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        try:
            mtime = child.stat().st_mtime
        except OSError:
            continue
        files, total = _size(child)
        age = (moment - mtime) / 86400.0
        rows.append({
            "path": str(child), "age_days": round(age, 3), "files": files, "bytes": total,
            "eligible": age >= older_than_days,
        })
    return rows


def apply(rows: list[dict], *, log_path: Path | None = None, now: float | None = None) -> dict:
    """Delete the eligible rows and append one audit line before each removal."""
    moment = time.time() if now is None else now
    removed = []
    log = Path(log_path) if log_path else None
    if log is not None:
        log.parent.mkdir(parents=True, exist_ok=True)
    for row in rows:
        if not row.get("eligible"):
            continue
        path = Path(row["path"])
        if log is not None:
            with log.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({
                    "action": "gc", "path": str(path), "files": row.get("files"), "bytes": row.get("bytes"),
                    "age_days": row.get("age_days"), "at": moment,
                }, ensure_ascii=False) + "\n")
        shutil.rmtree(path, ignore_errors=True)
        removed.append(str(path))
    return {"removed": removed, "bytes": sum(row.get("bytes", 0) for row in rows if row.get("eligible"))}
