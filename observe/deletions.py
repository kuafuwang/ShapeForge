"""Archive one finished experiment directory or one search directory, and append a reason. The log is append-only.

The directories move under the observe data directory, next to the index, and out of ``runs/``.
The spend ledger is not reduced. Nothing in this module rewrites or truncates the log.
"""

from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path

from observe import detail
from observe.actions import Refused
from observe.index import borrow, db_path, get_index
from observe.ledger import operator_name
from observe.query import refresh

_SKIP = {"candidates", "__pycache__"}
_REASON_LIMIT = 500


def log_path() -> Path:
    return db_path().parent / "deletion_log.jsonl"


def archive_root() -> Path:
    return db_path().parent / "deleted"


def _reason(text: object) -> str:
    if not isinstance(text, str):
        raise Refused(400, "a deletion needs a reason")
    reason = text.strip()
    if not reason:
        raise Refused(400, "a deletion needs a reason")
    if len(reason) > _REASON_LIMIT:
        raise Refused(400, f"the reason must be at most {_REASON_LIMIT} characters")
    return reason


def _under_runs(relative: str) -> Path:
    if not isinstance(relative, str) or not relative.strip() or "\x00" in relative:
        raise Refused(400, "the record name is empty")
    rel = Path(relative.strip())
    if rel.is_absolute() or ".." in rel.parts:
        raise Refused(400, "that path is not inside runs/")
    root = detail.runs_root().resolve()
    try:
        path = (root / rel).resolve()
    except OSError as exc:
        raise Refused(400, "that path is not inside runs/") from exc
    if path == root or root not in path.parents:
        raise Refused(400, "that path is not inside runs/")
    if not path.is_dir():
        raise Refused(404, "no such record")
    return path


def _operator(name: object) -> str:
    if not isinstance(name, str):
        raise Refused(400, "the operator name is empty")
    cleaned = operator_name(name.strip())
    if not cleaned or cleaned == "unknown" or "/" in cleaned or "\\" in cleaned or ".." in cleaned:
        raise Refused(400, "the operator name is not a task name")
    return cleaned


def _searches(root: Path) -> list[Path]:
    found: list[Path] = []
    if (root / "state.json").exists() or (root / "result.json").exists():
        found.append(root)
    for current, dirnames, filenames in os.walk(root):
        dirnames[:] = [name for name in dirnames if name not in _SKIP and not name.startswith(".")]
        path = Path(current)
        if path != root and ("state.json" in filenames or "result.json" in filenames):
            found.append(path)
    return found


def _live(directory: Path) -> bool:
    state = detail._load(directory / "state.json")
    result = detail._load(directory / "result.json") if (directory / "result.json").exists() else None
    return detail.liveness(state, result, directory) == "running"


def _relative(path: Path) -> str:
    return path.resolve().relative_to(detail.runs_root().resolve()).as_posix()


def _read_log() -> list[dict]:
    path = log_path()
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except ValueError:
            continue
        if isinstance(item, dict):
            rows.append(item)
    return rows


def list_deletions() -> dict:
    """The log, oldest first. This is the only way the page reads it."""
    return {"deletions": _read_log()}


def _append(entry: dict) -> None:
    path = log_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(entry, ensure_ascii=False, sort_keys=True) + "\n"
    fd = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o644)
    try:
        os.write(fd, line.encode("utf-8"))
        os.fsync(fd)
    finally:
        os.close(fd)


def _archive(paths: list[Path], stamp: str) -> str:
    folder = archive_root() / stamp
    folder.mkdir(parents=True, exist_ok=False)
    root = detail.runs_root().resolve()
    for path in sorted(paths, key=lambda item: len(item.parts), reverse=True):
        relative = path.resolve().relative_to(root).as_posix()
        dest = folder / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(path), str(dest))
    return folder.name


def _runs_for_operator(name: str) -> list[str]:
    refresh()
    with borrow() as index:
        rows = index.conn.execute("SELECT run_id, task, operator FROM runs").fetchall()
    matched = []
    for row in rows:
        label = row["operator"] or operator_name(row["task"])
        if label == name:
            matched.append(row["run_id"])
    return matched


def _client_ip(value: object) -> str:
    if not isinstance(value, str):
        return "unknown"
    host = value.strip()
    return host or "unknown"


def delete_records(kind: object, name: object, reason: object, client_ip: object = "") -> dict:
    """Move finished records out of ``runs/`` and append one log line. A live search refuses the whole request."""
    kept = _reason(reason)
    peer = _client_ip(client_ip)
    if kind == "experiment":
        if not isinstance(name, str):
            raise Refused(400, "the record name is empty")
        root = _under_runs(name)
        targets = [root]
        run_dirs = _searches(root)
        experiment_ids = [name.strip()]
    elif kind == "operator":
        cleaned = _operator(name)
        found = _runs_for_operator(cleaned)
        if not found:
            raise Refused(404, f"no records for operator {cleaned}")
        targets = []
        run_dirs = []
        for run_id in found:
            path = detail.resolve_run(run_id)
            if path is None:
                continue
            targets.append(path)
            run_dirs.append(path)
        if not targets:
            raise Refused(404, f"no records for operator {cleaned}")
        experiment_ids = []
        name = cleaned
    elif kind == "run":
        if not isinstance(name, str):
            raise Refused(400, "the record name is empty")
        root = _under_runs(name)
        if not (root / "state.json").exists() and not (root / "result.json").exists():
            raise Refused(404, "no such record")
        targets = [root]
        run_dirs = [root]
        experiment_ids = []
    else:
        raise Refused(400, "kind must be experiment, operator or run")
    live = [_relative(path) for path in run_dirs if _live(path)]
    if live:
        raise Refused(409, "still running, nothing was deleted: " + ", ".join(live))
    run_ids = [_relative(path) for path in run_dirs]
    stamp = f"{len(_read_log()) + 1:04d}-{time.strftime('%Y%m%dT%H%M%S')}"
    folder = _archive(targets, stamp)
    recorded = name.strip() if isinstance(name, str) else str(name)
    entry = {
        "archive": folder,
        "kind": kind,
        "name": recorded,
        "reason": kept,
        "runs": run_ids,
        "ip": peer,
        "t": time.time(),
        "who": "dashboard",
    }
    try:
        get_index().drop_archived(run_ids, experiment_ids)
    finally:
        _append(entry)
    return {"ok": True, "archive": folder, "kind": kind, "name": recorded, "runs": run_ids}
