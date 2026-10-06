"""Atomic JSON snapshots and an append-only GPU event log.

Every function swallows errors. A dashboard file must never fail a search or an evaluation.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path


def cache_root() -> Path:
    env = os.environ.get("AI_LOWERLING_OBSERVE_DIR")
    if env:
        return Path(env)
    return Path.home() / ".cache" / "ai_lowerling"


def gpu_dir() -> Path:
    return cache_root() / "gpu.d"


def observe_dir() -> Path:
    return cache_root() / "observe"


def gpu_events_path() -> Path:
    return observe_dir() / "gpu_events.jsonl"


def tcenv_live_path(pid: int | None = None) -> Path:
    return observe_dir() / f"tcenv-{pid or os.getpid()}.json"


def publish(path: Path, doc: dict) -> None:
    """Write ``doc`` as JSON via a temp file in the same directory, then replace. Never raises."""
    try:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = json.dumps(doc, ensure_ascii=False, default=str)
        tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        tmp.write_text(data, encoding="utf-8")
        os.replace(tmp, path)
    except Exception:
        return


def append_jsonl(path: Path, doc: dict) -> None:
    """Append one JSON line. Never raises."""
    try:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(doc, ensure_ascii=False, default=str) + "\n"
        with path.open("a", encoding="utf-8") as handle:
            handle.write(line)
    except Exception:
        return


def job_state(path: Path, *, script: str, current, queue, done, phase: str = "run") -> None:
    """Live status of a long experiment script."""
    publish(
        path,
        {
            "pid": os.getpid(),
            "updated_at": time.time(),
            "script": script,
            "phase": phase,
            "current": current,
            "queue": queue,
            "done": done,
        },
    )


def _gpu_file(pid: int, waiter: bool) -> Path:
    name = f"{pid}.wait.json" if waiter else f"{pid}.json"
    return gpu_dir() / name


def _unlink(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except Exception:
        return


def note_gpu(kind: str, mode: str, what: str) -> None:
    """Record a GPU-lock wait, acquire, release, or give-up. ``kind`` is one of those four words."""
    try:
        pid = os.getpid()
        now = time.time()
        if kind == "wait":
            publish(_gpu_file(pid, waiter=True), {"pid": pid, "role": "waiter", "mode": mode, "what": what, "since": now})
        elif kind == "acquire":
            _unlink(_gpu_file(pid, waiter=True))
            publish(_gpu_file(pid, waiter=False), {"pid": pid, "role": "holder", "mode": mode, "what": what, "since": now})
        elif kind in ("release", "give_up"):
            _unlink(_gpu_file(pid, waiter=True))
            if kind == "release":
                _unlink(_gpu_file(pid, waiter=False))
        else:
            return
        append_jsonl(gpu_events_path(), {"t": now, "pid": pid, "kind": kind, "mode": mode, "what": what})
    except Exception:
        return
