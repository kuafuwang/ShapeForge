"""In-process events for progress and dashboard files. The standard library only.

With no subscriber, :func:`emit` does nothing. :func:`install_file_log` is the subscriber that writes the JSON files the dashboard reads. Search and evaluation call this module; they do not import the web package.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Callable

Listener = Callable[[str, dict], None]
_listeners: list[Listener] = []
_file_log_on = False

_progress_path: Path | None = None
_progress_doc: dict = {}


def subscribe(listener: Listener) -> None:
    """Register ``listener``. A second registration of the same function is ignored."""
    if listener not in _listeners:
        _listeners.append(listener)


def reset() -> None:
    """Drop every subscriber and the in-memory progress document. Tests use this."""
    global _file_log_on, _progress_path, _progress_doc
    _listeners.clear()
    _file_log_on = False
    _progress_path = None
    _progress_doc = {}


def emit(kind: str, **payload) -> None:
    """Tell subscribers. No subscriber means this returns without writing."""
    for listener in list(_listeners):
        try:
            listener(kind, payload)
        except Exception:
            continue


def install_file_log() -> None:
    """Subscribe the file writer. Calling it again does not add a second writer."""
    global _file_log_on
    if _file_log_on:
        return
    subscribe(_on_event)
    _file_log_on = True


def cache_root() -> Path:
    """Dashboard cache. ``AI_LOWERLING_OBSERVE_DIR`` overrides it. Not inside ``runs/``."""
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
    """Ask the file log to replace ``path`` with ``doc``. Never raises. A no-op until :func:`install_file_log`."""
    try:
        install_file_log()
        emit("snapshot", path=path, doc=doc)
    except Exception:
        return


def append_jsonl(path: Path, doc: dict) -> None:
    """Ask the file log to append one JSON line. Never raises."""
    try:
        install_file_log()
        emit("append", path=path, doc=doc)
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


def note_gpu(kind: str, mode: str, what: str) -> None:
    """Record a GPU-lock wait, acquire, release, or give-up. Never raises."""
    try:
        install_file_log()
        emit("gpu", action=kind, mode=mode, what=what)
    except Exception:
        return


def bind_progress(path: str | Path | None) -> None:
    """Send later :func:`update_progress` calls to this file. ``None`` turns them off."""
    try:
        install_file_log()
        emit("progress-bind", path=None if path is None else str(path))
    except Exception:
        return


def update_progress(**fields) -> None:
    """Merge ``fields`` into the stage file. Never raises. A new ``stage`` records ``entered_at``."""
    try:
        install_file_log()
        emit("progress", fields=fields)
    except Exception:
        return


def _on_event(kind: str, payload: dict) -> None:
    if kind == "snapshot":
        _write_json(payload["path"], payload["doc"])
    elif kind == "append":
        _append_jsonl(payload["path"], payload["doc"])
    elif kind == "gpu":
        _record_gpu(payload["action"], payload["mode"], payload["what"])
    elif kind == "progress-bind":
        _bind_progress(payload.get("path"))
    elif kind == "progress":
        _update_progress(**payload.get("fields") or {})


def _write_json(path: Path, doc: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(doc, ensure_ascii=False, default=str)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(data, encoding="utf-8")
    os.replace(tmp, path)


def _append_jsonl(path: Path, doc: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(doc, ensure_ascii=False, default=str) + "\n"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line)


def _gpu_file(pid: int, waiter: bool) -> Path:
    name = f"{pid}.wait.json" if waiter else f"{pid}.json"
    return gpu_dir() / name


def _unlink(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except Exception:
        return


def _record_gpu(kind: str, mode: str, what: str) -> None:
    pid = os.getpid()
    now = time.time()
    if kind == "wait":
        _write_json(_gpu_file(pid, waiter=True), {"pid": pid, "role": "waiter", "mode": mode, "what": what, "since": now})
    elif kind == "acquire":
        _unlink(_gpu_file(pid, waiter=True))
        _write_json(_gpu_file(pid, waiter=False), {"pid": pid, "role": "holder", "mode": mode, "what": what, "since": now})
    elif kind in ("release", "give_up"):
        _unlink(_gpu_file(pid, waiter=True))
        if kind == "release":
            _unlink(_gpu_file(pid, waiter=False))
    else:
        return
    _append_jsonl(gpu_events_path(), {"t": now, "pid": pid, "kind": kind, "mode": mode, "what": what})


def _bind_progress(path: str | None) -> None:
    global _progress_path, _progress_doc
    _progress_path = Path(path) if path else None
    _progress_doc = {}


def _update_progress(**fields) -> None:
    global _progress_doc
    if _progress_path is None:
        return
    now = time.time()
    if "stage" in fields and fields["stage"] != _progress_doc.get("stage"):
        fields["entered_at"] = now
    _progress_doc.update(fields)
    _progress_doc["updated_at"] = now
    _write_json(_progress_path, _progress_doc)
