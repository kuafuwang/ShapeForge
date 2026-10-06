"""Stage file written by an evaluation worker. The parent copies it next to the heartbeat."""

from __future__ import annotations

import time
from pathlib import Path

from observe.publish import publish

_path: Path | None = None
_doc: dict = {}


def bind(path: str | Path | None) -> None:
    """Send later :func:`update` calls to this file. ``None`` turns publishing off."""
    global _path, _doc
    _path = Path(path) if path else None
    _doc = {}


def update(**fields) -> None:
    """Merge ``fields`` into the stage file. Never raises. A new ``stage`` records ``entered_at``."""
    global _doc
    if _path is None:
        return
    try:
        now = time.time()
        if "stage" in fields and fields["stage"] != _doc.get("stage"):
            fields["entered_at"] = now
        _doc.update(fields)
        _doc["updated_at"] = now
        publish(_path, _doc)
    except Exception:
        return
