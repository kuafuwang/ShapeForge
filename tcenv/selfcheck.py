"""A timed self-check writes one line next to the kernel. A timeout or a crash closes that directory."""

from __future__ import annotations

import json
from pathlib import Path

LOG_NAME = "self_checks.jsonl"
_HANG = frozenset({"timeout", "crash"})


def earlier_hang(directory: Path) -> str | None:
    """The reason a later timed check must be refused, or None when this directory may still time a kernel."""
    path = directory / LOG_NAME
    if not path.is_file():
        return None
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict) and row.get("status") in _HANG:
            return f"self-check already recorded {row.get('status')}; this workspace will not time another kernel"
    return None


def append(directory: Path, row: dict) -> None:
    """Append one result. The orchestrator reads the same file after the session."""
    path = directory / LOG_NAME
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, default=str) + "\n")
