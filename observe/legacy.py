"""Guess experiment, arm and trial for a run directory that has no ``experiment.json``.

This is the only module allowed to guess. Callers mark the row ``source=inferred`` and ``binding=none``.
A comparison may show the group and must not judge it. Delete this file when the old ``runs/`` tree is archived.
"""

from __future__ import annotations

import re
from pathlib import Path

_SWITCH = re.compile(r"^(?P<head>.+)_(?P<arm>on|off)(?:_(?P<trial>\d+))?$")
_TRIAL = re.compile(r"^trial_(?P<trial>\d+)$")


def _has(path: Path, name: str) -> bool:
    return (path / name).is_file()


def infer(run_dir: Path, runs_root: Path) -> dict | None:
    """Return ``experiment_id``, ``arm_id``, ``trial`` and ``kind``, or ``None`` when nothing about the path is recognizable.

    ``None`` means the caller should record the directory as a single search, not as a guessed arm.
    """
    try:
        rel = run_dir.resolve().relative_to(runs_root.resolve())
    except ValueError:
        return None
    parent = run_dir.resolve().parent
    root = runs_root.resolve()
    trial_match = _TRIAL.match(run_dir.name)
    if trial_match and parent.name and _has(parent.parent, "menu.json"):
        return {
            "experiment_id": _rel(parent.parent, runs_root),
            "arm_id": parent.name,
            "trial": int(trial_match.group("trial")),
            "kind": "contract_menu",
        }
    if trial_match and _has(parent.parent, "repeatability.json"):
        return {
            "experiment_id": _rel(parent.parent, runs_root),
            "arm_id": parent.name,
            "trial": int(trial_match.group("trial")),
            "kind": "repeatability",
        }
    switch = _SWITCH.match(run_dir.name)
    if switch and switch.group("arm") in ("on", "off"):
        return {
            "experiment_id": _rel(parent, root) if parent != root else run_dir.name,
            "arm_id": switch.group("arm"),
            "trial": int(switch.group("trial") or 1),
            "kind": "search_ab",
        }
    if parent != root and (_has(parent, "summary.json") or _has(parent, "menu.json") or _has(parent, "repeatability.json")):
        return {
            "experiment_id": _rel(parent, runs_root),
            "arm_id": run_dir.name,
            "trial": 1,
            "kind": "suite",
        }
    return None


def _rel(path: Path, root: Path) -> str:
    try:
        text = path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.name
    return text or "."
