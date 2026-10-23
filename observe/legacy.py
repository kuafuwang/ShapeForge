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
    """@brief 判断目录下是否存在指定名字的文件。

    @param path 目录路径。
    @param name 文件名。
    @return 是普通文件时为 True。
    """
    return (path / name).is_file()


def infer(run_dir: Path, runs_root: Path) -> dict | None:
    """@brief 为一个没有 ``experiment.json`` 的运行目录推测所属实验、arm 与 trial。

    返回 ``experiment_id``、``arm_id``、``trial`` 与 ``kind``；路径完全无法识别时返回 ``None``，
    此时调用方应把该目录记为单次搜索，而不是一个推测出来的 arm。

    @param run_dir 待推测的运行目录。
    @param runs_root ``runs/`` 根目录，用于计算相对的 experiment id。
    @return 推测结果字典；目录不在 ``runs_root`` 下或命名不匹配任何模式时返回 ``None``。
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
    """@brief 取 ``path`` 相对 ``root`` 的 POSIX 路径；无法相对化时退回目录名。

    @param path 目标路径。
    @param root 基准根目录。
    @return 相对路径文本，根目录自身返回 ``"."``；不在根目录下时返回 ``path.name``。
    """
    try:
        text = path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.name
    return text or "."
