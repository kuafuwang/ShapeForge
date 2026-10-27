"""Guess experiment, arm and trial for a run directory that has no ``experiment.json``.

This is the only module allowed to guess. Callers mark the row ``source=inferred`` and ``binding=none``.
A comparison may show the group and must not judge it. Delete this file when the old ``runs/`` tree is archived.
"""

from __future__ import annotations

import os
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


def _absolute(path: Path | str, root: str) -> str:
    """@brief 把路径规整成绝对文本；相对路径按根拼接，符号链接只在必要时解析。

    @param path 待规整路径。
    @param root 已解析的 ``runs/`` 根文本。
    @return ``normpath`` 后的绝对路径。
    """
    text = os.fspath(path)
    if not os.path.isabs(text):
        text = os.path.join(root, text)
    text = os.path.normpath(text)
    if os.path.islink(text):
        return os.path.normpath(os.fspath(Path(text).resolve()))
    return text


def _relative(text: str, root: str) -> str | None:
    """@brief 字符串相对路径；不在根内时返回 None。

    @param text 已规整的绝对路径。
    @param root 已规整的根路径。
    @return 相对路径（根自身为 ``"."``）或 None。
    """
    if text == root:
        return "."
    prefix = root + os.sep
    if not text.startswith(prefix):
        return None
    return text[len(prefix):].replace(os.sep, "/")


def infer(run_dir: Path, runs_root: Path, *, resolved_root: Path | None = None) -> dict | None:
    """@brief 为一个没有 ``experiment.json`` 的运行目录推测所属实验、arm 与 trial。

    返回 ``experiment_id``、``arm_id``、``trial`` 与 ``kind``；路径完全无法识别时返回 ``None``，
    此时调用方应把该目录记为单次搜索，而不是一个推测出来的 arm。

    @param run_dir 待推测的运行目录。
    @param runs_root ``runs/`` 根目录，用于计算相对的 experiment id。
    @param resolved_root 已解析的 ``runs/`` 根；索引器传入缓存值，避免每次摄取重复 ``realpath``。
    @return 推测结果字典；目录不在 ``runs_root`` 下或命名不匹配任何模式时返回 ``None``。
    """
    root = Path(resolved_root) if resolved_root is not None else Path(runs_root).resolve()
    root_text = os.path.normpath(os.fspath(root))
    run_text = _absolute(run_dir, root_text)
    if _relative(run_text, root_text) is None:
        return None
    parent_text = os.path.dirname(run_text)
    run_name = os.path.basename(run_text)
    parent = Path(parent_text)
    root = Path(root_text)
    trial_match = _TRIAL.match(run_name)
    if trial_match and parent.name and _has(parent.parent, "menu.json"):
        return {
            "experiment_id": _rel(parent.parent, root),
            "arm_id": parent.name,
            "trial": int(trial_match.group("trial")),
            "kind": "contract_menu",
        }
    if trial_match and _has(parent.parent, "repeatability.json"):
        return {
            "experiment_id": _rel(parent.parent, root),
            "arm_id": parent.name,
            "trial": int(trial_match.group("trial")),
            "kind": "repeatability",
        }
    switch = _SWITCH.match(run_name)
    if switch and switch.group("arm") in ("on", "off"):
        return {
            "experiment_id": _rel(parent, root) if parent != root else run_name,
            "arm_id": switch.group("arm"),
            "trial": int(switch.group("trial") or 1),
            "kind": "search_ab",
        }
    if parent != root and (_has(parent, "summary.json") or _has(parent, "menu.json") or _has(parent, "repeatability.json")):
        return {
            "experiment_id": _rel(parent, root),
            "arm_id": run_name,
            "trial": 1,
            "kind": "suite",
        }
    return None


def _rel(path: Path, root: Path) -> str:
    """@brief 取 ``path`` 相对 ``root`` 的 POSIX 路径；无法相对化时退回目录名。

    @param path 目标路径。
    @param root 基准根目录（已解析）。
    @return 相对路径文本，根目录自身返回 ``"."``；不在根目录下时返回 ``path.name``。
    """
    root_text = os.path.normpath(os.fspath(root))
    text = _absolute(path, root_text)
    relative = _relative(text, root_text)
    return relative if relative is not None else os.path.basename(text)
