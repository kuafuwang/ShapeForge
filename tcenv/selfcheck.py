"""Timed self-check budgets and their host-authoritative records.

A timed self-check writes one line next to the kernel for agent feedback. The
same line is also written to a host-side state directory. Quota and hang checks
prefer that host record, so clearing the workspace log cannot reset the budget.
The host directory is derived from the canonical workspace path and is not the
agent's submission directory.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import pwd
from pathlib import Path

LOG_NAME = "self_checks.jsonl"
_HANG = frozenset({"timeout", "crash"})

# 自测是给代理迭代用的，不是让它把整场会话花在计时循环里。
MAX_CHECKS = 20
MAX_GPU_S = 120.0


def _authority_root() -> Path:
    # Never derive the authority root from the evaluated process environment.
    # An agent could otherwise set TCENV_SELFCHECK_HOME back into its workspace.
    # Do not use /tmp: the hmz fence grants temp writes to an agent, so that
    # would let it delete the very record that is supposed to outrank its
    # workspace log. The user cache is host-owned and outside the fence.
    # The local CLI runs inside the agent process, so HOME/XDG_CACHE_HOME are
    # agent-controlled too. Resolve the account home from the OS database.
    cache = Path(pwd.getpwuid(os.getuid()).pw_dir) / ".cache"
    return cache / "taic-selfcheck-state"


def authority_log(directory: Path) -> Path:
    """Return the host-side log for a canonical self-check directory."""
    path = Path(directory)
    # Resolve only the host-controlled parent. Resolving the final component
    # would let an agent retarget .selfcheck with a symlink and change the key.
    canonical = path.parent.resolve() / path.name
    key = hashlib.sha256(str(canonical).encode("utf-8")).hexdigest()
    return _authority_root() / key / LOG_NAME


def _read_rows(directory: Path) -> tuple[list[dict], bool]:
    authority = authority_log(directory)
    if authority.is_file():
        path, strict = authority, True
    else:
        path, strict = directory / LOG_NAME, False
    if not path.is_file():
        return [], strict
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            if strict:
                raise ValueError(f"corrupt host selfcheck record: {path}") from exc
            continue
        if not isinstance(row, dict):
            if strict:
                raise ValueError(f"non-object host selfcheck record: {path}")
            continue
        rows.append(row)
    return rows, strict


def earlier_hang(directory: Path) -> str | None:
    """@brief 本目录是否已记下一次超时或崩溃，从而禁止再次计时。

    The reason a later timed check must be refused, or None when this directory may still time a kernel.

    @param directory 自测记录所在目录。
    @return 拒绝原因字符串；目录尚可计时时为 None。
    """
    rows, _ = _read_rows(directory)
    for row in rows:
        if row.get("status") in _HANG:
            return f"self-check already recorded {row.get('status')}; this workspace will not time another kernel"
    return None


def budget_reason(directory: Path) -> str | None:
    """@brief 本目录是否已用完计时自测的次数或 GPU 时间预算。

    The reason a timed check must be refused for running out of self-check budget,
    or None when this directory may still time a kernel.

    @param directory 自测记录所在目录。
    @return 拒绝原因字符串；预算尚足时为 None。
    """
    rows, _ = _read_rows(directory)
    checks = len(rows)
    gpu_s = 0.0
    for row in rows:
        elapsed = row.get("elapsed_s")
        if elapsed is not None and (isinstance(elapsed, bool) or not isinstance(elapsed, (int, float)) or not math.isfinite(elapsed) or elapsed < 0):
            return "self-check record has invalid elapsed_s; this workspace will not time another kernel"
        if isinstance(elapsed, (int, float)) and not isinstance(elapsed, bool):
            gpu_s += float(elapsed)
    if checks >= MAX_CHECKS:
        return f"this workspace already ran {checks} timed self-checks (limit {MAX_CHECKS}); hand the kernel back and let the search measure it"
    if gpu_s >= MAX_GPU_S:
        return f"this workspace already spent {gpu_s:.0f}s in timed self-checks (limit {MAX_GPU_S:.0f}s); hand the kernel back and let the search measure it"
    return None


def _append(path: Path, row: dict, *, private: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if private:
        try:
            path.parent.chmod(0o700)
        except OSError:
            pass
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, default=str) + "\n")
    if private:
        try:
            path.chmod(0o600)
        except OSError:
            pass


def _append_mirror(directory: Path, row: dict) -> None:
    """Best-effort workspace mirror; never follow an agent-created symlink."""
    try:
        if directory.is_symlink():
            return
        directory.mkdir(parents=True, exist_ok=True)
        if directory.is_symlink() or not directory.is_dir():
            return
        target = directory / LOG_NAME
        flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND
        flags |= getattr(os, "O_NOFOLLOW", 0)
        fd = os.open(target, flags, 0o600)
        try:
            with os.fdopen(fd, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, default=str) + "\n")
            fd = -1
        finally:
            if fd >= 0:
                os.close(fd)
    except OSError:
        # The mirror is diagnostic only; the host record above is authoritative.
        return


def append(directory: Path, row: dict) -> None:
    """@brief 向自测记录文件追加一条结果，并同步 host 权威记录。

    Append one result. The directory is created when the first timed check arrives.
    The host record is authoritative for subsequent budget/hang decisions.

    @param directory 自测记录目录，不存在时会创建。
    @param row 要写入的一行结果字典。
    """
    # Persist the host record before touching the mirror. Even a non-directory
    # .selfcheck path must not erase evidence for an evaluation that already ran.
    _append(authority_log(directory), row, private=True)
    _append_mirror(directory, row)
