"""A timed self-check writes one line next to the kernel. A timeout or a crash closes that directory."""

from __future__ import annotations

import json
from pathlib import Path

LOG_NAME = "self_checks.jsonl"
_HANG = frozenset({"timeout", "crash"})

# 自测是给代理迭代用的，不是让它把整场会话花在计时循环里。
# 依据：一次 `--timed` 是 15 次重复、与基线交错，实测约 0.2–1 秒 GPU；20 次 ≈ 4–20 秒，
# 相对它拿到的 25 分钟思考时间是很小的一份，但已经够改三到五版。120 秒是同一件事的另一种量法
# （核变慢时次数不变而时间会涨）。两者都是"不许拿它当训练集"的上限，不是性能目标。
# 超限后不自动重置：代理仍可以交卷，外层照常测；只是它自己不能再测。
MAX_CHECKS = 20
MAX_GPU_S = 120.0


def earlier_hang(directory: Path) -> str | None:
    """@brief 本目录是否已记下一次超时或崩溃，从而禁止再次计时。

    The reason a later timed check must be refused, or None when this directory may still time a kernel.

    @param directory 自测记录所在目录。
    @return 拒绝原因字符串；目录尚可计时时为 None。
    """
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


def budget_reason(directory: Path) -> str | None:
    """@brief 本目录是否已用完计时自测的次数或 GPU 时间预算。

    The reason a timed check must be refused for running out of self-check budget, or None.

    @param directory 自测记录所在目录。
    @return 拒绝原因字符串；预算尚足时为 None。
    """
    path = directory / LOG_NAME
    if not path.is_file():
        return None
    checks = 0
    gpu_s = 0.0
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(row, dict):
            continue
        checks += 1
        elapsed = row.get("elapsed_s")
        if isinstance(elapsed, (int, float)) and not isinstance(elapsed, bool):
            gpu_s += float(elapsed)
    if checks >= MAX_CHECKS:
        return f"this workspace already ran {checks} timed self-checks (limit {MAX_CHECKS}); hand the kernel back and let the search measure it"
    if gpu_s >= MAX_GPU_S:
        return f"this workspace already spent {gpu_s:.0f}s in timed self-checks (limit {MAX_GPU_S:.0f}s); hand the kernel back and let the search measure it"
    return None


def append(directory: Path, row: dict) -> None:
    """@brief 向自测记录文件追加一条结果。

    Append one result. The directory is created when the first timed check arrives.

    @param directory 自测记录所在目录，不存在时会创建。
    @param row 要写入的一行结果字典。
    """
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / LOG_NAME
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, default=str) + "\n")
