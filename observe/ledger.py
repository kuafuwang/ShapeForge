"""Spend totals stored in the observe index. A later reading may raise a total. It never lowers one."""

from __future__ import annotations

import time


def operator_name(task: str | None) -> str:
    """@brief 取一次搜索所属的 operator；tile 变体沿用其 operator。

    @param task 任务名，形如 ``name@variant``，可为 None。
    @return operator 名；任务为空时返回 ``"unknown"``。
    """
    name = (task or "").strip()
    if not name:
        return "unknown"
    return name.split("@", 1)[0]


def month_of(started_at: float | None) -> str:
    """@brief 把起始时间格式化为本地时区的 ``YYYY-MM``；缺失或非法时返回 ``unknown``。

    @param started_at 起始 unix 时间戳，可为 None。
    @return 本地时区月份字符串 ``YYYY-MM``，无法解析时为 ``"unknown"``。
    """
    if started_at is None:
        return "unknown"
    try:
        return time.strftime("%Y-%m", time.localtime(float(started_at)))
    except (TypeError, ValueError, OSError):
        return "unknown"
