"""Comparability key for cross-run analysis: the run's own config, not the experiment record.

A run is comparable with another when the task, dtype, size, budget and protocol epoch agree (the
frozen experiment conditions are folded in as well). An arm's ``delta`` is deliberately not part of
the key: it is constant inside an arm, and ``binding=drift`` already compares it against the frozen
conditions at ingest time, so it is the authority on whether a run may be compared at all.
"""

from __future__ import annotations

import hashlib
import json
import math

# 只把“题目/规模/预算”这类决定可比性的冻结字段折进 key；delta、预算事件、运行时刻都不进。
FROZEN_FIELDS = ("task", "tasks", "dtype", "size", "budget_usd")


def _number(value: object) -> float | int | None:
    """@brief 只接受真正的数值（bool 不算），并把浮点噪声抹到 6 位。

    @param value 原始值。
    @return 规范化后的数值，或 None。
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if not math.isfinite(float(value)):
        return None
    return round(float(value), 6)


def protocol_epoch(row: dict) -> str:
    """@brief 运行的协议纪元；``flame_chase_1`` 记成 ``2``，与索引的 ``protocol_epoch`` 口径一致。

    @param row 运行行（可含 ``protocol_epoch`` 或原始 ``protocol_version``）。
    @return 纪元字符串。
    """
    recorded = row.get("protocol_epoch")
    if isinstance(recorded, str) and recorded:
        return recorded
    version = str(row.get("protocol_version") or "")
    return "2" if version == "flame_chase_1" else version


def config_key(row: dict, frozen: dict | None = None) -> str:
    """@brief 可比配置键：``sha256(canonical_json(...))[:12]``，不含 ``delta``。

    参与哈希的只有决定“能不能放在一起比”的字段：运行的 task/dtype/size/budget_usd/protocol_epoch，
    加上冻结条件里的题目/规模/预算字段。时间、花费明细、发布结果与 ``delta`` 都不参与。

    @param row 运行行（索引的一行）。
    @param frozen 实验的冻结条件，可为 None。
    @return 12 个十六进制字符的配置键。
    """
    conditions = frozen if isinstance(frozen, dict) else {}
    payload = {
        "task": str(row.get("task") or ""),
        "dtype": str(row.get("dtype") or ""),
        "size": str(row.get("size") or ""),
        "budget_usd": _number(row.get("budget_usd")),
        "protocol_epoch": protocol_epoch(row),
        "frozen": {name: conditions[name] for name in FROZEN_FIELDS if conditions.get(name) not in (None, "")},
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]


def comparison_group(row: dict) -> tuple[str, str, str, str, str, str]:
    """@brief 分析面板的跨运行分组键：``(task, dtype, size, target_id, protocol_epoch, arm_id)``。

    题目/规模/目标/协议纪元任一不同就不该分到同一组；``arm_id`` 缺失时用空串（单 run 来源）。

    @param row 运行行。
    @return 六元组分组键。
    """
    return (
        str(row.get("task") or ""),
        str(row.get("dtype") or ""),
        str(row.get("size") or ""),
        str(row.get("target_id") or ""),
        protocol_epoch(row),
        str(row.get("arm_id") or ""),
    )
