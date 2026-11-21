"""Frozen H0 contracts for the read-only harness scorecard.

These helpers define units and missing-data semantics only. They do not compute
the H1 scorecard, open observe's index, use a GPU, or call a model.
"""

from __future__ import annotations

import math
import re
from typing import Any

SCHEMA = "shapeforge.harness-scorecard-inventory.1"
IDENTITY_FIELDS = ("candidate", "geometry", "bundle", "artifact", "native", "evaluation")
MIN_FIELD_COVERAGE = 0.8
SUFFICIENCY_THRESHOLDS = {
    "run_level_min_units": 5,
    "session_level_min_units": 5,
    "claim_level_min_units": 5,
    "delivery_receipt_min_units": 5,
    "paired_conclusion_min_units": 5,
    "min_field_coverage": MIN_FIELD_COVERAGE,
    "cross_task_min_tasks": 2,
    "cross_backend_min_real_backends": 2,
}
_RETRY = re.compile(r"\.retry[0-9]+$")


def metric_status(metric: str, *, units: int, known_units: int) -> str:
    """@brief 按 H0 冻结的独立单位和字段覆盖率判定指标是否充足。"""
    if units <= 0 or known_units <= 0:
        return "unavailable"
    minimum = int(SUFFICIENCY_THRESHOLDS[metric])
    if known_units < minimum or known_units / units < MIN_FIELD_COVERAGE:
        return "insufficient"
    return "measured"


def compare_identity(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    """@brief 比较六元身份；部分相同是 partial，缺失不伪装成完整 match。"""
    common = [name for name in IDENTITY_FIELDS if left.get(name) is not None and right.get(name) is not None]
    conflicts = [name for name in common if left[name] != right[name]]
    if conflicts:
        return {"status": "mismatch", "identity_scope": "partial" if len(common) < len(IDENTITY_FIELDS) else "full", "conflicts": conflicts}
    if len(common) == len(IDENTITY_FIELDS):
        return {"status": "match", "identity_scope": "full", "conflicts": []}
    if common:
        return {"status": "partial", "identity_scope": "partial", "conflicts": []}
    return {"status": "unavailable", "identity_scope": "unavailable", "conflicts": []}


def canonical_session_label(label: Any) -> str:
    """@brief 将 ``r1.p0.retry1`` 映射回逻辑会话，同时保留原标签给原始行。"""
    return _RETRY.sub("", str(label or ""))


def resolve_candidate_status(feedback_status: Any, outer_status: Any) -> dict[str, Any]:
    """@brief 外层候选报告优先；与 feedback 冲突时必须显式保留冲突。"""
    feedback = str(feedback_status or "")
    outer = str(outer_status or "")
    if outer:
        return {"status": outer, "source": "outer_report", "conflict": bool(feedback and feedback != outer)}
    if feedback:
        return {"status": feedback, "source": "feedback", "conflict": False}
    return {"status": None, "source": "unavailable", "conflict": False}


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def cost_metrics(spent_usd: Any, model_ok: int, *, incomplete: Any) -> dict[str, Any]:
    """@brief 分开账本估值与精确成本；不完整时精确值永远是 None。"""
    spent = _number(spent_usd)
    ok = max(0, int(model_ok))
    ledger = spent / ok if spent is not None and ok > 0 else None
    return {
        "ledger_cost_per_ok": ledger,
        "exact_cost_per_ok": None if incomplete is not False else ledger,
        "ledger_spent_without_ok_usd": spent if spent is not None and ok == 0 else (0.0 if spent is not None else None),
        "cost_complete": False if incomplete is True else (True if incomplete is False else None),
    }


def evidence_kind(*records: dict[str, Any]) -> str | None:
    """@brief 只有显式 measured/simulated 声明能决定真实或模拟；缺省为 None。"""
    measured = False
    for record in records:
        if not isinstance(record, dict):
            continue
        if record.get("simulated") is True:
            return "simulated"
        for key in ("measurement_kind", "measurement_type", "evidence_level"):
            value = str(record.get(key) or "")
            if value == "simulated":
                return "simulated"
            measured = measured or value == "measured"
    return "real" if measured else None


__all__ = [
    "IDENTITY_FIELDS", "MIN_FIELD_COVERAGE", "SCHEMA", "SUFFICIENCY_THRESHOLDS",
    "canonical_session_label", "compare_identity", "cost_metrics", "evidence_kind",
    "metric_status", "resolve_candidate_status",
]
