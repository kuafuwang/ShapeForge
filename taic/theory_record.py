"""Checks for a theory-practice record.

The prose lives in the architecture note. This module only rejects a record
that would present an explanation as a measurement or mark a claim supported
when it has no evidence pointer. It does not inject text, grant trust, or
call a model.
"""

from __future__ import annotations

import datetime
import re
from pathlib import Path

STATUSES = frozenset({"proposed", "supported", "refuted", "superseded"})
KINDS = frozenset({"观察事实", "理论解释", "人为决策"})
_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")


def _evidence_file_reviewer(reviewer: str, evidence: str) -> bool:
    """@brief 复核人是不是一份证据文件，而不是一次复核。"""
    text = reviewer.strip()
    if not text:
        return False
    if "](" in text:
        return True
    normalized = text.replace("\\", "/")
    if normalized.endswith((".md", ".markdown")):
        return True
    if " " not in text and "/" in normalized:
        return True
    evidence_text = evidence.strip()
    if evidence_text and text == evidence_text:
        return True
    for match in re.findall(r"\(([^)]+)\)", evidence_text):
        target = match.strip()
        if text == target or text == Path(target).name:
            return True
    return False


def validate_record(row: dict) -> list[str]:
    """@brief 返回这条理论记录违反合同的地方。空列表表示通过。

    ``supported`` 和 ``refuted`` 必须有证据指针和适用范围。
    没有证据只能是 ``proposed``。``superseded`` 必须留下被替换的命题和指针。
    理论解释不能被标成观察事实。每种状态都要有可追踪的复核人、真实复核日期，
    以及反例或替代解释。占位复核人（空、以「待」开头、「人工复核」）和无效日期
    一律拒绝。证据文件、证据路径或只指向证据的链接不能充当复核人。
    没有证据的命题只能是 ``proposed``。

    @param row 一条理论记录。
    @return 违规说明列表。
    """
    errors: list[str] = []
    kind = str(row.get("kind") or "")
    status = str(row.get("status") or "")
    evidence = str(row.get("evidence") or "").strip()
    scope = str(row.get("scope") or "").strip()
    if kind not in KINDS:
        errors.append("kind must be 观察事实, 理论解释, or 人为决策")
    if status not in STATUSES:
        errors.append("status must be proposed, supported, refuted, or superseded")
    if status in {"supported", "refuted"} and not evidence:
        errors.append("empty evidence cannot be supported or refuted")
    if status in {"supported", "refuted"} and not scope:
        errors.append("a supported or refuted claim needs a scope")
    if status == "proposed" and evidence and row.get("promote_without_scope"):
        errors.append("evidence without a declared scope cannot be promoted")
    if status == "superseded" and (not str(row.get("replaced_by") or "").strip() or not str(row.get("prior") or "").strip()):
        errors.append("superseded keeps the old proposition and a replacement pointer")
    if kind == "理论解释" and row.get("as_fact") is True:
        errors.append("a theoretical explanation cannot be rewritten as an observed fact")
    if kind == "人为决策" and row.get("as_effect") is True:
        errors.append("a human decision is not an effect measurement")
    reviewer = str(row.get("reviewer") or "").strip()
    if (
        not reviewer
        or reviewer == "人工复核"
        or reviewer == "待复核"
        or reviewer.startswith("待")
        or "待本阶段" in reviewer
    ):
        errors.append("a record needs a traceable reviewer; 人工复核 is not a review")
    elif _evidence_file_reviewer(reviewer, evidence):
        errors.append("an evidence file is not a reviewer")
    reviewed_on = str(row.get("reviewed_on") or "").strip()
    if _DATE.fullmatch(reviewed_on) is None:
        errors.append("a record needs a review date")
    else:
        try:
            datetime.date.fromisoformat(reviewed_on)
        except ValueError:
            errors.append("a record needs a review date")
    if not str(row.get("counterexample") or "").strip() and not str(row.get("alternative") or "").strip():
        errors.append("a record needs a counterexample or an alternative explanation")
    return errors
