"""G4 evidence levels: classify who ran evidence, with simulation kept separate.

The source is supplied by a host call site.  A model-authored record cannot
promote itself by writing ``evidence_level=host_process``.  Declared simulation
can only move a record into the separate, non-promotable simulated category.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Mapping

MODEL_CLAIM = "model_claim"
MODEL_SCRIPT = "model_script"
FENCED_HOST_TOOL = "fenced_host_tool"
HOST_PROCESS = "host_process"
HUMAN_CONFIRMED = "human_confirmed"
SIMULATED = "simulated"
LEVELS = (MODEL_CLAIM, MODEL_SCRIPT, FENCED_HOST_TOOL, HOST_PROCESS, HUMAN_CONFIRMED, SIMULATED)
RANK = {
    MODEL_CLAIM: 0,
    MODEL_SCRIPT: 1,
    FENCED_HOST_TOOL: 2,
    HOST_PROCESS: 3,
    HUMAN_CONFIRMED: 4,
}
_SOURCE_LEVEL = {
    "model_statement": MODEL_CLAIM,
    "model_probe": MODEL_SCRIPT,
    "model_script": MODEL_SCRIPT,
    "workspace_self_check": MODEL_SCRIPT,
    "host_bridge_self_check": FENCED_HOST_TOOL,
    "fenced_paired_compare": FENCED_HOST_TOOL,
    "outer_candidate_report": HOST_PROCESS,
    "host_paired_comparison": HOST_PROCESS,
    "human_confirmation": HUMAN_CONFIRMED,
}


def _declares_simulation(record: Mapping[str, Any] | None, *, nested: bool) -> bool:
    """Return whether this mapping declares simulation.

    ``measurement_kind`` and ``measurement_type`` are checked separately. A
    truthy kind such as ``measured`` must not hide ``measurement_type``.
    Nested ``provenance`` and ``evidence_provenance`` are one level deep.
    """
    if not isinstance(record, Mapping):
        return False
    if record.get("simulated") is True:
        return True
    if str(record.get("measurement_kind") or "") == SIMULATED or str(record.get("measurement_type") or "") == SIMULATED:
        return True
    if str(record.get("evidence_level") or "") == SIMULATED:
        return True
    if nested and str(record.get("level") or "") == SIMULATED:
        return True
    if nested:
        return False
    for key in ("provenance", "evidence_provenance"):
        extra = record.get(key)
        if isinstance(extra, Mapping) and _declares_simulation(extra, nested=True):
            return True
    return False


def declares_simulation(record: Mapping[str, Any] | None) -> bool:
    """Return whether the record or its evaluator provenance declares simulation."""
    return _declares_simulation(record, nested=False)


def grade_evidence(
    source: str,
    *,
    record: Mapping[str, Any] | None = None,
    simulated: bool = False,
    host_verified: bool = False,
    confirmation: Mapping[str, Any] | None = None,
) -> dict:
    """Grade a host-identified source without accepting a self-declared level.

    ``human_confirmation`` additionally requires an explicit host verification
    and a confirmation with a person and evidence reference.  Conversion of free
    text into a proposal is not such a confirmation.
    """
    normalized = str(source or "unknown")
    runner_level = _SOURCE_LEVEL.get(normalized, MODEL_CLAIM)
    reason = "" if normalized in _SOURCE_LEVEL else "unrecognized evidence source; fail closed to model_claim"
    if runner_level == HUMAN_CONFIRMED:
        valid_confirmation = (
            host_verified
            and isinstance(confirmation, Mapping)
            and bool(str(confirmation.get("confirmed_by") or "").strip())
            and bool(str(confirmation.get("evidence_ref") or "").strip())
        )
        if not valid_confirmation:
            runner_level = MODEL_CLAIM
            normalized = "unverified_human_confirmation"
            reason = "human confirmation needs host verification, confirmed_by and evidence_ref"
    is_simulated = bool(simulated) or declares_simulation(record)
    level = SIMULATED if is_simulated else runner_level
    return {
        "level": level,
        "runner_level": runner_level,
        "source": normalized,
        "rank": None if is_simulated else RANK[runner_level],
        "simulated": is_simulated,
        "promotion_allowed": bool(not is_simulated and runner_level == HUMAN_CONFIRMED),
        "permission_effect": "evidence_label_only",
        "reason": "simulated evidence is separate and cannot be promoted" if is_simulated else reason,
    }


def evidence_fields(grade: Mapping[str, Any]) -> dict:
    """Return the stable top-level fields persisted beside an evidence record."""
    return {
        "evidence_level": str(grade.get("level") or MODEL_CLAIM),
        "evidence_provenance": dict(grade),
    }


def grade_proposal(row: Mapping[str, Any], *, trusted_report: Mapping[str, Any] | None = None, trusted_reason: str = "") -> dict:
    """Grade a proposal from host-computed probe/report matching, not model fields."""
    report_simulated = trusted_report is not None and declares_simulation(trusted_report)
    simulated = declares_simulation(row) or report_simulated
    probe = bool(str(row.get("probe_sha256") or "").strip())
    source = "model_probe" if probe else "model_statement"
    evidence_record: Mapping[str, Any] | None = trusted_report if report_simulated else row
    if row.get("trusted_record") is True:
        if str(trusted_reason).startswith("outer candidate report"):
            source = "outer_candidate_report"
            if trusted_report is not None:
                evidence_record = trusted_report
        elif probe:
            # The premise reader accepts only the host-authority log.  It records a
            # fenced invocation of the host tool, not an outer host measurement.
            source = "host_bridge_self_check"
    return grade_evidence(source, record=evidence_record, simulated=simulated)


def summarize_levels(rows: list[Mapping[str, Any]]) -> dict:
    """Count explicit levels for status; old unlabelled rows stay model_claim."""
    counts = Counter()
    simulated_runner_levels = Counter()
    for row in rows:
        level = str(row.get("evidence_level") or MODEL_CLAIM)
        if level not in LEVELS:
            level = MODEL_CLAIM
        counts[level] += 1
        provenance = row.get("evidence_provenance")
        if level == SIMULATED and isinstance(provenance, Mapping):
            simulated_runner_levels[str(provenance.get("runner_level") or MODEL_CLAIM)] += 1
    return {
        "counts": {level: counts.get(level, 0) for level in LEVELS},
        "simulated_runner_levels": dict(sorted(simulated_runner_levels.items())),
        "simulated_promotion_allowed": False,
    }
