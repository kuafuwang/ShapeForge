"""Versioned acceptance contracts connecting a proposal, its review and its implementation.

An acceptance contract is data for the implementer and the experiment planner. It never
executes a command or grants a capability. A useful review is not evidence that the
implemented change improves search; those outcomes remain separate records.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

SCHEMA_VERSION = 1
KINDS = ("hardware_premise", "tool", "reporting", "wording")
PATTERNS = {
    "tool": re.compile(r"\b(?:tool|probe|script|command|cli|python|diagnostic)\b|工具|脚本|探针|命令|诊断", re.I),
    "reporting": re.compile(r"\b(?:report|display|print|field|metric|dashboard|output)\b|报告|显示|打印|字段|指标|看板", re.I),
    "wording": re.compile(r"\b(?:prompt|wording|instruction|skill|overlay)\b|提示|措辞|指导|技能", re.I),
}
MAX_TEXT = 2000
MAX_ITEMS = 16
RISK_TIERS = ("offline_only", "paired_required", "human_gated")
METRIC_IDS = ("first_eligible_cost_usd", "first_eligible_elapsed_s", "tool_success_rate", "invalid_feedback_rate", "confirmed_speedup")


class AcceptanceError(ValueError):
    """The proposal cannot produce a well-scoped acceptance contract."""


def _text(value, *, name: str, required: bool = True, limit: int = MAX_TEXT) -> str:
    if not isinstance(value, str):
        raise AcceptanceError(f"{name} must be text")
    result = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", " ", value).strip()
    if required and not result:
        raise AcceptanceError(f"{name} is required")
    return result[:limit]


def _items(value, *, name: str, required: bool = False) -> list[str]:
    if value is None:
        value = []
    if not isinstance(value, list) or len(value) > MAX_ITEMS:
        raise AcceptanceError(f"{name} must be a list of at most {MAX_ITEMS} strings")
    result = [_text(item, name=name, limit=500) for item in value]
    if required and not result:
        raise AcceptanceError(f"{name} needs at least one item")
    return result


def _paths(value) -> list[str]:
    result = []
    for raw in _items(value, name="allowed_paths"):
        path = raw.replace("\\", "/")
        parts = [part for part in path.split("/") if part]
        if path.startswith("/") or any(part in {"", ".", ".."} for part in parts):
            raise AcceptanceError("allowed_paths must contain repository-relative paths")
        result.append("/".join(parts))
    return sorted(set(result))


def kind_of(proposal: dict) -> str:
    """Classify the evidence needed by a suggestion without changing its existing surface."""
    surface = str(proposal.get("surface") or "")
    if surface == "premise":
        return "hardware_premise"
    if surface in {"skill", "task_wording"}:
        return "wording"
    text = f"{proposal.get('action') or ''} {proposal.get('rationale') or ''}"
    for kind in ("tool", "reporting", "wording"):
        if PATTERNS[kind].search(text):
            return kind
    return "tool"


def digest(document: dict) -> str:
    payload = json.dumps(document, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def make_spec(proposal_row: dict, review: dict | None = None) -> dict:
    """Create a deterministic spec. Old rows without a category or review remain readable."""
    proposal = proposal_row.get("proposal") if isinstance(proposal_row.get("proposal"), dict) else {}
    key = _text(proposal_row.get("proposal_key"), name="proposal_key")
    kind = proposal.get("kind") or kind_of(proposal)
    if kind not in KINDS:
        raise AcceptanceError("unknown proposal kind")
    evidence = proposal.get("evidence") if isinstance(proposal.get("evidence"), dict) else {}
    review = review if isinstance(review, dict) else {}
    reproduction = _text(evidence.get("reproduction") or review.get("reason") or "Use the recorded evidence and reproduce the reported limitation.", name="reproduction")
    expected = _text(evidence.get("expected") or proposal.get("action"), name="expected")
    checks = _items(evidence.get("acceptance_checks"), name="acceptance_checks")
    host_checks = evidence.get("host_checks")
    if host_checks is not None:
        from .acceptance_checks import validate_items
        host_checks = validate_items(host_checks)
    requested_risk = evidence.get("risk_tier") or proposal.get("risk_tier") or None
    if requested_risk is not None and requested_risk not in RISK_TIERS:
        raise AcceptanceError("unknown risk_tier")
    risk_tier = requested_risk or ("paired_required" if kind != "hardware_premise" else "human_gated")
    if kind == "hardware_premise" and risk_tier != "human_gated":
        raise AcceptanceError("hardware premises retain the probe/trust gate")
    if not checks and review.get("recommended_test"):
        checks = [_text(review["recommended_test"], name="recommended_test")]
    if not checks:
        checks = ["Add a regression test for the expected behavior; the host runs the applicable offline suite."]
    effect = "A measured hardware premise agrees with the bound harness record." if kind == "hardware_premise" else "Compare the declared harness metric between matched control and treatment runs."
    metric_id = _text(evidence.get("effect_metric_id") or "first_eligible_cost_usd", name="effect_metric_id")
    if metric_id not in METRIC_IDS:
        raise AcceptanceError("unknown effect_metric_id")
    spec = {
        "schema_version": SCHEMA_VERSION,
        "proposal_key": key,
        "kind": kind,
        "risk_tier": risk_tier,
        "host_checks": host_checks,
        "scope": proposal.get("scope") if isinstance(proposal.get("scope"), dict) else {},
        "source": {"run_id": str(proposal_row.get("run_id") or ""), "seq": proposal_row.get("seq"), "claim_digest": str(proposal_row.get("claim_digest") or "")},
        "reproduction": reproduction,
        "allowed_paths": _paths(evidence.get("allowed_paths") or []),
        "expected": expected,
        "acceptance_checks": checks,
        "effect_metric": _text(evidence.get("effect_metric") or effect, name="effect_metric"),
        "effect_metric_id": metric_id,
        "guardrail_metrics": _items(evidence.get("guardrail_metrics") or ["safety and correctness do not regress", "failure types and additional review cost remain visible"], name="guardrail_metrics", required=True),
        "control": _text(evidence.get("control") or "The same task, target, model, effort, seeds, budget and protocol without the proposed change.", name="control"),
        "rollback": _text(evidence.get("rollback") or "Disable the run overlay or revert the reviewed change; retain all evidence.", name="rollback"),
    }
    if review:
        spec["review"] = {
            "useful": review.get("useful") is True,
            "confidence": review.get("confidence"),
            "risk": _text(review.get("risk") or "", name="risk", required=False),
            "recommended_test": _text(review.get("recommended_test") or "", name="recommended_test", required=False),
            "review_sha256": _text(review.get("review_sha256") or "", name="review_sha256", required=False, limit=64),
        }
    spec["spec_sha256"] = digest(spec)
    return spec


def read_spec(document: dict) -> dict:
    """Validate a stored spec before it is used for implementation or an experiment."""
    if not isinstance(document, dict) or document.get("schema_version") != SCHEMA_VERSION:
        raise AcceptanceError("unsupported acceptance spec")
    allowed = {"schema_version", "proposal_key", "kind", "risk_tier", "host_checks", "scope", "source", "reproduction", "allowed_paths", "expected", "acceptance_checks", "effect_metric", "effect_metric_id", "guardrail_metrics", "control", "rollback", "review", "spec_sha256"}
    if set(document) - allowed:
        raise AcceptanceError("acceptance spec has unknown fields")
    if document.get("kind") not in KINDS:
        raise AcceptanceError("unknown proposal kind")
    normalized = make_spec({"proposal_key": document.get("proposal_key"), "proposal": {"surface": "premise" if document.get("kind") == "hardware_premise" else "skill" if document.get("kind") == "wording" else "tip", "scope": document.get("scope"), "action": document.get("expected"), "evidence": {"reproduction": document.get("reproduction"), "expected": document.get("expected"), "risk_tier": document.get("risk_tier"), "host_checks": document.get("host_checks"), "acceptance_checks": document.get("acceptance_checks"), "effect_metric": document.get("effect_metric"), "effect_metric_id": document.get("effect_metric_id"), "guardrail_metrics": document.get("guardrail_metrics"), "control": document.get("control"), "rollback": document.get("rollback"), "allowed_paths": document.get("allowed_paths")}}, "run_id": (document.get("source") or {}).get("run_id"), "seq": (document.get("source") or {}).get("seq"), "claim_digest": (document.get("source") or {}).get("claim_digest")}, document.get("review"))
    # Restoration above deliberately uses a generic surface; category is an explicit stored fact.
    normalized["kind"] = document["kind"]
    normalized["spec_sha256"] = digest({key: value for key, value in normalized.items() if key != "spec_sha256"})
    if document.get("spec_sha256") != normalized["spec_sha256"]:
        raise AcceptanceError("acceptance spec digest does not match its content")
    return normalized


def report_checks(report: dict, spec: dict) -> dict:
    """Check machine-readable implementation evidence; prose alone never means accepted."""
    if not isinstance(report, dict):
        return {"accepted": False, "missing": ["implementation report is not an object"]}
    raw = report.get("acceptance")
    if not isinstance(raw, list) or not raw:
        return {"accepted": False, "missing": ["acceptance must list host-checkable results"]}
    missing = []
    seen = set()
    for index, item in enumerate(raw):
        if not isinstance(item, dict) or not isinstance(item.get("check"), str) or not item["check"].strip():
            missing.append(f"acceptance[{index}] needs a check name")
            continue
        if not isinstance(item.get("passed"), bool):
            missing.append(f"acceptance[{index}] needs a boolean passed")
        if item.get("passed") is False:
            missing.append(f"failed: {item['check']}")
        seen.add(item["check"].strip())
    for check in spec["acceptance_checks"]:
        if check not in seen:
            missing.append(f"unreported: {check}")
    return {"accepted": not missing, "missing": missing}


def check_paths(paths: list[str], allowed: list[str]) -> list[str]:
    """An empty allow-list is unrestricted for backwards compatibility; a non-empty one is a capability boundary."""
    if not allowed:
        return []
    out = []
    for raw in paths:
        path = str(raw).replace("\\", "/")
        if not any(path == parent or path.startswith(parent.rstrip("/") + "/") for parent in allowed):
            out.append(path)
    return sorted(set(out))


def classify_change(spec: dict, patch: str) -> dict:
    """Derive risk from actual patch content as well as the proposal's declared scope."""
    from .code_agent import SECURITY_CRITICAL, patch_paths

    spec = read_spec(spec)
    try:
        paths = patch_paths(patch)
    except ValueError as exc:
        raise AcceptanceError(f"cannot classify patch paths: {exc}") from exc
    if not paths:
        raise AcceptanceError("a change needs at least one reviewed path")
    if any(any(path == root.rstrip("/") or path.startswith(root) for root in SECURITY_CRITICAL) for path in paths):
        return {"risk_tier": "human_gated", "reason": "the patch touches security- or evaluator-critical paths", "paths": paths}
    if any(path.startswith("tests/") or path.startswith("taic/acceptance") for path in paths):
        return {"risk_tier": "human_gated", "reason": "changing acceptance tests or check definitions needs a named human review", "paths": paths}
    if spec["kind"] == "hardware_premise" or spec["risk_tier"] == "human_gated":
        return {"risk_tier": "human_gated", "reason": "the declared scope requires human review", "paths": paths}
    if not spec["allowed_paths"] or check_paths(paths, spec["allowed_paths"]):
        return {"risk_tier": "human_gated", "reason": "the patch is outside the declared paths", "paths": paths}
    added = "\n".join(line[1:] for line in patch.splitlines() if line.startswith("+") and not line.startswith("+++"))
    if re.search(r"\b(?:subprocess|os\.system|eval\s*\(|exec\s*\(|importlib|__import__|gpu_lock|evaluate\s*\(|tcenv\.evaluate|taic\.prompts|TASK\.md|HARNESS\.md|FEEDBACK\.md)\b", added):
        return {"risk_tier": "human_gated", "reason": "the patch adds execution, GPU or prompt-affecting behavior", "paths": paths}
    if spec["risk_tier"] == "offline_only":
        if not spec["host_checks"] or any(path == "tcenv/report.py" for path in paths):
            return {"risk_tier": "paired_required", "reason": "report feedback or missing host checks can change model behavior", "paths": paths}
        return {"risk_tier": "offline_only", "reason": "host checks cover an additive read-only path", "paths": paths}
    return {"risk_tier": "paired_required", "reason": "the default change class needs paired evidence", "paths": paths}


def implementation_report(report: dict, spec: dict, paths: list[str], patch_sha256: str) -> dict:
    """Bind implementation evidence to the exact contract and patch, not just the proposal text."""
    if not isinstance(patch_sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", patch_sha256):
        raise AcceptanceError("a patch sha256 is required")
    checks = report_checks(report, spec)
    outside = check_paths(paths, spec["allowed_paths"])
    missing = [*checks["missing"], *(f"path outside acceptance scope: {path}" for path in outside)]
    return {"schema_version": 1, "event": "implementation", "proposal_key": spec["proposal_key"], "spec_sha256": spec["spec_sha256"], "patch_sha256": patch_sha256, "accepted": not missing, "missing": missing, "acceptance": checks, "outside_paths": outside}
