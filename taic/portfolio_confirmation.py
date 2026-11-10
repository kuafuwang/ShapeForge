"""Backend-neutral subject-confirmation semantics shared by execution and activation.

The Metal executor produces schema-4 reports; this module independently recomputes the R5 identity,
verdict and digest.  DispatchManifest uses the same verifier instead of trusting sidecar fields.
"""

from __future__ import annotations

import copy
import hashlib
from typing import Any, Mapping

from taic.policy import compare_confirmation_identity, expected_report_identity
from tcenv.bench import BENCH_ATTEMPTS, timing_is_uncertain
from tcenv.contract_errors import TaskContractError
from tcenv.evaluate import EvalReport
from tcenv.identity import MAX_SAFE_INTEGER, canonical_bytes, fingerprint
from tcenv.portfolio.candidate import project_provider
from tcenv.portfolio_evidence.projection import _json_closure, _keys, strict_json_equal
from tcenv.portfolio_validation.validation import ValidationPlan

CONFIRMATION_SCHEMA_ID = "tcenv.portfolio.subject-confirmation.1"
CONFIRMATION_COVERAGE_KINDS = frozenset({"sampled_unseen", "tune_confirmation"})
CONFIRMATION_KEYS = frozenset(
    {
        "schema_id",
        "instance_key",
        "usage",
        "provider",
        "variant_id",
        "fallback_id",
        "candidate_sha256",
        "credits_variant",
        "search_seed",
        "confirmation_seed",
        "confirmation_attempts",
        "expected",
        "identity",
        "search_report",
        "confirmation_report",
        "status",
        "reason",
        "coverage_kind",
        "confirmation_digest",
    }
)
SEARCH_SEED = 7


def confirmation_seed_for(validation: ValidationPlan, instance_key: str) -> int:
    """@brief unseen 用 master-derived seed；tune 用 plan-derived 独立 seed，二者不混账。"""
    if instance_key in validation.unseen_instance_keys:
        return int(validation.fresh_seeds[instance_key])
    if instance_key in validation.tune_shards:
        message = canonical_bytes(
            {
                "purpose": "tune-confirmation-seed",
                "validation_plan_digest": validation.digest,
                "instance_key": instance_key,
            }
        )
        return int.from_bytes(hashlib.sha256(message).digest()[:8], "big") & MAX_SAFE_INTEGER
    raise TaskContractError("confirmation seed requested for an instance outside tune/unseen")


def _subject_verdict(document: Mapping[str, Any]) -> tuple[str, str | None]:
    """@brief 从原始 reports/expected 重算确认资格，不从 status 字段推断。"""
    search = document["search_report"]
    confirmation = document["confirmation_report"]
    parts = document["instance_key"].split("/")
    if len(parts) != 3 or document["usage"] != parts[2]:
        raise TaskContractError("subject usage is not the instance_key's declared usage")
    expected_usage = document["usage"]
    expected_mode = "full" if expected_usage == "benchmark" else "quick"
    for report in (search, confirmation):
        EvalReport.from_json(copy.deepcopy(report))
    if document["identity"].get("kind") != "match":
        return "rejected", str(document["identity"].get("kind"))
    for field in ("opspec_digest", "lowering_profile_digest"):
        if document["expected"]["build"].get(field) is None:
            return "rejected", "build_identity_unknown"
    for period, report in (("search", search), ("confirmation", confirmation)):
        if report.get("status") != "ok":
            return "rejected", f"{period}_status_not_ok"
        if report.get("cache_hit") is not False:
            return "rejected", "cache_hit"
        build = report.get("build") or {}
        if build.get("shape") != "full" or build.get("build_kind") != "native":
            return "rejected", "confirmation_ineligible"
        if build.get("candidate_sha256") != document["candidate_sha256"]:
            return "rejected", "candidate_sha256_mismatch"
        incoherent = expected_report_identity(document["expected"], report)
        if incoherent is not None:
            return "rejected", incoherent
        if report.get("dtype") != parts[0]:
            return "rejected", "instance_key_mismatch"
        if report.get("size") != parts[1]:
            return "rejected", "instance_key_mismatch"
        if report.get("mode") != expected_mode:
            return "rejected", "measurement_mode_mismatch"
        if report.get("target_id") != document["expected"]["target_id"]:
            return "rejected", "target_mismatch"
    if document["search_seed"] != SEARCH_SEED or search.get("seed") != SEARCH_SEED or confirmation.get("seed") != document["confirmation_seed"]:
        return "rejected", "seed_mismatch"
    if document["search_seed"] == document["confirmation_seed"]:
        return "rejected", "not_fresh"
    if document["usage"] == "benchmark" and timing_is_uncertain(confirmation):
        return "rejected", "timing_uncertain"
    attempts = document["confirmation_attempts"]
    if type(attempts) is not int or not 1 <= attempts <= BENCH_ATTEMPTS:
        return "rejected", "invalid_attempt_count"
    if document["provider"] == "fallback" and document["credits_variant"] is not False:
        return "rejected", "fallback_credits_variant"
    if document["provider"] == "candidate" and document["credits_variant"] is not True:
        return "rejected", "variant_credit_mismatch"
    return "confirmed", None


def _finalize_subject_document(document: Mapping[str, Any]) -> dict[str, Any]:
    status, reason = _subject_verdict(document)
    expected = dict(document)
    expected["status"] = status
    expected["reason"] = reason
    expected["confirmation_digest"] = fingerprint(
        {key: value for key, value in expected.items() if key != "confirmation_digest"}
    )
    return expected


def verify_subject_document(value: Any) -> dict[str, Any]:
    """@brief 严格重建一份 subject sidecar 的 verdict/digest 并整树比较。"""
    _json_closure(value, where="subject confirmation")
    document = _keys(value, CONFIRMATION_KEYS, where="subject confirmation")
    if document["schema_id"] != CONFIRMATION_SCHEMA_ID:
        raise TaskContractError("unknown subject-confirmation schema")
    if document["coverage_kind"] not in CONFIRMATION_COVERAGE_KINDS:
        raise TaskContractError("a confirmation must retain its sampled_unseen or tune_confirmation label")
    identity = compare_confirmation_identity(
        candidate_sha256=document["candidate_sha256"],
        expected=document["expected"],
        search=document["search_report"],
        confirmation=document["confirmation_report"],
        scope="local",
    )
    if identity != document["identity"]:
        raise TaskContractError("the archived identity comparison is not the R5 comparator's result")
    expected = _finalize_subject_document(document)
    if not strict_json_equal(document, expected):
        raise TaskContractError("the archived subject confirmation is not its semantic projection")
    return expected


def validate_confirmation_for_activation(
    document: Mapping[str, Any],
    freeze: PortfolioFreeze,
    validation: ValidationPlan,
    *,
    expected_record,
    expected: Mapping[str, Any],
    track_version: int | None = None,
) -> dict[str, Any]:
    """@brief 激活前的完整 subject 校验：verdict + plan seed + target + 冻结投影。

    @param track_version 该 run 冻结的选择版本；``None`` 表示按 registry 默认解析（只用于没有
        记录版本的归档重放）。激活重投影必须用同一个版本，不能从 live registry 重选。
    """
    validation.validate_for(freeze)
    verified = verify_subject_document(document)
    if verified["provider"] != expected_record.provider:
        raise TaskContractError(f"{verified['instance_key']}: provider differs from dispatch replay")
    if verified["variant_id"] != expected_record.variant_id:
        raise TaskContractError(f"{verified['instance_key']}: variant identity differs from dispatch replay")
    if verified["fallback_id"] != expected_record.fallback_id:
        raise TaskContractError(f"{verified['instance_key']}: fallback identity differs from dispatch replay")
    if verified["credits_variant"] is not (expected_record.provider == "candidate"):
        raise TaskContractError(f"{verified['instance_key']}: variant credit differs from dispatch replay")
    instance_key = verified["instance_key"]
    if verified["confirmation_seed"] != confirmation_seed_for(validation, instance_key):
        raise TaskContractError(f"{instance_key}: confirmation seed is not derived from this validation plan")
    target_id = freeze.authority.target.id
    if verified["expected"].get("target_id") != target_id:
        raise TaskContractError(f"{instance_key}: expected target differs from the frozen authority")
    for period in ("search_report", "confirmation_report"):
        if verified[period].get("target_id") != target_id:
            raise TaskContractError(f"{instance_key}: {period} target differs from the frozen authority")
    portfolio = freeze.portfolio()
    if verified["provider"] == "candidate":
        provider = portfolio.variant_for_provider(str(verified["variant_id"]))
    else:
        provider = portfolio.fallback
    if provider is None:
        raise TaskContractError(f"{instance_key}: provider is not in the frozen portfolio")
    projection = project_provider(portfolio, provider, authority=freeze.authority, track_version=track_version)
    if verified["candidate_sha256"] != projection.candidate.candidate_sha256:
        raise TaskContractError(f"{instance_key}: candidate bytes differ from the frozen projection")
    if not strict_json_equal(verified["expected"], expected):
        raise TaskContractError(f"{instance_key}: expected identity is not independently rebuilt from this pack/probe")
    return verified


__all__ = [
    "CONFIRMATION_COVERAGE_KINDS",
    "CONFIRMATION_KEYS",
    "CONFIRMATION_SCHEMA_ID",
    "SEARCH_SEED",
    "confirmation_seed_for",
    "validate_confirmation_for_activation",
    "verify_subject_document",
]
