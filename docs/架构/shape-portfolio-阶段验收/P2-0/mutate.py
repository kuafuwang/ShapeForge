#!/usr/bin/env python3
"""Automatic, stage-attributed mutation coverage for the P2-0 evidence prototype.

Every scalar leaf is changed (including numeric type and magnitude boundaries), every object gains
an unknown field, and every array is reordered or extended.  Public derived digests are rebalanced
when the mutated value remains canonical; a deliberately non-canonical magnitude still reaches the
production verifier, whose strict boundary must reject it.  Only ``observation.elapsed_s`` may
accept a valid in-range mutation, and its stable core must remain unchanged.

Only :class:`EvidenceRejection` counts as coverage.  Its explicit ``stage``/``rejected_by`` fields
feed the report; internal exceptions and fuzzy message matching never count as a rejection.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping

from tcenv.identity import ContractError, IdentityError, fingerprint
from tcenv.portfolio_evidence.projection import (
    EVIDENCE_FIELD_ORIGINS,
    EVIDENCE_SCHEMA_ID,
    REJECTION_STAGES,
    EvidenceRejection,
    stable_core,
    strict_json_equal,
    verify_document,
)

MUTATION_REPORT_SCHEMA_ID = "tcenv.portfolio.p2-0-mutation-report.2"
MUTATION_SUITE_SCHEMA_ID = "tcenv.portfolio.p2-0-mutation-suite.3"
VOLATILE_PATHS = frozenset({"observation.elapsed_s"})
DIGEST_LEAVES = frozenset({"freeze_digest", "view_digest", "root_digest"})
Verifier = Callable[..., Mapping[str, Any]]


@dataclass(frozen=True)
class MutationCase:
    """@brief 一个自动生成的变异（路径 + 攻击方式）。"""

    path: str
    kind: str
    origin: str
    apply: Callable[[dict[str, Any]], bool]


def _origin(path: str) -> str:
    if path == "observation":
        return "observed_stable"
    if path.startswith("observation."):
        return EVIDENCE_FIELD_ORIGINS[path]
    return EVIDENCE_FIELD_ORIGINS[path.split(".", 1)[0]]


def _parent(document: dict[str, Any], parts: list[str]) -> tuple[Any, str | int]:
    node: Any = document
    for part in parts[:-1]:
        node = node[int(part) if isinstance(node, list) else part]
    leaf: str | int = parts[-1]
    if isinstance(node, list):
        leaf = int(leaf)
    return node, leaf


def _scalar_mutation(value: Any) -> list[tuple[str, Any]]:
    """@brief 标量主变异、类型变形与有语义的数值量级边界。"""
    if value is None:
        return [("null-to-zero", 0)]
    if isinstance(value, bool):
        return [("bool-flip", not value)]
    if isinstance(value, int):
        return [
            ("int-plus-one", value + 1),
            ("int-to-float", float(value)),
            ("int-to-bool", bool(value)),
            # Reproduces the former float() OverflowError and exceeds MAX_SAFE_INTEGER.
            ("int-magnitude-overflow", 10**400),
        ]
    if isinstance(value, float):
        return [
            ("float-plus-one", value + 1.0),
            ("float-to-int", int(value)),
            ("float-to-bool", bool(value)),
            ("float-magnitude-overflow", 10**400),
            ("float-positive-infinity", float("inf")),
        ]
    if isinstance(value, str):
        return [("string-suffix", value + "__P2_0_MUTATED__")]
    raise AssertionError(f"unsupported JSON scalar {type(value).__name__}")


def _cases(document: Mapping[str, Any]) -> Iterable[MutationCase]:
    """@brief 递归生成标量、未知字段和数组结构变异。"""
    def walk(node: Any, parts: list[str]) -> Iterable[MutationCase]:
        path = ".".join(parts)
        if isinstance(node, dict):
            if path:
                def add_unknown(forged: dict[str, Any], parts=parts) -> bool:
                    parent, leaf = _parent(forged, parts)
                    parent[leaf]["__P2_0_UNCLASSIFIED__"] = True
                    return True

                yield MutationCase(path, "object-unknown-field", _origin(path), add_unknown)
            for key, value in node.items():
                yield from walk(value, [*parts, str(key)])
            return
        if isinstance(node, list):
            def mutate_array(forged: dict[str, Any], parts=parts) -> bool:
                parent, leaf = _parent(forged, parts)
                array = parent[leaf]
                before = copy.deepcopy(array)
                if len(array) > 1:
                    array.reverse()
                else:
                    array.append("__P2_0_MUTATED__")
                # Python == would miss bool/int and signed-zero changes; JSON strictness is required.
                return not strict_json_equal(array, before)

            yield MutationCase(path, "array-order-or-append", _origin(path), mutate_array)
            for index, value in enumerate(node):
                yield from walk(value, [*parts, str(index)])
            return
        if path in DIGEST_LEAVES:
            return
        for kind, replacement in _scalar_mutation(node):
            def replace(forged: dict[str, Any], parts=parts, replacement=replacement) -> bool:
                parent, leaf = _parent(forged, parts)
                if strict_json_equal(parent[leaf], replacement):
                    return False
                parent[leaf] = replacement
                return True

            yield MutationCase(path, kind, _origin(path), replace)

    yield from walk(document, [])


def rebalance(document: dict[str, Any]) -> None:
    """@brief 重算所有公开派生摘要，模拟最强自洽伪造。"""
    document["freeze_digest"] = fingerprint(document["freeze"])
    document["view_digest"] = fingerprint(document["view"])
    document["root_digest"] = fingerprint(
        {
            "schema_id": EVIDENCE_SCHEMA_ID,
            "freeze_digest": document["freeze_digest"],
            "observation": document["observation"],
            "view_digest": document["view_digest"],
        }
    )


def _rebalance_for_attack(document: dict[str, Any]) -> None:
    """@brief 可 canonical 化时配平；极值仍交给 verifier，而不把配平异常算作拒绝。"""
    try:
        rebalance(document)
    except (IdentityError, ContractError):
        # A huge integer/non-finite float has no canonical digest.  The strict JSON entry must
        # reject it before digest verification; swallowing an internal verifier error is forbidden.
        return


def _origin_bucket(counts: dict[str, dict[str, int]], origin: str) -> dict[str, int]:
    return counts.setdefault(
        origin, {"cases": 0, "rejected": 0, "volatile_accepted": 0, "unexpected_accepted": 0}
    )


def run_mutation_gate(
    document: Mapping[str, Any],
    *,
    expected_freeze_digest: str,
    verifier: Verifier = verify_document,
    case_filter: Callable[[MutationCase], bool] | None = None,
) -> dict[str, Any]:
    """@brief 对一份已通过验收的证据运行自动变异覆盖并保留结构化拒绝归因。"""
    baseline_core = verifier(document, expected_freeze_digest=expected_freeze_digest)
    if not strict_json_equal(baseline_core, stable_core(document)):
        raise AssertionError("the mutation gate's baseline verifier returned another stable core")
    by_origin: dict[str, dict[str, int]] = {}
    by_stage: dict[str, int] = {stage: 0 for stage in REJECTION_STAGES}
    by_rejected_by: dict[str, int] = {}
    rejected = 0
    volatile_accepted = 0
    noop = 0
    unexpected: list[dict[str, str]] = []
    cases = 0

    for case in _cases(document):
        if case_filter is not None and not case_filter(case):
            continue
        forged = copy.deepcopy(document)
        changed = case.apply(forged)
        if not changed:
            noop += 1
            continue
        cases += 1
        _origin_bucket(by_origin, case.origin)["cases"] += 1
        _rebalance_for_attack(forged)
        try:
            core = verifier(forged, expected_freeze_digest=expected_freeze_digest)
        except EvidenceRejection as exc:
            if exc.stage not in REJECTION_STAGES or not exc.rejected_by:
                raise AssertionError(f"{case.path} ({case.kind}): unstructured evidence rejection") from exc
            rejected += 1
            _origin_bucket(by_origin, case.origin)["rejected"] += 1
            by_stage[exc.stage] = by_stage.get(exc.stage, 0) + 1
            by_rejected_by[exc.rejected_by] = by_rejected_by.get(exc.rejected_by, 0) + 1
            continue
        except Exception as exc:  # noqa: BLE001 - internal verifier bugs must fail the gate
            raise AssertionError(
                f"{case.path} ({case.kind}): verifier raised an internal error instead of a contract refusal: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        if case.path in VOLATILE_PATHS:
            if not strict_json_equal(core, baseline_core):
                raise AssertionError(f"{case.path} ({case.kind}): volatile mutation changed the stable core")
            volatile_accepted += 1
            _origin_bucket(by_origin, case.origin)["volatile_accepted"] += 1
        else:
            unexpected.append({"path": case.path, "kind": case.kind, "origin": case.origin})
            _origin_bucket(by_origin, case.origin)["unexpected_accepted"] += 1
    return {
        "schema_id": MUTATION_REPORT_SCHEMA_ID,
        "document_digest": fingerprint(document),
        "scalar_leaves": _count_scalar_leaves(document),
        "container_nodes": _count_containers(document),
        "digest_leaves_skipped": len(DIGEST_LEAVES),
        "noop_mutations": noop,
        "cases": cases,
        "rejected": rejected,
        "volatile_accepted": volatile_accepted,
        "unexpected_accepted": unexpected,
        "by_origin": {key: by_origin[key] for key in sorted(by_origin)},
        "by_stage": {key: by_stage[key] for key in sorted(by_stage)},
        "by_rejected_by": {key: by_rejected_by[key] for key in sorted(by_rejected_by)},
    }


def run_mutation_suite(
    documents: Mapping[str, Mapping[str, Any]],
    *,
    verifier: Verifier = verify_document,
) -> dict[str, Any]:
    """@brief 聚合 candidate/fallback/refusal 三类正例，保留每层与每文档贡献。"""
    if not documents or any(type(name) is not str or not name for name in documents):
        raise AssertionError("the mutation suite needs named documents")
    reports = {
        name: run_mutation_gate(
            document,
            expected_freeze_digest=document["freeze_digest"],
            verifier=verifier,
        )
        for name, document in sorted(documents.items())
    }
    by_origin: dict[str, dict[str, int]] = {}
    by_stage: dict[str, int] = {stage: 0 for stage in REJECTION_STAGES}
    by_rejected_by: dict[str, int] = {}
    unexpected: list[dict[str, str]] = []
    for name, report in reports.items():
        for origin, counts in report["by_origin"].items():
            target = _origin_bucket(by_origin, origin)
            for key, value in counts.items():
                target[key] += value
        for stage, count in report["by_stage"].items():
            by_stage[stage] = by_stage.get(stage, 0) + count
        for rejected_by, count in report["by_rejected_by"].items():
            by_rejected_by[rejected_by] = by_rejected_by.get(rejected_by, 0) + count
        unexpected.extend({"document": name, **item} for item in report["unexpected_accepted"])
    per_document = {
        name: {
            key: report[key]
            for key in (
                "document_digest",
                "scalar_leaves",
                "container_nodes",
                "noop_mutations",
                "cases",
                "rejected",
                "volatile_accepted",
                "by_stage",
                "by_rejected_by",
            )
        }
        for name, report in reports.items()
    }
    return {
        "schema_id": MUTATION_SUITE_SCHEMA_ID,
        "documents": sorted(documents),
        "document_digests": {name: report["document_digest"] for name, report in reports.items()},
        "per_document": per_document,
        "scalar_leaves": sum(report["scalar_leaves"] for report in reports.values()),
        "container_nodes": sum(report["container_nodes"] for report in reports.values()),
        "digest_leaves_skipped": sum(report["digest_leaves_skipped"] for report in reports.values()),
        "noop_mutations": sum(report["noop_mutations"] for report in reports.values()),
        "cases": sum(report["cases"] for report in reports.values()),
        "rejected": sum(report["rejected"] for report in reports.values()),
        "volatile_accepted": sum(report["volatile_accepted"] for report in reports.values()),
        "unexpected_accepted": unexpected,
        "by_origin": {key: by_origin[key] for key in sorted(by_origin)},
        "by_stage": {key: by_stage[key] for key in sorted(by_stage)},
        "by_rejected_by": {key: by_rejected_by[key] for key in sorted(by_rejected_by)},
    }


def _count_scalar_leaves(document: Any) -> int:
    if isinstance(document, dict):
        return sum(_count_scalar_leaves(value) for value in document.values())
    if isinstance(document, list):
        return sum(_count_scalar_leaves(value) for value in document)
    return 1


def _count_containers(document: Any) -> int:
    if isinstance(document, dict):
        return 1 + sum(_count_containers(value) for value in document.values())
    if isinstance(document, list):
        return 1 + sum(_count_containers(value) for value in document)
    return 0


__all__ = [
    "MUTATION_REPORT_SCHEMA_ID",
    "MUTATION_SUITE_SCHEMA_ID",
    "VOLATILE_PATHS",
    "rebalance",
    "run_mutation_gate",
    "run_mutation_suite",
]
