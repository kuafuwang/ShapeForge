#!/usr/bin/env python3
"""P2-0 CPU-only acceptance: typed evidence, deterministic projection and trust-root mutation.

This is a prototype gate, not P2 confirmation/activation.  It starts no model, search or GPU work.
``--write-golden`` and ``--write-manifest`` are explicit recording actions; the default run only
verifies.  Every negative asserts a reason fragment, and the volatile positive control must pass.
"""

from __future__ import annotations

import argparse
import base64
import copy
import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable, Mapping
from unittest.mock import patch

from tcenv.identity import ContractError, IdentityError, fingerprint
from tcenv.opspec.v2 import instantiate_v2
from tcenv.portfolio.candidate import parse_portfolio, project_provider
from tcenv.portfolio.dispatch import DispatchRequest
from tcenv.portfolio.domain import normalize_semantic_domain_plan
from tcenv.portfolio_evidence.projection import (
    DERIVED_VIEW,
    EVIDENCE_FIELD_ORIGINS,
    EXTERNAL_PIN,
    EXTERNAL_SECRET,
    FREEZE_SELF_CONSISTENCY,
    OBSERVED_VOLATILE,
    SEMANTIC_REPROJECTION,
    STRICT_PARSE,
    EvidenceRejection,
    FrozenPortfolioInputs,
    RawPortfolioObservation,
    loads_document,
    project_document,
    stable_core,
    strict_json_equal,
    verify_document,
    verify_json,
)
from tcenv.target import load_target
from tcenv.track.plane import CandidateEntry
from tests.support.shape_portfolio_metal import (
    evidence_all_valid,
    load_p1_task,
    p1_instances,
    parent_entries,
    portfolio_document,
)

ROOT = Path(__file__).resolve().parents[4]
P20 = Path(__file__).resolve().parent
GOLDEN = P20 / "golden.json"
MUTATION_REPORT = P20 / "mutation-report.json"
MANIFEST = P20 / "manifest.sha256"
PLAN = "docs/架构/shape-portfolio实施计划.md"
MODEL = "docs/架构/shape-portfolio-阶段验收/P2-0/evidence-model.md"
VERIFIER = "docs/架构/shape-portfolio-阶段验收/P2-0/verify.py"
PRODUCTION = "tcenv/portfolio_evidence/projection.py"
PACKAGE_INIT = "tcenv/portfolio_evidence/__init__.py"
GOLDEN_NAME = "docs/架构/shape-portfolio-阶段验收/P2-0/golden.json"
MUTATOR_NAME = "docs/架构/shape-portfolio-阶段验收/P2-0/mutate.py"
MUTATION_REPORT_NAME = "docs/架构/shape-portfolio-阶段验收/P2-0/mutation-report.json"
P20_TEST = "tests/shared/test_shape_portfolio_evidence.py"
P20_TEST_NODE_IDS = (
    f"{P20_TEST}::test_p2_0_core_api_strict_json_and_numeric_boundaries",
)
STRUCTURE_TEST = "tests/shared/test_structure.py"
P0_MANIFEST = ROOT / "docs/架构/shape-portfolio-阶段验收/P0/manifest.sha256"
P1_MANIFEST = ROOT / "docs/架构/shape-portfolio-阶段验收/P1/manifest.sha256"
DELIVERABLES = (
    PRODUCTION,
    PACKAGE_INIT,
    PLAN,
    MODEL,
    VERIFIER,
    GOLDEN_NAME,
    MUTATOR_NAME,
    MUTATION_REPORT_NAME,
    P20_TEST,
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_manifest(path: Path) -> dict[str, str]:
    entries: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, name = line.split("  ", 1)
        entries[name] = digest
    return entries


def write_manifest() -> None:
    lines = [f"{sha256(ROOT / name)}  {name}" for name in sorted(DELIVERABLES)]
    MANIFEST.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"OK: wrote {MANIFEST.relative_to(ROOT)} ({len(lines)} files)")


def verify_manifest() -> None:
    entries = parse_manifest(MANIFEST)
    require(set(entries) == set(DELIVERABLES), "P2-0 manifest file set drifted")
    for name, digest in entries.items():
        require(sha256(ROOT / name) == digest, f"P2-0 manifest mismatch: {name}")
    print(f"OK: P2-0 manifest ({len(entries)} files)")


def verify_package_boundary() -> None:
    """P2 证据包的封闭文件集与禁止架构边界，和 P0 核心包同等严格。"""
    package = ROOT / "tcenv/portfolio_evidence"
    sources = sorted(package.glob("*.py"))
    require(
        [path.name for path in sources] == ["__init__.py", "projection.py"],
        f"P2-0 evidence package drifted: {[path.name for path in sources]}",
    )
    forbidden = ("PatternResolver", "LoweringRuleRegistry", "StagePartitioner", "TileConnector", "LoweringProfile")
    for path in sources:
        text = path.read_text(encoding="utf-8")
        for token in forbidden:
            require(token not in text, f"{path.name}: forbidden architecture token {token}")
        for line in text.splitlines():
            stripped = line.strip()
            require(
                not stripped.startswith(("import tcenv.backends", "from tcenv.backends", "import taic", "from taic")),
                f"{path.name}: backend/search import crosses the evidence-package boundary: {stripped}",
            )
    print("OK: P2 evidence package boundary (2 closed files, no forbidden architecture/backend import)")


def verify_historical_manifests() -> None:
    """P2-0 只新增文件和 living plan，不回改 P0/P1 的验收资产。"""
    for label, path, allowed in (
        ("P0", P0_MANIFEST, {PLAN, "tcenv/portfolio/dispatch.py"}),
        ("P1", P1_MANIFEST, {PLAN}),
    ):
        entries = parse_manifest(path)
        drifted = {name for name, digest in entries.items() if sha256(ROOT / name) != digest}
        require(drifted <= allowed, f"{label} historical files drifted outside {sorted(allowed)}: {sorted(drifted)}")
        print(f"OK: {label} historical manifest (declared drift only: {sorted(drifted & allowed)})")


def _mutation_tools():
    spec = importlib.util.spec_from_file_location("p2_0_mutate", P20 / "mutate.py")
    if spec is None or spec.loader is None:
        raise AssertionError("cannot load the P2-0 mutation gate")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


MUTATE = _mutation_tools()
rebalance = MUTATE.rebalance


class Fixture:
    """@brief 从固定 P1 人工候选重建两个单源 CPU 原型实例。"""

    def __init__(self, root: Path) -> None:
        self.keeper = tempfile.TemporaryDirectory(prefix="p2-0-fixture-", dir=root)
        task_root = Path(self.keeper.name)
        self.contract = load_p1_task(task_root)
        self.instances = p1_instances(self.contract)
        self.target = load_target("apple-m4-max")
        self.plan = normalize_semantic_domain_plan(
            self.contract.spec, self.instances, target=self.target
        )
        self.entries = parent_entries(portfolio_document(self.plan.digest))
        self.portfolio = parse_portfolio(self.entries, domain_plan=self.plan)

    def freeze(self, size: str) -> FrozenPortfolioInputs:
        instance = instantiate_v2(self.contract.spec, "fp16", size)
        point = self.plan.find("fp16", size, "golden")
        require(point is not None, f"the fixed domain has no fp16/{size}/golden point")
        request = DispatchRequest.from_instance(
            instance, target_abi_digest=point.target_abi_digest, usage="golden"
        )
        return FrozenPortfolioInputs(
            protocol_epoch=1,
            opspec_document=self.contract.spec.document,
            opspec_tensor_order=tuple(self.contract.spec.document["tensors"]),
            domain_plan_document=self.plan.document(),
            candidate_entries=self.entries,
            evidence=evidence_all_valid(self.portfolio),
            target_id=self.target.id,
            max_buffer_length=None,
            request=request,
        )

    def project(self, freeze: FrozenPortfolioInputs, *, elapsed_s: int | float = 0.0) -> dict[str, Any]:
        portfolio, record = freeze.resolve()
        provider = None
        if record.outcome == "resolved":
            provider = portfolio.variant_for_provider(str(record.variant_id))
        elif record.outcome == "fallback":
            provider = portfolio.fallback
        if provider is None:
            observation = RawPortfolioObservation(
                None, None, None, freeze.request.instance_key, elapsed_s
            )
        else:
            projection = project_provider(portfolio, provider, authority=freeze.authority())
            observation = RawPortfolioObservation(
                projection.provider,
                projection.track_id,
                projection.candidate.candidate_sha256,
                freeze.request.instance_key,
                elapsed_s,
            )
        return project_document(freeze, observation)

    def positives(self) -> dict[str, dict[str, Any]]:
        return {size: self.project(self.freeze(size)) for size in ("below", "hole")}

    def refusal(self) -> dict[str, Any]:
        """@brief 一个语义拒绝正例：没有执行 provider，projection 必须为 null。"""
        frozen = self.freeze("below").document()
        frozen["request"]["op_name"] = "p2_0_not_the_semantic_op"
        document = self.project(FrozenPortfolioInputs.from_document(frozen))
        require(document["view"]["projection"] is None, "the refusal fixture unexpectedly has a projection")
        require(document["view"]["record"]["outcome"] == "semantic_unresolved", "wrong refusal fixture outcome")
        return document

    def mutation_documents(
        self, positives: Mapping[str, dict[str, Any]]
    ) -> dict[str, dict[str, Any]]:
        return {"below": positives["below"], "hole": positives["hole"], "refusal": self.refusal()}


def golden_document(positives: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "schema_id": "tcenv.portfolio.p2-0-golden.1",
        "protocol_epoch": 1,
        "cases": {
            size: {
                "freeze_digest": document["freeze_digest"],
                "view_digest": document["view_digest"],
                "root_digest": document["root_digest"],
                "stable_core_digest": fingerprint(stable_core(document)),
            }
            for size, document in sorted(positives.items())
        },
    }


def verify_golden(positives: Mapping[str, Mapping[str, Any]]) -> None:
    expected = json.loads(GOLDEN.read_text(encoding="utf-8"))
    actual = golden_document(positives)
    require(actual == expected, "the frozen P2-0 identity golden drifted")
    for size, document in positives.items():
        stable = verify_document(document, expected_freeze_digest=document["freeze_digest"])
        require(stable == stable_core(document), f"{size}: verify_document returned another stable core")
    print(f"OK: deterministic projection + frozen golden ({len(positives)} single-source instances)")


def expect_rejection(label: str, fragment: str, action: Callable[[], Any]) -> None:
    try:
        action()
    except Exception as exc:  # noqa: BLE001 - reason is checked below, not any-exception acceptance
        require(
            fragment in str(exc),
            f"{label}: rejected for the wrong reason: {exc}; expected fragment {fragment!r}",
        )
        return
    raise AssertionError(f"{label}: forged evidence was accepted")


def expect_evidence_rejection(
    label: str,
    fragment: str,
    action: Callable[[], Any],
    *,
    stage: str,
    rejected_by: str | None = None,
) -> None:
    """@brief 新攻击必须得到指定层的合同拒绝，内部异常绝不记为覆盖。"""
    try:
        action()
    except EvidenceRejection as exc:
        require(fragment in str(exc), f"{label}: wrong refusal reason: {exc}")
        require(exc.stage == stage, f"{label}: wrong stage {exc.stage!r}, expected {stage!r}")
        if rejected_by is not None:
            require(
                exc.rejected_by == rejected_by,
                f"{label}: wrong rejected_by {exc.rejected_by!r}, expected {rejected_by!r}",
            )
        return
    except Exception as exc:  # noqa: BLE001 - turn an internal error into an explicit gate failure
        raise AssertionError(f"{label}: internal {type(exc).__name__} escaped: {exc}") from exc
    raise AssertionError(f"{label}: forged evidence was accepted")


def verify_strict_types(document: Mapping[str, Any]) -> int:
    cases: list[tuple[str, str, Callable[[dict[str, Any]], None]]] = [
        ("unknown top-level field", "has fields", lambda d: d.__setitem__("unclassified", True)),
        ("bool epoch", "protocol_epoch must be an integer", lambda d: d["freeze"].__setitem__("protocol_epoch", True)),
        ("float epoch", "protocol_epoch must be an integer", lambda d: d["freeze"].__setitem__("protocol_epoch", 1.0)),
        ("list evidence map", "evidence must be a JSON object", lambda d: d["freeze"].__setitem__("evidence", [])),
        ("bool elapsed", "elapsed_s must be a finite non-negative number", lambda d: d["observation"].__setitem__("elapsed_s", True)),
        ("negative elapsed", "elapsed_s must be a finite non-negative number", lambda d: d["observation"].__setitem__("elapsed_s", -1.0)),
        ("bool provider", "provider must be candidate/fallback/null", lambda d: d["observation"].__setitem__("provider", True)),
        ("unknown observation field", "observation has fields", lambda d: d["observation"].__setitem__("extra", 1)),
        ("invalid candidate base64", "data_base64", lambda d: d["freeze"]["candidate_entries"][0].__setitem__("data_base64", "not-base64!")),
        ("overflow elapsed integer", "integer must be within", lambda d: d["observation"].__setitem__("elapsed_s", 10**400)),
        ("infinite elapsed", "finite JSON number", lambda d: d["observation"].__setitem__("elapsed_s", float("inf"))),
        ("overflow epoch integer", "integer must be within", lambda d: d["freeze"].__setitem__("protocol_epoch", 10**400)),
        (
            "tuple tensor dims",
            "JSON type closure",
            lambda d: d["freeze"]["opspec"]["tensors"]["x"].__setitem__(
                "dims", tuple(d["freeze"]["opspec"]["tensors"]["x"]["dims"])
            ),
        ),
        (
            "non-string object key",
            "non-string JSON object key",
            lambda d: d["freeze"]["opspec"]["symbols"].__setitem__(1, 1),
        ),
    ]
    for label, fragment, mutate in cases:
        forged = copy.deepcopy(document)
        mutate(forged)
        try:
            rebalance(forged)
        except (IdentityError, ContractError):
            # Non-canonical numeric attacks must still reach the strict JSON boundary.
            pass
        expect_rejection(
            label,
            fragment,
            lambda forged=forged: verify_document(
                forged, expected_freeze_digest=forged["freeze_digest"]
            ),
        )
    deep: Any = 0
    for _ in range(40):
        deep = [deep]
    forged_deep = copy.deepcopy(document)
    forged_deep["freeze"]["opspec"]["deep"] = deep
    expect_evidence_rejection(
        "excess JSON nesting",
        "maximum JSON nesting depth",
        lambda: verify_document(forged_deep, expected_freeze_digest=document["freeze_digest"]),
        stage=STRICT_PARSE,
        rejected_by="json_type_closure_and_schema",
    )
    expect_evidence_rejection(
        "recursive JSON text",
        "maximum JSON nesting depth",
        lambda: loads_document("[" * 2000 + "0" + "]" * 2000, expected_freeze_digest=document["freeze_digest"]),
        stage=STRICT_PARSE,
        rejected_by="json_type_closure_and_schema",
    )
    require(not strict_json_equal(True, 1), "strict JSON equality still conflates bool and int")
    require(not strict_json_equal(1, 1.0), "strict JSON equality still conflates int and float")
    require(not strict_json_equal([], {}), "strict JSON equality still conflates array and object")
    require(not strict_json_equal((1,), (True,)), "strict JSON equality accepted non-JSON tuples")
    require(not strict_json_equal({1: "x"}, {True: "x"}), "strict JSON equality accepted non-string keys")
    require(not strict_json_equal(-0.0, 0.0), "strict JSON equality conflated signed zero")
    require(strict_json_equal(-0.0, -0.0), "strict JSON equality rejected identical signed zero")
    require(strict_json_equal({"a": [1, False]}, {"a": [1, False]}), "strict JSON equality rejected identical data")
    print(f"OK: strict typed boundary ({len(cases) + 2} reasoned negatives + strict equality self-tests)")
    return len(cases) + 2


def verify_tensor_order_and_strict_entries(below: Mapping[str, Any]) -> int:
    """@brief R1/R2：顺序权威可往返，重复 key 必须在文件/文本入口消失前拒绝。"""
    expected_digest = below["freeze_digest"]

    object_reordered = copy.deepcopy(below)
    tensors = object_reordered["freeze"]["opspec"]["tensors"]
    object_reordered["freeze"]["opspec"]["tensors"] = dict(reversed(list(tensors.items())))
    rebalance(object_reordered)
    require(
        object_reordered["freeze_digest"] == expected_digest,
        "opspec.tensors object key order incorrectly changed the freeze pin",
    )
    require(
        strict_json_equal(
            verify_document(object_reordered, expected_freeze_digest=expected_digest),
            stable_core(below),
        ),
        "opspec.tensors object key order changed the verdict",
    )

    with tempfile.TemporaryDirectory(prefix="p2-0-json-entry-") as temporary:
        path = Path(temporary) / "evidence.json"
        path.write_text(json.dumps(object_reordered, sort_keys=True), encoding="utf-8")
        from_file = verify_json(path, expected_freeze_digest=expected_digest)
        from_text = loads_document(path.read_text(encoding="utf-8"), expected_freeze_digest=expected_digest)
        require(
            strict_json_equal(from_file, stable_core(below))
            and strict_json_equal(from_text, stable_core(below)),
            "sort_keys file/text round trip changed the stable verdict",
        )

        # The first formula is forged and the second is the original.  Permissive json.loads keeps
        # the last one and therefore accepts; both production entry points must reject earlier.
        duplicate_text = json.dumps(below)
        marker = '"formula":'
        position = duplicate_text.find(marker)
        require(position >= 0, "the fixture has no serialized formula key")
        duplicate_text = (
            duplicate_text[:position]
            + '"formula": "P2_0_FORGED_DUPLICATE", '
            + duplicate_text[position:]
        )
        permissive = json.loads(duplicate_text)
        verify_document(permissive, expected_freeze_digest=expected_digest)
        duplicate_path = Path(temporary) / "duplicate.json"
        duplicate_path.write_text(duplicate_text, encoding="utf-8")
        expect_evidence_rejection(
            "duplicate formula file key",
            "repeats the key 'formula'",
            lambda: verify_json(duplicate_path, expected_freeze_digest=expected_digest),
            stage=STRICT_PARSE,
            rejected_by="load_strict_json",
        )
        expect_evidence_rejection(
            "duplicate formula text key",
            "repeats the key 'formula'",
            lambda: loads_document(duplicate_text, expected_freeze_digest=expected_digest),
            stage=STRICT_PARSE,
            rejected_by="loads_strict_json",
        )

    reversed_order = copy.deepcopy(below)
    reversed_order["freeze"]["opspec_tensor_order"].reverse()
    require(
        fingerprint(reversed_order["freeze"]) != expected_digest,
        "the explicit tensor declaration order is not bound by the freeze pin",
    )
    rebalance(reversed_order)
    expect_evidence_rejection(
        "tensor declaration order root",
        "externally pinned freeze",
        lambda: verify_document(reversed_order, expected_freeze_digest=expected_digest),
        stage=EXTERNAL_PIN,
        rejected_by="expected_freeze_digest",
    )

    malformed_order = copy.deepcopy(below)
    malformed_order["freeze"]["opspec_tensor_order"].pop()
    rebalance(malformed_order)
    expect_evidence_rejection(
        "incomplete tensor declaration order",
        "opspec_tensor_order must contain every",
        lambda: verify_document(
            malformed_order, expected_freeze_digest=malformed_order["freeze_digest"]
        ),
        stage=SEMANTIC_REPROJECTION,
    )
    print("OK: tensor-order authority + sort_keys/file/text positives + duplicate-key negatives")
    return 4


def verify_origin_classification(document: Mapping[str, Any]) -> int:
    """@brief R5：键集齐全但来源标签错误的实现也必须失败。"""
    for label, field, wrong in (
        ("view mislabeled volatile", "view", OBSERVED_VOLATILE),
        ("freeze mislabeled secret", "freeze", EXTERNAL_SECRET),
    ):
        with patch.dict(EVIDENCE_FIELD_ORIGINS, {field: wrong}):
            expect_evidence_rejection(
                label,
                "incorrect semantic classifications",
                lambda: verify_document(document, expected_freeze_digest=document["freeze_digest"]),
                stage=STRICT_PARSE,
                rejected_by="json_type_closure_and_schema",
            )
    print("OK: field-origin inventory checks exact semantic labels, not just key coverage")
    return 2


def verify_derived_and_observations(fixture: Fixture, below: Mapping[str, Any], hole: Mapping[str, Any]) -> int:
    expected_digest = below["freeze_digest"]
    derived_cases: list[tuple[str, str, Callable[[dict[str, Any]], None]]] = [
        ("derived provider forgery", "deterministic projection", lambda d: d["view"]["record"].__setitem__("provider", "fallback")),
        ("derived extra field", "deterministic projection", lambda d: d["view"].__setitem__("extra", {"forged": True})),
    ]
    for label, fragment, mutate in derived_cases:
        forged = copy.deepcopy(below)
        mutate(forged)
        rebalance(forged)
        expect_rejection(label, fragment, lambda forged=forged: verify_document(forged, expected_freeze_digest=expected_digest))
    forged_freeze_digest = copy.deepcopy(below)
    forged_freeze_digest["freeze_digest"] = "0" * 64
    expect_evidence_rejection(
        "freeze digest forgery",
        "freeze_digest is not the digest",
        lambda: verify_document(forged_freeze_digest, expected_freeze_digest=expected_digest),
        stage=FREEZE_SELF_CONSISTENCY,
        rejected_by="freeze_digest",
    )
    forged_root = copy.deepcopy(below)
    forged_root["root_digest"] = "0" * 64
    expect_rejection(
        "root digest forgery",
        "deterministic projection",
        lambda: verify_document(forged_root, expected_freeze_digest=expected_digest),
    )
    forged_view_digest = copy.deepcopy(below)
    forged_view_digest["view_digest"] = "0" * 64
    expect_evidence_rejection(
        "view digest forgery",
        "deterministic projection",
        lambda: verify_document(forged_view_digest, expected_freeze_digest=expected_digest),
        stage=DERIVED_VIEW,
        rejected_by="strict_json_equal",
    )
    forged_credit = copy.deepcopy(hole)
    forged_credit["view"]["credits_variant"] = True
    rebalance(forged_credit)
    expect_rejection(
        "fallback credits variant",
        "deterministic projection",
        lambda: verify_document(forged_credit, expected_freeze_digest=hole["freeze_digest"]),
    )
    forged_observation = copy.deepcopy(below)
    forged_observation["observation"]["candidate_sha256"] = "0" * 64
    rebalance(forged_observation)
    expect_rejection(
        "stable observation forgery",
        "raw observation",
        lambda: verify_document(forged_observation, expected_freeze_digest=expected_digest),
    )
    forged_instance = copy.deepcopy(below)
    forged_instance["observation"]["instance_key"] = "fp16/tail/golden"
    rebalance(forged_instance)
    expect_evidence_rejection(
        "stable observation instance-key forgery",
        "observation.instance_key",
        lambda: verify_document(forged_instance, expected_freeze_digest=expected_digest),
        stage=SEMANTIC_REPROJECTION,
    )
    tail = fixture.project(fixture.freeze("tail"))
    require(
        stable_core(below)["observation"]["instance_key"]
        != stable_core(tail)["observation"]["instance_key"],
        "below/tail still have indistinguishable stable observations",
    )
    # Positive control: volatile time is allowed to change.  The document is still projected from
    # the new observation, while the stable core is byte-for-byte unchanged.
    volatile = fixture.project(fixture.freeze("below"), elapsed_s=12.5)
    rebalance(volatile)
    stable = verify_document(volatile, expected_freeze_digest=expected_digest)
    require(stable == stable_core(below), "volatile elapsed_s changed the stable evidence core")
    require(volatile["root_digest"] != below["root_digest"], "the root digest failed to bind the raw observation")
    print("OK: derived view/observation attacks rejected; instance keys distinguish shapes; volatile control passes")
    return len(derived_cases) + 6


def verify_defense_in_depth(below: Mapping[str, Any]) -> int:
    """@brief 外部 pin 自洽时，两个纵深防御分支也必须各自拒绝。"""
    duplicated_order = copy.deepcopy(below)
    order = duplicated_order["freeze"]["opspec_tensor_order"]
    order.append(order[0])
    rebalance(duplicated_order)
    expect_evidence_rejection(
        "duplicate tensor declaration name",
        "exactly once",
        lambda: verify_document(duplicated_order, expected_freeze_digest=duplicated_order["freeze_digest"]),
        stage=SEMANTIC_REPROJECTION,
        rejected_by="frozen_inputs_observation_projector",
    )
    extra_provider = copy.deepcopy(below)
    extra_provider["freeze"]["evidence"]["some_other_provider"] = "validated"
    rebalance(extra_provider)
    expect_evidence_rejection(
        "evidence names an extra provider",
        "exactly the portfolio's providers",
        lambda: verify_document(extra_provider, expected_freeze_digest=extra_provider["freeze_digest"]),
        stage=SEMANTIC_REPROJECTION,
        rejected_by="frozen_inputs_observation_projector",
    )
    print("OK: defense-in-depth branches (duplicate tensor name + extra evidence provider)")
    return 2


def verify_trust_roots(fixture: Fixture, below: Mapping[str, Any], hole: Mapping[str, Any]) -> int:
    old_expected = below["freeze_digest"]

    def projected_mutation(label: str, fragment: str, mutate: Callable[[dict[str, Any]], None]) -> None:
        def action() -> None:
            forged = copy.deepcopy(below)
            mutate(forged["freeze"])
            changed_freeze = FrozenPortfolioInputs.from_document(forged["freeze"])
            changed_document = fixture.project(changed_freeze)
            require(
                changed_document["view"] != below["view"],
                f"{label}: the trust-root mutation did not change the semantic view",
            )
            verify_document(changed_document, expected_freeze_digest=old_expected)

        expect_rejection(label, fragment, action)

    def candidate_source(freeze: dict[str, Any]) -> None:
        for entry in freeze["candidate_entries"]:
            if entry["path"] == "device/variant_b.metal":
                data = base64.b64decode(entry["data_base64"])
                entry["data_base64"] = base64.b64encode(data + b"\n// P2-0 trust-root mutation\n").decode("ascii")
                break
        else:
            raise AssertionError("the fixed variant B source is missing")
        # The source byte changes variant_id by design.  Re-register evidence under the new provider
        # identities too, so this is a fully rebalanced valid alternative freeze, not a stale map.
        entries = tuple(
            CandidateEntry(
                path=entry["path"],
                role=entry["role"],
                media_type=entry["media_type"],
                data=base64.b64decode(entry["data_base64"]),
            )
            for entry in freeze["candidate_entries"]
        )
        changed_portfolio = parse_portfolio(entries, domain_plan=fixture.plan)
        freeze["evidence"] = evidence_all_valid(changed_portfolio)

    projected_mutation("candidate root", "externally pinned freeze", candidate_source)
    projected_mutation(
        "protocol epoch root",
        "externally pinned freeze",
        lambda freeze: freeze.__setitem__("protocol_epoch", 2),
    )
    projected_mutation(
        "request root",
        "externally pinned freeze",
        lambda freeze: freeze.__setitem__("request", copy.deepcopy(hole["freeze"]["request"])),
    )
    projected_mutation(
        "evidence-status root",
        "externally pinned freeze",
        lambda freeze: freeze.__setitem__("evidence", {key: "unknown" for key in freeze["evidence"]}),
    )

    def invalid_root(label: str, fragment: str, mutate: Callable[[dict[str, Any]], None]) -> None:
        def action() -> None:
            forged = copy.deepcopy(below)
            mutate(forged["freeze"])
            changed_freeze = FrozenPortfolioInputs.from_document(forged["freeze"])
            changed_document = fixture.project(changed_freeze)
            verify_document(changed_document, expected_freeze_digest=old_expected)

        expect_rejection(label, fragment, action)

    invalid_root(
        "OpSpec root",
        "not the finite plan normalized from the authoritative OpSpec/target/instances",
        lambda freeze: freeze["opspec"].__setitem__("formula", "y = -2 * x"),
    )
    invalid_root(
        "DomainAuthority ABI root",
        "portfolio is bound to the domain plan",
        lambda freeze: freeze["domain_plan"]["instances"][0].__setitem__("target_abi_digest", "0" * 64),
    )
    invalid_root(
        "ABI buffer root",
        "buffer",
        lambda freeze: freeze.__setitem__("max_buffer_length", 1),
    )
    print("OK: trust-root perturbation (4 valid alternative roots + 3 invalid authority roots, all reasoned)")
    return 7


def verify_mutation_coverage(documents: Mapping[str, Mapping[str, Any]]) -> None:
    actual = MUTATE.run_mutation_suite(documents)
    unexpected = actual["unexpected_accepted"]
    require(not unexpected, f"automatic mutation found unchecked evidence leaves: {unexpected}")
    require(
        actual["rejected"] + actual["volatile_accepted"] == actual["cases"],
        "mutation accounting does not add up",
    )
    require(
        sum(actual["by_stage"].values()) == actual["rejected"]
        and sum(actual["by_rejected_by"].values()) == actual["rejected"],
        "structured rejection attribution does not add up",
    )
    require(
        set(actual["documents"]) == {"below", "hole", "refusal"},
        "mutation coverage lost the candidate/fallback/refusal document classes",
    )
    expected = json.loads(MUTATION_REPORT.read_text(encoding="utf-8"))
    require(actual == expected, "the archived mutation-coverage report drifted; re-record it explicitly")
    document = documents["below"]
    # The gate must also fail when the verifier is deliberately made permissive.  Restrict this
    # negative control to three paths so the default gate does not pay for a second full sweep.
    control = MUTATE.run_mutation_gate(
        document,
        expected_freeze_digest=document["freeze_digest"],
        verifier=lambda forged, **kwargs: stable_core(document),
        case_filter=lambda case: case.path in {"schema_id", "view", "observation.provider"},
    )
    require(control["cases"] >= 3, "the permissive-verifier control ran too few cases")
    require(
        len(control["unexpected_accepted"]) == control["cases"],
        "the mutation gate failed to report a deliberately permissive verifier",
    )
    print(
        f"OK: automatic mutation gate ({actual['cases']} effective cases over "
        f"{actual['scalar_leaves']} scalar leaves + {actual['container_nodes']} containers in "
        f"{len(actual['documents'])} document classes; {actual['rejected']} rejected, "
        f"{actual['volatile_accepted']} volatile accepted, 0 unchecked; stages={actual['by_stage']}; "
        f"{actual['noop_mutations']} no-op structural mutations skipped)"
    )


def _acceptance_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    require(spec is not None and spec.loader is not None, f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _pytest(nodes: list[str], *, where: str, expected: int) -> str:
    """@brief 运行明确 node id；无 skip/xfail/deselect，passed 数必须精确相等。"""
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT) + (
        os.pathsep + environment["PYTHONPATH"] if environment.get("PYTHONPATH") else ""
    )
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-n", "0", *nodes],
        cwd=ROOT,
        capture_output=True,
        text=True,
        env=environment,
    )
    output = (result.stdout or "") + (result.stderr or "")
    require(result.returncode == 0, f"{where} failed:\n{output}")
    summaries = [line for line in output.splitlines() if re.search(r"\b(passed|failed|error)\b", line)]
    require(summaries, f"{where}: no pytest summary\n{output}")
    summary = summaries[-1]
    require(
        " skipped" not in summary and "xfailed" not in summary and " deselected" not in summary,
        f"{where}: fixed nodes were skipped/xfailed/deselected: {summary}",
    )
    counts = re.findall(r"(\d+) passed\b", summary)
    require(len(counts) == 1 and int(counts[0]) == expected, f"{where}: expected {expected} passed: {summary}")
    return summary


def verify_fixed_pytest() -> None:
    """@brief P2-0 的显式扩展：历史 54 + 结构 6 + 新核心 API 1，绝不递归调用本验收器。"""
    p0 = _acceptance_module(P0_MANIFEST.parent / "verify.py", "p2_0_historical_p0_verify")
    p1 = _acceptance_module(P1_MANIFEST.parent / "verify.py", "p2_0_historical_p1_verify")
    historical = [*p0.FIXED_NODE_IDS, *p1.FIXED_CPU_NODE_IDS]
    require(len(historical) == 54 and len(set(historical)) == 54, "P0/P1 fixed node inventory drifted")
    historical_summary = _pytest(historical, where="P0/P1 fixed CPU nodes", expected=54)
    structure_summary = _pytest([STRUCTURE_TEST], where="test_structure.py", expected=6)
    p20_summary = _pytest(list(P20_TEST_NODE_IDS), where="P2-0 declared shared extension", expected=1)
    print(f"OK: P0/P1 fixed CPU pytest (54 nodes): {historical_summary}")
    print(f"OK: tests/shared/test_structure.py (6 nodes): {structure_summary}")
    print(f"OK: P2-0 declared shared pytest extension (1 node): {p20_summary}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write-golden", action="store_true")
    parser.add_argument("--write-mutation-report", action="store_true")
    parser.add_argument("--write-manifest", action="store_true")
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="p2-0-") as temporary:
        fixture = Fixture(Path(temporary))
        positives = fixture.positives()
        mutation_documents = fixture.mutation_documents(positives)
        if args.write_golden:
            GOLDEN.write_text(json.dumps(golden_document(positives), indent=2, sort_keys=True) + "\n", encoding="utf-8")
            print(f"OK: wrote {GOLDEN.relative_to(ROOT)}")
            return
        if args.write_mutation_report:
            report = MUTATE.run_mutation_suite(mutation_documents)
            require(not report["unexpected_accepted"], f"cannot record a report with unchecked leaves: {report['unexpected_accepted']}")
            MUTATION_REPORT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            print(f"OK: wrote {MUTATION_REPORT.relative_to(ROOT)} ({report['cases']} effective cases)")
            return
        if args.write_manifest:
            write_manifest()
            return
        verify_manifest()
        verify_package_boundary()
        verify_historical_manifests()
        verify_golden(positives)
        negatives = 0
        negatives += verify_strict_types(positives["below"])
        negatives += verify_tensor_order_and_strict_entries(positives["below"])
        negatives += verify_origin_classification(positives["below"])
        negatives += verify_derived_and_observations(fixture, positives["below"], positives["hole"])
        negatives += verify_defense_in_depth(positives["below"])
        negatives += verify_trust_roots(fixture, positives["below"], positives["hole"])
        verify_mutation_coverage(mutation_documents)
        verify_fixed_pytest()
        print(
            f"OK: P2-0 prototype acceptance (2 golden positive instances, {negatives} handwritten "
            "reasoned negatives, 1 handwritten volatile positive control, 3-class mutation suite, "
            "54+6+1 fixed pytest nodes, CPU-only; P2 confirmation/activation not claimed)"
        )


if __name__ == "__main__":
    main()
