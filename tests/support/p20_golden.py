"""P2-0 golden replay helpers copied from the closed ``P2-0/verify.py``.

The original verifier is a closed package and must not change.  If these copied
functions have a bug, fix them here and record the original verifier's drift in
a new acceptance epoch.  This module is a library, not an acceptance verifier.
"""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
from typing import Any, Mapping

from tcenv.identity import fingerprint
from tcenv.opspec.v2 import instantiate_v2
from tcenv.portfolio.candidate import parse_portfolio, project_provider
from tcenv.portfolio.dispatch import DispatchRequest
from tcenv.portfolio.domain import normalize_semantic_domain_plan
from tcenv.portfolio_evidence.projection import (
    FrozenPortfolioInputs, RawPortfolioObservation, project_document, stable_core, verify_document,
)
from tcenv.target import load_target
from tests.support.shape_portfolio_metal import (
    evidence_all_valid, load_p1_task, p1_instances, parent_entries, portfolio_document,
)

ROOT = Path(__file__).resolve().parents[2]
GOLDEN = ROOT / "tests/support/golden/shared/shape-portfolio-阶段验收/P2-0/golden.json"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


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

