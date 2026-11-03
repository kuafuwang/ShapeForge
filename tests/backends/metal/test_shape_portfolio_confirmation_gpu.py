"""P2-B real-Mac confirmation slices: fresh rebuild + R5 identity on unseen shards."""

from __future__ import annotations

import copy
import json
import os
import platform

import pytest

from tcenv.identity import fingerprint
from tcenv.opspec.v2 import instantiate_v2
from tcenv.portfolio.candidate import parse_portfolio
from tcenv.portfolio.dispatch import DispatchRequest
from tcenv.portfolio.domain import DomainAuthority, normalize_semantic_domain_plan
from tcenv.backends.metal.portfolio_confirmation import confirm_subject, verify_subject_document
from tcenv.portfolio_validation.validation import PortfolioFreeze, ValidationPlan
from tcenv.target import DEFAULT_TARGET_ID, load_target
from tcenv.taskcontract.locator import discover_standard_tasks
from tcenv.track import REPO_ROOT
from tests.support.shape_portfolio_metal import (
    evidence_all_valid,
    load_p1_task,
    p1_instances,
    parent_entries,
    portfolio_document,
)

MASTER = b"0123456789abcdef"
TUNE = ("fp16/large/golden", "fp16/large/benchmark")
REQUIRED = tuple(
    f"fp16/{size}/{usage}"
    for size in ("below", "threshold", "above", "tail", "hole", "boundary")
    for usage in ("golden", "benchmark")
)


def _fixture(tmp_path):
    root = tmp_path / "tasks"
    contract = load_p1_task(root)
    locator = discover_standard_tasks(root, code_root=REPO_ROOT)[0]
    contract = locator.load()
    instances = p1_instances(contract)
    target = load_target(DEFAULT_TARGET_ID)
    plan = normalize_semantic_domain_plan(contract.spec, instances, target=target)
    authority = DomainAuthority(
        plan=plan,
        spec=contract.spec,
        target=target,
        instances=instances,
        contract=contract,
    )
    portfolio = parse_portfolio(parent_entries(portfolio_document(plan.digest)), domain_plan=plan)
    freeze = PortfolioFreeze(
        protocol_epoch=1,
        authority=authority,
        candidate_entries=portfolio.entries,
        task_pack_digest=locator.deployment_pack_digest,
        guidance_digest=fingerprint({"guidance": "P2-A fixed guidance"}),
    )
    validation = ValidationPlan.derive(
        freeze,
        tune_shards=TUNE,
        unseen_count=len(REQUIRED),
        master_seed=MASTER,
        required_instance_keys=REQUIRED,
    )
    return locator, plan, portfolio, freeze, validation


def _request(contract, plan, size, usage):
    instance = instantiate_v2(contract.spec, "fp16", size)
    point = plan.find("fp16", size, usage)
    assert point is not None
    return DispatchRequest.from_instance(instance, target_abi_digest=point.target_abi_digest, usage=usage)


gpu = pytest.mark.gpu


@gpu
def test_p2_b_unseen_golden_confirmation_is_fresh_and_identity_bound(tmp_path):
    locator, plan, portfolio, freeze, validation = _fixture(tmp_path)
    contract = locator.load()
    request = _request(contract, plan, "below", "golden")
    document = confirm_subject(
        locator=locator,
        domain_plan=plan,
        portfolio=portfolio,
        request=request,
        evidence=evidence_all_valid(portfolio),
        validation=validation,
        freeze=freeze,
    )
    assert document["status"] == "confirmed", document["reason"]
    assert document["identity"]["kind"] == "match"
    assert document["search_seed"] != document["confirmation_seed"]
    assert document["search_report"]["seed"] == 7
    assert document["confirmation_report"]["seed"] == validation.fresh_seeds[request.instance_key]
    assert document["confirmation_report"]["cache_hit"] is False
    assert document["confirmation_report"]["build"]["shape"] == "full"
    assert document["confirmation_report"]["build"]["build_kind"] == "native"
    assert document["credits_variant"] is True
    assert verify_subject_document(copy.deepcopy(document)) == document


@gpu
def test_p2_b_unseen_benchmark_confirmation_preserves_the_timing_gate(tmp_path):
    locator, plan, portfolio, freeze, validation = _fixture(tmp_path)
    contract = locator.load()
    request = _request(contract, plan, "above", "benchmark")
    document = confirm_subject(
        locator=locator,
        domain_plan=plan,
        portfolio=portfolio,
        request=request,
        evidence=evidence_all_valid(portfolio),
        validation=validation,
        freeze=freeze,
    )
    if document["status"] == "rejected":
        assert document["reason"] == "timing_uncertain"
        from tcenv.bench import timing_is_uncertain

        assert timing_is_uncertain(document["confirmation_report"])
    else:
        assert document["status"] == "confirmed", document["reason"]
        from tcenv.bench import timing_is_uncertain

        assert not timing_is_uncertain(document["confirmation_report"])
    assert document["usage"] == "benchmark"
    assert document["confirmation_report"]["mode"] == "full"
    assert document["confirmation_report"]["benchmark"]
    assert document["identity"]["kind"] == "match"
    # This fixture is correctness-only: the timing gate applies, but no speedup is published.
    assert document["confirmation_report"]["speedup"] is None


def _record_suite(documents, freeze, validation, filename="unseen-confirmations.json") -> None:
    directory = os.environ.get("SHAPE_PORTFOLIO_P2_EVIDENCE_DIR")
    if not directory:
        return
    from pathlib import Path

    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_id": "tcenv.portfolio.p2-confirmation-suite.1",
        "host": {"chip": platform.processor(), "machine": platform.machine(), "system": platform.system()},
        "portfolio_freeze_digest": freeze.digest,
        "validation_plan_digest": validation.digest,
        "documents": documents,
    }
    (target / filename).write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


@gpu
def test_p2_b_every_unseen_subject_gets_an_independent_fresh_confirmation(tmp_path):
    locator, plan, portfolio, freeze, validation = _fixture(tmp_path)
    contract = locator.load()
    documents = []
    for instance_key in validation.unseen_instance_keys:
        _dtype, size, usage = instance_key.split("/")
        document = confirm_subject(
            locator=locator,
            domain_plan=plan,
            portfolio=portfolio,
            request=_request(contract, plan, size, usage),
            evidence=evidence_all_valid(portfolio),
            validation=validation,
            freeze=freeze,
        )
        if document["status"] == "rejected":
            assert usage == "benchmark" and document["reason"] == "timing_uncertain", (instance_key, document["reason"])
        else:
            assert document["status"] == "confirmed", (instance_key, document["reason"])
        assert document["identity"]["kind"] == "match", instance_key
        assert document["confirmation_report"]["cache_hit"] is False
        assert document["confirmation_report"]["build"]["shape"] == "full"
        documents.append(document)
    assert len(documents) == len(REQUIRED) == 12
    assert len({document["confirmation_digest"] for document in documents}) == len(documents)
    assert any(document["provider"] == "fallback" for document in documents)
    assert all(document["credits_variant"] is False for document in documents if document["provider"] == "fallback")
    _record_suite(documents, freeze, validation)


@gpu
def test_p2_c_tune_instances_have_independent_fresh_confirmation_for_activation(tmp_path):
    locator, plan, portfolio, freeze, validation = _fixture(tmp_path)
    contract = locator.load()
    documents = []
    for instance_key in validation.tune_shards:
        _dtype, size, usage = instance_key.split("/")
        document = confirm_subject(
            locator=locator,
            domain_plan=plan,
            portfolio=portfolio,
            request=_request(contract, plan, size, usage),
            evidence=evidence_all_valid(portfolio),
            validation=validation,
            freeze=freeze,
        )
        if document["status"] == "rejected":
            assert usage == "benchmark" and document["reason"] == "timing_uncertain", (instance_key, document["reason"])
        else:
            assert document["status"] == "confirmed", (instance_key, document["reason"])
        assert document["coverage_kind"] == "tune_confirmation"
        assert document["confirmation_seed"] != 7
        documents.append(document)
    assert len(documents) == 2
    _record_suite(documents, freeze, validation, filename="tune-confirmations.json")


@gpu
def test_p2_b_unseen_fallback_confirmation_never_credits_the_variant(tmp_path):
    locator, plan, portfolio, freeze, validation = _fixture(tmp_path)
    contract = locator.load()
    request = _request(contract, plan, "hole", "golden")
    document = confirm_subject(
        locator=locator,
        domain_plan=plan,
        portfolio=portfolio,
        request=request,
        evidence=evidence_all_valid(portfolio),
        validation=validation,
        freeze=freeze,
    )
    assert document["status"] == "confirmed", document["reason"]
    assert document["provider"] == "fallback"
    assert document["credits_variant"] is False
    assert document["confirmation_report"]["status"] == "ok"
    forged = copy.deepcopy(document)
    forged["credits_variant"] = True
    with pytest.raises(ValueError, match="fallback_credits_variant|semantic projection"):
        verify_subject_document(forged)
