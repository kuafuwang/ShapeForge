"""P2-C CPU tests: full-coverage DispatchManifest and active pointer lifecycle."""

from __future__ import annotations

import copy

import pytest

from taic.policy import compare_confirmation_identity

from tcenv.contract_errors import TaskContractError
from tcenv.identity import fingerprint, load_strict_json
from tcenv.portfolio.candidate import parse_portfolio
from tcenv.portfolio.domain import DomainAuthority, normalize_semantic_domain_plan
from tcenv.portfolio_validation.validation import PortfolioFreeze, ValidationPlan
from tcenv.target import load_target
from tcenv.taskcontract.locator import discover_standard_tasks
from tcenv.track import REPO_ROOT
from taic.portfolio_manifest import (
    ActivePortfolioPointer,
    DispatchManifest,
    verify_manifest_document,
)
from tcenv.backends.metal.portfolio_confirmation import activation_expectation
from tests.support.shape_portfolio_metal import (
    load_p1_task,
    p1_instances,
    parent_entries,
    portfolio_document,
)

ROOT = REPO_ROOT
MASTER = b"0123456789abcdef"
TUNE = ("fp16/large/golden", "fp16/large/benchmark")
REQUIRED = tuple(
    f"fp16/{size}/{usage}"
    for size in ("below", "threshold", "above", "tail", "hole", "boundary")
    for usage in ("golden", "benchmark")
)


def _fixture(tmp_path):
    root = tmp_path / "tasks"
    load_p1_task(root)
    locator = discover_standard_tasks(root, code_root=REPO_ROOT)[0]
    contract = locator.load()
    instances = p1_instances(contract)
    target = load_target("apple-m4-max")
    plan = normalize_semantic_domain_plan(contract.spec, instances, target=target)
    authority = DomainAuthority(plan=plan, spec=contract.spec, target=target, instances=instances, contract=contract)
    portfolio = parse_portfolio(parent_entries(portfolio_document(plan.digest)), domain_plan=plan)
    freeze = PortfolioFreeze(
        protocol_epoch=1,
        authority=authority,
        candidate_entries=portfolio.entries,
        task_pack_digest=locator.deployment_pack_digest,
        guidance_digest=fingerprint({"guidance": "P2-A fixed guidance"}),
    )
    validation = ValidationPlan.derive(
        freeze, tune_shards=TUNE, unseen_count=len(REQUIRED), master_seed=MASTER, required_instance_keys=REQUIRED
    )
    documents = []
    for name in ("unseen-confirmations.json", "tune-confirmations.json"):
        documents.extend(load_strict_json(ROOT / f"tests/support/golden/shared/shape-portfolio-阶段验收/P2-G/evidence/{name}")["documents"])
    return locator, freeze, validation, documents


def _from_documents(locator, freeze, validation, documents):
    expectations = {
        document["instance_key"]: activation_expectation(
            locator=locator,
            freeze=freeze,
            validation=validation,
            document=document,
        )
        for document in documents
    }
    return DispatchManifest.from_confirmations(freeze, validation, documents, expectations, locator=locator)


def _reseal(document):
    document["confirmation_digest"] = fingerprint(
        {key: value for key, value in document.items() if key != "confirmation_digest"}
    )


def test_p2_c_full_confirmation_activates_and_replays_manifest(tmp_path):
    locator, freeze, validation, documents = _fixture(tmp_path)
    validated = _from_documents(locator, freeze, validation, documents)
    assert validated.status == "validated" and validated.activation_generation == 0
    active = validated.activate(freeze, validation)
    assert active.status == "active" and active.activation_generation == 1
    assert len(active.subjects) == 14
    assert verify_manifest_document(active.document(), freeze, validation).document() == active.document()


def test_p2_c_missing_duplicate_or_unconfirmed_subjects_cannot_activate(tmp_path):
    locator, freeze, validation, documents = _fixture(tmp_path)
    with pytest.raises(TaskContractError, match="coverage mismatch"):
        _from_documents(locator, freeze, validation, documents[:-1])
    with pytest.raises(TaskContractError, match="duplicate instance keys"):
        _from_documents(locator, freeze, validation, [*documents, documents[0]])
    rejected = copy.deepcopy(documents)
    rejected[0]["status"] = "rejected"
    rejected[0]["reason"] = "timing_uncertain"
    with pytest.raises(TaskContractError, match="semantic projection"):
        _from_documents(locator, freeze, validation, rejected)


def test_p2_c_revocation_is_a_new_generation_and_cannot_be_reactivated(tmp_path):
    locator, freeze, validation, documents = _fixture(tmp_path)
    active = _from_documents(locator, freeze, validation, documents).activate(freeze, validation)
    revoked = active.revoke(freeze, reason="candidate source superseded")
    assert revoked.status == "revoked" and revoked.activation_generation == 2
    with pytest.raises(TaskContractError, match="only a validated manifest"):
        revoked.activate(freeze, validation)


def test_p2_c_active_pointer_round_trip_and_stale_rejection(tmp_path):
    locator, freeze, validation, documents = _fixture(tmp_path)
    active = _from_documents(locator, freeze, validation, documents).activate(freeze, validation)
    pointer = ActivePortfolioPointer(active.digest, active.activation_generation)
    path = tmp_path / "active_portfolio_pointer.json"
    pointer.write_atomic(path)
    assert ActivePortfolioPointer.read(path, active).document() == pointer.document()
    stale = ActivePortfolioPointer("0" * 64, active.activation_generation)
    stale.write_atomic(path)
    with pytest.raises(TaskContractError, match="stale"):
        ActivePortfolioPointer.read(path, active)
    revoked = active.revoke(freeze, reason="retired")
    with pytest.raises(TaskContractError, match="non-active"):
        ActivePortfolioPointer.read(path, revoked)


def test_p2_c_manifest_digest_and_status_cannot_be_forged(tmp_path):
    locator, freeze, validation, documents = _fixture(tmp_path)
    active = _from_documents(locator, freeze, validation, documents).activate(freeze, validation)
    forged = active.document()
    forged["dispatch_manifest_digest"] = "0" * 64
    with pytest.raises(TaskContractError, match="canonical projection"):
        verify_manifest_document(forged, freeze, validation)
    forged = active.document()
    forged["status"] = "revoked"
    with pytest.raises(TaskContractError, match="revocation_reason"):
        verify_manifest_document(forged, freeze, validation)


def test_p2_c_fallback_credit_and_subject_identity_are_closed(tmp_path):
    locator, freeze, validation, documents = _fixture(tmp_path)
    fallback = next(document for document in documents if document["provider"] == "fallback")
    fallback = copy.deepcopy(fallback)
    fallback["credits_variant"] = True
    fallback["status"] = "rejected"
    fallback["reason"] = "fallback_credits_variant"
    _reseal(fallback)
    forged_documents = [
        fallback if document["instance_key"] == fallback["instance_key"] else document for document in documents
    ]
    with pytest.raises(TaskContractError, match="variant credit differs from dispatch replay"):
        _from_documents(locator, freeze, validation, forged_documents)


def test_p2_c_manifest_recomputes_sidecar_verdict_before_activation(tmp_path):
    locator, freeze, validation, documents = _fixture(tmp_path)
    forged = copy.deepcopy(documents)
    forged[0]["confirmation_attempts"] = 99
    forged[0]["status"] = "rejected"
    forged[0]["reason"] = "invalid_attempt_count"
    _reseal(forged[0])
    with pytest.raises(TaskContractError, match="invalid_attempt_count"):
        _from_documents(locator, freeze, validation, forged)
    forged = copy.deepcopy(documents)
    forged[0]["search_report"] = {"build": {}, "status": "failed", "correctness": {"ok": False}}
    with pytest.raises(Exception):
        _from_documents(locator, freeze, validation, forged)


def test_p2_c_manifest_binds_provider_plan_and_coverage_kind(tmp_path):
    locator, freeze, validation, documents = _fixture(tmp_path)
    forged = copy.deepcopy(documents)
    candidate = next(item for item in forged if item["provider"] == "candidate")
    candidate["variant_id"] = "a" * 64
    _reseal(candidate)
    with pytest.raises(TaskContractError, match="variant identity differs from dispatch replay"):
        _from_documents(locator, freeze, validation, forged)
    forged = copy.deepcopy(documents)
    fallback = next(item for item in forged if item["provider"] == "fallback")
    fallback["fallback_id"] = "b" * 64
    _reseal(fallback)
    with pytest.raises(TaskContractError, match="fallback identity differs from dispatch replay"):
        _from_documents(locator, freeze, validation, forged)
    forged = copy.deepcopy(documents)
    tune = next(item for item in forged if item["coverage_kind"] == "tune_confirmation")
    tune["coverage_kind"] = "sampled_unseen"
    _reseal(tune)
    with pytest.raises(TaskContractError, match="coverage_kind"):
        _from_documents(locator, freeze, validation, forged)


def test_p2_c_manifest_rejects_another_validation_plan_digest(tmp_path):
    locator, freeze, validation, documents = _fixture(tmp_path)
    active = _from_documents(locator, freeze, validation, documents).activate(freeze, validation)
    forged = active.document()
    forged["validation_plan_digest"] = "0" * 64
    forged["dispatch_manifest_digest"] = fingerprint(
        {key: value for key, value in forged.items() if key != "dispatch_manifest_digest"}
    )
    with pytest.raises(TaskContractError, match="another freeze or validation plan"):
        verify_manifest_document(forged, freeze, validation)


def test_p2_c_pointer_generation_floor_blocks_rollback(tmp_path):
    locator, freeze, validation, documents = _fixture(tmp_path)
    active = _from_documents(locator, freeze, validation, documents).activate(freeze, validation)
    pointer = ActivePortfolioPointer(active.digest, active.activation_generation)
    path = tmp_path / "active_portfolio_pointer.json"
    pointer.write_atomic(path)
    revoked = active.revoke(freeze, reason="superseded")
    assert revoked.activation_generation == 2
    with pytest.raises(TaskContractError, match="below the required generation floor"):
        ActivePortfolioPointer.read(path, active, minimum_generation=revoked.activation_generation)


def test_p2_c_document_only_manifest_cannot_be_activated(tmp_path):
    locator, freeze, validation, documents = _fixture(tmp_path)
    active = _from_documents(locator, freeze, validation, documents).activate(freeze, validation)
    document = active.document()
    document["status"] = "validated"
    document["activation_generation"] = 0
    document["dispatch_manifest_digest"] = fingerprint(
        {key: value for key, value in document.items() if key != "dispatch_manifest_digest"}
    )
    restored = DispatchManifest.from_document(document)
    with pytest.raises(TaskContractError, match="without confirmed sidecars"):
        restored.activate(freeze, validation)


def test_p2_c_activation_binds_candidate_bytes_to_selected_variant(tmp_path):
    locator, freeze, validation, documents = _fixture(tmp_path)
    forged = copy.deepcopy(documents)
    source = next(item for item in forged if item["instance_key"] == "fp16/above/golden")
    target = next(item for item in forged if item["instance_key"] == "fp16/below/golden")
    source["variant_id"] = target["variant_id"]
    _reseal(source)
    with pytest.raises(TaskContractError, match="variant identity differs from dispatch replay"):
        _from_documents(locator, freeze, validation, forged)


def test_p2_c_activation_binds_plan_seed(tmp_path):
    locator, freeze, validation, documents = _fixture(tmp_path)
    forged = copy.deepcopy(documents)
    source = next(item for item in forged if item["instance_key"] == "fp16/below/golden")
    other = next(item for item in forged if item["instance_key"] == "fp16/above/golden")
    source["confirmation_seed"] = other["confirmation_seed"]
    source["confirmation_report"]["seed"] = other["confirmation_seed"]
    _reseal(source)
    with pytest.raises(TaskContractError, match="not derived from this validation plan"):
        _from_documents(locator, freeze, validation, forged)


def test_p2_c_activation_binds_frozen_target(tmp_path):
    locator, freeze, validation, documents = _fixture(tmp_path)
    forged = copy.deepcopy(documents)
    source = forged[0]
    source["expected"]["target_id"] = "apple-m4-other"
    source["expected"]["evaluation"]["target_id"] = "apple-m4-other"
    source["search_report"]["target_id"] = "apple-m4-other"
    source["confirmation_report"]["target_id"] = "apple-m4-other"
    source["identity"] = compare_confirmation_identity(
        candidate_sha256=source["candidate_sha256"],
        expected=source["expected"],
        search=source["search_report"],
        confirmation=source["confirmation_report"],
        scope="local",
    )
    _reseal(source)
    with pytest.raises(TaskContractError, match="target differs from the frozen authority"):
        _from_documents(locator, freeze, validation, forged)


def test_p2_c_activation_requires_expectation_for_every_subject(tmp_path):
    locator, freeze, validation, documents = _fixture(tmp_path)
    with pytest.raises(TaskContractError, match="cover exactly the confirmation subjects"):
        DispatchManifest.from_confirmations(freeze, validation, documents, {}, locator=locator)


def test_p2_c_document_only_active_manifest_cannot_have_an_active_pointer(tmp_path):
    locator, freeze, validation, documents = _fixture(tmp_path)
    active = _from_documents(locator, freeze, validation, documents).activate(freeze, validation)
    restored = DispatchManifest.from_document(active.document())
    assert restored.status == "active" and not restored._derived_from_confirmations
    path = tmp_path / "active_portfolio_pointer.json"
    ActivePortfolioPointer(restored.digest, restored.activation_generation).write_atomic(path)
    with pytest.raises(TaskContractError, match="document-only active manifest"):
        ActivePortfolioPointer.read(path, restored)
