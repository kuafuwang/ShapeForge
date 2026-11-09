"""P2-B CPU negatives: pack binding and archive-suite semantics (no GPU work)."""

from __future__ import annotations

import copy
import importlib.util
import json
import shutil
from dataclasses import replace

import pytest

from taic.policy import compare_confirmation_identity
from taic.portfolio_manifest import DispatchManifest
from taic.task_binding import bind_task

from tcenv.backends.metal import portfolio_confirmation
from tcenv.backends.metal.portfolio_confirmation import activation_expectation, confirm_subject, verify_subject_document
from tcenv.contract_errors import TaskContractError
from tcenv.identity import fingerprint, load_strict_json
from tcenv.opspec.v2 import instantiate_v2
from tcenv.portfolio.candidate import parse_portfolio
from tcenv.portfolio.dispatch import DispatchRequest
from tcenv.portfolio.domain import DomainAuthority, normalize_semantic_domain_plan
from tcenv.portfolio_validation.validation import PortfolioFreeze, ValidationPlan
from tcenv.target import load_target
from tcenv.taskcontract.locator import discover_standard_tasks
from tcenv.track import REPO_ROOT
from tests.support.shape_portfolio_metal import (
    evidence_all_valid,
    load_p1_task,
    p1_instances,
    parent_entries,
    portfolio_document,
)

ROOT = REPO_ROOT
ARCHIVE = ROOT / "tests/support/golden/shared/shape-portfolio-阶段验收/P2/evidence/unseen-confirmations.json"
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
    return locator, plan, portfolio, freeze, validation


def _request(contract, plan, size="above", usage="benchmark"):
    instance = instantiate_v2(contract.spec, "fp16", size)
    point = plan.find("fp16", size, usage)
    return DispatchRequest.from_instance(instance, target_abi_digest=point.target_abi_digest, usage=usage)


def _archived_subject(key):
    return copy.deepcopy(next(document for document in load_strict_json(ARCHIVE)["documents"] if document["instance_key"] == key))


def _write_suite(tmp_path, documents):
    source = load_strict_json(ARCHIVE)
    source["documents"] = documents
    directory = tmp_path / "evidence"
    directory.mkdir()
    (directory / "unseen-confirmations.json").write_text(json.dumps(source), encoding="utf-8")
    return directory


def _verify_suite(directory, locator, freeze, validation):
    path = ROOT / "tests/support/golden/shared/shape-portfolio-阶段验收/P2/verify.py"
    spec = importlib.util.spec_from_file_location("p2_acceptance_verify", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    from tests.support.boundaries import rebind_acceptance_root

    rebind_acceptance_root(module)
    return module.verify_confirmation_suite(directory, locator, freeze, validation)


def test_confirmation_uses_the_tasks_declared_cases_instead_of_normal(monkeypatch):
    locator = bind_task("varlen_kda", "fp16", "mixed").locator
    captured = {}

    class FakeEvaluator:
        def __init__(self, **kwargs):
            captured["evaluator"] = kwargs

    class FakeBuilder:
        def __init__(self, **kwargs):
            captured["builder"] = kwargs

    sentinel = object()
    monkeypatch.setattr(portfolio_confirmation, "MetalTrackEvaluator", FakeEvaluator)
    monkeypatch.setattr(portfolio_confirmation, "MetalTrackBuilder", FakeBuilder)
    monkeypatch.setattr(portfolio_confirmation, "run_portfolio_task", lambda **kwargs: sentinel)
    result = portfolio_confirmation._evaluate(
        locator,
        object(),
        object(),
        object(),
        object(),
        mode="quick",
        seed=7,
        reps=3,
        target_id="apple-m4-max",
    )
    assert result is sentinel
    assert captured["evaluator"]["cases"] == tuple(locator.load().instance.case_domain.cases)
    assert captured["evaluator"]["cases"][0] == "canonical"
    assert "normal" not in captured["evaluator"]["cases"]


def test_p2_b_a_different_task_pack_is_refused_before_any_build(tmp_path, monkeypatch):
    locator, plan, portfolio, freeze, validation = _fixture(tmp_path)
    changed_root = tmp_path / "changed"
    shutil.copytree(tmp_path / "tasks", changed_root)
    (changed_root / locator.task_id / "known_good.metal").write_text("// changes the standard pack closure\n", encoding="utf-8")
    changed = discover_standard_tasks(changed_root, code_root=REPO_ROOT)[0]
    assert changed.deployment_pack_digest != locator.deployment_pack_digest
    calls = []
    monkeypatch.setattr(portfolio_confirmation, "_evaluate", lambda *args, **kwargs: calls.append((args, kwargs)))
    with pytest.raises(TaskContractError, match="not the frozen task pack"):
        confirm_subject(
            locator=changed,
            domain_plan=plan,
            portfolio=portfolio,
            request=_request(locator.load(), plan),
            evidence=evidence_all_valid(portfolio),
            validation=validation,
            freeze=freeze,
        )
    assert calls == []


def test_p2_b_archive_verifier_rejects_usage_and_mode_flips():
    changed_usage = _archived_subject("fp16/above/benchmark")
    changed_usage["usage"] = "golden"
    with pytest.raises(ValueError, match="declared usage"):
        verify_subject_document(changed_usage)
    changed_mode = _archived_subject("fp16/above/benchmark")
    changed_mode["confirmation_report"]["mode"] = "quick"
    with pytest.raises(ValueError, match="measurement_mode_mismatch|semantic projection"):
        verify_subject_document(changed_mode)


def test_p2_b_suite_rejects_a_seed_not_derived_from_the_validation_plan(tmp_path):
    locator, plan, portfolio, freeze, validation = _fixture(tmp_path)
    document = load_strict_json(ARCHIVE)
    changed = copy.deepcopy(document["documents"])
    target = next(item for item in changed if item["instance_key"] == "fp16/above/benchmark")
    target["confirmation_seed"] += 1
    target["confirmation_report"]["seed"] = target["confirmation_seed"]
    target["confirmation_digest"] = fingerprint(
        {key: value for key, value in target.items() if key != "confirmation_digest"}
    )
    directory = _write_suite(tmp_path, changed)
    with pytest.raises(TaskContractError, match="not derived from this validation plan"):
        _verify_suite(directory, locator, freeze, validation)


def test_p2_b_suite_rejects_a_locator_whose_pack_differs_from_the_freeze(tmp_path):
    locator, plan, portfolio, freeze, validation = _fixture(tmp_path)
    changed_root = tmp_path / "changed-suite"
    shutil.copytree(tmp_path / "tasks", changed_root)
    (changed_root / locator.task_id / "known_good.metal").write_text("// pack drift\n", encoding="utf-8")
    changed = discover_standard_tasks(changed_root, code_root=REPO_ROOT)[0]
    directory = _write_suite(tmp_path, load_strict_json(ARCHIVE)["documents"])
    with pytest.raises(AssertionError, match="not the frozen task pack"):
        _verify_suite(directory, changed, freeze, validation)


def test_p2_b_suite_rejects_expected_and_report_coordinated_forgery(tmp_path):
    locator, plan, portfolio, freeze, validation = _fixture(tmp_path)
    changed = copy.deepcopy(load_strict_json(ARCHIVE)["documents"])
    target = next(item for item in changed if item["instance_key"] == "fp16/above/benchmark")
    target["expected"]["build"]["lowering_profile_digest"] = "f" * 64
    target["search_report"]["build"]["lowering_profile_digest"] = "f" * 64
    target["confirmation_report"]["build"]["lowering_profile_digest"] = "f" * 64
    target["identity"] = compare_confirmation_identity(
        candidate_sha256=target["candidate_sha256"],
        expected=target["expected"],
        search=target["search_report"],
        confirmation=target["confirmation_report"],
        scope="local",
    )
    target["confirmation_digest"] = fingerprint(
        {key: value for key, value in target.items() if key != "confirmation_digest"}
    )
    directory = _write_suite(tmp_path, changed)
    with pytest.raises(TaskContractError, match="not independently rebuilt"):
        _verify_suite(directory, locator, freeze, validation)


def test_p2_b_suite_replays_provider_and_credit_instead_of_trusting_sidecar(tmp_path):
    locator, plan, portfolio, freeze, validation = _fixture(tmp_path)
    document = load_strict_json(ARCHIVE)
    changed = copy.deepcopy(document["documents"])
    target = next(item for item in changed if item["instance_key"] == "fp16/hole/golden")
    target["provider"] = "candidate"
    target["variant_id"] = next(item["variant_id"] for item in changed if item["provider"] == "candidate")
    target["fallback_id"] = None
    target["credits_variant"] = True
    target["confirmation_digest"] = fingerprint(
        {key: value for key, value in target.items() if key != "confirmation_digest"}
    )
    directory = _write_suite(tmp_path, changed)
    with pytest.raises(TaskContractError, match="provider differs from dispatch replay"):
        _verify_suite(directory, locator, freeze, validation)


def test_p2_b_activation_expectation_rejects_a_consistent_cross_pack_repin(tmp_path):
    locator, plan, portfolio, freeze, validation = _fixture(tmp_path)
    changed_root = tmp_path / "changed-activation-pack"
    shutil.copytree(tmp_path / "tasks", changed_root)
    reference = changed_root / locator.task_id / "reference.py"
    reference.write_text(reference.read_text(encoding="utf-8") + "\n# pack B\n", encoding="utf-8")
    changed = discover_standard_tasks(changed_root, code_root=REPO_ROOT)[0]
    assert changed.deployment_pack_digest != freeze.task_pack_digest
    documents = []
    for name in ("unseen-confirmations.json", "tune-confirmations.json"):
        path = ROOT / f"tests/support/golden/shared/shape-portfolio-阶段验收/P2/evidence/{name}"
        documents.extend(copy.deepcopy(load_strict_json(path)["documents"]))

    def activate_with_pack_b_expectations():
        expectations = {}
        for document in documents:
            record, expected = activation_expectation(
                locator=changed, freeze=freeze, validation=validation, document=document
            )
            expectations[document["instance_key"]] = (record, expected)
            document["expected"] = copy.deepcopy(expected)
            for period in ("search_report", "confirmation_report"):
                report = document[period]
                report["build"]["lowering_profile_digest"] = expected["build"]["lowering_profile_digest"]
                for key, value in expected["evaluation"].items():
                    report[key] = copy.deepcopy(value)
            document["identity"] = compare_confirmation_identity(
                candidate_sha256=document["candidate_sha256"],
                expected=document["expected"],
                search=document["search_report"],
                confirmation=document["confirmation_report"],
                scope="local",
            )
            document["status"] = "confirmed"
            document["reason"] = None
            document["confirmation_digest"] = fingerprint(
                {key: value for key, value in document.items() if key != "confirmation_digest"}
            )
        return DispatchManifest.from_confirmations(
            freeze, validation, documents, expectations, locator=locator
        ).activate(freeze, validation)

    with pytest.raises(TaskContractError, match="not the frozen task pack"):
        activate_with_pack_b_expectations()
    wrong_op = replace(locator, op_name="another_shape_portfolio_task")
    with pytest.raises(TaskContractError, match="activation locator names another task"):
        activation_expectation(locator=wrong_op, freeze=freeze, validation=validation, document=documents[0])


def test_p2_b_manifest_rejects_expectations_rebuilt_under_another_freeze(tmp_path):
    locator, plan, portfolio, freeze, validation = _fixture(tmp_path)
    changed_root = tmp_path / "independently-frozen-pack-b"
    shutil.copytree(tmp_path / "tasks", changed_root)
    reference = changed_root / locator.task_id / "reference.py"
    reference.write_text(reference.read_text(encoding="utf-8") + "\n# independently frozen pack B\n", encoding="utf-8")
    changed = discover_standard_tasks(changed_root, code_root=REPO_ROOT)[0]
    freeze_b = PortfolioFreeze(
        protocol_epoch=freeze.protocol_epoch,
        authority=freeze.authority,
        candidate_entries=freeze.candidate_entries,
        task_pack_digest=changed.deployment_pack_digest,
        guidance_digest=freeze.guidance_digest,
    )
    validation_b = ValidationPlan.derive(
        freeze_b,
        tune_shards=validation.tune_shards,
        unseen_count=len(validation.unseen_instance_keys),
        master_seed=MASTER,
        required_instance_keys=validation.required_instance_keys,
    )
    documents = []
    for name in ("unseen-confirmations.json", "tune-confirmations.json"):
        path = ROOT / f"tests/support/golden/shared/shape-portfolio-阶段验收/P2/evidence/{name}"
        documents.extend(copy.deepcopy(load_strict_json(path)["documents"]))
    expectations = {}
    changed_expected = 0
    for document in documents:
        previous_expected = copy.deepcopy(document["expected"])
        record, expected = activation_expectation(
            locator=changed, freeze=freeze_b, validation=validation_b, document=document
        )
        expectations[document["instance_key"]] = (record, expected)
        document["expected"] = copy.deepcopy(expected)
        changed_expected += int(expected != previous_expected)
        for period in ("search_report", "confirmation_report"):
            report = document[period]
            report["build"]["lowering_profile_digest"] = expected["build"]["lowering_profile_digest"]
            for key, value in expected["evaluation"].items():
                report[key] = copy.deepcopy(value)
        document["identity"] = compare_confirmation_identity(
            candidate_sha256=document["candidate_sha256"],
            expected=document["expected"],
            search=document["search_report"],
            confirmation=document["confirmation_report"],
            scope="local",
        )
        document["status"] = "confirmed"
        document["reason"] = None
        document["confirmation_digest"] = fingerprint(
            {key: value for key, value in document.items() if key != "confirmation_digest"}
        )
    assert changed_expected == len(documents) == 14
    with pytest.raises(TaskContractError, match="not the frozen task pack"):
        DispatchManifest.from_confirmations(
            freeze, validation, documents, expectations, locator=locator
        ).activate(freeze, validation)


def test_p2_b_lying_locator_cannot_bind_pack_b_bytes_to_freeze_a(tmp_path):
    locator, plan, portfolio, freeze, validation = _fixture(tmp_path)
    changed_root = tmp_path / "lying-locator-pack-b"
    shutil.copytree(tmp_path / "tasks", changed_root)
    reference = changed_root / locator.task_id / "reference.py"
    reference.write_text(reference.read_text(encoding="utf-8") + "\n# lying locator pack B\n", encoding="utf-8")
    changed = discover_standard_tasks(changed_root, code_root=REPO_ROOT)[0]
    lying = replace(changed, deployment_pack_digest=locator.deployment_pack_digest)
    assert lying.identity == locator.identity and lying.task_dir != locator.task_dir
    documents = []
    for name in ("unseen-confirmations.json", "tune-confirmations.json"):
        path = ROOT / f"tests/support/golden/shared/shape-portfolio-阶段验收/P2/evidence/{name}"
        documents.extend(copy.deepcopy(load_strict_json(path)["documents"]))

    def activate_with_lying_locator():
        expectations = {}
        for document in documents:
            record, expected = activation_expectation(
                locator=lying, freeze=freeze, validation=validation, document=document
            )
            expectations[document["instance_key"]] = (record, expected)
            document["expected"] = copy.deepcopy(expected)
            for period in ("search_report", "confirmation_report"):
                report = document[period]
                report["build"]["lowering_profile_digest"] = expected["build"]["lowering_profile_digest"]
                for key, value in expected["evaluation"].items():
                    report[key] = copy.deepcopy(value)
            document["identity"] = compare_confirmation_identity(
                candidate_sha256=document["candidate_sha256"],
                expected=document["expected"],
                search=document["search_report"],
                confirmation=document["confirmation_report"],
                scope="local",
            )
            document["status"] = "confirmed"
            document["reason"] = None
            document["confirmation_digest"] = fingerprint(
                {key: value for key, value in document.items() if key != "confirmation_digest"}
            )
        return DispatchManifest.from_confirmations(
            freeze, validation, documents, expectations, locator=lying
        ).activate(freeze, validation)

    with pytest.raises(ValueError, match="drifted|does not match|not the frozen task pack"):
        activation_expectation(locator=lying, freeze=freeze, validation=validation, document=documents[0])
    with pytest.raises(ValueError, match="drifted|does not match|not the frozen task pack"):
        activate_with_lying_locator()


def test_p2_b_confirmation_attempts_require_a_strict_bounded_integer():
    for attempts in (True, 1.5, 3.0):
        forged = _archived_subject("fp16/above/benchmark")
        forged["confirmation_attempts"] = attempts
        forged["confirmation_digest"] = fingerprint(
            {key: value for key, value in forged.items() if key != "confirmation_digest"}
        )
        with pytest.raises(TaskContractError, match="invalid_attempt_count|semantic projection"):
            verify_subject_document(forged)
