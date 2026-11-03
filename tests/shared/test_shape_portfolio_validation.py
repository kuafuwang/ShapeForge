"""P2-A CPU tests: authoritative freeze and two-stage validation-shard isolation."""

from __future__ import annotations

import json

import pytest

from tcenv.contract_errors import TaskContractError
from tcenv.identity import fingerprint
from tcenv.portfolio.candidate import parse_portfolio
from tcenv.portfolio.domain import DomainAuthority, normalize_semantic_domain_plan
from tcenv.portfolio_validation.validation import (
    PortfolioFreeze,
    ValidationPlan,
    loads_portfolio_freeze,
    loads_validation_plan,
)
from tcenv.target import load_target
from tcenv.taskcontract.locator import discover_standard_tasks
from tcenv.track import REPO_ROOT
from tcenv.track.plane import CandidateEntry
from tests.support.shape_portfolio_metal import (
    load_p1_task,
    p1_instances,
    parent_entries,
    portfolio_document,
)

MASTER = b"0123456789abcdef"
OTHER_MASTER = b"fedcba9876543210"
TUNE = ("fp16/large/golden", "fp16/large/benchmark")
REQUIRED = tuple(
    f"fp16/{size}/{usage}"
    for size in ("below", "threshold", "above", "tail", "hole", "boundary")
    for usage in ("golden", "benchmark")
)


def _freeze(tmp_path):
    root = tmp_path / "tasks"
    load_p1_task(root)
    locator = discover_standard_tasks(root, code_root=REPO_ROOT)[0]
    contract = locator.load()
    target = load_target("apple-m4-max")
    instances = p1_instances(contract)
    plan = normalize_semantic_domain_plan(contract.spec, instances, target=target)
    authority = DomainAuthority(
        plan=plan,
        spec=contract.spec,
        target=target,
        instances=instances,
        contract=contract,
    )
    entries = parent_entries(portfolio_document(plan.digest))
    parse_portfolio(entries, domain_plan=plan)
    return PortfolioFreeze(
        protocol_epoch=1,
        authority=authority,
        candidate_entries=entries,
        task_pack_digest=locator.deployment_pack_digest,
        guidance_digest=fingerprint({"guidance": "P2-A fixed guidance"}),
    )


def _plan(freeze):
    return ValidationPlan.derive(
        freeze,
        tune_shards=TUNE,
        unseen_count=len(REQUIRED),
        master_seed=MASTER,
        required_instance_keys=REQUIRED,
    )


def test_p2_a_freeze_is_typed_deterministic_and_byte_bound(tmp_path):
    freeze = _freeze(tmp_path)
    restored = PortfolioFreeze.from_document(freeze.document(), authority=freeze.authority)
    assert restored.digest == freeze.digest
    assert restored.document() == freeze.document()
    assert freeze.opspec_digest == freeze.domain_plan.opspec_digest
    changed = json.loads(json.dumps(freeze.document(), sort_keys=True))
    assert PortfolioFreeze.from_document(changed, authority=freeze.authority).digest == freeze.digest
    with pytest.raises(ValueError, match="schema_id"):
        PortfolioFreeze.from_document({**freeze.document(), "schema_id": "wrong"}, authority=freeze.authority)
    with pytest.raises(ValueError, match="authoritative domain plan|domain_plan"):
        PortfolioFreeze.from_document({**freeze.document(), "domain_plan": None}, authority=freeze.authority)
    with pytest.raises(ValueError, match="repeats the key"):
        loads_portfolio_freeze('{"schema_id":"first","schema_id":"second"}', authority=freeze.authority)


def test_p2_a_required_points_are_always_unseen_and_cover_both_usages(tmp_path):
    freeze = _freeze(tmp_path)
    plan = _plan(freeze)
    assert not set(plan.tune_shards) & set(plan.unseen_instance_keys)
    assert set(REQUIRED) == set(plan.unseen_instance_keys)
    assert not set(REQUIRED) & set(plan.tune_shards)
    assert {key.rsplit("/", 1)[1] for key in plan.required_instance_keys} == {"golden", "benchmark"}
    assert set(plan.fresh_seeds) == set(plan.unseen_instance_keys)
    restored = ValidationPlan.from_document(plan.document())
    assert restored.digest == plan.digest
    restored.verify_fresh_seeds(freeze, master_seed=MASTER, expected_plan_digest=plan.digest)


def test_p2_a_candidate_view_cannot_leak_unseen_or_fresh_seeds(tmp_path):
    freeze = _freeze(tmp_path)
    plan = _plan(freeze)
    view = plan.candidate_view(freeze)
    payload = json.dumps(view.document(), sort_keys=True)
    assert set(view.tune_shards) == set(TUNE)
    for key in plan.unseen_instance_keys:
        assert key not in payload
    for seed in plan.fresh_seeds.values():
        assert str(seed) not in payload
    assert "master_seed" not in payload and "fresh_seeds" not in payload
    assert "unseen_instance_keys" not in payload
    assert type(view.tune_shards) is tuple
    restored = type(view).from_document(view.document())
    assert restored.document() == view.document()


def test_p2_a_resume_requires_same_master_freeze_and_plan_digest(tmp_path):
    freeze = _freeze(tmp_path)
    plan = _plan(freeze)
    assert ValidationPlan.derive(
        freeze, tune_shards=TUNE, unseen_count=len(REQUIRED), master_seed=MASTER, required_instance_keys=REQUIRED
    ).document() == plan.document()
    with pytest.raises(ValueError, match="fresh seeds"):
        plan.verify_fresh_seeds(freeze, master_seed=OTHER_MASTER, expected_plan_digest=plan.digest)
    with pytest.raises(ValueError, match="externally pinned plan"):
        plan.verify_fresh_seeds(freeze, master_seed=MASTER, expected_plan_digest="0" * 64)
    other = PortfolioFreeze(
        protocol_epoch=1,
        authority=freeze.authority,
        candidate_entries=freeze.candidate_entries,
        task_pack_digest="0" * 64,
        guidance_digest=freeze.guidance_digest,
    )
    with pytest.raises(ValueError, match="different portfolio freeze"):
        plan.verify_fresh_seeds(other, master_seed=MASTER, expected_plan_digest=plan.digest)


def test_p2_a_plan_and_candidate_view_are_immutable(tmp_path):
    freeze = _freeze(tmp_path)
    plan = _plan(freeze)
    digest = plan.digest
    with pytest.raises(TypeError):
        plan.fresh_seeds[next(iter(plan.fresh_seeds))] = 0
    with pytest.raises(AttributeError):
        plan.tune_shards.append("fp16/tail/golden")
    assert plan.digest == digest
    document = plan.document()
    restored = ValidationPlan.from_document(document)
    document["fresh_seeds"][next(iter(document["fresh_seeds"]))] = 0
    assert restored.digest == plan.digest
    view = restored.candidate_view(freeze)
    assert view.tune_shards is not restored.tune_shards or isinstance(view.tune_shards, tuple)
    with pytest.raises(AttributeError):
        view.tune_shards.append("fp16/tail/golden")


def test_p2_a_invalid_shard_seed_and_authority_contracts_are_refused(tmp_path):
    freeze = _freeze(tmp_path)
    with pytest.raises(ValueError, match="must be unseen, not tuned"):
        ValidationPlan.derive(
            freeze,
            tune_shards=["fp16/threshold/golden"],
            unseen_count=1,
            master_seed=MASTER,
            required_instance_keys=["fp16/threshold/golden"],
        )
    with pytest.raises(ValueError, match="outside the frozen domain"):
        ValidationPlan.derive(
            freeze,
            tune_shards=["fp16/not-a-size/golden"],
            unseen_count=1,
            master_seed=MASTER,
            required_instance_keys=["fp16/not-a-size/golden"],
        )
    with pytest.raises(ValueError, match="unseen_count"):
        ValidationPlan.derive(
            freeze, tune_shards=TUNE, unseen_count=True, master_seed=MASTER, required_instance_keys=REQUIRED
        )
    with pytest.raises(ValueError, match="unseen instances"):
        ValidationPlan.derive(
            freeze, tune_shards=TUNE, unseen_count=999, master_seed=MASTER, required_instance_keys=REQUIRED
        )
    document = _plan(freeze).document()
    with pytest.raises(ValueError, match="fields"):
        ValidationPlan.from_document({**document, "master_seed": "leaked"})
    bad = dict(document)
    bad["fresh_seeds"] = dict(document["fresh_seeds"])
    key = next(iter(bad["fresh_seeds"]))
    bad["fresh_seeds"][key] = True
    with pytest.raises(ValueError, match="fresh_seeds"):
        ValidationPlan.from_document(bad)
    with pytest.raises(ValueError, match="DomainAuthority"):
        PortfolioFreeze(
            protocol_epoch=1,
            authority=object(),
            candidate_entries=freeze.candidate_entries,
            task_pack_digest=freeze.task_pack_digest,
            guidance_digest=freeze.guidance_digest,
        )
    with pytest.raises(TaskContractError, match="CandidateEntry values"):
        PortfolioFreeze(
            protocol_epoch=1,
            authority=freeze.authority,
            candidate_entries=(object(),),
            task_pack_digest=freeze.task_pack_digest,
            guidance_digest=freeze.guidance_digest,
        )
    with pytest.raises(ValueError, match="repeats the key"):
        loads_validation_plan('{"schema_id":"first","schema_id":"second"}')


def test_p2_a_hmac_selects_remaining_unseen_after_required(tmp_path):
    freeze = _freeze(tmp_path)
    required = tuple(f"fp16/{size}/golden" for size in ("below", "threshold", "above", "tail", "hole", "boundary"))
    plan = ValidationPlan.derive(
        freeze,
        tune_shards=["fp16/large/golden"],
        unseen_count=8,
        master_seed=MASTER,
        required_instance_keys=required,
    )
    assert set(required) <= set(plan.unseen_instance_keys)
    assert len(plan.unseen_instance_keys) == 8
    assert len(set(plan.unseen_instance_keys) - set(required)) == 2
    same = ValidationPlan.derive(
        freeze,
        tune_shards=["fp16/large/golden"],
        unseen_count=8,
        master_seed=MASTER,
        required_instance_keys=required,
    )
    assert same.document() == plan.document()
    other = ValidationPlan.derive(
        freeze,
        tune_shards=["fp16/large/golden"],
        unseen_count=8,
        master_seed=OTHER_MASTER,
        required_instance_keys=required,
    )
    assert other.digest != plan.digest
    assert set(required) <= set(other.unseen_instance_keys)


def test_p2_a_candidate_byte_drift_requires_a_new_freeze_and_plan(tmp_path):
    freeze = _freeze(tmp_path)
    plan = _plan(freeze)
    entries = []
    for entry in freeze.candidate_entries:
        data = bytes(entry.data)
        if entry.path == "device/variant_b.metal":
            data += b"\n// P2-A freeze drift\n"
        entries.append(type(entry)(path=entry.path, role=entry.role, media_type=entry.media_type, data=data))
    changed = PortfolioFreeze(
        protocol_epoch=1,
        authority=freeze.authority,
        candidate_entries=tuple(entries),
        task_pack_digest=freeze.task_pack_digest,
        guidance_digest=freeze.guidance_digest,
    )
    assert changed.digest != freeze.digest
    with pytest.raises(ValueError, match="different portfolio freeze"):
        plan.validate_for(changed)
    new_plan = ValidationPlan.derive(
        changed, tune_shards=TUNE, unseen_count=len(REQUIRED), master_seed=MASTER, required_instance_keys=REQUIRED
    )
    assert new_plan.digest != plan.digest
    assert new_plan.portfolio_freeze_digest == changed.digest


def test_p2_a_oversized_freeze_fails_at_construction_with_contract_error(tmp_path):
    freeze = _freeze(tmp_path)
    big = CandidateEntry(
        path="device/big.metal",
        role="device_kernel",
        media_type="text/plain",
        data=b"x" * 1_100_000,
    )
    with pytest.raises(TaskContractError, match="not canonicalizable"):
        PortfolioFreeze(
            protocol_epoch=1,
            authority=freeze.authority,
            candidate_entries=(*freeze.candidate_entries, big),
            task_pack_digest=freeze.task_pack_digest,
            guidance_digest=freeze.guidance_digest,
        )
