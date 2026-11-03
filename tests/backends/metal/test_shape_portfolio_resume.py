"""P2-D CPU tests: portfolio sidecar, legacy-resume interlock and revocation floors."""

from __future__ import annotations

import argparse
import json

import pytest

from taic import resume as legacy_resume
from taic.cli import ResumeRefused, prepare_resume
from taic.portfolio_manifest import DispatchManifest
from taic.shape_portfolio import resume_checkpoint, revoke_checkpoint, write_checkpoint
from tcenv.backends.metal.portfolio_confirmation import activation_expectation
from tcenv.contract_errors import TaskContractError
from tcenv.identity import fingerprint, load_strict_json
from tcenv.portfolio.candidate import parse_portfolio
from tcenv.portfolio.domain import DomainAuthority, normalize_semantic_domain_plan
from taic.portfolio_store import PortfolioRun
from tcenv.portfolio_validation.validation import PortfolioFreeze, ValidationPlan
from tcenv.target import load_target
from tcenv.taskcontract.locator import discover_standard_tasks
from tcenv.track import REPO_ROOT
from observe.shape_portfolio import read_shape_portfolio
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
        documents.extend(load_strict_json(ROOT / f"docs/架构/shape-portfolio-阶段验收/P2/evidence/{name}")["documents"])
    expectations = {
        document["instance_key"]: activation_expectation(
            locator=locator,
            freeze=freeze,
            validation=validation,
            document=document,
        )
        for document in documents
    }
    manifest = DispatchManifest.from_confirmations(
        freeze, validation, documents, expectations, locator=locator
    ).activate(freeze, validation)
    return freeze, validation, manifest


def test_p2_d_checkpoint_resume_observe_and_legacy_interlock(tmp_path):
    freeze, validation, manifest = _fixture(tmp_path)
    run_dir = tmp_path / "run"
    write_checkpoint(run_dir, freeze=freeze, validation=validation, manifest=manifest, run_config_digest=fingerprint({"schema_version": 3}))
    completed = resume_checkpoint(run_dir, freeze=freeze, validation=validation, manifest=manifest, run_config_digest=fingerprint({"schema_version": 3}))
    assert len(completed) == 14
    observed = read_shape_portfolio(run_dir)
    assert observed["status"] == "active" and len(observed["subjects"]) == 14
    assert observed["coverage_kind"] == "full_finite"
    info = legacy_resume.inspect_run(run_dir)
    assert info["resumable"] is False and info["kind"] == "portfolio_run" and info["reason"] == "use portfolio resume"
    with pytest.raises(legacy_resume.ResumeError, match="portfolio run"):
        legacy_resume.load_snapshot(run_dir, None, "")
    with pytest.raises(ResumeRefused) as caught:
        prepare_resume(argparse.Namespace(run_dir=run_dir))
    assert caught.value.code == 3


def test_p2_d_revocation_records_reason_and_blocks_resume(tmp_path):
    freeze, validation, manifest = _fixture(tmp_path)
    run_dir = tmp_path / "run"
    write_checkpoint(run_dir, freeze=freeze, validation=validation, manifest=manifest, run_config_digest=fingerprint({"schema_version": 3}))
    revoked = revoke_checkpoint(run_dir, reason="candidate source superseded")
    assert revoked.status == "revoked"
    assert revoked.invalidated_artifacts[-1]["reason"] == "candidate source superseded"
    with pytest.raises(TaskContractError, match="revoked portfolio_run cannot resume"):
        resume_checkpoint(run_dir, freeze=freeze, validation=validation, manifest=manifest, run_config_digest=fingerprint({"schema_version": 3}))
    observed = read_shape_portfolio(run_dir)
    assert observed["status"] == "revoked"
    assert observed["invalidated_artifacts"][-1]["reason"] == "candidate source superseded"


def test_p2_d_generation_floor_and_manifest_drift_are_refused(tmp_path):
    freeze, validation, manifest = _fixture(tmp_path)
    run_dir = tmp_path / "run"
    write_checkpoint(run_dir, freeze=freeze, validation=validation, manifest=manifest, run_config_digest=fingerprint({"schema_version": 3}))
    with pytest.raises(TaskContractError, match="generation floor"):
        PortfolioRun.read(run_dir, freeze=freeze, validation=validation, manifest=manifest, minimum_generation=2, run_config_digest=fingerprint({"schema_version": 3}))
    other = PortfolioFreeze(
        protocol_epoch=1,
        authority=freeze.authority,
        candidate_entries=freeze.candidate_entries,
        task_pack_digest="0" * 64,
        guidance_digest=freeze.guidance_digest,
    )
    with pytest.raises(TaskContractError, match="another freeze"):
        PortfolioRun.read(run_dir, freeze=other, validation=validation, manifest=manifest, run_config_digest=fingerprint({"schema_version": 3}))


def test_p2_d_resume_rejects_ghost_subjects_and_run_config_drift(tmp_path):
    freeze, validation, manifest = _fixture(tmp_path)
    run_dir = tmp_path / "run"
    write_checkpoint(run_dir, freeze=freeze, validation=validation, manifest=manifest, run_config_digest=fingerprint({"schema_version": 3}))
    document = load_strict_json(run_dir / "portfolio_run.json")
    original = dict(document)
    ghost = dict(document["subjects"][0])
    ghost["instance_key"] = "fp16/ghost/golden"
    ghost["usage"] = "golden"
    document["subjects"] = [ghost, *document["subjects"][1:]]
    (run_dir / "portfolio_run.json").write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(TaskContractError, match="subjects differ"):
        resume_checkpoint(run_dir, freeze=freeze, validation=validation, manifest=manifest, run_config_digest=fingerprint({"schema_version": 3}))
    document["subjects"] = original["subjects"][:13]
    (run_dir / "portfolio_run.json").write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(TaskContractError, match="subjects differ|coverage mismatch"):
        resume_checkpoint(run_dir, freeze=freeze, validation=validation, manifest=manifest, run_config_digest=fingerprint({"schema_version": 3}))
    (run_dir / "portfolio_run.json").write_text(json.dumps(original), encoding="utf-8")
    with pytest.raises(TaskContractError, match="run_config_digest does not match"):
        resume_checkpoint(run_dir, freeze=freeze, validation=validation, manifest=manifest, run_config_digest="0" * 64)


def test_p2_d_document_only_or_partial_manifest_cannot_checkpoint(tmp_path):
    freeze, validation, manifest = _fixture(tmp_path)
    document = manifest.document()
    document["subjects"] = document["subjects"][:3]
    document["dispatch_manifest_digest"] = fingerprint(
        {key: value for key, value in document.items() if key != "dispatch_manifest_digest"}
    )
    partial = DispatchManifest.from_document(document)
    with pytest.raises(TaskContractError, match="document-only manifest|coverage mismatch"):
        write_checkpoint(
            tmp_path / "run",
            freeze=freeze,
            validation=validation,
            manifest=partial,
            run_config_digest=fingerprint({"schema_version": 3}),
        )


def test_p2_d_revoked_generation_cannot_be_checkpointed_again(tmp_path):
    freeze, validation, manifest = _fixture(tmp_path)
    run_dir = tmp_path / "run"
    write_checkpoint(run_dir, freeze=freeze, validation=validation, manifest=manifest, run_config_digest=fingerprint({"schema_version": 3}))
    revoke_checkpoint(run_dir, reason="retire")
    with pytest.raises(TaskContractError, match="over a revoked portfolio generation"):
        write_checkpoint(run_dir, freeze=freeze, validation=validation, manifest=manifest, run_config_digest=fingerprint({"schema_version": 3}))


def test_p2_d_resume_binds_the_manifest_to_the_callers_freeze_and_plan(tmp_path):
    freeze, validation, manifest = _fixture(tmp_path)
    run_dir = tmp_path / "run"
    run_config_digest = fingerprint({"schema_version": 3})
    write_checkpoint(
        run_dir, freeze=freeze, validation=validation, manifest=manifest, run_config_digest=run_config_digest
    )
    other_freeze = PortfolioFreeze(
        protocol_epoch=freeze.protocol_epoch,
        authority=freeze.authority,
        candidate_entries=freeze.candidate_entries,
        task_pack_digest="0" * 64,
        guidance_digest=freeze.guidance_digest,
    )
    other_validation = ValidationPlan.derive(
        other_freeze,
        tune_shards=validation.tune_shards,
        unseen_count=len(validation.unseen_instance_keys),
        master_seed=MASTER,
        required_instance_keys=validation.required_instance_keys,
    )
    path = run_dir / "portfolio_run.json"
    document = load_strict_json(path)
    document["portfolio_freeze_digest"] = other_freeze.digest
    document["validation_plan_digest"] = other_validation.digest
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(TaskContractError, match="manifest belongs to another freeze or validation plan"):
        resume_checkpoint(
            run_dir,
            freeze=other_freeze,
            validation=other_validation,
            manifest=manifest,
            run_config_digest=run_config_digest,
        )


def test_p2_d_resume_rejects_a_revoked_manifest_before_pointer_fallback(tmp_path):
    freeze, validation, manifest = _fixture(tmp_path)
    run_dir = tmp_path / "run"
    run_config_digest = fingerprint({"schema_version": 3})
    write_checkpoint(
        run_dir, freeze=freeze, validation=validation, manifest=manifest, run_config_digest=run_config_digest
    )
    revoked = manifest.revoke(freeze, reason="cross-stage status attack")
    path = run_dir / "portfolio_run.json"
    document = load_strict_json(path)
    document["dispatch_manifest_digest"] = revoked.digest
    document["activation_generation"] = revoked.activation_generation
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(TaskContractError, match="non-active DispatchManifest"):
        PortfolioRun.read(
            run_dir,
            freeze=freeze,
            validation=validation,
            manifest=revoked,
            run_config_digest=run_config_digest,
        )
    with pytest.raises(TaskContractError, match="non-active DispatchManifest"):
        resume_checkpoint(
            run_dir,
            freeze=freeze,
            validation=validation,
            manifest=revoked,
            run_config_digest=run_config_digest,
        )


def test_p2_d_active_checkpoint_cannot_be_taken_over_by_another_run_config(tmp_path):
    freeze, validation, manifest = _fixture(tmp_path)
    run_dir = tmp_path / "run"
    original_digest = fingerprint({"schema_version": 3, "run": "A"})
    takeover_digest = fingerprint({"schema_version": 3, "run": "B"})
    write_checkpoint(
        run_dir, freeze=freeze, validation=validation, manifest=manifest, run_config_digest=original_digest
    )
    with pytest.raises(TaskContractError, match="cannot change an existing active checkpoint"):
        write_checkpoint(
            run_dir, freeze=freeze, validation=validation, manifest=manifest, run_config_digest=takeover_digest
        )
    write_checkpoint(
        run_dir, freeze=freeze, validation=validation, manifest=manifest, run_config_digest=original_digest
    )
    assert len(
        resume_checkpoint(
            run_dir, freeze=freeze, validation=validation, manifest=manifest, run_config_digest=original_digest
        )
    ) == 14
    with pytest.raises(TaskContractError, match="run_config_digest does not match"):
        resume_checkpoint(
            run_dir, freeze=freeze, validation=validation, manifest=manifest, run_config_digest=takeover_digest
        )


def test_p2_d_revocation_marker_blocks_restoring_the_old_active_sidecar(tmp_path):
    freeze, validation, manifest = _fixture(tmp_path)
    run_dir = tmp_path / "run"
    run_config_digest = fingerprint({"schema_version": 3})
    write_checkpoint(
        run_dir, freeze=freeze, validation=validation, manifest=manifest, run_config_digest=run_config_digest
    )
    path = run_dir / "portfolio_run.json"
    active_sidecar = path.read_text(encoding="utf-8")
    revoke_checkpoint(run_dir, reason="persistent revocation marker test")
    path.write_text(active_sidecar, encoding="utf-8")
    with pytest.raises(TaskContractError, match="revoked"):
        resume_checkpoint(
            run_dir, freeze=freeze, validation=validation, manifest=manifest, run_config_digest=run_config_digest
        )
    observed = read_shape_portfolio(run_dir)
    assert observed["status"] == "revoked"
    assert observed["invalidated_artifacts"][-1]["reason"] == "persistent revocation marker test"
    with pytest.raises(TaskContractError, match="revoked"):
        write_checkpoint(
            run_dir, freeze=freeze, validation=validation, manifest=manifest, run_config_digest=run_config_digest
        )
