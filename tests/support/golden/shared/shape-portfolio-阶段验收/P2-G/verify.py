#!/usr/bin/env python3
"""P2-G acceptance: per-launch SharedEvent protocol plus the current-pack P2 replay."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import statistics
from pathlib import Path

from tcenv.contract_errors import TaskContractError
from tcenv.identity import fingerprint, load_strict_json
from tcenv.opspec.v2 import instantiate_v2
from tcenv.portfolio.candidate import project_provider
from tcenv.portfolio.dispatch import DispatchRequest, ExactDispatchResolver
from tcenv.portfolio.domain import DomainAuthority, normalize_semantic_domain_plan
from tcenv.portfolio_evidence.projection import strict_json_equal
from tcenv.backends.metal.portfolio_confirmation import activation_expectation
from tcenv.backends.metal.portfolio_confirmation import confirmation_expected as rebuild_confirmation_expected
from tcenv.backends.metal.portfolio_confirmation import _resolved
from taic.portfolio_confirmation import confirmation_seed_for, validate_confirmation_for_activation, verify_subject_document
from tcenv.backends.metal.track_adapter import msl_single_geometry_resolver
from tcenv.bench import BENCH_ATTEMPTS
from taic.portfolio_manifest import ActivePortfolioPointer, DispatchManifest, verify_manifest_document
from taic.shape_portfolio import resume_checkpoint, revoke_checkpoint, write_checkpoint
from observe.shape_portfolio import read_shape_portfolio
from tcenv.portfolio_validation.validation import PortfolioFreeze, ValidationPlan
from tcenv.target import load_target
from tcenv.taskcontract.locator import discover_standard_tasks
from tcenv.taskcontract.portfolio_runner import _preflight_hook, locator_for_instance
from tcenv.track import REPO_ROOT
from tests.support.shape_portfolio_metal import (
    evidence_all_valid,
    load_p1_task,
    p1_instances,
    parent_entries,
    portfolio_document,
)

ROOT = Path(__file__).resolve().parents[6]
P2 = Path(__file__).resolve().parent
BASELINE = P2 / "acceptance-baseline.json"
GOLDEN = P2 / "golden.json"
MANIFEST = P2 / "manifest.sha256"
PLAN = "docs/架构/shape-portfolio实施计划.md"
PRODUCTION = "tcenv/portfolio_validation/validation.py"
CONFIRMATION = "tcenv/backends/metal/portfolio_confirmation.py"
CONFIRMATION_SEMANTICS = "taic/portfolio_confirmation.py"
STORE = "taic/portfolio_store.py"
SHAPE_PORTFOLIO = "taic/shape_portfolio.py"
OBSERVE = "observe/shape_portfolio.py"
MANIFEST_PRODUCTION = "taic/portfolio_manifest.py"
PACKAGE_INIT = "tcenv/portfolio_validation/__init__.py"
TEST = "tests/shared/test_shape_portfolio_validation.py"
MANIFEST_TEST = "tests/backends/metal/test_shape_portfolio_manifest.py"
RESUME_TEST = "tests/backends/metal/test_shape_portfolio_resume.py"
SEARCH_RESUME_TEST = "tests/shared/test_resume.py"
P4_TEST = "tests/shared/test_shape_portfolio_npu_sim.py"
P4_SIMULATION = "tests/support/npu_sim/shape_portfolio.py"
P4_EVIDENCE = "tests/support/golden/shared/shape-portfolio-阶段验收/P4/evidence/simulation.json"
GPU_TEST = "tests/backends/metal/test_shape_portfolio_confirmation_gpu.py"
CPU_CONFIRMATION_TEST = "tests/backends/metal/test_shape_portfolio_confirmation.py"
A_DOC = "tests/support/golden/shared/shape-portfolio-阶段验收/P2/A.md"
B_DOC = "tests/support/golden/shared/shape-portfolio-阶段验收/P2/B.md"
C_DOC = "tests/support/golden/shared/shape-portfolio-阶段验收/P2/C.md"
D_DOC = "tests/support/golden/shared/shape-portfolio-阶段验收/P2/D.md"
E_DOC = "tests/support/golden/shared/shape-portfolio-阶段验收/P2/E.md"
F_DOC = "tests/support/golden/shared/shape-portfolio-阶段验收/P2/F.md"
G_DOC = "tests/support/golden/shared/shape-portfolio-阶段验收/P2-G/G.md"
VERIFIER = "tests/support/golden/shared/shape-portfolio-阶段验收/P2-G/verify.py"
BASELINE_NAME = "tests/support/golden/shared/shape-portfolio-阶段验收/P2-G/acceptance-baseline.json"
PACK_CLOSURE_NAME = "tests/support/golden/shared/shape-portfolio-阶段验收/P2-G/pack-closure.json"
GOLDEN_NAME = "tests/support/golden/shared/shape-portfolio-阶段验收/P2-G/golden.json"
EVIDENCE_NAME = "tests/support/golden/shared/shape-portfolio-阶段验收/P2-G/evidence/unseen-confirmations.json"
TUNE_EVIDENCE_NAME = "tests/support/golden/shared/shape-portfolio-阶段验收/P2-G/evidence/tune-confirmations.json"
MANIFEST_EVIDENCE_NAME = "tests/support/golden/shared/shape-portfolio-阶段验收/P2-G/evidence/dispatch-manifest.json"
POINTER_EVIDENCE_NAME = "tests/support/golden/shared/shape-portfolio-阶段验收/P2-G/evidence/active_portfolio_pointer.json"
PER_LAUNCH_CONFIRMATION_NAME = "tests/support/golden/shared/shape-portfolio-阶段验收/P2-G/evidence/per-launch-softmax-confirmation.json"
PER_LAUNCH_PAIRS_NAME = "tests/support/golden/shared/shape-portfolio-阶段验收/P2-G/evidence/per-launch-softmax-pairs.json"
PER_LAUNCH_TEST = "tests/backends/metal/test_per_launch_submission.py"
DELIVERABLES = (
    PRODUCTION,
    CONFIRMATION,
    CONFIRMATION_SEMANTICS,
    STORE,
    SHAPE_PORTFOLIO,
    OBSERVE,
    MANIFEST_PRODUCTION,
    PACKAGE_INIT,
    TEST,
    MANIFEST_TEST,
    RESUME_TEST,
    SEARCH_RESUME_TEST,
    P4_TEST,
    P4_SIMULATION,
    P4_EVIDENCE,
    GPU_TEST,
    CPU_CONFIRMATION_TEST,
    A_DOC,
    B_DOC,
    C_DOC,
    D_DOC,
    E_DOC,
    F_DOC,
    G_DOC,
    VERIFIER,
    BASELINE_NAME,
    PACK_CLOSURE_NAME,
    GOLDEN_NAME,
    EVIDENCE_NAME,
    TUNE_EVIDENCE_NAME,
    MANIFEST_EVIDENCE_NAME,
    POINTER_EVIDENCE_NAME,
    PER_LAUNCH_CONFIRMATION_NAME,
    PER_LAUNCH_PAIRS_NAME,
    PER_LAUNCH_TEST,
    "tcenv/backends/metal/runtime.py",
    "tcenv/backends/metal/v2_runtime.py",
    "tcenv/backends/metal/kernel_pipeline.py",
    "tcenv/backends/metal/track_adapter.py",
    "tcenv/backends/metal/tilelang_runtime.py",
    "tcenv/backends/metal/search_candidate.py",
    "tcenv/v2_worker.py",
    "tcenv/evaluate.py",
    "tcenv/cli.py",
    "taic/llm.py",
    "taic/cli.py",
    "taic/orchestrator.py",
    "taic/v2_evaluation.py",
    "experiments/softmax_command_buffer_probe.py",
    PLAN,
)
MASTER = b"0123456789abcdef"
TUNE = ("fp16/large/golden", "fp16/large/benchmark")
REQUIRED = tuple(
    f"fp16/{size}/{usage}"
    for size in ("below", "threshold", "above", "tail", "hole", "boundary")
    for usage in ("golden", "benchmark")
)
P0_P1_NODES = (
    "tests/shared/test_shape_portfolio_domain.py",
    "tests/shared/test_shape_portfolio_candidate.py",
    "tests/shared/test_shape_portfolio_dispatch.py",
    "tests/shared/test_shape_portfolio_runner.py",
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_manifest() -> None:
    MANIFEST.write_text(
        "\n".join(f"{sha256(ROOT / name)}  {name}" for name in sorted(DELIVERABLES)) + "\n",
        encoding="utf-8",
    )
    print(f"OK: wrote {MANIFEST.relative_to(ROOT)} ({len(DELIVERABLES)} files)")


def verify_manifest() -> None:
    entries = {}
    for line in MANIFEST.read_text(encoding="utf-8").splitlines():
        digest, name = line.split("  ", 1)
        entries[name] = digest
    require(set(entries) == set(DELIVERABLES), "P2-A manifest file set drifted")
    for name, digest in entries.items():
        require(sha256(ROOT / name) == digest, f"P2-A manifest mismatch: {name}")
    print(f"OK: P2-A manifest ({len(entries)} files)")


def verify_baseline() -> None:
    from tcenv.taskcontract.pack import code_import_closure

    document = json.loads(BASELINE.read_text(encoding="utf-8"))
    require(document["epoch"] == "p2-g-per-launch-1", "wrong P2-G baseline epoch")
    require(document["parent_head"] == "37e80f3aaa9c29ccfbb8cd58265f8598437ac578", "P2-G parent HEAD drifted")
    closure = load_strict_json(P2 / "pack-closure.json")
    files = code_import_closure(ROOT)
    digest = hashlib.sha256(b"".join((ROOT / name).read_bytes() for name in files)).hexdigest()
    require(list(closure["files"]) == list(files), "P2-G task-pack file closure drifted")
    require(closure["sha256"] == digest == document["pack_closure_sha256"], "P2-G task-pack digest drifted")
    require(len(files) == document["pack_closure_files"] == 74, "P2-G task-pack file count drifted")
    print(f"OK: P2-G stage baseline (epoch=p2-g-per-launch-1, {len(files)} pack files, digest={digest[:12]}…)")


def verify_package_boundary() -> None:
    package = ROOT / "tcenv/portfolio_validation"
    sources = sorted(package.glob("*.py"))
    require([path.name for path in sources] == ["__init__.py", "validation.py"], "P2 validation package is not closed")
    forbidden = ("PatternResolver", "LoweringRuleRegistry", "StagePartitioner", "TileConnector", "LoweringProfile")
    for path in sources:
        text = path.read_text(encoding="utf-8")
        for token in forbidden:
            require(token not in text, f"{path.name}: forbidden architecture token {token}")
        for line in text.splitlines():
            stripped = line.strip()
            require(
                not stripped.startswith(("import tcenv.backends", "from tcenv.backends", "import taic", "from taic")),
                f"{path.name}: backend/search import crosses the P2 validation boundary",
            )
    print("OK: P2 validation package boundary (2 closed files, no backend/search/forbidden architecture)")


def fixture(root: Path):
    task_root = root / "tasks"
    load_p1_task(task_root)
    locator = discover_standard_tasks(task_root, code_root=REPO_ROOT)[0]
    contract = locator.load()
    target = load_target("apple-m4-max")
    domain = normalize_semantic_domain_plan(contract.spec, p1_instances(contract), target=target)
    instances = p1_instances(contract)
    authority = DomainAuthority(
        plan=domain,
        spec=contract.spec,
        target=target,
        instances=instances,
        contract=contract,
    )
    entries = parent_entries(portfolio_document(domain.digest))
    freeze = PortfolioFreeze(
        protocol_epoch=1,
        authority=authority,
        candidate_entries=entries,
        task_pack_digest=locator.deployment_pack_digest,
        guidance_digest=fingerprint({"guidance": "P2-A fixed guidance"}),
    )
    plan = ValidationPlan.derive(
        freeze,
        tune_shards=TUNE,
        unseen_count=len(REQUIRED),
        master_seed=MASTER,
        required_instance_keys=REQUIRED,
    )
    return locator, freeze, plan


def golden_document(freeze: PortfolioFreeze, plan: ValidationPlan) -> dict:
    return {
        "schema_id": "tcenv.portfolio.p2-a-golden.1",
        "portfolio_freeze_digest": freeze.digest,
        "validation_plan_digest": plan.digest,
        "opspec_digest": freeze.opspec_digest,
        "tune_shards": list(plan.tune_shards),
        "unseen_instance_keys": list(plan.unseen_instance_keys),
        "required_instance_keys": list(plan.required_instance_keys),
        "fresh_seeds": dict(sorted(plan.fresh_seeds.items())),
        "candidate_view": plan.candidate_view(freeze).document(),
    }


def verify_golden(freeze: PortfolioFreeze, plan: ValidationPlan) -> None:
    actual = golden_document(freeze, plan)
    expected = load_strict_json(GOLDEN)
    require(strict_json_equal(actual, expected), "P2-A deterministic derivation golden drifted")
    restored = PortfolioFreeze.from_document(freeze.document(), authority=freeze.authority)
    require(restored.digest == freeze.digest, "freeze round trip changed its identity")
    restored_plan = ValidationPlan.from_document(plan.document())
    restored_plan.verify_fresh_seeds(restored, master_seed=MASTER, expected_plan_digest=plan.digest)
    candidate_payload = json.dumps(restored_plan.candidate_view(restored).document(), sort_keys=True)
    for key in restored_plan.unseen_instance_keys:
        require(key not in candidate_payload, f"candidate view leaked unseen key {key}")
    for seed in restored_plan.fresh_seeds.values():
        require(str(seed) not in candidate_payload, "candidate view leaked a fresh seed")
    print("OK: deterministic P2-A golden + candidate-view leak check")


SUBJECT_STABLE_KEYS = (
    "instance_key",
    "usage",
    "provider",
    "variant_id",
    "fallback_id",
    "candidate_sha256",
    "credits_variant",
    "search_seed",
    "confirmation_seed",
    "expected",
    "identity",
    "coverage_kind",
)
REPORT_STABLE_KEYS = (
    "status",
    "cache_hit",
    "dtype",
    "size",
    "mode",
    "target_id",
    "backend",
    "backend_version",
    "protocol_version",
    "device_name",
    "gpu_family",
    "evaluation_fingerprint",
    "baseline_implementation_digest",
    "baseline_environment_digest",
)
BUILD_STABLE_KEYS = (
    "candidate_sha256",
    "opspec_digest",
    "lowering_profile_digest",
    "shape",
    "build_kind",
    "artifact_set_digest",
    "native_digest",
)


def _stable_report(report):
    build = report.get("build") or {}
    return {
        **{key: report.get(key) for key in REPORT_STABLE_KEYS},
        "build": {key: build.get(key) for key in BUILD_STABLE_KEYS},
        "correctness_ok": (report.get("correctness") or {}).get("ok"),
        "safety_ok": (report.get("safety") or {}).get("ok"),
        "abi_ok": (report.get("abi") or {}).get("ok"),
    }


def _stable_subject(document):
    return {
        **{key: document.get(key) for key in SUBJECT_STABLE_KEYS},
        "search_report": _stable_report(document["search_report"]),
        "confirmation_report": _stable_report(document["confirmation_report"]),
    }


def verify_confirmation_suite(
    directory: Path,
    locator,
    freeze: PortfolioFreeze,
    validation: ValidationPlan,
    *,
    expected_instance_keys: set[str] | None = None,
    filename: str = "unseen-confirmations.json",
) -> dict[str, dict]:
    path = directory / filename
    document = load_strict_json(path)
    require(
        set(document) == {"schema_id", "host", "portfolio_freeze_digest", "validation_plan_digest", "documents"},
        "P2 confirmation suite is not the closed shape",
    )
    require(document["schema_id"] == "tcenv.portfolio.p2-confirmation-suite.1", "unknown confirmation suite schema")
    golden = load_strict_json(GOLDEN)
    require(document["portfolio_freeze_digest"] == golden["portfolio_freeze_digest"], "evidence freeze drifted")
    require(document["validation_plan_digest"] == golden["validation_plan_digest"], "evidence validation plan drifted")
    require(locator.deployment_pack_digest == freeze.task_pack_digest, "confirmation suite task pack is not the frozen task pack")
    require(locator.op_name == freeze.authority.spec.op_name, "confirmation suite locator names another task")
    validation.validate_for(freeze)
    portfolio = freeze.portfolio()
    resolver = ExactDispatchResolver(
        domain_plan=freeze.domain_plan,
        portfolio=portfolio,
        evidence=evidence_all_valid(portfolio),
        spec=freeze.authority.spec,
        target=freeze.authority.target,
        instances=freeze.authority.instances,
        source_preflight=True,
    )
    subjects = {}
    for subject in document["documents"]:
        verified = verify_subject_document(subject)
        if verified["status"] == "rejected":
            require(
                verified["usage"] == "benchmark"
                and verified["reason"] == "timing_uncertain"
                and verified["confirmation_attempts"] == BENCH_ATTEMPTS,
                f"{verified['instance_key']}: confirmation was rejected before the bounded attempt budget was exhausted",
            )
        else:
            require(verified["status"] == "confirmed", f"{verified['instance_key']}: invalid confirmation status")
        require(verified["instance_key"] not in subjects, "duplicate confirmation subject")
        dtype, size, usage = verified["instance_key"].split("/")
        point = freeze.domain_plan.find(dtype, size, usage)
        require(point is not None, f"{verified['instance_key']}: not in the frozen domain")
        expected_record, expected = activation_expectation(
            locator=locator,
            freeze=freeze,
            validation=validation,
            document=verified,
        )
        verified = validate_confirmation_for_activation(
            verified,
            freeze,
            validation,
            expected_record=expected_record,
            expected=expected,
        )
        require(verified["search_seed"] == 7, f"{verified['instance_key']}: wrong search seed")
        subjects[verified["instance_key"]] = verified
    expected = validation.unseen_instance_keys if expected_instance_keys is None else expected_instance_keys
    require(set(subjects) == set(expected), "not every required subject was confirmed")
    blocked = sum(subject["status"] == "rejected" for subject in subjects.values())
    print(f"OK: archived fresh confirmations ({len(subjects)} unseen subjects, {blocked} timing-blocked, all others confirmed)")
    return subjects


def verify_full_confirmation_suite(directory: Path, locator, freeze: PortfolioFreeze, validation: ValidationPlan) -> dict[str, dict]:
    subjects = verify_confirmation_suite(directory, locator, freeze, validation)
    tune = verify_confirmation_suite(
        directory,
        locator,
        freeze,
        validation,
        expected_instance_keys=set(validation.tune_shards),
        filename="tune-confirmations.json",
    )
    require(not set(subjects) & set(tune), "tune and unseen confirmation subjects overlap")
    subjects.update(tune)
    require(set(subjects) == set(freeze.instance_keys()), "not every domain instance has fresh evidence")
    print(f"OK: full fresh evidence ({len(subjects)} domain instances: unseen + tune)")
    return subjects


def verify_live_archived_confirmations(live: Path, archived: Path, locator, freeze: PortfolioFreeze, validation: ValidationPlan) -> None:
    live_subjects = verify_full_confirmation_suite(live, locator, freeze, validation)
    archived_subjects = verify_full_confirmation_suite(archived, locator, freeze, validation)
    for key, archived_subject in archived_subjects.items():
        live_subject = live_subjects[key]
        require(
            strict_json_equal(_stable_subject(live_subject), _stable_subject(archived_subject)),
            f"{key}: live and archived stable confirmation identities differ",
        )
    print("OK: live == archived stable confirmation identities (timing values excluded, semantics rechecked)")


def pytest_nodes(arguments: list[str], *, where: str, expected: int, env=None) -> None:
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-n", "0", "-p", "no:cacheprovider", *arguments],
        cwd=ROOT,
        text=True,
        capture_output=True,
        env=None if env is None else {**os.environ, **env},
    )
    output = result.stdout + result.stderr
    require(result.returncode == 0, f"{where} failed:\n{output}")
    require(f"{expected} passed" in output, f"{where}: expected exactly {expected} passed:\n{output}")
    require(" skipped" not in output and "xfail" not in output and "deselected" not in output, f"{where}: skip/xfail/deselect is not accepted:\n{output}")
    print(f"OK: {where} ({expected} passed)")


def _active_from_archive(locator, freeze: PortfolioFreeze, validation: ValidationPlan) -> DispatchManifest:
    subjects = verify_full_confirmation_suite(P2 / "evidence", locator, freeze, validation)
    expectations = {
        key: activation_expectation(locator=locator, freeze=freeze, validation=validation, document=document)
        for key, document in subjects.items()
    }
    return DispatchManifest.from_confirmations(
        freeze,
        validation,
        list(subjects.values()),
        expectations,
        locator=locator,
    ).activate(freeze, validation)


def verify_activation_evidence(locator, freeze: PortfolioFreeze, validation: ValidationPlan) -> None:
    active = _active_from_archive(locator, freeze, validation)
    archived = verify_manifest_document(load_strict_json(P2 / "evidence/dispatch-manifest.json"), freeze, validation)
    require(
        strict_json_equal(active.document(), archived.document()),
        "the activated DispatchManifest is not the archived stable confirmation projection",
    )
    ActivePortfolioPointer.read(P2 / "evidence/active_portfolio_pointer.json", active)
    print(f"OK: active DispatchManifest + pointer (generation={active.activation_generation}, subjects={len(active.subjects)})")


def verify_checkpoint_lifecycle(locator, freeze: PortfolioFreeze, validation: ValidationPlan) -> None:
    with tempfile.TemporaryDirectory(prefix="p2-d-checkpoint-") as temporary:
        run_dir = Path(temporary) / "run"
        active = _active_from_archive(locator, freeze, validation)
        write_checkpoint(
            run_dir,
            freeze=freeze,
            validation=validation,
            manifest=active,
            run_config_digest=fingerprint({"schema_version": 3, "task": active.subjects[0].instance_key}),
        )
        completed = resume_checkpoint(run_dir, freeze=freeze, validation=validation, manifest=active, run_config_digest=fingerprint({"schema_version": 3, "task": active.subjects[0].instance_key}))
        require(set(completed) == {subject.instance_key for subject in active.subjects}, "resume would repeat a confirmed fresh batch or return a forged subject set")
        observed = read_shape_portfolio(run_dir)
        require(observed["status"] == "active" and len(observed["subjects"]) == len(active.subjects), "observe sidecar is incomplete")
        revoked = revoke_checkpoint(run_dir, reason="P2-D retirement drill")
        require(revoked.status == "revoked" and revoked.invalidated_artifacts, "revocation lost its invalidation record")
        try:
            resume_checkpoint(run_dir, freeze=freeze, validation=validation, manifest=active, run_config_digest=fingerprint({"schema_version": 3, "task": active.subjects[0].instance_key}))
        except TaskContractError as exc:
            require("revoked" in str(exc), f"revoked checkpoint resumed: {exc}")
        else:
            raise AssertionError("a revoked checkpoint resumed execution")
        print(f"OK: portfolio checkpoint/resume/observe (completed={len(completed)}, revoked generation={revoked.activation_generation})")


def _per_launch_frozen():
    from taic.task_binding import bind_task
    from tcenv.backends.metal.search_candidate import confirmation_input

    binding = bind_task("softmax", "fp16", "paper", target_id="apple-m4-max")
    source = (ROOT / "kernels/apple-m4-max/softmax/fp16/paper/best.metal").read_text(encoding="utf-8")
    frozen = confirmation_input(
        binding.locator,
        source,
        target=load_target("apple-m4-max"),
        guidance_digest=fingerprint({"guidance": "P2-G per-launch softmax", "source_sha256": hashlib.sha256(source.encode()).hexdigest()}),
        master_seed=7,
        batch_submission="per_launch_set_event",
    )
    return binding, source, frozen


def verify_per_launch_confirmation(directory: Path, *, archived: Path | None = None) -> None:
    binding, source, frozen = _per_launch_frozen()
    payload = load_strict_json(directory / "per-launch-softmax-confirmation.json")
    require(payload["schema_id"] == "shapeforge.p2g-per-launch-confirmation.1", "unknown P2-G confirmation schema")
    require(payload["publication_allowed"] is False, "P2-G is protocol evidence, not a speedup publication")
    require(payload["portfolio_freeze_digest"] == frozen.freeze.digest, "P2-G freeze drifted")
    require(payload["validation_plan_digest"] == frozen.validation.digest, "P2-G validation plan drifted")
    documents = [verify_subject_document(document) for document in payload["documents"]]
    require(len(documents) == 2 and all(document["status"] == "confirmed" for document in documents), "P2-G golden+benchmark are not both confirmed")
    for document in documents:
        require(document["identity"]["kind"] == "match", "P2-G confirmation identity mismatch")
        require(document["confirmation_report"]["protocol_version"] == "3+per_launch_set_event", "P2-G report protocol drifted")
        require(document["expected"]["protocol_version"] == "3+per_launch_set_event", "P2-G expected protocol drifted")
    expectations = {
        document["instance_key"]: activation_expectation(
            locator=binding.locator, freeze=frozen.freeze, validation=frozen.validation, document=document,
        )
        for document in documents
    }
    active = DispatchManifest.from_confirmations(
        frozen.freeze, frozen.validation, documents, expectations, locator=binding.locator,
    ).activate(frozen.freeze, frozen.validation)
    archived_manifest = verify_manifest_document(payload["dispatch_manifest"], frozen.freeze, frozen.validation)
    require(strict_json_equal(active.document(), archived_manifest.document()), "P2-G active manifest is not the canonical projection")
    require(payload["active_pointer"] == ActivePortfolioPointer(active.digest, active.activation_generation).document(), "P2-G active pointer is stale")
    if archived is not None:
        previous = load_strict_json(archived / "per-launch-softmax-confirmation.json")
        for live_document, archived_document in zip(documents, previous["documents"]):
            require(
                strict_json_equal(_stable_subject(live_document), _stable_subject(archived_document)),
                "P2-G live and archived confirmation identities differ",
            )
    print("OK: P2-G softmax golden+benchmark fresh confirmed, active manifest generation=1, subjects=2")


def verify_per_launch_pairs(directory: Path, *, archived: Path | None = None) -> None:
    _binding, source, _frozen = _per_launch_frozen()
    payload = load_strict_json(directory / "per-launch-softmax-pairs.json")
    require(payload["schema_id"] == "shapeforge.p2g-per-launch-pairs.1", "unknown P2-G pairs schema")
    require(payload["publication_allowed"] is False, "P2-G pairs are not a speedup publication")
    require(payload["candidate_sha256"] == hashlib.sha256(source.encode()).hexdigest(), "P2-G pair candidate drifted")
    pairs = payload["pairs"]
    require(len(pairs) == 3, "P2-G needs three paired fresh seeds")
    for row in pairs:
        for arm, protocol in (("single_buffer", "3"), ("per_launch_set_event", "3+per_launch_set_event")):
            report = row[arm]["report"]
            require(report["status"] == "ok" and report["protocol_version"] == protocol, f"P2-G {arm} report is not a full ok evaluation")
            require(report["correctness"]["ok"] and report["safety"]["ok"], f"P2-G {arm} correctness/safety failed")
    old_median = statistics.median(row["single_buffer"]["speedup"] for row in pairs)
    new_median = statistics.median(row["per_launch_set_event"]["speedup"] for row in pairs)
    require(new_median > 1.0 and new_median > old_median + 0.03, f"P2-G paired gate failed: {old_median} -> {new_median}")
    if archived is not None:
        previous = load_strict_json(archived / "per-launch-softmax-pairs.json")
        for live_row, archived_row in zip(pairs, previous["pairs"]):
            require(live_row["seed"] == archived_row["seed"], "P2-G pair seed order drifted")
            for arm in ("single_buffer", "per_launch_set_event"):
                require(
                    strict_json_equal(_stable_report(live_row[arm]["report"]), _stable_report(archived_row[arm]["report"])),
                    f"P2-G live and archived {arm} stable identities differ",
                )
    print(f"OK: P2-G paired fresh median {old_median:.6f}x -> {new_median:.6f}x (live identity stable)")


def run_confirmation_gpu(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    pytest_nodes(
        [GPU_TEST],
        where="P2-B real Mac confirmation nodes",
        expected=5,
        env={"SHAPE_PORTFOLIO_P2_EVIDENCE_DIR": str(directory)},
    )
    pytest_nodes(
        [
            "tests/backends/metal/test_per_launch_submission.py::test_per_launch_preserves_read_write_generation_dependencies",
            "tests/backends/metal/test_per_launch_submission.py::test_per_launch_rejects_a_fill_instead_of_dropping_workspace_zeroing",
            "tests/backends/metal/test_per_launch_submission.py::test_p2_g_softmax_per_launch_fresh_confirmation",
            "tests/backends/metal/test_per_launch_submission.py::test_p2_g_softmax_paired_fresh_evaluations",
        ],
        where="P2-G real Mac SharedEvent nodes",
        expected=4,
        env={"SHAPE_PORTFOLIO_P2G_EVIDENCE_DIR": str(directory)},
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write-golden", action="store_true")
    parser.add_argument("--write-manifest", action="store_true")
    parser.add_argument("--record-evidence", action="store_true")
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="p2-a-fixture-") as temporary:
        locator, freeze, plan = fixture(Path(temporary))
        if args.write_golden:
            require(MASTER == b"0123456789abcdef", "P2-A golden may only be recorded with the public fixture master")
            GOLDEN.write_text(json.dumps(golden_document(freeze, plan), indent=2, sort_keys=True) + "\n", encoding="utf-8")
            print(f"OK: wrote {GOLDEN.relative_to(ROOT)}")
            return
        if args.record_evidence:
            run_confirmation_suite_dir = P2 / "evidence"
            run_confirmation_gpu(run_confirmation_suite_dir)
            active = _active_from_archive(locator, freeze, plan)
            (P2 / "evidence/dispatch-manifest.json").write_text(
                json.dumps(active.document(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            ActivePortfolioPointer(active.digest, active.activation_generation).write_atomic(
                P2 / "evidence/active_portfolio_pointer.json"
            )
            verify_activation_evidence(locator, freeze, plan)
            verify_per_launch_confirmation(run_confirmation_suite_dir)
            verify_per_launch_pairs(run_confirmation_suite_dir)
            return
        if args.write_manifest:
            write_manifest()
            return
        verify_manifest()
        verify_baseline()
        verify_package_boundary()
        verify_golden(freeze, plan)
        pytest_nodes(list(P0_P1_NODES), where="P0/P1 fixed CPU anchors", expected=58)
        pytest_nodes(["tests/shared/test_structure.py"], where="repository structure", expected=6)
        pytest_nodes(["tests/shared/test_shape_portfolio_evidence.py"], where="P2-0 anchor", expected=1)
        pytest_nodes([TEST], where="P2-A fixed CPU nodes", expected=9)
        pytest_nodes([CPU_CONFIRMATION_TEST], where="P2-B pack/archive negative nodes", expected=11)
        pytest_nodes([MANIFEST_TEST], where="P2-C DispatchManifest CPU nodes", expected=16)
        pytest_nodes([RESUME_TEST], where="P2-D checkpoint/resume CPU nodes", expected=10)
        pytest_nodes(
            ["tests/shared/test_resume.py::test_resume_cannot_switch_the_batch_submission_epoch"],
            where="P2-G resume epoch guard",
            expected=1,
        )
        pytest_nodes(
            [
                "tests/backends/metal/test_per_launch_submission.py::test_per_launch_eligibility_is_narrower_than_the_native_batch",
                "tests/backends/metal/test_per_launch_submission.py::test_evaluator_requires_an_explicit_known_submission",
                "tests/backends/metal/test_per_launch_submission.py::test_per_launch_confirmation_uses_a_new_epoch",
                "tests/backends/metal/test_per_launch_submission.py::test_taic_config_maps_the_explicit_submission_to_the_protocol_epoch",
                "tests/backends/metal/test_per_launch_submission.py::test_per_launch_emits_one_wait_and_signal_per_set_generation",
            ],
            where="P2-G SharedEvent CPU nodes",
            expected=5,
        )
        pytest_nodes([P4_TEST], where="P4 historical offline replay (unchanged bytes)", expected=8)
        verify_confirmation_suite(P2 / "evidence", locator, freeze, plan)
        with tempfile.TemporaryDirectory(prefix="p2-b-live-") as live:
            live_path = Path(live)
            run_confirmation_gpu(live_path)
            verify_live_archived_confirmations(live_path, P2 / "evidence", locator, freeze, plan)
            verify_per_launch_confirmation(live_path, archived=P2 / "evidence")
            verify_per_launch_pairs(live_path, archived=P2 / "evidence")
        verify_activation_evidence(locator, freeze, plan)
        verify_per_launch_confirmation(P2 / "evidence")
        verify_per_launch_pairs(P2 / "evidence")
        verify_checkpoint_lifecycle(locator, freeze, plan)
        print("OK: P2-G acceptance (60 CPU + 9 real Mac nodes, two active manifests + checkpoint lifecycle + P4 replay; no model/search)")


if __name__ == "__main__":
    main()
