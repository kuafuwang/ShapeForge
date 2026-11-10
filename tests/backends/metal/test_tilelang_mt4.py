"""MT4 TileLang generation/evaluation/resume seams (CPU + authorized real Metal)."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from scripts.tilelang_pack import pack
from taic.kernel_agent import AgentResult, SessionBrief, render_contract
from taic.llm import LLMConfig, SearchConfig
from taic.orchestrator import Orchestrator
from taic.prompts import system_prompt_v2
from taic.resume import _kernel_name
from taic.task_binding import TaskBinding
from taic.v2_evaluation import V2LocalEvaluator
from tcenv.backends.metal.search_candidate import (
    confirmation_input, domain_for, evaluate_candidate, portfolio_for_submission, preflight_submission,
)
from tcenv.contract_errors import UnsupportedContractCapability
from tcenv.portfolio.candidate import project_provider
from tcenv.target import load_target
from tcenv.track import decode_envelope, encode_envelope
from tests.support.tilelang_mt2 import (
    SINGLE_TILELANG, WORKLIST_TILELANG, geometry_document, tilelang_candidate, write_mt2_task,
)


def _binding(locator) -> TaskBinding:
    return TaskBinding(
        mode="v2", name="mt2_scale", dtype="fp32", size="small", target_id="apple-m4-max",
        locator=locator, contract=locator.load(), track_id="tilelang-pipeline", track_version=2,
    )


def test_mt4_prompt_and_signature_describe_tilelang_not_an_msl_submission(tmp_path):
    binding = _binding(write_mt2_task(tmp_path))
    signature = binding.signature_stub()
    assert "T.Tensor" in signature and "acc" in signature and "tmp" in signature
    assert "KERNELS" in signature and "kernel void" not in signature
    prompt = system_prompt_v2(binding, use_tools=False)
    assert "TileLang 0.1.15" in prompt and "KERNELS" in prompt
    assert "scripts/tilelang_pack.py" in prompt and "msl-single/msl-bundle" in prompt
    brief = SessionBrief(
        repo=str(Path(__file__).resolve().parents[3]), task_name="mt2_scale", dtype="fp32", size="small",
        kernel_name="mt2_scale_kernel", triton_source="", description="fixed MT4 prompt test",
        contract_text="{}", abi_stub=signature, rtol=1e-5, atol_rel=1e-6,
        v2=True, reference_source="", file_ext=".taic", track_id="tilelang-pipeline", source_fence="python",
    )
    task = render_contract(brief)
    assert "scripts/tilelang_pack.py" in task and "--track tilelang-pipeline" in task
    assert "HostTilingPlan" in task  # explicitly forbidden, not a hidden fallback


def test_mt4_packer_preserves_bytes_and_never_executes_candidate(tmp_path):
    src = tmp_path / "src"
    (src / "host").mkdir(parents=True)
    # This would raise if the packer imported the candidate.  It is legal text for packaging only.
    (src / "kernel.py").write_text("raise RuntimeError('packer must not execute this')\n", encoding="utf-8")
    (src / "host/execution_geometry.json").write_bytes(geometry_document(worklist=False))
    out = tmp_path / "submission.taic"
    digest = pack(src, out)
    candidate = decode_envelope(out.read_text(encoding="utf-8"))
    assert candidate.candidate_sha256 == digest
    by_path = {entry.path: entry for entry in candidate.entries}
    assert by_path["kernel.py"].data == b"raise RuntimeError('packer must not execute this')\n"
    with pytest.raises(ValueError, match="must be regular files"):
        pack(tmp_path / "missing", tmp_path / "other.taic")


def test_mt4_projection_and_preflight_keep_dsl_identity_and_msl_guard(tmp_path, monkeypatch):
    locator = write_mt2_task(tmp_path)
    contract = locator.load()
    target = load_target("apple-m4-max")
    plan, authority = domain_for(contract, target)
    source = encode_envelope(tilelang_candidate(worklist=True))
    candidate, portfolio = portfolio_for_submission(
        source, plan, authority=authority, track_id="tilelang-pipeline", track_version=2
    )
    projection = project_provider(
        portfolio, portfolio.variants[0], authority=authority, track_id="tilelang-pipeline", track_version=2
    )
    assert candidate.candidate_sha256 == projection.candidate.candidate_sha256
    assert [(entry.path, entry.role) for entry in projection.candidate.entries] == [
        ("host/execution_geometry.json", "schedule"), ("kernel.py", "dsl_source")
    ]
    assert preflight_submission(contract, source, target=target, track_id="tilelang-pipeline", track_version=2).ok
    # The default MSL projection remains a hard boundary, not an implicit DSL route.
    with pytest.raises(UnsupportedContractCapability, match="dsl_source role"):
        project_provider(portfolio, portfolio.variants[0], authority=authority)
    malformed = encode_envelope(tilelang_candidate(worklist=False, geometry=b'{"schema_version":1,"source":"candidate","stages":[]}'))
    assert not preflight_submission(contract, malformed, target=target, track_id="tilelang-pipeline", track_version=2).ok

    # Extracting MetalKernelPipelineEvaluator must preserve the A4 track_adapter._evaluate_in_process
    # seam so MetalTrackEvaluator(worker_child=True) still honors monkeypatched in-process hooks and
    # rejects a drifted full build.
    from tcenv.backends.metal import track_adapter
    from tcenv.backends.metal.track_adapter import MetalTrackEvaluator, msl_single_geometry_resolver
    from tcenv.build_projection import project_full
    from tcenv.contract_errors import TaskContractError
    from tcenv.evaluate import EvalReport
    from tcenv.remote.artifact_builder import BuildArtifact, BuildEvidence, BuildOutcome, BuildProfile, BuildProvenance
    from tcenv.remote.artifact_bundle import ArtifactBundle, BundleEntry, NativeExecutable
    from tcenv.taskcontract.runner import evaluate_task
    from tests.backends.metal.test_track_a4 import _rmsnorm_single_source, _rmsnorm_task
    from tests.support.report_build import StructuralNativeHandle

    assert callable(track_adapter._evaluate_in_process)
    rmsnorm_locator, rmsnorm_contract, tracks = _rmsnorm_task(tmp_path / "a4_seam")
    rmsnorm_identity = {
        "opspec_digest": rmsnorm_contract.spec.digest,
        "lowering_profile_digest": rmsnorm_contract.fingerprint(),
    }

    class _StructuralBuilder:
        build_kind = "native"

        def prepare(self, request):
            pass

        def build(self, request):
            entry = request.candidate.entries[0]
            src_entry = BundleEntry(
                path=entry.path, role="SOURCE", media_type=entry.media_type,
                size_bytes=len(entry.data), sha256=entry.sha256, data=bytes(entry.data),
            )
            assembled = "a" * 64
            native = NativeExecutable(
                backend_id="metal", kind="compute-pipeline", entry_path=entry.path,
                entry_sha256=entry.sha256, descriptor={"assembled_sha256": assembled},
                handle=StructuralNativeHandle(),
            )
            bundle = ArtifactBundle(entries=(src_entry,), native=native)
            profile = BuildProfile(
                build_profile="metal-msl-control-host-v1", builder_id="metal-source-builder",
                builder_version="1", build_host="control_host", toolchain_id="apple-metal",
                toolchain_version="unknown", sdk_version="unknown",
                compile_profile={"language": "msl", "math": "safe", "device": "fixture", "gpu_family": "fixture"},
            )
            provenance = BuildProvenance(
                profile=profile, source_sha256=request.candidate.candidate_sha256,
                artifact_set_digest=bundle.artifact_set_digest,
            )
            evidence = BuildEvidence(
                mode="prebuilt", status="ok", provenance=provenance,
                artifacts=(BuildArtifact.from_entry(src_entry),), placement="control_host", native=native,
            )
            return BuildOutcome(
                bundle=bundle, provenance=provenance, evidence=evidence,
                payloads={}, assembled_sha256=assembled,
            )

    def _drifted(*args, **kwargs):
        rep = EvalReport(
            task=rmsnorm_contract.op_name, dtype=rmsnorm_contract.instance.dtype,
            size=rmsnorm_contract.instance.size, mode="quick", status="ok", stage="ok", seed=0,
        )
        rep.build = project_full(kwargs["build_outcome"], build_kind="native", identity=rmsnorm_identity)
        rep.build["assembled_sha256"] = "0" * 64
        rep.build["provenance"]["artifact_set_digest"] = "f" * 64
        rep.build["artifacts"]["artifact_set_digest"] = "f" * 64
        return rep

    monkeypatch.setattr(track_adapter, "_evaluate_in_process", _drifted)
    with pytest.raises(TaskContractError, match="drifted|does not match"):
        evaluate_task(
            rmsnorm_locator,
            track_id="msl-single",
            candidate_source=_rmsnorm_single_source(),
            builder=_StructuralBuilder(),
            evaluator=MetalTrackEvaluator(mode="quick", cases=("normal",), worker_child=True),
            geometry_resolver=msl_single_geometry_resolver(rmsnorm_contract),
            track_registry_path=tracks,
        )


def test_mt4_dsl_confirmation_is_explicitly_deferred_not_relabeled_as_msl(tmp_path):
    locator = write_mt2_task(tmp_path)
    source = encode_envelope(tilelang_candidate(worklist=False))
    with pytest.raises(UnsupportedContractCapability, match="MT6|confirmation"):
        confirmation_input(
            locator, source, target=load_target("apple-m4-max"), guidance_digest="0" * 64,
            master_seed=b"0123456789abcdef", track_id="tilelang-pipeline", track_version=2,
        )
    evaluator = V2LocalEvaluator(_binding(locator))
    with pytest.raises(UnsupportedContractCapability, match="MT6"):
        evaluator.confirmation_expected(
            task="mt2_scale", dtype="fp32", size="small", target_id="apple-m4-max", source=source
        )


def test_mt4_resume_uses_taic_not_a_fake_metal_extension(tmp_path):
    (tmp_path / "run_config.json").write_text(json.dumps({
        "target": {"id": "apple-m4-max"},
        "track": {"track_id": "tilelang-pipeline", "track_version": 2, "delivery": "dsl_source_pipeline"},
    }), encoding="utf-8")
    assert _kernel_name(tmp_path) == "kernel.taic"


@pytest.mark.gpu
def test_mt4_evaluate_candidate_uses_the_production_portfolio_runner(tmp_path):
    result = evaluate_candidate(
        write_mt2_task(tmp_path), encode_envelope(tilelang_candidate(worklist=True)),
        target=load_target("apple-m4-max"), mode="quick", seed=7, reps=1,
        track_id="tilelang-pipeline", track_version=2,
    )
    assert result.record.outcome == "resolved"
    assert result.result is not None and result.result.report["status"] == "ok"
    assert result.result.candidate_sha256 == tilelang_candidate(worklist=True).candidate_sha256
    assert result.result.artifact_set_digest == "95e3796e38867e96be7b12efe57aa08e31d66795d9b731e13715a28e74702b03"
    assert [row["kernel"] for row in result.result.execution["dispatch"]] == ["scale_kernel_kernel", "bias_kernel_kernel"]


@pytest.mark.gpu
def test_mt4_fake_agent_drives_real_orchestrator_and_resume_without_api(tmp_path):
    locator = write_mt2_task(tmp_path)
    binding = _binding(locator)
    source = encode_envelope(tilelang_candidate(worklist=False))
    calls = []

    def agent(brief, tag):
        calls.append((brief.file_ext, brief.track_id, tag))
        return AgentResult(source=source, cost_usd=0.0, cost_incomplete=False, status="ok")

    llm = LLMConfig(
        base_url="http://x", api_key="sk-test-0123456789abcdef", model="fake",
        reasoning_effort="off", use_tools=False, truncation_retry=False,
    )
    search = SearchConfig(
        budget_usd=1.0, branches=1, max_rounds=0, eval_reps=1, kernel_agent="pi", correctness_only=True,
        stop_at_roofline=False, abi_guard=False, target_note=False, rich_history=False,
        status_lines=False, plan_ledger=False,
    )
    run_dir = tmp_path / "run"
    orch = Orchestrator(
        "mt2_scale", "fp32", "small", llm, search, run_dir=run_dir,
        binding=binding, track_id="tilelang-pipeline", track_version=2,
        agent_runner=agent, log=lambda _message: None,
    )
    result = asyncio.run(orch.run())
    assert orch.best is not None and orch.best.report.status == "ok"
    assert result.best_label == orch.best.label
    assert calls == [(".taic", "tilelang-pipeline", "init")]
    assert (run_dir / "best.taic").is_file() and not (run_dir / "best.metal").exists()
    assert decode_envelope(orch.best.source).candidate_sha256 == tilelang_candidate(worklist=False).candidate_sha256
    assert json.loads((run_dir / "run_config.json").read_text())["track"]["track_version"] == 2

    # Resume must restore the recorded /2 and must not need another agent/model call.
    resumed = Orchestrator(
        "mt2_scale", "fp32", "small", llm, search, run_dir=run_dir,
        binding=binding, track_id="tilelang-pipeline", track_version=2, resuming=True,
        agent_runner=lambda *_args: (_ for _ in ()).throw(AssertionError("resume started a new agent")),
        log=lambda _message: None,
    )
    from taic import resume as resume_module

    snapshot = resume_module.load_snapshot(run_dir, resumed.search, "mt2_scale_kernel")
    resumed.restore(snapshot)
    resumed_result = asyncio.run(resumed.run())
    assert resumed_result.best_label == result.best_label
    assert len(resumed.candidates) == 1
    assert resumed.candidates[0].report.status == "ok"
    assert resumed.candidates[0].source == orch.best.source
