"""MT1a/MT5 Metal adapter fixtures; all checks are CPU-only and never create a device context."""

from __future__ import annotations

import copy
import json
from dataclasses import replace
from pathlib import Path

import pytest

from taic.artifacts import NeedsInspection
from taic.llm import LLMConfig, SearchConfig
from taic.orchestrator import Orchestrator, RunStartError, confirmation_master_seed
from taic.task_binding import bind_task
from taic.v2_evaluation import V2LocalEvaluator
from tcenv.backends.metal.search_candidate import evaluate_candidate, guidance_fingerprint
from tcenv.backends.metal.track_adapter import MetalTrackBuilder, MetalTrackEvaluator, configured_compile_deadline
from tcenv.backends.metal.track_factory import adapter_for_track
from tcenv.contract_errors import UnsupportedContractCapability
from tcenv.target import load_target
from tcenv.taskcontract.locator import REPO_ROOT, discover_standard_tasks
from tcenv.track.registry import TrackImplementation, load_track_registry
from tests.support.fakes import FakeClient
from tests.support.shape_portfolio_metal import DEVICE_B, load_p1_task
from tests.support.track_explore import source_bytes


def test_factory_constructs_only_the_registered_current_metal_identities():
    for track_id in ("msl-single", "msl-bundle"):
        spec = load_track_registry().get(track_id)
        adapter = adapter_for_track(spec)
        assert adapter.resolve("builder") is MetalTrackBuilder
        assert adapter.resolve("evaluator") is MetalTrackEvaluator
        resolver = adapter.resolve("geometry_resolver")
        if track_id == "msl-single":
            assert resolver is not None
        else:
            assert resolver is None


def test_factory_and_lazy_resolution_do_not_read_the_default_registry(monkeypatch):
    spec = load_track_registry().get("msl-single")
    adapter = adapter_for_track(spec)
    monkeypatch.setattr(
        "tcenv.track.registry.load_track_registry",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("default registry read")),
    )
    assert adapter.resolve("builder") is MetalTrackBuilder
    assert adapter.resolve("geometry_resolver") is not None


def test_an_unknown_implementation_version_fails_closed():
    spec = load_track_registry().get("msl-single")
    changed = replace(spec, build_adapter=TrackImplementation("metal.track.msl-single.build", 2))
    with pytest.raises(UnsupportedContractCapability, match="no Metal adapter implements"):
        adapter_for_track(changed)


def test_an_unregistered_track_is_rejected_before_any_msl_projection(tmp_path, monkeypatch):
    root = tmp_path / "tasks"
    load_p1_task(root)
    locator = discover_standard_tasks(root, code_root=REPO_ROOT)[0]
    document = copy.deepcopy(load_track_registry().document())
    future = copy.deepcopy(document["tracks"][0])
    future["track_id"] = "future-unimplemented"
    future["renderer"]["implementation_id"] = "future.renderer"
    future["validator"]["implementation_id"] = "future.validator"
    future["build_adapter"]["implementation_id"] = "future.build"
    document["tracks"] = [future]
    registry_path = tmp_path / "tracks.json"
    registry_path.write_text(json.dumps(document))
    monkeypatch.setattr(
        "tcenv.backends.metal.search_candidate.domain_for",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("MSL projection was reached")),
    )
    with pytest.raises(UnsupportedContractCapability, match="no Metal adapter implements"):
        evaluate_candidate(
            locator,
            DEVICE_B.decode(),
            target=load_target("apple-m4-max"),
            track_registry_path=registry_path,
            track_id="future-unimplemented",
        )


def test_mt5_compile_deadline_is_unset_by_default_and_validated_when_injected(monkeypatch):
    monkeypatch.delenv("TAIC_TRACK_BUILD_TIMEOUT_SEC", raising=False)
    assert configured_compile_deadline(worker_timeout=420) is None
    monkeypatch.setenv("TAIC_TRACK_BUILD_TIMEOUT_SEC", "45.5")
    assert configured_compile_deadline(worker_timeout=420) == 45.5
    for value in ("0", "-1", "nan", "inf", "not-a-number", "421"):
        monkeypatch.setenv("TAIC_TRACK_BUILD_TIMEOUT_SEC", value)
        with pytest.raises(Exception, match="TAIC_TRACK_BUILD_TIMEOUT_SEC"):
            configured_compile_deadline(worker_timeout=420)


def test_run_config_binds_the_guidance_used_by_existing_fresh_confirmation(tmp_path):
    root = tmp_path / "tasks"
    load_p1_task(root)
    binding = replace(
        bind_task("shape_portfolio_pointwise", "fp16", "threshold", tasks_root=root),
        track_id="msl-single",
    )
    llm = LLMConfig(base_url="http://x", api_key="sk-test-0123456789abcdef", model="fake", reasoning_effort="off", use_tools=False)
    search = SearchConfig(
        budget_usd=1.0, branches=1, max_rounds=0, eval_reps=3, correctness_only=True,
        stop_at_roofline=False, abi_guard=False, target_note=False, rich_history=False, status_lines=False,
    )
    orch = Orchestrator(
        "shape_portfolio_pointwise", "fp16", "threshold", llm, search,
        run_dir=tmp_path / "run", binding=binding, evaluator=V2LocalEvaluator(binding),
        client=FakeClient([]), log=lambda _s: None, track_id="msl-single",
    )
    config = json.loads((orch.run_dir / "run_config.json").read_text())
    assert config["confirmation_guidance_digest"] == guidance_fingerprint(
        orch.system, binding.presentation().statement
    )
    assert config["run_seed"] == orch.run_seed


def _resuming_orchestrator(tmp_path, digest: str):
    root = tmp_path / "tasks"
    load_p1_task(root)
    binding = replace(
        bind_task("shape_portfolio_pointwise", "fp16", "threshold", tasks_root=root),
        track_id="msl-single",
    )
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "run_config.json").write_text(json.dumps({
        "run_seed": 1,
        "confirmation_guidance_digest": digest,
    }))
    llm = LLMConfig(base_url="http://x", api_key="sk-test-0123456789abcdef", model="fake", reasoning_effort="off", use_tools=False)
    search = SearchConfig(
        budget_usd=1.0, branches=1, max_rounds=0, eval_reps=3, correctness_only=True,
        stop_at_roofline=False, abi_guard=False, target_note=False, rich_history=False, status_lines=False,
    )
    return Orchestrator(
        "shape_portfolio_pointwise", "fp16", "threshold", llm, search,
        run_dir=run_dir, resuming=True, binding=binding, evaluator=V2LocalEvaluator(binding),
        client=FakeClient([]), log=lambda _s: None, track_id="msl-single",
    )


def test_resume_uses_the_recorded_guidance_and_rejects_a_snapshot_mismatch(tmp_path):
    recorded = "c" * 64
    orch = _resuming_orchestrator(tmp_path, recorded)
    assert orch._confirmation_guidance_digest == recorded
    with pytest.raises(NeedsInspection, match="does not match the snapshot system prompt"):
        orch._restore_confirmation_guidance("a different system prompt")
    orch._restore_confirmation_guidance(None)
    assert orch._confirmation_guidance_digest == recorded


def test_resume_rejects_a_malformed_recorded_guidance_digest(tmp_path):
    with pytest.raises(RunStartError, match="not a lowercase sha256"):
        _resuming_orchestrator(tmp_path, "not-a-digest")


def test_resume_recomputes_only_a_matching_snapshot_digest(tmp_path):
    orch = _resuming_orchestrator(tmp_path, "0" * 64)
    recorded = guidance_fingerprint(orch.system, orch.binding.presentation().statement)
    orch._confirmation_guidance_digest = recorded
    orch._restore_confirmation_guidance(orch.system)
    assert orch._confirmation_guidance_digest == recorded


def test_scheme_a_deriver_replays_each_tracks_own_freeze_and_seed_without_device_work(tmp_path):
    root = tmp_path / "tasks"
    load_p1_task(root)
    locator = discover_standard_tasks(root, code_root=REPO_ROOT)[0]
    target = load_target("apple-m4-max")
    master = confirmation_master_seed(1234)
    outputs = {}
    for track_id, source in (("msl-single", DEVICE_B.decode()), ("msl-bundle", source_bytes("msl-bundle").decode())):
        spec = load_track_registry().get(track_id)
        derive = adapter_for_track(spec).resolve("confirmation")
        first = derive(
            locator, source, target=target, guidance_digest="a" * 64,
            master_seed=master, instance_key="fp16/threshold/benchmark",
        )
        second = derive(
            locator, source, target=target, guidance_digest="a" * 64,
            master_seed=master, instance_key="fp16/threshold/benchmark",
        )
        assert first == second
        assert first["confirmation_seed"] != 7
        outputs[track_id] = first
    assert outputs["msl-single"]["portfolio_freeze_digest"] != outputs["msl-bundle"]["portfolio_freeze_digest"]
    assert outputs["msl-single"]["confirmation_seed"] != outputs["msl-bundle"]["confirmation_seed"]
