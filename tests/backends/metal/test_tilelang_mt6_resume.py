"""MT6 /3 run_config and in-memory snapshot resume tests; no agent, GPU or model call."""

from __future__ import annotations

import json

import pytest

from taic.llm import LLMConfig, SearchConfig
from taic.orchestrator import Orchestrator, RunStartError
from taic.resume import Snapshot, _kernel_name
from taic.cli import _recorded_track_selection
from taic.task_binding import TaskBinding
from taic.v2_evaluation import V2LocalEvaluator
from tcenv.evaluate import EvalReport
from tcenv.track.envelope import decode_envelope, encode_envelope
from tests.backends.metal.test_tilelang_mt6_confirmation import _registry
from tests.support.tilelang_mt2 import tilelang_candidate, write_mt2_task


@pytest.fixture
def fresh_registry(monkeypatch):
    registry = _registry()
    # Import direct-reference consumers before patching the registry module.  Otherwise a first
    # import inside the fixture can permanently capture the test lambda after monkeypatch.undo().
    import taic.track_prompt as track_prompt

    monkeypatch.setattr(track_prompt, "load_track_registry", lambda *args, **kwargs: registry)
    monkeypatch.setattr("tcenv.track.registry.load_track_registry", lambda *args, **kwargs: registry)
    monkeypatch.setattr("tcenv.backends.metal.search_candidate.load_track_registry", lambda *args, **kwargs: registry)
    monkeypatch.setattr("tcenv.backends.metal.portfolio_confirmation.load_track_registry", lambda *args, **kwargs: registry)
    monkeypatch.setattr("tcenv.track.tilelang_confirmation.load_track_registry", lambda *args, **kwargs: registry)
    monkeypatch.setattr("tcenv.portfolio.candidate.load_track_registry", lambda *args, **kwargs: registry)
    return registry


def _config(run_dir, version=3):
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run_config.json").write_text(json.dumps({
        "task_mode": "v2",
        "target": {"id": "apple-m4-max"},
        "track": {
            "track_id": "tilelang-pipeline",
            "track_version": version,
            "delivery": "dsl_source_pipeline",
        },
    }), encoding="utf-8")


def _search():
    return SearchConfig(
        budget_usd=1.0,
        branches=1,
        max_rounds=0,
        eval_reps=1,
        kernel_agent="pi",
        correctness_only=True,
        stop_at_roofline=False,
        abi_guard=False,
        target_note=False,
        rich_history=False,
        status_lines=False,
        plan_ledger=False,
    )


def _llm():
    return LLMConfig(
        base_url="http://x",
        api_key="sk-test-0123456789abcdef",
        model="fake",
        reasoning_effort="off",
        use_tools=False,
        truncation_retry=False,
    )


def _binding(locator, version=3):
    return TaskBinding(
        mode="v2",
        name=locator.op_name,
        dtype="fp32",
        size="small",
        target_id="apple-m4-max",
        locator=locator,
        contract=locator.load(),
        track_id="tilelang-pipeline",
        track_version=version,
    )


def _orchestrator(tmp_path, locator, *, binding=None, evaluator=None):
    return Orchestrator(
        locator.op_name,
        "fp32",
        "small",
        _llm(),
        _search(),
        run_dir=tmp_path / "run",
        binding=binding or _binding(locator),
        evaluator=evaluator,
        track_id="tilelang-pipeline",
        track_version=3,
        resuming=True,
        agent_runner=lambda *_args: (_ for _ in ()).throw(AssertionError("resume started an agent")),
        log=lambda _message: None,
    )


def test_mt6_run_config_and_orchestrator_freeze_v3(fresh_registry, tmp_path):
    locator = write_mt2_task(tmp_path)
    _config(tmp_path / "run")
    assert _recorded_track_selection(tmp_path / "run") == ("tilelang-pipeline", 3)
    assert _kernel_name(tmp_path / "run") == "kernel.taic"
    orch = _orchestrator(tmp_path, locator)
    assert (orch.track_id, orch.track_version) == ("tilelang-pipeline", 3)
    assert (orch.binding.track_id, orch.binding.track_version) == ("tilelang-pipeline", 3)
    assert orch.evaluator.binding.track.track_version == 3


def test_mt6_snapshot_dry_run_restores_taic_without_writing_or_agent(fresh_registry, tmp_path):
    locator = write_mt2_task(tmp_path)
    _config(tmp_path / "run")
    orch = _orchestrator(tmp_path, locator)
    source = encode_envelope(tilelang_candidate(worklist=False))
    report = EvalReport(
        task=locator.op_name,
        dtype="fp32",
        size="small",
        mode="quick",
        status="ok",
        stage="ok",
        seed=7,
    )
    snapshot = Snapshot(
        run_dir=str(tmp_path / "run"),
        task=locator.op_name,
        dtype="fp32",
        size="small",
        model="fake",
        requests=[],
        candidates=[{"idx": 0, "label": "init", "source": source, "report": report, "plan": {}}],
        trace=[],
        history=[],
        best_label="init",
        rounds_done=0,
        next_round=0,
        stagnant=0,
        partial=None,
    )
    orch.restore(snapshot, dry_run=True)
    assert len(orch.candidates) == 1
    assert orch.candidates[0].source == source
    assert decode_envelope(orch.candidates[0].source).candidate_sha256 == tilelang_candidate(worklist=False).candidate_sha256
    assert not (tmp_path / "run" / "state.json").exists()
    assert not (tmp_path / "run" / "candidates" / "000_init" / "kernel.taic").exists()


def test_mt6_resume_refuses_an_evaluator_from_v2(fresh_registry, tmp_path):
    locator = write_mt2_task(tmp_path)
    _config(tmp_path / "run")
    old_binding = _binding(locator, version=2)
    with pytest.raises(RunStartError, match="never silently upgrades"):
        _orchestrator(
            tmp_path,
            locator,
            binding=_binding(locator),
            evaluator=V2LocalEvaluator(old_binding),
        )
