"""MT1b: a resumed run restores the exact ``(track_id, track_version)``, end to end.

The chain under test is the real one: ``run_config.json`` bytes -> ``taic.cli`` parsing ->
``Orchestrator`` -> ``TaskBinding`` -> renderer/evaluator/gates.  Two versions of ``msl-single``
are registered and both are selectable, so any place that drops the version silently resolves
version 2 instead of the recorded version 1.
"""

from __future__ import annotations

import dataclasses
import json

import pytest

from taic.cli import (
    ResumeRefused,
    _recorded_track,
    _recorded_track_selection,
    _recorded_track_version,
    _resume_track_choices,
    _run_track_choices,
)
from taic.llm import LLMConfig, SearchConfig
from taic.orchestrator import Orchestrator, RunStartError, _resolve_run_track
from taic.task_binding import bind_task
from taic.v2_evaluation import V2LocalEvaluator
import tcenv.track.registry as registry_module
from tcenv.track.registry import TrackRegistry


def _two_version_registry() -> tuple[TrackRegistry, object, object]:
    live = registry_module.load_track_registry()
    v1 = live.get("msl-single")
    v2 = dataclasses.replace(v1, track_version=2)
    bundle = live.get("msl-bundle")
    tilelang = live.get("tilelang-pipeline", version=1)
    return TrackRegistry(backend_id="metal", tracks=(v1, v2, bundle, tilelang)), v1, v2


@pytest.fixture()
def two_versions(monkeypatch):
    registry, v1, v2 = _two_version_registry()
    original = registry_module.load_track_registry
    monkeypatch.setattr(registry_module, "load_track_registry", lambda path=None: registry)
    yield v1, v2
    monkeypatch.setattr(registry_module, "load_track_registry", original)


def _write_config(run_dir, *, track_id="msl-single", version=1, delivery="single_source", task_mode="v2"):
    run_dir.mkdir(parents=True, exist_ok=True)
    document = {"task_mode": task_mode, "track": {"track_id": track_id, "track_version": version, "delivery": delivery}}
    (run_dir / "run_config.json").write_text(json.dumps(document), encoding="utf-8")


def _orchestrator(tmp_path, *, track_id, version):
    binding = bind_task("gemm", "fp16", "small")
    llm = LLMConfig(
        base_url="http://x", api_key="sk-test-0123456789abcdef", model="fake",
        reasoning_effort="off", use_tools=False, truncation_retry=False,
    )
    off = dict(
        stop_at_roofline=False, abi_guard=False, target_note=False, rich_history=False,
        status_lines=False, plan_ledger=False,
    )
    search = SearchConfig(
        budget_usd=5.0, branches=1, max_rounds=0, eval_reps=3, kernel_agent="pi",
        correctness_only=True, **off,
    )
    # No injected evaluator: the production path builds V2LocalEvaluator from the binding the
    # Orchestrator has just frozen, so this exercises the real assembly order.
    return Orchestrator(
        "gemm", "fp16", "small", llm, search,
        run_dir=tmp_path / "run",
        log=lambda _message: None,
        track_id=track_id,
        track_version=version,
    )


def test_a_resumed_v1_run_selects_v1_even_when_v2_is_available(tmp_path, two_versions):
    """@brief 记录 version=1、registry 同时有 1/2：续跑必须落在 1，而不是最高可选版本。"""
    v1, v2 = two_versions
    run_dir = tmp_path / "old-run"
    _write_config(run_dir, version=1)

    # Both versions are selectable, so the default is 2: the assertion below is not vacuous.
    registry = registry_module.load_track_registry()
    assert {track.track_version for track in registry.tracks if track.track_id == "msl-single"} == {1, 2}
    assert registry.require_available("msl-single").track_version == 2

    recorded_id, recorded_version = _recorded_track_selection(run_dir)
    assert (recorded_id, recorded_version) == ("msl-single", 1)
    assert _recorded_track(run_dir) == "msl-single"
    assert _recorded_track_version(run_dir) == 1

    orch = _orchestrator(tmp_path, track_id=recorded_id, version=recorded_version)
    assert (orch.track_id, orch.track_version) == ("msl-single", 1)
    assert (orch.binding.track_id, orch.binding.track_version) == ("msl-single", 1)
    assert orch.binding.track.track_version == 1
    from tcenv.identity import fingerprint

    assert fingerprint(orch.track.document()) == fingerprint(v1.document())
    assert fingerprint(orch.track.document()) != fingerprint(v2.document())
    # The evaluator is assembled from the same frozen selection.
    assert orch.evaluator.binding.track.track_version == 1
    # And the new run that this resume must not imitate would pick version 2.
    assert _resolve_run_track(bind_task("gemm", "fp16", "small"), "msl-single")[0].track_version == 2


def test_a_new_run_records_the_version_it_selected(tmp_path, two_versions):
    """@brief 新 run（无记录版本）取最高可选版本，并把版本写进 run_config。"""
    _v1, v2 = two_versions
    orch = _orchestrator(tmp_path, track_id="msl-single", version=None)
    assert (orch.track_id, orch.track_version) == ("msl-single", 2)
    recorded = json.loads((tmp_path / "run" / "run_config.json").read_text(encoding="utf-8"))
    assert recorded["track"] == {"track_id": "msl-single", "track_version": 2, "delivery": "single_source"}
    assert _recorded_track_selection(tmp_path / "run") == ("msl-single", 2)


def test_a_recorded_version_without_a_registered_row_is_refused(tmp_path, monkeypatch):
    """@brief 记录的版本已不存在：拒绝续跑，不回退到别的版本。"""
    registry, _v1, _v2 = _two_version_registry()
    only_v2 = TrackRegistry(
        backend_id="metal", tracks=tuple(track for track in registry.tracks if track.key != ("msl-single", 1))
    )
    monkeypatch.setattr(registry_module, "load_track_registry", lambda path=None: only_v2)
    run_dir = tmp_path / "stale"
    _write_config(run_dir, version=1)
    with pytest.raises(ResumeRefused) as refusal:
        _recorded_track_version(run_dir)
    assert "not registered at version 1" in str(refusal.value)
    with pytest.raises(RunStartError, match="msl-single/1"):
        _orchestrator(tmp_path, track_id="msl-single", version=1)


@pytest.mark.parametrize("value", [True, False, 1.0, "1", 0, -1, None])
def test_a_recorded_version_is_never_coerced(tmp_path, two_versions, value):
    """@brief bool/浮点/字符串/0/None 都不是版本：typed 拒绝，不做 int() 强转。"""
    run_dir = tmp_path / "bad"
    _write_config(run_dir, version=value)
    with pytest.raises(ResumeRefused, match="no coercion"):
        _recorded_track_version(run_dir)
    with pytest.raises(ResumeRefused, match="no coercion"):
        _recorded_track(run_dir)


def test_a_v1_resume_has_no_version_and_a_legacy_v2_block_still_means_version_one(tmp_path, two_versions):
    """@brief v1 run 没有版本；schema<3 的 v2 run 按记录重建为 msl-single/1。"""
    v1_run = tmp_path / "v1"
    v1_run.mkdir()
    (v1_run / "run_config.json").write_text(json.dumps({"task_mode": "v1"}), encoding="utf-8")
    assert _recorded_track_selection(v1_run) == (None, None)

    legacy = tmp_path / "legacy"
    legacy.mkdir()
    (legacy / "run_config.json").write_text(json.dumps({"task_mode": "v2"}), encoding="utf-8")
    assert _recorded_track_selection(legacy) == ("msl-single", 1)


def test_track_choices_are_deduplicated_and_asymmetric(two_versions):
    """@brief resume 接受每个 id 一次（版本由记录决定）；新 run 只列可选 track。"""
    choices = _resume_track_choices()
    assert choices == ["msl-single", "msl-bundle", "tilelang-pipeline"]
    assert len(choices) == len(set(choices))
    assert _run_track_choices() == ["msl-single", "msl-bundle"]


def test_an_injected_evaluator_pinned_to_another_version_is_refused(tmp_path, two_versions):
    """@brief 注入的评测器 binding 解析到 v2 而 run 冻结在 v1：启动拒绝。"""
    binding = bind_task("gemm", "fp16", "small")
    llm = LLMConfig(
        base_url="http://x", api_key="sk-test-0123456789abcdef", model="fake",
        reasoning_effort="off", use_tools=False, truncation_retry=False,
    )
    search = SearchConfig(
        budget_usd=5.0, branches=1, max_rounds=0, eval_reps=3, kernel_agent="pi",
        correctness_only=True, stop_at_roofline=False, abi_guard=False, target_note=False,
        rich_history=False, status_lines=False, plan_ledger=False,
    )
    with pytest.raises(RunStartError, match="never silently upgrades"):
        Orchestrator(
            "gemm", "fp16", "small", llm, search,
            run_dir=tmp_path / "run",
            evaluator=V2LocalEvaluator(binding),  # binding.track_version is None -> default v2
            log=lambda _message: None,
            track_id="msl-single",
            track_version=1,
        )
