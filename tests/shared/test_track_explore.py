"""MT5 deterministic fake-agent/CPU-fixture coverage. No real run, model, GPU, or NPU."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from taic.track_explore import (
    SingleTrackRequest,
    SubprocessSingleTrackRunner,
    TrackExploreConfig,
    TrackExploreError,
    comparability_digest,
    exploration_bundle_digest,
    explore_tracks,
    summarize_track_explore,
    validate_track_explore,
)
from taic.llm import LLMConfig, SearchConfig
from tcenv.identity import fingerprint
from tcenv.track.registry import load_track_registry

from ..support.track_explore import FixedTrackExploreFixture


def _run(fixture: FixedTrackExploreFixture, **kwargs):
    return explore_tracks(
        fixture.config,
        runner=fixture,
        adapter_factory=fixture.adapter_factory,
        metadata_provider=fixture.metadata,
        **kwargs,
    )


def _track(summary, track_id):
    return next(row for row in summary["tracks"] if row["track_id"] == track_id)


def _instance(summary, track_id, instance_key="fp16/above/benchmark"):
    return next(row for row in _track(summary, track_id)["instances"] if row["instance_key"] == instance_key)


def _production_request(tmp_path, *, task="kda", size="single", dtype="fp16", track_id="msl-single"):
    fixture = FixedTrackExploreFixture(tmp_path)
    track = load_track_registry().get(track_id)
    benchmark = {"dtype": dtype, "size": size, "usage": "benchmark", "instance_key": f"{dtype}/{size}/benchmark"}
    return SingleTrackRequest(
        task=task,
        target="apple-m4-max",
        track=track,
        benchmark=benchmark,
        run_dir=tmp_path / task / track_id / "run",
        run_seed=1,
        max_agent_turns=3,
        build_timeout_sec=45.0,
        generation={"config": None, "preset": None, "model": None},
        shared=fixture.config.document["shared"],
        budget_usd=0.001,
        budget_scope="per_track_instance",
        max_wall_sec=30.0,
    ), fixture


def test_full_success_scheme_a_costs_and_independent_fresh_identities(tmp_path):
    fixture = FixedTrackExploreFixture(tmp_path)
    summary = _run(fixture)
    assert summary["status"] == "exploration_winner"
    assert summary["winner"]["track_id"] == "msl-bundle"
    assert summary["winner"]["formal_publication"] is False
    assert summary["same_fresh_input"] is False
    single = _instance(summary, "msl-single")
    bundle = _instance(summary, "msl-bundle")
    assert single["run_dir"] != bundle["run_dir"]
    assert single["actual_confirmation"]["portfolio_freeze_digest"] != bundle["actual_confirmation"]["portfolio_freeze_digest"]
    assert single["actual_confirmation"]["confirmation_seed"] != bundle["actual_confirmation"]["confirmation_seed"]
    assert single["actual_confirmation"]["fresh_input_identity"]["digest"] != bundle["actual_confirmation"]["fresh_input_identity"]["digest"]
    for row in (single, bundle):
        identity = row["actual_confirmation"]["fresh_input_identity"]
        assert "fixed-shape: schema-4 has no per-case byte digest" in identity["identity_scope"]
        assert identity["cases"] and all(case["input_sha256"] is None for case in identity["cases"])
        assert all(len(case["derived_input_identity"]) == 64 for case in identity["cases"])
    assert single["search_cost"]["build_timeout_sec"] == 30.0
    assert bundle["search_cost"]["build_timeout_sec"] == 60.0
    assert single["search_cost"]["agent_turns"] == bundle["search_cost"]["agent_turns"] == 2
    assert single["search_cost"]["build_seconds"] > 0
    assert summary["budget"]["actual_total_spent_usd"] == pytest.approx(0.75)
    assert summary["budget"]["actual_total_search_wall_sec"] == pytest.approx(21.0)
    assert summary["budget"]["cost_complete"] is True
    assert summary["quiet_confirmation_serial"] is True
    assert [call.track.track_id for call in fixture.calls] == ["msl-single", "msl-bundle"]
    assert all(call.serial_confirmation for call in fixture.calls)
    on_disk = json.loads(fixture.config.output_path.read_text())
    assert on_disk == summary


def test_repeated_exploration_is_idempotent_and_does_not_restart_paid_work(tmp_path):
    fixture = FixedTrackExploreFixture(tmp_path)
    first = _run(fixture)
    before = fixture.config.output_path.read_bytes()
    calls = len(fixture.calls)
    second = _run(fixture)
    assert second == first
    assert fixture.config.output_path.read_bytes() == before
    assert len(fixture.calls) == calls


def test_a_partial_run_hits_the_resume_interlock_instead_of_starting_over(tmp_path):
    fixture = FixedTrackExploreFixture(tmp_path)
    run_dir = tmp_path / "runs" / "msl-single" / "fp16__above__benchmark"
    run_dir.mkdir(parents=True)
    (run_dir / "state.json").write_text("{}")
    with pytest.raises(TrackExploreError, match="resume_interlock"):
        _run(fixture)
    assert fixture.calls == []


def test_incorrect_and_unsafe_candidates_do_not_produce_a_winner(tmp_path):
    fixture = FixedTrackExploreFixture(tmp_path)
    fixture.overrides[("msl-single", "fp16/above/benchmark")] = {"result_status": "incorrect"}
    fixture.overrides[("msl-bundle", "fp16/above/benchmark")] = {"result_status": "unsafe"}
    summary = _run(fixture)
    assert summary["status"] == "no_winner" and summary["winner"] is None
    assert "incorrect" in _instance(summary, "msl-single")["elimination_reasons"]
    assert "unsafe" in _instance(summary, "msl-bundle")["elimination_reasons"]
    failed = _instance(summary, "msl-single")
    assert failed["search_cost"]["agent_turns"] == 2
    assert failed["search_cost"]["build_cost_details"]["reports"]
    assert failed["run_identity"]["track"]["track_id"] == "msl-single"
    assert failed["confirmation_status"] == "incorrect"


def test_no_baseline_is_correctness_only_and_cannot_create_a_performance_winner(tmp_path):
    fixture = FixedTrackExploreFixture(tmp_path)
    fixture.config.document["shared"]["baseline"] = None
    fixture.config.document["comparability_digest"] = comparability_digest(fixture.config.document)
    for track in fixture.tracks:
        fixture.overrides[(track, "fp16/above/benchmark")] = {"no_baseline": True}
    summary = _run(fixture)
    assert summary["status"] == "no_winner"
    assert "no_baseline" in _instance(summary, "msl-single")["elimination_reasons"]
    assert "baseline" not in summary["tracks"][0]["instances"][0].get("comparability_differences", [])
    from taic.track_explore import comparability_document

    assert "baseline" not in comparability_document(fixture.config.document)


def test_golden_only_exploration_is_unranked_and_never_starts_a_run_in_summary_mode(tmp_path):
    fixture = FixedTrackExploreFixture(tmp_path)
    fixture.config.document["benchmarks"] = [{
        "dtype": "fp16", "size": "above", "usage": "golden", "instance_key": "fp16/above/golden",
    }]
    fixture.config.document["comparability_digest"] = comparability_digest(fixture.config.document)
    summary = _run(fixture, summarize_only=True)
    assert summary["status"] == "no_winner"
    assert fixture.calls == []
    assert _instance(summary, "msl-single", "fp16/above/golden")["ranked"] is False
    assert summary["budget"]["cost_complete"] is False
    assert summary["budget"]["actual_total_spent_usd"] is None
    assert summary["budget"]["actual_total_search_wall_sec"] is None


def test_golden_runs_contribute_wall_time_and_cost_even_without_a_performance_winner(tmp_path):
    fixture = FixedTrackExploreFixture(tmp_path)
    fixture.config.document["benchmarks"] = [{
        "dtype": "fp16", "size": "above", "usage": "golden", "instance_key": "fp16/above/golden",
    }]
    fixture.config.document["comparability_digest"] = comparability_digest(fixture.config.document)
    summary = _run(fixture)
    assert summary["status"] == "no_winner"
    assert summary["budget"]["actual_total_spent_usd"] == pytest.approx(0.75)
    assert summary["budget"]["actual_total_search_wall_sec"] == pytest.approx(21.0)
    assert summary["budget"]["cost_complete"] is True
    for track_id in ("msl-single", "msl-bundle"):
        row = _instance(summary, track_id, "fp16/above/golden")
        assert row["ranked"] is False
        assert row["run_identity"]["track"]["track_id"] == track_id
        assert row["candidate_sha256"] and row["build_identity"] and row["evaluation_identity"]
        assert row["confirmation_status"] == "ok"
        assert row["search_cost"]["agent_turns"] == 2
        assert row["search_cost"]["build_seconds"] > 0
        assert row["search_cost"]["build_cost_details"]["reports"]


def test_timing_uncertain_eliminates_only_the_affected_track(tmp_path):
    fixture = FixedTrackExploreFixture(tmp_path)
    fixture.overrides[("msl-single", "fp16/above/benchmark")] = {"timing_uncertain": True}
    summary = _run(fixture)
    assert summary["winner"]["track_id"] == "msl-bundle"
    assert "timing_uncertain" in _instance(summary, "msl-single")["elimination_reasons"]


def test_shared_run_directories_are_refused_before_cache_or_evidence_can_be_reused(tmp_path):
    fixture = FixedTrackExploreFixture(tmp_path)
    fixture.config.document["participating_tracks"][1]["run_dir"] = "runs/msl-single"
    with pytest.raises(TrackExploreError, match="independent, non-nested run_dir"):
        validate_track_explore(fixture.config.document)


def test_hardlinked_confirmation_evidence_is_cache_reuse_not_independent_confirmation(tmp_path):
    fixture = FixedTrackExploreFixture(tmp_path)
    _run(fixture)
    single = tmp_path / "runs" / "msl-single" / "fp16__above__benchmark" / "confirm_report.json"
    bundle = tmp_path / "runs" / "msl-bundle" / "fp16__above__benchmark" / "confirm_report.json"
    bundle.unlink()
    os.link(single, bundle)
    summary = summarize_track_explore(
        fixture.config, adapter_factory=fixture.adapter_factory, metadata_provider=fixture.metadata,
    )
    assert summary["status"] == "exploration_winner"
    assert summary["winner"]["track_id"] == "msl-single"
    assert _instance(summary, "msl-single")["elimination_reasons"] == []
    assert _instance(summary, "msl-bundle")["elimination_reasons"] == ["cache_reuse"]


def test_track_drift_is_rejected_even_when_the_other_track_has_valid_evidence(tmp_path):
    fixture = FixedTrackExploreFixture(tmp_path)
    fixture.overrides[("msl-single", "fp16/above/benchmark")] = {"actual_track": "msl-bundle"}
    summary = _run(fixture)
    assert summary["winner"]["track_id"] == "msl-bundle"
    assert _instance(summary, "msl-single")["elimination_reasons"] == ["track_mismatch"]


def test_a_different_baseline_is_display_only_and_cannot_join_the_comparison(tmp_path):
    fixture = FixedTrackExploreFixture(tmp_path)
    fixture.overrides[("msl-single", "fp16/above/benchmark")] = {"different_baseline": True}
    summary = _run(fixture)
    row = _instance(summary, "msl-single")
    assert summary["winner"]["track_id"] == "msl-bundle"
    assert row["eligible"] is False
    assert row["comparability_digest"] != summary["comparability_digest"]
    assert "comparability_mismatch" in row["elimination_reasons"]
    assert "baseline" in row["comparability_differences"]


def test_all_incomparable_results_produce_no_winner(tmp_path):
    fixture = FixedTrackExploreFixture(tmp_path)
    for track in fixture.tracks:
        fixture.overrides[(track, "fp16/above/benchmark")] = {"different_baseline": True}
    summary = _run(fixture)
    assert summary["status"] == "no_winner"


def test_a_nonfresh_confirmation_seed_is_rejected(tmp_path):
    fixture = FixedTrackExploreFixture(tmp_path)
    fixture.overrides[("msl-single", "fp16/above/benchmark")] = {"confirmation_seed": 7}
    summary = _run(fixture)
    assert summary["winner"]["track_id"] == "msl-bundle"
    assert "not_fresh" in _instance(summary, "msl-single")["elimination_reasons"]


def test_single_instance_allows_the_exact_110_boundary_without_an_extra_speedup_gate(tmp_path):
    fixture = FixedTrackExploreFixture(tmp_path)
    fixture.overrides[("msl-single", "fp16/above/benchmark")] = {"ratio": 1.10}
    fixture.overrides[("msl-bundle", "fp16/above/benchmark")] = {"ratio": 1.100001}
    summary = _run(fixture)
    assert summary["winner"]["track_id"] == "msl-single"
    assert summary["winner"]["geomean_latency_ratio"] == pytest.approx(1.10)
    assert "latency_gate" in _instance(summary, "msl-bundle")["elimination_reasons"]


def test_single_instance_above_110_on_every_track_is_no_winner(tmp_path):
    fixture = FixedTrackExploreFixture(tmp_path)
    for track in fixture.tracks:
        fixture.overrides[(track, "fp16/above/benchmark")] = {"ratio": 1.100001}
    summary = _run(fixture)
    assert summary["status"] == "no_winner"


def test_multi_instance_geomean_prevents_the_long_instance_from_monopolizing_selection(tmp_path):
    fixture = FixedTrackExploreFixture(tmp_path, instances=["above", "below"])
    fixture.overrides[("msl-single", "fp16/above/benchmark")] = {"ratio": 0.20, "baseline_us": 10000.0}
    fixture.overrides[("msl-single", "fp16/below/benchmark")] = {"ratio": 1.09, "baseline_us": 10.0}
    fixture.overrides[("msl-bundle", "fp16/above/benchmark")] = {"ratio": 0.90, "baseline_us": 10000.0}
    fixture.overrides[("msl-bundle", "fp16/below/benchmark")] = {"ratio": 0.20, "baseline_us": 10.0}
    summary = _run(fixture)
    single, bundle = _track(summary, "msl-single"), _track(summary, "msl-bundle")
    assert single["unnormalized_sum_sum_latency_ratio"] < bundle["unnormalized_sum_sum_latency_ratio"]
    assert bundle["geomean_latency_ratio"] < single["geomean_latency_ratio"]
    assert summary["winner"]["track_id"] == "msl-bundle"
    vector = summary["winner"]["per_instance_candidate_vector"]
    assert [row["instance_key"] for row in vector] == ["fp16/above/benchmark", "fp16/below/benchmark"]
    assert {row["candidate_sha256"] for row in vector} == {vector[0]["candidate_sha256"]}
    assert all(row["candidate_sha256"].startswith("22c5880e60") for row in vector)


def test_worst_shard_is_only_a_post_gate_tie_break(tmp_path):
    fixture = FixedTrackExploreFixture(tmp_path, instances=["above", "below"])
    fixture.overrides[("msl-single", "fp16/above/benchmark")] = {"ratio": 0.5}
    fixture.overrides[("msl-single", "fp16/below/benchmark")] = {"ratio": 1.1}
    fixture.overrides[("msl-bundle", "fp16/above/benchmark")] = {"ratio": 0.55}
    fixture.overrides[("msl-bundle", "fp16/below/benchmark")] = {"ratio": 1.0}
    summary = _run(fixture)
    single, bundle = _track(summary, "msl-single"), _track(summary, "msl-bundle")
    assert single["geomean_latency_ratio"] == pytest.approx(bundle["geomean_latency_ratio"])
    assert summary["winner"]["track_id"] == "msl-bundle"
    assert summary["winner"]["worst_shard_latency_ratio"] == pytest.approx(1.0)


def test_a_complete_tie_is_broken_by_candidate_identity_then_track_identity(tmp_path):
    fixture = FixedTrackExploreFixture(tmp_path)
    for track in fixture.tracks:
        fixture.overrides[(track, "fp16/above/benchmark")] = {"ratio": 0.8}
    summary = _run(fixture)
    # The fixed msl-single candidate digest sorts before the bundle digest.
    assert summary["winner"]["track_id"] == "msl-single"


def test_a_missing_instance_eliminates_the_whole_track_bundle(tmp_path):
    fixture = FixedTrackExploreFixture(tmp_path, instances=["above", "below"], tracks=["msl-single"])
    fixture.overrides[("msl-single", "fp16/below/benchmark")] = {"missing": True}
    summary = _run(fixture)
    assert summary["status"] == "no_winner"
    assert any(reason.startswith("missing_instance:") for reason in _track(summary, "msl-single")["elimination_reasons"])


def test_geomean_cannot_hide_a_local_latency_failure(tmp_path):
    fixture = FixedTrackExploreFixture(tmp_path, instances=["above", "below"])
    fixture.overrides[("msl-single", "fp16/above/benchmark")] = {"ratio": 0.01}
    fixture.overrides[("msl-single", "fp16/below/benchmark")] = {"ratio": 1.10001}
    fixture.overrides[("msl-bundle", "fp16/above/benchmark")] = {"ratio": 0.9}
    fixture.overrides[("msl-bundle", "fp16/below/benchmark")] = {"ratio": 0.9}
    summary = _run(fixture)
    assert summary["winner"]["track_id"] == "msl-bundle"
    assert "latency_gate" in _instance(summary, "msl-single", "fp16/below/benchmark")["elimination_reasons"]


def test_versioned_input_binds_master_comparability_and_serial_policy(tmp_path):
    fixture = FixedTrackExploreFixture(tmp_path)
    loaded = TrackExploreConfig.load(fixture.config.path)
    assert loaded.digest == fixture.config.digest
    assert validate_track_explore(loaded.document) == loaded.document
    for path, value, match in (
        (("master_seed",), "0" * 64, "master_seed"),
        (("comparability_digest",), "0" * 64, "comparability_digest"),
        (("generation_policy",), "parallel", "serial"),
        (("budget_scope",), "global", "budget_scope"),
        (("max_agent_turns",), 0, "max_agent_turns"),
    ):
        changed = json.loads(json.dumps(loaded.document))
        changed[path[0]] = value
        with pytest.raises(TrackExploreError, match=match):
            validate_track_explore(changed)
    duplicate = json.loads(json.dumps(loaded.document))
    duplicate["benchmarks"].append(dict(duplicate["benchmarks"][0]))
    with pytest.raises(TrackExploreError, match="unique"):
        validate_track_explore(duplicate)


def test_per_track_budget_scope_reserves_remaining_budget_and_records_overrun(tmp_path):
    fixture = FixedTrackExploreFixture(tmp_path, instances=["above", "below"])
    fixture.config.document["budget_scope"] = "per_track"
    fixture.config.document["budget_usd"] = 0.4
    summary = _run(fixture)
    # msl-bundle has no budget left for its second instance and cannot win a full bundle.
    assert summary["winner"]["track_id"] == "msl-single"
    skipped = _instance(summary, "msl-bundle", "fp16/below/benchmark")
    assert skipped["observation"]["skipped_reason"] == "track_budget_exhausted"
    first = _instance(summary, "msl-bundle", "fp16/above/benchmark")
    assert first["observation"]["budget_usd"] == pytest.approx(0.4)
    second = _instance(summary, "msl-single", "fp16/below/benchmark")
    assert second["observation"]["budget_usd"] == pytest.approx(0.15)
    # Existing model billing is not a hard in-flight cap; actual overspend is retained, not hidden.
    assert summary["budget"]["actual_total_spent_usd"] == pytest.approx(1.0)
    assert summary["budget"]["scope"] == "per_track"


def test_timeout_and_failed_runner_batches_are_not_eligible_even_with_fixture_reports(tmp_path):
    fixture = FixedTrackExploreFixture(tmp_path)
    fixture.overrides[("msl-single", "fp16/above/benchmark")] = {"timeout": True}
    fixture.overrides[("msl-bundle", "fp16/above/benchmark")] = {"returncode": 6}
    summary = _run(fixture)
    assert summary["status"] == "no_winner"
    assert "runner_timeout" in _instance(summary, "msl-single")["elimination_reasons"]
    assert "runner_failed" in _instance(summary, "msl-bundle")["elimination_reasons"]


def test_observation_budget_allocation_mismatch_is_rejected(tmp_path):
    fixture = FixedTrackExploreFixture(tmp_path)
    _run(fixture)
    observations = json.loads(fixture.config.observation_path.read_text())
    observations["runs"]["msl-single/fp16/above/benchmark"]["budget_usd"] = 0.123
    fixture.config.observation_path.write_text(json.dumps(observations))
    summary = summarize_track_explore(
        fixture.config, adapter_factory=fixture.adapter_factory, metadata_provider=fixture.metadata,
    )
    assert summary["winner"]["track_id"] == "msl-bundle"
    assert "generation_config_mismatch" in _instance(summary, "msl-single")["elimination_reasons"]


def test_result_run_seed_mismatch_is_rejected(tmp_path):
    fixture = FixedTrackExploreFixture(tmp_path)
    _run(fixture)
    result_path = tmp_path / "runs" / "msl-single" / "fp16__above__benchmark" / "result.json"
    result = json.loads(result_path.read_text())
    result["run_seed"] += 1
    result_path.write_text(json.dumps(result))
    summary = summarize_track_explore(
        fixture.config, adapter_factory=fixture.adapter_factory, metadata_provider=fixture.metadata,
    )
    assert summary["winner"]["track_id"] == "msl-bundle"
    assert "run_identity_mismatch" in _instance(summary, "msl-single")["elimination_reasons"]


def test_selection_policy_digest_is_a_fixed_versioned_golden(tmp_path):
    fixture = FixedTrackExploreFixture(tmp_path)
    summary = _run(fixture)
    assert summary["selection_policy_digest"] == "22e4283765d35cba939699a10cfac70592f8f6147d2ec8100fb0ac06cd3ca5f8"
    assert summary["selection_policy"]["additional_speedup_gt_one_gate"] is False
    assert summary["selection_policy"]["unnormalized_sum_sum_is_ranking"] is False


def test_compile_error_build_cost_is_retained_with_a_failure_list(tmp_path):
    fixture = FixedTrackExploreFixture(tmp_path)
    fixture.overrides[("msl-single", "fp16/above/benchmark")] = {"extra_compile_error": True}
    summary = _run(fixture)
    row = _instance(summary, "msl-single")
    assert row["eligible"] is True
    assert row["search_cost"]["build_seconds"] > 0
    assert row["search_cost"]["build_cost_details"]["failed_reports"] == ["candidates/001_bad/report.json"]
    assert row["search_cost"]["build_cost_details"]["missing_duration_reports"] == []
    assert row["search_cost"]["cost_complete"] is True


def test_missing_build_duration_is_explicitly_incomplete_without_losing_known_cost(tmp_path):
    fixture = FixedTrackExploreFixture(tmp_path)
    fixture.overrides[("msl-single", "fp16/above/benchmark")] = {
        "extra_compile_error": True,
        "compile_error_missing_duration": True,
    }
    summary = _run(fixture)
    row = _instance(summary, "msl-single")
    assert row["search_cost"]["build_seconds"] > 0
    assert row["search_cost"]["build_cost_details"]["missing_duration_reports"] == ["candidates/001_bad/report.json"]
    assert row["search_cost"]["cost_complete"] is False


def test_pre_mt1a_guidance_is_recomputed_from_the_archived_system_prompt(tmp_path):
    fixture = FixedTrackExploreFixture(tmp_path)
    fixture.overrides[("msl-single", "fp16/above/benchmark")] = {"legacy_guidance": True}
    summary = _run(fixture)
    row = _instance(summary, "msl-single")
    assert row["eligible"] is True
    assert row["run_identity"]["confirmation_guidance_digest"] == fingerprint({
        "guidance": ["fixture legacy system", "fixture statement"]
    })
    assert "seed_derivation_unavailable" not in row["elimination_reasons"]


def test_pre_mt1a_guidance_without_system_prompt_is_a_typed_refusal(tmp_path):
    fixture = FixedTrackExploreFixture(tmp_path)
    fixture.overrides[("msl-single", "fp16/above/benchmark")] = {
        "legacy_guidance": True,
        "missing_legacy_system": True,
    }
    summary = _run(fixture)
    row = _instance(summary, "msl-single")
    assert summary["winner"]["track_id"] == "msl-bundle"
    assert "confirmation_guidance_unavailable" in row["elimination_reasons"]


def test_confirm_batches_run_config_mismatch_is_rejected(tmp_path):
    fixture = FixedTrackExploreFixture(tmp_path)
    fixture.overrides[("msl-single", "fp16/above/benchmark")] = {"confirm_batches": 3}
    summary = _run(fixture)
    assert "generation_config_mismatch" in _instance(summary, "msl-single")["elimination_reasons"]


def test_confirmation_retry_attempts_are_not_independent_batches(tmp_path):
    fixture = FixedTrackExploreFixture(tmp_path)
    fixture.overrides[("msl-single", "fp16/above/benchmark")] = {
        "confirmation_attempts": 2,
        "benchmark_attempts": 2,
    }
    summary = _run(fixture)
    assert "confirmation_batch_mismatch" not in _instance(summary, "msl-single")["elimination_reasons"]
    (tmp_path / "rejected").mkdir()
    rejected = FixedTrackExploreFixture(tmp_path / "rejected")
    rejected.overrides[("msl-single", "fp16/above/benchmark")] = {
        "confirmation_attempts": 4,
        "benchmark_attempts": 4,
    }
    summary = _run(rejected)
    assert "invalid_attempt_count" in _instance(summary, "msl-single")["elimination_reasons"]


def test_a_fully_failed_run_still_has_identity_agent_turns_and_build_cost(tmp_path):
    fixture = FixedTrackExploreFixture(tmp_path)
    fixture.overrides[("msl-single", "fp16/above/benchmark")] = {"primary_compile_error": True}
    summary = _run(fixture)
    row = _instance(summary, "msl-single")
    assert row["eligible"] is False
    assert row["run_identity"]["track"]["track_id"] == "msl-single"
    assert row["search_cost"]["agent_turns"] == 2
    assert row["search_cost"]["build_seconds"] > 0
    assert "candidates/000_init/report.json" in row["search_cost"]["build_cost_details"]["failed_reports"]
    assert row["search_cost"]["cost_complete"] is True


def test_agent_turn_cap_includes_initial_restarts_and_provider_retries(tmp_path):
    fixture = FixedTrackExploreFixture(tmp_path)
    fixture.overrides[("msl-single", "fp16/above/benchmark")] = {"agent_turns": 4}
    summary = _run(fixture)
    assert summary["winner"]["track_id"] == "msl-bundle"
    row = _instance(summary, "msl-single")
    assert "generation_config_mismatch" in row["elimination_reasons"]
    assert "restarts and provider retries" in row["search_cost"]["agent_turns_semantics"]


@pytest.mark.parametrize(("task", "size"), [("kda", "single"), ("varlen_kda", "mixed")])
def test_production_runner_argv_is_accepted_by_taic_run_for_declared_instances(tmp_path, monkeypatch, task, size):
    request, _fixture = _production_request(tmp_path, task=task, size=size)
    runner = SubprocessSingleTrackRunner()
    argv = runner.argv(request)
    captured = {}

    def fake_cmd_run(args):
        captured["args"] = args
        return 0

    monkeypatch.setattr("taic.cli.cmd_run", fake_cmd_run)
    from taic import cli as taic_cli

    assert taic_cli.main(argv[5:]) == 0
    args = captured["args"]
    assert (args.task, args.dtype, args.size) == (task, "fp16", size)
    assert args.eval_reps == 15 and args.confirm_batches == 1
    assert args.correctness_only is False
    binding = taic_cli._validate_run_instance(args)
    assert binding.contract.instance.dtype == "fp16"
    assert binding.contract.instance.size == size
    environment = runner.environment(request)
    assert environment["TCENV_EVAL_CACHE_DIR"] == str(request.cache_dir.resolve())


def test_an_undeclared_instance_is_a_typed_refusal_not_an_argparse_choice_error(tmp_path, capsys):
    from taic import cli as taic_cli

    code = taic_cli.main([
        "run", "shape_portfolio_pointwise", "--dtype", "fp16", "--size", "above",
        "--run-dir", str(tmp_path / "never"), "--seed", "1", "--budget", "0.001",
    ])
    assert code == 5
    assert "not declared" in capsys.readouterr().err


def test_shared_evaluation_config_reaches_the_existing_run_without_a_real_agent(tmp_path, monkeypatch):
    request, _fixture = _production_request(tmp_path)
    argv = SubprocessSingleTrackRunner().argv(request)
    from taic import cli as taic_cli

    monkeypatch.setattr(taic_cli, "_validate_run_instance", lambda _args: None)
    monkeypatch.setattr("taic.llm.load_raw_config", lambda *_args, **_kwargs: {"search": {}})
    cfg = LLMConfig(base_url="http://x", api_key="sk-test-0123456789abcdef", model="fake", reasoning_effort="off", use_tools=False)
    search = SearchConfig()

    def fake_load_config(*_args, **kwargs):
        search.max_rounds = kwargs["max_rounds"]
        return cfg, search

    monkeypatch.setattr("taic.llm.load_config", fake_load_config)
    monkeypatch.setattr("taic.committees.load_committees", lambda *_args, **_kwargs: ["first", "second"])
    seen = {}

    class FakeOrchestrator:
        def __init__(self, *args, **kwargs):
            seen["search"] = args[4]
            seen["run_dir"] = kwargs["run_dir"]

        async def run(self):
            return SimpleNamespace(
                task="kda", dtype="fp16", model="fake",
                best_speedup=1.0, confirmed_speedup=1.0, spent_usd=0.0,
                stop_reason="done", run_dir=str(seen["run_dir"]),
            )

    monkeypatch.setattr("taic.orchestrator.Orchestrator", FakeOrchestrator)
    assert taic_cli.main(argv[5:]) == 0
    effective = seen["search"]
    assert effective.eval_reps == 15
    assert effective.confirm_batches == 1
    assert effective.correctness_only is False
    assert effective.max_rounds == 2


def test_two_tracks_cannot_share_an_evaluation_cache_directory(tmp_path):
    fixture = FixedTrackExploreFixture(tmp_path)
    fixture.overrides[("msl-bundle", "fp16/above/benchmark")] = {
        "cache_dir": str((tmp_path / "runs" / "msl-single" / "fp16__above__benchmark" / "cache").resolve())
    }
    summary = _run(fixture)
    assert summary["winner"]["track_id"] == "msl-single"
    assert "cache_reuse" in _instance(summary, "msl-bundle")["elimination_reasons"]


def test_a_search_cache_hit_is_recorded_and_rejected_not_hidden_by_fresh_confirmation(tmp_path):
    fixture = FixedTrackExploreFixture(tmp_path)
    fixture.overrides[("msl-single", "fp16/above/benchmark")] = {"search_cache_hit": True}
    summary = _run(fixture)
    row = _instance(summary, "msl-single")
    assert summary["winner"]["track_id"] == "msl-bundle"
    assert row["cache_identity"]["hit_count"] == 1
    assert row["cache_identity"]["confirmation_cache_hit"] is False
    assert "cache_hit" in row["elimination_reasons"]


def test_empty_input_identity_is_display_only_and_cannot_rank(tmp_path):
    fixture = FixedTrackExploreFixture(tmp_path)
    fixture.overrides[("msl-single", "fp16/above/benchmark")] = {"empty_input_identity": True, "variable_length_kind": "l1"}
    summary = _run(fixture)
    assert summary["winner"]["track_id"] == "msl-bundle"
    assert "input_identity_unavailable" in _instance(summary, "msl-single")["elimination_reasons"]


def test_bundle_digest_binds_the_winning_track_and_not_only_the_candidate_vector(tmp_path):
    fixture = FixedTrackExploreFixture(tmp_path)
    summary = _run(fixture)
    winner = summary["winner"]
    vector = winner["per_instance_candidate_vector"]
    bundle = load_track_registry().get("msl-bundle")
    single = load_track_registry().get("msl-single")
    assert winner["exploration_bundle_digest"] == exploration_bundle_digest(bundle, vector)
    assert winner["exploration_bundle_digest"] != exploration_bundle_digest(single, vector)
    from tcenv.identity import fingerprint

    assert winner["exploration_bundle_digest"] != fingerprint({"per_instance_candidate_vector": vector})


def test_run_dir_aliases_and_symlinks_cannot_bypass_independence(tmp_path):
    fixture = FixedTrackExploreFixture(tmp_path)
    fixture.config.document["participating_tracks"][1]["run_dir"] = "runs/b/../msl-single"
    with pytest.raises(TrackExploreError, match="independent, non-nested run_dir"):
        validate_track_explore(fixture.config.document, base_dir=tmp_path)
    actual = tmp_path / "actual"
    actual.mkdir()
    first = tmp_path / "link-a"
    second = tmp_path / "link-b"
    first.symlink_to(actual, target_is_directory=True)
    second.symlink_to(actual, target_is_directory=True)
    fixture.config.document["participating_tracks"][0]["run_dir"] = str(first)
    fixture.config.document["participating_tracks"][1]["run_dir"] = str(second)
    with pytest.raises(TrackExploreError, match="independent, non-nested run_dir"):
        validate_track_explore(fixture.config.document, base_dir=tmp_path)


def test_cli_summarize_only_reads_missing_runs_without_calling_a_model_or_gpu(tmp_path):
    fixture = FixedTrackExploreFixture(tmp_path)
    completed = subprocess.run(
        [
            sys.executable, "-m", "taic.cli", "track-explore",
            "--config", str(fixture.config.path), "--summarize-only",
        ],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["status"] == "no_winner" and payload["exploration_only"] is True
    assert fixture.calls == []
