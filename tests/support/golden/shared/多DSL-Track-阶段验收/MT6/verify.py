#!/usr/bin/env python3
"""Read-only verifier for the MT6 CPU contract slice; no GPU/model execution."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from tcenv.identity import fingerprint, load_strict_json  # noqa: E402
from tcenv.bench import QUIET_CV  # noqa: E402
from taic.trust import current_prompt_sha1  # noqa: E402
from tcenv.track import load_track_registry  # noqa: E402
from tcenv.track import tilelang  # noqa: E402
from tcenv.track import tilelang_confirmation as fresh  # noqa: E402

HERE = Path(__file__).resolve().parent


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def verify_real_metal(identity: dict, nodes: dict) -> None:
    summary = load_strict_json(HERE / "evidence/real-metal-r1/summary.json")
    require(summary["real_metal"] is True and summary["model_executed"] is False, "real Metal evidence scope drift")
    require(len(summary["runs"]) == 3 and summary["golden_observations"] == 2, "initial real Metal run count drift")
    require(summary["stable_fresh_performance"] is False and summary["performance_gate_passed"] is False, "noisy observations became a performance pass")
    require(summary["publication_allowed"] is False and summary["done"] is False, "real Metal observations self-signed publication")
    require(summary["mechanism_identity_match_all"] is True and summary["native_and_artifact_identity_stable_all"] is True, "real Metal identity summary is false")
    rejected = 0
    for row in summary["runs"]:
        benchmark = load_strict_json(HERE / "evidence/real-metal-r1" / row["run"] / "fresh-benchmark.json")
        require(benchmark["status"] == row["status"] and benchmark["reason"] == row["reason"], f"{row['run']}: verdict drift")
        require(benchmark["identity"]["kind"] == row["identity_kind"] == "match", f"{row['run']}: identity mismatch")
        require(benchmark["confirmation_attempts"] == row["confirmation_attempts"], f"{row['run']}: attempt count drift")
        require(benchmark["confirmation_report"]["candidate_us"] == row["candidate_us"], f"{row['run']}: candidate time drift")
        require(benchmark["confirmation_report"]["benchmark"]["noise_cv"] == row["noise_cv"], f"{row['run']}: noise drift")
        require(row["native_digest_match"] and row["artifact_set_match"] and row["compile_profile_match"], f"{row['run']}: build identity drift")
        golden_path = HERE / "evidence/real-metal-r1" / row["run"] / "fresh-golden.json"
        require(golden_path.exists() == row["golden_executed"], f"{row['run']}: golden presence drift")
        if row["golden_executed"]:
            golden = load_strict_json(golden_path)
            require(golden["status"] == row["golden_status"] == "confirmed", f"{row['run']}: golden not confirmed")
            require(golden["identity"]["kind"] == "match", f"{row['run']}: golden identity mismatch")
        if row["status"] == "rejected":
            require(row["reason"] == "timing_uncertain", f"{row['run']}: unexpected rejection")
            rejected += 1
    require(rejected == 2, "the archived timing_uncertain history was rewritten")
    quiet = load_strict_json(HERE / nodes["quiet_series"])
    require(quiet["runs_completed"] == quiet["runs_requested"] == 3, "quiet series is incomplete")
    require(quiet["stable_fresh_performance"] is False and quiet["performance_gate_passed"] is False, "quiet series became a performance pass")
    require(quiet["publication_allowed"] is False and quiet["done"] is False and quiet["reviewer_done"] is False, "quiet series self-signed completion")
    quiet_rejected = 0
    for row in quiet["runs"]:
        run_dir = HERE / "evidence/real-metal-quiet-r2" / row["run"]
        benchmark = load_strict_json(run_dir / "fresh-benchmark.json")
        golden = load_strict_json(run_dir / "fresh-golden.json")
        require(benchmark["status"] == row["status"] and benchmark["reason"] == row["reason"], f"quiet {row['run']}: verdict drift")
        require(benchmark["identity"]["kind"] == row["identity_kind"] == "match", f"quiet {row['run']}: identity mismatch")
        require(benchmark["confirmation_attempts"] == row["confirmation_attempts"], f"quiet {row['run']}: attempt drift")
        require(benchmark["confirmation_report"]["candidate_us"] == row["candidate_us"], f"quiet {row['run']}: candidate time drift")
        require(benchmark["confirmation_report"]["benchmark"]["noise_cv"] == row["noise_cv"], f"quiet {row['run']}: noise drift")
        require(row["native_digest_match"] and row["artifact_set_match"] and row["compile_profile_match"], f"quiet {row['run']}: build identity drift")
        require(golden["status"] == row["golden_status"] == "confirmed" and golden["identity"]["kind"] == "match", f"quiet {row['run']}: golden drift")
        if row["status"] == "rejected":
            require(row["reason"] == "timing_uncertain", f"quiet {row['run']}: unexpected rejection")
            quiet_rejected += 1
    require(quiet_rejected == 1, "quiet series timing history was rewritten")
    latest = load_strict_json(HERE / nodes["quiet_series_r3"])
    require(latest["runs_completed"] == latest["runs_requested"] == 3, "latest quiet series is incomplete")
    require(latest["stable_fresh_performance"] is False and latest["performance_gate_passed"] is False, "latest quiet series became a performance pass")
    require(latest["publication_allowed"] is False and latest["done"] is False and latest["reviewer_done"] is False, "latest quiet series self-signed completion")
    latest_rejected = 0
    for row in latest["runs"]:
        run_dir = HERE / "evidence/real-metal-quiet-r3" / row["run"]
        benchmark = load_strict_json(run_dir / "fresh-benchmark.json")
        golden = load_strict_json(run_dir / "fresh-golden.json")
        require(benchmark["status"] == row["status"] and benchmark["reason"] == row["reason"], f"latest {row['run']}: verdict drift")
        require(benchmark["identity"]["kind"] == row["identity_kind"] == "match", f"latest {row['run']}: identity mismatch")
        require(benchmark["confirmation_attempts"] == row["confirmation_attempts"], f"latest {row['run']}: attempt drift")
        require(benchmark["confirmation_report"]["candidate_us"] == row["candidate_us"], f"latest {row['run']}: candidate time drift")
        require(benchmark["confirmation_report"]["benchmark"]["noise_cv"] == row["noise_cv"], f"latest {row['run']}: noise drift")
        require(row["native_digest_match"] and row["artifact_set_match"] and row["compile_profile_match"], f"latest {row['run']}: build identity drift")
        require(golden["status"] == row["golden_status"] == "confirmed" and golden["identity"]["kind"] == "match", f"latest {row['run']}: golden drift")
        if row["status"] == "rejected":
            require(row["reason"] == "timing_uncertain", f"latest {row['run']}: unexpected rejection")
            latest_rejected += 1
    require(latest_rejected == 2, "latest quiet series timing history was rewritten")
    recovered = load_strict_json(HERE / nodes["quiet_series_r5"])
    require(recovered["runs_completed"] == recovered["runs_requested"] == 3, "recovered quiet series is incomplete")
    require(recovered["stable_fresh_performance"] is False and recovered["performance_gate_passed"] is False, "recovered quiet series became a performance pass")
    require(recovered["publication_allowed"] is False and recovered["done"] is False and recovered["reviewer_done"] is False, "recovered quiet series self-signed completion")
    recovered_rejected = 0
    for row in recovered["runs"]:
        run_dir = HERE / "evidence/real-metal-quiet-r5" / row["run"]
        benchmark = load_strict_json(run_dir / "fresh-benchmark.json")
        golden = load_strict_json(run_dir / "fresh-golden.json")
        require(benchmark["status"] == row["status"] and benchmark["reason"] == row["reason"], f"recovered {row['run']}: verdict drift")
        require(benchmark["identity"]["kind"] == row["identity_kind"] == "match", f"recovered {row['run']}: identity mismatch")
        require(benchmark["confirmation_attempts"] == row["confirmation_attempts"], f"recovered {row['run']}: attempt drift")
        require(benchmark["confirmation_report"]["candidate_us"] == row["candidate_us"], f"recovered {row['run']}: candidate time drift")
        require(benchmark["confirmation_report"]["benchmark"]["noise_cv"] == row["noise_cv"], f"recovered {row['run']}: noise drift")
        require(row["native_digest_match"] and row["artifact_set_match"] and row["compile_profile_match"], f"recovered {row['run']}: build identity drift")
        require(golden["status"] == row["golden_status"] == "confirmed" and golden["identity"]["kind"] == "match", f"recovered {row['run']}: golden drift")
        require(row["status"] == "rejected" and row["reason"] == "timing_uncertain", f"recovered {row['run']}: unexpected verdict")
        recovered_rejected += 1
    require(recovered_rejected == 3, "recovered quiet series timing history was rewritten")
    stable = load_strict_json(HERE / nodes["quiet_series_r6"])
    require(stable["stable_fresh_timing"] is True and stable["performance_gate_passed"] is False, "stable timing became a performance pass")
    require(stable["summary_revision"] == 2 and (HERE / "evidence/real-metal-quiet-r6/summary-initial.json").is_file(), "initial r6 summary was not preserved")
    require(stable["publication_allowed"] is False and stable["done"] is False and stable["reviewer_done"] is False, "stable timing self-signed completion")
    for row in stable["runs"]:
        run_dir = HERE / "evidence/real-metal-quiet-r6" / row["run"]
        benchmark = load_strict_json(run_dir / "fresh-benchmark.json")
        golden = load_strict_json(run_dir / "fresh-golden.json")
        require(benchmark["status"] == row["status"] == "confirmed" and benchmark["reason"] is None, f"stable {row['run']}: verdict drift")
        require(benchmark["identity"]["kind"] == row["identity_kind"] == "match", f"stable {row['run']}: identity mismatch")
        require(benchmark["confirmation_attempts"] == row["confirmation_attempts"], f"stable {row['run']}: attempt drift")
        require(benchmark["confirmation_report"]["candidate_us"] == row["candidate_us"], f"stable {row['run']}: candidate time drift")
        require(row["latency_ratio"] == row["candidate_us"] / row["baseline_mlx_us"] and row["latency_ratio"] > 1.10, f"stable {row['run']}: latency gate drift")
        require(row["native_digest_match"] and row["artifact_set_match"] and row["compile_profile_match"], f"stable {row['run']}: build identity drift")
        require(golden["status"] == row["golden_status"] == "confirmed" and golden["identity"]["kind"] == "match", f"stable {row['run']}: golden drift")
    interval_one = load_strict_json(HERE / nodes["quiet_series_r7"])
    require(interval_one["stable_fresh_timing"] is False and interval_one["performance_gate_passed"] is False, "interval-one experiment became a pass")
    require([row["status"] for row in interval_one["runs"]] == ["confirmed", "confirmed", "rejected"], "r7 verdict history drift")
    require(all(row["latency_ratio"] > 1.10 for row in interval_one["runs"]), "r7 unexpectedly met the latency gate")
    for row in interval_one["runs"]:
        run_dir = HERE / "evidence/real-metal-quiet-r7" / row["run"]
        benchmark = load_strict_json(run_dir / "fresh-benchmark.json")
        golden = load_strict_json(run_dir / "fresh-golden.json")
        require(benchmark["status"] == row["status"] and benchmark["reason"] == row["reason"], f"r7 {row['run']}: verdict drift")
        require(benchmark["identity"]["kind"] == row["identity_kind"] == "match", f"r7 {row['run']}: identity mismatch")
        require(benchmark["confirmation_attempts"] == row["confirmation_attempts"], f"r7 {row['run']}: attempt drift")
        require(benchmark["confirmation_report"]["candidate_us"] == row["candidate_us"], f"r7 {row['run']}: candidate time drift")
        require(row["latency_ratio"] == row["candidate_us"] / row["baseline_mlx_us"], f"r7 {row['run']}: ratio drift")
        require(golden["status"] == row["golden_status"] == "confirmed" and golden["identity"]["kind"] == "match", f"r7 {row['run']}: golden drift")
    interval_two = load_strict_json(HERE / nodes["quiet_series_r8"])
    require(interval_two["stable_fresh_timing"] is True and interval_two["stable_fresh_performance"] is True, "interval-two fresh series did not pass")
    require(interval_two["performance_gate_passed"] is True and interval_two["publication_allowed"] is False, "r8 performance/publication scope drift")
    for row in interval_two["runs"]:
        run_dir = HERE / "evidence/real-metal-quiet-r8" / row["run"]
        benchmark = load_strict_json(run_dir / "fresh-benchmark.json")
        golden = load_strict_json(run_dir / "fresh-golden.json")
        require(benchmark["status"] == row["status"] == "confirmed" and benchmark["reason"] is None, f"r8 {row['run']}: verdict drift")
        require(benchmark["identity"]["kind"] == row["identity_kind"] == "match", f"r8 {row['run']}: identity mismatch")
        require(benchmark["confirmation_attempts"] == row["confirmation_attempts"], f"r8 {row['run']}: attempt drift")
        require(benchmark["confirmation_report"]["candidate_us"] == row["candidate_us"], f"r8 {row['run']}: candidate time drift")
        require(row["latency_ratio"] == row["candidate_us"] / row["baseline_mlx_us"] and row["latency_ratio"] <= 1.10, f"r8 {row['run']}: latency gate drift")
        require(row["native_digest_match"] and row["artifact_set_match"] and row["compile_profile_match"], f"r8 {row['run']}: build identity drift")
        require(golden["status"] == row["golden_status"] == "confirmed" and golden["identity"]["kind"] == "match", f"r8 {row['run']}: golden drift")
    frontend = load_strict_json(HERE / nodes["frontend_recovery"])
    require(frontend["status"] == "staged_and_verified" and frontend["source_only_double_build_passed"] is True, "frontend recovery was not verified")
    require(frontend["frontend"]["package_files"] == 3651 and frontend["record_distributions"] == 20, "frontend recovery identity drift")
    require(frontend["torch"]["version"] == "2.15.0" and frontend["original_full_tree_identity_claimed"] is False, "frontend recovery overclaims the original tree")
    require(frontend["original_staging_path_reused"] is False and frontend["staging_root"].endswith("-recovered"), "frontend recovery conflates the original MT0 staging path")
    failure = load_strict_json(HERE / "evidence/real-metal-quiet-r4/failure.json")
    require(failure["status"] == "backend_capability_unavailable" and failure["real_metal_benchmark_completed"] is False, "r4 environment failure was rewritten")
    total_runs = len(summary["runs"]) + len(quiet["runs"]) + len(latest["runs"]) + len(recovered["runs"]) + len(stable["runs"]) + len(interval_one["runs"]) + len(interval_two["runs"])
    total_golden = summary["golden_observations"] + len(quiet["runs"]) + len(latest["runs"]) + len(recovered["runs"]) + len(stable["runs"]) + len(interval_one["runs"]) + len(interval_two["runs"])
    require(identity["real_metal_observations"] == nodes["real_metal_observations"] == total_runs, "total real Metal observations drift")
    require(identity["golden_observations"] == nodes["golden_observations"] == total_golden, "total golden observations drift")
    require(identity["quiet_series_observations"] == nodes["quiet_series_observations"] == len(quiet["runs"]) + len(latest["runs"]) + len(recovered["runs"]) + len(stable["runs"]) + len(interval_one["runs"]) + len(interval_two["runs"]), "quiet series count drift")
    require(identity["stable_fresh_timing"] is True and nodes["stable_fresh_timing"] is True, "stable timing was not recorded")
    require(identity["fresh_confirmation_stable"] is True and identity["performance_gate_passed"] is True, "r8 performance pass was not recorded")
    require(nodes["gpu_executed"] is True and nodes["real_metal_gpu_passed"] == 2, "real Metal node record drift")
    analysis = load_strict_json(HERE / nodes["timing_analysis"])
    require(analysis["diagnostic_only"] is True and analysis["publication_allowed"] is False, "timing analysis became publication")
    require(analysis["quiet_cv_threshold"] == QUIET_CV == 0.05 and analysis["threshold_changed"] is False, "noise threshold drift")
    require(analysis["environment_ready_for_rerecord"] is True and identity["environment_ready_for_rerecord"] is True, "stable timing environment was not recorded")
    require(len(analysis["observations"]) == identity["real_metal_observations"], "timing analysis omitted an observation")
    drivers = {"candidate": 0, "mlx": 0, "torch_mps": 0}
    for row in analysis["observations"]:
        benchmark = load_strict_json(HERE / "evidence" / row["series"] / row["run"] / "fresh-benchmark.json")
        report = benchmark["confirmation_report"]["benchmark"]
        values = {"candidate": report["candidate"]["cv"], **{name: value["cv"] for name, value in report["baselines"].items()}}
        driver = max(values, key=values.get)
        require(row["noise_driver"] == driver, f"{row['series']}/{row['run']}: noise driver drift")
        drivers[driver] += 1
    require(drivers == nodes["noise_drivers"] == analysis["noise_drivers"], "noise driver summary drift")


def verify_track_explore(identity: dict, nodes: dict) -> None:
    from taic.track_explore import TrackExploreConfig, comparability_digest
    from taic.portfolio_confirmation import verify_subject_document
    from experiments.common import _manifest_sha256, _sha256_bytes

    root = HERE / "evidence/track-explore-r15"
    config = TrackExploreConfig.load(root / "track_explore.json")
    summary = load_strict_json(root / "summary.json")
    observations = load_strict_json(root / "observations.json")
    require(summary["config_digest"] == config.digest == observations["config_digest"], "track-explore config identity drift")
    require(summary["comparability_digest"] == comparability_digest(config.document), "track-explore comparability drift")
    require(summary["status"] == "exploration_winner" and summary["winner"]["track_id"] == "tilelang-pipeline", "track-explore winner drift")
    require(summary["winner"]["track_version"] == 3 and summary["winner"]["formal_publication"] is False, "track-explore became formal publication")
    require(summary["winner"]["same_fresh_input"] is False and summary["winner"]["cross_track_runtime_portfolio"] is False, "track-explore overclaims a shared input/runtime portfolio")
    require(identity["multi_track_batch_validated"] is True and nodes["multi_track_batch_validated"] is True, "track-explore validation was not recorded")
    require(identity["track_explore_reviewer_done"] is True and nodes["track_explore_reviewer_done"] is True, "track-explore independent review is missing")
    review = load_strict_json(HERE / identity["track_explore_review"]["document"])
    actor_request = HERE / identity["track_explore_review"]["actor_request"]
    require(review["done"] is True, "independent reviewer did not accept r15")
    require(hashlib.sha256(actor_request.read_bytes()).hexdigest() == identity["track_explore_review"]["actor_request_sha256"], "r15 actor review request identity drift")
    rejected = load_strict_json(HERE / identity["track_explore_previous_rejected"])
    require(rejected["done"] is False and rejected["reviewer_done"] is False, "r10 rejection was rewritten")
    require(len(rejected["blockers"]) == 3 and rejected["publication_allowed"] is False, "r10 blocker record drift")
    require(len(summary["tracks"]) == 3, "track-explore must include all three registered routes")
    for track in summary["tracks"]:
        require(track["eligible"] is True and len(track["instances"]) == 1, f"{track['track_id']}: not eligible")
        instance = track["instances"][0]
        require(instance["eligible"] is True and instance["confirmation_status"] == "confirmed", f"{track['track_id']}: not confirmed")
        require(instance["elimination_reasons"] == [] and instance["latency_ratio"] <= 1.10, f"{track['track_id']}: latency gate failed")
        run_dir = root / "runs" / track["track_id"] / "fp16__small__benchmark"
        run_config = load_strict_json(run_dir / "run_config.json")
        result = load_strict_json(run_dir / "result.json")
        confirmation = load_strict_json(run_dir / "confirmations/fp16_small_benchmark.json")
        require(verify_subject_document(confirmation)["status"] == "confirmed", f"{track['track_id']}: sidecar not confirmed")
        require(confirmation["identity"]["kind"] == "match", f"{track['track_id']}: identity mismatch")
        require(instance["candidate_sha256"] == confirmation["candidate_sha256"], f"{track['track_id']}: candidate drift")
        require(run_config["run_seed"] == 4 and result["confirmed_status"] == "ok", f"{track['track_id']}: run identity/verdict drift")
        require(run_config["track"]["track_id"] == track["track_id"] and run_config["track"]["track_version"] == track["track_version"], f"{track['track_id']}: track drift")
        actual = instance["actual_confirmation"]
        require(actual["seed_derivation_version"] == config.document["input_generator"]["seed_derivation_version"], f"{track['track_id']}: seed derivation drift")
        require(actual["fresh_input_identity"]["identity_scope"].startswith("fixed-shape:"), f"{track['track_id']}: fresh identity scope drift")
        if track["track_id"] == "tilelang-pipeline":
            best = run_dir / "best.taic"
            require(best.is_file(), "TileLang best.taic missing")
            require(_manifest_sha256(best.read_bytes()) == confirmation["candidate_sha256"], "TileLang manifest mismatch")
        else:
            best = run_dir / "best.metal"
            require(best.is_file(), f"{track['track_id']}: best.metal missing")
            require(_manifest_sha256(best.read_bytes()) == confirmation["candidate_sha256"], f"{track['track_id']}: manifest mismatch")
        require(result["confirmation_source_sha256"] == _sha256_bytes(best.read_bytes()), f"{track['track_id']}: transport hash mismatch")
        observation = observations["runs"][f"{track['track_id']}/fp16/small/benchmark"]
        require(observation["cache_dir"] == str((Path(instance["run_dir"]) / "cache").resolve()), f"{track['track_id']}: cache identity drift")
        require(observation["encoder_strategy"] == "pyobjc_fallback", f"{track['track_id']}: encode strategy drift")
        require(observation["native_batch_calls"] == 0, f"{track['track_id']}: native fallback was not observed")
        require(observation["command_buffers"] > 0, f"{track['track_id']}: command buffer evidence missing")
        execution = load_strict_json(run_dir / "observed-execution.json")
        require(execution and all(int(row["execution"].get("benchmark_protocol", {}).get("native_batch_calls", 0)) == 0 for row in execution), f"{track['track_id']}: native execution was not fallback")
        require(instance["search_cost"]["cost_complete"] is True, f"{track['track_id']}: incomplete cost")
        require(instance["search_cost"]["agent_turns"] is not None and instance["search_cost"]["agent_turns"] <= 1, f"{track['track_id']}: turn cap drift")
        require(instance["search_cost"]["build_seconds"] > 0, f"{track['track_id']}: build duration missing")
        require(observation["actual_spent_usd"] == 0.0 and observation["returncode"] == 0 and observation["timeout"] is False, f"{track['track_id']}: execution observation drift")
    require(summary["winner"]["geomean_latency_ratio"] <= 1.10, "track-explore winner exceeds the latency gate")


def verify() -> dict:
    identity = load_strict_json(HERE / "identity.json")
    nodes = load_strict_json(HERE / "nodes.json")
    require(identity["done"] is True and identity["reviewer_done"] is True, "MT6 completion or independent review is missing")
    scope = load_strict_json(HERE / "scope-decision.json")
    require(scope["scope"] == "msl_and_tilelang" and scope["additional_dsl_not_required"] is True, "third DSL was not formally rejected")
    require(scope["second_dsl_scope_decision_pending"] is False and scope["full_mt6_done"] is True and scope["reviewer_done"] is True, "scope review was not signed")
    scope_review = load_strict_json(HERE / "scope-review.json")
    require(scope_review["done"] is True, "independent scope review did not accept MSL + TileLang")
    require(scope["scope_review_log_sha256"] == "6fb693ef9b483c9415533dcc58b45a31201a0bcf3ef0ff749746f130ad82b4ae", "scope review log digest drift")
    review = load_strict_json(HERE / "review.json")
    actor_request = HERE / identity["review"]["actor_request"]
    require(review["done"] is True, "independent reviewer did not accept the /3 registration slice")
    require(hashlib.sha256(actor_request.read_bytes()).hexdigest() == identity["review"]["actor_request_sha256"], "actor review request identity drift")
    require(identity["registered"] is True and identity["selectable"] is True, "registered /3 is not selectable")
    require(identity["builder_profile_cpu"] is True and identity["structured_probe_cpu"] is True, "builder/probe CPU evidence missing")
    require(identity["artifact_audit_cpu"] is True and identity["real_metal_build_executed"] is True, "real Metal build evidence missing")
    require(identity["confirmation_route_cpu"] is True and identity["activation_route_cpu"] is True, "confirmation/activation route evidence missing")
    require(identity["v2_expected_route_cpu"] is True and identity["resume_route_cpu"] is True, "resume route evidence missing")
    require(identity["fresh_confirmation_executed"] is True, "real fresh execution was not recorded")
    require(identity["publication_allowed"] is False and identity["performance_publication"] is False, "CPU slice claims publication")
    require(tilelang.SUPPORTED_TRACK_VERSIONS == (1, 2), "/3 entered the registered/buildable version set")
    require(fresh.TILELANG_CONFIRMATION_TRACK_VERSION == 3, "MT6 contract version drifted")
    require(fingerprint(tilelang.frozen_contract(track_version=1)) == identity["v1_contract_digest"], "/1 contract drift")
    require(fingerprint(tilelang.frozen_contract(track_version=2)) == identity["v2_contract_digest"], "/2 contract drift")
    require(fingerprint(fresh.frozen_confirmation_contract()) == identity["contract_digest"], "/3 contract drift")
    registry = load_track_registry()
    require(fingerprint(registry.document()) == identity["registry_digest"], "registry drifted after the MT6 registration epoch")
    require(registry.versions("tilelang-pipeline") == (1, 2, 3), "registered TileLang version set drift")
    require(registry.require_available("tilelang-pipeline").track_version == 3, "default selection is not /3")
    epoch = load_strict_json(HERE / nodes["registry_epoch"])
    require(epoch["previous_registry_digest"] == identity["previous_registry_digest"], "previous registry digest drift")
    require(epoch["registry_digest"] == identity["registry_digest"], "registry epoch digest drift")
    require(epoch["previous_prompt_sha1"] == identity["previous_prompt_sha1"] and epoch["prompt_sha1"] == identity["prompt_sha1"], "prompt epoch record drift")
    require(current_prompt_sha1() == identity["prompt_sha1"], "prompt source digest drift after registration")
    require(epoch["closed_packages_rewritten"] is False and epoch["expected_mt2_mt4_verifier_epoch_drift"] is True, "closed-package drift was not classified")
    default = HERE / "evidence" / identity["default_registry_evidence"]
    evaluate = load_strict_json(default / "evaluate.json")
    require(evaluate["status"] == "ok" and evaluate["candidate_sha256"] == "bbe388556769de007eab9089a4d6e6a445c7115830d8bd41a5062b062fd57f45", "default /3 evaluate failed")
    require(any(row["kernel"] == "relu_kernel_kernel" for row in evaluate["dispatch"]), "default /3 dispatch missing")
    benchmark = load_strict_json(default / "confirmation-benchmark.json")
    golden = load_strict_json(default / "confirmation-golden.json")
    require(benchmark["status"] == golden["status"] == "confirmed", "default fresh confirmation failed")
    require(benchmark["identity"]["kind"] == golden["identity"]["kind"] == "match", "default fresh identity mismatch")
    ratio = benchmark["confirmation_report"]["candidate_us"] / benchmark["confirmation_report"]["baseline_us"]["mlx"]
    require(ratio == nodes["default_benchmark_latency_ratio"] and ratio <= 1.10, "default latency gate failed")
    activation = load_strict_json(default / "activation.json")
    require(activation["outcome"] == "resolved" and activation["provider"] == "candidate", "default activation failed")
    require(activation["candidate_sha256"] == evaluate["candidate_sha256"] and activation["fresh_subject"], "default activation identity drift")
    resume = load_strict_json(default / "resume.json")
    require(resume["non_dry_run"] is True and resume["model_calls"] == 0 and resume["source_matches"] is True, "default resume failed")
    require(resume["track"]["track_version"] == 3 and resume["state_json_exists"] is True, "default resume did not persist /3 state")
    require(len(nodes["mt6_cpu"]) == 3 and len(nodes["mt6_builder_cpu"]) == 5, "fixed MT6 CPU node set drift")
    require(len(nodes["mt6_confirmation_cpu"]) == 8 and len(nodes["mt6_resume_cpu"]) == 3, "fixed MT6 confirmation/resume nodes drift")
    require(nodes["compatibility"]["total_passed"] == 41 and nodes["focused_total_passed"] == 46, "MT6 compatibility count drift")
    require(nodes["extended_resume_compatibility_passed"] == 75, "extended resume compatibility count drift")
    require(nodes["msl_compatibility_passed"] == 148, "MSL compatibility count drift")
    require(nodes["model_executed"] is False, "MT6 claims an unrequested model run")
    verify_real_metal(identity, nodes)
    verify_track_explore(identity, nodes)
    require(nodes["done"] is True and nodes["reviewer_done"] is True, "node package review state drift")
    manifest = HERE / "manifest.sha256"
    if manifest.exists():
        for line in manifest.read_text(encoding="utf-8").splitlines():
            expected, name = line.split("  ", 1)
            require(hashlib.sha256((HERE / name).read_bytes()).hexdigest() == expected, f"manifest mismatch: {name}")
        listed = {line.split("  ", 1)[1] for line in manifest.read_text(encoding="utf-8").splitlines()}
        actual = {str(path.relative_to(HERE)) for path in HERE.rglob("*") if path.is_file() and path.name != "manifest.sha256" and path.suffix != ".pyc" and "__pycache__" not in path.parts}
        require(listed == actual, "MT6 manifest is not the closed package file set")
    return {
        "stage": "MT6",
        "status": identity["status"],
        "done": True,
        "reviewer_done": True,
        "registered": True,
        "selectable": True,
        "default_registry_validated": True,
        "fresh_confirmation_executed": True,
        "fresh_confirmation_stable": True,
        "performance_gate_passed": True,
        "track_explore": "exploration_winner",
        "track_explore_reviewer_done": True,
        "publication_allowed": False,
        "verifier_passed": True,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.parse_args()
    print(json.dumps(verify(), sort_keys=True))
