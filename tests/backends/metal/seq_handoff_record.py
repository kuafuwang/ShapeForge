"""Real-Metal recorder for P2-Q: SEQ arms, fresh confirmations, scratch attempts, decision."""

from __future__ import annotations

import argparse
import statistics

from taic.task_binding import bind_task
from tcenv.backends.metal.search_candidate import (
    confirm_candidate,
    evaluate_candidate,
    guidance_fingerprint,
    preflight_submission,
)
from tcenv.target import load_target
from tcenv.taskcontract.locator import REPO_ROOT
from experiments.seq_handoff_probe import run_probe
from tests.support import seq_handoff as sh

TARGET = load_target("apple-m4-max")
REPEATS = 3
REPS = 15


def master_key() -> bytes:
    from tests.support.acceptance_keys import P2H_MASTER_KEY
    return P2H_MASTER_KEY


def _locator(task: str, size: str):
    return bind_task(task, "fp16", size).locator


def _one_evaluation(task: str, size: str, source: str, *, mode: str = "full"):
    run = evaluate_candidate(
        _locator(task, size),
        source,
        target=TARGET,
        mode=mode,
        seed=7,
        reps=REPS,
        track_id="msl-bundle" if task == "kda" else "msl-single",
    )
    return run


def _host_wall_ns(run) -> dict:
    """Steady-state Host Program wall time from the harness's own benchmark phase."""
    evidence = run.evaluation.get("host_runs") or []
    bench = [item for item in evidence if item.get("evaluation_phase") == "benchmark"]
    correctness = [item for item in evidence if item.get("evaluation_phase") == "correctness"]
    if not bench:
        return {"benchmark_ns": None, "correctness_ns": [], "calls": 0}
    return {
        "benchmark_ns": int(bench[0]["host_tiling_wall_ns"]),
        "end_to_end_ns": None if bench[0].get("end_to_end_wall_ns") is None else int(bench[0]["end_to_end_wall_ns"]),
        "device_kernel_ns": None if bench[0].get("device_kernel_ns") is None else int(bench[0]["device_kernel_ns"]),
        "correctness_ns": [int(item["host_tiling_wall_ns"]) for item in correctness],
        "calls": len(evidence),
    }


def record_measurements() -> dict:
    """Three independent full evaluations per arm, on the search seed, same inputs everywhere."""
    measurements: dict[str, dict] = {}
    for size in sh.KDA_SIZES:
        measurements[f"kda:{size}"] = {}
        for arm in sh.KDA_ARMS:
            source = sh.kda_source(arm)
            runs = [_one_evaluation("kda", size, source) for _ in range(REPEATS)]
            reports = [run.result.report for run in runs]
            statuses = {report["status"] for report in reports}
            if statuses != {"ok"}:
                raise AssertionError(f"kda:{size}/{arm} did not pass every repeat: {statuses}")
            cases = reports[0]["correctness"]["cases"]
            measurements[f"kda:{size}"][arm] = {
                "status": "ok",
                "median_us": float(statistics.median(float(report["candidate_us"]) for report in reports)),
                "samples_us": [float(report["candidate_us"]) for report in reports],
                "cv": max(float(report["benchmark"]["candidate"]["cv"]) for report in reports),
                "repeats": REPEATS,
                "reps": REPS,
                "cases_ok": [case["case"] for case in cases if case["ok"]],
                "scaled_err": {case["case"]: case["scaled_err"] for case in cases},
                "safety_ok": all(report["safety"]["ok"] for report in reports),
                "nondeterminism": [
                    item for report in reports for item in (report.get("safety") or {}).get("nondeterminism", [])
                ],
                "dispatch": runs[0].evaluation["dispatch"],
                "workspace_bytes": runs[0].evaluation["workspace_bytes"],
                "source_sha256": sh.sha256_bytes(source.encode("utf-8")),
                "host": _host_wall_ns(runs[0]),
                "baseline_us": {name: one["median_us"] for name, one in reports[0]["benchmark"]["baselines"].items()},
                "speedup": reports[0]["speedup"],
                "benchmark_note": reports[0]["benchmark"]["protocol"],
            }
            print(
                f"OK: kda {size:6s} {arm:28s} {measurements[f'kda:{size}'][arm]['median_us']:12.3f} us "
                f"cv={measurements[f'kda:{size}'][arm]['cv']:.4f}",
                flush=True,
            )
    for size in sh.VARLEN_SIZES:
        measurements[f"varlen_kda:{size}"] = {}
        for arm in sh.VARLEN_ARMS:
            source = sh.varlen_source(arm)
            runs = [_one_evaluation("varlen_kda", size, source) for _ in range(REPEATS)]
            reports = [run.result.report for run in runs]
            statuses = {report["status"] for report in reports}
            if statuses != {"ok"}:
                raise AssertionError(f"varlen_kda:{size}/{arm} did not pass every repeat: {statuses}")
            cases = reports[0]["correctness"]["cases"]
            measurements[f"varlen_kda:{size}"][arm] = {
                "status": "ok",
                "median_us": float(statistics.median(float(report["candidate_us"]) for report in reports)),
                "samples_us": [float(report["candidate_us"]) for report in reports],
                "cv": max(float(report["benchmark"]["candidate"]["cv"]) for report in reports),
                "repeats": REPEATS,
                "reps": REPS,
                "cases_ok": [case["case"] for case in cases if case["ok"]],
                "scaled_err": {case["case"]: case["scaled_err"] for case in cases},
                "safety_ok": all(report["safety"]["ok"] for report in reports),
                "nondeterminism": [
                    item for report in reports for item in (report.get("safety") or {}).get("nondeterminism", [])
                ],
                "dispatch": runs[0].evaluation["dispatch"],
                "workspace_bytes": runs[0].evaluation["workspace_bytes"],
                "source_sha256": sh.sha256_bytes(source.encode("utf-8")),
                "host": _host_wall_ns(runs[0]),
                "baseline_us": {name: one["median_us"] for name, one in reports[0]["benchmark"]["baselines"].items()},
                "speedup": reports[0]["speedup"],
                "benchmark_note": reports[0]["benchmark"]["protocol"],
            }
            print(
                f"OK: varlen {size:6s} {arm:14s} {measurements[f'varlen_kda:{size}'][arm]['median_us']:12.3f} us "
                f"cv={measurements[f'varlen_kda:{size}'][arm]['cv']:.4f}",
                flush=True,
            )
    document = {
        "schema_id": sh.SCHEMA_MEASUREMENTS,
        "epoch": "p2-q-seq-state-handoff-1",
        "not_a_candidate": False,
        "publication_allowed": False,
        "npu_simulated_only": True,
        "search_seed": 7,
        "reps": REPS,
        "repeats": REPEATS,
        "source_digests": sh.source_digests(),
        "arms": {
            "kda": {
                "seq_three_piece": "the shipped cut: up to three pieces over three serial stages",
                "single_piece_three_stage": "one piece, the same three dispatched stages",
                "single_piece_single_stage": "one piece, one dispatched stage",
            },
            "varlen_kda": {
                "pieces_three": "the shipped cut: min(3, len) pieces over plan + three piece stages",
                "pieces_one": "one piece per chain, the same four dispatched stages",
            },
        },
        "measurements": measurements,
    }
    sh.dump_json(sh.MEASUREMENTS, document)
    return document


def record_confirmations() -> dict:
    """Fresh confirmation for every measured arm; the fresh seed is never the search seed."""
    subjects: dict[str, dict] = {}
    for size in ("small", "paper"):
        for arm in sh.KDA_ARMS:
            source = sh.kda_source(arm)
            subject = confirm_candidate(
                _locator("kda", size),
                source,
                target=TARGET,
                guidance_digest=guidance_fingerprint(sh.GUIDANCE, source),
                master_seed=master_key(),
                track_id="msl-bundle",
            )
            subjects[f"kda:{size}/{arm}"] = subject
            print(
                f"OK: kda {size} {arm} fresh {subject['status']} seed={subject['confirmation_seed']}",
                flush=True,
            )
    for arm in sh.VARLEN_ARMS:
        source = sh.varlen_source(arm)
        subject = confirm_candidate(
            _locator("varlen_kda", "mixed"),
            source,
            target=TARGET,
            guidance_digest=guidance_fingerprint(sh.GUIDANCE, source),
            master_seed=master_key(),
            track_id="msl-single",
        )
        subjects[f"varlen_kda:mixed/{arm}"] = subject
        print(f"OK: varlen mixed {arm} fresh {subject['status']} seed={subject['confirmation_seed']}", flush=True)
    document = {
        "schema_id": "shapeforge.seq-handoff-confirmations.1",
        "epoch": "p2-q-seq-state-handoff-1",
        "publication_allowed": False,
        "subjects": subjects,
        "source_digests": sh.source_digests(),
    }
    sh.dump_json(sh.CONFIRMATIONS, document)
    return document


def record_scratch_attempt() -> dict:
    """Both ways to put the handoff state on candidate scratch, recorded as they actually behave."""
    sources = sh.scratch_sources()
    contract = bind_task("kda", "fp16", "small").contract
    attempts: dict[str, dict] = {}
    poison = preflight_submission(contract, sources["uninitialized_poison"], target=TARGET, track_id="msl-bundle")
    attempts["uninitialized_poison"] = {
        "accepted": bool(poison.ok),
        "stage": poison.stage,
        "detail": str(poison.rejection or ""),
    }
    zero_source = sources["device_zero"]
    zero_preflight = preflight_submission(contract, zero_source, target=TARGET, track_id="msl-bundle")
    zero = {
        "accepted": bool(zero_preflight.ok),
        "preflight_stage": zero_preflight.stage,
        "preflight_detail": str(zero_preflight.rejection or ""),
    }
    if zero_preflight.ok:
        run = evaluate_candidate(
            _locator("kda", "small"), zero_source, target=TARGET, mode="quick", reps=3, track_id="msl-bundle"
        )
        report = run.result.report
        zero.update(
            {
                "evaluation_status": report["status"],
                "evaluation_message": str(report.get("message") or ""),
                "outcome": "ok" if report["status"] == "ok" else "blocked",
                "workspace_bytes": run.evaluation.get("workspace_bytes"),
            }
        )
    else:
        zero["outcome"] = "blocked"
    attempts["device_zero"] = zero
    # A two-buffer ping-pong is the one shape that *does* satisfy the lifecycle rule: each buffer's
    # first use is a write, and the state alternates a -> b -> final.  It passes preflight, so the
    # protocol can express a SEQ state; what stops it is the worker lifecycle identity below.
    pingpong_source = sh.pingpong_source()
    pingpong_preflight = preflight_submission(
        contract, pingpong_source, target=TARGET, track_id="msl-bundle"
    )
    algebra = sh.state_algebra()["kda:small"]
    pingpong = {
        "accepted_by_lifecycle": bool(pingpong_preflight.ok),
        "preflight_stage": pingpong_preflight.stage,
        "preflight_detail": str(pingpong_preflight.rejection or ""),
        "buffers": [spec["name"] for spec in sh.pingpong_geometry()["scratch"]],
        "init_policy": "uninitialized_poison",
        # Read off the geometry document, not hand-written: a wrong use list here once contradicted
        # the envelope it claimed to describe.
        "uses": sh.pingpong_uses(),
        # Two buffers of H*D*D fp32 each: the same total as the declared 2*H*D*D workspace capacity.
        "scratch_bytes": 2 * algebra["heads"] * algebra["state_bytes_per_chain"],
        "workspace_bytes_at_the_same_capacity": algebra["capacity_bytes"],
        "storage_penalty": False,
    }
    if pingpong_preflight.ok:
        run = evaluate_candidate(
            _locator("kda", "small"), pingpong_source, target=TARGET, mode="quick", reps=3, track_id="msl-bundle"
        )
        report = run.result.report
        pingpong.update(
            {
                "evaluation_status": report["status"],
                "evaluation_message": str(report.get("message") or ""),
            }
        )
    attempts["two_buffer_pingpong"] = pingpong

    document = {
        "schema_id": sh.SCHEMA_SCRATCH,
        "epoch": "p2-q-seq-state-handoff-1",
        "scratch_candidate_added": False,
        "reason": "no_measured_benefit_plus_bundle_scratch_blocker",
        "state_bytes_for_small": sh.state_algebra()["kda:small"]["capacity_bytes"],
        "unused_states_on_single_token": sh.state_algebra()["kda:single"]["unused_states"],
        "attempts": attempts,
        "publication_allowed": False,
    }
    sh.dump_json(sh.SCRATCH_ATTEMPT, document)
    print(f"OK: scratch poison refused: {attempts['uninitialized_poison']['detail'][:120]}", flush=True)
    print(f"OK: scratch device_zero outcome={attempts['device_zero']['outcome']}", flush=True)
    return document


def record_decision() -> dict:
    measurements = sh.load_json(sh.MEASUREMENTS)
    probe = sh.load_json(sh.PROBE)
    scratch = sh.load_json(sh.SCRATCH_ATTEMPT)
    sh.assert_measurements(measurements)
    sh.assert_scratch_attempt(scratch)
    document = sh.decision_document(measurements, probe, scratch)
    sh.assert_decision(document)
    sh.dump_json(sh.DECISION, document)
    for name, one in document["sizes"].items():
        print(
            f"OK: {name} SEQ {one['seq_three_piece_us']:.3f} us vs single stage {one['single_stage_us']:.3f} us "
            f"(+{100.0 * one['seq_overhead_fraction_of_single_stage']:.1f}%: handoff "
            f"{one['handoff_us']:.3f} us, stage floor {one['stage_floor_us']:.3f} us)",
            flush=True,
        )
    return document


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--measurements", action="store_true")
    parser.add_argument("--confirmations", action="store_true")
    parser.add_argument("--scratch", action="store_true")
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--decision", action="store_true")
    args = parser.parse_args()
    if not any((args.measurements, args.confirmations, args.scratch, args.probe, args.decision)):
        record_measurements()
        record_confirmations()
        record_scratch_attempt()
        run_probe(sh.PROBE)
        record_decision()
        return
    if args.measurements:
        record_measurements()
    if args.confirmations:
        record_confirmations()
    if args.scratch:
        record_scratch_attempt()
    if args.probe:
        run_probe(sh.PROBE)
    if args.decision:
        record_decision()


if __name__ == "__main__":
    main()
