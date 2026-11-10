"""Deterministic CPU-only fixtures for MT5; no model, subprocess search, or device work."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

from taic.orchestrator import confirmation_master_seed
from taic.policy import compare_confirmation_identity
from taic.portfolio_confirmation import _finalize_subject_document
from taic.track_explore import (
    COMPARISON_MODE,
    TrackExploreConfig,
    comparability_digest,
)
from tcenv.identity import MAX_SAFE_INTEGER, fingerprint
from tcenv.track.envelope import encode_envelope
from tcenv.track.plane import CandidateArtifact, CandidateEntry
from tcenv.track.registry import load_track_registry

from .shape_portfolio_metal import DEVICE_A, DEVICE_B, HOST_TILING_A, geometry_bytes, geometry_document

OP = "shape_portfolio_pointwise"
RUN_SEED = 123456789
PACK_DIGEST = fingerprint({"fixture": "shape-portfolio-pointwise-pack"})
GUIDANCE_DIGEST = fingerprint({"fixture": "guidance"})
PRESENTATION_DIGEST = fingerprint({"fixture": "presentation"})
REFERENCE_DIGEST = fingerprint({"fixture": "reference.py"})
EVAL_REPS = 15
ROOT = Path(__file__).resolve().parents[2]
ARCHIVE = ROOT / "tests/support/golden/shared/shape-portfolio-阶段验收/P2/evidence/unseen-confirmations.json"


def _templates() -> dict[str, dict]:
    documents = json.loads(ARCHIVE.read_text(encoding="utf-8"))["documents"]
    single = next(row for row in documents if row["candidate_sha256"].startswith("1209a6d886"))
    bundle = next(row for row in documents if row["candidate_sha256"].startswith("22c5880e60"))
    return {"msl-single": single, "msl-bundle": bundle}


def bundle_bytes() -> bytes:
    candidate = CandidateArtifact(
        track_id="msl-bundle",
        entries=(
            CandidateEntry(path="device/variant_a.metal", role="device_kernel", media_type="text/plain", data=DEVICE_A),
            CandidateEntry(path="host/host_tiling_a.json", role="host_tiling", media_type="application/json", data=HOST_TILING_A),
            CandidateEntry(
                path="host/execution_geometry.json", role="schedule", media_type="application/json",
                data=geometry_bytes(geometry_document("variant_a")),
            ),
        ),
    )
    assert candidate.candidate_sha256.startswith("22c5880e60")
    return encode_envelope(candidate).encode("utf-8")


def source_bytes(track_id: str) -> bytes:
    return DEVICE_B if track_id == "msl-single" else bundle_bytes()


def derive(track_id: str, instance_key: str, source: str, master_seed: bytes) -> dict:
    freeze = fingerprint({
        "fixture": "freeze", "track": track_id, "instance": instance_key,
        "source_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
        "master_sha256": hashlib.sha256(master_seed).hexdigest(),
    })
    validation = fingerprint({"fixture": "validation", "freeze": freeze})
    seed = int.from_bytes(hashlib.sha256(f"{validation}:{instance_key}".encode()).digest()[:8], "big") & MAX_SAFE_INTEGER
    if seed == 7:
        seed += 1
    return {
        "portfolio_freeze_digest": freeze,
        "validation_plan_digest": validation,
        "confirmation_seed": seed,
        "seed_derivation_version": "fixture.validation-plan.1",
    }


class _Adapter:
    def __init__(self, track_id: str):
        self.track_id = track_id

    def resolve(self, service: str):
        if service == "guidance":
            return lambda system, statement: fingerprint({"guidance": [system, statement]})
        if service != "confirmation":
            raise AssertionError(service)

        def derive_here(_locator, source: str, **kwargs):
            return derive(self.track_id, kwargs["instance_key"], source, kwargs["master_seed"])

        return derive_here


class FixedTrackExploreFixture:
    """A scriptable fake agent that writes complete, publishable v2 run fixtures."""

    def __init__(self, tmp_path: Path, *, instances: list[str] | None = None, tracks: list[str] | None = None):
        self.root = tmp_path
        self.instances = instances or ["above"]
        self.tracks = tracks or ["msl-single", "msl-bundle"]
        self.overrides: dict[tuple[str, str], dict] = {}
        self.calls = []
        self.config = self.make_config()
        self.templates = _templates()

    def make_config(self) -> TrackExploreConfig:
        template = _templates()["msl-single"]["confirmation_report"]
        participants = []
        for index, track_id in enumerate(self.tracks):
            track = load_track_registry().get(track_id)
            participants.append({
                "track_id": track_id,
                "track_version": track.track_version,
                "run_dir": f"runs/{track_id}",
                "build_timeout_sec": 30.0 + 30.0 * index,
                "generation": {"config": None, "preset": None, "model": None},
            })
        benchmarks = [
            {"dtype": "fp16", "size": size, "usage": "benchmark", "instance_key": f"fp16/{size}/benchmark"}
            for size in self.instances
        ]
        document = {
            "schema_id": "taic.track-explore.1",
            "task": OP,
            "target": "apple-m4-max",
            "participating_tracks": participants,
            "benchmarks": benchmarks,
            "run_seed": RUN_SEED,
            "master_seed": confirmation_master_seed(RUN_SEED).hex(),
            "input_generator": {
                "id": "fixture.opspec.gen_inputs",
                "version": "1",
                "seed_derivation_version": "fixture.validation-plan.1",
            },
            "comparability_digest": "0" * 64,
            "shared": {
                "generation": {"config": None, "preset": None, "model": None},
                "correctness": {"correctness_only": False},
                "baseline": {
                    "implementation_digest": template["baseline_implementation_digest"],
                    "environment_digest": template["baseline_environment_digest"],
                },
                "evaluation": {
                    "opspec_digest": template["build"]["opspec_digest"],
                    "reference_implementation_digest": REFERENCE_DIGEST,
                    "target_device": {
                        "target_id": "apple-m4-max",
                        "device_name": template["device_name"],
                        "gpu_family": template["gpu_family"],
                    },
                    "protocol_version": template["protocol_version"],
                    "timing_discipline": template["benchmark"]["protocol"],
                    "eval_reps": EVAL_REPS,
                },
                "confirmation": {
                    "mode": COMPARISON_MODE,
                    "quiet_required": True,
                    "confirm_batches": 1,
                    "search_reps": 3,
                    "confirmation_reps": 15,
                },
            },
            "generation_policy": "serial",
            "max_agent_turns": 3,
            "budget_scope": "per_track_instance",
            "budget_usd": 2.0,
            "max_total_search_wall_sec": 3600.0,
            "output": "out/track-explore-summary.json",
            "selection_policy_version": "track-explore-selection.1",
        }
        document["comparability_digest"] = comparability_digest(document)
        path = self.root / "track_explore.json"
        path.write_text(json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return TrackExploreConfig(document, path.resolve())

    def metadata(self, _task: str, _dtype: str, _size: str, _track_id: str) -> dict:
        option = self.overrides.get((_track_id, f"fp16/{_size}/benchmark"), {})
        return {
            "opspec_digest": self.config.document["shared"]["evaluation"]["opspec_digest"],
            "reference_implementation_digest": REFERENCE_DIGEST,
            "task_pack_digest": PACK_DIGEST,
            "locator": None,
            "variable_length_kind": option.get("variable_length_kind", "fixed"),
            "presentation_statement": option.get("presentation_statement", "fixture statement"),
        }

    def adapter_factory(self, track):
        return _Adapter(track.track_id)

    def options(self, track_id: str, instance_key: str) -> dict:
        default_ratio = {
            "msl-single": {"above": 0.90, "below": 0.95},
            "msl-bundle": {"above": 0.80, "below": 0.85},
        }[track_id].get(instance_key.split("/")[1], 0.90)
        row = {
            "ratio": default_ratio,
            "baseline_us": 100.0,
            "wall_sec": 10.0 + max(0, len(self.calls) - 1),
            "spent_usd": 0.25 if track_id == "msl-single" else 0.50,
            "agent_turns": 2,
        }
        row.update(self.overrides.get((track_id, instance_key), {}))
        return row

    def _subject(self, track_id: str, instance_key: str, option: dict) -> dict:
        document = copy.deepcopy(self.templates[track_id])
        size = instance_key.split("/")[1]
        usage = instance_key.split("/")[2]
        source = source_bytes(track_id).decode("utf-8")
        derived = derive(track_id, instance_key, source, bytes.fromhex(self.config.document["master_seed"]))
        confirmation_seed = option.get("confirmation_seed", derived["confirmation_seed"])
        ratio = float(option["ratio"])
        baseline = float(option["baseline_us"])
        candidate = baseline * ratio
        speedup = baseline / candidate
        evaluation_fingerprint = fingerprint({"fixture": "evaluation", "instance": instance_key})
        for period in ("search_report", "confirmation_report"):
            report = document[period]
            if period == "search_report" and option.get("search_cache_hit", False):
                report["cache_hit"] = True
            if period == "search_report" and option.get("primary_compile_error", False):
                report["status"] = "compile_error"
                report["eligible"] = False
                report["stage"] = "compile"
                report["compile"] = {"ok": False, "errors": "fixture compile error"}
                for key in ("placement", "artifacts", "evidence", "native", "native_digest", "native_absent_reason"):
                    report["build"].pop(key, None)
                report["build"]["provenance"]["artifact_set_digest"] = None
                report["build"].update({
                    "shape": "partial",
                    "error_class": "compile",
                    "duration_ms": 250,
                    "log": {"sha256": "0" * 64, "size_bytes": 0},
                })
            report["task"] = OP
            report["dtype"] = "fp16"
            report["size"] = size
            report["mode"] = "full" if usage == "benchmark" else "quick"
            report["evaluation_fingerprint"] = evaluation_fingerprint
            if option.get("different_baseline"):
                report["baseline_implementation_digest"] = "f" * 64
        report = document["confirmation_report"]
        report["seed"] = confirmation_seed
        report["candidate_us"] = candidate
        report["baseline_us"] = {"mlx": baseline}
        report["speedup"] = speedup
        report["speedups"] = {"mlx": speedup}
        report["benchmark"].update({
            "noisy": bool(option.get("timing_uncertain", False)),
            "attempts": 1,
            "noise_cv": 0.01,
            "best_baseline": "mlx",
            "baselines": {"mlx": {"cv": 0.01, "reps": 15}},
        })
        report["benchmark"]["candidate"].update({"cv": 0.01, "reps": 15})
        for index, case in enumerate(report["correctness"]["cases"]):
            case["seed"] = (confirmation_seed + 7919 * (index + 1)) & MAX_SAFE_INTEGER
            if option.get("variable_length_kind") in {"l1", "l2"} and not option.get("empty_input_identity", False):
                case["input_sha256"] = hashlib.sha256(
                    f"{instance_key}:{case['case']}:{case['seed']}".encode("utf-8")
                ).hexdigest()
                case["evaluation_input_digest"] = case["input_sha256"]
            else:
                # Real fixed-shape reports leave both fields empty. Do not invent a byte hash.
                case["input_sha256"] = ""
                case["evaluation_input_digest"] = None
        document["instance_key"] = instance_key
        document["usage"] = usage
        document["confirmation_seed"] = confirmation_seed
        document["confirmation_attempts"] = option.get("confirmation_attempts", 1)
        report["benchmark"]["attempts"] = option.get("benchmark_attempts", 1)
        document["expected"]["evaluation"]["evaluation_fingerprint"] = evaluation_fingerprint
        if option.get("different_baseline"):
            document["expected"]["evaluation"]["baseline_implementation_digest"] = "f" * 64
        document["identity"] = compare_confirmation_identity(
            candidate_sha256=document["candidate_sha256"],
            expected=document["expected"],
            search=document["search_report"],
            confirmation=document["confirmation_report"],
            scope="local",
        )
        return _finalize_subject_document(document)

    def _write_run(self, request, option: dict) -> None:
        actual_track = option.get("actual_track", request.track.track_id)
        instance_key = request.instance_key
        subject = self._subject(actual_track, instance_key, option)
        raw = source_bytes(actual_track)
        run_dir = request.run_dir
        (run_dir / "candidates" / "000_init").mkdir(parents=True, exist_ok=True)
        (run_dir / "confirmations").mkdir(exist_ok=True)
        (run_dir / "best.metal").write_bytes(raw)
        (run_dir / "candidates" / "000_init" / "kernel.metal").write_bytes(raw)
        (run_dir / "candidates" / "000_init" / "report.json").write_text(json.dumps(subject["search_report"]), encoding="utf-8")
        if option.get("extra_compile_error", False):
            failed = copy.deepcopy(subject["search_report"])
            failed["status"] = "compile_error"
            failed["eligible"] = False
            failed["stage"] = "compile"
            failed["compile"] = {"ok": False, "errors": "fixture compile error"}
            for key in ("placement", "artifacts", "evidence", "native", "native_digest", "native_absent_reason"):
                failed["build"].pop(key, None)
            failed["build"]["provenance"]["artifact_set_digest"] = None
            failed["build"].update({
                "shape": "partial",
                "error_class": "compile",
                "log": {"sha256": "0" * 64, "size_bytes": 0},
            })
            if option.get("compile_error_missing_duration", False):
                failed["build"].pop("duration_ms", None)
            else:
                failed["build"]["duration_ms"] = 250
            (run_dir / "candidates" / "001_bad").mkdir(exist_ok=True)
            (run_dir / "candidates" / "001_bad" / "report.json").write_text(json.dumps(failed), encoding="utf-8")
        (run_dir / "confirm_report.json").write_text(json.dumps(subject["confirmation_report"]), encoding="utf-8")
        (run_dir / "confirmations" / f"{instance_key.replace('/', '_')}.json").write_text(json.dumps(subject), encoding="utf-8")
        track = load_track_registry().get(actual_track)
        track_block = {"track_id": actual_track, "track_version": 1, "delivery": track.delivery}
        no_baseline = bool(option.get("no_baseline", False))
        run_config = {
            "schema_version": 3,
            "task_mode": "v2",
            "track": track_block,
            "task_pack_digest": PACK_DIGEST,
            "presentation_sha256": PRESENTATION_DIGEST,
            "confirmation_guidance_digest": GUIDANCE_DIGEST,
            "effective_correctness_only": no_baseline,
            "task": OP,
            "dtype": "fp16",
            "size": instance_key.split("/")[1],
            "run_seed": RUN_SEED,
            "target": {"id": "apple-m4-max"},
            "search": {
                "max_rounds": 2,
                "eval_reps": EVAL_REPS,
                "correctness_only": no_baseline,
                "confirm_batches": option.get("confirm_batches", 1),
            },
        }
        if option.get("legacy_guidance", False):
            run_config.pop("confirmation_guidance_digest", None)
            if not option.get("missing_legacy_system", False):
                (run_dir / "system_prompt.txt").write_text(
                    option.get("legacy_system", "fixture legacy system"), encoding="utf-8"
                )
        (run_dir / "run_config.json").write_text(json.dumps(run_config), encoding="utf-8")
        confirmed = subject["status"] == "confirmed" and not no_baseline
        report = subject["confirmation_report"]
        status = "ok" if confirmed else str(subject.get("reason") or "not_confirmed")
        status = option.get("result_status", status)
        speedup = report["speedup"]
        result = {
            "task_mode": "v2",
            "track": track_block,
            "task": OP,
            "dtype": "fp16",
            "size": instance_key.split("/")[1],
            "run_seed": RUN_SEED,
            "target_id": "apple-m4-max",
            "protocol_version": report["protocol_version"],
            "confirm_winner_label": "init",
            "confirmed_status": status,
            "confirmed_speedup": speedup if status == "ok" else None,
            "confirmed_values": [speedup] if status == "ok" else [],
            "candidate_us": report["candidate_us"] if status == "ok" else None,
            "baseline_us": report["baseline_us"] if status == "ok" else {},
            "confirmation_source_sha256": hashlib.sha256(raw).hexdigest(),
            "confirmations": [{
                "label": "init", "status": "ok" if confirmed else status,
                "failed_batches": 0, "identity": subject["identity"],
                "confirmation_seed": subject["confirmation_seed"],
            }],
            "model_totals": [{"sessions": option["agent_turns"]}],
            "spent_usd": option["spent_usd"],
            "best_speedup": speedup,
        }
        if no_baseline:
            (run_dir / "confirm_report.json").unlink()
            (run_dir / "confirmations" / f"{instance_key.replace('/', '_')}.json").unlink()
            result.update({
                "confirmed_status": None, "confirmed_speedup": None, "confirmed_values": [],
                "candidate_us": None, "baseline_us": {}, "best_speedup": None, "confirmations": [],
            })
        (run_dir / "result.json").write_text(json.dumps(result), encoding="utf-8")

    def __call__(self, request) -> dict:
        self.calls.append(request)
        option = self.options(request.track.track_id, request.instance_key)
        if not option.get("missing", False):
            self._write_run(request, option)
        return {
            "wall_sec": option["wall_sec"],
            "returncode": option.get("returncode", -9 if option.get("timeout") else 0),
            "timeout": bool(option.get("timeout", False)),
            "cache_dir": option.get("cache_dir", str(request.cache_dir.resolve())),
        }


__all__ = [
    "EVAL_REPS",
    "GUIDANCE_DIGEST",
    "OP",
    "PACK_DIGEST",
    "PRESENTATION_DIGEST",
    "REFERENCE_DIGEST",
    "RUN_SEED",
    "FixedTrackExploreFixture",
    "derive",
    "source_bytes",
]
