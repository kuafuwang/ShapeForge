"""Lightweight multi-Track exploration and deterministic result summarisation.

The outer loop only starts existing single-Track runs and reads their evidence.  It is
not a Host scheduler, a second evaluator, or a publication authority.  Scheme A keeps
run/master seed derivation and task instances common while each Track retains its own
fresh confirmation seed and input identity.  A summary is always exploration-only.

No function in this module is called by the CPU tests with the production subprocess
runner: tests inject a deterministic runner.  Invoking the real runner is equivalent to
``taic run`` and therefore requires the user's separate model/GPU/cost authorization.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from experiments.common import conclusion_verdict
from tcenv.identity import fingerprint, is_hex64, load_strict_json
from tcenv.track.registry import TrackSpec, load_track_registry

from .orchestrator import confirmation_master_seed
from .portfolio_confirmation import verify_subject_document

SCHEMA_ID = "taic.track-explore.1"
SUMMARY_SCHEMA_ID = "taic.track-explore-summary.1"
OBSERVATION_SCHEMA_ID = "taic.track-explore-observations.1"
SELECTION_POLICY_VERSION = "track-explore-selection.1"
COMPARISON_MODE = "scheme_A_independent_fresh_confirmation_same_distribution"
LATENCY_GATE = 1.10


class TrackExploreError(ValueError):
    """A versioned input, run layout, or summary identity is invalid."""


def _object(value: Any, keys: set[str], *, where: str) -> dict:
    if not isinstance(value, dict) or set(value) != keys:
        raise TrackExploreError(f"{where} must be an object with exactly {sorted(keys)}")
    return value


def _text(value: Any, *, where: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TrackExploreError(f"{where} must be a non-empty string")
    return value


def _digest(value: Any, *, where: str) -> str:
    if not is_hex64(value):
        raise TrackExploreError(f"{where} must be a lowercase sha256 digest")
    return value


def _integer(value: Any, *, where: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise TrackExploreError(f"{where} must be an integer >= {minimum}")
    return value


def _number(value: Any, *, where: str, positive: bool = True) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TrackExploreError(f"{where} must be a finite number")
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0) or (not positive and result < 0):
        raise TrackExploreError(f"{where} is outside its finite range")
    return result


def _generation(value: Any, *, where: str) -> dict:
    row = _object(value, {"config", "preset", "model"}, where=where)
    for key in row:
        if row[key] is not None:
            _text(row[key], where=f"{where}.{key}")
    return dict(row)


def validate_track_explore(document: Any, *, base_dir: str | Path | None = None) -> dict:
    """Validate and return one versioned ``track_explore.json`` document."""
    top = _object(
        document,
        {
            "schema_id", "task", "target", "participating_tracks", "benchmarks",
            "run_seed", "master_seed", "input_generator", "comparability_digest",
            "shared", "generation_policy", "max_agent_turns", "budget_scope",
            "budget_usd", "max_total_search_wall_sec", "output", "selection_policy_version",
        },
        where="track_explore",
    )
    if top["schema_id"] != SCHEMA_ID:
        raise TrackExploreError(f"schema_id must be {SCHEMA_ID}")
    _text(top["task"], where="task")
    _text(top["target"], where="target")
    if top["selection_policy_version"] != SELECTION_POLICY_VERSION:
        raise TrackExploreError("unsupported selection policy")
    if top["generation_policy"] != "serial":
        # Parallel generation needs a single-Track interface that separates generation from
        # quiet confirmation.  The current run interface does not expose that split, so v1
        # refuses it rather than racing full confirmations.
        raise TrackExploreError("only generation_policy='serial' is supported in schema 1")
    _integer(top["run_seed"], where="run_seed", minimum=0)
    if top["run_seed"] >= 2**64:
        raise TrackExploreError("run_seed must be an unsigned 64-bit integer")
    _digest(top["master_seed"], where="master_seed")
    if bytes.fromhex(top["master_seed"]) != confirmation_master_seed(top["run_seed"]):
        raise TrackExploreError("master_seed is not the existing taic-v2-search derivation of run_seed")
    _integer(top["max_agent_turns"], where="max_agent_turns", minimum=1)
    if top["budget_scope"] not in {"per_track_instance", "per_track"}:
        raise TrackExploreError("unknown budget_scope")
    _number(top["budget_usd"], where="budget_usd")
    _number(top["max_total_search_wall_sec"], where="max_total_search_wall_sec")
    _text(top["output"], where="output")

    generator = _object(
        top["input_generator"], {"id", "version", "seed_derivation_version"}, where="input_generator"
    )
    for key in generator:
        _text(generator[key], where=f"input_generator.{key}")

    benchmarks = top["benchmarks"]
    if not isinstance(benchmarks, list) or not benchmarks:
        raise TrackExploreError("benchmarks must be a non-empty array")
    instances: list[dict] = []
    seen: set[tuple[str, str, str]] = set()
    for index, raw in enumerate(benchmarks):
        if not isinstance(raw, dict) or set(raw) not in (
            {"dtype", "size", "usage"}, {"dtype", "size", "usage", "instance_key"}
        ):
            raise TrackExploreError(
                f"benchmarks[{index}] must contain dtype/size/usage and only an optional derived instance_key"
            )
        row = raw
        dtype, size, usage = (_text(row[key], where=f"benchmarks[{index}].{key}") for key in ("dtype", "size", "usage"))
        if usage not in {"benchmark", "golden"}:
            raise TrackExploreError("usage must be benchmark or golden")
        key = (dtype, size, usage)
        if key in seen:
            raise TrackExploreError("benchmark instances must be unique")
        seen.add(key)
        instance_key = "/".join(key)
        if row.get("instance_key", instance_key) != instance_key:
            raise TrackExploreError("a recorded instance_key does not match dtype/size/usage")
        instances.append({"dtype": dtype, "size": size, "usage": usage, "instance_key": instance_key})
    top["benchmarks"] = instances

    tracks = top["participating_tracks"]
    if not isinstance(tracks, list) or not tracks:
        raise TrackExploreError("participating_tracks must be a non-empty array")
    registry = load_track_registry()
    normalized: list[dict] = []
    track_ids: set[str] = set()
    run_roots: list[Path] = []
    for index, raw in enumerate(tracks):
        where = f"participating_tracks[{index}]"
        row = _object(
            raw, {"track_id", "track_version", "run_dir", "build_timeout_sec", "generation"}, where=where
        )
        track_id = _text(row["track_id"], where=f"{where}.track_id")
        if track_id in track_ids:
            raise TrackExploreError("a Track may participate only once per exploration")
        track_ids.add(track_id)
        version = _integer(row["track_version"], where=f"{where}.track_version", minimum=1)
        track = registry.get(track_id, version=version)
        track.require_available()
        raw_root = Path(_text(row["run_dir"], where=f"{where}.run_dir"))
        base = Path(base_dir) if base_dir is not None else Path.cwd()
        root = (base.resolve() / raw_root).resolve() if not raw_root.is_absolute() else raw_root.resolve()
        for other in run_roots:
            if root == other or root in other.parents or other in root.parents:
                raise TrackExploreError("independent, non-nested run_dir roots are required")
        run_roots.append(root)
        normalized.append({
            "track_id": track_id,
            "track_version": version,
            "run_dir": row["run_dir"],
            "build_timeout_sec": _number(row["build_timeout_sec"], where=f"{where}.build_timeout_sec"),
            "generation": _generation(row["generation"], where=f"{where}.generation"),
        })
    top["participating_tracks"] = normalized

    shared = _object(
        top["shared"], {"generation", "correctness", "baseline", "evaluation", "confirmation"}, where="shared"
    )
    shared["generation"] = _generation(shared["generation"], where="shared.generation")
    correctness = _object(shared["correctness"], {"correctness_only"}, where="shared.correctness")
    if not isinstance(correctness["correctness_only"], bool):
        raise TrackExploreError("correctness_only must be boolean")
    if shared["baseline"] is not None:
        baseline = _object(
            shared["baseline"], {"implementation_digest", "environment_digest"}, where="shared.baseline"
        )
        if baseline["implementation_digest"] is None and baseline["environment_digest"] is None:
            shared["baseline"] = None
        else:
            _digest(baseline["implementation_digest"], where="baseline.implementation_digest")
            _digest(baseline["environment_digest"], where="baseline.environment_digest")
    evaluation = _object(
        shared["evaluation"],
        {
            "opspec_digest", "reference_implementation_digest", "target_device",
            "protocol_version", "timing_discipline", "eval_reps",
        },
        where="shared.evaluation",
    )
    _digest(evaluation["opspec_digest"], where="evaluation.opspec_digest")
    _digest(evaluation["reference_implementation_digest"], where="evaluation.reference_implementation_digest")
    device = _object(
        evaluation["target_device"], {"target_id", "device_name", "gpu_family"}, where="evaluation.target_device"
    )
    for key in device:
        _text(device[key], where=f"evaluation.target_device.{key}")
    if device["target_id"] != top["target"]:
        raise TrackExploreError("target_device.target_id must equal target")
    _text(evaluation["protocol_version"], where="evaluation.protocol_version")
    _text(evaluation["timing_discipline"], where="evaluation.timing_discipline")
    _integer(evaluation["eval_reps"], where="evaluation.eval_reps", minimum=2)
    confirmation = _object(
        shared["confirmation"],
        {"mode", "quiet_required", "confirm_batches", "search_reps", "confirmation_reps"},
        where="shared.confirmation",
    )
    if confirmation["mode"] != COMPARISON_MODE or confirmation["quiet_required"] is not True:
        raise TrackExploreError("schema 1 requires scheme A and quiet existing confirmation")
    if confirmation["confirm_batches"] != 1:
        raise TrackExploreError("the existing v2 subject confirmation requires confirm_batches=1")
    _integer(confirmation["search_reps"], where="shared.confirmation.search_reps", minimum=2)
    _integer(confirmation["confirmation_reps"], where="shared.confirmation.confirmation_reps", minimum=2)
    if confirmation["search_reps"] != 3 or confirmation["confirmation_reps"] != 15:
        raise TrackExploreError("the existing v2 subject protocol fixes search_reps=3 and confirmation_reps=15")
    expected = comparability_digest(top)
    if top["comparability_digest"] != expected:
        raise TrackExploreError("comparability_digest does not match the common comparability tuple")
    return top


def _validated_config(config: "TrackExploreConfig") -> "TrackExploreConfig":
    return TrackExploreConfig(validate_track_explore(config.document, base_dir=config.path.parent), config.path)


@dataclass(frozen=True)
class TrackExploreConfig:
    document: dict
    path: Path

    @classmethod
    def load(cls, path: str | Path) -> "TrackExploreConfig":
        source = Path(path)
        document = load_strict_json(source)
        return cls(validate_track_explore(document, base_dir=source.resolve().parent), source.resolve())

    @property
    def digest(self) -> str:
        return fingerprint(self.document)

    def resolve(self, value: str) -> Path:
        path = Path(value)
        return path.resolve() if path.is_absolute() else (self.path.parent / path).resolve()

    @property
    def output_path(self) -> Path:
        return self.resolve(self.document["output"])

    @property
    def observation_path(self) -> Path:
        output = self.output_path
        return output.with_name(output.stem + "-observations.json")


def comparability_document(document: Mapping[str, Any], *, actual: Mapping[str, Any] | None = None) -> dict:
    """Return the one canonical common tuple used for cross-Track comparability."""
    shared = document["shared"]
    evaluation = shared["evaluation"]
    common = {
        "schema_id": "taic.track-explore-comparability.1",
        "opspec_digest": evaluation["opspec_digest"],
        "instances": [
            {"dtype": row["dtype"], "size": row["size"], "usage": row["usage"], "instance_key": row["instance_key"]}
            for row in sorted(document["benchmarks"], key=lambda row: row["instance_key"])
        ],
        "reference_implementation_digest": evaluation["reference_implementation_digest"],
        "target_device": dict(evaluation["target_device"]),
        "protocol_version": evaluation["protocol_version"],
        "timing_discipline": evaluation["timing_discipline"],
        "eval_reps": evaluation["eval_reps"],
        "input_generator": dict(document["input_generator"]),
    }
    if shared["baseline"] is not None:
        common["baseline"] = dict(shared["baseline"])
    if actual:
        common.update(actual)
    return common


def comparability_digest(document: Mapping[str, Any], *, actual: Mapping[str, Any] | None = None) -> str:
    return fingerprint(comparability_document(document, actual=actual))


def selection_policy_document() -> dict:
    return {
        "schema_id": SELECTION_POLICY_VERSION,
        "comparison_mode": COMPARISON_MODE,
        "latency_ratio": "candidate_us/baseline_us",
        "speedup": "baseline_us/candidate_us",
        "latency_gate": LATENCY_GATE,
        "additional_speedup_gt_one_gate": False,
        "single_instance": ["latency_ratio", "candidate_identity", "track_identity"],
        "multi_instance": "per-track per-instance best bundle",
        "multi_instance_sort": ["equal_weight_geomean", "worst_shard", "candidate_identity_vector", "track_identity"],
        "unnormalized_sum_sum_is_ranking": False,
        "cross_track_runtime_portfolio": False,
        "same_fresh_input_claim": False,
        "formal_publication": False,
    }


def selection_policy_digest() -> str:
    return fingerprint(selection_policy_document())


def _instance_dir(root: Path, instance_key: str) -> Path:
    return root / instance_key.replace("/", "__")


def _read_json(path: Path) -> Any:
    return load_strict_json(path)


def _best_source_path(run_dir: Path) -> Path:
    """Return the recorded candidate envelope for either MSL or a DSL Track."""
    taic = run_dir / "best.taic"
    if taic.is_file():
        return taic
    return run_dir / "best.metal"


def _write_json(path: Path, document: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(document, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    temporary.write_bytes(data)
    os.replace(temporary, path)


def _load_observations(config: TrackExploreConfig) -> dict:
    path = config.observation_path
    if not path.is_file():
        return {}
    document = _object(
        _read_json(path), {"schema_id", "config_digest", "runs"}, where="track-explore observations"
    )
    if document["schema_id"] != OBSERVATION_SCHEMA_ID or document["config_digest"] != config.digest:
        raise TrackExploreError("the observation file belongs to another config or schema")
    if not isinstance(document["runs"], dict):
        raise TrackExploreError("observation runs must be an object")
    for key, observation in document["runs"].items():
        if not isinstance(key, str) or not isinstance(observation, dict):
            raise TrackExploreError("each observation needs a string key and an object value")
        for field in ("wall_sec", "actual_spent_usd"):
            if observation.get(field) is not None:
                _number(observation[field], where=f"observations.{key}.{field}", positive=False)
        for field in ("budget_usd", "build_timeout_sec"):
            if observation.get(field) is not None:
                _number(observation[field], where=f"observations.{key}.{field}")
        if observation.get("returncode") is not None and type(observation["returncode"]) is not int:
            raise TrackExploreError("observation returncode must be an integer or null")
        if observation.get("timeout") is not None and not isinstance(observation["timeout"], bool):
            raise TrackExploreError("observation timeout must be boolean or null")
    return document["runs"]


def _record_observation(config: TrackExploreConfig, key: str, observation: Mapping[str, Any]) -> None:
    rows = _load_observations(config)
    rows[key] = dict(observation)
    _write_json(
        config.observation_path,
        {"schema_id": OBSERVATION_SCHEMA_ID, "config_digest": config.digest, "runs": rows},
    )


@dataclass(frozen=True)
class SingleTrackRequest:
    task: str
    target: str
    track: TrackSpec
    benchmark: dict
    run_dir: Path
    run_seed: int
    max_agent_turns: int
    build_timeout_sec: float
    generation: dict
    shared: dict
    budget_usd: float
    budget_scope: str
    max_wall_sec: float
    serial_confirmation: bool = True

    @property
    def instance_key(self) -> str:
        return self.benchmark["instance_key"]

    @property
    def cache_dir(self) -> Path:
        """The evaluation cache owned by this one track×instance run."""
        return self.run_dir / "cache"


class SubprocessSingleTrackRunner:
    """The production adapter to the existing ``taic run`` CLI (never used by CPU fixtures)."""

    def __init__(self, repo: str | Path | None = None) -> None:
        self.repo = Path(repo or Path(__file__).resolve().parent.parent).resolve()

    def argv(self, request: SingleTrackRequest) -> list[str]:
        """Build the existing single-Track CLI argv without starting a process."""
        generation = dict(request.shared["generation"])
        generation.update({key: value for key, value in request.generation.items() if value is not None})
        command = [
            "uv", "run", "python", "-m", "taic.cli", "run", request.task,
            "--dtype", request.benchmark["dtype"], "--size", request.benchmark["size"],
            "--target", request.target, "--track", request.track.track_id,
            "--run-dir", str(request.run_dir), "--seed", str(request.run_seed),
            "--budget", str(request.budget_usd),
            "--max-rounds", str(request.max_agent_turns - 1),
            "--eval-reps", str(request.shared["evaluation"]["eval_reps"]),
            "--confirm-batches", str(request.shared["confirmation"]["confirm_batches"]),
            "--correctness-only" if request.shared["correctness"]["correctness_only"] else "--no-correctness-only",
        ]
        for key in ("config", "preset", "model"):
            if generation.get(key) is not None:
                command.extend([f"--{key}", str(generation[key])])
        return command

    def environment(self, request: SingleTrackRequest) -> dict:
        """Return the per-run timeout and isolated evaluation-cache environment."""
        environment = dict(os.environ)
        environment["TAIC_TRACK_BUILD_TIMEOUT_SEC"] = str(request.build_timeout_sec)
        environment["TCENV_EVAL_CACHE_DIR"] = str(request.cache_dir.resolve())
        return environment

    def __call__(self, request: SingleTrackRequest) -> dict:
        started = time.monotonic()
        process = subprocess.Popen(self.argv(request), cwd=self.repo, env=self.environment(request), start_new_session=True)
        timed_out = False
        try:
            returncode = process.wait(timeout=request.max_wall_sec)
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(process.pid, signal.SIGKILL)
            returncode = process.wait()
        return {
            "wall_sec": time.monotonic() - started,
            "returncode": returncode,
            "timeout": timed_out,
            "argv": self.argv(request),
            "cache_dir": str(request.cache_dir.resolve()),
        }


def _default_adapter_factory(target_id: str) -> Callable[[TrackSpec], Any]:
    from tcenv.backends.registry import load_backend
    from tcenv.target import load_target

    backend = load_backend(load_target(target_id).backend)

    def factory(track: TrackSpec):
        return backend.track_adapter(track)

    return factory


def reference_implementation_digest(locator) -> str:
    """Hash the exact reference.py/reference.json payload in the discovered task pack."""
    files = []
    for name, data in locator.read_pack():
        if name in {"reference.py", "reference.json"}:
            files.append({"path": name, "sha256": hashlib.sha256(data).hexdigest()})
    if not any(row["path"] == "reference.py" for row in files):
        raise TrackExploreError("the task pack has no reference.py")
    return fingerprint({"schema_id": "taic.reference-implementation.1", "files": files})


def _default_task_metadata(task: str, dtype: str, size: str, track_id: str) -> dict:
    from dataclasses import replace

    from .task_binding import bind_task

    binding = replace(bind_task(task, dtype, size), track_id=track_id)
    return {
        "opspec_digest": binding.contract.spec.digest,
        "reference_implementation_digest": reference_implementation_digest(binding.locator),
        "task_pack_digest": binding.task_pack_digest,
        "locator": binding.locator,
        "variable_length_kind": binding.contract.instance.variable_length_kind,
        "presentation_statement": binding.presentation().statement,
    }


def _fresh_input_identity(
    report: Mapping[str, Any],
    confirmation_seed: int,
    *,
    task_pack_digest: str,
    instance_key: str,
    variable_length_kind: str | None,
) -> dict:
    correctness = report.get("correctness")
    cases = correctness.get("cases") if isinstance(correctness, dict) else None
    if not isinstance(cases, list) or not cases:
        raise TrackExploreError("fresh_input_identity_unknown")
    variable = variable_length_kind in {"l1", "l2"}
    if not is_hex64(task_pack_digest):
        raise TrackExploreError("fresh_input_identity_unknown")
    rows = []
    for case in cases:
        if not isinstance(case, dict) or not isinstance(case.get("case"), str) or type(case.get("seed")) is not int:
            raise TrackExploreError("fresh_input_identity_unknown")
        input_sha256 = case.get("input_sha256") or None
        evaluation_digest = case.get("evaluation_input_digest") or None
        if variable and not input_sha256 and not evaluation_digest:
            raise TrackExploreError("input_identity_unavailable")
        for value in (input_sha256, evaluation_digest):
            if value is not None and not is_hex64(value):
                raise TrackExploreError("input_identity_unavailable")
        row = {
            "case": case["case"],
            "seed": case["seed"],
            "input_sha256": input_sha256,
            "evaluation_input_digest": evaluation_digest,
        }
        if not variable:
            # Fixed-shape reports intentionally leave input_sha256 empty. Preserve that fact and
            # derive only the deterministic generator identity; never call it a byte hash.
            row["derived_input_identity"] = fingerprint({
                "task_pack_digest": task_pack_digest,
                "instance_key": instance_key,
                "case": case["case"],
                "seed": case["seed"],
            })
        rows.append(row)
    payload = {
        "schema_id": "taic.track-explore-fresh-input.1",
        "identity_scope": (
            "variable-length: ordered schema-4 case name/seed/input hashes; missing digests are fatal"
            if variable else
            "fixed-shape: schema-4 has no per-case byte digest; identity binds task pack, instance, case and seed, not a hash of input bytes"
        ),
        "task_pack_digest": task_pack_digest,
        "instance_key": instance_key,
        "variable_length_kind": variable_length_kind,
        "confirmation_seed": confirmation_seed,
        "cases": rows,
    }
    return payload | {"digest": fingerprint(payload)}


def _build_identity(report: Mapping[str, Any], run_config: Mapping[str, Any]) -> dict:
    from .policy import build_identity_view

    return build_identity_view(report) | {
        "task_pack_digest": run_config.get("task_pack_digest"),
        "track": run_config.get("track"),
    }


def _evaluation_identity(report: Mapping[str, Any], run_config: Mapping[str, Any]) -> dict:
    from .policy import evaluation_identity_view

    return evaluation_identity_view(report) | {
        "task_pack_digest": run_config.get("task_pack_digest"),
        "presentation_sha256": run_config.get("presentation_sha256"),
    }


def _compile_seconds(report: Any) -> float | None:
    if not isinstance(report, dict):
        return None
    # The complete build evidence is authoritative for DSL tracks: TileLang source-only
    # lowering dominates the tiny Metal compile/pipeline fields.  Prefer it over summaries.
    build = report.get("build")
    if isinstance(build, dict):
        evidence = build.get("evidence")
        if isinstance(evidence, dict) and isinstance(evidence.get("duration_ms"), (int, float)) and not isinstance(evidence.get("duration_ms"), bool):
            value = float(evidence["duration_ms"]) / 1000.0
            if math.isfinite(value) and value > 0:
                return value
        if isinstance(build.get("duration_ms"), (int, float)) and not isinstance(build.get("duration_ms"), bool):
            value = float(build["duration_ms"]) / 1000.0
            if math.isfinite(value) and value > 0:
                return value
    compile_block = report.get("compile")
    if isinstance(compile_block, dict) and isinstance(compile_block.get("duration_ms"), (int, float)) and not isinstance(compile_block.get("duration_ms"), bool):
        value = float(compile_block["duration_ms"]) / 1000.0
        return value if math.isfinite(value) and value >= 0 else None
    if isinstance(compile_block, dict):
        values = []
        complete_pair = True
        for key in ("compile_s", "pipeline_s"):
            value = compile_block.get(key)
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(float(value)) or value < 0:
                complete_pair = False
                break
            values.append(float(value))
        if complete_pair and values:
            return sum(values)
    return None


def _search_build_seconds(run_dir: Path, confirm_report: Any) -> tuple[dict, bool]:
    """Sum available build durations while retaining every failed-build record."""
    reports = []
    root = run_dir / "candidates"
    if root.is_dir():
        for path in sorted(root.glob("*/report.json")):
            try:
                reports.append((path.relative_to(run_dir).as_posix(), _read_json(path)))
            except Exception:
                return {"seconds": None, "reports": [], "failed_reports": [], "missing_duration_reports": []}, False
    if confirm_report is not None:
        reports.append((Path("confirm_report.json").as_posix(), confirm_report))
    rows = []
    complete = True
    for path, report in reports:
        seconds = _compile_seconds(report)
        status = report.get("status") if isinstance(report, dict) else None
        failed = status not in {None, "ok"}
        if seconds is None:
            # A failed candidate with no build duration is genuinely incomplete evidence, not
            # a reason to discard otherwise successful build costs.
            complete = False
        rows.append({
            "path": path,
            "status": status,
            "seconds": seconds,
            "failed": failed,
            "missing_duration": seconds is None,
        })
    known = [row["seconds"] for row in rows if row["seconds"] is not None]
    return {
        "seconds": sum(known) if known else (0.0 if complete else None),
        "reports": rows,
        "failed_reports": [row["path"] for row in rows if row["failed"]],
        "missing_duration_reports": [row["path"] for row in rows if row["missing_duration"]],
    }, complete


def _agent_turns(result: Mapping[str, Any]) -> int | None:
    """Count billable hmz session attempts, not inner provider/tool messages.

    ``event=agent`` is emitted once per session attempt. The count therefore includes the
    initial generation, subsequent max_rounds sessions, initial-generation restarts and any
    provider retry. ``max_rounds = max_agent_turns - 1`` is only a normal-path upper bound;
    retries/restarts can exceed it and are rejected by the post-run check below.
    """
    totals = result.get("model_totals")
    if not isinstance(totals, list):
        return None
    values = []
    for row in totals:
        if not isinstance(row, dict) or type(row.get("sessions")) is not int or row["sessions"] < 0:
            return None
        values.append(row["sessions"])
    return sum(values) if values else None


def _metric(report: Mapping[str, Any], verdict_speedup: Any) -> tuple[float, float, str]:
    candidate = report.get("candidate_us")
    baselines = report.get("baseline_us")
    benchmark = report.get("benchmark")
    best = benchmark.get("best_baseline") if isinstance(benchmark, dict) else None
    baseline = baselines.get(best) if isinstance(baselines, dict) else None
    candidate_value = _number(candidate, where="candidate_us")
    baseline_value = _number(baseline, where="baseline_us")
    speedup = _number(verdict_speedup, where="confirmed_speedup")
    ratio = candidate_value / baseline_value
    if not math.isclose(ratio, 1.0 / speedup, rel_tol=1e-6, abs_tol=1e-9):
        raise TrackExploreError("metric_identity_mismatch")
    if not math.isclose(float(report.get("speedup")), speedup, rel_tol=1e-9, abs_tol=1e-9):
        raise TrackExploreError("metric_identity_mismatch")
    return ratio, speedup, str(best)


def _publication_subject(run_dir: Path, confirm_report: Any, instance_key: str) -> dict:
    folder = run_dir / "confirmations"
    matches = []
    if folder.is_dir():
        for path in sorted(folder.glob("*.json")):
            document = verify_subject_document(_read_json(path))
            if (
                document.get("status") == "confirmed"
                and document.get("instance_key") == instance_key
                and document.get("confirmation_report") == confirm_report
            ):
                matches.append(document)
    if len(matches) != 1:
        raise TrackExploreError("not_confirmed")
    subject = matches[0]
    if subject.get("provider") != "candidate" or subject.get("credits_variant") is not True:
        raise TrackExploreError("fallback_credit")
    return subject


def _bundle_vector(rows: list[dict]) -> list[dict]:
    return [
        {
            "instance_key": row["instance_key"],
            "candidate_sha256": row["candidate_sha256"],
            "build_identity": row["build_identity"],
            "evaluation_identity": row["evaluation_identity"],
        }
        for row in sorted(rows, key=lambda row: row["instance_key"])
    ]


def exploration_bundle_digest(track: TrackSpec, vector: list[dict]) -> str:
    """Bind a per-instance candidate vector to the Track that owns the whole bundle."""
    return fingerprint({
        "track_id": track.track_id,
        "track_version": track.track_version,
        "per_instance_candidate_vector": vector,
    })


def _search_cache_evidence(run_dir: Path, observation: Mapping[str, Any]) -> tuple[dict, list[str]]:
    """Bind search reports to this run's isolated evaluation cache and expose every hit."""
    expected = (run_dir / "cache").resolve()
    recorded = observation.get("cache_dir")
    if not isinstance(recorded, str) or not recorded:
        raise TrackExploreError("cache_identity_unavailable")
    if Path(recorded).resolve() != expected:
        raise TrackExploreError("cache_reuse")
    reports = []
    for path in sorted((run_dir / "candidates").glob("*/report.json")):
        if path.is_symlink():
            raise TrackExploreError("symlinked_evidence")
        document = _read_json(path)
        cache_hit = document.get("cache_hit")
        if not isinstance(cache_hit, bool):
            raise TrackExploreError("cache_hit_unknown")
        reports.append({
            "path": path.relative_to(run_dir).as_posix(),
            "cache_hit": cache_hit,
            "candidate_sha256": (document.get("build") or {}).get("candidate_sha256"),
            "seed": document.get("seed"),
        })
    if not reports:
        raise TrackExploreError("cache_identity_unavailable")
    hits = [row["path"] for row in reports if row["cache_hit"]]
    identity = {
        "eval_cache_dir": str(expected),
        "expected_eval_cache_dir": str(expected),
        "independent": True,
        "reports": reports,
        "hit_count": len(hits),
        "miss_count": len(reports) - len(hits),
        "hits": hits,
        "confirmation_cache_hit": (
            _read_json(run_dir / "confirm_report.json").get("cache_hit")
            if (run_dir / "confirm_report.json").is_file() else None
        ),
    }
    return identity, (["cache_hit"] if hits else [])


def _attach_run_record(
    config: TrackExploreConfig,
    row: dict,
    run_dir: Path,
    result: Mapping[str, Any],
    run_config: Mapping[str, Any],
    participant: Mapping[str, Any],
    observation: Mapping[str, Any],
) -> None:
    """Attach cost and identity before any eligibility branch can return."""
    row["run_identity"] = {
        "run_config_sha256": hashlib.sha256((run_dir / "run_config.json").read_bytes()).hexdigest(),
        "run_seed": run_config.get("run_seed"),
        "task_pack_digest": run_config.get("task_pack_digest"),
        "presentation_sha256": run_config.get("presentation_sha256"),
        "confirmation_guidance_digest": run_config.get("confirmation_guidance_digest"),
        "track": run_config.get("track"),
    }
    confirm_path = run_dir / "confirm_report.json"
    confirm_report = _read_json(confirm_path) if confirm_path.is_file() else None
    search_report = confirm_report
    if search_report is None:
        for path in sorted((run_dir / "candidates").glob("*/report.json")):
            try:
                search_report = _read_json(path)
                break
            except Exception:
                continue
    if isinstance(search_report, dict) and isinstance(search_report.get("build"), dict):
        row["candidate_sha256"] = search_report["build"].get("candidate_sha256")
        row["build_identity"] = _build_identity(search_report, run_config)
        row["evaluation_identity"] = _evaluation_identity(search_report, run_config)
    row["confirmation_status"] = result.get("confirmed_status") or "not_confirmed"
    build_cost, build_complete = _search_build_seconds(run_dir, confirm_report)
    turns = _agent_turns(result)
    spent = result.get("spent_usd")
    wall = observation.get("wall_sec")
    cost_complete = bool(
        build_complete and turns is not None
        and isinstance(wall, (int, float)) and not isinstance(wall, bool)
        and math.isfinite(float(wall)) and wall >= 0
        and isinstance(spent, (int, float)) and not isinstance(spent, bool)
        and math.isfinite(float(spent)) and spent >= 0
    )
    row["search_cost"] = {
        "agent_turns": turns,
        "build_seconds": build_cost["seconds"],
        "build_cost_details": build_cost,
        "build_time_basis": "sum available schema-4 compile/build durations over candidate reports and the independent confirmation build; failed builds remain listed",
        "total_search_wall_sec": wall,
        "spent_usd": spent,
        "cost_complete": cost_complete,
        "build_timeout_sec": participant["build_timeout_sec"],
        "budget_scope": config.document["budget_scope"],
        "budget_usd": observation.get("budget_usd", config.document["budget_usd"]),
        "max_agent_turns": config.document["max_agent_turns"],
        "max_search_rounds": config.document["max_agent_turns"] - 1,
        "agent_turns_semantics": "one per event=agent session attempt; includes initial, rounds, restarts and provider retries, not inner tool messages",
        "generation": participant["generation"],
        "returncode": observation.get("returncode"),
        "timeout": observation.get("timeout"),
    }


def _validate_confirmation_attempts(subject: Mapping[str, Any], confirm_report: Mapping[str, Any], confirm_batches: int) -> None:
    """Validate confirmation retry attempts without conflating them with independent batches."""
    from tcenv.bench import BENCH_ATTEMPTS

    attempts = subject.get("confirmation_attempts")
    report_attempts = (confirm_report.get("benchmark") or {}).get("attempts")
    if type(attempts) is not int or type(report_attempts) is not int:
        raise TrackExploreError("confirmation_batch_mismatch")
    if not 1 <= attempts <= BENCH_ATTEMPTS or not 1 <= report_attempts <= BENCH_ATTEMPTS:
        raise TrackExploreError("confirmation_batch_mismatch")
    if confirm_batches != 1:
        raise TrackExploreError("confirmation_batch_mismatch")


def _run_row(
    config: TrackExploreConfig,
    participant: Mapping[str, Any],
    benchmark: Mapping[str, Any],
    track: TrackSpec,
    adapter: Any,
    metadata_provider: Callable,
    observation: Mapping[str, Any],
    evidence_registry: dict,
    expected_budget_usd: float,
) -> dict:
    run_dir = _instance_dir(config.resolve(participant["run_dir"]), benchmark["instance_key"])
    instance_key = benchmark["instance_key"]
    row = {
        "instance_key": instance_key,
        "track_id": track.track_id,
        "track_version": track.track_version,
        "run_dir": str(run_dir),
        "ranked": benchmark["usage"] == "benchmark",
        "eligible": False,
        "elimination_reasons": [],
        "observation": dict(observation),
    }
    reasons: list[str] = row["elimination_reasons"]
    if observation.get("timeout") is True:
        reasons.append("runner_timeout")
    if observation.get("returncode") not in (None, 0):
        reasons.append("runner_failed")
    if observation and observation.get("budget_usd") is not None and not math.isclose(
        float(observation["budget_usd"]), expected_budget_usd, rel_tol=0.0, abs_tol=1e-9
    ):
        reasons.append("generation_config_mismatch")
    result_path = run_dir / "result.json"
    config_path = run_dir / "run_config.json"
    confirm_path = run_dir / "confirm_report.json"
    if not result_path.is_file() or not config_path.is_file():
        reasons.append("no_baseline" if observation.get("skipped_reason") == "no_baseline" else "missing_result")
        return row
    try:
        result = _read_json(result_path)
        run_config = _read_json(config_path)
        if not isinstance(result, dict) or not isinstance(run_config, dict):
            raise TrackExploreError("unreadable_evidence")
        _attach_run_record(config, row, run_dir, result, run_config, participant, observation)
        if benchmark["usage"] != "benchmark":
            row["display_note"] = "correctness-only exploration is reported but never creates a performance winner"
            try:
                row["cache_identity"], _ = _search_cache_evidence(run_dir, observation)
            except TrackExploreError as exc:
                row["cache_identity"] = {"error": str(exc)}
            return row
        evidence_files = [result_path, config_path, confirm_path, _best_source_path(run_dir)]
        candidate_root = run_dir / "candidates"
        if candidate_root.is_dir():
            evidence_files.extend(sorted(candidate_root.glob("*/kernel.metal")))
            evidence_files.extend(sorted(candidate_root.glob("*/kernel.taic")))
        confirmation_root = run_dir / "confirmations"
        if confirmation_root.is_dir():
            evidence_files.extend(sorted(confirmation_root.glob("*.json")))
        for path in evidence_files:
            if path.is_symlink():
                raise TrackExploreError("symlinked_evidence")
            if not path.is_file():
                continue
            stat = path.stat()
            identity = (stat.st_dev, stat.st_ino)
            previous = evidence_registry.get(identity)
            if previous is not None and previous != str(run_dir):
                reasons.append("cache_reuse")
                return row
            evidence_registry[identity] = str(run_dir)
        if run_dir.is_symlink():
            raise TrackExploreError("symlinked_evidence")
    except Exception:
        reasons.append("unreadable_evidence")
        return row
    expected_track = {
        "track_id": track.track_id, "track_version": track.track_version, "delivery": track.delivery,
    }
    if run_config.get("track") != expected_track or result.get("track") != expected_track:
        reasons.append("track_mismatch")
    for key, expected in (("task", config.document["task"]), ("dtype", benchmark["dtype"]), ("size", benchmark["size"]), ("run_seed", config.document["run_seed"])):
        if run_config.get(key) != expected or result.get(key) != expected:
            reasons.append("run_identity_mismatch")
    if run_config.get("task_mode") != "v2" or result.get("task_mode") != "v2" or result.get("target_id") != config.document["target"]:
        reasons.append("run_identity_mismatch")
    if (run_config.get("target") or {}).get("id") != config.document["target"]:
        reasons.append("run_identity_mismatch")
    search = run_config.get("search") if isinstance(run_config.get("search"), dict) else {}
    if search.get("max_rounds") != config.document["max_agent_turns"] - 1:
        reasons.append("generation_config_mismatch")
    if (
        run_config.get("effective_correctness_only") is not True
        and search.get("correctness_only") != config.document["shared"]["correctness"]["correctness_only"]
    ):
        reasons.append("generation_config_mismatch")
    if search.get("eval_reps") != config.document["shared"]["evaluation"]["eval_reps"]:
        reasons.append("generation_config_mismatch")
    if search.get("confirm_batches") != config.document["shared"]["confirmation"]["confirm_batches"]:
        reasons.append("generation_config_mismatch")
    requested_model = participant["generation"].get("model") or config.document["shared"]["generation"].get("model")
    if requested_model is not None and (run_config.get("llm") or {}).get("model") != requested_model:
        reasons.append("generation_config_mismatch")
    if observation and observation.get("build_timeout_sec") != participant["build_timeout_sec"]:
        reasons.append("generation_config_mismatch")
    try:
        cache_identity, cache_reasons = _search_cache_evidence(run_dir, observation)
        row["cache_identity"] = cache_identity
        reasons.extend(cache_reasons)
    except TrackExploreError as exc:
        reasons.append(str(exc))
    except Exception:
        reasons.append("cache_identity_unavailable")
    if reasons:
        return row
    try:
        verdict, publication_reason = conclusion_verdict(run_dir, result)
    except Exception:
        verdict, publication_reason = {"speedup": None, "publication": "未确认"}, "not_confirmed"
    row["publication"] = verdict["publication"]
    row["publication_reason"] = publication_reason
    if publication_reason != "confirmed" or verdict["publication"] != "confirmed":
        status = result.get("confirmed_status")
        if status in {"incorrect", "unsafe"}:
            reasons.append(str(status))
        elif run_config.get("effective_correctness_only") is True:
            reasons.append("no_baseline")
        else:
            reasons.append(str(publication_reason or "not_confirmed"))
    if reasons:
        return row
    try:
        confirm_report = _read_json(confirm_path)
        subject = _publication_subject(run_dir, confirm_report, instance_key)
        confirmation_config = config.document["shared"]["confirmation"]
        for period, expected_reps in (("search_report", confirmation_config["search_reps"]), ("confirmation_report", confirmation_config["confirmation_reps"])):
            benchmark_stats = (subject[period].get("benchmark") or {}).get("candidate") or {}
            if benchmark_stats.get("reps") != expected_reps:
                raise TrackExploreError("generation_config_mismatch")
        _validate_confirmation_attempts(subject, confirm_report, confirmation_config["confirm_batches"])
        ratio, speedup, baseline_name = _metric(confirm_report, verdict["speedup"])
        cache_identity["confirmation_cache_hit"] = confirm_report.get("cache_hit")
        metadata = metadata_provider(config.document["task"], benchmark["dtype"], benchmark["size"], track.track_id)
        fresh = _fresh_input_identity(
            confirm_report,
            subject["confirmation_seed"],
            task_pack_digest=run_config.get("task_pack_digest"),
            instance_key=instance_key,
            variable_length_kind=metadata.get("variable_length_kind"),
        )
    except TrackExploreError as exc:
        reasons.append(str(exc))
        return row
    except Exception:
        reasons.append("unreadable_evidence")
        return row
    row.update({
        "candidate_sha256": subject["candidate_sha256"],
        "build_identity": _build_identity(confirm_report, run_config),
        "evaluation_identity": _evaluation_identity(confirm_report, run_config),
        "confirmation_digest": subject["confirmation_digest"],
        "latency_ratio_kind": "single-candidate confirmed candidate/best-registered-baseline; not a P2 multi-shard portfolio account",
        "latency_ratio": ratio,
        "confirmed_speedup": speedup,
        "baseline": baseline_name,
        "candidate_us": confirm_report["candidate_us"],
        "baseline_us": confirm_report["baseline_us"][baseline_name],
        "actual_confirmation": {
            "confirmation_seed": subject["confirmation_seed"],
            "search_seed": subject["search_seed"],
            "coverage_kind": subject["coverage_kind"],
            "fresh_input_identity": fresh,
        },
    })
    try:
        if run_config.get("task_pack_digest") != metadata.get("task_pack_digest"):
            raise TrackExploreError("task_pack_mismatch")
        actual = {
            "opspec_digest": metadata["opspec_digest"],
            "reference_implementation_digest": metadata["reference_implementation_digest"],
            "target_device": {
                "target_id": confirm_report.get("target_id"),
                "device_name": confirm_report.get("device_name"),
                "gpu_family": confirm_report.get("gpu_family"),
            },
            "protocol_version": confirm_report.get("protocol_version"),
            "timing_discipline": (confirm_report.get("benchmark") or {}).get("protocol"),
            "eval_reps": search.get("eval_reps"),
        }
        if config.document["shared"]["baseline"] is not None:
            actual["baseline"] = {
                "implementation_digest": confirm_report.get("baseline_implementation_digest"),
                "environment_digest": confirm_report.get("baseline_environment_digest"),
            }
        actual_digest = comparability_digest(config.document, actual=actual)
        row["comparability_digest"] = actual_digest
        if actual_digest != config.document["comparability_digest"]:
            reasons.append("comparability_mismatch")
        expected_common = comparability_document(config.document)
        actual_common = comparability_document(config.document, actual=actual)
        row["comparability_differences"] = sorted(
            key for key in expected_common if expected_common[key] != actual_common[key]
        )
    except Exception as exc:
        row["comparability_digest"] = None
        reasons.append("comparability_unavailable" if isinstance(exc, TrackExploreError) else "task_metadata_unavailable")
    try:
        guidance_digest = run_config.get("confirmation_guidance_digest")
        if guidance_digest is None:
            system_path = run_dir / "system_prompt.txt"
            if not system_path.is_file() or system_path.is_symlink():
                raise TrackExploreError("confirmation_guidance_unavailable")
            statement = metadata.get("presentation_statement")
            if not isinstance(statement, str):
                raise TrackExploreError("confirmation_guidance_unavailable")
            guidance_digest = adapter.resolve("guidance")(system_path.read_text(encoding="utf-8"), statement)
            row["run_identity"]["confirmation_guidance_digest"] = guidance_digest
        deriver = adapter.resolve("confirmation")
        locator = metadata.get("locator")
        source = _best_source_path(run_dir).read_bytes().decode("utf-8")
        from tcenv.target import load_target

        derived = deriver(
            locator,
            source,
            target=load_target(config.document["target"]),
            guidance_digest=guidance_digest,
            master_seed=bytes.fromhex(config.document["master_seed"]),
            instance_key=instance_key,
        )
        row["actual_confirmation"].update({
            "portfolio_freeze_digest": derived["portfolio_freeze_digest"],
            "validation_plan_digest": derived["validation_plan_digest"],
            "seed_derivation_version": derived["seed_derivation_version"],
        })
        if (
            derived["confirmation_seed"] != subject["confirmation_seed"]
            or derived["seed_derivation_version"] != config.document["input_generator"]["seed_derivation_version"]
        ):
            reasons.append("seed_derivation_mismatch")
    except TrackExploreError as exc:
        reasons.append(str(exc))
    except Exception:
        reasons.append("seed_derivation_unavailable")
    row["confirmation_status"] = "confirmed"
    turns = row["search_cost"]["agent_turns"]
    if turns is None or turns > config.document["max_agent_turns"]:
        reasons.append("generation_config_mismatch")
    if ratio > LATENCY_GATE:
        reasons.append("latency_gate")
    row["elimination_reasons"] = sorted(set(reasons))
    row["eligible"] = not row["elimination_reasons"]
    return row


def summarize_track_explore(
    config: TrackExploreConfig,
    *,
    adapter_factory: Callable[[TrackSpec], Any] | None = None,
    metadata_provider: Callable | None = None,
) -> dict:
    """Read independent run evidence and deterministically select an exploration winner."""
    config = _validated_config(config)
    registry = load_track_registry()
    adapter_factory = adapter_factory or _default_adapter_factory(config.document["target"])
    metadata_provider = metadata_provider or _default_task_metadata
    observations = _load_observations(config)
    evidence_registry: dict = {}
    track_rows = []
    all_rows = []
    spent_by_track: dict[str, float] = {}
    for participant in config.document["participating_tracks"]:
        track = registry.get(participant["track_id"], version=participant["track_version"])
        adapter = adapter_factory(track)
        spent_by_track[track.track_id] = 0.0
        rows = []
        for benchmark in config.document["benchmarks"]:
            key = f"{track.track_id}/{benchmark['instance_key']}"
            row = _run_row(
                config, participant, benchmark, track, adapter, metadata_provider,
                observations.get(key, {}), evidence_registry,
                config.document["budget_usd"] if config.document["budget_scope"] == "per_track_instance" else max(
                    0.0, config.document["budget_usd"] - spent_by_track[track.track_id]
                ),
            )
            rows.append(row)
            spent = row["observation"].get("actual_spent_usd")
            if isinstance(spent, (int, float)) and not isinstance(spent, bool):
                spent_by_track[track.track_id] += float(spent)
        ranked = [row for row in rows if row["ranked"]]
        all_rows.extend(rows)
        missing = sorted({row["instance_key"] for row in ranked if row["elimination_reasons"] == ["missing_result"]})
        valid = [row for row in ranked if row["eligible"]]
        complete = len(valid) == len(ranked) and bool(ranked)
        track_result = {
            "track_id": track.track_id,
            "track_version": track.track_version,
            "eligible": complete,
            "instances": rows,
            "elimination_reasons": (["missing_instance:" + ",".join(missing)] if missing else []) + sorted({
                reason for row in ranked for reason in row["elimination_reasons"] if reason != "missing_result"
            }),
        }
        if complete:
            vector = _bundle_vector(valid)
            ratios = [row["latency_ratio"] for row in sorted(valid, key=lambda row: row["instance_key"])]
            geomean = math.exp(sum(math.log(value) for value in ratios) / len(ratios))
            track_result.update({
                "exploration_bundle_digest": exploration_bundle_digest(track, vector),
                "per_instance_candidate_vector": vector,
                "geomean_latency_ratio": geomean,
                "worst_shard_latency_ratio": max(ratios),
                # Descriptive only: this is deliberately not the selection metric.
                "unnormalized_sum_sum_latency_ratio": sum(row["candidate_us"] for row in valid) / sum(row["baseline_us"] for row in valid),
            })
        track_rows.append(track_result)
    eligible = [row for row in track_rows if row["eligible"]]
    multi = len([row for row in config.document["benchmarks"] if row["usage"] == "benchmark"]) > 1
    if multi:
        eligible.sort(key=lambda row: (
            row["geomean_latency_ratio"], row["worst_shard_latency_ratio"],
            [item["candidate_sha256"] for item in row["per_instance_candidate_vector"]],
            row["track_id"], row["track_version"],
        ))
    else:
        eligible.sort(key=lambda row: (
            row["geomean_latency_ratio"],
            [item["candidate_sha256"] for item in row["per_instance_candidate_vector"]],
            row["track_id"], row["track_version"],
        ))
    winner = None
    if eligible:
        best = eligible[0]
        winner = {
            "track_id": best["track_id"],
            "track_version": best["track_version"],
            "per_instance_candidate_vector": best["per_instance_candidate_vector"],
            "exploration_bundle_digest": best["exploration_bundle_digest"],
            "geomean_latency_ratio": best["geomean_latency_ratio"],
            "worst_shard_latency_ratio": best["worst_shard_latency_ratio"],
            "kind": "exploration_winner",
            "formal_publication": False,
            "same_fresh_input": False,
            "cross_track_runtime_portfolio": False,
        }
    wall_values = [
        row["search_cost"]["total_search_wall_sec"] if "search_cost" in row else row["observation"].get("wall_sec")
        for row in all_rows
    ]
    spent_values = [
        row["search_cost"]["spent_usd"] if "search_cost" in row else row["observation"].get("actual_spent_usd")
        for row in all_rows
    ]
    wall_complete = bool(all_rows) and len(wall_values) == len(all_rows) and all(
        isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)) and value >= 0
        for value in wall_values
    )
    spent_complete = bool(all_rows) and len(spent_values) == len(all_rows) and all(
        isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)) and value >= 0
        for value in spent_values
    )
    row_cost_complete = bool(all_rows) and all(row.get("search_cost", {}).get("cost_complete", False) for row in all_rows)
    summary = {
        "schema_id": SUMMARY_SCHEMA_ID,
        "config_digest": config.digest,
        "task": config.document["task"],
        "target": config.document["target"],
        "benchmarks": config.document["benchmarks"],
        "comparison_mode": COMPARISON_MODE,
        "exploration_only": True,
        "formal_publication": False,
        "same_fresh_input": False,
        "run_seed": config.document["run_seed"],
        "master_seed": config.document["master_seed"],
        "input_generator": config.document["input_generator"],
        "comparability_digest": config.document["comparability_digest"],
        "selection_policy_version": SELECTION_POLICY_VERSION,
        "selection_policy": selection_policy_document(),
        "selection_policy_digest": selection_policy_digest(),
        "generation_policy": "serial",
        "quiet_confirmation_serial": True,
        "budget": {
            "scope": config.document["budget_scope"],
            "budget_usd": config.document["budget_usd"],
            "max_total_search_wall_sec": config.document["max_total_search_wall_sec"],
            "actual_total_spent_usd": sum(spent_values) if spent_complete else None,
            "actual_total_search_wall_sec": sum(wall_values) if wall_complete else None,
            "cost_complete": wall_complete and spent_complete and row_cost_complete,
        },
        "tracks": track_rows,
        "winner": winner,
        "no_winner": winner is None,
        "status": "exploration_winner" if winner is not None else "no_winner",
    }
    return summary


def explore_tracks(
    config: TrackExploreConfig,
    *,
    runner: Callable[[SingleTrackRequest], Mapping[str, Any] | None] | None = None,
    summarize_only: bool = False,
    adapter_factory: Callable[[TrackSpec], Any] | None = None,
    metadata_provider: Callable | None = None,
) -> dict:
    """Serially start missing single-Track runs, then summarize without re-evaluating."""
    config = _validated_config(config)
    output = config.output_path
    for participant in config.document["participating_tracks"]:
        for benchmark in config.document["benchmarks"]:
            run_dir = _instance_dir(config.resolve(participant["run_dir"]), benchmark["instance_key"])
            if output == run_dir or run_dir in output.parents or config.observation_path == run_dir or run_dir in config.observation_path.parents:
                raise TrackExploreError("summary and observation outputs must be outside every single-Track run_dir")
    if output.is_file():
        old = _read_json(output)
        if not isinstance(old, dict) or old.get("config_digest") != config.digest:
            raise TrackExploreError("refusing to overwrite another exploration's summary")
    registry = load_track_registry()
    adapter_factory = adapter_factory or _default_adapter_factory(config.document["target"])
    metadata_provider = metadata_provider or _default_task_metadata
    runner = runner or SubprocessSingleTrackRunner()
    started = time.monotonic()
    spent_by_track: dict[str, float] = {}
    for participant in config.document["participating_tracks"]:
        track = registry.get(participant["track_id"], version=participant["track_version"])
        adapter_factory(track)  # validate the implementation before any run starts
        spent_by_track[track.track_id] = 0.0
        for benchmark in config.document["benchmarks"]:
            instance_key = benchmark["instance_key"]
            run_dir = _instance_dir(config.resolve(participant["run_dir"]), instance_key)
            key = f"{track.track_id}/{instance_key}"
            if (run_dir / "result.json").is_file() or summarize_only:
                continue
            if run_dir.exists() and any(run_dir.iterdir()):
                raise TrackExploreError(
                    f"resume_interlock: {run_dir} is non-empty but has no result.json; use the existing single-Track resume flow"
                )
            remaining_wall = config.document["max_total_search_wall_sec"] - (time.monotonic() - started)
            if remaining_wall <= 0:
                _record_observation(config, key, {"skipped_reason": "total_wall_exhausted", "wall_sec": 0.0, "actual_spent_usd": 0.0})
                continue
            budget = config.document["budget_usd"]
            if config.document["budget_scope"] == "per_track":
                budget = max(0.0, budget - spent_by_track[track.track_id])
                if budget <= 0:
                    _record_observation(config, key, {"skipped_reason": "track_budget_exhausted", "wall_sec": 0.0, "actual_spent_usd": 0.0})
                    continue
            generation = dict(config.document["shared"]["generation"])
            generation.update({key: value for key, value in participant["generation"].items() if value is not None})
            request = SingleTrackRequest(
                task=config.document["task"], target=config.document["target"], track=track,
                benchmark=benchmark, run_dir=run_dir, run_seed=config.document["run_seed"],
                max_agent_turns=config.document["max_agent_turns"],
                build_timeout_sec=participant["build_timeout_sec"], generation=generation,
                shared=config.document["shared"], budget_usd=budget,
                budget_scope=config.document["budget_scope"], max_wall_sec=remaining_wall,
            )
            observation = dict(runner(request) or {})
            observation["budget_usd"] = budget
            observation["build_timeout_sec"] = request.build_timeout_sec
            observation["actual_spent_usd"] = None
            if (run_dir / "result.json").is_file():
                try:
                    result = _read_json(run_dir / "result.json")
                    spent = result.get("spent_usd")
                    if isinstance(spent, (int, float)) and not isinstance(spent, bool):
                        observation["actual_spent_usd"] = float(spent)
                        spent_by_track[track.track_id] += float(spent)
                except Exception:
                    pass
            _record_observation(config, key, observation)
    summary = summarize_track_explore(
        config, adapter_factory=adapter_factory, metadata_provider=metadata_provider,
    )
    _write_json(output, summary)
    return summary


__all__ = [
    "COMPARISON_MODE",
    "LATENCY_GATE",
    "SCHEMA_ID",
    "SELECTION_POLICY_VERSION",
    "SUMMARY_SCHEMA_ID",
    "SingleTrackRequest",
    "SubprocessSingleTrackRunner",
    "TrackExploreConfig",
    "TrackExploreError",
    "comparability_digest",
    "comparability_document",
    "explore_tracks",
    "exploration_bundle_digest",
    "reference_implementation_digest",
    "selection_policy_digest",
    "selection_policy_document",
    "summarize_track_explore",
    "validate_track_explore",
]
