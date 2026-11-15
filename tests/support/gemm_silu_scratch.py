"""gemm_silu REG candidates, probe evidence and the scratch decision.

Pure Python on purpose: the shared CPU test imports this module, so it must not import
``tcenv.backends.metal``.  The Metal record/replay code lives in
``tests/backends/metal/gemm_silu_scratch_record.py``.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
METAL_FIXTURES = ROOT / "tests" / "support" / "golden" / "metal" / "fixtures"
TREE = ROOT / "tests" / "support" / "golden" / "metal" / "gemm_silu_reg"
_FIXTURE = ROOT / "tests/support/golden/shared/acceptance-epochs/fixtures/p2o"
P2O = _FIXTURE
EVIDENCE = _FIXTURE / "evidence"

SMALL_SHA256 = "d5203e459ac2c8bca4d1d59a0fb22ba5910e568fde98fe9b83da7e43f9f07d4c"
PAPER_SHA256 = "461cff911579129e2a6209fc31c2a14db68eac867c1d15d18d6378c713cc5330"
SIZES = ("small", "paper")
SHAPES = {"small": (256, 256, 128), "paper": (2048, 2048, 2048)}
DECLARED_CASES = ("normal", "wide_range", "large", "tiny", "outlier", "ieee")
PROBE_SCHEMA = "shapeforge.gemm-silu-scratch-probe.2"
DECISION_SCHEMA = "shapeforge.gemm-silu-scratch-decision.1"
CONTROL_SCHEMA = "shapeforge.gemm-silu-plain-gemm-control.1"
REG_TIMEOUT_SECONDS = 900

SMALL_CONFIRMATION = EVIDENCE / "reg-small-confirmation.json"
PAPER_CONFIRMATION = EVIDENCE / "reg-paper-confirmation.json"
SMALL_CONTROL = EVIDENCE / "plain-gemm-small-control.json"
PAPER_CONTROL = EVIDENCE / "plain-gemm-paper-control.json"
PROBE = EVIDENCE / "scratch-probe.json"
DECISION = EVIDENCE / "decision.json"

_LAUNCH = {
    "small": "/* taic-launch\nmath fast\nlaunch gemm_kernel grid=4,4,1 threads=32,2,2\n*/\n",
    "paper": "/* taic-launch\nmath fast\nlaunch gemm_kernel grid=32,32,1 threads=128,1,1\n*/\n",
}
_CONSTANTS = "constant constexpr uint BLOCK_M = 64;\nconstant constexpr uint BLOCK_N = 64;\nconstant constexpr uint BLOCK_K = 32;\n"


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha256(path: Path) -> str:
    return sha256_bytes(Path(path).read_bytes())


def load_json(path: Path) -> dict:
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise AssertionError(f"{path} is not a JSON object")
    return document


def dump_json(path: Path, document: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _kernel_comments() -> dict[str, str]:
    return json.loads((_FIXTURE / "kernel-comments.json").read_text(encoding="utf-8"))


def generated_source(size: str) -> str:
    """Rebuild a REG candidate from the registered GEMM fixture plus the SiLU epilogue."""
    comments = _kernel_comments()
    if size == "small":
        body = (METAL_FIXTURES / "gemm_small_steel.metal").read_text(encoding="utf-8")
        anchor = "} // namespace steel\n} // namespace mlx\n\nconstant mlx::steel::GEMMParams p3_params"
        if body.count(anchor) != 1:
            raise AssertionError("gemm_small_steel.metal no longer has the expected namespace boundary")
        epilogue = comments["small_epilogue"]
        body = body.replace(anchor, "} // namespace steel\n} // namespace mlx\n" + epilogue + "\nconstant mlx::steel::GEMMParams p3_params")
        old = "using Kernel = mlx::steel::GEMMKernel<half, half, 64, 64, 16, 2, 2, false, false, true, true, float>;"
        if body.count(old) != 1:
            raise AssertionError("the small steel kernel instantiation drifted")
        body = body.replace(
            old,
            "using Kernel = mlx::steel::GEMMKernel<half, half, 64, 64, 16, 2, 2, false, false, true, true, float, mlx::steel::TransformSiluFused<float, float>>;",
        )
        old_store = """  METAL_FUNC void store_result(device U* D, const int ldd) thread {
    // The plain output store performs the single fp32 -> fp16 conversion directly.
    // Do not add a redundant float -> half -> float pre-round trip here.
    D += sm * ldd + sn;"""
        if body.count(old_store) != 1:
            raise AssertionError("the small steel aligned store path drifted")
        body = body.replace(old_store, comments["small_store"])
        head = comments["small_head"]
        return head + _CONSTANTS + _LAUNCH[size] + body
    if size == "paper":
        body = (METAL_FIXTURES / "gemm_paper_best.metal").read_text(encoding="utf-8")
        old = """            thread auto& te = acc[i][j].thread_elements();
            uint mine = as_type<uint>(half2((half)te[0], (half)te[1]));"""
        if body.count(old) != 1:
            raise AssertionError("the paper epilogue drifted")
        body = body.replace(old, comments["paper_body"])
        head = comments["paper_head"]
        return head + _CONSTANTS + _LAUNCH[size] + body
    raise ValueError(f"unknown size {size!r}")


def stored_source(size: str) -> str:
    return (TREE / "variants" / size / "kernel.metal").read_text(encoding="utf-8")


def plain_gemm_source(size: str) -> str:
    """The registered GEMM candidate, byte-for-byte, used as the same-algorithm control."""
    fixture = "gemm_small_steel.metal" if size == "small" else "gemm_paper_best.metal"
    body = (METAL_FIXTURES / fixture).read_text(encoding="utf-8")
    return _CONSTANTS + _LAUNCH[size] + body


def candidate_sha256(size: str) -> str:
    return sha256(TREE / "variants" / size / "kernel.metal")


def assert_source_provenance() -> None:
    for size in SIZES:
        stored = stored_source(size)
        if stored != generated_source(size):
            raise AssertionError(f"{size} REG source is not the generated fixture + SiLU epilogue")
        expected = SMALL_SHA256 if size == "small" else PAPER_SHA256
        if candidate_sha256(size) != expected:
            raise AssertionError(f"{size} REG candidate SHA-256 drifted")
        if "scratch" in stored.split("/* taic-launch", 1)[1].split("*/", 1)[0]:
            raise AssertionError(f"{size} REG candidate declares scratch; REG must stay on-chip")
        launches = [line for line in stored.splitlines() if line.startswith("launch ")]
        if len(launches) != 1:
            raise AssertionError(f"{size} REG candidate must be one stage, found {launches}")
        if size == "small" and "TransformSiluFused<float, float>" not in stored:
            raise AssertionError("the small REG candidate lost the fused SiLU epilogue")
        if size == "paper" and "te[0] / (1.0f + metal::exp(-te[0]))" not in stored:
            raise AssertionError("the paper REG candidate lost the fused SiLU epilogue")


def assert_no_global_candidate() -> None:
    files = sorted(path.name for path in (TREE / "variants").rglob("*") if path.is_file())
    if len(files) != 2 or any(name != "kernel.metal" for name in files):
        raise AssertionError(f"the REG tree grew an unexpected file: {files}")
    for size in SIZES:
        source = stored_source(size)
        body = source.split("/* taic-launch", 1)[1].split("*/", 1)[0]
        if "scratch" in body:
            raise AssertionError(f"{size} contains a global-scratch declaration")


def _report(subject: dict) -> dict:
    report = subject.get("confirmation_report")
    if not isinstance(report, dict):
        raise AssertionError("the subject has no confirmation_report")
    return report


def _case_ok(report: dict) -> None:
    if report.get("status") != "ok":
        raise AssertionError(f"fresh report is not ok: {report.get('status')!r}")
    cases = (report.get("correctness") or {}).get("cases") or []
    if [item.get("case") for item in cases] != list(DECLARED_CASES):
        raise AssertionError("the fresh report did not run the six declared cases")
    if not all(item.get("ok") is True for item in cases):
        raise AssertionError("a declared case failed")
    safety = report.get("safety") or {}
    if safety.get("ok") is not True or safety.get("nondeterminism") != []:
        raise AssertionError("the fresh report is not safe/deterministic")


def assert_subject(subject: dict, size: str) -> None:
    expected_sha = SMALL_SHA256 if size == "small" else PAPER_SHA256
    if subject.get("candidate_sha256") != expected_sha:
        raise AssertionError(f"{size} subject candidate SHA-256 drifted")
    if subject.get("provider") != "candidate" or subject.get("fallback_id") is not None:
        raise AssertionError(f"{size} subject did not resolve to its REG candidate")
    if subject.get("status") != "confirmed":
        raise AssertionError(f"{size} subject is not confirmed")
    if (subject.get("identity") or {}).get("kind") != "match":
        raise AssertionError(f"{size} subject identity is not a match")
    if subject.get("search_seed") == subject.get("confirmation_seed"):
        raise AssertionError(f"{size} fresh seed equals the search seed")
    report = _report(subject)
    _case_ok(report)
    if report.get("dtype") != "fp16" or report.get("size") != size or report.get("task") != "gemm_silu":
        raise AssertionError(f"{size} subject report names the wrong instance")
    if (report.get("pipeline") or {}).get("stages") != 1:
        raise AssertionError(f"{size} REG candidate is not a single-stage pipeline")
    if not report.get("candidate_us") or report["candidate_us"] <= 0:
        raise AssertionError(f"{size} has no positive candidate time")
    if (report.get("benchmark") or {}).get("noisy") is not False:
        raise AssertionError(f"{size} fresh timing is flagged noisy")
    if report.get("build", {}).get("candidate_sha256") != expected_sha:
        raise AssertionError(f"{size} build identity does not name the REG candidate")


def stable_identity(subject: dict) -> dict:
    """Fields that must survive a replay; wall-clock microseconds are deliberately absent."""
    report = _report(subject)
    build = report.get("build") or {}
    expected = subject.get("expected") or {}
    return {
        "candidate_sha256": subject.get("candidate_sha256"),
        "confirmation_seed": subject.get("confirmation_seed"),
        "instance_key": subject.get("instance_key"),
        "usage": subject.get("usage"),
        "identity": (subject.get("identity") or {}).get("kind"),
        "expected": expected,
        "build": {
            "candidate_sha256": build.get("candidate_sha256"),
            "assembled_sha256": build.get("assembled_sha256"),
            "native_digest": build.get("native_digest"),
            "lowering_profile_digest": build.get("lowering_profile_digest"),
            "opspec_digest": build.get("opspec_digest"),
            "artifact_set_digest": (build.get("artifacts") or {}).get("artifact_set_digest"),
        },
        "evaluation": {
            key: report.get(key)
            for key in (
                "evaluation_fingerprint",
                "baseline_environment_digest",
                "baseline_implementation_digest",
                "target_id",
                "protocol_version",
            )
        },
        "pipeline": report.get("pipeline"),
        "eligible": report.get("eligible"),
    }


def assert_control(document: dict, size: str) -> None:
    if document.get("schema_id") != CONTROL_SCHEMA:
        raise AssertionError("unknown plain-GEMM control schema")
    if document.get("publication_allowed") is not False or document.get("fresh_confirmation") is not False:
        raise AssertionError("a control run may not be published or confirmed")
    report = document.get("report") or {}
    if report.get("status") != "ok" or report.get("task") != "gemm" or report.get("size") != size:
        raise AssertionError(f"{size} plain-GEMM control is not a safe/ok gemm run")
    cases = (report.get("correctness") or {}).get("cases") or []
    if [item.get("case") for item in cases] != list(DECLARED_CASES) or not all(item.get("ok") for item in cases):
        raise AssertionError(f"{size} plain-GEMM control did not pass all six cases")
    if report.get("mode") != "full" or report.get("seed") != 7:
        raise AssertionError(f"{size} control is not the search-seed full run")
    if (report.get("benchmark") or {}).get("noisy") is not False:
        raise AssertionError(f"{size} control timing is noisy")


def assert_probe(document: dict) -> None:
    if document.get("schema_id") != PROBE_SCHEMA:
        raise AssertionError("unknown scratch-probe schema")
    if document.get("not_a_candidate") is not True or document.get("publication_allowed") is not False:
        raise AssertionError("the probe must stay non-publishable and not a candidate")
    for size in SIZES:
        one = (document.get("sizes") or {}).get(size)
        if not isinstance(one, dict):
            raise AssertionError(f"the probe has no {size} entry")
        if one.get("symbols") != dict(zip(("M", "N", "K"), SHAPES[size])):
            raise AssertionError(f"{size} probe shape drifted")
        m, n, _k = SHAPES[size]
        if one.get("scratch_roundtrip_bytes") != 8 * m * n:
            raise AssertionError(f"{size} round-trip byte count drifted")
        arms = one.get("arms") or {}
        if set(arms) != {
            "single_stage_fp16_write",
            "two_stage_floor",
            "two_stage_texture_roundtrip",
            "two_stage_buffer_roundtrip",
        }:
            raise AssertionError(f"{size} probe arms drifted")
        for name, arm in arms.items():
            if arm["wall_us_median"] <= 0 or arm["gpu_us_median"] <= 0:
                raise AssertionError(f"{size}/{name} has no positive time")
            if len(arm["wall_samples_us"]) != 9:
                raise AssertionError(f"{size}/{name} does not carry nine wall samples")
        derived = one["derived"]
        if arms["two_stage_texture_roundtrip"]["fill_bytes_per_iteration"] != 0:
            raise AssertionError(f"{size}: the texture arm must not pay a fill")
        if arms["two_stage_buffer_roundtrip"]["fill_bytes_per_iteration"] != 4 * m * n:
            raise AssertionError(f"{size}: the buffer arm must bill its device_zero fill")
        texture = arms["two_stage_texture_roundtrip"]
        buffer = arms["two_stage_buffer_roundtrip"]
        for name, arm in (("texture", texture), ("buffer", buffer)):
            coverage = arm.get("coverage")
            if not isinstance(coverage, dict):
                raise AssertionError(f"{size}/{name}: the round trip has no coverage read-back")
            if coverage.get("elements") != m * n:
                raise AssertionError(f"{size}/{name}: coverage does not cover every element")
            if coverage.get("scratch_elements_read_back") != m * n:
                raise AssertionError(f"{size}/{name}: the scratch read-back is incomplete")
            if coverage.get("output_elements_read_back") != m * n:
                raise AssertionError(f"{size}/{name}: the output read-back is incomplete")
            if coverage.get("scratch_elements_match") is not True or coverage.get("output_elements_match") is not True:
                raise AssertionError(f"{size}/{name}: an element did not carry the probe pattern")
            if coverage.get("ok") is not True:
                raise AssertionError(f"{size}/{name}: coverage failed")
        negative = one.get("quarter_coverage_negative") or {}
        if negative.get("detected") is not True:
            raise AssertionError(f"{size}: the quarter-coverage control was not detected")
        if negative.get("scratch_elements_match") is not False:
            raise AssertionError(f"{size}: the quarter-coverage control did not leave scratch holes")


def _best_baseline(report: dict) -> tuple[str, float]:
    baseline = report.get("baseline_us") or {}
    name, value = min(baseline.items(), key=lambda item: float(item[1]))
    return str(name), float(value)


def decision_document(subjects: dict[str, dict], controls: dict[str, dict], probe: dict) -> dict:
    """Recompute the REG-vs-GLOBAL decision from the archived measurements.

    This function is the single source of truth for ``evidence/decision.json``; the
    recorder calls it and the verifier recomputes it byte-for-byte.
    """
    assert_probe(probe)
    sizes = {}
    for size in SIZES:
        subject, control = subjects[size], controls[size]
        assert_subject(subject, size)
        assert_control(control, size)
        report = _report(subject)
        m, n, k = SHAPES[size]
        reg_us = float(report["candidate_us"])
        reg_gpu_us = float(report["benchmark"]["candidate_gpu_us_median"])
        reg_cv = float(report["benchmark"]["noise_cv"])
        best_name, best_us = _best_baseline(report)
        plain_us = float(control["report"]["candidate_us"])
        fusion_us = reg_us - plain_us
        peak_gbps = float(report["roofline"]["peak_gbps_measured"])
        serialized_bytes = 8 * m * n
        serialized_roofline_us = serialized_bytes / (peak_gbps * 1e9) * 1e6
        # A conservative floor only charges the stage-2 read+output traffic:
        # the producer's extra 2*M*N fp32 write can hide inside the compute-bound GEMM,
        # and GLOBAL does not pay stage-1's SiLU. The full 8*M*N serialized number is
        # kept as a reference estimate, not as a lower bound.
        consumer_bytes = 6 * m * n
        consumer_roofline_us = consumer_bytes / (peak_gbps * 1e9) * 1e6
        net_consumer_roofline_us = max(0.0, consumer_roofline_us - max(fusion_us, 0.0))
        one = probe["sizes"][size]
        single = float(one["arms"]["single_stage_fp16_write"]["wall_us_median"])
        floor = float(one["derived"]["two_stage_floor_us"])
        texture = float(one["derived"]["two_stage_texture_roundtrip_us"])
        buffer = float(one["derived"]["two_stage_buffer_roundtrip_us"])
        texture_extra = float(one["derived"]["texture_roundtrip_over_single_us"])
        buffer_extra = float(one["derived"]["buffer_roundtrip_over_single_us"])
        serialized_global = reg_us + serialized_roofline_us
        texture_probe_global = reg_us + texture_extra
        buffer_probe_global = reg_us + buffer_extra
        # Any two-stage pipeline also pays the measured launch/barrier floor.
        hard_lower_bound = max(floor, reg_us + net_consumer_roofline_us)
        best_probe_global = min(texture_probe_global, buffer_probe_global)
        stop_reasons = ["global_saves_no_flops"]
        if floor >= reg_us:
            stop_reasons.append("two_stage_floor_exceeds_reg")
        if consumer_roofline_us > max(fusion_us, 0.0):
            stop_reasons.append("consumer_scratch_roofline_exceeds_fusion_saving")
        if hard_lower_bound >= reg_us:
            stop_reasons.append("hard_lower_bound_exceeds_reg")
        if best_probe_global >= reg_us:
            stop_reasons.append("probe_extrapolated_global_is_slower")
        if fusion_us <= reg_us * 0.01:
            stop_reasons.append("fusion_cost_below_one_percent")
        sizes[size] = {
            "symbols": {"M": m, "N": n, "K": k},
            "reg_candidate_sha256": subject["candidate_sha256"],
            "reg_candidate_us": reg_us,
            "reg_candidate_gpu_us_median": reg_gpu_us,
            "reg_cv": reg_cv,
            "reg_noisy": report["benchmark"]["noisy"],
            "reg_confirmation_seed": subject["confirmation_seed"],
            "reg_search_seed": subject["search_seed"],
            "best_baseline": best_name,
            "best_baseline_us": best_us,
            "reg_latency_ratio_vs_best": reg_us / best_us,
            "reg_speedup_vs_best": best_us / reg_us,
            "plain_gemm_us": plain_us,
            "fusion_cost_us": fusion_us,
            "fusion_cost_fraction_of_reg": fusion_us / reg_us,
            "peak_gbps_measured": peak_gbps,
            "scratch_fp32_bytes": 4 * m * n,
            "scratch_roundtrip_bytes": serialized_bytes,
            "serialized_scratch_roofline_us": serialized_roofline_us,
            "consumer_scratch_bytes": consumer_bytes,
            "consumer_scratch_roofline_us": consumer_roofline_us,
            "net_consumer_roofline_us": net_consumer_roofline_us,
            "single_stage_fp16_write_us": single,
            "two_stage_floor_us": floor,
            "two_stage_texture_roundtrip_us": texture,
            "two_stage_buffer_roundtrip_us": buffer,
            "texture_roundtrip_over_single_us": texture_extra,
            "buffer_roundtrip_over_single_us": buffer_extra,
            "serialized_global_estimate_us": serialized_global,
            "serialized_global_estimate_over_reg": serialized_global / reg_us,
            "texture_probe_global_estimate_us": texture_probe_global,
            "buffer_probe_global_estimate_us": buffer_probe_global,
            "best_probe_global_estimate_us": best_probe_global,
            "best_probe_global_estimate_over_reg": best_probe_global / reg_us,
            "probe_estimates_are_not_candidates": True,
            "hard_lower_bound_us": hard_lower_bound,
            "hard_lower_bound_over_reg": hard_lower_bound / reg_us,
            "global_would_win": best_probe_global < reg_us,
            "stop_reasons": stop_reasons,
        }
    any_win = any(one["global_would_win"] for one in sizes.values())
    if any_win:
        raise AssertionError("a measured GLOBAL lower bound is below REG; the negative decision is invalid")
    return {
        "schema_id": DECISION_SCHEMA,
        "task": "gemm_silu",
        "dtype": "fp16",
        "target_id": "apple-m4-max",
        "publication_allowed": False,
        "adopted_route": "REG",
        "implementation": {
            "reg_candidate_added": True,
            "global_candidate_source_added": False,
            "same_algorithm_ab_run": False,
            "reg_tree": "tests/support/golden/metal/gemm_silu_reg",
            "reg_small_sha256": SMALL_SHA256,
            "reg_paper_sha256": PAPER_SHA256,
        },
        "sizes": sizes,
        "decision": "reg_only",
        "rationale": (
            "GLOBAL splits the same GEMM and the same SiLU into producer + barrier + consumer and "
            "adds an unavoidable fp32 intermediate write+read of 8*M*N bytes. It saves no FLOPs. "
            "At small the measured two-stage floor alone (about 9 us) is already several times the "
            "REG candidate (about 1.8 us). At paper a conservative floor charges only the stage-2 "
            "read+output traffic (6*M*N, about 53 us at peak) minus the measured fusion cost "
            "(about 2 us), giving about 50.6 us; even that is 4.4% of REG. The full serialized "
            "32 MiB estimate is about 70.6 us, and the full-coverage texture probe adds about 80 us, but those "
            "are estimates, not lower bounds. The device_zero buffer route additionally pays a "
            "timed fill. Therefore no GLOBAL candidate was written; REG is kept."
        ),
        "probe_not_a_candidate": True,
        "npu": {"simulated_only": True, "publication_allowed": False},
        "replay": {
            "probe": "uv run python experiments/gemm_silu_scratch_probe.py <output>",
            "cpu_test": "uv run pytest -q -n 0 tests/shared/test_gemm_silu_scratch_decision.py",
            "gpu_test": "uv run pytest -q -n 0 -m gpu tests/backends/metal/test_gemm_silu_scratch_gpu.py",
        },
    }


def _historical_replay_verifier() -> str:
    """One historical command from the fixture. It is not emitted into a new decision."""
    raw = (_FIXTURE / "historical-replay-verifier.txt").read_text(encoding="utf-8")
    if raw.endswith("\n"):
        raw = raw[:-1]
    if not raw or raw != raw.strip() or "\n" in raw:
        raise AssertionError("historical replay verifier fixture is not one command")
    return raw


def _without_archived_verifier(document: dict) -> dict:
    """Drop only the historical command. Any other verifier string fails."""
    replay = document.get("replay")
    if not isinstance(replay, dict) or "verifier" not in replay:
        return document
    if replay.get("verifier") != _historical_replay_verifier():
        raise AssertionError("archived replay.verifier is not the historical command")
    compared = dict(document)
    compared_replay = dict(replay)
    del compared_replay["verifier"]
    compared["replay"] = compared_replay
    return compared


def assert_decision(document: dict, subjects: dict[str, dict], controls: dict[str, dict], probe: dict) -> None:
    expected = decision_document(subjects, controls, probe)
    if "verifier" in expected["replay"]:
        raise AssertionError("a recomputed decision must not name a package verifier")
    if _without_archived_verifier(document) != expected:
        raise AssertionError("decision.json is not the recomputation of its measurement sources")
    if document["adopted_route"] != "REG" or document["decision"] != "reg_only":
        raise AssertionError("the decision did not keep REG")
    if document["implementation"]["global_candidate_source_added"] is not False:
        raise AssertionError("a GLOBAL candidate source was added")
    if document["publication_allowed"] is not False:
        raise AssertionError("gemm_silu may not publish a speedup")
