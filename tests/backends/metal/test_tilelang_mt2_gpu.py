"""MT2 real-Metal vertical slice: fixed manual TileLang /2 candidates, never an LLM run."""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path

import pytest

from tcenv.remote.execution_geometry import EvaluatorBuildError, verify_evaluator_build
from tcenv.track import (
    CanonicalBundle, abi_limits_from_target, candidate_geometry,
    geometry_pair, load_track_registry,
)
from tcenv.target import load_target
from tcenv.track.abi import derive_tensor_memory_contract
from tests.support.tilelang_mt2 import (
    MSL_COMPARISON, SINGLE_TILELANG, TRACK_ID, TRACK_VERSION,
    evaluate_tilelang, tilelang_candidate, write_mt2_registry, write_mt2_task,
)

gpu = pytest.mark.gpu


def _record(name: str, observation: str, result, *, publication_allowed: bool = False) -> None:
    """Archive one fresh observation only in an explicitly named new evidence run directory."""
    root = os.environ.get("SHAPEFORGE_MT2_EVIDENCE_DIR")
    if not root:
        return
    directory = Path(root)
    directory.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_id": "shapeforge.mt2.real-metal-observation.1",
        "observation": observation,
        "real_metal": True,
        "model_calls": 0,
        "paid_search": False,
        "publication_allowed": bool(publication_allowed),
        "result": result.document(),
        "execution": result.execution,
        "build": result.build,
    }
    # Exclusive create: a rerun may not silently overwrite an earlier observation.
    (directory / name).open("x", encoding="utf-8").write(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")


def _assert_full_ok(result) -> None:
    report = result.report
    assert report["status"] == "ok", report.get("message") or report.get("compile")
    assert report["abi"]["ok"] is True
    assert report["correctness"]["ok"] is True
    assert report["benchmark"]["candidate"]["median_us"] > 0
    assert report["benchmark"]["noisy"] is False
    assert report["speedup"] is None  # no registered baseline: mechanism evidence, not a speedup claim
    assert result.execution["benchmark_protocol"]["rotation_sets"] == 16
    assert result.execution["benchmark_guard_violations"] == []
    assert result.execution["host_runs"] == []
    assert result.execution["tilelang_build"]["attempt"]["returncode"] == 0
    assert result.execution["tilelang_build"]["attempt"]["online"] is False
    build = result.build
    assert build["shape"] == "full" and build["candidate_sha256"] == result.candidate_sha256
    assert build["provenance"]["source_sha256"] == result.candidate_sha256
    assert build["provenance"]["builder_id"] == "tilelang-metal-track-builder"
    assert build["provenance"]["compile_profile"]["frontend"] == "tilelang"
    assert build["native"]["entry_path"] == "kernel.metal"
    assert build["native"]["descriptor"]["tilelang"]["track"]["track_version"] == 2
    paths = {entry["path"] for entry in build["artifacts"]["entries"]}
    assert paths == {
        "kernel.py", "host/execution_geometry.json", "candidate_manifest.json", "execution_geometry.json",
        "frontend_identity.json", "kernel.metal", "device_ir.json", "kernel_params.json", "host_module.txt",
    }
    # The frozen geometry enters the artifact set before its final identity is computed.
    assert build["provenance"]["artifact_set_digest"] == result.artifact_set_digest
    assert build["native"]["descriptor"]["execution_geometry"]["sha256"] == next(
        entry["sha256"] for entry in build["artifacts"]["entries"] if entry["path"] == "execution_geometry.json"
    )


def _canonical(locator, candidate):
    task = locator.load()
    target = load_target("apple-m4-max")
    abi = derive_tensor_memory_contract(task.instance, target=target)
    pair = geometry_pair(
        candidate_geometry(candidate), None, instance=task.instance, contract=abi,
        limits=abi_limits_from_target(target),
    )
    return CanonicalBundle(candidate=candidate, pair=pair)


@gpu
def test_mt2_tilelang_single_full_real_metal(tmp_path):
    locator = write_mt2_task(tmp_path)
    registry = write_mt2_registry(tmp_path / "tracks.json")
    candidate = tilelang_candidate(worklist=False)
    result = evaluate_tilelang(locator, registry, candidate, mode="full", seed=7, reps=3)
    _record("tilelang-single.json", "single-stage full fresh-rotation", result)
    _assert_full_ok(result)
    observed = [(row["kernel"], row["grid"], row["threads"]) for row in result.execution["dispatch"]]
    assert observed == [("scale_kernel_kernel", [4, 1, 1], [256, 1, 1])]
    assert any(arg["name"] == "acc" and arg["access"] == "read_write" for arg in result.execution["bindings"][0]["arguments"])
    assert any(case["worst"] and case["worst"]["output"] == "acc" for case in result.report["correctness"]["cases"])


@gpu
def test_mt2_tilelang_candidate_owned_two_stage_worklist_and_artifact_audit(tmp_path):
    locator = write_mt2_task(tmp_path)
    registry = write_mt2_registry(tmp_path / "tracks.json")
    candidate = tilelang_candidate(worklist=True)
    result = evaluate_tilelang(locator, registry, candidate, mode="full", seed=7, reps=3)
    _record("tilelang-worklist.json", "candidate-owned two-stage work-list full fresh-rotation + audit negatives", result)
    _assert_full_ok(result)
    observed = [(row["kernel"], row["grid"], row["threads"], row["workspace"]) for row in result.execution["dispatch"]]
    assert observed == [
        ("scale_kernel_kernel", [8, 1, 1], [128, 1, 1], ["tmp"]),
        ("bias_kernel_kernel", [8, 1, 1], [128, 1, 1], ["tmp"]),
    ]
    assert result.execution["workspace_bytes"] == {"tmp": 4096}
    assert result.execution["workspace_zero"] == {"tmp": False}
    assert len(result.execution["bindings"]) == 2

    spec = load_track_registry(registry).get(TRACK_ID, version=TRACK_VERSION)
    from tcenv.backends.metal.track_factory import adapter_for_track

    builder = adapter_for_track(spec).resolve("builder")(target_id="apple-m4-max")
    canonical = _canonical(locator, candidate)
    verified = verify_evaluator_build(
        result.build, canonical_bundle=canonical,
        identity={"opspec_digest": locator.load().spec.digest, "lowering_profile_digest": locator.load().fingerprint()},
        artifact_audit=builder.audit_evaluator_build,
    )
    assert verified.shape == "full" and verified.artifact_set_digest == result.artifact_set_digest
    entries = [dict(row) for row in result.build["artifacts"]["entries"]]
    candidate_paths = {"kernel.py", "host/execution_geometry.json"}
    kwargs = dict(
        candidate_entries=[row for row in entries if row["path"] in candidate_paths],
        manifest_entry=next(row for row in entries if row["path"] == "candidate_manifest.json"),
        geometry_entry=next(row for row in entries if row["path"] == "execution_geometry.json"),
        native=result.build["native"], canonical_bundle=canonical,
    )
    expected, native_paths = builder.audit_evaluator_build(entries=entries, **kwargs)
    assert expected == entries and native_paths == {"kernel.metal"}
    with pytest.raises(EvaluatorBuildError, match="kernel.metal.*missing"):
        builder.audit_evaluator_build(entries=[row for row in entries if row["path"] != "kernel.metal"], **kwargs)
    forged = copy.deepcopy(result.build["native"])
    forged["descriptor"]["tilelang"]["candidate_sha256"] = "0" * 64
    with pytest.raises(EvaluatorBuildError, match="another candidate"):
        builder.audit_evaluator_build(entries=entries, **{**kwargs, "native": forged})


@gpu
def test_mt2_same_semantics_msl_single_stage_comparison(tmp_path):
    from tcenv.backends.metal.track_adapter import MetalTrackBuilder, MetalTrackEvaluator, msl_single_geometry_resolver
    from tcenv.taskcontract.runner import evaluate_task

    locator = write_mt2_task(tmp_path)
    registry = write_mt2_registry(tmp_path / "tracks.json")
    result = evaluate_task(
        locator, track_id="msl-single", candidate_source=MSL_COMPARISON,
        builder=MetalTrackBuilder(),
        evaluator=MetalTrackEvaluator(mode="full", seed=7, reps=3, cases=("normal", "wide_range", "outlier")),
        geometry_resolver=msl_single_geometry_resolver(locator.load()), track_registry_path=registry,
    )
    _record("msl-single-comparison.json", "same-semantics one-stage MSL comparison", result)
    report = result.report
    assert report["status"] == "ok", report.get("message")
    assert report["abi"]["ok"] and report["correctness"]["ok"]
    assert report["benchmark"]["candidate"]["median_us"] > 0 and report["benchmark"]["noisy"] is False
    assert result.execution["benchmark_protocol"]["rotation_sets"] == 16
    assert [(row["kernel"], row["grid"], row["threads"]) for row in result.execution["dispatch"]] == [
        ("scale_msl", [4, 1, 1], [256, 1, 1])
    ]


@gpu
def test_mt2_shipped_registry_selects_v2_for_a_real_metal_worklist(tmp_path):
    """Use the actual shipped registry bytes, not the test-only registry writer."""
    locator = write_mt2_task(tmp_path)
    registry = Path(__file__).resolve().parents[3] / "backends/metal/tracks.json"
    result = evaluate_tilelang(
        locator, registry, tilelang_candidate(worklist=True), mode="full", seed=7, reps=3
    )
    _record("shipped-worklist.json", "shipped registry /2 real Metal work-list; /1 preserved", result)
    _assert_full_ok(result)
    assert result.track["track_version"] == 2
    assert result.execution["worker"]["track_version"] == 2
    from tcenv.track import load_track_registry

    live = load_track_registry(registry)
    assert live.get(TRACK_ID, version=1).status == "interface_only"
    assert live.get(TRACK_ID).track_version == 2


@gpu
@pytest.mark.parametrize("kind", ["invalid_kernel", "external_read"])
def test_mt2_frontend_or_sandbox_failure_is_partial_without_device_dispatch(tmp_path, kind):
    locator = write_mt2_task(tmp_path)
    registry = write_mt2_registry(tmp_path / "tracks.json")
    if kind == "invalid_kernel":
        source = SINGLE_TILELANG.replace("KERNEL = scale_kernel", "KERNEL = 42").encode()
    else:
        canary = tmp_path / "outside-attempt-canary.bin"
        canary.write_bytes(b"MT2 outside-attempt read must be denied without changing this canary\n")
        source = SINGLE_TILELANG.replace(
            "import tilelang\n",
            f"from pathlib import Path\nPath({str(canary)!r}).read_bytes()\nimport tilelang\n",
            1,
        ).encode()
    result = evaluate_tilelang(
        locator, registry, tilelang_candidate(worklist=False, source=source), mode="quick", reps=1
    )
    # Record the actual outcome before assertions so a wrong expectation can never erase the observation.
    _record(f"negative-{kind}.json", f"{kind}: expected partial build, zero dispatch", result)
    report = result.report
    assert report["status"] == "compile_error", report
    assert report["build"]["shape"] == "partial"
    assert report["build"]["candidate_sha256"] == result.candidate_sha256
    assert report["build"]["provenance"]["source_sha256"] == result.candidate_sha256
    assert report["build"]["provenance"]["artifact_set_digest"] is None
    assert report["build"]["assembled_sha256"] is not None
    assert not result.execution or not result.execution.get("dispatch")
    if kind == "external_read":
        assert canary.read_bytes() == b"MT2 outside-attempt read must be denied without changing this canary\n"


@gpu
def test_mt2_out_of_bounds_write_is_unsafe_and_cannot_be_published(tmp_path):
    locator = write_mt2_task(tmp_path)
    registry = write_mt2_registry(tmp_path / "tracks.json")
    source = SINGLE_TILELANG.replace("y[index] = value", "y[index + 1] = value").encode()
    result = evaluate_tilelang(
        locator, registry, tilelang_candidate(worklist=False, source=source), mode="quick", reps=1
    )
    _record("negative-out-of-bounds.json", "real guard violation: full build but unsafe/non-publishable", result)
    assert result.report["status"] == "unsafe", result.report
    assert result.build["shape"] == "full"  # a real compile happened; safety still rejects execution
    assert result.report["correctness"]["ok"] is False
