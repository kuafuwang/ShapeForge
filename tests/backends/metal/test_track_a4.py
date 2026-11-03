"""A4: real Mac vertical slices for the two Metal tracks and the v2 evaluate_task seam.

The tests in this file open the Metal device and build/execute real kernels.  They are marked
``gpu``; the CPU portion is limited to contract/geometry construction and dispatch comparison.
"""

from __future__ import annotations
from tests.support.v1_tasks import V1_TASK

import hashlib
import json
import os
import struct
from pathlib import Path

import pytest

from tcenv.track import CandidateArtifact, CandidateEntry

from ...support.v2_tasks import (
    RMSNORM_READ_WRITE_GEN_REFERENCE,
    RMSNORM_SILU_REFERENCE,
    rmsnorm_read_write_opspec,
    rmsnorm_silu_opspec,
    rmsnorm_workspace_opspec,
    write_task,
)

gpu = pytest.mark.gpu


def _record(name: str, payload: dict) -> None:
    """@brief A4 verify sets A4_EVIDENCE_DIR; tests archive one real-slice record."""
    directory = os.environ.get("A4_EVIDENCE_DIR")
    if not directory:
        return
    Path(directory).mkdir(parents=True, exist_ok=True)
    (Path(directory) / name).write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _observed(result) -> list[tuple]:
    """@brief Return the runtime's recorded actual dispatch, not a source re-parse."""
    assert result.execution is not None
    seen: list[tuple] = []
    for record in result.execution["dispatch"]:
        item = (
            record["kernel"],
            tuple(record["grid"]),
            tuple(record["threads"]),
            int(record["threadgroup_bytes"]),
            tuple(record["workspace"]),
        )
        if item not in seen:
            seen.append(item)
    return seen


def _canonical_single(source: str, contract):
    """@brief Rebuild the A2 canonical bundle for a single-source candidate."""
    from tcenv.target import DEFAULT_TARGET_ID, load_target
    from tcenv.backends.metal.track_adapter import msl_single_geometry_resolver
    from tcenv.track import (
        CanonicalBundle,
        abi_limits_from_target,
        candidate_from_single_source,
        geometry_pair,
    )
    from tcenv.track.abi import derive_tensor_memory_contract

    candidate = candidate_from_single_source(source)
    geometry = msl_single_geometry_resolver(contract)(candidate)
    abi = derive_tensor_memory_contract(contract.instance, target=load_target(DEFAULT_TARGET_ID))
    pair = geometry_pair(
        geometry, None, instance=contract.instance, contract=abi, limits=abi_limits_from_target(load_target(DEFAULT_TARGET_ID))
    )
    return CanonicalBundle(candidate=candidate, pair=pair)


def _structural_full_build(canonical):
    """@brief CPU 结构 full build：候选 SOURCE + 冻结几何，不编译任何 Metal。

    The evaluator-build audit must reject a worker record that listed a wrong/missing/extra
    artifact without opening the device.  This fixture is the same typed control-host shape a
    real MTL build produces (a live handle stand-in, an assembled digest in the native
    descriptor, and the candidate bytes as SOURCE), so the audit can be exercised on CPU.
    """
    from tcenv.remote.artifact_builder import (
        BuildArtifact,
        BuildEvidence,
        BuildOutcome,
        BuildProfile,
        BuildProvenance,
    )
    from tcenv.remote.artifact_bundle import ArtifactBundle, BundleEntry, NativeExecutable
    from tcenv.remote.execution_geometry import attach_execution_geometry
    from tests.support.report_build import StructuralNativeHandle

    candidate = canonical.candidate
    entry = candidate.entries[0]
    source = BundleEntry(
        path=entry.path,
        role="SOURCE",
        media_type=entry.media_type,
        size_bytes=len(entry.data),
        sha256=entry.sha256,
        data=bytes(entry.data),
    )
    assembled = hashlib.sha256(b"a4 structural assembled fixture").hexdigest()
    native = NativeExecutable(
        backend_id="metal",
        kind="compute-pipeline",
        entry_path=entry.path,
        entry_sha256=entry.sha256,
        descriptor={"assembled_sha256": assembled},
        handle=StructuralNativeHandle(),
    )
    bundle = ArtifactBundle(entries=(source,), native=native)
    profile = BuildProfile(
        build_profile="metal-msl-control-host-v1",
        builder_id="metal-source-builder",
        builder_version="1",
        build_host="control_host",
        toolchain_id="apple-metal",
        toolchain_version="unknown",
        sdk_version="unknown",
        compile_profile={"language": "msl", "math": "safe", "device": "fixture", "gpu_family": "fixture"},
    )
    provenance = BuildProvenance(
        profile=profile,
        source_sha256=candidate.candidate_sha256,
        artifact_set_digest=bundle.artifact_set_digest,
    )
    evidence = BuildEvidence(
        mode="prebuilt",
        status="ok",
        provenance=provenance,
        artifacts=(BuildArtifact.from_entry(source),),
        placement="control_host",
        native=native,
    )
    outcome = BuildOutcome(
        bundle=bundle,
        provenance=provenance,
        evidence=evidence,
        payloads={},
        assembled_sha256=assembled,
    )
    return attach_execution_geometry(outcome, canonical_bundle=canonical)


def _structural_partial_build(canonical, *, build_kind: str, identity):
    """@brief CPU 结构 partial build：与 canonical 候选同身份的 compile failure。"""
    import hashlib

    from tcenv.build_projection import project_partial
    from tcenv.remote.artifact_builder import BuildError, BuildProfile, BuildProvenance

    candidate = canonical.candidate
    profile = BuildProfile(
        build_profile="metal-msl-control-host-v1",
        builder_id="metal-source-builder",
        builder_version="1",
        build_host="control_host",
        toolchain_id="apple-metal",
        toolchain_version="unknown",
        sdk_version="unknown",
        compile_profile={"language": "msl", "math": "safe", "device": "fixture", "gpu_family": "fixture"},
    )
    provenance = BuildProvenance(
        profile=profile,
        source_sha256=candidate.candidate_sha256,
        artifact_set_digest=None,
    )
    error = BuildError(
        "compile",
        "structural fixture: MSL frontend rejected the kernel",
        log=b"kernel.metal:1:1: error: structural fixture rejection\n",
        duration_ms=1,
        provenance=provenance,
        assembled_sha256=hashlib.sha256(b"a4 structural partial fixture").hexdigest(),
    )
    return project_partial(error, build_kind=build_kind, identity=identity)


def _reseal_evaluator_build(fields):
    """@brief 重新封口一个被测试改写的 full build，使只有目标不变量破坏。"""
    from tcenv.remote.artifact_bundle import BUNDLE_SCHEMA_ID, NativeExecutable, artifact_set_digest

    fields["artifacts"]["entries"].sort(key=lambda item: item["path"])
    fields["artifacts"]["artifact_set_digest"] = artifact_set_digest(
        BUNDLE_SCHEMA_ID, fields["artifacts"]["entries"]
    )
    fields["provenance"]["artifact_set_digest"] = fields["artifacts"]["artifact_set_digest"]
    fields["evidence"]["artifacts"] = json.loads(json.dumps(fields["artifacts"]))
    fields["evidence"]["provenance"] = json.loads(json.dumps(fields["provenance"]))
    by_path = {item["path"]: item for item in fields["artifacts"]["entries"]}
    native = fields["native"]
    native["entry_sha256"] = by_path[native["entry_path"]]["sha256"]
    rebuilt = NativeExecutable(
        backend_id=native["backend_id"],
        kind=native["kind"],
        entry_path=native["entry_path"],
        entry_sha256=native["entry_sha256"],
        descriptor=native["descriptor"],
    )
    fields["native"] = rebuilt.document()
    fields["native_digest"] = rebuilt.digest


def _tracks(path: Path, *, version: int = 1) -> Path:
    """@brief 写出 A4 的可用 track registry（两个 track 都有真实证据）。

    ``version`` lets an A4 negative prove the worker selects the control host's frozen
    ``track_version`` instead of silently falling back to the shipped default registry.
    """
    document = json.loads((Path(__file__).resolve().parents[3] / "backends/metal/tracks.json").read_text())
    for track in document["tracks"]:
        track["status"] = "available"
        track["track_version"] = int(version)
        evidence = list(track.get("evidence", []))
        if "tests/backends/metal/test_track_a4.py" not in evidence:
            evidence.insert(0, "tests/backends/metal/test_track_a4.py")
        track["evidence"] = evidence
        if int(version) != 1:
            # A distinct build-adapter identity keeps the frozen track digest different from the
            # shipped default track, so a default-registry worker is visible as drift.
            track["build_adapter"] = {
                "implementation_id": track["build_adapter"]["implementation_id"],
                "implementation_version": 2,
            }
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def _rmsnorm_task(tmp_path: Path):
    """@brief 写 v2 RMSNorm + workspace 任务并返回 locator/contract/registry。"""
    from tcenv.taskcontract import discover_standard_tasks
    from tcenv.track import REPO_ROOT

    root = tmp_path / "tasks"
    write_task(root, op_name="rmsnorm_v2", opspec=rmsnorm_workspace_opspec("rmsnorm_v2"))
    locator = discover_standard_tasks(root, code_root=REPO_ROOT)[0]
    contract = locator.load()
    return locator, contract, _tracks(tmp_path / "tracks.json")


def _fusion_task(tmp_path: Path):
    """@brief 写融合 RMSNorm+SiLU 的 v2 任务。"""
    from tcenv.taskcontract import discover_standard_tasks
    from tcenv.track import REPO_ROOT

    root = tmp_path / "tasks"
    write_task(
        root,
        op_name="rmsnorm_silu",
        opspec=rmsnorm_silu_opspec("rmsnorm_silu"),
        reference=RMSNORM_SILU_REFERENCE,
    )
    locator = discover_standard_tasks(root, code_root=REPO_ROOT)[0]
    return locator, locator.load(), _tracks(tmp_path / "tracks.json")


def _rmsnorm_single_source() -> str:
    return """#include <metal_stdlib>
using namespace metal;
kernel void rmsnorm_single(device const half* x [[buffer(0)]],
                           device const half* w [[buffer(1)]],
                           device half* y [[buffer(2)]],
                           device half* scratch [[buffer(3)]],
                           uint row [[threadgroup_position_in_grid]],
                           uint lid [[thread_index_in_threadgroup]],
                           uint lane [[thread_index_in_simdgroup]],
                           uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float partial[8];
    device const half* xr = x + (ulong)row * H;
    float ss = 0.0f;
    for (uint i = lid; i < H; i += 256) { float v = float(xr[i]); ss += v * v; }
    ss = simd_sum(ss);
    if (lane == 0) partial[sg] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float tot = 0.0f;
    for (uint s = 0; s < 8; ++s) tot += partial[s];
    float r = rsqrt(tot / float(H) + EPS);
    for (uint i = lid; i < H; i += 256) y[(ulong)row * H + i] = half(float(xr[i]) * r * float(w[i]));
    (void)scratch;
}
/* taic-launch
math safe
workspace scratch M*H*2
launch rmsnorm_single grid=M,1,1 threads=256,1,1 workspace=scratch
*/
"""


def _bundle_entries(zero: bool = False) -> tuple[str, str, dict]:
    host_plan = {
        "schema_id": "tcenv.metal.host-tiling.1",
        "schema_version": 1,
        "writes": [
            {
                "workspace": "scratch",
                "kind": "linear",
                "start": 0,
                "step": 1,
                "count": "M",
                "stride": 1,
                "offset": 0,
                "dtype": "fp16",
            }
        ],
    }
    device = """#include <metal_stdlib>
using namespace metal;
kernel void rmsnorm_bundle(device const half* x [[buffer(0)]],
                           device const half* w [[buffer(1)]],
                           device half* y [[buffer(2)]],
                           device half* scratch [[buffer(3)]],
                           uint tg [[threadgroup_position_in_grid]],
                           uint lid [[thread_index_in_threadgroup]],
                           uint lane [[thread_index_in_simdgroup]],
                           uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float partial[8];
    uint row = uint(scratch[tg]);
    device const half* xr = x + (ulong)row * H;
    float ss = 0.0f;
    for (uint i = lid; i < H; i += 256) { float v = float(xr[i]); ss += v * v; }
    ss = simd_sum(ss);
    if (lane == 0) partial[sg] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float tot = 0.0f;
    for (uint s = 0; s < 8; ++s) tot += partial[s];
    float r = rsqrt(tot / float(H) + EPS);
    for (uint i = lid; i < H; i += 256) y[(ulong)row * H + i] = half(float(xr[i]) * r * float(w[i]));
}
/* taic-launch
math safe
workspace scratch M*H*2
launch rmsnorm_bundle grid=M,1,1 threads=256,1,1 workspace=scratch
*/
"""
    schedule = {
        "schema_version": 1,
        "source": "candidate",
        "stages": [
            {"kernel": "rmsnorm_bundle", "grid": ["M", 1, 1], "threads": [256, 1, 1], "workspace": [{"name": "scratch", "zero": zero}]},
        ],
    }
    return json.dumps(host_plan, sort_keys=True), device, schedule

def _bundle_candidate(zero: bool = False):
    from tcenv.track import CandidateArtifact, CandidateEntry

    host_plan, device, schedule = _bundle_entries(zero=zero)
    return CandidateArtifact(
        track_id="msl-bundle",
        entries=(
            CandidateEntry(path="host/tiling.json", role="host_tiling", media_type="application/json", data=host_plan.encode()),
            CandidateEntry(path="device/kernel.metal", role="device_kernel", media_type="text/plain", data=device.encode()),
            CandidateEntry(
                path="host/execution_geometry.json",
                role="schedule",
                media_type="application/json",
                data=json.dumps(schedule, sort_keys=True).encode(),
            ),
        ),
    )


def _bundle_lifecycle(candidate, contract, inside_timing: bool | None = None):
    from tcenv.target import DEFAULT_TARGET_ID
    from tcenv.track import CandidateLifecycle, HostStep, PrepareDeclaration

    def make(resolved):
        return CandidateLifecycle(
            track_id="msl-bundle",
            geometry=resolved,
            host_steps=(
                HostStep(
                    PrepareDeclaration(name="tiling", input_dependent=False, inside_timing=inside_timing),
                    role="host_tiling",
                ),
                HostStep(PrepareDeclaration(name="schedule", input_dependent=False), role="schedule"),
            ),
            candidate_sha256=candidate.candidate_sha256,
            target_id=DEFAULT_TARGET_ID,
            instance_key=contract.fingerprint(),
        )

    return make


def _fusion_source(threads: int) -> str:
    return f"""#include <metal_stdlib>
using namespace metal;
kernel void rmsnorm_silu(device const half* x [[buffer(0)]],
                         device const half* w [[buffer(1)]],
                         device half* y [[buffer(2)]],
                         device half* scratch [[buffer(3)]],
                         uint row [[threadgroup_position_in_grid]],
                         uint lid [[thread_index_in_threadgroup]],
                         uint lane [[thread_index_in_simdgroup]],
                         uint sg [[simdgroup_index_in_threadgroup]]) {{
    threadgroup float partial[8];
    if (lid < 8) partial[lid] = 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint stride = {threads};
    device const half* xr = x + (ulong)row * H;
    float ss = 0.0f;
    for (uint i = lid; i < H; i += stride) {{ float v = float(xr[i]); ss += v * v; }}
    ss = simd_sum(ss);
    if (lane == 0) partial[sg] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float tot = 0.0f;
    for (uint s = 0; s < 8; ++s) tot += partial[s];
    float r0 = rsqrt(tot / float(H) + EPS);
    for (uint i = lid; i < H; i += stride) {{
        float r = float(xr[i]) * r0 * float(w[i]);
        y[(ulong)row * H + i] = half(r * (1.0f / (1.0f + exp(-r))));
    }}
    (void)scratch;
}}
/* taic-launch
math safe
workspace scratch M*H*2
launch rmsnorm_silu grid=M,1,1 threads={threads},1,1 workspace=scratch
*/
"""


def _stable(result) -> dict:
    """@brief 不含计时值的重放身份。"""
    return {
        "candidate_sha256": result.candidate_sha256,
        "geometry_identity": result.geometry_identity,
        "bundle_digest": result.bundle_digest,
        "artifact_set_digest": result.artifact_set_digest,
        "native_digest": result.native_digest,
    }


@gpu
def test_a4_msl_single_rmsnorm_real_build_abi_correctness_and_timing(tmp_path):
    """@brief msl-single：真实构建、ABI、safety、correctness、计时，identity 与几何 artifact 绑定。"""
    from tcenv.backends.metal.track_adapter import MetalTrackBuilder, MetalTrackEvaluator, msl_single_geometry_resolver
    from tcenv.taskcontract.runner import evaluate_task

    locator, contract, tracks = _rmsnorm_task(tmp_path)
    source = _rmsnorm_single_source()
    result = evaluate_task(
        locator,
        track_id="msl-single",
        candidate_source=source,
        builder=MetalTrackBuilder(),
        evaluator=MetalTrackEvaluator(mode="full", reps=3, cases=("normal", "wide_range", "outlier")),
        geometry_resolver=msl_single_geometry_resolver(contract),
        track_registry_path=tracks,
    )
    report = result.report
    assert report["status"] == "ok", report.get("message")
    assert report["abi"]["ok"] is True
    assert report["correctness"]["ok"] is True
    assert report["benchmark"]["candidate"]["median_us"] > 0
    assert result.candidate_sha256 == hashlib.sha256(source.encode()).hexdigest()
    build = report["build"]
    assert build["shape"] == "full"
    assert build["candidate_sha256"] == result.candidate_sha256
    assert build["provenance"]["artifact_set_digest"] == result.artifact_set_digest
    entries = {entry["path"]: entry for entry in build["artifacts"]["entries"]}
    assert entries["kernel.metal"]["role"] == "SOURCE"
    assert entries["kernel.metal"]["sha256"] == result.candidate_sha256
    assert report["build"]["native"]["entry_path"] == "kernel.metal"
    assert "execution_geometry.json" in entries
    assert build["native"]["descriptor"]["execution_geometry"]["sha256"] == entries["execution_geometry.json"]["sha256"]
    assert result.effective_correctness_only is True
    assert result.execution is not None
    assert result.execution["workspace_bytes"]["scratch"] == 64 * 512 * 2
    assert result.execution["workspace_fills"] == []
    from tcenv.remote.execution_geometry import canonical_geometry_bytes
    from tcenv.target import DEFAULT_TARGET_ID

    canonical = _canonical_single(source, contract)
    assert canonical.bundle_digest == result.bundle_digest
    _record(
        "single.json",
        {
            "slice": "msl-single",
            "task": "rmsnorm_v2",
            "candidate_sha256": result.candidate_sha256,
            "presentation_sha256": result.presentation_sha256,
            "track": {
                "track_id": result.track["track_id"],
                "track_version": result.track["track_version"],
                "track_digest": result.track["track_digest"],
                "registry_digest": result.track["registry_digest"],
            },
            "worker": result.execution["worker"],
            "geometry_identity": result.geometry_identity,
            "bundle_digest": result.bundle_digest,
            "artifact_set_digest": result.artifact_set_digest,
            "native_digest": result.native_digest,
            "geometry_artifact_sha256": hashlib.sha256(canonical_geometry_bytes(canonical)).hexdigest(),
            "canonical_bundle": canonical.document(),
            "dispatch": _observed(result),
            "provider": result.execution["provider"],
            "bindings": result.execution["bindings"],
            "workspace_bytes": result.execution["workspace_bytes"],
            "workspace_zero": result.execution["workspace_zero"],
            "workspace_fills": result.execution["workspace_fills"],
            "status": report["status"],
            "abi_ok": report["abi"]["ok"],
            "correctness_ok": report["correctness"]["ok"],
            "target": DEFAULT_TARGET_ID,
        },
    )


@gpu
def test_a4_msl_bundle_real_host_tiling_and_device_stage(tmp_path):
    """@brief msl-bundle：候选 Host Tiling 先真实执行，device stage 依赖其结果，几何 artifact 与 dispatch 一致。"""
    from tcenv.backends.metal.track_adapter import MetalTrackBuilder, MetalTrackEvaluator
    from tcenv.taskcontract.runner import evaluate_task
    from tcenv.track import (
        CanonicalBundle,
        candidate_geometry,
        dispatch_parity,
        geometry_pair,
        select_geometry,
        selection_parity,
    )
    from tcenv.target import DEFAULT_TARGET_ID, load_target
    from tcenv.track.abi import abi_limits_from_target, derive_tensor_memory_contract

    locator, contract, tracks = _rmsnorm_task(tmp_path)
    candidate = _bundle_candidate()
    result = evaluate_task(
        locator,
        track_id="msl-bundle",
        candidate=candidate,
        builder=MetalTrackBuilder(),
        evaluator=MetalTrackEvaluator(mode="full", reps=3, cases=("normal", "wide_range", "outlier")),
        lifecycle_factory=_bundle_lifecycle(candidate, contract),
        track_registry_path=tracks,
    )
    report = result.report
    assert report["status"] == "ok"
    assert report["abi"]["ok"] is True
    assert report["correctness"]["ok"] is True
    assert "launch rmsnorm_bundle" in report["compile"]["launch"]
    assert "rmsnorm_tiling" not in report["compile"]["launch"]
    entries = {entry["path"]: entry for entry in report["build"]["artifacts"]["entries"]}
    assert entries["host/tiling.json"]["role"] == "AUXILIARY"
    assert entries["device/kernel.metal"]["role"] == "AUXILIARY"
    assert entries["host/execution_geometry.json"]["role"] == "AUXILIARY"
    assert entries["candidate_manifest.json"]["role"] == "SOURCE"
    assert report["build"]["native"]["entry_path"] == "device/kernel.metal"
    assert result.execution is not None
    assert result.execution["host_tiling"] is not None
    assert result.execution["host_runs"], "Host Tiling plan was never executed"
    assert result.execution["host_runs"][0]["writes"] == 64 * 2
    assert report["benchmark"]["candidate"]["median_us"] > 0
    assert result.execution["workspace_bytes"]["scratch"] == 64 * 512 * 2
    assert result.execution["workspace_zero"] == {"scratch": False}
    assert result.execution["workspace_fills"] == []
    assert result.execution["provider"] == "candidate"

    geometry = candidate_geometry(candidate)
    abi = derive_tensor_memory_contract(contract.instance, target=load_target(DEFAULT_TARGET_ID))
    limits = abi_limits_from_target(load_target(DEFAULT_TARGET_ID))
    pair = geometry_pair(geometry, None, instance=contract.instance, contract=abi, limits=limits)
    chosen = select_geometry(pair)
    canonical = CanonicalBundle(candidate=candidate, pair=pair, lifecycle=_bundle_lifecycle(candidate, contract)(chosen.geometry))
    assert canonical.bundle_digest == result.bundle_digest
    observed = _observed(result)
    assert list(dict.fromkeys(record["kernel"] for record in result.execution["dispatch"])) == ["rmsnorm_bundle"]
    assert all(record["provider"] == "candidate" for record in result.execution["dispatch"])
    assert dispatch_parity(chosen.geometry, observed) == []
    assert selection_parity(chosen, provider=result.execution["provider"], observed=observed) == []
    assert dispatch_parity(chosen.geometry, [("rmsnorm_bundle", (32, 1, 1), (256, 1, 1), 0, ("scratch",))])
    assert selection_parity(chosen, provider="fallback", observed=observed)
    from tcenv.remote.execution_geometry import canonical_geometry_bytes

    _record(
        "bundle.json",
        {
            "slice": "msl-bundle",
            "task": "rmsnorm_v2",
            "candidate_sha256": result.candidate_sha256,
            "geometry_identity": result.geometry_identity,
            "bundle_digest": result.bundle_digest,
            "artifact_set_digest": result.artifact_set_digest,
            "native_digest": result.native_digest,
            "geometry_artifact_sha256": hashlib.sha256(canonical_geometry_bytes(canonical)).hexdigest(),
            "canonical_bundle": canonical.document(),
            "entries": sorted(entries),
            "dispatch": observed,
            "provider": result.execution["provider"],
            "bindings": result.execution["bindings"],
            "workspace_bytes": result.execution["workspace_bytes"],
            "workspace_zero": result.execution["workspace_zero"],
            "workspace_fills": result.execution["workspace_fills"],
            "workspace_probe": result.execution["workspace_probe"],
            "host_runs": result.execution["host_runs"],
            "status": report["status"],
            "abi_ok": report["abi"]["ok"],
            "correctness_ok": report["correctness"]["ok"],
        },
    )


@gpu
def test_a4_new_operator_is_discovered_and_evaluated_without_a_central_list(tmp_path):
    """@brief 全新算子只有标准合同+reference：自动发现、真实 correctness-only 评测。"""
    from tasks import get_task
    from tcenv.backends.metal.track_adapter import MetalTrackBuilder, MetalTrackEvaluator, msl_single_geometry_resolver
    from tcenv.taskcontract import discover_standard_tasks
    from tcenv.taskcontract.runner import evaluate_task
    from tcenv.track import REPO_ROOT

    found = [locator for locator in discover_standard_tasks(REPO_ROOT / "tasks", code_root=REPO_ROOT) if locator.op_name == "sigmoid_shift"]
    assert len(found) == 1
    locator = found[0]
    contract = locator.load()
    # The v1 registry has no central entry for the new operator.  Since A5/A6 dispatch by
    # ``spec_version`` this is a *typed* refusal that points at the v2 loader, not an opaque
    # KeyError; either way the new operator is never assembled from a v1 name list.
    from tcenv.contract_errors import TaskContractError

    with pytest.raises(TaskContractError) as refused:
        get_task("sigmoid_shift", "fp16", "small")
    assert "v2 standard entry" in str(refused.value)
    source = """#include <metal_stdlib>
using namespace metal;
kernel void sigmoid_shift(device const half* x [[buffer(0)]],
                          device half* y [[buffer(1)]],
                          uint i [[thread_position_in_grid]]) {
    if (i < N) {
        float v = float(x[i]);
        y[i] = half(1.0f / (1.0f + exp(-v)) + 1.0f);
    }
}
/* taic-launch
math safe
launch sigmoid_shift grid=N/256,1,1 threads=256,1,1
*/
"""
    result = evaluate_task(
        locator,
        track_id="msl-single",
        candidate_source=source,
        builder=MetalTrackBuilder(),
        evaluator=MetalTrackEvaluator(mode="quick", cases=("normal", "wide_range", "outlier")),
        geometry_resolver=msl_single_geometry_resolver(contract),
        track_registry_path=_tracks(tmp_path / "tracks.json"),
    )
    assert result.effective_correctness_only is True
    assert result.report["status"] == "ok"
    assert result.report["eligible"] is True
    assert result.report["correctness"]["ok"] is True
    from tcenv.remote.execution_geometry import canonical_geometry_bytes

    canonical = _canonical_single(source, contract)
    assert canonical.bundle_digest == result.bundle_digest
    _record(
        "newop.json",
        {
            "slice": "new-operator",
            "task": "sigmoid_shift",
            "task_pack_digest": result.task_pack_digest,
            "candidate_sha256": result.candidate_sha256,
            "geometry_identity": result.geometry_identity,
            "bundle_digest": result.bundle_digest,
            "artifact_set_digest": result.artifact_set_digest,
            "native_digest": result.native_digest,
            "geometry_artifact_sha256": hashlib.sha256(canonical_geometry_bytes(canonical)).hexdigest(),
            "canonical_bundle": canonical.document(),
            "dispatch": _observed(result),
            "provider": result.execution["provider"],
            "bindings": result.execution["bindings"],
            "workspace_bytes": result.execution["workspace_bytes"],
            "workspace_zero": result.execution["workspace_zero"],
            "workspace_fills": result.execution["workspace_fills"],
            "status": result.report["status"],
            "effective_correctness_only": result.effective_correctness_only,
        },
    )


@gpu
def test_a4_fusion_geometry_comparison_replay_and_evidence(tmp_path):
    """@brief 同一融合 ABI 的两个几何：candidate/geometry/build 摘要分离，可 replay，dispatch/fallback 偏离可见。"""
    from tcenv.backends.metal.track_adapter import MetalTrackBuilder, MetalTrackEvaluator, msl_single_geometry_resolver
    from tcenv.taskcontract.runner import evaluate_task

    locator, contract, tracks = _fusion_task(tmp_path)
    resolver = msl_single_geometry_resolver(contract)
    runs = {}
    for threads in (128, 256):
        runs[threads] = evaluate_task(
            locator,
            track_id="msl-single",
            candidate_source=_fusion_source(threads),
            builder=MetalTrackBuilder(),
            evaluator=MetalTrackEvaluator(mode="quick", cases=("normal", "wide_range", "outlier")),
            geometry_resolver=resolver,
            track_registry_path=tracks,
        )
        assert runs[threads].report["status"] == "ok"
        assert runs[threads].report["correctness"]["ok"] is True
        canonical = _canonical_single(_fusion_source(threads), contract)
        assert canonical.bundle_digest == runs[threads].bundle_digest
    a, b = runs[128], runs[256]
    assert a.candidate_sha256 != b.candidate_sha256
    assert a.locator["deployment_pack_digest"] == b.locator["deployment_pack_digest"]
    assert a.geometry_identity != b.geometry_identity
    assert a.bundle_digest != b.bundle_digest
    assert a.artifact_set_digest != b.artifact_set_digest
    replay = evaluate_task(
        locator,
        track_id="msl-single",
        candidate_source=_fusion_source(128),
        builder=MetalTrackBuilder(),
        evaluator=MetalTrackEvaluator(mode="quick", cases=("normal", "wide_range", "outlier")),
        geometry_resolver=resolver,
        track_registry_path=tracks,
    )
    assert _stable(replay) == _stable(a)
    assert replay.execution is not None
    assert {tuple(r["threads"]) for r in replay.execution["dispatch"]} == {(128, 1, 1)}
    assert _observed(replay) == _observed(a)

    from tcenv.remote.execution_geometry import canonical_geometry_bytes

    _record(
        "fusion.json",
        {
            "slice": "fusion",
            "task": "rmsnorm_silu",
            "task_pack_digest": a.task_pack_digest,
            "candidates": {
                str(threads): {
                    **_stable(run),
                    "geometry_artifact_sha256": hashlib.sha256(
                        canonical_geometry_bytes(_canonical_single(_fusion_source(threads), contract))
                    ).hexdigest(),
                    "canonical_bundle": _canonical_single(_fusion_source(threads), contract).document(),
                    "dispatch": _observed(run),
                    "provider": run.execution["provider"],
                    "bindings": run.execution["bindings"],
                    "workspace_bytes": run.execution["workspace_bytes"],
                    "workspace_zero": run.execution["workspace_zero"],
                    "workspace_fills": run.execution["workspace_fills"],
                }
                for threads, run in ((128, a), (256, b))
            },
            "replay_equal": _stable(replay) == _stable(a),
        },
    )


def test_a4_msl_single_refuses_to_invent_a_geometry_without_a_launch_block(tmp_path):
    """@brief 负例：msl-single 缺 ``taic-launch`` 时拒绝，而不是由 harness 猜 grid/threads。"""
    from tcenv.backends.metal.track_adapter import MetalTrackBuilder, MetalTrackEvaluator, msl_single_geometry_resolver
    from tcenv.contract_errors import GeometryError
    from tcenv.taskcontract.runner import evaluate_task

    locator, contract, tracks = _rmsnorm_task(tmp_path)
    bare = """#include <metal_stdlib>
using namespace metal;
kernel void rmsnorm_single(device const half* x [[buffer(0)]],
                           device const half* w [[buffer(1)]],
                           device half* y [[buffer(2)]],
                           uint row [[threadgroup_position_in_grid]]) {
    y[row] = x[row];
}
"""
    with pytest.raises(GeometryError, match="taic-launch"):
        evaluate_task(
            locator,
            track_id="msl-single",
            candidate_source=bare,
            builder=MetalTrackBuilder(),
            evaluator=MetalTrackEvaluator(mode="quick"),
            geometry_resolver=msl_single_geometry_resolver(contract),
            track_registry_path=tracks,
        )


def test_a4_a_fallback_or_dispatch_deviation_is_a_mismatch():
    """@brief 负例：几何/提供者/逐段 dispatch 任一偏离都必须留下可见的 mismatch。"""
    from tcenv.target import DEFAULT_TARGET_ID, load_target
    from tcenv.track import (
        candidate_geometry,
        dispatch_parity,
        geometry_pair,
        select_geometry,
        selection_parity,
    )
    from tcenv.track.abi import abi_limits_from_target, derive_tensor_memory_contract

    # Build the same real bundle contract without a device: the mismatch check is pure CPU.
    from tcenv.taskcontract import discover_standard_tasks
    from tcenv.track import REPO_ROOT

    # Use a throwaway v2 task only to obtain the semantic instance/ABI.
    import tempfile

    with tempfile.TemporaryDirectory() as scratch:
        root = Path(scratch) / "tasks"
        write_task(root, op_name="rmsnorm_v2", opspec=rmsnorm_workspace_opspec("rmsnorm_v2"))
        contract = discover_standard_tasks(root, code_root=REPO_ROOT)[0].load()
        geometry = candidate_geometry(_bundle_candidate())
        abi = derive_tensor_memory_contract(contract.instance, target=load_target(DEFAULT_TARGET_ID))
        pair = geometry_pair(
            geometry,
            None,
            instance=contract.instance,
            contract=abi,
            limits=abi_limits_from_target(load_target(DEFAULT_TARGET_ID)),
        )
        selection = select_geometry(pair)
    observed = [("rmsnorm_bundle", (64, 1, 1), (256, 1, 1), 0, ("scratch",))]
    assert dispatch_parity(selection.geometry, observed) == []
    assert selection_parity(selection, provider="candidate", observed=observed) == []
    wrong_grid = [("rmsnorm_bundle", (32, 1, 1), (256, 1, 1), 0, ("scratch",))]
    assert dispatch_parity(selection.geometry, wrong_grid)
    assert selection_parity(selection, provider="fallback", observed=observed)
    assert selection_parity(selection, provider="candidate", observed=wrong_grid)




@gpu
def test_a4_derived_scalar_is_set_by_name_and_inactive_when_stage_is_not_dispatched(tmp_path):
    """@brief 真实 reflection 把 Derived Scalar 按名 setBytes；when=false 的 stage 不进入 dispatch。"""
    from tcenv.backends.metal.track_adapter import MetalTrackBuilder, MetalTrackEvaluator
    from tcenv.taskcontract.runner import evaluate_task
    from tcenv.target import DEFAULT_TARGET_ID, load_target
    from tcenv.track import ExecutionGeometry, GeometryStage, abi_limits_from_target, derive_tensor_memory_contract

    locator, contract, tracks = _rmsnorm_task(tmp_path)
    abi = derive_tensor_memory_contract(contract.instance, target=load_target(DEFAULT_TARGET_ID))
    limits = abi_limits_from_target(load_target(DEFAULT_TARGET_ID))
    source = """#include <metal_stdlib>
using namespace metal;
kernel void rmsnorm_derived(device const half* x [[buffer(0)]],
                            device const half* w [[buffer(1)]],
                            device half* y [[buffer(2)]],
                            constant uint& BLOCKS [[buffer(3)]],
                            uint row [[threadgroup_position_in_grid]],
                            uint lid [[thread_index_in_threadgroup]],
                            uint lane [[thread_index_in_simdgroup]],
                            uint sg [[simdgroup_index_in_threadgroup]]) {
    if (BLOCKS != 2) { return; }
    threadgroup float partial[8];
    device const half* xr = x + (ulong)row * H;
    float ss = 0.0f;
    for (uint i = lid; i < H; i += 256) { float v = float(xr[i]); ss += v * v; }
    ss = simd_sum(ss);
    if (lane == 0) partial[sg] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float tot = 0.0f;
    for (uint s = 0; s < 8; ++s) tot += partial[s];
    float r = rsqrt(tot / float(H) + EPS);
    for (uint i = lid; i < H; i += 256) y[(ulong)row * H + i] = half(float(xr[i]) * r * float(w[i]));
}
/* taic-launch
math safe
workspace scratch M*H*2
launch rmsnorm_derived grid=M,1,1 threads=256,1,1 workspace=scratch
*/
"""
    raw_geometry = ExecutionGeometry(
        source="candidate",
        stages=(
            GeometryStage(
                kernel="rmsnorm_derived",
                grid=("M", 1, 1),
                threads=(256, 1, 1),
                workspace=({"name": "scratch", "zero": False},),
                derived_scalars=(("BLOCKS", 2),),
            ),
            GeometryStage(
                kernel="rmsnorm_never",
                grid=(1, 1, 1),
                threads=(1, 1, 1),
                when={"eq": ["H", 0]},
            ),
        ),
    )
    result = evaluate_task(
        locator,
        track_id="msl-single",
        candidate_source=source,
        builder=MetalTrackBuilder(),
        evaluator=MetalTrackEvaluator(mode="quick", cases=("normal",)),
        geometry_resolver=lambda _candidate: raw_geometry,
        track_registry_path=tracks,
        max_buffer_length=limits.max_buffer_length,
    )
    assert result.report["status"] == "ok", result.report.get("message")
    assert result.report["correctness"]["ok"] is True
    assert result.execution is not None
    bound = result.execution["bindings"][0]["arguments"]
    assert any(item["role"] == "derived_scalar" and item["name"] == "BLOCKS" and item["value"] == 2 for item in bound)
    assert [record["kernel"] for record in result.execution["dispatch"] if record["kernel"] == "rmsnorm_never"] == []
    assert any(record["kernel"] == "rmsnorm_derived" for record in result.execution["dispatch"])
    _record(
        "derived.json",
        {
            "slice": "derived-scalar-when",
            "task": "rmsnorm_v2",
            "candidate_sha256": result.candidate_sha256,
            "geometry_identity": result.geometry_identity,
            "dispatch": _observed(result),
            "bindings": result.execution["bindings"],
            "provider": result.execution["provider"],
            "workspace_bytes": result.execution["workspace_bytes"],
            "workspace_zero": result.execution["workspace_zero"],
            "workspace_fills": result.execution["workspace_fills"],
            "inactive_kernels": [
                record["kernel"] for record in result.execution["dispatch"] if record["kernel"] == "rmsnorm_never"
            ],
        },
    )


@gpu
def test_a4_read_write_initial_values_are_restored_on_every_trial(tmp_path):
    """@brief 真实 read_write：y/acc 初值在 out1/out2/out3 前恢复，终值参与差分。"""
    from tcenv.backends.metal.track_adapter import MetalTrackBuilder, MetalTrackEvaluator, msl_single_geometry_resolver
    from tcenv.taskcontract import discover_standard_tasks
    from tcenv.taskcontract.runner import evaluate_task
    from tcenv.track import REPO_ROOT

    root = tmp_path / "tasks"
    write_task(
        root,
        op_name="rmsnorm_rw",
        opspec=rmsnorm_read_write_opspec("rmsnorm_rw"),
        reference=RMSNORM_READ_WRITE_GEN_REFERENCE,
    )
    locator = discover_standard_tasks(root, code_root=REPO_ROOT)[0]
    contract = locator.load()
    source = """#include <metal_stdlib>
using namespace metal;
kernel void rmsnorm_rw(device const half* x [[buffer(0)]],
                       device const half* w [[buffer(1)]],
                       device half* y [[buffer(2)]],
                       device half* acc [[buffer(3)]],
                       uint row [[threadgroup_position_in_grid]],
                       uint lid [[thread_index_in_threadgroup]],
                       uint lane [[thread_index_in_simdgroup]],
                       uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float partial[8];
    device const half* xr = x + (ulong)row * H;
    float ss = 0.0f;
    for (uint i = lid; i < H; i += 256) { float v = float(xr[i]); ss += v * v; }
    ss = simd_sum(ss);
    if (lane == 0) partial[sg] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float tot = 0.0f;
    for (uint s = 0; s < 8; ++s) tot += partial[s];
    float r = rsqrt(tot / float(H) + EPS);
    for (uint i = lid; i < H; i += 256) {
        ulong j = (ulong)row * H + i;
        float y0 = float(y[j]);
        float a0 = float(acc[j]);
        y[j] = half(y0 + float(xr[i]) * r * float(w[i]));
        acc[j] = half(a0 + y0);
    }
}
/* taic-launch
math safe
launch rmsnorm_rw grid=M,1,1 threads=256,1,1
*/
"""
    result = evaluate_task(
        locator,
        track_id="msl-single",
        candidate_source=source,
        builder=MetalTrackBuilder(),
        evaluator=MetalTrackEvaluator(mode="quick", cases=("normal", "wide_range", "outlier")),
        geometry_resolver=msl_single_geometry_resolver(contract),
        track_registry_path=_tracks(tmp_path / "tracks.json"),
    )
    assert result.report["status"] == "ok"
    assert result.report["correctness"]["ok"] is True
    assert result.execution is not None
    actual = _observed(result)
    assert actual and actual[0][0] == "rmsnorm_rw"
    _record(
        "readwrite.json",
        {
            "slice": "read-write",
            "task": "rmsnorm_rw",
            "candidate_sha256": result.candidate_sha256,
            "geometry_identity": result.geometry_identity,
            "dispatch": actual,
            "bindings": result.execution["bindings"],
            "provider": result.execution["provider"],
            "workspace_bytes": result.execution["workspace_bytes"],
            "workspace_zero": result.execution["workspace_zero"],
            "workspace_fills": result.execution["workspace_fills"],
        },
    )


@gpu
def test_a4_actual_runtime_dispatch_cannot_satisfy_fallback_or_a_changed_grid(tmp_path):
    """@brief 真实 runtime 记录与 fallback/不同 grid 比较必须 mismatch；不是重解析源文本。"""
    from tcenv.backends.metal.track_adapter import MetalTrackBuilder, MetalTrackEvaluator, msl_single_geometry_resolver
    from tcenv.taskcontract.runner import evaluate_task
    from tcenv.track import (
        candidate_from_single_source,
        dispatch_parity,
        geometry_pair,
        select_geometry,
        selection_parity,
    )
    from tcenv.target import DEFAULT_TARGET_ID, load_target
    from tcenv.track.abi import abi_limits_from_target, derive_tensor_memory_contract

    locator, contract, tracks = _rmsnorm_task(tmp_path)
    source = _rmsnorm_single_source()
    result = evaluate_task(
        locator,
        track_id="msl-single",
        candidate_source=source,
        builder=MetalTrackBuilder(),
        evaluator=MetalTrackEvaluator(mode="quick", cases=("normal",)),
        geometry_resolver=msl_single_geometry_resolver(contract),
        track_registry_path=tracks,
    )
    assert result.execution is not None
    actual = _observed(result)
    assert result.execution["provider"] == "candidate"
    assert all(record["provider"] == "candidate" for record in result.execution["dispatch"])
    resolver = msl_single_geometry_resolver(contract)
    geometry = resolver(candidate_from_single_source(source))
    pair = geometry_pair(
        geometry,
        None,
        instance=contract.instance,
        contract=derive_tensor_memory_contract(contract.instance, target=load_target(DEFAULT_TARGET_ID)),
        limits=abi_limits_from_target(load_target(DEFAULT_TARGET_ID)),
    )
    chosen = select_geometry(pair)
    assert dispatch_parity(chosen.geometry, actual) == []
    # A worker that used fallback while evidence selected candidate must mismatch even if the
    # per-stage signature happens to agree.
    assert selection_parity(chosen, provider="fallback", observed=actual)
    assert dispatch_parity(chosen.geometry, [("rmsnorm_single", (32, 1, 1), (256, 1, 1), 0, ("scratch",))])


def test_a4_workspace_size_mismatch_is_refused_before_allocation(tmp_path):
    """@brief 负例：launch 的 ``M*2`` 不能覆盖 ABI ``(M,H)`` workspace；按 ABI span 拒绝。"""
    from tcenv.backends.metal.launch import LaunchSpec, StageDecl, WorkspaceDecl
    from tcenv.backends.metal.v2_runtime import workspace_size_problems
    from tcenv.target import DEFAULT_TARGET_ID, load_target
    from tcenv.track import ExecutionGeometry, GeometryStage, abi_limits_from_target, derive_tensor_memory_contract, resolve_geometry

    locator, contract, _tracks_path = _rmsnorm_task(tmp_path)
    abi = derive_tensor_memory_contract(contract.instance, target=load_target(DEFAULT_TARGET_ID))
    raw = ExecutionGeometry(
        source="candidate",
        stages=(
            GeometryStage(
                kernel="rmsnorm_single",
                grid=("M", 1, 1),
                threads=(256, 1, 1),
                workspace=({"name": "scratch", "zero": False},),
            ),
        ),
    )
    geometry = resolve_geometry(
        raw,
        instance=contract.instance,
        contract=abi,
        limits=abi_limits_from_target(load_target(DEFAULT_TARGET_ID)),
    )
    spec = LaunchSpec(
        "safe",
        (WorkspaceDecl("scratch", 128, False),),
        (StageDecl("rmsnorm_single", (64, 1, 1), (256, 1, 1), 0, ("scratch",)),),
    )
    problems = workspace_size_problems(abi, geometry, spec)
    assert any("launch declares 128 bytes" in item and "65536" in item for item in problems)


def test_a4_host_tiling_plan_is_closed_and_required_not_a_renamed_kernel(tmp_path):
    """@brief 负例：Host Tiling entry 必须是闭集 host plan；空计划不能静默跳过。"""
    from tcenv.backends.metal.v2_runtime import HostTilingPlan
    from tcenv.contract_errors import TaskContractError
    from tcenv.taskcontract import discover_standard_tasks
    from tcenv.track import CandidateArtifact, CandidateEntry, REPO_ROOT

    locator, contract, _tracks_path = _rmsnorm_task(tmp_path)
    candidate = _bundle_candidate()
    assert HostTilingPlan.from_candidate(candidate, contract) is not None
    bad = CandidateArtifact(
        track_id="msl-bundle",
        entries=tuple(
            CandidateEntry(
                path=entry.path,
                role=entry.role,
                media_type=entry.media_type,
                data=json.dumps({"schema_id": "tcenv.metal.host-tiling.1", "schema_version": 1, "writes": []}).encode(),
            )
            if entry.role == "host_tiling"
            else entry
            for entry in candidate.entries
        ),
    )
    with pytest.raises(TaskContractError, match="at least one write"):
        HostTilingPlan.from_candidate(bad, contract)


@gpu
def test_a4_a_bundle_that_ships_host_work_cannot_skip_its_host_step(tmp_path):
    """@brief 负例：候选带 host_tiling entry 但 lifecycle 没有 host step 时拒绝，不允许跳过 Host Tiling。"""
    from dataclasses import replace

    from tcenv.backends.metal.track_adapter import MetalTrackBuilder, MetalTrackEvaluator
    from tcenv.contract_errors import TaskContractError
    from tcenv.taskcontract.runner import evaluate_task

    locator, contract, tracks = _rmsnorm_task(tmp_path)
    candidate = _bundle_candidate()
    full_lifecycle = _bundle_lifecycle(candidate, contract)

    def no_host_step(resolved):
        lifecycle = full_lifecycle(resolved)
        return replace(lifecycle, host_steps=tuple(step for step in lifecycle.host_steps if step.role != "host_tiling"))

    with pytest.raises(TaskContractError, match="host_tiling"):
        evaluate_task(
            locator,
            track_id="msl-bundle",
            candidate=candidate,
            builder=MetalTrackBuilder(),
            evaluator=MetalTrackEvaluator(mode="quick", cases=("normal",)),
            lifecycle_factory=no_host_step,
            track_registry_path=tracks,
        )



@gpu
def test_a4_declared_zero_workspace_is_cleared_inside_timing_and_host_tiling_restores_it(tmp_path):
    """@brief zero:true 的 workspace 在计时内清零；Host Tiling 在清零之后、首个 device stage 之前执行。"""
    from tcenv.backends.metal.track_adapter import MetalTrackBuilder, MetalTrackEvaluator
    from tcenv.taskcontract.runner import evaluate_task
    from tcenv.track import candidate_geometry

    locator, contract, tracks = _rmsnorm_task(tmp_path)
    candidate = _bundle_candidate(zero=True)
    result = evaluate_task(
        locator,
        track_id="msl-bundle",
        candidate=candidate,
        builder=MetalTrackBuilder(),
        evaluator=MetalTrackEvaluator(mode="full", reps=3, cases=("normal",)),
        geometry_resolver=lambda _candidate: candidate_geometry(candidate),
        lifecycle_factory=_bundle_lifecycle(candidate, contract, inside_timing=True),
        track_registry_path=tracks,
    )
    assert result.report["status"] == "ok"
    assert result.report["correctness"]["ok"] is True
    assert result.report["benchmark"]["candidate"]["median_us"] > 0
    assert result.execution is not None
    assert result.execution["workspace_zero"] == {"scratch": True}
    assert result.execution["workspace_fills"] == ["scratch"]
    assert result.execution["host_runs"]
    assert result.execution["host_runs"][0]["writes"] == 64 * 2
    probe = bytes.fromhex(result.execution["workspace_probe"]["scratch"])
    assert struct.unpack("<8e", probe[:16]) == tuple(float(i) for i in range(8))


@gpu
def test_a4_real_runtime_refuses_a_fallback_geometry_against_candidate_selection(tmp_path):
    """@brief 负例：真实 worker 用 fallback geometry 时，runtime 记录到的 provider 与 selection 不一致。"""
    from dataclasses import replace

    from tcenv.backends.metal.track_adapter import MetalTrackBuilder, MetalTrackEvaluator, msl_single_geometry_resolver
    from tcenv.taskcontract.runner import evaluate_task

    class FallbackEvaluator(MetalTrackEvaluator):
        """@brief 模拟 worker 把已选候选几何替换成同 stage 的 fallback。"""

        def evaluate(self, request):
            fallback = replace(request.geometry, source="fallback")
            return super().evaluate(replace(request, geometry=fallback))

    locator, contract, tracks = _rmsnorm_task(tmp_path)
    # In-process: the substituted geometry has to reach MetalV2Runtime.  A subprocess
    # rebuilds the candidate launch and would not see this evaluator's replacement.
    evaluator = FallbackEvaluator(mode="quick", cases=("normal",), worker_child=True)
    result = evaluate_task(
        locator,
        track_id="msl-single",
        candidate_source=_rmsnorm_single_source(),
        builder=MetalTrackBuilder(),
        evaluator=evaluator,
        geometry_resolver=msl_single_geometry_resolver(contract),
        track_registry_path=tracks,
    )
    # The fallback really reached the runtime; the runtime recorded provider=fallback and
    # selection parity failed.  The report is a normal runtime_error with the attached build,
    # not a fabricated identity-drift exception.
    assert result.report["status"] == "runtime_error"
    assert result.report["build"]["provenance"]["artifact_set_digest"] == result.artifact_set_digest
    assert result.report["build"]["artifacts"]["artifact_set_digest"] == result.artifact_set_digest
    assert result.execution is not None
    assert result.execution["provider"] == "fallback"
    assert result.execution["dispatch"] == []


def _wrong_device_source(device: str) -> str:
    """@brief A4 negative fixture: compile a device kernel whose RMSNorm result is deterministically wrong."""
    needle = "y[(ulong)row * H + i] = half(float(xr[i]) * r * float(w[i]));"
    assert needle in device
    return device.replace(needle, "y[(ulong)row * H + i] = half(0.0f);")


def _unwritten_device_source(device: str) -> str:
    """@brief A4 negative fixture: compile a device kernel that never writes its output."""
    needle = "y[(ulong)row * H + i] = half(float(xr[i]) * r * float(w[i]));"
    assert needle in device
    return device.replace(needle, "(void)r;")


@gpu
def test_a4_incorrect_single_is_reported_incorrect_with_the_attached_build(tmp_path):
    """@brief 负例：确定性错误候选应是 incorrect，不能伪装成 artifact_set_digest 身份漂移。"""
    from tcenv.backends.metal.track_adapter import MetalTrackBuilder, MetalTrackEvaluator, msl_single_geometry_resolver
    from tcenv.taskcontract.runner import evaluate_task

    locator, contract, tracks = _rmsnorm_task(tmp_path)
    source = _wrong_device_source(_rmsnorm_single_source())
    result = evaluate_task(
        locator,
        track_id="msl-single",
        candidate_source=source,
        builder=MetalTrackBuilder(),
        evaluator=MetalTrackEvaluator(mode="quick", cases=("normal",)),
        geometry_resolver=msl_single_geometry_resolver(contract),
        track_registry_path=tracks,
    )
    assert result.report["status"] == "incorrect"
    assert result.report["correctness"]["ok"] is False
    assert result.report["build"]["provenance"]["artifact_set_digest"] == result.artifact_set_digest
    assert result.report["build"]["artifacts"]["artifact_set_digest"] == result.artifact_set_digest


@gpu
def test_a4_incorrect_bundle_is_reported_incorrect_with_the_attached_build(tmp_path):
    """@brief 负例：bundle 的确定性错误也走同一 verdict，而不是身份漂移。"""
    from tcenv.backends.metal.track_adapter import MetalTrackBuilder, MetalTrackEvaluator
    from tcenv.taskcontract.runner import evaluate_task
    from tcenv.track import candidate_geometry

    locator, contract, tracks = _rmsnorm_task(tmp_path)
    candidate = _bundle_candidate()
    wrong_entries = tuple(
        CandidateEntry(
            path=entry.path,
            role=entry.role,
            media_type=entry.media_type,
            data=_wrong_device_source(entry.data.decode()).encode() if entry.role == "device_kernel" else entry.data,
        )
        for entry in candidate.entries
    )
    wrong = CandidateArtifact(track_id="msl-bundle", entries=wrong_entries)
    result = evaluate_task(
        locator,
        track_id="msl-bundle",
        candidate=wrong,
        builder=MetalTrackBuilder(),
        evaluator=MetalTrackEvaluator(mode="quick", cases=("normal",)),
        geometry_resolver=lambda _candidate: candidate_geometry(wrong),
        lifecycle_factory=_bundle_lifecycle(wrong, contract),
        track_registry_path=tracks,
    )
    assert result.report["status"] == "incorrect"
    assert result.report["build"]["provenance"]["artifact_set_digest"] == result.artifact_set_digest
    assert result.report["build"]["artifacts"]["artifact_set_digest"] == result.artifact_set_digest


@gpu
def test_a4_unwritten_output_is_reported_unsafe_with_the_attached_build(tmp_path):
    """@brief 负例：不写输出的候选应是 unsafe（三 trial 依赖），不是身份漂移。"""
    from tcenv.backends.metal.track_adapter import MetalTrackBuilder, MetalTrackEvaluator, msl_single_geometry_resolver
    from tcenv.taskcontract.runner import evaluate_task

    locator, contract, tracks = _rmsnorm_task(tmp_path)
    source = _unwritten_device_source(_rmsnorm_single_source())
    result = evaluate_task(
        locator,
        track_id="msl-single",
        candidate_source=source,
        builder=MetalTrackBuilder(),
        evaluator=MetalTrackEvaluator(mode="quick", cases=("normal",)),
        geometry_resolver=msl_single_geometry_resolver(contract),
        track_registry_path=tracks,
    )
    assert result.report["status"] == "unsafe"
    assert result.report["build"]["provenance"]["artifact_set_digest"] == result.artifact_set_digest
    assert result.report["build"]["artifacts"]["artifact_set_digest"] == result.artifact_set_digest


@gpu
def test_a4_benchmark_checks_guards_after_reps_and_catches_a_second_launch_oob(tmp_path):
    """@brief 负例：第二次 benchmark launch 才越界；reps 后 guard 检查必须仍能看见，不能擦除。"""
    from tcenv.backends.metal.track_adapter import MetalTrackBuilder, MetalTrackEvaluator, msl_single_geometry_resolver
    from tcenv.taskcontract.runner import evaluate_task

    locator, contract, tracks = _rmsnorm_task(tmp_path)
    source = """#include <metal_stdlib>
using namespace metal;
kernel void rmsnorm_second_oob(device const half* x [[buffer(0)]],
                               device const half* w [[buffer(1)]],
                               device half* y [[buffer(2)]],
                               device half* scratch [[buffer(3)]],
                               uint row [[threadgroup_position_in_grid]],
                               uint lid [[thread_index_in_threadgroup]],
                               uint lane [[thread_index_in_simdgroup]],
                               uint sg [[simdgroup_index_in_threadgroup]]) {
    device uint* flags = reinterpret_cast<device uint*>(scratch);
    // Only one thread touches the flag.  A threadgroup-wide read/write would let later
    // threadgroups see the marker set by the first group during the same launch.
    bool second = false;
    if (row == 0 && lid == 0) {
        second = (flags[0] == 0x12345678u);
        if (!second) { flags[0] = 0x12345678u; }
    }
    threadgroup float partial[8];
    device const half* xr = x + (ulong)row * H;
    float ss = 0.0f;
    for (uint i = lid; i < H; i += 256) { float v = float(xr[i]); ss += v * v; }
    ss = simd_sum(ss);
    if (lane == 0) partial[sg] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float tot = 0.0f;
    for (uint s = 0; s < 8; ++s) tot += partial[s];
    float r = rsqrt(tot / float(H) + EPS);
    for (uint i = lid; i < H; i += 256) y[(ulong)row * H + i] = half(float(xr[i]) * r * float(w[i]));
    if (row == 0 && lid == 0 && second) { y[(ulong)M * H + 1] = half(0.0f); }
}
/* taic-launch
math safe
workspace scratch M*H*2
launch rmsnorm_second_oob grid=M,1,1 threads=256,1,1 workspace=scratch
*/
"""
    result = evaluate_task(
        locator,
        track_id="msl-single",
        candidate_source=source,
        builder=MetalTrackBuilder(),
        evaluator=MetalTrackEvaluator(mode="full", reps=3, cases=("normal",)),
        geometry_resolver=msl_single_geometry_resolver(contract),
        track_registry_path=tracks,
    )
    assert result.report["status"] == "unsafe"
    assert result.report["correctness"]["ok"] is True
    assert any("past the end" in item or "out of bounds" in item.lower() for item in result.report["safety"]["violations"])
    assert result.report["build"]["provenance"]["artifact_set_digest"] == result.artifact_set_digest
    assert result.report["build"]["artifacts"]["artifact_set_digest"] == result.artifact_set_digest


@gpu
def test_a4_real_evaluator_refuses_a_drifted_full_build(tmp_path, monkeypatch):
    """@brief 负例：真实 MetalTrackEvaluator 不能把漂移的 full build 覆盖成 attached build。"""
    from tcenv.backends.metal import track_adapter
    from tcenv.backends.metal.track_adapter import MetalTrackBuilder, MetalTrackEvaluator, msl_single_geometry_resolver
    from tcenv.contract_errors import TaskContractError
    from tcenv.taskcontract.runner import evaluate_task

    original = track_adapter._evaluate_in_process

    def drifted(*args, **kwargs):
        report = original(*args, **kwargs)
        if isinstance(report.build, dict) and report.build.get("shape") == "full":
            report.build = json.loads(json.dumps(report.build))
            report.build["assembled_sha256"] = "0" * 64
            report.build["provenance"]["artifact_set_digest"] = "f" * 64
            report.build["artifacts"]["artifact_set_digest"] = "f" * 64
        return report

    monkeypatch.setattr(track_adapter, "_evaluate_in_process", drifted)
    locator, contract, tracks = _rmsnorm_task(tmp_path)
    with pytest.raises(TaskContractError, match="drifted|does not match"):
        evaluate_task(
            locator,
            track_id="msl-single",
            candidate_source=_rmsnorm_single_source(),
            builder=MetalTrackBuilder(),
            # In-process so the patched evaluator is the one that builds the report.
            # A subprocess would import the unpatched function and hide the drift.
            evaluator=MetalTrackEvaluator(mode="quick", cases=("normal",), worker_child=True),
            geometry_resolver=msl_single_geometry_resolver(contract),
            track_registry_path=tracks,
        )


@gpu
def test_a4_fill_workspaces_clears_a_non_word_aligned_span():
    """@brief fill_workspaces 的独立命令缓冲必须把 6 字节 span 的尾部也清零。"""
    from tcenv.backends.metal.runtime import MetalContext
    from tcenv.backends.metal.sanitize import GUARD_PATTERN, GuardedBuffer
    from tcenv.backends.registry import load_backend

    backend = load_backend("metal")
    ctx = backend.context()
    backend.prepare_context(ctx)
    assert isinstance(ctx, MetalContext)
    buf = GuardedBuffer(ctx, 6, pattern=GUARD_PATTERN)
    buf.mem[buf.guard : buf.guard + buf.nbytes] = bytes([GUARD_PATTERN]) * 6
    ctx.fill_workspaces([(buf.buf, buf.guard, 6)])
    assert buf.read() == bytes(6)


@gpu
def test_a4_run_pipeline_fill_clears_a_non_word_aligned_span():
    """@brief run_pipeline 的计时内 fill 路径也必须清除 6 字节 span，而不是只清一个 word。"""
    from tcenv.backends.metal.runtime import MetalContext
    from tcenv.backends.metal.sanitize import GUARD_PATTERN, GuardedBuffer
    from tcenv.backends.registry import load_backend

    backend = load_backend("metal")
    ctx = backend.context()
    backend.prepare_context(ctx)
    assert isinstance(ctx, MetalContext)
    kernel = ctx.compile(
        "#include <metal_stdlib>\nusing namespace metal;\nkernel void empty_kernel() {}\n",
        "empty_kernel",
        math="safe",
    )
    buf = GuardedBuffer(ctx, 6, pattern=GUARD_PATTERN)
    buf.mem[buf.guard : buf.guard + buf.nbytes] = bytes([GUARD_PATTERN]) * 6
    ctx.run_pipeline(
        [kernel],
        [(1, 1, 1)],
        [(1, 1, 1)],
        [0],
        [[[]]],
        [[(buf.buf, buf.guard, 6)]],
        iters=1,
        dispatch="serial",
    )
    assert buf.read() == bytes(6)


@gpu
def test_a4_two_active_device_stages_chain_output_and_reflect_each_stage(tmp_path):
    """@brief 真实两激活 stage：stage1 写 y，stage2 读取并写回 y；逐段反射/实际顺序/摘要可重放。"""
    from tcenv.backends.metal.track_adapter import MetalTrackBuilder, MetalTrackEvaluator, msl_single_geometry_resolver
    from tcenv.taskcontract.runner import evaluate_task
    from tcenv.remote.execution_geometry import canonical_geometry_bytes

    locator, contract, tracks = _rmsnorm_task(tmp_path)
    source = """#include <metal_stdlib>
using namespace metal;
kernel void stage_one(device const half* x [[buffer(0)]],
                      device const half* w [[buffer(1)]],
                      device half* y [[buffer(2)]],
                      uint row [[threadgroup_position_in_grid]],
                      uint lid [[thread_index_in_threadgroup]],
                      uint lane [[thread_index_in_simdgroup]],
                      uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float partial[8];
    device const half* xr = x + (ulong)row * H;
    float ss = 0.0f;
    for (uint i = lid; i < H; i += 256) { float v = float(xr[i]); ss += v * v; }
    ss = simd_sum(ss);
    if (lane == 0) partial[sg] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float tot = 0.0f;
    for (uint s = 0; s < 8; ++s) tot += partial[s];
    float r = rsqrt(tot / float(H) + EPS);
    for (uint i = lid; i < H; i += 256) y[(ulong)row * H + i] = half(float(xr[i]) * r * float(w[i]));
}
kernel void stage_two(device half* y [[buffer(0)]],
                      uint row [[threadgroup_position_in_grid]],
                      uint lid [[thread_index_in_threadgroup]]) {
    // This stage consumes the previous stage's output and writes the final result.
    for (uint i = lid; i < H; i += 256) {
        ulong j = (ulong)row * H + i;
        y[j] = half(float(y[j]) * 1.0f);
    }
}
/* taic-launch
math safe
launch stage_one grid=M,1,1 threads=256,1,1
launch stage_two grid=M,1,1 threads=256,1,1
*/
"""
    evaluator = MetalTrackEvaluator(mode="quick", cases=("normal",))
    result = evaluate_task(
        locator,
        track_id="msl-single",
        candidate_source=source,
        builder=MetalTrackBuilder(),
        evaluator=evaluator,
        geometry_resolver=msl_single_geometry_resolver(contract),
        track_registry_path=tracks,
    )
    assert result.report["status"] == "ok"
    assert result.report["correctness"]["ok"] is True
    assert result.execution is not None
    observed = _observed(result)
    assert [item[0] for item in observed] == ["stage_one", "stage_two"]
    assert observed[0][1] == (64, 1, 1) and observed[1][1] == (64, 1, 1)
    assert observed[0][2] == (256, 1, 1) and observed[1][2] == (256, 1, 1)
    bindings = result.execution["bindings"]
    assert [item["kernel"] for item in bindings] == ["stage_one", "stage_two"]
    assert any(arg["name"] == "y" and arg["access"] == "write" for arg in bindings[0]["arguments"])
    assert any(arg["name"] == "y" for arg in bindings[1]["arguments"])

    canonical = _canonical_single(source, contract)
    assert canonical.bundle_digest == result.bundle_digest
    assert len(canonical.document()["geometry"]["candidate_resolution"]["stages"]) == 2
    assert canonical_geometry_bytes(canonical)

    replay_evaluator = MetalTrackEvaluator(mode="quick", cases=("normal",))
    replay = evaluate_task(
        locator,
        track_id="msl-single",
        candidate_source=source,
        builder=MetalTrackBuilder(),
        evaluator=replay_evaluator,
        geometry_resolver=msl_single_geometry_resolver(contract),
        track_registry_path=tracks,
    )
    assert _stable(replay) == _stable(result)
    assert _observed(replay) == observed
    _record(
        "multistage.json",
        {
            "slice": "two-active-stages",
            "task": "rmsnorm_v2",
            "candidate_sha256": result.candidate_sha256,
            "geometry_identity": result.geometry_identity,
            "bundle_digest": result.bundle_digest,
            "artifact_set_digest": result.artifact_set_digest,
            "native_digest": result.native_digest,
            "canonical_bundle": canonical.document(),
            "geometry_artifact_sha256": hashlib.sha256(canonical_geometry_bytes(canonical)).hexdigest(),
            "dispatch": observed,
            "bindings": bindings,
            "replay_equal": _stable(replay) == _stable(result),
        },
    )


@gpu
def test_a4_real_stage_refuses_a_workspace_it_did_not_declare(tmp_path):
    """@brief 真实 GPU 路径：kernel 绑定 scratch 但 stage 未声明时，在 ABI 阶段拒绝，不能执行。"""
    from tcenv.backends.metal.track_adapter import MetalTrackBuilder, MetalTrackEvaluator, msl_single_geometry_resolver
    from tcenv.taskcontract.runner import evaluate_task

    locator, contract, tracks = _rmsnorm_task(tmp_path)
    source = _rmsnorm_single_source().replace(" workspace=scratch", "")
    result = evaluate_task(
        locator,
        track_id="msl-single",
        candidate_source=source,
        builder=MetalTrackBuilder(),
        evaluator=MetalTrackEvaluator(mode="quick", cases=("normal",)),
        geometry_resolver=msl_single_geometry_resolver(contract),
        track_registry_path=tracks,
    )
    assert result.report["status"] == "abi_error"
    assert any("not declared by this stage" in item for item in result.report["abi"]["violations"])
    assert result.execution is None or result.execution.get("dispatch") == []


def test_a4_v2_host_path_include_is_refused_before_the_compiler_is_called(tmp_path, monkeypatch):
    """@brief v2 首次编译前拒绝宿主路径 include；编译器（MetalSourceBuilder.build）未被调用。"""
    from tcenv.backends.metal import source_builder
    from tcenv.backends.metal.track_adapter import MetalTrackBuilder, MetalTrackEvaluator, msl_single_geometry_resolver
    from tcenv.taskcontract.runner import evaluate_task
    from tcenv.contract_errors import TaskContractError

    locator, contract, tracks = _rmsnorm_task(tmp_path)
    source = _rmsnorm_single_source().replace("#include <metal_stdlib>", "#include <stdio.h>")
    called: list[object] = []

    def _never(self, *args, **kwargs):
        called.append((args, kwargs))
        raise AssertionError("the compiler must not be called for a refused v2 source")

    monkeypatch.setattr(source_builder.MetalSourceBuilder, "build", _never)
    with pytest.raises(TaskContractError, match="refused before compilation"):
        evaluate_task(
            locator,
            track_id="msl-single",
            candidate_source=source,
            builder=MetalTrackBuilder(),
            evaluator=MetalTrackEvaluator(mode="quick", cases=("normal",)),
            geometry_resolver=msl_single_geometry_resolver(contract),
            track_registry_path=tracks,
        )
    assert called == []


def test_a4_external_deadline_kills_a_stuck_v2_worker(monkeypatch):
    """@brief 外部 supervisor 在单次 launch 心跳过期后杀 v2 worker；不使用真实 GPU。"""
    import time
    from tcenv import evaluate as ev_mod
    from tcenv import v2_worker

    monkeypatch.setattr(ev_mod, "POST_KILL_COOLDOWN_S", 0.0)
    start = time.perf_counter()
    result = v2_worker.run_bounded_v2(
        {"target_id": "apple-m4-max"},
        timeout=60.0,
        launch_deadline_s=1.0,
        worker_module="tests.support.fake_v2_worker_stuck",
    )
    assert result["status"] == "timeout"
    assert "launch" in result["message"]
    assert time.perf_counter() - start < 10


def test_a4_external_deadline_kills_a_stuck_compile_in_the_worker(monkeypatch):
    """@brief 首次真实编译在外部 worker 内：卡住的编译由外部 compile 截止时间 SIGKILL。"""
    import time

    from tcenv import evaluate as ev_mod
    from tcenv import v2_worker

    monkeypatch.setattr(ev_mod, "POST_KILL_COOLDOWN_S", 0.0)
    start = time.perf_counter()
    result = v2_worker.run_bounded_v2(
        {"target_id": "apple-m4-max"},
        timeout=60.0,
        launch_deadline_s=60.0,
        compile_deadline_s=1.0,
        worker_module="tests.support.fake_v2_worker_stuck_compile",
    )
    assert result["status"] == "timeout"
    assert "compile" in result["message"]
    assert time.perf_counter() - start < 10


def test_a4_frozen_track_selection_never_reads_the_default_registry(tmp_path, monkeypatch):
    """@brief worker 只用父进程冻结的 TrackSpec/registry；冻结路径不再读任何默认 registry。"""
    from tcenv.contract_errors import TaskContractError
    from tcenv.taskcontract import runner as runner_mod

    parent = runner_mod.select_track("msl-single", registry_path=_tracks(tmp_path / "parent.json", version=2))
    frozen = parent.frozen_document(presentation_sha256="a" * 64)

    def _never(*args, **kwargs):
        raise AssertionError("a frozen track selection must not read a registry")

    monkeypatch.setattr(runner_mod, "load_track_registry", _never)
    child = runner_mod.select_track("msl-single", frozen=frozen)
    assert child.track.track_version == 2
    assert child.frozen is True
    assert child.track_digest == parent.track_digest
    assert child.registry_digest == parent.registry_digest
    with pytest.raises(TaskContractError, match="registry path"):
        runner_mod.select_track(
            "msl-single", frozen=frozen, registry_path=_tracks(tmp_path / "other.json")
        )


def test_a4_frozen_track_drift_is_refused_before_any_build(tmp_path):
    """@brief 冻结 registry/track/渲染身份被改即拒绝；漂移不能靠默认 registry 或重渲染掩盖。"""
    from tcenv.backends.metal.track_adapter import MetalTrackBuilder, MetalTrackEvaluator, msl_single_geometry_resolver
    from tcenv.contract_errors import TaskContractError
    from tcenv.taskcontract import runner as runner_mod
    from tcenv.taskcontract.runner import evaluate_task

    locator, contract, _ = _rmsnorm_task(tmp_path)
    tracks = _tracks(tmp_path / "tracks-v2.json", version=2)
    parent = runner_mod.select_track("msl-single", registry_path=tracks)
    frozen = parent.frozen_document(presentation_sha256="a" * 64)

    registry_drift = json.loads(json.dumps(frozen))
    registry_drift["registry"]["tracks"][0]["track_version"] += 1
    with pytest.raises(TaskContractError, match="registry_digest|does not hash"):
        runner_mod.select_track("msl-single", frozen=registry_drift)

    track_drift = json.loads(json.dumps(frozen))
    track_drift["track"]["track_version"] += 1
    with pytest.raises(TaskContractError, match="track_digest"):
        runner_mod.select_track("msl-single", frozen=track_drift)

    with pytest.raises(TaskContractError, match="version"):
        runner_mod.select_track("msl-single", frozen=frozen, version=1)

    # A worker whose rendering no longer reproduces the control-host presentation digest is
    # refused before the builder prepares/compiles anything.
    with pytest.raises(TaskContractError, match="rendering|presentation"):
        evaluate_task(
            locator,
            track_id="msl-single",
            candidate_source=_rmsnorm_single_source(),
            builder=MetalTrackBuilder(),
            evaluator=MetalTrackEvaluator(mode="quick", cases=("normal",)),
            geometry_resolver=msl_single_geometry_resolver(contract),
            frozen_track=frozen,
        )


def test_a4_a_timed_out_bounded_worker_has_no_fabricated_build(tmp_path, monkeypatch):
    """@brief 父进程外部时限杀掉 worker 时，报告没有 full build，身份摘要保持 null。"""
    from tcenv import evaluate as ev_mod
    from tcenv.backends.metal.track_adapter import MetalTrackBuilder, MetalTrackEvaluator, msl_single_geometry_resolver
    from tcenv.taskcontract.runner import evaluate_task

    monkeypatch.setattr(ev_mod, "POST_KILL_COOLDOWN_S", 0.0)
    locator, contract, tracks = _rmsnorm_task(tmp_path)
    result = evaluate_task(
        locator,
        track_id="msl-single",
        candidate_source=_rmsnorm_single_source(),
        builder=MetalTrackBuilder(),
        evaluator=MetalTrackEvaluator(
            mode="quick",
            cases=("normal",),
            worker_timeout=60.0,
            worker_launch_deadline_s=1.0,
            worker_module="tests.support.fake_v2_worker_stuck",
        ),
        geometry_resolver=msl_single_geometry_resolver(contract),
        track_registry_path=tracks,
    )
    assert result.report["status"] == "timeout"
    assert result.report["build"] is None
    assert result.artifact_set_digest is None
    assert result.native_digest is None
    assert result.build is None
    assert result.execution["status"] == "timeout"
    assert result.execution["dispatch"] == []


@gpu
def test_a4_the_worker_uses_the_control_host_frozen_track_not_the_default_registry(tmp_path):
    """@brief 真实 worker 用父进程冻结的 track_version=2/registry 身份；默认 registry 仍是 v1。"""
    from tcenv.backends.metal.track_adapter import MetalTrackBuilder, MetalTrackEvaluator, msl_single_geometry_resolver
    from tcenv.identity import fingerprint
    from tcenv.taskcontract.runner import evaluate_task
    from tcenv.track import load_track_registry

    locator, contract, _ = _rmsnorm_task(tmp_path)
    tracks = _tracks(tmp_path / "tracks-v2.json", version=2)
    # The reviewed reproduction: the parent froze version 2 while the default registry is v1, and
    # the worker still returned status=ok.  The default must no longer be selectable here.
    assert load_track_registry().get("msl-single").track_version == 1
    frozen_track = load_track_registry(tracks).get("msl-single", version=2)
    result = evaluate_task(
        locator,
        track_id="msl-single",
        candidate_source=_rmsnorm_single_source(),
        builder=MetalTrackBuilder(),
        evaluator=MetalTrackEvaluator(mode="quick", cases=("normal",)),
        geometry_resolver=msl_single_geometry_resolver(contract),
        track_registry_path=tracks,
    )
    assert result.report["status"] == "ok", result.report.get("message")
    assert result.track["track_version"] == 2
    assert result.track["track_digest"] == fingerprint(frozen_track.document())
    worker = result.execution["worker"]
    assert worker["track_version"] == 2
    assert worker["track_digest"] == result.track["track_digest"]
    assert worker["registry_digest"] == result.track["registry_digest"]
    assert worker["presentation_sha256"] == result.presentation_sha256
    assert result.report["build"]["provenance"]["artifact_set_digest"] == result.artifact_set_digest


@gpu
def test_a4_parent_never_compiles_the_candidate_before_the_bounded_worker(tmp_path, monkeypatch):
    """@brief 控制机父进程不编译：首次真实编译只在有界 worker 内发生，候选只编译一次。"""
    from tcenv.backends.metal import source_builder
    from tcenv.backends.metal.track_adapter import MetalTrackBuilder, MetalTrackEvaluator, msl_single_geometry_resolver
    from tcenv.taskcontract.runner import evaluate_task

    calls: list[object] = []

    def _never(self, *args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError(
            "the control host compiled the v2 candidate; the first real build belongs to the bounded worker"
        )

    monkeypatch.setattr(source_builder.MetalSourceBuilder, "build", _never)
    locator, contract, tracks = _rmsnorm_task(tmp_path)
    result = evaluate_task(
        locator,
        track_id="msl-single",
        candidate_source=_rmsnorm_single_source(),
        builder=MetalTrackBuilder(),
        evaluator=MetalTrackEvaluator(mode="quick", cases=("normal",)),
        geometry_resolver=msl_single_geometry_resolver(contract),
        track_registry_path=tracks,
    )
    assert calls == [], "the parent process called the Metal compiler"
    assert result.report["status"] == "ok", result.report.get("message")
    assert result.build["shape"] == "full"
    assert result.report["build"]["provenance"]["artifact_set_digest"] == result.artifact_set_digest


@gpu
def test_a4_a_worker_track_drift_is_refused(tmp_path):
    """@brief 负例：worker 真实构建成功但报告了不同 Track 版本/摘要，父进程拒绝。"""
    from tcenv.backends.metal.track_adapter import MetalTrackBuilder, MetalTrackEvaluator, msl_single_geometry_resolver
    from tcenv.contract_errors import TaskContractError
    from tcenv.taskcontract.runner import evaluate_task

    locator, contract, tracks = _rmsnorm_task(tmp_path)
    with pytest.raises(TaskContractError, match="TrackSpec|drift|frozen"):
        evaluate_task(
            locator,
            track_id="msl-single",
            candidate_source=_rmsnorm_single_source(),
            builder=MetalTrackBuilder(),
            evaluator=MetalTrackEvaluator(
                mode="quick",
                cases=("normal",),
                worker_module="tests.support.fake_v2_worker_drifting_track",
            ),
            geometry_resolver=msl_single_geometry_resolver(contract),
            track_registry_path=tracks,
        )


@gpu
def test_a4_a_worker_build_that_disagrees_with_the_canonical_geometry_is_refused(tmp_path):
    """@brief 负例：worker 自洽但几何审计漂移的 full build 被父进程交叉审计拒绝。"""
    from tcenv.backends.metal.track_adapter import MetalTrackBuilder, MetalTrackEvaluator, msl_single_geometry_resolver
    from tcenv.contract_errors import TaskContractError
    from tcenv.taskcontract.runner import evaluate_task

    locator, contract, tracks = _rmsnorm_task(tmp_path)
    with pytest.raises(TaskContractError, match="build|geometry|drift|audit"):
        evaluate_task(
            locator,
            track_id="msl-single",
            candidate_source=_rmsnorm_single_source(),
            builder=MetalTrackBuilder(),
            evaluator=MetalTrackEvaluator(
                mode="quick",
                cases=("normal",),
                worker_module="tests.support.fake_v2_worker_drifting_build",
            ),
            geometry_resolver=msl_single_geometry_resolver(contract),
            track_registry_path=tracks,
        )


@gpu
def test_a4_a_worker_compile_error_is_a_partial_build_not_a_fabricated_full_one(tmp_path):
    """@brief 负例：worker 内首次编译失败只报 partial build，不伪造 full build 或 artifact set。"""
    from tcenv.backends.metal.track_adapter import MetalTrackBuilder, MetalTrackEvaluator, msl_single_geometry_resolver
    from tcenv.taskcontract.runner import evaluate_task

    locator, contract, tracks = _rmsnorm_task(tmp_path)
    source = _rmsnorm_single_source().replace("rsqrt", "not_a_registered_msl_builtin")
    result = evaluate_task(
        locator,
        track_id="msl-single",
        candidate_source=source,
        builder=MetalTrackBuilder(),
        evaluator=MetalTrackEvaluator(mode="quick", cases=("normal",)),
        geometry_resolver=msl_single_geometry_resolver(contract),
        track_registry_path=tracks,
    )
    assert result.report["status"] == "compile_error"
    build = result.report["build"]
    assert build["shape"] == "partial"
    assert build["error_class"] == "compile"
    assert build["provenance"]["artifact_set_digest"] is None
    assert build["assembled_sha256"] is not None
    assert result.artifact_set_digest is None
    assert result.native_digest is None
    assert result.build["shape"] == "partial"
    assert result.execution["stage"] == "compile"
    assert result.execution["dispatch"] == []


def test_a4_the_report_status_and_build_shape_must_stay_in_the_closed_table():
    """@brief 负例：ok/compile_error/timeout/crash 与 partial/full 的错配在读取层拒绝。"""
    from tcenv.evaluate import EvalReport
    from tcenv.schema import SchemaError
    from tests.support.report_build import report_for_request

    request = {
        "task": V1_TASK,
        "dtype": "fp16",
        "size": "small",
        "mode": "quick",
        "seed": 1,
        "target_id": "apple-m4-max",
    }
    ok = report_for_request(request, status="ok", build_kind="native")
    compile_error = report_for_request(request, status="compile_error", build_kind="native")
    assert ok["build"]["shape"] == "full"
    assert compile_error["build"]["shape"] == "partial"
    for status, build in (
        ("compile_error", ok["build"]),
        ("ok", compile_error["build"]),
        ("timeout", ok["build"]),
        ("crash", ok["build"]),
    ):
        document = json.loads(json.dumps(ok))
        document["status"] = status
        document["eligible"] = status == "ok"
        document["build"] = json.loads(json.dumps(build))
        with pytest.raises(SchemaError, match="build"):
            EvalReport.from_json(document)


def test_a4_an_evaluator_build_must_carry_exactly_the_candidate_and_frozen_geometry(tmp_path):
    """@brief 负例：worker build 的缺失/额外 artifact entry 被父进程逐项拒绝。"""
    from tcenv.build_projection import project_full
    from tcenv.remote.artifact_bundle import BUNDLE_SCHEMA_ID, NativeExecutable, artifact_set_digest
    from tcenv.remote.execution_geometry import EvaluatorBuildError, verify_evaluator_build

    _locator, contract, _tracks_path = _rmsnorm_task(tmp_path)
    canonical = _canonical_single(_rmsnorm_single_source(), contract)
    identity = {
        "opspec_digest": contract.spec.digest,
        "lowering_profile_digest": contract.fingerprint(),
    }
    document = project_full(_structural_full_build(canonical), build_kind="native", identity=identity)
    assert verify_evaluator_build(document, canonical_bundle=canonical, identity=identity).shape == "full"

    missing = json.loads(json.dumps(document))
    missing["artifacts"]["entries"] = [
        item for item in missing["artifacts"]["entries"] if item["path"] != "execution_geometry.json"
    ]
    _reseal_evaluator_build(missing)
    with pytest.raises(EvaluatorBuildError, match="frozen execution geometry|artifact set"):
        verify_evaluator_build(missing, canonical_bundle=canonical, identity=identity)

    extra = json.loads(json.dumps(document))
    extra["artifacts"]["entries"].append(
        {
            "path": "extra/note.txt",
            "role": "AUXILIARY",
            "media_type": "text/plain",
            "size_bytes": 3,
            "sha256": hashlib.sha256(b"abc").hexdigest(),
        }
    )
    _reseal_evaluator_build(extra)
    with pytest.raises(EvaluatorBuildError, match="frozen execution geometry|artifact set"):
        verify_evaluator_build(extra, canonical_bundle=canonical, identity=identity)

    # A self-consistent duplicate path is not a valid artifact bundle: the native executable
    # would no longer be identified by a unique SOURCE entry.  The auditor must refuse it
    # before an exact-entry comparison can accidentally accept one of the aliases.
    duplicate = json.loads(json.dumps(document))
    source = next(item for item in duplicate["artifacts"]["entries"] if item["role"] == "SOURCE")
    geometry = next(item for item in duplicate["artifacts"]["entries"] if item["path"] == "execution_geometry.json")
    source["path"] = geometry["path"]
    duplicate["artifacts"]["entries"] = [source, geometry]
    duplicate["native"]["entry_path"] = geometry["path"]
    duplicate["native"]["entry_sha256"] = geometry["sha256"]
    _reseal_evaluator_build(duplicate)
    with pytest.raises(EvaluatorBuildError, match="duplicate|alias"):
        verify_evaluator_build(duplicate, canonical_bundle=canonical, identity=identity)

    # Casefold alias: one filesystem would collapse these into the same entry, even though the
    # strings differ.  The native executable must not be allowed to point at either alias.
    casefold = json.loads(json.dumps(document))
    source = next(item for item in casefold["artifacts"]["entries"] if item["role"] == "SOURCE")
    geometry = next(item for item in casefold["artifacts"]["entries"] if item["path"] == "execution_geometry.json")
    source["path"] = geometry["path"].upper()
    casefold["native"]["entry_path"] = source["path"]
    _reseal_evaluator_build(casefold)
    with pytest.raises(EvaluatorBuildError, match="duplicate|alias"):
        verify_evaluator_build(casefold, canonical_bundle=canonical, identity=identity)

    # Decomposed-NFC alias: U+00E9 and U+0065 U+0301 are distinct strings but one NFC path.
    nfc = json.loads(json.dumps(document))
    for path in ("caf\u00e9/note.txt", "cafe\u0301/note.txt"):
        nfc["artifacts"]["entries"].append(
            {
                "path": path,
                "role": "AUXILIARY",
                "media_type": "text/plain",
                "size_bytes": 3,
                "sha256": hashlib.sha256(path.encode("utf-8")).hexdigest(),
            }
        )
    _reseal_evaluator_build(nfc)
    with pytest.raises(EvaluatorBuildError, match="duplicate|alias"):
        verify_evaluator_build(nfc, canonical_bundle=canonical, identity=identity)


def test_a4_a_single_source_build_must_carry_the_canonical_source_manifest(tmp_path):
    """@brief 负例：SOURCE 不能靠自报 path/media_type 绕过 canonical 候选 entry。"""
    from tcenv.build_projection import project_full
    from tcenv.remote.execution_geometry import EvaluatorBuildError, verify_evaluator_build

    _locator, contract, _tracks_path = _rmsnorm_task(tmp_path)
    canonical = _canonical_single(_rmsnorm_single_source(), contract)
    identity = {
        "opspec_digest": contract.spec.digest,
        "lowering_profile_digest": contract.fingerprint(),
    }
    document = project_full(_structural_full_build(canonical), build_kind="native", identity=identity)
    assert verify_evaluator_build(document, canonical_bundle=canonical, identity=identity).shape == "full"

    renamed = json.loads(json.dumps(document))
    source = next(item for item in renamed["artifacts"]["entries"] if item["role"] == "SOURCE")
    source["path"] = "attacker/other-name.metal"
    renamed["native"]["entry_path"] = source["path"]
    _reseal_evaluator_build(renamed)
    with pytest.raises(EvaluatorBuildError, match="SOURCE|canonical"):
        verify_evaluator_build(renamed, canonical_bundle=canonical, identity=identity)

    retyped = json.loads(json.dumps(document))
    source = next(item for item in retyped["artifacts"]["entries"] if item["role"] == "SOURCE")
    source["media_type"] = "application/octet-stream"
    _reseal_evaluator_build(retyped)
    with pytest.raises(EvaluatorBuildError, match="SOURCE|canonical"):
        verify_evaluator_build(retyped, canonical_bundle=canonical, identity=identity)


def test_a4_a_partial_build_must_still_be_a_native_build(tmp_path):
    """@brief 负例：Metal worker 的 compile_error partial 不能是 simulator/source_only。"""
    from tcenv.remote.execution_geometry import EvaluatorBuildError, verify_evaluator_build

    _locator, contract, _tracks_path = _rmsnorm_task(tmp_path)
    canonical = _canonical_single(_rmsnorm_single_source(), contract)
    identity = {
        "opspec_digest": contract.spec.digest,
        "lowering_profile_digest": contract.fingerprint(),
    }
    native = _structural_partial_build(canonical, build_kind="native", identity=identity)
    assert verify_evaluator_build(native, canonical_bundle=canonical, identity=identity).shape == "partial"
    for build_kind in ("simulator", "source_only"):
        document = _structural_partial_build(canonical, build_kind=build_kind, identity=identity)
        with pytest.raises(EvaluatorBuildError, match="native"):
            verify_evaluator_build(document, canonical_bundle=canonical, identity=identity)


def test_a4_a_crashed_bounded_worker_has_no_fabricated_build(tmp_path, monkeypatch):
    """@brief 负例：外部 worker 崩溃时报告没有 full build，摘要保持 null。"""
    from tcenv import v2_worker
    from tcenv.backends.metal.track_adapter import MetalTrackBuilder, MetalTrackEvaluator, msl_single_geometry_resolver
    from tcenv.taskcontract.runner import evaluate_task

    monkeypatch.setattr(
        v2_worker,
        "run_bounded_v2",
        lambda *args, **kwargs: {"status": "crash", "stage": "crash", "message": "fixture crash"},
    )
    locator, contract, tracks = _rmsnorm_task(tmp_path)
    result = evaluate_task(
        locator,
        track_id="msl-single",
        candidate_source=_rmsnorm_single_source(),
        builder=MetalTrackBuilder(),
        evaluator=MetalTrackEvaluator(mode="quick", cases=("normal",)),
        geometry_resolver=msl_single_geometry_resolver(contract),
        track_registry_path=tracks,
    )
    assert result.report["status"] == "crash"
    assert result.report["build"] is None
    assert result.artifact_set_digest is None
    assert result.native_digest is None
    assert result.build is None
    assert result.execution["status"] == "crash"
    assert result.execution["dispatch"] == []

