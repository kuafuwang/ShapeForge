"""Frozen geometry enters the build artifact set and the confirmation digest (A3, CPU)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from tcenv.contract_errors import GeometryError
from tcenv.remote.artifact_builder import (
    BuildArtifact,
    BuildEvidence,
    BuildOutcome,
    BuildProfile,
    BuildProvenance,
)
from tcenv.remote.artifact_bundle import ArtifactBundle, BundleEntry, NativeExecutable
from tcenv.remote.execution_geometry import (
    CANDIDATE_MANIFEST_ARTIFACT_PATH,
    EXECUTION_GEOMETRY_ARTIFACT_PATH,
    GeometryArtifactError,
    attach_execution_geometry,
    canonical_geometry_bytes,
    verify_execution_geometry_artifact,
)
from tcenv.target import DEFAULT_TARGET_ID, load_target
from tcenv.taskcontract import load_task_contract
from tcenv.track import (
    CandidateArtifact,
    CandidateEntry,
    CanonicalBundle,
    ExecutionGeometry,
    GeometryStage,
    TargetAbiLimits,
    candidate_from_single_source,
    derive_tensor_memory_contract,
    geometry_pair,
    resolve_geometry,
)

from ..support.v2_tasks import write_task

LIMITS = TargetAbiLimits(max_buffer_index=30, max_threads_per_threadgroup=1024, max_threadgroup_memory_bytes=32768)
PROFILE = BuildProfile(
    build_profile="metal-msl-control-host-v1",
    builder_id="metal-source-builder",
    builder_version="1",
    build_host="control_host",
    toolchain_id="apple-metal",
    toolchain_version="unknown",
    sdk_version="unknown",
    compile_profile={"language": "msl", "math": "safe", "device": "test", "gpu_family": "test"},
)
IDENTITY = {"opspec_digest": "a" * 64, "lowering_profile_digest": "b" * 64}


def _candidate_and_bundle(tmp_path: Path, *, grid=(2, 1, 1), fallback_grid=(1, 1, 1), single=False):
    """@brief 构造候选与 canonical bundle。

    @param tmp_path 临时目录。
    @param grid 候选 grid。
    @param fallback_grid 独立 fallback grid。
    @param single 是否单文件候选。
    @return ``(candidate, canonical_bundle)``。
    """
    task = load_task_contract(write_task(tmp_path / "task"))
    abi = derive_tensor_memory_contract(task.instance, target=load_target(DEFAULT_TARGET_ID))
    geometry_document = {
        "schema_version": 1,
        "source": "candidate",
        "stages": [{"kernel": "fill", "grid": list(grid), "threads": [256, 1, 1]}],
    }
    geometry = ExecutionGeometry(
        source="candidate", stages=(GeometryStage(kernel="fill", grid=grid, threads=(256, 1, 1)),)
    )
    fallback = ExecutionGeometry(
        source="fallback", stages=(GeometryStage(kernel="fb", grid=fallback_grid, threads=(64, 1, 1)),)
    )
    pair = geometry_pair(geometry, fallback, instance=task.instance, contract=abi, limits=LIMITS)
    if single:
        candidate = candidate_from_single_source("kernel void fill() {}\n")
    else:
        candidate = CandidateArtifact(
            track_id="msl-bundle",
            entries=(
                CandidateEntry(
                    path="kernel.metal", role="device_kernel", media_type="text/plain", data=b"kernel void fill(){}\n"
                ),
                CandidateEntry(path="host/tiling.metal", role="host_tiling", media_type="text/plain", data=b"// t\n"),
                CandidateEntry(
                    path="host/execution_geometry.json",
                    role="schedule",
                    media_type="application/json",
                    data=json.dumps(geometry_document, sort_keys=True).encode(),
                ),
            ),
        )
    return candidate, CanonicalBundle(candidate=candidate, pair=pair)


def _outcome(candidate, canonical, *, with_source=False) -> BuildOutcome:
    """@brief 一个模拟的真实构建结果（含 native handle）。

    @param candidate 候选。
    @param canonical canonical bundle。
    @param with_source bundle 构建是否错误地带了 SOURCE（负例）。
    @return BuildOutcome。
    """
    handle = object()
    entries: list[BundleEntry]
    if candidate.single_source:
        entry = candidate.entries[0]
        entries = [
            BundleEntry(
                path=entry.path,
                role="SOURCE",
                media_type=entry.media_type,
                size_bytes=len(entry.data),
                sha256=entry.sha256,
                data=entry.data,
            )
        ]
        native_path = entry.path
    else:
        entries = []
        for item in canonical.entries:
            entries.append(
                BundleEntry(
                    path=item.path,
                    role="SOURCE" if with_source and item.role == "device_kernel" else "AUXILIARY",
                    media_type=item.media_type,
                    size_bytes=len(item.data),
                    sha256=item.sha256,
                    data=bytes(item.data),
                )
            )
        native_path = "kernel.metal"
    native = NativeExecutable(
        backend_id="metal",
        kind="compute-pipeline",
        entry_path=native_path,
        entry_sha256=next(entry.sha256 for entry in entries if entry.path == native_path),
        descriptor={"backend": "metal", "kernel_name": "fill"},
        handle=handle,
    )
    bundle = ArtifactBundle(entries=tuple(entries), native=native)
    provenance = BuildProvenance(
        profile=PROFILE, source_sha256=candidate.candidate_sha256, artifact_set_digest=bundle.artifact_set_digest
    )
    evidence = BuildEvidence(
        mode="prebuilt",
        status="ok",
        provenance=provenance,
        artifacts=tuple(BuildArtifact.from_entry(entry) for entry in bundle.entries),
        placement="control_host",
        native=native,
    )
    return BuildOutcome(bundle=bundle, provenance=provenance, evidence=evidence)


def test_the_geometry_entry_is_attached_after_the_build_and_changes_the_confirmation_digest(tmp_path):
    """@brief 几何在构建后冻结，artifact_set_digest 覆盖它；native 只做交叉审计。"""
    candidate, canonical = _candidate_and_bundle(tmp_path / "bundle")
    before = _outcome(candidate, canonical)
    attached = attach_execution_geometry(before, canonical_bundle=canonical)
    assert attached.provenance.source_sha256 == candidate.candidate_sha256
    assert attached.bundle.entry(EXECUTION_GEOMETRY_ARTIFACT_PATH).data == canonical_geometry_bytes(canonical)
    assert attached.bundle.entry(CANDIDATE_MANIFEST_ARTIFACT_PATH).role == "SOURCE"
    assert (
        attached.bundle.entry(CANDIDATE_MANIFEST_ARTIFACT_PATH).sha256 == candidate.candidate_sha256
    )
    assert attached.artifact_set_digest != before.artifact_set_digest
    audit = attached.bundle.native.descriptor["execution_geometry"]
    assert audit["path"] == EXECUTION_GEOMETRY_ARTIFACT_PATH
    assert audit["sha256"] == attached.bundle.entry(EXECUTION_GEOMETRY_ARTIFACT_PATH).sha256
    assert attached.bundle.native.digest != before.bundle.native.digest
    verify_execution_geometry_artifact(attached, canonical)

    # The frozen schema-4 projection accepts the attached set and binds the same digest.
    from tcenv.build_projection import project_full

    projected = project_full(attached, build_kind="native", identity=IDENTITY)
    assert projected["candidate_sha256"] == candidate.candidate_sha256
    assert projected["provenance"]["artifact_set_digest"] == attached.artifact_set_digest
    assert any(
        entry["path"] == EXECUTION_GEOMETRY_ARTIFACT_PATH for entry in projected["artifacts"]["entries"]
    )
    # A resolution/lifecycle drift of the *same* candidate is a different build subject.
    _, other_canonical = _candidate_and_bundle(tmp_path / "other", fallback_grid=(2, 1, 1))
    assert other_canonical.candidate_sha256 == canonical.candidate_sha256
    assert other_canonical.geometry_identity != canonical.geometry_identity
    other = attach_execution_geometry(_outcome(candidate, other_canonical), canonical_bundle=other_canonical)
    assert other.provenance.source_sha256 == attached.provenance.source_sha256
    assert other.artifact_set_digest != attached.artifact_set_digest
    assert other.bundle.native.digest != attached.bundle.native.digest


def test_a_single_source_build_keeps_its_byte_identity_and_adds_only_the_geometry(tmp_path):
    """@brief msl-single 的 SOURCE 仍是文件字节；只多一个几何 entry。"""
    candidate, canonical = _candidate_and_bundle(tmp_path, single=True)
    before = _outcome(candidate, canonical)
    attached = attach_execution_geometry(before, canonical_bundle=canonical)
    assert attached.provenance.source_sha256 == hashlib.sha256(b"kernel void fill() {}\n").hexdigest()
    assert attached.bundle.entry("kernel.metal").role == "SOURCE"
    assert attached.bundle.entry("kernel.metal").sha256 == candidate.candidate_sha256
    assert attached.bundle.entry(EXECUTION_GEOMETRY_ARTIFACT_PATH).data == canonical_geometry_bytes(canonical)
    assert CANDIDATE_MANIFEST_ARTIFACT_PATH not in {entry.path for entry in attached.bundle.entries}


def test_attaching_a_geometry_from_another_candidate_or_twice_is_refused(tmp_path):
    """@brief 身份不符/重复附加/构建 SOURCE 冲突都拒绝。"""
    candidate, canonical = _candidate_and_bundle(tmp_path / "a")
    other_candidate, other_canonical = _candidate_and_bundle(tmp_path / "b", grid=(4, 1, 1))
    with pytest.raises(GeometryArtifactError, match="source_sha256"):
        attach_execution_geometry(_outcome(candidate, canonical), canonical_bundle=other_canonical)
    attached = attach_execution_geometry(_outcome(candidate, canonical), canonical_bundle=canonical)
    with pytest.raises(GeometryArtifactError, match="already carries"):
        attach_execution_geometry(attached, canonical_bundle=canonical)

    bad_candidate, bad_canonical = _candidate_and_bundle(tmp_path / "c", single=False)
    with pytest.raises(GeometryArtifactError, match="SOURCE"):
        attach_execution_geometry(_outcome(bad_candidate, bad_canonical, with_source=True), canonical_bundle=bad_canonical)


def test_a_tampered_descriptor_or_geometry_entry_fails_cross_audit(tmp_path):
    """@brief native 交叉审计与 entry 字节都会被重算，篡改即拒绝。"""
    candidate, canonical = _candidate_and_bundle(tmp_path)
    attached = attach_execution_geometry(_outcome(candidate, canonical), canonical_bundle=canonical)
    tampered = attached.bundle.native.descriptor["execution_geometry"]
    assert tampered["sha256"] == attached.bundle.entry(EXECUTION_GEOMETRY_ARTIFACT_PATH).sha256
    _, other_canonical = _candidate_and_bundle(tmp_path / "other", fallback_grid=(2, 1, 1))
    other = attach_execution_geometry(_outcome(candidate, other_canonical), canonical_bundle=other_canonical)
    with pytest.raises(GeometryArtifactError):
        verify_execution_geometry_artifact(attached, other_canonical)
    assert other.artifact_set_digest != attached.artifact_set_digest
