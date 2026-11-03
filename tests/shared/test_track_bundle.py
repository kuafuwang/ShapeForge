"""Canonical candidate bundle: determinism, separate identities and geometry sensitivity (A2).

The canonical bundle is the A2 delivery object: candidate source entries (sorted) plus the frozen
geometry pair.  ``candidate_sha256`` stays the submission identity (candidate sources only) while
``bundle_digest`` is the build-subject digest, so a geometry drift changes the subject digest
without changing what the candidate submitted.  A3 binds the latter into
``provenance.artifact_set_digest``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tcenv.contract_errors import GeometryError
from tcenv.target import DEFAULT_TARGET_ID, load_target
from tcenv.taskcontract import load_task_contract
from tcenv.track import (
    CandidateArtifact,
    CandidateEntry,
    CandidateLifecycle,
    CanonicalBundle,
    ExecutionGeometry,
    GeometryStage,
    HostStep,
    PrepareDeclaration,
    TargetAbiLimits,
    derive_tensor_memory_contract,
    geometry_pair,
)

from ..support.v2_tasks import write_task

LIMITS = TargetAbiLimits(max_buffer_index=30, max_threads_per_threadgroup=1024, max_threadgroup_memory_bytes=32768)


def _candidate(track_id: str = "msl-bundle") -> CandidateArtifact:
    """@brief 一个两文件候选（Host Tiling 属于候选）。

    @param track_id track。
    @return 候选。
    """
    entries = (
        CandidateEntry(path="kernel.metal", role="device_kernel", media_type="text/plain", data=b"kernel void k(){}\n"),
        CandidateEntry(path="host/tiling.metal", role="host_tiling", media_type="text/plain", data=b"// tiling\n"),
    )
    if track_id == "msl-single":
        entries = entries[:1]
    return CandidateArtifact(track_id=track_id, entries=entries)


def _pair(tmp_path: Path, grid=({"ceil_div": ["M*H", 256]}, 1, 1)):
    """@brief 构造几何配对。

    @param tmp_path 临时目录。
    @param grid 候选 grid。
    @return 配对。
    """
    task = load_task_contract(write_task(tmp_path))
    abi = derive_tensor_memory_contract(task.instance, target=load_target(DEFAULT_TARGET_ID))
    candidate = ExecutionGeometry(source="candidate", stages=(GeometryStage(kernel="k", grid=grid, threads=(256, 1, 1)),))
    fallback = ExecutionGeometry(
        source="fallback", stages=(GeometryStage(kernel="fallback_k", grid=(1, 1, 1), threads=(64, 1, 1)),)
    )
    return geometry_pair(candidate, fallback, instance=task.instance, contract=abi, limits=LIMITS)


def test_the_bundle_is_canonical_in_entry_order_and_carries_both_identities(tmp_path):
    """@brief entry 顺序不同不影响字节/digest；提交身份与构建对象摘要分离。"""
    first = CandidateArtifact(
        track_id="msl-bundle",
        entries=(
            CandidateEntry(path="kernel.metal", role="device_kernel", media_type="text/plain", data=b"k\n"),
            CandidateEntry(path="host/tiling.metal", role="host_tiling", media_type="text/plain", data=b"t\n"),
        ),
    )
    second = CandidateArtifact(track_id="msl-bundle", entries=tuple(reversed(first.entries)))
    bundle_a = CanonicalBundle(candidate=first, pair=_pair(tmp_path / "a"))
    bundle_b = CanonicalBundle(candidate=second, pair=_pair(tmp_path / "b"))
    assert [entry.path for entry in bundle_a.entries] == ["host/tiling.metal", "kernel.metal"]
    assert bundle_a.candidate_sha256 == bundle_b.candidate_sha256 == first.candidate_sha256
    assert bundle_a.bundle_digest == bundle_b.bundle_digest
    assert bundle_a.geometry_identity == bundle_b.geometry_identity
    assert bundle_a.document()["bundle_digest"] == bundle_a.bundle_digest
    assert set(bundle_a.document()) == {
        "schema_id",
        "track_id",
        "candidate_sha256",
        "candidate_manifest",
        "geometry",
        "lifecycle",
        "bundle_digest",
    }


def test_a_geometry_change_moves_the_subject_digest_but_not_the_candidate_identity(tmp_path):
    """@brief 几何漂移改变构建对象摘要，候选提交身份不变（A3 的确认比较依赖这点）。"""
    candidate = _candidate()
    small = CanonicalBundle(candidate=candidate, pair=_pair(tmp_path / "small"))
    large_grid = ({"ceil_div": ["M*H", 128]}, 1, 1)
    large = CanonicalBundle(candidate=candidate, pair=_pair(tmp_path / "large", grid=large_grid))
    assert small.candidate_sha256 == large.candidate_sha256
    assert small.geometry_identity != large.geometry_identity
    assert small.bundle_digest != large.bundle_digest
    assert small.document()["geometry"]["candidate_resolution"]["stages"][0]["grid"] != large.document()["geometry"][
        "candidate_resolution"
    ]["stages"][0]["grid"]


def test_the_bundle_refuses_a_fallback_as_candidate_and_an_unknown_schema(tmp_path):
    """@brief fallback 不能冒充候选；未知 schema 拒绝。"""
    pair = _pair(tmp_path)
    candidate = _candidate()
    with pytest.raises(GeometryError, match="schema"):
        CanonicalBundle(candidate=candidate, pair=pair, schema_id="tcenv.track.canonical_bundle.9")
    fallback_only = geometry_pair(
        ExecutionGeometry(source="candidate", stages=(GeometryStage(kernel="k", grid=(1, 1, 1), threads=(4096, 1, 1)),)),
        instance=load_task_contract(write_task(tmp_path / "t")).instance,
        contract=derive_tensor_memory_contract(
            load_task_contract(write_task(tmp_path / "t")).instance, target=load_target(DEFAULT_TARGET_ID)
        ),
        limits=LIMITS,
    )
    assert fallback_only.candidate is None
    bundle = CanonicalBundle(candidate=candidate, pair=fallback_only)
    with pytest.raises(GeometryError, match="unusable"):
        _ = bundle.selection


def test_the_bundle_binds_the_prepare_run_lifecycle(tmp_path):
    """@brief 生命周期进入构建对象摘要；换 prepare 或换几何都改变摘要。"""
    candidate = _candidate()
    pair = _pair(tmp_path)
    assert pair.candidate is not None
    task = load_task_contract(write_task(tmp_path / "subject"))

    def lifecycle(geometry=pair.candidate, second_inside_timing=False):
        return CandidateLifecycle(
            track_id=candidate.track_id,
            geometry=geometry,
            host_steps=(
                HostStep(PrepareDeclaration(name="compile", input_dependent=False), role="harness"),
                HostStep(
                    PrepareDeclaration(name="tiling", input_dependent=False, inside_timing=second_inside_timing),
                    role="host_tiling",
                ),
            ),
            candidate_sha256=candidate.candidate_sha256,
            target_id=DEFAULT_TARGET_ID,
            instance_key=task.fingerprint(),
        )

    base_lifecycle = lifecycle()
    with_lifecycle = CanonicalBundle(candidate=candidate, pair=pair, lifecycle=base_lifecycle)
    assert with_lifecycle.document()["lifecycle"]["run"][0]["kernel"] == "k"
    assert with_lifecycle.lifecycle_identity == base_lifecycle.identity
    without = CanonicalBundle(candidate=candidate, pair=pair)
    assert without.document()["lifecycle"] is None
    assert with_lifecycle.candidate_sha256 == without.candidate_sha256
    assert with_lifecycle.bundle_digest != without.bundle_digest

    changed = CanonicalBundle(candidate=candidate, pair=pair, lifecycle=lifecycle(second_inside_timing=True))
    assert changed.bundle_digest != with_lifecycle.bundle_digest

    other = _pair(tmp_path, grid=({"ceil_div": ["M*H", 128]}, 1, 1))
    with pytest.raises(GeometryError, match="different resolved geometry"):
        CanonicalBundle(candidate=candidate, pair=pair, lifecycle=lifecycle(geometry=other.candidate))
    with pytest.raises(GeometryError, match="CandidateLifecycle"):
        CanonicalBundle(candidate=candidate, pair=pair, lifecycle="not-a-lifecycle")


def test_the_single_source_track_keeps_the_byte_sha256_identity(tmp_path):
    """@brief msl-single 的对象摘要仍以单文件字节 SHA-256 为提交身份。"""
    import hashlib

    candidate = _candidate("msl-single")
    bundle = CanonicalBundle(candidate=candidate, pair=_pair(tmp_path))
    assert bundle.candidate_sha256 == hashlib.sha256(b"kernel void k(){}\n").hexdigest()
    assert bundle.document()["track_id"] == "msl-single"
