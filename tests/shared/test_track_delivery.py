"""Candidate bundle delivery: geometry entry and manifest identity (A3, CPU)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from tcenv.contract_errors import GeometryError, TaskContractError
from tcenv.target import DEFAULT_TARGET_ID, load_target
from tcenv.taskcontract import load_task_contract
from tcenv.track import (
    CandidateArtifact,
    CandidateEntry,
    ExecutionGeometry,
    TargetAbiLimits,
    candidate_from_single_source,
    candidate_geometry,
    derive_tensor_memory_contract,
    parse_candidate_geometry,
    resolve_geometry,
)

from ..support.v2_tasks import write_task

LIMITS = TargetAbiLimits(max_buffer_index=30, max_threads_per_threadgroup=1024, max_threadgroup_memory_bytes=32768)


def _geometry_document(grid=None, second_when=None) -> dict:
    """@brief 一个合法的候选几何文档。

    @param grid 第一段 grid。
    @param second_when 第二段 when。
    @return 文档。
    """
    return {
        "schema_version": 1,
        "source": "candidate",
        "stages": [
            {
                "kernel": "fill",
                "grid": list(grid or [{"ceil_div": ["M*H", 256]}, 1, 1]),
                "threads": [256, 1, 1],
                "threadgroup_bytes": {"*": ["BLOCKS", 64]},
                "workspace": [],
                "when": None,
                "derived_scalars": [{"name": "BLOCKS", "term": {"ceil_div": ["M*H", 256]}}],
            },
            {
                "kernel": "reduce",
                "grid": [1, 1, 1],
                "threads": [64, 1, 1],
                "when": second_when or ["gt", "M", 10**9],
            },
        ],
    }


def _candidate(document=None, *, path="host/execution_geometry.json", media="application/json") -> CandidateArtifact:
    """@brief 一个含 schedule entry 的 msl-bundle 候选。

    @param document 几何文档；省略时用合法文档。
    @param path schedule entry 路径。
    @param media schedule entry 媒体类型。
    @return 候选。
    """
    geometry = json.dumps(document if document is not None else _geometry_document(), sort_keys=True).encode()
    return CandidateArtifact(
        track_id="msl-bundle",
        entries=(
            CandidateEntry(path="kernel.metal", role="device_kernel", media_type="text/plain", data=b"kernel void fill(){}\n"),
            CandidateEntry(path="host/tiling.metal", role="host_tiling", media_type="text/plain", data=b"// tiling\n"),
            CandidateEntry(path=path, role="schedule", media_type=media, data=geometry),
        ),
    )


def test_the_candidate_geometry_entry_is_parsed_by_the_one_typed_resolver(tmp_path):
    """@brief 候选几何 entry 进入 A2 typed 几何；同文档同求值。"""
    task = load_task_contract(write_task(tmp_path))
    abi = derive_tensor_memory_contract(task.instance, target=load_target(DEFAULT_TARGET_ID))
    geometry = candidate_geometry(_candidate())
    assert isinstance(geometry, ExecutionGeometry)
    assert geometry.source == "candidate"
    assert [stage.kernel for stage in geometry.stages] == ["fill", "reduce"]
    resolved = resolve_geometry(geometry, instance=task.instance, contract=abi, limits=LIMITS)
    elements = task.instance.symbols["M"] * task.instance.symbols["H"]
    assert resolved.active_stages[0].grid[0] == elements // 256
    assert resolved.active_stages[0].threadgroup_bytes == (elements // 256) * 64
    assert [stage.kernel for stage in resolved.active_stages] == ["fill"]


def test_the_candidate_geometry_document_is_closed_and_refuses_harness_shortcuts():
    """@brief 几何 entry 是闭集字段；fallback/开放字段/tiling 选项一律拒绝。"""
    for mutate, message in (
        (lambda d: d.update(source="fallback"), "source 'candidate'"),
        (lambda d: d.update(schema_version=2), "unsupported candidate geometry schema"),
        (lambda d: d.update(stages=[]), "at least one stage"),
        (lambda d: d.update(tile=16), "expected exactly"),
        (lambda d: d.pop("stages"), "expected exactly"),
        (lambda d: d["stages"][0].update(grid=[1, 1]), "three terms"),
        (lambda d: d["stages"][0].update(num_warps=4), "unregistered fields"),
        (lambda d: d["stages"][1].pop("threads"), "missing"),
        (lambda d: d["stages"][0].update(derived_scalars=[{"name": "BLOCKS", "slot": 3}]), "name and term"),
    ):
        document = _geometry_document()
        mutate(document)
        with pytest.raises(GeometryError, match=message):
            parse_candidate_geometry(document)


def test_a_bundle_owns_exactly_one_geometry_entry():
    """@brief 没有、重复、错路径或错媒体类型的 schedule entry 都拒绝。"""
    with pytest.raises(GeometryError, match="ships no schedule entry"):
        candidate_geometry(
            CandidateArtifact(
                track_id="msl-bundle",
                entries=(
                    CandidateEntry(path="kernel.metal", role="device_kernel", media_type="text/plain", data=b"k\n"),
                    CandidateEntry(path="host/tiling.metal", role="host_tiling", media_type="text/plain", data=b"t\n"),
                ),
            )
        )
    two = _candidate()
    duplicate = CandidateArtifact(
        track_id=two.track_id,
        entries=two.entries + (CandidateEntry(path="second.json", role="schedule", media_type="application/json", data=b"{}"),),
    )
    with pytest.raises(GeometryError, match="2 schedule entries"):
        candidate_geometry(duplicate)
    with pytest.raises(GeometryError, match="fixed candidate geometry path"):
        candidate_geometry(_candidate(path="host/geometry.json"))
    with pytest.raises(GeometryError, match="media type"):
        candidate_geometry(_candidate(media="text/plain"))
    with pytest.raises(GeometryError, match="not UTF-8"):
        candidate_geometry(
            CandidateArtifact(
                track_id="msl-bundle",
                entries=(
                    CandidateEntry(path="kernel.metal", role="device_kernel", media_type="text/plain", data=b"k\n"),
                    CandidateEntry(path="host/tiling.metal", role="host_tiling", media_type="text/plain", data=b"t\n"),
                    CandidateEntry(path="host/execution_geometry.json", role="schedule", media_type="application/json", data=b"\xff"),
                ),
            )
        )


def test_the_bundle_identity_is_the_candidate_manifest_digest_not_the_envelope_hash():
    """@brief 提交身份是 manifest digest；顺序不变，任一 entry 字节变化都改变它。"""
    candidate = _candidate()
    assert not candidate.single_source
    assert candidate.candidate_sha256 == hashlib.sha256(candidate.manifest_bytes).hexdigest()
    reordered = CandidateArtifact(track_id=candidate.track_id, entries=tuple(reversed(candidate.entries)))
    assert reordered.candidate_sha256 == candidate.candidate_sha256
    assert reordered.manifest_bytes == candidate.manifest_bytes
    changed = CandidateArtifact(
        track_id=candidate.track_id,
        entries=tuple(
            CandidateEntry(path=entry.path, role=entry.role, media_type=entry.media_type, data=entry.data + b"\n")
            if entry.path == "host/tiling.metal"
            else entry
            for entry in candidate.entries
        ),
    )
    assert changed.candidate_sha256 != candidate.candidate_sha256
    single = candidate_from_single_source("kernel void k(){}\n")
    assert single.candidate_sha256 == hashlib.sha256(b"kernel void k(){}\n").hexdigest()
    with pytest.raises(TaskContractError, match="non-empty source"):
        candidate_from_single_source("  \n")
