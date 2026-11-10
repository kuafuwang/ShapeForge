"""The frozen A1 track plane: registry, ABI contract, geometry, candidate and renderer.

CPU only, backend free: the shipped Metal registry is read as JSON and the neutral fixture
supplies a v2 contract instance.  No backend module is imported, no build or device is used.
"""

from __future__ import annotations

import copy
import dataclasses
import hashlib

import pytest

from tcenv.contract_errors import TaskContractError, UnsupportedContractCapability
from tcenv.taskcontract import load_task_contract
from tcenv.track import (
    CANDIDATE_MANIFEST_SCHEMA_ID,
    CandidateArtifact,
    CandidateEntry,
    ExecutionGeometry,
    GeometryStage,
    TensorMemoryArg,
    TensorMemoryContract,
    TrackRenderer,
    registry_from_document,
)

from ..support.v2_tasks import write_task


def _registry_document() -> dict:
    """@brief 一份 backend-neutral 的 track registry 文档（负例的变异基底）。

    @return 文档。
    """
    return {
        "schema_version": 1,
        "backend_id": "metal",
        "tracks": [
            {
                "track_id": "msl-single",
                "track_version": 1,
                "delivery": "single_source",
                "status": "interface_only",
                "candidate_roles": ["device_kernel"],
                "capabilities": ["single_file_msl"],
                "renderer": {"implementation_id": "metal.track.msl-single.renderer", "implementation_version": 1},
                "validator": {"implementation_id": "metal.track.msl-single.validator", "implementation_version": 1},
                "build_adapter": {"implementation_id": "metal.track.msl-single.build", "implementation_version": 1},
            },
            {
                "track_id": "msl-bundle",
                "track_version": 1,
                "delivery": "candidate_bundle",
                "status": "interface_only",
                "candidate_roles": ["device_kernel", "host_tiling", "schedule"],
                "capabilities": ["candidate_host_tiling", "candidate_device_kernel", "workspace", "multi_stage"],
                "renderer": {"implementation_id": "metal.track.msl-bundle.renderer", "implementation_version": 1},
                "validator": {"implementation_id": "metal.track.msl-bundle.validator", "implementation_version": 1},
                "build_adapter": {"implementation_id": "metal.track.msl-bundle.build", "implementation_version": 1},
            },
        ],
    }


def _rejects(document: dict) -> None:
    """@brief 断言一份 tracks 文档被拒绝。

    @param document 文档。
    """
    with pytest.raises(TaskContractError):
        registry_from_document(document)


def _abi(backend_id: str = "metal") -> TensorMemoryContract:
    """@brief 一个手工构造的 ABI 合同（A1 没有 TargetAbiDeriver）。

    @param backend_id 后端 id。
    @return 合同。
    """
    return TensorMemoryContract(
        abi_id="abi.rmsnorm.fp16.small",
        backend_id=backend_id,
        target_id="apple-m4-max",
        tensors=(
            TensorMemoryArg(name="x", role="input", dtype="fp16", space="global", access="read", shape=(8, 64), strides=(64, 1)),
            TensorMemoryArg(name="w", role="input", dtype="fp16", space="global", access="read", shape=(64,), strides=(1,)),
            TensorMemoryArg(name="y", role="output", dtype="fp16", space="global", access="write", shape=(8, 64), strides=(64, 1)),
        ),
        acc_dtype="fp32",
        rtol=0.002,
        atol_rel=0.001,
        exact=False,
    )


def test_a_registry_with_interface_only_tracks_cannot_be_selected():
    """@brief 只有元数据、没有真实证据的 track 不可 available，也不能被选择。"""
    registry = registry_from_document(_registry_document())
    assert registry.backend_id == "metal"
    assert {track.track_id for track in registry.tracks} == {"msl-single", "msl-bundle"}
    assert all(track.status == "interface_only" for track in registry.tracks)
    assert registry.available() == ()
    assert registry.get("msl-single").delivery == "single_source"
    assert registry.get("msl-bundle").delivery == "candidate_bundle"
    assert registry.get("msl-bundle").candidate_roles == ("device_kernel", "host_tiling", "schedule")
    with pytest.raises(UnsupportedContractCapability):
        registry.require_available("msl-single")
    with pytest.raises(UnsupportedContractCapability, match="unknown track"):
        registry.get("npu-magic")
    with pytest.raises(UnsupportedContractCapability, match="version"):
        registry.get("msl-single", version=2)


def test_a_track_is_delivery_metadata_not_an_algorithm_or_op_specific_library():
    """@brief schema 与 dataclass 字段都没有 op_name/shape/tile/grid/lowering，Track 无法按题给 tiling。"""
    document = _registry_document()
    for track in document["tracks"]:
        for forbidden in ("op_name", "shapes", "tile", "grid", "threads", "lowering_profile", "kernel"):
            assert forbidden not in track
    names = {field.name for field in dataclasses.fields(type(registry_from_document(_registry_document()).get("msl-single")))}
    for forbidden in ("op_name", "shape", "tile", "grid", "threads", "lowering_profile", "algorithm"):
        assert forbidden not in names


def test_an_op_specific_or_tiling_field_is_refused_at_the_schema_level():
    """@brief 尝试把 Track 变成算法库的字段（op_name/tile/grid）一律拒绝。"""
    for field, value in (
        ("op_name", "rmsnorm"),
        ("tile", [64, 64]),
        ("grid", [1, 1, 1]),
        ("threads", [256, 1, 1]),
        ("lowering_profile", "mlx-gemm"),
    ):
        document = _registry_document()
        document["tracks"][0][field] = value
        _rejects(document)


def test_registry_version_duplicate_and_capability_negatives():
    """@brief 版本、重复 track、未知能力、交付角色组合都 fail closed。"""
    # MT1b: schema 2 is the live registry, schema 1 is a frozen replay input.  An *unknown*
    # version still fails closed, and a live loader refuses the replay-only schema.
    from tcenv.track.registry import load_track_registry

    document = _registry_document()
    document["schema_version"] = 3
    _rejects(document)

    # The same track id may coexist at two versions; the same (id, version) may not.
    document = _registry_document()
    second = copy.deepcopy(document["tracks"][0])
    second["track_version"] = 2
    document["tracks"].append(second)
    versioned = registry_from_document(document)
    assert versioned.versions("msl-single") == (1, 2)

    document = _registry_document()
    document["tracks"].append(copy.deepcopy(document["tracks"][0]))
    _rejects(document)

    # A schema-1 document is replay-only, never a new-run registry.
    import json as _json
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "tracks.json"
        path.write_text(_json.dumps(_registry_document()), encoding="utf-8")
        with pytest.raises(TaskContractError, match="frozen replay inputs only"):
            load_track_registry(path)

    document = _registry_document()
    document["tracks"][0]["capabilities"] = ["magic"]
    _rejects(document)

    document = _registry_document()
    document["tracks"][0]["candidate_roles"] = ["device_kernel", "host_tiling"]
    _rejects(document)

    document = _registry_document()
    document["tracks"][1]["candidate_roles"] = ["device_kernel"]
    _rejects(document)

    document = _registry_document()
    document["tracks"][0]["status"] = "available"
    _rejects(document)

    document = _registry_document()
    document["tracks"][0]["status"] = "interface_only"
    document["tracks"][0]["evidence"] = ["fake"]
    _rejects(document)


def test_tensor_memory_contract_locks_abi_memory_and_numerics_only():
    """@brief TensorMemoryContract 无 tile/并行度/fallback 字段；只含 ABI/内存/数值。"""
    contract = _abi()
    document = contract.document()
    assert set(document) == {"schema_version", "abi_id", "backend_id", "target_id", "tensors", "acc_dtype", "tolerances"}
    names = {field.name for field in dataclasses.fields(TensorMemoryContract)}
    for forbidden in ("tile", "tiles", "grid", "threads", "num_warps", "warps", "workspace_fills", "fallback"):
        assert forbidden not in names
    assert contract.tensor("x").access == "read"
    with pytest.raises(TaskContractError, match="duplicate tensor"):
        TensorMemoryContract(
            abi_id="a",
            backend_id="metal",
            target_id="t",
            tensors=contract.tensors + (contract.tensors[0],),
            acc_dtype="fp32",
            rtol=0.0,
            atol_rel=0.0,
            exact=False,
        )
    with pytest.raises(TaskContractError, match="output needs write"):
        TensorMemoryArg(name="y", role="output", dtype="fp16", space="global", access="read", shape=(1,), strides=(1,))


def test_execution_geometry_is_only_the_frozen_a1_field_set():
    """@brief A1 只冻结 ExecutionGeometry 字段；执行身份/几何校验属 A2/A3。"""
    candidate = ExecutionGeometry(
        source="candidate",
        stages=(GeometryStage(kernel="rmsnorm", grid=("ceil_div(N, 256)", 1, 1), threads=(256, 1, 1)),),
    )
    fallback = ExecutionGeometry(
        source="fallback",
        stages=(GeometryStage(kernel="rmsnorm", grid=(1, 1, 1), threads=(64, 1, 1)),),
    )
    assert candidate.document()["source"] == "candidate"
    assert fallback.document()["source"] == "fallback"
    assert candidate.is_fallback is False and fallback.is_fallback is True
    fields = {field.name for field in dataclasses.fields(ExecutionGeometry)}
    assert fields == {"source", "stages", "schema_version"}
    assert "fallback" not in fields and "op_name" not in fields and "tile" not in fields
    # A1 intentionally does not expose an execution identity helper.
    import tcenv.track.plane as plane

    assert not hasattr(plane, "geometry_identity")


def test_geometry_stage_is_a_field_projection_not_a_validator():
    """@brief stage 只做字段投影；表达式/名字校验留给 A2。"""
    stage = GeometryStage(
        kernel="k",
        grid=({"ceil_div": ["N", 256]}, 1, 1),
        threads=(256, 1, 1),
        threadgroup_bytes=1024,
        workspace=(("M", 4),),
        when=("gt", "K", 0),
        derived_scalars=(("BLOCK_COUNT", {"ceil_div": ["N", 256]}),),
    )
    document = stage.document()
    assert document["derived_scalars"] == [{"name": "BLOCK_COUNT", "term": {"ceil_div": ["N", 256]}}]
    fields = {field.name for field in dataclasses.fields(GeometryStage)}
    assert fields == {
        "kernel",
        "grid",
        "threads",
        "threadgroup_bytes",
        "workspace",
        "when",
        "derived_scalars",
    }
    # A1 does not reject a short grid or duplicate derived names; A2 owns those rules.
    assert GeometryStage(kernel="k", grid=(1, 1), threads=(1, 1, 1)).document()["grid"] == [1, 1]
    assert GeometryStage(
        kernel="k", grid=(1, 1, 1), threads=(1, 1, 1), derived_scalars=(("A", 1), ("A", 2))
    ).document()["derived_scalars"] == [{"name": "A", "term": 1}, {"name": "A", "term": 2}]


def _single_candidate(data: bytes = b"kernel void rmsnorm_kernel() {}\n") -> CandidateArtifact:
    """@brief 一个单文件 msl-single 候选。

    @param data 文件字节。
    @return 候选工件。
    """
    return CandidateArtifact(
        track_id="msl-single",
        entries=(CandidateEntry(path="kernel.metal", role="device_kernel", media_type="text/plain", data=data),),
    )


def _bundle_candidate() -> CandidateArtifact:
    """@brief 一个 msl-bundle 候选（Host Tiling 属于候选）。"""
    return CandidateArtifact(
        track_id="msl-bundle",
        entries=(
            CandidateEntry(path="kernel.metal", role="device_kernel", media_type="text/plain", data=b"kernel void k(){}\n"),
            CandidateEntry(path="host/tiling.metal", role="host_tiling", media_type="text/plain", data=b"// host tiling\n"),
            CandidateEntry(path="host/schedule.json", role="schedule", media_type="application/json", data=b"{}\n"),
        ),
    )


def test_a_single_source_candidate_keeps_the_existing_bytes_sha256():
    """@brief msl-single 的 candidate_sha256 就是文件字节 SHA-256（与现有单文件身份一致）。"""
    artifact = _single_candidate()
    assert artifact.single_source is True
    assert artifact.candidate_sha256 == hashlib.sha256(b"kernel void rmsnorm_kernel() {}\n").hexdigest()
    registry = registry_from_document(_registry_document())
    assert artifact.validate_for(registry.get("msl-single")) == "accepted"


def test_a_bundle_candidate_digest_is_a_manifest_over_candidate_sources_only():
    """@brief bundle 身份是仅候选源 entries 的 canonical manifest digest，与顺序无关。"""
    artifact = _bundle_candidate()
    reversed_artifact = CandidateArtifact(track_id=artifact.track_id, entries=tuple(reversed(artifact.entries)))
    assert artifact.candidate_sha256 == reversed_artifact.candidate_sha256
    assert artifact.manifest()["schema_id"] == CANDIDATE_MANIFEST_SCHEMA_ID
    changed = CandidateArtifact(
        track_id=artifact.track_id,
        entries=artifact.entries[:-1]
        + (CandidateEntry(path="host/schedule.json", role="schedule", media_type="application/json", data=b'{"x":1}\n'),),
    )
    assert changed.candidate_sha256 != artifact.candidate_sha256
    assert registry_from_document(_registry_document()).get("msl-bundle").delivery == "candidate_bundle"
    assert artifact.validate_for(registry_from_document(_registry_document()).get("msl-bundle")) == "accepted"


def test_the_same_candidate_bytes_have_one_verdict_regardless_of_author_source():
    """@brief 同一候选字节 + 同一 track 的判决键与结论不因作者/LLM/committee/会话改变。"""
    bytes_ = b"kernel void k(){}\n"
    human = _single_candidate(bytes_)
    llm = _single_candidate(bytes_)
    committee = _single_candidate(bytes_)
    labels = {"human": human, "llm": llm, "committee-first": committee, "session-7": _single_candidate(bytes_)}
    assert len({artifact.candidate_sha256 for artifact in labels.values()}) == 1
    assert len({artifact.verdict_key() for artifact in labels.values()}) == 1
    registry = registry_from_document(_registry_document())
    assert {artifact.validate_for(registry.get("msl-single")) for artifact in labels.values()} == {"accepted"}
    names = {field.name for field in dataclasses.fields(CandidateArtifact)}
    for forbidden in ("author", "model", "llm", "committee", "session", "source_label"):
        assert forbidden not in names


def test_candidate_role_and_track_delivery_mismatches_are_refused():
    """@brief 单文件 track 拒绝 bundle；bundle track 必须含候选 Host Tiling。"""
    registry = registry_from_document(_registry_document())
    mislabelled = CandidateArtifact(
        track_id="msl-single",
        entries=(
            CandidateEntry(path="a.metal", role="device_kernel", media_type="text/plain", data=b"a"),
            CandidateEntry(path="b.metal", role="device_kernel", media_type="text/plain", data=b"b"),
        ),
    )
    with pytest.raises(TaskContractError, match="single-source"):
        mislabelled.validate_for(registry.get("msl-single"))
    with pytest.raises(TaskContractError, match="not 'msl-single'"):
        _bundle_candidate().validate_for(registry.get("msl-single"))
    missing_host = CandidateArtifact(
        track_id="msl-bundle",
        entries=(CandidateEntry(path="k.metal", role="device_kernel", media_type="text/plain", data=b"k\n"),),
    )
    with pytest.raises(TaskContractError, match="Host Tiling"):
        missing_host.validate_for(registry.get("msl-bundle"))


def test_candidate_roles_must_be_registered_on_the_target_track():
    """@brief candidate entries 的 role 必须是该 track 的 candidate_roles 子集。"""
    registry = registry_from_document(_registry_document())
    with_launch = CandidateArtifact(
        track_id="msl-bundle",
        entries=_bundle_candidate().entries
        + (CandidateEntry(path="host/launch.json", role="launch", media_type="application/json", data=b"{}\n"),),
    )
    with pytest.raises(TaskContractError, match="not registered"):
        with_launch.validate_for(registry.get("msl-bundle"))
    single_with_launch = CandidateArtifact(
        track_id="msl-single",
        entries=_single_candidate().entries
        + (CandidateEntry(path="launch.json", role="launch", media_type="application/json", data=b"{}\n"),),
    )
    with pytest.raises(TaskContractError, match="not registered"):
        single_with_launch.validate_for(registry.get("msl-single"))


def test_candidate_paths_and_bytes_fail_closed():
    """@brief 绝对/穿越/反斜杠路径、大小写冲突和空字节都拒绝。"""
    for path in (
        "/abs.metal",
        "../up.metal",
        "a\\b.metal",
        "./a.metal",
        "a/../b.metal",
        "a/../../b.metal",
        "a/b/../../c.metal",
        "a/./b.metal",
        "a/..",
        ".",
        "..",
    ):
        with pytest.raises(TaskContractError, match="relative POSIX path|normalized|must not contain"):
            CandidateEntry(path=path, role="device_kernel", media_type="text/plain", data=b"k")
    with pytest.raises(TaskContractError, match="collide"):
        CandidateArtifact(
            track_id="msl-bundle",
            entries=(
                CandidateEntry(path="A.metal", role="device_kernel", media_type="text/plain", data=b"1"),
                CandidateEntry(path="a.metal", role="host_tiling", media_type="text/plain", data=b"2"),
            ),
        )
    with pytest.raises(TaskContractError, match="non-empty"):
        CandidateEntry(path="k.metal", role="device_kernel", media_type="text/plain", data=b"")
    with pytest.raises(TaskContractError, match="role"):
        CandidateEntry(path="k.metal", role="oracle", media_type="text/plain", data=b"k")


def test_the_track_renderer_requires_an_abi_and_is_deterministic(tmp_path):
    """@brief renderer 不伪造 ABI；有 ABI 时只格式化声明事实，三次渲染完全相同。"""
    task_dir = write_task(tmp_path)
    contract = load_task_contract(task_dir)
    registry = registry_from_document(_registry_document())
    track = registry.get("msl-single")
    renderer = TrackRenderer.for_track(track)
    with pytest.raises(UnsupportedContractCapability, match="TensorMemoryContract"):
        renderer.render(
            backend_id="metal",
            instance=contract.instance,
            reference_source=contract.reference_source,
            contract=None,
            display=contract.display,
            track=track,
        )
    first = renderer.render(
        backend_id="metal",
        instance=contract.instance,
        reference_source=contract.reference_source,
        contract=_abi(),
        display=contract.display,
        track=track,
    )
    second = renderer.render(
        backend_id="metal",
        instance=contract.instance,
        reference_source=contract.reference_source,
        contract=_abi(),
        display=contract.display,
        track=track,
    )
    assert first.document() == second.document()
    assert first.abi_document == _abi().document()
    assert first.statement == contract.statement
    assert first.reference_source == contract.reference_source
    names = {field.name for field in dataclasses.fields(type(first))}
    for forbidden in ("author", "llm", "committee", "session", "tile"):
        assert forbidden not in names


def test_the_track_renderer_rejects_unregistered_identity_and_backend(tmp_path):
    """@brief renderer 身份/后端不匹配时拒绝，不能借渲染串用 track。"""
    contract = load_task_contract(write_task(tmp_path))
    registry = registry_from_document(_registry_document())
    track = registry.get("msl-single")
    wrong = TrackRenderer(renderer_id="someone.else", renderer_version=1)
    with pytest.raises(UnsupportedContractCapability, match="not the renderer registered"):
        wrong.render(
            backend_id="metal",
            instance=contract.instance,
            reference_source=contract.reference_source,
            contract=_abi(),
            display=contract.display,
            track=track,
        )
    with pytest.raises(TaskContractError, match="backend"):
        TrackRenderer.for_track(track).render(
            backend_id="另一个后端",
            instance=contract.instance,
            reference_source=contract.reference_source,
            contract=_abi(),
            display=contract.display,
            track=track,
        )
