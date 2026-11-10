"""``evaluate_task``: formal track, lifecycle, build, frozen geometry, report identity (A3, CPU)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tcenv.contract_errors import TaskContractError, UnsupportedContractCapability
from tcenv.remote.artifact_builder import (
    BuildArtifact,
    BuildEvidence,
    BuildOutcome,
    BuildProfile,
    BuildProvenance,
)
from tcenv.build_projection import project_full
from tcenv.evaluate import EvalReport
from tcenv.remote.artifact_bundle import ArtifactBundle, BundleEntry, NativeExecutable
from tcenv.schema import PROTOCOL_VERSION
from tcenv.target import DEFAULT_TARGET_ID
from tcenv.taskcontract import discover_standard_tasks
from tcenv.taskcontract.runner import (
    CandidateBuildRequest,
    CandidateEvaluationRequest,
    default_geometry_resolver,
    evaluate_task,
    select_track,
)
from tcenv.track import (
    CandidateArtifact,
    CandidateEntry,
    CandidateLifecycle,
    HostStep,
    PrepareDeclaration,
    candidate_from_single_source,
)
from tcenv.track import REPO_ROOT

from ..support.v2_tasks import write_task

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


def _tracks(tmp_path: Path, *, status="available") -> Path:
    """@brief 写一份测试用 tracks.json（schema 2，闭集，两份 MSL 交付机制）。

    MT1b moved the live registry to schema 2: a new run selects from it, while a schema-1 document
    is a frozen replay input.  This fixture is a *live* registry, so it must be schema 2.
    """
    document = {
        "schema_version": 2,
        "backend_id": "metal",
        "tracks": [
            {
                "track_id": "msl-single",
                "track_version": 1,
                "delivery": "single_source",
                "status": status,
                "candidate_roles": ["device_kernel"],
                "capabilities": ["single_file_msl"],
                "renderer": {"implementation_id": "metal.track.msl-single.renderer", "implementation_version": 1},
                "validator": {"implementation_id": "metal.track.msl-single.validator", "implementation_version": 1},
                "build_adapter": {"implementation_id": "metal.track.msl-single.build", "implementation_version": 1},
                **({"evidence": ["test"]} if status == "available" else {}),
            },
            {
                "track_id": "msl-bundle",
                "track_version": 1,
                "delivery": "candidate_bundle",
                "status": status,
                "candidate_roles": ["device_kernel", "host_tiling", "schedule"],
                "capabilities": ["candidate_host_tiling", "candidate_device_kernel", "workspace", "multi_stage"],
                "renderer": {"implementation_id": "metal.track.msl-bundle.renderer", "implementation_version": 1},
                "validator": {"implementation_id": "metal.track.msl-bundle.validator", "implementation_version": 1},
                "build_adapter": {"implementation_id": "metal.track.msl-bundle.build", "implementation_version": 1},
                **({"evidence": ["test"]} if status == "available" else {}),
            },
        ],
    }
    path = tmp_path / "tracks.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


class FakeBuilder:
    """@brief 用候选 entry 组装 BuildOutcome 的假适配器；SOURE 规则由真实构建器负责。"""

    def __init__(self, *, source_mode=False, wrong_source=False):
        self.source_mode = source_mode
        self.wrong_source = wrong_source
        self.requests: list[CandidateBuildRequest] = []

    def build(self, request: CandidateBuildRequest) -> BuildOutcome:
        self.requests.append(request)
        candidate = request.candidate
        entries: list[BundleEntry] = []
        if self.source_mode:
            entry = candidate.entries[0]
            entries.append(
                BundleEntry(
                    path=entry.path,
                    role="SOURCE",
                    media_type=entry.media_type,
                    size_bytes=len(entry.data),
                    sha256=entry.sha256,
                    data=entry.data,
                )
            )
            native_path = entry.path
        else:
            for entry in candidate.entries:
                entries.append(
                    BundleEntry(
                        path=entry.path,
                        role="AUXILIARY",
                        media_type=entry.media_type,
                        size_bytes=len(entry.data),
                        sha256=entry.sha256,
                        data=entry.data,
                    )
                )
            native_path = "kernel.metal"
        native = NativeExecutable(
            backend_id="metal",
            kind="compute-pipeline",
            entry_path=native_path,
            entry_sha256=next(entry.sha256 for entry in entries if entry.path == native_path),
            descriptor={"backend": "metal", "kernel_name": "fill"},
            handle=object(),
        )
        bundle = ArtifactBundle(entries=tuple(entries), native=native)
        source_sha256 = ("0" * 64) if self.wrong_source else candidate.candidate_sha256
        provenance = BuildProvenance(
            profile=PROFILE, source_sha256=source_sha256, artifact_set_digest=bundle.artifact_set_digest
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


class FakeEvaluator:
    """@brief 返回真实合法 schema-4 报告；可注入 digest/结构负例。"""

    def __init__(self, *, wrong_digest=False, bad_schema=None):
        self.wrong_digest = wrong_digest
        self.bad_schema = bad_schema
        self.requests: list[CandidateEvaluationRequest] = []

    def evaluate(self, request: CandidateEvaluationRequest) -> dict:
        self.requests.append(request)
        build = project_full(
            request.build,
            build_kind="native",
            identity={"opspec_digest": "a" * 64, "lowering_profile_digest": "b" * 64},
        )
        task = request.build_request.locator
        report = EvalReport(
            task=task.op_name,
            dtype=task.dtype,
            size=task.size,
            mode="full",
            status="ok",
            stage="bench",
            seed=0,
            build=build,
            correctness={"passed": True},
            protocol_version=PROTOCOL_VERSION,
            target_id=DEFAULT_TARGET_ID,
            backend="metal",
            backend_version="1",
        )
        document = report.to_json()
        if self.wrong_digest:
            document["build"]["provenance"]["artifact_set_digest"] = "f" * 64
            document["build"]["artifacts"]["artifact_set_digest"] = "f" * 64
        if self.bad_schema == "missing_version":
            document.pop("schema_version")
        elif self.bad_schema == "free_field":
            document["free_field"] = True
        elif self.bad_schema == "missing_build":
            document.pop("build")
        return document


def _bundle_candidate(geometry_document: dict) -> CandidateArtifact:
    return CandidateArtifact(
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


def _geometry_document(grid=(2, 1, 1)) -> dict:
    return {"schema_version": 1, "source": "candidate", "stages": [{"kernel": "fill", "grid": list(grid), "threads": [256, 1, 1]}]}


def _lifecycle(resolved, candidate, *, cached_schedule=False) -> CandidateLifecycle:
    steps = [
        HostStep(PrepareDeclaration(name="tiling", input_dependent=True, inside_timing=True), role="host_tiling"),
        HostStep(
            PrepareDeclaration(
                name="schedule",
                input_dependent=not cached_schedule,
                cached=cached_schedule,
                inside_timing=not cached_schedule,
            ),
            role="schedule",
        ),
    ]
    return CandidateLifecycle(
        track_id="msl-bundle",
        geometry=resolved,
        host_steps=tuple(steps),
        candidate_sha256=candidate.candidate_sha256,
        target_id=DEFAULT_TARGET_ID,
        instance_key="task-instance",
    )


def test_track_selection_refuses_missing_unknown_and_interface_only_tracks(tmp_path):
    """@brief v2 没有隐式 track；interface_only 不能产出证据。"""
    with pytest.raises(UnsupportedContractCapability, match="no implicit Track"):
        select_track(None)
    with pytest.raises(TaskContractError, match="unknown track"):
        select_track("does-not-exist", registry_path=_tracks(tmp_path))
    with pytest.raises(UnsupportedContractCapability, match="interface_only|available"):
        select_track("msl-bundle", registry_path=_tracks(tmp_path, status="interface_only"))
    selection = select_track("msl-bundle", registry_path=_tracks(tmp_path))
    assert selection.document()["track_id"] == "msl-bundle"
    assert selection.document()["track_version"] == 1
    assert selection.document()["delivery"] == "candidate_bundle"
    assert selection.document()["renderer"] == {
        "implementation_id": "metal.track.msl-bundle.renderer",
        "implementation_version": 1,
    }
    # Track 只描述交付机制：即使未来加了字段，也不能出现按算子/shape/tile 的 tiling。
    assert not {"op_name", "shape", "tiling", "grid", "threads"} & set(selection.track.document())
    with pytest.raises(TaskContractError):
        select_track("msl-bundle", version=9, registry_path=_tracks(tmp_path))


def test_evaluate_task_composes_the_whole_v2_pipeline(tmp_path):
    """@brief locator→track→ABI→几何→构建→附加几何→报告；生命周期改变构建立即改摘要。"""
    root = tmp_path / "tasks"
    write_task(root)
    locator = discover_standard_tasks(root, code_root=REPO_ROOT)[0]
    candidate = _bundle_candidate(_geometry_document())
    builder, evaluator = FakeBuilder(), FakeEvaluator()

    result = evaluate_task(
        locator,
        track_id="msl-bundle",
        candidate=candidate,
        builder=builder,
        evaluator=evaluator,
        lifecycle_factory=lambda resolved: _lifecycle(resolved, candidate),
        track_registry_path=_tracks(tmp_path),
    )
    assert result.track["track_id"] == "msl-bundle"
    assert result.candidate_sha256 == candidate.candidate_sha256
    assert result.task_pack_digest == locator.deployment_pack_digest
    assert result.effective_correctness_only is True
    assert result.geometry_identity
    assert result.presentation_sha256 == result.document()["presentation_sha256"]
    assert result.artifact_set_digest == evaluator.requests[0].build.artifact_set_digest
    assert result.report["correctness"]["passed"] is True
    assert builder.requests[0].candidate.candidate_sha256 == candidate.candidate_sha256
    assert builder.requests[0].selection.provider == "candidate"
    assert locator.document()["deployment_pack_digest"] in result.document()["locator"]["deployment_pack_digest"]

    other = evaluate_task(
        locator,
        track_id="msl-bundle",
        candidate=candidate,
        builder=FakeBuilder(),
        evaluator=FakeEvaluator(),
        lifecycle_factory=lambda resolved: _lifecycle(resolved, candidate, cached_schedule=True),
        track_registry_path=_tracks(tmp_path),
    )
    assert other.candidate_sha256 == result.candidate_sha256
    assert other.bundle_digest != result.bundle_digest
    assert other.artifact_set_digest != result.artifact_set_digest
    assert other.identity != result.identity


def test_evaluate_task_refuses_a_bundle_without_a_lifecycle_or_a_drifting_report(tmp_path):
    """@brief host work 未登记、builder 身份错、报告 digest 错都拒绝。"""
    root = tmp_path / "tasks"
    write_task(root)
    locator = discover_standard_tasks(root, code_root=REPO_ROOT)[0]
    candidate = _bundle_candidate(_geometry_document())
    with pytest.raises(UnsupportedContractCapability, match="CandidateLifecycle"):
        evaluate_task(
            locator,
            track_id="msl-bundle",
            candidate=candidate,
            builder=FakeBuilder(),
            evaluator=FakeEvaluator(),
            track_registry_path=_tracks(tmp_path),
        )
    with pytest.raises(TaskContractError, match="source identity"):
        evaluate_task(
            locator,
            track_id="msl-bundle",
            candidate=candidate,
            builder=FakeBuilder(wrong_source=True),
            evaluator=FakeEvaluator(),
            lifecycle_factory=lambda resolved: _lifecycle(resolved, candidate),
            track_registry_path=_tracks(tmp_path),
        )
    with pytest.raises(TaskContractError, match="artifact_set_digest"):
        evaluate_task(
            locator,
            track_id="msl-bundle",
            candidate=candidate,
            builder=FakeBuilder(),
            evaluator=FakeEvaluator(wrong_digest=True),
            lifecycle_factory=lambda resolved: _lifecycle(resolved, candidate),
            track_registry_path=_tracks(tmp_path),
        )
    with pytest.raises(TaskContractError, match="exactly one"):
        evaluate_task(
            locator,
            track_id="msl-bundle",
            candidate_source="x",
            candidate=candidate,
            builder=FakeBuilder(),
            evaluator=FakeEvaluator(),
            track_registry_path=_tracks(tmp_path),
        )


def test_a_single_source_candidate_uses_the_injected_launch_parser(tmp_path):
    """@brief msl-single 无默认几何：必须注入后端 launch 解析器；否则拒绝。"""
    root = tmp_path / "tasks"
    write_task(root)
    locator = discover_standard_tasks(root, code_root=REPO_ROOT)[0]
    candidate = candidate_from_single_source("kernel void fill() {}\n")
    with pytest.raises(UnsupportedContractCapability, match="backend adapter"):
        default_geometry_resolver(select_track("msl-single", registry_path=_tracks(tmp_path)).track)

    def resolve(_candidate):
        from tcenv.track import ExecutionGeometry, GeometryStage

        return ExecutionGeometry(
            source="candidate", stages=(GeometryStage(kernel="fill", grid=(2, 1, 1), threads=(256, 1, 1)),)
        )

    result = evaluate_task(
        locator,
        track_id="msl-single",
        candidate_source="kernel void fill() {}\n",
        builder=FakeBuilder(source_mode=True),
        evaluator=FakeEvaluator(),
        geometry_resolver=resolve,
        track_registry_path=_tracks(tmp_path),
    )
    assert result.candidate_sha256 == candidate.candidate_sha256
    assert result.artifact_set_digest


def test_evaluate_task_decodes_a_single_submission_envelope_without_splitting_segments(tmp_path):
    """@brief 分段 envelope 作为一次提交进入 evaluate_task，身份仍是候选 manifest digest。"""
    from tcenv.track import encode_envelope

    root = tmp_path / "tasks"
    write_task(root)
    locator = discover_standard_tasks(root, code_root=REPO_ROOT)[0]
    candidate = _bundle_candidate(_geometry_document())
    text = encode_envelope(candidate)
    result = evaluate_task(
        locator,
        track_id="msl-bundle",
        candidate_source=text,
        builder=FakeBuilder(),
        evaluator=FakeEvaluator(),
        lifecycle_factory=lambda resolved: _lifecycle(resolved, candidate),
        track_registry_path=_tracks(tmp_path),
    )
    assert result.candidate_sha256 == candidate.candidate_sha256
    assert result.report["build"]["provenance"]["source_sha256"] == candidate.candidate_sha256


@pytest.mark.parametrize("bad_schema", ["missing_version", "free_field", "missing_build"])
def test_evaluate_task_refuses_a_non_schema4_report(tmp_path, bad_schema):
    """@brief 评测报告必须过唯一 schema-4 读取器；缺版本/自由字段/缺 build 都拒绝。"""
    root = tmp_path / "tasks"
    write_task(root)
    locator = discover_standard_tasks(root, code_root=REPO_ROOT)[0]
    candidate = _bundle_candidate(_geometry_document())
    with pytest.raises(TaskContractError, match="schema-4"):
        evaluate_task(
            locator,
            track_id="msl-bundle",
            candidate=candidate,
            builder=FakeBuilder(),
            evaluator=FakeEvaluator(bad_schema=bad_schema),
            lifecycle_factory=lambda resolved: _lifecycle(resolved, candidate),
            track_registry_path=_tracks(tmp_path),
        )


def test_evaluate_task_refuses_a_task_that_drifted_after_discovery(tmp_path):
    """@brief locator 后修改任务文件，evaluate_task 不能拿旧 digest 评测新 reference。"""
    from tcenv.taskcontract import TaskDiscoveryError

    root = tmp_path / "tasks"
    task_dir = write_task(root)
    locator = discover_standard_tasks(root, code_root=REPO_ROOT)[0]
    with open(task_dir / "reference.py", "a", encoding="utf-8") as handle:
        handle.write("\n# drift after discovery\n")
    with pytest.raises(TaskDiscoveryError, match="drifted since discovery"):
        evaluate_task(
            locator,
            track_id="msl-bundle",
            candidate=_bundle_candidate(_geometry_document()),
            builder=FakeBuilder(),
            evaluator=FakeEvaluator(),
            lifecycle_factory=lambda resolved: _lifecycle(resolved, _bundle_candidate(_geometry_document())),
            track_registry_path=_tracks(tmp_path),
        )
