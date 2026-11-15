"""Worker geometry identity for bundle + Host Program + scratch.

The pre-fix reconstruction below is the old ``_lifecycle_from_document`` body.  It is kept so a
CPU test can show the failure the P2-Q ping-pong candidate hit, without deleting the production
identity comparison or editing that phase's evidence.
"""

from __future__ import annotations

import json
from typing import Any, Mapping

from tcenv.contract_errors import GeometryError
from tcenv.remote.artifact_builder import (
    BuildArtifact,
    BuildEvidence,
    BuildOutcome,
    BuildProfile,
    BuildProvenance,
)
from tcenv.remote.artifact_bundle import ArtifactBundle, BundleEntry, NativeExecutable
from tcenv.remote.execution_geometry import attach_execution_geometry
from tcenv.taskcontract import load_task_contract
from tcenv.target import DEFAULT_TARGET_ID, load_target
from tcenv.track import (
    CandidateLifecycle,
    HostStep,
    PrepareDeclaration,
    abi_limits_from_target,
    derive_tensor_memory_contract,
    geometry_pair,
)
from tcenv.track.delivery import parse_candidate_geometry
from tcenv.track.envelope import decode_envelope
from tcenv.track.plane import CandidateArtifact
from tcenv.v2_worker import _lifecycle_from_document

ROOT_TASK = "tasks/kda"
HISTORICAL_EMPTY_RESOLUTION = (
    '{"resolution_version":1,"source":"candidate","stages":[{"active":true,"derived_scalars":[],'
    '"grid":[2,1,1],"index":0,"kernel":"fill","threadgroup_bytes":0,"threads":[256,1,1],'
    '"when":null,"workspace":[]}]}'
)
_PROFILE = BuildProfile(
    build_profile="metal-msl-control-host-v1",
    builder_id="metal-source-builder",
    builder_version="1",
    build_host="control_host",
    toolchain_id="apple-metal",
    toolchain_version="unknown",
    sdk_version="unknown",
    compile_profile={"language": "msl", "math": "safe", "device": "test", "gpu_family": "test"},
)
_SHA = "ab" * 32


def kda_contract(size: str = "small"):
    """@brief 固定长度 KDA 合同。scratch 测试不改它的 workspace。"""
    return load_task_contract(ROOT_TASK, dtype="fp16", size=size)


def _abi_limits(contract):
    target = load_target(DEFAULT_TARGET_ID)
    abi = derive_tensor_memory_contract(contract.instance, target=target)
    return target, abi, abi_limits_from_target(target)


def resolve_source(source: str, *, size: str = "small"):
    """@brief 解析一份 msl-bundle 信封，Host Program 视为在场。"""
    contract = kda_contract(size)
    _target, abi, limits = _abi_limits(contract)
    artifact = decode_envelope(source)
    raw = next(entry for entry in artifact.entries if entry.path == "host/execution_geometry.json")
    geometry = parse_candidate_geometry(json.loads(raw.data))
    pair = geometry_pair(
        geometry,
        instance=contract.instance,
        contract=abi,
        limits=limits,
        host_program_present=True,
    )
    if pair.candidate is None:
        raise GeometryError(pair.candidate_error or "the candidate geometry did not resolve")
    return contract, artifact, pair.candidate


def _lifecycle(resolved, *, candidate_sha256: str, instance_key: str) -> CandidateLifecycle:
    steps = (
        HostStep(
            PrepareDeclaration(
                name="host_tiling",
                input_dependent=True,
                preprocesses_inputs=True,
                inside_timing=True,
            ),
            role="host_tiling",
        ),
        HostStep(PrepareDeclaration(name="schedule", input_dependent=False), role="schedule"),
    )
    return CandidateLifecycle(
        track_id="msl-bundle",
        geometry=resolved,
        host_steps=steps,
        candidate_sha256=candidate_sha256,
        target_id=DEFAULT_TARGET_ID,
        instance_key=instance_key,
    )


def production_lifecycle(source: str, *, size: str = "small") -> tuple[Any, CandidateLifecycle]:
    """@brief 按 bundle 的 host 角色登记一份生命周期，几何就是求值结果。"""
    contract, artifact, resolved = resolve_source(source, size=size)
    return resolved, _lifecycle(
        resolved,
        candidate_sha256=artifact.candidate_sha256,
        instance_key=contract.fingerprint(),
    )


def lifecycle_from_document_before_p2r(document: Mapping[str, Any]) -> CandidateLifecycle:
    """@brief 修复前的重建：几何和 stage 都不读 ``scratch``。

    这是修复前 ``tcenv/v2_worker.py::_lifecycle_from_document`` 的几何部分。
    非空 scratch 的冻结文档和这份重建不相等，身份检查因此失败。
    """
    from tcenv.track import ResolvedGeometry, ResolvedStage, WorkspaceUse

    raw_geometry = document["geometry"]
    stages = tuple(
        ResolvedStage(
            index=int(item["index"]),
            kernel=str(item["kernel"]),
            grid=tuple(int(value) for value in item["grid"]),
            threads=tuple(int(value) for value in item["threads"]),
            threadgroup_bytes=int(item["threadgroup_bytes"]),
            workspace=tuple(WorkspaceUse(name=str(use["name"]), zero=bool(use["zero"])) for use in item["workspace"]),
            active=bool(item["active"]),
            when=item["when"],
            derived_scalars=tuple((str(pair["name"]), int(pair["value"])) for pair in item["derived_scalars"]),
        )
        for item in raw_geometry["stages"]
    )
    geometry = ResolvedGeometry(
        source=str(raw_geometry["source"]),
        stages=stages,
        resolution_version=int(raw_geometry.get("resolution_version", 1)),
    )
    steps = []
    for item in document.get("prepare", ()):
        fields = {
            key: item[key]
            for key in (
                "name",
                "input_dependent",
                "cached",
                "inside_timing",
                "computes_reference",
                "preprocesses_inputs",
            )
            if key in item
        }
        steps.append(HostStep(PrepareDeclaration(**fields), role=str(item["role"])))
    return CandidateLifecycle(
        track_id=str(document["track_id"]),
        geometry=geometry,
        host_steps=tuple(steps),
        candidate_sha256=str(document["candidate_sha256"]),
        target_id=str(document["target_id"]),
        instance_key=str(document["instance_key"]),
    )


def round_trip(source: str, *, size: str = "small") -> dict[str, Any]:
    """@brief 比较修复前重建、现行重建和求值几何。"""
    resolved, lifecycle = production_lifecycle(source, size=size)
    frozen = lifecycle.document()
    legacy = lifecycle_from_document_before_p2r(frozen)
    current = _lifecycle_from_document(frozen)
    return {
        "frozen_has_scratch": "scratch" in resolved.document(),
        "legacy_equal": legacy.geometry.document() == resolved.document(),
        "current_equal": current.geometry.document() == resolved.document(),
        "lifecycle_equal": current.document() == frozen,
        "legacy_omits_scratch": "scratch" not in legacy.geometry.document(),
        "current_scratch": current.geometry.document().get("scratch"),
        "frozen_scratch": resolved.document().get("scratch"),
    }


def historical_empty_round_trip() -> dict[str, Any]:
    """@brief 一份没有 scratch 键的历史 resolution，重建后键仍然不存在。"""
    geometry = json.loads(HISTORICAL_EMPTY_RESOLUTION)
    document = {
        "track_id": "msl-bundle",
        "candidate_sha256": _SHA,
        "target_id": DEFAULT_TARGET_ID,
        "instance_key": "historical.empty",
        "geometry": geometry,
        "prepare": [
            {
                "role": "host_tiling",
                "name": "host_tiling",
                "input_dependent": True,
                "preprocesses_inputs": True,
                "inside_timing": True,
            }
        ],
    }
    current = _lifecycle_from_document(document)
    legacy = lifecycle_from_document_before_p2r(document)
    rebuilt = current.geometry.document()
    return {
        "current_equal": rebuilt == geometry,
        "legacy_equal": legacy.geometry.document() == geometry,
        "scratch_key_absent": "scratch" not in rebuilt and all("scratch" not in stage for stage in rebuilt["stages"]),
        "canonical_equal": json.dumps(rebuilt, sort_keys=True) == json.dumps(geometry, sort_keys=True),
    }


def _control_host_outcome(candidate: CandidateArtifact) -> BuildOutcome:
    """@brief 控制机侧的预构建产物。几何由 ``attach_execution_geometry`` 再附上。"""
    entries = tuple(
        BundleEntry(
            path=item.path,
            role="AUXILIARY",
            media_type=item.media_type,
            size_bytes=len(item.data),
            sha256=item.sha256,
            data=bytes(item.data),
        )
        for item in sorted(candidate.entries, key=lambda item: item.path)
    )
    native_path = "device/host_piece_handoff.metal"
    native = NativeExecutable(
        backend_id="metal",
        kind="compute-pipeline",
        entry_path=native_path,
        entry_sha256=next(entry.sha256 for entry in entries if entry.path == native_path),
        descriptor={"backend": "metal", "kernel_name": "kda_piece0"},
        handle=object(),
    )
    bundle = ArtifactBundle(entries=entries, native=native)
    provenance = BuildProvenance(
        profile=_PROFILE,
        source_sha256=candidate.candidate_sha256,
        artifact_set_digest=bundle.artifact_set_digest,
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


def _attach(candidate: CandidateArtifact, canonical) -> Any:
    return attach_execution_geometry(_control_host_outcome(candidate), canonical_bundle=canonical)


def _kda_locator():
    """@brief ``tasks/kda`` fp16/small 的 locator。两边求值共用这一份任务包。"""
    from tcenv.taskcontract.locator import TaskLocator
    from tcenv.taskcontract.pack import task_pack_closure
    from tcenv.track import REPO_ROOT

    contract = kda_contract()
    task_dir = REPO_ROOT / "tasks" / "kda"
    closure = task_pack_closure(task_dir, root=REPO_ROOT)
    return TaskLocator(
        task_id="kda",
        op_name=contract.op_name,
        dtype=contract.dtype,
        size=contract.size,
        pack_digest=contract.pack_digest,
        deployment_pack_digest=closure.digest,
        task_dir=task_dir,
        code_root=REPO_ROOT,
    )


class _GeometryBuilder:
    """@brief 不编译设备代码。``evaluate_task`` 会把冻结几何附到这份产物上。"""

    def build(self, request):
        return _control_host_outcome(request.candidate)


class _BoundReportEvaluator:
    """@brief 把已附加的构建投影成 schema-4 报告。

    状态是 ``rejected``：这份 CPU 报告只为了让 ``TaskEvaluationResult.identity``
    绑住真实构建，不是设备正确性或计时结论。两边用同一句说明，所以说明本身
    不会造成身份差异。
    """

    def __init__(self, identity: Mapping[str, str]):
        self.identity = dict(identity)

    def evaluate(self, request):
        from tcenv.build_projection import project_full
        from tcenv.evaluate import EvalReport
        from tcenv.schema import PROTOCOL_VERSION

        build = project_full(request.build, build_kind="native", identity=self.identity)
        task = request.build_request.locator
        report = EvalReport(
            task=task.op_name,
            dtype=task.dtype,
            size=task.size,
            mode="quick",
            status="rejected",
            stage="identity",
            message=(
                "cpu identity comparison binds the attached build; "
                "this report is not a device correctness or timing verdict"
            ),
            seed=0,
            build=build,
            protocol_version=PROTOCOL_VERSION,
            target_id=DEFAULT_TARGET_ID,
            backend="metal",
            backend_version="1",
        )
        return report.to_json()


def _evaluate_resolved(candidate, geometry, lifecycle):
    """@brief 同一份候选字节，用注入的几何走一遍真实 ``evaluate_task``。"""
    from tcenv.taskcontract.runner import evaluate_task

    locator = _kda_locator()
    contract = locator.load()

    def resolve(_candidate):
        return geometry

    return evaluate_task(
        locator,
        track_id="msl-bundle",
        candidate=candidate,
        builder=_GeometryBuilder(),
        evaluator=_BoundReportEvaluator(
            {
                "opspec_digest": contract.spec.digest,
                "lowering_profile_digest": contract.fingerprint(),
            }
        ),
        geometry_resolver=resolve,
        lifecycle=lifecycle,
    )


def _json_leaf(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Mapping):
        return {"object_keys": sorted(str(key) for key in value)}
    if isinstance(value, list):
        return {"length": len(value)}
    return {"type": type(value).__name__}


def _entry_key(item: Any) -> str | None:
    if isinstance(item, Mapping) and isinstance(item.get("path"), str):
        return str(item["path"])
    return None


def _leaf_diff(left: Any, right: Any, prefix: str) -> list[dict[str, Any]]:
    """@brief 逐叶记录两边求值文档的差异。相等的叶不出现。

    带 ``path`` 的清单按路径对齐，避免把冻结几何文件写成一个序号。
    """
    if left == right:
        return []
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        rows: list[dict[str, Any]] = []
        for key in sorted(set(left) | set(right), key=str):
            path = f"{prefix}.{key}" if prefix else str(key)
            if key not in left:
                rows.append({"path": path, "kind": "only_scratch", "scratch": _json_leaf(right[key])})
            elif key not in right:
                rows.append({"path": path, "kind": "only_empty", "empty": _json_leaf(left[key])})
            else:
                rows.extend(_leaf_diff(left[key], right[key], path))
        return rows
    if isinstance(left, list) and isinstance(right, list):
        left_keys = [_entry_key(item) for item in left]
        right_keys = [_entry_key(item) for item in right]
        if (
            left_keys
            and None not in left_keys
            and None not in right_keys
            and len(set(left_keys)) == len(left_keys)
            and len(set(right_keys)) == len(right_keys)
        ):
            rows = []
            left_by_path = dict(zip(left_keys, left))
            right_by_path = dict(zip(right_keys, right))
            for key in sorted(set(left_keys) | set(right_keys)):
                path = f"{prefix}[{key}]"
                if key not in left_by_path:
                    rows.append({"path": path, "kind": "only_scratch", "scratch": _json_leaf(right_by_path[key])})
                elif key not in right_by_path:
                    rows.append({"path": path, "kind": "only_empty", "empty": _json_leaf(left_by_path[key])})
                else:
                    rows.extend(_leaf_diff(left_by_path[key], right_by_path[key], path))
            return rows
        rows = []
        if len(left) != len(right):
            rows.append({"path": prefix, "kind": "length", "empty": len(left), "scratch": len(right)})
        for index in range(min(len(left), len(right))):
            rows.extend(_leaf_diff(left[index], right[index], f"{prefix}[{index}]"))
        return rows
    return [{"path": prefix, "kind": "changed", "empty": _json_leaf(left), "scratch": _json_leaf(right)}]


def identity_delta(empty_source: str, scratch_source: str) -> dict[str, Any]:
    """@brief 同一份候选字节配两份已求值几何：非空 scratch 改掉记录身份。

    两份信封的源字节不同，所以这里固定用空 scratch 候选的 SHA，只换
    ``ResolvedGeometry``。execution、lifecycle、artifact、native 四个因此不同。

    评测身份是 ``evaluate_task`` 返回的 ``TaskEvaluationResult.identity``
    （``fingerprint(document())``）。两边的 ``candidate_sha256`` 和
    ``task_pack_digest`` 用的是同一份候选、同一份任务包。文档里其余不同的叶
    记在 ``evaluation_differences``，不省略。
    """
    from tcenv.track import CanonicalBundle

    contract = kda_contract()
    _target, abi, limits = _abi_limits(contract)
    artifact = decode_envelope(empty_source)
    empty_geometry = _raw_geometry(empty_source)
    scratch_geometry = _raw_geometry(scratch_source)
    empty_pair = geometry_pair(
        empty_geometry,
        instance=contract.instance,
        contract=abi,
        limits=limits,
        host_program_present=True,
    )
    scratch_pair = geometry_pair(
        scratch_geometry,
        instance=contract.instance,
        contract=abi,
        limits=limits,
        host_program_present=True,
    )
    if empty_pair.candidate is None or scratch_pair.candidate is None:
        raise GeometryError("both geometries must resolve for the identity comparison")
    empty_life = _lifecycle(
        empty_pair.candidate,
        candidate_sha256=artifact.candidate_sha256,
        instance_key=contract.fingerprint(),
    )
    scratch_life = _lifecycle(
        scratch_pair.candidate,
        candidate_sha256=artifact.candidate_sha256,
        instance_key=contract.fingerprint(),
    )
    empty_canonical = CanonicalBundle(candidate=artifact, pair=empty_pair, lifecycle=empty_life)
    scratch_canonical = CanonicalBundle(candidate=artifact, pair=scratch_pair, lifecycle=scratch_life)
    empty_attached = _attach(artifact, empty_canonical)
    scratch_attached = _attach(artifact, scratch_canonical)
    empty_result = _evaluate_resolved(artifact, empty_geometry, empty_life)
    scratch_result = _evaluate_resolved(artifact, scratch_geometry, scratch_life)
    _require_evaluation_matches(empty_result, empty_canonical, empty_attached)
    _require_evaluation_matches(scratch_result, scratch_canonical, scratch_attached)
    empty_document = empty_result.document()
    scratch_document = scratch_result.document()
    changed_fields = sorted(
        key for key in empty_document if empty_document[key] != scratch_document[key]
    )
    unchanged_fields = sorted(
        key for key in empty_document if empty_document[key] == scratch_document[key]
    )
    return {
        "execution_changed": empty_pair.identity != scratch_pair.identity,
        "lifecycle_changed": empty_life.identity != scratch_life.identity,
        "artifact_changed": empty_attached.provenance.artifact_set_digest
        != scratch_attached.provenance.artifact_set_digest,
        "native_changed": empty_attached.bundle.native.digest != scratch_attached.bundle.native.digest,
        "evaluation_changed": empty_result.identity != scratch_result.identity,
        "evaluation_identity": "TaskEvaluationResult.identity",
        "evaluation_device_verdict": False,
        "empty_execution": empty_pair.identity,
        "scratch_execution": scratch_pair.identity,
        "empty_lifecycle": empty_life.identity,
        "scratch_lifecycle": scratch_life.identity,
        "empty_artifact": empty_attached.provenance.artifact_set_digest,
        "scratch_artifact": scratch_attached.provenance.artifact_set_digest,
        "empty_native": empty_attached.bundle.native.digest,
        "scratch_native": scratch_attached.bundle.native.digest,
        "empty_evaluation": empty_result.identity,
        "scratch_evaluation": scratch_result.identity,
        "empty_report_status": empty_result.report["status"],
        "scratch_report_status": scratch_result.report["status"],
        "empty_candidate_sha256": empty_result.candidate_sha256,
        "scratch_candidate_sha256": scratch_result.candidate_sha256,
        "empty_task_pack_digest": empty_result.task_pack_digest,
        "scratch_task_pack_digest": scratch_result.task_pack_digest,
        "evaluation_changed_fields": changed_fields,
        "evaluation_unchanged_fields": unchanged_fields,
        "evaluation_differences": _leaf_diff(empty_document, scratch_document, ""),
        "empty_resolved_has_scratch": "scratch" in empty_pair.candidate.document(),
        "scratch_resolved_has_scratch": "scratch" in scratch_pair.candidate.document(),
    }


def _require_evaluation_matches(result, canonical, attached) -> None:
    """@brief ``evaluate_task`` 的身份必须就是上面附上的那份几何，不能另算一套。"""
    from tcenv.identity import fingerprint

    if result.identity != fingerprint(result.document()):
        raise GeometryError("TaskEvaluationResult.identity is not fingerprint(document())")
    if result.geometry_identity != canonical.geometry_identity:
        raise GeometryError("evaluate_task geometry identity diverged from the resolved pair")
    if result.bundle_digest != canonical.bundle_digest:
        raise GeometryError("evaluate_task bundle digest diverged from the canonical bundle")
    if result.artifact_set_digest != attached.provenance.artifact_set_digest:
        raise GeometryError("evaluate_task artifact identity diverged from attach_execution_geometry")
    if result.native_digest != attached.bundle.native.digest:
        raise GeometryError("evaluate_task native identity diverged from attach_execution_geometry")
    if result.candidate_sha256 != canonical.candidate_sha256:
        raise GeometryError("evaluate_task candidate identity diverged from the pinned candidate")


def _raw_geometry(source: str):
    artifact = decode_envelope(source)
    raw = next(entry for entry in artifact.entries if entry.path == "host/execution_geometry.json")
    return parse_candidate_geometry(json.loads(raw.data))


def pre_fix_record() -> dict[str, Any]:
    """@brief 修复前失败的可重算记录。现行重建必须相等，旧重建必须不相等。"""
    from tests.support.seq_handoff import kda_source, pingpong_source, scratch_sources

    ping = round_trip(pingpong_source())
    zero = round_trip(scratch_sources()["device_zero"])
    empty = round_trip(kda_source("seq_three_piece"))
    historical = historical_empty_round_trip()
    return {
        "schema_id": "shapeforge.p2r-scratch-identity.1",
        "epoch": "p2-r-bundle-scratch-geometry-1",
        "publication_allowed": False,
        "npu_simulated_only": True,
        "root_cause": (
            "tcenv/v2_worker.py::_lifecycle_from_document rebuilt ResolvedGeometry without scratch, "
            "so lifecycle.geometry.document() did not match the re-resolved geometry"
        ),
        "identity_check_kept": True,
        "pre_fix": {
            "pingpong_documents_equal": ping["legacy_equal"],
            "device_zero_documents_equal": zero["legacy_equal"],
            "empty_documents_equal": empty["legacy_equal"],
            "pingpong_dropped_scratch": ping["legacy_omits_scratch"] and ping["frozen_has_scratch"],
            "device_zero_dropped_scratch": zero["legacy_omits_scratch"] and zero["frozen_has_scratch"],
        },
        "post_fix": {
            "pingpong_documents_equal": ping["current_equal"],
            "device_zero_documents_equal": zero["current_equal"],
            "empty_documents_equal": empty["current_equal"] and empty["lifecycle_equal"],
            "empty_scratch_key_absent": not empty["frozen_has_scratch"] and empty["legacy_omits_scratch"],
            "historical_empty_equal": historical["current_equal"] and historical["canonical_equal"],
            "historical_scratch_key_absent": historical["scratch_key_absent"],
            "pingpong_lifecycle_equal": ping["lifecycle_equal"],
            "device_zero_lifecycle_equal": zero["lifecycle_equal"],
        },
    }
