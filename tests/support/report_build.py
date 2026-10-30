"""Schema-4 report ``build`` records for test doubles.

Every "ok" a test double reports must come from a registered builder (§6.2): no path may
report ``ok`` with an empty ``build``.  The simulator registered here is the same
``FakeNpu`` builder the remote test service uses; it is a test-support builder and can
never be published (its ``build_kind`` is ``simulator``).

The helpers project a real ``BuildOutcome`` through the production projector, so the test
doubles exercise the same closed-set shape the production worker writes.
"""

from __future__ import annotations

from typing import Any

from tcenv.build_projection import project_full, project_partial
from tcenv.evaluate import EvalReport
from tcenv.remote.artifact_builder import BuildError, SourceBundle


def _builder(task_obj: Any):
    """Return the registered simulator builder bound to one task.

    @param task_obj 任务对象。
    @return ``(builder, task_contract)``。
    """
    from tests.support.npu_sim.fake_backend import FakeNpuBackend

    backend = FakeNpuBackend()
    builder = backend.artifact_builder(
        kernel_contract=task_obj.contract, specialization=task_obj.specialization,
    )
    return builder, task_obj.contract


def simulated_build(task_obj: Any, source: bytes) -> dict[str, Any]:
    """Project a simulator ``full`` build for one task.

    @param task_obj 任务对象。
    @param source 候选源码字节（simulator 的 SOURCE 工件）。
    @return 合法的 schema-4 ``build`` 对象。
    """
    builder, contract = _builder(task_obj)
    outcome = builder.build(SourceBundle(source), task_contract=contract)
    return project_full(
        outcome, build_kind=builder.build_kind, identity=builder.lowering_identity,
    )


def _structural_native_profile() -> Any:
    """Return the Metal-native structural build profile used by CPU fixtures.

    This profile is a *test double*, named as such.  It is not the FakeNpu simulator profile and
    it is not the real Metal builder either: it exists so the R4/R5 positive cases can exercise a
    ``native`` build shape without compiling MSL.  No FakeNpu outcome is relabeled.

    @return ``BuildProfile``（``build_kind="native"``，``build_host="control_host"``）。
    """
    import platform

    from tcenv.remote.artifact_builder import BuildProfile

    release = str(platform.mac_ver()[0] or "")
    return BuildProfile(
        build_profile="metal-native-structural-fixture",
        builder_id="metal-native-structural-fixture",
        builder_version="0",
        build_host="control_host",
        toolchain_id="metal",
        toolchain_version="fixture",
        # Same form the cache key uses, so a structural fixture can also exercise the cache.
        sdk_version=f"macos-{release}" if release else "",
        compile_profile={"fixture": True, "compiled": False},
        build_kind="native",
    )


def _identity_of(task_obj: Any) -> dict[str, str] | None:
    """Return the task's ``Specialization`` identity as the projection's mapping or ``None``.

    @param task_obj 任务对象。
    @return ``{"opspec_digest", "lowering_profile_digest"}`` 或 ``None``（无 Specialization）。
    """
    specialization = getattr(task_obj, "specialization", None)
    if specialization is None:
        return None
    return {
        "opspec_digest": specialization.opspec_digest,
        "lowering_profile_digest": specialization.lowering_profile_digest,
    }


class StructuralNativeHandle:
    """A process-local stand-in for a Metal pipeline handle (structural fixtures only).

    It carries no Metal object, compiles nothing and is never serialized.  ``BuildOutcome``
    requires *some* live handle for a ``native`` build, so a structural fixture must supply one
    and say plainly what it is.
    """

    def __repr__(self) -> str:
        return "StructuralNativeHandle(no MSL was compiled)"


def _structural_native_outcome(task_obj: Any, source: bytes) -> Any:
    """Assemble a typed native ``BuildOutcome`` whose artefacts are fixture bytes.

    The ``EXECUTABLE_MODULE`` entry is synthetic and labelled as such in the provenance
    (``compile_profile.compiled=False``); nothing here claims a real Metal compile happened.

    @param task_obj 任务对象。
    @param source 候选源码字节。
    @return ``BuildOutcome``。
    """
    import hashlib

    from tcenv.remote.artifact_builder import (
        BuildArtifact, BuildEvidence, BuildOutcome, BuildProvenance, SourceBundle,
    )
    from tcenv.remote.artifact_bundle import (
        NATIVE_EXECUTABLE_SCHEMA_ID, ArtifactBundle, BundleEntry, NativeExecutable,
    )

    profile = _structural_native_profile()
    metallib = b"// structural fixture module; never compiled or loaded\n"
    metallib_sha = hashlib.sha256(metallib).hexdigest()
    source_bundle = SourceBundle(source)
    bundle = ArtifactBundle(entries=(
        BundleEntry(
            path=source_bundle.path, role="SOURCE", media_type=source_bundle.media_type,
            size_bytes=len(source), sha256=hashlib.sha256(source).hexdigest(), data=source,
        ),
        BundleEntry(
            path="build/kernel.metallib", role="EXECUTABLE_MODULE",
            media_type="application/octet-stream", size_bytes=len(metallib),
            sha256=metallib_sha, data=metallib,
        ),
    ))
    native = NativeExecutable(
        backend_id="metal", kind="compute-pipeline", entry_path="build/kernel.metallib",
        entry_sha256=metallib_sha, handle=StructuralNativeHandle(),
        descriptor={
            "schema_id": NATIVE_EXECUTABLE_SCHEMA_ID,
            "compiler": "metal-native-structural-fixture",
            "compiled": "false",
            "language": "MSL",
            "kernel_name": "relu_kernel",
        },
    )
    bundle = ArtifactBundle(entries=bundle.entries, native=native)
    artifacts = tuple(BuildArtifact.from_entry(entry) for entry in bundle.entries)
    provenance = BuildProvenance(
        profile=profile, source_sha256=hashlib.sha256(source).hexdigest(),
        artifact_set_digest=bundle.artifact_set_digest,
    )
    evidence = BuildEvidence(
        mode="prebuilt", status="ok", provenance=provenance, artifacts=artifacts,
        placement="control_host", native=native,
    )
    return BuildOutcome(
        bundle=bundle, provenance=provenance, evidence=evidence, payloads={},
        assembled_sha256=hashlib.sha256(source).hexdigest(),
    )


def structural_expected(*, task: str, dtype: str, size: str, target_id: str, source: str) -> dict:
    """Return the current executor expected for the structural native fixture.

    It has the same shape as ``tcenv.evaluate.confirmation_expected_document``: recomputable
    fields only, no assembled/artifact/native digests, and the provenance of the fixture's
    registered profile (not a copy of any report).

    @param task 任务名。
    @param dtype 元素类型短名。
    @param size 问题规模名。
    @param target_id 目标设备 id。
    @param source 候选源码。
    @return expected 文档。
    """
    import hashlib

    from tasks import get_task

    from tcenv.evaluate import (
        _evaluation_fingerprint, baseline_environment_digest, baseline_implementation_digest,
    )
    from tcenv.schema import PROTOCOL_VERSION

    task_obj = get_task(task, dtype, size)
    source_sha256 = hashlib.sha256(source.encode("utf-8")).hexdigest()
    profile = _structural_native_profile()
    build = {
        "candidate_sha256": source_sha256,
        "provenance": profile.provenance_document(source_sha256=source_sha256, artifact_set_digest=None),
    }
    specialization = getattr(task_obj, "specialization", None)
    if specialization is not None:
        build["opspec_digest"] = specialization.opspec_digest
        build["lowering_profile_digest"] = specialization.lowering_profile_digest
    return {
        "build": build,
        "evaluation": {
            "evaluation_fingerprint": _evaluation_fingerprint(task_obj, "metal", target_id),
            "baseline_implementation_digest": baseline_implementation_digest(task_obj, "metal"),
            "baseline_environment_digest": baseline_environment_digest(task_obj, "metal", target_id),
            "protocol_version": PROTOCOL_VERSION,
            "target_id": target_id,
            "backend": "metal",
            "backend_version": "1",
        },
        "protocol_version": PROTOCOL_VERSION,
        "target_id": target_id,
    }


def native_build(task_obj: Any, source: bytes) -> dict[str, Any]:
    """Project a Metal/native ``full`` build for CPU structural tests.

    This is an independent structural fixture (§6.2 shape only): a typed control-host
    ``BuildOutcome`` with a Metal-native provenance and a synthetic module entry.  It makes no
    claim that a real compile happened, and it is the *only* native fixture; nothing relabels a
    FakeNpu/simulator outcome as native.

    @param task_obj 任务对象。
    @param source 候选源码字节。
    @return 合法的 schema-4 ``native`` ``full`` ``build`` 对象。
    """
    from tcenv.build_projection import project_full

    outcome = _structural_native_outcome(task_obj, source)
    return project_full(outcome, build_kind="native", identity=_identity_of(task_obj))


def native_compile_failure(task_obj: Any, source: bytes) -> dict[str, Any]:
    """Project a typed ``native`` compile failure (structural fixture) as ``partial``.

    @param task_obj 任务对象。
    @param source 候选源码字节。
    @return 合法的 schema-4 ``native`` ``partial`` ``build`` 对象。
    """
    import hashlib

    from tcenv.build_projection import project_partial
    from tcenv.remote.artifact_builder import BuildError, BuildProvenance

    profile = _structural_native_profile()
    provenance = BuildProvenance(
        profile=profile, source_sha256=hashlib.sha256(source).hexdigest(),
        artifact_set_digest=None,
    )
    error = BuildError(
        "compile", "structural fixture: MSL frontend rejected the kernel",
        log=b"kernel.metal:1:1: error: structural fixture rejection\n",
        duration_ms=1, provenance=provenance,
        assembled_sha256=hashlib.sha256(source).hexdigest(),
    )
    return project_partial(error, build_kind="native", identity=_identity_of(task_obj))


def simulated_compile_failure(task_obj: Any, source: bytes) -> dict[str, Any]:
    """Project a simulator ``partial`` build (a typed compile failure).

    @param task_obj 任务对象。
    @param source 含 ``COMPILE_FAIL`` 标记的候选源码。
    @return 合法的 schema-4 ``partial`` ``build`` 对象。
    """
    builder, contract = _builder(task_obj)
    try:
        builder.build(SourceBundle(source), task_contract=contract)
    except BuildError as exc:
        return project_partial(
            exc, build_kind=builder.build_kind, identity=builder.lowering_identity,
        )
    raise AssertionError("the simulator did not reject the marked source")


def task_identity(task: str, dtype: str, size: str) -> dict[str, str]:
    """Return the task Specialization identity a test-double report must carry.

    @param task 任务名。
    @param dtype 元素类型短名。
    @param size 问题规模名。
    @return ``{"opspec_digest", "lowering_profile_digest"}``。
    """
    from tasks import get_task

    specialization = get_task(task, dtype, size).specialization
    return {
        "opspec_digest": specialization.opspec_digest,
        "lowering_profile_digest": specialization.lowering_profile_digest,
    }


def report_for_request(
    request: Any, *, status: str = "ok", source: bytes | None = None,
    backend_id: str = "sim-npu", build_kind: str = "simulator", **overrides: Any,
) -> dict[str, Any]:
    """Build a schema-4 report dict for a fake worker request.

    The request may be a dict (worker request file) or an object with ``task``/``dtype``/
    ``size``/``mode``/``seed``/``target_id`` attributes.

    @param request 请求对象或字典。
    @param status 报告状态。
    @param source 模拟候选源码；缺省为固定占位源码。
    @param backend_id 身份所用后端 id。
    @param build_kind 默认 ``simulator``（FakeNpu）；本地 Metal 结构正例必须显式请求 ``native``。
    @param overrides 覆盖报告顶层字段（例如测试用的错任务名）。
    @param build_kind 默认 ``simulator``（FakeNpu）；本地 Metal 结构正例必须显式请求 ``native``。
    @return 报告字典。
    """
    from tasks import get_task

    def field(name: str, default: Any = None) -> Any:
        if isinstance(request, dict):
            return request.get(name, default)
        return getattr(request, name, default)

    task, dtype, size = str(field("task")), str(field("dtype")), str(field("size"))
    seed = field("seed_int", field("seed", 0))
    if not isinstance(seed, int) or isinstance(seed, bool):
        seed = 0
    target_id = str(field("target_id") or "apple-m4-max")
    task_obj = get_task(task, dtype, size)
    from tcenv.schema import PROTOCOL_VERSION

    report = EvalReport(
        task=task, dtype=dtype, size=size, mode=str(field("mode", "quick")), seed=seed,
        status=status, stage="done", target_id=target_id,
        backend=backend_id, backend_version="1", protocol_version=PROTOCOL_VERSION,
    )
    payload = b"kernel void demo() {}\n" if source is None else source
    if status == "ok":
        report.build = simulated_build(task_obj, payload) if build_kind == "simulator" else native_build(task_obj, payload)
    elif status == "compile_error":
        report.build = (
            simulated_compile_failure(task_obj, b"// COMPILE_FAIL\n" + payload)
            if build_kind == "simulator" else native_compile_failure(task_obj, payload)
        )
    attach_identity(report, task_obj, backend_id=backend_id)
    document = report.to_json()
    document.update(overrides)
    return document


def attach_build(
    report: EvalReport, task_obj: Any, *, source: bytes = b"kernel void demo() {}\n",
    build_kind: str = "native",
) -> EvalReport:
    """Give a test-double report the one ``build`` record its status requires.

    @param report 报告。
    @param task_obj 任务对象。
    @param source 模拟候选源码。
    @param build_kind ``native``（本地 Metal 替身，默认）或 ``simulator``（负面用例）。
    @return 同一份报告。
    """
    from tcenv.build_projection import STATUS_SHAPES

    shapes = STATUS_SHAPES.get(report.status, frozenset())
    if shapes == frozenset({"full"}):
        report.build = simulated_build(task_obj, source) if build_kind == "simulator" else native_build(task_obj, source)
    elif "partial" in shapes:
        report.build = (
            simulated_compile_failure(task_obj, b"// COMPILE_FAIL\n" + source)
            if build_kind == "simulator" else native_compile_failure(task_obj, source)
        )
    else:
        report.build = None
    return report


def attach_identity(report: EvalReport, task_obj: Any, *, backend_id: str = "sim-npu") -> EvalReport:
    """Fill the top-level evaluation identities on a test-double report.

    @param report 报告。
    @param task_obj 任务对象。
    @param backend_id 后端 id。
    @return 同一份报告。
    """
    from tcenv.evaluate import (
        _evaluation_fingerprint,
        baseline_environment_digest,
        baseline_implementation_digest,
    )

    report.evaluation_fingerprint = _evaluation_fingerprint(task_obj, backend_id, report.target_id or "apple-m4-max")
    report.baseline_implementation_digest = baseline_implementation_digest(task_obj, backend_id)
    report.baseline_environment_digest = baseline_environment_digest(
        task_obj, backend_id, report.target_id or "apple-m4-max",
    )
    return report


def ok_build(task: str, dtype: str, size: str, *, source: bytes = b"kernel void demo() {}\n") -> dict[str, Any]:
    """Project the simulator ``full`` build for one task/dtype/size.

    @param task 任务名。
    @param dtype 元素类型短名。
    @param size 问题规模名。
    @param source 模拟候选源码。
    @return ``build`` 对象。
    """
    from tasks import get_task

    return native_build(get_task(task, dtype, size), source)


def ok_report_dict(
    task: str, dtype: str, size: str, *, source: bytes = b"kernel void demo() {}\n",
    speedup: float | None = None, mode: str = "quick", target_id: str = "apple-m4-max",
) -> dict[str, Any]:
    """Return a serialized schema-4 ``ok`` report for resume/checkpoint fixtures.

    @param task 任务名。
    @param dtype 元素类型短名。
    @param size 问题规模名。
    @param source 模拟候选源码。
    @param speedup 可选的加速比。
    @param mode 评测模式。
    @param target_id 目标设备 id。
    @return 报告字典。
    """
    from tasks import get_task

    task_obj = get_task(task, dtype, size)
    report = EvalReport(
        task=task, dtype=dtype, size=size, mode=mode, seed=0, status="ok", stage="done",
        target_id=target_id, backend="metal", speedup=speedup,
    )
    report.build = native_build(task_obj, source)
    attach_identity(report, task_obj)
    return report.to_json()


def ok_report_from_request(request: dict[str, Any], *, source: bytes = b"kernel void demo() {}\n") -> EvalReport:
    """Build a schema-4 ``ok`` report a fake worker can write.

    @param request worker 请求字典。
    @param source 模拟候选源码。
    @return 可序列化的报告。
    """
    from tasks import get_task

    task_obj = get_task(request["task"], request["dtype"], request["size"])
    report = EvalReport(
        task=request["task"], dtype=request["dtype"], size=request["size"], mode=request["mode"],
        seed=request["seed"], status="ok", stage="done",
        target_id=str(request.get("target_id") or "apple-m4-max"),
        backend="sim-npu",
    )
    report.build = simulated_build(task_obj, source)
    return attach_identity(report, task_obj)


_BATCH_IDENTITY = {
    "kind": "match", "differences": [], "shape": "full", "build_kind": "native",
    "measurement_kind": "measured", "cache_hit": False, "timing_uncertain": False,
}


def confirmation_batch(label: str) -> dict[str, Any]:
    """The per-batch identity record a publishable confirmation must carry.

    @param label 候选标签。
    @return 批次记录。
    """
    return {
        "label": label, "shape": "full", "build_kind": "native",
        "measurement_kind": "measured", "cache_hit": False, "timing_uncertain": False,
        "identity": dict(_BATCH_IDENTITY),
    }


def confirmation_report(
    task: str, dtype: str, size: str, *, source: bytes, label: str,
    values: list[float], speedup: float, candidate_us: float | None = None,
    target_id: str = "apple-m4-max",
) -> dict[str, Any]:
    """Return a whole schema-4 ``ok`` report with a publishable confirmation block.

    The confirm report on disk is a full ``EvalReport`` document, and the publication read boundary
    validates it as one, so a fixture must be one too: a native ``full`` build bound to
    ``sha256(source)``, the top-level evaluation identity, and a ``benchmark.confirmation`` block
    whose frozen expected repeats that identity (minus the outcome-only artifact digest).

    @param task 任务名。
    @param dtype 元素类型短名。
    @param size 问题规模名。
    @param source 候选源码字节。
    @param label 胜者标签。
    @param values 逐批加速比（每条对应一条批次记录）。
    @param speedup 中位数。
    @param candidate_us 候选微秒数；``None`` 表示不记。
    @param target_id 目标设备 id。
    @return 报告字典。
    """
    import hashlib

    from tcenv.schema import PROTOCOL_VERSION

    document = ok_report_dict(task, dtype, size, source=source, speedup=speedup, mode="full", target_id=target_id)
    # A worker-written report always states the protocol and the backend version; ``ok_report_dict``
    # leaves them at their ``EvalReport`` defaults, and an empty one is an unknown identity.
    document["protocol_version"] = PROTOCOL_VERSION
    document["backend_version"] = "1"
    build = document["build"]
    expected = {
        "protocol_version": document["protocol_version"],
        "target_id": document["target_id"],
        "build": {
            "candidate_sha256": build["candidate_sha256"],
            "opspec_digest": build["opspec_digest"],
            "lowering_profile_digest": build["lowering_profile_digest"],
            "provenance": {
                key: value for key, value in build["provenance"].items() if key != "artifact_set_digest"
            },
        },
        "evaluation": {
            field: document[field] for field in (
                "evaluation_fingerprint", "baseline_implementation_digest",
                "baseline_environment_digest", "backend", "backend_version",
                "protocol_version", "target_id",
            )
        },
    }
    assert build["candidate_sha256"] == hashlib.sha256(source).hexdigest()
    document["benchmark"] = {
        "noisy": False, "candidate": {"cv": 0.01, "reps": 15},
        "baselines": {"mlx": {"cv": 0.01, "reps": 15}}, "best_baseline": "mlx",
        "confirmation": {
            "kernel_sha256": build["candidate_sha256"], "winner_label": label,
            "confirmed_values": list(values), "speedup": speedup, "candidate_us": candidate_us,
            "expected": expected,
            "batches": [confirmation_batch(label) for _value in values],
        },
    }
    return document
