"""The real Metal ``ArtifactBuilder``: local build-and-execute placement.

Two kinds of coverage live here:

* device-free invariants of the neutral model the builder uses — a
  backend-native executable is local-only (its bundle refuses every transport
  document), its identity is canonical, and the builder classifies a source that
  never reaches the compiler before touching a device;
* real-Mac tests (``gpu`` marker): the builder compiles valid MSL through the
  production Metal runtime, creates a usable pipeline, reports the historical
  compile/pipeline values unchanged, refuses invalid MSL as a *compile* failure
  and keeps ``candidate_sha256`` identical to the legacy evaluator.

No performance benchmark, no publication and no NPU contract is exercised here.
"""

from __future__ import annotations

import hashlib
import json

import pytest

from tcenv.remote.artifact_builder import (
    BuildArtifact,
    BuildError,
    BuildEvidence,
    BuildProvenance,
    BuildOutcome,
    SourceBundle,
)
from tcenv.remote.artifact_bundle import (
    ArtifactBundle,
    BundleError,
    NativeExecutable,
)

gpu = pytest.mark.gpu


def native_language_version(builder) -> str:
    """The MSL version this machine's real Metal compiler reports."""

    return builder.probe().profile.toolchain_version


def _native(entry, *, kernel: str = "relu_kernel", language: str = "3.1") -> NativeExecutable:
    return NativeExecutable(
        backend_id="metal",
        kind="compute-pipeline",
        entry_path=entry.path,
        entry_sha256=entry.sha256,
        descriptor={
            "backend": "metal", "kernel_name": kernel, "language_version": language,
            "bindings": [{"name": "out", "index": 0, "access": "write"}],
        },
    )


# -- device-free model invariants -----------------------------------------
def test_a_backend_native_artifact_is_local_only_and_its_identity_is_canonical():
    source = SourceBundle(b"kernel void relu_kernel() {}\n")
    entry = source.entry()
    native = _native(entry)
    bundle = ArtifactBundle(entries=(entry,), native=native)

    # The artifact-set digest still describes only the manifest/bytes; the
    # native executable has its own descriptor digest.
    assert bundle.artifact_set_digest == ArtifactBundle(entries=(entry,)).artifact_set_digest
    assert bundle.native_digest == native.digest
    assert len(native.digest) == 64
    # Same descriptor in another insertion order is the same executable.
    reordered = NativeExecutable(
        backend_id="metal", kind="compute-pipeline",
        entry_path=entry.path, entry_sha256=entry.sha256,
        descriptor=json.loads(json.dumps(dict(reversed(list(native.descriptor.items()))))),
    )
    assert reordered.digest == native.digest
    # A local-only artifact set has no request/worker/transport document.
    with pytest.raises(BundleError, match="local-only"):
        bundle.to_dict()
    with pytest.raises(BundleError, match="local-only"):
        bundle.to_worker_dict()

    provenance = BuildProvenance(
        profile=_profile(), source_sha256=source.sha256,
        artifact_set_digest=bundle.artifact_set_digest,
    )
    evidence = BuildEvidence(
        mode="prebuilt", status="ok", provenance=provenance,
        artifacts=(BuildArtifact.from_entry(entry),), native=native,
        loaded=((entry.path, entry.sha256),),
    )
    # The wire BuildEvidence keeps its exact shared shape (no native field), and
    # the local record is a separate document.
    assert "native" not in evidence.to_dict()
    assert evidence.native_document()["digest"] == native.digest
    outcome = BuildOutcome(bundle=bundle, provenance=provenance, evidence=evidence)
    assert outcome.artifact_set_digest == bundle.artifact_set_digest


def test_a_native_executable_must_be_built_from_the_bundled_source_bytes():
    source = SourceBundle(b"kernel void relu_kernel() {}\n")
    entry = source.entry()
    other = SourceBundle(b"kernel void relu_kernel() { /* other */ }\n").entry()
    with pytest.raises(BundleError, match="source bytes"):
        ArtifactBundle(
            entries=(entry,),
            native=NativeExecutable(
                backend_id="metal", kind="compute-pipeline",
                entry_path=entry.path, entry_sha256=other.sha256, descriptor={"kernel": "k"},
            ),
        )


def test_the_metal_builder_refuses_a_source_that_cannot_reach_a_compiler():
    from tcenv.backends.metal.source_builder import MetalSourceBuilder

    builder = MetalSourceBuilder()
    # Not UTF-8: classified as a candidate compile failure, and this happens
    # before any device call, so it is valid without a GPU.
    with pytest.raises(BuildError) as error:
        builder.build(
            SourceBundle(b"\xff\xfe not msl"), task_contract={"kernel_name": "relu_kernel"},
        )
    assert error.value.error_class == "compile"
    # A task contract without a kernel name is a build-identity refusal, not a
    # candidate compile error.
    with pytest.raises(BuildError) as missing:
        builder.build(SourceBundle(b"kernel void relu_kernel() {}\n"), task_contract={})
    assert missing.value.error_class == "toolchain_profile"


def test_a_local_producer_evidence_is_valid_but_the_remote_rule_is_not_weakened():
    from tests.support.vendor_sim import module_builder as mb
    from tcenv.remote import contracts

    for profile, placement in (
        (mb.control_host_profile(), "control_host"),
        (mb.evaluator_profile(), "evaluator"),
    ):
        builder = mb.FakeTsmModuleBuilder(profile)
        outcome = builder.build(
            SourceBundle(b"npu-kernel relu_kernel\n  tx_thread_id\n"), task_contract={},
        )
        evidence = outcome.evidence
        assert evidence.placement == placement
        # The local producer rule accepts a fresh compile with its own log and
        # duration, under both placements.
        document = evidence.verified(report_status="ok", complete=True)
        assert document["status"] == "ok"
        assert document["mode"] == evidence.mode
        assert document["log"] is not None and document["duration_ms"] >= 1
        if placement == "control_host":
            record = evidence.producer_record()
            assert record["status"] == "ok" and record["duration_ms"] == document["duration_ms"]
            assert evidence.matches_producer_record(record)
            assert not evidence.matches_producer_record({**record, "duration_ms": record["duration_ms"] + 1})
            # A prebuilt producer row is still not a valid *received* result.
            with pytest.raises(contracts.ContractError, match="only an evaluator build"):
                contracts.npu_build_evidence_document(evidence.to_dict())
        else:
            with pytest.raises(ValueError):
                evidence.producer_record()


def test_only_the_metal_compile_error_is_classified_as_a_compile_failure():
    from tcenv.backends.metal.source_builder import MetalSourceBuilder

    class MetalCompileLike(Exception):
        pass

    class StubContext:
        def info(self):
            return {
                "name": "stub", "gpu_family": "Apple9", "language_version": "4.0",
                "unified_memory": True,
            }

    class StubBackend:
        version = "stub-1"

        def __init__(self, failure):
            self.failure = failure

        def assemble(self, contract, source):
            return source, 0

        def context(self):
            return StubContext()

        def compile_error(self):
            return MetalCompileLike

        def compile(self, context, source, contract, *, math="fast"):
            raise self.failure

    contract = {"kernel_name": "relu_kernel"}
    # A compiler rejection is the only exception classified as a candidate
    # compile failure.
    builder = MetalSourceBuilder(backend=StubBackend(MetalCompileLike("bad MSL")))
    with pytest.raises(BuildError) as classified:
        builder.build(SourceBundle(b"kernel void relu_kernel() {}"), task_contract=contract)
    assert classified.value.error_class == "compile"
    # An unexpected/device/infrastructure exception is not masked as a compile
    # failure (the legacy Metal evaluator only catches its compile error too).
    builder = MetalSourceBuilder(backend=StubBackend(RuntimeError("device lost")))
    with pytest.raises(RuntimeError, match="device lost"):
        builder.build(SourceBundle(b"kernel void relu_kernel() {}"), task_contract=contract)


def _profile():
    from tcenv.remote.artifact_builder import BuildProfile

    return BuildProfile(
        build_profile="metal-msl-control-host-v1", builder_id="metal-source-builder",
        builder_version="1", build_host="control_host", toolchain_id="apple-metal",
        toolchain_version="3.1", sdk_version="macos-test",
        compile_profile={"language": "msl", "math": "fast"},
    )


# -- real Mac tests --------------------------------------------------------
@gpu
def test_the_real_metal_builder_compiles_msl_and_creates_a_usable_pipeline():
    from tasks import get_task
    from tcenv.backends.metal.runtime import get_context
    from tcenv.backends.metal.sanitize import ArgSet
    from tcenv.backends.metal.source_builder import MetalSourceBuilder
    from tcenv.backends.metal.testing import check_case, prepare_case

    task = get_task("relu", "fp16", "small")
    contract = task.contract
    ctx = get_context()
    # The builder consumes the *candidate* MSL (the same bytes the legacy
    # candidate_sha256 covers) and applies the contract header itself.
    source = task.naive_msl
    builder = MetalSourceBuilder(kernel_contract=contract)
    capabilities = builder.probe()
    assert capabilities.available is True
    outcome = builder.build(SourceBundle(source.encode("utf-8")), task_contract=contract)

    # Real composed documents, in the same shape the NPU paths use.
    evidence = outcome.evidence
    provenance = evidence.provenance
    bundle = outcome.bundle
    assert evidence.status == "ok" and evidence.mode == "prebuilt"
    assert evidence.placement == "control_host"
    assert evidence.error_class is None
    # The producer's own placement is coherent and validated; the remote NPU
    # rule that a *received* prebuilt set is `reused` is tested separately.
    assert evidence.verified(report_status="ok", complete=True)["status"] == "ok"
    producer = evidence.producer_record()
    assert producer["status"] == "ok"
    assert producer["duration_ms"] == evidence.duration_ms
    assert evidence.matches_producer_record(producer)
    assert provenance.source_sha256 == hashlib.sha256(source.encode("utf-8")).hexdigest()
    assert provenance.source_sha256 == SourceBundle(source.encode("utf-8")).sha256
    assert provenance.artifact_set_digest == bundle.artifact_set_digest
    document = provenance.to_dict()
    assert document["build_host"] == "control_host"
    assert document["toolchain"]["id"] == "apple-metal"
    assert document["toolchain"]["version"] == native_language_version(builder)
    assert document["compile_profile"]["device"] == (
        capabilities.identity()["compile_profile"]["device"]
    )
    assert bundle.local_only is True and bundle.native is not None
    native = bundle.native
    assert native.handle is not None and native.handle.pso is not None
    assert native.descriptor["kernel_name"] == contract.kernel_name
    assert native.descriptor["language_version"] != ""
    assert native.descriptor["bindings"] == [
        {
            "name": binding.name, "index": binding.index, "kind": binding.kind,
            "dtype_code": binding.dtype_code, "access": binding.access,
            "elem_size": binding.elem_size, "used": binding.used,
            "tg_bytes": binding.tg_bytes,
        }
        for binding in native.handle.bindings
    ]
    # The local record names the exact executable that was built.
    local = evidence.native_document()
    assert local["digest"] == native.digest
    assert local["entry_sha256"] == hashlib.sha256(source.encode("utf-8")).hexdigest()
    # The historical compile/pipeline values are the ones the runtime produced;
    # the builder neither fabricates nor rewrites them.
    log = evidence.log.decode("utf-8")
    assert f"compile_s={native.handle.compile_s:.6f}" in log
    assert f"pipeline_s={native.handle.pipeline_s:.6f}" in log
    assert f"language={native.handle.language_version}" in log
    assert evidence.duration_ms == round(
        (native.handle.compile_s + native.handle.pipeline_s) * 1000
    )

    # The pipeline is genuinely usable: one small correctness run on the GPU.
    data = prepare_case(task, "normal", 4)
    argset = ArgSet(ctx, contract)
    argset.load_inputs(data.raw)
    argset.poison_outputs()
    ctx.run(native.handle, [argset.args()], contract.grid, contract.threads)
    assert not argset.safety_violations()
    assert check_case(task, data, argset.read_outputs(), None).ok


@gpu
def test_the_builder_reports_the_same_pipeline_shape_as_the_legacy_compile():
    from tasks import get_task
    from tcenv.backends.metal.backend import MetalBackend
    from tcenv.backends.metal.runtime import get_context
    from tcenv.backends.metal.source_builder import MetalSourceBuilder
    from tcenv.evaluation import EvaluationRequest

    task = get_task("relu", "fp16", "small")
    contract = task.contract
    ctx = get_context()
    source = task.naive_msl
    assembled, _header = contract.assemble(source)
    backend = MetalBackend()
    # The legacy path the evaluator uses, unchanged.
    legacy = backend.compile(ctx, assembled, contract)
    built = MetalSourceBuilder(backend=backend, kernel_contract=contract).build(
        SourceBundle(source.encode("utf-8")), task_contract=contract,
    )
    compiled = built.bundle.native.handle
    for name in (
        "name", "max_total_threads", "execution_width", "static_tg_bytes", "language_version",
    ):
        assert getattr(compiled, name) == getattr(legacy, name), name
    assert compiled.pso is not None
    # candidate_sha256, the legacy report identity, is exactly the source digest
    # the builder reports.
    request = EvaluationRequest(
        task="relu", dtype="fp16", size="small", source=source,
        kind="formal_eval", request_id="metal-builder-sha", run_id="run-1",
        session_id="session-1", target_id="apple-m4-max", task_fingerprint="a" * 64,
    )
    assert request.candidate_sha256 == built.provenance.source_sha256


@gpu
def test_invalid_msl_is_a_typed_compile_failure_not_infrastructure():
    from tasks import get_task
    from tcenv.backends.metal.runtime import MetalCompileError, get_context
    from tcenv.backends.metal.source_builder import MetalSourceBuilder

    task = get_task("relu", "fp16", "small")
    contract = task.contract
    ctx = get_context()
    broken = task.naive_msl + "\nthis is not valid metal;\n"
    # Prove the failure actually fires on the real compiler before asserting the
    # classification.
    with pytest.raises(MetalCompileError):
        ctx.compile(contract.assemble(broken)[0], contract.kernel_name)
    builder = MetalSourceBuilder(kernel_contract=contract)
    with pytest.raises(BuildError) as error:
        builder.build(SourceBundle(broken.encode("utf-8")), task_contract=contract)
    assert error.value.error_class == "compile"
    assert "MetalCompileError" in str(error.value)


@gpu
def test_the_registered_metal_profile_gates_a_changed_toolchain_identity():
    from tasks import get_task
    from tcenv.backends.metal.source_builder import MetalSourceBuilder
    from tcenv.remote.artifact_builder import ArtifactBuilderRegistry

    task = get_task("relu", "fp16", "small")
    contract = task.contract
    source = task.naive_msl
    builder = MetalSourceBuilder(kernel_contract=contract)
    capabilities = builder.probe()
    registry = ArtifactBuilderRegistry([builder])
    assert registry.profiles == (capabilities.build_profile,)
    # A request naming another builder/toolchain identity is refused before any
    # compile; the registered profile is the only one served.
    document = capabilities.identity()
    tampered = {
        "schema_id": "tcenv.remote.npu-build.provenance.1",
        **document,
        "toolchain": {**document["toolchain"], "version": "0.0.0"},
        "source_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
        "artifact_set_digest": None,
    }
    with pytest.raises(BuildError) as error:
        registry.build(
            source_bundle=SourceBundle(source.encode("utf-8")),
            task_contract=contract, provenance=tampered,
        )
    assert error.value.error_class == "toolchain_profile"
    # The genuine identity builds through the registry and keeps its profile.
    genuine = BuildProvenance(
        profile=builder.probe().profile,
        source_sha256=hashlib.sha256(source.encode("utf-8")).hexdigest(),
    ).to_dict()
    outcome = registry.build(
        source_bundle=SourceBundle(source.encode("utf-8")),
        task_contract=contract, provenance=genuine,
    )
    assert outcome.provenance.profile.build_profile == capabilities.build_profile
