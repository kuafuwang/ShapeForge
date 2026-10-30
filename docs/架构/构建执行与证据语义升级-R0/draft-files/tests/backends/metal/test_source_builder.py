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


def _native(entry, *, kernel: str = "relu_kernel", language: str = "3.1", handle=None) -> NativeExecutable:
    return NativeExecutable(
        backend_id="metal",
        kind="compute-pipeline",
        entry_path=entry.path,
        entry_sha256=entry.sha256,
        descriptor={
            "backend": "metal", "kernel_name": kernel, "language_version": language,
            "bindings": [{"name": "out", "index": 0, "access": "write"}],
        },
        handle=handle,
    )


# -- device-free model invariants -----------------------------------------
def test_a_backend_native_artifact_is_local_only_and_its_identity_is_canonical():
    source = SourceBundle(b"kernel void relu_kernel() {}\n")
    entry = source.entry()
    native = _native(entry, handle=object())
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
        def __init__(self, failure):
            self.failure = failure

        def info(self):
            return {
                "name": "stub", "gpu_family": "Apple9", "language_version": "4.0",
                "unified_memory": True,
            }

        def compile(self, source, kernel_name, *, math="fast"):
            raise self.failure

    class StubBackend:
        version = "stub-1"

        def __init__(self, failure):
            self.failure = failure

        def assemble(self, contract, source):
            return source, 0

        def context(self):
            return StubContext(self.failure)

        def compile_error(self):
            return MetalCompileLike

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
    # The runtime primitive itself: the builder calls exactly this, so the test
    # can compare the builder's output with the pre-refactor compiler result.
    legacy = ctx.compile(assembled, contract.kernel_name, math="fast")
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


def test_a_compile_failure_carries_the_real_diagnostic_and_duration():
    """A typed compile failure is not a bare message: it keeps log and duration.

    The successful path records the compiler identity and the measured
    ``compile_s``/``pipeline_s``; the failure path has to record the compiler's
    own diagnostic and the real elapsed time, otherwise the only durable record
    of why a candidate failed is a truncated one-liner with ``duration_ms=0``.
    """

    from tcenv.backends.metal.source_builder import MetalSourceBuilder

    class StubCompileError(Exception):
        pass

    class StubContext:
        def info(self):
            return {
                "name": "stub-device", "gpu_family": "Apple9", "language_version": "4.0",
                "unified_memory": True,
            }

        def compile(self, source, kernel_name, *, math="fast"):
            raise StubCompileError("program_source:9:1: error: unknown type name 'tx_bad'")

    class StubBackend:
        version = "stub-1"

        def assemble(self, contract, source):
            return source, 3

        def context(self):
            return StubContext()

        def compile_error(self):
            return StubCompileError

        def rewrite_compiler_lines(self, contract, message, header_lines):
            return f"rewritten: {message}"

    builder = MetalSourceBuilder(backend=StubBackend())
    with pytest.raises(BuildError) as error:
        builder.build(
            SourceBundle(b"kernel void relu_kernel() {}\n"),
            task_contract={"kernel_name": "relu_kernel"},
        )
    failure = error.value
    assert failure.error_class == "compile"
    # The report-facing text keeps the historical ``type: rewritten message``
    # shape the evaluator already reports.
    assert failure.detail.startswith("StubCompileError: rewritten: ")
    assert "unknown type name 'tx_bad'" in failure.detail
    # The failure now carries a real log: the compiler identity, the diagnostic
    # the evaluator would report, and the raw text when the rewrite changed it.
    log = failure.log.decode("utf-8")
    assert "metal: kernel=relu_kernel" in log
    assert "status=compile_failed error=StubCompileError" in log
    assert "rewritten: program_source:9:1: error: unknown type name 'tx_bad'" in log
    assert "metal: raw diagnostic:" in log
    assert "program_source:9:1" in log.split("metal: raw diagnostic:", 1)[1]
    assert "assembled_sha256=" + hashlib.sha256(
        b"kernel void relu_kernel() {}\n"
    ).hexdigest() in log
    # A build that ran never reports the contract's "did not run" duration (0).
    assert failure.duration_ms >= 1


@gpu
def test_invalid_msl_keeps_the_real_compiler_diagnostic_and_the_legacy_report_text():
    """Real Metal: the failure log holds the compiler's own diagnostics."""

    from tasks import get_task
    from tcenv.backends.metal.runtime import MetalCompileError
    from tcenv.backends.metal.source_builder import MetalSourceBuilder

    task = get_task("relu", "fp16", "small")
    contract = task.contract
    broken = task.naive_msl + "\nthis is not valid metal;\n"
    assembled = contract.assemble(broken)[0]
    # The legacy path is untouched: the same exception type with the same text.
    with pytest.raises(MetalCompileError) as legacy:
        MetalSourceBuilder().backend.context().compile(assembled, contract.kernel_name)
    legacy_text = str(legacy.value)
    assert "program_source" in legacy_text

    builder = MetalSourceBuilder(kernel_contract=contract)
    with pytest.raises(BuildError) as error:
        builder.build(SourceBundle(broken.encode("utf-8")), task_contract=contract)
    failure = error.value
    assert failure.error_class == "compile"
    # The report text the evaluator would surface is the compiler's diagnostic
    # (rewritten back to candidate line numbers), unchanged in shape.
    assert "MetalCompileError" in failure.detail
    assert "error:" in failure.detail and "kernel.metal:" in failure.detail
    log = failure.log.decode("utf-8")
    # The log carries the real diagnostic: the rewritten line the evaluator
    # reports *and* the compiler's raw text (which still names program_source),
    # not a synthesised summary.
    assert "metal: status=compile_failed error=MetalCompileError" in log
    assert "kernel.metal:12:1: error:" in log
    raw_section = log.split("metal: raw diagnostic:", 1)[1]
    assert "program_source" in raw_section
    assert "expected unqualified-id" in raw_section
    assert f"metal: kernel={contract.kernel_name}" in log
    # The real elapsed compile time is recorded (never 0 = "did not run").
    assert failure.duration_ms >= 1
    match = next(
        line for line in log.splitlines() if line.startswith("metal: status=compile_failed")
    )
    compile_s = float(match.split("compile_s=")[1].split()[0])
    assert compile_s > 0.0


@gpu
def test_the_probe_only_compiles_the_fixed_msl_version_probe():
    """``probe`` measures the device's MSL version, never the candidate.

    ``MetalContext.info()`` -> ``language_version()`` does compile a fixed
    two-line MSL library (that is how the highest usable MSL version is chosen
    on this machine) but it never creates a pipeline, never sees the candidate
    source and never runs a kernel.  The candidate is only compiled by
    ``build``, which this test never calls.
    """

    from tcenv.backends.metal.backend import MetalBackend
    from tcenv.backends.metal.runtime import MetalContext
    from tcenv.backends.metal.source_builder import MetalSourceBuilder

    class CountingDevice:
        def __init__(self, device):
            self._device = device
            self.library_calls = 0
            self.pipeline_calls = 0
            #: The exact source string handed to the compiler, so the fixed
            #: probe text is what the assertion checks (not just a call count).
            self.library_sources: list[str] = []

        def __getattr__(self, name):
            attribute = getattr(self._device, name)
            if name == "newLibraryWithSource_options_error_":
                def counted(*args, **kwargs):
                    self.library_calls += 1
                    self.library_sources.append(args[0])
                    return attribute(*args, **kwargs)
                return counted
            if name.startswith("newComputePipelineState"):
                def counted(*args, **kwargs):
                    self.pipeline_calls += 1
                    return attribute(*args, **kwargs)
                return counted
            return attribute

    context = MetalContext()
    context._lang_cache = None
    counting = CountingDevice(context.device)
    context.device = counting

    class FixedContextBackend(MetalBackend):
        def context(self):
            return context

    builder = MetalSourceBuilder(backend=FixedContextBackend())
    capabilities = builder.probe()
    assert capabilities.available is True
    # The version probe compiled exactly one fixed MSL library, selected the
    # version it reported, and created no pipeline.
    assert counting.library_calls == 1
    assert counting.pipeline_calls == 0
    # The compiled source is the runtime's fixed version probe: one include plus
    # one `kernel void k`, no candidate code, exactly two lines.
    probe_source = counting.library_sources[0]
    probe_lines = [line for line in probe_source.splitlines() if line.strip()]
    assert len(probe_lines) == 2, probe_lines
    assert probe_lines[0].startswith("#include <metal_stdlib>")
    assert "kernel void k(" in probe_lines[1]
    assert "relu_kernel" not in probe_source
    assert "threadposition" not in probe_source
    assert capabilities.profile.toolchain_version == context.language_version()
    assert capabilities.profile.toolchain_version in {"4.0", "3.2", "3.1", "3.0", "2.4"}
    # The result is cached: a second probe does no further compile.
    assert builder.probe().profile == capabilities.profile
    assert counting.library_calls == 1
    # No candidate was compiled: that only happens in ``build``.
    assert counting.pipeline_calls == 0


def test_the_native_handle_is_never_hashed_serialized_or_transported():
    """The live pipeline is a local capability, not a document field.

    Two executables with the same descriptor and different live handles are the
    same *executable identity*: the handle is excluded from equality and from
    every digest, it never appears in a document, and a local-only artifact set
    has no transport payload to mistake for one.
    """

    class Handle:
        def __init__(self, name):
            self.name = name

        def __repr__(self):  # pragma: no cover - only used if the invariant breaks
            return f"<live pipeline {self.name}>"

    source = SourceBundle(b"kernel void relu_kernel() {}\n")
    entry = source.entry()
    made = [
        NativeExecutable(
            backend_id="metal", kind="compute-pipeline",
            entry_path=entry.path, entry_sha256=entry.sha256,
            descriptor={"kernel_name": "relu_kernel", "language_version": "4.0"},
            handle=Handle(name),
        )
        for name in ("first", "second")
    ]
    first, second = made
    assert first.handle is not second.handle
    # Same executable identity: the live object is not part of it.
    assert first == second
    assert first.digest == second.digest
    assert "handle" not in first.document()
    assert "live pipeline" not in repr(first)

    # A descriptor that tries to carry the live object is not a document at all.
    with pytest.raises((TypeError, ValueError)):
        NativeExecutable(
            backend_id="metal", kind="compute-pipeline",
            entry_path=entry.path, entry_sha256=entry.sha256,
            descriptor={"kernel_name": "relu_kernel", "pipeline": Handle("x")},
        )

    # The producer link of the evidence is a digest of the evidence document, so
    # it too is unmoved by the handle: a local digest can never depend on a
    # capability that only exists in this process.
    def _evidence(native):
        bundle = ArtifactBundle(entries=(entry,), native=native)
        provenance = BuildProvenance(
            profile=_profile(), source_sha256=source.sha256,
            artifact_set_digest=bundle.artifact_set_digest,
        )
        return BuildOutcome(
            bundle=bundle, provenance=provenance,
            evidence=BuildEvidence(
                mode="prebuilt", status="ok", provenance=provenance,
                artifacts=(BuildArtifact.from_entry(entry),), native=native,
                loaded=(), placement="control_host",
            ),
        )

    outcome = _evidence(first)
    other = _evidence(second)
    assert outcome.evidence.producer_record() == other.evidence.producer_record()
    assert outcome.evidence.to_dict() == other.evidence.to_dict()
    assert "native" not in outcome.evidence.to_dict()
    # ...and there is no portable payload to send it with.
    with pytest.raises(BundleError, match="local-only"):
        outcome.payload()


@gpu
def test_two_real_compiles_of_one_source_share_an_identity_and_not_a_handle():
    """Real Metal: a second build is a second pipeline with the same identity."""

    from tasks import get_task
    from tcenv.backends.metal.source_builder import MetalSourceBuilder

    task = get_task("relu", "fp16", "small")
    contract = task.contract
    source = SourceBundle(task.naive_msl.encode("utf-8"))
    first = MetalSourceBuilder(kernel_contract=contract).build(source, task_contract=contract)
    second = MetalSourceBuilder(kernel_contract=contract).build(source, task_contract=contract)
    assert first.bundle.native.handle is not second.bundle.native.handle
    assert first.bundle.native.digest == second.bundle.native.digest
    assert first.artifact_set_digest == second.artifact_set_digest
    # The evidence documents are identical apart from this build's own measured
    # duration and log: nothing about the live pipeline entered either document.
    documents = []
    for outcome in (first, second):
        document = outcome.evidence.to_dict()
        assert "native" not in document
        assert "pso" not in json.dumps(document)
        documents.append(
            {key: value for key, value in document.items() if key not in ("duration_ms", "log")}
        )
    assert documents[0] == documents[1]


def test_one_build_outcome_must_name_one_live_executable():
    """Descriptor equality is not enough: the outcome closes over one handle.

    ``NativeExecutable`` deliberately excludes the live handle from equality and
    from the digest (a handle is a process-local capability, not a document
    field).  That means two executables with the same descriptor compare equal,
    so `BuildOutcome` has to check the handle *identity* as well: a bundle
    carrying one pipeline and evidence claiming another would otherwise describe
    two different live executables as one build.
    """

    source = SourceBundle(b"kernel void relu_kernel() {}\n")
    entry = source.entry()
    handles = [object(), object()]

    def native(handle):
        return NativeExecutable(
            backend_id="metal", kind="compute-pipeline",
            entry_path=entry.path, entry_sha256=entry.sha256,
            descriptor={"kernel_name": "relu_kernel", "language_version": "4.0"},
            handle=handle,
        )

    bundle_native, evidence_native = native(handles[0]), native(handles[1])
    # The two are the same *documented* identity...
    assert bundle_native == evidence_native and bundle_native.digest == evidence_native.digest
    bundle = ArtifactBundle(entries=(entry,), native=bundle_native)
    provenance = BuildProvenance(
        profile=_profile(), source_sha256=source.sha256,
        artifact_set_digest=bundle.artifact_set_digest,
    )

    def evidence_for(candidate):
        return BuildEvidence(
            mode="prebuilt", status="ok", provenance=provenance,
            artifacts=(BuildArtifact.from_entry(entry),), native=candidate,
            placement="control_host",
        )

    # ...but one outcome may not pair two live executables.
    with pytest.raises(ValueError, match="live executable"):
        BuildOutcome(
            bundle=bundle, provenance=provenance, evidence=evidence_for(evidence_native),
        )
    # A success that omits the live handle on *both* records is still not a
    # build: equality ignores the handle, so both-None would otherwise pass as
    # two records of one executable.
    missing = native(None)
    missing_bundle = ArtifactBundle(entries=(entry,), native=missing)
    missing_provenance = BuildProvenance(
        profile=_profile(), source_sha256=source.sha256,
        artifact_set_digest=missing_bundle.artifact_set_digest,
    )
    with pytest.raises(ValueError, match="non-empty live executable handle"):
        BuildOutcome(
            bundle=missing_bundle, provenance=missing_provenance,
            evidence=BuildEvidence(
                mode="prebuilt", status="ok", provenance=missing_provenance,
                artifacts=(BuildArtifact.from_entry(entry),), native=missing,
                placement="control_host",
            ),
        )
    # The same live handle (even through another equal wrapper) is coherent.
    coherent = BuildOutcome(
        bundle=bundle, provenance=provenance, evidence=evidence_for(native(handles[0])),
    )
    assert coherent.bundle.native.handle is coherent.evidence.native.handle
