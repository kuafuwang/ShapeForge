"""P1c: unified ArtifactBuilder and control_host artifact handoff over six RPCs.

Both build timings run through the real six RPCs and (for a prebuilt artifact
set) the real BlobChannel:

* prebuilt — the control host builds, uploads the executable module by content
  id and only then submits; the evaluator verifies the artifact set and loads the
  module without compiling.
* evaluator — the inline source is submitted and built inside the attempt by the
  registered ArtifactBuilder before the vendor driver loads it.

The fake toolchain and fake NPU vendor live in ``tests/support/vendor_sim/``;
nothing here is a real TX compiler, ABI or device, so every result is
``simulated=true`` and never publishable.
"""

from __future__ import annotations

import hashlib
import json
import os
import time

import grpc
import pytest

from tcenv.evaluation import EvaluationRequest, EvaluationUnavailable
from tcenv.remote import contracts
from tcenv.remote import grpc_contract as gc
from tcenv.remote.artifact_builder import BuildError, BuildProvenance
from tcenv.remote.blob_channel import BlobError
from tcenv.remote.job_types import JobError, NpuBuildRequest, StateCorrupt
from tcenv.remote.v1 import evaluator_pb2 as pb
from tests.support.job_engine import NPU_TASK_FINGERPRINT, NPU_TOKEN
from tests.support.vendor_sim import module_builder as mb
from tests.support.vendor_sim.harness import ModuleBuildGrpcHarness
from tests.support.v1_tasks import V1_TASK

METADATA = (("authorization", f"Bearer {NPU_TOKEN}"),)
SOURCE = b"npu-kernel relu_kernel\n  tx_thread_id\n"
FAILING_SOURCE = b"npu-kernel relu_kernel\nTXCC_FAIL undefined tx_bad\n"


# -- helpers ---------------------------------------------------------------
def _host_request(harness, request_id: str, *, source: bytes = SOURCE, timeout_s: float = 10.0):
    return EvaluationRequest(
        kind="formal_eval", request_id=request_id, run_id="npu-run-1",
        session_id="npu-session-1", task=V1_TASK, dtype="fp16", size="small",
        target_id=harness.engine.identity.target_id,
        task_fingerprint=NPU_TASK_FINGERPRINT, source=source.decode("utf-8"),
        mode="quick", seed=3, reps=2, queue_timeout_s=timeout_s,
        candidate_timeout_s=timeout_s,
    )


def _expected_pb(harness) -> pb.ExpectedEvaluator:
    expected = harness.engine.expected_identities.evaluator
    return pb.ExpectedEvaluator(
        site_id=expected.site_id, device_id=expected.device_id,
        target_id=expected.target_id, worker_build=expected.worker_build,
        task_pack_digest=expected.task_pack_digest,
        evaluator_epoch=expected.evaluator_epoch,
    )


def _capabilities(harness, nonce: str = "c" * 32):
    return harness.factory.stub.GetCapabilities(
        pb.CapabilitiesRequest(
            api_contract_ids=[contracts.NPU_BUILD_API_CONTRACT_ID],
            evidence_contract_ids=[contracts.NPU_BUILD_EVIDENCE_CONTRACT_ID],
            required_features=list(harness.factory.required_features),
            client_nonce=nonce,
            expected=_expected_pb(harness),
        ),
        metadata=METADATA, timeout=10,
    )


def _bind(harness, host: EvaluationRequest) -> EvaluationRequest:
    """Bind the host identity exactly the way the production binder does."""

    from taic.evaluation_outbox import bind_wire_digest

    return bind_wire_digest(
        host, harness.factory._for_request(host), npu_contract=True,
    )


def _wire(harness, host):
    evaluator = harness.factory._for_request(host)
    return evaluator._wire_request(evaluator._external_request(host))


def _submit_wire(harness, external, capability=None):
    capability = capability or _capabilities(harness)
    return harness.factory.stub.Submit(
        pb.SubmitRequest(
            request=gc.request_to_proto(external, dialect=harness.factory.contract),
            request_digest=external.request_digest,
            expected=gc.expected_identities_to_proto(harness.engine.expected_identities),
            capability_id=capability.capability_id,
            required_features=list(harness.factory.required_features),
        ),
        metadata=METADATA, timeout=10,
    )


def _get(harness, external):
    return harness.factory.stub.GetJob(
        pb.JobRef(
            request_id=external.request_id,
            request_digest=external.request_digest,
            expected=gc.expected_identities_to_proto(harness.engine.expected_identities),
        ),
        metadata=METADATA, timeout=10,
    )


def _retry(harness, external, snapshot):
    import secrets

    capability = _capabilities(harness, nonce=secrets.token_hex(16))
    return harness.factory.stub.Retry(
        pb.RetryRequest(
            job=pb.JobRef(
                request_id=external.request_id, request_digest=external.request_digest,
                expected=gc.expected_identities_to_proto(harness.engine.expected_identities),
            ),
            interrupted_attempt_id=snapshot.current_attempt.attempt_id,
            capability_id=capability.capability_id,
            required_features=list(harness.factory.required_features),
        ),
        metadata=METADATA, timeout=10,
    )


def _wait(harness, external, states, timeout: float = 25.0):
    deadline = time.monotonic() + timeout
    snapshot = None
    while time.monotonic() < deadline:
        snapshot = _get(harness, external)
        if gc.state_from_proto(snapshot.state) in states:
            return snapshot
        time.sleep(0.03)
    raise AssertionError(
        f"job did not reach {states}: {gc.state_from_proto(snapshot.state)} {snapshot.failure}"
    )


def _formal(harness, request_id: str, *, source: bytes = SOURCE):
    host = _bind(harness, _host_request(harness, request_id, source=source))
    return harness.factory._for_request(host).formal(host), host


def _run_prebuilt(harness, request_id: str, *, source: bytes = SOURCE):
    outcome = harness.host_build(source)
    harness.upload_module(outcome)
    harness.set_prebuilt(outcome, outcome.bundle, source)
    result, host = _formal(harness, request_id, source=b"")
    return outcome, result


def _assert_no_device_work(harness) -> None:
    """The device capability probe may have run; nothing may have been built/loaded.

    A capabilities handshake legitimately probes the group, so ``ledger.exists()``
    alone proves nothing.  The build contract must still leave no job row, no
    attempt resource, no module load and no build record.
    """

    assert not harness.engine.store._conn.execute(
        "SELECT COUNT(*) FROM jobs"
    ).fetchone()[0]
    if not harness.controller.ledger.exists():
        return
    assert harness.controller.ledger.builds() == []
    assert harness.controller.ledger.module_loads() == []
    assert harness.controller.ledger.live_resources() == []
    assert harness.controller.ledger.running_markers() == []


# -- prebuilt timing -------------------------------------------------------
def test_the_control_host_builds_once_uploads_the_module_and_the_npu_only_loads(tmp_path):
    with ModuleBuildGrpcHarness(tmp_path / "prebuilt") as harness:
        capability = _capabilities(harness)
        assert capability.api_contract_id == contracts.NPU_BUILD_API_CONTRACT_ID
        target = capability.targets[0]
        assert list(target.artifact_kinds) == ["bundle"]
        assert list(target.launch_kinds) == ["KERNEL"]
        assert "build-provenance" in list(target.features)

        outcome = harness.host_build(SOURCE)
        assert harness.builder_calls == 1
        bundle = harness.upload_module(outcome)
        module = next(entry for entry in bundle.entries if entry.role == "EXECUTABLE_MODULE")
        assert module.data is None and module.blob_ref is not None
        harness.set_prebuilt(outcome, bundle, SOURCE)

        result, host = _formal(harness, "build-prebuilt-1", source=b"")

        assert result.report["status"] == "ok"
        assert result.report["build"]["build_kind"] == "simulator"
        evidence = result.evidence
        assert evidence["schema_id"] == contracts.NPU_BUILD_TIMING_EVIDENCE_SCHEMA_ID
        assert evidence["publication_allowed"] is False
        build = evidence["build"]
        assert build["status"] == "reused" and build["mode"] == "prebuilt"
        assert build["error_class"] is None
        assert build["provenance"]["build_host"] == "control_host"
        assert build["provenance"]["artifact_set_digest"] == outcome.bundle.artifact_set_digest
        assert build["provenance"] == harness.build.provenance_document()
        # The receiver really loaded the control-host module bytes.
        assert build["loaded"] == [{
            "path": module.path, "sha256": module.sha256,
        }]
        # The producer's own build record is not discarded: the received row
        # repeats its status/duration/log identity and the digest of the
        # evidence document the control host kept.
        producer = outcome.evidence.producer_record()
        assert build["producer"] == producer
        assert producer["status"] == "ok" and producer["mode"] == "prebuilt"
        assert producer["duration_ms"] == outcome.evidence.duration_ms >= 1
        assert producer["log"] == {
            "sha256": hashlib.sha256(outcome.evidence.log).hexdigest(),
            "size_bytes": len(outcome.evidence.log),
        }
        assert producer["log"]["size_bytes"] > 0
        assert producer["evidence_sha256"] == hashlib.sha256(
            contracts.canonical_digest_document(outcome.evidence.to_dict())
        ).hexdigest()
        assert outcome.evidence.matches_producer_record(build["producer"])
        assert harness.build.to_dict()["producer"] == producer
        # The receiver's own verification is recorded too: a positive duration
        # and a load log, never the old hardcoded zero/empty row.
        assert build["duration_ms"] >= 1
        assert build["log"] is not None and build["log"]["size_bytes"] > 0
        # The producer's local placement is coherent (fresh prebuilt compile),
        # while the remote NPU rule still refuses that row as a received result.
        assert outcome.evidence.placement == "control_host"
        assert outcome.evidence.verified(report_status="ok", complete=True)["status"] == "ok"
        with pytest.raises(contracts.ContractError, match="only an evaluator build"):
            contracts.npu_build_evidence_document(
                outcome.evidence.to_dict(),
                expected_build=harness.build.to_dict(),
                request_artifact={"bundle": outcome.bundle.to_dict()},
            )
        loads = harness.controller.ledger.module_loads()
        assert loads and loads[-1]["sha256"] == module.sha256
        builds = harness.controller.ledger.builds()
        assert builds and builds[-1]["provenance"] == build["provenance"]
        assert builds[-1]["mode"] == "prebuilt"
        assert builds[-1]["loads"][0]["sha256"] == hashlib.sha256(
            outcome.payloads[module.path]
        ).hexdigest()
        # No compiler ran on the NPU side: the builder was invoked once, on the
        # control host, and the worker only verified/loaded.
        assert harness.builder_calls == 1
        assert result.report["candidate_us"] == pytest.approx(
            evidence["timing"]["device_elapsed_ns"] / 1000.0
        )


def test_the_two_build_timings_share_the_build_contract_and_differ_only_by_host(tmp_path):
    with ModuleBuildGrpcHarness(tmp_path / "prebuilt") as prebuilt:
        prebuilt_outcome, prebuilt_result = _run_prebuilt(prebuilt, "timing-prebuilt")
    with ModuleBuildGrpcHarness(tmp_path / "evaluator", mode="evaluator") as evaluator:
        evaluator.set_evaluator_build(SOURCE)
        evaluator_result, _host = _formal(evaluator, "timing-evaluator")

    prebuilt_build = prebuilt_result.evidence["build"]
    evaluator_build = evaluator_result.evidence["build"]
    # The two rows share one contract; the prebuilt one additionally carries the
    # producer link (the received set never claims a fresh compile).
    assert set(prebuilt_build) - {"producer"} == set(evaluator_build)
    assert "producer" not in evaluator_build
    assert prebuilt_build["schema_id"] == evaluator_build["schema_id"]
    assert prebuilt_build["provenance"]["builder_id"] == evaluator_build["provenance"]["builder_id"]
    assert prebuilt_build["provenance"]["toolchain"] == evaluator_build["provenance"]["toolchain"]
    assert prebuilt_build["provenance"]["source_sha256"] == hashlib.sha256(SOURCE).hexdigest()
    assert prebuilt_build["provenance"]["build_host"] == "control_host"
    assert evaluator_build["provenance"]["build_host"] == "evaluator"
    # Same roles and same manifest rules for both produced artifact sets.
    assert {entry["role"] for entry in prebuilt_build["artifacts"]["entries"]} == {
        entry["role"] for entry in evaluator_build["artifacts"]["entries"]
    }
    assert evaluator_build["status"] == "ok"
    assert prebuilt_build["status"] == "reused"
    assert evaluator_build["duration_ms"] >= 1
    assert evaluator_build["log"]["size_bytes"] > 0
    # The control-host compile duration/log are not lost: the receiver repeats
    # the producer link and its own verification takes measurable time.
    assert prebuilt_build["producer"]["status"] == "ok"
    assert prebuilt_build["producer"]["duration_ms"] >= 1
    assert prebuilt_build["producer"]["log"]["size_bytes"] > 0
    assert prebuilt_build["duration_ms"] >= 1
    assert evaluator_build["provenance"]["artifact_set_digest"] != (
        prebuilt_build["provenance"]["artifact_set_digest"]
    )
    assert prebuilt_result.report["build"]["build_kind"] == "simulator"
    assert evaluator_result.report["build"]["build_kind"] == "simulator"


def test_the_control_host_build_failure_never_submits(tmp_path):
    with ModuleBuildGrpcHarness(tmp_path / "prebuilt") as harness:
        capability = _capabilities(harness)
        with pytest.raises(BuildError) as error:
            harness.host_build(FAILING_SOURCE)
        assert error.value.error_class == "compile"
        assert b"--txcc: error:" in error.value.log
        assert harness.builder_calls == 1
        # The failed build produced no artifact set, so nothing can be submitted:
        # there is no bundle to name and no job row was ever accepted.
        assert harness.bundle is None
        assert not harness.engine.store.job_exists("build-failure-1")
        _assert_no_device_work(harness)


def test_an_interrupted_upload_is_refused_then_resumes_without_rebuilding(tmp_path):
    with ModuleBuildGrpcHarness(tmp_path / "prebuilt") as harness:
        outcome = harness.host_build(SOURCE)
        ref, data = harness.start_partial_upload(outcome, chunks=1)
        received = harness.blob_store.state(harness.scope(), ref.content_id).received_bytes
        assert 0 < received < len(data)
        harness.set_prebuilt(outcome, outcome.bundle, SOURCE)
        host = _host_request(harness, "build-resume-1", source=b"")
        external = _wire(harness, host)
        with pytest.raises(grpc.RpcError) as refused:
            _submit_wire(harness, external, _capabilities(harness))
        assert refused.value.code() == grpc.StatusCode.FAILED_PRECONDITION
        assert not harness.engine.store.job_exists("build-resume-1")
        _assert_no_device_work(harness)
        harness.finish_upload(ref, data)
        assert harness.blob_store.state(harness.scope(), ref.content_id).received_bytes == len(data)
        result = harness.factory._for_request(host).formal(_bind(harness, host))
        assert result.report["status"] == "ok"
        # Resume never rebuilt the artifact: the control host built exactly once.
        assert harness.builder_calls == 1
        assert len(harness.controller.ledger.builds()) == 1
        # The completed row still answers a replay after the artifact was cleaned.
        with pytest.raises(BlobError):
            harness.blob_store.state(harness.scope(), ref.content_id)
        replay = harness.factory._for_request(host).formal(_bind(harness, host))
        assert replay.report["status"] == "ok"
        assert harness.builder_calls == 1


def test_a_rewritten_build_provenance_artifact_digest_is_refused_before_a_job(tmp_path):
    with ModuleBuildGrpcHarness(tmp_path / "prebuilt") as harness:
        outcome = harness.host_build(SOURCE)
        harness.upload_module(outcome)
        # Build a *second*, valid artifact set; name it in the request but claim
        # the first set's provenance digest.  The tamper is real: the two digests
        # differ.
        other = harness.host_build(SOURCE + b"// extra\n")
        assert other.bundle.artifact_set_digest != outcome.bundle.artifact_set_digest
        harness.bundle = other.bundle
        harness.build = mb.request_build(
            harness.profile, source=SOURCE + b"// extra\n", mode="prebuilt",
            artifact_set_digest=outcome.bundle.artifact_set_digest,
            producer=other.evidence.producer_record(),
        )
        host = _host_request(harness, "build-tamper-digest-1", source=b"")
        with pytest.raises(JobError) as refused:
            _bind(harness, host)
        assert "artifact_set_digest" in str(refused.value)
        assert not harness.engine.store.job_exists("build-tamper-digest-1")
        _assert_no_device_work(harness)


def test_a_build_whose_outcome_claims_another_builder_version_is_refused(tmp_path):
    # The control-host builder itself reports a provenance that does not match
    # the registered profile: the closed loop refuses it before any upload or
    # Submit, instead of signing a build identity nobody probed.
    builder = mb.FakeTsmModuleBuilder(
        mb.control_host_profile(), build_reports_version="9.9.9",
    )
    with ModuleBuildGrpcHarness(tmp_path / "prebuilt", builder=builder) as harness:
        from tcenv.remote.artifact_builder import BuildProvenance, SourceBundle

        with pytest.raises(BuildError) as refused:
            harness.registry.build(
                source_bundle=SourceBundle(SOURCE),
                task_contract={"task_family": "kernel_module"},
                provenance=BuildProvenance(
                    profile=harness.profile,
                    source_sha256=hashlib.sha256(SOURCE).hexdigest(),
                ).to_dict(),
            )
        assert refused.value.error_class == "toolchain_profile"
        assert harness.builder is builder
        _assert_no_device_work(harness)


def test_a_build_identity_cannot_carry_a_shell_path_or_build_command():
    from tcenv.remote.artifact_builder import BuildProvenance, SourceBundle
    from tcenv.remote.job_types import JobError, NpuBuildRequest

    source = SourceBundle(SOURCE)
    document = BuildProvenance(
        profile=mb.evaluator_profile(), source_sha256=source.sha256,
    ).request_document(mode="evaluator")
    assert document["mode"] == "evaluator"
    for extra in ("shell", "path", "command", "build_command", "argv"):
        # Neither the request document nor the provenance document has a slot
        # for a command or a host path: the closed field set refuses both.
        with pytest.raises(JobError):
            NpuBuildRequest.from_dict({**document, extra: "/bin/sh -c true"})
        with pytest.raises(JobError):
            NpuBuildRequest.from_dict({
                **document,
                "provenance": {**document["provenance"], extra: "/tmp/build"},
            })


def test_a_build_that_misses_its_deadline_is_typed_and_never_runs(tmp_path):
    from tcenv.remote.artifact_builder import BuildProvenance, SourceBundle

    with ModuleBuildGrpcHarness(tmp_path / "prebuilt") as harness:
        with pytest.raises(BuildError) as refused:
            harness.registry.build(
                source_bundle=SourceBundle(SOURCE),
                task_contract={"task_family": "kernel_module"},
                provenance=BuildProvenance(
                    profile=harness.profile,
                    source_sha256=hashlib.sha256(SOURCE).hexdigest(),
                ).to_dict(),
                deadline_monotonic=time.monotonic() - 1.0,
            )
        assert refused.value.error_class == "deadline"
        # The deadline is not a candidate compile failure and the builder never
        # started a build.
        assert harness.builder_calls == 0
        _assert_no_device_work(harness)


def test_executable_bytes_that_differ_from_the_manifest_fail_the_upload(tmp_path):
    from tests.support.blob_sim.client import descriptor_for

    with ModuleBuildGrpcHarness(tmp_path / "prebuilt") as harness:
        outcome = harness.host_build(SOURCE)
        module = next(
            entry for entry in outcome.bundle.entries if entry.role == "EXECUTABLE_MODULE"
        )
        data = bytearray(outcome.payloads[module.path])
        data[len(data) // 2] ^= 0x01
        # Upload under the *manifest's* descriptor: the channel verifies the
        # exact bytes and fails the blob closed.
        ref = descriptor_for(
            outcome.payloads[module.path], media_type=module.media_type, chunk_size=8192,
        )
        assert ref.content_id == module.blob_ref["content_id"]
        harness.blob_client.begin(ref)
        for offset in range(0, len(data), 8192):
            harness.blob_client.write(ref.content_id, offset, bytes(data[offset:offset + 8192]))
        with pytest.raises(grpc.RpcError) as refused_commit:
            harness.blob_client.commit(ref.content_id)
        assert "digest verification failed" in refused_commit.value.details()
        # The tampered bytes are gone, so the artifact set cannot be submitted.
        harness.set_prebuilt(outcome, outcome.bundle, SOURCE)
        host = _host_request(harness, "build-bytes-tamper-1", source=b"")
        external = _wire(harness, host)
        with pytest.raises(grpc.RpcError) as refused_submit:
            _submit_wire(harness, external, _capabilities(harness))
        assert refused_submit.value.code() == grpc.StatusCode.FAILED_PRECONDITION
        assert not harness.engine.store.job_exists("build-bytes-tamper-1")
        _assert_no_device_work(harness)


def test_a_blob_reference_that_does_not_bind_the_manifest_is_refused(tmp_path):
    from dataclasses import replace

    from tcenv.remote.artifact_bundle import ArtifactBundle, BundleError

    with ModuleBuildGrpcHarness(tmp_path / "prebuilt") as harness:
        outcome = harness.host_build(SOURCE)
        harness.upload_module(outcome)
        module = next(
            entry for entry in outcome.bundle.entries if entry.role == "EXECUTABLE_MODULE"
        )
        # The blob reference is content-addressed and must bind the manifest
        # identity: a rewritten content id, digest or size is refused by the
        # neutral artifact model itself, before a request can name it.
        for field, value in (
            ("content_id", "blob-sha256-v1:" + "0" * 64),
            ("sha256", "0" * 64),
            ("size_bytes", module.size_bytes + 1),
        ):
            tampered = replace(module, blob_ref={**dict(module.blob_ref), field: value})
            with pytest.raises(BundleError):
                ArtifactBundle(entries=tuple(
                    tampered if entry.path == module.path else entry
                    for entry in outcome.bundle.entries
                ))
        # And the genuine artifact set only becomes submittable after the real
        # upload; the tampered entries above never produced a job row.
        assert not harness.engine.store.job_exists("build-blob-tamper-1")
        _assert_no_device_work(harness)


def test_a_prebuilt_request_without_or_with_a_rewritten_producer_link_is_refused(tmp_path):
    with ModuleBuildGrpcHarness(tmp_path / "prebuilt") as harness:
        from tcenv.remote.job_types import NpuBuildRequest

        outcome = harness.host_build(SOURCE)
        harness.upload_module(outcome)
        harness.set_prebuilt(outcome, outcome.bundle, SOURCE)
        document = harness.build.to_dict()
        assert document["producer"] == outcome.evidence.producer_record()
        # The producer link is required for a prebuilt request.
        with pytest.raises(JobError, match="producer build-evidence link"):
            NpuBuildRequest.from_dict({k: v for k, v in document.items() if k != "producer"})

        host = _host_request(harness, "build-producer-tamper-1", source=b"")
        external = _wire(harness, host)
        _submit_wire(harness, external, _capabilities(harness))
        final = _wait(harness, external, {"completed"})
        assert final.HasField("result")
        signed = gc.evidence_json(gc.blob_from_proto(final.result.evidence, allow_empty=True)[0])
        assert signed["build"]["producer"] == outcome.evidence.producer_record()
        # Rewriting the producer link in the signed evidence no longer repeats
        # the record the accepted request bound: the validator refuses it.  The
        # link is a whole record, so its duration, its log identity and its
        # evidence digest are each bound on their own.
        for name, mutate in (
            ("duration_ms", lambda producer: producer.__setitem__("duration_ms", producer["duration_ms"] + 1)),
            ("log.sha256", lambda producer: producer["log"].__setitem__("sha256", "0" * 64)),
            ("log.size_bytes", lambda producer: producer["log"].__setitem__("size_bytes", 1)),
            ("evidence_sha256", lambda producer: producer.__setitem__("evidence_sha256", "1" * 64)),
        ):
            tampered = json.loads(json.dumps(signed))
            mutate(tampered["build"]["producer"])
            with pytest.raises(contracts.ContractError, match="producer record"):
                contracts.npu_evidence_document(
                    tampered,
                    device_group=harness.engine.npu_execution.device_group.to_dict(),
                    expected_artifact_set_digest=outcome.bundle.artifact_set_digest,
                    expected_resources=harness.engine.npu_execution.declared_resources.to_dict(),
                    expected_build=harness.build.to_dict(),
                    request_artifact={"bundle": outcome.bundle.to_dict()},
                    measurement_kind="simulated", publication_allowed=False,
                    report_status="ok", dialect=contracts.NPU_BUILD_DIALECT,
                ), name
        # A received row must always carry the link.
        without = json.loads(json.dumps(signed))
        without["build"].pop("producer")
        with pytest.raises(contracts.ContractError, match="producer"):
            contracts.npu_evidence_document(
                without,
                device_group=harness.engine.npu_execution.device_group.to_dict(),
                expected_artifact_set_digest=outcome.bundle.artifact_set_digest,
                expected_resources=harness.engine.npu_execution.declared_resources.to_dict(),
                expected_build=harness.build.to_dict(),
                request_artifact={"bundle": outcome.bundle.to_dict()},
                measurement_kind="simulated", publication_allowed=False,
                report_status="ok", dialect=contracts.NPU_BUILD_DIALECT,
            )


def test_a_manifest_rewritten_on_the_wire_is_refused_before_a_job(tmp_path):
    with ModuleBuildGrpcHarness(tmp_path / "prebuilt") as harness:
        outcome = harness.host_build(SOURCE)
        harness.upload_module(outcome)
        harness.set_prebuilt(outcome, outcome.bundle, SOURCE)
        host = _host_request(harness, "build-manifest-tamper-1", source=b"")
        external = _wire(harness, host)
        message = gc.request_to_proto(external, dialect=harness.factory.contract)
        assert message.bundle.entries
        target = next(
            entry for entry in message.bundle.entries if entry.role == "EXECUTABLE_MODULE"
        )
        # Keep the declared artifact_set_digest, rewrite one manifest digest.
        target.sha256 = "0" * 64
        with pytest.raises(grpc.RpcError) as refused:
            harness.factory.stub.Submit(
                pb.SubmitRequest(
                    request=message,
                    request_digest=external.request_digest,
                    expected=gc.expected_identities_to_proto(harness.engine.expected_identities),
                    capability_id=_capabilities(harness).capability_id,
                    required_features=list(harness.factory.required_features),
                ),
                metadata=METADATA, timeout=10,
            )
        assert refused.value.code() in {
            grpc.StatusCode.FAILED_PRECONDITION, grpc.StatusCode.INVALID_ARGUMENT,
        }
        assert not harness.engine.store.job_exists("build-manifest-tamper-1")
        _assert_no_device_work(harness)


def test_a_request_naming_another_builder_version_is_refused_before_a_job(tmp_path):
    with ModuleBuildGrpcHarness(tmp_path / "prebuilt") as harness:
        outcome = harness.host_build(SOURCE)
        harness.upload_module(outcome)
        harness.bundle = outcome.bundle
        wrong = mb.control_host_profile(builder_version="9.9.9")
        harness.build = mb.request_build(
            wrong, source=SOURCE, mode="prebuilt",
            artifact_set_digest=outcome.bundle.artifact_set_digest,
            producer=outcome.evidence.producer_record(),
        )
        host = _host_request(harness, "build-version-tamper-1", source=b"")
        with pytest.raises(EvaluationUnavailable) as refused:
            harness.factory._for_request(host).formal(_bind(harness, host))
        assert "build profile is not served" in str(refused.value)
        assert not harness.engine.store.job_exists("build-version-tamper-1")
        _assert_no_device_work(harness)


def test_an_unregistered_or_mismatched_toolchain_is_refused_before_a_job(tmp_path):
    # (a) An unavailable builder makes the whole capability unavailable, before
    # any device probe.
    with ModuleBuildGrpcHarness(
        tmp_path / "unavailable",
        builder=mb.FakeTsmModuleBuilder(
            mb.control_host_profile(), available=False, detail="txcc is not installed",
        ),
    ) as harness:
        with pytest.raises(grpc.RpcError) as refused:
            _capabilities(harness)
        assert refused.value.code() == grpc.StatusCode.FAILED_PRECONDITION
        assert "txcc is not installed" in refused.value.details()
        assert harness.controller.ledger.exists() is False
        assert not harness.engine.store._conn.execute(
            "SELECT COUNT(*) FROM jobs"
        ).fetchone()[0]
    # (b) A builder whose probe reports another version than the request names is
    # a failed precondition at Submit; no job, attempt or device call is created.
    with ModuleBuildGrpcHarness(
        tmp_path / "mismatch", mode="evaluator",
        builder=mb.FakeTsmModuleBuilder(mb.evaluator_profile(), reported_version="9.9.9"),
    ) as harness:
        with pytest.raises(BuildError) as host_build:
            harness.registry.build(
                source_bundle=__import__(
                    "tcenv.remote.artifact_builder", fromlist=["SourceBundle"]
                ).SourceBundle(SOURCE),
                task_contract={},
                provenance=BuildProvenance(
                    profile=harness.profile,
                    source_sha256=hashlib.sha256(SOURCE).hexdigest(),
                ).to_dict(),
            )
        # The control-host side of the same closed loop refuses the probe first.
        assert host_build.value.error_class == "toolchain_profile"
        harness.set_evaluator_build(SOURCE)
        host = _host_request(harness, "build-mismatch-1")
        with pytest.raises(EvaluationUnavailable) as refused:
            harness.factory._for_request(host).formal(_bind(harness, host))
        assert "build profile is not served" in str(refused.value)
        assert not harness.engine.store.job_exists("build-mismatch-1")
        _assert_no_device_work(harness)


def test_an_unregistered_build_profile_is_refused_by_the_evaluator(tmp_path):
    with ModuleBuildGrpcHarness(tmp_path / "prebuilt") as harness:
        outcome = harness.host_build(SOURCE)
        harness.upload_module(outcome)
        harness.bundle = outcome.bundle
        unknown = mb.control_host_profile(build_profile="not-registered-v9")
        harness.build = mb.request_build(
            unknown, source=SOURCE, mode="prebuilt",
            artifact_set_digest=outcome.bundle.artifact_set_digest,
            producer=outcome.evidence.producer_record(),
        )
        host = _host_request(harness, "build-unregistered-1", source=b"")
        with pytest.raises(EvaluationUnavailable) as refused:
            harness.factory._for_request(host).formal(_bind(harness, host))
        assert "not registered" in str(refused.value)
        assert not harness.engine.store.job_exists("build-unregistered-1")
        _assert_no_device_work(harness)
        assert harness.builder_calls == 1


# -- evaluator timing ------------------------------------------------------
def test_the_evaluator_builds_the_source_in_the_attempt_and_signs_the_evidence(tmp_path):
    with ModuleBuildGrpcHarness(tmp_path / "evaluator", mode="evaluator") as harness:
        capability = _capabilities(harness)
        assert list(capability.targets[0].artifact_kinds) == ["source"]
        harness.set_evaluator_build(SOURCE)
        result, _host = _formal(harness, "build-evaluator-1")
        assert result.report["status"] == "ok"
        assert result.report["build"]["build_kind"] == "simulator"
        evidence = result.evidence
        build = evidence["build"]
        assert build["status"] == "ok" and build["mode"] == "evaluator"
        assert build["provenance"]["build_host"] == "evaluator"
        assert build["provenance"]["builder_id"] == mb.BUILDER_ID
        assert build["provenance"]["toolchain"] == {
            "id": mb.TOOLCHAIN_ID, "version": mb.TOOLCHAIN_VERSION,
            "sdk_version": mb.SDK_VERSION,
        }
        module = next(
            entry for entry in build["artifacts"]["entries"]
            if entry["role"] == "EXECUTABLE_MODULE"
        )
        assert all(
            load["sha256"] == module["sha256"] and load["path"] == module["path"]
            for load in build["loaded"]
        )
        # The control host never compiled this candidate.
        assert harness.builder_calls == 0
        builds = harness.controller.ledger.builds()
        assert len(builds) == 1 and builds[0]["mode"] == "evaluator"
        assert builds[0]["provenance"] == build["provenance"]
        loads = harness.controller.ledger.module_loads()
        assert [load["sha256"] for load in loads] == [module["sha256"]]
        assert result.report["candidate_us"] == pytest.approx(
            evidence["timing"]["device_elapsed_ns"] / 1000.0
        )
        assert not (harness.engine.store.state_dir / "build").exists()


def test_an_in_attempt_build_failure_is_a_compile_failure_not_a_timeout(tmp_path):
    with ModuleBuildGrpcHarness(tmp_path / "evaluator", mode="evaluator") as harness:
        harness.set_evaluator_build(FAILING_SOURCE)
        result, _host = _formal(harness, "build-compile-fail-1", source=FAILING_SOURCE)
        assert result.report["status"] == "compile_error"
        assert result.report["stage"] == "compile"
        assert result.report["candidate_us"] == 0
        evidence = result.evidence
        build = evidence["build"]
        assert build["status"] == "compile_failed"
        assert build["error_class"] == "compile"
        assert build["artifacts"] is None and build["loaded"] == []
        assert build["log"]["size_bytes"] > 0
        # A build that really ran records the compiler's own duration, never the
        # contract's "did not run" sentinel (0).
        assert build["duration_ms"] >= 1
        assert evidence["complete"] is False
        assert evidence["timing"]["launches"] == 0
        assert evidence["timing"]["device_elapsed_ns"] == 0
        assert evidence["publication_allowed"] is False
        # It is a committed candidate failure, not an infrastructure timeout.
        snapshot = harness.engine.store._job_by_id("build-compile-fail-1")
        assert snapshot.state == "completed"
        assert snapshot.failure is None
        # No loader record: the failed build never reached the device.
        assert harness.controller.ledger.module_loads() == []
        assert harness.controller.ledger.builds() == []
        assert harness.builder_calls == 0
        assert not (harness.engine.store.state_dir / "build").exists()


def test_a_worker_killed_after_its_build_retries_with_a_fresh_attempt_and_no_reuse(tmp_path):
    root = tmp_path / "evaluator"
    env = {"TCENV_VENDOR_C_ABORT_AFTER_BUILD": "1"}
    with ModuleBuildGrpcHarness(root, mode="evaluator", worker_env=env) as harness:
        harness.set_evaluator_build(SOURCE)
        host = _host_request(harness, "build-retry-1")
        external = _wire(harness, host)
        _submit_wire(harness, external, _capabilities(harness))
        interrupted = _wait(harness, external, {"interrupted", "failed", "completed"})
        assert gc.state_from_proto(interrupted.state) == "interrupted", interrupted.failure
        old_attempt = harness.engine.store._attempt(
            harness.engine.store._conn.execute(
                "SELECT * FROM attempts WHERE attempt_id=?",
                (interrupted.current_attempt.attempt_id,),
            ).fetchone()
        )
        # The killed worker left its build scratch behind: the temporary build
        # artifact is observable and is not claimed as cleaned.
        scratch = harness.engine.store.state_dir / "build" / old_attempt.attempt_id
        assert scratch.is_dir()
        assert harness.controller.ledger.builds() == []

        retried = _retry(harness, external, interrupted)
        assert gc.state_from_proto(retried.state) in {"queued", "running"}
        final = _wait(harness, external, {"completed", "failed", "interrupted"})
        assert gc.state_from_proto(final.state) == "completed", final.failure
        new_attempt = final.current_attempt
        assert new_attempt.attempt_id != old_attempt.attempt_id
        builds = harness.controller.ledger.builds()
        assert len(builds) == 1 and builds[0]["attempt_id"] == new_attempt.attempt_id
        # A result from the old attempt/fence is refused: the new attempt's
        # receipt and result stay the committed ones.
        snapshot = harness.engine.store.complete_attempt(
            "build-retry-1", old_attempt,
            report={"task": V1_TASK, "status": "ok"},
            evidence={},
        )
        assert snapshot.state == "completed"
        assert snapshot.current_attempt.attempt_id == new_attempt.attempt_id
        assert harness.engine.store._job_by_id("build-retry-1").result is not None
        assert harness.builder_calls == 0
    # Closing the frontend sweeps the orphaned build scratch of the dead attempt.
    with ModuleBuildGrpcHarness(root, mode="evaluator") as reopened:
        assert not (reopened.engine.store.state_dir / "build" / old_attempt.attempt_id).exists()
        assert reopened.engine.close_error is None


def test_an_unprovable_device_cleanup_quarantines_the_build_result(tmp_path):
    # The worker skips its own close *and* the controller cannot prove an
    # external release, so the loaded module's cleanup is genuinely unprovable.
    with ModuleBuildGrpcHarness(
        tmp_path / "evaluator", mode="evaluator", release_supported=False,
        worker_env={"TCENV_VENDOR_C_SKIP_CLOSE": "1"},
    ) as harness:
        harness.set_evaluator_build(SOURCE)
        host = _host_request(harness, "build-quarantine-1")
        external = _wire(harness, host)
        _submit_wire(harness, external, _capabilities(harness))
        deadline = time.monotonic() + 20
        snapshot = harness.engine.store._job_by_id("build-quarantine-1")
        while time.monotonic() < deadline and not snapshot.quarantined:
            time.sleep(0.03)
            snapshot = harness.engine.store._job_by_id("build-quarantine-1")
        assert snapshot.quarantined is True
        assert snapshot.state != "completed"
        assert snapshot.result is None
        assert harness.engine.close_error is not None
        # The build did happen, but a result whose device cleanup cannot be
        # proven is never committed.
        assert len(harness.controller.ledger.builds()) == 1


def test_a_candidate_timeout_reports_a_not_run_build_not_a_compile_failure(tmp_path):
    with ModuleBuildGrpcHarness(
        tmp_path / "evaluator", mode="evaluator",
        worker_env={"TCENV_VENDOR_C_SLOW": "1"},
    ) as harness:
        harness.set_evaluator_build(SOURCE)
        host = _host_request(harness, "build-timeout-1", timeout_s=0.2)
        external = _wire(harness, host)
        _submit_wire(harness, external, _capabilities(harness))
        final = _wait(harness, external, {"completed"})
        assert final.HasField("result")
        report = gc.report_json(gc.blob_from_proto(final.result.report, allow_empty=False)[0])
        assert report["status"] == "timeout"
        evidence = gc.evidence_json(gc.blob_from_proto(final.result.evidence, allow_empty=True)[0])
        build = evidence["build"]
        # The attempt never built: `not_run` claims no artifact, no load and no
        # compile error, so a timeout can never be read as an ok build.
        assert build["status"] == "not_run"
        assert build["error_class"] is None
        assert build["artifacts"] is None and build["loaded"] == []
        assert build["provenance"] == harness.build.provenance_document()
        assert evidence["complete"] is False
        assert evidence["publication_allowed"] is False
        assert evidence["timing"]["launches"] == 0
        # No build and no device load happened.
        assert harness.builder_calls == 0
        assert harness.controller.ledger.builds() == []
        assert harness.controller.ledger.module_loads() == []


def test_a_reported_build_identity_that_differs_from_the_request_is_never_signed(tmp_path):
    with ModuleBuildGrpcHarness(
        tmp_path / "evaluator", mode="evaluator",
        worker_env={"TCENV_VENDOR_C_FORGE_BUILD_PROVENANCE": "1"},
    ) as harness:
        harness.set_evaluator_build(SOURCE)
        host = _host_request(harness, "build-forged-provenance-1")
        external = _wire(harness, host)
        _submit_wire(harness, external, _capabilities(harness))
        final = _wait(harness, external, {"completed", "failed", "interrupted", "quarantined"})
        # The vendor really reported the forged identity: the worker's own
        # signed-evidence check refuses it (exit 22) instead of writing a result.
        assert final.HasField("failure") and "exited 22" in final.failure.message
        assert not final.HasField("result")
        # The ledger kept the identity the receiver actually accepted, not the
        # forged one that was reported to the worker.
        builds = harness.controller.ledger.builds()
        assert len(builds) == 1
        assert builds[0]["provenance"]["builder_version"] == mb.BUILDER_VERSION


def test_terminal_cleanup_removes_the_blob_lease_bytes_and_build_scratch(tmp_path):
    with ModuleBuildGrpcHarness(tmp_path / "prebuilt") as harness:
        outcome, result = _run_prebuilt(harness, "build-cleanup-1")
        assert result.report["status"] == "ok"
        module = next(
            entry for entry in outcome.bundle.entries if entry.role == "EXECUTABLE_MODULE"
        )
        with pytest.raises(BlobError):
            harness.blob_store.state(harness.scope(), module.blob_ref["content_id"])
        assert not harness.blob_store.leases()
        assert harness.controller.ledger.live_resources() == []
        assert not (harness.engine.store.state_dir / "build").exists()
        assert harness.engine.close_error is None


def test_a_tampered_stored_build_provenance_fails_closed(tmp_path):
    root = tmp_path / "prebuilt"
    with ModuleBuildGrpcHarness(root) as harness:
        _run_prebuilt(harness, "build-store-tamper-1")
        connection = harness.engine.store._conn
        row = connection.execute(
            "SELECT request_json FROM jobs WHERE request_id=?", ("build-store-tamper-1",)
        ).fetchone()
        document = json.loads(row["request_json"])
        assert document["build"]["provenance"]["builder_version"] == mb.BUILDER_VERSION
        document["build"]["provenance"]["builder_version"] = "9.9.9"
        connection.execute(
            "UPDATE jobs SET request_json=? WHERE request_id=?",
            (json.dumps(document), "build-store-tamper-1"),
        )
        connection.commit()
        with pytest.raises((StateCorrupt, JobError)):
            harness.engine.store._job_by_id("build-store-tamper-1")
    with pytest.raises((StateCorrupt, JobError)):
        reopened = ModuleBuildGrpcHarness(root)
        try:
            reopened.engine.store._job_by_id("build-store-tamper-1")
        finally:
            reopened.close()


def test_a_leftover_build_scratch_is_swept_and_an_unprovable_one_is_recorded(tmp_path):
    root = tmp_path / "sweep"
    state = root / "state"
    orphan = state / "build" / "dead-attempt"
    orphan.mkdir(parents=True)
    (orphan / "kernel.tx").write_bytes(SOURCE)
    with ModuleBuildGrpcHarness(root) as harness:
        # Startup removed the dead attempt's temporary build directory.
        assert not orphan.exists()
        assert harness.engine.close_error is None
        # Scratch an attempt leaves behind while this frontend is running is
        # removed by the close sweep (startup cannot have seen it).
        leaked = state / "build" / "leaked-attempt"
        leaked.mkdir(parents=True)
        (leaked / "kernel.tx").write_bytes(SOURCE)
    assert not leaked.exists()

    # A directory that cannot be removed is not reported as a clean teardown.
    stuck = state / "build" / "stuck-attempt"
    stuck.mkdir(parents=True)
    (stuck / "kernel.tx").write_bytes(SOURCE)
    os.chmod(stuck, 0o000)
    try:
        harness = ModuleBuildGrpcHarness(root)
        try:
            assert stuck.exists()
            assert harness.engine.close_error is not None
            assert "build scratch" in str(harness.engine.close_error)
        finally:
            harness.close()
    finally:
        os.chmod(stuck, 0o700)


def test_a_symlinked_build_scratch_root_never_deletes_outside_the_state_dir(tmp_path):
    """The sweep root is worker-reachable: a planted link must never be followed.

    ``<state_dir>/build`` is created by the worker (``mkdir(parents=True)``), so a
    compromised or racing worker can make it a symbolic link.  A sweep that
    followed it would recursively delete an arbitrary directory outside the
    state directory while ``close()`` reported a clean teardown.
    """

    root = tmp_path / "root"
    state = root / "state"
    state.mkdir(parents=True)
    outside = tmp_path / "outside"
    victim = outside / "user-data"
    victim.mkdir(parents=True)
    (victim / "important.txt").write_text("not the engine's to delete", encoding="utf-8")
    (outside / "sibling").mkdir()
    (outside / "sibling" / "also-important.txt").write_text("keep", encoding="utf-8")
    link = state / "build"
    os.symlink(outside, link, target_is_directory=True)

    harness = ModuleBuildGrpcHarness(root)
    try:
        # Nothing outside the state directory was touched.
        assert (victim / "important.txt").read_text(encoding="utf-8") == (
            "not the engine's to delete"
        )
        assert (outside / "sibling" / "also-important.txt").read_text() == "keep"
        # The link itself is refused, not silently accepted as the scratch root.
        assert link.is_symlink()
        error = harness.engine.close_error
        assert error is not None and "build scratch root" in str(error)
        assert "not a plain directory" in str(error)
    finally:
        harness.close()
    assert (victim / "important.txt").exists()
    assert link.is_symlink()


def test_a_planted_symlink_inside_the_build_scratch_is_left_alone_and_recorded(tmp_path):
    """An entry the builder never creates is quarantined, never followed."""

    root = tmp_path / "root"
    state = root / "state"
    outside = tmp_path / "outside"
    victim = outside / "kept"
    victim.mkdir(parents=True)
    (victim / "payload.bin").write_bytes(b"outside bytes")
    orphan = state / "build" / "dead-attempt"
    orphan.mkdir(parents=True)
    (orphan / "kernel.tx").write_bytes(SOURCE)
    planted = orphan / "escape"
    os.symlink(outside, planted, target_is_directory=True)

    harness = ModuleBuildGrpcHarness(root)
    try:
        # The planted link is neither followed nor removed; the engine cannot
        # prove what it points at, so it records an unverified cleanup.
        assert planted.is_symlink()
        assert (victim / "payload.bin").read_bytes() == b"outside bytes"
        error = harness.engine.close_error
        assert error is not None and "build scratch" in str(error)
        assert "symbolic link" in str(error)
    finally:
        harness.close()
    assert (victim / "payload.bin").read_bytes() == b"outside bytes"
    assert planted.is_symlink()


def test_a_regular_file_as_the_build_scratch_root_is_refused_not_deleted(tmp_path):
    root = tmp_path / "root"
    state = root / "state"
    state.mkdir(parents=True)
    (state / "build").write_bytes(b"not a directory")

    harness = ModuleBuildGrpcHarness(root)
    try:
        assert (state / "build").read_bytes() == b"not a directory"
        error = harness.engine.close_error
        assert error is not None and "not a plain directory" in str(error)
    finally:
        harness.close()
    assert (state / "build").read_bytes() == b"not a directory"


def test_an_absurdly_deep_build_scratch_is_refused_instead_of_walked(tmp_path):
    """The sweep must not recurse over a depth the worker invented."""

    from tcenv.remote.job_engine import MAX_SCRATCH_DEPTH

    root = tmp_path / "root"
    state = root / "state"
    deep = state / "build" / "deep-attempt"
    deep.mkdir(parents=True)
    cursor = deep
    for index in range(MAX_SCRATCH_DEPTH + 5):
        cursor = cursor / f"level-{index}"
    cursor.mkdir(parents=True)
    (cursor / "kernel.tx").write_bytes(SOURCE)

    harness = ModuleBuildGrpcHarness(root)
    try:
        # The whole tree is left alone (fail closed) and the refusal is visible.
        assert cursor.is_dir() and (cursor / "kernel.tx").read_bytes() == SOURCE
        error = harness.engine.close_error
        assert error is not None and "build scratch" in str(error)
        assert "deeper than" in str(error)
    finally:
        harness.close()
