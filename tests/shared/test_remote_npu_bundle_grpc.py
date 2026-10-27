"""P1 bundle contract over the real six gRPC RPCs, with fake vendor B.

Vendor B is graph-only and asynchronous: the submission returns before the
device finishes and completion is observed through a service-issued completion
token.  These tests drive that through GetCapabilities/Submit/GetJob/Watch,
Cancel and Retry with the protobuf wire format, and keep the fake vendor in the
test support tree only.
"""

from __future__ import annotations

import hashlib
import json
import secrets
import time
import grpc
import pytest
from grpc_status import rpc_status

from tcenv.evaluation import EvaluationRequest
from tcenv.remote import contracts
from tcenv.remote import grpc_contract as gc
from tcenv.remote.v1 import evaluator_pb2 as pb
from tcenv.remote.artifact_bundle import BundleEntry
from tests.support.job_engine import (
    NPU_BUNDLE_FEATURES,
    NPU_TASK_FINGERPRINT,
    NPU_TOKEN,
    npu_bundle_features,
)
from tests.support.remote_contract import load_bundle_golden
from tests.support.vendor_sim.harness import (
    BundleSourceGrpcHarness,
    GraphGrpcHarness,
)
from tests.support.vendor_sim.vendor_b import PARAMETER_BLOB, build_graph_bundle

METADATA = (("authorization", f"Bearer {NPU_TOKEN}"),)


def _expected_pb(harness) -> pb.ExpectedEvaluator:
    expected = harness.engine.expected_identities.evaluator
    return pb.ExpectedEvaluator(
        site_id=expected.site_id, device_id=expected.device_id,
        target_id=expected.target_id, worker_build=expected.worker_build,
        task_pack_digest=expected.task_pack_digest,
        evaluator_epoch=expected.evaluator_epoch,
    )


def _capabilities(harness, *, features=None, api=contracts.NPU_BUNDLE_API_CONTRACT_ID,
                  evidence=contracts.NPU_BUNDLE_EVIDENCE_CONTRACT_ID, nonce=None):
    return harness.factory.stub.GetCapabilities(
        pb.CapabilitiesRequest(
            api_contract_ids=[api],
            evidence_contract_ids=[evidence],
            required_features=list(features if features is not None else NPU_BUNDLE_FEATURES),
            client_nonce=nonce or secrets.token_hex(16),
            expected=_expected_pb(harness),
        ),
        metadata=METADATA, timeout=10,
    )


def _wire_request(request_id: str, harness, *, source: str = "", timeout_s: float = 10.0):
    return EvaluationRequest(
        kind="formal_eval", request_id=request_id, run_id="npu-run-1",
        session_id="npu-session-1", task="relu", dtype="fp16", size="small",
        target_id=harness.engine.identity.target_id,
        task_fingerprint=NPU_TASK_FINGERPRINT, source=source, mode="quick",
        seed=3, reps=2, queue_timeout_s=timeout_s, candidate_timeout_s=timeout_s,
    )


def _features(harness=None) -> list[str]:
    """The feature set this harness' frontend advertises."""

    blob = harness is not None and getattr(harness, "blob_store", None) is not None
    return list(npu_bundle_features(blob=blob))


def _submit(harness, request_id: str, capability, *, source: str = "", timeout_s: float = 10.0):
    host = _wire_request(request_id, harness, source=source, timeout_s=timeout_s)
    request = harness.factory._for_request(host)._wire_request(
        harness.factory._for_request(host)._external_request(host)
    )
    snapshot = harness.factory.stub.Submit(
        pb.SubmitRequest(
            request=gc.request_to_proto(request, dialect=harness.factory.contract),
            request_digest=request.request_digest,
            expected=gc.expected_identities_to_proto(harness.engine.expected_identities),
            capability_id=capability.capability_id,
            required_features=_features(harness),
        ),
        metadata=METADATA, timeout=10,
    )
    return request, snapshot


def _get(harness, request, *, request_digest: str | None = None):
    return harness.factory.stub.GetJob(
        pb.JobRef(
            request_id=request.request_id,
            request_digest=request_digest or request.request_digest,
            expected=gc.expected_identities_to_proto(harness.engine.expected_identities),
        ),
        metadata=METADATA, timeout=10,
    )


def _wait_snapshot(harness, request, states, timeout: float = 25.0):
    deadline = time.monotonic() + timeout
    snapshot = None
    while time.monotonic() < deadline:
        snapshot = _get(harness, request)
        if gc.state_from_proto(snapshot.state) in states:
            return snapshot
        time.sleep(0.03)
    raise AssertionError(f"job did not reach {states}: {gc.state_from_proto(snapshot.state)}")


def test_the_six_rpcs_complete_a_graph_artifact_set(tmp_path):
    with GraphGrpcHarness(tmp_path) as harness:
        capability = _capabilities(harness)
        assert capability.api_contract_id == contracts.NPU_BUNDLE_API_CONTRACT_ID
        target = capability.targets[0]
        assert list(target.artifact_kinds) == ["bundle"]
        assert list(target.launch_kinds) == ["GRAPH"]
        assert list(target.launch_subtypes) == ["SERVICE"]
        assert list(target.timing_sources) == ["DEVICE_EVENT"]
        group = target.device_groups[0]
        assert group.state == "available" and group.logical_devices == 1

        request, submitted = _submit(harness, "bundle-grpc-1", capability)
        assert gc.state_from_proto(submitted.state) in {"queued", "running"}

        final = _wait_snapshot(harness, request, {"completed", "failed", "interrupted"})
        assert gc.state_from_proto(final.state) == "completed", final.failure
        assert final.HasField("result")
        # Read the descriptor the way a wire consumer does: straight off the
        # protobuf OpaqueBlob, not through the internal document.  The schema id
        # and the media type must be the version-2 identity the golden pins, so
        # a version-1 consumer can never mistake a bundle result for its own.
        golden = load_bundle_golden()["npu_bundle"]["signature"]["unsigned_result"]
        envelope_evidence = final.result.evidence
        assert envelope_evidence.schema_id == golden["evidence"]["schema_id"]
        assert envelope_evidence.schema_id == contracts.NPU_BUNDLE_TIMING_EVIDENCE_SCHEMA_ID
        assert envelope_evidence.media_type == golden["evidence"]["media_type"]
        assert envelope_evidence.media_type == contracts.NPU_BUNDLE_EVIDENCE_MEDIA_TYPE
        assert envelope_evidence.size_bytes == len(bytes(envelope_evidence.data))
        assert envelope_evidence.sha256 == hashlib.sha256(bytes(envelope_evidence.data)).hexdigest()
        assert envelope_evidence.schema_id != contracts.NPU_TIMING_EVIDENCE_SCHEMA_ID
        report_bytes, _report_descriptor, evidence_bytes, _evidence_descriptor, provenance, receipt, _signature = (
            gc.result_from_proto(final.result, dialect=harness.factory.contract)
        )
        report_document = json.loads(report_bytes)
        evidence = json.loads(evidence_bytes)
        assert report_document["status"] == "ok"
        assert evidence["schema_id"] == contracts.NPU_BUNDLE_TIMING_EVIDENCE_SCHEMA_ID
        assert evidence["complete"] is True
        assert evidence["publication_allowed"] is False
        assert evidence["artifact_set_digest"] == harness.bundle.artifact_set_digest
        # The device time is the service's completion time, not the host launch.
        assert evidence["timing"]["device_elapsed_ns"] > 0
        assert evidence["timing"]["host_api_elapsed_ns"] > 0
        assert report_document["candidate_us"] == evidence["timing"]["device_elapsed_ns"] / 1000.0
        assert [span["device_id"] for span in evidence["spans"]] == ["device-0"]
        assert receipt["candidate_digest"] == {
            "algorithm": contracts.BUNDLE_DIGEST_ALGORITHM,
            "value": harness.bundle.artifact_set_digest,
        }
        assert receipt["device_group"] == evidence["device_group"]
        assert provenance["request_digest"] == request.request_digest
        assert receipt["request_digest"] == request.request_digest
        # The service really ran the asynchronous path.
        events = [event["event"] for event in harness.controller.service.device_events()]
        assert events[:3] == ["submitted", "running", "completed"]


def test_watch_streams_the_snapshot_of_a_bundle_job(tmp_path):
    with GraphGrpcHarness(tmp_path) as harness:
        capability = _capabilities(harness)
        request, _submitted = _submit(harness, "bundle-watch-1", capability)
        events = list(harness.factory.stub.Watch(
            pb.WatchRequest(
                job=pb.JobRef(
                    request_id=request.request_id,
                    request_digest=request.request_digest,
                    expected=gc.expected_identities_to_proto(harness.engine.expected_identities),
                ),
                after_sequence=0,
            ),
            metadata=METADATA, timeout=20,
        ))
        assert events
        terminals = [event for event in events if event.HasField("snapshot")]
        assert terminals
        assert gc.state_from_proto(terminals[-1].snapshot.state) == "completed"
        assert terminals[-1].snapshot.request_digest == request.request_digest


def test_the_source_branch_of_the_same_contract_still_serves_vendor_a(tmp_path):
    with BundleSourceGrpcHarness(tmp_path) as harness:
        capability = _capabilities(harness)
        target = capability.targets[0]
        assert list(target.artifact_kinds) == ["source"]
        assert list(target.launch_kinds) == ["KERNEL"]
        request, _submitted = _submit(
            harness, "bundle-source-1", capability, source="npu-kernel relu_kernel\n",
        )
        assert request.npu.artifact_form == "source"
        final = _wait_snapshot(harness, request, {"completed", "failed", "interrupted"})
        assert gc.state_from_proto(final.state) == "completed", final.failure
        _report, _report_descriptor, evidence_bytes, _evidence_descriptor, _provenance, receipt, _sig = (
            gc.result_from_proto(final.result, dialect=harness.factory.contract)
        )
        evidence = json.loads(evidence_bytes)
        assert evidence["artifact_set_digest"] is None
        assert evidence["timing"]["device_elapsed_ns"] > 0
        assert receipt["candidate_digest"]["algorithm"] == "sha256"


def test_explicit_cancel_stops_the_service_submission(tmp_path):
    with GraphGrpcHarness(tmp_path, worker_env={"TCENV_VENDOR_B_FAULT": "never_complete"}) as harness:
        capability = _capabilities(harness)
        request, _submitted = _submit(harness, "bundle-cancel-1", capability, timeout_s=8.0)
        service = harness.controller.service
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            if service.submissions():
                break
            time.sleep(0.02)
        assert service.submissions(), "the service never received the submission"
        snapshot = harness.factory.stub.Cancel(
            pb.CancelRequest(
                job=pb.JobRef(
                    request_id=request.request_id,
                    request_digest=request.request_digest,
                    expected=gc.expected_identities_to_proto(harness.engine.expected_identities),
                ),
                reason="test cancel",
            ),
            metadata=METADATA, timeout=10,
        )
        assert gc.state_from_proto(snapshot.state) in {"cancelled", "running", "queued"}
        final = _wait_snapshot(harness, request, {"cancelled", "completed", "failed"})
        assert gc.state_from_proto(final.state) == "cancelled", final.failure
        # The worker-external controller proved the service resources are gone.
        assert service.submissions() == {}
        assert service.live_markers() == []
        assert not final.HasField("result")


def test_an_interrupted_attempt_is_retried_with_a_new_capability_and_fence(tmp_path):
    with GraphGrpcHarness(tmp_path, worker_env={"TCENV_VENDOR_B_FAULT": "lost_token"}) as harness:
        capability = _capabilities(harness)
        request, submitted = _submit(harness, "bundle-retry-1", capability, timeout_s=4.0)
        first_attempt = submitted.current_attempt
        interrupted = _wait_snapshot(harness, request, {"interrupted", "completed", "failed"})
        assert gc.state_from_proto(interrupted.state) == "interrupted", interrupted.failure
        harness.engine.worker_env["TCENV_VENDOR_B_FAULT"] = ""
        # Retry needs a fresh capability, and the new attempt gets a new fence.
        capability2 = _capabilities(harness)
        retried = harness.factory.stub.Retry(
            pb.RetryRequest(
                job=pb.JobRef(
                    request_id=request.request_id,
                    request_digest=request.request_digest,
                    expected=gc.expected_identities_to_proto(harness.engine.expected_identities),
                ),
                interrupted_attempt_id=first_attempt.attempt_id,
                capability_id=capability2.capability_id,
                required_features=list(npu_bundle_features()),
            ),
            metadata=METADATA, timeout=10,
        )
        assert retried.current_attempt.attempt_id != first_attempt.attempt_id
        assert retried.current_attempt.lease_fence > first_attempt.lease_fence
        final = _wait_snapshot(harness, request, {"completed", "failed", "interrupted"})
        assert gc.state_from_proto(final.state) == "completed", final.failure
        assert final.current_attempt.lease_fence == retried.current_attempt.lease_fence
        # The stale attempt cannot be retried again, and its old identity cannot
        # be used to observe the new attempt's result.
        with pytest.raises(grpc.RpcError) as error:
            harness.factory.stub.Retry(
                pb.RetryRequest(
                    job=pb.JobRef(
                        request_id=request.request_id,
                        request_digest=request.request_digest,
                        expected=gc.expected_identities_to_proto(harness.engine.expected_identities),
                    ),
                    interrupted_attempt_id=first_attempt.attempt_id,
                    capability_id=capability2.capability_id,
                    required_features=list(npu_bundle_features()),
                ),
                metadata=METADATA, timeout=10,
            )
        assert error.value.code() == grpc.StatusCode.FAILED_PRECONDITION
        with pytest.raises(grpc.RpcError) as stale:
            list(harness.factory.stub.Watch(
                pb.WatchRequest(
                    job=pb.JobRef(
                        request_id=request.request_id,
                        request_digest=request.request_digest,
                        expected=gc.expected_identities_to_proto(harness.engine.expected_identities),
                    ),
                    after_sequence=0,
                    attempt_id=first_attempt.attempt_id,
                ),
                metadata=METADATA, timeout=10,
            ))
        assert stale.value.code() == grpc.StatusCode.FAILED_PRECONDITION


def test_unknown_protobuf_fields_are_refused_by_the_contract(tmp_path):
    with GraphGrpcHarness(tmp_path) as harness:
        capability = _capabilities(harness)
        request, _submitted = _submit(harness, "bundle-unknown-1", capability)
        message = gc.request_to_proto(request, dialect=harness.factory.contract)
        raw = message.SerializeToString()
        # Append an unknown field (tag 19000, length-delimited) and parse it back:
        # protobuf keeps it, so the server must reject it explicitly.
        unknown_tag = (19000 << 3) | 2
        encoded = bytearray()
        while unknown_tag > 0x7F:
            encoded.append((unknown_tag & 0x7F) | 0x80)
            unknown_tag >>= 7
        encoded.append(unknown_tag)
        payload = b"\x01tsm-shell"
        encoded.append(len(payload))
        encoded.extend(payload)
        parsed = pb.EvaluationRequest()
        parsed.ParseFromString(raw + bytes(encoded))
        assert parsed.SerializeToString() != raw
        with pytest.raises(grpc.RpcError) as error:
            harness.factory.stub.Submit(
                pb.SubmitRequest(
                    request=parsed,
                    request_digest=request.request_digest,
                    expected=gc.expected_identities_to_proto(harness.engine.expected_identities),
                    capability_id=capability.capability_id,
                    required_features=list(npu_bundle_features()),
                ),
                metadata=METADATA, timeout=10,
            )
        assert error.value.code() == grpc.StatusCode.INVALID_ARGUMENT
        detail = rpc_status.from_call(error.value)
        assert detail is not None
        assert "unknown" in detail.message

        # The same rule applies inside a bundle entry.
        tampered = gc.request_to_proto(request, dialect=harness.factory.contract)
        entry = tampered.bundle.entries[0]
        entry_path = entry.path
        raw_entry = entry.SerializeToString()
        parsed_bundle = pb.EvaluationRequest()
        # Rebuild a request whose first entry carries an unknown field.
        rebuilt = pb.EvaluationRequest()
        rebuilt.CopyFrom(tampered)
        del rebuilt.bundle.entries[0]
        new_entry = rebuilt.bundle.entries.add()
        new_entry.ParseFromString(raw_entry + bytes(encoded))
        with pytest.raises(grpc.RpcError) as error:
            harness.factory.stub.Submit(
                pb.SubmitRequest(
                    request=rebuilt,
                    request_digest=request.request_digest,
                    expected=gc.expected_identities_to_proto(harness.engine.expected_identities),
                    capability_id=capability.capability_id,
                    required_features=list(npu_bundle_features()),
                ),
                metadata=METADATA, timeout=10,
            )
        assert error.value.code() == grpc.StatusCode.INVALID_ARGUMENT
        assert entry_path


def test_a_request_from_another_contract_version_is_refused(tmp_path):
    with GraphGrpcHarness(tmp_path) as harness:
        capability = _capabilities(harness)
        # A version 3 capability handshake is not served by this frontend.
        with pytest.raises(grpc.RpcError) as error:
            _capabilities(
                harness, api=contracts.NPU_SOURCE_API_CONTRACT_ID,
                evidence=contracts.NPU_SOURCE_EVIDENCE_CONTRACT_ID,
                features=(),
            )
        assert error.value.code() in {
            grpc.StatusCode.FAILED_PRECONDITION, grpc.StatusCode.INVALID_ARGUMENT,
        }
        # A version 3 request body (source/KERNEL) cannot ride the version 4
        # capability: the execution spec presence differs from the contract.
        source = b"npu-kernel relu\n"
        v3 = gc.pb.EvaluationRequest(
            kind=pb.EVALUATION_KIND_FORMAL_EVAL,
            request_id="bundle-mixed-1", run_id="npu-run-1", session_id="npu-session-1",
            task="relu", dtype="fp16", size="small",
            target_id=harness.engine.identity.target_id,
            task_fingerprint=NPU_TASK_FINGERPRINT, source=source,
            candidate_sha256=hashlib.sha256(source).hexdigest(),
            mode=pb.EVALUATION_MODE_QUICK, reps=2,
            queue_timeout_ms=3000, candidate_timeout_ms=3000,
            cache_policy=pb.CACHE_POLICY_REUSE,
        )
        v3.candidate_digest.algorithm = "sha256"
        v3.candidate_digest.value = hashlib.sha256(source).hexdigest()
        v3.execution.artifact_kind = "source"
        v3.execution.launch_kind = "KERNEL"
        v3.execution.launch_subtype = "SIMT"
        v3.execution.source_media_type = "text/plain; charset=utf-8"
        v3.execution.synchronization = "stream-event"
        v3.execution.required_timing_sources.append("DEVICE_EVENT")
        v3.execution.device_group.CopyFrom(
            pb.DeviceGroupBinding(**harness.engine.npu_execution.device_group.to_dict())
        )
        with pytest.raises(grpc.RpcError) as error:
            harness.factory.stub.Submit(
                pb.SubmitRequest(
                    request=v3,
                    request_digest="0" * 64,
                    expected=gc.expected_identities_to_proto(harness.engine.expected_identities),
                    capability_id=capability.capability_id,
                    required_features=list(npu_bundle_features()),
                ),
                metadata=METADATA, timeout=10,
            )
        assert error.value.code() == grpc.StatusCode.INVALID_ARGUMENT


def test_a_blob_backed_bundle_entry_is_read_through_the_channel(tmp_path):
    with GraphGrpcHarness(tmp_path, blob_store=True) as harness:
        # Build a bundle whose parameter table is a large, binary artifact
        # transported by content id: the control plane only carries metadata.
        large = PARAMETER_BLOB * 4000
        descriptor = harness.upload(large, chunk_size=8192)
        harness.blob_ref = descriptor
        bundle = build_graph_bundle()
        graph_entry = bundle.entries[0]
        harness.bundle = type(bundle)(entries=(
            graph_entry,
            BundleEntry(
                path="graph/params.bin", role="PARAMETER_TABLE",
                media_type="application/octet-stream", size_bytes=len(large),
                sha256=descriptor.sha256, blob_ref=descriptor,
            ),
        ))
        capability = _capabilities(harness, features=npu_bundle_features(blob=True))
        request, _submitted = _submit(harness, "bundle-blob-1", capability)
        assert request.npu.bundle.blob_entries
        assert request.source == b""
        final = _wait_snapshot(harness, request, {"completed", "failed", "interrupted"})
        assert gc.state_from_proto(final.state) == "completed", final.failure
        _report, _report_descriptor, evidence_bytes, _evidence_descriptor, _provenance, _receipt, _sig = (
            gc.result_from_proto(final.result, dialect=harness.factory.contract)
        )
        evidence = json.loads(evidence_bytes)
        assert evidence["artifact_set_digest"] == harness.bundle.artifact_set_digest
        # The worker really read the large entry through the channel, in chunks.
        reads = harness.controller.service.blob_reads()
        assert reads, "the worker never read the blob-backed entry"
        assert reads[0]["chunks"] > 1
        assert max(reads[0]["max_chunk_bytes"], 1) <= 8192
        assert reads[0]["entry_digests"] == [descriptor.sha256]
        # The blob lease is released when the job reaches a terminal state; the
        # bytes may already be cleaned up by the retention policy.
        leases = harness.blob_store.leases()
        assert descriptor.content_id not in {
            content_id for values in leases.values() for content_id in values
        }
