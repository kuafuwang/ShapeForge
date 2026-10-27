"""P0 NPU source contract: canonical digest, receipt, signature and outbox chain."""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from tcenv.evaluation import EvaluationRequest, EvaluationUnavailable
from tcenv.remote import contracts
from tests.support.job_engine import (
    NPU_TARGET,
    NPU_TASK_FINGERPRINT,
    NPU_TOKEN,
    npu_get,
    npu_request,
)
from tests.support.vendor_sim.harness import (
    NPU_SIGNING_KEY_ID,
    NpuGrpcHarness,
)
from taic.evaluation_outbox import EvaluationOutbox

SOURCE = b"npu-kernel relu_kernel\n"


def _legacy_request(request_id: str = "npu-outbox-1", *, source: bytes = SOURCE) -> EvaluationRequest:
    return EvaluationRequest(
        kind="formal_eval", request_id=request_id, run_id="npu-run-1",
        session_id="npu-session-1", task="relu", dtype="fp16", size="small",
        target_id=NPU_TARGET, task_fingerprint=NPU_TASK_FINGERPRINT,
        source=source.decode("utf-8"), mode="quick", seed=3, reps=2,
        queue_timeout_s=3.0, candidate_timeout_s=3.0,
    )


def _npu_outbox(root: Path) -> EvaluationOutbox:
    return EvaluationOutbox(
        root,
        transport="grpc",
        api_contract_id=contracts.NPU_SOURCE_API_CONTRACT_ID,
        evidence_contract_id=contracts.NPU_SOURCE_EVIDENCE_CONTRACT_ID,
    )


def test_npu_canonical_request_binds_source_device_group_and_new_document_type():
    request = npu_request("npu-digest-1", source=SOURCE)
    document = request.digest_document()
    assert document["document_type"] == contracts.NPU_SOURCE_REQUEST_DOCUMENT_TYPE
    assert document["document_version"] == contracts.NPU_SOURCE_DOCUMENT_VERSION
    assert "candidate_digest" in document["request"]
    assert "candidate_sha256" not in document["request"]
    assert document["request"]["candidate_digest"] == request.candidate_digest
    assert document["request"]["execution"]["device_group"]["device_group_id"] == "vendor-a-group-0"

    changed_source = npu_request("npu-digest-1", source=SOURCE + b"\n")
    assert changed_source.request_digest != request.request_digest

    execution = request.npu.execution
    from tcenv.remote.job_types import DeviceGroupBinding, NpuExecutionSpec, NpuSourceRequest

    changed_group = DeviceGroupBinding(
        device_group_id=execution.device_group.device_group_id,
        vendor_id=execution.device_group.vendor_id,
        profile_digest=execution.device_group.profile_digest,
        topology_digest=execution.device_group.topology_digest,
        resource_generation="2",
    )
    changed = replace(request, npu=NpuSourceRequest(
        candidate_digest=request.npu.candidate_digest,
        execution=NpuExecutionSpec(device_group=changed_group),
    ))
    assert changed.request_digest != request.request_digest


def test_npu_contract_rejects_legacy_or_unsupported_execution_shape():
    request = npu_request("npu-shape-1", source=SOURCE)
    fields = request.digest_fields()
    fields["execution"] = {**fields["execution"], "artifact_kind": "bundle"}
    with pytest.raises(contracts.ContractError):
        contracts.request_digest_document(
            fields, source_bytes=SOURCE, dialect=contracts.NPU_SOURCE_DIALECT,
        )
    fields = request.digest_fields()
    fields["execution"] = {**fields["execution"], "required_timing_sources": ["HOST_WALL"]}
    with pytest.raises(contracts.ContractError):
        contracts.request_digest_document(
            fields, source_bytes=SOURCE, dialect=contracts.NPU_SOURCE_DIALECT,
        )
    with pytest.raises(contracts.ContractError):
        contracts.dialect_for(
            contracts.NPU_SOURCE_API_CONTRACT_ID, contracts.EVIDENCE_CONTRACT_ID,
        )


def test_npu_result_signature_rejects_every_tampered_chain_link(tmp_path):
    with NpuGrpcHarness(tmp_path) as harness:
        request = _legacy_request()
        external = npu_request(request.request_id, source=SOURCE)
        result = harness.factory.formal(request)
        snapshot = harness.engine.get_job(
            request_id=external.request_id, request_digest=external.request_digest,
            expected=harness.engine.expected_identities, grant_token=NPU_TOKEN,
        )
        stored = snapshot.result
        assert stored is not None
        document = {
            "provenance": stored.provenance,
            "report": stored.report_descriptor,
            "evidence": stored.evidence_descriptor,
            "receipt": stored.receipt,
        }
        signature = stored.signature["signature"]
        assert contracts.verify_result_signature(
            document, key_id=NPU_SIGNING_KEY_ID, key=harness.key,
            signature=signature, dialect=contracts.NPU_SOURCE_DIALECT,
        )
        assert result.receipt["candidate_digest"] == external.candidate_digest
        assert "candidate_sha256" not in result.receipt
        assert result.provenance.measurement_kind == "simulated"
        assert result.evidence["publication_allowed"] is False

        tampered_receipt = copy.deepcopy(document)
        tampered_receipt["receipt"]["candidate_digest"]["value"] = "0" * 64
        with pytest.raises(contracts.ContractError):
            contracts.verify_result_signature(
                tampered_receipt, key_id=NPU_SIGNING_KEY_ID, key=harness.key,
                signature=signature, dialect=contracts.NPU_SOURCE_DIALECT,
            )
        with pytest.raises(contracts.ContractError):
            contracts.receipt_document(
                tampered_receipt["receipt"], dialect=contracts.NPU_SOURCE_DIALECT,
            )

        tampered_group = copy.deepcopy(document)
        tampered_group["provenance"]["device_group"]["resource_generation"] = "2"
        with pytest.raises(contracts.ContractError):
            contracts.verify_result_signature(
                tampered_group, key_id=NPU_SIGNING_KEY_ID, key=harness.key,
                signature=signature, dialect=contracts.NPU_SOURCE_DIALECT,
            )

        tampered_evidence = copy.deepcopy(document)
        tampered_evidence["evidence"]["sha256"] = "1" * 64
        assert not contracts.verify_result_signature(
            tampered_evidence, key_id=NPU_SIGNING_KEY_ID, key=harness.key,
            signature=signature, dialect=contracts.NPU_SOURCE_DIALECT,
        )

        # The legacy verifier must not accept a NPU document merely because the
        # HMAC key happens to match.
        with pytest.raises(contracts.ContractError):
            contracts.verify_result_signature(
                document, key_id=NPU_SIGNING_KEY_ID, key=harness.key,
                signature=signature, dialect=contracts.LEGACY_DIALECT,
            )


def test_npu_outbox_row_version_digest_and_resume_tamper(tmp_path):
    root = tmp_path / "run"
    with NpuGrpcHarness(tmp_path) as harness:
        legacy = _legacy_request()
        external = npu_request(legacy.request_id, source=SOURCE)
        wire_request = replace(legacy, contract_request_digest=external.request_digest)
        outbox = _npu_outbox(root)
        assert outbox.begin(wire_request, epoch=harness.factory.epoch) is None
        result = harness.factory.formal(wire_request)
        outbox.complete(wire_request, result)

        state_path = outbox._state_path(wire_request.request_id)
        row = json.loads(state_path.read_text(encoding="utf-8"))
        assert row["schema_version"] == 2
        assert row["candidate_digest"] == external.candidate_digest
        assert row["device_group"] == result.receipt["device_group"]
        assert row["request_digest"] == external.request_digest

        resumed = _npu_outbox(root)
        replay = resumed.begin(wire_request, epoch=harness.factory.epoch)
        assert replay is not None
        assert replay.receipt["candidate_digest"] == external.candidate_digest

        # Tamper the durable result envelope: resume must fail closed.
        result_path = outbox._result_path(wire_request.request_id)
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        payload["receipt"]["candidate_digest"]["value"] = "2" * 64
        result_path.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(EvaluationUnavailable):
            _npu_outbox(root).begin(wire_request, epoch=harness.factory.epoch)

        # Restore the result, then tamper the outbox row's CandidateDigest.
        payload["receipt"]["candidate_digest"]["value"] = external.candidate_digest["value"]
        result_path.write_text(json.dumps(payload), encoding="utf-8")
        row = json.loads(state_path.read_text(encoding="utf-8"))
        row["candidate_digest"]["value"] = "3" * 64
        state_path.write_text(json.dumps(row), encoding="utf-8")
        with pytest.raises(EvaluationUnavailable):
            _npu_outbox(root).begin(wire_request, epoch=harness.factory.epoch)


def test_npu_request_rejects_a_dual_candidate_identity_and_legacy_mapping():
    from tcenv.remote.job_types import ExternalEvaluationRequest, JobError
    from tcenv.remote import grpc_contract as gc

    request = npu_request("npu-dual-1", source=SOURCE)
    with pytest.raises(JobError, match="must not also carry"):
        ExternalEvaluationRequest(
            kind=request.kind, request_id=request.request_id, run_id=request.run_id,
            session_id=request.session_id, task=request.task, dtype=request.dtype,
            size=request.size, target_id=request.target_id,
            task_fingerprint=request.task_fingerprint,
            candidate_sha256=request.npu.candidate_digest.value,
            source=request.source, mode=request.mode, seed=request.seed,
            reps=request.reps, queue_timeout_ms=request.queue_timeout_ms,
            candidate_timeout_ms=request.candidate_timeout_ms, cases=None,
            npu=request.npu,
        )
    # The NPU contract never silently downgrades to the legacy worker/evaluator
    # request, and it cannot be announced under the legacy contract id.
    with pytest.raises(JobError, match="no legacy request mapping"):
        request.to_legacy()
    with pytest.raises(JobError, match="request contract differs"):
        gc.request_to_proto(request, dialect=contracts.LEGACY_DIALECT)


def test_npu_outbox_row_device_group_tamper_fails_closed(tmp_path):
    root = tmp_path / "run"
    with NpuGrpcHarness(tmp_path) as harness:
        legacy = _legacy_request("npu-outbox-group-1")
        wire = harness.factory.wire_request_digest(legacy)
        external = npu_request(legacy.request_id, source=SOURCE)
        assert wire == external.request_digest
        request = replace(legacy, contract_request_digest=wire)
        outbox = _npu_outbox(root)
        assert outbox.begin(request, epoch=harness.factory.epoch) is None
        result = harness.factory.formal(request)
        outbox.complete(request, result)

        state_path = outbox._state_path(request.request_id)
        row = json.loads(state_path.read_text(encoding="utf-8"))
        assert row["device_group"] == result.receipt["device_group"]

        # A durable row whose device group was rewritten must not replay the
        # retained result: the row is part of the CandidateDigest chain.
        row["device_group"]["resource_generation"] = "9"
        state_path.write_text(json.dumps(row), encoding="utf-8")
        with pytest.raises(EvaluationUnavailable):
            _npu_outbox(root).begin(request, epoch=harness.factory.epoch)

        # A truncated group is corrupt rather than "absent".
        row["device_group"] = {"device_group_id": "vendor-a-group-0"}
        state_path.write_text(json.dumps(row), encoding="utf-8")
        with pytest.raises(EvaluationUnavailable):
            _npu_outbox(root).begin(request, epoch=harness.factory.epoch)


def test_npu_stale_durable_request_digest_fails_before_submit(tmp_path):
    with NpuGrpcHarness(tmp_path) as harness:
        legacy = _legacy_request("npu-stale-wire-1")
        stale = replace(legacy, contract_request_digest="0" * 64)
        with pytest.raises(EvaluationUnavailable, match="durable host request digest"):
            harness.factory.formal(stale)
        # Nothing was durably accepted: a changed device group or request can
        # never be resumed into a silently different job.
        assert harness.engine.store._conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == 0

        fresh = replace(legacy, contract_request_digest=harness.factory.wire_request_digest(legacy))
        assert harness.factory.formal(fresh).receipt["candidate_digest"]["value"] == (
            hashlib.sha256(SOURCE).hexdigest()
        )


def test_legacy_outbox_stays_version_one_and_has_no_candidate_digest(tmp_path):
    request = _legacy_request("legacy-outbox-1")
    outbox = EvaluationOutbox(tmp_path, transport="grpc")
    outbox.begin(request, epoch="e" * 64)
    row = json.loads(outbox._state_path(request.request_id).read_text(encoding="utf-8"))
    assert row["schema_version"] == 1
    assert row["candidate_sha256"] == request.candidate_sha256
    assert "candidate_digest" not in row
