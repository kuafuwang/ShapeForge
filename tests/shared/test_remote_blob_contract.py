"""P0b: blob reference inside the NPU contract, outbox, resume and cleanup.

The blob is only an artifact source; the identity chain is the contract:
canonical request -> request digest -> receipt -> evidence -> outbox row ->
HMAC result signature -> resume.  Every test here uses the real store, the real
engine, the real fake vendor worker and (where stated) the real six gRPC RPCs.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import threading
import threading
from dataclasses import replace
import time
from pathlib import Path

import pytest

from tcenv.evaluation import EvaluationRequest, EvaluationUnavailable
from taic.evaluation_outbox import EvaluationOutbox, bind_wire_digest
from tcenv.remote import contracts, grpc_contract as gc
from tcenv.remote.blob_channel import BlobLimits, BlobScope, BlobStore
from tcenv.remote.job_types import BlobDescriptor, JobError
from tests.support.job_engine import (
    NPU_TARGET,
    NPU_TASK_FINGERPRINT,
    NPU_TOKEN,
    npu_capabilities,
    npu_features,
    npu_grant,
    npu_request,
)
from tests.support.vendor_sim.harness import (
    NPU_HOST_SIGNATURE_KEY,
    NPU_SIGNING_KEY,
    BlobNpuGrpcHarness,
)
from tests.support.vendor_sim.ledger import VendorLedger

SOURCE = b"blob-kernel relu_kernel\n"


def submit_blob_job(harness, descriptor, *, request_id: str = "blob-req-1"):
    capability = npu_capabilities(harness.engine)
    request = npu_request(request_id, blob=descriptor)
    snapshot = harness.engine.submit(
        request=request, expected=harness.engine.expected_identities,
        capability_id=capability.capability_id,
        required_features=list(npu_features(contracts.NPU_BLOB_DIALECT)),
        grant_token=NPU_TOKEN,
    )
    return request, snapshot


def wait_for(harness, request, states: set[str], timeout: float = 15.0):
    deadline = time.monotonic() + timeout
    snapshot = harness.engine.get_job(
        request_id=request.request_id, request_digest=request.request_digest,
        expected=harness.engine.expected_identities, grant_token=NPU_TOKEN,
    )
    while snapshot.state not in states and time.monotonic() < deadline:
        time.sleep(0.02)
        snapshot = harness.engine.get_job(
            request_id=request.request_id, request_digest=request.request_digest,
            expected=harness.engine.expected_identities, grant_token=NPU_TOKEN,
        )
    assert snapshot.state in states, f"timed out waiting for {states}: {snapshot.state}"
    return snapshot


# -- Submit admission ------------------------------------------------------
def test_an_uncommitted_blob_can_never_be_submitted(tmp_path):
    harness = BlobNpuGrpcHarness(tmp_path / "blob-submit")
    try:
        descriptor = harness.upload(SOURCE, chunk_size=4, commit=False)
        capability = npu_capabilities(harness.engine)
        request = npu_request("blob-not-ready", blob=descriptor)
        with pytest.raises(JobError) as refused:
            harness.engine.submit(
                request=request, expected=harness.engine.expected_identities,
                capability_id=capability.capability_id,
                required_features=list(npu_features(contracts.NPU_BLOB_DIALECT)),
                grant_token=NPU_TOKEN,
            )
        assert refused.value.code == "failed_precondition"
        # Nothing durable was written and no lease was taken.
        assert harness.engine.store.live_request_ids() == []
        assert harness.engine.blob_store.leases() == {}
    finally:
        harness.close()


def test_another_grants_blob_is_not_submittable(tmp_path):
    other = npu_grant(token="blob-other-token", subject="blob-other-subject", session_id="blob-other-session")
    harness = BlobNpuGrpcHarness(tmp_path / "blob-scope", grants=[npu_grant(), other])
    try:
        # Uploaded under the *other* grant's scope.
        other_scope = harness.scope(token=other.token)
        descriptor = harness.upload(SOURCE, chunk_size=4, scope=other_scope)
        capability = npu_capabilities(harness.engine)
        request = npu_request("blob-foreign", blob=descriptor)
        with pytest.raises(JobError) as denied:
            harness.engine.submit(
                request=request, expected=harness.engine.expected_identities,
                capability_id=capability.capability_id,
                required_features=list(npu_features(contracts.NPU_BLOB_DIALECT)),
                grant_token=NPU_TOKEN,
            )
        assert denied.value.code == "permission_denied"
        assert harness.engine.store.live_request_ids() == []
    finally:
        harness.close()


def test_a_rejected_submit_releases_the_lease_it_took(tmp_path):
    # A full queue rejects Submit after the blob was verified and leased.
    harness = BlobNpuGrpcHarness(tmp_path / "blob-lease", max_queue_depth=1)
    try:
        first = harness.upload(SOURCE, chunk_size=4)
        second = harness.upload(os.urandom(16), chunk_size=4)
        capability = npu_capabilities(harness.engine)
        request = npu_request("blob-lease-1", blob=first)
        harness.engine.submit(
            request=request, expected=harness.engine.expected_identities,
            capability_id=capability.capability_id,
            required_features=list(npu_features(contracts.NPU_BLOB_DIALECT)),
            grant_token=NPU_TOKEN,
        )
        assert harness.engine.blob_store.leases() == {"blob-lease-1": [first.content_id]}

        blocked = npu_request("blob-lease-2", blob=second)
        with pytest.raises(JobError) as full:
            harness.engine.submit(
                request=blocked, expected=harness.engine.expected_identities,
                capability_id=capability.capability_id,
                required_features=list(npu_features(contracts.NPU_BLOB_DIALECT)),
                grant_token=NPU_TOKEN,
            )
        assert full.value.code == "resource_exhausted"
        # The rejected request left no job and no dangling lease; the live job keeps its own.
        assert not harness.engine.store.job_exists("blob-lease-2")
        assert harness.engine.blob_store.leases() == {"blob-lease-1": [first.content_id]}
    finally:
        harness.close()


# -- end to end ------------------------------------------------------------
def test_a_blob_job_completes_and_the_whole_digest_chain_agrees(tmp_path):
    harness = BlobNpuGrpcHarness(tmp_path / "blob-e2e")
    try:
        descriptor = harness.upload(SOURCE, chunk_size=4)
        request, _ = submit_blob_job(harness, descriptor)
        snapshot = wait_for(harness, request, {"completed", "failed", "cancelled"})
        assert snapshot.state == "completed"
        result = snapshot.result
        receipt = result.receipt
        evidence = json.loads(result.evidence)

        candidate = {"algorithm": contracts.BLOB_DIGEST_ALGORITHM, "value": descriptor.sha256}
        # request digest binds the descriptor, the CandidateDigest and the execution.
        document = contracts.request_digest_document(
            request.digest_fields(), source_bytes=b"", dialect=contracts.NPU_BLOB_DIALECT,
        )
        assert document["document_version"] == contracts.NPU_BLOB_DOCUMENT_VERSION
        assert document["request"]["blob_ref"] == descriptor.to_dict()
        assert document["request"]["candidate_digest"] == candidate
        assert request.request_digest == contracts.request_digest(
            request.digest_fields(), source_bytes=b"", dialect=contracts.NPU_BLOB_DIALECT,
        )
        assert receipt["candidate_digest"] == candidate
        assert receipt["request_digest"] == request.request_digest
        assert receipt["receipt_id"] == contracts.attempt_receipt_id(
            request_id=request.request_id, attempt_id=receipt["attempt_id"],
            request_digest=request.request_digest,
            candidate_digest=candidate, device_group=receipt["device_group"],
            dialect=contracts.NPU_BLOB_DIALECT,
        )
        assert evidence["candidate_digest"] == candidate
        assert evidence["device_group"] == receipt["device_group"]
        assert contracts.verify_result_signature(
            {"provenance": result.provenance, "report": result.report_descriptor,
             "evidence": result.evidence_descriptor, "receipt": receipt},
            key_id=result.signature["key_id"], key=NPU_SIGNING_KEY,
            signature=result.signature["signature"], dialect=contracts.NPU_BLOB_DIALECT,
        )

        # The worker really streamed the artifact chunk by chunk.
        reads = VendorLedger(harness.ledger_path).blob_reads()
        assert len(reads) == 1
        assert reads[0]["content_id"] == descriptor.content_id
        assert reads[0]["size_bytes"] == len(SOURCE)
        assert reads[0]["digest"] == descriptor.sha256
        assert reads[0]["chunks"] == -(-len(SOURCE) // 4)
        assert reads[0]["max_chunk_bytes"] <= 4

        # Terminal cleanup released the lease and removed the bytes.
        assert harness.engine.blob_store.leases() == {}
        assert harness.engine.blob_store.usage()["blobs"] == 0
    finally:
        harness.close()


# -- host outbox and resume ------------------------------------------------
def host_request(request_id: str) -> EvaluationRequest:
    return EvaluationRequest(
        task="relu", dtype="fp16", size="small", source="",
        mode="quick", reps=2, seed=3, target_id=NPU_TARGET,
        task_fingerprint=NPU_TASK_FINGERPRINT, kind="formal_eval",
        request_id=request_id, run_id="npu-run-1", session_id="npu-session-1",
    )


def blob_outbox(run_dir: Path) -> EvaluationOutbox:
    return EvaluationOutbox(
        run_dir, transport="grpc",
        api_contract_id=contracts.NPU_BLOB_API_CONTRACT_ID,
        evidence_contract_id=contracts.NPU_BLOB_EVIDENCE_CONTRACT_ID,
        signature_key=NPU_HOST_SIGNATURE_KEY,
    )


def test_outbox_binds_the_blob_candidate_and_replays_the_same_identity(tmp_path):
    harness = BlobNpuGrpcHarness(tmp_path / "blob-outbox")
    outbox = blob_outbox(tmp_path / "run")
    try:
        descriptor = harness.upload(SOURCE, chunk_size=4)
        harness.blob_ref = descriptor
        request = host_request("blob-host-1")
        request = bind_wire_digest(request, harness.factory, outbox)
        assert request.wire_candidate_digest == descriptor.sha256
        assert request.contract_request_digest

        assert outbox.begin(request) is None
        result = harness.factory.formal(request)
        outbox.complete(request, result)
        replay = outbox.begin(request)
        assert replay is not None
        assert replay.receipt["candidate_digest"]["value"] == descriptor.sha256
        assert replay.receipt["candidate_digest"]["algorithm"] == contracts.BLOB_DIGEST_ALGORITHM

        row = json.loads(
            next((tmp_path / "run" / "evaluation_outbox").glob("*.json")).read_text(encoding="utf-8")
        )
        assert row["schema_version"] == 2
        assert row["candidate_digest"] == {
            "algorithm": contracts.BLOB_DIGEST_ALGORITHM, "value": descriptor.sha256,
        }
        assert row["request_digest"] == request.contract_request_digest
        assert row["device_group"] == replay.receipt["device_group"]
    finally:
        harness.close()


def test_outbox_refuses_a_missing_rewritten_or_mismatched_blob_candidate(tmp_path):
    harness = BlobNpuGrpcHarness(tmp_path / "blob-outbox-tamper")
    outbox = blob_outbox(tmp_path / "run")
    try:
        descriptor = harness.upload(SOURCE, chunk_size=4)
        harness.blob_ref = descriptor
        bound = bind_wire_digest(host_request("blob-host-2"), harness.factory, outbox)
        assert outbox.begin(bound) is None
        result = harness.factory.formal(bound)
        outbox.complete(bound, result)

        # A row without the wire CandidateDigest cannot be opened or completed.
        unbind = replace(bound, wire_candidate_digest="")
        with pytest.raises(EvaluationUnavailable) as missing:
            outbox.begin(unbind)
        assert missing.value.code == "request_digest_missing"
        with pytest.raises(EvaluationUnavailable):
            outbox.complete(unbind, result)

        # A different candidate digest than the recorded row is refused.
        swapped = replace(bound, wire_candidate_digest=hashlib.sha256(b"other").hexdigest())
        with pytest.raises(EvaluationUnavailable):
            outbox.begin(swapped)

        # A rewritten host envelope signature is refused even if every field is
        # otherwise still the blob identity.
        forged = replace(result, signature="0" * 64)
        target = blob_outbox(tmp_path / "forged-run")
        assert target.begin(bound) is None
        with pytest.raises(EvaluationUnavailable) as bad_signature:
            target.complete(bound, forged)
        assert bad_signature.value.code == "signature_mismatch"

        # Rewriting the retained row's CandidateDigest is refused, not replayed.
        state = outbox._state_path(bound.request_id)
        document = json.loads(state.read_text(encoding="utf-8"))
        document["candidate_digest"] = {
            "algorithm": contracts.BLOB_DIGEST_ALGORITHM,
            "value": hashlib.sha256(b"forged").hexdigest(),
        }
        state.write_text(json.dumps(document), encoding="utf-8")
        with pytest.raises(EvaluationUnavailable):
            outbox.begin(bound)

        # Rewriting the retained result's receipt digest is refused too.
        state.write_text(json.dumps(
            {**document, "candidate_digest": bound.wire_candidate_digest and {
                "algorithm": contracts.BLOB_DIGEST_ALGORITHM, "value": bound.wire_candidate_digest,
            }},
        ), encoding="utf-8")
        result_path = outbox._result_path(bound.request_id)
        envelope = json.loads(result_path.read_text(encoding="utf-8"))
        envelope["receipt"]["candidate_digest"] = {
            "algorithm": contracts.BLOB_DIGEST_ALGORITHM,
            "value": hashlib.sha256(b"forged").hexdigest(),
        }
        result_path.write_text(json.dumps(envelope), encoding="utf-8")
        with pytest.raises(EvaluationUnavailable):
            outbox.begin(bound)
    finally:
        harness.close()


# -- idempotent replay after terminal cleanup ------------------------------
def _resubmit(harness, request):
    """Re-issue GetCapabilities + Submit for the same request/digest."""

    capability = npu_capabilities(harness.engine)
    return harness.engine.submit(
        request=request, expected=harness.engine.expected_identities,
        capability_id=capability.capability_id,
        required_features=list(npu_features(contracts.NPU_BLOB_DIALECT)),
        grant_token=NPU_TOKEN,
    )


def test_a_completed_job_replays_after_the_terminal_cleanup(tmp_path):
    harness = BlobNpuGrpcHarness(tmp_path / "blob-replay-done")
    try:
        descriptor = harness.upload(SOURCE, chunk_size=4)
        request, first = submit_blob_job(harness, descriptor)
        completed = wait_for(harness, request, {"completed"})
        # Terminal cleanup really removed the artifact and dropped the lease.
        assert harness.engine.blob_store.usage()["blobs"] == 0
        assert harness.engine.blob_store.leases() == {}

        replay = _resubmit(harness, request)
        assert replay.state == "completed"
        assert replay.current_attempt.attempt_id == completed.current_attempt.attempt_id
        assert replay.result.receipt["receipt_id"] == completed.result.receipt["receipt_id"]
        assert replay.result.receipt["candidate_digest"]["value"] == descriptor.sha256
        # The replay came from the committed row: no new lease, no new bytes.
        assert harness.engine.blob_store.leases() == {}
        assert harness.engine.blob_store.usage()["blobs"] == 0
        assert first.state in {"accepted", "queued", "running", "completed"}
    finally:
        harness.close()


def test_a_failed_job_replays_after_the_terminal_cleanup(tmp_path):
    harness = BlobNpuGrpcHarness(
        tmp_path / "blob-replay-failed", worker_env={"TCENV_VENDOR_A_BAD_REPORT": "1"},
    )
    try:
        descriptor = harness.upload(SOURCE, chunk_size=4)
        request, _ = submit_blob_job(harness, descriptor)
        failed = wait_for(harness, request, {"failed", "completed"})
        assert failed.state == "failed"
        assert harness.engine.blob_store.usage()["blobs"] == 0
        assert harness.engine.blob_store.leases() == {}

        replay = _resubmit(harness, request)
        assert replay.state == "failed"
        assert replay.current_attempt.attempt_id == failed.current_attempt.attempt_id
        assert replay.failure is not None and replay.failure.code == failed.failure.code
        assert harness.engine.blob_store.leases() == {}
        assert harness.engine.blob_store.usage()["blobs"] == 0
    finally:
        harness.close()


def test_a_terminal_job_replays_after_a_frontend_restart(tmp_path):
    root = tmp_path / "blob-replay-restart"
    harness = BlobNpuGrpcHarness(root)
    try:
        descriptor = harness.upload(SOURCE, chunk_size=4)
        request, _ = submit_blob_job(harness, descriptor)
        completed = wait_for(harness, request, {"completed"})
        assert harness.engine.blob_store.usage()["blobs"] == 0
    finally:
        harness.close()

    reopened = BlobNpuGrpcHarness(root)
    try:
        # The artifact is gone, but the durable row still answers the replay.
        assert reopened.engine.blob_store.usage()["blobs"] == 0
        replay = _resubmit(reopened, request)
        assert replay.state == "completed"
        assert replay.result.receipt["receipt_id"] == completed.result.receipt["receipt_id"]
        # A replay never re-creates the lease or the bytes.
        assert reopened.engine.blob_store.leases() == {}
        assert reopened.engine.blob_store.usage()["blobs"] == 0

        # A different digest for the same request id is still refused, and it also
        # does not take a lease for content that was never uploaded.
        conflicted = npu_request(
            request.request_id,
            blob=BlobDescriptor.for_bytes(os.urandom(8), media_type="text/plain; charset=utf-8", chunk_size=4),
        )
        with pytest.raises(JobError) as reused:
            _resubmit(reopened, conflicted)
        assert reused.value.code in {"already_exists", "not_found"}
        assert reopened.engine.blob_store.leases() == {}
    finally:
        reopened.close()


# -- concurrent Submit of one request id with two artifacts ---------------
def test_two_concurrent_submits_of_one_request_id_leave_only_the_winner_lease(tmp_path):
    harness = BlobNpuGrpcHarness(tmp_path / "blob-race")
    try:
        blobs = [
            harness.upload(os.urandom(8), chunk_size=4),
            harness.upload(os.urandom(8), chunk_size=4),
        ]
        engine = harness.engine
        capability = npu_capabilities(engine)

        # Force the dangerous interleaving deterministically: the replay check
        # reads "no job", then stalls long enough for the other Submit to reach
        # the same check.  Without one critical section both would lease their
        # own artifact and the loser would leave an orphan lease keyed by the
        # shared request id.
        real_exists = engine.store.job_exists
        start = threading.Barrier(2, timeout=10)

        def delayed_exists(request_id: str) -> bool:
            exists = real_exists(request_id)
            if not exists:
                time.sleep(0.15)
            return exists

        engine.store.job_exists = delayed_exists
        results: list[tuple[int, Any]] = []

        def submit(index: int) -> None:
            request = npu_request("blob-race-1", blob=blobs[index])
            try:
                start.wait()
                snapshot = engine.submit(
                    request=request, expected=engine.expected_identities,
                    capability_id=capability.capability_id,
                    required_features=list(npu_features(contracts.NPU_BLOB_DIALECT)),
                    grant_token=NPU_TOKEN,
                )
                results.append((index, snapshot))
            except JobError as exc:
                results.append((index, exc))

        threads = [threading.Thread(target=submit, args=(index,)) for index in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=30)
        assert len(results) == 2, "a concurrent Submit never returned"

        successes = [(index, value) for index, value in results if not isinstance(value, JobError)]
        failures = [(index, value) for index, value in results if isinstance(value, JobError)]
        assert len(successes) == 1, results
        assert len(failures) == 1
        winner_index, _winner = successes[0]
        assert failures[0][1].code == "already_exists"
        winner_blob = blobs[winner_index]

        # The durable job names exactly the winner's artifact...
        connection = sqlite3.connect(engine.store.state_dir / "jobs.sqlite3")
        try:
            row = connection.execute(
                "SELECT request_json FROM jobs WHERE request_id=?", ("blob-race-1",),
            ).fetchone()
        finally:
            connection.close()
        assert row is not None
        assert json.loads(row[0])["blob_ref"] == winner_blob.to_dict()

        # ...and only the winner's lease survives: the loser neither added an
        # orphan lease nor deleted the winner's.
        assert engine.blob_store.leases() == {"blob-race-1": [winner_blob.content_id]}

        # The job still runs to completion and its terminal cleanup releases the
        # single lease.
        request = npu_request("blob-race-1", blob=winner_blob)
        wait_for(harness, request, {"completed"}, timeout=20)
        assert engine.blob_store.leases() == {}
    finally:
        harness.close()


# -- host binding never silently rewrites the blob candidate ---------------
class _StubBlobEvaluator:
    contract = contracts.NPU_BLOB_DIALECT

    def __init__(self, request_digest: str, candidate: str | None) -> None:
        self.request_digest = request_digest
        self.candidate = candidate

    def wire_request_digest(self, request) -> str:
        return self.request_digest

    def wire_candidate_digest(self, request) -> str:
        if self.candidate is None:
            raise AssertionError("this evaluator must not expose a candidate")
        return self.candidate


class _StubBlobEvaluatorWithoutCandidate(_StubBlobEvaluator):
    def __init__(self, request_digest: str) -> None:
        super().__init__(request_digest, None)

    wire_candidate_digest = None  # type: ignore[assignment]


def test_bind_wire_digest_refuses_to_rebind_a_changed_blob_candidate():
    evaluator = _StubBlobEvaluator("1" * 64, "a" * 64)
    request = host_request("blob-rebind-1")
    bound = bind_wire_digest(request, evaluator, None, npu_contract=True)
    assert bound.contract_request_digest == "1" * 64
    assert bound.wire_candidate_digest == "a" * 64

    # The same candidate keeps the committed identity.
    again = bind_wire_digest(bound, evaluator, None, npu_contract=True)
    assert again.wire_candidate_digest == "a" * 64
    assert again.contract_request_digest == "1" * 64

    # Another candidate for an already-bound request is refused, not applied.
    changed = _StubBlobEvaluator("1" * 64, "c" * 64)
    with pytest.raises(EvaluationUnavailable) as mismatch:
        bind_wire_digest(bound, changed, None, npu_contract=True)
    assert mismatch.value.code == "request_digest_mismatch"

    # An evaluator that no longer exposes a candidate cannot bind at all.
    with pytest.raises(EvaluationUnavailable) as missing:
        bind_wire_digest(request, _StubBlobEvaluatorWithoutCandidate("1" * 64), None, npu_contract=True)
    assert missing.value.code == "request_digest_missing"

    # A first binding still accepts a fresh candidate for a fresh request.
    fresh = bind_wire_digest(host_request("blob-rebind-2"), changed, None, npu_contract=True)
    assert fresh.wire_candidate_digest == "c" * 64


def test_bind_wire_digest_binds_a_zero_byte_blob_candidate_that_equals_the_empty_source():
    empty_digest = hashlib.sha256(b"").hexdigest()
    request = host_request("blob-rebind-empty")
    assert request.candidate_sha256 == empty_digest
    evaluator = _StubBlobEvaluator("2" * 64, empty_digest)
    bound = bind_wire_digest(request, evaluator, None, npu_contract=True)
    # The value equals the empty-source digest, but the blob form must still
    # record it; inferring the form from equality would drop the identity.
    assert bound.wire_candidate_digest == empty_digest
    assert bound.contract_request_digest == "2" * 64


# -- durable tamper --------------------------------------------------------
def _rewrite_request_json(state: Path, request_id: str, mutate) -> None:
    connection = sqlite3.connect(state / "jobs.sqlite3")
    try:
        row = connection.execute(
            "SELECT request_json FROM jobs WHERE request_id=?", (request_id,),
        ).fetchone()
        document = json.loads(row[0])
        mutate(document)
        connection.execute(
            "UPDATE jobs SET request_json=? WHERE request_id=?",
            (json.dumps(document, sort_keys=True), request_id),
        )
        connection.commit()
    finally:
        connection.close()


def _tamper_blob_sha(document):
    document["blob_ref"]["sha256"] = hashlib.sha256(b"forged").hexdigest()
    document["blob_ref"]["content_id"] = contracts.blob_content_id(document["blob_ref"]["sha256"])


def _tamper_blob_size(document):
    document["blob_ref"]["size_bytes"] += 1


def _tamper_blob_chunk(document):
    document["blob_ref"]["chunk_size"] += 1


def _tamper_blob_media(document):
    document["blob_ref"]["media_type"] = "application/x-forged"


@pytest.mark.parametrize("mutate", [
    _tamper_blob_sha, _tamper_blob_size, _tamper_blob_chunk, _tamper_blob_media,
])
def test_a_tampered_stored_blob_reference_is_never_served(tmp_path, mutate):
    root = tmp_path / "blob-tamper"
    harness = BlobNpuGrpcHarness(root)
    request = None
    try:
        descriptor = harness.upload(SOURCE, chunk_size=4)
        request, _ = submit_blob_job(harness, descriptor)
        wait_for(harness, request, {"completed"})
        state = harness.engine.store.state_dir
        _rewrite_request_json(state, request.request_id, mutate)
        with pytest.raises(Exception) as corrupt:
            harness.engine.store._job_by_id(request.request_id)
        assert corrupt.value.__class__.__name__ in {"StateCorrupt", "JobError"}
    finally:
        harness.close()

    # Reopening the same durable state also refuses to serve it, and it is the
    # request/provenance check that fails (not an unrelated lock or schema error).
    with pytest.raises(Exception) as reopened_error:
        BlobNpuGrpcHarness(root)
    assert "request" in str(reopened_error.value).lower() or "provenance" in str(
        reopened_error.value
    ).lower()


# -- retry, fence and cleanup ---------------------------------------------
def test_an_interrupted_blob_attempt_retries_with_the_same_artifact(tmp_path):
    harness = BlobNpuGrpcHarness(
        tmp_path / "blob-retry", worker_env={"TCENV_VENDOR_A_ABORT_AFTER_LAUNCH": "1"},
    )
    try:
        descriptor = harness.upload(SOURCE, chunk_size=4)
        request, _ = submit_blob_job(harness, descriptor)
        interrupted = wait_for(harness, request, {"interrupted", "failed", "completed"}, timeout=20)
        assert interrupted.state in {"interrupted", "failed"}
        assert harness.engine.store.live_request_ids() == [request.request_id]
        assert harness.engine.blob_store.leases() == {request.request_id: [descriptor.content_id]}

        harness.engine.worker_env.pop("TCENV_VENDOR_A_ABORT_AFTER_LAUNCH", None)
        capability = npu_capabilities(harness.engine)
        harness.engine.retry(
            request_id=request.request_id, request_digest=request.request_digest,
            expected=harness.engine.expected_identities,
            interrupted_attempt_id=interrupted.current_attempt.attempt_id,
            capability_id=capability.capability_id,
            required_features=list(npu_features(contracts.NPU_BLOB_DIALECT)),
            grant_token=NPU_TOKEN,
        )
        completed = wait_for(harness, request, {"completed", "failed"}, timeout=20)
        assert completed.state == "completed"
        assert completed.result.receipt["candidate_digest"]["value"] == descriptor.sha256
        # The second attempt streamed the same committed artifact again.
        reads = VendorLedger(harness.ledger_path).blob_reads()
        assert len(reads) == 2
        assert {read["digest"] for read in reads} == {descriptor.sha256}
    finally:
        harness.close()


def test_a_failing_cleanup_quarantines_the_blob_and_the_job(tmp_path):
    from tests.support.blob_sim.faults import FaultIO

    io = FaultIO()
    harness = BlobNpuGrpcHarness(tmp_path / "blob-quarantine")
    harness.blob_store = BlobStore(
        harness.engine.store.state_dir / "blobs",
        limits=BlobLimits(max_size_bytes=1 << 20, max_chunk_size=4096),
        io=io,
    )
    harness.engine.blob_store = harness.blob_store  # the engine keeps its own reference
    try:
        descriptor = harness.upload(SOURCE, chunk_size=4)
        request, _ = submit_blob_job(harness, descriptor)
        io.fail("remove", times=1)
        snapshot = wait_for(harness, request, {"completed", "failed"})
        assert snapshot.state == "completed"
        state = harness.engine.blob_store.state(harness.scope(), descriptor.content_id)
        assert state.state == "quarantine"
        assert state.cleanup["state"] == "quarantine"
        # The job records the hazard instead of claiming a clean teardown.
        assert harness.engine.close_error is not None
        assert "blob cleanup failed" in str(harness.engine.close_error)
    finally:
        harness.close()


def test_a_restart_releases_an_orphaned_blob_lease(tmp_path):
    root = tmp_path / "blob-restart"
    harness = BlobNpuGrpcHarness(root)
    try:
        descriptor = harness.upload(SOURCE, chunk_size=4)
        # A lease with no durable job models a crash between commit and cleanup.
        harness.engine.blob_store.attach(harness.scope(), descriptor.content_id, "vanished-request")
        assert harness.engine.blob_store.leases() == {"vanished-request": [descriptor.content_id]}
    finally:
        harness.close()

    reopened = BlobNpuGrpcHarness(root)
    try:
        # Startup reconciliation released the orphan and cleaned the bytes; a live
        # job's lease is what keeps an artifact (covered by the store test).
        assert reopened.engine.blob_store.leases() == {}
        assert reopened.engine.blob_store.usage()["blobs"] == 0
    finally:
        reopened.close()


# -- the blob outbox never invents a candidate ------------------------------
EMPTY_SOURCE_SHA256 = hashlib.sha256(b"").hexdigest()


def test_unavailable_without_the_wire_candidate_writes_no_blob_row(tmp_path):
    harness = BlobNpuGrpcHarness(tmp_path / "blob-unavailable")
    outbox = blob_outbox(tmp_path / "run")
    try:
        # No durable row: nothing was accepted, so nothing is written and the
        # empty-source digest must never become a blob identity.
        request = host_request("blob-unavailable-1")
        outbox.unavailable(request, EvaluationUnavailable("transport_timeout", "boom"))
        assert not outbox._state_path(request.request_id).is_file()
        assert outbox.pending() == []
        assert not any(
            path for path in (tmp_path / "run" / "evaluation_outbox").glob("*.json")
            if path.name != "receipts.json"
        )

        # A durable row exists: an unbound candidate is refused and the row keeps
        # its committed identity and pending status.
        descriptor = harness.upload(SOURCE, chunk_size=4)
        harness.blob_ref = descriptor
        bound = bind_wire_digest(host_request("blob-unavailable-2"), harness.factory, outbox)
        assert outbox.begin(bound) is None
        state = outbox._state_path(bound.request_id)
        before = state.read_bytes()

        unbound_candidate = replace(bound, wire_candidate_digest="")
        with pytest.raises(EvaluationUnavailable) as missing_candidate:
            outbox.unavailable(unbound_candidate, EvaluationUnavailable("transport_timeout", "boom"))
        assert missing_candidate.value.code == "request_digest_missing"
        assert state.read_bytes() == before

        # Neither wire identity bound (a fresh host request carrying the same
        # request id): the pre-existing missing-wire-digest rule still applies.
        unbound_digest = host_request(bound.request_id)
        with pytest.raises(EvaluationUnavailable) as missing_digest:
            outbox.unavailable(unbound_digest, EvaluationUnavailable("transport_timeout", "boom"))
        assert missing_digest.value.code == "request_digest_missing"
        assert state.read_bytes() == before

        # The correctly bound request still records the real candidate, never the
        # empty-source digest.
        outbox.unavailable(bound, EvaluationUnavailable("transport_timeout", "boom"))
        row = json.loads(state.read_text(encoding="utf-8"))
        assert row["status"] == "unavailable"
        assert row["candidate_digest"] == {
            "algorithm": contracts.BLOB_DIGEST_ALGORITHM, "value": descriptor.sha256,
        }
        assert row["candidate_digest"]["value"] != EMPTY_SOURCE_SHA256
        assert json.dumps(row).count(EMPTY_SOURCE_SHA256) == 0
    finally:
        harness.close()


# -- the old contracts are unchanged --------------------------------------
def test_the_source_contract_rejects_a_blob_reference_and_keeps_its_algorithm():
    assert contracts.NPU_SOURCE_DIALECT.candidate_algorithms == ("sha256",)
    assert contracts.NPU_BLOB_DIALECT.candidate_algorithms == (contracts.BLOB_DIGEST_ALGORITHM,)
    blob = BlobDescriptor.for_bytes(SOURCE, media_type="text/plain; charset=utf-8", chunk_size=4)
    fields = {
        "kind": "formal_eval", "run_id": "r", "session_id": "s", "task": "relu",
        "dtype": "fp16", "size": "small", "target_id": NPU_TARGET,
        "task_fingerprint": NPU_TASK_FINGERPRINT,
        "candidate_digest": {"algorithm": contracts.BLOB_DIGEST_ALGORITHM, "value": blob.sha256},
        "execution": npu_request("probe").npu.execution.to_dict(),
        "blob_ref": blob.to_dict(),
        "mode": "quick", "seed": "3", "reps": 2,
        "queue_timeout_ms": 1000, "candidate_timeout_ms": 1000,
        "cases": None, "fast_bench": False, "require_contract_launch": False,
        "validate": False, "measure_serial": False, "cache_policy": "reuse",
    }
    with pytest.raises(contracts.ContractError):
        contracts.request_digest_document(fields, source_bytes=b"", dialect=contracts.NPU_SOURCE_DIALECT)
    # The legacy contract still hashes the exact bytes and never sees a blob.
    legacy = {key: value for key, value in fields.items() if key not in {
        "candidate_digest", "execution", "blob_ref",
    }}
    legacy["candidate_sha256"] = hashlib.sha256(SOURCE).hexdigest()
    document = contracts.request_digest_document(legacy, source_bytes=SOURCE)
    assert document["document_version"] == 1
    assert "blob_ref" not in document["request"]
    assert gc.request_from_proto(
        gc.request_to_proto(
            __import__("tests.support.job_engine", fromlist=["x"]).npu_request("inline"),
            dialect=contracts.NPU_SOURCE_DIALECT,
        ),
        dialect=contracts.NPU_SOURCE_DIALECT,
    ).request_digest == __import__("tests.support.job_engine", fromlist=["x"]).npu_request("inline").request_digest
