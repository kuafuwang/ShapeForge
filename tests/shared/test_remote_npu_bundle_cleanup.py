"""P1 vendor-B failure matrix: asynchronous service completion, cancel, reconcile.

Vendor B's completion is a service-issued token, so these are the ways a device
observation can be wrong without looking wrong: the token is lost, duplicated or
never issued for this attempt; the worker dies after the submission; the service
cannot prove it released the submission; or the evidence claims completeness it
does not have.  Every one of them must produce no ``ok``/publishable result.
"""

from __future__ import annotations

import json
import time

import pytest

from tcenv.remote.job_types import JobError, NpuResources
from tests.support.job_engine import (
    NPU_TOKEN,
    npu_bundle_capabilities,
    npu_bundle_engine,
    npu_bundle_execution,
    npu_bundle_get,
    npu_bundle_request,
    npu_bundle_features,
    npu_bundle_retry,
    npu_bundle_submit,
)
from tests.support.vendor_sim.vendor_b import FakeVendorBController

SIGNING = {"signing_key": b"s" * 32, "signing_key_id": "sim-npu-site/key-1"}


def _engine(root, *, worker_env=None, release_supported=True):
    service = root / "graph-service.json"
    controller = FakeVendorBController(service, release_supported=release_supported)
    engine = npu_bundle_engine(
        root / "state", service_path=service, controller=controller,
        worker_env=worker_env, **SIGNING,
    )
    return engine, controller, service


def _run(engine, request, states=("completed", "failed", "interrupted", "cancelled"), timeout=25.0):
    deadline = time.monotonic() + timeout
    snapshot = npu_bundle_get(engine, request)
    while snapshot.state not in states and time.monotonic() < deadline:
        time.sleep(0.02)
        snapshot = npu_bundle_get(engine, request)
    return snapshot


@pytest.mark.parametrize("fault", ["lost_token", "duplicate_token", "wrong_token"])
def test_a_broken_completion_token_never_becomes_an_ok_result(tmp_path, fault):
    engine, controller, service = _engine(
        tmp_path / "run", worker_env={"TCENV_VENDOR_B_FAULT": fault},
    )
    try:
        request = npu_bundle_request(f"bundle-token-{fault}", candidate_timeout_ms=4000)
        npu_bundle_submit(engine, request)
        final = _run(engine, request)
        assert final.state in {"interrupted", "failed"}, final.state
        assert final.result is None, "a broken completion token produced a result"
        service_state = controller.service.read()
        events = [event["event"] for event in service_state["device_events"]]
        assert "submitted" in events and "fault_fired" in events
        failures = [event for event in service_state["device_events"] if event["event"] == "device_failed"]
        assert failures, "the worker did not record the device failure"
        assert "token" in failures[0]["reason"]
        # The worker released its session before exiting, so the group is clean.
        assert service_state["submissions"] == {}
        assert service_state["live"] == []
    finally:
        engine.close(wait=5)


def test_a_never_completing_service_times_out_without_device_time(tmp_path):
    engine, controller, service = _engine(
        tmp_path / "run", worker_env={"TCENV_VENDOR_B_FAULT": "never_complete"},
    )
    try:
        request = npu_bundle_request("bundle-stall-1", candidate_timeout_ms=1200)
        npu_bundle_submit(engine, request)
        final = _run(engine, request)
        assert final.state == "completed", final.state
        evidence = json.loads(final.result.evidence)
        report = json.loads(final.result.report)
        assert report["status"] == "timeout"
        assert evidence["complete"] is False
        assert evidence["timing"]["device_elapsed_ns"] == 0
        assert evidence["timing"]["launches"] == 0
        assert evidence["publication_allowed"] is False
        assert evidence["spans"] == []
        # The host wait is recorded as host time only.
        assert evidence["timing"]["host_api_elapsed_ns"] > 0
    finally:
        engine.close(wait=5)


def test_a_worker_killed_after_submission_is_reconciled_by_the_controller(tmp_path):
    engine, controller, service = _engine(
        tmp_path / "run", worker_env={"TCENV_VENDOR_B_ABORT_AFTER_SUBMIT": "1"},
    )
    try:
        request = npu_bundle_request("bundle-kill-execute")
        npu_bundle_submit(engine, request)
        final = _run(engine, request)
        events = [event["event"] for event in controller.service.read()["device_events"]]
        # The device really was running before the kill: the service accepted the
        # submission and the worker died without collecting it.
        assert "submitted" in events
        assert final.state in {"interrupted", "failed"}, final.state
        assert final.result is None
        # Reconcile (not the dead worker's close) proved the cleanup.
        assert controller.service.submissions() == {}
        assert controller.service.live_markers() == []
        assert not final.quarantined
        # An interrupted attempt is retryable with a new capability and fence.
        if final.state == "interrupted":
            retried = npu_bundle_retry(engine, request, final.current_attempt.attempt_id)
            assert retried.current_attempt.lease_fence > final.current_attempt.lease_fence
    finally:
        engine.close(wait=5)


def test_an_unproven_service_cleanup_quarantines_and_refuses_retry(tmp_path):
    engine, controller, service = _engine(
        tmp_path / "run", worker_env={"TCENV_VENDOR_B_FAULT": "keep_service_job"},
    )
    try:
        request = npu_bundle_request("bundle-quarantine-1")
        npu_bundle_submit(engine, request)
        deadline = time.monotonic() + 20
        snapshot = npu_bundle_get(engine, request)
        while not snapshot.quarantined and time.monotonic() < deadline:
            time.sleep(0.02)
            snapshot = npu_bundle_get(engine, request)
        assert snapshot.quarantined is True
        assert "device cleanup was not proven" in snapshot.recovery_error
        assert snapshot.result is None
        assert engine.close_error is not None
        # The service kept the submission, so the live evidence stays observable.
        assert controller.service.submissions(), "failed cleanup must retain live service state"
        with pytest.raises(JobError) as error:
            npu_bundle_retry(engine, request, snapshot.current_attempt.attempt_id)
        assert error.value.code == "failed_precondition"
    finally:
        engine.close(wait=5)


def test_the_controller_cleans_up_even_when_the_worker_skips_its_close(tmp_path):
    engine, controller, service = _engine(
        tmp_path / "run", worker_env={"TCENV_VENDOR_B_SKIP_CLOSE": "1"},
    )
    try:
        request = npu_bundle_request("bundle-skip-close")
        npu_bundle_submit(engine, request)
        final = _run(engine, request)
        assert final.state == "completed", final.failure
        assert json.loads(final.result.report)["status"] == "ok"
        # The worker-local close was skipped; the worker-external controller is
        # what proved the release, so the service has no live submission left.
        assert controller.service.submissions() == {}
        assert controller.service.live_markers() == []
    finally:
        engine.close(wait=5)


@pytest.mark.parametrize("degradation", ["drop_span", "buffer_drop"])
def test_evidence_that_claims_completeness_is_refused(tmp_path, degradation):
    engine, controller, service = _engine(
        tmp_path / "run", worker_env={"TCENV_VENDOR_B_DEGRADE": degradation},
    )
    try:
        request = npu_bundle_request(f"bundle-degraded-{degradation}")
        npu_bundle_submit(engine, request)
        final = _run(engine, request)
        assert final.state != "completed" or final.result is None or (
            json.loads(final.result.report)["status"] != "ok"
        ), "incomplete evidence was published as an ok result"
        if final.result is not None:
            evidence = json.loads(final.result.evidence)
            assert evidence["complete"] is False or evidence["publication_allowed"] is False
    finally:
        engine.close(wait=5)


@pytest.mark.parametrize("degradation", ["wrong_report_time", "host_time_report"])
def test_a_report_that_rewrites_the_device_time_is_refused(tmp_path, degradation):
    engine, controller, service = _engine(
        tmp_path / "run", worker_env={"TCENV_VENDOR_B_DEGRADE": degradation},
    )
    try:
        request = npu_bundle_request(f"bundle-wrong-time-{degradation}")
        npu_bundle_submit(engine, request)
        final = _run(engine, request)
        assert final.state != "completed" or final.result is None, (
            "a report whose candidate time is not the device time was accepted"
        )
        if final.failure is not None:
            assert final.failure.domain in {"infrastructure", "protocol"}
    finally:
        engine.close(wait=5)


def test_a_slow_host_does_not_change_the_device_elapsed_time(tmp_path):
    engine, controller, service = _engine(
        tmp_path / "run", worker_env={"TCENV_VENDOR_B_SLOW_HOST": "1"},
    )
    try:
        request = npu_bundle_request("bundle-slow-host")
        npu_bundle_submit(engine, request)
        final = _run(engine, request)
        assert final.state == "completed", final.failure
        evidence = json.loads(final.result.evidence)
        report = json.loads(final.result.report)
        device_ns = evidence["timing"]["device_elapsed_ns"]
        host_ns = evidence["timing"]["host_api_elapsed_ns"]
        assert host_ns > device_ns * 10, "the host fault did not inflate the host time"
        assert report["candidate_us"] == device_ns / 1000.0
        # The device time is the service's, which the host loop never computes.
        assert evidence["spans"][0]["elapsed_ns"] == device_ns
    finally:
        engine.close(wait=5)


def test_an_agent_cannot_demand_a_rank_shape_the_profile_does_not_have(tmp_path):
    root = tmp_path / "run"
    service = root / "graph-service.json"
    controller = FakeVendorBController(service)
    # The trusted vendor-B profile has no rank/die dimension; a request that
    # declares two rank spans must be refused before any device work starts.
    execution = npu_bundle_execution("vendor-b", resources=NpuResources(required_rank_spans=2))
    engine = npu_bundle_engine(
        root / "state", service_path=service, controller=controller,
        vendor="vendor-b", **SIGNING,
    )
    engine.npu_execution = execution
    try:
        with pytest.raises(JobError) as error:
            npu_bundle_capabilities(engine)
        assert error.value.code == "failed_precondition"
        assert "rank span" in str(error.value)
        assert engine.store._conn.execute("SELECT COUNT(*) FROM capabilities").fetchone()[0] == 0
        assert controller.service.read()["submissions"] == {}
    finally:
        engine.close(wait=5)


def test_an_unprovable_cancel_is_quarantined_not_reported_as_cancelled(tmp_path):
    engine, controller, service = _engine(
        tmp_path / "run", worker_env={"TCENV_VENDOR_B_FAULT": "never_complete"},
        release_supported=False,
    )
    try:
        request = npu_bundle_request("bundle-cancel-unproven", candidate_timeout_ms=8000)
        npu_bundle_submit(engine, request)
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline and not controller.service.submissions():
            time.sleep(0.02)
        assert controller.service.submissions()
        engine.cancel(
            request_id=request.request_id, request_digest=request.request_digest,
            expected=engine.expected_identities, grant_token=NPU_TOKEN, reason="test",
        )
        deadline = time.monotonic() + 10
        snapshot = npu_bundle_get(engine, request)
        while not snapshot.quarantined and time.monotonic() < deadline:
            time.sleep(0.02)
            snapshot = npu_bundle_get(engine, request)
        # The cancel could not be proven, so the job is neither cancelled nor
        # completed, and the service still shows the live submission.
        assert snapshot.state != "cancelled"
        assert snapshot.result is None
        assert controller.service.submissions()
        assert snapshot.cancel_requested is True
        # A quarantine is recorded for the unproven cleanup, so a later reader
        # cannot mistake this attempt for a cleanly cancelled one.
        assert snapshot.quarantined is True
        assert engine.close_error is not None
    finally:
        engine.close(wait=5)


# -- artifact channel presence --------------------------------------------
def _blob_descriptor(data: bytes, *, chunk_size: int = 8192) -> dict:
    from tcenv.remote.job_types import BlobDescriptor

    return BlobDescriptor.for_bytes(
        data, media_type="application/octet-stream", chunk_size=chunk_size,
    ).to_dict()


def _blob_bundle(descriptors, data):
    """A graph bundle plus one blob-backed entry per descriptor."""

    from tcenv.remote.artifact_bundle import BundleEntry
    from tests.support.vendor_sim.vendor_b import build_graph_bundle

    template = build_graph_bundle()
    entries = [template.entries[0]]
    for index, descriptor in enumerate(descriptors):
        entries.append(BundleEntry(
            path=f"graph/params-{index}.bin", role="PARAMETER_TABLE",
            media_type="application/octet-stream", size_bytes=descriptor["size_bytes"],
            sha256=descriptor["sha256"], blob_ref=descriptor,
        ))
    return type(template)(entries=tuple(entries))


def test_a_bundle_that_needs_the_blob_channel_is_refused_before_it_exists(tmp_path):
    """No channel means no durable job -- not a worker that fails later.

    Without this check the Submit was accepted, an attempt leased the device and
    the worker only discovered the missing blob root, so a permanent
    configuration error looked like a retryable infrastructure failure.
    """

    from tcenv.remote.artifact_bundle import MAX_BUNDLE_INLINE_ENTRY_BYTES

    engine, controller, service = _engine(tmp_path / "run")
    try:
        assert engine.blob_store is None
        data = b"P" * (MAX_BUNDLE_INLINE_ENTRY_BYTES + 64)
        request = npu_bundle_request(
            "bundle-no-channel", bundle=_blob_bundle([_blob_descriptor(data)], data),
        )
        with pytest.raises(JobError, match="no blob channel"):
            npu_bundle_submit(engine, request)
        # Nothing durable was created and the device was never given a job.
        assert engine.store.job_exists(request.request_id) is False
        assert controller.service.submissions() == {}
        # An inline bundle on the same frontend is still served.
        npu_bundle_submit(engine, npu_bundle_request("bundle-inline-only"))
        snapshot = _run(engine, npu_bundle_request("bundle-inline-only"))
        assert snapshot.state == "completed", snapshot.failure
    finally:
        engine.close(wait=5)


# -- lease rollback scope -------------------------------------------------
def _blob_engine(root):
    """A bundle frontend with the real P0b channel attached."""

    from tcenv.remote.blob_channel import BlobLimits, BlobStore

    service = root / "graph-service.json"
    controller = FakeVendorBController(service)
    store = BlobStore(root / "state" / "blobs", limits=BlobLimits(
        max_size_bytes=4 << 20, max_chunk_size=65536, min_chunk_size=1,
        max_chunks=8192, max_uploads_per_scope=8,
    ))
    engine = npu_bundle_engine(
        root / "state", service_path=service, controller=controller,
        blob_store=store, **SIGNING,
    )
    return engine, controller, store


def _upload(engine, data: bytes, *, commit: bool = True) -> dict:
    from tcenv.remote.job_types import BlobDescriptor

    descriptor = BlobDescriptor.for_bytes(
        data, media_type="application/octet-stream", chunk_size=8192,
    )
    scope = _scope(engine)
    engine.blob_store.begin(scope, descriptor.to_dict())
    offset = 0
    while offset < len(data):
        engine.blob_store.write_chunk(scope, descriptor.content_id, offset, data[offset:offset + 8192])
        offset += 8192
    if commit:
        engine.blob_store.commit(scope, descriptor.content_id)
    return descriptor.to_dict()


def _scope(engine):
    from tcenv.remote.blob_channel import BlobScope

    grant = engine.store.blob_scope(NPU_TOKEN)
    identity = engine.identity
    return BlobScope(
        site_id=identity.site_id, device_id=identity.device_id,
        target_id=grant["target_id"], subject=grant["subject"],
        run_id=grant["run_id"], session_id=grant["session_id"],
    )


def test_a_rejected_multi_entry_submit_only_rolls_back_its_own_leases(tmp_path):
    """Rollback is scoped to the attach loop, never to the request id.

    A request id may already own a lease taken under an earlier attempt/fence
    (the P0b channel binds leases to attempts).  A failed multi-entry Submit must
    drop only the leases *it* created: releasing by request id would also free
    the earlier attempt's lease, which another execution may still be reading.
    """

    from tcenv.remote.artifact_bundle import MAX_BUNDLE_INLINE_ENTRY_BYTES
    from tcenv.remote.blob_channel import BlobAttempt

    engine, controller, store = _blob_engine(tmp_path / "run")
    del controller
    try:
        older = _upload(engine, b"O" * (MAX_BUNDLE_INLINE_ENTRY_BYTES + 16))
        fresh = _upload(engine, b"F" * (MAX_BUNDLE_INLINE_ENTRY_BYTES + 24))
        pending = _upload(engine, b"P" * (MAX_BUNDLE_INLINE_ENTRY_BYTES + 32), commit=False)
        scope = _scope(engine)
        # An earlier attempt of this request still holds ``older``.
        store.attach(
            scope, older["content_id"], "bundle-rollback-1",
            attempt=BlobAttempt(attempt_id="attempt-old", lease_fence=1),
        )
        assert store.state(scope, older["content_id"]).leases == ("bundle-rollback-1",)

        request = npu_bundle_request(
            "bundle-rollback-1",
            bundle=_blob_bundle([fresh, pending], b"unused"),
        )
        with pytest.raises(JobError, match="only a committed blob can be submitted"):
            npu_bundle_submit(engine, request)

        # The Submit attached ``fresh`` and then failed: only that lease is gone.
        assert store.state(scope, fresh["content_id"]).leases == ()
        # The earlier attempt's *bound* lease survived the rejected Submit: the
        # same request cannot take the blob under a different attempt identity.
        assert store.state(scope, older["content_id"]).leases == ("bundle-rollback-1",)
        from tcenv.remote.blob_channel import BlobError

        with pytest.raises(BlobError, match="another attempt/fence"):
            store.attach(scope, older["content_id"], "bundle-rollback-1")
        # And the rejected Submit left no durable job behind.
        assert engine.store.job_exists(request.request_id) is False
        # The blob that was never attached has no lease either.
        assert store.state(scope, pending["content_id"]).leases == ()

        # The bytes the caller uploaded are still available for a corrected
        # Submit: the rollback dropped the lease, not the artifact.
        corrected = npu_bundle_request(
            "bundle-rollback-1", bundle=_blob_bundle([fresh], b"unused"),
        )
        npu_bundle_submit(engine, corrected)
        assert store.state(scope, fresh["content_id"]).leases == ("bundle-rollback-1",)
    finally:
        engine.close(wait=5)


def test_a_failing_bundle_blob_cleanup_is_quarantined_not_hidden(tmp_path):
    """A terminal bundle job must not look clean if its blob bytes remain."""

    from tcenv.remote.blob_channel import BlobLimits, BlobStore
    from tests.support.blob_sim.faults import FaultIO

    engine, controller, _store = _blob_engine(tmp_path / "run")
    del controller
    io = FaultIO()
    engine.blob_store = BlobStore(
        engine.store.state_dir / "blobs",
        limits=BlobLimits(max_size_bytes=2 << 20, max_chunk_size=65536),
        io=io,
    )
    try:
        blob = _upload(engine, b"Q" * (262_144 + 64))
        request = npu_bundle_request(
            "bundle-quarantine-1", bundle=_blob_bundle([blob], b"unused"),
        )
        npu_bundle_submit(engine, request)
        scope = _scope(engine)
        assert engine.blob_store.state(scope, blob["content_id"]).leases
        io.fail("remove", times=1)
        snapshot = _run(engine, request)
        assert snapshot.state == "completed", snapshot.failure
        state = engine.blob_store.state(scope, blob["content_id"])
        assert state.state == "quarantine"
        assert state.cleanup["state"] == "quarantine"
        # The job records the hazard instead of claiming a clean teardown.
        assert engine.close_error is not None
        assert "blob cleanup failed" in str(engine.close_error)
    finally:
        engine.close(wait=5)


def test_a_store_rejection_after_attach_keeps_an_older_attempt_lease(tmp_path):
    """The durable-insert failure path must roll back the same narrow scope.

    A rejected capability is refused *after* the artifact was leased (the lease
    happens before the durable insert).  Releasing by request id there would drop
    the lease an earlier attempt/fence of this request still owns, exactly like
    the attach-failure path did before it was narrowed.
    """

    from tcenv.remote.artifact_bundle import MAX_BUNDLE_INLINE_ENTRY_BYTES
    from tcenv.remote.blob_channel import BlobAttempt, BlobError

    engine, controller, store = _blob_engine(tmp_path / "run")
    del controller
    try:
        older = _upload(engine, b"O" * (MAX_BUNDLE_INLINE_ENTRY_BYTES + 8))
        fresh = _upload(engine, b"F" * (MAX_BUNDLE_INLINE_ENTRY_BYTES + 16))
        scope = _scope(engine)
        store.attach(
            scope, older["content_id"], "bundle-post-attach-1",
            attempt=BlobAttempt(attempt_id="attempt-old", lease_fence=1),
        )
        assert store.state(scope, older["content_id"]).leases == ("bundle-post-attach-1",)

        request = npu_bundle_request(
            "bundle-post-attach-1", bundle=_blob_bundle([fresh], b"unused"),
        )
        with pytest.raises(JobError, match="capability is expired"):
            engine.submit(
                request=request, expected=engine.expected_identities,
                capability_id="capability-never-issued",
                required_features=list(npu_bundle_features(blob=True)),
                grant_token=NPU_TOKEN,
            )

        # The lease this Submit created is gone; the older attempt's is not.
        assert store.state(scope, fresh["content_id"]).leases == ()
        earlier = store.state(scope, older["content_id"])
        assert earlier.leases == ("bundle-post-attach-1",)
        with pytest.raises(BlobError, match="another attempt/fence"):
            store.attach(scope, older["content_id"], "bundle-post-attach-1")
        # A rejected Submit leaves no durable job and both uploads stay usable.
        assert engine.store.job_exists(request.request_id) is False
        assert store.state(scope, fresh["content_id"]).ready is True
        assert store.state(scope, older["content_id"]).ready is True
    finally:
        engine.close(wait=5)
