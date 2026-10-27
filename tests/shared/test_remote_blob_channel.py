"""P0b blob channel store: chunking, resume, authorization, quota, cleanup.

Every fault is injected through :class:`tests.support.blob_sim.faults.FaultIO`,
so the assertions run against real filesystem failures rather than a mock that
always succeeds.  The store is the production ``tcenv.remote.blob_channel``.
"""

from __future__ import annotations

import hashlib
import os
import threading

import pytest

from tcenv.remote import contracts
from tcenv.remote.blob_channel import (
    BlobAttempt,
    BlobError,
    BlobLimits,
    BlobReader,
    BlobScope,
    BlobStore,
)
from tests.support.blob_sim.faults import FakeCrash, FaultIO

SCOPE = BlobScope(
    site_id="blob-site", device_id="blob-device", target_id="blob-target",
    subject="blob-subject", run_id="blob-run", session_id="blob-session",
)
OTHER_SCOPE = BlobScope(
    site_id="blob-site", device_id="blob-device", target_id="blob-target",
    subject="other-subject", run_id="blob-run", session_id="other-session",
)
MEDIA = "application/octet-stream"


class Clock:
    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        return self.value


def make_store(root, *, io=None, clock=None, **limits):
    defaults = {
        "max_size_bytes": 4096, "min_chunk_size": 1, "max_chunk_size": 64,
        "max_chunks": 512, "max_uploads_per_scope": 2,
        "max_scope_bytes": 8192, "max_total_bytes": 16384,
        "partial_ttl_ms": 60_000, "ready_ttl_ms": 600_000,
    }
    defaults.update(limits)
    return BlobStore(root, limits=BlobLimits(**defaults), io=io, clock=clock or Clock())


def descriptor_for(data: bytes, *, chunk_size: int, media_type: str = MEDIA) -> dict:
    digest = hashlib.sha256(data).hexdigest()
    return {
        "content_id": contracts.blob_content_id(digest), "sha256": digest,
        "size_bytes": len(data), "media_type": media_type, "chunk_size": chunk_size,
    }


def read_all(store: BlobStore, scope, content_id: str) -> bytes:
    collected = bytearray()
    for chunk in store.iter_chunks(scope, content_id):
        collected += chunk
    return bytes(collected)


def upload(store: BlobStore, data: bytes, *, chunk_size: int = 4, scope=SCOPE, attempt=None):
    descriptor = descriptor_for(data, chunk_size=chunk_size)
    store.begin(scope, descriptor, attempt=attempt)
    offset = 0
    while offset < len(data):
        chunk = data[offset:offset + chunk_size]
        store.write_chunk(scope, descriptor["content_id"], offset, chunk, attempt=attempt)
        offset += len(chunk)
    store.commit(scope, descriptor["content_id"], attempt=attempt)
    return descriptor


# -- byte identity across chunk boundaries ---------------------------------
@pytest.mark.parametrize("size", [0, 1, 4, 5, 9, 64, 129])
def test_round_trip_is_byte_identical_at_every_chunk_boundary(tmp_path, size):
    data = os.urandom(size)
    store = make_store(tmp_path / "blobs")
    descriptor = upload(store, data, chunk_size=4)

    collected = bytearray()
    for chunk in store.iter_chunks(SCOPE, descriptor["content_id"]):
        assert len(chunk) <= 4
        collected += chunk
    assert bytes(collected) == data
    assert store.state(SCOPE, descriptor["content_id"]).state == "ready"


def test_binary_with_nul_and_non_utf8_bytes_is_opaque(tmp_path):
    data = bytes(range(256)) + b"\x00\xff\xfe" + b"\x00"
    store = make_store(tmp_path / "blobs", max_size_bytes=8192)
    descriptor = upload(store, data, chunk_size=7)

    assert store.read_chunk(SCOPE, descriptor["content_id"], 0, 7) == data[:7]
    assert store.read_chunk(SCOPE, descriptor["content_id"], 250, 7) == data[250:257]
    assert BlobReader(tmp_path / "blobs").digest(descriptor["content_id"]) == (
        hashlib.sha256(data).hexdigest(), len(data),
    )


def test_reads_and_writes_are_bounded_by_the_declared_chunk(tmp_path):
    data = os.urandom(200)
    io = FaultIO()
    store = make_store(tmp_path / "blobs", io=io)
    descriptor = upload(store, data, chunk_size=8)

    assert sum(1 for _ in store.iter_chunks(SCOPE, descriptor["content_id"])) == 25
    # Every artifact write is one declared chunk; only the small JSON manifest is
    # written as one unit, and no content read ever asks for more than chunk_size + 1.
    chunk_writes = [size for name, size in io.write_log if name.endswith(".chunk")]
    chunk_reads = [size for name, size in io.read_log if name.endswith(".chunk")]
    manifest_reads = [size for name, size in io.read_log if name.endswith(".json")]
    assert chunk_writes and max(chunk_writes) <= 8
    assert chunk_reads and max(chunk_reads) <= 9
    assert manifest_reads and max(manifest_reads) < 4096
    assert sum(1 for _ in store.iter_chunks(SCOPE, descriptor["content_id"])) == 25


def test_iter_chunks_is_lazy_and_reads_one_chunk_per_step(tmp_path):
    io = FaultIO()
    store = make_store(tmp_path / "blobs", io=io)
    data = os.urandom(20)
    descriptor = upload(store, data, chunk_size=4)
    chunk_reads = lambda: len([name for name, _ in io.read_log if name.endswith(".chunk")])  # noqa: E731

    before = chunk_reads()
    stream = store.iter_chunks(SCOPE, descriptor["content_id"])
    first = next(stream)
    assert first == data[0:4]
    # A generator, not a materialized list: one step = one chunk read.
    assert chunk_reads() - before == 1
    second = next(stream)
    assert second == data[4:8]
    assert chunk_reads() - before == 2
    rest = bytearray(first + second)
    for chunk in stream:
        rest += chunk
    assert bytes(rest) == data


# -- crash and resume -------------------------------------------------------
def test_a_crash_mid_upload_resumes_from_the_verified_offset(tmp_path):
    root = tmp_path / "blobs"
    clock = Clock()
    io = FaultIO()
    store = make_store(root, io=io, clock=clock)
    data = os.urandom(20)
    descriptor = descriptor_for(data, chunk_size=4)
    store.begin(SCOPE, descriptor)
    for offset in (0, 4, 8):
        store.write_chunk(SCOPE, descriptor["content_id"], offset, data[offset:offset + 4])

    io.crash("write_atomic", match="manifest.json")
    with pytest.raises(FakeCrash):
        # A crash is not a typed refusal: the process dies with the chunk on disk
        # but unacknowledged, which is what the restarted process must distrust.
        store.write_chunk(SCOPE, descriptor["content_id"], 12, data[12:16])
    assert io.crashes, "the injected crash did not fire"

    # A fresh store object re-reads the manifest exactly as a restarted process would.
    reopened = make_store(root, clock=clock)
    state = reopened.state(SCOPE, descriptor["content_id"])
    assert state.state == "uploading"
    assert state.verified_chunks == (0, 1, 2)
    assert state.received_bytes == 12

    # Resume from the verified offset, not from the beginning.
    for offset in (12, 16):
        reopened.write_chunk(SCOPE, descriptor["content_id"], offset, data[offset:offset + 4])
    assert reopened.commit(SCOPE, descriptor["content_id"]).state == "ready"
    assert read_all(reopened, SCOPE, descriptor["content_id"]) == data


def test_a_committed_blob_survives_a_restart(tmp_path):
    root = tmp_path / "blobs"
    data = os.urandom(30)
    store = make_store(root)
    descriptor = upload(store, data, chunk_size=4)

    reopened = make_store(root)
    assert reopened.state(SCOPE, descriptor["content_id"]).state == "ready"
    assert read_all(reopened, SCOPE, descriptor["content_id"]) == data


# -- idempotency and conflict ----------------------------------------------
def test_replaying_an_acknowledged_chunk_is_idempotent_only_for_identical_bytes(tmp_path):
    store = make_store(tmp_path / "blobs")
    data = os.urandom(16)
    descriptor = descriptor_for(data, chunk_size=4)
    store.begin(SCOPE, descriptor)
    store.write_chunk(SCOPE, descriptor["content_id"], 0, data[0:4])

    again = store.write_chunk(SCOPE, descriptor["content_id"], 0, data[0:4])
    assert again.received_bytes == 4 and again.verified_chunks == (0,)
    with pytest.raises(BlobError) as conflict:
        store.write_chunk(SCOPE, descriptor["content_id"], 0, b"XXXX")
    assert conflict.value.code == "failed_precondition"


def test_out_of_order_offsets_are_refused(tmp_path):
    store = make_store(tmp_path / "blobs")
    data = os.urandom(12)
    descriptor = descriptor_for(data, chunk_size=4)
    store.begin(SCOPE, descriptor)
    with pytest.raises(BlobError) as ahead:
        store.write_chunk(SCOPE, descriptor["content_id"], 4, data[4:8])
    assert ahead.value.code == "failed_precondition"
    assert store.state(SCOPE, descriptor["content_id"]).received_bytes == 0


def test_the_same_content_id_cannot_be_redeclared_with_another_size(tmp_path):
    store = make_store(tmp_path / "blobs")
    data = os.urandom(8)
    descriptor = descriptor_for(data, chunk_size=4)
    store.begin(SCOPE, descriptor)
    tampered = {**descriptor, "size_bytes": 9}
    with pytest.raises(BlobError) as conflict:
        store.begin(SCOPE, tampered)
    assert conflict.value.code == "failed_precondition"


def test_a_content_id_that_does_not_bind_its_digest_is_refused(tmp_path):
    store = make_store(tmp_path / "blobs")
    data = os.urandom(8)
    descriptor = descriptor_for(data, chunk_size=4)
    tampered = {**descriptor, "sha256": "0" * 64}
    with pytest.raises(BlobError) as refused:
        store.begin(SCOPE, tampered)
    assert refused.value.code == "failed_precondition"


def test_a_wrong_digest_declaration_fails_at_commit_and_is_unusable(tmp_path):
    store = make_store(tmp_path / "blobs")
    data = os.urandom(8)
    descriptor = descriptor_for(data, chunk_size=4)
    # Declare a different digest for content that will not match it: the content
    # id must still bind the declared digest, so this is a *valid* declaration
    # whose bytes simply do not verify.
    other = hashlib.sha256(b"not the uploaded bytes").hexdigest()
    wrong = {**descriptor, "sha256": other, "content_id": contracts.blob_content_id(other)}
    store.begin(SCOPE, wrong)
    store.write_chunk(SCOPE, wrong["content_id"], 0, data[0:4])
    store.write_chunk(SCOPE, wrong["content_id"], 4, data[4:8])
    with pytest.raises(BlobError) as mismatch:
        store.commit(SCOPE, wrong["content_id"])
    assert mismatch.value.code == "failed_precondition"
    assert store.state(SCOPE, wrong["content_id"]).state == "failed"
    with pytest.raises(BlobError):
        store.read_chunk(SCOPE, wrong["content_id"], 0, 4)
    with pytest.raises(BlobError):
        store.attach(SCOPE, wrong["content_id"], "request-1")


def test_repeated_commit_is_idempotent_and_writes_after_ready_are_refused(tmp_path):
    store = make_store(tmp_path / "blobs")
    data = os.urandom(8)
    descriptor = upload(store, data, chunk_size=4)
    assert store.commit(SCOPE, descriptor["content_id"]).state == "ready"
    with pytest.raises(BlobError) as ready:
        store.write_chunk(SCOPE, descriptor["content_id"], 0, data[0:4])
    assert ready.value.code == "failed_precondition"


def test_a_declared_blob_is_accepted_then_uploading_and_both_are_resumable(tmp_path):
    root = tmp_path / "blobs"
    store = make_store(root)
    data = os.urandom(8)
    descriptor = descriptor_for(data, chunk_size=4)
    declared = store.begin(SCOPE, descriptor)
    assert declared.state == "accepted"
    assert declared.received_bytes == 0

    # A restart in `accepted` resumes as `accepted`, not as a lost declaration.
    assert make_store(root).begin(SCOPE, descriptor).state == "accepted"
    store.write_chunk(SCOPE, descriptor["content_id"], 0, data[0:4])
    assert store.state(SCOPE, descriptor["content_id"]).state == "uploading"
    assert make_store(root).begin(SCOPE, descriptor).state == "uploading"


def test_an_uncommitted_blob_cannot_be_attached_to_a_request(tmp_path):
    store = make_store(tmp_path / "blobs")
    data = os.urandom(8)
    descriptor = descriptor_for(data, chunk_size=4)
    store.begin(SCOPE, descriptor)
    store.write_chunk(SCOPE, descriptor["content_id"], 0, data[0:4])
    with pytest.raises(BlobError) as incomplete:
        store.attach(SCOPE, descriptor["content_id"], "request-1")
    assert incomplete.value.code == "failed_precondition"


# -- limits ----------------------------------------------------------------
def test_oversized_blob_chunk_and_quota_are_refused_before_bytes_land(tmp_path):
    store = make_store(tmp_path / "blobs", max_size_bytes=8, max_scope_bytes=12)
    too_big = descriptor_for(os.urandom(9), chunk_size=4)
    with pytest.raises(BlobError) as size:
        store.begin(SCOPE, too_big)
    assert size.value.code == "resource_exhausted"

    too_wide = descriptor_for(os.urandom(8), chunk_size=65)
    with pytest.raises(BlobError) as chunk:
        store.begin(SCOPE, too_wide)
    assert chunk.value.code == "resource_exhausted"

    first = upload(store, os.urandom(8), chunk_size=4)
    assert store.state(SCOPE, first["content_id"]).state == "ready"
    over_quota = descriptor_for(os.urandom(8), chunk_size=4)
    with pytest.raises(BlobError) as quota:
        store.begin(SCOPE, over_quota)
    assert quota.value.code == "resource_exhausted"
    assert store.usage()["ready_bytes"] == 8


def test_more_chunks_than_declared_and_oversized_write_are_refused(tmp_path):
    store = make_store(tmp_path / "blobs")
    data = os.urandom(8)
    descriptor = descriptor_for(data, chunk_size=4)
    store.begin(SCOPE, descriptor)
    with pytest.raises(BlobError) as wide:
        store.write_chunk(SCOPE, descriptor["content_id"], 0, data[0:5])
    assert wide.value.code == "resource_exhausted"
    with pytest.raises(BlobError) as beyond:
        store.write_chunk(SCOPE, descriptor["content_id"], 0, data[0:1])
    assert store.state(SCOPE, descriptor["content_id"]).received_bytes == 0
    del beyond


def test_concurrent_uploads_of_one_content_id_never_interleave_corruptly(tmp_path):
    store = make_store(tmp_path / "blobs", max_uploads_per_scope=4)
    data = os.urandom(40)
    descriptor = descriptor_for(data, chunk_size=4)
    store.begin(SCOPE, descriptor)
    errors: list[BlobError] = []

    def writer(index: int) -> None:
        offset = 0
        try:
            while offset < len(data):
                store.write_chunk(SCOPE, descriptor["content_id"], offset, data[offset:offset + 4])
                offset += 4
        except BlobError as exc:  # a lost race is a typed refusal, not corruption
            errors.append(exc)

    threads = [threading.Thread(target=writer, args=(index,)) for index in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert errors and all(error.code == "failed_precondition" for error in errors)
    state = store.state(SCOPE, descriptor["content_id"])
    # Whoever won wrote every chunk exactly once; the store is either complete or
    # still missing a suffix, never a hole in the middle.
    assert state.verified_chunks == tuple(range(len(state.verified_chunks)))
    if state.received_bytes == len(data):
        assert store.commit(SCOPE, descriptor["content_id"]).state == "ready"


# -- authorization ---------------------------------------------------------
def test_another_scope_cannot_read_or_attach_even_with_the_content_id(tmp_path):
    store = make_store(tmp_path / "blobs")
    data = os.urandom(8)
    descriptor = upload(store, data, chunk_size=4)

    with pytest.raises(BlobError) as read:
        store.read_chunk(OTHER_SCOPE, descriptor["content_id"], 0, 4)
    assert read.value.code == "permission_denied"
    with pytest.raises(BlobError) as stream:
        list(store.iter_chunks(OTHER_SCOPE, descriptor["content_id"]))
    assert stream.value.code == "permission_denied"
    with pytest.raises(BlobError) as attach:
        store.attach(OTHER_SCOPE, descriptor["content_id"], "request-2")
    assert attach.value.code == "permission_denied"
    assert store.leases() == {}


def test_a_stale_attempt_cannot_finish_or_abort_another_attempts_upload(tmp_path):
    store = make_store(tmp_path / "blobs")
    data = os.urandom(8)
    descriptor = descriptor_for(data, chunk_size=4)
    owner = BlobAttempt(attempt_id="attempt-new", lease_fence=7)
    stale = BlobAttempt(attempt_id="attempt-old", lease_fence=3)
    store.begin(SCOPE, descriptor, attempt=owner)
    for offset in (0, 4):
        store.write_chunk(
            SCOPE, descriptor["content_id"], offset, data[offset:offset + 4], attempt=owner,
        )

    with pytest.raises(BlobError) as write:
        store.write_chunk(SCOPE, descriptor["content_id"], 4, data[4:8], attempt=stale)
    assert write.value.code == "permission_denied"
    with pytest.raises(BlobError) as commit:
        store.commit(SCOPE, descriptor["content_id"], attempt=stale)
    assert commit.value.code == "permission_denied"
    with pytest.raises(BlobError) as abort:
        store.abort(SCOPE, descriptor["content_id"], attempt=stale)
    assert abort.value.code == "permission_denied"
    assert store.state(SCOPE, descriptor["content_id"]).state == "uploading"
    assert store.commit(SCOPE, descriptor["content_id"], attempt=owner).state == "ready"


# -- expiry and cleanup ----------------------------------------------------
def test_an_expired_upload_is_refused_and_swept(tmp_path):
    clock = Clock()
    store = make_store(tmp_path / "blobs", clock=clock, partial_ttl_ms=1000)
    data = os.urandom(8)
    descriptor = descriptor_for(data, chunk_size=4)
    store.begin(SCOPE, descriptor)
    store.write_chunk(SCOPE, descriptor["content_id"], 0, data[0:4])

    clock.value = 2.0
    with pytest.raises(BlobError) as expired:
        store.write_chunk(SCOPE, descriptor["content_id"], 4, data[4:8])
    assert expired.value.code == "not_found"
    swept = store.sweep()
    assert [record["cleanup"] for record in swept] == ["done"]
    assert store.usage()["blobs"] == 0


def test_releasing_the_last_lease_cleans_up_and_a_failing_cleanup_quarantines(tmp_path):
    clock = Clock()
    io = FaultIO()
    store = make_store(tmp_path / "blobs", io=io, clock=clock)
    data = os.urandom(8)
    descriptor = upload(store, data, chunk_size=4)
    store.attach(SCOPE, descriptor["content_id"], "request-clean")

    records = store.release("request-clean")
    assert [record["cleanup"] for record in records] == ["done"]
    assert store.usage()["blobs"] == 0

    # Now make the delete fail: the blob must be quarantined, never "cleaned".
    descriptor = upload(store, data, chunk_size=4)
    store.attach(SCOPE, descriptor["content_id"], "request-quarantine")
    io.fail("remove", times=1)
    records = store.release("request-quarantine")
    assert records[0]["cleanup"] == "quarantine"
    assert records[0]["error"]
    state = store.state(SCOPE, descriptor["content_id"])
    assert state.state == "quarantine"
    assert state.cleanup["state"] == "quarantine"
    with pytest.raises(BlobError):
        store.read_chunk(SCOPE, descriptor["content_id"], 0, 4)


def test_release_keeps_a_blob_that_another_request_still_leases(tmp_path):
    store = make_store(tmp_path / "blobs")
    data = os.urandom(8)
    descriptor = upload(store, data, chunk_size=4)
    store.attach(SCOPE, descriptor["content_id"], "request-a")
    store.attach(SCOPE, descriptor["content_id"], "request-b")

    records = store.release("request-a")
    assert records[0]["remaining_leases"] == 1
    assert store.state(SCOPE, descriptor["content_id"]).state == "ready"
    store.release("request-b")
    assert store.usage()["blobs"] == 0


def test_reconcile_releases_only_requests_that_are_gone(tmp_path):
    store = make_store(tmp_path / "blobs")
    live = upload(store, os.urandom(8), chunk_size=4)
    orphan = upload(store, os.urandom(8), chunk_size=4)
    store.attach(SCOPE, live["content_id"], "request-live")
    store.attach(SCOPE, orphan["content_id"], "request-orphan")

    released = store.reconcile({"request-live"})
    assert list(released) == ["request-orphan"]
    assert store.state(SCOPE, live["content_id"]).state == "ready"
    assert store.leases() == {"request-live": [live["content_id"]]}


# -- trusted reader --------------------------------------------------------
def test_the_worker_reader_detects_a_corrupted_chunk(tmp_path):
    root = tmp_path / "blobs"
    store = make_store(root)
    data = os.urandom(8)
    descriptor = upload(store, data, chunk_size=4)

    chunk = next((root / "blobs").rglob("00000001.chunk"))
    chunk.write_bytes(b"XXXX")
    with pytest.raises(BlobError) as corrupt:
        list(BlobReader(root).iter_chunks(descriptor["content_id"]))
    assert corrupt.value.code == "unavailable"
