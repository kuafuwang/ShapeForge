"""P0b BlobChannel over real gRPC: chunking, streaming, auth and refusals.

The service is the production ``tcenv.remote.blob_service`` on the same listener
as the six-RPC Evaluator, but it is a different proto package and service.  Each
call carries the same per-RPC bearer grant; the store is the real one.
"""

from __future__ import annotations

import hashlib
import os
import secrets
import time

import grpc
import pytest

from tcenv.remote import contracts, grpc_contract as gc
from tcenv.remote.blob.v1 import blob_pb2 as bp
from tcenv.remote.blob_channel import BlobLimits
from tests.support.blob_sim import BlobClient, descriptor_for
from tests.support.job_engine import NPU_TOKEN, npu_grant
from tests.support.vendor_sim.harness import BlobNpuGrpcHarness

MEDIA = "text/plain; charset=utf-8"


class Frontend:
    def __init__(self, root, **kwargs) -> None:
        self.harness = BlobNpuGrpcHarness(root, **kwargs)
        self.channel = grpc.insecure_channel(self.harness.server.address)
        self.client = BlobClient(self.channel, NPU_TOKEN)

    def close(self) -> None:
        self.channel.close()
        self.harness.close()


@pytest.fixture()
def frontend(tmp_path):
    fe = Frontend(tmp_path / "blob-grpc")
    try:
        yield fe
    finally:
        fe.close()


def _status(exc_info) -> grpc.StatusCode:
    return exc_info.value.code()


def test_upload_commit_and_stream_are_byte_identical_multi_chunk(frontend):
    data = os.urandom(9) + b"\x00\xff"
    ref = frontend.client.upload_and_commit(data, chunk_size=4, media_type=MEDIA)

    state = frontend.client.status(ref.content_id)
    assert state.state == bp.BLOB_STATE_READY
    assert state.received_bytes == len(data)
    assert state.descriptor == ref
    assert frontend.client.download(ref.content_id) == data
    assert frontend.client.download(ref.content_id, offset=4) == data[4:]
    assert frontend.client.read(ref.content_id, 2, 4) == data[2:6]
    # One chunk per RPC and one chunk per streamed response: no whole-file call.
    assert frontend.client.max_write <= 4
    assert frontend.client.max_read <= 4


def test_empty_file_round_trips_and_is_marked_ready(frontend):
    ref = frontend.client.upload_and_commit(b"", chunk_size=4, media_type=MEDIA)
    assert ref.size_bytes == 0
    assert frontend.client.status(ref.content_id).state == bp.BLOB_STATE_READY
    assert frontend.client.download(ref.content_id) == b""
    with pytest.raises(grpc.RpcError) as read:
        frontend.client.read(ref.content_id, 0, 1)
    assert _status(read) == grpc.StatusCode.OUT_OF_RANGE


def test_a_missing_grant_is_refused_before_any_store_write(frontend):
    data = os.urandom(8)
    ref = descriptor_for(data, chunk_size=4, media_type=MEDIA)
    anonymous = BlobClient(frontend.channel, "")
    with pytest.raises(grpc.RpcError) as denied:
        anonymous.begin(ref)
    assert _status(denied) == grpc.StatusCode.UNAUTHENTICATED
    unknown = BlobClient(frontend.channel, "not-a-registered-token")
    with pytest.raises(grpc.RpcError) as refused:
        unknown.begin(ref)
    assert _status(refused) == grpc.StatusCode.UNAUTHENTICATED
    assert frontend.harness.blob_store.usage()["blobs"] == 0


def test_an_expired_grant_cannot_upload_or_read(tmp_path):
    expired = npu_grant(expires_at=time.time() - 3600)
    fe = Frontend(tmp_path / "expired", grants=[expired])
    try:
        client = BlobClient(fe.channel, expired.token)
        data = os.urandom(8)
        with pytest.raises(grpc.RpcError) as denied:
            client.begin(descriptor_for(data, chunk_size=4, media_type=MEDIA))
        assert _status(denied) == grpc.StatusCode.UNAUTHENTICATED
        assert fe.harness.blob_store.usage()["blobs"] == 0
    finally:
        fe.close()


def test_a_revoked_grant_loses_access_to_an_already_uploaded_blob(frontend):
    data = os.urandom(8)
    ref = frontend.client.upload_and_commit(data, chunk_size=4, media_type=MEDIA)
    shared = frontend.client.upload_and_commit(os.urandom(8), chunk_size=4, media_type=MEDIA)

    frontend.harness.engine.revoke_grant(NPU_TOKEN)
    with pytest.raises(grpc.RpcError) as read:
        frontend.client.read(shared.content_id, 0, 4)
    assert _status(read) == grpc.StatusCode.PERMISSION_DENIED
    with pytest.raises(grpc.RpcError) as stream:
        list(frontend.client.stream(ref.content_id))
    assert _status(stream) == grpc.StatusCode.PERMISSION_DENIED


def test_another_scope_cannot_read_the_content_id(tmp_path):
    owner = npu_grant()
    other = npu_grant(token="other-grant-token", subject="other-subject", session_id="other-session")
    fe = Frontend(tmp_path / "scopes", grants=[owner, other])
    try:
        data = os.urandom(8)
        mine = BlobClient(fe.channel, owner.token)
        ref = mine.upload_and_commit(data, chunk_size=4, media_type=MEDIA)
        theirs = BlobClient(fe.channel, other.token)
        with pytest.raises(grpc.RpcError) as denied:
            theirs.read(ref.content_id, 0, 4)
        assert _status(denied) == grpc.StatusCode.PERMISSION_DENIED
        with pytest.raises(grpc.RpcError) as begin:
            theirs.begin(ref)
        assert _status(begin) == grpc.StatusCode.PERMISSION_DENIED
    finally:
        fe.close()


def test_ranges_chunks_and_offsets_are_refused_outside_the_declaration(frontend):
    data = os.urandom(10)
    ref = frontend.client.upload_and_commit(data, chunk_size=4, media_type=MEDIA)

    with pytest.raises(grpc.RpcError) as beyond:
        frontend.client.read(ref.content_id, 8, 4)
    assert _status(beyond) == grpc.StatusCode.OUT_OF_RANGE
    with pytest.raises(grpc.RpcError) as wide:
        frontend.client.read(ref.content_id, 0, 5)
    assert _status(wide) == grpc.StatusCode.OUT_OF_RANGE

    payload = os.urandom(10)
    second = descriptor_for(payload, chunk_size=4, media_type=MEDIA)
    frontend.client.begin(second)
    with pytest.raises(grpc.RpcError) as ahead:
        frontend.client.write(second.content_id, 4, payload[4:8])
    assert _status(ahead) == grpc.StatusCode.FAILED_PRECONDITION
    with pytest.raises(grpc.RpcError) as oversized:
        frontend.client.write(second.content_id, 0, payload[0:5])
    assert _status(oversized) == grpc.StatusCode.RESOURCE_EXHAUSTED
    with pytest.raises(grpc.RpcError) as tail:
        frontend.client.write(second.content_id, 0, payload[0:2])
    assert _status(tail) == grpc.StatusCode.FAILED_PRECONDITION
    with pytest.raises(grpc.RpcError) as beyond:
        frontend.client.write(second.content_id, 8, payload[8:12])
    assert _status(beyond) == grpc.StatusCode.FAILED_PRECONDITION
    assert frontend.client.status(second.content_id).received_bytes == 0


def test_an_interrupted_upload_resumes_from_the_acknowledged_offset(frontend):
    data = os.urandom(13)
    ref = descriptor_for(data, chunk_size=4, media_type=MEDIA)
    state = frontend.client.begin(ref)
    frontend.client.write(ref.content_id, 0, data[0:4])
    state = frontend.client.write(ref.content_id, 4, data[4:8])
    assert state.received_bytes == 8
    assert list(state.verified_chunks) == [0, 1]

    # Reconnecting (a new client, a new channel) resumes from the durable offset.
    resumed = BlobClient(grpc.insecure_channel(frontend.harness.server.address), NPU_TOKEN)
    state = resumed.begin(ref)
    assert state.received_bytes == 8
    for offset in (8, 12):
        state = resumed.write(ref.content_id, offset, data[offset:offset + 4])
    assert state.received_bytes == len(data)
    assert resumed.commit(ref.content_id).state == bp.BLOB_STATE_READY
    assert resumed.download(ref.content_id) == data


def test_duplicate_chunk_and_commit_are_idempotent(frontend):
    data = os.urandom(8)
    ref = descriptor_for(data, chunk_size=4, media_type=MEDIA)
    frontend.client.begin(ref)
    frontend.client.write(ref.content_id, 0, data[0:4])
    repeat = frontend.client.write(ref.content_id, 0, data[0:4])
    assert repeat.received_bytes == 4
    frontend.client.write(ref.content_id, 4, data[4:8])
    assert frontend.client.commit(ref.content_id).state == bp.BLOB_STATE_READY
    assert frontend.client.commit(ref.content_id).state == bp.BLOB_STATE_READY
    with pytest.raises(grpc.RpcError) as after_ready:
        frontend.client.write(ref.content_id, 0, data[0:4])
    assert _status(after_ready) == grpc.StatusCode.FAILED_PRECONDITION


def test_begin_upload_reserves_the_declared_size_in_the_scope_quota(tmp_path):
    limits = BlobLimits(
        max_size_bytes=64, min_chunk_size=1, max_chunk_size=8, max_chunks=64,
        max_scope_bytes=12, max_total_bytes=12, max_uploads_per_scope=2,
    )
    fe = Frontend(tmp_path / "reserve-scope", limits=limits)
    try:
        first = fe.client.begin(descriptor_for(os.urandom(8), chunk_size=4, media_type=MEDIA))
        assert first.state == bp.BLOB_STATE_ACCEPTED
        # Nothing has been written yet, but the declaration already reserves its
        # bytes: a second 8-byte declaration cannot slip under the 12-byte quota.
        with pytest.raises(grpc.RpcError) as reserved:
            fe.client.begin(descriptor_for(os.urandom(8), chunk_size=4, media_type=MEDIA))
        assert _status(reserved) == grpc.StatusCode.RESOURCE_EXHAUSTED
        assert fe.harness.blob_store.usage()["reserved_bytes"] == 8
        assert fe.harness.blob_store.usage()["ready_bytes"] == 0
    finally:
        fe.close()


def test_begin_upload_reserves_the_declared_size_in_the_global_quota(tmp_path):
    owner = npu_grant()
    other = npu_grant(token="reserve-other", subject="reserve-other", session_id="reserve-other")
    limits = BlobLimits(
        max_size_bytes=64, min_chunk_size=1, max_chunk_size=8, max_chunks=64,
        max_scope_bytes=64, max_total_bytes=12, max_uploads_per_scope=2,
    )
    fe = Frontend(tmp_path / "reserve-total", grants=[owner, other], limits=limits)
    try:
        BlobClient(fe.channel, owner.token).begin(
            descriptor_for(os.urandom(8), chunk_size=4, media_type=MEDIA),
        )
        with pytest.raises(grpc.RpcError) as reserved:
            BlobClient(fe.channel, other.token).begin(
                descriptor_for(os.urandom(8), chunk_size=4, media_type=MEDIA),
            )
        assert _status(reserved) == grpc.StatusCode.RESOURCE_EXHAUSTED
    finally:
        fe.close()


def test_replaying_an_older_chunk_is_idempotent_and_off_boundary_is_refused(frontend):
    data = os.urandom(12)
    ref = descriptor_for(data, chunk_size=4, media_type=MEDIA)
    frontend.client.begin(ref)
    for offset in (0, 4, 8):
        frontend.client.write(ref.content_id, offset, data[offset:offset + 4])
    assert frontend.client.status(ref.content_id).received_bytes == 12

    # Chunk 0 and chunk 1 are still replayable after later chunks were acked.
    for offset in (0, 4):
        replay = frontend.client.write(ref.content_id, offset, data[offset:offset + 4])
        assert replay.received_bytes == 12
        assert list(replay.verified_chunks) == [0, 1, 2]

    with pytest.raises(grpc.RpcError) as boundary:
        frontend.client.write(ref.content_id, 2, data[2:6])
    assert _status(boundary) == grpc.StatusCode.FAILED_PRECONDITION
    with pytest.raises(grpc.RpcError) as different:
        frontend.client.write(ref.content_id, 0, b"ZZZZ")
    assert _status(different) == grpc.StatusCode.FAILED_PRECONDITION
    assert frontend.client.commit(ref.content_id).state == bp.BLOB_STATE_READY
    assert frontend.client.download(ref.content_id) == data


def test_a_digest_mismatch_fails_the_blob_closed(frontend):
    data = os.urandom(8)
    wrong_digest = hashlib.sha256(b"other").hexdigest()
    ref = bp.BlobRef(
        content_id=contracts.blob_content_id(wrong_digest), sha256=wrong_digest,
        size_bytes=len(data), media_type=MEDIA, chunk_size=4,
    )
    frontend.client.upload(data, chunk_size=4, media_type=MEDIA, ref=ref)
    with pytest.raises(grpc.RpcError) as mismatch:
        frontend.client.commit(ref.content_id)
    assert _status(mismatch) == grpc.StatusCode.FAILED_PRECONDITION
    state = frontend.client.status(ref.content_id)
    assert state.state == bp.BLOB_STATE_FAILED
    with pytest.raises(grpc.RpcError) as unreadable:
        frontend.client.read(ref.content_id, 0, 4)
    assert _status(unreadable) == grpc.StatusCode.FAILED_PRECONDITION


def test_the_same_content_id_with_another_size_is_a_conflict(frontend):
    data = os.urandom(8)
    ref = frontend.client.upload_and_commit(data, chunk_size=4, media_type=MEDIA)
    tampered = bp.BlobRef(
        content_id=ref.content_id, sha256=ref.sha256, size_bytes=9,
        media_type=MEDIA, chunk_size=4,
    )
    with pytest.raises(grpc.RpcError) as conflict:
        frontend.client.begin(tampered)
    assert _status(conflict) == grpc.StatusCode.FAILED_PRECONDITION


def test_abort_frees_the_upload_and_keeps_it_unusable(frontend):
    data = os.urandom(8)
    ref = descriptor_for(data, chunk_size=4, media_type=MEDIA)
    frontend.client.begin(ref)
    frontend.client.write(ref.content_id, 0, data[0:4])
    aborted = frontend.client.abort(ref.content_id)
    assert aborted.state == bp.BLOB_STATE_FAILED
    assert frontend.harness.blob_store.usage()["partial_bytes"] == 0
    with pytest.raises(grpc.RpcError) as read:
        frontend.client.read(ref.content_id, 0, 4)
    assert _status(read) == grpc.StatusCode.FAILED_PRECONDITION


def test_a_slow_consumer_does_not_make_the_server_prefetch_the_artifact(tmp_path):
    from tests.support.blob_sim.faults import FaultIO

    io = FaultIO()
    fe = Frontend(tmp_path / "slow", io=io)
    try:
        data = os.urandom(64)
        ref = fe.client.upload_and_commit(data, chunk_size=8, media_type=MEDIA)
        stream = fe.client.stream(ref.content_id)
        first = next(stream)
        # One response carries at most one declared chunk, even though the slow
        # consumer is not draining the stream.
        assert len(first) <= 8
        rest = bytearray(first)
        for chunk in stream:
            assert len(chunk) <= 8
            rest += chunk
        assert bytes(rest) == data
        # No single read ever asked for the whole artifact: the server reads one
        # chunk (or a small manifest record) at a time.
        chunk_reads = [size for name, size in io.read_log if name.endswith(".chunk")]
        assert chunk_reads and max(chunk_reads) <= 9
    finally:
        fe.close()


def test_the_channel_messages_carry_no_path_url_or_command(tmp_path):
    forbidden = {"path", "url", "uri", "command", "shell", "mount", "argv", "host_path", "bdf"}
    for name in ("BeginUploadRequest", "WriteChunkRequest", "CommitUploadRequest",
                 "AbortUploadRequest", "BlobRefRequest", "ReadChunkRequest", "StreamBlobRequest"):
        fields = {field.name for field in bp.DESCRIPTOR.message_types_by_name[name].fields}
        assert not fields & forbidden, (name, sorted(fields & forbidden))
    blob_fields = {field.name for field in bp.DESCRIPTOR.message_types_by_name["BlobRef"].fields}
    assert blob_fields == {"content_id", "sha256", "size_bytes", "media_type", "chunk_size"}


def test_the_blob_service_is_separate_from_the_six_rpc_evaluator(frontend):
    from tcenv.remote.grpc_server import EvaluatorService

    methods = sorted(
        name for name, value in vars(EvaluatorService).items()
        if not name.startswith("_") and callable(value)
    )
    assert methods == sorted(["GetCapabilities", "Submit", "GetJob", "Watch", "Cancel", "Retry"])
    blob_methods = {method.name for method in bp.DESCRIPTOR.services_by_name["BlobChannel"].methods}
    assert blob_methods == {
        "AbortUpload", "BeginUpload", "CommitUpload", "GetBlob", "ReadChunk",
        "StreamBlob", "WriteChunk",
    }
    # The Evaluator proto still has exactly the six RPCs and the legacy field numbers.
    from tcenv.remote.v1 import evaluator_pb2 as pb

    request_fields = {field.number: field.name for field in pb.EvaluationRequest.DESCRIPTOR.fields}
    assert request_fields[23] == "candidate_digest"
    assert request_fields[24] == "execution"
    assert request_fields[25] == "blob_ref"
    assert request_fields[10] == "candidate_sha256"
    assert request_fields[11] == "source"
    assert gc.blob_ref_to_proto.__name__ == "blob_ref_to_proto"
