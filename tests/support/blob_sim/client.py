"""Chunk-at-a-time gRPC client for the P0b blob channel (test-only).

The client deliberately never holds a whole artifact to upload or download: it
sends one chunk and waits for its acknowledgement, and a download consumes one
response at a time.  Tests use it to drive the real BlobChannel service the same
way an agent would.
"""

from __future__ import annotations

import hashlib
from typing import Iterable, Iterator

import grpc

from tcenv.remote import contracts
from tcenv.remote.blob.v1 import blob_pb2 as bp
from tcenv.remote.blob.v1 import blob_pb2_grpc as bp_grpc


def descriptor_for(data: bytes, *, media_type: str, chunk_size: int) -> bp.BlobRef:
    digest = hashlib.sha256(data).hexdigest()
    return bp.BlobRef(
        content_id=contracts.blob_content_id(digest),
        sha256=digest,
        size_bytes=len(data),
        media_type=media_type,
        chunk_size=chunk_size,
    )


def _binding(binding: tuple[str, int] | None) -> bp.BlobBinding | None:
    if binding is None:
        return None
    attempt_id, lease_fence = binding
    return bp.BlobBinding(attempt_id=attempt_id, lease_fence=lease_fence)


class BlobClient:
    """A minimal typed client; every call carries the session grant."""

    def __init__(self, channel: grpc.Channel, token: str) -> None:
        self.stub = bp_grpc.BlobChannelStub(channel)
        self.token = token
        #: Largest single read/write the client ever performed, for the
        #: "no whole-file buffering" assertion.
        self.max_read = 0
        self.max_write = 0

    def _metadata(self) -> tuple[tuple[str, str], ...]:
        return (("authorization", f"Bearer {self.token}"),)

    def begin(self, ref: bp.BlobRef, *, binding: tuple[str, int] | None = None, timeout: float = 10):
        return self.stub.BeginUpload(
            bp.BeginUploadRequest(descriptor=ref, client_nonce="blob-test", binding=_binding(binding)),
            metadata=self._metadata(), timeout=timeout,
        )

    def write(self, content_id: str, offset: int, data: bytes, *, binding: tuple[str, int] | None = None, timeout: float = 10):
        self.max_write = max(self.max_write, len(data))
        return self.stub.WriteChunk(
            bp.WriteChunkRequest(
                content_id=content_id, offset=offset, data=data, binding=_binding(binding),
            ),
            metadata=self._metadata(), timeout=timeout,
        )

    def commit(self, content_id: str, *, binding: tuple[str, int] | None = None, timeout: float = 10):
        return self.stub.CommitUpload(
            bp.CommitUploadRequest(content_id=content_id, binding=_binding(binding)),
            metadata=self._metadata(), timeout=timeout,
        )

    def abort(self, content_id: str, *, binding: tuple[str, int] | None = None, timeout: float = 10):
        return self.stub.AbortUpload(
            bp.AbortUploadRequest(content_id=content_id, binding=_binding(binding)),
            metadata=self._metadata(), timeout=timeout,
        )

    def status(self, content_id: str, *, timeout: float = 10):
        return self.stub.GetBlob(
            bp.BlobRefRequest(content_id=content_id), metadata=self._metadata(), timeout=timeout,
        )

    def read(self, content_id: str, offset: int, length: int, *, timeout: float = 10) -> bytes:
        response = self.stub.ReadChunk(
            bp.ReadChunkRequest(content_id=content_id, offset=offset, length=length),
            metadata=self._metadata(), timeout=timeout,
        )
        self.max_read = max(self.max_read, len(response.data))
        return response.data

    def stream(self, content_id: str, *, offset: int = 0, timeout: float = 10) -> Iterator[bytes]:
        responses = self.stub.StreamBlob(
            bp.StreamBlobRequest(content_id=content_id, offset=offset),
            metadata=self._metadata(), timeout=timeout,
        )
        for response in responses:
            if response.data:
                self.max_read = max(self.max_read, len(response.data))
                yield response.data

    # -- composite helpers ------------------------------------------------
    def upload(
        self, data: bytes, *, chunk_size: int, media_type: str = "application/octet-stream",
        binding: tuple[str, int] | None = None, ref: bp.BlobRef | None = None,
    ) -> bp.BlobRef:
        """Upload in order, one acknowledged chunk at a time."""

        declared = ref if ref is not None else descriptor_for(data, media_type=media_type, chunk_size=chunk_size)
        self.begin(declared, binding=binding)
        offset = 0
        while offset < len(data):
            chunk = data[offset:offset + chunk_size]
            self.write(declared.content_id, offset, chunk, binding=binding)
            offset += len(chunk)
        return declared

    def download(self, content_id: str, *, offset: int = 0) -> bytes:
        collected = bytearray()
        for chunk in self.stream(content_id, offset=offset):
            collected += chunk
        return bytes(collected)

    def upload_and_commit(
        self, data: bytes, *, chunk_size: int, media_type: str = "application/octet-stream",
        binding: tuple[str, int] | None = None,
    ) -> bp.BlobRef:
        ref = self.upload(data, chunk_size=chunk_size, media_type=media_type, binding=binding)
        self.commit(ref.content_id, binding=binding)
        return ref
