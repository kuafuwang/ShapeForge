"""Authenticated gRPC adapter for the independent content-addressed blob channel.

The adapter is intentionally thin: authorize the per-RPC grant into a
:class:`~tcenv.remote.blob_channel.BlobScope`, map Protobuf to typed store calls,
and map store refusals to gRPC status codes.  It never accepts a path, a URL, a
shell command or vendor parameters, and it never reads a whole blob into memory:
uploads are chunk-acked and downloads are either one bounded range or a stream of
chunk-sized responses.

The six-RPC Evaluator service is untouched by this module.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Iterable, Protocol

import grpc

from .blob_channel import BLOB_STATES, BlobAttempt, BlobError, BlobScope, BlobState, BlobStore
from .job_types import JobError
from .blob.v1 import blob_pb2 as bp
from .blob.v1 import blob_pb2_grpc as bp_grpc

BLOB_STATES_PROTO = {
    "accepted": bp.BLOB_STATE_ACCEPTED,
    "uploading": bp.BLOB_STATE_UPLOADING,
    "ready": bp.BLOB_STATE_READY,
    "failed": bp.BLOB_STATE_FAILED,
    "quarantine": bp.BLOB_STATE_QUARANTINE,
}
_GRPC_CODES = {
    "unauthenticated": grpc.StatusCode.UNAUTHENTICATED,
    "not_found": grpc.StatusCode.NOT_FOUND,
    "permission_denied": grpc.StatusCode.PERMISSION_DENIED,
    "failed_precondition": grpc.StatusCode.FAILED_PRECONDITION,
    "resource_exhausted": grpc.StatusCode.RESOURCE_EXHAUSTED,
    "out_of_range": grpc.StatusCode.OUT_OF_RANGE,
    "unavailable": grpc.StatusCode.UNAVAILABLE,
}
_MAX_DETAIL = 512


class BlobAuthorizer(Protocol):
    """Resolve one raw session token into its blob scope.

    Implementations must raise on an unknown, expired or revoked grant; returning
    a scope is the authorization decision.  The production wiring resolves the
    grant through the durable job store, so blob access can never outlive the
    session grant that admitted the job.
    """

    def __call__(self, token: str) -> BlobScope: ...


def _bearer_token(values: Iterable[tuple[str, str]]) -> str:
    for key, value in values:
        if str(key).lower() != "authorization":
            continue
        scheme, _, token = str(value).partition(" ")
        if scheme.lower() != "bearer":
            raise BlobError("permission_denied", "authorization metadata must use the Bearer scheme")
        token = token.strip()
        if not token:
            raise BlobError("permission_denied", "session grant is required")
        if len(token.encode("utf-8")) > 4096:
            raise BlobError("permission_denied", "session grant is too large")
        return token
    raise BlobError("permission_denied", "session grant is required")


def blob_status(state: BlobState) -> bp.BlobStatus:
    """Return the wire view of one blob; scope internals are never exposed."""

    return bp.BlobStatus(
        descriptor=bp.BlobRef(**state.descriptor),
        state=BLOB_STATES_PROTO[state.state],
        received_bytes=state.received_bytes,
        verified_chunks=list(state.verified_chunks),
        request_ids=list(state.leases),
        expires_at_ms=state.expires_at_ms,
        cleanup_state=str(state.cleanup.get("state") or ""),
        quarantine_reason="" if state.quarantine is None else str(state.quarantine.get("reason") or "")[:512],
    )


def _binding(message: Any) -> BlobAttempt | None:
    if message is None or not message.attempt_id:
        return None
    return BlobAttempt(attempt_id=message.attempt_id, lease_fence=int(message.lease_fence))


def _descriptor(message: Any) -> dict[str, Any]:
    return {
        "content_id": message.content_id,
        "sha256": message.sha256,
        "size_bytes": int(message.size_bytes),
        "media_type": message.media_type,
        "chunk_size": int(message.chunk_size),
    }


class BlobChannelService(bp_grpc.BlobChannelServicer):
    """One authenticated blob data plane over one single-writer store."""

    def __init__(
        self,
        store: BlobStore,
        authorize: BlobAuthorizer,
        *,
        require_client_identity: bool = False,
        logger: logging.Logger | None = None,
    ) -> None:
        if not isinstance(store, BlobStore):
            raise ValueError("the blob channel requires a typed BlobStore")
        if not callable(authorize):
            raise ValueError("the blob channel requires a grant authorizer")
        self.store = store
        self.authorize = authorize
        self.require_client_identity = bool(require_client_identity)
        self.logger = logger or logging.getLogger("tcenv.remote.blob")

    # -- helpers -----------------------------------------------------------
    def _scope(self, context: grpc.ServicerContext) -> BlobScope:
        token = _bearer_token(context.invocation_metadata())
        try:
            scope = self.authorize(token)
        except JobError as exc:
            # The durable grant check classifies its own refusal; keep it instead
            # of degrading every failure (expired, revoked, unknown) to one code.
            raise BlobError(exc.code or "permission_denied", str(exc)) from exc
        if not isinstance(scope, BlobScope):
            raise BlobError("permission_denied", "authorizer returned a non-scope value")
        return scope

    def _abort(self, context: grpc.ServicerContext, error: BlobError) -> None:
        context.abort(_GRPC_CODES.get(error.code, grpc.StatusCode.INTERNAL), str(error.detail)[:_MAX_DETAIL])

    # -- RPCs --------------------------------------------------------------
    def BeginUpload(self, request: bp.BeginUploadRequest, context: grpc.ServicerContext) -> bp.BlobStatus:
        try:
            scope = self._scope(context)
            state = self.store.begin(scope, _descriptor(request.descriptor), attempt=_binding(request.binding))
            return blob_status(state)
        except BlobError as exc:
            self._abort(context, exc)
            raise  # pragma: no cover - abort raises

    def WriteChunk(self, request: bp.WriteChunkRequest, context: grpc.ServicerContext) -> bp.BlobStatus:
        try:
            scope = self._scope(context)
            state = self.store.write_chunk(
                scope, request.content_id, int(request.offset), request.data,
                attempt=_binding(request.binding),
            )
            return blob_status(state)
        except BlobError as exc:
            self._abort(context, exc)
            raise  # pragma: no cover - abort raises

    def CommitUpload(self, request: bp.CommitUploadRequest, context: grpc.ServicerContext) -> bp.BlobStatus:
        try:
            scope = self._scope(context)
            state = self.store.commit(scope, request.content_id, attempt=_binding(request.binding))
            return blob_status(state)
        except BlobError as exc:
            self._abort(context, exc)
            raise  # pragma: no cover - abort raises

    def AbortUpload(self, request: bp.AbortUploadRequest, context: grpc.ServicerContext) -> bp.BlobStatus:
        try:
            scope = self._scope(context)
            state = self.store.abort(scope, request.content_id, attempt=_binding(request.binding))
            return blob_status(state)
        except BlobError as exc:
            self._abort(context, exc)
            raise  # pragma: no cover - abort raises

    def GetBlob(self, request: bp.BlobRefRequest, context: grpc.ServicerContext) -> bp.BlobStatus:
        try:
            scope = self._scope(context)
            return blob_status(self.store.state(scope, request.content_id))
        except BlobError as exc:
            self._abort(context, exc)
            raise  # pragma: no cover - abort raises

    def ReadChunk(self, request: bp.ReadChunkRequest, context: grpc.ServicerContext) -> bp.ChunkData:
        try:
            scope = self._scope(context)
            data = self.store.read_chunk(
                scope, request.content_id, int(request.offset), int(request.length),
            )
            return bp.ChunkData(data=data, offset=int(request.offset), done=False)
        except BlobError as exc:
            self._abort(context, exc)
            raise  # pragma: no cover - abort raises

    def StreamBlob(self, request: bp.StreamBlobRequest, context: grpc.ServicerContext):
        try:
            scope = self._scope(context)
            iterator = self.store.iter_chunks(scope, request.content_id, offset=int(request.offset))
        except BlobError as exc:
            self._abort(context, exc)
            return
        offset = int(request.offset)
        for index, chunk in enumerate(iterator):
            # One chunk per response: the client's flow control is the backpressure.
            yield bp.ChunkData(data=chunk, offset=offset, done=False)
            offset += len(chunk)
            del index
        yield bp.ChunkData(data=b"", offset=offset, done=True)


class StoreAuthorizer:
    """Resolve a grant through a callable that returns the grant's durable scope.

    ``scope_of(token)`` must raise on an unknown/expired/revoked grant.  The
    deployment identity is supplied here, so a token can never carry its own
    site/device claim into the blob store.
    """

    def __init__(
        self, scope_of: Callable[[str], dict[str, str]], *, site_id: str, device_id: str,
    ) -> None:
        self._scope_of = scope_of
        self.site_id = site_id
        self.device_id = device_id

    def __call__(self, token: str) -> BlobScope:
        scope = self._scope_of(token)
        try:
            return BlobScope(
                site_id=self.site_id,
                device_id=self.device_id,
                target_id=scope["target_id"],
                subject=scope["subject"],
                run_id=scope["run_id"],
                session_id=scope["session_id"],
            )
        except (KeyError, TypeError, BlobError) as exc:
            raise BlobError("permission_denied", f"grant scope is unusable: {exc}") from exc


def build_blob_service(
    store: BlobStore, authorize: BlobAuthorizer, *, require_client_identity: bool = False,
    logger: logging.Logger | None = None,
) -> BlobChannelService:
    """Return the servicer a composition root registers (production or test)."""

    return BlobChannelService(
        store, authorize, require_client_identity=require_client_identity, logger=logger,
    )


def blob_states() -> frozenset[str]:
    """Return the store's valid state names (adapter self-check helpers)."""

    return BLOB_STATES


def blob_root(store: BlobStore) -> Path:
    return store.root
