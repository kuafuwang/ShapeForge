"""Test-only composition roots for the P0 / P0b NPU gRPC loops."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from tcenv.remote import contracts
from tcenv.remote.blob_channel import BlobLimits, BlobStore
from tcenv.remote.blob_service import StoreAuthorizer, build_blob_service
from tcenv.remote.grpc_client import GrpcEvaluatorFactory, GrpcExpected, SignatureKeyring
from tcenv.remote.grpc_server import GrpcServerConfig, build_server
from tests.support.job_engine import (
    NPU_FEATURES,
    NPU_TARGET,
    NPU_TOKEN,
    npu_engine,
    npu_features,
    npu_identity,
)
from tests.support.vendor_sim.vendor_a import FakeVendorAController

NPU_SIGNING_KEY = b"n" * 32
NPU_SIGNING_KEY_ID = "sim-npu-site/key-1"
#: Host-side HMAC over the retained result envelope.  The NPU contract refuses
#: an outbox without it, so a retained envelope can never be replayed without
#: authenticating every field (report, evidence, provenance, receipt).
NPU_HOST_SIGNATURE_KEY = b"host-envelope-key-32-bytes-long!!"


class NpuGrpcHarness:
    def __init__(
        self, root: Path, *, ledger: Path | None = None,
        controller: FakeVendorAController | None = None,
        worker_env: dict[str, str] | None = None,
        release_supported: bool = True,
        signing_key: bytes | None = NPU_SIGNING_KEY,
        required_features: tuple[str, ...] = NPU_FEATURES,
        token: str = NPU_TOKEN,
        generation: str = "1",
    ) -> None:
        self.root = root
        self.ledger_path = ledger or (root / "vendor-ledger.json")
        self.controller = controller or FakeVendorAController(
            self.ledger_path, release_supported=release_supported, generation=generation,
        )
        self.key = signing_key
        keyring = {} if signing_key is None else {NPU_SIGNING_KEY_ID: signing_key}
        self.engine = npu_engine(
            root / "state", ledger_path=self.ledger_path, controller=self.controller,
            worker_env=worker_env, signing_key=signing_key,
            signing_key_id=NPU_SIGNING_KEY_ID if signing_key is not None else "",
            signing_keyring=keyring, generation=generation,
        )
        self.server = build_server(
            self.engine,
            config=GrpcServerConfig(
                address="127.0.0.1:0", allow_insecure=True,
                require_client_identity=False, watch_poll_s=0.02,
            ),
        )
        identity = npu_identity()
        self.factory = GrpcEvaluatorFactory(
            endpoint=self.server.address,
            token_provider=lambda _run, _session: token,
            run_id="npu-run-1",
            target_id=NPU_TARGET,
            expected=GrpcExpected(
                site_id=identity.site_id, device_id=identity.device_id,
                target_id=NPU_TARGET, worker_build=identity.worker_build,
                task_pack_digest=identity.task_pack_digest,
            ),
            api_contract_ids=(contracts.NPU_SOURCE_API_CONTRACT_ID,),
            evidence_contract_ids=(contracts.NPU_SOURCE_EVIDENCE_CONTRACT_ID,),
            required_features=required_features,
            keyring=SignatureKeyring(keyring),
            legacy_signature_key=NPU_HOST_SIGNATURE_KEY,
            allow_insecure=True,
            watch_slice_s=1.0,
            job_margin_s=6.0,
        )

    def close(self) -> None:
        self.factory.close()
        self.server.stop(grace_s=0.2)
        self.engine.close(wait=5.0)

    def __enter__(self) -> "NpuGrpcHarness":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


class BlobNpuGrpcHarness(NpuGrpcHarness):
    """P0b composition root: the NPU blob contract plus the real BlobChannel.

    Both services share one listener but not one contract: the blob channel is a
    separate proto package and never reuses an Evaluator message.
    """

    def __init__(
        self, root: Path, *, limits: BlobLimits | None = None,
        ledger: Path | None = None, controller: Any | None = None,
        worker_env: dict[str, str] | None = None,
        release_supported: bool = True,
        generation: str = "1",
        signing_key: bytes | None = NPU_SIGNING_KEY,
        required_features: tuple[str, ...] | None = None,
        token: str = NPU_TOKEN,
        grants: list | None = None,
        authorize: Any | None = None,
        io: Any | None = None,
        **engine_options: Any,
    ) -> None:
        from tcenv.remote.grpc_client import GrpcEvaluatorFactory, GrpcExpected, SignatureKeyring
        from tcenv.remote.grpc_server import GrpcServerConfig, build_server
        from tests.support.vendor_sim.vendor_a import FakeVendorAController

        self.root = root
        self.ledger_path = ledger or (root / "vendor-ledger.json")
        self.controller = controller or FakeVendorAController(
            self.ledger_path, release_supported=release_supported, generation=generation,
        )
        self.key = signing_key
        keyring = {} if signing_key is None else {NPU_SIGNING_KEY_ID: signing_key}
        self.blob_store = BlobStore(root / "state" / "blobs", limits=limits or BlobLimits(
            max_size_bytes=1 << 20, max_chunk_size=4096, min_chunk_size=1,
            max_chunks=4096, max_uploads_per_scope=2,
        ), io=io)
        #: Set by the test after uploading the candidate this run should submit.
        self.blob_ref: Any | None = None
        engine_kwargs: dict[str, Any] = {}
        if grants is not None:
            engine_kwargs["grants"] = grants
        self.engine = npu_engine(
            root / "state", ledger_path=self.ledger_path, controller=self.controller,
            worker_env=worker_env, signing_key=signing_key,
            signing_key_id=NPU_SIGNING_KEY_ID if signing_key is not None else "",
            signing_keyring=keyring, generation=generation,
            contract=contracts.NPU_BLOB_DIALECT, blob_store=self.blob_store,
            **engine_kwargs, **engine_options,
        )
        identity = npu_identity()
        authorizer = authorize or StoreAuthorizer(
            self.engine.store.blob_scope, site_id=identity.site_id, device_id=identity.device_id,
        )
        self.blob_service = build_blob_service(self.blob_store, authorizer)
        self.server = build_server(
            self.engine,
            config=GrpcServerConfig(
                address="127.0.0.1:0", allow_insecure=True,
                require_client_identity=False, watch_poll_s=0.02,
            ),
            blob_channel=self.blob_service,
        )
        self.factory = GrpcEvaluatorFactory(
            endpoint=self.server.address,
            token_provider=lambda _run, _session: token,
            run_id="npu-run-1",
            target_id=NPU_TARGET,
            expected=GrpcExpected(
                site_id=identity.site_id, device_id=identity.device_id,
                target_id=NPU_TARGET, worker_build=identity.worker_build,
                task_pack_digest=identity.task_pack_digest,
            ),
            api_contract_ids=(contracts.NPU_BLOB_API_CONTRACT_ID,),
            evidence_contract_ids=(contracts.NPU_BLOB_EVIDENCE_CONTRACT_ID,),
            required_features=required_features or npu_features(contracts.NPU_BLOB_DIALECT),
            keyring=SignatureKeyring(keyring),
            legacy_signature_key=NPU_HOST_SIGNATURE_KEY,
            allow_insecure=True,
            watch_slice_s=1.0,
            job_margin_s=6.0,
            blob_descriptor=lambda _request: self.blob_ref,
        )

    # -- store-level helpers for engine (non-gRPC) tests -------------------
    def scope(self, token: str = NPU_TOKEN):
        """Return the blob scope the frontend derives for this grant."""

        from tcenv.remote.blob_channel import BlobScope

        grant_scope = self.engine.store.blob_scope(token)
        identity = npu_identity()
        return BlobScope(
            site_id=identity.site_id, device_id=identity.device_id,
            target_id=grant_scope["target_id"], subject=grant_scope["subject"],
            run_id=grant_scope["run_id"], session_id=grant_scope["session_id"],
        )

    def upload(
        self, data: bytes, *, chunk_size: int = 7,
        media_type: str = "text/plain; charset=utf-8", commit: bool = True,
        scope=None,
    ):
        """Upload through the store (engine-level tests); returns the descriptor."""

        from tcenv.remote.job_types import BlobDescriptor

        descriptor = BlobDescriptor.for_bytes(data, media_type=media_type, chunk_size=chunk_size)
        scope = scope or self.scope()
        self.blob_store.begin(scope, descriptor.to_dict())
        offset = 0
        while offset < len(data):
            chunk = data[offset:offset + chunk_size]
            self.blob_store.write_chunk(scope, descriptor.content_id, offset, chunk)
            offset += len(chunk)
        if commit:
            self.blob_store.commit(scope, descriptor.content_id)
        return descriptor
