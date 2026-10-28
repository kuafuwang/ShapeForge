"""Test-only composition roots for the P0 / P0b NPU gRPC loops."""

from __future__ import annotations

import hashlib
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
    npu_bundle_engine,
    npu_bundle_features,
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


class _BundleHarnessBase:
    """Shared P1 wiring: the bundle contract over one fake vendor."""

    vendor = ""

    def close(self) -> None:
        self.factory.close()
        self.server.stop(grace_s=0.2)
        self.engine.close(wait=5.0)

    def __enter__(self) -> "_BundleHarnessBase":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


class BundleSourceGrpcHarness(_BundleHarnessBase):
    """Vendor A (source -> ELF/SIMT) served under the version 4 contract.

    The same contract version that carries an artifact bundle also carries the
    inline source branch, so the source/bundle oneof is exercised by two
    genuinely different vendors rather than by one vendor with two payloads.
    """

    vendor = "vendor-a"

    def __init__(
        self, root: Path, *, ledger: Path | None = None,
        controller: Any | None = None, worker_env: dict[str, str] | None = None,
        release_supported: bool = True, generation: str = "1",
        signing_key: bytes | None = NPU_SIGNING_KEY,
        required_features: tuple[str, ...] | None = None,
        token: str = NPU_TOKEN,
        **engine_options: Any,
    ) -> None:
        from tests.support.vendor_sim.vendor_a import FakeVendorAController

        self.root = root
        self.ledger_path = ledger or (root / "vendor-ledger.json")
        self.controller = controller or FakeVendorAController(
            self.ledger_path, release_supported=release_supported, generation=generation,
        )
        keyring = {} if signing_key is None else {NPU_SIGNING_KEY_ID: signing_key}
        self.engine = npu_bundle_engine(
            root / "state", service_path=self.ledger_path, controller=self.controller,
            vendor="vendor-a", worker_env=worker_env, signing_key=signing_key,
            signing_key_id=NPU_SIGNING_KEY_ID if signing_key is not None else "",
            signing_keyring=keyring, generation=generation, **engine_options,
        )
        self.server = build_server(
            self.engine,
            config=GrpcServerConfig(
                address="127.0.0.1:0", allow_insecure=True,
                require_client_identity=False, watch_poll_s=0.02,
            ),
        )
        identity = self.engine.identity
        self.factory = GrpcEvaluatorFactory(
            endpoint=self.server.address,
            token_provider=lambda _run, _session: token,
            run_id="npu-run-1",
            target_id=identity.target_id,
            expected=GrpcExpected(
                site_id=identity.site_id, device_id=identity.device_id,
                target_id=identity.target_id, worker_build=identity.worker_build,
                task_pack_digest=identity.task_pack_digest,
            ),
            api_contract_ids=(contracts.NPU_BUNDLE_API_CONTRACT_ID,),
            evidence_contract_ids=(contracts.NPU_BUNDLE_EVIDENCE_CONTRACT_ID,),
            required_features=required_features or npu_bundle_features(),
            keyring=SignatureKeyring(keyring),
            legacy_signature_key=NPU_HOST_SIGNATURE_KEY,
            allow_insecure=True,
            watch_slice_s=1.0,
            job_margin_s=6.0,
        )


class GraphGrpcHarness(_BundleHarnessBase):
    """Vendor B: a graph artifact set executed by an asynchronous service.

    It optionally carries the P0b blob channel so a large bundle entry can be
    transported by content id instead of being inlined.
    """

    vendor = "vendor-b"

    def __init__(
        self, root: Path, *, service: Path | None = None,
        controller: Any | None = None, worker_env: dict[str, str] | None = None,
        release_supported: bool = True, generation: str = "1",
        signing_key: bytes | None = NPU_SIGNING_KEY,
        required_features: tuple[str, ...] | None = None,
        token: str = NPU_TOKEN,
        blob_store: bool = False,
        blob_limits: Any | None = None,
        **engine_options: Any,
    ) -> None:
        from tests.support.vendor_sim.vendor_b import FakeVendorBController, build_graph_bundle

        self.root = root
        self.service_path = service or (root / "graph-service.json")
        self.controller = controller or FakeVendorBController(
            self.service_path, release_supported=release_supported, generation=generation,
        )
        keyring = {} if signing_key is None else {NPU_SIGNING_KEY_ID: signing_key}
        self.blob_store = None
        self.blob_channel = None
        self.blob_ref = None
        if blob_store:
            self.blob_store = BlobStore(root / "state" / "blobs", limits=blob_limits or BlobLimits(
                max_size_bytes=4 << 20, max_chunk_size=65536, min_chunk_size=1,
                max_chunks=8192, max_uploads_per_scope=4,
            ))
        self.engine = npu_bundle_engine(
            root / "state", service_path=self.service_path, controller=self.controller,
            vendor="vendor-b", worker_env=worker_env, signing_key=signing_key,
            signing_key_id=NPU_SIGNING_KEY_ID if signing_key is not None else "",
            signing_keyring=keyring, generation=generation, blob_store=self.blob_store,
            **engine_options,
        )
        #: Set by the test before submitting; defaults to the deterministic graph.
        self.bundle = build_graph_bundle()
        if self.blob_store is not None:
            authorizer = StoreAuthorizer(
                self.engine.store.blob_scope, site_id=self.engine.identity.site_id,
                device_id=self.engine.identity.device_id,
            )
            self.blob_channel = build_blob_service(self.blob_store, authorizer)
        self.server = build_server(
            self.engine,
            config=GrpcServerConfig(
                address="127.0.0.1:0", allow_insecure=True,
                require_client_identity=False, watch_poll_s=0.02,
            ),
            blob_channel=self.blob_channel,
        )
        identity = self.engine.identity
        self.factory = GrpcEvaluatorFactory(
            endpoint=self.server.address,
            token_provider=lambda _run, _session: token,
            run_id="npu-run-1",
            target_id=identity.target_id,
            expected=GrpcExpected(
                site_id=identity.site_id, device_id=identity.device_id,
                target_id=identity.target_id, worker_build=identity.worker_build,
                task_pack_digest=identity.task_pack_digest,
            ),
            api_contract_ids=(contracts.NPU_BUNDLE_API_CONTRACT_ID,),
            evidence_contract_ids=(contracts.NPU_BUNDLE_EVIDENCE_CONTRACT_ID,),
            required_features=required_features or npu_bundle_features(blob=blob_store),
            keyring=SignatureKeyring(keyring),
            legacy_signature_key=NPU_HOST_SIGNATURE_KEY,
            allow_insecure=True,
            watch_slice_s=1.0,
            job_margin_s=6.0,
            bundle_provider=lambda _request: self.bundle,
            blob_descriptor=(
                (lambda _request: self.blob_ref) if blob_store else None
            ),
        )

    def scope(self, token: str = NPU_TOKEN):
        from tcenv.remote.blob_channel import BlobScope

        grant_scope = self.engine.store.blob_scope(token)
        identity = self.engine.identity
        return BlobScope(
            site_id=identity.site_id, device_id=identity.device_id,
            target_id=grant_scope["target_id"], subject=grant_scope["subject"],
            run_id=grant_scope["run_id"], session_id=grant_scope["session_id"],
        )

    def upload(self, data: bytes, *, chunk_size: int = 4096, media_type="application/octet-stream"):
        """Upload a large bundle entry through the P0b blob channel."""

        from tcenv.remote.job_types import BlobDescriptor

        descriptor = BlobDescriptor.for_bytes(data, media_type=media_type, chunk_size=chunk_size)
        scope = self.scope()
        self.blob_store.begin(scope, descriptor.to_dict())
        offset = 0
        while offset < len(data):
            chunk = data[offset:offset + chunk_size]
            self.blob_store.write_chunk(scope, descriptor.content_id, offset, chunk)
            offset += len(chunk)
        self.blob_store.commit(scope, descriptor.content_id)
        return descriptor


class ModuleBuildGrpcHarness(_BundleHarnessBase):
    """P1c composition root: the build contract over fake vendor C.

    One frontend serves one execution shape.  ``mode='prebuilt'`` advertises the
    ``kernel_module``/bundle shape and transports the control-host-built module
    through the real BlobChannel; ``mode='evaluator'`` advertises the inline
    ``kernel``/source shape and builds inside the attempt.  Both use the same
    BuildProvenance/BuildEvidence contract and the same fake toolchain profile
    shape (only the registered profile name/host differs).
    """

    vendor = "vendor-c"

    def __init__(
        self, root: Path, *, mode: str = "prebuilt", ledger: Path | None = None,
        controller: Any | None = None, worker_env: dict[str, str] | None = None,
        release_supported: bool = True, generation: str = "1",
        signing_key: bytes | None = NPU_SIGNING_KEY,
        required_features: tuple[str, ...] | None = None,
        token: str = NPU_TOKEN,
        profile: Any | None = None,
        builder: Any | None = None,
        blob_limits: Any | None = None,
        **engine_options: Any,
    ) -> None:
        import grpc

        from tcenv.remote import contracts
        from tcenv.remote.artifact_builder import ArtifactBuilderRegistry
        from tcenv.remote.artifact_bundle import BundleEntry
        from tcenv.remote.blob_channel import BlobLimits, BlobStore
        from tcenv.remote.grpc_client import GrpcEvaluatorFactory, GrpcExpected, SignatureKeyring
        from tcenv.remote.grpc_server import GrpcServerConfig, build_server
        from tests.support.blob_sim.client import BlobClient
        from tests.support.job_engine import npu_build_engine, npu_build_features, npu_build_identity
        from tests.support.vendor_sim.module_builder import (
            FakeTsmModuleBuilder, control_host_profile, evaluator_profile,
        )
        from tests.support.vendor_sim.vendor_c import FakeModuleVendorController

        if mode not in ("prebuilt", "evaluator"):
            raise ValueError("module harness mode must be prebuilt or evaluator")
        self.mode = mode
        self.root = root
        self.ledger_path = ledger or (root / "vendor-ledger.json")
        self.controller = controller or FakeModuleVendorController(
            self.ledger_path, release_supported=release_supported, generation=generation,
            request_artifact_kind="bundle" if mode == "prebuilt" else "source",
        )
        keyring = {} if signing_key is None else {NPU_SIGNING_KEY_ID: signing_key}
        self.profile = profile or (
            control_host_profile() if mode == "prebuilt" else evaluator_profile()
        )
        self.builder = builder or FakeTsmModuleBuilder(self.profile)
        self.registry = ArtifactBuilderRegistry([self.builder])
        self.builder_calls = 0
        self.blob_store = None
        self.blob_channel = None
        self.blob_client = None
        if mode == "prebuilt":
            self.blob_store = BlobStore(root / "state" / "blobs", limits=blob_limits or BlobLimits(
                max_size_bytes=4 << 20, max_chunk_size=65536, min_chunk_size=1,
                max_chunks=8192, max_uploads_per_scope=4,
            ))
        self.engine = npu_build_engine(
            root / "state", ledger_path=self.ledger_path, controller=self.controller,
            worker_env=worker_env, signing_key=signing_key,
            signing_key_id=NPU_SIGNING_KEY_ID if signing_key is not None else "",
            signing_keyring=keyring, generation=generation, mode=mode,
            build_registry=self.registry, blob_store=self.blob_store,
            **engine_options,
        )
        if self.blob_store is not None:
            from tcenv.remote.blob_service import StoreAuthorizer, build_blob_service

            authorizer = StoreAuthorizer(
                self.engine.store.blob_scope, site_id=self.engine.identity.site_id,
                device_id=self.engine.identity.device_id,
            )
            self.blob_channel = build_blob_service(self.blob_store, authorizer)
        self.server = build_server(
            self.engine,
            config=GrpcServerConfig(
                address="127.0.0.1:0", allow_insecure=True,
                require_client_identity=False, watch_poll_s=0.02,
            ),
            blob_channel=self.blob_channel,
        )
        if self.blob_channel is not None:
            self._blob_rpc_channel = grpc.insecure_channel(self.server.address)
            self.blob_client = BlobClient(self._blob_rpc_channel, token)
        else:
            self._blob_rpc_channel = None
        #: Set by the test before submitting.
        self.bundle = None
        self.build = None
        identity = self.engine.identity
        self.factory = GrpcEvaluatorFactory(
            endpoint=self.server.address,
            token_provider=lambda _run, _session: token,
            run_id="npu-run-1",
            target_id=identity.target_id,
            expected=GrpcExpected(
                site_id=identity.site_id, device_id=identity.device_id,
                target_id=identity.target_id, worker_build=identity.worker_build,
                task_pack_digest=identity.task_pack_digest,
            ),
            api_contract_ids=(contracts.NPU_BUILD_API_CONTRACT_ID,),
            evidence_contract_ids=(contracts.NPU_BUILD_EVIDENCE_CONTRACT_ID,),
            required_features=required_features or npu_build_features(blob=mode == "prebuilt"),
            keyring=SignatureKeyring(keyring),
            legacy_signature_key=NPU_HOST_SIGNATURE_KEY,
            allow_insecure=True,
            watch_slice_s=1.0,
            job_margin_s=6.0,
            bundle_provider=lambda _request: self.bundle,
            build_provider=lambda _request: self.build,
        )

    # -- control-host build and upload -------------------------------------
    def host_build(self, source: bytes, *, deadline_s: float = 10.0):
        """Build on the control host; count invocations for "build once" tests."""

        import time

        from tcenv.remote.artifact_builder import SourceBundle

        from tcenv.remote.artifact_builder import BuildProvenance

        self.builder_calls += 1
        return self.registry.build(
            source_bundle=SourceBundle(source),
            task_contract={"task_family": "kernel_module", "media_type": "text/plain; charset=utf-8"},
            provenance=BuildProvenance(
                profile=self.profile,
                source_sha256=hashlib.sha256(source).hexdigest(),
                artifact_set_digest=None,
            ).to_dict(),
            deadline_monotonic=time.monotonic() + deadline_s,
        )

    def upload_module(self, outcome: Any, *, chunk_size: int | None = None):
        """Upload every blob-backed artifact through the real BlobChannel."""

        if self.blob_client is None:
            raise RuntimeError("this frontend has no blob channel")
        for entry in outcome.bundle.blob_entries:
            data = outcome.payloads[entry.path]
            ref = self.blob_client.upload_and_commit(
                data, chunk_size=chunk_size or int(entry.blob_ref["chunk_size"]),
                media_type=entry.media_type,
            )
            if (
                ref.content_id != entry.blob_ref["content_id"]
                or int(ref.size_bytes) != entry.size_bytes
            ):
                raise AssertionError("the uploaded descriptor does not bind the built artifact")
        return outcome.bundle

    def start_partial_upload(self, outcome: Any, *, chunk_size: int = 8192, chunks: int = 1):
        """Begin the module upload and write only a prefix (no commit)."""

        from tests.support.blob_sim.client import descriptor_for

        module = next(
            entry for entry in outcome.bundle.entries if entry.role == "EXECUTABLE_MODULE"
        )
        assert module.data is None and module.blob_ref is not None
        data = outcome.payloads[module.path]
        ref = descriptor_for(data, media_type=module.media_type, chunk_size=chunk_size)
        self.blob_client.begin(ref)
        offset = 0
        for _ in range(chunks):
            chunk = data[offset:offset + chunk_size]
            self.blob_client.write(ref.content_id, offset, chunk)
            offset += len(chunk)
        return ref, data

    def finish_upload(self, ref: Any, data: bytes) -> None:
        """Resume the interrupted upload from the acknowledged offset."""

        offset = (self.blob_store.state(self.scope(), ref.content_id).received_bytes)
        while offset < len(data):
            chunk = data[offset:offset + ref.chunk_size]
            self.blob_client.write(ref.content_id, offset, chunk)
            offset += len(chunk)
        self.blob_client.commit(ref.content_id)

    def scope(self, token: str = NPU_TOKEN):
        from tcenv.remote.blob_channel import BlobScope

        grant_scope = self.engine.store.blob_scope(token)
        identity = self.engine.identity
        return BlobScope(
            site_id=identity.site_id, device_id=identity.device_id,
            target_id=grant_scope["target_id"], subject=grant_scope["subject"],
            run_id=grant_scope["run_id"], session_id=grant_scope["session_id"],
        )

    def set_prebuilt(self, outcome: Any, bundle: Any, source: bytes) -> None:
        """Publish the built (and uploaded) artifact set + build identity.

        The prebuilt request carries the producer's build-evidence link, so the
        receiver's signed evidence repeats the control-host status/duration/log
        identity and the digest of the evidence document the host kept.
        """

        from tests.support.vendor_sim.module_builder import request_build

        self.bundle = bundle
        self.build = request_build(
            self.profile, source=source, mode="prebuilt",
            artifact_set_digest=bundle.artifact_set_digest,
            producer=outcome.evidence.producer_record(),
        )
        self.outcome = outcome

    def set_evaluator_build(self, source: bytes) -> None:
        from tests.support.vendor_sim.module_builder import request_build

        self.bundle = None
        self.build = request_build(
            self.profile, source=source, mode="evaluator", artifact_set_digest=None,
        )

    def close(self) -> None:
        if self._blob_rpc_channel is not None:
            self._blob_rpc_channel.close()
        self.factory.close()
        self.server.stop(grace_s=0.2)
        self.engine.close(wait=5.0)
