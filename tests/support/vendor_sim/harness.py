"""Test-only composition root for the P0 NPU gRPC loop."""

from __future__ import annotations

from pathlib import Path

from tcenv.remote import contracts
from tcenv.remote.grpc_client import GrpcEvaluatorFactory, GrpcExpected, SignatureKeyring
from tcenv.remote.grpc_server import GrpcServerConfig, build_server
from tests.support.job_engine import (
    NPU_FEATURES,
    NPU_TARGET,
    NPU_TOKEN,
    npu_engine,
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
