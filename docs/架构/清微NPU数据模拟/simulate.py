#!/usr/bin/env python3
"""Offline 清微 NPU data/protocol simulation.

This is deliberately not a Qingmicro compiler, ABI, device model, or performance test.  It uses
one real v2 task contract to generate deterministic tensors, checks exact-byte ArtifactBundle
transport, then submits the same candidate source to the existing fake vendor-A gRPC service.
All device execution and timing are fake; no result can publish or resolve NPU-P0.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tcenv.remote import contracts
from tcenv.remote.artifact_bundle import (
    ArtifactBundle,
    BundleEntry,
    BundleError,
    payload_bundle,
    payload_bytes,
)
from tcenv.evaluation import EvaluationRequest
from taic.evaluation_outbox import EvaluationOutbox, bind_wire_digest
from tcenv.taskcontract import discover_standard_tasks
from tests.support.job_engine import NPU_TARGET, NPU_TASK_FINGERPRINT
from tests.support.v1_tasks import V1_TASK
from tests.support.vendor_sim.harness import NPU_HOST_SIGNATURE_KEY, NpuGrpcHarness

SCHEMA_ID = "local.qingmicro-npu-data-simulation.1"
REPORT_PATH = Path(__file__).resolve().parent / "report.json"


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical(document: Any) -> bytes:
    from tcenv.identity import canonical_bytes

    return canonical_bytes(document)


def _tensor_record(name: str, role: str, tensor: Any) -> tuple[dict[str, Any], bytes]:
    value = tensor.detach().to("cpu").contiguous()
    full_data = value.numpy().tobytes(order="C")
    flat = value.reshape(-1)
    indices = sorted(set([0, 1, 2, 3, max(0, flat.numel() - 1)]))
    # Keep the full tensor identity without turning this small offline example into a BlobChannel
    # test.  Only the deterministic first-256-element window travels in the inline bundle.
    window = flat[: min(256, flat.numel())].contiguous()
    window_data = window.numpy().tobytes(order="C")
    return {
        "name": name,
        "role": role,
        "dtype": str(value.dtype).removeprefix("torch."),
        "shape": list(value.shape),
        "numel": int(value.numel()),
        "nbytes": len(full_data),
        "sha256": _sha256(full_data),
        "bundle_window": {
            "offset_elements": 0,
            "numel": int(window.numel()),
            "nbytes": len(window_data),
            "sha256": _sha256(window_data),
        },
        "samples": [[index, float(flat[index].item())] for index in indices],
    }, window_data


def _entry(path: str, role: str, media_type: str, data: bytes) -> BundleEntry:
    return BundleEntry(
        path=path,
        role=role,
        media_type=media_type,
        size_bytes=len(data),
        sha256=_sha256(data),
        data=data,
    )


def _run_remote(root: Path) -> dict[str, Any]:
    """Run the preserved-v1 protocol fixture; it is not the sigmoid data task."""
    source = b"npu-kernel rope_kernel\nnpu-launch\n"
    with NpuGrpcHarness(root) as harness:
        legacy = EvaluationRequest(
            kind="formal_eval",
            request_id="qingmicro-sim-rope-protocol-7",
            run_id="npu-run-1",
            session_id="npu-session-1",
            task=V1_TASK,
            dtype="fp16",
            size="small",
            target_id=NPU_TARGET,
            task_fingerprint=NPU_TASK_FINGERPRINT,
            source=source.decode("utf-8"),
            mode="quick",
            seed=7,
            reps=1,
            queue_timeout_s=3.0,
            candidate_timeout_s=3.0,
        )
        outbox = EvaluationOutbox(
            root / "outbox",
            signature_key=NPU_HOST_SIGNATURE_KEY,
            transport="grpc",
            api_contract_id=contracts.NPU_SOURCE_API_CONTRACT_ID,
            evidence_contract_id=contracts.NPU_SOURCE_EVIDENCE_CONTRACT_ID,
        )
        request = bind_wire_digest(legacy, harness.factory, outbox)
        wire_digest = harness.factory.wire_request_digest(legacy)
        assert request.request_digest == wire_digest
        result = harness.factory.formal(request)
        evidence = result.evidence
        assert result.provenance.measurement_kind == "simulated"
        assert evidence["measurement_kind"] == "simulated"
        assert evidence["publication_allowed"] is False
        assert evidence["complete"] is True
        assert result.receipt["candidate_digest"] == {"algorithm": "sha256", "value": _sha256(source)}
        return {
            "fixture_task": V1_TASK,
            "fixture_boundary": "independent preserved-v1 protocol fixture, not the v2 sigmoid tensor task",
            "api_contract_id": contracts.NPU_SOURCE_API_CONTRACT_ID,
            "evidence_contract_id": contracts.NPU_SOURCE_EVIDENCE_CONTRACT_ID,
            "request_digest": wire_digest,
            "candidate_digest": result.receipt["candidate_digest"],
            "device_group": result.receipt["device_group"],
            "measurement_kind": result.provenance.measurement_kind,
            "publication_allowed": evidence["publication_allowed"],
            "complete": evidence["complete"],
            "synchronization_completed": evidence["timing"]["synchronization_completed"],
            "artifact_format": evidence["diagnostics"]["artifact_format"],
            "artifact_sha256": evidence["diagnostics"]["artifact_sha256"],
            "timing_numbers_included": False,
        }


def simulate() -> dict[str, Any]:
    locators = {
        locator.op_name: locator
        for locator in discover_standard_tasks(ROOT / "tasks", code_root=ROOT)
    }
    locator = locators["sigmoid"]
    contract = locator.load()
    instance = contract.instance
    assert (instance.dtype, instance.size) == ("fp16", "small")
    inputs = contract.reference.generate(instance, "normal", 7)
    reference = contract.reference.execute(instance, inputs)
    import torch

    tensor_inputs = {
        "x-reference-fp32": ("reference_input", inputs["x"]),
        "x-candidate-fp16": ("simulated_candidate_input", inputs["x"].to(torch.float16)),
        "y-reference-fp64": ("reference_output_not_device_result", reference["y"]),
    }
    tensors: list[dict[str, Any]] = []
    tensor_blobs: dict[str, bytes] = {}
    for name, (role, tensor) in tensor_inputs.items():
        record, data = _tensor_record(name, role, tensor)
        tensors.append(record)
        tensor_blobs[name] = data

    source = b"npu-kernel sigmoid_kernel\nnpu-launch\n"
    data_manifest = {
        "schema_id": SCHEMA_ID,
        "simulated": True,
        "publication_allowed": False,
        "task": {
            "op_name": contract.op_name,
            "spec_version": 2,
            "dtype": instance.dtype,
            "size": instance.size,
            "case": "normal",
            "seed": 7,
            "opspec_digest": contract.spec.digest,
            "task_fingerprint": contract.fingerprint(),
            "contract_pack_digest": contract.pack_digest,
            "deployment_pack_digest": locator.deployment_pack_digest,
        },
        "reference": contract.reference.document(),
        "tensors": tensors,
        "candidate_source_sha256": _sha256(source),
        "tensor_execution": "not executed on a real or fake NPU; exact-byte offline data simulation only",
    }
    entries = [
        _entry("candidate/kernel.npu", "SOURCE", "text/plain; charset=utf-8", source),
        *[
            _entry(f"data/{name}.bin", "AUXILIARY", "application/octet-stream", tensor_blobs[name])
            for name in sorted(tensor_blobs)
        ],
        _entry("data/manifest.json", "AUXILIARY", "application/json", _canonical(data_manifest)),
    ]
    bundle = ArtifactBundle(entries=tuple(entries))
    payload = payload_bytes(bundle)
    restored = payload_bundle(payload)
    assert restored.to_dict() == bundle.to_dict()

    with tempfile.TemporaryDirectory(prefix="qingmicro-npu-data-sim-") as temporary:
        root = Path(temporary)
        written = restored.materialize(root / "materialized", anchor=root)
        for entry in restored.entries:
            assert written[entry.path].read_bytes() == entry.data
        tampered = payload[:-1] + bytes([payload[-1] ^ 1])
        try:
            payload_bundle(tampered)
        except BundleError:
            tamper_rejected = True
        else:  # pragma: no cover - fail closed
            raise AssertionError("a one-byte payload mutation was not rejected")
        remote = _run_remote(root / "remote")

    return {
        "schema_id": SCHEMA_ID,
        "simulated": True,
        "publication_allowed": False,
        "npu_p0": {
            "status": "blocked",
            "real_device_available": False,
            "real_toolchain_validated": False,
            "correctness_on_device_proven": False,
            "performance_on_device_proven": False,
            "reason": "offline data/protocol simulation cannot resolve NPU-P0 or publish NPU evidence",
        },
        "data_manifest": data_manifest,
        "bundle": {
            "artifact_set_digest": bundle.artifact_set_digest,
            "payload_sha256": _sha256(payload),
            "entries": bundle.manifest["entries"],
            "payload_roundtrip_exact": True,
            "materialized_bytes_exact": True,
            "one_byte_tamper_rejected": tamper_rejected,
        },
        "remote": remote,
        "checks": [
            "v2 contract fingerprint and deployment pack bind the offline tensor manifest",
            "reference input/output and simulated candidate tensors have stable exact-byte digests",
            "the independent preserved-v1 fake-NPU protocol fixture completes with its own bound identity",
            "ArtifactBundle payload roundtrip and materialization preserve every byte",
            "one-byte payload mutation is rejected",
            "fake service reports simulated=true and publication_allowed=false",
            "no device correctness, timing, speedup or publication claim is produced",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--replay", action="store_true")
    parser.add_argument("--replay-against", type=Path)
    args = parser.parse_args()
    report = simulate()
    encoded = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    replay_path = args.replay_against
    if args.replay:
        replay_path = REPORT_PATH
    if replay_path is not None:
        expected = replay_path.read_text(encoding="utf-8")
        if encoded != expected:
            raise SystemExit(f"simulation replay differs from {replay_path}")
        print("OK: 清微 NPU 离线数据/协议模拟逐字段重放一致；simulated=true, publication_allowed=false, NPU-P0=blocked")
        return
    if args.output is None:
        print(encoded, end="")
    else:
        args.output.write_text(encoded, encoding="utf-8")
        print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
