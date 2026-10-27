"""Locations, loaders and builders for backend-neutral evaluator goldens.

Two frozen cross-language vectors live here:

* ``evaluator_v1.json`` is the legacy Metal/source contract golden.
* ``npu_blob_v3.json`` is the P0b blob-reference contract golden.  It is built by
  :func:`build_blob_golden` from the production contract functions, so the Python
  test can detect drift and the independent Go verifier can rebuild the same
  documents from the raw fields.
"""

from __future__ import annotations

import base64
import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from tcenv.remote import contracts

SUPPORT = Path(__file__).resolve().parent
GOLDEN = SUPPORT / "golden" / "remote" / "evaluator_v1.json"
BLOB_GOLDEN = SUPPORT / "golden" / "remote" / "npu_blob_v3.json"
GO_VERIFIER = SUPPORT / "remotecontract"

#: Fixed blob artifact for the cross-language vector (chunked, text media type).
BLOB_SOURCE = b"blob-kernel relu_kernel\n"
BLOB_MEDIA_TYPE = "text/plain; charset=utf-8"
BLOB_CHUNK_SIZE = 4
BLOB_REQUEST_ID = "blob-golden-1"
BLOB_ATTEMPT_ID = "attempt-7-golden"
BLOB_LEASE_FENCE = "7"
BLOB_SESSION_SEQUENCE = "1"
BLOB_ISSUED_AT = "2026-10-28T00:00:00Z"
BLOB_KEY_ID = "npu-site/key-1"
BLOB_KEY = b"b" * 32
BLOB_DEVICE_GROUP = {
    "device_group_id": "vendor-a-group-0",
    "vendor_id": "vendor-a",
    "profile_digest": "a" * 64,
    "topology_digest": "b" * 64,
    "resource_generation": "1",
}


def load_evaluator_golden() -> dict[str, Any]:
    """Load the frozen cross-language contract vectors without recording them."""

    return json.loads(GOLDEN.read_text(encoding="utf-8"))


def load_blob_golden() -> dict[str, Any]:
    return json.loads(BLOB_GOLDEN.read_text(encoding="utf-8"))


def _descriptor(data: bytes, *, schema_id: str, media_type: str) -> dict[str, Any]:
    return {
        "schema_id": schema_id,
        "media_type": media_type,
        "size_bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def build_blob_golden() -> dict[str, Any]:
    """Return the complete P0b blob contract vector (request, receipt, signature)."""

    digest = hashlib.sha256(BLOB_SOURCE).hexdigest()
    blob_ref = {
        "content_id": contracts.blob_content_id(digest),
        "sha256": digest,
        "size_bytes": len(BLOB_SOURCE),
        "media_type": BLOB_MEDIA_TYPE,
        "chunk_size": BLOB_CHUNK_SIZE,
    }
    execution = {
        "artifact_kind": "source",
        "launch_kind": "KERNEL",
        "launch_subtype": "SIMT",
        "source_media_type": BLOB_MEDIA_TYPE,
        "device_group": dict(BLOB_DEVICE_GROUP),
        "required_timing_sources": ["DEVICE_EVENT"],
        "synchronization": "stream-event",
    }
    request = {
        "kind": "formal_eval",
        "run_id": "golden-run",
        "session_id": "golden-session",
        "task": "relu",
        "dtype": "fp16",
        "size": "small",
        "target_id": "sim-npu-source",
        "task_fingerprint": "d" * 64,
        "candidate_digest": {"algorithm": contracts.BLOB_DIGEST_ALGORITHM, "value": digest},
        "execution": execution,
        "blob_ref": blob_ref,
        "mode": "quick",
        "seed": "3",
        "reps": 2,
        "queue_timeout_ms": 3000,
        "candidate_timeout_ms": 3000,
        "cases": None,
        "fast_bench": False,
        "require_contract_launch": False,
        "validate": False,
        "measure_serial": False,
        "cache_policy": "reuse",
    }
    request_document = contracts.request_digest_document(
        request, source_bytes=b"", dialect=contracts.NPU_BLOB_DIALECT,
    )
    request_digest = contracts.request_digest(
        request, source_bytes=b"", dialect=contracts.NPU_BLOB_DIALECT,
    )
    receipt = {
        "receipt_id": contracts.attempt_receipt_id(
            request_id=BLOB_REQUEST_ID, attempt_id=BLOB_ATTEMPT_ID,
            request_digest=request_digest,
            candidate_digest=request["candidate_digest"], device_group=BLOB_DEVICE_GROUP,
            dialect=contracts.NPU_BLOB_DIALECT,
        ),
        "request_id": BLOB_REQUEST_ID,
        "attempt_id": BLOB_ATTEMPT_ID,
        "request_digest": request_digest,
        "lease_fence": BLOB_LEASE_FENCE,
        "session_sequence": BLOB_SESSION_SEQUENCE,
        "issued_at": BLOB_ISSUED_AT,
        "candidate_digest": request["candidate_digest"],
        "device_group": dict(BLOB_DEVICE_GROUP),
    }
    receipt_document = contracts.receipt_id_document(
        request_id=BLOB_REQUEST_ID, attempt_id=BLOB_ATTEMPT_ID, request_digest=request_digest,
        candidate_digest=request["candidate_digest"], device_group=BLOB_DEVICE_GROUP,
        dialect=contracts.NPU_BLOB_DIALECT,
    )
    provenance = {
        "schema_version": 1,
        "evaluator_epoch": "e" * 64,
        "site_id": "sim-npu-site",
        "device_id": "sim-npu-device",
        "backend": "vendor-a",
        "backend_version": "1",
        "target_id": "sim-npu-source",
        "worker_build": "fake-vendor-a-worker-1",
        "task_pack_digest": "f" * 64,
        "calibration_digest": "c" * 64,
        "protocol_version": "3",
        "api_contract_id": contracts.NPU_BLOB_API_CONTRACT_ID,
        "evidence_contract_id": contracts.NPU_BLOB_EVIDENCE_CONTRACT_ID,
        "measurement_kind": "simulated",
        "publication_allowed": False,
        "device_group": dict(BLOB_DEVICE_GROUP),
        "artifact_kind": "source",
        "launch_kind": "KERNEL",
        "launch_subtype": "SIMT",
        "timing_sources": ["DEVICE_EVENT"],
        "attempt_id": BLOB_ATTEMPT_ID,
        "request_digest": request_digest,
    }
    report = b'{"status":"ok","simulated":true}'
    evidence = json.dumps({
        "schema_id": contracts.NPU_TIMING_EVIDENCE_SCHEMA_ID,
        "candidate_digest": request["candidate_digest"],
        "device_group": dict(BLOB_DEVICE_GROUP),
        "timing": {
            "source": "DEVICE_EVENT", "clock_domain": "vendor-a-device-event-clock-v1",
            "synchronization_completed": True, "device_elapsed_ns": 1_250_000,
            "host_api_elapsed_ns": 2_500_000, "launches": 1,
        },
        "measurement_kind": "simulated",
        "publication_allowed": False,
        "complete": True,
    }, sort_keys=True, separators=(",", ":")).encode("utf-8")
    unsigned = {
        "provenance": provenance,
        "report": _descriptor(report, schema_id="tcenv.eval-report.v3", media_type="application/json"),
        "evidence": _descriptor(
            evidence, schema_id=contracts.NPU_TIMING_EVIDENCE_SCHEMA_ID,
            media_type=contracts.NPU_EVIDENCE_MEDIA_TYPE,
        ),
        "receipt": receipt,
    }
    signature_document = contracts.result_signature_document(
        unsigned, key_id=BLOB_KEY_ID, dialect=contracts.NPU_BLOB_DIALECT,
    )
    signature = contracts.sign_result(
        unsigned, key_id=BLOB_KEY_ID, key=BLOB_KEY, dialect=contracts.NPU_BLOB_DIALECT,
    )
    return {
        "schema_id": "tcenv.remote.npu-blob.contract-golden.1",
        "npu_blob": {
            "request": request,
            "request_digest_document": request_document,
            "request_canonical_json": contracts.canonical_digest_document(request_document).decode("utf-8"),
            "request_digest": request_digest,
            "receipt": {
                "request_id": BLOB_REQUEST_ID,
                "attempt_id": BLOB_ATTEMPT_ID,
                "document": receipt_document,
                "canonical_json": contracts.canonical_digest_document(receipt_document).decode("utf-8"),
                "receipt_id": receipt["receipt_id"],
            },
            "signature": {
                "key_id": BLOB_KEY_ID,
                "key_hex": BLOB_KEY.hex(),
                "report_base64": base64.b64encode(report).decode("ascii"),
                "evidence_base64": base64.b64encode(evidence).decode("ascii"),
                "unsigned_result": contracts.result_signature_document(
                    unsigned, key_id=BLOB_KEY_ID, dialect=contracts.NPU_BLOB_DIALECT,
                ),
                "document": signature_document,
                "canonical_json": contracts.canonical_digest_document(signature_document).decode("utf-8"),
                "hmac_sha256": signature,
            },
        },
    }


def write_blob_golden(path: Path | None = None) -> Path:
    """Write the P0b vector (fixture maintenance only; the test never records)."""

    target = path or BLOB_GOLDEN
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(build_blob_golden(), indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    return target


# -- P1 bundle contract vector --------------------------------------------
BUNDLE_GOLDEN = SUPPORT / "golden" / "remote" / "npu_bundle_v4.json"
BUNDLE_REQUEST_ID = "bundle-golden-1"
BUNDLE_ATTEMPT_ID = "attempt-9-golden"
BUNDLE_LEASE_FENCE = "9"
BUNDLE_SESSION_SEQUENCE = "2"
BUNDLE_ISSUED_AT = "2026-10-29T00:00:00Z"
BUNDLE_KEY_ID = "npu-bundle-site/key-1"
BUNDLE_KEY = b"k" * 32
BUNDLE_DEVICE_ELAPSED_NS = 1_250_000
BUNDLE_HOST_ELAPSED_NS = 2_500_000
BUNDLE_CLOCK_DOMAIN = "vendor-b-service-clock-v1"
BUNDLE_DEVICE_GROUP = {
    "device_group_id": "vendor-b-group-0",
    "vendor_id": "vendor-b",
    "profile_digest": "1" * 64,
    "topology_digest": "2" * 64,
    "resource_generation": "1",
}
BUNDLE_ENTRIES = (
    ("graph/main.json", "GRAPH_DEFINITION", "application/json",
     b'{"schema":"fake-graph-v1","nodes":[{"op":"matmul"},{"op":"relu"}]}\n'),
    ("graph/params.bin", "PARAMETER_TABLE", "application/octet-stream",
     bytes(range(256)) * 4),
)


def load_bundle_golden() -> dict[str, Any]:
    return json.loads(BUNDLE_GOLDEN.read_text(encoding="utf-8"))


def bundle_completeness_vectors(
    evidence_document: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """TimingEvidence completeness vectors shared with the Go verifier.

    ``valid`` is decided by the production validator here, and re-derived
    independently in Go, so the device/rank/die coverage rule cannot drift
    between the two languages: the wire ``spans`` of one device with two ranks
    are two spans, a missing coordinate is a missing span, and a graph-only
    vendor is not asked for a dimension it does not have.
    """

    base = copy.deepcopy(dict(evidence_document))
    clock = base["timing"]["clock_domain"]

    def span(device: str, *, rank=None, die=None, elapsed: int) -> dict[str, Any]:
        return {"device_id": device, "rank": rank, "die": die,
                "clock_domain": clock, "elapsed_ns": elapsed}

    def with_shape(*, devices, ranks, dies, spans, elapsed) -> dict[str, Any]:
        document = copy.deepcopy(base)
        document["topology"] = {
            "logical_devices": len(devices), "declared_devices": list(devices),
            "required_rank_spans": ranks, "required_die_spans": dies,
        }
        document["spans"] = spans
        document["timing"] = dict(
            document["timing"], device_elapsed_ns=elapsed, launches=len(spans),
        )
        return document

    two_devices = ["device-0", "device-1"]
    rank_complete = [
        span("device-0", rank=0, elapsed=1_100), span("device-0", rank=1, elapsed=1_250),
        span("device-1", rank=0, elapsed=1_150), span("device-1", rank=1, elapsed=1_300),
    ]
    documents = [
        ("graph-service-one-device", with_shape(
            devices=["device-0"], ranks=None, dies=None,
            spans=[span("device-0", elapsed=1_250)], elapsed=1_250,
        )),
        ("two-devices-two-ranks", with_shape(
            devices=two_devices, ranks=2, dies=None, spans=rank_complete, elapsed=1_300,
        )),
        ("missing-second-rank-of-one-device", with_shape(
            devices=two_devices, ranks=2, dies=None,
            spans=[rank_complete[0], rank_complete[1], rank_complete[2]], elapsed=1_250,
        )),
        ("duplicate-device-rank-coordinate", with_shape(
            devices=two_devices, ranks=2, dies=None,
            spans=[rank_complete[0], rank_complete[0], rank_complete[2], rank_complete[3]],
            elapsed=1_300,
        )),
        ("graph-only-vendor-invents-a-rank", with_shape(
            devices=["device-0"], ranks=None, dies=None,
            spans=[span("device-0", rank=0, elapsed=1_250)], elapsed=1_250,
        )),
        ("rank-the-topology-does-not-cover", with_shape(
            devices=two_devices, ranks=2, dies=None,
            spans=rank_complete[:3] + [span("device-1", rank=2, elapsed=1_300)], elapsed=1_300,
        )),
    ]
    vectors: list[dict[str, Any]] = []
    for name, document in documents:
        try:
            checked = contracts.npu_timing_evidence_document_v2(document)
        except contracts.ContractError:
            vectors.append({"name": name, "valid": False, "document": document})
        else:
            vectors.append({
                "name": name, "valid": True, "document": checked,
                "device_elapsed_ns": checked["timing"]["device_elapsed_ns"],
            })
    return vectors


def build_bundle_golden() -> dict[str, Any]:
    """Return the complete P1 bundle contract vector.

    It is built from the production functions so the Python test detects drift,
    while the independent Go verifier rebuilds every document from the raw
    fields (manifest, request, receipt, result signature).
    """

    from tcenv.remote.artifact_bundle import (
        BUNDLE_SCHEMA_ID,
        ArtifactBundle,
        BundleEntry,
        canonical_manifest,
    )

    entries = [
        BundleEntry(
            path=path, role=role, media_type=media_type, size_bytes=len(data),
            sha256=hashlib.sha256(data).hexdigest(), data=data,
        )
        for path, role, media_type, data in BUNDLE_ENTRIES
    ]
    bundle = ArtifactBundle(entries=tuple(entries))
    artifact_set = bundle.artifact_set_digest
    execution = {
        "schema_id": contracts.NPU_BUNDLE_EXECUTION_SCHEMA_ID,
        "task_family": "graph",
        "artifact_kind": "bundle",
        "launch_kind": "GRAPH",
        "launch_subtype": "SERVICE",
        "source_media_type": "application/octet-stream",
        "device_group": dict(BUNDLE_DEVICE_GROUP),
        "resources": {
            "logical_devices": 1,
            "required_rank_spans": None,
            "required_die_spans": None,
        },
        "required_timing_sources": ["DEVICE_EVENT"],
        "synchronization": "service-completion-token",
    }
    request = {
        "kind": "formal_eval",
        "run_id": "golden-run",
        "session_id": "golden-session",
        "task": "relu",
        "dtype": "fp16",
        "size": "small",
        "target_id": "sim-npu-graph",
        "task_fingerprint": "d" * 64,
        "candidate_digest": {"algorithm": contracts.BUNDLE_DIGEST_ALGORITHM, "value": artifact_set},
        "execution": execution,
        "artifact": {"bundle": bundle.to_dict()},
        "mode": "quick",
        "seed": "3",
        "reps": 2,
        "queue_timeout_ms": 3000,
        "candidate_timeout_ms": 3000,
        "cases": None,
        "fast_bench": False,
        "require_contract_launch": False,
        "validate": False,
        "measure_serial": False,
        "cache_policy": "reuse",
    }
    request_document = contracts.request_digest_document(
        request, source_bytes=b"", dialect=contracts.NPU_BUNDLE_DIALECT,
    )
    request_digest = contracts.request_digest(
        request, source_bytes=b"", dialect=contracts.NPU_BUNDLE_DIALECT,
    )
    receipt = {
        "receipt_id": contracts.attempt_receipt_id(
            request_id=BUNDLE_REQUEST_ID, attempt_id=BUNDLE_ATTEMPT_ID,
            request_digest=request_digest, candidate_digest=request["candidate_digest"],
            device_group=BUNDLE_DEVICE_GROUP, dialect=contracts.NPU_BUNDLE_DIALECT,
        ),
        "request_id": BUNDLE_REQUEST_ID,
        "attempt_id": BUNDLE_ATTEMPT_ID,
        "request_digest": request_digest,
        "lease_fence": BUNDLE_LEASE_FENCE,
        "session_sequence": BUNDLE_SESSION_SEQUENCE,
        "issued_at": BUNDLE_ISSUED_AT,
        "candidate_digest": request["candidate_digest"],
        "device_group": dict(BUNDLE_DEVICE_GROUP),
    }
    receipt_document = contracts.receipt_id_document(
        request_id=BUNDLE_REQUEST_ID, attempt_id=BUNDLE_ATTEMPT_ID,
        request_digest=request_digest, candidate_digest=request["candidate_digest"],
        device_group=BUNDLE_DEVICE_GROUP, dialect=contracts.NPU_BUNDLE_DIALECT,
    )
    provenance = {
        "schema_version": 1,
        "evaluator_epoch": "e" * 64,
        "site_id": "sim-npu-site",
        "device_id": "sim-npu-graph-device",
        "backend": "vendor-b",
        "backend_version": "1",
        "target_id": "sim-npu-graph",
        "worker_build": "fake-vendor-b-worker-1",
        "task_pack_digest": "f" * 64,
        "calibration_digest": "c" * 64,
        "protocol_version": "3",
        "api_contract_id": contracts.NPU_BUNDLE_API_CONTRACT_ID,
        "evidence_contract_id": contracts.NPU_BUNDLE_EVIDENCE_CONTRACT_ID,
        "measurement_kind": "simulated",
        "publication_allowed": False,
        "device_group": dict(BUNDLE_DEVICE_GROUP),
        "artifact_kind": "bundle",
        "launch_kind": "GRAPH",
        "launch_subtype": "SERVICE",
        "timing_sources": ["DEVICE_EVENT"],
        "attempt_id": BUNDLE_ATTEMPT_ID,
        "request_digest": request_digest,
    }
    report = (b'{"status":"ok","simulated":true,"candidate_us":%d}'
              % (BUNDLE_DEVICE_ELAPSED_NS // 1000))
    evidence_document = {
        "schema_id": contracts.NPU_BUNDLE_TIMING_EVIDENCE_SCHEMA_ID,
        "candidate_digest": request["candidate_digest"],
        "artifact_set_digest": artifact_set,
        "device_group": dict(BUNDLE_DEVICE_GROUP),
        "topology": {
            "logical_devices": 1,
            "declared_devices": ["device-0"],
            "required_rank_spans": None,
            "required_die_spans": None,
        },
        "spans": [{
            "device_id": "device-0", "rank": None, "die": None,
            "clock_domain": BUNDLE_CLOCK_DOMAIN, "elapsed_ns": BUNDLE_DEVICE_ELAPSED_NS,
        }],
        "timing": {
            "source": "DEVICE_EVENT", "clock_domain": BUNDLE_CLOCK_DOMAIN,
            "synchronization_completed": True,
            "device_elapsed_ns": BUNDLE_DEVICE_ELAPSED_NS,
            "host_api_elapsed_ns": BUNDLE_HOST_ELAPSED_NS,
            "launches": 1,
        },
        "buffer": None,
        "measurement_kind": "simulated",
        "publication_allowed": False,
        "complete": True,
    }
    evidence = json.dumps(
        evidence_document, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")
    unsigned = {
        "provenance": provenance,
        "report": _descriptor(report, schema_id="tcenv.eval-report.v3", media_type="application/json"),
        "evidence": _descriptor(
            evidence, schema_id=contracts.NPU_BUNDLE_TIMING_EVIDENCE_SCHEMA_ID,
            media_type=contracts.NPU_BUNDLE_EVIDENCE_MEDIA_TYPE,
        ),
        "receipt": receipt,
    }
    signature_document = contracts.result_signature_document(
        unsigned, key_id=BUNDLE_KEY_ID, dialect=contracts.NPU_BUNDLE_DIALECT,
    )
    signature = contracts.sign_result(
        unsigned, key_id=BUNDLE_KEY_ID, key=BUNDLE_KEY, dialect=contracts.NPU_BUNDLE_DIALECT,
    )
    return {
        "schema_id": "tcenv.remote.npu-bundle.contract-golden.1",
        "npu_bundle": {
            "request": request,
            "bundle_manifest": canonical_manifest(
                BUNDLE_SCHEMA_ID, [entry.manifest() for entry in bundle.entries],
            ),
            "artifact_set_digest": artifact_set,
            "request_digest_document": request_document,
            "request_canonical_json": contracts.canonical_digest_document(request_document).decode("utf-8"),
            "request_digest": request_digest,
            "receipt": {
                "request_id": BUNDLE_REQUEST_ID,
                "attempt_id": BUNDLE_ATTEMPT_ID,
                "document": receipt_document,
                "canonical_json": contracts.canonical_digest_document(receipt_document).decode("utf-8"),
                "receipt_id": receipt["receipt_id"],
            },
            "evidence": evidence_document,
            "completeness_vectors": bundle_completeness_vectors(evidence_document),
            "signature": {
                "key_id": BUNDLE_KEY_ID,
                "key_hex": BUNDLE_KEY.hex(),
                "report_base64": base64.b64encode(report).decode("ascii"),
                "evidence_base64": base64.b64encode(evidence).decode("ascii"),
                "unsigned_result": contracts.result_signature_document(
                    unsigned, key_id=BUNDLE_KEY_ID, dialect=contracts.NPU_BUNDLE_DIALECT,
                ),
                "document": signature_document,
                "canonical_json": contracts.canonical_digest_document(signature_document).decode("utf-8"),
                "hmac_sha256": signature,
            },
        },
    }


def write_bundle_golden(path: Path | None = None) -> Path:
    """Write the P1 vector (fixture maintenance only; the test never records)."""

    target = path or BUNDLE_GOLDEN
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(build_bundle_golden(), indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    return target
