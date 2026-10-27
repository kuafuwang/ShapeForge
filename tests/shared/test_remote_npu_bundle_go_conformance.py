"""Non-Python conformance for the P1 bundle contract.

The independent Go verifier (``tests/support/remotecontract``) rebuilds the
canonical bundle manifest, the artifact-set digest, the request document, the
receipt and the signed result from the raw fields; it also re-decides the
version-2 evidence completeness rules (exact device x rank x die coverage, one
span per coordinate, aggregate device time, host/device separation) against the
shared vectors.  It never imports Python.
"""

from __future__ import annotations

import copy
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

from tests.support.remote_contract import (
    BUNDLE_GOLDEN,
    GO_VERIFIER,
    build_bundle_golden,
    load_bundle_golden,
)

pytestmark = pytest.mark.skipif(
    shutil.which("go") is None, reason="the Go toolchain is required for cross-language conformance",
)


def _go_environment() -> dict[str, str]:
    env = dict(os.environ)
    base = Path(tempfile.gettempdir())
    env.setdefault("GOMODCACHE", str(base / "go-modcache"))
    env.setdefault("GOCACHE", str(base / "go-build-cache"))
    env.setdefault("GOPATH", str(base / "gopath"))
    for name in ("GOMODCACHE", "GOCACHE", "GOPATH"):
        Path(env[name]).mkdir(parents=True, exist_ok=True)
    env["GOFLAGS"] = "-mod=mod"
    return env


def _run_verifier(golden: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["go", "run", ".", "-golden", str(golden)],
        cwd=GO_VERIFIER, env=_go_environment(), text=True, capture_output=True, timeout=300,
    )


def test_the_frozen_bundle_vector_matches_the_production_documents():
    frozen = load_bundle_golden()
    assert frozen == json.loads(json.dumps(build_bundle_golden()))
    assert frozen["schema_id"] == "tcenv.remote.npu-bundle.contract-golden.1"
    bundle = frozen["npu_bundle"]
    assert bundle["request"]["candidate_digest"]["algorithm"] == "sha256-bundle-v1"
    assert bundle["request"]["artifact"]["bundle"]["artifact_set_digest"] == (
        bundle["artifact_set_digest"]
    )
    assert bundle["evidence"]["schema_id"] == "tcenv.remote.npu-bundle.timing-evidence.2"
    assert bundle["evidence"]["artifact_set_digest"] == bundle["artifact_set_digest"]


def test_every_completeness_vector_agrees_with_the_production_validator():
    """The shared device/rank/die vectors must decide the same way on both sides.

    The golden carries the documents *and* the verdict the production validator
    reached; the Go verifier re-derives each verdict from the declared topology.
    A drift in either implementation turns one of them into a disagreement.
    """

    from tcenv.remote import contracts

    vectors = load_bundle_golden()["npu_bundle"]["completeness_vectors"]
    names = {vector["name"] for vector in vectors}
    assert {"two-devices-two-ranks", "graph-service-one-device"} <= names
    assert any(vector["valid"] for vector in vectors)
    assert not all(vector["valid"] for vector in vectors)
    for vector in vectors:
        try:
            checked = contracts.npu_timing_evidence_document_v2(vector["document"])
        except contracts.ContractError:
            assert vector["valid"] is False, vector["name"]
        else:
            assert vector["valid"] is True, vector["name"]
            assert checked["complete"] is True
            assert checked["timing"]["device_elapsed_ns"] == vector["device_elapsed_ns"]
            # One span per declared device x rank x die coordinate, no extras.
            topology = checked["topology"]
            ranks = topology["required_rank_spans"] or 1
            dies = topology["required_die_spans"] or 1
            assert len(checked["spans"]) == len(topology["declared_devices"]) * ranks * dies


def test_the_go_verifier_rebuilds_the_bundle_request_receipt_and_signature():
    result = _run_verifier(BUNDLE_GOLDEN)
    assert result.returncode == 0, result.stderr
    assert "ok bundle artifact_set=" in result.stdout


@pytest.mark.parametrize("mutation", [
    "entry_sha256",
    "artifact_set_digest",
    "manifest_path",
    "execution_shape",
    "request_digest",
    "receipt_id",
    "evidence_span",
    "evidence_missing_rank",
    "evidence_host_as_device",
    "signature",
])
def test_the_go_verifier_rejects_a_tampered_bundle_vector(tmp_path, mutation):
    golden = copy.deepcopy(load_bundle_golden())
    bundle = golden["npu_bundle"]
    if mutation == "entry_sha256":
        bundle["request"]["artifact"]["bundle"]["entries"][0]["sha256"] = "0" * 64
    elif mutation == "artifact_set_digest":
        bundle["request"]["artifact"]["bundle"]["artifact_set_digest"] = "1" * 64
    elif mutation == "manifest_path":
        bundle["bundle_manifest"]["entries"][0]["path"] = "graph/evil.json"
    elif mutation == "execution_shape":
        bundle["request"]["execution"]["launch_kind"] = "KERNEL"
    elif mutation == "request_digest":
        bundle["request_canonical_json"] = bundle["request_canonical_json"].replace(
            "relu", "gelu"
        )
    elif mutation == "receipt_id":
        bundle["receipt"]["receipt_id"] = "0" * 64
    elif mutation == "evidence_span":
        bundle["evidence"]["spans"] = []
    elif mutation == "evidence_missing_rank":
        topology = bundle["evidence"]["topology"]
        topology["logical_devices"] = 2
        topology["declared_devices"] = ["device-0", "device-1"]
        topology["required_rank_spans"] = 2
        bundle["evidence"]["spans"] = [
            dict(bundle["evidence"]["spans"][0], rank=0),
            dict(bundle["evidence"]["spans"][0], rank=1),
        ]
    elif mutation == "evidence_host_as_device":
        bundle["evidence"]["timing"]["device_elapsed_ns"] = bundle["evidence"]["timing"][
            "host_api_elapsed_ns"
        ]
    else:
        bundle["signature"]["hmac_sha256"] = "0" * 64
    path = tmp_path / "tampered_bundle.json"
    path.write_text(json.dumps(golden), encoding="utf-8")

    result = _run_verifier(path)
    assert result.returncode != 0, "a tampered bundle vector was accepted"
    assert "verification failed" in result.stderr


def test_the_go_verifier_still_accepts_the_earlier_contract_vectors(tmp_path):
    from tests.support.remote_contract import BLOB_GOLDEN, GOLDEN

    for vector in (GOLDEN, BLOB_GOLDEN):
        result = _run_verifier(vector)
        assert result.returncode == 0, (vector, result.stderr)
