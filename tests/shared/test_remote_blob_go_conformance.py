"""Non-Python conformance for the P0b blob-reference control metadata.

The independent Go verifier (``tests/support/remotecontract``) rebuilds the
canonical request document, the receipt document and the signed result document
from the raw fields and compares their RFC 8785 bytes, digests and HMAC with the
frozen vector.  Tampering with any of them must make it fail.
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
    BLOB_GOLDEN,
    GO_VERIFIER,
    build_blob_golden,
    load_blob_golden,
)

ROOT = Path(__file__).resolve().parents[2]

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


def test_the_frozen_blob_vector_matches_the_production_documents():
    frozen = load_blob_golden()
    assert frozen == json.loads(json.dumps(build_blob_golden()))
    assert frozen["schema_id"] == "tcenv.remote.npu-blob.contract-golden.1"
    request = frozen["npu_blob"]["request"]
    assert request["candidate_digest"]["algorithm"] == "sha256-blob-v1"
    assert request["blob_ref"]["content_id"].startswith("blob-sha256-v1:")


def test_the_go_verifier_rebuilds_the_blob_request_receipt_and_signature():
    result = _run_verifier(BLOB_GOLDEN)
    assert result.returncode == 0, result.stderr
    assert "ok blob request=" in result.stdout


@pytest.mark.parametrize("mutation", [
    "candidate_digest",
    "blob_size",
    "request_digest",
    "receipt_id",
    "signature",
])
def test_the_go_verifier_rejects_a_tampered_blob_vector(tmp_path, mutation):
    golden = copy.deepcopy(load_blob_golden())
    blob = golden["npu_blob"]
    if mutation == "candidate_digest":
        blob["request"]["candidate_digest"]["value"] = "0" * 64
    elif mutation == "blob_size":
        blob["request"]["blob_ref"]["size_bytes"] += 1
    elif mutation == "request_digest":
        blob["request_canonical_json"] = blob["request_canonical_json"].replace("relu", "gelu")
    elif mutation == "receipt_id":
        blob["receipt"]["receipt_id"] = "0" * 64
    else:
        blob["signature"]["hmac_sha256"] = "0" * 64
    path = tmp_path / "tampered_blob.json"
    path.write_text(json.dumps(golden), encoding="utf-8")

    result = _run_verifier(path)
    assert result.returncode != 0, "a tampered blob vector was accepted"
    assert "verification failed" in result.stderr
