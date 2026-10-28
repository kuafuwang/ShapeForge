"""Non-Python conformance for the P1c build contract (document version 5).

The independent Go verifier (``tests/support/remotecontract``) rebuilds the
version-5 request digest document from the raw fields, including the independent
``build`` document, and re-derives the received build-evidence rules: the
artifact set is ``reused``, the producer link is repeated, the link hashes its
own producer evidence document, the manifest and load record cover the executable
module, and the receiver recorded its own verify/load cost.  It never imports
Python, so the build identity is not only checked by the implementation that
emitted it.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

from tests.support.remote_contract import (
    BUILD_GOLDEN,
    GO_VERIFIER,
    build_build_golden,
    load_build_golden,
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


def test_the_frozen_build_vector_matches_the_production_documents():
    frozen = load_build_golden()
    assert frozen == json.loads(json.dumps(build_build_golden()))
    vector = frozen["npu_build"]
    assert frozen["schema_id"] == "tcenv.remote.npu-build.contract-golden.1"
    assert vector["request_digest_document"]["document_version"] == 5
    assert vector["request_digest_document"]["document_type"] == (
        "tcenv.remote.npu-build.request-digest"
    )
    # The build field is the only addition to the version-4 request document.
    bundle_fields = set(vector["request"]) - {"build"}
    assert set(vector["request_digest_document"]["request"]) == set(vector["request"])
    assert "build" in vector["request_digest_document"]["request"]
    assert bundle_fields
    # The producer link names the producer's own evidence document.
    assert vector["producer_record"]["evidence_sha256"] != ""
    assert vector["producer_record"] == vector["request"]["build"]["producer"]
    assert vector["evidence"]["build"]["producer"] == vector["producer_record"]


def test_the_production_validators_accept_the_build_vector():
    """The frozen vector is admissible under the production contract, not just Go."""

    from tcenv.remote import contracts

    vector = load_build_golden()["npu_build"]
    request_document = contracts.request_digest_document(
        vector["request"], source_bytes=b"", dialect=contracts.NPU_BUILD_DIALECT,
    )
    assert contracts.canonical_digest_document(request_document).decode("utf-8") == (
        vector["request_canonical_json"]
    )
    assert contracts.request_digest(
        vector["request"], source_bytes=b"", dialect=contracts.NPU_BUILD_DIALECT,
    ) == vector["request_digest"]
    build_document = contracts.build_request_document(vector["request"]["build"])
    producer = contracts.build_producer_record(vector["producer_record"])
    assert build_document["producer"] == producer
    contracts.npu_evidence_document(
        vector["evidence"],
        source_bytes=None,
        device_group=vector["request"]["execution"]["device_group"],
        expected_resources=vector["request"]["execution"]["resources"],
        expected_build=vector["request"]["build"],
        request_artifact=vector["request"]["artifact"],
        measurement_kind="simulated", publication_allowed=False,
        report_status="ok", dialect=contracts.NPU_BUILD_DIALECT,
    )


def test_the_go_verifier_rebuilds_the_build_request_producer_link_and_evidence():
    result = _run_verifier(BUILD_GOLDEN)
    assert result.returncode == 0, result.stderr
    vector = load_build_golden()["npu_build"]
    assert "ok build artifact_set=" in result.stdout
    assert vector["artifact_set_digest"] in result.stdout
    assert vector["request_digest"] in result.stdout
    assert vector["receipt"]["receipt_id"] in result.stdout
    assert str(vector["producer_record"]["evidence_sha256"]) in result.stdout


def _resign_after_producer_change(golden: dict) -> None:
    """Re-sign a producer-record change so only its bound can reject the vector.

    The producer link is authenticated twice: by the request digest and by the
    digest of the producer's own evidence document.  A test that wants to check
    the *bounds* (rather than the digests) has to keep both consistent, which is
    what a hostile-but-consistent sender would do.
    """

    from tcenv.remote import contracts

    vector = golden["npu_build"]
    evidence_document = vector["producer_evidence"]
    old_record = copy.deepcopy(vector["producer_record"])
    record = vector["producer_record"]
    record["duration_ms"] = evidence_document["duration_ms"]
    record["log"] = copy.deepcopy(evidence_document["log"])
    record["evidence_sha256"] = hashlib.sha256(
        contracts.canonical_digest_document(evidence_document)
    ).hexdigest()
    vector["request"]["build"]["producer"] = copy.deepcopy(record)
    vector["evidence"]["build"]["producer"] = copy.deepcopy(record)
    # The frozen expected digest document has to describe the same request, or
    # the vector would be rejected for disagreeing with itself instead of for
    # the rule under test.
    vector["request_digest_document"]["request"]["build"]["producer"] = copy.deepcopy(record)

    old_json = contracts.canonical_digest_document(old_record).decode("utf-8")
    new_json = contracts.canonical_digest_document(record).decode("utf-8")
    canonical_json = vector["request_canonical_json"].replace(old_json, new_json, 1)
    assert canonical_json != vector["request_canonical_json"]
    vector["request_canonical_json"] = canonical_json
    digest = hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()
    vector["request_digest"] = digest
    vector["receipt"]["receipt_id"] = hashlib.sha256(
        contracts.canonical_digest_document(
            contracts.receipt_id_document(
                request_id=vector["receipt"]["request_id"],
                attempt_id=vector["receipt"]["attempt_id"],
                request_digest=digest,
                candidate_digest=vector["request"]["candidate_digest"],
                device_group=vector["request"]["execution"]["device_group"],
                dialect=contracts.NPU_BUILD_DIALECT,
            )
        )
    ).hexdigest()


@pytest.mark.parametrize("mutation,reason", [
    # The build identity is part of the request digest.
    ("provenance_builder_version", "build request"),
    ("provenance_toolchain_version", "build request"),
    ("provenance_source_sha256", "source_sha256 does not bind the SOURCE entry"),
    ("provenance_compile_profile", "build request"),
    ("build_mode", "build request"),
    ("build_schema_id", "build schema id is not the version 1 identity"),
    ("artifact_set_digest", "artifact_set_digest does not bind the artifact set"),
    # The producer link binds status/duration/log/digest of the producer row.
    ("producer_evidence_sha256", "build request"),
    ("producer_duration", "build request"),
    ("producer_log_size", "build request"),
    ("producer_status", "must describe a successful control-host compile"),
    ("producer_evidence_document", "does not bind the producer evidence document"),
    # The received build evidence must repeat the accepted row and cover the load.
    ("evidence_status_ok", "must be reported as reused"),
    ("evidence_producer_duration", "does not repeat the accepted producer link"),
    ("evidence_provenance_version", "differs from the accepted request"),
    ("evidence_loaded", "load record does not cover the executable modules"),
    ("evidence_receiver_log_is_producer_log", "log is the producer's log identity"),
    ("evidence_receiver_duration_zero", "must record its own duration"),
    ("evidence_artifact_set_digest", "artifact_set_digest does not bind its entries"),
    # A field the version-1 build evidence does not have is not extra credit.
    ("evidence_unexpected_field", "fields are missing or unknown"),
    # The evidence manifest is recomputed from its own entries and compared with
    # the accepted bundle, entry by entry: rewriting the SOURCE digest (or any
    # other entry) cannot survive by keeping the declared artifact-set digest.
    ("evidence_source_entry_sha256", "does not bind its entries"),
    ("evidence_manifest_rewritten_consistently", "entries differ from the accepted prebuilt bundle"),
    # Integer bounds and the receiver's own log are contract rules, not cosmetics.
    ("evidence_log_size_negative", "build.log.size_bytes"),
    ("evidence_duration_over_limit", "build.duration_ms"),
    ("evidence_log_null", "must record its own log"),
    # The producer link is a bounded record too; these vectors are re-signed so
    # only the bound can reject them.
    ("producer_duration_over_limit", "build.producer.duration_ms"),
    ("producer_log_size_over_limit", "build.producer.log.size_bytes"),
    ("receipt_request_digest", "receipt id does not bind the request"),
])
def test_the_go_verifier_rejects_a_tampered_build_vector(tmp_path, mutation, reason):
    golden = copy.deepcopy(load_build_golden())
    vector = golden["npu_build"]
    request = vector["request"]
    build = request["build"]
    provenance = build["provenance"]
    evidence_build = vector["evidence"]["build"]
    if mutation == "provenance_builder_version":
        provenance["builder_version"] = "9.9.9"
    elif mutation == "provenance_toolchain_version":
        provenance["toolchain"]["version"] = "0.0.0"
    elif mutation == "provenance_source_sha256":
        provenance["source_sha256"] = "0" * 64
    elif mutation == "provenance_compile_profile":
        provenance["compile_profile"]["opt_level"] = "O0"
    elif mutation == "build_mode":
        build["mode"] = "evaluator"
    elif mutation == "build_schema_id":
        build["schema_id"] = "tcenv.remote.npu-build.build-request.2"
    elif mutation == "artifact_set_digest":
        provenance["artifact_set_digest"] = "1" * 64
    elif mutation == "producer_evidence_sha256":
        build["producer"]["evidence_sha256"] = "2" * 64
        evidence_build["producer"]["evidence_sha256"] = "2" * 64
    elif mutation == "producer_duration":
        build["producer"]["duration_ms"] = 13
        evidence_build["producer"]["duration_ms"] = 13
    elif mutation == "producer_log_size":
        build["producer"]["log"]["size_bytes"] = 1
        evidence_build["producer"]["log"]["size_bytes"] = 1
    elif mutation == "producer_status":
        build["producer"]["status"] = "reused"
        evidence_build["producer"]["status"] = "reused"
    elif mutation == "producer_evidence_document":
        # The link is unchanged but the document it hashes is not.
        vector["producer_evidence"]["duration_ms"] = 99
    elif mutation == "evidence_status_ok":
        evidence_build["status"] = "ok"
    elif mutation == "evidence_producer_duration":
        evidence_build["producer"]["duration_ms"] = 99
    elif mutation == "evidence_provenance_version":
        evidence_build["provenance"]["builder_version"] = "9.9.9"
    elif mutation == "evidence_loaded":
        evidence_build["loaded"] = []
    elif mutation == "evidence_receiver_log_is_producer_log":
        evidence_build["log"] = copy.deepcopy(vector["producer_evidence"]["log"])
    elif mutation == "evidence_receiver_duration_zero":
        evidence_build["duration_ms"] = 0
    elif mutation == "evidence_artifact_set_digest":
        evidence_build["artifacts"]["artifact_set_digest"] = "3" * 64
    elif mutation == "evidence_unexpected_field":
        evidence_build["unexpected"] = "x"
    elif mutation == "evidence_source_entry_sha256":
        entry = next(
            item for item in evidence_build["artifacts"]["entries"] if item["role"] == "SOURCE"
        )
        entry["sha256"] = "0" * 64
    elif mutation == "evidence_log_size_negative":
        evidence_build["log"]["size_bytes"] = -1
    elif mutation == "evidence_duration_over_limit":
        evidence_build["duration_ms"] = 3_600_001
    elif mutation == "evidence_log_null":
        evidence_build["log"] = None
    elif mutation == "producer_duration_over_limit":
        vector["producer_evidence"]["duration_ms"] = 3_600_001
        _resign_after_producer_change(golden)
    elif mutation == "producer_log_size_over_limit":
        vector["producer_evidence"]["log"]["size_bytes"] = 1_000_001
        _resign_after_producer_change(golden)
    elif mutation == "evidence_manifest_rewritten_consistently":
        # The declared digest is recomputed, so the evidence is internally
        # consistent: only the comparison with the accepted bundle can see it.
        from tcenv.remote.artifact_bundle import BUNDLE_SCHEMA_ID, canonical_manifest
        from tcenv.remote.contracts import canonical_digest_document

        entries = evidence_build["artifacts"]["entries"]
        entry = next(item for item in entries if item["role"] == "SOURCE")
        entry["sha256"] = "5" * 64
        manifest = canonical_manifest(BUNDLE_SCHEMA_ID, entries)
        evidence_build["artifacts"]["artifact_set_digest"] = hashlib.sha256(
            canonical_digest_document(manifest)
        ).hexdigest()
    else:
        vector["receipt"]["receipt_id"] = "4" * 64
    path = tmp_path / "tampered_build.json"
    path.write_text(json.dumps(golden), encoding="utf-8")

    result = _run_verifier(path)
    assert result.returncode != 0, f"a tampered build vector was accepted: {mutation}"
    assert "verification failed" in result.stderr
    # Rejected for the reason under test, not incidentally by another check.
    assert reason in result.stderr, result.stderr
