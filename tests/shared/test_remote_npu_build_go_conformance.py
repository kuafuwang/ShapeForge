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


def _resign_request_change(golden: dict, mutate) -> None:
    """Apply ``mutate`` to the request and to everything that authenticates it.

    The request document is authenticated twice over: the raw request is
    canonicalized into the digest document, and the receipt id is the digest of
    the receipt document that binds that request digest.  A mutation that wants
    to test a *rule* (rather than the digest) has to keep all of it consistent,
    which is what a hostile-but-consistent sender would do.
    """

    from tcenv.remote import contracts

    vector = golden["npu_build"]
    mutate(vector["request"])
    mutate(vector["request_digest_document"]["request"])
    canonical_json = contracts.canonical_digest_document(
        vector["request_digest_document"]
    ).decode("utf-8")
    vector["request_canonical_json"] = canonical_json
    digest = hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()
    vector["request_digest"] = digest
    # The receipt document is assembled here instead of through the production
    # helper: a mutation may target the very field that helper validates (the
    # device group, the candidate digest), and the vector still has to be signed
    # consistently enough for the rule under test to be the only reason it fails.
    receipt_document = {
        "document_type": contracts.NPU_BUILD_RECEIPT_DOCUMENT_TYPE,
        "document_version": contracts.NPU_BUILD_DOCUMENT_VERSION,
        "digest_algorithm": contracts.RECEIPT_DIGEST_ALGORITHM,
        "request_id": vector["receipt"]["request_id"],
        "attempt_id": vector["receipt"]["attempt_id"],
        "request_digest": digest,
        "candidate_digest": vector["request"]["candidate_digest"],
        "device_group": vector["request"]["execution"]["device_group"],
    }
    vector["receipt"]["receipt_id"] = hashlib.sha256(
        contracts.canonical_digest_document(receipt_document)
    ).hexdigest()


def _mutate_provenance(golden: dict, mutate) -> None:
    """Apply ``mutate`` to the accepted and the signed provenance alike.

    The request's provenance is authenticated by the request digest and repeated
    field for field in the signed evidence, so a provenance rule can only be
    tested when both copies (and the digest) agree.
    """

    _resign_request_change(golden, lambda request: mutate(request["build"]["provenance"]))
    mutate(golden["npu_build"]["evidence"]["build"]["provenance"])


def _manifest_core(entries: list[dict]) -> tuple[dict, str]:
    """The canonical manifest and its digest, computed without the entry domain.

    The vectors for the entry domain (an unknown role, a non-hex digest, a
    negative size, ...) cannot be built through the production helper, because it
    refuses exactly those entries.  The independent verifier still has to see
    them, so this builds the same canonical document by hand.
    """

    from tcenv.remote.artifact_bundle import BUNDLE_SCHEMA_ID, BUNDLE_SCHEMA_VERSION
    from tcenv.remote.contracts import canonical_digest_document

    manifest = {
        "schema_id": BUNDLE_SCHEMA_ID,
        "schema_version": BUNDLE_SCHEMA_VERSION,
        "entries": sorted(
            (
                {key: entry[key] for key in ("path", "role", "media_type", "size_bytes", "sha256")}
                for entry in entries
            ),
            key=lambda item: item["path"],
        ),
    }
    return manifest, hashlib.sha256(canonical_digest_document(manifest)).hexdigest()


def _rebuild_artifact_set(golden: dict, entries: list[dict], *, loaded=None) -> None:
    """Rewrite a *valid* artifact set through every document that binds it."""

    from tcenv.remote.artifact_bundle import BUNDLE_SCHEMA_ID, canonical_manifest

    canonical_manifest(BUNDLE_SCHEMA_ID, entries)
    _rebuild_artifact_set_unchecked(golden, entries, loaded=loaded)


def _rebuild_artifact_set_unchecked(golden: dict, entries: list[dict], *, loaded=None) -> None:
    """Rewrite the artifact set even when its entries are outside the domain.

    The bundle manifest, the artifact-set digest, the CandidateDigest, the build
    provenance, the signed evidence artifacts and the receiver's load record are
    all one chain; replacing a transport entry without recomputing the whole
    chain would only prove that some digest mismatch is detected.
    """

    from tcenv.remote.artifact_bundle import BUNDLE_SCHEMA_ID

    vector = golden["npu_build"]
    manifest, digest = _manifest_core(entries)
    # A vector that removes the SOURCE role on purpose has nothing to bind; the
    # provenance keeps its digest and the entry-domain rule is what must refuse it.
    first_source = next(
        (entry for entry in manifest["entries"] if entry["role"] == "SOURCE"), None,
    )

    def mutate(request):
        bundle = request["artifact"]["bundle"]
        bundle["entries"] = copy.deepcopy(entries)
        bundle["artifact_set_digest"] = digest
        request["candidate_digest"]["value"] = digest
        provenance = request["build"]["provenance"]
        provenance["artifact_set_digest"] = digest
        if first_source is not None:
            provenance["source_sha256"] = first_source["sha256"]

    _resign_request_change(golden, mutate)

    vector["artifact_set_digest"] = digest
    vector["bundle_manifest"] = manifest
    evidence = vector["evidence"]
    evidence["artifact_set_digest"] = digest
    evidence["candidate_digest"]["value"] = digest
    evidence["build"]["provenance"] = copy.deepcopy(vector["request"]["build"]["provenance"])
    evidence["build"]["artifacts"] = {
        "schema_id": BUNDLE_SCHEMA_ID,
        "entries": copy.deepcopy(manifest["entries"]),
        "artifact_set_digest": digest,
    }
    if loaded is None:
        loaded = [
            {"path": entry["path"], "sha256": entry["sha256"]}
            for entry in manifest["entries"] if entry["role"] == "EXECUTABLE_MODULE"
        ]
    evidence["build"]["loaded"] = copy.deepcopy(loaded)


def _inline_entry(path: str, role: str, media_type: str, data: bytes) -> dict:
    return {
        "path": path, "role": role, "media_type": media_type,
        "size_bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(),
        "content": "inline",
    }


def _resign_after_producer_change(golden: dict) -> None:
    """Re-sign a producer-record change so only its bound can reject the vector."""

    from tcenv.remote import contracts

    vector = golden["npu_build"]
    evidence_document = vector["producer_evidence"]
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
    _resign_request_change(golden, lambda request: None)


def _apply_mutation(golden: dict, mutation: str) -> None:
    """Apply one named tamper to a copy of the frozen build vector."""

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
        # Consistently re-signed: only the prebuilt-shape rule can see it.
        _resign_request_change(
            golden, lambda request: request["build"].__setitem__("mode", "evaluator"),
        )
    elif mutation == "provenance_build_host":
        # Neither side may pair prebuilt with an evaluator build host.
        _resign_request_change(
            golden,
            lambda request: request["build"]["provenance"].__setitem__(
                "build_host", "evaluator",
            ),
        )
        vector["evidence"]["build"]["provenance"]["build_host"] = "evaluator"
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
    elif mutation.startswith("provenance_") and mutation not in (
        "provenance_builder_version", "provenance_toolchain_version",
        "provenance_source_sha256", "provenance_compile_profile",
        "provenance_build_host",
    ):
        if mutation == "provenance_builder_id_type":
            _mutate_provenance(golden, lambda prov: prov.__setitem__("builder_id", 7))
        elif mutation == "provenance_builder_version_too_long":
            _mutate_provenance(golden, lambda prov: prov.__setitem__("builder_version", "v" * 65))
        elif mutation == "provenance_build_profile_type":
            _mutate_provenance(golden, lambda prov: prov.__setitem__("build_profile", 7))
        elif mutation == "provenance_toolchain_id_type":
            _mutate_provenance(golden, lambda prov: prov["toolchain"].__setitem__("id", 7))
        elif mutation == "provenance_toolchain_version_type":
            _mutate_provenance(golden, lambda prov: prov["toolchain"].__setitem__("version", []))
        elif mutation == "provenance_compile_profile_type":
            _mutate_provenance(golden, lambda prov: prov.__setitem__("compile_profile", []))
        else:
            _mutate_provenance(
                golden,
                lambda prov: prov.__setitem__(
                    "compile_profile", {f"opt_{index}": "v" for index in range(17)},
                ),
            )
    elif mutation == "kernel_module_source_media_type":
        _resign_request_change(
            golden,
            lambda request: request["execution"].__setitem__(
                "source_media_type", "application/octet-stream",
            ),
        )
    elif mutation == "second_source_entry":
        entries = copy.deepcopy(vector["request"]["artifact"]["bundle"]["entries"])
        entries.append(
            _inline_entry(
                # Same bytes as the first SOURCE, so the only illegal thing about
                # this vector is that there are two of them: a verifier that binds
                # whichever SOURCE it sees last still accepts it.
                "kernel2.tx", "SOURCE", "text/plain; charset=utf-8",
                b"npu-kernel relu_kernel\n  tx_thread_id\n",
            )
        )
        _rebuild_artifact_set(golden, entries)
    elif mutation == "duplicate_loaded_module":
        entries = copy.deepcopy(vector["request"]["artifact"]["bundle"]["entries"])
        entries.append(
            _inline_entry(
                "module/extra.txelf", "EXECUTABLE_MODULE", "application/x-tx-elf",
                b"TXELF-SIM\x00second fake module\n",
            )
        )
        module = next(entry for entry in entries if entry["role"] == "EXECUTABLE_MODULE")
        _rebuild_artifact_set(
            golden, entries,
            loaded=[
                {"path": module["path"], "sha256": module["sha256"]},
                {"path": module["path"], "sha256": module["sha256"]},
            ],
        )
    elif mutation == "resource_generation_zero":
        _resign_request_change(
            golden,
            lambda request: request["execution"]["device_group"].__setitem__(
                "resource_generation", "0",
            ),
        )
    elif mutation == "resource_generation_over_uint64":
        _resign_request_change(
            golden,
            lambda request: request["execution"]["device_group"].__setitem__(
                "resource_generation", "18446744073709551616",
            ),
        )
    elif mutation == "timing_sources_empty":
        _resign_request_change(
            golden,
            lambda request: request["execution"].__setitem__(
                "required_timing_sources", [],
            ),
        )
    elif mutation == "timing_sources_not_device_event":
        _resign_request_change(
            golden,
            lambda request: request["execution"].__setitem__(
                "required_timing_sources", ["HOST_WALL"],
            ),
        )
    elif mutation == "device_group_digest_not_hex":
        _resign_request_change(
            golden,
            lambda request: request["execution"]["device_group"].__setitem__(
                "profile_digest", "z" * 64,
            ),
        )
    elif mutation == "resource_generation_not_decimal":
        _resign_request_change(
            golden,
            lambda request: request["execution"]["device_group"].__setitem__(
                "resource_generation", "01",
            ),
        )
    elif mutation == "logical_devices_out_of_bounds":
        _resign_request_change(
            golden,
            lambda request: request["execution"]["resources"].__setitem__(
                "logical_devices", 9,
            ),
        )
    elif mutation == "execution_unknown_field":
        _resign_request_change(
            golden,
            lambda request: request["execution"].__setitem__(
                "unexpected", "not-in-the-contract",
            ),
        )
    elif mutation == "device_group_unknown_field":
        _resign_request_change(
            golden,
            lambda request: request["execution"]["device_group"].__setitem__(
                "unexpected", "not-in-the-contract",
            ),
        )
    elif mutation == "resources_unknown_field":
        _resign_request_change(
            golden,
            lambda request: request["execution"]["resources"].__setitem__(
                "unexpected", "not-in-the-contract",
            ),
        )
    elif mutation == "candidate_digest_unknown_field":
        _resign_request_change(
            golden,
            lambda request: request["candidate_digest"].__setitem__(
                "unexpected", "not-in-the-contract",
            ),
        )
    elif mutation == "bundle_unknown_field":
        _resign_request_change(
            golden,
            lambda request: request["artifact"]["bundle"].__setitem__(
                "unexpected", "not-in-the-contract",
            ),
        )
    elif mutation == "entry_unknown_field":
        _resign_request_change(
            golden,
            lambda request: request["artifact"]["bundle"]["entries"][0].__setitem__(
                "unexpected", "not-in-the-contract",
            ),
        )
    elif mutation.startswith("entry_") or mutation == "too_many_entries":
        entries = copy.deepcopy(vector["request"]["artifact"]["bundle"]["entries"])
        source = next(entry for entry in entries if entry["role"] == "SOURCE")
        if mutation == "entry_role_unsupported":
            # Keep exactly one SOURCE and one module: only the extra entry's role
            # is outside the domain, so nothing else can refuse the vector.
            entries.append(
                _inline_entry("aux/extra.bin", "AUXILIARY", "application/octet-stream", b"x")
            )
            entries[-1]["role"] = "NOT_A_ROLE"
        elif mutation == "entry_media_type_invalid":
            source["media_type"] = ""
        elif mutation == "entry_sha256_not_hex":
            source["sha256"] = "z" * 64
        elif mutation == "entry_size_negative":
            source["size_bytes"] = -1
        elif mutation == "entry_path_too_deep":
            source["path"] = "/".join(f"d{index}" for index in range(17)) + "/kernel.tx"
        elif mutation == "too_many_entries":
            for index in range(64):
                entries.append(
                    _inline_entry(
                        f"aux/{index:02d}.bin", "AUXILIARY", "application/octet-stream",
                        b"x",
                    )
                )
        else:
            # 60 entries of 300 kB declare 18 MB, over the 16 MB total limit.
            entries = [
                _inline_entry(
                    f"bulk/{index:02d}.bin", "AUXILIARY", "application/octet-stream", b"x",
                )
                for index in range(60)
            ]
            for entry in entries:
                entry["size_bytes"] = 300_000
            entries.append(
                _inline_entry(
                    "module/kernel.txelf", "EXECUTABLE_MODULE", "application/x-tx-elf", b"y",
                )
            )
            entries.append(
                _inline_entry(
                    "kernel.tx", "SOURCE", "text/plain; charset=utf-8", b"z",
                )
            )
        _rebuild_artifact_set_unchecked(golden, entries)
    else:
        vector["receipt"]["receipt_id"] = "4" * 64


@pytest.mark.parametrize("mutation,reason", [
    # The build identity is part of the request digest.
    ("provenance_builder_version", "build request"),
    ("provenance_toolchain_version", "build request"),
    ("provenance_source_sha256", "source_sha256 does not bind the SOURCE entry"),
    ("provenance_compile_profile", "build request"),
    ("build_mode", "requires the prebuilt build shape"),
    ("provenance_build_host", "requires build_host"),
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
    # Provenance value domains: identifiers, bounded text, toolchain and the
    # bounded compile profile.  Each is reached only after a consistent re-sign.
    ("provenance_builder_id_type", "build.provenance.builder_id must be a bounded opaque identifier"),
    ("provenance_builder_version_too_long", "build.provenance.builder_version exceeds its UTF-8 byte limit"),
    ("provenance_build_profile_type", "build.provenance.build_profile must be a bounded opaque identifier"),
    ("provenance_toolchain_id_type", "build.toolchain.id must be a bounded opaque identifier"),
    ("provenance_toolchain_version_type", "build.toolchain.version must be a non-empty string"),
    ("provenance_compile_profile_type", "compile_profile must be a bounded object"),
    ("provenance_compile_profile_too_many", "compile_profile must be a bounded object"),
    # The prebuilt module shape has one media type and exactly one SOURCE entry.
    ("kernel_module_source_media_type", "requires source_media_type"),
    ("second_source_entry", "requires exactly one SOURCE entry"),
    # The receiver's load record is a set: no duplicates, full coverage.
    ("duplicate_loaded_module", "build.loaded reports the same artifact twice"),
    # The artifact entries themselves are a closed domain: an unknown role, an
    # empty media type, a non-hex digest, a negative size, a too-deep path, too
    # many entries and an over-limit total are different documents.
    ("entry_role_unsupported", "role is not a supported public role"),
    ("entry_media_type_invalid", "media type is not a bounded ASCII media type"),
    ("entry_sha256_not_hex", "sha256 must be 64 lowercase hexadecimal characters"),
    ("entry_size_negative", "size is outside its integer bounds"),
    ("entry_path_too_deep", "path is too deep"),
    ("too_many_entries", "entry-count limit"),
    ("entry_total_bytes_over_limit", "declared total byte limit"),
    # The bundle and each entry are closed documents: an added field is a
    # different document, even though it never enters the manifest digest.
    ("bundle_unknown_field", "bundle fields are missing or unknown"),
    ("entry_unknown_field", "bundle entry has unknown fields"),
    # ...and so is every other document the request binds.
    ("execution_unknown_field", "execution spec fields are missing or unknown"),
    ("device_group_unknown_field", "device_group fields are missing or unknown"),
    ("resources_unknown_field", "execution.resources fields are missing or unknown"),
    ("candidate_digest_unknown_field", "candidate_digest fields are missing or unknown"),
    # ...and the values inside those documents are typed too.
    ("device_group_digest_not_hex", "device_group.profile_digest must be 64 lowercase hexadecimal characters"),
    ("resource_generation_not_decimal", "resource_generation must be a canonical unsigned decimal string"),
    ("logical_devices_out_of_bounds", "execution.resources.logical_devices"),
    # resource_generation is a uint64 with minimum 1, not a 20-digit shape.
    ("resource_generation_zero", "resource_generation is outside its unsigned 64-bit bounds"),
    ("resource_generation_over_uint64", "resource_generation is outside its unsigned 64-bit bounds"),
    # required_timing_sources is a non-empty array of known sources.
    ("timing_sources_empty", "requires at least one timing source"),
    ("timing_sources_not_device_event", "supports only DEVICE_EVENT timing"),
])
def test_the_go_verifier_rejects_a_tampered_build_vector(tmp_path, mutation, reason):
    golden = copy.deepcopy(load_build_golden())
    _apply_mutation(golden, mutation)
    path = tmp_path / "tampered_build.json"
    path.write_text(json.dumps(golden), encoding="utf-8")

    result = _run_verifier(path)
    assert result.returncode != 0, f"a tampered build vector was accepted: {mutation}"
    assert "verification failed" in result.stderr
    # Rejected for the reason under test, not incidentally by another check.
    assert reason in result.stderr, result.stderr


@pytest.mark.parametrize("mutation,level", [
    # Request-level rules: the provenance value domains, the derived media type
    # and the single-SOURCE identity all live in the request document.
    ("provenance_builder_id_type", "request"),
    ("provenance_builder_version_too_long", "request"),
    ("provenance_build_profile_type", "request"),
    ("provenance_toolchain_id_type", "request"),
    ("provenance_toolchain_version_type", "request"),
    ("provenance_compile_profile_type", "request"),
    ("provenance_compile_profile_too_many", "request"),
    ("kernel_module_source_media_type", "request"),
    ("second_source_entry", "request"),
    ("build_mode", "request"),
    ("provenance_build_host", "request"),
    # The receiver's load record is checked with the signed evidence.
    ("duplicate_loaded_module", "evidence"),
    # The artifact-entry domain is a request-level rule.
    ("entry_role_unsupported", "request"),
    ("entry_media_type_invalid", "request"),
    ("entry_sha256_not_hex", "request"),
    ("entry_size_negative", "request"),
    ("entry_path_too_deep", "request"),
    ("too_many_entries", "request"),
    ("entry_total_bytes_over_limit", "request"),
    ("bundle_unknown_field", "request"),
    ("entry_unknown_field", "request"),
    ("execution_unknown_field", "request"),
    ("device_group_unknown_field", "request"),
    ("resources_unknown_field", "request"),
    ("candidate_digest_unknown_field", "request"),
    ("device_group_digest_not_hex", "request"),
    ("resource_generation_not_decimal", "request"),
    ("logical_devices_out_of_bounds", "request"),
    ("resource_generation_zero", "request"),
    ("resource_generation_over_uint64", "request"),
    ("timing_sources_empty", "request"),
    ("timing_sources_not_device_event", "request"),
])
def test_the_python_contract_rejects_the_same_tampered_vector(mutation, level):
    """The Go verifier is not stricter or looser than the production contract.

    Each vector here is the frozen golden with one named mutation applied (and,
    where the mutation targets an authenticated field, consistently re-signed).
    The Python contract must refuse the same input at the same layer, otherwise
    the two languages disagree about what the document means.
    """

    from tcenv.remote import contracts

    golden = copy.deepcopy(load_build_golden())
    _apply_mutation(golden, mutation)
    vector = golden["npu_build"]
    if level == "request":
        with pytest.raises(contracts.ContractError):
            contracts.request_digest_document(
                vector["request"], source_bytes=b"", dialect=contracts.NPU_BUILD_DIALECT,
            )
        return
    with pytest.raises(contracts.ContractError):
        contracts.npu_evidence_document(
            vector["evidence"], source_bytes=None,
            device_group=vector["request"]["execution"]["device_group"],
            expected_resources=vector["request"]["execution"]["resources"],
            expected_build=vector["request"]["build"],
            request_artifact=vector["request"]["artifact"],
            measurement_kind="simulated", publication_allowed=False,
            report_status="ok", dialect=contracts.NPU_BUILD_DIALECT,
        )


def test_the_uint64_boundary_generation_is_valid_on_both_sides(tmp_path):
    """The bound is inclusive: 18446744073709551615 is a legal generation."""

    golden = copy.deepcopy(load_build_golden())
    _resign_request_change(
        golden,
        lambda request: request["execution"]["device_group"].__setitem__(
            "resource_generation", "18446744073709551615",
        ),
    )
    path = tmp_path / "boundary_generation.json"
    path.write_text(json.dumps(golden), encoding="utf-8")
    result = _run_verifier(path)
    assert result.returncode == 0, result.stderr
    assert "ok build" in result.stdout

    from tcenv.remote import contracts

    vector = golden["npu_build"]
    document = contracts.request_digest_document(
        vector["request"], source_bytes=b"", dialect=contracts.NPU_BUILD_DIALECT,
    )
    assert contracts.canonical_digest_document(document).decode("utf-8") == (
        vector["request_canonical_json"]
    )


def test_a_duplicated_timing_source_normalises_like_the_contract(tmp_path):
    """Both sides canonicalise the array (sorted, unique), not just validate it.

    The frozen digest document already holds the canonical ``["DEVICE_EVENT"]``,
    so a raw request that repeats the source can only verify when the verifier
    applies the same normalisation the production contract applies.
    """

    golden = copy.deepcopy(load_build_golden())
    vector = golden["npu_build"]
    vector["request"]["execution"]["required_timing_sources"] = [
        "DEVICE_EVENT", "DEVICE_EVENT",
    ]
    path = tmp_path / "duplicate_timing_source.json"
    path.write_text(json.dumps(golden), encoding="utf-8")
    result = _run_verifier(path)
    assert result.returncode == 0, result.stderr

    from tcenv.remote import contracts

    document = contracts.request_digest_document(
        vector["request"], source_bytes=b"", dialect=contracts.NPU_BUILD_DIALECT,
    )
    assert document["request"]["execution"]["required_timing_sources"] == ["DEVICE_EVENT"]
    assert contracts.canonical_digest_document(document).decode("utf-8") == (
        vector["request_canonical_json"]
    )
