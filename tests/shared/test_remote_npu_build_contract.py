"""P1c build contract (document version 5) at the document level.

The end-to-end timings live in ``test_remote_npu_build_grpc.py`` and the outbox
row in ``test_remote_npu_build_outbox.py``; this file pins the document layer:
the build request/evidence rules, the version-5 execution shape, and — the part
that keeps P0/P0b/P1 intact — that every *earlier* contract version still
refuses the new shape instead of silently accepting it.
"""

from __future__ import annotations

import copy

import pytest

from tcenv.remote import contracts
from tests.support.remote_contract import build_build_golden


def _vector():
    return copy.deepcopy(build_build_golden()["npu_build"])


def test_a_wrong_source_media_type_is_a_contract_error_not_a_key_error():
    """Every task family has its own media type; the refusal must be typed.

    ``kernel_module`` exists only in the build contract's media-type table.  An
    error message that indexed the older table raised ``KeyError`` out of the
    validator, which is neither a ``ContractError`` nor a ``BuildError`` and
    would escape the caller's classification.
    """

    execution = _vector()["request"]["execution"]
    assert execution["task_family"] == "kernel_module"
    broken = dict(execution, source_media_type="application/octet-stream")
    with pytest.raises(contracts.ContractError, match="requires source_media_type"):
        contracts.npu_build_execution_document(broken)
    # The other two families keep their own messages through the same path.
    for family, media_type in (("kernel", "text/plain; charset=utf-8"), ("graph", "application/octet-stream")):
        document = dict(
            execution, task_family=family, artifact_kind="source" if family == "kernel" else "bundle",
            launch_kind="KERNEL" if family == "kernel" else "GRAPH",
            launch_subtype="SIMT" if family == "kernel" else "SERVICE",
            synchronization="stream-event" if family == "kernel" else "service-completion-token",
            source_media_type="application/json" if media_type != "application/json" else "text/plain; charset=utf-8",
        )
        with pytest.raises(contracts.ContractError):
            contracts.npu_build_execution_document(document)


def test_a_prebuilt_build_request_needs_its_producer_link_and_the_artifact_set():
    vector = _vector()
    build = vector["request"]["build"]

    without_producer = copy.deepcopy(build)
    without_producer.pop("producer")
    with pytest.raises(contracts.ContractError, match="producer"):
        contracts.build_request_document(without_producer)

    # A prebuilt provenance must name the artifact set the request submits; that
    # is a request-level consistency rule between the build document, the
    # bundle and the CandidateDigest.
    wrong_set = copy.deepcopy(vector["request"])
    wrong_set["build"]["provenance"]["artifact_set_digest"] = "0" * 64
    with pytest.raises(contracts.ContractError, match="artifact_set_digest"):
        contracts.request_digest_document(
            wrong_set, source_bytes=b"", dialect=contracts.NPU_BUILD_DIALECT,
        )

    # The producer link is a closed record: an added field is not admissible.
    extra = copy.deepcopy(build)
    extra["producer"]["extra"] = "x"
    with pytest.raises(contracts.ContractError):
        contracts.build_request_document(extra)


def test_a_build_document_is_refused_by_every_earlier_contract_version():
    """P0/P0b/P1 keep their exact field sets: the new field is not smuggled in."""

    vector = _vector()
    request = vector["request"]
    # Version 4 (the P1 bundle contract) does not know the `build` field.
    with pytest.raises(contracts.ContractError, match="build"):
        contracts.request_digest_document(
            request, source_bytes=b"", dialect=contracts.NPU_BUNDLE_DIALECT,
        )
    # Nor does it know the `kernel_module` task family.
    with pytest.raises(contracts.ContractError, match="task_family"):
        contracts.npu_bundle_execution_document(request["execution"])
    # The build request is only admissible through the build dialect, and there
    # the execution shape is the prebuilt module shape.
    document = contracts.request_digest_document(
        request, source_bytes=b"", dialect=contracts.NPU_BUILD_DIALECT,
    )
    assert document["document_version"] == 5
    assert request["execution"] in (document["request"]["execution"],) or (
        document["request"]["execution"]["task_family"] == "kernel_module"
    )


def test_a_signed_evidence_build_document_is_refused_by_the_bundle_dialect():
    vector = _vector()
    evidence = vector["evidence"]
    # The version-5 evidence is validated with the build expected identity...
    contracts.npu_evidence_document(
        evidence, source_bytes=None,
        device_group=vector["request"]["execution"]["device_group"],
        expected_resources=vector["request"]["execution"]["resources"],
        expected_build=vector["request"]["build"],
        request_artifact=vector["request"]["artifact"],
        measurement_kind="simulated", publication_allowed=False,
        report_status="ok", dialect=contracts.NPU_BUILD_DIALECT,
    )
    # ...and the version-4 dialect refuses the same build block instead of
    # validating only the fields it happens to know.
    with pytest.raises(contracts.ContractError):
        contracts.npu_evidence_document(
            evidence, source_bytes=None,
            device_group=vector["request"]["execution"]["device_group"],
            expected_resources=vector["request"]["execution"]["resources"],
            expected_artifact_set_digest=vector["artifact_set_digest"],
            measurement_kind="simulated", publication_allowed=False,
            report_status="ok", dialect=contracts.NPU_BUNDLE_DIALECT,
        )


def test_the_build_dialect_keeps_the_six_documented_rpc_names_and_versions():
    """The six RPC names, the artifact field numbers and the digests are frozen."""

    from pathlib import Path

    root = Path(contracts.__file__).resolve().parents[2]
    proto = (root / "proto/tcenv/remote/v1/evaluator.proto").read_text(encoding="utf-8")
    for rpc in (
        "rpc GetCapabilities(", "rpc Submit(", "rpc GetJob(",
        "rpc Watch(", "rpc Cancel(", "rpc Retry(",
    ):
        assert rpc in proto, rpc
    assert "service Evaluator" in proto
    # The build request is one new field number; nothing below it moved.
    assert "BuildRequest build = 27;" in proto
    assert "BuildProducerEvidence producer_evidence = 4;" in proto
    assert contracts.NPU_BUILD_DOCUMENT_VERSION == 5
    assert contracts.NPU_BUILD_TIMING_EVIDENCE_SCHEMA_ID.endswith("timing-evidence.3")
    assert contracts.NPU_BUNDLE_DOCUMENT_VERSION == 4
    assert contracts.NPU_BUNDLE_TIMING_EVIDENCE_SCHEMA_ID.endswith("timing-evidence.2")


def test_the_worker_resolves_only_the_registered_profile_not_the_request_one():
    """The in-attempt builder must not adopt the request's own identity.

    A worker that rebuilt its profile out of the request's ``compile_profile``
    could never fail the registry's probe/build comparison: any declared profile
    would become its own baseline.  The worker resolves the registered profile by
    name instead, so a request naming another toolchain/compile profile is
    refused before a build runs.
    """

    import hashlib
    from dataclasses import replace

    from tests.support.vendor_sim import module_builder as mb
    from tcenv.remote.artifact_builder import (
        ArtifactBuilderRegistry, BuildError, BuildProvenance, SourceBundle,
    )

    source = b"npu-kernel relu_kernel\n  tx_thread_id\n"
    registered = mb.evaluator_profile()
    genuine = BuildProvenance(
        profile=registered, source_sha256=hashlib.sha256(source).hexdigest(),
    ).to_dict()
    builder, _ = mb.builder_for_provenance(genuine)
    assert builder.probe().profile == registered
    ArtifactBuilderRegistry([builder]).build(
        source_bundle=SourceBundle(source), task_contract={}, provenance=genuine,
    )

    # Another compile profile names a build environment that is not registered.
    tampered = BuildProvenance(
        profile=replace(registered, compile_profile={**registered.compile_profile, "opt_level": "O0"}),
        source_sha256=hashlib.sha256(source).hexdigest(),
    ).to_dict()
    builder, _ = mb.builder_for_provenance(tampered)
    with pytest.raises(BuildError) as refused:
        ArtifactBuilderRegistry([builder]).build(
            source_bundle=SourceBundle(source), task_contract={}, provenance=tampered,
        )
    assert refused.value.error_class == "toolchain_profile"
    assert "compile_profile" in refused.value.detail

    # An unknown profile name is refused by name, not mapped onto a default.
    unknown = BuildProvenance(
        profile=mb.evaluator_profile(build_profile="not-registered-v9"),
        source_sha256=hashlib.sha256(source).hexdigest(),
    ).to_dict()
    with pytest.raises(BuildError, match="not registered"):
        mb.builder_for_provenance(unknown)


def test_a_received_row_must_record_its_own_duration_not_the_not_run_sentinel():
    """Both languages agree that a loaded artifact set cost real time.

    ``0`` is the duration of a row that never ran (a timeout's ``not_run``).  A
    ``reused`` row that verified and loaded an artifact set must record its own
    cost; otherwise an all-zero receiver row would look like an attempt that
    never touched the artifact.
    """

    vector = _vector()
    evidence = vector["evidence"]
    assert evidence["build"]["status"] == "reused"
    evidence["build"]["duration_ms"] = 0
    with pytest.raises(contracts.ContractError, match="must record its own duration"):
        contracts.npu_evidence_document(
            evidence, source_bytes=None,
            device_group=vector["request"]["execution"]["device_group"],
            expected_resources=vector["request"]["execution"]["resources"],
            expected_build=vector["request"]["build"],
            request_artifact=vector["request"]["artifact"],
            measurement_kind="simulated", publication_allowed=False,
            report_status="ok", dialect=contracts.NPU_BUILD_DIALECT,
        )
    # A row that never ran keeps the sentinel: that is what it means.
    not_run = copy.deepcopy(vector["evidence"])
    not_run["build"] = {
        "schema_id": contracts.NPU_BUILD_EVIDENCE_SCHEMA_ID,
        "mode": "prebuilt", "status": "not_run", "error_class": None,
        "provenance": vector["request"]["build"]["provenance"],
        "artifacts": None, "duration_ms": 0, "log": None, "loaded": [],
    }
    not_run["complete"] = False
    not_run["timing"] = dict(
        not_run["timing"], launches=0, device_elapsed_ns=0,
        synchronization_completed=False, clock_domain=contracts.NPU_UNOBSERVED_CLOCK_DOMAIN,
    )
    not_run["spans"] = []
    contracts.npu_evidence_document(
        not_run, source_bytes=None,
        device_group=vector["request"]["execution"]["device_group"],
        expected_resources=vector["request"]["execution"]["resources"],
        expected_build=vector["request"]["build"],
        request_artifact=vector["request"]["artifact"],
        measurement_kind="simulated", publication_allowed=False,
        report_status="timeout", dialect=contracts.NPU_BUILD_DIALECT,
    )


def test_a_received_row_records_a_bounded_own_duration_and_log():
    """The received row's own numbers live in the same domains as the producer's.

    ``duration_ms`` is milliseconds of the receiver's own verify/load pass and
    ``log.size_bytes`` is a bounded byte count; a negative size or a duration
    beyond the contract's ceiling is not "unusual", it is a different document.
    The receiver must also carry its own log at all: a reused row with no log
    cannot show that anything was read back.
    """

    vector = _vector()
    for name, reason in (
        ("duration_over_limit", "build.duration_ms"),
        ("log_size_negative", "build.log.size_bytes"),
        ("log_missing", "must record its own log"),
    ):
        evidence = copy.deepcopy(vector["evidence"])
        if name == "duration_over_limit":
            evidence["build"]["duration_ms"] = 3_600_001
        elif name == "log_size_negative":
            evidence["build"]["log"]["size_bytes"] = -1
        else:
            evidence["build"]["log"] = None
        with pytest.raises(contracts.ContractError, match=reason):
            contracts.npu_evidence_document(
                evidence, source_bytes=None,
                device_group=vector["request"]["execution"]["device_group"],
                expected_resources=vector["request"]["execution"]["resources"],
                expected_build=vector["request"]["build"],
                request_artifact=vector["request"]["artifact"],
                measurement_kind="simulated", publication_allowed=False,
                report_status="ok", dialect=contracts.NPU_BUILD_DIALECT,
            )


def test_the_producer_link_and_the_received_row_share_the_integer_domains():
    """One domain for both sides: a producer value the receiver may not repeat is
    not a valid link, and the request digest refuses it."""

    vector = _vector()
    for field, value, reason in (
        (("duration_ms",), 3_600_001, "build.producer.duration_ms"),
        (("log", "size_bytes"), 1_000_001, "build.producer.log.size_bytes"),
        (("log", "size_bytes"), -1, "build.producer.log.size_bytes"),
        (("duration_ms",), -1, "build.producer.duration_ms"),
    ):
        tampered = copy.deepcopy(vector["request"])
        target = tampered["build"]["producer"]
        for key in field[:-1]:
            target = target[key]
        target[field[-1]] = value
        with pytest.raises(contracts.ContractError, match=reason):
            contracts.request_digest_document(
                tampered, source_bytes=b"", dialect=contracts.NPU_BUILD_DIALECT,
            )
        # The receiver cannot repeat such a link either.
        with pytest.raises(contracts.ContractError, match=reason):
            contracts.build_producer_record(tampered["build"]["producer"])
