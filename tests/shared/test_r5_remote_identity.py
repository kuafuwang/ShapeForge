"""R5: the remote (dialect) half of the confirmation identity comparison.

CPU only: the contexts here are the persisted outbox shapes (projection / row), and the expected
document is the host-side assembly.  Nothing probes a device and nothing reads the host's own
library versions for a remote measurement.
"""

from __future__ import annotations

import hashlib

import pytest

from taic.policy import (
    DEVICE_GROUP_FIELDS,
    IDENTITY_SCOPES,
    compare_confirmation_identity,
    remote_context_view,
)
from tcenv.evaluate import _evaluation_fingerprint, remote_confirmation_expected
from tests.support.v1_tasks import v1_task as get_task
from tests.support.v1_tasks import V1_TASK

SOURCE = "kernel void k() {}\n"
SOURCE_SHA = hashlib.sha256(SOURCE.encode("utf-8")).hexdigest()
PACK = "11" * 32
FROZEN_ENV = "22" * 32
DEVICE_GROUP = {
    "device_group_id": "npu-0", "vendor_id": "ascend",
    "profile_digest": "33" * 32, "topology_digest": "44" * 32, "resource_generation": "3",
}
CANDIDATE_DIGEST = {"algorithm": "sha256", "value": SOURCE_SHA}
BUILD_REQUEST = {"schema_id": "npu.build.request/3", "mode": "evaluator", "provenance": {"builder_id": "b"}}
EPOCH = "e" * 64


def _task():
    return get_task(V1_TASK, "fp16", "small")


def _expected(**over) -> dict:
    document = remote_confirmation_expected(
        V1_TASK, "fp16", "small", "apple-m4-max", SOURCE,
        task_pack_digest=PACK, baseline_environment_digest=FROZEN_ENV,
        build_provenance=BUILD_REQUEST, device_group=DEVICE_GROUP,
        evaluator_epoch=EPOCH, candidate_digest=CANDIDATE_DIGEST,
    )
    document.update(over)
    return document


def _report(expected: dict) -> dict:
    """A schema-4 report projection whose identity equals ``expected``."""
    build, evaluation = expected["build"], expected["evaluation"]
    return {
        "build": {
            "shape": "full", "build_kind": "native",
            "candidate_sha256": build["candidate_sha256"],
            "opspec_digest": build.get("opspec_digest"),
            "lowering_profile_digest": build.get("lowering_profile_digest"),
            "provenance": build.get("provenance"),
        },
        "evaluation_fingerprint": evaluation["evaluation_fingerprint"],
        "baseline_implementation_digest": evaluation["baseline_implementation_digest"],
        "baseline_environment_digest": evaluation["baseline_environment_digest"],
        "protocol_version": evaluation["protocol_version"],
        "target_id": evaluation["target_id"],
        "backend": evaluation["backend"],
        "backend_version": evaluation["backend_version"],
    }


def _context(*, group=DEVICE_GROUP, epoch=EPOCH, candidate=CANDIDATE_DIGEST, build=BUILD_REQUEST) -> dict:
    return {
        "request_id": "req-1", "request_digest": "55" * 32, "candidate": candidate,
        "evaluator_epoch": epoch, "device_group": group, "build_provenance": build,
    }


def test_the_scope_is_explicit_and_a_local_comparison_cannot_carry_a_context():
    assert IDENTITY_SCOPES == ("local", "remote")
    report = _report(_expected())
    with pytest.raises(ValueError, match="scope"):
        compare_confirmation_identity(
            candidate_sha256=SOURCE_SHA, expected=_expected(), search=report, confirmation=report,
            scope="remote-ish",
        )
    with pytest.raises(ValueError, match="local"):
        compare_confirmation_identity(
            candidate_sha256=SOURCE_SHA, expected=_expected(), search=report, confirmation=report,
            scope="local", search_context=_context(),
        )


def test_a_remote_match_needs_the_expected_the_reports_and_the_context():
    expected = _expected()
    report = _report(expected)
    got = compare_confirmation_identity(
        candidate_sha256=SOURCE_SHA, expected=expected, search=report, confirmation=report,
        scope="remote", search_context=_context(), confirmation_context=_context(),
    )
    assert got["kind"] == "match", got["differences"]
    assert got["differences"] == []


@pytest.mark.parametrize(
    "field,context",
    [
        ("device_group.profile_digest", dict(_context(), device_group={**DEVICE_GROUP, "profile_digest": "99" * 32})),
        ("device_group.topology_digest", dict(_context(), device_group={**DEVICE_GROUP, "topology_digest": "99" * 32})),
        ("device_group.resource_generation", dict(_context(), device_group={**DEVICE_GROUP, "resource_generation": "4"})),
        ("evaluator_epoch", dict(_context(), evaluator_epoch="f" * 64)),
        ("candidate_digest", dict(_context(), candidate={"algorithm": "sha256", "value": "99" * 32})),
        ("build_provenance", dict(_context(), build_provenance={"schema_id": "x", "mode": "prebuilt"})),
    ],
)
def test_a_changed_context_component_is_a_build_mismatch(field, context):
    """The persisted context is a third party: a changed device, epoch or build identity fails."""
    expected = _expected()
    report = _report(expected)
    got = compare_confirmation_identity(
        candidate_sha256=SOURCE_SHA, expected=expected, search=report, confirmation=report,
        scope="remote", search_context=_context(), confirmation_context=context,
    )
    assert got["kind"] == "build_identity_mismatch"
    assert field in [item["field"] for item in got["differences"] if item["kind"] == "mismatch"]


def test_a_missing_confirmation_context_is_unknown_not_a_match():
    """F-10: no row/result for this batch means the remote-only fields cannot be compared."""
    expected = _expected()
    report = _report(expected)
    got = compare_confirmation_identity(
        candidate_sha256=SOURCE_SHA, expected=expected, search=report, confirmation=report,
        scope="remote", search_context=_context(), confirmation_context=None,
    )
    assert got["kind"] == "build_identity_unknown"
    unknown = {item["field"] for item in got["differences"] if item["kind"] == "unknown"}
    assert {"candidate_digest", "evaluator_epoch", "build_provenance"} <= unknown
    assert any(field.startswith("device_group.") for field in unknown)


def test_a_local_comparison_never_looks_at_remote_fields():
    """The local path's difference list is unchanged by the remote additions."""
    expected = _expected()
    report = _report(expected)
    got = compare_confirmation_identity(
        candidate_sha256=SOURCE_SHA, expected=expected, search=report, confirmation=report,
        scope="local",
    )
    assert not [item for item in got["differences"] if "device_group" in item["field"] or item["field"] == "evaluator_epoch"]


def test_the_remote_expected_uses_the_frozen_environment_not_the_host_libraries():
    """The environment component and the fingerprint come from the frozen value, not this machine."""
    expected = _expected()
    evaluation = expected["evaluation"]
    assert evaluation["baseline_environment_digest"] == FROZEN_ENV
    assert evaluation["task_pack_digest"] == PACK
    task = _task()
    assert evaluation["evaluation_fingerprint"] == _evaluation_fingerprint(
        task, "metal", "apple-m4-max", task_pack_digest=PACK, environment_digest=FROZEN_ENV,
    )
    # The host's own environment digest is a different value; using it would silently disagree.
    assert evaluation["evaluation_fingerprint"] != _evaluation_fingerprint(task, "metal", "apple-m4-max")


def test_the_remote_expected_carries_the_persisted_capability_not_the_report():
    """CandidateDigest / device group / epoch / build request come from the validated context."""
    expected = _expected()
    assert expected["build"]["candidate_digest"] == CANDIDATE_DIGEST
    assert expected["build"]["device_group"] == DEVICE_GROUP
    assert expected["evaluation"]["evaluator_epoch"] == EPOCH
    assert expected["build"]["build_provenance"] == BUILD_REQUEST


def test_an_unknown_environment_makes_the_fingerprint_null_so_nothing_can_publish():
    """A contract-allowed ``null`` environment is unknown, not a licence to recompute it locally."""
    expected = remote_confirmation_expected(
        V1_TASK, "fp16", "small", "apple-m4-max", SOURCE,
        task_pack_digest=PACK, baseline_environment_digest=None,
    )
    assert expected["evaluation"]["baseline_environment_digest"] is None
    assert expected["evaluation"]["evaluation_fingerprint"] is None
    report = _report(_expected())
    got = compare_confirmation_identity(
        candidate_sha256=SOURCE_SHA, expected=expected, search=report, confirmation=report,
        scope="remote", search_context=_context(), confirmation_context=_context(),
    )
    assert got["kind"] != "match"
    evaluation_unknown = {
        item["field"] for item in got["differences"]
        if item["domain"] == "evaluation" and item["kind"] == "unknown"
    }
    assert "evaluation_fingerprint" in evaluation_unknown
    assert "baseline_environment_digest" in evaluation_unknown


def test_the_device_group_fields_are_the_closed_set_the_plan_names():
    assert DEVICE_GROUP_FIELDS == ("profile_digest", "topology_digest", "resource_generation")
    view = remote_context_view(_context())
    assert set(view) == {"candidate_digest", "evaluator_epoch", "device_group", "build_provenance"}
    assert remote_context_view(None) is None
