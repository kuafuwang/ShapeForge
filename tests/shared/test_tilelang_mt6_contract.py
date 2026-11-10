"""MT6 CPU-only TileLang /3 fresh-identity contract slice.

No registry entry, Metal builder, frontend execution, GPU or model call is used here.  The
purpose is to keep /1 and /2 byte-identical while separating a stable pre-build subject from
attempt-specific source-only evidence for the future fresh-confirmation Track version.
"""

from __future__ import annotations

import pytest

from tcenv.contract_errors import UnsupportedContractCapability
from tcenv.identity import fingerprint
from tcenv.track import load_track_registry
from tcenv.track import tilelang as contract
from tcenv.track import tilelang_confirmation as fresh
from tests.shared.test_track_mt1b_tilelang_contract import (
    ATTEMPT,
    GOOD_ENVIRONMENT,
    GOOD_FRONTEND,
    R5_CANDIDATE,
    _production_artifacts,
    _source_candidate,
)


def test_mt6_v3_is_a_new_interface_only_contract_and_v1_v2_digests_do_not_move():
    v1 = contract.frozen_contract(track_version=1)
    v2 = contract.frozen_contract(track_version=2)
    v3 = fresh.frozen_confirmation_contract()
    assert fingerprint(v1) == "689f86802ed8e040a37461677c4ffb922b9be1b8ea4c3e73b18faeeed9100022"
    assert fingerprint(v2) == "03541a31c94ef815de38c43e1ed260fd31c7c74955598c3652031d641429d1df"
    assert fresh.TILELANG_CONFIRMATION_TRACK_VERSION == 3
    assert fingerprint(v3) not in {fingerprint(v1), fingerprint(v2)}
    assert v3["build"]["implemented_modes"] == ["source_only"]
    assert v3["evidence_boundary"]["availability"].startswith("interface_only")
    registry = load_track_registry()
    document = registry.document()
    pre_v3_document = dict(
        document,
        tracks=[
            row
            for row in document["tracks"]
            if not (row["track_id"] == contract.TILELANG_TRACK_ID and row["track_version"] == 3)
        ],
    )
    assert fingerprint(pre_v3_document) == "596474db2d9132c16998d6f406eb1f10286acf86e501da734caa958cb4350fec"
    assert fingerprint(document) == "9b85fa111f7ccde5c95dfc61b84c2c2c6a142ae5cc8e171c0fae340ac042d138"
    assert registry.get(contract.TILELANG_TRACK_ID, version=3).status == "available"


def test_mt6_stable_subject_survives_attempt_identity_changes():
    """The pre-build cache key is stable; full source-only identity is attempt evidence.

    A fresh confirmation must put the former in BuildProfile provenance and keep the latter in
    the native descriptor.  Comparing two attempt-specific full identities as toolchain
    provenance would reject the same candidate before correctness or timing is considered.
    """
    source = R5_CANDIDATE.read_bytes()
    candidate = _source_candidate(source=source)
    artifacts = _production_artifacts(source)
    subject = fresh.expected_fresh_subject(candidate=candidate, environment=GOOD_ENVIRONMENT)
    first = contract.source_only_build_identity(
        candidate=candidate,
        observed_frontend=GOOD_FRONTEND,
        environment=GOOD_ENVIRONMENT,
        artifacts=artifacts,
        attempt=ATTEMPT,
        expected_kernel_names=("scale_kernel",),
        track_version=contract.TILELANG_METAL_TRACK_VERSION,
    )
    second_attempt = dict(ATTEMPT, request_sha256="c" * 64)
    second = contract.source_only_build_identity(
        candidate=candidate,
        observed_frontend=GOOD_FRONTEND,
        environment=GOOD_ENVIRONMENT,
        artifacts=artifacts,
        attempt=second_attempt,
        expected_kernel_names=("scale_kernel",),
        track_version=contract.TILELANG_METAL_TRACK_VERSION,
    )
    assert subject.full_build is False
    assert first.full_build is True and second.full_build is True
    assert first.cache_key == second.cache_key == subject.source_only_subject_cache_key
    assert subject.cache_key != subject.source_only_subject_cache_key
    assert first.digest != second.digest
    assert first.artifact_set_sha256 == second.artifact_set_sha256
    assert first.attempt != second.attempt


def test_mt6_v3_subject_binds_candidate_geometry_and_version():
    candidate = _source_candidate()
    changed_geometry = _source_candidate(grid=8)
    subject = fresh.expected_fresh_subject(candidate=candidate, environment=GOOD_ENVIRONMENT)
    assert subject.cache_key != fresh.expected_fresh_subject(
        candidate=changed_geometry, environment=GOOD_ENVIRONMENT
    ).cache_key
    assert subject.cache_key != subject.source_only_subject_cache_key
    assert subject.cache_key != fresh.expected_fresh_subject(
        candidate=candidate,
        environment=dict(GOOD_ENVIRONMENT, machine="x86_64"),
    ).cache_key
    assert subject.cache_key != fresh.expected_fresh_subject(
        candidate=_source_candidate(source=b"import tilelang\n# candidate drift\n"),
        environment=GOOD_ENVIRONMENT,
    ).cache_key
