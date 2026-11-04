"""§1.5 回归：多 device 源拼接的顺序无关性与重复 launch 块的拦截。

``combined_candidate_source`` used to call ``sorted(...)`` on ``CandidateEntry`` objects, which are
not orderable, so any two-device bundle raised ``TypeError`` at the first build.  The fix sorts by
path.  These tests pin the observable consequences: the join is deterministic, the candidate
identity does not depend on the order the entries were delivered in, and the entries are joined in
path order so a compiler sees one stable compilation unit.
"""

from __future__ import annotations

import pytest

from tcenv.backends.metal.track_adapter import combined_candidate_source
from tcenv.contract_errors import TaskContractError
from tcenv.track.plane import CandidateArtifact, CandidateEntry
from tests.support.shape_portfolio_metal import DEVICE_A, DEVICE_B, HOST_TILING_A, geometry_bytes, geometry_document

GEOMETRY = geometry_bytes(geometry_document("variant_a"))


def _entries(*, reverse: bool = False) -> tuple[CandidateEntry, ...]:
    devices = [
        CandidateEntry(path="device/variant_a.metal", role="device_kernel", media_type="text/plain", data=DEVICE_A),
        CandidateEntry(path="device/variant_b.metal", role="device_kernel", media_type="text/plain", data=DEVICE_B),
    ]
    if reverse:
        devices.reverse()
    return tuple(
        devices
        + [
            CandidateEntry(
                path="host/tiling.json", role="host_tiling", media_type="application/json", data=HOST_TILING_A
            ),
            CandidateEntry(
                path="host/execution_geometry.json", role="schedule", media_type="application/json", data=GEOMETRY
            ),
        ]
    )


def test_two_device_sources_are_joined_rather_than_raising_a_type_error():
    joined = combined_candidate_source(CandidateArtifact(track_id="msl-bundle", entries=_entries()))
    assert "variant_a" in joined and "variant_b" in joined
    # Path order, not delivery order: the compiler sees one stable unit.
    assert joined.index("variant_a") < joined.index("variant_b")


def test_the_join_is_identical_whatever_order_the_entries_arrived_in():
    forward = combined_candidate_source(CandidateArtifact(track_id="msl-bundle", entries=_entries()))
    backward = combined_candidate_source(CandidateArtifact(track_id="msl-bundle", entries=_entries(reverse=True)))
    assert forward == backward


def test_reordering_the_entries_does_not_change_the_candidate_identity():
    forward = CandidateArtifact(track_id="msl-bundle", entries=_entries())
    backward = CandidateArtifact(track_id="msl-bundle", entries=_entries(reverse=True))
    assert forward.candidate_sha256 == backward.candidate_sha256
    # The join order is the path order, so both produce the same build input bytes too.
    assert forward.entries[0].sha256 != forward.entries[1].sha256


def test_a_bundle_needs_exactly_one_host_tiling_plan():
    duplicated = CandidateArtifact(
        track_id="msl-bundle",
        entries=(
            CandidateEntry(path="device/variant_a.metal", role="device_kernel", media_type="text/plain", data=DEVICE_A),
            CandidateEntry(path="host/tiling.json", role="host_tiling", media_type="application/json", data=HOST_TILING_A),
            CandidateEntry(
                path="host/tiling_2.json", role="host_tiling", media_type="application/json", data=HOST_TILING_A
            ),
            CandidateEntry(
                path="host/execution_geometry.json", role="schedule", media_type="application/json", data=GEOMETRY
            ),
        ),
    )
    with pytest.raises(TaskContractError, match="exactly one host_tiling"):
        combined_candidate_source(duplicated)
