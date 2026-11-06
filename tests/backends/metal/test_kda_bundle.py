"""Fixed-length KDA complex msl-bundle delivery checks (CPU preflight only)."""

from __future__ import annotations

from taic.task_binding import bind_task
from tcenv.backends.metal.search_candidate import preflight_submission
from tcenv.target import load_target
from tcenv.track.delivery import candidate_geometry
from tcenv.track.envelope import encode_envelope
from tcenv.track.plane import CandidateArtifact

from ...support.kda_fixed_bundle import DEVICE_SOURCE, HOST_ENTRY_PATH, bundle_entries, bundle_source

TARGET = load_target("apple-m4-max")


def _contract(size="small"):
    return bind_task("kda", "fp16", size).contract


def test_fixed_kda_bundle_preflight_preserves_native_host_and_three_stage_handoff():
    outcome = preflight_submission(
        _contract(), bundle_source(), target=TARGET, track_id="msl-bundle"
    )
    assert outcome.ok, outcome.rejection
    candidate = outcome.projection.candidate
    assert candidate.track_id == "msl-bundle"
    assert {entry.path for entry in candidate.entries} == {
        "device/host_piece_handoff.metal",
        "host/tiling.c",
        "host/execution_geometry.json",
    }
    geometry = candidate_geometry(candidate)
    assert [stage.kernel for stage in geometry.stages] == ["kda_piece0", "kda_piece1", "kda_piece2"]
    assert all(stage.grid == ("H", 1, 1) for stage in geometry.stages)
    assert all(stage.threads == (1, 1, 1) for stage in geometry.stages)
    assert all(
        [item["name"] for item in stage.workspace] == ["work_items", "item_counts", "handoff_states"]
        for stage in geometry.stages
    )
    assert all(item["zero"] is False for stage in geometry.stages for item in stage.workspace)


def test_fixed_kda_single_device_source_cannot_be_mislabelled_as_a_bundle():
    outcome = preflight_submission(
        _contract(), DEVICE_SOURCE, target=TARGET, track_id="msl-bundle"
    )
    assert not outcome.ok
    assert outcome.stage == "delivery"
    assert "not a complete candidate envelope" in outcome.rejection


def test_fixed_kda_bundle_requires_the_declared_host_program():
    entries = tuple(entry for entry in bundle_entries() if entry.path != HOST_ENTRY_PATH)
    submission = encode_envelope(CandidateArtifact(track_id="msl-bundle", entries=entries))
    outcome = preflight_submission(_contract(), submission, target=TARGET, track_id="msl-bundle")
    assert not outcome.ok
    assert outcome.stage == "delivery"
    assert "Host Tiling plan" in outcome.rejection
