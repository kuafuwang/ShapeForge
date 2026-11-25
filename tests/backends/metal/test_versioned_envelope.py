"""Versioned delivery boundaries that belong to the Metal search/confirmation path."""

from __future__ import annotations

from dataclasses import replace

import pytest

from taic.task_binding import bind_task
from taic.v2_evaluation import V2LocalEvaluator
from tcenv.backends.metal.search_candidate import confirmation_input
from tcenv.contract_errors import UnsupportedContractCapability
from tcenv.target import load_target
from tcenv.track import CandidateArtifact, CandidateEntry, encode_envelope


def _versioned_source() -> str:
    candidate = CandidateArtifact(
        track_id="msl-bundle", track_version=1,
        entries=(
            CandidateEntry(path="kernel.metal", role="device_kernel", media_type="text/plain", data=b"kernel\n"),
            CandidateEntry(path="host/tiling.metal", role="host_tiling", media_type="text/plain", data=b"host\n"),
            CandidateEntry(path="host/execution_geometry.json", role="schedule", media_type="application/json", data=b"{}\n"),
        ),
    )
    return encode_envelope(candidate)


def test_versioned_confirmation_is_rejected_before_identity_rebuild_or_build():
    """Versioned remote candidates must not silently downgrade to the unversioned confirmation hash."""
    source = _versioned_source()
    binding = bind_task("rmsnorm", "fp16", "small", target_id="apple-m4-max")
    with pytest.raises(UnsupportedContractCapability, match="quick/formal"):
        confirmation_input(
            binding.locator, source, target=load_target("apple-m4-max"),
            guidance_digest="0" * 64, master_seed=7, track_id="msl-bundle", track_version=1,
        )
    evaluator = V2LocalEvaluator(replace(binding, track_id="msl-bundle", track_version=1))
    with pytest.raises(UnsupportedContractCapability, match="quick/formal"):
        evaluator.confirmation_expected(
            task="rmsnorm", dtype="fp16", size="small", target_id="apple-m4-max", source=source,
        )
