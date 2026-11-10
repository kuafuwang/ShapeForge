"""Renderer-resolved prompt/submission styles for existing Tracks.

The returned style is generation-side metadata, not a new Track schema.  Model-visible
text stays in the fingerprinted functions that consume this style.  Unknown renderer
implementations fail closed rather than receiving MSL bundle instructions by accident.
"""

from __future__ import annotations

from dataclasses import dataclass

from tcenv.contract_errors import UnsupportedContractCapability
from tcenv.track.registry import TrackSpec, load_track_registry


@dataclass(frozen=True)
class TrackPromptStyle:
    style_id: str
    delivery: str
    explicit_track: bool
    envelope: bool


def prompt_style_for_track(track: TrackSpec) -> TrackPromptStyle:
    """Resolve the current renderer implementation, never infer style from a file suffix."""
    identity = (
        track.renderer.implementation_id,
        track.renderer.implementation_version,
        track.delivery,
    )
    if identity == ("metal.track.msl-single.renderer", 1, "single_source"):
        return TrackPromptStyle("msl-single", track.delivery, False, False)
    if identity == ("metal.track.msl-bundle.renderer", 1, "candidate_bundle"):
        return TrackPromptStyle("msl-bundle", track.delivery, True, True)
    if identity == ("metal.track.tilelang-pipeline.renderer", 1, "dsl_source_pipeline") and track.track_version in (2, 3):
        # /3 changes fresh-build identity and the native barrier interval, not the candidate DSL or
        # delivery wording. This is a new prompt epoch, recorded by MT6; MT4 evidence is not rewritten.
        return TrackPromptStyle("tilelang-pipeline", track.delivery, True, True)
    raise UnsupportedContractCapability(
        f"renderer {track.renderer.implementation_id}/{track.renderer.implementation_version} "
        f"for {track.track_id}/{track.track_version} has no registered prompt adapter"
    )


def prompt_style_for_id(track_id: str) -> TrackPromptStyle:
    """Compatibility lookup for a SessionBrief; unknown IDs fail before agent execution."""
    return prompt_style_for_track(load_track_registry().get(str(track_id)))


__all__ = ["TrackPromptStyle", "prompt_style_for_id", "prompt_style_for_track"]
