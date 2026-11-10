"""Fail-closed construction of the registered Metal Track adapters.

Only construction mechanics live here.  This is not a DSL registry and does not make a newly
named Track implemented.  The closed tuples below are the already-published renderer/validator/
build identities in ``backends/metal/tracks.json``; any other identity is rejected instead of
being routed through an MSL contract.

MT1b adds the TileLang ``kernel_pipeline`` Track to the same construction path: its services are
the TileLang contract validator/builder and the shared candidate-geometry resolver, while its
evaluator/confirmation services resolve to a typed refusal.  The factory never substitutes an MSL
adapter for a DSL Track and never reads a default registry.
"""

from __future__ import annotations

from typing import Any

from ...contract_errors import UnsupportedContractCapability
from ...track.adapter import LazyTrackAdapter
from ...track.registry import TrackSpec


def _implementation_tuple(spec: TrackSpec) -> tuple:
    return (
        spec.renderer.implementation_id,
        spec.renderer.implementation_version,
        spec.validator.implementation_id,
        spec.validator.implementation_version,
        spec.build_adapter.implementation_id,
        spec.build_adapter.implementation_version,
        spec.delivery,
    )


_SINGLE = (
    "metal.track.msl-single.renderer",
    1,
    "metal.track.msl-single.validator",
    1,
    "metal.track.msl-single.build",
    1,
    "single_source",
)
_BUNDLE = (
    "metal.track.msl-bundle.renderer",
    1,
    "metal.track.msl-bundle.validator",
    1,
    "metal.track.msl-bundle.build",
    1,
    "candidate_bundle",
)
_TILELANG = (
    "metal.track.tilelang-pipeline.renderer",
    1,
    "metal.track.tilelang-pipeline.validator",
    1,
    "metal.track.tilelang-pipeline.build",
    1,
    "dsl_source_pipeline",
)
_TILELANG_V2 = (
    "metal.track.tilelang-pipeline.renderer",
    1,
    "metal.track.tilelang-pipeline.validator",
    1,
    "metal.track.tilelang-pipeline.build",
    2,
    "dsl_source_pipeline",
)
_TILELANG_V3 = (
    "metal.track.tilelang-pipeline.renderer",
    1,
    "metal.track.tilelang-pipeline.validator",
    1,
    "metal.track.tilelang-pipeline.build",
    3,
    "dsl_source_pipeline",
)


def _profile(spec: TrackSpec) -> str:
    identity = _implementation_tuple(spec)
    if identity == _SINGLE:
        return "msl-single"
    if identity == _BUNDLE:
        return "msl-bundle"
    if (identity == _TILELANG and spec.track_version == 1) or (
        identity == _TILELANG_V2 and spec.track_version == 2
    ) or (identity == _TILELANG_V3 and spec.track_version == 3):
        return "tilelang-pipeline"
    raise UnsupportedContractCapability(
        f"no Metal adapter implements the registered identity {identity!r} for "
        f"{spec.track_id}/{spec.track_version}"
    )


def _confirmation_identity(spec: TrackSpec):
    """Return the CPU-only existing fresh-seed derivation for MT5 scheme A."""
    profile = _profile(spec)

    def derive(
        locator,
        source: str,
        *,
        target,
        guidance_digest: str,
        master_seed: bytes,
        instance_key: str,
    ) -> dict[str, Any]:
        # Lazy by design: this parses the fixed winning candidate and derives the existing
        # ValidationPlan, but never compiles or touches a device.  The frozen Track version is
        # part of that derivation: deriving with the registry default would let a new version
        # change an old run's confirmation seeds.
        from taic.portfolio_confirmation import confirmation_seed_for
        from tcenv.portfolio_validation.validation import VALIDATION_PLAN_SCHEMA_ID
        from .search_candidate import confirmation_input

        frozen = confirmation_input(
            locator,
            source,
            target=target,
            guidance_digest=guidance_digest,
            master_seed=master_seed,
            track_id=profile,
            track_version=spec.track_version,
        )
        return {
            "portfolio_freeze_digest": frozen.freeze.digest,
            "validation_plan_digest": frozen.validation.digest,
            "confirmation_seed": confirmation_seed_for(frozen.validation, instance_key),
            "seed_derivation_version": (
                f"taic-v2-search-master-1/{VALIDATION_PLAN_SCHEMA_ID}/"
                f"derivation-{frozen.validation.derivation_version}"
            ),
        }

    return derive


def adapter_for_track(spec: TrackSpec) -> LazyTrackAdapter:
    """Resolve the registered implementation without reading a default registry."""
    profile = _profile(spec)

    if profile == "tilelang-pipeline":
        from .tilelang_adapter import adapter_services

        return LazyTrackAdapter(spec, adapter_services(spec))

    def builder():
        from .track_adapter import MetalTrackBuilder

        return MetalTrackBuilder

    def evaluator():
        from .track_adapter import MetalTrackEvaluator

        return MetalTrackEvaluator

    def geometry_resolver():
        if profile == "msl-bundle":
            return None
        from .track_adapter import msl_single_geometry_resolver

        return msl_single_geometry_resolver

    def renderer():
        from ...track.plane import TrackRenderer

        return TrackRenderer.for_track(spec)

    def validator():
        return lambda candidate: candidate.validate_for(spec)

    def guidance():
        from .search_candidate import guidance_fingerprint

        return guidance_fingerprint

    return LazyTrackAdapter(
        spec,
        {
            "builder": builder,
            "evaluator": evaluator,
            # This service is intentionally an existing kernel-pipeline implementation detail,
            # not a claim that every future Track must use GeometryPair.
            "geometry_resolver": geometry_resolver,
            "renderer": renderer,
            "validator": validator,
            "confirmation": lambda: _confirmation_identity(spec),
            "guidance": guidance,
        },
    )


__all__ = ["adapter_for_track"]
