"""MT6 CPU-only TileLang /3 fresh-confirmation identity contract.

This module deliberately does **not** modify the frozen :mod:`tcenv.track.tilelang` /1 or /2
contract, registry or task-pack closure.  A future /3 Track may wrap the already frozen /2
source-only frontend semantics, but must keep two identities separate:

* the stable pre-build subject below is suitable for BuildProfile provenance;
* the /3 native descriptor stays stable across attempts; full attempt evidence is retained in
  BuildEvidence.log rather than forcing native_digest to differ.

Nothing here registers a Track, imports a frontend, builds Metal or executes a candidate.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Mapping

from ..identity import fingerprint
from . import tilelang as base
from .registry import TrackSpec, load_track_registry

TILELANG_CONFIRMATION_TRACK_VERSION = 3
FRESH_SUBJECT_SCHEMA_ID = "tcenv.track.tilelang.fresh-subject.1"


@dataclass(frozen=True)
class FreshBuildSubject:
    """Stable /3 subject derived from the complete frozen /2 pre-build subject."""

    candidate_sha256: str
    source_only_subject_cache_key: str
    cache_key: str
    track_version: int = TILELANG_CONFIRMATION_TRACK_VERSION
    full_build: bool = False


def frozen_confirmation_contract() -> dict[str, Any]:
    """Return the unregistered /3 contract proposal without changing /2 bytes or closure."""
    document = copy.deepcopy(base.frozen_contract(track_version=base.TILELANG_METAL_TRACK_VERSION))
    document["track_version"] = TILELANG_CONFIRMATION_TRACK_VERSION
    document["build"]["implemented_modes"] = [base.SOURCE_ONLY_BUILD_MODE]
    document["evidence_boundary"] = {
        "source_only": "reuses the frozen /2 source-only subject; no frontend execution in this CPU contract slice",
        "metal_build": "MT6 fresh-stable subject contract only; no registered Metal or fresh-confirmation evidence",
        "availability": "interface_only pending the MT6 builder, fresh Metal confirmation and independent review",
    }
    document["fresh_confirmation"] = {
        "subject_schema_id": FRESH_SUBJECT_SCHEMA_ID,
        "build_profile_field": "tilelang_fresh_subject",
        "build_profile_value": "stable expected_fresh_subject.cache_key",
        "native_descriptor_value": "stable /3 candidate/frontend/artifact identity; full attempt evidence in BuildEvidence.log",
        "forbidden_comparator_relaxation": True,
        "registered": False,
    }
    return document


def audit_fresh_registration(track: TrackSpec, *, source_track: TrackSpec | None = None) -> dict[str, str]:
    """Audit an unregistered /3 outer Track against the exact frozen /2 source-only contract.

    This is intentionally separate from ``base.audit_registration``: /3 is not added to the old
    supported-version set or task-pack closure.  The outer builder/evaluator identity is new, while
    the nested frontend lowering remains explicitly /2 rather than silently relabelled.
    """
    source = source_track or load_track_registry().get(
        base.TILELANG_TRACK_ID, version=base.TILELANG_METAL_TRACK_VERSION
    )
    base.audit_registration(source)
    if source.track_version != base.TILELANG_METAL_TRACK_VERSION:
        raise base.TileLangContractError("the /3 source-only sub-build must use the frozen /2 TrackSpec")
    if track.track_id != base.TILELANG_TRACK_ID or track.track_version != TILELANG_CONFIRMATION_TRACK_VERSION:
        raise base.TileLangContractError(f"not the TileLang fresh-confirmation /3 Track: {track.key}")
    if track.status not in ("interface_only", "available"):
        raise base.TileLangContractError(f"/3 has unknown Track status {track.status!r}")
    if track.status == "interface_only" and (track.evidence or "metal_dispatch" in set(track.capabilities)):
        raise base.TileLangContractError("an interface_only /3 Track cannot claim evidence or metal_dispatch")
    if track.status == "available" and (not track.evidence or "metal_dispatch" not in set(track.capabilities)):
        raise base.TileLangContractError("an available /3 Track needs real evidence and metal_dispatch")
    for name in ("delivery", "candidate_roles", "candidate_source", "frontend", "renderer", "validator"):
        if getattr(track, name) != getattr(source, name):
            raise base.TileLangIdentityDrift(f"/3 {name} does not preserve the frozen /2 source contract")
    if track.build_adapter.implementation_id != source.build_adapter.implementation_id or int(
        track.build_adapter.implementation_version
    ) != TILELANG_CONFIRMATION_TRACK_VERSION:
        raise base.TileLangIdentityDrift("/3 needs its own build_adapter implementation_version=3")
    return {
        "track": f"{track.track_id}/{track.track_version}",
        "source_only_track": f"{source.track_id}/{source.track_version}",
        "contract_digest": fingerprint(frozen_confirmation_contract()),
    }


def expected_fresh_subject(*, candidate: Any, environment: Mapping[str, Any]) -> FreshBuildSubject:
    """Derive a stable /3 subject from candidate, geometry, frontend and environment.

    The wrapped /2 subject already binds the complete candidate manifest, pinned frontend,
    options and closed environment.  The /3 fingerprint adds only the new Track/version and
    fresh-subject schema; it does not read attempt paths, generated artifacts or a report.
    """
    source_only = base.expected_build_subject(
        candidate=candidate,
        environment=environment,
        track_version=base.TILELANG_METAL_TRACK_VERSION,
    )
    cache_key = fingerprint(
        {
            "schema_id": FRESH_SUBJECT_SCHEMA_ID,
            "track": {
                "track_id": base.TILELANG_TRACK_ID,
                "track_version": TILELANG_CONFIRMATION_TRACK_VERSION,
            },
            "source_only_subject_cache_key": source_only.cache_key,
        }
    )
    return FreshBuildSubject(
        candidate_sha256=source_only.candidate_sha256,
        source_only_subject_cache_key=source_only.cache_key,
        cache_key=cache_key,
    )


__all__ = [
    "FRESH_SUBJECT_SCHEMA_ID",
    "FreshBuildSubject",
    "TILELANG_CONFIRMATION_TRACK_VERSION",
    "audit_fresh_registration",
    "expected_fresh_subject",
    "frozen_confirmation_contract",
]
