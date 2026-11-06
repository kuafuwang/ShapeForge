"""Packaging for the candidate-owned fixed-length KDA native Host+Device bundle."""

from __future__ import annotations

import json
from pathlib import Path

from tcenv.track.envelope import encode_envelope
from tcenv.track.plane import CandidateArtifact, CandidateEntry

ROOT = Path(__file__).resolve().parents[2]
PLANNER_PATH = ROOT / "tests/support/golden/shared/hostprogram/kda_fixed_planner.c"
DEVICE_PATH = ROOT / "tests/support/golden/metal/kda/host_piece_handoff.metal"
HOST_ENTRY_PATH = "host/tiling.c"
DEVICE_ENTRY_PATH = "device/host_piece_handoff.metal"
GEOMETRY_ENTRY_PATH = "host/execution_geometry.json"
PLANNER_SOURCE = PLANNER_PATH.read_text(encoding="utf-8")
DEVICE_SOURCE = DEVICE_PATH.read_text(encoding="utf-8")
WORKSPACES = ("work_items", "item_counts", "handoff_states")


def geometry_document() -> dict:
    stages = []
    for piece in range(3):
        stages.append(
            {
                "kernel": f"kda_piece{piece}",
                "grid": ["H", 1, 1],
                "threads": [1, 1, 1],
                "threadgroup_bytes": 0,
                "workspace": [{"name": name, "zero": False} for name in WORKSPACES],
                "when": None,
                "derived_scalars": [],
            }
        )
    return {"schema_version": 1, "source": "candidate", "stages": stages}


def geometry_bytes(document: dict | None = None) -> bytes:
    return json.dumps(geometry_document() if document is None else document, indent=2, sort_keys=True).encode() + b"\n"


def bundle_entries(*, planner_source: str | None = None, device_source: str | None = None, geometry: dict | None = None):
    return (
        CandidateEntry(
            path=DEVICE_ENTRY_PATH,
            role="device_kernel",
            media_type="text/x-metal",
            data=(DEVICE_SOURCE if device_source is None else device_source).encode("utf-8"),
        ),
        CandidateEntry(
            path=HOST_ENTRY_PATH,
            role="host_tiling",
            media_type="text/x-c",
            data=(PLANNER_SOURCE if planner_source is None else planner_source).encode("utf-8"),
        ),
        CandidateEntry(
            path=GEOMETRY_ENTRY_PATH,
            role="schedule",
            media_type="application/json",
            data=geometry_bytes(geometry),
        ),
    )


def bundle_source(**overrides) -> str:
    return encode_envelope(CandidateArtifact(track_id="msl-bundle", entries=tuple(bundle_entries(**overrides))))


__all__ = [
    "DEVICE_ENTRY_PATH",
    "DEVICE_SOURCE",
    "GEOMETRY_ENTRY_PATH",
    "HOST_ENTRY_PATH",
    "PLANNER_SOURCE",
    "WORKSPACES",
    "bundle_entries",
    "bundle_source",
    "geometry_bytes",
    "geometry_document",
]
