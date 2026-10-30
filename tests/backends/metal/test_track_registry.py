"""The shipped Metal track registry: msl-single/msl-bundle are A1 metadata skeletons.

This file is Metal-specific because it pins the backend's own ``tracks.json``.  It imports
no Metal runtime and starts no GPU work; it only verifies that the backend's delivery
metadata exists, is closed, and does not claim availability before A2/A4 evidence.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tcenv.contract_errors import UnsupportedContractCapability
from tcenv.track import METAL_TRACKS_PATH, load_track_registry

ROOT = Path(__file__).resolve().parents[3]


def test_the_shipped_metal_tracks_are_the_two_interface_only_skeletons():
    """@brief 首批 Metal track 只有 msl-single/msl-bundle，且都是 interface_only。"""
    assert METAL_TRACKS_PATH == ROOT / "backends" / "metal" / "tracks.json"
    registry = load_track_registry()
    assert registry.backend_id == "metal"
    assert {track.track_id for track in registry.tracks} == {"msl-single", "msl-bundle"}
    assert all(track.status == "interface_only" for track in registry.tracks)
    assert registry.available() == ()
    assert registry.get("msl-single").delivery == "single_source"
    assert registry.get("msl-bundle").delivery == "candidate_bundle"
    assert registry.get("msl-bundle").candidate_roles == ("device_kernel", "host_tiling", "schedule")
    with pytest.raises(UnsupportedContractCapability):
        registry.require_available("msl-single")


def test_the_metal_registry_document_is_closed_and_op_agnostic():
    """@brief tracks.json 是 schema_version 1 的闭集元数据，没有题名/shape/tiling 字段。"""
    document = json.loads(METAL_TRACKS_PATH.read_text(encoding="utf-8"))
    assert document["schema_version"] == 1
    assert document["backend_id"] == "metal"
    for track in document["tracks"]:
        for forbidden in ("op_name", "shape", "shapes", "tile", "grid", "threads", "num_warps", "lowering_profile"):
            assert forbidden not in track
        assert track["status"] == "interface_only"
        assert "evidence" not in track

