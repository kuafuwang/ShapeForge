"""The shipped Metal track registry: msl-single/msl-bundle are A4-available tracks.

This file is Metal-specific because it pins the backend's own ``tracks.json``.  It imports
no Metal runtime and starts no GPU work; it verifies that the closed delivery metadata is
present and that availability now cites the A4 real-slice evidence.
"""

from __future__ import annotations

import json
from pathlib import Path

from tcenv.track import METAL_TRACKS_PATH, load_track_registry

ROOT = Path(__file__).resolve().parents[3]


def test_the_shipped_metal_tracks_are_available_with_a4_evidence():
    """@brief 首批 Metal track 只有 msl-single/msl-bundle，且 A4 后都可用。"""
    assert METAL_TRACKS_PATH == ROOT / "backends" / "metal" / "tracks.json"
    registry = load_track_registry()
    assert registry.backend_id == "metal"
    assert {track.track_id for track in registry.tracks} == {"msl-single", "msl-bundle"}
    assert all(track.status == "available" for track in registry.tracks)
    assert [track.track_id for track in registry.available()] == ["msl-single", "msl-bundle"]
    assert registry.get("msl-single").delivery == "single_source"
    assert registry.get("msl-bundle").delivery == "candidate_bundle"
    assert registry.get("msl-bundle").candidate_roles == ("device_kernel", "host_tiling", "schedule")
    assert registry.require_available("msl-single").evidence == (
        "tests/backends/metal/test_track_a4.py",
        "docs/架构/跨硬件算子生成分工-阶段验收/A4.md",
    )


def test_the_metal_registry_document_is_closed_and_op_agnostic():
    """@brief tracks.json 是 schema_version 1 的闭集元数据，没有题名/shape/tiling 字段。"""
    document = json.loads(METAL_TRACKS_PATH.read_text(encoding="utf-8"))
    assert document["schema_version"] == 1
    assert document["backend_id"] == "metal"
    for track in document["tracks"]:
        for forbidden in ("op_name", "shape", "shapes", "tile", "grid", "threads", "num_warps", "lowering_profile"):
            assert forbidden not in track
        assert track["status"] == "available"
        assert track["evidence"]

