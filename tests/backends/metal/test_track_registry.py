"""The shipped Metal track registry: msl-single/msl-bundle are A4-available tracks.

MT1b turned this into a versioned (schema 2) registry that also registers the TileLang
``kernel_pipeline`` Track as ``interface_only``.  This file is Metal-specific because it pins the
backend's own ``tracks.json``.  It imports no Metal runtime and starts no GPU work; it verifies
that the closed delivery metadata is present, that availability still cites the A4 real-slice
evidence, and that a source-only DSL Track can never be selected.
"""

from __future__ import annotations

import json
from pathlib import Path

from tcenv.track import METAL_TRACKS_PATH, load_track_registry

ROOT = Path(__file__).resolve().parents[3]


def test_the_shipped_metal_tracks_are_available_with_a4_evidence():
    """@brief 首批 Metal track 只有 msl-single/msl-bundle 可选；A4 后都可用。"""
    assert METAL_TRACKS_PATH == ROOT / "backends" / "metal" / "tracks.json"
    registry = load_track_registry()
    assert registry.backend_id == "metal"
    # The two MSL Tracks keep version 1; TileLang /1 stays replay-only while /2 and /3 are available.
    assert set(registry.track_ids) == {"msl-single", "msl-bundle", "tilelang-pipeline"}
    assert all(registry.get(track_id).status == "available" for track_id in ("msl-single", "msl-bundle"))
    assert [track.track_id for track in registry.available()] == ["msl-single", "msl-bundle", "tilelang-pipeline"]
    assert [track.track_version for track in registry.available()] == [1, 1, 3]
    assert registry.get("tilelang-pipeline", version=1).status == "interface_only"
    assert registry.get("tilelang-pipeline", version=2).supports("metal_dispatch")
    assert registry.get("msl-single").delivery == "single_source"
    assert registry.get("msl-bundle").delivery == "candidate_bundle"
    assert registry.get("msl-bundle").candidate_roles == ("device_kernel", "host_tiling", "schedule")
    assert registry.require_available("msl-single").evidence == (
        "tests/backends/metal/test_track_a4.py",
        "docs/架构/跨硬件算子生成分工-阶段验收/A4.md",
    )


def test_the_metal_registry_document_is_closed_and_op_agnostic():
    """@brief tracks.json 是 schema_version 2 的闭集元数据，没有题名/shape/tiling 字段。"""
    document = json.loads(METAL_TRACKS_PATH.read_text(encoding="utf-8"))
    assert document["schema_version"] == 2
    assert document["backend_id"] == "metal"
    for track in document["tracks"]:
        for forbidden in ("op_name", "shape", "shapes", "tile", "grid", "threads", "num_warps", "lowering_profile"):
            assert forbidden not in track
        if track["track_id"].startswith("msl-"):
            assert track["status"] == "available"
            assert track["evidence"]
        else:
            assert track["delivery"] == "dsl_source_pipeline"
            if track["track_version"] == 1:
                # The MT1b source-only contract remains frozen and cannot be selected.
                assert track["status"] == "interface_only"
                assert "evidence" not in track
                assert "metal_dispatch" not in track["capabilities"]
            else:
                # MT2 /2 and the MT6 /3 fresh-confirmation Track carry real Metal evidence.
                assert track["track_version"] in (2, 3) and track["status"] == "available"
                assert track["evidence"] and "metal_dispatch" in track["capabilities"]
