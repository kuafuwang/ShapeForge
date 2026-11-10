"""Read-only observe Track exploration API tests; no GPU or model calls."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from observe.app import api_v2_track_explore
from observe.track_explore import read_track_explore

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "tests/support/golden/shared/多DSL-Track-阶段验收/MT6/evidence/track-explore-r15"


def test_observe_track_explore_reports_winner_and_all_tracks():
    document = read_track_explore(EVIDENCE)
    assert document["status"] == "exploration_winner"
    assert document["winner"]["track_id"] == "tilelang-pipeline"
    assert document["winner"]["track_version"] == 3
    assert document["winner"]["latency_ratio"] < 1.10
    assert document["publication_authority"] is False
    assert document["formal_publication"] is False
    assert document["same_fresh_input"] is False
    assert document["cross_track_runtime_portfolio"] is False
    assert len(document["tracks"]) == 3
    assert all(row["eligible"] and row["confirmation_status"] == "confirmed" for row in document["tracks"])


def test_observe_track_explore_preserves_execution_and_cost_evidence():
    document = read_track_explore(EVIDENCE)
    for row in document["tracks"]:
        assert row["native_batch_calls"] == 0
        assert row["command_buffers"] > 0
        assert row["encoder_strategy"] == "pyobjc_fallback"
        assert row["cost_complete"] is True
        assert row["agent_turns"] == 1
        assert row["build_seconds"] > 0
        assert row["actual_spent_usd"] == 0.0
    tilelang = next(row for row in document["tracks"] if row["track_id"] == "tilelang-pipeline")
    assert tilelang["build_seconds"] > 1.0
    assert document["config_digest"] == document["observations_digest"]


def test_track_explore_endpoint_is_read_only_json():
    response = asyncio.run(api_v2_track_explore(None))
    assert response.status_code == 200
    document = json.loads(response.body)
    assert document["status"] == "exploration_winner"
    assert document["winner"]["track_id"] == "tilelang-pipeline"
    assert document["publication_authority"] is False
    assert len(document["tracks"]) == 3
