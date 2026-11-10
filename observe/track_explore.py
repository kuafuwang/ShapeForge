"""Read-only Track exploration summary for observe; never a publication authority."""

from __future__ import annotations

from pathlib import Path

from tcenv.identity import load_strict_json


def read_track_explore(evidence_dir: str | Path) -> dict:
    """Read one archived track-explore summary without starting any run or model."""
    root = Path(evidence_dir)
    summary = load_strict_json(root / "summary.json")
    observations = load_strict_json(root / "observations.json")
    tracks = []
    for row in summary["tracks"]:
        instance = row["instances"][0]
        key = f"{row['track_id']}/fp16/small/benchmark"
        observation = observations["runs"][key]
        if instance["observation"] != observation:
            raise ValueError(f"observation snapshot differs from summary: {key}")
        tracks.append({
            "track_id": row["track_id"],
            "track_version": row["track_version"],
            "eligible": row["eligible"],
            "confirmation_status": instance["confirmation_status"],
            "latency_ratio": instance["latency_ratio"],
            "candidate_sha256": instance["candidate_sha256"],
            "run_dir": instance["run_dir"],
            "cost_complete": instance["search_cost"]["cost_complete"],
            "agent_turns": instance["search_cost"]["agent_turns"],
            "build_seconds": instance["search_cost"]["build_seconds"],
            "encoder_strategy": observation["encoder_strategy"],
            "native_batch_calls": observation["native_batch_calls"],
            "command_buffers": observation["command_buffers"],
            "actual_spent_usd": observation["actual_spent_usd"],
        })
    winner = summary.get("winner")
    return {
        "schema_id": "observe.track-explore.1",
        "status": summary["status"],
        "config_digest": summary["config_digest"],
        "observations_digest": observations["config_digest"],
        "exploration_only": True,
        "formal_publication": False,
        "same_fresh_input": False,
        "cross_track_runtime_portfolio": False,
        "winner": None if winner is None else {
            "track_id": winner["track_id"],
            "track_version": winner["track_version"],
            "latency_ratio": winner["geomean_latency_ratio"],
            "worst_shard_latency_ratio": winner["worst_shard_latency_ratio"],
            "exploration_bundle_digest": winner["exploration_bundle_digest"],
        },
        "tracks": tracks,
        "budget": summary["budget"],
        "evidence_dir": str(root),
        "publication_authority": False,
    }


__all__ = ["read_track_explore"]
