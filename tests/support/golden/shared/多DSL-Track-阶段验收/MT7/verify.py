#!/usr/bin/env python3
"""Read-only MT7 documentation and observability verifier; no GPU or model call."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from observe.track_explore import read_track_explore  # noqa: E402

HERE = Path(__file__).resolve().parent


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def load(name: str):
    return json.loads((HERE / name).read_text(encoding="utf-8"))


def verify() -> dict:
    identity = load("identity.json")
    nodes = load("nodes.json")
    require(identity["documentation_complete"] is True and identity["observability_complete"] is True, "MT7 docs/observability incomplete")
    require(identity["track_explore_reviewer_done"] is True, "r15 independent review missing")
    require(identity["second_dsl_scope_decision_pending"] is False and identity["additional_dsl_not_required"] is True, "second DSL scope was not explicitly narrowed")
    require(identity["full_mt6_done"] is True and identity["done"] is True and identity["reviewer_done"] is True, "MT6/MT7 completion or final review is missing")
    require(identity["publication_allowed"] is False, "MT7 became publication")
    for name in ("README.md", "README.zh-CN.md"):
        text = (ROOT / name).read_text(encoding="utf-8")
        require("Multi-DSL" in text or "多 DSL" in text, f"{name} has no Multi-DSL section")
        require("track-explore" in text, f"{name} has no track-explore link")
        require("MSL + TileLang" in text and "full_mt6_done=true" in text, f"{name} does not record the signed MT6 scope")
        require("remain deferred" not in text and "仍 deferred" not in text, f"{name} still defers the signed MT6 scope")
    guide = (ROOT / "backends/metal/wiki/tilelang.md").read_text(encoding="utf-8")
    require("tilelang.md" in (ROOT / "backends/metal/wiki/INDEX.md").read_text(encoding="utf-8"), "TileLang guide is not linked")
    require("MSL + TileLang" in guide and "full_mt6_done=true" in guide, "TileLang guide omits the signed MT6 scope")
    require("完整 MT6、无界" not in guide, "TileLang guide still denies the narrowed MT6 completion")
    skills = (ROOT / "docs/agent-skills.md").read_text(encoding="utf-8")
    require("TileLang Track" in skills, "agent skills has no TileLang entry")
    require("full_mt6_done=true" in skills and "仍未完成" not in skills, "agent skills still says MT6 is unfinished")
    acceptance = (HERE / "acceptance.md").read_text(encoding="utf-8")
    require("待独立范围复核" not in acceptance and "仍 deferred" not in acceptance, "MT7 acceptance still treats scope review as pending")
    require("full_mt6_done=true" in acceptance, "MT7 acceptance omits the signed MT6 completion")
    document = read_track_explore(ROOT / "docs/架构/多DSL-Track-阶段验收/MT6/evidence/track-explore-r15")
    require(document["status"] == "exploration_winner" and document["winner"]["track_id"] == "tilelang-pipeline", "observe winner drift")
    require(document["publication_authority"] is False and document["formal_publication"] is False, "observe became publication authority")
    require(len(document["tracks"]) == 3 and all(row["eligible"] for row in document["tracks"]), "observe track eligibility drift")
    require(all(row["native_batch_calls"] == 0 and row["cost_complete"] and row["agent_turns"] == 1 for row in document["tracks"]), "observe execution/cost evidence drift")
    require(nodes["observe_and_structure_passed"] == 9 and nodes["track_explore_passed"] == 45 and nodes["compatibility_passed"] == 133, "MT7 node record drift")
    require(nodes["done"] is True and nodes["reviewer_done"] is True, "MT7 node package review state drift")
    manifest = HERE / "manifest.sha256"
    if manifest.exists():
        for line in manifest.read_text(encoding="utf-8").splitlines():
            expected, name = line.split("  ", 1)
            require(hashlib.sha256((HERE / name).read_bytes()).hexdigest() == expected, f"manifest mismatch: {name}")
        listed = {line.split("  ", 1)[1] for line in manifest.read_text(encoding="utf-8").splitlines()}
        actual = {path.name for path in HERE.iterdir() if path.is_file() and path.name != "manifest.sha256" and path.suffix != ".pyc"}
        require(listed == actual, "MT7 manifest is not the closed file set")
    return {
        "stage": "MT7",
        "status": identity["status"],
        "documentation_complete": True,
        "observability_complete": True,
        "second_dsl_scope_decision_pending": identity["second_dsl_scope_decision_pending"],
        "done": True,
        "reviewer_done": True,
        "publication_allowed": False,
        "verifier_passed": True,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.parse_args()
    print(json.dumps(verify(), sort_keys=True))
