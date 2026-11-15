"""P2-M: the ScratchSpec contract is present in the actual hashed prompt surface."""

from __future__ import annotations

import json
from pathlib import Path

from taic.trust import current_prompt_sha1, prompt_fingerprint_parts

ROOT = Path(__file__).resolve().parents[2]
EPOCH = json.loads((ROOT / "tests/support/golden/shared/shape-portfolio-阶段验收/P2-M/prompt-epoch.json").read_text())


def test_prompt_epoch_is_explicit_and_not_aliased():
    assert EPOCH["previous_prompt_sha1"] == "27e2bd2c2dfe3c009b065e1e83c9e2051bed9381fce79aa36b92618c2ac0fe3c"
    assert EPOCH["current_prompt_sha1"] == "e9d28ff61819a041610d763f5c1464ea860265be2d414f78c15a1f8aea1fdbcf"
    from tests.support.live_anchors import load_live_anchors
    live = load_live_anchors(ROOT)["prompt_sha1"]
    assert current_prompt_sha1() == live
    assert live != EPOCH["previous_prompt_sha1"]


def test_reference_and_session_skill_cover_the_full_scratch_lifecycle():
    reference = (ROOT / "backends/metal/skills/msl_reference.md").read_text(encoding="utf-8")
    skill = (ROOT / "taic/flows/kernel_once/skills/taic-tools/SKILL.md").read_text(encoding="utf-8")
    for fragment in (
        "Candidate scratch workspaces",
        "scratch uwin fp32 36*C*OC zero",
        "uninitialized_poison",
        "device_zero",
        "host_written",
        "out1/out3 poison scratch even when the declaration says `zero`",
        "Every iteration redispatches all active stages",
        "Never use unordered floating-point atomics",
    ):
        assert fragment in reference, fragment
    for fragment in (
        "Scratch workspaces",
        "out1/out3 poison scratch even with `zero` declared",
        "Every timed iteration must regenerate the intermediate",
        "Split-K also adds partial traffic and a reduction",
    ):
        assert fragment in skill, fragment


def test_scratch_sentences_are_inside_the_registered_fingerprint_parts():
    parts = dict(prompt_fingerprint_parts())
    assert "backends/metal/skills/msl_reference.md" in parts
    assert "taic/flows/kernel_once/skills/taic-tools/SKILL.md" in parts
    prompt_source = parts["prompts.system_prompt_v2"].decode("utf-8")
    for fragment in (
        "Candidate-owned ScratchSpec",
        "optional top-level `scratch` array",
        "uninitialized_poison",
        "out1/out3 even when you declare `zero`",
        "never compute scratch once in prepare",
    ):
        assert fragment in prompt_source, fragment
    assert b"Candidate scratch workspaces" in parts["backends/metal/skills/msl_reference.md"]
    assert b"Scratch workspaces" in parts["taic/flows/kernel_once/skills/taic-tools/SKILL.md"]
