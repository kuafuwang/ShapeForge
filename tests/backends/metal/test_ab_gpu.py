"""Real-Metal AB runner integration. Guard corruption itself is covered by Host/MTLBuffer negatives."""

from __future__ import annotations

from pathlib import Path

import pytest

from tcenv.backends.metal.ab_bench import compare_many
from tcenv.backends.metal.paired_compare import assemble_v2_group, runner_for_prepared
from tcenv.backends.metal.runtime import get_context
from tcenv.backends.registry import load_backend
from tcenv.taskcontract.locator import REPO_ROOT

pytestmark = pytest.mark.gpu


def _ok() -> dict:
    return {"status": "ok", "ok": True, "worst_scaled_err": 0.0, "allow": 1.0, "message": "ok"}


def test_real_runner_stops_when_the_post_timing_guard_reports_a_violation():
    """@brief 真实 Metal 启动后，安全钩子报违规就停止；另一臂不得再启动或排名。"""
    paths = (
        REPO_ROOT / "kernels/apple-m4-max/conv2d/fp16/paper/best.metal",
        REPO_ROOT / "tests/support/golden/metal/conv2d_portfolio/variants/paper/kernel.metal",
    )
    context = get_context()
    load_backend("metal").prepare_context(context)
    bound = assemble_v2_group(
        paths, task="conv2d", dtype="fp16", size="paper", seed=7,
        track_id="msl-single", context=context,
    )
    real_bad = runner_for_prepared(bound[0])
    real_good = runner_for_prepared(bound[1])
    calls = {"bad": 0, "good": 0}
    broken = {"value": False}

    def bad(k: int) -> float:
        calls["bad"] += 1
        seconds = real_bad(k)
        broken["value"] = True
        return seconds

    bad.clock_of = real_bad.clock_of
    bad.safety_violations = lambda: (
        ["injected post-timing guard violation"] if broken["value"] else real_bad.safety_violations()
    )
    bad.rotation_sets = real_bad.rotation_sets

    def good(k: int) -> float:
        calls["good"] += 1
        return real_good(k)

    good.clock_of = real_good.clock_of
    good.safety_violations = real_good.safety_violations
    good.rotation_sets = real_good.rotation_sets
    document = compare_many(
        {"bad.metal": bad, "good.metal": good},
        verdicts={"bad.metal": _ok(), "good.metal": _ok()},
        reps=1, warmup_reps=0, kmin=6,
    )
    assert document["status"] == "unsafe"
    assert document["rows"] == [] and document["ranking"]["order"] == []
    assert any("injected post-timing guard violation" in item for item in document["safety"])
    assert calls == {"bad": 1, "good": 0}
