"""Roofline class -> one existing Apple GPU skill section. No GPU and no API key."""

from __future__ import annotations

from pathlib import Path

from taic import prompts
from taic.feedback_extra import status_lines
from taic.symptom import SymptomIndex, bound_of, section_for
from tasks import get_task
from tcenv.evaluate import EvalReport

SKILL = Path(__file__).resolve().parent.parent / "backends" / "metal" / "skills" / "apple_gpu_optimization.md"
MEMORY_HEADING = "## Memory-bound kernels"
COMPUTE_HEADING = "## Compute-bound kernels"
# Copied from tests/support/golden/metal/fixtures/relu_best.report.json and gemm_paper_best.report.json.
# These are the roofline objects tcenv.backends.metal.profile.roofline writes; bound is the structured field.
_RELU_ROOFLINE = {
    "bound": "memory",
    "min_bytes": 67108864.0,
    "flops": 16777216,
    "arithmetic_intensity": 0.25,
    "ridge_intensity": 32.86610531024317,
    "achieved_gbps": 460.8119125251599,
    "bandwidth_fraction": 1.0256817845352935,
    "achieved_tflops": 0.11520297813128998,
    "compute_fraction": 0.00780197238806712,
    "lower_bound_us": 149.37187497707782,
    "time_over_lower_bound": 0.9749612551158554,
    "peak_gbps_measured": 449.2737606078677,
    "peak_tflops_measured": 14.765878729267158,
}
_GEMM_ROOFLINE = {
    "bound": "compute",
    "min_bytes": 25165824.0,
    "flops": 17179869184.0,
    "arithmetic_intensity": 682.6666666666666,
    "ridge_intensity": 31.08155136177421,
    "achieved_gbps": 21.2878800743148,
    "bandwidth_fraction": 0.0448100888572003,
    "achieved_tflops": 14.532526130732236,
    "compute_fraction": 0.9841964976948918,
    "lower_bound_us": 1163.4843749561696,
    "time_over_lower_bound": 1.0160572633027267,
    "peak_gbps_measured": 475.0689100875139,
    "peak_tflops_measured": 14.765878729267158,
}


def _task():
    return get_task("relu", "fp16", "small")


def _plan(bound):
    return prompts.plan_user(_task(), "kernel void relu_kernel() {}", "feedback text", ["h1"], 3, bound=bound, source_fence="metal")


def test_the_index_is_deterministic_and_slices_the_existing_file():
    file_text = SKILL.read_text()
    a, b = SymptomIndex(), SymptomIndex()
    assert a.by_bound == b.by_bound
    assert section_for("memory") == a.section("memory") == section_for("memory")
    for bound, heading, other in (
        ("memory", MEMORY_HEADING, COMPUTE_HEADING),
        ("compute", COMPUTE_HEADING, MEMORY_HEADING),
    ):
        text = section_for(bound)
        assert text.startswith(heading) and other not in text
        assert text in file_text
        assert text.count("\n## ") == 0
        assert "## Roofline section" not in text and "## Diagnose" not in text


def test_memory_selects_the_memory_chapter_and_not_the_compute_chapter():
    text = section_for("memory")
    assert "move each byte exactly once" in text
    assert "Register-block" not in text
    msg = _plan("memory")
    assert MEMORY_HEADING in msg and COMPUTE_HEADING not in msg
    assert "move each byte exactly once" in msg and "Register-block" not in msg


def test_compute_selects_the_compute_chapter_and_not_the_memory_chapter():
    text = section_for("compute")
    assert "Register-block" in text
    assert "move each byte exactly once" not in text
    msg = _plan("compute")
    assert COMPUTE_HEADING in msg and MEMORY_HEADING not in msg
    assert "Register-block" in msg and "move each byte exactly once" not in msg


def test_bound_of_reads_the_structured_roofline_field():
    assert bound_of({"roofline": _RELU_ROOFLINE}) == "memory"
    assert bound_of({"roofline": _GEMM_ROOFLINE}) == "compute"
    report = EvalReport(task="relu", dtype="fp16", size="paper", mode="full", status="ok")
    report.roofline = dict(_RELU_ROOFLINE)
    assert bound_of(report) == "memory"
    # A feedback sentence that names the other class does not override the field.
    msg = prompts.plan_user(_task(), "k", "bound: compute; achieved 10 GB/s", [], 3, bound=bound_of(report), source_fence="metal")
    assert MEMORY_HEADING in msg and COMPUTE_HEADING not in msg


def test_a_missing_or_unknown_bound_does_not_crash_or_invent_one():
    plain = prompts.plan_user(_task(), "k", "bound: compute; achieved 10 GB/s", ["h"], 3, source_fence="metal")
    for report in (
        None,
        {},
        {"roofline": None},
        {"roofline": {}},
        {"roofline": {"lower_bound_us": 10.0}},
        {"roofline": {"bound": None}},
        {"roofline": {"bound": ""}},
        {"roofline": {"bound": "latency"}},
        {"roofline": {"bound": "Memory"}},
        {"roofline": {"bound": "memory-bound"}},
        EvalReport(task="relu", dtype="fp16", size="small", mode="full"),
    ):
        assert bound_of(report) is None
    for bound in (None, "", "latency", "Memory", "unknown"):
        assert section_for(bound) is None
        assert _plan(bound) == prompts.plan_user(_task(), "kernel void relu_kernel() {}", "feedback text", ["h1"], 3, source_fence="metal")
        assert MEMORY_HEADING not in _plan(bound) and COMPUTE_HEADING not in _plan(bound)
    assert plain == prompts.plan_user(_task(), "k", "bound: compute; achieved 10 GB/s", ["h"], 3, bound=None, source_fence="metal")
    assert COMPUTE_HEADING not in plain and MEMORY_HEADING not in plain
    system = prompts.system_prompt(_task(), use_tools=False)
    assert prompts.planning_system(system, None) is system
    assert prompts.planning_system(system, "") is system
    assert prompts.planning_system(system, "latency") is system
    # no calibration on the report: the default roofline is empty, and the stored prompt stays whole
    assert MEMORY_HEADING in system and COMPUTE_HEADING in system


def test_a_known_bound_puts_only_the_matching_chapter_in_the_planning_request():
    status = status_lines(rnd=2, max_rounds=12, patience_left=3, spent=1.5, budget=3, required_gain=1.014)
    for tools in (False, True):
        system = prompts.system_prompt(_task(), use_tools=tools)
        assert system.startswith(prompts.HARNESS_RULES)
        for bound, keep, drop, phrase, other in (
            ("memory", MEMORY_HEADING, COMPUTE_HEADING, "move each byte exactly once", "Register-block"),
            ("compute", COMPUTE_HEADING, MEMORY_HEADING, "Register-block", "move each byte exactly once"),
        ):
            sent = prompts.planning_system(system, bound)
            assert sent != system and sent.startswith(prompts.HARNESS_RULES)
            assert keep not in sent and drop not in sent
            assert "## Machine model" in sent and "## Divergence, barriers and scheduling" in sent
            assert "## A suggested optimization order" in sent and "## Roofline section" in sent
            user = prompts.plan_user(_task(), "k", "fb", [], 3, status=status, bound=bound, source_fence="metal")
            assert keep in user and drop not in user and phrase in user and other not in user
            assert user.index("noise floor") < user.index(keep) < user.index("Diagnose the dominant bottleneck")
            request = sent + "\n" + user
            assert keep in request and drop not in request and other not in request
            assert prompts.planning_system(system, bound) == sent
