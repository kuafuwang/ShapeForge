"""Source-level accumulator pressure is a heuristic, not a compiler register count."""

from __future__ import annotations

from tcenv.evaluate import EvalReport
from tcenv.metal_runtime import CompiledKernel
from tcenv.pressure import pressure_hint, source_pressure
from tcenv.profile import pipeline_stats
from tcenv.report import format_feedback


def test_declared_fragment_arrays_and_comments():
    source = """
    // simdgroup_float8x8 ignored[128];
    /* simdgroup_float8x8 also_ignored[64]; */
    simdgroup_float8x8 yacc[32];
    simdgroup_float8x8 acc[8], temp;
    kernel void k() {}
    """
    pressure = source_pressure(source)
    assert pressure["fragment_slots"] == 41  # 32 + 8 + 1
    assert pressure["largest_array"] == 32
    assert pressure["arrays"] == [{"name": "yacc", "count": 32}, {"name": "acc", "count": 8}]
    assert "not compiler register or spill data" in pressure["basis"]


def test_a_symbolic_array_extent_is_not_guessed():
    pressure = source_pressure("simdgroup_float8x8 yacc[FRAGMENTS];")
    assert pressure["fragment_slots"] == 1
    assert pressure["largest_array"] == 0
    assert pressure["arrays"] == []


def test_only_a_large_declared_array_gets_an_advisory_hint():
    assert pressure_hint({"largest_array": 16}) == ""
    hint = pressure_hint({"largest_array": 32})
    assert "32 slots" in hint
    assert "690 us" in hint and "1069 us" in hint
    assert "source upper bound" in hint


def test_report_labels_the_source_pressure_as_a_heuristic():
    report = EvalReport(task="conv2d", dtype="fp16", size="paper", mode="full", status="ok", stage="done")
    report.pipeline = {
        "max_total_threads_per_threadgroup": 1024,
        "launch_threads_per_threadgroup": 128,
        "thread_execution_width": 32,
        "static_threadgroup_bytes": 12432,
        "source_pressure": {
            "fragment_slots": 46,
            "largest_array": 32,
            "arrays": [{"name": "yacc", "count": 32}],
            "basis": "declared simdgroup_float8x8 slots; not compiler register or spill data",
        },
        "hints": [pressure_hint({"largest_array": 32})],
    }
    text = format_feedback(report)
    assert "source pressure: 46 declared simdgroup_float8x8 slots, largest array 32 slots" in text
    assert "Metal exposes no compiler register count" in text
    assert "not compiler register usage or a spill report" in text


def test_pipeline_stats_attaches_the_source_pressure_and_hint():
    class Contract:
        threads_per_tg = 128
        num_threadgroups = 1458

    kernel = CompiledKernel("conv2d_kernel", None, [], 1024, 32, 12432, 0.0, 0.0)
    stats = pipeline_stats(
        kernel,
        Contract(),
        {"max_threads_per_threadgroup": 1024, "max_threadgroup_memory_bytes": 32768},
        kernel_source="simdgroup_float8x8 yacc[32];",
    )
    assert stats["source_pressure"]["largest_array"] == 32
    assert any("32 slots" in hint for hint in stats["hints"])
