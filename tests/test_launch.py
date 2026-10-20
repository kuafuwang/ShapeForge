"""Parser and limit checks for ``/* taic-launch */``. No GPU and no model."""

import pytest

from tcenv.abi import check_pipeline, only_static_threadgroup_limit
from tcenv.metal_runtime import pipeline_ops
from tcenv.launch import LaunchRejected, check_launch_limits, parse_launch_spec
from taic.policy import feature_signature

CONST = {"M_TOTAL": 93312, "BLOCK_M": 64}


def _spec(body: str, constants=None):
    return parse_launch_spec("kernel void k() {}\n" + body, constants or CONST)


def test_no_block_is_the_old_path():
    assert parse_launch_spec("kernel void conv2d_kernel() {}\n", CONST) is None


def test_one_block_parses_contract_arithmetic():
    spec = _spec("""
/* taic-launch
math safe
workspace scratch M_TOTAL * 2
launch gemm grid=M_TOTAL/BLOCK_M,2,1 threads=128,1,1 threadgroup=4096 workspace=scratch
*/
""")
    assert spec.math == "safe"
    assert spec.workspaces[0].nbytes == 93312 * 2
    assert spec.stages[0].grid == (93312 // 64, 2, 1)
    assert spec.stages[0].threadgroup_bytes == 4096


def test_marker_in_a_line_comment_or_a_second_block_is_rejected():
    with pytest.raises(LaunchRejected):
        parse_launch_spec("// taic-launch\nkernel void k() {}\n", CONST)
    src = "/* taic-launch\nlaunch k grid=1,1,1 threads=32,1,1\n*/\n/* taic-launch\nlaunch k grid=1,1,1 threads=32,1,1\n*/\n"
    with pytest.raises(LaunchRejected):
        parse_launch_spec(src, CONST)


def test_uneven_division_and_a_short_workspace_are_rejected():
    with pytest.raises(LaunchRejected):
        _spec("/* taic-launch\nlaunch k grid=3/2,1,1 threads=32,1,1\n*/")
    with pytest.raises(LaunchRejected):
        _spec("/* taic-launch\nworkspace s 2\nlaunch k grid=1,1,1 threads=32,1,1 workspace=s\n*/")


def test_too_many_stages_and_an_oversized_grid_are_rejected():
    lines = "\n".join(f"launch k grid=1,1,1 threads=32,1,1" for _ in range(33))
    with pytest.raises(LaunchRejected):
        _spec(f"/* taic-launch\n{lines}\n*/")
    with pytest.raises(LaunchRejected):
        _spec("/* taic-launch\nlaunch k grid=16777217,1,1 threads=32,1,1\n*/")


def test_an_expression_workspace_marked_zero_stays_zero():
    spec = _spec("""
/* taic-launch
workspace scratch M_TOTAL * 2 zero
launch gemm grid=1,1,1 threads=32,1,1 workspace=scratch
*/
""")
    assert spec.workspaces[0].zero is True
    assert spec.workspaces[0].nbytes == 93312 * 2


def test_sixteen_stages_are_allowed():
    lines = "\n".join("launch k grid=1,1,1 threads=32,1,1" for _ in range(16))
    spec = _spec(f"/* taic-launch\n{lines}\n*/")
    assert len(spec.stages) == 16


def test_65536_threadgroups_is_allowed_and_the_device_cap_is_applied_later():
    spec = _spec("/* taic-launch\nlaunch k grid=65536,1,1 threads=256,1,1\n*/")
    assert spec.stages[0].grid[0] == 65536
    check_launch_limits(
        spec, contract_names=set(), max_threads_xyz=(1024, 1024, 1024), max_threads_product=1024,
        max_threadgroup_memory_bytes=32768, recommended_working_set_bytes=100 * 2**30,
        max_buffer_length=80 * 2**30, rotation_sets=6,
    )
    fat = _spec("/* taic-launch\nworkspace scratch 8000000000\nlaunch k grid=1,1,1 threads=32,1,1 workspace=scratch\n*/")
    with pytest.raises(LaunchRejected):
        check_launch_limits(
            fat, contract_names={"x"}, max_threads_xyz=(1024, 1024, 1024), max_threads_product=1024,
            max_threadgroup_memory_bytes=32768, recommended_working_set_bytes=8 * 2**30,
            max_buffer_length=80 * 2**30, rotation_sets=6,
        )


class _Arg:
    def __init__(self, name, index, role):
        self.name, self.index, self.role = name, index, role


class _Contract:
    def __init__(self):
        self.args = [_Arg("x", 0, "input"), _Arg("y", 1, "output")]
        self.outputs = [self.args[1]]

    def arg(self, name):
        return next(a for a in self.args if a.name == name)


class _Bind:
    def __init__(self, name, index, kind="buffer", access="read"):
        self.name, self.index, self.kind, self.access = name, index, kind, access


class _Kern:
    def __init__(self, bindings, static=0, max_total=1024):
        self.bindings = bindings
        self.static_tg_bytes = static
        self.max_total_threads = max_total


def test_pipeline_abi_accepts_12kib_dynamic_and_rejects_index_31_and_a_combined_overrun():
    spec = parse_launch_spec(
        "/* taic-launch\nlaunch k grid=1,1,1 threads=32,1,1 threadgroup=12288\n*/\n", {}
    )
    kernel = _Kern([
        _Bind("x", 0, access="read"),
        _Bind("y", 1, access="write"),
        _Bind("smem", 0, kind="threadgroup"),
    ])
    assert check_pipeline(_Contract(), {"k": kernel}, spec, max_tg_memory=32768) == []
    high = _Kern([
        _Bind("x", 0, access="read"),
        _Bind("y", 31, access="write"),
        _Bind("smem", 0, kind="threadgroup"),
    ])
    assert any("above 30" in item for item in check_pipeline(_Contract(), {"k": high}, spec, max_tg_memory=32768))
    fat = parse_launch_spec(
        "/* taic-launch\nlaunch k grid=1,1,1 threads=32,1,1 threadgroup=20000\n*/\n", {}
    )
    heavy = _Kern([
        _Bind("x", 0, access="read"),
        _Bind("y", 1, access="write"),
        _Bind("smem", 0, kind="threadgroup"),
    ], static=20000)
    problems = check_pipeline(_Contract(), {"k": heavy}, fat, max_tg_memory=32768)
    assert any("combined threadgroup" in item for item in problems)
    assert only_static_threadgroup_limit(problems) is False
    bare = _Kern([_Bind("x", 0, access="read"), _Bind("y", 1, access="write"), _Bind("smem", 0, kind="threadgroup")])
    undeclared = parse_launch_spec("/* taic-launch\nlaunch k grid=1,1,1 threads=32,1,1\n*/\n", {})
    assert any("does not declare threadgroup" in item for item in check_pipeline(_Contract(), {"k": bare}, undeclared))


def test_an_unused_kernel_does_not_count_as_a_launched_feature():
    from taic.hwfeatures import analyze_source

    source = """
/* taic-launch
launch used grid=1,1,1 threads=32,1,1
*/
kernel void unused() { simdgroup_multiply_accumulate(a, b, c, d); }
kernel void used(uint i [[thread_position_in_grid]]) {}
"""
    assert analyze_source(source, kernel_name="unused")["matrix"]["present"] is True
    assert feature_signature(source)[0] is False


def test_a_later_bench_iteration_fills_the_same_workspace_again():
    ops = pipeline_ops(8, 3, 2, concurrent=True, filled=[True, True, True])
    fills = [op for op in ops if op[0] == "fill"]
    assert [op[1] for op in fills] == [0, 1, 2, 0, 1, 2, 0, 1]
    # The fill of a set is followed by a barrier before that iteration's first dispatch.
    for index, op in enumerate(ops):
        if op[0] != "fill":
            continue
        assert ops[index + 1][0] == "barrier"
        assert ops[index + 2] == ("dispatch", op[1], 0)


def test_a_pipeline_warning_names_the_stage_that_is_at_the_register_limit():
    from taic.hwfeatures import measured_from_report
    from taic.hwfeedback import _warnings

    report = {
        "pipeline": {
            "stages": [
                {"kernel": "gemm", "max_total_threads_per_threadgroup": 128, "launch_threads_per_threadgroup": 128},
                {"kernel": "epi", "max_total_threads_per_threadgroup": 1024, "launch_threads_per_threadgroup": 32},
            ],
            "max_total_threads_per_threadgroup": 128,
            "launch_threads_per_threadgroup": 32,
            "static_threadgroup_bytes": 0,
        }
    }
    measured = measured_from_report(report)
    assert measured["launch_threads"] is None
    notes = _warnings(type("H", (), {"source": {"matrix": {"acc_tiles": 0}}})(), measured)
    assert any("stage gemm" in note for note in notes)
    assert not any("launch size 32" in note for note in notes)


def test_a_pipeline_command_buffer_waits_two_seconds_per_stage_up_to_eight():
    from tcenv.evaluate import LAUNCH_DEADLINE_S, command_buffer_deadline_s, deadline_for_source

    assert command_buffer_deadline_s(1) == LAUNCH_DEADLINE_S
    assert command_buffer_deadline_s(3) == 6.0
    assert command_buffer_deadline_s(32) == 8.0
    three = """/* taic-launch
launch a grid=1,1,1 threads=32,1,1
launch b grid=1,1,1 threads=32,1,1
launch c grid=1,1,1 threads=32,1,1
*/
"""
    assert deadline_for_source("conv2d", "fp16", "paper", three) == 6.0
    assert deadline_for_source("conv2d", "fp16", "paper", "kernel void k() {}\n") == LAUNCH_DEADLINE_S
