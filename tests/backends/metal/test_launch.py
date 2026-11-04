"""Parser and limit checks for ``/* taic-launch */``. No GPU and no model."""

import pytest

from tcenv.backends.metal.abi import check_pipeline, only_static_threadgroup_limit, validation_problems
from tcenv.backends.metal.runtime import MetalContext, pipeline_ops
from tcenv.backends.metal.launch import LaunchRejected, check_launch_limits, parse_launch_spec
from taic.policy import feature_signature

CONST = {"M_TOTAL": 93312, "BLOCK_M": 64}


def _spec(body: str, constants=None):
    """@brief 在最小 kernel 后拼接 taic-launch 注释并解析。

    @param body 追加到 kernel 后的源码，通常是注释块。
    @param constants 解析时可用的常量表，None 表示用 CONST。
    @return 解析出的 launch spec。
    """
    return parse_launch_spec("kernel void k() {}\n" + body, constants or CONST)


def test_no_block_is_the_old_path():
    """@brief 没有 taic-launch 块时解析结果为 None，走旧路径。
    """
    assert parse_launch_spec("kernel void conv2d_kernel() {}\n", CONST) is None


def test_one_block_parses_contract_arithmetic():
    """@brief 单个块能解析 math、带表达式的 workspace 字节数以及 grid/threadgroup。
    """
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


def test_spaces_and_parentheses_are_the_same_grid():
    """@brief 带空格或括号的 grid/threads 写法解析成相同的数值。
    """
    spec = _spec("""
/* taic-launch
launch gemm grid=(16, 1, 1) threads=(128, 1, 1)
launch other grid=M_TOTAL / BLOCK_M, 2, 1 threads=32, 1, 1
*/
""")
    assert spec.stages[0].grid == (16, 1, 1)
    assert spec.stages[0].threads == (128, 1, 1)
    assert spec.stages[1].grid == (93312 // 64, 2, 1)
    assert spec.stages[1].threads == (32, 1, 1)


def test_marker_in_a_line_comment_or_a_second_block_is_rejected():
    """@brief 行注释里的标记或第二个 taic-launch 块都被拒绝。
    """
    with pytest.raises(LaunchRejected):
        parse_launch_spec("// taic-launch\nkernel void k() {}\n", CONST)
    src = "/* taic-launch\nlaunch k grid=1,1,1 threads=32,1,1\n*/\n/* taic-launch\nlaunch k grid=1,1,1 threads=32,1,1\n*/\n"
    with pytest.raises(LaunchRejected):
        parse_launch_spec(src, CONST)


def test_uneven_division_and_a_short_workspace_are_rejected():
    """@brief 不能整除的 grid 和小于需求的 workspace 被拒绝。
    """
    with pytest.raises(LaunchRejected):
        _spec("/* taic-launch\nlaunch k grid=3/2,1,1 threads=32,1,1\n*/")
    with pytest.raises(LaunchRejected):
        _spec("/* taic-launch\nworkspace s 2\nlaunch k grid=1,1,1 threads=32,1,1 workspace=s\n*/")


def test_too_many_stages_and_an_oversized_grid_are_rejected():
    """@brief 超过 32 个 stage 和超大的 grid 被拒绝。
    """
    lines = "\n".join(f"launch k grid=1,1,1 threads=32,1,1" for _ in range(33))
    with pytest.raises(LaunchRejected):
        _spec(f"/* taic-launch\n{lines}\n*/")
    with pytest.raises(LaunchRejected):
        _spec("/* taic-launch\nlaunch k grid=16777217,1,1 threads=32,1,1\n*/")


def test_an_expression_workspace_marked_zero_stays_zero():
    """@brief 表达式 workspace 标了 zero 时 zero 为 True，字节数仍正确计算。
    """
    spec = _spec("""
/* taic-launch
workspace scratch M_TOTAL * 2 zero
launch gemm grid=1,1,1 threads=32,1,1 workspace=scratch
*/
""")
    assert spec.workspaces[0].zero is True
    assert spec.workspaces[0].nbytes == 93312 * 2


def test_sixteen_stages_are_allowed():
    """@brief 16 个 stage 是合法上限内的配置，全部解析。
    """
    lines = "\n".join("launch k grid=1,1,1 threads=32,1,1" for _ in range(16))
    spec = _spec(f"/* taic-launch\n{lines}\n*/")
    assert len(spec.stages) == 16


def test_65536_threadgroups_is_allowed_and_the_device_cap_is_applied_later():
    """@brief 65536 个 threadgroup 可以解析；设备上限留给 check_launch_limits 检查，超大 workspace 在那里被拒绝。
    """
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
    """@brief pipeline ABI 接受 12KiB 动态 threadgroup，并报告 buffer index 31、静态加动态超限和未声明 threadgroup。
    """
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


def test_a_pipeline_stage_checks_the_scalar_abi_like_the_single_kernel_path():
    """@brief pipeline stage 标量参数与单 kernel 路径用同一套 ABI 规则。

    以前矢量类型或可写 scalar 配上 4 字节 setBytes 也能通过 check_pipeline；单 kernel 路径会拒绝，
    现在 pipeline 路径应用同样规则，避免 stage 从 2 字节 scalar 里读出 16 字节并返回静默错误答案。
    """
    from tcenv.dtypes import get_dtype

    class _ScalarArg:
        name, index, role, dtype = "scale", 2, "scalar", "fp16"

    class _ScalarBind(_Bind):
        def __init__(self, name, index, *, dtype_code, elem_size, access="read"):
            super().__init__(name, index, access=access)
            self.dtype_code, self.elem_size = dtype_code, elem_size

    contract = _Contract()
    contract.args = [*contract.args, _ScalarArg()]
    spec = parse_launch_spec("/* taic-launch\nlaunch k grid=1,1,1 threads=32,1,1\n*/\n", {})
    half = get_dtype("fp16")
    good = _Kern([
        _Bind("x", 0, access="read"), _Bind("y", 1, access="write"),
        _ScalarBind("scale", 2, dtype_code=half.mtl_code, elem_size=half.nbytes),
    ])
    assert check_pipeline(contract, {"k": good}, spec, max_tg_memory=32768) == []
    wide = _Kern([
        _Bind("x", 0, access="read"), _Bind("y", 1, access="write"),
        _ScalarBind("scale", 2, dtype_code=get_dtype("fp32").mtl_code, elem_size=16),
    ])
    assert any("scalar `scale`" in item for item in check_pipeline(contract, {"k": wide}, spec, max_tg_memory=32768))
    writable = _Kern([
        _Bind("x", 0, access="read"), _Bind("y", 1, access="write"),
        _ScalarBind("scale", 2, dtype_code=half.mtl_code, elem_size=half.nbytes, access="write"),
    ])
    assert any("must be read-only" in item for item in check_pipeline(contract, {"k": writable}, spec, max_tg_memory=32768))


def test_an_unused_kernel_does_not_count_as_a_launched_feature():
    """@brief 只在未 launch 的 kernel 里出现的矩阵指令不算作本次使用的特性。
    """
    from taic.hwfeatures import analyze_source

    source = """
/* taic-launch
launch used grid=1,1,1 threads=32,1,1
*/
kernel void unused() { simdgroup_multiply_accumulate(a, b, c, d); }
kernel void used(uint i [[thread_position_in_grid]]) {}
"""
    assert analyze_source(source, kernel_name="unused")["matrix"]["present"] is True
    assert feature_signature(source, backend_id="metal")[0] is False


def test_a_later_bench_iteration_fills_the_same_workspace_again():
    """@brief 后续 benchmark 迭代会重新填充同一 workspace，且填充后先 barrier 再 dispatch。
    """
    ops = pipeline_ops(8, 3, 2, concurrent=True, filled=[True, True, True])
    fills = [op for op in ops if op[0] == "fill"]
    assert [op[1] for op in fills] == [0, 1, 2, 0, 1, 2, 0, 1]
    # The fill of a set is followed by a barrier before that iteration's first dispatch.
    for index, op in enumerate(ops):
        if op[0] != "fill":
            continue
        assert ops[index + 1][0] == "barrier"
        assert ops[index + 2] == ("dispatch", op[1], 0)


def test_run_pipeline_reuses_stage_state_but_rebinds_after_a_fill():
    """@brief 编码优化不改变 pipeline 状态：单阶段复用，fill 后重新绑定候选。"""
    import Metal

    class Encoder:
        def __init__(self):
            self.pipeline = []
            self.dispatches = 0
            self.barriers = 0

        def setComputePipelineState_(self, pso): self.pipeline.append(pso)
        def setBuffer_offset_atIndex_(self, buf, offset, index): pass
        def setBytes_length_atIndex_(self, data, length, index): pass
        def dispatchThreadgroups_threadsPerThreadgroup_(self, grid, threads): self.dispatches += 1
        def memoryBarrierWithScope_(self, scope): self.barriers += 1
        def endEncoding(self): pass

    class CommandBuffer:
        def __init__(self): self.encoder = Encoder()
        def computeCommandEncoder(self): return self.encoder
        def computeCommandEncoderWithDispatchType_(self, kind): return self.encoder
        def commit(self): pass
        def waitUntilCompleted(self): pass
        def status(self): return Metal.MTLCommandBufferStatusCompleted
        def GPUStartTime(self): return 1.0
        def GPUEndTime(self): return 2.0

    class Queue:
        def __init__(self): self.cb = CommandBuffer()
        def commandBuffer(self): return self.cb

    class Stage:
        def __init__(self, name): self.name, self.pso, self.bindings = name, name, ()

    ctx = object.__new__(MetalContext)
    ctx.queue = Queue()
    stage = Stage("candidate")
    ctx.run_pipeline([stage], [(1, 1, 1)], [(32, 1, 1)], [0], [[[]], [[]]], [[], []], iters=4, dispatch="concurrent")
    assert ctx.queue.cb.encoder.pipeline == ["candidate"]
    assert ctx.queue.cb.encoder.dispatches == 4
    assert ctx.queue.cb.encoder.barriers == 1

    ctx.queue = Queue()
    fill = Stage("fill")
    ctx._fill_kernel = lambda: fill
    ctx.run_pipeline(
        [stage], [(1, 1, 1)], [(32, 1, 1)], [0], [[[]], [[]]],
        [[(object(), 0, 4)], [(object(), 0, 4)]], iters=2, dispatch="serial",
    )
    assert ctx.queue.cb.encoder.pipeline == ["fill", "candidate", "fill", "candidate"]
    assert ctx.queue.cb.encoder.dispatches == 4  # two fill dispatches + two candidate dispatches


def test_a_pipeline_warning_names_the_stage_that_is_at_the_register_limit():
    """@brief pipeline 警告会点名处于寄存器上限的 stage，不把它当成整体 launch size，也不把上限当成 spill 诊断。
    """
    from taic.hwfeatures import measured_from_report
    from taic.hwfeedback import _warnings

    report = {
        # A candidate report is current evidence only at schema 4; the pipeline block below is the
        # part this test is about, but the document still has to look like today's report.
        "schema_version": 4,
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
    assert any("does not prove register pressure" in note for note in notes)  # the cap alone is not a spill diagnosis


def test_instrumented_threadgroup_over_the_device_max_is_reported_and_not_a_signature_failure():
    """@brief 插桩后静态 threadgroup 超过设备上限会被报告，且不被误判为签名失败。
    """
    inflated = validation_problems([], [53280], 32768)
    assert inflated == ["static threadgroup memory 53280 B exceeds the device limit of 32768 B"]
    assert only_static_threadgroup_limit(inflated)
    assert validation_problems(["missing kernel parameter `x`"], [53280], 32768) == ["missing kernel parameter `x`"]
    assert validation_problems(["static threadgroup memory 100 B exceeds the device limit of 32768 B"], [16000], 32768) == [
        "static threadgroup memory 100 B exceeds the device limit of 32768 B"
    ]


def test_a_pipeline_command_buffer_waits_two_seconds_per_stage_up_to_eight():
    """@brief pipeline command buffer 的截止时间是每 stage 2 秒、最多 8 秒；没有 launch 时用单次截止时间。
    """
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
    assert deadline_for_source("sem_attention_odd_scale", "fp16", "paper", three) == 6.0
    assert deadline_for_source("sem_attention_odd_scale", "fp16", "paper", "kernel void k() {}\n") == LAUNCH_DEADLINE_S


def test_shader_validation_threadgroup_size_is_not_treated_as_the_device_limit():
    """@brief only_static_threadgroup_limit 只对单条静态 threadgroup 超限消息返回 True。
    """
    from tcenv.backends.metal.abi import only_static_threadgroup_limit

    inflated = "static threadgroup memory 40960 B exceeds the device limit of 32768 B"
    assert only_static_threadgroup_limit([inflated])
    assert not only_static_threadgroup_limit([])
    assert not only_static_threadgroup_limit([inflated, "missing kernel parameter `x`"])
    assert not only_static_threadgroup_limit(["launch infeasible: the contract launches 128 threads"])
