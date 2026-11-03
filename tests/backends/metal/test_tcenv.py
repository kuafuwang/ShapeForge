"""Tests of the evaluation environment itself: it must accept good kernels and reject every kind of bad one.

Run with:  uv run pytest -q tests/backends/metal/test_tcenv.py
"""

from __future__ import annotations

import os

import pytest
import torch

from tasks import get_task, list_tasks
from tcenv.backends.metal.abi import check_abi
from tcenv.contract_errors import TaskContractError
from tcenv.dtypes import get_dtype
from tcenv.evaluate import evaluate
from tcenv.backends.metal.runtime import BufferArg, MetalCompileError, get_context
from tcenv.backends.metal.sanitize import ArgSet
from tcenv.backends.metal.testing import prepare_case
from tcenv.tensors import from_raw, round_to, to_raw

from ...support.hang_candidate import LION_BODY as GOOD_BODY
from ...support.hang_candidate import lion_hang_findings, lion_hang_source
from ...support.v1_tasks import V1_TASK, v1_task

pytestmark = pytest.mark.gpu

#: The evaluation environment is a **v1 mechanism** that survives A6 for the preserved tasks.
#: A5 retired the migrated relu assembly, so every case below drives the same mechanism with
#: lion (the preserved elementwise task with an ``if (j < N)`` body) instead of reviving relu.
TASK = "lion"
SIZE = "small"
#: ``lion`` ABI: three inputs then two outputs (declaration order).
P_IN, G_IN, M_IN, P_OUT, M_OUT = 0, 1, 2, 3, 4


def task_src() -> str:
    """@brief 返回保留的 v1 题（lion）在 fp16/small 下的 naive Metal 源码。

    @return naive MSL 源码。
    """
    return v1_task(TASK).naive_msl


def ev(src: str, *, mode: str = "quick", task: str = TASK, dtype: str = "fp16", seed: int = 7, timeout: float | None = None):
    """@brief 用固定 small 尺寸调用 evaluate 的薄封装。

    @param src 待评估的 Metal 源码。
    @param mode 评估模式（quick 或 full）。
    @param task 任务名。
    @param dtype 数据类型。
    @param seed 测试用例种子。
    @param timeout 评估超时秒数，None 表示使用默认值。
    @return 评估报告 EvalReport。
    """
    return evaluate(task, dtype, SIZE, src, mode=mode, seed=seed, timeout=timeout)


def mutate(body: str) -> str:
    """@brief 把 naive lion 的标准循环体替换成给定实现，用于构造错误内核。

    @param body 用来替换 GOOD_BODY 的新循环体。
    @return 替换后的完整源码。
    """
    src = task_src()
    assert GOOD_BODY in src
    return src.replace(GOOD_BODY, body)


# --------------------------------------------------------------------------- tensors / dtypes


@pytest.mark.parametrize("dtype", ["fp16", "bf16", "fp32", "i32", "u32"])
def test_raw_roundtrip(dtype):
    """@brief 各 dtype 的 to_raw/from_raw 往返后数值不变。

    @param dtype 参数化的存储类型。
    """
    t = torch.arange(-8, 8, dtype=torch.float32) * (1 if dtype in ("i32",) else 0.37)
    if dtype == "u32":
        t = t.abs()
    r = round_to(t, dtype)
    back = from_raw(to_raw(r, dtype), dtype, r.shape)
    assert torch.equal(back.to(torch.float32), r.to(torch.float32))


def test_bf16_rounding_is_real():
    """@brief bf16 会把 1+2^-10 舍入到 1.0，而 fp16 能保留该值。
    """
    x = torch.tensor([1.0 + 2**-10])
    assert round_to(x, "bf16").item() == 1.0
    assert round_to(x, "fp16").item() == x.item()


# --------------------------------------------------------------------------- runtime


def test_compile_error_lines_are_remapped():
    """@brief 编译错误行号会按 contract 前缀偏移映射回 kernel.metal。
    """
    task = get_task(TASK, "fp16", SIZE)
    ctx = get_context()
    name = task.contract.kernel_name
    src, off = task.contract.assemble(f"kernel void {name}(device half* y [[buffer(0)]]) {{\n  y[0] = nope;\n}}\n")
    with pytest.raises(MetalCompileError) as e:
        ctx.compile(src, name)
    msg = task.contract.rewrite_compiler_lines(str(e.value), off)
    assert "kernel.metal:2:" in msg and "nope" in msg


def test_missing_kernel_name_is_reported():
    """@brief 源码里没有目标 kernel 时，编译报 not found。
    """
    task = get_task(TASK, "fp16", SIZE)
    src, _ = task.contract.assemble("kernel void other(device half* y [[buffer(0)]]) { y[0] = 1; }")
    with pytest.raises(MetalCompileError) as e:
        get_context().compile(src, task.contract.kernel_name)
    assert "not found" in str(e.value)


def test_naive_is_abi_conformant():
    """@brief 任务自带的 naive kernel 能编译且 ABI 检查无违规。
    """
    task = get_task(TASK, "fp16", SIZE)
    src, _ = task.contract.assemble(task.naive_msl)
    k = get_context().compile(src, task.contract.kernel_name)
    assert check_abi(task.contract, k) == []


def test_guard_bands_detect_out_of_bounds_write():
    """@brief guard band 能发现越界写并报 past the end。
    """
    task = get_task(TASK, "fp16", SIZE)
    c = task.contract
    ctx = get_context()
    src, _ = c.assemble(task.naive_msl.replace("p_out[j]", "p_out[j + 8]"))
    k = ctx.compile(src, c.kernel_name)
    data = prepare_case(task, "normal", 1)
    a = ArgSet(ctx, c)
    a.load_inputs(data.raw)
    a.poison_outputs()
    ctx.run(k, [a.args()], c.grid, c.threads)
    assert any("past the end" in v for v in a.safety_violations())


# --------------------------------------------------------------------------- evaluation: acceptance


@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
def test_good_kernel_is_accepted_full_pipeline(dtype):
    """@brief 正确 kernel 在 full 模式下通过：状态 ok、加速比为正、roofline 已判定且 shader validation 干净。

    @param dtype 参数化的数据类型。
    """
    task = get_task(TASK, dtype, SIZE)
    rep = evaluate(TASK, dtype, SIZE, task.naive_msl, mode="full", seed=11, reps=5)
    assert rep.status == "ok", rep.message
    assert rep.speedup is not None and rep.speedup > 0
    assert rep.roofline["bound"] in ("memory", "compute")
    assert rep.safety["shader_validation"] == "clean"


def test_all_tasks_are_self_consistent_small():
    """@brief 每个任务自己的 naive kernel 都能通过自己的测试（fp16）。
    """
    for name in list_tasks():
        rep = evaluate(name, "fp16", SIZE, get_task(name, "fp16", SIZE).naive_msl, mode="quick", seed=3)
        assert rep.status == "ok", f"{name}: {rep.message}"


# --------------------------------------------------------------------------- evaluation: rejection


def test_rejects_compile_error():
    """@brief 引用不存在符号的内核被判 compile_error，错误信息带源文件名和符号名。
    """
    rep = ev(mutate("if (j < N) { p_out[j] = nonexistent_symbol; }"))
    assert rep.status == "compile_error"
    assert "nonexistent_symbol" in rep.compile["errors"]
    assert "kernel.metal:" in rep.compile["errors"]


P_DECL = "device const half* p [[buffer(0)]]"
P_OUT_DECL = "device half* p_out [[buffer(3)]],"
P_OUT_TAIL = "device half* p_out [[buffer(3)]]"
M_OUT_DECL = "device half* m_out [[buffer(4)]]"


@pytest.mark.parametrize(
    "old,new,needle,mutate_src",
    [
        (P_DECL, "device const half* pp [[buffer(0)]]", "must be named", "rename"),
        (P_DECL, "device half* p [[buffer(0)]]", "must be declared `device const", None),
        (P_OUT_DECL, P_OUT_DECL + " device half* extra [[buffer(5)]],", "extra buffer", None),
        (P_OUT_DECL, P_OUT_DECL + " threadgroup float* dyn [[threadgroup(0)]],", "dynamic threadgroup", None),
        (P_OUT_TAIL, "device const half* p_out [[buffer(3)]]", "read-only", "readonly"),
    ],
)
def test_rejects_abi_violations(old, new, needle, mutate_src):
    """@brief 参数改名、去掉 const、多余 buffer、动态 threadgroup、只读输出等 ABI 违规都被拒绝。

    @param old 原源码里被替换的片段。
    @param new 替换后的片段。
    @param needle 期望出现在 ABI 违规信息里的关键词。
    @param mutate_src 需要同步改写正文的额外构造（``rename``/``readonly``）。
    """
    src = task_src()
    assert old in src
    if mutate_src == "readonly":
        src = src.replace(GOOD_BODY, "if (j < N) { }")  # a const output cannot be written
    if mutate_src == "rename":
        src = src.replace("float(p[j])", "float(pp[j])")  # keep the body consistent with the renamed parameter
    rep = ev(src.replace(old, new, 1))
    assert rep.status == "abi_error", (rep.status, rep.message, rep.compile)
    assert any(needle in v for v in rep.abi["violations"]), rep.abi["violations"]


def test_rejects_wrong_math():
    """@brief 数学错误的内核被判 incorrect，错误信息含 out of tolerance。
    """
    rep = ev(mutate("if (j < N) { float pv = float(p[j]), gv = float(g[j]), mv = float(m[j]); float c = BETA1 * mv + (1.0f - BETA1) * gv; float u = c > 0.0f ? 1.0f : (c < 0.0f ? -1.0f : 0.0f); p_out[j] = half(0.5f * pv); m_out[j] = half(BETA2 * mv + (1.0f - BETA2) * gv); }"))
    assert rep.status == "incorrect"
    assert "out of tolerance" in rep.message


def test_rejects_unwritten_outputs_and_says_so():
    """@brief 只写部分输出时被判 incorrect，并明确提示从未写入的元素。
    """
    rep = ev(mutate("if (j < N && (j % 1024) < 1000) { float pv = float(p[j]), gv = float(g[j]), mv = float(m[j]); float c = BETA1 * mv + (1.0f - BETA1) * gv; float u = c > 0.0f ? 1.0f : (c < 0.0f ? -1.0f : 0.0f); p_out[j] = half(pv * (1.0f - LR * WD) - LR * u); m_out[j] = half(BETA2 * mv + (1.0f - BETA2) * gv); }"))
    assert rep.status == "incorrect"
    assert "never written" in rep.message


def test_rejects_out_of_bounds_write_as_unsafe():
    """@brief 越界写被判 unsafe，错误信息含 past the end。
    """
    rep = ev(mutate("if (j < N) { float pv = float(p[j]), gv = float(g[j]), mv = float(m[j]); float c = BETA1 * mv + (1.0f - BETA1) * gv; float u = c > 0.0f ? 1.0f : (c < 0.0f ? -1.0f : 0.0f); p_out[j + 16] = half(pv * (1.0f - LR * WD) - LR * u); m_out[j] = half(BETA2 * mv + (1.0f - BETA2) * gv); }"))
    assert rep.status == "unsafe"
    assert "past the end" in rep.message


def test_rejects_dependence_on_initial_output_contents():
    """@brief 读取输出旧值的内核被判 unsafe，错误信息含 initial contents。
    """
    rep = ev(mutate("if (j < N) { float pv = float(p[j]), gv = float(g[j]), mv = float(m[j]); float c = BETA1 * mv + (1.0f - BETA1) * gv; float u = c > 0.0f ? 1.0f : (c < 0.0f ? -1.0f : 0.0f); float r = pv * (1.0f - LR * WD) - LR * u; half prev = p_out[j]; p_out[j] = isnan(prev) ? half(r) : half(r + 1.0f); m_out[j] = half(BETA2 * mv + (1.0f - BETA2) * gv); }"))
    assert rep.status == "unsafe"
    assert "initial contents" in rep.message


def test_shader_validation_catches_far_out_of_bounds_read():
    """@brief 远离缓冲区的越界读在 full 模式下被判 unsafe/crash/runtime_error，且不可用。
    """
    rep = ev(mutate("if (j < N) { float pv = float(p[j + 4000000]), gv = float(g[j]), mv = float(m[j]); float c = BETA1 * mv + (1.0f - BETA1) * gv; float u = c > 0.0f ? 1.0f : (c < 0.0f ? -1.0f : 0.0f); p_out[j] = half(pv * (1.0f - LR * WD) - LR * u); m_out[j] = half(BETA2 * mv + (1.0f - BETA2) * gv); }"), mode="full")
    assert rep.status in ("unsafe", "crash", "runtime_error"), (rep.status, rep.message)
    assert not rep.eligible


def test_input_threadgroup_size_limit_is_reported():
    """@brief 编译出的 kernel 支持的最大线程数不低于 contract 要求的 threadgroup 大小。
    """
    # Force a register-heavy kernel; if the compiler cannot run 128 threads/tg the ABI check must say so.
    task = get_task(TASK, "fp16", SIZE)
    src, _ = task.contract.assemble(task.naive_msl)
    k = get_context().compile(src, task.contract.kernel_name)
    assert k.max_total_threads >= task.contract.threads_per_tg


def test_dtype_registry():
    """@brief dtype 注册表把 bf16 映射到 bfloat，未知的 fp8 抛 KeyError。
    """
    assert get_dtype("bf16").msl == "bfloat"
    with pytest.raises(KeyError):
        get_dtype("fp8")


# --------------------------------------------------------------------------- GPU safety (none of these touch the GPU)


@pytest.mark.parametrize(
    "body",
    ["while (true) { }", "while(1){}", "for (;;) { }", "for (uint i = 0; ; ++i) { }", "do { } while (true);"],
)
def test_unbounded_loops_are_refused_before_any_worker_starts(body):
    """@brief 各种无界循环在 screen 阶段就被拒绝，不启动 worker，耗时不到 1 秒。

    @param body 替换进 lion kernel 的无界循环写法。
    """
    rep = evaluate(TASK, "fp16", SIZE, task_src().replace(GOOD_BODY, body), mode="full", seed=1)
    assert rep.status == "rejected", rep.message
    assert rep.elapsed_s < 1.0 and rep.stage == "screen"


def test_loops_inside_comments_do_not_trigger_the_screen():
    """@brief 注释里的循环写法不触发循环筛查。
    """
    from tcenv.backends.metal.screen import screen_source

    assert screen_source("// while (true) {}\n/* for (;;) */ kernel void k() { for (uint i = 0; i < 4; ++i) {} }") == []


def test_parentheses_do_not_hide_an_unconditional_loop():
    """@brief 括号、恒真条件和空条件不能绕过循环筛查；字符串字面量不算循环，向后 goto 会被拒绝。
    """
    from tcenv.backends.metal.screen import screen_source

    for src in ("while((true)) {}", "while(true == true) {}", "for(int i = (0); ; i++) {}", "for(; (true); ) {}"):
        assert screen_source(src), src
    assert screen_source("for (uint i = 0; i < 4; ++i) {}") == []
    assert screen_source("while (i < n) {}") == []
    # a string literal is data, not a loop; a backward goto is refused
    assert screen_source('kernel void k() { const char *s = "for (;;)"; }') == []
    assert screen_source("kernel void k() { again: ; goto again; }")


def test_a_stack_trace_is_not_a_shader_validation_failure():
    """@brief 栈回溯文本不算 shader validation 失败，真正的 validation error 才算。
    """
    from tcenv.backends.metal.sanitize import validation_problems

    assert validation_problems("stack trace follows\ntraceback (most recent call last)\ngraceful shutdown\n") == []
    assert validation_problems("validation error: race condition on device memory")


def test_sem_relu_gelu_comment_reference_follows_the_written_comparison():
    """@brief sem_relu_gelu_comment 的 v2 参考实现按 x > 0 的写法处理 NaN、-inf 和 -0。

    A5 moved the task to the v2 standard entry, so the written comparison is the document
    ``formula`` and the authority is ``reference.py`` -- there is no Triton prompt or v1
    known-good text left to read (A6 asserts they are not resurrected elsewhere).
    """
    from pathlib import Path

    from tcenv.taskcontract import load_task_contract

    directory = Path(__file__).resolve().parents[3] / "tasks" / "sem_relu_gelu_comment"
    contract = load_task_contract(directory, dtype="fp16", size="small")
    # The v2 reference validates the declared shape, so the five special values live at the front
    # of a full-size fp16 tensor; the remaining positions are ordinary negative/positive values.
    n = int(contract.instance.symbols["N"])
    x = torch.linspace(-1.0, 1.0, n, dtype=torch.float16)
    x[:5] = torch.tensor([float("nan"), -1.0, 2.0, float("-inf"), -0.0], dtype=torch.float16)
    y = contract.reference.execute(contract.instance, {"x": x})["y"]
    assert y[0].item() == 0.0 and y[1].item() == 0.0 and y[2].item() == 2.0 and y[3].item() == 0.0
    assert str(y[4].item()) == "0.0"
    assert torch.all(y[5:] == torch.where(x[5:] > 0, x[5:].double(), torch.zeros(n - 5, dtype=torch.float64)))
    assert "x[i] > 0" in str(contract.spec.document["formula"])
    with pytest.raises(TaskContractError):
        get_task("sem_relu_gelu_comment", "fp16", "small")


def test_non_metal_includes_are_refused():
    """@brief 非 Metal 的 #include 被拒绝，metal_stdlib 和 simdgroup 头文件允许。
    """
    from tcenv.backends.metal.screen import screen_source

    assert screen_source('#include <metal_stdlib>\n#include "/etc/hosts"\n')
    assert not screen_source("#include <metal_stdlib>\n#include <metal_simdgroup_matrix>\n")


def _fake_worker_report(module: str, **kw):
    """@brief 用指定的 fake worker 模块跑一次 _run_worker 并返回报告。

    @param module fake worker 的模块名。
    @param kw 传给 _run_worker 的超时、launch deadline 等参数。
    @return worker 产出的 EvalReport。
    """
    from tcenv import evaluate as ev_mod

    return ev_mod._run_worker(TASK, "fp16", SIZE, __import__("pathlib").Path("/dev/null"), "quick", 1, 3, validation=False, cases=None, worker_module=module, **kw)


def test_supervisor_kills_a_stuck_launch_within_the_deadline(monkeypatch):
    """@brief 单次 launch 卡死会在 launch_deadline 内被杀，远早于 60 秒总超时。

    @param monkeypatch pytest 补丁工具，用于把杀后冷却设为 0。
    """
    import time

    from tcenv import evaluate as ev_mod

    monkeypatch.setattr(ev_mod, "POST_KILL_COOLDOWN_S", 0.0)
    t0 = time.perf_counter()
    rep = _fake_worker_report("tests.support.fake_worker_stuck_launch", timeout=60.0, launch_deadline_s=1.0)
    assert rep.status == "timeout" and "single GPU launch" in rep.message
    assert time.perf_counter() - t0 < 10  # nowhere near the 60 s overall timeout


def test_supervisor_enforces_the_overall_timeout(monkeypatch):
    """@brief worker 空闲不推进时整体超时生效并报 exceeded。

    @param monkeypatch pytest 补丁工具，用于把杀后冷却设为 0。
    """
    from tcenv import evaluate as ev_mod

    monkeypatch.setattr(ev_mod, "POST_KILL_COOLDOWN_S", 0.0)
    rep = _fake_worker_report("tests.support.fake_worker_idle", timeout=2.0, launch_deadline_s=1.0)
    assert rep.status == "timeout" and "exceeded" in rep.message


def test_supervisor_does_not_kill_launches_that_finish(monkeypatch):
    """@brief 正常完成的 launch 不会被 supervisor 杀掉，报告为 ok。

    @param monkeypatch pytest 补丁工具，用于把杀后冷却设为 0。
    """
    from tcenv import evaluate as ev_mod

    monkeypatch.setattr(ev_mod, "POST_KILL_COOLDOWN_S", 0.0)
    rep = _fake_worker_report("tests.support.fake_worker_ok", timeout=30.0, launch_deadline_s=1.0)
    assert rep.status == "ok", rep.message


# --------------------------------------------------------------------------- slow / exhaustive


@pytest.mark.skipif(
    os.environ.get("TCENV_ALLOW_GPU_HANG_TESTS") != "1",
    reason="DANGEROUS: runs a non-terminating GPU kernel, which can freeze the display for tens of seconds or force a hard reset. "
    "The supervisor logic is covered by the fake-worker tests above; set TCENV_ALLOW_GPU_HANG_TESTS=1 only on a machine you can afford to lose.",
)
def test_real_gpu_hang_is_killed():
    """@brief 真实 GPU 死循环会被杀掉，之后正常的 kernel 仍能通过。

    默认跳过，只有设置 TCENV_ALLOW_GPU_HANG_TESTS=1 才运行。
    """
    # The candidate must use lion's ABI (three inputs, two outputs); a relu-shaped ``x[j]``/``y[j]``
    # body would fail to compile here and the test would see ``compile_error`` instead of a hang.
    hang = lion_hang_source(task_src())
    assert lion_hang_findings(task_src(), hang) == []
    rep = ev(hang, timeout=20)
    assert rep.status == "timeout"
    assert ev(task_src()).status == "ok"


@pytest.mark.slow
@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
def test_reference_outputs_are_representable_at_paper_size(dtype):
    """@brief paper 尺寸下参考输出必须能被输出 dtype 表示，否则 prepare_case 报错（属于 harness bug）。

    @param dtype 参数化的数据类型。
    """
    from tcenv.backends.metal.testing import prepare_case

    for name in list_tasks():
        task = get_task(name, dtype, "paper")
        for i, case in enumerate(task.cases):
            prepare_case(task, case, 1000 + i)


def test_benchmark_uses_every_rotating_buffer_set_and_spot_checks_only_launched_ones():
    """@brief 每个轮转 buffer set 都会被填充，spot check 只检查真正 launch 过的 set。

    回归：慢 kernel（每批 launch 次数少）以前会留下大部分轮转 set 未使用，
    spot check 又把未使用的 set 当成 never written；paper 尺寸的 FlashAttention 就是这种 kernel。
    """
    task = get_task("flash_attention", "fp16", "paper")
    rep = evaluate("flash_attention", "fp16", "paper", task.naive_msl, mode="full", seed=5, reps=3)
    assert rep.status == "ok", rep.message
    assert rep.benchmark["candidate"]["iters"] >= rep.benchmark["rotation_sets"]


@pytest.mark.parametrize("name", list(list_tasks(include_phase2=True))[len(list_tasks()):] + ["sem_int_1", "sem_int_2", "sem_attention_odd_scale"])
@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
def test_phase2_and_semantic_tasks_are_self_consistent_small(name, dtype):
    """@brief phase2 与语义任务的 naive kernel 在 small 尺寸下自洽。

    @param name 参数化的任务名。
    @param dtype 参数化的数据类型。
    """
    rep = evaluate(name, dtype, SIZE, get_task(name, dtype, SIZE).naive_msl, mode="quick", seed=3)
    assert rep.status == "ok", f"{name} {dtype}: {rep.message}"


def test_verify_interface_is_a_stub_that_never_claims_a_proof():
    """@brief verify 目前只是 stub：返回 unsupported，不声称证明也不给反例。
    """
    from tasks import get_task
    from tcenv.verify import STATUSES, UNSUPPORTED, VerifyResult, verify

    t = get_task(TASK, "fp16", "small")
    r = verify(t, t.naive_msl)
    assert isinstance(r, VerifyResult) and r.status == UNSUPPORTED and r.status in STATUSES
    assert not r.proven and r.counterexample is None and "Metal" in r.message
    assert r.to_json()["status"] == "unsupported"


def test_every_evaluation_report_records_that_formal_verification_is_unsupported():
    """@brief 每份评估报告都记录形式化验证为 unsupported，且经 worker 到父进程的 JSON 往返不变。
    """
    from tasks import get_task
    from tcenv.evaluate import EvalReport, evaluate
    from tcenv.report import format_feedback

    t = get_task(TASK, "fp16", "small")
    rep = evaluate(TASK, "fp16", "small", t.naive_msl, mode="quick")
    assert rep.status == "ok" and rep.verification["status"] == "unsupported"
    assert EvalReport.from_json(rep.to_json()).verification == rep.verification  # survives the worker -> parent JSON round trip
    full = evaluate(TASK, "fp16", "small", t.naive_msl, mode="full", reps=5)
    assert "Formal verification: unsupported" in format_feedback(full)
