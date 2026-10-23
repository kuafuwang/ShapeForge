"""Tests of the evaluation environment itself: it must accept good kernels and reject every kind of bad one.

Run with:  uv run pytest -q tests/test_tcenv.py
"""

from __future__ import annotations

import os

import pytest
import torch

from tasks import get_task, list_tasks
from tcenv.backends.metal.abi import check_abi
from tcenv.dtypes import get_dtype
from tcenv.evaluate import evaluate
from tcenv.backends.metal.runtime import BufferArg, MetalCompileError, get_context
from tcenv.backends.metal.sanitize import ArgSet
from tcenv.tensors import from_raw, round_to, to_raw

pytestmark = pytest.mark.gpu

TASK = "relu"
SIZE = "small"


def relu_src() -> str:
    return get_task(TASK, "fp16", SIZE).naive_msl


def ev(src: str, *, mode: str = "quick", task: str = TASK, dtype: str = "fp16", seed: int = 7, timeout: float | None = None):
    return evaluate(task, dtype, SIZE, src, mode=mode, seed=seed, timeout=timeout)


GOOD_BODY = "if (j < N) { float v = float(x[j]); y[j] = half(v < 0.0f ? 0.0f : v); }"


def mutate(body: str) -> str:
    src = relu_src()
    assert GOOD_BODY in src
    return src.replace(GOOD_BODY, body)


# --------------------------------------------------------------------------- tensors / dtypes


@pytest.mark.parametrize("dtype", ["fp16", "bf16", "fp32", "i32", "u32"])
def test_raw_roundtrip(dtype):
    t = torch.arange(-8, 8, dtype=torch.float32) * (1 if dtype in ("i32",) else 0.37)
    if dtype == "u32":
        t = t.abs()
    r = round_to(t, dtype)
    back = from_raw(to_raw(r, dtype), dtype, r.shape)
    assert torch.equal(back.to(torch.float32), r.to(torch.float32))


def test_bf16_rounding_is_real():
    x = torch.tensor([1.0 + 2**-10])
    assert round_to(x, "bf16").item() == 1.0
    assert round_to(x, "fp16").item() == x.item()


# --------------------------------------------------------------------------- runtime


def test_compile_error_lines_are_remapped():
    task = get_task(TASK, "fp16", SIZE)
    ctx = get_context()
    src, off = task.contract.assemble("kernel void relu_kernel(device half* y [[buffer(0)]]) {\n  y[0] = nope;\n}\n")
    with pytest.raises(MetalCompileError) as e:
        ctx.compile(src, "relu_kernel")
    msg = task.contract.rewrite_compiler_lines(str(e.value), off)
    assert "kernel.metal:2:" in msg and "nope" in msg


def test_missing_kernel_name_is_reported():
    task = get_task(TASK, "fp16", SIZE)
    src, _ = task.contract.assemble("kernel void other(device half* y [[buffer(0)]]) { y[0] = 1; }")
    with pytest.raises(MetalCompileError) as e:
        get_context().compile(src, "relu_kernel")
    assert "not found" in str(e.value)


def test_naive_is_abi_conformant():
    task = get_task(TASK, "fp16", SIZE)
    src, _ = task.contract.assemble(task.naive_msl)
    k = get_context().compile(src, task.contract.kernel_name)
    assert check_abi(task.contract, k) == []


def test_guard_bands_detect_out_of_bounds_write():
    task = get_task(TASK, "fp16", SIZE)
    c = task.contract
    ctx = get_context()
    src, _ = c.assemble(
        "kernel void relu_kernel(device const half* x [[buffer(0)]], device half* y [[buffer(1)]], uint g [[thread_position_in_grid]]) { y[g + 8] = x[g]; }"
    )
    k = ctx.compile(src, "relu_kernel")
    a = ArgSet(ctx, c)
    a.load_inputs({"x": to_raw(torch.randn(c.arg("x").numel), "fp16")})
    a.poison_outputs()
    ctx.run(k, [a.args()], (c.arg("x").numel // 256, 1, 1), (256, 1, 1))
    assert any("past the end" in v for v in a.safety_violations())


# --------------------------------------------------------------------------- evaluation: acceptance


@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
def test_good_kernel_is_accepted_full_pipeline(dtype):
    task = get_task(TASK, dtype, SIZE)
    rep = evaluate(TASK, dtype, SIZE, task.naive_msl, mode="full", seed=11, reps=5)
    assert rep.status == "ok", rep.message
    assert rep.speedup is not None and rep.speedup > 0
    assert rep.roofline["bound"] in ("memory", "compute")
    assert rep.safety["shader_validation"] == "clean"


def test_all_tasks_are_self_consistent_small():
    """Every task's own naive kernel passes its own tests (fp16)."""
    for name in list_tasks():
        rep = evaluate(name, "fp16", SIZE, get_task(name, "fp16", SIZE).naive_msl, mode="quick", seed=3)
        assert rep.status == "ok", f"{name}: {rep.message}"


# --------------------------------------------------------------------------- evaluation: rejection


def test_rejects_compile_error():
    rep = ev(mutate("if (j < N) { y[j] = nonexistent_symbol; }"))
    assert rep.status == "compile_error"
    assert "nonexistent_symbol" in rep.compile["errors"]
    assert "kernel.metal:" in rep.compile["errors"]


@pytest.mark.parametrize(
    "old,new,needle",
    [
        ("device const half* x [[buffer(0)]]", "device const half* xx [[buffer(0)]]", "must be named"),
        ("device const half* x [[buffer(0)]]", "device half* x [[buffer(0)]]", "must be declared `device const"),
        ("device half* y [[buffer(1)]],", "device half* y [[buffer(1)]], device half* extra [[buffer(5)]],", "extra buffer"),
        ("device half* y [[buffer(1)]],", "device half* y [[buffer(1)]], threadgroup float* dyn [[threadgroup(0)]],", "dynamic threadgroup"),
        ("device half* y [[buffer(1)]],", "device const half* y [[buffer(1)]],", "read-only"),
    ],
)
def test_rejects_abi_violations(old, new, needle):
    src = relu_src()
    assert old in src
    if "const half* y" in new:
        src = src.replace(GOOD_BODY, "if (j < N) { }")  # a const output cannot be written
    if "xx [[" in new:
        src = src.replace("float(x[j])", "float(xx[j])")  # keep the body consistent with the renamed parameter
    rep = ev(src.replace(old, new, 1))
    assert rep.status == "abi_error", (rep.status, rep.message, rep.compile)
    assert any(needle in v for v in rep.abi["violations"]), rep.abi["violations"]


def test_rejects_wrong_math():
    rep = ev(mutate("if (j < N) { float v = float(x[j]); y[j] = half(v > 0.0f ? v : 0.01f * v); }"))
    assert rep.status == "incorrect"
    assert "out of tolerance" in rep.message


def test_rejects_unwritten_outputs_and_says_so():
    rep = ev(mutate("if (j < N && (j % 1024) < 1000) { float v = float(x[j]); y[j] = half(v > 0.0f ? v : 0.0f); }"))
    assert rep.status == "incorrect"
    assert "never written" in rep.message


def test_rejects_out_of_bounds_write_as_unsafe():
    rep = ev(mutate("if (j < N) { float v = float(x[j]); y[j + 16] = half(v > 0.0f ? v : 0.0f); }"))
    assert rep.status == "unsafe"
    assert "past the end" in rep.message


def test_rejects_dependence_on_initial_output_contents():
    rep = ev(mutate("if (j < N) { float v = float(x[j]); half r = half(v > 0.0f ? v : 0.0f); half prev = y[j]; y[j] = isnan(prev) ? r : half(r + half(1.0f)); }"))
    assert rep.status == "unsafe"
    assert "initial contents" in rep.message


def test_shader_validation_catches_far_out_of_bounds_read():
    rep = ev(mutate("if (j < N) { float v = float(x[j + 4000000]); y[j] = half(v > 0.0f ? v : 0.0f); }"), mode="full")
    assert rep.status in ("unsafe", "crash", "runtime_error"), (rep.status, rep.message)
    assert not rep.eligible


def test_input_threadgroup_size_limit_is_reported():
    # Force a register-heavy kernel; if the compiler cannot run 128 threads/tg the ABI check must say so.
    task = get_task("relu", "fp16", SIZE)
    src, _ = task.contract.assemble(task.naive_msl)
    k = get_context().compile(src, task.contract.kernel_name)
    assert k.max_total_threads >= task.contract.threads_per_tg


def test_dtype_registry():
    assert get_dtype("bf16").msl == "bfloat"
    with pytest.raises(KeyError):
        get_dtype("fp8")


# --------------------------------------------------------------------------- GPU safety (none of these touch the GPU)


@pytest.mark.parametrize(
    "body",
    ["while (true) { }", "while(1){}", "for (;;) { }", "for (uint i = 0; ; ++i) { }", "do { } while (true);"],
)
def test_unbounded_loops_are_refused_before_any_worker_starts(body):
    rep = evaluate(TASK, "fp16", SIZE, relu_src().replace(GOOD_BODY, body), mode="full", seed=1)
    assert rep.status == "rejected", rep.message
    assert rep.elapsed_s < 1.0 and rep.stage == "screen"


def test_loops_inside_comments_do_not_trigger_the_screen():
    from tcenv.backends.metal.screen import screen_source

    assert screen_source("// while (true) {}\n/* for (;;) */ kernel void k() { for (uint i = 0; i < 4; ++i) {} }") == []


def test_parentheses_do_not_hide_an_unconditional_loop():
    from tcenv.backends.metal.screen import screen_source

    for src in ("while((true)) {}", "while(true == true) {}", "for(int i = (0); ; i++) {}", "for(; (true); ) {}"):
        assert screen_source(src), src
    assert screen_source("for (uint i = 0; i < 4; ++i) {}") == []
    assert screen_source("while (i < n) {}") == []
    # a string literal is data, not a loop; a backward goto is refused
    assert screen_source('kernel void k() { const char *s = "for (;;)"; }') == []
    assert screen_source("kernel void k() { again: ; goto again; }")


def test_a_stack_trace_is_not_a_shader_validation_failure():
    from tcenv.backends.metal.sanitize import validation_problems

    assert validation_problems("stack trace follows\ntraceback (most recent call last)\ngraceful shutdown\n") == []
    assert validation_problems("validation error: race condition on device memory")


def test_sem_relu_gelu_comment_reference_follows_the_written_comparison():
    task = get_task("sem_relu_gelu_comment", "fp16", "small")
    y = task.reference({"x": torch.tensor([float("nan"), -1.0, 2.0, float("-inf"), -0.0], dtype=torch.float16)})["y"]
    assert y[0].item() == 0.0 and y[1].item() == 0.0 and y[2].item() == 2.0 and y[3].item() == 0.0
    assert str(y[4].item()) == "0.0"
    assert "x > 0" in task.triton_source and "v > 0.0f" in task.naive_msl


def test_non_metal_includes_are_refused():
    from tcenv.backends.metal.screen import screen_source

    assert screen_source('#include <metal_stdlib>\n#include "/etc/hosts"\n')
    assert not screen_source("#include <metal_stdlib>\n#include <metal_simdgroup_matrix>\n")


def _fake_worker_report(module: str, **kw):
    from tcenv import evaluate as ev_mod

    return ev_mod._run_worker(TASK, "fp16", SIZE, __import__("pathlib").Path("/dev/null"), "quick", 1, 3, validation=False, cases=None, worker_module=module, **kw)


def test_supervisor_kills_a_stuck_launch_within_the_deadline(monkeypatch):
    import time

    from tcenv import evaluate as ev_mod

    monkeypatch.setattr(ev_mod, "POST_KILL_COOLDOWN_S", 0.0)
    t0 = time.perf_counter()
    rep = _fake_worker_report("tests.support.fake_worker_stuck_launch", timeout=60.0, launch_deadline_s=1.0)
    assert rep.status == "timeout" and "single GPU launch" in rep.message
    assert time.perf_counter() - t0 < 10  # nowhere near the 60 s overall timeout


def test_supervisor_enforces_the_overall_timeout(monkeypatch):
    from tcenv import evaluate as ev_mod

    monkeypatch.setattr(ev_mod, "POST_KILL_COOLDOWN_S", 0.0)
    rep = _fake_worker_report("tests.support.fake_worker_idle", timeout=2.0, launch_deadline_s=1.0)
    assert rep.status == "timeout" and "exceeded" in rep.message


def test_supervisor_does_not_kill_launches_that_finish(monkeypatch):
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
    hang = mutate("if (j < N) { float v = float(x[j]); volatile uint spin = 0; while (spin < 0xFFFFFFFFu) { spin += (v == v) ? 0u : 1u; } y[j] = half(v); }")
    rep = ev(hang, timeout=20)
    assert rep.status == "timeout"
    assert ev(relu_src()).status == "ok"


@pytest.mark.slow
@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
def test_reference_outputs_are_representable_at_paper_size(dtype):
    """prepare_case raises if any reference output would overflow the output dtype (a harness bug, not a candidate bug)."""
    from tcenv.backends.metal.testing import prepare_case

    for name in list_tasks():
        task = get_task(name, dtype, "paper")
        for i, case in enumerate(task.cases):
            prepare_case(task, case, 1000 + i)


def test_benchmark_uses_every_rotating_buffer_set_and_spot_checks_only_launched_ones():
    """Regression: a slow kernel (few launches per batch) used to leave most rotating sets untouched, and the
    spot check then flagged the untouched sets as 'never written'. FlashAttention at paper size is such a kernel."""
    task = get_task("flash_attention", "fp16", "paper")
    rep = evaluate("flash_attention", "fp16", "paper", task.naive_msl, mode="full", seed=5, reps=3)
    assert rep.status == "ok", rep.message
    assert rep.benchmark["candidate"]["iters"] >= rep.benchmark["rotation_sets"]


@pytest.mark.parametrize("name", list(list_tasks(include_phase2=True))[len(list_tasks()):] + ["sem_int_1", "sem_int_2", "sem_gemm_neg", "sem_relu_gelu_comment", "sem_gelu_named_relu", "sem_attention_odd_scale"])
@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
def test_phase2_and_semantic_tasks_are_self_consistent_small(name, dtype):
    rep = evaluate(name, dtype, SIZE, get_task(name, dtype, SIZE).naive_msl, mode="quick", seed=3)
    assert rep.status == "ok", f"{name} {dtype}: {rep.message}"


def test_verify_interface_is_a_stub_that_never_claims_a_proof():
    from tasks import get_task
    from tcenv.verify import STATUSES, UNSUPPORTED, VerifyResult, verify

    t = get_task("relu", "fp16", "small")
    r = verify(t, t.naive_msl)
    assert isinstance(r, VerifyResult) and r.status == UNSUPPORTED and r.status in STATUSES
    assert not r.proven and r.counterexample is None and "Metal" in r.message
    assert r.to_json()["status"] == "unsupported"


def test_every_evaluation_report_records_that_formal_verification_is_unsupported():
    from tasks import get_task
    from tcenv.evaluate import EvalReport, evaluate
    from tcenv.report import format_feedback

    t = get_task("relu", "fp16", "small")
    rep = evaluate("relu", "fp16", "small", t.naive_msl, mode="quick")
    assert rep.status == "ok" and rep.verification["status"] == "unsupported"
    assert EvalReport.from_json(rep.to_json()).verification == rep.verification  # survives the worker -> parent JSON round trip
    full = evaluate("relu", "fp16", "small", t.naive_msl, mode="full", reps=5)
    assert "Formal verification: unsupported" in format_feedback(full)
