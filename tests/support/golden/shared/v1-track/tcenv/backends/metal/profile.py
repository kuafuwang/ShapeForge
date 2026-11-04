"""Profiler feedback (the Nsight Compute analogue).

Metal exposes far fewer hardware counters to user code than NCU, so feedback is assembled
from what is reliably available:

  * pipeline statistics: ``maxTotalThreadsPerThreadgroup`` (the pipeline's threadgroup launch limit;
    lowers it when a kernel uses many registers), SIMD width, static threadgroup memory;
  * GPU timestamps of the command buffer;
  * an empirical roofline: measured peak memory bandwidth (copy kernel) and measured peak
    fp16 matmul throughput (large MLX GEMM) on *this* machine.

``tcenv.backends.metal.xctrace`` can add Instruments GPU counters on top of that (see that module).
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

from .runtime import BufferArg, CompiledKernel, MetalContext, get_context
from .pressure import pressure_hint, source_pressure

from ...target import DEFAULT_TARGET_ID, calibration_file

def get_calibration(force: bool = False, target_id: str | None = None) -> dict:
    """@brief 读取标定缓存，缺失或过期时在本机重测并写回。

    @param force 是否忽略已有缓存强制重测。
    @param target_id 目标机型标识；省略时用默认目标。
    @return 标定数据字典（带宽、算力、设备名与时间）。
    """
    ctx = get_context()
    info = ctx.info()
    path, _source = calibration_file(target_id or DEFAULT_TARGET_ID)
    if path.exists() and not force:
        try:
            data = json.loads(path.read_text())
            if data.get("device") == info["name"]:
                return data
        except Exception:
            pass
    from ...gpulock import gpu_lock

    with gpu_lock("bandwidth / FLOP calibration"):
        bw = measure_peak_bandwidth(ctx)
        data = {
            "device": info["name"],
            "peak_gbps": max(bw["copy4_gbps"], bw["read4_gbps"]),
            "copy_gbps": bw["copy4_gbps"],
            "read_gbps": bw["read4_gbps"],
            "peak_tflops": {d: measure_peak_flops(d) for d in ("fp16", "bf16", "fp32")},
            "measured_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
    from ...target import calibration_cache_path

    dest = Path(os.environ["TCENV_CALIBRATION"]) if os.environ.get("TCENV_CALIBRATION") else calibration_cache_path(target_id or DEFAULT_TARGET_ID)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(data, indent=2))
    return data

_COPY_SRC = """
#include <metal_stdlib>
using namespace metal;
kernel void copy4(device const float4* x [[buffer(0)]], device float4* y [[buffer(1)]], uint i [[thread_position_in_grid]]) { y[i] = x[i]; }
kernel void read4(device const float4* x [[buffer(0)]], device float4* y [[buffer(1)]], uint i [[thread_position_in_grid]]) {
    float4 v = x[i]; if (v.x == 123456.789f && v.y == 0.0f) y[i] = v;  // read-dominated: store practically never happens
}
"""


def measure_peak_bandwidth(ctx: MetalContext | None = None, nbytes: int = 512 * 2**20, iters: int = 20) -> dict:
    """@brief 用 copy/read 核测本机可达的显存带宽。

    @param ctx Metal 上下文；省略时取进程单例。
    @param nbytes 每个缓冲区的大小。
    @param iters 每轮计时的迭代次数。
    @return 含 ``copy4_gbps`` 与 ``read4_gbps`` 的字典（GB/s）。
    """
    ctx = ctx or get_context()
    n4 = nbytes // 16
    bx, by = ctx.new_buffer(nbytes), ctx.new_buffer(nbytes)
    ctx.view(bx)[:] = bytes(nbytes)
    # Calibration has no task contract: it goes through the backend factory with the registered
    # ``metal.raw-source-calibration`` diagnostic profile (raw-source assemble, no header), and
    # the fixed copy/read kernels are built as *one* BuildOutcome (plan §5.2).
    from tcenv.backends.registry import load_backend
    from tcenv.remote.artifact_builder import SourceBundle

    from .source_builder import DiagnosticContract

    names = ("copy4", "read4")
    contract = DiagnosticContract(kernel_name=names[0])
    builder = load_backend("metal").artifact_builder(
        kernel_contract=contract,
        kernel_names=names,
        diagnostic_profile="metal.raw-source-calibration",
    )
    outcome = builder.build(
        SourceBundle(source=_COPY_SRC.encode("utf-8")), task_contract=contract, deadline_monotonic=None
    )
    handles = outcome.executable_handles()
    if tuple(str(getattr(handle, "name", "")) for handle in handles) != names:
        raise RuntimeError(
            "the calibration build did not return the registered kernels: "
            f"{[str(getattr(handle, 'name', '')) for handle in handles]}"
        )
    out = {}
    for name, k in zip(names, handles):
        grid = (n4 // 256, 1, 1)
        args = [[(0, BufferArg(bx)), (1, BufferArg(by))]]
        ctx.run(k, args, grid, (256, 1, 1), iters=2)
        best = 0.0
        for _ in range(5):
            tm = ctx.run(k, args, grid, (256, 1, 1), iters=iters)
            moved = nbytes * (2 if name == "copy4" else 1)
            best = max(best, moved / tm.gpu_per_iter_s / 1e9)
        out[name + "_gbps"] = best
    return out


def measure_peak_flops(dtype: str = "fp16", n: int = 4096) -> float:
    """@brief 用厂商 GEMM（MLX）测本芯片可达的算力。

    @param dtype 数据类型，``fp16``、``bf16`` 或 ``fp32``。
    @param n 方阵边长。
    @return 实测 TFLOP/s。

    Achievable TFLOP/s of the vendor GEMM (MLX) on this chip.
    """
    import mlx.core as mx

    dt = {"fp16": mx.float16, "bf16": mx.bfloat16, "fp32": mx.float32}[dtype]
    a = mx.random.normal((n, n)).astype(dt)
    b = mx.random.normal((n, n)).astype(dt)
    mx.eval(a, b)
    for _ in range(3):
        mx.eval(a @ b)
    best = 0.0
    for _ in range(5):
        outs = [a @ b for _ in range(8)]
        t0 = time.perf_counter()
        mx.eval(*outs)
        dt_s = (time.perf_counter() - t0) / 8
        best = max(best, 2.0 * n**3 / dt_s / 1e12)
    return best


def pipeline_stats(kernel: CompiledKernel, contract, device_info: dict, *, kernel_source: str | None = None) -> dict:
    """@brief 汇总单个 pipeline 的线程组上限、SIMD 宽度与源码级压力。

    @param kernel 已编译的核。
    @param contract 任务契约，提供启动线程数。
    @param device_info 设备信息映射。
    @param kernel_source 候选源码，用于统计 fragment 声明。
    @return 统计字典，压力可见时带 ``source_pressure``。
    """
    max_threads = kernel.max_total_threads
    cap = int(device_info.get("max_threads_per_threadgroup") or 0)
    pressure = source_pressure(kernel_source)
    hints = []
    if cap and max_threads < cap:
        hints.append(
            f"maxTotalThreadsPerThreadgroup={max_threads} < {cap}: this pipeline is limited to {max_threads} threads per threadgroup. "
            "The field alone does not distinguish register pressure, threadgroup memory, or an explicit limit; "
            "do not raise the launch size solely from this hint."
        )
    tg_frac = kernel.static_tg_bytes / max(device_info["max_threadgroup_memory_bytes"], 1)
    if tg_frac > 0.5:
        hints.append(f"static threadgroup memory is {kernel.static_tg_bytes} B ({tg_frac:.0%} of the {device_info['max_threadgroup_memory_bytes']} B limit); occupancy limited")
    hint = pressure_hint(pressure)
    if hint:
        hints.append(hint)
    out = {
        "max_total_threads_per_threadgroup": max_threads,
        "thread_execution_width": kernel.execution_width,
        "static_threadgroup_bytes": kernel.static_tg_bytes,
        "max_threadgroup_memory_bytes": int(device_info["max_threadgroup_memory_bytes"]),
        "launch_threads_per_threadgroup": contract.threads_per_tg,
        "launch_threadgroups": contract.num_threadgroups,
        "hints": hints,
    }
    if pressure:
        out["source_pressure"] = pressure
    return out


def pipeline_stats_many(pairs, device_info: dict, *, kernel_source: str | None = None) -> dict:
    """@brief 汇总多段流水线的统计，顶层字段沿用单核形状。

    @param pairs ``(compiled kernel, stage)`` 的序列。
    @param device_info 设备信息映射。
    @param kernel_source 候选源码，用于统计 fragment 声明。
    @return 统计字典，``stages`` 为逐段结果列表。

    ``pairs`` is ``(compiled kernel, stage)``. Top-level keys stay the old shape; ``stages`` is the per-stage list.
    """
    class _View:
        def __init__(self, stage):
            self.threads_per_tg = stage.threads[0] * stage.threads[1] * stage.threads[2]
            self.num_threadgroups = stage.grid[0] * stage.grid[1] * stage.grid[2]

    parts = []
    for kernel, stage in pairs:
        one = pipeline_stats(kernel, _View(stage), device_info)
        one["kernel"] = stage.kernel
        one["grid"] = list(stage.grid)
        one["threads"] = list(stage.threads)
        one["threadgroup_bytes"] = stage.threadgroup_bytes
        one["workspace"] = list(stage.workspaces)
        parts.append(one)
    hints = [h for part in parts for h in part["hints"]]
    pressure = source_pressure(kernel_source)
    hint = pressure_hint(pressure)
    if hint:
        hints.append(hint)
    out = {
        "max_total_threads_per_threadgroup": min(p["max_total_threads_per_threadgroup"] for p in parts),
        "thread_execution_width": parts[0]["thread_execution_width"],
        "static_threadgroup_bytes": max(p["static_threadgroup_bytes"] for p in parts),
        "max_threadgroup_memory_bytes": parts[0]["max_threadgroup_memory_bytes"],
        "launch_threads_per_threadgroup": parts[-1]["launch_threads_per_threadgroup"],
        "launch_threadgroups": parts[-1]["launch_threadgroups"],
        "hints": hints,
        "stages": parts,
    }
    if pressure:
        out["source_pressure"] = pressure
    return out


def effective_peak_gbps(task, calib: dict) -> float:
    """@brief 按任务的读写比例插值出可达带宽。

    @param task 任务对象，用契约输入输出的字节数算写比例。
    @param calib 标定数据。
    @return 可达带宽（GB/s）。

    Achievable bandwidth depends on the read/write mix: pure reads reach ``read_gbps``, an even read/write
    stream (copy) only ``copy_gbps``. Interpolate on the write fraction of the task's minimal traffic.
    """
    c = task.contract
    rd = sum(a.nbytes for a in c.inputs)
    wr = sum(a.nbytes for a in c.outputs)
    wf = wr / max(rd + wr, 1)
    read, copy = calib.get("read_gbps", calib["peak_gbps"]), calib.get("copy_gbps", calib["peak_gbps"])
    return read - (read - copy) * min(wf / 0.5, 1.0)


def roofline(task, time_s: float, calib: dict) -> dict:
    """@brief 给出该任务这次测量的 roofline 位置与上下界。

    @param task 任务对象。
    @param time_s 实测耗时（秒）。
    @param calib 标定数据。
    @return roofline 指标字典。
    """
    peak_gbps = effective_peak_gbps(task, calib)
    peak_bw = peak_gbps * 1e9
    peak_fl = calib["peak_tflops"].get(task.dtype, calib["peak_tflops"]["fp16"]) * 1e12
    t_mem = task.min_bytes / peak_bw
    t_cmp = task.flops / peak_fl if task.flops else 0.0
    bound = "compute" if t_cmp > t_mem else "memory"
    t_lb = max(t_mem, t_cmp)
    ai = task.flops / task.min_bytes if task.min_bytes else 0.0
    return {
        "bound": bound,
        "min_bytes": task.min_bytes,
        "flops": task.flops,
        "arithmetic_intensity": ai,
        "ridge_intensity": peak_fl / peak_bw,
        "achieved_gbps": task.min_bytes / time_s / 1e9,
        "bandwidth_fraction": (task.min_bytes / time_s) / peak_bw,
        "achieved_tflops": task.flops / time_s / 1e12 if task.flops else 0.0,
        "compute_fraction": (task.flops / time_s) / peak_fl if task.flops else 0.0,
        "lower_bound_us": t_lb * 1e6,
        "time_over_lower_bound": time_s / t_lb if t_lb > 0 else None,
        "peak_gbps_measured": peak_gbps,
        "peak_tflops_measured": calib["peak_tflops"].get(task.dtype),
        "peak_tflops_by_dtype": dict(calib["peak_tflops"]),
        "bound_kind": "reference_algorithm",
    }
