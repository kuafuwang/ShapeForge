"""Isolate command-buffer structure from output-buffer dependencies for softmax.

This is a local diagnostic, not a search and not publication evidence.  It uses one
already-evaluated kernel and never calls a model.  The production arm is the real
``run_pipeline`` path.  The per-launch arms distinguish three dependency structures:

* ``per_cb_group_wait``: one command buffer per launch, but every rotation waits for
  all three launches of the preceding rotation (the same dependency granularity as
  the production rotate barrier).
* ``per_cb_set_wait``: one command buffer per launch and one shared-event dependency
  per buffer set.  For this single-stage, read-only-x/write-only-y kernel this is the
  minimal safe dependency: launch i+3 waits for launch i, independent sets may overlap.
* ``per_cb_no_wait``: diagnostic only.  Reusing three outputs without waits is not a
  generally safe protocol even when repeated identical writes happen to look correct.
* ``per_cb_fresh_y``: diagnostic only.  Every launch gets a distinct preallocated y,
  which approximates MLX's independent-output dependency graph but does not price
  allocation the same way.  It must not become a published speedup.

Usage (GPU required):
    uv run python experiments/softmax_command_buffer_probe.py \
      --kernel runs/apple-m4-max/softmax-ledger-verify-20261110/best.metal \
      --output runs/apple-m4-max/softmax-ledger-verify-20261110/scratch/cb-probe.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import statistics
import time
from pathlib import Path

import numpy as np

from tcenv.backends.metal.runtime import BufferArg, get_context
from tcenv.backends.registry import load_backend
from tcenv.gpulock import gpu_lock

M, N = 8192, 2048
BYTES = M * N * 2
NSETS = 3
HEADER = (
    "#include <metal_stdlib>\n"
    "#include <metal_simdgroup>\n"
    "using namespace metal;\n"
    "constant constexpr uint N = 2048u;\n"
    "constant constexpr uint M = 8192u;\n"
)


def _median(values: list[float]) -> float:
    return float(statistics.median(values))


def _stats(samples: list[float]) -> dict:
    mean = statistics.fmean(samples)
    return {
        "median_us": _median(samples) * 1e6,
        "mean_us": mean * 1e6,
        "min_us": min(samples) * 1e6,
        "cv": (statistics.pstdev(samples) / mean) if mean > 0 and len(samples) > 1 else None,
        "samples_us": [x * 1e6 for x in samples],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kernel", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--k", type=int, default=36, help="launches per batch; divisible by 3")
    parser.add_argument("--reps", type=int, default=15)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    if args.k < NSETS or args.k % NSETS:
        raise SystemExit("--k must be a positive multiple of 3")
    if args.reps < 1:
        raise SystemExit("--reps must be positive")

    import Metal
    import mlx.core as mx

    ctx = get_context()
    load_backend("metal").prepare_context(ctx)
    source = args.kernel.read_bytes()
    kern = ctx.compile(HEADER + source.decode("utf-8"), "softmax_kernel")
    grid, threads = (M // 4, 1, 1), (512, 1, 1)
    mtl_grid = Metal.MTLSizeMake(*grid)
    mtl_threads = Metal.MTLSizeMake(*threads)

    rng = np.random.default_rng(23)
    data = rng.standard_normal((M, N), dtype=np.float32).astype(np.float16)
    blob = data.tobytes()

    def make_sets(n: int) -> tuple[list, list]:
        xs, ys = [], []
        for _ in range(n):
            x = ctx.new_buffer(BYTES)
            ctx.view(x)[:] = blob
            xs.append(x)
            ys.append(ctx.new_buffer(BYTES))
        return xs, ys

    xs3, ys3 = make_sets(NSETS)
    fresh_y = [ctx.new_buffer(BYTES) for _ in range(args.k)]
    mx_sets = [mx.array(data) for _ in range(NSETS)]
    expected = np.array(mx.softmax(mx_sets[0], axis=-1, precise=True).astype(mx.float32))

    def encode_launch(cb, x, y):
        enc = cb.computeCommandEncoderWithDispatchType_(Metal.MTLDispatchTypeConcurrent)
        enc.setComputePipelineState_(kern.pso)
        enc.setBuffer_offset_atIndex_(x, 0, 0)
        enc.setBuffer_offset_atIndex_(y, 0, 1)
        enc.dispatchThreadgroups_threadsPerThreadgroup_(mtl_grid, mtl_threads)
        enc.endEncoding()

    def check_cbs(cbs: list) -> None:
        for cb in cbs:
            cb.waitUntilCompleted()
            if cb.status() != Metal.MTLCommandBufferStatusCompleted:
                err = cb.error()
                raise RuntimeError(str(err.localizedDescription()) if err is not None else f"command buffer status {cb.status()}")

    def production(k: int) -> float:
        sets = [[[(0, BufferArg(xs3[i])), (1, BufferArg(ys3[i]))]] for i in range(NSETS)]
        timing = ctx.run_pipeline([kern], [grid], [threads], [0], sets, [[] for _ in range(NSETS)], iters=k, dispatch="concurrent")
        return timing.wall_s

    def one_cb_python(k: int) -> float:
        t0 = time.perf_counter()
        cb = ctx.queue.commandBuffer()
        enc = cb.computeCommandEncoderWithDispatchType_(Metal.MTLDispatchTypeConcurrent)
        for i in range(k):
            if i and i % NSETS == 0:
                enc.memoryBarrierWithScope_(Metal.MTLBarrierScopeBuffers)
            s = i % NSETS
            enc.setComputePipelineState_(kern.pso)
            enc.setBuffer_offset_atIndex_(xs3[s], 0, 0)
            enc.setBuffer_offset_atIndex_(ys3[s], 0, 1)
            enc.dispatchThreadgroups_threadsPerThreadgroup_(mtl_grid, mtl_threads)
        enc.endEncoding()
        cb.commit()
        check_cbs([cb])
        return time.perf_counter() - t0

    def per_cb_group_wait(k: int) -> float:
        t0 = time.perf_counter()
        events = [ctx.device.newSharedEvent() for _ in range(k)]
        cbs = []
        for i in range(k):
            s = i % NSETS
            cb = ctx.queue.commandBuffer()
            if i >= NSETS and s == 0:
                # Match the production rotate barrier: every member of the preceding
                # rotation must finish before this rotation starts.
                for previous in range(i - NSETS, i):
                    cb.encodeWaitForEvent_value_(events[previous], 1)
            encode_launch(cb, xs3[s], ys3[s])
            cb.encodeSignalEvent_value_(events[i], 1)
            cb.commit()
            cbs.append(cb)
        check_cbs(cbs)
        return time.perf_counter() - t0

    def per_cb_set_wait(k: int, distinct_x: list | None = None) -> float:
        t0 = time.perf_counter()
        events = [ctx.device.newSharedEvent() for _ in range(NSETS)]
        cbs = []
        for i in range(k):
            s = i % NSETS
            generation = i // NSETS
            cb = ctx.queue.commandBuffer()
            if generation:
                cb.encodeWaitForEvent_value_(events[s], generation)
            encode_launch(cb, distinct_x[i] if distinct_x is not None else xs3[s], ys3[s])
            cb.encodeSignalEvent_value_(events[s], generation + 1)
            cb.commit()
            cbs.append(cb)
        check_cbs(cbs)
        return time.perf_counter() - t0

    def per_cb_no_wait(k: int) -> float:
        t0 = time.perf_counter()
        cbs = []
        for i in range(k):
            s = i % NSETS
            cb = ctx.queue.commandBuffer()
            encode_launch(cb, xs3[s], ys3[s])
            cb.commit()
            cbs.append(cb)
        check_cbs(cbs)
        return time.perf_counter() - t0

    def per_cb_fresh_y(k: int) -> float:
        t0 = time.perf_counter()
        cbs = []
        for i in range(k):
            cb = ctx.queue.commandBuffer()
            encode_launch(cb, xs3[i % NSETS], fresh_y[i])
            cb.commit()
            cbs.append(cb)
        check_cbs(cbs)
        return time.perf_counter() - t0

    def mlx_baseline(k: int) -> float:
        outs = [mx.softmax(mx_sets[i % NSETS], axis=-1, precise=True) for i in range(k)]
        t0 = time.perf_counter()
        mx.eval(*outs)
        return time.perf_counter() - t0

    runners = {
        "production_1cb": production,
        "python_1cb_group_barrier": one_cb_python,
        "per_cb_group_wait": per_cb_group_wait,
        "per_cb_set_wait": per_cb_set_wait,
        "per_cb_no_wait_UNSAFE": per_cb_no_wait,
        "per_cb_fresh_y_DIAGNOSTIC": per_cb_fresh_y,
        "mlx": mlx_baseline,
    }

    def output_errors() -> dict[str, float]:
        def error_for(buf) -> float:
            got = np.array(ctx.view(buf).cast("h"), dtype=np.uint16).view(np.float16).astype(np.float32).reshape(M, N)
            return float(np.max(np.abs(got - expected)))

        return {
            "shared_y_set0": error_for(ys3[0]),
            "per_cb_fresh_y_last": error_for(fresh_y[-1]),
        }

    load_before = os.getloadavg()
    with gpu_lock("softmax_command_buffer_probe"):
        # Warm every path, then verify that the final stored values are the expected softmax.
        for fn in runners.values():
            fn(args.k)
        errors = output_errors()
        if max(errors.values()) > 0.003:
            raise RuntimeError(f"probe correctness failed: {errors}")
        samples: dict[str, list[float]] = {name: [] for name in runners}
        order_rng = random.Random(args.seed)
        for rep in range(args.reps + 2):
            order = list(runners)
            order_rng.shuffle(order)
            for name in order:
                elapsed = runners[name](args.k) / args.k
                if rep >= 2:
                    samples[name].append(elapsed)
        errors_after = output_errors()

        # A repeated identical write cannot expose a missing dependency.  Use six distinct
        # inputs over the same three outputs: each final y must belong to the last launch
        # for that set, not whichever overlapping launch happened to finish last.
        verify_data = [rng.standard_normal((M, N), dtype=np.float32).astype(np.float16) + np.float16(i * 0.25) for i in range(6)]
        verify_x = []
        for arr in verify_data:
            buf = ctx.new_buffer(BYTES)
            ctx.view(buf)[:] = arr.tobytes()
            verify_x.append(buf)
        per_cb_set_wait(6, verify_x)
        dependency_errors = {}
        for s in range(NSETS):
            last = s + NSETS
            want = np.array(mx.softmax(mx.array(verify_data[last]), axis=-1, precise=True).astype(mx.float32))
            got = np.array(ctx.view(ys3[s]).cast("h"), dtype=np.uint16).view(np.float16).astype(np.float32).reshape(M, N)
            dependency_errors[f"set{s}_last_launch{last}"] = float(np.max(np.abs(got - want)))
        if max(dependency_errors.values()) > 0.003:
            raise RuntimeError(f"distinct-input dependency test failed: {dependency_errors}")

    stats = {name: _stats(values) for name, values in samples.items()}
    mlx_us = stats["mlx"]["median_us"]
    production_us = stats["production_1cb"]["median_us"]
    comparison = {
        name: {
            "speedup_vs_mlx": mlx_us / row["median_us"],
            "delta_vs_production_us": row["median_us"] - production_us,
        }
        for name, row in stats.items()
    }
    doc = {
        "schema_id": "shapeforge.softmax-command-buffer-probe.1",
        "publication_allowed": False,
        "kernel": str(args.kernel),
        "kernel_sha256": hashlib.sha256(source).hexdigest(),
        "shape": [M, N],
        "dtype": "fp16",
        "logical_bytes_per_launch": BYTES * 2,
        "k": args.k,
        "reps": args.reps,
        "rotation_sets": NSETS,
        "seed": args.seed,
        "device": ctx.info().get("name"),
        "architecture": platform.machine(),
        "loadavg_before_after": [load_before, os.getloadavg()],
        "correctness_max_abs_before": errors,
        "correctness_max_abs_after": errors_after,
        "distinct_input_dependency_test": {"k": 6, "rotation_sets": NSETS, "max_abs": dependency_errors},
        "stats": stats,
        "comparison": comparison,
        "notes": [
            "per_cb_no_wait_UNSAFE is not a legal general harness protocol.",
            "per_cb_fresh_y_DIAGNOSTIC preallocates outputs and is not allocation-cost comparable to MLX.",
            "per_cb_set_wait is checked only for this single-stage read-only-x/write-only-y task; it is not a general multi-stage protocol.",
            "This probe does not change production timing or any acceptance record.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"k={args.k} reps={args.reps} kernel={args.kernel}")
    print(f"correctness max abs: {max(errors_after.values()):.6g}")
    for name, row in stats.items():
        cmp = comparison[name]
        print(
            f"{name:30s} {row['median_us']:9.3f} us  cv={row['cv']:.3%}  "
            f"vs_prod={cmp['delta_vs_production_us']:+8.3f} us  vs_mlx={cmp['speedup_vs_mlx']:.5f}x"
        )
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
