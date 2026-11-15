"""Re-runnable, non-publishable gemm_silu global-scratch cost probe.

P2-O asks one question: for ``C = silu(A @ B)``, should the fp32 accumulator stay in
registers (REG) or be written to global scratch and read back by a second kernel
(GLOBAL)?  This probe measures the costs that decision needs on real Metal:

* ``single_stage_fp16_write``: one kernel writes the whole fp16 output.  This is the
  REG output-store baseline, without a GEMM.
* ``two_stage_floor``: producer -> barrier -> consumer over a four-texel texture
  scratch, no fill.  This is the extra launch/barrier floor.
* ``two_stage_texture_roundtrip``: the fairest GLOBAL byte flow.  Every logical fp32
  element is written to and read from its own R32 texel (four scalar texel writes /
  reads per thread), then the consumer writes the fp16 output.  No fill, but the full
  fp32 write + read is there.
* ``two_stage_buffer_roundtrip``: the same byte flow through a buffer scratch, which the
  lifecycle only accepts with ``device_zero``.  The fill is inside the timed command
  buffer, exactly as the harness would bill it, and is included here.

The texture arms carry a **coverage check**, not just a byte-count claim: the producer
writes ``float((i & 1023) + 1)`` into element ``i`` and the consumer writes the same
value to output ``i``.  After a run the probe reads the whole scratch and output back
and requires every element to match.  A deliberately broken quarter-coverage kernel
(the previous ``float4``-write-at-one-coordinate bug) is run through the same check and
must be detected; this is the regression guard.

The kernels are deliberately incomplete: they do not compute a GEMM and cannot pass
the task's correctness oracle.  ``not_a_candidate=true``; nothing here is a candidate,
a speedup or a publication.  The real REG candidates are measured separately through
``evaluate_candidate`` / ``confirm_candidate``.
"""
from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from tcenv.backends.metal.launch import parse_launch_spec
from tcenv.backends.metal.runtime import get_context
from tcenv.backends.metal.scratch_texture import scratch_texture_layout
from tcenv.backends.metal.stage_binding import bind_pipeline_arguments
from tcenv.backends.metal.track_adapter import msl_single_geometry_resolver
from tcenv.backends.metal.v2_runtime import V2ArgSet, reflection_for_kernel
from tcenv.backends.registry import load_backend
from tcenv.gpulock import gpu_lock
from tcenv.target import load_target
from tcenv.taskcontract import load_task_contract
from tcenv.taskcontract.locator import REPO_ROOT
from tcenv.track import (
    GUARD_BYTES,
    abi_limits_from_target,
    candidate_from_single_source,
    derive_tensor_memory_contract,
    resolve_geometry,
)

SCHEMA_ID = "shapeforge.gemm-silu-scratch-probe.2"
TARGET_ID = "apple-m4-max"
SIZES = ("small", "paper")
CASES = ("normal", "wide_range", "large", "tiny", "outlier", "ieee")
ARMS = (
    "single_stage_fp16_write",
    "two_stage_floor",
    "two_stage_texture_roundtrip",
    "two_stage_buffer_roundtrip",
)
PATTERN_MODULUS = 1024


def _source(symbols: dict[str, int], arm: str, texture: tuple[int, int, int] | None = None, *, quarter: bool = False) -> str:
    """Build one intentionally incomplete probe arm for this shape."""
    total = int(symbols["M"]) * int(symbols["N"])
    header = "#include <metal_stdlib>\nusing namespace metal;\n" + "".join(
        f"constant constexpr uint {name} = {value}u;\n" for name, value in symbols.items()
    ) + (
        f"inline float probe_value(uint i) {{ return float((i & {PATTERN_MODULUS - 1}u) + 1u); }}\n"
    )
    if arm == "single_stage_fp16_write":
        return header + f"""
// Probe only: writes {total} fp16 elements with 16 B stores; no GEMM.
kernel void probe_write_half(device const half* A [[buffer(0)]],
                             device const half* B [[buffer(1)]],
                             device half* C [[buffer(2)]],
                             uint gid [[thread_position_in_grid]]) {{
  const uint i = gid * 8u;
  if (i + 8u <= M * N) {{
    *reinterpret_cast<device uint4*>(C + i) = uint4(0u);
  }}
}}
/* taic-launch
math fast
launch probe_write_half grid={total // (8 * 256)},1,1 threads=256,1,1
*/
"""
    if arm == "two_stage_floor":
        return header + """
// Probe only: four fp32 texel writes and reads; no fill, no traffic meaning.
kernel void probe_touch_producer(device const half* A [[buffer(0)]],
                                 texture2d<float, access::write> partial [[texture(0)]],
                                 uint lid [[thread_index_in_threadgroup]]) {
  if (lid == 0u) {
    partial.write(float4(1.0f), uint2(0, 0));
    partial.write(float4(2.0f), uint2(1, 0));
    partial.write(float4(3.0f), uint2(2, 0));
    partial.write(float4(4.0f), uint2(3, 0));
  }
}
kernel void probe_touch_consumer(device const half* A [[buffer(0)]],
                                 device const half* B [[buffer(1)]],
                                 device half* C [[buffer(2)]],
                                 texture2d<float, access::read> partial [[texture(0)]],
                                 uint lid [[thread_index_in_threadgroup]]) {
  if (lid == 0u) {
    const float v = partial.read(uint2(0, 0)).x + partial.read(uint2(1, 0)).x
                  + partial.read(uint2(2, 0)).x + partial.read(uint2(3, 0)).x;
    C[0] = half(v);
  }
}
/* taic-launch
math fast
scratch partial fp32 4
launch probe_touch_producer grid=1,1,1 threads=32,1,1 scratch=partial:write
launch probe_touch_consumer grid=1,1,1 threads=32,1,1 scratch=partial:read
*/
"""
    if arm == "two_stage_texture_roundtrip":
        assert texture is not None
        width, _height, _row = texture
        assert total % width == 0 and width % 4 == 0
        if quarter:
            # Deliberately broken control: one float4 write at one R32 coordinate
            # reaches only one of the four logical elements. Kept only to prove the
            # coverage check catches the old bug.
            return header + f"""
// Negative control only: quarter coverage, the old buggy texture kernel.
kernel void probe_tex_producer(device const half* A [[buffer(0)]],
                               texture2d<float, access::write> partial [[texture(0)]],
                               uint gid [[thread_position_in_grid]]) {{
  const uint i = gid * 4u;
  if (i + 4u <= M * N) {{
    const uint y = i / {width}u;
    const uint x = i - y * {width}u;
    partial.write(float4(probe_value(i + 0u), probe_value(i + 1u),
                         probe_value(i + 2u), probe_value(i + 3u)), uint2(x, y));
  }}
}}
kernel void probe_tex_consumer(device const half* A [[buffer(0)]],
                               device const half* B [[buffer(1)]],
                               device half* C [[buffer(2)]],
                               texture2d<float, access::read> partial [[texture(0)]],
                               uint gid [[thread_position_in_grid]]) {{
  const uint i = gid * 4u;
  if (i + 4u <= M * N) {{
    const uint y = i / {width}u;
    const uint x = i - y * {width}u;
    const float v = partial.read(uint2(x, y)).x;
    *reinterpret_cast<device half4*>(C + i) = half4(half(v), half(v), half(v), half(v));
  }}
}}
/* taic-launch
math fast
scratch partial fp32 M*N
launch probe_tex_producer grid={total // (4 * 256)},1,1 threads=256,1,1 scratch=partial:write
launch probe_tex_consumer grid={total // (4 * 256)},1,1 threads=256,1,1 scratch=partial:read
*/
"""
        return header + f"""
// Probe only: every logical fp32 has its own R32 texel write and read. texture width {width}.
kernel void probe_tex_producer(device const half* A [[buffer(0)]],
                               texture2d<float, access::write> partial [[texture(0)]],
                               uint gid [[thread_position_in_grid]]) {{
  const uint i = gid * 4u;
  if (i + 4u <= M * N) {{
    const uint y = i / {width}u;
    const uint x = i - y * {width}u;
    partial.write(float4(probe_value(i + 0u)), uint2(x + 0u, y));
    partial.write(float4(probe_value(i + 1u)), uint2(x + 1u, y));
    partial.write(float4(probe_value(i + 2u)), uint2(x + 2u, y));
    partial.write(float4(probe_value(i + 3u)), uint2(x + 3u, y));
  }}
}}
kernel void probe_tex_consumer(device const half* A [[buffer(0)]],
                               device const half* B [[buffer(1)]],
                               device half* C [[buffer(2)]],
                               texture2d<float, access::read> partial [[texture(0)]],
                               uint gid [[thread_position_in_grid]]) {{
  const uint i = gid * 4u;
  if (i + 4u <= M * N) {{
    const uint y = i / {width}u;
    const uint x = i - y * {width}u;
    const float v0 = partial.read(uint2(x + 0u, y)).x;
    const float v1 = partial.read(uint2(x + 1u, y)).x;
    const float v2 = partial.read(uint2(x + 2u, y)).x;
    const float v3 = partial.read(uint2(x + 3u, y)).x;
    *reinterpret_cast<device half4*>(C + i) = half4(half(v0), half(v1), half(v2), half(v3));
  }}
}}
/* taic-launch
math fast
scratch partial fp32 M*N
launch probe_tex_producer grid={total // (4 * 256)},1,1 threads=256,1,1 scratch=partial:write
launch probe_tex_consumer grid={total // (4 * 256)},1,1 threads=256,1,1 scratch=partial:read
*/
"""
    if arm == "two_stage_buffer_roundtrip":
        return header + f"""
// Probe only: the same GLOBAL byte flow through a buffer. The lifecycle only accepts
// this with device_zero, so the timed fill (harness byte-granular fill kernel) is here.
kernel void probe_buf_producer(device const half* A [[buffer(0)]],
                               device float* partial [[buffer(1)]],
                               uint gid [[thread_position_in_grid]]) {{
  const uint i = gid * 4u;
  if (i + 4u <= M * N) {{
    *reinterpret_cast<device float4*>(partial + i) =
        float4(probe_value(i + 0u), probe_value(i + 1u),
               probe_value(i + 2u), probe_value(i + 3u));
  }}
}}
kernel void probe_buf_consumer(device const half* A [[buffer(0)]],
                               device const half* B [[buffer(1)]],
                               device half* C [[buffer(2)]],
                               device const float* partial [[buffer(3)]],
                               uint gid [[thread_position_in_grid]]) {{
  const uint i = gid * 4u;
  if (i + 4u <= M * N) {{
    const float4 v = *reinterpret_cast<device const float4*>(partial + i);
    *reinterpret_cast<device half4*>(C + i) =
        half4(half(v.x), half(v.y), half(v.z), half(v.w));
  }}
}}
/* taic-launch
math fast
scratch partial fp32 M*N zero
launch probe_buf_producer grid={total // (4 * 256)},1,1 threads=256,1,1 scratch=partial:write
launch probe_buf_consumer grid={total // (4 * 256)},1,1 threads=256,1,1 scratch=partial:read
*/
"""
    raise ValueError(f"unknown probe arm {arm!r}")


@dataclass
class _Prepared:
    spec: object
    resolved: object
    kernels: list
    bindings: list
    grids: list
    threads: list
    sets: list
    args: list
    fills: list


def _prepare(ctx, contract, abi, limits, symbols: dict[str, int], source: str, *, nsets: int) -> _Prepared:
    spec = parse_launch_spec(source, symbols)
    resolved = resolve_geometry(
        msl_single_geometry_resolver(contract)(candidate_from_single_source(source)),
        instance=contract.instance,
        contract=abi,
        limits=limits,
    )
    kernels = list(ctx.compile_all(source, [stage.kernel for stage in spec.stages], math=spec.math))
    reflection = {kernel.name: tuple(reflection_for_kernel(kernel)) for kernel in kernels}
    bindings = bind_pipeline_arguments(resolved, contract=abi, reflection=reflection, limits=limits)
    sets = [V2ArgSet(ctx, abi, resolved) for _ in range(nsets)]
    for one in sets:
        one.buffers["A"].write(np.ones(int(symbols["M"]) * int(symbols["K"]), dtype=np.float16).tobytes())
        one.buffers["B"].write(np.ones(int(symbols["K"]) * int(symbols["N"]), dtype=np.float16).tobytes())
    args = [[one.stage_args(binding) for binding in bindings] for one in sets]
    fills = [one.workspace_fills_declared_zero() for one in sets]
    return _Prepared(
        spec=spec,
        resolved=resolved,
        kernels=kernels,
        bindings=bindings,
        grids=[stage.grid for stage in spec.stages],
        threads=[stage.threads for stage in spec.stages],
        sets=sets,
        args=args,
        fills=fills,
    )


def _jsonable(value: float) -> float | str:
    return float(value) if np.isfinite(value) else ("nan" if np.isnan(value) else ("inf" if value > 0 else "-inf"))


def _first_mismatch(got: np.ndarray, want: np.ndarray) -> dict:
    bad = np.nonzero(got != want)[0]
    index = int(bad[0])
    return {"index": index, "got": _jsonable(float(got[index])), "want": _jsonable(float(want[index]))}


def _coverage(prepared: _Prepared, symbols: dict[str, int], ctx) -> dict:
    """Run one iteration, then read back every scratch element and every output element."""
    total = int(symbols["M"]) * int(symbols["N"])
    ctx.run_pipeline(
        prepared.kernels,
        prepared.grids,
        prepared.threads,
        [0 for _ in prepared.spec.stages],
        prepared.args[:1],
        prepared.fills[:1],
        iters=1,
        dispatch="concurrent",
    )
    want = (np.arange(total, dtype=np.int64) % PATTERN_MODULUS + 1).astype(np.float32)
    raw = prepared.sets[0].workspace_bufs["partial"].read()
    got_scratch = np.frombuffer(raw, dtype=np.float32)[:total]
    out = prepared.sets[0].buffers["C"].read()
    got_output = np.frombuffer(out, dtype=np.float16)[:total]
    want_output = want.astype(np.float16)
    scratch_ok = bool(np.array_equal(got_scratch, want))
    output_ok = bool(np.array_equal(got_output, want_output))
    document = {
        "elements": total,
        "pattern": f"float((i & {PATTERN_MODULUS - 1}) + 1)",
        "scratch_elements_read_back": int(got_scratch.size),
        "output_elements_read_back": int(got_output.size),
        "scratch_elements_match": scratch_ok,
        "output_elements_match": output_ok,
        "ok": scratch_ok and output_ok,
    }
    if not scratch_ok:
        document["first_scratch_mismatch"] = _first_mismatch(got_scratch, want)
    if not output_ok:
        document["first_output_mismatch"] = _first_mismatch(got_output, want_output)
    return document


def _measure(ctx, contract, abi, limits, symbols: dict[str, int], source: str, *, with_coverage: bool) -> tuple[dict, dict | None]:
    prepared = _prepare(ctx, contract, abi, limits, symbols, source, nsets=6)
    ctx.run_pipeline(
        prepared.kernels, prepared.grids, prepared.threads, [0 for _ in prepared.spec.stages],
        prepared.args, prepared.fills, iters=20, dispatch="concurrent",
    )
    walls, gpus = [], []
    for _ in range(9):
        timing = ctx.run_pipeline(
            prepared.kernels, prepared.grids, prepared.threads, [0 for _ in prepared.spec.stages],
            prepared.args, prepared.fills, iters=100, dispatch="concurrent",
        )
        walls.append(float(timing.wall_s) / 100 * 1e6)
        gpus.append(float(timing.gpu_s) / 100 * 1e6)
    coverage = _coverage(prepared, symbols, ctx) if with_coverage else None
    arm = {
        "wall_us_median": statistics.median(walls),
        "gpu_us_median": statistics.median(gpus),
        "wall_samples_us": walls,
        "gpu_samples_us": gpus,
        "iters_per_sample": 100,
        "rotation_sets": len(prepared.sets),
        "dispatch": "concurrent, one command buffer, native batch when eligible",
        "stages": len(prepared.spec.stages),
        "stage_kernels": [stage.kernel for stage in prepared.spec.stages],
        "grids": [list(stage.grid) for stage in prepared.spec.stages],
        "threads": [list(stage.threads) for stage in prepared.spec.stages],
        "scratch_name": prepared.spec.scratches[0].name if getattr(prepared.spec, "scratches", ()) else None,
        "scratch_bytes": sum(int(item.nbytes) for item in getattr(prepared.resolved, "scratch", ())),
        "fill_bytes_per_iteration": sum(int(fill[2]) for fill in prepared.fills[0]),
    }
    if coverage is not None:
        arm["coverage"] = coverage
    return arm, coverage


def _quarter_negative(ctx, contract, abi, limits, symbols: dict[str, int], texture) -> dict:
    source = _source(symbols, "two_stage_texture_roundtrip", texture, quarter=True)
    prepared = _prepare(ctx, contract, abi, limits, symbols, source, nsets=1)
    coverage = _coverage(prepared, symbols, ctx)
    return {
        "description": "the old float4-write-at-one-R32-coordinate texture kernel",
        "detected": not coverage["ok"],
        "scratch_elements_match": coverage["scratch_elements_match"],
        "output_elements_match": coverage["output_elements_match"],
        "first_scratch_mismatch": coverage.get("first_scratch_mismatch"),
        "first_output_mismatch": coverage.get("first_output_mismatch"),
    }


def run_probe(output: Path) -> dict:
    target = load_target(TARGET_ID)
    ctx = get_context()
    # The composition root owns the fixed internal fill kernel; the runtime may not
    # compile it on its own. prepare_context also installs the native batch encoder.
    load_backend("metal").prepare_context(ctx)
    document = {
        "schema_id": SCHEMA_ID,
        "not_a_candidate": True,
        "publication_allowed": False,
        "target_id": TARGET_ID,
        "device_name": target.calibration_device_name,
        "correctness": "intentionally incomplete launch/byte-flow kernels; never evaluated and not publishable",
        "cases_not_run": list(CASES),
        "coverage_method": (
            "producer writes float((i & 1023) + 1) into element i; after a run the probe reads "
            "the entire scratch and output back and requires every element to match. The quarter-"
            "coverage control is run through the same check and must be detected."
        ),
        "sizes": {},
    }
    with gpu_lock("gemm-silu-scratch-probe"):
        for size in SIZES:
            contract = load_task_contract(REPO_ROOT / "tasks" / "gemm_silu", dtype="fp16", size=size)
            symbols = {str(name): int(value) for name, value in contract.instance.symbols.items()}
            abi = derive_tensor_memory_contract(contract.instance, target=target)
            limits = abi_limits_from_target(target)
            m, n, k = (symbols[name] for name in ("M", "N", "K"))
            total = m * n
            texture = scratch_texture_layout(total, guard_bytes=GUARD_BYTES)
            arms = {}
            for arm in ARMS:
                arms[arm], _ = _measure(
                    ctx,
                    contract,
                    abi,
                    limits,
                    symbols,
                    _source(symbols, arm, texture),
                    with_coverage=arm in ("two_stage_texture_roundtrip", "two_stage_buffer_roundtrip"),
                )
                print(
                    size, arm, arms[arm]["wall_us_median"], arms[arm]["gpu_us_median"],
                    arms[arm].get("coverage", {}).get("ok"),
                    flush=True,
                )
            negative = _quarter_negative(ctx, contract, abi, limits, symbols, texture)
            print(size, "quarter_coverage_negative", negative, flush=True)
            single = arms["single_stage_fp16_write"]["wall_us_median"]
            floor = arms["two_stage_floor"]["wall_us_median"]
            texture_roundtrip = arms["two_stage_texture_roundtrip"]["wall_us_median"]
            buffer_roundtrip = arms["two_stage_buffer_roundtrip"]["wall_us_median"]
            document["sizes"][size] = {
                "symbols": {"M": m, "N": n, "K": k},
                "texture_layout": {"width": texture[0], "height": texture[1], "bytes_per_row": texture[2]},
                "scratch_fp32_bytes": 4 * total,
                "fp32_fill_bytes": 4 * total,
                "fp32_write_bytes": 4 * total,
                "fp32_read_bytes": 4 * total,
                "fp16_output_bytes": 2 * total,
                "scratch_roundtrip_bytes": 8 * total,
                "arms": arms,
                "quarter_coverage_negative": negative,
                "derived": {
                    "single_stage_fp16_write_us": single,
                    "two_stage_floor_us": floor,
                    "two_stage_texture_roundtrip_us": texture_roundtrip,
                    "two_stage_buffer_roundtrip_us": buffer_roundtrip,
                    "floor_over_single_us": floor - single,
                    "texture_roundtrip_over_single_us": texture_roundtrip - single,
                    "buffer_roundtrip_over_single_us": buffer_roundtrip - single,
                    "texture_roundtrip_over_floor_us": texture_roundtrip - floor,
                    "buffer_roundtrip_over_floor_us": buffer_roundtrip - floor,
                    "texture_roundtrip_gbps": 10 * total / (max(texture_roundtrip, 1e-9) * 1e-6) / 1e9,
                    "buffer_roundtrip_gbps": 14 * total / (max(buffer_roundtrip, 1e-9) * 1e-6) / 1e9,
                },
            }
    document["summary"] = {
        size: {
            arm: document["sizes"][size]["arms"][arm]["wall_us_median"] for arm in ARMS
        }
        | {
            "texture_coverage_ok": document["sizes"][size]["arms"]["two_stage_texture_roundtrip"]["coverage"]["ok"],
            "buffer_coverage_ok": document["sizes"][size]["arms"]["two_stage_buffer_roundtrip"]["coverage"]["ok"],
            "quarter_negative_detected": document["sizes"][size]["quarter_coverage_negative"]["detected"],
            "texture_roundtrip_over_single_us": document["sizes"][size]["derived"]["texture_roundtrip_over_single_us"],
            "buffer_roundtrip_over_single_us": document["sizes"][size]["derived"]["buffer_roundtrip_over_single_us"],
        }
        for size in SIZES
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("SUMMARY", json.dumps(document["summary"], indent=2), flush=True)
    return document


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    run_probe(parser.parse_args().output)


if __name__ == "__main__":
    main()
