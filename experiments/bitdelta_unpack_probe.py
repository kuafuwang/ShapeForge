"""Re-runnable, non-publishable bitdelta_gemm REG-vs-SHARED probe.

The probe asks one question about ``bitdelta_gemm``: for the registered Apple M4 Max shapes, is
it better to decode the packed sign words *inside* each output tile (REG, on-chip, no global
intermediate) or to expand ``W`` into a global matrix once and let the GEMM read that
(SHARED: producer -> barrier -> consumer)?  This probe measures every term of the comparison
on real Metal with the same ABI and the same 64x64x32 GEMM algorithm.

The scratch lifecycle decides which SHARED forms are even legal.  On this toolchain every
non-const ``device`` pointer reflects ``readwrite``, and an ``uninitialized_poison`` buffer
must be written by a kernel that reflects ``write``.  Only an fp32 scratch has the
``R32Float`` write-texture view, so a poison SHARED producer is necessarily fp32; the fp16
alternative must declare ``device_zero`` and pay the harness fill inside the timed window.
Both legal forms are measured here, plus the floor and the packed REG arm:

* ``floor``: producer -> barrier -> consumer over a four-texel fp32 scratch.  The extra
  two-stage launch/barrier floor, with no traffic meaning.
* ``expand_fp32``: the legal poison producer.  It decodes every packed word into exact
  ``+-1`` floats through a write texture and writes the whole ``[K, N]`` fp32 matrix.
  After the run the probe reads the whole scratch back and requires element-wise equality
  with the CPU decode of the same words.  Coverage, not a byte-count claim.
* ``packed``: the REG candidate source, byte for byte.  One stage, per-tile decode into
  threadgroup memory, no global intermediate.
* ``shared_fp32``: ``expand_fp32`` followed by the same GEMM tiling reading the expanded
  fp32 matrix from the scratch buffer.  Producer + barrier + consumer + expansion traffic
  are all inside the timed command buffer.
* ``fill_fp16``: a ``device_zero`` fp16 scratch of ``K*N`` halves with a touch producer and
  a sink consumer.  This isolates the harness fill that the fp16 SHARED form must pay.
* ``expand_fp16``: the fp16 ``device_zero`` producer (threadgroup transpose, exact +-1
  halves) plus a sink consumer: fill + producer.
* ``shared_fp16``: the fp16 producer plus the GEMM consumer reading fp16: fill + producer +
  consumer.  ``shared_fp16 - fill_fp16`` is the estimate for the same route if the contract
  ever allowed a poison fp16 buffer (no fill).

The probe kernels are intentionally not candidates: nothing here is evaluated against the
task oracle or published.  ``not_a_candidate=true``, ``publication_allowed=false``.  The
real REG baseline is measured separately through ``evaluate_candidate``/``confirm_candidate``.
"""
from __future__ import annotations

import argparse
import json
import re
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

SCHEMA_ID = "shapeforge.bitdelta-unpack-probe.2"
SUPERSEDED_SCHEMA_ID = "shapeforge.bitdelta-unpack-probe.1"
TARGET_ID = "apple-m4-max"
SIZES = ("small", "paper")
SHAPES = {"small": (128, 256, 256), "paper": (512, 4096, 4096)}
ARMS = (
    "floor",
    "expand_fp32",
    "packed",
    "shared_fp32",
    "fill_fp16",
    "expand_fp16",
    "shared_fp16",
    "shared_packed16",
)
CASES_NOT_RUN = ("random", "edge_bits", "large", "tiny", "outlier", "ieee")
CANDIDATE_TREE = REPO_ROOT / "tests/support/golden/metal/bitdelta_reg"
SAMPLE_COUNT = 9
ROTATION_SETS = 16
PAIR_ROUNDS = 12
SAMPLE_TARGET_SECONDS = 0.06
WARM_TARGET_SECONDS = 0.15
MAX_ITERS = 200_000
BITS_SEED = 0x5EED_B17D
# Audit of the superseded r1 probe: four uint4 passes per thread wrote 64 k rows into the
# BLOCK_K=32-row threadgroup region (2296 halves = 4592 B past the end) and read the expanded
# B matrix twice per k step.  Kept as a constant so r1 can never be mistaken for a measurement
# of the route.
_R1_OVERRUN_BYTES = 4592
_R1_OVER_READ_RATIO = 2.0

_HEADER = """#include <metal_stdlib>
#include <metal_simdgroup>
#include <metal_simdgroup_matrix>
using namespace metal;
"""


def _header(symbols: dict[str, int]) -> str:
    declarations = "".join(f"constant constexpr uint {name} = {value}u;\n" for name, value in symbols.items())
    return _HEADER + declarations + "constant constexpr float SCALE = 0.02f;\n"


def _constants(m: int, n: int, k: int) -> dict[str, int]:
    return {"M": m, "N": n, "K": k, "KB": k // 32}


def _floor_arm() -> str:
    width, _height, _row = scratch_texture_layout(4, guard_bytes=GUARD_BYTES)
    return f"""
kernel void probe_floor_producer(device const uint* bits [[buffer(0)]],
                                 texture2d<float, access::write> ctrl [[texture(0)]],
                                 uint lid [[thread_index_in_threadgroup]]) {{
  if (lid == 0u) {{
    ctrl.write(float4(1.0f, 2.0f, 3.0f, 4.0f), uint2(0, 0));
  }}
}}

kernel void probe_floor_consumer(device const half* x [[buffer(0)]],
                                 device const uint* bits [[buffer(1)]],
                                 device half* y [[buffer(2)]],
                                 texture2d<float, access::read> ctrl [[texture(0)]],
                                 uint lid [[thread_index_in_threadgroup]]) {{
  if (lid == 0u) {{
    const float4 v = ctrl.read(uint2(0, 0));
    y[0] = half(v.x + v.y + v.z + v.w);
  }}
}}
/* taic-launch
math fast
scratch ctrl fp32 4
launch probe_floor_producer grid=1,1,1 threads=32,1,1 scratch=ctrl:write
launch probe_floor_consumer grid=1,1,1 threads=32,1,1 scratch=ctrl:read
*/
"""


def _expand_fp32_producer(total: int, n: int) -> str:
    width, _height, _row = scratch_texture_layout(total, guard_bytes=GUARD_BYTES)
    return f"""
// Legal poison producer: one R32Float write per decoded element.  Thread `gid` owns four
// consecutive k-major elements, so it reads four consecutive packed words (same bit) and
// issues one 16-byte float4 texture write.
constant constexpr uint ELEM_TOTAL = {total}u;
constant constexpr uint TEX_WIDTH = {width}u;

kernel void probe_expand_f32(device const uint* bits [[buffer(0)]],
                             texture2d<float, access::write> wfull [[texture(0)]],
                             uint gid [[thread_position_in_grid]]) {{
  const uint i = gid * 4u;
  if (i >= ELEM_TOTAL) {{
    return;
  }}
  const uint k = i / N;
  const uint n = i - k * N;
  const uint base = (k >> 5) * N + n;
  const uint j = k & 31u;
  const uint m0 = (bits[base + 0u] >> j) & 1u;
  const uint m1 = (bits[base + 1u] >> j) & 1u;
  const uint m2 = (bits[base + 2u] >> j) & 1u;
  const uint m3 = (bits[base + 3u] >> j) & 1u;
  const uint y = i / TEX_WIDTH;
  const uint x = i - y * TEX_WIDTH;
  // R32Float texture writes address ONE texel per call; a float4 write at one coordinate
  // would cover only the first of the four logical elements (the earlier quarter-coverage bug).
  wfull.write(float4(m0 ? 1.0f : -1.0f), uint2(x + 0u, y));
  wfull.write(float4(m1 ? 1.0f : -1.0f), uint2(x + 1u, y));
  wfull.write(float4(m2 ? 1.0f : -1.0f), uint2(x + 2u, y));
  wfull.write(float4(m3 ? 1.0f : -1.0f), uint2(x + 3u, y));
}}

kernel void probe_expand_f32_sink(device const half* x [[buffer(0)]],
                                  device const uint* bits [[buffer(1)]],
                                  device half* y [[buffer(2)]],
                                  device const float* wfull [[buffer(3)]],
                                  uint lid [[thread_index_in_threadgroup]]) {{
  if (lid == 0u) {{
    y[0] = half(wfull[0]);
  }}
}}
"""


def _expand_fp16_producer(m: int, n: int, k: int) -> str:
    kb = k // 32
    return f"""
// fp16 producer under device_zero: a threadgroup owns 128 weight columns and up to 16
// packed k-rows, decodes them into a 32 x 128 exact +-1 tile and transposes through
// threadgroup memory so the global stores are coalesced 16-byte rows.
constant constexpr uint EXPAND_N = 128u;
constant constexpr uint EXPAND_KB = 16u;

kernel void probe_expand_f16(device const uint* bits [[buffer(0)]],
                             device half* wfull [[buffer(1)]],
                             uint2 tg [[threadgroup_position_in_grid]],
                             uint lid [[thread_index_in_threadgroup]]) {{
  const uint n0 = tg.x * EXPAND_N;
  const uint kb0 = tg.y * EXPAND_KB;
  const uint kb_end = min(kb0 + EXPAND_KB, KB);
  threadgroup half tile[32u * EXPAND_N];
  for (uint kb = kb0; kb < kb_end; ++kb) {{
    const uint word = bits[kb * N + n0 + lid];
#pragma unroll
    for (uint j = 0; j < 32u; ++j) {{
      tile[j * EXPAND_N + lid] = half((((word >> j) & 1u) != 0u) ? 1.0f : -1.0f);
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint u = lid; u < 32u * (EXPAND_N / 8u); u += 128u) {{
      const uint row = u / (EXPAND_N / 8u);
      const uint chunk = u - row * (EXPAND_N / 8u);
      *reinterpret_cast<device uint4*>(wfull + (kb * 32u + row) * N + n0 + chunk * 8u) =
          *reinterpret_cast<threadgroup const uint4*>(tile + row * EXPAND_N + chunk * 8u);
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }}
}}

kernel void probe_touch_f16(device const uint* bits [[buffer(0)]],
                            device half* wfull [[buffer(1)]],
                            uint lid [[thread_index_in_threadgroup]]) {{
  if (lid == 0u) {{
    wfull[0] = half(-1.0f);
  }}
}}

kernel void probe_sink_f16(device const half* x [[buffer(0)]],
                           device const uint* bits [[buffer(1)]],
                           device half* y [[buffer(2)]],
                           device const half* wfull [[buffer(3)]],
                           uint lid [[thread_index_in_threadgroup]]) {{
  if (lid == 0u) {{
    y[0] = wfull[0];
  }}
}}
"""


def _expand_fp16_launch(m: int, n: int, k: int) -> str:
    kb = k // 32
    return (
        "/* taic-launch\nmath fast\nscratch wfull fp16 K*N zero\n"
        f"launch probe_expand_f16 grid={n // 128},{max(1, (kb + 15) // 16)},1 threads=128,1,1 scratch=wfull:write\n"
        "launch probe_sink_f16 grid=1,1,1 threads=32,1,1 scratch=wfull:read\n*/\n"
    )


def _expand_fp32_launch(total: int) -> str:
    return (
        "/* taic-launch\nmath fast\nscratch wfull fp32 K*N\n"
        f"launch probe_expand_f32 grid={(total // 4 + 255) // 256},1,1 threads=256,1,1 scratch=wfull:write\n"
        "launch probe_expand_f32_sink grid=1,1,1 threads=32,1,1 scratch=wfull:read\n*/\n"
    )


def _fill_fp16_launch() -> str:
    return (
        "/* taic-launch\nmath fast\nscratch wfull fp16 K*N zero\n"
        "launch probe_touch_f16 grid=1,1,1 threads=32,1,1 scratch=wfull:write\n"
        "launch probe_sink_f16 grid=1,1,1 threads=32,1,1 scratch=wfull:read\n*/\n"
    )


def _b_staging(n: int, k: int) -> dict:
    """Static audit of the cooperative global -> threadgroup B staging loop.

    The r1 probe wrote four passes of ``uint4`` per thread, i.e. 64 k rows into a
    ``BLOCK_K``-row threadgroup region: 2296 halves (4592 B) past the declared array, and a
    2x over-read of the expanded B matrix.  Both shared arms were inflated by that bug, so
    the pass count is now derived here, emitted into the kernel and recorded as evidence.
    """
    threads = 128
    block_k, block_n = 32, 64
    halves = block_k * block_n
    per_pass = threads * 8
    passes, remainder = divmod(halves, per_pass)
    if remainder or block_n % 8:
        raise AssertionError("the B staging loop cannot cover the tile exactly")
    rows_written = (threads * passes - 1) >> 3
    c_stride = block_n + 8
    a_tile_h = 64 * (block_k + 8)
    declared = a_tile_h + block_k * c_stride
    last = a_tile_h + rows_written * c_stride + ((threads * passes - 1) & 7) * 8 + 8
    overrun = max(0, last - declared)
    return {
        "threads": threads,
        "passes": passes,
        "rows_declared": block_k,
        "rows_written": rows_written + 1,
        "halves_declared": halves,
        "halves_written": threads * passes * 8,
        "over_read_ratio": (threads * passes * 8) / halves,
        "threadgroup_overrun_halves": overrun,
        "threadgroup_overrun_bytes": 2 * overrun,
        "expanded_b_read_bytes_per_k_iter": 2 * threads * passes * 8,
        "expanded_b_needed_bytes_per_k_iter": 2 * halves,
    }


_GEMM_COMMON = """
constant constexpr uint BLOCK_M = 64;
constant constexpr uint BLOCK_N = 64;
constant constexpr uint BLOCK_K = 32;
constant constexpr uint B_PASSES = {b_passes}u;
constant constexpr uint A_STRIDE = BLOCK_K + 8;          // 40 halves
constant constexpr uint C_STRIDE = BLOCK_N + 8;          // 72 halves
constant constexpr uint A_TILE_H = BLOCK_M * A_STRIDE;   // 2560 halves
constant constexpr uint SMEM_H   = A_TILE_H + BLOCK_K * C_STRIDE;  // A, then B, then C staging
constant constexpr uint SMEM_V   = SMEM_H / 8;
constant constexpr uint K_ITERS  = K / BLOCK_K;

inline void load_a(device const half* A, uint m0, uint k0, uint lid,
                   thread uint4& x0, thread uint4& x1) {
    uint s = lid;
    x0 = *reinterpret_cast<device const uint4*>(A + (m0 + (s >> 2)) * K + k0 + ((s & 3u) << 3));
    s += 128u;
    x1 = *reinterpret_cast<device const uint4*>(A + (m0 + (s >> 2)) * K + k0 + ((s & 3u) << 3));
}

inline void store_a(threadgroup half* As, uint lid, uint4 x0, uint4 x1) {
    uint s = lid;
    *reinterpret_cast<threadgroup uint4*>(As + (s >> 2) * A_STRIDE + ((s & 3u) << 3)) = x0;
    s += 128u;
    *reinterpret_cast<threadgroup uint4*>(As + (s >> 2) * A_STRIDE + ((s & 3u) << 3)) = x1;
}
"""


_GEMM_EPILOGUE = """
    // Single rounding point; the fp32 accumulator is scaled by the OpSpec attr and stored once.
    threadgroup half* Cs = smem;
    const ushort qid = (ushort)(lane >> 2);
    const ushort rr = (ushort)((qid & 4) + ((lane >> 1) & 3u));
    const ushort cc = (ushort)((qid & 2) * 2 + (lane & 1u) * 2u);
#pragma unroll
    for (uint i = 0; i < 4; ++i) {
#pragma unroll
        for (uint j = 0; j < 4; ++j) {
            thread auto& te = acc[i][j].thread_elements();
            uint mine = as_type<uint>(half2((half)(te[0] * SCALE), (half)(te[1] * SCALE)));
            uint peer = simd_shuffle_xor(mine, (ushort)1u);
            if ((lane & 1u) == 0u) {
                half2 lo = as_type<half2>(mine);
                half2 hi = as_type<half2>(peer);
                *reinterpret_cast<threadgroup half4*>(
                    Cs + (r0 + i * 8u + rr) * C_STRIDE + c0 + j * 8u + cc) =
                    half4(lo.x, lo.y, hi.x, hi.y);
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
#pragma unroll
    for (uint s = lid; s < BLOCK_M * (BLOCK_N / 8u); s += 128u) {
        uint r = s >> 3;
        uint c = (s & 7u) << 3;
        *reinterpret_cast<device uint4*>(y + (m0 + r) * N + n0 + c) =
            *reinterpret_cast<threadgroup const uint4*>(Cs + r * C_STRIDE + c);
    }
"""


def _gemm_common(n: int, k: int) -> str:
    # Not str.format: the Metal body is full of braces.
    return _GEMM_COMMON.replace("{b_passes}", str(_b_staging(n, k)["passes"]))


def _gemm_fp(load_and_stage: str, dtype: str, n: int, k: int) -> str:
    """One fp16/fp32/packed-fp32 B GEMM kernel sharing the A path, MMA loop and epilogue."""
    pointer = {
        "fp16": "device const half* wfull",
        "fp32": "device const float* wfull",
        "p16": "device const float* wfull",
    }[dtype]
    return (
        """
kernel void probe_gemm_%s(device const half* x [[buffer(0)]],
                          %s [[buffer(1)]],
                          device half* y [[buffer(2)]],
                          uint2 tgp [[threadgroup_position_in_grid]],
                          uint lid [[thread_index_in_threadgroup]],
                          uint sgid [[simdgroup_index_in_threadgroup]],
                          uint lane [[thread_index_in_simdgroup]])
{
    threadgroup uint4 smem_raw[SMEM_V];
    threadgroup half* smem = reinterpret_cast<threadgroup half*>(smem_raw);
    threadgroup half* As = smem;
    threadgroup half* Bs = smem + A_TILE_H;

    const uint m0 = tgp.y * BLOCK_M;
    const uint n0 = tgp.x * BLOCK_N;

    uint4 pa0, pa1;
    load_a(x, m0, 0u, lid, pa0, pa1);

    simdgroup_float8x8 acc[4][4];
#pragma unroll
    for (uint i = 0; i < 4; ++i)
#pragma unroll
        for (uint j = 0; j < 4; ++j) acc[i][j] = simdgroup_float8x8(0);

    const uint r0 = (sgid >> 1) * 32u;
    const uint c0 = (sgid & 1u) * 32u;
    threadgroup half* a_frag = As + r0 * A_STRIDE;
    threadgroup half* b_frag = Bs + c0;

    for (uint ki = 0; ki < K_ITERS; ++ki) {
        store_a(As, lid, pa0, pa1);
%s
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (ki + 1u < K_ITERS) {
            load_a(x, m0, (ki + 1u) * BLOCK_K, lid, pa0, pa1);
        }

#pragma unroll
        for (uint kk = 0; kk < 4; ++kk) {
            simdgroup_half8x8 af[4], bf[4];
#pragma unroll
            for (uint i = 0; i < 4; ++i)
                simdgroup_load(af[i], a_frag + (i * 8u * A_STRIDE) + (kk * 8u), A_STRIDE);
#pragma unroll
            for (uint j = 0; j < 4; ++j)
                simdgroup_load(bf[j], b_frag + (j * 8u) + (kk * 8u * C_STRIDE), C_STRIDE);
#pragma unroll
            for (uint i = 0; i < 4; ++i)
#pragma unroll
                for (uint j = 0; j < 4; ++j)
                    simdgroup_multiply_accumulate(acc[i][j], af[i], bf[j], acc[i][j]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
%s
}
"""
        % (dtype, pointer, load_and_stage, _GEMM_EPILOGUE)
    )


def _fp16_gemm_body(n: int, k: int) -> str:
    stage = """
        // fp16 B tile: B_PASSES x uint4 per thread (32 x 64 halves) staged row-major [k][n].
        // B_PASSES is derived in Python so the loop cannot overrun the BLOCK_K-row region.
        for (uint pass = 0; pass < B_PASSES; ++pass) {
            const uint s = lid + pass * 128u;
            *reinterpret_cast<threadgroup uint4*>(Bs + (s >> 3) * C_STRIDE + ((s & 7u) << 3)) =
                *reinterpret_cast<device const uint4*>(
                    wfull + (ki * BLOCK_K + (s >> 3)) * N + n0 + ((s & 7u) << 3));
        }"""
    return _gemm_common(n, k) + _gemm_fp(stage, "fp16", n, k)


def _fp32_gemm_body(n: int, k: int) -> str:
    stage = """
        // fp32 B tile: B_PASSES x (2 x float4) per thread, converted to half on the way in.
        for (uint pass = 0; pass < B_PASSES; ++pass) {
            const uint s = lid + pass * 128u;
            const uint krow = s >> 3;
            const uint ncol = (s & 7u) << 3;
            const float4 f0 = *reinterpret_cast<device const float4*>(
                wfull + (ki * BLOCK_K + krow) * N + n0 + ncol);
            const float4 f1 = *reinterpret_cast<device const float4*>(
                wfull + (ki * BLOCK_K + krow) * N + n0 + ncol + 4u);
            *reinterpret_cast<threadgroup half4*>(Bs + krow * C_STRIDE + ncol) =
                half4(half(f0.x), half(f0.y), half(f0.z), half(f0.w));
            *reinterpret_cast<threadgroup half4*>(Bs + krow * C_STRIDE + ncol + 4u) =
                half4(half(f1.x), half(f1.y), half(f1.z), half(f1.w));
        }"""
    return _gemm_common(n, k) + _gemm_fp(stage, "fp32", n, k)


def _shared_fp32_arm(total: int, m: int, n: int, k: int) -> str:
    launch = (
        "/* taic-launch\nmath fast\nscratch wfull fp32 K*N\n"
        f"launch probe_expand_f32 grid={(total // 4 + 255) // 256},1,1 threads=256,1,1 scratch=wfull:write\n"
        f"launch probe_gemm_fp32 grid={n // 64},{m // 64},1 threads=128,1,1 scratch=wfull:read\n*/\n"
    )
    return _expand_fp32_producer(total, n) + _fp32_gemm_body(n, k) + launch


def _shared_fp16_arm(m: int, n: int, k: int) -> str:
    launch = (
        "/* taic-launch\nmath fast\nscratch wfull fp16 K*N zero\n"
        f"launch probe_expand_f16 grid={n // 128},{max(1, (k // 32 + 15) // 16)},1 threads=128,1,1 scratch=wfull:write\n"
        f"launch probe_gemm_fp16 grid={n // 64},{m // 64},1 threads=128,1,1 scratch=wfull:read\n*/\n"
    )
    return _expand_fp16_producer(m, n, k) + _fp16_gemm_body(n, k) + launch


def _expand_packed16_producer(n: int, k: int) -> str:
    """Legal poison producer that packs two decoded halves into one fp32 scratch texel.

    This is the cheapest SHARED producer the lifecycle allows: an ``uninitialized_poison``
    scratch needs a write-reflected producer, only an R32Float texture reflects ``write``, so
    the buffer must be fp32.  Storing the exact fp16 bit patterns of two decoded weights per
    texel keeps the 2 B/weight footprint of the fp16 expansion (4 B per two weights) while
    removing the ``device_zero`` fill entirely.  The scratch read-back in ``_coverage``
    proves the texture path carried every bit, including the bit patterns that are
    denormal/NaN as floats.
    """
    total = (k * n) // 2
    width, _height, _row = scratch_texture_layout(total, guard_bytes=GUARD_BYTES)
    return f"""
constant constexpr uint P16_TEX_WIDTH = {width}u;
constant constexpr uint EXPAND_N = 128u;
constant constexpr uint EXPAND_KB = 16u;

kernel void probe_expand_p16(device const uint* bits [[buffer(0)]],
                             texture2d<float, access::write> wfull [[texture(0)]],
                             uint2 tg [[threadgroup_position_in_grid]],
                             uint lid [[thread_index_in_threadgroup]]) {{
  const uint n0 = tg.x * EXPAND_N;
  const uint kb0 = tg.y * EXPAND_KB;
  const uint kb_end = min(kb0 + EXPAND_KB, KB);
  threadgroup half tile[32u * EXPAND_N];
  for (uint kb = kb0; kb < kb_end; ++kb) {{
    const uint word = bits[kb * N + n0 + lid];
#pragma unroll
    for (uint j = 0; j < 32u; ++j) {{
      tile[j * EXPAND_N + lid] = half((((word >> j) & 1u) != 0u) ? 1.0f : -1.0f);
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // 16 decoded halves = 8 texels; the half2 pairs are stored as their exact 32-bit pattern.
    for (uint u = lid; u < 32u * (EXPAND_N / 16u); u += 128u) {{
      const uint row = u / (EXPAND_N / 16u);
      const uint chunk = u - row * (EXPAND_N / 16u);
      const uint4 v0 = *reinterpret_cast<threadgroup const uint4*>(tile + row * EXPAND_N + chunk * 16u);
      const uint4 v1 = *reinterpret_cast<threadgroup const uint4*>(tile + row * EXPAND_N + chunk * 16u + 8u);
      thread uint values[8] = {{v0.x, v0.y, v0.z, v0.w, v1.x, v1.y, v1.z, v1.w}};
      // Packed element index: row k, half-pair (n0/2 + chunk*8).
      const uint base = (kb * 32u + row) * (N / 2u) + (n0 / 2u) + chunk * 8u;
#pragma unroll
      for (uint t = 0; t < 8u; ++t) {{
        const uint e = base + t;
        wfull.write(float4(as_type<float>(values[t])), uint2(e % P16_TEX_WIDTH, e / P16_TEX_WIDTH));
      }}
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }}
}}
"""


def _packed16_gemm_body(n: int, k: int) -> str:
    stage = """
        // The packed fp32 scratch already holds the exact fp16 bit patterns two weights at a
        // time, so staging B is a 16-byte copy: no conversion and no bit twiddling here.
        for (uint pass = 0; pass < B_PASSES; ++pass) {
            const uint s = lid + pass * 128u;
            const uint krow = s >> 3;
            *reinterpret_cast<threadgroup uint4*>(Bs + krow * C_STRIDE + ((s & 7u) << 3)) =
                *reinterpret_cast<device const uint4*>(
                    wfull + (ki * BLOCK_K + krow) * (N / 2u) + (n0 / 2u) + ((s & 7u) << 2));
        }"""
    return _gemm_common(n, k) + _gemm_fp(stage, "p16", n, k)


def _shared_packed16_arm(m: int, n: int, k: int) -> str:
    launch = (
        "/* taic-launch\nmath fast\nscratch wfull fp32 K*KB*16\n"
        f"launch probe_expand_p16 grid={n // 128},{max(1, (k // 32 + 15) // 16)},1 threads=128,1,1 scratch=wfull:write\n"
        f"launch probe_gemm_p16 grid={n // 64},{m // 64},1 threads=128,1,1 scratch=wfull:read\n*/\n"
    )
    return _expand_packed16_producer(n, k) + _packed16_gemm_body(n, k) + launch


def candidate_source(size: str) -> str:
    return (CANDIDATE_TREE / "variants" / size / "kernel.metal").read_text(encoding="utf-8")


def arm_source(size: str, arm: str) -> str:
    m, n, k = SHAPES[size]
    # The expanded weight matrix has K*N elements; y has M*N.  These are different
    # (paper: 16.7M vs 2.1M), so the producer must cover K*N, not M*N.
    weight_total = k * n
    header = _header(_constants(m, n, k))
    if arm == "floor":
        return header + _floor_arm()
    if arm == "expand_fp32":
        return header + _expand_fp32_producer(weight_total, n) + _expand_fp32_launch(weight_total)
    if arm == "packed":
        return header + candidate_source(size)
    if arm == "shared_fp32":
        return header + _shared_fp32_arm(weight_total, m, n, k)
    if arm == "fill_fp16":
        return header + _expand_fp16_producer(m, n, k) + _fill_fp16_launch()
    if arm == "expand_fp16":
        return header + _expand_fp16_producer(m, n, k) + _expand_fp16_launch(m, n, k)
    if arm == "shared_fp16":
        return header + _shared_fp16_arm(m, n, k)
    if arm == "shared_packed16":
        return header + _shared_packed16_arm(m, n, k)
    raise ValueError(f"unknown probe arm {arm!r}")


@dataclass
class _Prepared:
    spec: object
    resolved: object
    kernels: list
    grids: list
    threads: list
    sets: list
    args: list
    fills: list


def _prepare(ctx, task_contract, abi, limits, symbols: dict[str, int], source: str, *, nsets: int) -> _Prepared:
    spec = parse_launch_spec(source, symbols)
    resolved = resolve_geometry(
        msl_single_geometry_resolver(task_contract)(candidate_from_single_source(source)),
        instance=task_contract.instance,
        contract=abi,
        limits=limits,
    )
    kernels = list(ctx.compile_all(source, [stage.kernel for stage in spec.stages], math=spec.math))
    reflection = {kernel.name: tuple(reflection_for_kernel(kernel)) for kernel in kernels}
    bindings = bind_pipeline_arguments(resolved, contract=abi, reflection=reflection, limits=limits)
    sets = [V2ArgSet(ctx, abi, resolved) for _ in range(nsets)]
    return _Prepared(
        spec=spec,
        resolved=resolved,
        kernels=kernels,
        grids=[stage.grid for stage in spec.stages],
        threads=[stage.threads for stage in spec.stages],
        sets=sets,
        args=[[one.stage_args(binding) for binding in bindings] for one in sets],
        fills=[one.workspace_fills_declared_zero() for one in sets],
    )


def _bits(m: int, n: int, k: int) -> np.ndarray:
    kb = k // 32
    rng = np.random.default_rng(BITS_SEED)
    words = rng.integers(0, 1 << 32, size=(kb, n), dtype=np.uint64).astype(np.uint32)
    edge = (
        np.uint32(0x00000000),
        np.uint32(0xFFFFFFFF),
        np.uint32(0xAAAAAAAA),
        np.uint32(0x55555555),
        np.uint32(0x80000001),
        np.uint32(0x7FFFFFFE),
    )
    for row, word in enumerate(edge):
        if row < kb:
            words[row, :] = word
    return words


def expanded_fp16(words: np.ndarray, k: int, n: int) -> np.ndarray:
    """The exact CPU decode: bit j of bits[k//32, n] is W[k, n] = +1, else -1."""
    kb = k // 32
    flat_bits = np.zeros((kb, 32, n), dtype=np.uint32)
    for j in range(32):
        flat_bits[:, j, :] = (words >> np.uint32(j)) & np.uint32(1)
    flat = flat_bits.reshape(kb * 32, n)
    half_bits = np.where(flat == 1, np.uint16(0x3C00), np.uint16(0xBC00)).astype(np.uint16)
    return half_bits.view(np.float16).reshape(k, n)


def _fill_inputs(prepared: _Prepared, words: np.ndarray, m: int, n: int, k: int) -> None:
    x = np.ones((m, k), dtype=np.float16)
    for one in prepared.sets:
        one.buffers["x"].write(x.tobytes())
        one.buffers["bits"].write(words.tobytes())


def _read_scratch_bytes(prepared: _Prepared, index: int) -> bytes:
    return prepared.sets[index].workspace_bufs["wfull"].read()


def _read_y(prepared: _Prepared, index: int, count: int) -> np.ndarray:
    return np.frombuffer(prepared.sets[index].buffers["y"].read(), dtype=np.float16)[:count]


def _measure_interleaved(ctx, arms: dict[str, _Prepared], *, rounds: int = PAIR_ROUNDS) -> dict:
    """Paired A/B of the REG arm against each legal SHARED arm inside one run.

    Sequential arms are not enough to decide a few-percent question: the REG arm moved 6.7%
    between the r2 and r3 runs of this probe while the packed SHARED arm moved 0.2%.  Here
    every round runs each arm back to back with its own rotation set, so clock and thermal
    drift hit all arms in the same round; the reported ratio is the median of the per-round
    paired ratios, not a ratio of medians.
    """
    per_arm: dict[str, list[float]] = {name: [] for name in arms}
    gpus: dict[str, list[float]] = {name: [] for name in arms}
    iters: dict[str, int] = {}
    for name, prepared in arms.items():
        probe = _run(ctx, prepared, 20)
        per_iter = max(float(probe.wall_s) / 20.0, 1e-9)
        iters[name] = int(min(MAX_ITERS, max(20, SAMPLE_TARGET_SECONDS / per_iter)))
        warm = int(min(MAX_ITERS, max(20, WARM_TARGET_SECONDS / per_iter)))
        _run(ctx, prepared, warm)
    for _ in range(rounds):
        for name, prepared in arms.items():
            timing = _run(ctx, prepared, iters[name])
            per_arm[name].append(float(timing.wall_s) / iters[name] * 1e6)
            gpus[name].append(float(timing.gpu_s) / iters[name] * 1e6)
    reg = per_arm["packed"]
    document = {
        "schema": "paired interleave inside one run; the REG arm is the candidate source",
        "rounds": rounds,
        "iters_per_sample": iters,
        "arms": {
            name: {
                "wall_us_median": statistics.median(values),
                "wall_samples_us": values,
                "gpu_us_median": statistics.median(gpus[name]),
            }
            for name, values in per_arm.items()
        },
        "paired_ratio_over_reg": {},
    }
    for name, values in per_arm.items():
        if name == "packed":
            continue
        ratios = [value / base for value, base in zip(values, reg)]
        document["paired_ratio_over_reg"][name] = {
            "median": statistics.median(ratios),
            "min": min(ratios),
            "max": max(ratios),
            "rounds_over_one": sum(1 for ratio in ratios if ratio > 1.0),
            "per_round": ratios,
        }
    return document


def _run(ctx, prepared: _Prepared, iters: int, *, use_fills: bool = True):
    return ctx.run_pipeline(
        prepared.kernels,
        prepared.grids,
        prepared.threads,
        [0 for _ in prepared.spec.stages],
        prepared.args,
        prepared.fills if use_fills else [[] for _ in prepared.sets],
        iters=iters,
        dispatch="concurrent",
    )


def _measure(ctx, prepared: _Prepared, *, use_fills: bool = True) -> dict:
    stages = len(prepared.spec.stages)
    # Short kernels measured without other GPU work report a down-clocked time: the harness
    # interleaves registered baselines, so its tiny-shape numbers sit at boost clock.  Ramp
    # the clock with a bounded warm-up and give every sample the same ~60 ms window.
    probe = _run(ctx, prepared, 20, use_fills=use_fills)
    per_iter = max(float(probe.wall_s) / 20.0, 1e-9)
    warm_iters = int(min(MAX_ITERS, max(20, WARM_TARGET_SECONDS / per_iter)))
    sample_iters = int(min(MAX_ITERS, max(20, SAMPLE_TARGET_SECONDS / per_iter)))
    _run(ctx, prepared, warm_iters, use_fills=use_fills)
    walls, gpus = [], []
    for _ in range(SAMPLE_COUNT):
        timing = _run(ctx, prepared, sample_iters, use_fills=use_fills)
        walls.append(float(timing.wall_s) / sample_iters * 1e6)
        gpus.append(float(timing.gpu_s) / sample_iters * 1e6)
    fills = [int(fill[2]) for fill in prepared.fills[0]] if use_fills else []
    return {
        "wall_us_median": statistics.median(walls),
        "gpu_us_median": statistics.median(gpus),
        "wall_samples_us": walls,
        "gpu_samples_us": gpus,
        "iters_per_sample": sample_iters,
        "warmup_iters": warm_iters,
        "rotation_sets": len(prepared.sets),
        "stages": stages,
        "stage_kernels": [stage.kernel for stage in prepared.spec.stages],
        "grids": [list(stage.grid) for stage in prepared.spec.stages],
        "threads": [list(stage.threads) for stage in prepared.spec.stages],
        "scratch_name": prepared.spec.scratches[0].name if prepared.spec.scratches else None,
        "scratch_bytes": sum(int(item.nbytes) for item in getattr(prepared.resolved, "scratch", ())),
        "fill_bytes_per_iteration": sum(fills),
        "fill_count": len(fills),
        "dispatch": "concurrent, one command buffer, native batch when eligible",
    }


def _measure_subset(ctx, prepared: _Prepared, stage_index: int, *, use_fills: bool) -> dict:
    """Measure one stage of a prepared pipeline, for decomposition only.

    ``use_fills`` is False on purpose for the producer/consumer split: this is not a legal
    route, it only separates the terms of the end-to-end arms.  The scratch is left with the
    previous run's contents (a "prepare cache" the contract forbids in a candidate).
    """
    kernels = [prepared.kernels[stage_index]]
    grids = [prepared.grids[stage_index]]
    threads = [prepared.threads[stage_index]]
    args = [[one[stage_index] for one in prepared.args]]
    fills = [prepared.fills[0]] if use_fills else [[]]
    probe = ctx.run_pipeline(kernels, grids, threads, [0], args, fills, iters=20, dispatch="concurrent")
    per_iter = max(float(probe.wall_s) / 20.0, 1e-9)
    warm_iters = int(min(MAX_ITERS, max(20, WARM_TARGET_SECONDS / per_iter)))
    sample_iters = int(min(MAX_ITERS, max(20, SAMPLE_TARGET_SECONDS / per_iter)))
    ctx.run_pipeline(kernels, grids, threads, [0], args, fills, iters=warm_iters, dispatch="concurrent")
    walls, gpus = [], []
    for _ in range(SAMPLE_COUNT):
        timing = ctx.run_pipeline(
            kernels, grids, threads, [0], args, fills, iters=sample_iters, dispatch="concurrent"
        )
        walls.append(float(timing.wall_s) / sample_iters * 1e6)
        gpus.append(float(timing.gpu_s) / sample_iters * 1e6)
    return {
        "stage_index": int(stage_index),
        "stage_kernel": prepared.spec.stages[stage_index].kernel,
        "wall_us_median": statistics.median(walls),
        "gpu_us_median": statistics.median(gpus),
        "wall_samples_us": walls,
        "iters_per_sample": sample_iters,
        "warmup_iters": warm_iters,
        "fill_bytes_per_iteration": sum(int(fill[2]) for fill in fills[0]),
        "not_a_route": True,
    }


def _coverage(prepared: _Prepared, expected: np.ndarray, dtype: str) -> dict:
    raw = _read_scratch_bytes(prepared, 0)
    if dtype == "fp32":
        got = np.frombuffer(raw, dtype=np.float32)[: expected.size]
        want = expected.astype(np.float32).reshape(-1)
    else:
        got = np.frombuffer(raw, dtype=np.uint16)[: expected.size]
        want = expected.view(np.uint16).reshape(-1)
    mismatch = int(np.count_nonzero(got != want)) if got.shape == want.shape else -1
    document = {
        "expected_elements": int(want.size),
        "read_back_elements": int(got.size),
        "mismatched_elements": mismatch,
        "ok": mismatch == 0,
    }
    if mismatch > 0:
        first = int(np.nonzero(got != want)[0][0])
        document["first_mismatch_index"] = first
        document["first_mismatch_got"] = got[first].item()
        document["first_mismatch_want"] = want[first].item()
    return document


def _lanes_per_word(source: str) -> int:
    """How many lanes of a threadgroup share one packed word in the REG decode.

    The small candidate decodes with ``const uint nn = lid >> 2;`` (one word per four lanes) and
    the paper candidate with ``const uint n = lid >> 1;`` (one word per lane pair).  Derived from
    the candidate bytes, not from the shape.
    """
    assignments = re.findall(r"const uint (\w+) = lid >> (\d+);", source)
    # The decode index is the only `lid >> n` variable that also indexes the packed words; the
    # other one is the output-row index (`orow` in the small candidate).
    hits = [
        (name, int(shift))
        for name, shift in assignments
        if re.search(r"bits\[[^\]]*\b" + re.escape(name) + r"\b", source)
    ]
    if len(hits) != 1:
        raise AssertionError(f"the REG decode no longer has exactly one packed-word lane mapping: {hits}")
    name, shift = hits[0]
    if shift not in (1, 2):
        raise AssertionError(f"unexpected lane-per-word shift {shift} for {name!r}")
    return 1 << shift


def _traffic(m: int, n: int, k: int, *, reg_grid: tuple[int, int], lanes_per_word: int) -> dict:
    """Byte accounting for the two routes, each at *its own* tiling.

    The REG candidate and the SHARED probe do not share a tiling: at small the candidate owns
    32x32 tiles (grid 8x4 -> eight column blocks, four row blocks, 32 threadgroups) while the
    probe's SHARED consumer owns 64x64 tiles (four column blocks, two row blocks, eight
    threadgroups).  Using one ``m // 64`` row-block count for both under-counted the small REG
    decode repeats, packed reads and x reads; the two geometries are separate from here on.
    """
    packed = 4 * (k // 32) * n
    x_bytes = 2 * m * k
    y_bytes = 2 * m * n
    expanded16 = 2 * k * n
    expanded32 = 4 * k * n

    reg_col_blocks, reg_row_blocks = int(reg_grid[0]), int(reg_grid[1])
    if reg_col_blocks * reg_row_blocks <= 0:
        raise AssertionError(f"bad REG grid {reg_grid}")
    reg = {
        "packed_read_bytes": reg_row_blocks * packed,
        "packed_read_bytes_issued": lanes_per_word * reg_row_blocks * packed,
        "x_read_bytes": reg_col_blocks * x_bytes,
        "y_write_bytes": y_bytes,
        "fill_bytes": 0,
        "tile_m": m // reg_row_blocks,
        "tile_n": n // reg_col_blocks,
        "row_blocks": reg_row_blocks,
        "col_blocks": reg_col_blocks,
        "output_threadgroups": reg_row_blocks * reg_col_blocks,
        "lanes_per_word": lanes_per_word,
    }
    reg["total_bytes"] = reg["packed_read_bytes"] + reg["x_read_bytes"] + reg["y_write_bytes"]

    shared_m, shared_n = 64, 64
    shared_row_blocks = m // shared_m
    shared_col_blocks = n // shared_n
    if shared_row_blocks * shared_m != m or shared_col_blocks * shared_n != n:
        raise AssertionError(f"the SHARED tiling {shared_m}x{shared_n} does not cover {m}x{n}")
    shared_tiling = {
        "tile_m": shared_m,
        "tile_n": shared_n,
        "row_blocks": shared_row_blocks,
        "col_blocks": shared_col_blocks,
        "output_threadgroups": shared_row_blocks * shared_col_blocks,
    }
    shared32 = {
        "packed_word_read_bytes_issued": expanded32,   # 4 elements per thread, 16 B read
        "expanded_write_bytes": expanded32,
        "consumer_b_read_bytes": shared_row_blocks * expanded32,
        "x_read_bytes": shared_col_blocks * x_bytes,
        "y_write_bytes": y_bytes,
        "fill_bytes": 0,
        **shared_tiling,
    }
    shared32["total_bytes"] = sum(
        shared32[key] for key in ("packed_word_read_bytes_issued", "expanded_write_bytes",
                                 "consumer_b_read_bytes", "x_read_bytes", "y_write_bytes", "fill_bytes")
    )
    shared16 = {
        "packed_word_read_bytes": packed,
        "expanded_write_bytes": expanded16,
        "consumer_b_read_bytes": shared_row_blocks * expanded16,
        "x_read_bytes": shared_col_blocks * x_bytes,
        "y_write_bytes": y_bytes,
        "fill_bytes": expanded16,
        **shared_tiling,
    }
    shared16["total_bytes"] = sum(
        shared16[key] for key in ("packed_word_read_bytes", "expanded_write_bytes",
                                  "consumer_b_read_bytes", "x_read_bytes", "y_write_bytes", "fill_bytes")
    )
    return {
        "reg_tiling": {key: reg[key] for key in ("tile_m", "tile_n", "row_blocks", "col_blocks",
                                                "output_threadgroups", "lanes_per_word")},
        "shared_tiling": shared_tiling,
        "decode": {
            "elements": int(k * n),
            "reg_repeats": reg_row_blocks,
            "reg_decoded_elements": reg_row_blocks * k * n,
            "shared_repeats": 1,
            "shared_decoded_elements": k * n,
            "reg_repeats_per_weight": f"once per REG row block ({reg_row_blocks}x at this tiling)",
        },
        "reg": reg,
        "shared_fp32": shared32,
        "shared_fp16": shared16,
        "b_staging": _b_staging(n, k),
        "note": (
            "packed word reads are L2-resident; the REG and SHARED routes have different tilings and "
            "therefore different row/column block counts, so each route carries its own geometry. "
            "packed_read_bytes counts distinct words per row block; packed_read_bytes_issued counts "
            "what the hardware is asked for (one word per lane pair at paper, one per lane quad at "
            "small). consumer_b_read_bytes counts the expanded matrix re-read by every output row "
            "block of the 64x64 SHARED consumer."
        ),
    }


def run_probe(output: Path) -> dict:
    target = load_target(TARGET_ID)
    ctx = get_context()
    load_backend("metal").prepare_context(ctx)
    document = {
        "schema_id": SCHEMA_ID,
        "not_a_candidate": True,
        "publication_allowed": False,
        "target_id": TARGET_ID,
        "device_name": target.calibration_device_name,
        "correctness": (
            "probe kernels are never evaluated against the task oracle; the two expand arms carry a "
            "full element-wise coverage check against the CPU decode"
        ),
        "cases_not_run": list(CASES_NOT_RUN),
        "bits_seed": BITS_SEED,
        "supersedes": {
            "schema_id": SUPERSEDED_SCHEMA_ID,
            "r1_defect": (
                "every GEMM arm staged B with four uint4 passes per thread, so it wrote 64 k rows "
                "into a BLOCK_K=32-row threadgroup region (2296 halves = 4592 B past the declared "
                "array) and read the expanded matrix twice per k step. r1 is kept verbatim as "
                "unpack-probe-r1-superseded.json and is not used for the decision."
            ),
            "r2_fix": "the pass count is derived from BLOCK_K*BLOCK_N and recorded as b_staging_audit",
        },
        "coverage_method": (
            "Each expand producer writes the exact +-1 decode of every packed word into scratch[K, N]. "
            "After a timed run the probe reads the whole scratch back and requires element-wise equality "
            "with the CPU decode of the same words (uint16 half bits for fp16/packed16, float32 for fp32). "
            "The packed and all three shared GEMM arms receive the same bits and x=1 and must produce "
            "identical y."
        ),
        "sizes": {},
    }
    with gpu_lock("bitdelta-unpack-probe"):
        for size in SIZES:
            m, n, k = SHAPES[size]
            kb = k // 32
            total = m * n
            task_contract = load_task_contract(REPO_ROOT / "tasks" / "bitdelta_gemm", dtype="fp16", size=size)
            symbols = {str(name): int(value) for name, value in task_contract.instance.symbols.items()}
            abi = derive_tensor_memory_contract(task_contract.instance, target=target)
            limits = abi_limits_from_target(target)
            words = _bits(m, n, k)
            expected_half = expanded_fp16(words, k, n)
            arms: dict[str, dict] = {}
            prepared_map: dict[str, _Prepared] = {}
            for arm in ARMS:
                prepared = _prepare(ctx, task_contract, abi, limits, symbols, arm_source(size, arm), nsets=ROTATION_SETS)
                _fill_inputs(prepared, words, m, n, k)
                prepared_map[arm] = prepared
                arms[arm] = _measure(ctx, prepared)
                if arm == "expand_fp32":
                    arms[arm]["coverage"] = _coverage(prepared, expected_half, "fp32")
                elif arm in ("expand_fp16", "shared_packed16"):
                    # The packed fp32 scratch holds the exact half bit patterns, so the same
                    # uint16 comparison covers it element by element.
                    arms[arm]["coverage"] = _coverage(prepared, expected_half, "fp16")
                print(size, arm, arms[arm]["wall_us_median"], arms[arm]["gpu_us_median"], flush=True)
            # Measured no-fill form: the same two-stage pipeline with the declared fill removed
            # from the timed window.  Not a legal route (a device_zero scratch must be zeroed
            # every iteration, and the contents here are left from the previous run), but it is
            # the only way to measure the route's cost without the fill on the same kernels,
            # same tiling and same rotation instead of subtracting one arm from another.
            no_fill = {}
            for arm in ("shared_fp16", "shared_packed16"):
                measured = _measure(ctx, prepared_map[arm], use_fills=False)
                measured["declared_fill_bytes_per_iteration"] = arms[arm]["fill_bytes_per_iteration"]
                measured["policy_fill_removed"] = True
                measured["not_a_route"] = True
                no_fill[arm] = measured
                print(size, arm, "NO_FILL", measured["wall_us_median"], flush=True)
            # Decompose the legal SHARED routes: producer and consumer measured on their own.
            # This is not a route (the scratch is left from the previous run); it only splits
            # the end-to-end numbers into producer, fill and consumer terms.
            decomposition = {}
            for arm, label in (("shared_fp32", "fp32"), ("shared_fp16", "fp16_zero")):
                prepared = prepared_map[arm]
                decomposition[label] = {
                    "end_to_end_us": arms[arm]["wall_us_median"],
                    "producer_only": _measure_subset(ctx, prepared, 0, use_fills=False),
                    "consumer_only": _measure_subset(ctx, prepared, 1, use_fills=False),
                    "fill_bytes_per_iteration": arms[arm]["fill_bytes_per_iteration"],
                    "barrier_and_launch_floor_us": arms["floor"]["wall_us_median"],
                }
            # Paired, interleaved A/B of REG against every legal SHARED arm.  This is the
            # measurement the few-percent question actually needs.
            interleaved = _measure_interleaved(
                ctx,
                {
                    "packed": prepared_map["packed"],
                    "shared_packed16": prepared_map["shared_packed16"],
                    "shared_fp32": prepared_map["shared_fp32"],
                    "shared_fp16": prepared_map["shared_fp16"],
                },
            )
            print(
                size,
                "INTERLEAVED",
                {name: round(one["median"], 4) for name, one in interleaved["paired_ratio_over_reg"].items()},
                flush=True,
            )
            y_packed = _read_y(prepared_map["packed"], 0, total)
            y_shared_fp32 = _read_y(prepared_map["shared_fp32"], 0, total)
            y_shared_fp16 = _read_y(prepared_map["shared_fp16"], 0, total)
            y_shared_packed16 = _read_y(prepared_map["shared_packed16"], 0, total)
            agreement = {
                "packed_vs_shared_fp32_equal": bool(np.array_equal(y_packed, y_shared_fp32)),
                "packed_vs_shared_fp16_equal": bool(np.array_equal(y_packed, y_shared_fp16)),
                "packed_vs_shared_packed16_equal": bool(np.array_equal(y_packed, y_shared_packed16)),
                "elements": int(total),
                "packed_nan": int(np.count_nonzero(np.isnan(y_packed.astype(np.float32)))),
                "packed_nonzero": int(np.count_nonzero(y_packed)),
            }
            for key in (
                "packed_vs_shared_fp32_equal",
                "packed_vs_shared_fp16_equal",
                "packed_vs_shared_packed16_equal",
            ):
                if not agreement[key]:
                    print(size, "ARM DISAGREEMENT", key, flush=True)
            document["sizes"][size] = {
                "symbols": {"M": m, "N": n, "K": k, "KB": kb},
                "packed_bytes": int(words.nbytes),
                "expanded_fp16_bytes": 2 * k * n,
                "expanded_packed16_bytes": 2 * k * n,
                "expanded_fp32_bytes": 4 * k * n,
                "expand_ratio_fp16": (2 * k * n) / int(words.nbytes),
                "bits_words": int(words.size),
                "decode_elements": int(k * n),
                "arms": arms,
                "no_fill": no_fill,
                "interleaved": interleaved,
                "decomposition": decomposition,
                "output_agreement": agreement,
                "traffic": _traffic(
                    m,
                    n,
                    k,
                    reg_grid=(
                        int(prepared_map["packed"].grids[0][0]),
                        int(prepared_map["packed"].grids[0][1]),
                    ),
                    lanes_per_word=_lanes_per_word(candidate_source(size)),
                ),
                "b_staging_audit": {
                    "schema": "the GEMM arms stage exactly one BLOCK_K x BLOCK_N tile per k step",
                    "rows_written": _b_staging(n, k)["rows_written"],
                    "rows_declared": _b_staging(n, k)["rows_declared"],
                    "over_read_ratio": _b_staging(n, k)["over_read_ratio"],
                    "threadgroup_overrun_bytes": _b_staging(n, k)["threadgroup_overrun_bytes"],
                    "r1_defect_overrun_bytes": _R1_OVERRUN_BYTES,
                    "r1_defect_over_read_ratio": _R1_OVER_READ_RATIO,
                },
            }
    document["summary"] = {
        size: {
            arm: document["sizes"][size]["arms"][arm]["wall_us_median"] for arm in ARMS
        }
        | {
            "expand_fp32_coverage_ok": document["sizes"][size]["arms"]["expand_fp32"]["coverage"]["ok"],
            "expand_fp16_coverage_ok": document["sizes"][size]["arms"]["expand_fp16"]["coverage"]["ok"],
            "shared_packed16_coverage_ok": document["sizes"][size]["arms"]["shared_packed16"]["coverage"]["ok"],
            "shared_fp16_no_fill": document["sizes"][size]["no_fill"]["shared_fp16"]["wall_us_median"],
            "interleaved_over_reg": {
                name: one["median"]
                for name, one in document["sizes"][size]["interleaved"]["paired_ratio_over_reg"].items()
            },
            "packed_vs_shared_fp32_equal": document["sizes"][size]["output_agreement"]["packed_vs_shared_fp32_equal"],
            "packed_vs_shared_fp16_equal": document["sizes"][size]["output_agreement"]["packed_vs_shared_fp16_equal"],
            "packed_vs_shared_packed16_equal": document["sizes"][size]["output_agreement"]["packed_vs_shared_packed16_equal"],
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
