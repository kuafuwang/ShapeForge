# Apple GPU optimization principles

General principles for writing fast Metal kernels, written for an engineer who knows CUDA/Triton. Peaks for the chip you are on are in `device.md` in the working directory, and on the scorecard roofline. Where Apple differs from NVIDIA this says so.

## Machine model
Keywords: simd width threadgroup size registers occupancy unified memory slc cache bandwidth peak

- 32-wide SIMD groups (warps). A threadgroup (TG) has at most **1024 threads** and **32 KiB** of threadgroup (shared) memory.
  Without a `/* taic-launch */` block the contract fixes grid and TG size, so occupancy is controlled by register use and threadgroup memory. With that block, each stage chooses its own grid and TG size.
- Many TGs are resident per GPU core at once when registers and threadgroup memory allow it; latency is hidden by switching between resident SIMD groups.
  `maxTotalThreadsPerThreadgroup` is the pipeline's launch limit. A value below the launch's threads per TG makes the contract infeasible and the harness reports
  an ABI error. A lower value by itself is **not** proof of register pressure; it can also come from threadgroup memory or an explicit attribute.
- Unified memory with a large system-level cache: small working sets run out of cache, so the harness rotates over several buffer sets and
  reports bandwidth against the *DRAM* roofline.
- Peaks (read bandwidth, copy bandwidth, vendor GEMM by dtype, scalar FMA) are measured per chip. Read `device.md` and the scorecard. `simdgroup_matrix` is what gets close to the matrix peak; a scalar-FMA GEMM does not.
- fp16 and bf16 are both native storage types; do arithmetic in `float` unless you have a reason not to (half math has no throughput benefit for
  most kernels here and loses accuracy).

## The launch contract shapes the design
Keywords: contract grid program tile persistent loop fixed launch

Without a `/* taic-launch */` block, the Triton program maps to one threadgroup and the grid and TG size stay on the contract. With that block, you choose each stage's grid and TG size. You decide how the TG's threads cover the tile that its program
owns: how many elements per thread, which SIMD group handles which sub-tile, how the K loop is staged. Do not assume `gid` covers the data one
element per thread; loop (with a bounded trip count) over `BLOCK_SIZE`-sized tiles. Use the contract constants (`BLOCK_M`, `N`, ...) so the
compiler can fully unroll and fold index math.

## Memory-bound kernels (elementwise, normalization, softmax, reductions, RoPE, GEMV)
Keywords: memory bound bandwidth vectorized load coalescing streaming elementwise float4 half4 uint4

Goal: move each byte exactly once and keep enough bytes in flight.

1. **Vectorize**: each thread should issue 16-byte loads/stores (`uint4`, `half4 x2`, `float4`) at consecutive addresses; consecutive lanes should
   touch consecutive 16-byte chunks so a SIMD group reads 512 contiguous bytes per instruction.
2. **Unroll for memory-level parallelism**: issue several independent loads before using any of them (e.g. load 2-4 `uint4` per thread, then compute).
3. **Single pass over global memory**: a row that fits in registers/threadgroup memory should be read once. For softmax and RMSNorm, keep the row in
   registers (e.g. 8 elements per thread for a 2048-wide row on 256 threads), do the reduction, then write once. The online-softmax recurrence
   (running max and rescaled running sum) removes the second read when the row does not fit.
4. **Reductions**: `simd_sum`/`simd_max` inside a SIMD group (no barrier, no memory), then one threadgroup-memory exchange across SIMD groups. Avoid atomics.
5. Keep the arithmetic cheap (`exp2` with a folded constant, `fma`, `rsqrt`) but do not over-engineer: these kernels become bandwidth bound quickly.
   The naive scalar versions in this suite already reach 80-87% of the measured peak; the remaining headroom is in vector width and loads in flight.
6. Tail handling: the contract sizes are compile-time constants. When `N % (elements per thread * threads) == 0` the masks can be removed entirely.

```metal
// 16-byte vectorized copy-with-scale skeleton: 128 threads, each thread handles 8 consecutive halves per iteration.
kernel void k(device const half* x [[buffer(0)]], device half* y [[buffer(1)]],
              uint tg [[threadgroup_position_in_grid]], uint lid [[thread_index_in_threadgroup]]) {
    constexpr uint BLOCK = 1024;
    uint base = tg * BLOCK + lid * 8;                                  // 128 threads x 8 halves = 1024 per TG
    uint4 raw = *reinterpret_cast<device const uint4*>(x + base);
    float4 a = float4(as_type<half4>(raw.xy)) * 0.5f;
    float4 b = float4(as_type<half4>(raw.zw)) * 0.5f;
    uint4 out = uint4(as_type<uint2>(half4(a)), as_type<uint2>(half4(b)));
    *reinterpret_cast<device uint4*>(y + base) = out;
}
```

## Compute-bound kernels (GEMM, fused GEMM epilogues, convolution as implicit GEMM)
Keywords: gemm matmul tile simdgroup_matrix register blocking accumulator reuse double buffering bank conflict implicit gemm convolution epilogue

1. **Use `simdgroup_matrix`** (8x8 MMA, half/bfloat inputs, float accumulators). It is the only way to approach the matrix throughput; scalar FMA tops out
   near a third of it.
2. **Register-block**: one SIMD group should own a sub-tile of the TG's output tile held in several accumulator tiles (for a 64x64 TG tile and 4 SIMD
   groups: a 32x32 sub-tile = 4x4 accumulator tiles = 16 `simdgroup_float8x8`). Each K step loads 4 A tiles and 4 B tiles and performs 16
   multiply-accumulates, so loads are amortized 2x over MMAs.
3. **Stage through threadgroup memory** when several SIMD groups share operand tiles: all threads cooperatively copy the A (BLOCK_M x BLOCK_K) and B
   (BLOCK_K x BLOCK_N) tiles with vectorized 16-byte loads, barrier, multiply, barrier. Budget: a 64x32 + 32x64 fp16 pair is 8 KiB; double buffering still fits
   comfortably under 32 KiB and lets the next tile's loads overlap the current MMAs.
4. **Bank conflicts**: pad threadgroup tile row strides by a few elements (e.g. 8 halves) so that `simdgroup_load` rows land in different banks.
5. **Accumulate in float.** Half accumulation fails the correctness tolerance on long K.
6. **Fuse the epilogue** (SiLU, scaling, casts) on the accumulator registers (see `thread_elements` in the MSL reference) before one rounding and store, rather than writing
   the matrix and reading it back.
7. **Convolution as implicit GEMM**: the A tile is a gather of input windows; consecutive output pixels in a row share input values across the 3x3 window.
   Gathering with 16-byte vector loads along the channel dimension (channels are innermost in NHWC) and staging through threadgroup memory removes redundant
   global reads; K blocks of 32 channels never straddle a (kh, kw) boundary when `C % 32 == 0`.
8. Do not let the compiler spill: a thread-local array indexed by a runtime value lives in memory. Use compile-time indices (fully unrolled loops) for accumulator arrays.

## Divergence, barriers and scheduling
Keywords: divergence barrier uniform control flow occupancy register pressure spill unroll

- A barrier must be executed by every thread of the TG; compute the loop trip count from compile-time constants so it is uniform.
- Divergent branches inside a SIMD group serialize both paths; prefer `select` for short conditionals.
- Heavy unrolling can raise register use and spill; check wall/GPU time and, when available, compiler spill/register data after every structural change. Do not raise `maxTotalThreadsPerThreadgroup` just to satisfy a guess.
- Integer division and modulo by non-constants are expensive; use contract constants (powers of two become shifts) or precompute indices outside inner loops.
- 64-bit address arithmetic is costlier than 32-bit; use `uint` indices where the element offset fits and widen only for the base pointer.

## Numerical-safety checklist (the harness tests all of these)
Keywords: overflow underflow cancellation accuracy fp16 accumulation tolerance

- Sums of squares, dot products, softmax denominators: `float` accumulators (fp16 overflows at 65504 and loses precision quickly).
- Softmax: subtract the row max before exponentiation; the `large` test case uses logits of magnitude 1e4.
- Sigmoid/SiLU: evaluate `exp` on `-|x|` so it never overflows.
- RoPE: use `precise::sin/cos` for angles (up to ~1e3 rad); fast-math trig is inaccurate there.
- Do not rely on subnormals being preserved in float/bfloat arithmetic (flushed to zero on this GPU).
- Write every output element exactly once; outputs are pre-filled with NaN and the result must not depend on their initial contents.

## A suggested optimization order
Keywords: workflow plan hypothesis strategy

1. Read the feedback's roofline: is the kernel memory- or compute-bound, and how far above the lower bound (`time_over_lower_bound`) is it?
2. Memory-bound and > 1.15x above the bound: vectorize loads, add loads in flight, remove second passes over global memory, remove masks.
3. Compute-bound: move to `simdgroup_matrix` with float accumulators, then register blocking, then threadgroup staging and double buffering, then the fused epilogue.
4. If the pipeline reports a launch-size cap, look for real spill/register evidence before changing live state; the field alone is not a diagnosis.
5. Change one thing at a time; the harness only keeps a candidate that is correct, safe and faster than the current best.
