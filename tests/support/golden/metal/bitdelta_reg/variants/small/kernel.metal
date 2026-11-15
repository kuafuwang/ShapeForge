// P2-P REG baseline (small): packed sign bits decoded per output tile into threadgroup
// memory, one 32x32 tile per threadgroup, four SIMD groups of 2x2 simdgroup_matrix
// fragments.  Archived from
// runs/apple-m4-max/20261103-090118_bitdelta_gemm_fp16_deepseek-flash_hh8_ynro/
// candidates/000_init/kernel.metal (the 2026-11-03 paid-search init candidate, search
// best 2.765x).  The only edit is the v2 ABI rebinding of the OpSpec attr SCALE: it is a
// compile-time attribute constant in the assembled Metal header, not a buffer slot, so
// the output moves from buffer(3) to buffer(2) and the epilogue reads SCALE.  Semantics,
// tiling, decode mapping and store pattern are unchanged.
/* taic-launch
launch bitdelta_gemm_kernel grid=8,4,1 threads=128,1,1
*/

// BitDelta GEMM: y = scale * (x @ W), W[k,n] = +1 if bit k of column n is set else -1.
//
// One threadgroup owns a 32x32 output tile and is split into 4 SIMD groups, each owning a
// 16x16 quadrant (2x2 simdgroup_matrix fragments). The packed signs of the tile's 32
// columns are decoded into threadgroup memory once per 32-k chunk, so every decoded +-1 is
// reused by all 128 threads; x is consumed straight from device memory as 8x8 fragments
// (leading dimension K), which removes a staging round trip that measured slower. Products
// accumulate in fp32 fragments and are rounded to fp16 exactly once, in the epilogue.

static inline half pm1(uint w, uint j) {
    // bit j of the packed word -> +1 / -1 in the element type
    return (((w >> j) & 1u) != 0u) ? half(1.0f) : half(-1.0f);
}

kernel void bitdelta_gemm_kernel(
    device const half* x [[buffer(0)]],
    device const uint* bits [[buffer(1)]],
    device half* y [[buffer(2)]],
    uint2 tg [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]])
{
    constexpr uint TM = 32;   // output rows per threadgroup
    constexpr uint TN = 32;   // output cols per threadgroup
    constexpr uint LD = 40;   // padded leading dimension (halves / floats)

    const uint m0 = tg.y * TM;
    const uint n0 = tg.x * TN;

    // Decode mapping: four threads per packed word, eight bits each.
    const uint nn = lid >> 2;        // 0..31, weight column
    const uint kq = (lid & 3) * 8;   // 0..31, k offset inside the chunk

    threadgroup half Bt[TN][LD];     // decoded weights for this chunk, [n][k]
    threadgroup float Cs[TM][LD];    // fp32 epilogue staging

    // Each SIMD group owns a 16x16 quadrant of the tile.
    const uint r0 = (sg >> 1) * 16;
    const uint c0 = (sg & 1) * 16;

    simdgroup_float8x8 acc00 = simdgroup_float8x8(0.0f);
    simdgroup_float8x8 acc01 = simdgroup_float8x8(0.0f);
    simdgroup_float8x8 acc10 = simdgroup_float8x8(0.0f);
    simdgroup_float8x8 acc11 = simdgroup_float8x8(0.0f);

    for (uint kb = 0; kb < K / 32u; ++kb) {
        // ---- decode the packed signs of this 32-k chunk --------------------
        const uint word = bits[kb * N + n0 + nn];
        *reinterpret_cast<threadgroup half4*>(&Bt[nn][kq]) =
            half4(pm1(word, kq), pm1(word, kq + 1u), pm1(word, kq + 2u), pm1(word, kq + 3u));
        *reinterpret_cast<threadgroup half4*>(&Bt[nn][kq + 4u]) =
            half4(pm1(word, kq + 4u), pm1(word, kq + 5u), pm1(word, kq + 6u), pm1(word, kq + 7u));
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ---- four k steps of 8: matrix multiply-accumulate -----------------
        for (uint ks = 0; ks < 4; ++ks) {
            simdgroup_half8x8 a0, a1, b0, b1;
            simdgroup_load(a0, x + (m0 + r0) * K + kb * 32u + ks * 8u, K);
            simdgroup_load(a1, x + (m0 + r0 + 8u) * K + kb * 32u + ks * 8u, K);
            simdgroup_load(b0, &Bt[c0][ks * 8u], LD, ulong2(0, 0), true);
            simdgroup_load(b1, &Bt[c0 + 8u][ks * 8u], LD, ulong2(0, 0), true);
            simdgroup_multiply_accumulate(acc00, a0, b0, acc00);
            simdgroup_multiply_accumulate(acc01, a0, b1, acc01);
            simdgroup_multiply_accumulate(acc10, a1, b0, acc10);
            simdgroup_multiply_accumulate(acc11, a1, b1, acc11);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // ---- single rounding at the output -------------------------------------
    simdgroup_store(acc00, &Cs[r0][c0], LD);
    simdgroup_store(acc01, &Cs[r0][c0 + 8u], LD);
    simdgroup_store(acc10, &Cs[r0 + 8u][c0], LD);
    simdgroup_store(acc11, &Cs[r0 + 8u][c0 + 8u], LD);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint orow = lid >> 2;
    const uint ocol = (lid & 3) * 8;
    const float4 f0 = *reinterpret_cast<threadgroup float4*>(&Cs[orow][ocol]);
    const float4 f1 = *reinterpret_cast<threadgroup float4*>(&Cs[orow][ocol + 4]);
    device half* py = y + (m0 + orow) * N + n0 + ocol;
    *reinterpret_cast<device half4*>(py) = half4(f0 * SCALE);
    *reinterpret_cast<device half4*>(py + 4) = half4(f1 * SCALE);
}
