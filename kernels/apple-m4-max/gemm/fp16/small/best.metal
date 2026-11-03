// gemm fp16: C[256,256] = A[256,128] @ B[128,256], fp32 accumulation, one rounding.
//
// Geometry (measured, see NOTES.md): 16 threadgroups x 256 threads. A threadgroup owns a
// 32x128 strip of C and each of its 8 SIMD groups owns a 16x32 tile = 2x4
// simdgroup_float8x8 accumulators (8 fragments, no spill; 16 fragments with a split-K
// pair spilled and cost 5x). Operands are loaded straight from device memory with
// simdgroup_load: the whole problem is 128 KB of input so it stays cache resident, and
// staging tiles through threadgroup memory (one barrier per K-slice, plus the extra
// threadgroup traffic) measured 0.5-1.5 us slower on this ~1.6 us kernel.
//
// 16 threadgroups beat 32 and 64 by ~2% and 4 threadgroups (4 cores busy) by 5x.
// 2x k-unroll (two K-steps per iteration) is ~1% faster than the single-step loop.

/* taic-launch
launch gemm_kernel grid=8,2,1 threads=256,1,1
*/

kernel void gemm_kernel(
    device const half* A [[buffer(0)]],
    device const half* B [[buffer(1)]],
    device half* C [[buffer(2)]],
    uint2 tg [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    constexpr uint KK = 128;   // K: row stride of A, row count of B
    constexpr uint NN = 256;   // N: row stride of B and C
    constexpr uint TM = 2;     // 8x8 tiles along M per SIMD group -> 16 rows
    constexpr uint TN = 4;     // 8x8 tiles along N per SIMD group -> 32 columns

    // threadgroup (32 rows x 128 cols) = 2x4 SIMD groups; tg.x walks M, tg.y walks N.
    uint m0 = (tg.x * 2 + (sg >> 2)) * (TM * 8);
    uint n0 = (tg.y * 4 + (sg & 3)) * (TN * 8);

    simdgroup_float8x8 acc[TM][TN];
    for (uint i = 0; i < TM; ++i)
        for (uint j = 0; j < TN; ++j) acc[i][j] = simdgroup_float8x8(0.0f);

    for (uint k = 0; k < KK; k += 16) {
        simdgroup_half8x8 a[2][TM], b[2][TN];
        for (uint s = 0; s < 2; ++s) {
            for (uint i = 0; i < TM; ++i)
                simdgroup_load(a[s][i], A + (m0 + i * 8) * KK + k + s * 8, KK);
            for (uint j = 0; j < TN; ++j)
                simdgroup_load(b[s][j], B + (k + s * 8) * NN + n0 + j * 8, NN);
        }
        for (uint s = 0; s < 2; ++s)
            for (uint i = 0; i < TM; ++i)
                for (uint j = 0; j < TN; ++j)
                    simdgroup_multiply_accumulate(acc[i][j], a[s][i], b[s][j], acc[i][j]);
    }

    // Accumulator fragment layout: with q = lane/4, lane l owns (row, col) and
    // (row, col + 1) of every 8x8 tile, where row = (q & 4) + ((l/2) % 4) and
    // col = (q & 2) * 2 + (l % 2) * 2. Writing half2 keeps it to one 4-byte store per
    // tile per lane and narrows the fp32 accumulator exactly once.
    uint q = lane / 4;
    uint row = (q & 4) + ((lane / 2) % 4);
    uint col = (q & 2) * 2 + (lane % 2) * 2;
    for (uint i = 0; i < TM; ++i)
        for (uint j = 0; j < TN; ++j) {
            auto e = acc[i][j].thread_elements();
            *(reinterpret_cast<device half2*>(C + (m0 + i * 8 + row) * NN + n0 + j * 8 + col)) =
                half2(e[0], e[1]);
        }
}
