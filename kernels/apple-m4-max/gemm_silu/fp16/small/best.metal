// gemm_silu, fp16, M=256 N=256 K=128: C = silu(A @ B).
//
// One threadgroup owns a 16x32 output tile and runs four SIMD groups; the four groups split
// K into 32-wide slices, so every accumulator fragment is 8x8 fp32 over a quarter of K.
// Groups 1..3 hand their partial sums over through threadgroup memory, group 0 adds them,
// applies SiLU on the fp32 sum and rounds once to fp16 on the way out.
//
// Why this shape: the fragment count per SIMD group is what the register file tolerates here
// (8 tiles = 2x4), the four-way K split raises the resident SIMD-group count to 512 without
// shrinking the tile, and a single epilogue group is cheaper than spreading the handoff over
// all four (measured: 1.356 us GPU vs 1.819 us, same window).
/* taic-launch
math fast
launch gemm_silu_kernel grid=8,16,1 threads=128,1,1
*/

static inline float silu_f32(float x) {
    return x / (1.0f + exp(-x));
}

[[max_total_threads_per_threadgroup(128)]]
kernel void gemm_silu_kernel(
    device const half* A [[buffer(0)]],
    device const half* B [[buffer(1)]],
    device half* C [[buffer(2)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint warp [[simdgroup_index_in_threadgroup]]) {
    constexpr uint TILE_M = 16, TILE_N = 32;
    constexpr uint LDA = 128, LDB = 256, LDC = 256;   // row strides of A, B, C in halves
    constexpr uint FNM = TILE_M / 8, FNN = TILE_N / 8; // accumulator fragments along M, N
    constexpr uint FRAGS = FNM * FNN;                  // 8 fragments = 1 KB of fp32 output
    constexpr uint KSPAN = 128 / 4;                    // K handled by one group
    constexpr uint KSTEP = 8;                          // one 8x8x8 multiply per step

    threadgroup float2 part[3 * FRAGS * 32];           // partials from groups 1..3

    const uint m0 = tgid.y * TILE_M;
    const uint n0 = tgid.x * TILE_N;
    const uint k0 = warp * KSPAN;
    device const half* ap = A + m0 * LDA + k0;
    device const half* bp = B + n0 + k0 * LDB;

    simdgroup_float8x8 acc[FNM][FNN];
#pragma unroll
    for (uint i = 0; i < FNM; ++i)
#pragma unroll
        for (uint j = 0; j < FNN; ++j) acc[i][j] = simdgroup_float8x8(0.0f);

    // Operands are read straight from device memory as fragments: A stays in L2 across the
    // eight N blocks, B across the sixteen M blocks, and staging cost measured slower here.
#pragma unroll
    for (uint k = 0; k < KSPAN; k += KSTEP) {
        simdgroup_half8x8 a[FNM][KSTEP / 8], b[FNN][KSTEP / 8];
#pragma unroll
        for (uint s = 0; s < KSTEP / 8; ++s) {
#pragma unroll
            for (uint i = 0; i < FNM; ++i)
                simdgroup_load(a[i][s], ap + i * 8 * LDA + k + s * 8, LDA);
#pragma unroll
            for (uint j = 0; j < FNN; ++j)
                simdgroup_load(b[j][s], bp + (k + s * 8) * LDB + j * 8, LDB);
        }
#pragma unroll
        for (uint s = 0; s < KSTEP / 8; ++s)
#pragma unroll
            for (uint i = 0; i < FNM; ++i)
#pragma unroll
                for (uint j = 0; j < FNN; ++j)
                    simdgroup_multiply_accumulate(acc[i][j], a[i][s], b[j][s], acc[i][j]);
    }

    // Lane-owned element pair of an 8x8 fragment (verified on this GPU for float8x8).
    const uint q = lane >> 2;
    const uint row = (q & 4u) + ((lane >> 1) & 3u);
    const uint col = (q & 2u) * 2u + (lane & 1u) * 2u;

    if (warp != 0u) {
#pragma unroll
        for (uint i = 0; i < FNM; ++i)
#pragma unroll
            for (uint j = 0; j < FNN; ++j) {
                const uint f = i * FNN + j;
                auto e = acc[i][j].thread_elements();
                part[(warp - 1u) * (FRAGS * 32u) + f * 32u + lane] = float2(e[0], e[1]);
            }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (warp != 0u) return;

    device half* out = C + (m0 + row) * LDC + n0 + col;
#pragma unroll
    for (uint i = 0; i < FNM; ++i)
#pragma unroll
        for (uint j = 0; j < FNN; ++j) {
            const uint f = i * FNN + j;
            auto e = acc[i][j].thread_elements();
            float2 sum = float2(e[0], e[1]);           // this group's 32 of K
#pragma unroll
            for (uint w = 0; w + 1u < 4; ++w) {        // plus three fixed-order partials
                const float2 o = part[w * (FRAGS * 32u) + f * 32u + lane];
                sum.x += o.x;
                sum.y += o.y;
            }
            *reinterpret_cast<device half2*>(out + i * 8 * LDC + j * 8) =
                half2(silu_f32(sum.x), silu_f32(sum.y));
        }
}
