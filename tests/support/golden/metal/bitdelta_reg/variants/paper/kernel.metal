// P2-P REG baseline (paper): the registered P3 64x64x32 steel-style GEMM tiling with the
// B operand decoded from packed sign bits inside the kernel.  One threadgroup owns a
// 64x64 output tile (four SIMD groups of 2x2 simdgroup_matrix fragments) and decodes the
// 64 packed words of the current 32-k chunk into a n-major threadgroup tile.  Every
// decoded +-1 is reused by all 128 threads and by the four SIMD groups; nothing is written
// to global scratch.  Derived from
// tests/support/golden/metal/fixtures/gemm_paper_best.metal: same tile, same A staging, same
// MMA loop and the same single-rounding epilogue.  Only the B path changed: it reads
// u32 packed words (4 B per 32 weights) instead of an fp16 [K,N] matrix.
using namespace metal;

// ---- tile geometry -------------------------------------------------------
constant constexpr uint BLOCK_M = 64;
constant constexpr uint BLOCK_N = 64;
constant constexpr uint BLOCK_K = 32;

// A tile: BLOCK_M x BLOCK_K staged row-major, row stride padded by 8 halves
// B tile: BLOCK_N x BLOCK_K staged n-major, row stride padded by 8 halves.  A packed word
//         is exactly one 32-k row of one output column, so the decode writes 16
//         contiguous halves per thread.
constant constexpr uint A_STRIDE = BLOCK_K + 8;          // 40 halves
constant constexpr uint B_STRIDE = BLOCK_K + 8;          // 40 halves
constant constexpr uint C_STRIDE = BLOCK_N + 8;          // 72 halves
constant constexpr uint A_TILE_H = BLOCK_M * A_STRIDE;   // 2560 halves
constant constexpr uint B_TILE_H = BLOCK_N * B_STRIDE;   // 2560 halves
constant constexpr uint SMEM_H   = A_TILE_H + B_TILE_H;  // 5120 halves = 10240 B
constant constexpr uint SMEM_V   = SMEM_H / 8;           // 640 uint4
constant constexpr uint K_ITERS  = K / BLOCK_K;          // 128 for K = 4096

// ---- cooperative global -> register staging (16 bytes per thread per tile) --
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

// Two +-1 halves packed in one u32.  A bit of the packed word selects the sign bit of the
// fp16 1.0 constant (0x3C00), so a decoded value is exact: no multiply and no rounding.
inline uint pm1_pair(uint w, uint j) {
    // bit set -> +1.0h (0x3C00), bit clear -> -1.0h (0xBC00); the sign bit is the complement.
    const uint lo = 0x3C00u | ((((w >> j) & 1u) ^ 1u) << 15);
    const uint hi = 0x3C00u | ((((w >> (j + 1u)) & 1u) ^ 1u) << 15);
    return lo | (hi << 16);
}

// 128 threads decode 64 words x 32 bits = 2048 halves.  Thread `lid` owns word `lid >> 1`
// (output column n0 + n) and one 16-bit half of it, i.e. two 16-byte threadgroup stores.
inline void decode_b(device const uint* bits, uint k0, uint n0, uint lid, threadgroup half* Bs) {
    const uint n = lid >> 1;                   // 0..63, weight column inside the tile
    const uint j0 = (lid & 1u) * 16u;          // 0 or 16, k offset inside the chunk
    const uint word = bits[(k0 >> 5) * N + n0 + n];
    const uint4 v0 = uint4(pm1_pair(word, j0 + 0u), pm1_pair(word, j0 + 2u),
                           pm1_pair(word, j0 + 4u), pm1_pair(word, j0 + 6u));
    const uint4 v1 = uint4(pm1_pair(word, j0 + 8u), pm1_pair(word, j0 + 10u),
                           pm1_pair(word, j0 + 12u), pm1_pair(word, j0 + 14u));
    *reinterpret_cast<threadgroup uint4*>(Bs + n * B_STRIDE + j0) = v0;
    *reinterpret_cast<threadgroup uint4*>(Bs + n * B_STRIDE + j0 + 8u) = v1;
}

kernel void bitdelta_gemm_kernel(device const half* x [[buffer(0)]],
                                 device const uint* bits [[buffer(1)]],
                                 device half* y [[buffer(2)]],
                                 uint2 tgp [[threadgroup_position_in_grid]],
                                 uint lid [[thread_index_in_threadgroup]],
                                 uint sgid [[simdgroup_index_in_threadgroup]],
                                 uint lane [[thread_index_in_simdgroup]])
{
    // One shared buffer: A tile then B tile, reused as C staging after the K loop.
    threadgroup uint4 smem_raw[SMEM_V];
    threadgroup half* smem = reinterpret_cast<threadgroup half*>(smem_raw);
    threadgroup half* As = smem;
    threadgroup half* Bs = smem + A_TILE_H;

    const uint m0 = tgp.y * BLOCK_M;
    const uint n0 = tgp.x * BLOCK_N;

    // ---- prologue: prefetch K step 0 into registers ----------------------
    uint4 pa0, pa1;
    load_a(x, m0, 0u, lid, pa0, pa1);

    simdgroup_float8x8 acc[4][4];
#pragma unroll
    for (uint i = 0; i < 4; ++i)
#pragma unroll
        for (uint j = 0; j < 4; ++j) acc[i][j] = simdgroup_float8x8(0);

    const uint r0 = (sgid >> 1) * 32u;   // this SIMD group's sub-tile origin
    const uint c0 = (sgid & 1u) * 32u;

    threadgroup half* a_frag = As + r0 * A_STRIDE;
    threadgroup half* b_frag = Bs + c0 * B_STRIDE;

    // ---- main K loop ------------------------------------------------------
    for (uint ki = 0; ki < K_ITERS; ++ki) {
        store_a(As, lid, pa0, pa1);
        decode_b(bits, ki * BLOCK_K, n0, lid, Bs);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (ki + 1u < K_ITERS) {   // prefetch next A tile while the MMAs run
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
                simdgroup_load(bf[j], b_frag + (j * 8u * B_STRIDE) + (kk * 8u), B_STRIDE,
                               ulong2(0, 0), true);
#pragma unroll
            for (uint i = 0; i < 4; ++i)
#pragma unroll
                for (uint j = 0; j < 4; ++j)
                    simdgroup_multiply_accumulate(acc[i][j], af[i], bf[j], acc[i][j]);
        }
        // Cs aliases As/Bs later, so this drain barrier stops a fast SIMD group's stores
        // from landing in operand halves another SIMD group has not read yet.
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // ---- epilogue: scale, round once to fp16, shuffle-paired 8-byte smem stores ---
    threadgroup half* Cs = smem;
    const ushort qid = (ushort)(lane >> 2);
    const ushort rr = (ushort)((qid & 4) + ((lane >> 1) & 3u));
    const ushort cc = (ushort)((qid & 2) * 2 + (lane & 1u) * 2u);
#pragma unroll
    for (uint i = 0; i < 4; ++i) {
#pragma unroll
        for (uint j = 0; j < 4; ++j) {
            thread auto& te = acc[i][j].thread_elements();
            // SCALE is the OpSpec attr injected as a compile-time float constant.
            uint mine = as_type<uint>(half2((half)(te[0] * SCALE), (half)(te[1] * SCALE)));
            // lane^1 holds the neighbouring 2-half pair of the same C row
            // (rr depends on lane>>1 only, cc on lane&1), so the even lane of
            // each pair writes one 8-byte half4 covering both pairs.
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
}
/* taic-launch
math fast
launch bitdelta_gemm_kernel grid=64,8,1 threads=128,1,1
*/
