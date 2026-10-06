using namespace metal;

// ---- tile geometry -------------------------------------------------------
// A tile: BLOCK_M x BLOCK_K staged row-major, row stride padded by 8 halves
// B tile: BLOCK_K x BLOCK_N staged row-major, row stride padded by 8 halves
constant constexpr uint A_STRIDE = BLOCK_K + 8;          // 40 halves
constant constexpr uint B_STRIDE = BLOCK_N + 8;          // 72 halves
constant constexpr uint C_STRIDE = BLOCK_N + 8;          // 72 halves
constant constexpr uint A_TILE_H = BLOCK_M * A_STRIDE;   // 2560 halves
constant constexpr uint B_TILE_H = BLOCK_K * B_STRIDE;   // 2304 halves
constant constexpr uint SMEM_H   = A_TILE_H + B_TILE_H;  // 4864 halves = 9728 B
constant constexpr uint SMEM_V   = SMEM_H / 8;           // 608 uint4
constant constexpr uint K_ITERS  = K / BLOCK_K;          // 64

// ---- cooperative global -> register staging (16 bytes per thread per tile) --
inline void load_a(device const half* A, uint m0, uint k0, uint lid,
                   thread uint4& x0, thread uint4& x1) {
    uint s = lid;
    x0 = *reinterpret_cast<device const uint4*>(A + (m0 + (s >> 2)) * K + k0 + ((s & 3u) << 3));
    s += 128u;
    x1 = *reinterpret_cast<device const uint4*>(A + (m0 + (s >> 2)) * K + k0 + ((s & 3u) << 3));
}

inline void load_b(device const half* B, uint k0, uint n0, uint lid,
                   thread uint4& x0, thread uint4& x1) {
    uint s = lid;
    x0 = *reinterpret_cast<device const uint4*>(B + (k0 + (s >> 3)) * N + n0 + ((s & 7u) << 3));
    s += 128u;
    x1 = *reinterpret_cast<device const uint4*>(B + (k0 + (s >> 3)) * N + n0 + ((s & 7u) << 3));
}

inline void store_a(threadgroup half* As, uint lid, uint4 x0, uint4 x1) {
    uint s = lid;
    *reinterpret_cast<threadgroup uint4*>(As + (s >> 2) * A_STRIDE + ((s & 3u) << 3)) = x0;
    s += 128u;
    *reinterpret_cast<threadgroup uint4*>(As + (s >> 2) * A_STRIDE + ((s & 3u) << 3)) = x1;
}

inline void store_b(threadgroup half* Bs, uint lid, uint4 x0, uint4 x1) {
    uint s = lid;
    *reinterpret_cast<threadgroup uint4*>(Bs + (s >> 3) * B_STRIDE + ((s & 7u) << 3)) = x0;
    s += 128u;
    *reinterpret_cast<threadgroup uint4*>(Bs + (s >> 3) * B_STRIDE + ((s & 7u) << 3)) = x1;
}

kernel void gemm_kernel(device const half* A [[buffer(0)]],
                        device const half* B [[buffer(1)]],
                        device half* C [[buffer(2)]],
                        uint3 tgp [[threadgroup_position_in_grid]],
                        uint lid [[thread_index_in_threadgroup]],
                        uint sgid [[simdgroup_index_in_threadgroup]],
                        uint lane [[thread_index_in_simdgroup]])
{
    // One shared buffer: A tile then B tile, reused as C staging after the K loop.
    threadgroup uint4 smem_raw[SMEM_V];
    threadgroup half* smem = reinterpret_cast<threadgroup half*>(smem_raw);
    threadgroup half* As = smem;
    threadgroup half* Bs = smem + A_TILE_H;

    const uint m0 = tgp.y * BLOCK_M;   // pid_m = program_id(axis=1)
    const uint n0 = tgp.x * BLOCK_N;   // pid_n = program_id(axis=0)

    // ---- prologue: prefetch K step 0 into registers ----------------------
    uint4 pa0, pa1, pb0, pb1;
    load_a(A, m0, 0u, lid, pa0, pa1);
    load_b(B, 0u, n0, lid, pb0, pb1);

    simdgroup_float8x8 acc[4][4];
#pragma unroll
    for (uint i = 0; i < 4; ++i)
#pragma unroll
        for (uint j = 0; j < 4; ++j) acc[i][j] = simdgroup_float8x8(0);

    const uint r0 = (sgid >> 1) * 32u;   // this SIMD group's sub-tile origin
    const uint c0 = (sgid & 1u) * 32u;

    // (3) Operand fragment bases hoisted out of the K loop: with the kk loop
    // unrolled the remaining offsets (i*8*A_STRIDE + kk*8, j*8 + kk*8*B_STRIDE)
    // are compile-time constants, so the inner block emits address adds, not
    // four full 32-bit multiplies per K step.
    threadgroup half* a_frag = As + r0 * A_STRIDE;
    threadgroup half* b_frag = Bs + c0;

    // ---- main K loop ------------------------------------------------------
    for (uint ki = 0; ki < K_ITERS; ++ki) {
        store_a(As, lid, pa0, pa1);
        store_b(Bs, lid, pb0, pb1);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (ki + 1u < K_ITERS) {   // prefetch next tile while the MMAs run
            load_a(A, m0, (ki + 1u) * BLOCK_K, lid, pa0, pa1);
            load_b(B, (ki + 1u) * BLOCK_K, n0, lid, pb0, pb1);
        }

#pragma unroll
        for (uint kk = 0; kk < 4; ++kk) {
            simdgroup_half8x8 af[4], bf[4];
#pragma unroll
            for (uint i = 0; i < 4; ++i)
                simdgroup_load(af[i], a_frag + (i * 8u * A_STRIDE) + (kk * 8u), A_STRIDE);
#pragma unroll
            for (uint j = 0; j < 4; ++j)
                simdgroup_load(bf[j], b_frag + (j * 8u) + (kk * 8u * B_STRIDE), B_STRIDE);
#pragma unroll
            for (uint i = 0; i < 4; ++i)
#pragma unroll
                for (uint j = 0; j < 4; ++j)
                    simdgroup_multiply_accumulate(acc[i][j], af[i], bf[j], acc[i][j]);
        }
        // Kept (not dead): Cs aliases As/Bs, so this drain barrier is what stops
        // a fast SIMD group's C staging stores from landing in operand halves
        // another SIMD group has not read yet.
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // ---- epilogue: round once to fp16, shuffle-paired 8-byte smem stores ---
    threadgroup half* Cs = smem;
    const ushort qid = (ushort)(lane >> 2);
    const ushort rr = (ushort)((qid & 4) + ((lane >> 1) & 3u));
    const ushort cc = (ushort)((qid & 2) * 2 + (lane & 1u) * 2u);
#pragma unroll
    for (uint i = 0; i < 4; ++i) {
#pragma unroll
        for (uint j = 0; j < 4; ++j) {
            thread auto& te = acc[i][j].thread_elements();
            uint mine = as_type<uint>(half2((half)te[0], (half)te[1]));
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
        *reinterpret_cast<device uint4*>(C + (m0 + r) * N + n0 + c) =
            *reinterpret_cast<threadgroup const uint4*>(Cs + r * C_STRIDE + c);
    }
}
