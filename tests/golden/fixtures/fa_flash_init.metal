// ============================================================================
// FlashAttention forward (non-causal).  fp16 storage, fp32 accumulate.
//   grid = (S/BLOCK_M, B*H) threadgroups, 128 threads = 4 SIMD groups.
//   SIMD group `sgid` owns query rows [m0+8*sgid, m0+8*sgid+8) of the block.
//   Per K-iteration the tg stages K,V tiles (BLOCK_N x D) cooperatively, then
//     scores(8 x BLOCK_N) = Q(8 x D) @ K^T(D x BLOCK_N)   (8x8x8 MMA, fp32 acc)
//     online softmax on the accumulator registers (row max / rescale)
//     acc(8 x D) += P(8 x BLOCK_N) @ V(BLOCK_N x D)       (fp16 P)
//   The next K/V tile is prefetched into registers while the MMAs run.
// ============================================================================

constant constexpr uint  QS       = 72;                      // padded TG row stride (halves)
constant constexpr float QK_SCALE = SM_SCALE * 1.4426950408889634f;   // scale with log2(e) folded in

kernel void flash_attention_kernel(
    device const half* q [[buffer(0)]],
    device const half* k [[buffer(1)]],
    device const half* v [[buffer(2)]],
    device half*       o [[buffer(3)]],
    uint2 tgp  [[threadgroup_position_in_grid]],
    uint  tid  [[thread_index_in_threadgroup]],
    uint  sgid [[simdgroup_index_in_threadgroup]])
{
    const uint pid_m = tgp.x;                       // 0 .. S/BLOCK_M-1
    const uint bh    = tgp.y;                       // 0 .. B*H-1
    const uint m0    = pid_m * BLOCK_M;
    const uint r0    = sgid * 8u;                   // first query row of this SIMD group

    const size_t bh_off = (size_t)bh * (size_t)S * (size_t)D;
    device const half* qb = q + bh_off;
    device const half* kb = k + bh_off;
    device const half* vb = v + bh_off;
    device half*       ob = o + bh_off;

    // ---- threadgroup staging for K and V: row stride QS halves (144 B, conflict
    //      free for both plain and transposed 8x8 simdgroup loads) ----
    threadgroup uint4 Kstage[(BLOCK_N * QS) / 8];
    threadgroup uint4 Vstage[(BLOCK_N * QS) / 8];
    threadgroup half* Ks = reinterpret_cast<threadgroup half*>(Kstage);
    threadgroup half* Vs = reinterpret_cast<threadgroup half*>(Vstage);

    constexpr uint KT = D / 8;                      // 8 k-tiles of the Q/K reduction
    constexpr uint NT = BLOCK_N / 8;                // 4 score tiles
    constexpr uint ROWS_PER_PASS = 128 / 8;         // 16 rows staged by 128 threads

    // ---- Q fragments (rows [m0+r0, +8) x D), live in registers for the loop ----
    simdgroup_half8x8 qf[KT];
    #pragma unroll
    for (uint jt = 0; jt < KT; ++jt)
        simdgroup_load(qf[jt], qb + (size_t)(m0 + r0) * D + jt * 8, D);

    simdgroup_float8x8 acc[KT];
    #pragma unroll
    for (uint jt = 0; jt < KT; ++jt)
        acc[jt] = simdgroup_float8x8(0.0f);

    float m_i = -INFINITY;
    float l_i = 0.0f;

    // ---- staging registers: each thread copies 16 halves of K and of V ----
    const uint sr   = tid >> 3;                     // source row    0..15
    const uint sc   = (tid & 7u) * 8u;              // source column 0,8,..,56
    const uint stg0 = (sr * QS + sc) / 8;           // Kstage/Vstage slot for row sr
    const uint stg1 = ((sr + ROWS_PER_PASS) * QS + sc) / 8;   // row sr+16
    uint4 kr0, kr1, vr0, vr1;
    {
        kr0 = *reinterpret_cast<device const uint4*>(kb + (size_t)sr * D + sc);
        kr1 = *reinterpret_cast<device const uint4*>(kb + (size_t)(sr + ROWS_PER_PASS) * D + sc);
        vr0 = *reinterpret_cast<device const uint4*>(vb + (size_t)sr * D + sc);
        vr1 = *reinterpret_cast<device const uint4*>(vb + (size_t)(sr + ROWS_PER_PASS) * D + sc);
    }

    for (uint n0 = 0; n0 < S; n0 += BLOCK_N) {
        // ---- publish the staged tile (barrier first: last iteration's readers) ----
        threadgroup_barrier(mem_flags::mem_threadgroup);
        Kstage[stg0] = kr0;
        Kstage[stg1] = kr1;
        Vstage[stg0] = vr0;
        Vstage[stg1] = vr1;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ---- prefetch the next tile while the MMAs below run ----
        if (n0 + BLOCK_N < S) {
            const size_t g = (size_t)(n0 + BLOCK_N) * D + (size_t)sr * D + sc;
            kr0 = *reinterpret_cast<device const uint4*>(kb + g);
            kr1 = *reinterpret_cast<device const uint4*>(kb + g + (size_t)ROWS_PER_PASS * D);
            vr0 = *reinterpret_cast<device const uint4*>(vb + g);
            vr1 = *reinterpret_cast<device const uint4*>(vb + g + (size_t)ROWS_PER_PASS * D);
        }

        // ------------------- scores = Q @ K^T -------------------
        simdgroup_float8x8 sm[NT];
        #pragma unroll
        for (uint jt = 0; jt < NT; ++jt) sm[jt] = simdgroup_float8x8(0.0f);
        #pragma unroll
        for (uint kt = 0; kt < KT; ++kt) {
            #pragma unroll
            for (uint jt = 0; jt < NT; ++jt) {
                simdgroup_half8x8 bf;
                simdgroup_load(bf, Ks + (jt * 8) * QS + kt * 8, QS, ulong2(0, 0), true);
                simdgroup_multiply_accumulate(sm[jt], qf[kt], bf, sm[jt]);
            }
        }

        // ------------------- online softmax -------------------
        // lane l owns (row, col),(row, col+1) of the 8x8 tile; the four lanes of
        // one row are {2r,2r+1,2r+8,2r+9} -> xor 1 then xor 8 reduces a row.
        float rmax = -INFINITY;
        #pragma unroll
        for (uint jt = 0; jt < NT; ++jt) {
            auto e = sm[jt].thread_elements();
            const float a = e[0] * QK_SCALE;
            const float b = e[1] * QK_SCALE;
            sm[jt].thread_elements()[0] = a;
            sm[jt].thread_elements()[1] = b;
            rmax = max(rmax, max(a, b));
        }
        rmax = max(rmax, simd_shuffle_xor(rmax, 1));
        rmax = max(rmax, simd_shuffle_xor(rmax, 8));

        const float m_new = max(m_i, rmax);
        const float alpha = exp2(m_i - m_new);

        float rsum = 0.0f;
        #pragma unroll
        for (uint jt = 0; jt < NT; ++jt) {
            auto e = sm[jt].thread_elements();
            const float a = exp2(e[0] - m_new);
            const float b = exp2(e[1] - m_new);
            sm[jt].thread_elements()[0] = a;
            sm[jt].thread_elements()[1] = b;
            rsum += a + b;
        }
        rsum += simd_shuffle_xor(rsum, 1);
        rsum += simd_shuffle_xor(rsum, 8);

        l_i = l_i * alpha + rsum;
        m_i = m_new;

        #pragma unroll
        for (uint jt = 0; jt < KT; ++jt) {
            auto e = acc[jt].thread_elements();
            const float a = e[0] * alpha;
            const float b = e[1] * alpha;
            acc[jt].thread_elements()[0] = a;
            acc[jt].thread_elements()[1] = b;
        }

        // ------------------- acc += P @ V -------------------
        simdgroup_half8x8 pf[NT];
        #pragma unroll
        for (uint jt = 0; jt < NT; ++jt) {
            auto se = sm[jt].thread_elements();
            pf[jt].thread_elements()[0] = (half)se[0];
            pf[jt].thread_elements()[1] = (half)se[1];
        }
        #pragma unroll
        for (uint kt = 0; kt < NT; ++kt) {
            #pragma unroll
            for (uint jt = 0; jt < KT; ++jt) {
                simdgroup_half8x8 bf;
                simdgroup_load(bf, Vs + (kt * 8) * QS + jt * 8, QS);
                simdgroup_multiply_accumulate(acc[jt], pf[kt], bf, acc[jt]);
            }
        }
    }

    // ------------------- epilogue -------------------
    const float inv = 1.0f / l_i;
    device half* orow = ob + (size_t)(m0 + r0) * D;
    #pragma unroll
    for (uint jt = 0; jt < KT; ++jt) {
        auto e = acc[jt].thread_elements();
        simdgroup_half8x8 ho;
        ho.thread_elements()[0] = (half)(e[0] * inv);
        ho.thread_elements()[1] = (half)(e[1] * inv);
        simdgroup_store(ho, orow + jt * 8, D);
    }
}
