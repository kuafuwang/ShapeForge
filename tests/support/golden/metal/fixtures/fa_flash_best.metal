// ============================================================================
// FlashAttention forward (non-causal).  fp16 storage, fp32 accumulate.
//   grid = (S/BLOCK_M, B*H) threadgroups, 128 threads = 4 SIMD groups.
//   SIMD group `sgid` owns query rows [m0+8*sgid, m0+8*sgid+8) of the block.
//
//   Staging uses 8-key slices in a four-slice ring:
//       Ka,Kb,Va,Vb = 4 * (NF*FP*4 B) = 4 * 1056 B = 4224 B
//   i.e. half of the previous 8448 B, so twice as many threadgroups fit into a
//   core's 32 KiB threadgroup pool.
//
//   The online-softmax cadence is still 16 keys: the even 8-key slice is scored
//   into sm[0] (slice A) while the odd one is scored into sm[1] (slice B); the
//   row-max / exp2 / l-update / acc-rescale pass then runs once over both
//   halves and both P fragments feed the P @ V MMAs.  Since the even slice's V
//   buffer is refilled by the odd slice's staging step, its V fragments are
//   parked in registers for exactly that one step, which keeps the ring two
//   deep with a single barrier per 8-key step (two per 16 keys).
//   The QK reduction runs on two partial accumulators so two independent MMA
//   chains stay in flight per slice.
//
//   Fragment slot for lane l (8x8 tile lane map):
//     row(l) = ((l>>1)&3) | ((l&16)>>2),  col(l) = 2*(l&1) + 4*((l>>3)&1)
//   => slot(row,col) = (row&3)<<1 | ((col&1)) ... precisely:
//      slot = ((row&3)<<1) | ((col&2)>>1) | (((col&4)>>2)<<3) | (((row&4)>>2)<<4)
//   With NT = 1 a fragment index is simply the channel tile of the 8x8 tile,
//   for both K^T and V, so every operand fetch in the inner loop is one aligned
//   32-bit threadgroup load at a compile-time constant offset.
//
//   K staging: thread (kp, chq) loads the adjacent keys 2kp and 2kp+1 over
//   channels 4chq..4chq+4; each channel yields one complete (key,key+1)
//   operand word written whole by a single lane.
// ============================================================================

constant constexpr uint  CHUNK   = 8u;                    // keys staged per slice
constant constexpr uint  NCHUNK  = S / CHUNK;             // 128 slices along the keys
constant constexpr uint  KT      = D / 8u;                // 8 tiles of the Q/K reduction
constant constexpr uint  NT      = CHUNK / 8u;            // 1 score tile per slice
constant constexpr uint  NF      = KT * NT;               // operand fragments per slice
constant constexpr uint  FP      = 33u;                   // uint stride between fragments
constant constexpr float QK_SCALE = SM_SCALE * 1.4426950408889634f; // log2(e) folded in

// Stage the four (adjacent-key-pair, channel) operand words owned by one
// thread: keys 2*kp and 2*kp+1 at channels cb..cb+4.
inline void stage_k_row(threadgroup uint* dst, uint2 k0, uint2 k1, uint kbase)
{
    dst[kbase + 0u] = (k0.x & 0xffffu) | (k1.x << 16);
    dst[kbase + 2u] = (k0.x >> 16)    | (k1.x & 0xffff0000u);
    dst[kbase + 4u] = (k0.y & 0xffffu) | (k1.y << 16);
    dst[kbase + 6u] = (k0.y >> 16)    | (k1.y & 0xffff0000u);
}

// Stage one row of V (8 consecutive channels of one key).  The four uint-sized
// pieces of the loaded row are already the two-halves-per-lane pieces the MMA
// consumes, so they are stored whole.
inline void stage_v_row(threadgroup uint* dst, uint4 raw, uint vf, uint vl)
{
    dst[vf * FP + vl + 0u] = raw.x;
    dst[vf * FP + vl + 1u] = raw.y;
    dst[vf * FP + vl + 8u] = raw.z;
    dst[vf * FP + vl + 9u] = raw.w;
}

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
    const uint lane  = tid & 31u;                   // SIMD lane == fragment slot

    const size_t bh_off = (size_t)bh * (size_t)S * (size_t)D;
    device const half* qb = q + bh_off;
    device const half* kb = k + bh_off;
    device const half* vb = v + bh_off;
    device half*       ob = o + bh_off;

    // ---- four 8-key slices, each NF fragments of FP uints (1056 B) ----
    threadgroup uint Ka[NF * FP];
    threadgroup uint Kb[NF * FP];
    threadgroup uint Va[NF * FP];
    threadgroup uint Vb[NF * FP];

    // ---- Q fragments (rows [m0+r0, +8) x D), live in registers for the loop ----
    simdgroup_half8x8 qf[KT];
    #pragma unroll
    for (uint jt = 0; jt < KT; ++jt)
        simdgroup_load(qf[jt], qb + (size_t)(m0 + r0) * (size_t)D + jt * 8u, D);

    simdgroup_float8x8 acc[KT];
    #pragma unroll
    for (uint jt = 0; jt < KT; ++jt)
        acc[jt] = simdgroup_float8x8(0.0f);

    float m_i = -INFINITY;
    float l_i = 0.0f;

    // ---- K staging map (threads 0..63): thread (kp, chq) covers keys
    //      2kp, 2kp+1 over channels 4chq..4chq+4; kp in 0..3 -> 8 keys ----
    const bool   is_k  = (tid < 64u);
    const uint   kp    = tid >> 4;                  // 0..3
    const uint   chq   = tid & 15u;                 // channel quad 0..15
    const size_t k_off = (size_t)(2u * kp) * (size_t)D + 4u * chq;
    const uint   kb0   = (chq >> 1u) * FP           // channel tile (NT = 1)
                       + (kp & 1u) + (((kp >> 1u) & 1u) << 3) + ((chq & 1u) << 4);

    // ---- V staging map (threads 64..127): thread (sr, sc) covers key sr,
    //      channels sc..sc+8; sr in 0..7 -> 8 keys ----
    const uint   u       = tid - 64u;               // 0..63
    const uint   sr      = u >> 3;                  // source key row 0..7
    const uint   sc      = (u & 7u) * 8u;           // source channel 0,8,..,56
    const size_t src_off = (size_t)sr * (size_t)D + sc;
    const uint   vf      = (sc >> 3) * NT;          // V fragment (channel tile)
    const uint   vl      = ((sr & 3u) << 1) | (((sr >> 2u) & 1u) << 4);

    uint2 k0  = uint2(0u), k1 = uint2(0u);          // K registers of slice i+1
    uint4 cv  = uint4(0u);                          // V registers of slice i+1
    uint2 nk0 = uint2(0u), nk1 = uint2(0u);         // in-flight loads of slice i+2
    uint4 nv  = uint4(0u);

    // ---- prologue: slice A <- slice 0, registers <- slice 1 ----
    if (is_k) {
        k0 = *reinterpret_cast<device const uint2*>(kb + k_off);
        k1 = *reinterpret_cast<device const uint2*>(kb + k_off + (size_t)D);
        stage_k_row(Ka, k0, k1, kb0);
        const size_t g = (size_t)CHUNK * (size_t)D;
        k0 = *reinterpret_cast<device const uint2*>(kb + g + k_off);
        k1 = *reinterpret_cast<device const uint2*>(kb + g + k_off + (size_t)D);
    } else {
        cv = *reinterpret_cast<device const uint4*>(vb + src_off);
        stage_v_row(Va, cv, vf, vl);
        cv = *reinterpret_cast<device const uint4*>(vb + (size_t)CHUNK * (size_t)D + src_off);
    }

    // V fragments of the even slice parked here while the odd slice is scored
    // (its threadgroup buffer is refilled during that step).
    uint vr[KT];

    for (uint c = 0; c < NCHUNK; c += 2u) {         // 16-key online-softmax cadence
        simdgroup_float8x8 sm[2];

        #pragma unroll
        for (uint p = 0; p < 2u; ++p) {             // p is compile time: slice parity
            const uint i = c + p;                   // 8-key slice index

            // ---- global loads of slice i+2 in flight (issued before the barrier) ----
            if (i + 2u < NCHUNK) {
                const size_t g = (size_t)(i + 2u) * (size_t)CHUNK * (size_t)D;
                if (is_k) {
                    nk0 = *reinterpret_cast<device const uint2*>(kb + g + k_off);
                    nk1 = *reinterpret_cast<device const uint2*>(kb + g + k_off + (size_t)D);
                } else {
                    nv  = *reinterpret_cast<device const uint4*>(vb + g + src_off);
                }
            }

            // ---- the single barrier of the slice ----
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // ---- publish slice i+1 into slice (i+1)&1 (last read in step i-1) ----
            if (i + 1u < NCHUNK) {
                if (p == 0u) {
                    if (is_k) stage_k_row(Kb, k0, k1, kb0);
                    else      stage_v_row(Vb, cv, vf, vl);
                } else {
                    if (is_k) stage_k_row(Ka, k0, k1, kb0);
                    else      stage_v_row(Va, cv, vf, vl);
                }
            }

            threadgroup const uint* Kp = (p == 0u) ? Ka : Kb;
            threadgroup const uint* Vp = (p == 0u) ? Va : Vb;

            // ------------------- scores = Q @ K^T (8 x 8) -------------------
            // two partial accumulators keep two independent MMA chains in flight
            simdgroup_float8x8 s0 = simdgroup_float8x8(0.0f);
            simdgroup_float8x8 s1 = simdgroup_float8x8(0.0f);
            #pragma unroll
            for (uint t = 0; t < KT / 2u; ++t) {
                simdgroup_half8x8 ba, bb;
                half2 ha = as_type<half2>(Kp[(2u * t) * FP + lane]);
                half2 hb = as_type<half2>(Kp[(2u * t + 1u) * FP + lane]);
                ba.thread_elements()[0] = ha[0];
                ba.thread_elements()[1] = ha[1];
                bb.thread_elements()[0] = hb[0];
                bb.thread_elements()[1] = hb[1];
                simdgroup_multiply_accumulate(s0, qf[2u * t], ba, s0);
                simdgroup_multiply_accumulate(s1, qf[2u * t + 1u], bb, s1);
            }
            {
                // NOTE: thread_elements() hands back a reference to the fragment
                // storage; write through the accessor expression (an `auto` copy
                // would swallow the store).
                const float a0 = s0.thread_elements()[0] + s1.thread_elements()[0];
                const float a1 = s0.thread_elements()[1] + s1.thread_elements()[1];
                sm[p].thread_elements()[0] = a0;
                sm[p].thread_elements()[1] = a1;
            }

            if (p == 0u) {                          // park the even slice's V fragments
                #pragma unroll
                for (uint jt = 0; jt < KT; ++jt)
                    vr[jt] = Vp[jt * FP + lane];
            }

            k0 = nk0; k1 = nk1; cv = nv;
        }

        // ------------------- online softmax over the 16 keys -------------------
        // lane l owns (row, col),(row, col+1); the four lanes of one row are
        // {2r,2r+1,2r+8,2r+9} -> xor 1 then xor 8 reduces a row.
        float rmax = -INFINITY;
        #pragma unroll
        for (uint p = 0; p < 2u; ++p) {
            auto e = sm[p].thread_elements();
            const float a = e[0] * QK_SCALE;
            const float b = e[1] * QK_SCALE;
            sm[p].thread_elements()[0] = a;
            sm[p].thread_elements()[1] = b;
            rmax = max(rmax, max(a, b));
        }
        rmax = max(rmax, simd_shuffle_xor(rmax, 1));
        rmax = max(rmax, simd_shuffle_xor(rmax, 8));

        const float m_new = max(m_i, rmax);
        const float alpha = exp2(m_i - m_new);

        float rsum = 0.0f;
        #pragma unroll
        for (uint p = 0; p < 2u; ++p) {
            auto e = sm[p].thread_elements();
            const float a = exp2(e[0] - m_new);
            const float b = exp2(e[1] - m_new);
            sm[p].thread_elements()[0] = a;
            sm[p].thread_elements()[1] = b;
            rsum += a + b;
        }
        rsum += simd_shuffle_xor(rsum, 1);
        rsum += simd_shuffle_xor(rsum, 8);

        l_i = l_i * alpha + rsum;
        m_i = m_new;

        #pragma unroll
        for (uint jt = 0; jt < KT; ++jt) {
            auto e = acc[jt].thread_elements();
            acc[jt].thread_elements()[0] = e[0] * alpha;
            acc[jt].thread_elements()[1] = e[1] * alpha;
        }

        // ------------------- acc += P @ V (two 8-key key tiles) -------------------
        simdgroup_half8x8 pf[2];
        #pragma unroll
        for (uint p = 0; p < 2u; ++p) {
            auto se = sm[p].thread_elements();
            pf[p].thread_elements()[0] = (half)se[0];
            pf[p].thread_elements()[1] = (half)se[1];
        }
        #pragma unroll
        for (uint jt = 0; jt < KT; ++jt) {          // channel tile
            simdgroup_half8x8 bA, bB;
            half2 hA = as_type<half2>(vr[jt]);      // even slice, from registers
            half2 hB = as_type<half2>(Vb[jt * FP + lane]); // odd slice, still in threadgroup
            bA.thread_elements()[0] = hA[0];
            bA.thread_elements()[1] = hA[1];
            bB.thread_elements()[0] = hB[0];
            bB.thread_elements()[1] = hB[1];
            simdgroup_multiply_accumulate(acc[jt], pf[0], bA, acc[jt]);
            simdgroup_multiply_accumulate(acc[jt], pf[1], bB, acc[jt]);
        }
    }

    // ------------------- epilogue -------------------
    const float inv = 1.0f / l_i;
    device half* orow = ob + (size_t)(m0 + r0) * (size_t)D;
    #pragma unroll
    for (uint jt = 0; jt < KT; ++jt) {
        auto e = acc[jt].thread_elements();
        simdgroup_half8x8 ho;
        ho.thread_elements()[0] = (half)(e[0] * inv);
        ho.thread_elements()[1] = (half)(e[1] * inv);
        simdgroup_store(ho, orow + jt * 8u, D);
    }
}
