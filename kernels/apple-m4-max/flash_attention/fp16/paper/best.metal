/* taic-launch
launch flash_attention_kernel grid=32,16,1 threads=64,1,1
*/
// ---------------------------------------------------------------------------
// FlashAttention forward, non-causal.  fp16 storage, fp32 scores / row statistics /
// accumulator, output rounded once.
//
// One threadgroup owns BLOCK_M = 32 query rows of one (batch, head) and walks the
// S = 1024 keys in 32-key blocks.  Two SIMD groups of 16 query rows, i.e. two 8-row
// tiles (a, b) each, over all D = 64 head dims.  Every key tile is used by both row
// tiles, which is why 16 rows per group beats 8.
//
// K is transposed once into threadgroup memory.  A slot holds the 64 dims of 32 keys
// with a row stride of 36 halves (72 B = 18 banks), so the eight rows of an operand
// tile start in eight different banks.  Two slots ping-pong: the slot of the block
// after next is refilled right after the barrier and before the current block's matrix
// stream, so its latency hides behind it.  One threadgroup barrier per 32 keys, and
// the last refill is skipped.  9216 B of static threadgroup memory.
//
// V is never staged.  Each lane already owns the (row, col), (row, col + 1) pair of an
// 8x8 tile, which is exactly the right-operand map of P@V, so it reads those two halves
// straight from device memory and widens them.  The probabilities therefore stay in
// fp32 and P@V is a float8x8 product; on this GPU that runs at the same rate as the
// half one, while rounding P once costs accuracy the tolerance does not have.
//
// The exponential shift is taken once, from the first 16 keys of the row block, and is
// kept for all 1024 keys: the loop then carries no row maximum and no rescale of the 32
// accumulator fragments, only one dependent multiply-accumulate chain.  A later key far
// above that shift can overflow fp32, so the row sums and the accumulator are bounded
// after the loop and a row block that breaks a bound is recomputed against its true row
// maximum (two passes over K read transposed from device, no rescaling).
// ---------------------------------------------------------------------------

constant constexpr uint  KLD    = 36u;             // halves per K^T row (72 B)
constant constexpr uint  KSLOT  = D * KLD;         // halves per 32-key K^T slot
constant constexpr uint  NKB    = S / BLOCK_N;     // 32-key blocks per row block
constant constexpr uint  N16    = S / 16u;         // 16-key steps per row block
constant constexpr float Q2E    = SM_SCALE * 1.4426950408889634f;
constant constexpr float SUMCAP = 1.0e30f;         // a partial row sum this large overflowed
constant constexpr float ACCAP  = 1.0e37f;         // fp32 tops out at 3.4e38; Inf and NaN both trip this

// This lane's two elements of an 8x8 tile.  ``thread_elements()`` returns the whole
// per-thread storage (vec<T, 64> on this SDK); the lane's own pair is the first two
// elements of it, which probe_lane.py checks against simdgroup_load/simdgroup_store.
inline thread float2& flash_pair(thread simdgroup_float8x8& m)
{
    return *(thread float2*)&m.thread_elements();
}

inline thread half2& flash_hpair(thread simdgroup_half8x8& m)
{
    return *(thread half2*)&m.thread_elements();
}

// This SIMD group transposes 16 consecutive keys into columns [col, col + 16) of a slot:
// source element (key r, dim c) lands at (dim c, key r).
inline void flash_stage(threadgroup half* dst, const device half* src, uint col)
{
#pragma unroll
    for (uint i = 0; i < 8u; ++i) {
        simdgroup_half8x8 lo, hi;
        simdgroup_load(lo, src + (ulong)i * 8u, D);
        simdgroup_load(hi, src + (ulong)(8u * D) + (ulong)i * 8u, D);
        simdgroup_store(lo, dst + (i * 8u) * KLD + col, KLD, ulong2(0, 0), true);
        simdgroup_store(hi, dst + (i * 8u) * KLD + col + 8u, KLD, ulong2(0, 0), true);
    }
}

// 16 keys against a staged K^T slot: QK scores of both 8-row tiles, fp32.
__attribute__((always_inline)) inline void flash_qk(
    thread simdgroup_half8x8 (&qa)[8], thread simdgroup_half8x8 (&qb)[8], threadgroup half* Ks,
    thread simdgroup_float8x8& s0, thread simdgroup_float8x8& s1,
    thread simdgroup_float8x8& t0, thread simdgroup_float8x8& t1)
{
    s0 = simdgroup_float8x8(0.0f);
    s1 = simdgroup_float8x8(0.0f);
    t0 = simdgroup_float8x8(0.0f);
    t1 = simdgroup_float8x8(0.0f);
#pragma unroll
    for (uint i = 0; i < 8u; ++i) {
        simdgroup_half8x8 k0, k1;
        simdgroup_load(k0, Ks + (i * 8u) * KLD, KLD);
        simdgroup_load(k1, Ks + (i * 8u) * KLD + 8u, KLD);
        simdgroup_multiply_accumulate(s0, qa[i], k0, s0);
        simdgroup_multiply_accumulate(s1, qa[i], k1, s1);
        simdgroup_multiply_accumulate(t0, qb[i], k0, t0);
        simdgroup_multiply_accumulate(t1, qb[i], k1, t1);
    }
}

// The same scores with K read transposed straight from device memory; used by the exact
// path, which must not depend on what happens to be staged.
__attribute__((always_inline)) inline void flash_qk_dev(
    thread simdgroup_half8x8 (&qa)[8], thread simdgroup_half8x8 (&qb)[8], const device half* kb,
    thread simdgroup_float8x8& s0, thread simdgroup_float8x8& s1,
    thread simdgroup_float8x8& t0, thread simdgroup_float8x8& t1)
{
    s0 = simdgroup_float8x8(0.0f);
    s1 = simdgroup_float8x8(0.0f);
    t0 = simdgroup_float8x8(0.0f);
    t1 = simdgroup_float8x8(0.0f);
#pragma unroll
    for (uint i = 0; i < 8u; ++i) {
        simdgroup_half8x8 k0, k1;
        simdgroup_load(k0, kb + (ulong)i * 8u, D, ulong2(0, 0), true);
        simdgroup_load(k1, kb + (ulong)(8u * D) + (ulong)i * 8u, D, ulong2(0, 0), true);
        simdgroup_multiply_accumulate(s0, qa[i], k0, s0);
        simdgroup_multiply_accumulate(s1, qa[i], k1, s1);
        simdgroup_multiply_accumulate(t0, qb[i], k0, t0);
        simdgroup_multiply_accumulate(t1, qb[i], k1, t1);
    }
}

// Row maximum of the two score fragments, in log2 units.  The four lanes that share a
// query row are l, l ^ 1, l ^ 8, l ^ 9 (the lane owns columns c and c + 1 of the tile).
__attribute__((always_inline)) inline float flash_rowmax2(
    thread simdgroup_float8x8& s0, thread simdgroup_float8x8& s1)
{
    const float2 e0 = flash_pair(s0);
    const float2 e1 = flash_pair(s1);
    float m = max(max(e0.x, e0.y), max(e1.x, e1.y));
    m = max(m, simd_shuffle_xor(m, 1u));
    m = max(m, simd_shuffle_xor(m, 8u));
    return m * Q2E;
}

// The two halves this lane owns of an 8x8 V tile, widened to the float right operand.
inline simdgroup_float8x8 flash_vtile(const device half* p)
{
    const half2 hw = *reinterpret_cast<const device half2*>(p);
    simdgroup_float8x8 f = simdgroup_float8x8(0.0f);
    thread float2& e = flash_pair(f);
    e.x = float(hw.x);
    e.y = float(hw.y);
    return f;
}

// Exponentiate the 16 scores of a step against the fixed shifts, add them into the
// lane-local row sums, and accumulate P@V for both row tiles.
__attribute__((always_inline)) inline void flash_pv(
    thread simdgroup_float8x8& s0, thread simdgroup_float8x8& s1,
    thread simdgroup_float8x8& t0, thread simdgroup_float8x8& t1, float ma, float mb,
    thread simdgroup_float8x8 (&acca)[8], thread simdgroup_float8x8 (&accb)[8],
    thread float& la, thread float& lb,
    const device half* vk, uint vrow, uint vcol)
{
    thread float2& a0 = flash_pair(s0);
    thread float2& a1 = flash_pair(s1);
    thread float2& b0 = flash_pair(t0);
    thread float2& b1 = flash_pair(t1);

    a0.x = exp2(fma(a0.x, Q2E, -ma));
    a0.y = exp2(fma(a0.y, Q2E, -ma));
    a1.x = exp2(fma(a1.x, Q2E, -ma));
    a1.y = exp2(fma(a1.y, Q2E, -ma));
    b0.x = exp2(fma(b0.x, Q2E, -mb));
    b0.y = exp2(fma(b0.y, Q2E, -mb));
    b1.x = exp2(fma(b1.x, Q2E, -mb));
    b1.y = exp2(fma(b1.y, Q2E, -mb));

    la += (a0.x + a0.y) + (a1.x + a1.y);
    lb += (b0.x + b0.y) + (b1.x + b1.y);

    const device half* vlan = vk + (ulong)vrow * D + vcol;
#pragma unroll
    for (uint c = 0; c < 8u; ++c) {
        simdgroup_float8x8 v0 = flash_vtile(vlan + c * 8u);
        simdgroup_multiply_accumulate(acca[c], s0, v0, acca[c]);
        simdgroup_multiply_accumulate(accb[c], t0, v0, accb[c]);
    }
    vlan += (ulong)(8u * D);
#pragma unroll
    for (uint c = 0; c < 8u; ++c) {
        simdgroup_float8x8 v1 = flash_vtile(vlan + c * 8u);
        simdgroup_multiply_accumulate(acca[c], s1, v1, acca[c]);
        simdgroup_multiply_accumulate(accb[c], t1, v1, accb[c]);
    }
}

kernel void flash_attention_kernel(
    device const half* q [[buffer(0)]],
    device const half* k [[buffer(1)]],
    device const half* v [[buffer(2)]],
    device half*       o [[buffer(3)]],
    uint2 tg  [[threadgroup_position_in_grid]],
    uint  tid [[thread_index_in_threadgroup]],
    uint  sg  [[simdgroup_index_in_threadgroup]])
{
    threadgroup half kts[2u * KSLOT] __attribute__((aligned(16)));

    const ulong bh   = (ulong)tg.y * (ulong)(S * D);
    const uint  r0   = tg.x * BLOCK_M + sg * 16u;    // first query row of this group
    const uint  lane = tid & 31u;

    const device half* qp = q + bh + (ulong)r0 * D;
    const device half* kp = k + bh;
    const device half* vp = v + bh;
    device half*       op = o + bh + (ulong)r0 * D;

    // this lane's two elements of every 8x8 tile: (vrow, vcol) and (vrow, vcol + 1)
    const uint quad = lane >> 2u;
    const uint vrow = (quad & 4u) + ((lane >> 1u) & 3u);
    const uint vcol = ((quad & 2u) << 1u) + ((lane & 1u) << 1u);

    simdgroup_half8x8 qa[8], qb[8];
#pragma unroll
    for (uint i = 0; i < 8u; ++i) {
        simdgroup_load(qa[i], qp + i * 8u, D);                      // rows 0..7 of the group
        simdgroup_load(qb[i], qp + (8u * D) + i * 8u, D);           // rows 8..15
    }

    simdgroup_float8x8 acca[8], accb[8];
#pragma unroll
    for (uint c = 0; c < 8u; ++c) {
        acca[c] = simdgroup_float8x8(0.0f);
        accb[c] = simdgroup_float8x8(0.0f);
    }

    float la = 0.0f, lb = 0.0f;
    float ma = 0.0f, mb = 0.0f;

    const uint kcol = sg * 16u;                     // this group transposes keys 16s..16s+15
    const device half* ksrc = kp + (ulong)kcol * D;

    flash_stage(kts, ksrc, kcol);                                           // block 0
    threadgroup_barrier(mem_flags::mem_threadgroup);
    flash_stage(kts + KSLOT, ksrc + (ulong)(BLOCK_N * D), kcol);            // block 1, ahead

    {   // block 0, keys 0..15: this step fixes the shift for the whole row block
        simdgroup_float8x8 s0, s1, t0, t1;
        flash_qk(qa, qb, kts, s0, s1, t0, t1);
        ma = flash_rowmax2(s0, s1);
        mb = flash_rowmax2(t0, t1);
        flash_pv(s0, s1, t0, t1, ma, mb, acca, accb, la, lb, vp, vrow, vcol);
    }
    {   // block 0, keys 16..31
        simdgroup_float8x8 s0, s1, t0, t1;
        flash_qk(qa, qb, kts + 16u, s0, s1, t0, t1);
        flash_pv(s0, s1, t0, t1, ma, mb, acca, accb, la, lb, vp + (ulong)(16u * D), vrow, vcol);
    }

    for (uint blk = 1u; blk < NKB; ++blk) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (blk + 1u < NKB) {
            const uint nxt = blk + 1u;
            flash_stage(kts + (nxt & 1u) * KSLOT, kp + (ulong)(nxt * BLOCK_N + kcol) * D, kcol);
        }
        threadgroup half* kc = kts + (blk & 1u) * KSLOT;
        const device half* vb = vp + (ulong)(blk * BLOCK_N) * D;

        simdgroup_float8x8 s0, s1, t0, t1;
        flash_qk(qa, qb, kc, s0, s1, t0, t1);
        flash_pv(s0, s1, t0, t1, ma, mb, acca, accb, la, lb, vb, vrow, vcol);
        flash_qk(qa, qb, kc + 16u, s0, s1, t0, t1);
        flash_pv(s0, s1, t0, t1, ma, mb, acca, accb, la, lb, vb + (ulong)(16u * D), vrow, vcol);
    }

    // Did the fixed shift hold?  Bounded row sums and a bounded accumulator; Inf and NaN
    // both fail these comparisons.
    bool ok = (la < SUMCAP) && (lb < SUMCAP);
#pragma unroll
    for (uint c = 0; c < 8u; ++c) {
        const float2 ea = flash_pair(acca[c]);
        const float2 eb = flash_pair(accb[c]);
        ok = ok && fabs(ea.x) < ACCAP && fabs(ea.y) < ACCAP
                && fabs(eb.x) < ACCAP && fabs(eb.y) < ACCAP;
    }

    if (simd_any(!ok)) {
        // Exact path: the true row maximum first, then one accumulation pass against it.
        // Nothing is rescaled, so a row block cannot lose a term to a shifted exponent.
#pragma unroll
        for (uint c = 0; c < 8u; ++c) {
            acca[c] = simdgroup_float8x8(0.0f);
            accb[c] = simdgroup_float8x8(0.0f);
        }
        ma = -INFINITY;
        mb = -INFINITY;
        for (uint nb = 0u; nb < N16; ++nb) {
            simdgroup_float8x8 s0, s1, t0, t1;
            flash_qk_dev(qa, qb, kp + (ulong)(nb * 16u) * D, s0, s1, t0, t1);
            ma = max(ma, flash_rowmax2(s0, s1));
            mb = max(mb, flash_rowmax2(t0, t1));
        }
        la = 0.0f;
        lb = 0.0f;
        for (uint nb = 0u; nb < N16; ++nb) {
            simdgroup_float8x8 s0, s1, t0, t1;
            const device half* kb = kp + (ulong)(nb * 16u) * D;
            flash_qk_dev(qa, qb, kb, s0, s1, t0, t1);
            flash_pv(s0, s1, t0, t1, ma, mb, acca, accb, la, lb,
                     vp + (ulong)(nb * 16u) * D, vrow, vcol);
        }
    }

    // One reduction of the lane-local row sums, one division per row, one rounding.
    float La = la + simd_shuffle_xor(la, 1u);
    La += simd_shuffle_xor(La, 8u);
    float Lb = lb + simd_shuffle_xor(lb, 1u);
    Lb += simd_shuffle_xor(Lb, 8u);
    const float ua = 1.0f / La;
    const float ub = 1.0f / Lb;

#pragma unroll
    for (uint c = 0; c < 8u; ++c) {
        const float2 ea = flash_pair(acca[c]);
        simdgroup_half8x8 hha = simdgroup_half8x8((half)0.0f);
        thread half2& ha = flash_hpair(hha);
        ha.x = half(ea.x * ua);
        ha.y = half(ea.y * ua);
        simdgroup_store(hha, op + c * 8u, D);

        const float2 eb = flash_pair(accb[c]);
        simdgroup_half8x8 hhb = simdgroup_half8x8((half)0.0f);
        thread half2& hb = flash_hpair(hhb);
        hb.x = half(eb.x * ub);
        hb.y = half(eb.y * ub);
        simdgroup_store(hhb, op + (8u * D) + c * 8u, D);
    }
}
