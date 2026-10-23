/* taic-launch
launch flash_attention_kernel grid=32,16,1 threads=64,1,1
*/
// FlashAttention forward (non-causal), fp16 in / fp16 out, one (batch, head) x 32 query rows per
// threadgroup.  Two SIMD groups; group s owns query rows 16*s .. 16*s+15 (two 8-row tiles) and all
// D = 64 output columns.
//
// K^T lives in threadgroup memory: two 32-key slots whose rows are the 64 head dims, stride 36
// halves (72 B) so the eight row reads of an 8x8 tile spread over eight different bank pairs.  A
// key block is transposed once by the group that stages it (fragment store with transpose), so every
// matrix operand tile in the main loop is a plain row-major threadgroup load.
//
// V is not staged at all: each lane gathers its own (row, col), (row, col+1) pair of the 8x8 V tile
// straight from device memory into a float fragment, which is exactly the right-operand map of the
// P@V product.  Probabilities therefore stay fp32 and are never rounded; scores, row statistics and
// the accumulator are fp32 and the output is rounded once, in the epilogue.
//
// One threadgroup barrier per 32 keys.  A slot is refilled for the block after the next between the
// barrier and the current block's 128 matrix instructions, so the staging latency hides behind the
// matrix stream of the other resident threadgroups.
//
// The exponential shift is taken once, from the first 16 keys of the row block, and kept for all
// 1024 keys.  Recomputing the row maximum every 32 keys puts a read-modify-write of the 32
// accumulator registers on the matrix-unit dependency chain; with a fixed shift the whole loop is one
// chain of multiply-accumulates.  A later key far above the first block's maximum would overflow, so
// the tail bounds the row sums and the accumulator and a row block that breaks the bound is
// recomputed with the exact online-maximum recurrence (the only user of that slower path).
constant constexpr uint  KRW  = 36u;                       // K^T row stride, halves
constant constexpr uint  KSL  = D * KRW;                   // halves per 32-key K^T slot (4608 B)
constant constexpr uint  NSB  = S / 32u;                   // 32-key blocks
constant constexpr uint  NB16 = S / 16u;                   // 16-key steps (exact path)
constant constexpr float QSC  = SM_SCALE * 1.4426950408889634f;
constant constexpr float LMAX = 1.0e30f;                   // a partial row sum this large means overflow
constant constexpr float AMAX = 1.0e37f;                   // fp32 tops out at 3.4e38; Inf and NaN both fail

// This SIMD group writes 16 key rows of K into the 16 columns [col, col+16) of a K^T slot, transposed:
// element (key r, dim c) of the source tile lands at (dim c, key r).
inline void stage_keys(threadgroup half* dst, const device half* src, uint col)
{
#pragma unroll
    for (uint i = 0; i < 8u; ++i) {
        simdgroup_half8x8 lo, hi;
        simdgroup_load(lo, src + (ulong)i * 8u, D);
        simdgroup_load(hi, src + (ulong)(8u * D) + (ulong)i * 8u, D);
        simdgroup_store(lo, dst + (i * 8u) * KRW + col, KRW, ulong2(0, 0), true);
        simdgroup_store(hi, dst + (i * 8u) * KRW + col + 8u, KRW, ulong2(0, 0), true);
    }
}

// One 8x8 V tile read by this lane alone: (row, col) and (row, col + 1), the accumulator map and the
// map of the right operand of a P@V product.  ``p`` already points at this lane's pair.
inline simdgroup_float8x8 gather_v(const device half* p)
{
    simdgroup_float8x8 f;
    *(thread float2*)&f.thread_elements() = float2(*(const device half2*)p);
    return f;
}

// 16 keys against the fixed shifts ma / mb: scores, exp2, P back into the score fragments, P@V.
__attribute__((always_inline)) inline void keys16(
    thread simdgroup_half8x8 (&qa)[8], thread simdgroup_half8x8 (&qb)[8],
    thread simdgroup_float8x8 (&acca)[8], thread simdgroup_float8x8 (&accb)[8],
    thread float& la, thread float& lb, float ma, float mb,
    threadgroup half* Ks, const device half* vb, uint vrow, uint vcol)
{
    simdgroup_float8x8 sa0 = simdgroup_float8x8(0.0f), sa1 = simdgroup_float8x8(0.0f);
    simdgroup_float8x8 sb0 = simdgroup_float8x8(0.0f), sb1 = simdgroup_float8x8(0.0f);
#pragma unroll
    for (uint i = 0; i < 8u; ++i) {
        simdgroup_half8x8 k0, k1;
        simdgroup_load(k0, Ks + (i * 8u) * KRW, KRW);
        simdgroup_load(k1, Ks + (i * 8u) * KRW + 8u, KRW);
        simdgroup_multiply_accumulate(sa0, qa[i], k0, sa0);
        simdgroup_multiply_accumulate(sa1, qa[i], k1, sa1);
        simdgroup_multiply_accumulate(sb0, qb[i], k0, sb0);
        simdgroup_multiply_accumulate(sb1, qb[i], k1, sb1);
    }

    const float2 e0 = *(thread float2*)&sa0.thread_elements();
    const float2 e1 = *(thread float2*)&sa1.thread_elements();
    const float2 f0 = *(thread float2*)&sb0.thread_elements();
    const float2 f1 = *(thread float2*)&sb1.thread_elements();

    float pa0 = exp2(fma(e0.x, QSC, -ma)), pa1 = exp2(fma(e0.y, QSC, -ma));
    float pa2 = exp2(fma(e1.x, QSC, -ma)), pa3 = exp2(fma(e1.y, QSC, -ma));
    float pb0 = exp2(fma(f0.x, QSC, -mb)), pb1 = exp2(fma(f0.y, QSC, -mb));
    float pb2 = exp2(fma(f1.x, QSC, -mb)), pb3 = exp2(fma(f1.y, QSC, -mb));
    la += (pa0 + pa1) + (pa2 + pa3);
    lb += (pb0 + pb1) + (pb2 + pb3);

    *(thread float2*)&sa0.thread_elements() = float2(pa0, pa1);
    *(thread float2*)&sa1.thread_elements() = float2(pa2, pa3);
    *(thread float2*)&sb0.thread_elements() = float2(pb0, pb1);
    *(thread float2*)&sb1.thread_elements() = float2(pb2, pb3);

    // keys 0..7 of this step
    const device half* vl = vb + (ulong)vrow * D + vcol;
#pragma unroll
    for (uint c = 0; c < 8u; ++c) {
        simdgroup_float8x8 vt = gather_v(vl + c * 8u);
        simdgroup_multiply_accumulate(acca[c], sa0, vt, acca[c]);
        simdgroup_multiply_accumulate(accb[c], sb0, vt, accb[c]);
    }
    // keys 8..15
    vl += (ulong)(8u * D);
#pragma unroll
    for (uint c = 0; c < 8u; ++c) {
        simdgroup_float8x8 vt = gather_v(vl + c * 8u);
        simdgroup_multiply_accumulate(acca[c], sa1, vt, acca[c]);
        simdgroup_multiply_accumulate(accb[c], sb1, vt, accb[c]);
    }
}

// The first 16 keys of the row block: they define the fixed shift, so nothing is rescaled here.
__attribute__((always_inline)) inline void first16(
    thread simdgroup_half8x8 (&qa)[8], thread simdgroup_half8x8 (&qb)[8],
    thread simdgroup_float8x8 (&acca)[8], thread simdgroup_float8x8 (&accb)[8],
    thread float& la, thread float& lb, thread float& ma, thread float& mb,
    threadgroup half* Ks, const device half* vb, uint vrow, uint vcol)
{
    simdgroup_float8x8 sa0 = simdgroup_float8x8(0.0f), sa1 = simdgroup_float8x8(0.0f);
    simdgroup_float8x8 sb0 = simdgroup_float8x8(0.0f), sb1 = simdgroup_float8x8(0.0f);
#pragma unroll
    for (uint i = 0; i < 8u; ++i) {
        simdgroup_half8x8 k0, k1;
        simdgroup_load(k0, Ks + (i * 8u) * KRW, KRW);
        simdgroup_load(k1, Ks + (i * 8u) * KRW + 8u, KRW);
        simdgroup_multiply_accumulate(sa0, qa[i], k0, sa0);
        simdgroup_multiply_accumulate(sa1, qa[i], k1, sa1);
        simdgroup_multiply_accumulate(sb0, qb[i], k0, sb0);
        simdgroup_multiply_accumulate(sb1, qb[i], k1, sb1);
    }

    const float2 e0 = *(thread float2*)&sa0.thread_elements();
    const float2 e1 = *(thread float2*)&sa1.thread_elements();
    const float2 f0 = *(thread float2*)&sb0.thread_elements();
    const float2 f1 = *(thread float2*)&sb1.thread_elements();
    float ta[4] = { e0.x, e0.y, e1.x, e1.y };
    float tb[4] = { f0.x, f0.y, f1.x, f1.y };
    float pa[4], pb[4];

    // the four lanes of a query row differ in bit 0 and bit 3
    float mx = max(max(ta[0], ta[1]), max(ta[2], ta[3]));
    mx = max(mx, simd_shuffle_xor(mx, 1u));
    mx = max(mx, simd_shuffle_xor(mx, 8u));
    ma = mx * QSC;
    float ps = 0.0f;
#pragma unroll
    for (uint j = 0; j < 4u; ++j) {
        pa[j] = exp2(fma(ta[j], QSC, -ma));
        ps += pa[j];
    }
    la = ps;

    mx = max(max(tb[0], tb[1]), max(tb[2], tb[3]));
    mx = max(mx, simd_shuffle_xor(mx, 1u));
    mx = max(mx, simd_shuffle_xor(mx, 8u));
    mb = mx * QSC;
    ps = 0.0f;
#pragma unroll
    for (uint j = 0; j < 4u; ++j) {
        pb[j] = exp2(fma(tb[j], QSC, -mb));
        ps += pb[j];
    }
    lb = ps;

    *(thread float2*)&sa0.thread_elements() = float2(pa[0], pa[1]);
    *(thread float2*)&sa1.thread_elements() = float2(pa[2], pa[3]);
    *(thread float2*)&sb0.thread_elements() = float2(pb[0], pb[1]);
    *(thread float2*)&sb1.thread_elements() = float2(pb[2], pb[3]);

    const device half* vl = vb + (ulong)vrow * D + vcol;
#pragma unroll
    for (uint c = 0; c < 8u; ++c) {
        simdgroup_float8x8 vt = gather_v(vl + c * 8u);
        simdgroup_multiply_accumulate(acca[c], sa0, vt, acca[c]);
        simdgroup_multiply_accumulate(accb[c], sb0, vt, accb[c]);
    }
    vl += (ulong)(8u * D);
#pragma unroll
    for (uint c = 0; c < 8u; ++c) {
        simdgroup_float8x8 vt = gather_v(vl + c * 8u);
        simdgroup_multiply_accumulate(acca[c], sa1, vt, acca[c]);
        simdgroup_multiply_accumulate(accb[c], sb1, vt, accb[c]);
    }
}

kernel void flash_attention_kernel(
    device const half* q [[buffer(0)]],
    device const half* k [[buffer(1)]],
    device const half* v [[buffer(2)]],
    device half*       o [[buffer(3)]],
    uint2 tg  [[threadgroup_position_in_grid]],
    uint  lid [[thread_index_in_threadgroup]],
    uint  sg  [[simdgroup_index_in_threadgroup]])
{
    threadgroup half Kt[2u * KSL] __attribute__((aligned(16)));

    const ulong bias = (ulong)tg.y * (ulong)(S * D);
    const uint  r0   = tg.x * 32u + sg * 16u;

    const device half* qp = q + bias + (ulong)r0 * D;
    const device half* kp = k + bias;
    const device half* vp = v + bias;
    device half*       op = o + bias + (ulong)r0 * D;

    // where this lane sits in an 8x8 tile: rows 0..7, columns 0, 2, 4, 6 (two elements each)
    const uint lane = lid & 31u;
    const uint quad = lane >> 2u;
    const uint vrow = (quad & 4u) | ((lane >> 1u) & 3u);
    const uint vcol = ((quad & 2u) << 1u) | ((lane & 1u) << 1u);

    simdgroup_half8x8 qa[8], qb[8];
#pragma unroll
    for (uint i = 0; i < 8u; ++i) {
        simdgroup_load(qa[i], qp + i * 8u, D);
        simdgroup_load(qb[i], qp + (8u * D) + i * 8u, D);
    }

    simdgroup_float8x8 acca[8], accb[8];
#pragma unroll
    for (uint c = 0; c < 8u; ++c) {
        acca[c] = simdgroup_float8x8(0.0f);
        accb[c] = simdgroup_float8x8(0.0f);
    }

    float ma = -INFINITY, mb = -INFINITY;
    float la = 0.0f, lb = 0.0f;

    const uint kcol = sg * 16u;                 // this group stages keys 16*sg .. 16*sg+15
    const device half* ksrc = kp + (ulong)kcol * D;

    stage_keys(Kt, ksrc, kcol);

    // ---- first block, peeled: its first 16 keys fix the shift ----
    {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        stage_keys(Kt + KSL, ksrc + (ulong)(32u * D), kcol);
        first16(qa, qb, acca, accb, la, lb, ma, mb, Kt, vp, vrow, vcol);
        keys16(qa, qb, acca, accb, la, lb, ma, mb, Kt + 16u, vp + (ulong)(16u * D), vrow, vcol);
    }

    // ---- remaining blocks: one barrier and one staging per 32 keys ----
    for (uint sb = 1u; sb < NSB; ++sb) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const uint nxt = (sb + 1u < NSB) ? (sb + 1u) : (NSB - 1u);   // the last refill is never read
        stage_keys(Kt + ((sb + 1u) & 1u) * KSL, kp + (ulong)(nxt * 32u + kcol) * D, kcol);

        threadgroup half* Kc = Kt + (sb & 1u) * KSL;
        const device half* vb = vp + (ulong)sb * (32u * D);
        keys16(qa, qb, acca, accb, la, lb, ma, mb, Kc, vb, vrow, vcol);
        keys16(qa, qb, acca, accb, la, lb, ma, mb, Kc + 16u, vb + (ulong)(16u * D), vrow, vcol);
    }

    // ---- did the fixed shift hold?  bounded partial sums, bounded accumulator ----
    bool ok = (la < LMAX) && (lb < LMAX);
#pragma unroll
    for (uint c = 0; c < 8u; ++c) {
        const float2 e = *(thread float2*)&acca[c].thread_elements();
        const float2 f = *(thread float2*)&accb[c].thread_elements();
        ok = ok && fabs(e.x) < AMAX && fabs(e.y) < AMAX && fabs(f.x) < AMAX && fabs(f.y) < AMAX;
    }

    if (simd_any(!ok)) {
        // ---- exact online-maximum recurrence, 16 keys per step, K read straight from device ----
#pragma unroll
        for (uint c = 0; c < 8u; ++c) {
            acca[c] = simdgroup_float8x8(0.0f);
            accb[c] = simdgroup_float8x8(0.0f);
        }
        ma = -INFINITY; mb = -INFINITY;
        la = 0.0f; lb = 0.0f;

        for (uint blk = 0; blk < NB16; ++blk) {
            const device half* kb = kp + (ulong)(blk * 16u) * D;
            simdgroup_float8x8 sa0 = simdgroup_float8x8(0.0f), sa1 = simdgroup_float8x8(0.0f);
            simdgroup_float8x8 sb0 = simdgroup_float8x8(0.0f), sb1 = simdgroup_float8x8(0.0f);
#pragma unroll
            for (uint i = 0; i < 8u; ++i) {
                simdgroup_half8x8 k0, k1;
                simdgroup_load(k0, kb + (ulong)i * 8u, D, ulong2(0, 0), true);
                simdgroup_load(k1, kb + (ulong)(8u * D) + (ulong)i * 8u, D, ulong2(0, 0), true);
                simdgroup_multiply_accumulate(sa0, qa[i], k0, sa0);
                simdgroup_multiply_accumulate(sa1, qa[i], k1, sa1);
                simdgroup_multiply_accumulate(sb0, qb[i], k0, sb0);
                simdgroup_multiply_accumulate(sb1, qb[i], k1, sb1);
            }

            const float2 e0 = *(thread float2*)&sa0.thread_elements();
            const float2 e1 = *(thread float2*)&sa1.thread_elements();
            const float2 f0 = *(thread float2*)&sb0.thread_elements();
            const float2 f1 = *(thread float2*)&sb1.thread_elements();
            float ta[4] = { e0.x, e0.y, e1.x, e1.y };
            float tb[4] = { f0.x, f0.y, f1.x, f1.y };
            float pa[4], pb[4], ps;

            float mx = max(max(ta[0], ta[1]), max(ta[2], ta[3]));
            mx = max(mx, simd_shuffle_xor(mx, 1u));
            mx = max(mx, simd_shuffle_xor(mx, 8u));
            const float na = max(ma, mx * QSC);
            const float ala = exp2(ma - na);
            ps = 0.0f;
#pragma unroll
            for (uint j = 0; j < 4u; ++j) {
                pa[j] = exp2(fma(ta[j], QSC, -na));
                ps += pa[j];
            }
            la = la * ala + ps;
            ma = na;
#pragma unroll
            for (uint c = 0; c < 8u; ++c) {
                float2 t = *(thread float2*)&acca[c].thread_elements();
                t *= ala;
                *(thread float2*)&acca[c].thread_elements() = t;
            }

            mx = max(max(tb[0], tb[1]), max(tb[2], tb[3]));
            mx = max(mx, simd_shuffle_xor(mx, 1u));
            mx = max(mx, simd_shuffle_xor(mx, 8u));
            const float nb = max(mb, mx * QSC);
            const float alb = exp2(mb - nb);
            ps = 0.0f;
#pragma unroll
            for (uint j = 0; j < 4u; ++j) {
                pb[j] = exp2(fma(tb[j], QSC, -nb));
                ps += pb[j];
            }
            lb = lb * alb + ps;
            mb = nb;
#pragma unroll
            for (uint c = 0; c < 8u; ++c) {
                float2 t = *(thread float2*)&accb[c].thread_elements();
                t *= alb;
                *(thread float2*)&accb[c].thread_elements() = t;
            }

            *(thread float2*)&sa0.thread_elements() = float2(pa[0], pa[1]);
            *(thread float2*)&sa1.thread_elements() = float2(pa[2], pa[3]);
            *(thread float2*)&sb0.thread_elements() = float2(pb[0], pb[1]);
            *(thread float2*)&sb1.thread_elements() = float2(pb[2], pb[3]);

            const device half* vl = vp + (ulong)(blk * 16u + vrow) * D + vcol;
#pragma unroll
            for (uint c = 0; c < 8u; ++c) {
                simdgroup_float8x8 vt = gather_v(vl + c * 8u);
                simdgroup_multiply_accumulate(acca[c], sa0, vt, acca[c]);
                simdgroup_multiply_accumulate(accb[c], sb0, vt, accb[c]);
            }
            vl += (ulong)(8u * D);
#pragma unroll
            for (uint c = 0; c < 8u; ++c) {
                simdgroup_float8x8 vt = gather_v(vl + c * 8u);
                simdgroup_multiply_accumulate(acca[c], sa1, vt, acca[c]);
                simdgroup_multiply_accumulate(accb[c], sb1, vt, accb[c]);
            }
        }
    }

    // ---- epilogue: one reduction, one division per row, one rounding ----
    float La = la + simd_shuffle_xor(la, 1u);
    La += simd_shuffle_xor(La, 8u);
    float Lb = lb + simd_shuffle_xor(lb, 1u);
    Lb += simd_shuffle_xor(Lb, 8u);
    const float ra = 1.0f / La;
    const float rb = 1.0f / Lb;

#pragma unroll
    for (uint c = 0; c < 8u; ++c) {
        const float2 e = *(thread float2*)&acca[c].thread_elements();
        simdgroup_half8x8 oh;
        *(thread half2*)&oh.thread_elements() = half2((half)(e.x * ra), (half)(e.y * ra));
        simdgroup_store(oh, op + c * 8u, D);

        const float2 f = *(thread float2*)&accb[c].thread_elements();
        simdgroup_half8x8 og;
        *(thread half2*)&og.thread_elements() = half2((half)(f.x * rb), (half)(f.y * rb));
        simdgroup_store(og, op + (8u * D) + c * 8u, D);
    }
}
