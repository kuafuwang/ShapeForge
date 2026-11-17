// conv2d, stride 1, no padding, no bias, NHWC x, [OC,KH,KW,C] w, fp16 in/out.
// N=4 H=W=18 C=32 OC=64 -> 16x16 output = 64 disjoint 4x4 tiles.
//
// Winograd F(4x4,3x3):  y = A^T [ (G w G^T) . (B^T x B) ] A, 36 transform positions.
// MMA operands are fp32 (half-rounded V misses the tolerance: the fold's intermediate
// terms are ~1e3x the result).  Only the U value handed to the B fragment is rounded.
//
// Shape: 16 threadgroups x 128 threads.  One threadgroup owns 8 tiles x 32 output
// channels; each SIMD group owns an 8-wide channel fragment, so a lane's two fragment
// elements are the same (tile, oc) pair in all 36 positions and the fold stays
// lane-local.  The 36 positions stream as two strips of three p rows; a strip's V
// (18 positions x 32 channels x 8 tiles fp32 = 18 432 B) is staged by all 128 threads
// and shared by the four SIMD groups.
//
// Two work reductions against the direct form of the inner loop:
//  * U: the column transform G w of one p row is computed once per p row and shared by
//    the six q positions (6 MMAs per channel chunk) instead of being rebuilt for every
//    MMA, so a p row costs ~12 short transforms plus 48 two-term row transforms.
//  * fold: two stage.  Per p row and lane element the six M values are folded along q
//    into four t values, then folded along p into the 4x4 output.  That is ~480 fma
//    per lane for the whole problem instead of one product per (position, output) pair.
//
// Non-finite x or w: a Winograd transform would smear one Inf across a whole tile while
// the contract's reference sums products tap by tap, so every thread checks the x patch
// and the w taps it reads and the threadgroup falls back to a plain fp32 tap-by-tap sum
// over its own slice only (no cross-threadgroup writes).

#include <metal_stdlib>
#include <metal_simdgroup>
#include <metal_simdgroup_matrix>
using namespace metal;

// N, H, W, C, OC, OH, OW, KH, KW come from the contract header.
constant constexpr uint TB = 8u;                // tiles per threadgroup
constant constexpr uint OCG = 32u;              // output channels per threadgroup
constant constexpr uint NGO = 2u;               // channel groups (64 / 32)
constant constexpr uint THR = 128u;             // 4 SIMD groups
constant constexpr uint GRID = 8u * NGO;        // 16 threadgroups
constant constexpr uint SVPOS = 32u * TB;       // 256 floats per position: [channel][tile]

// A^T of F(4x4,3x3)
constant constexpr float AT[4][6] = {
    {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 0.0f},
    {0.0f, 1.0f, -1.0f, 2.0f, -2.0f, 0.0f},
    {0.0f, 1.0f, 1.0f, 4.0f, 4.0f, 0.0f},
    {0.0f, 1.0f, -1.0f, 8.0f, -8.0f, 1.0f}};

static inline bool nfh(const half v) { return (as_type<ushort>(v) & 0x7c00u) == 0x7c00u; }

// B^T row P applied to six values (one axis of the input transform).
template <uint P>
static inline float bt6(float x0, float x1, float x2, float x3, float x4, float x5)
{
    float r = 0.0f;
    if (P == 0u)      r = 4.0f * x0 - 5.0f * x2 + x4;
    else if (P == 1u) r = -4.0f * (x1 + x2) + x3 + x4;
    else if (P == 2u) r = 4.0f * x1 - 4.0f * x2 - x3 + x4;
    else if (P == 3u) r = -2.0f * x1 - x2 + 2.0f * x3 + x4;
    else if (P == 4u) r = 2.0f * x1 - x2 - 2.0f * x3 + x4;
    else              r = 4.0f * x1 - 5.0f * x3 + x5;
    return r;
}

// G row P applied to three values (weight transform).
template <uint P>
static inline float gr3(float a, float b, float c)
{
    float r = 0.0f;
    if (P == 0u)      r = 0.25f * a;
    else if (P == 1u) r = -(a + b + c) * (1.0f / 6.0f);
    else if (P == 2u) r = (-a + b - c) * (1.0f / 6.0f);
    else if (P == 3u) r = (a + 2.0f * b + 4.0f * c) * (1.0f / 24.0f);
    else if (P == 4u) r = (a - 2.0f * b + 4.0f * c) * (1.0f / 24.0f);
    else              r = c;
    return r;
}

// fp32 tap-by-tap sum over this threadgroup's own outputs.  Its accumulation order
// (tap outer, channel inner) is the reference's.
static inline void direct_slice(device const half* x, device const half* w, device half* y,
                                const uint tgrp, const uint ocbase, const uint tid)
{
    for (uint e = tid; e < TB * 16u * OCG; e += THR) {
        const uint ocl = e & 31u;
        const uint rr = e >> 5u;
        const uint tl = rr & 7u;
        const uint pos = rr >> 3u;
        const uint oi = pos >> 2u;
        const uint oj = pos & 3u;
        const uint gt = tgrp * TB + tl;
        const uint img = gt >> 4u;
        const uint rem = gt & 15u;
        const uint tr = rem >> 2u;
        const uint tc = rem & 3u;
        const uint oh = 4u * tr + oi;
        const uint ow = 4u * tc + oj;
        const uint oc = ocbase + ocl;
        device const half* wrow = w + oc * (9u * C);
        device const half* xb = x + (((img * H + oh) * W) + ow) * C;
        float acc = 0.0f;
        for (uint tap = 0; tap < 9u; ++tap) {
            const uint kh = tap / 3u;
            const uint kw = tap - kh * 3u;
            device const half* xr = xb + (kh * W + kw) * C;
            device const half* wr = wrow + tap * C;
            for (uint c = 0; c < C; ++c) acc = fma(float(xr[c]), float(wr[c]), acc);
        }
        y[(((img * OH + oh) * OW) + ow) * OC + oc] = half(acc);
    }
}

#define BT6(P, x0, x1, x2, x3, x4, x5) bt6<(P)>(x0, x1, x2, x3, x4, x5)
#define GR3(P, a, b, c) gr3<(P)>(a, b, c)

// Input transform for one (tile, channel) pair and one p row: B^T row P along p for
// each of the six patch columns, then B^T along q.  Writes six V values.
#define STAGE_ROW(P, PL)                                                                       \
    {                                                                                          \
        const float pp0 = BT6(P, float(px[0]), float(px[6]), float(px[12]),                      \
                              float(px[18]), float(px[24]), float(px[30]));                      \
        const float pp1 = BT6(P, float(px[1]), float(px[7]), float(px[13]),                      \
                              float(px[19]), float(px[25]), float(px[31]));                      \
        const float pp2 = BT6(P, float(px[2]), float(px[8]), float(px[14]),                      \
                              float(px[20]), float(px[26]), float(px[32]));                      \
        const float pp3 = BT6(P, float(px[3]), float(px[9]), float(px[15]),                      \
                              float(px[21]), float(px[27]), float(px[33]));                      \
        const float pp4 = BT6(P, float(px[4]), float(px[10]), float(px[16]),                     \
                              float(px[22]), float(px[28]), float(px[34]));                      \
        const float pp5 = BT6(P, float(px[5]), float(px[11]), float(px[17]),                     \
                              float(px[23]), float(px[29]), float(px[35]));                      \
        sV[((PL) * 6u + 0u) * SVPOS + sb] = BT6(0u, pp0, pp1, pp2, pp3, pp4, pp5);              \
        sV[((PL) * 6u + 1u) * SVPOS + sb] = BT6(1u, pp0, pp1, pp2, pp3, pp4, pp5);              \
        sV[((PL) * 6u + 2u) * SVPOS + sb] = BT6(2u, pp0, pp1, pp2, pp3, pp4, pp5);              \
        sV[((PL) * 6u + 3u) * SVPOS + sb] = BT6(3u, pp0, pp1, pp2, pp3, pp4, pp5);              \
        sV[((PL) * 6u + 4u) * SVPOS + sb] = BT6(4u, pp0, pp1, pp2, pp3, pp4, pp5);              \
        sV[((PL) * 6u + 5u) * SVPOS + sb] = BT6(5u, pp0, pp1, pp2, pp3, pp4, pp5);              \
    }

// q fold: t[oj] += A^T[oj][Q] * M, per lane element.
#define FOLD_Q(Q, M0, M1)                                                                      \
    {                                                                                          \
        t0[0] = fma(AT[0][Q], (M0), t0[0]);                                                     \
        t1[0] = fma(AT[0][Q], (M1), t1[0]);                                                     \
        t0[1] = fma(AT[1][Q], (M0), t0[1]);                                                     \
        t1[1] = fma(AT[1][Q], (M1), t1[1]);                                                     \
        t0[2] = fma(AT[2][Q], (M0), t0[2]);                                                     \
        t1[2] = fma(AT[2][Q], (M1), t1[2]);                                                     \
        t0[3] = fma(AT[3][Q], (M0), t0[3]);                                                     \
        t1[3] = fma(AT[3][Q], (M1), t1[3]);                                                     \
    }

// p fold: y[oi][oj] += A^T[oi][P] * t[oj], per lane element.
#define FOLD_P(S, PL)                                                                          \
    {                                                                                          \
        y0[0u] = fma(AT[0][(S) * 3u + (PL)], t0[0], y0[0u]);                                    \
        y1[0u] = fma(AT[0][(S) * 3u + (PL)], t1[0], y1[0u]);                                    \
        y0[1u] = fma(AT[0][(S) * 3u + (PL)], t0[1], y0[1u]);                                    \
        y1[1u] = fma(AT[0][(S) * 3u + (PL)], t1[1], y1[1u]);                                    \
        y0[2u] = fma(AT[0][(S) * 3u + (PL)], t0[2], y0[2u]);                                    \
        y1[2u] = fma(AT[0][(S) * 3u + (PL)], t1[2], y1[2u]);                                    \
        y0[3u] = fma(AT[0][(S) * 3u + (PL)], t0[3], y0[3u]);                                    \
        y1[3u] = fma(AT[0][(S) * 3u + (PL)], t1[3], y1[3u]);                                    \
        y0[4u] = fma(AT[1][(S) * 3u + (PL)], t0[0], y0[4u]);                                    \
        y1[4u] = fma(AT[1][(S) * 3u + (PL)], t1[0], y1[4u]);                                    \
        y0[5u] = fma(AT[1][(S) * 3u + (PL)], t0[1], y0[5u]);                                    \
        y1[5u] = fma(AT[1][(S) * 3u + (PL)], t1[1], y1[5u]);                                    \
        y0[6u] = fma(AT[1][(S) * 3u + (PL)], t0[2], y0[6u]);                                    \
        y1[6u] = fma(AT[1][(S) * 3u + (PL)], t1[2], y1[6u]);                                    \
        y0[7u] = fma(AT[1][(S) * 3u + (PL)], t0[3], y0[7u]);                                    \
        y1[7u] = fma(AT[1][(S) * 3u + (PL)], t1[3], y1[7u]);                                    \
        y0[8u] = fma(AT[2][(S) * 3u + (PL)], t0[0], y0[8u]);                                    \
        y1[8u] = fma(AT[2][(S) * 3u + (PL)], t1[0], y1[8u]);                                    \
        y0[9u] = fma(AT[2][(S) * 3u + (PL)], t0[1], y0[9u]);                                    \
        y1[9u] = fma(AT[2][(S) * 3u + (PL)], t1[1], y1[9u]);                                    \
        y0[10u] = fma(AT[2][(S) * 3u + (PL)], t0[2], y0[10u]);                                  \
        y1[10u] = fma(AT[2][(S) * 3u + (PL)], t1[2], y1[10u]);                                  \
        y0[11u] = fma(AT[2][(S) * 3u + (PL)], t0[3], y0[11u]);                                  \
        y1[11u] = fma(AT[2][(S) * 3u + (PL)], t1[3], y1[11u]);                                  \
        y0[12u] = fma(AT[3][(S) * 3u + (PL)], t0[0], y0[12u]);                                  \
        y1[12u] = fma(AT[3][(S) * 3u + (PL)], t1[0], y1[12u]);                                  \
        y0[13u] = fma(AT[3][(S) * 3u + (PL)], t0[1], y0[13u]);                                  \
        y1[13u] = fma(AT[3][(S) * 3u + (PL)], t1[1], y1[13u]);                                  \
        y0[14u] = fma(AT[3][(S) * 3u + (PL)], t0[2], y0[14u]);                                  \
        y1[14u] = fma(AT[3][(S) * 3u + (PL)], t1[2], y1[14u]);                                  \
        y0[15u] = fma(AT[3][(S) * 3u + (PL)], t0[3], y0[15u]);                                  \
        y1[15u] = fma(AT[3][(S) * 3u + (PL)], t1[3], y1[15u]);                                  \
    }

// MMA for channel chunk KB at position Q into the accumulator of that position.
#define MMA_K(ACC, PL, KB, Q)                                                                             \
    {                                                                                                     \
        simdgroup_float8x8 af;                                                                            \
        auto ae = af.thread_elements();                                                                   \
        ae[0] = sV[((PL) * 6u + (Q)) * SVPOS + ((KB) * 8u + fcol) * TB + frow];                           \
        ae[1] = sV[((PL) * 6u + (Q)) * SVPOS + ((KB) * 8u + fcol + 1u) * TB + frow];                      \
        af.thread_elements() = ae;                                                                        \
        simdgroup_float8x8 bf;                                                                            \
        auto be = bf.thread_elements();                                                                   \
        be[0] = GR3(Q, w1[(KB) * 6u + 0u], w1[(KB) * 6u + 1u], w1[(KB) * 6u + 2u]);                       \
        be[1] = GR3(Q, w1[(KB) * 6u + 3u], w1[(KB) * 6u + 4u], w1[(KB) * 6u + 5u]);                       \
        bf.thread_elements() = be;                                                                        \
        simdgroup_multiply_accumulate(ACC[Q], af, bf, ACC[Q]);                                            \
    }                                                                                                     \

// One channel chunk: the six positions, six independent accumulator chains.
#define MMA_KB(ACC, PL, KB)                                                                               \
    {                                                                                                     \
        MMA_K(ACC, PL, KB, 0u)                                                                            \
        MMA_K(ACC, PL, KB, 1u)                                                                            \
        MMA_K(ACC, PL, KB, 2u)                                                                            \
        MMA_K(ACC, PL, KB, 3u)                                                                            \
        MMA_K(ACC, PL, KB, 4u)                                                                            \
        MMA_K(ACC, PL, KB, 5u)                                                                            \
    }                                                                                                     \

// Issue one p row: G w row transform shared by the six q positions, then 24 MMAs.
#define MMA_ISSUE(ACC, S, PL)                                                                             \
    {                                                                                                     \
        float w1[24];                                                                                     \
        w1[0] = GR3((S) * 3u + (PL), float(wt[0]), float(wt[3]), float(wt[6]));                           \
        w1[1] = GR3((S) * 3u + (PL), float(wt[1]), float(wt[4]), float(wt[7]));                           \
        w1[2] = GR3((S) * 3u + (PL), float(wt[2]), float(wt[5]), float(wt[8]));                           \
        w1[3] = GR3((S) * 3u + (PL), float(wt[9]), float(wt[12]), float(wt[15]));                           \
        w1[4] = GR3((S) * 3u + (PL), float(wt[10]), float(wt[13]), float(wt[16]));                           \
        w1[5] = GR3((S) * 3u + (PL), float(wt[11]), float(wt[14]), float(wt[17]));                           \
        w1[6] = GR3((S) * 3u + (PL), float(wt[18]), float(wt[21]), float(wt[24]));                         \
        w1[7] = GR3((S) * 3u + (PL), float(wt[19]), float(wt[22]), float(wt[25]));                        \
        w1[8] = GR3((S) * 3u + (PL), float(wt[20]), float(wt[23]), float(wt[26]));                        \
        w1[9] = GR3((S) * 3u + (PL), float(wt[27]), float(wt[30]), float(wt[33]));                         \
        w1[10] = GR3((S) * 3u + (PL), float(wt[28]), float(wt[31]), float(wt[34]));                       \
        w1[11] = GR3((S) * 3u + (PL), float(wt[29]), float(wt[32]), float(wt[35]));                       \
        w1[12] = GR3((S) * 3u + (PL), float(wt[36]), float(wt[39]), float(wt[42]));                       \
        w1[13] = GR3((S) * 3u + (PL), float(wt[37]), float(wt[40]), float(wt[43]));                       \
        w1[14] = GR3((S) * 3u + (PL), float(wt[38]), float(wt[41]), float(wt[44]));                       \
        w1[15] = GR3((S) * 3u + (PL), float(wt[45]), float(wt[48]), float(wt[51]));                       \
        w1[16] = GR3((S) * 3u + (PL), float(wt[46]), float(wt[49]), float(wt[52]));                       \
        w1[17] = GR3((S) * 3u + (PL), float(wt[47]), float(wt[50]), float(wt[53]));                       \
        w1[18] = GR3((S) * 3u + (PL), float(wt[54]), float(wt[57]), float(wt[60]));                       \
        w1[19] = GR3((S) * 3u + (PL), float(wt[55]), float(wt[58]), float(wt[61]));                       \
        w1[20] = GR3((S) * 3u + (PL), float(wt[56]), float(wt[59]), float(wt[62]));                       \
        w1[21] = GR3((S) * 3u + (PL), float(wt[63]), float(wt[66]), float(wt[69]));                       \
        w1[22] = GR3((S) * 3u + (PL), float(wt[64]), float(wt[67]), float(wt[70]));                       \
        w1[23] = GR3((S) * 3u + (PL), float(wt[65]), float(wt[68]), float(wt[71]));                       \
        ACC[0u] = simdgroup_float8x8(0);                                                                  \
        ACC[1u] = simdgroup_float8x8(0);                                                                  \
        ACC[2u] = simdgroup_float8x8(0);                                                                  \
        ACC[3u] = simdgroup_float8x8(0);                                                                  \
        ACC[4u] = simdgroup_float8x8(0);                                                                  \
        ACC[5u] = simdgroup_float8x8(0);                                                                  \
        MMA_KB(ACC, PL, 0u)                                                                               \
        MMA_KB(ACC, PL, 1u)                                                                               \
        MMA_KB(ACC, PL, 2u)                                                                               \
        MMA_KB(ACC, PL, 3u)                                                                               \
    }                                                                                                     \

// Fold one p row: q fold of the six positions with the factored A^T rows, then p fold.
#define FOLD_ROW(ACC, S, PL)                                                                              \
    {                                                                                                     \
        const auto q0 = ACC[0u].thread_elements();                                                        \
        const auto q1 = ACC[1u].thread_elements();                                                        \
        const auto q2 = ACC[2u].thread_elements();                                                        \
        const auto q3 = ACC[3u].thread_elements();                                                        \
        const auto q4 = ACC[4u].thread_elements();                                                        \
        const auto q5 = ACC[5u].thread_elements();                                                        \
        const float pa0 = q1[0] + q2[0], mb0 = q1[0] - q2[0];                                             \
        const float pa1 = q3[0] + q4[0], mb1 = q3[0] - q4[0];                                             \
        const float pA0 = q1[1] + q2[1], mB0 = q1[1] - q2[1];                                             \
        const float pA1 = q3[1] + q4[1], mB1 = q3[1] - q4[1];                                             \
        t0[0] = q0[0] + pa0 + pa1;                                                                        \
        t0[1] = fma(2.0f, mb1, mb0);                                                                      \
        t0[2] = fma(4.0f, pa1, pa0);                                                                      \
        t0[3] = fma(8.0f, mb1, mb0 + q5[0]);                                                              \
        t1[0] = q0[1] + pA0 + pA1;                                                                        \
        t1[1] = fma(2.0f, mB1, mB0);                                                                      \
        t1[2] = fma(4.0f, pA1, pA0);                                                                      \
        t1[3] = fma(8.0f, mB1, mB0 + q5[1]);                                                              \
        FOLD_P(S, PL)                                                                                     \
    }                                                                                                     \

[[max_total_threads_per_threadgroup(THR)]]
kernel void conv2d_kernel(
    device const half* x [[buffer(0)]],
    device const half* w [[buffer(1)]],
    device half* y [[buffer(2)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint3 tidv [[thread_position_in_threadgroup]])
{
    threadgroup float sV[18u * SVPOS];          // 4608 floats = 18 432 B
    threadgroup uint s_bad;

    const uint tid = tidv.x;
    const uint gid = tgid.x;
    const uint tgrp = gid / NGO;
    const uint ocbase = (gid - tgrp * NGO) * OCG;
    const uint sg = tid >> 5u;
    const uint lane = tid & 31u;
    const uint oc8 = ocbase + sg * 8u;
    const uint qid = lane >> 2u;
    const uint frow = (qid & 4u) + ((lane >> 1u) & 3u);
    const uint fcol = (qid & 2u) * 2u + (lane & 1u) * 2u;

    if (tid == 0u) s_bad = 0u;

    bool bad = false;

    // ---- weight taps of this lane: 4 channel chunks x 2 oc x 9 taps ----
    half wt[72];
    #pragma unroll
    for (uint kb = 0u; kb < 4u; ++kb) {
        const uint c = kb * 8u + frow;
        #pragma unroll
        for (uint o = 0u; o < 2u; ++o) {
            const uint oc = oc8 + fcol + o;
            device const half* wrow = w + oc * (9u * C) + c;
            #pragma unroll
            for (uint tp = 0u; tp < 9u; ++tp) {
                const half v = wrow[tp * C];
                wt[kb * 18u + o * 9u + tp] = v;
                bad = bad || nfh(v);
            }
        }
    }

    float y0[16];
    float y1[16];
    #pragma unroll
    for (uint i = 0u; i < 16u; ++i) {
        y0[i] = 0.0f;
        y1[i] = 0.0f;
    }

    simdgroup_float8x8 accA[6];
    simdgroup_float8x8 accB[6];
    simdgroup_float8x8 accC[6];
    float t0[4];
    float t1[4];
        // ---- stage V for strip 0 ----
        #pragma unroll
        for (uint it = 0u; it < 2u; ++it) {
            const uint pi = tid + it * THR;
            const uint stile = pi >> 5u;
            const uint sc = pi & 31u;
            const uint gt = tgrp * TB + stile;
            const uint img = gt >> 4u;
            const uint rem = gt & 15u;
            const uint tr = rem >> 2u;
            const uint tc = rem & 3u;
            device const half* xp = x + ((img * H + 4u * tr) * W + 4u * tc) * C + sc;
            half px[36];
            #pragma unroll
            for (uint i = 0u; i < 6u; ++i) {
                #pragma unroll
                for (uint j = 0u; j < 6u; ++j) {
                    const half v = xp[(i * W + j) * C];
                    px[i * 6u + j] = v;
                    bad = bad || nfh(v);
                }
            }
            const uint sb = sc * TB + stile;
            STAGE_ROW(0u, 0u)
            STAGE_ROW(1u, 1u)
            STAGE_ROW(2u, 2u)
        }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    MMA_ISSUE(accA, 0u, 0u)
    MMA_ISSUE(accB, 0u, 1u)
    MMA_ISSUE(accC, 0u, 2u)
    FOLD_ROW(accA, 0u, 0u)
    threadgroup_barrier(mem_flags::mem_threadgroup);
        // ---- stage V for strip 1 ----
        #pragma unroll
        for (uint it = 0u; it < 2u; ++it) {
            const uint pi = tid + it * THR;
            const uint stile = pi >> 5u;
            const uint sc = pi & 31u;
            const uint gt = tgrp * TB + stile;
            const uint img = gt >> 4u;
            const uint rem = gt & 15u;
            const uint tr = rem >> 2u;
            const uint tc = rem & 3u;
            device const half* xp = x + ((img * H + 4u * tr) * W + 4u * tc) * C + sc;
            half px[36];
            #pragma unroll
            for (uint i = 0u; i < 6u; ++i) {
                #pragma unroll
                for (uint j = 0u; j < 6u; ++j) {
                    const half v = xp[(i * W + j) * C];
                    px[i * 6u + j] = v;
                    bad = bad || nfh(v);
                }
            }
            const uint sb = sc * TB + stile;
            STAGE_ROW(3u, 0u)
            STAGE_ROW(4u, 1u)
            STAGE_ROW(5u, 2u)
        }
    // the two folds of strip 0 run while strip 1's V is being staged
    FOLD_ROW(accB, 0u, 1u)
    FOLD_ROW(accC, 0u, 2u)
    threadgroup_barrier(mem_flags::mem_threadgroup);
    MMA_ISSUE(accA, 1u, 0u)
    MMA_ISSUE(accB, 1u, 1u)
    MMA_ISSUE(accC, 1u, 2u)
    FOLD_ROW(accA, 1u, 0u)
    FOLD_ROW(accB, 1u, 1u)
    FOLD_ROW(accC, 1u, 2u)

    const bool anybad = simd_any(bad);
    if (lane == 0u && anybad) s_bad = 1u;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (s_bad != 0u) {
        direct_slice(x, w, y, tgrp, ocbase, tid);
        return;
    }

    const uint lt = tgrp * TB + frow;
    const uint img = lt >> 4u;
    const uint rem = lt & 15u;
    const uint tr = rem >> 2u;
    const uint tc = rem & 3u;
    device half* yp = y + (((img * OH + 4u * tr) * OW) + 4u * tc) * OC + oc8 + fcol;
    #pragma unroll
    for (uint oi = 0u; oi < 4u; ++oi) {
        #pragma unroll
        for (uint oj = 0u; oj < 4u; ++oj) {
            const uint k = oi * 4u + oj;
            *(device half2*)(yp + (oi * OW + oj) * OC) = half2(half(y0[k]), half(y1[k]));
        }
    }
}

#undef BT6
#undef GR3
#undef STAGE_ROW
#undef FOLD_Q
#undef FOLD_P
#undef MMA_K
#undef MMA_KB
#undef MMA_ISSUE
#undef FOLD_ROW

/* taic-launch
math fast
launch conv2d_kernel grid=16,1,1 threads=128,1,1
*/
