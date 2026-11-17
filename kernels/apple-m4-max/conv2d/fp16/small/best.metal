/* taic-launch
math fast
launch conv2d_r12a grid=64,1,1 threads=128,1,1
*/

// conv2d fp16 3x3 stride 1, NHWC in / NHWC out, [OC,KH,KW,C] weights, no bias.
// fp32 transforms, fp32 accumulation, exactly one rounding to fp16 at the output.
//
// 2-D Winograd F(2x2,3x3).  Geometry (this chip's measured MMA-issue optimum): 64 threadgroups
// x 128 threads = 256 warps.  A warp owns one output row pair (8 tiles of 2x2 along ow) x 8 oc
// = 256 outputs, so the 16 transform coordinates are exactly the 16 accumulator fragments.  The
// four warps of a threadgroup share one oc group, so U = G g G^T (16 coordinates x 32 channels x
// 8 oc = 16 KB) is built once per threadgroup into threadgroup memory and read back as one
// B operand (8 channels x 8 oc) per coordinate and channel half.
//
// Implementation notes (all scaling by powers of two, so exact):
//  - U is stored with the two K-blocks of a channel half interleaved, four floats per
//    (coordinate, K row, oc pair): [blk0 oc c, blk0 oc c+1, blk1 oc c, blk1 oc c+1].  One
//    16-byte threadgroup read is then the B operand of two matrix multiplies.
//  - The lane's four contiguous window channels (one 8-byte load) are converted once to float4,
//    and every transform step runs on all four channels at once.
//  - G carries no 1/2 factors (G' rows are {1,0,0}, {1,1,1}, {1,-1,1}, {0,0,1}), which removes
//    28 multiplies from each U-build thread.  The matching 1/2 per column of A^T is folded into
//    the inverse transform as fma(0.5, t, m), i.e. the same fma count as before.
//  - The four output stores are issued before the non-finite guard; the tap-by-tap fallback
//    rewrites the same 2x16x8 slab, so the branch is off the store's critical path.
//  - The fallback recomputes its own slab indices from (threadgroup, thread), so img/oh0/oc0/lane
//    die before the MMA region.
//  - Block 1's window loads are issued right after block 0's row transform: the window registers
//    are dead there and the loads hide behind block 0's 32 matrix multiplies.
//
// Non-finite inputs: a warp whose folded outputs hold a non-finite exponent redoes its own
// 2x16x8 slab with the oracle's tap-by-tap fp32 sum (only the `ieee` case takes that path).

#define CI      32u
#define HW      18u
#define OW      16u
#define OPIX    256u
#define OCLS    64u
#define WROW    288u
#define NCOORD  16u
#define USLICE  256u

// Direct fp32 tap sum for one 2x16 x 8oc slab (non-finite fallback, same order as the oracle).
static __attribute__((noinline)) void
conv2d_tapslab(device const half *x, device const half *w, device half *y,
               uint tg, uint tid)
{
    const uint lane = tid & 31u;
    const uint rp   = ((tg >> 3u) << 2u) + (tid >> 5u);
    const uint img  = rp >> 3u;
    const uint oh0  = (rp & 7u) << 1u;
    const uint oc0  = (tg & 7u) << 3u;
    const uint r = lane >> 4u;
    const uint c = lane & 15u;
    const device half *xb = x + ((img * HW + oh0 + r) * HW + c) * CI;
    device half *yb = y + (img * OPIX + (oh0 + r) * OW + c) * OCLS + oc0;
    for (uint o = 0u; o < 8u; ++o) {
        const device half *wb = w + (oc0 + o) * WROW;
        float s = 0.0f;
        for (uint kh = 0u; kh < 3u; ++kh)
            for (uint kw = 0u; kw < 3u; ++kw) {
                const device half *xs = xb + (kh * HW + kw) * CI;
                const device half *ws = wb + (kh * 3u + kw) * CI;
                for (uint ci = 0u; ci < CI; ++ci)
                    s = fma(float(xs[ci]), float(ws[ci]), s);
            }
        yb[o] = (half)s;
    }
}

static inline float2 frag2(thread simdgroup_float8x8 &M)
{
    return *(thread float2 *)&M.thread_elements();
}
static inline void setfrag(thread simdgroup_float8x8 &M, float2 v)
{
    *(thread float2 *)&M.thread_elements() = v;
}

kernel void conv2d_r12a(
    device const half *x [[buffer(0)]],
    device const half *w [[buffer(1)]],
    device half *y [[buffer(2)]],
    uint3 tgp [[threadgroup_position_in_grid]],
    uint3 tid [[thread_position_in_threadgroup]])
{
    threadgroup float U[NCOORD * USLICE];

    const uint lane = tid.x & 31u;
    const uint warp = tid.x >> 5u;
    const uint ocg  = tgp.x & 7u;
    const uint rp   = ((tgp.x >> 3u) << 2u) + warp;   // 0..31  (image, row pair)
    const uint img  = rp >> 3u;
    const uint oh0  = (rp & 7u) << 1u;
    const uint oc0  = ocg << 3u;

    const uint qi   = lane >> 2u;
    const uint mt   = (qi & 4u) + ((lane >> 1u) & 3u);   // tile along ow (fragment row)
    const uint cp   = (qi & 2u) + (lane & 1u);           // channel quad 0..3

    // fragment element map (measured): row = ((lane>>1)&3) + 4*(lane>>4),
    //                                  col = 4*((lane>>3)&1) + 2*(lane&1)
    const uint brow = ((lane >> 1u) & 3u) + 4u * (lane >> 4u);
    const uint bcol = 2u * cp;                           // == col

    {   // ---- weight transform, once per threadgroup: 1 channel x 2 adjacent oc ----
        // Two adjacent oc live in one threadgroup float2 (the two K-blocks of a row are
        // interleaved), so a thread's 32 coordinate values become 16 float2 stores.
        const uint ci  = tid.x & 31u;
        const uint pos = 2u * (ci >> 4u) + ((ci >> 1u) & 1u);
        const uint row = 2u * ((ci & 15u) >> 2u) + (ci & 1u);
        const uint pq  = tid.x >> 5u;                       // oc = 2pq, 2pq+1
        threadgroup float *dp =
            U + (pos >> 1u) * 128u + row * 16u + 4u * pq + 2u * (pos & 1u);
        const device half *wp0 = w + (oc0 + 2u * pq) * WROW + ci;
        float g0[9], g1[9];
#pragma unroll
        for (uint t = 0u; t < 9u; ++t) {
            g0[t] = float(wp0[t * CI]);
            g1[t] = float(wp0[WROW + t * CI]);
        }
        float f0[4][3], f1[4][3];
#pragma unroll
        for (uint kw = 0u; kw < 3u; ++kw) {
            const float a0 = g0[kw], b0 = g0[3u + kw], c0 = g0[6u + kw];
            const float a1 = g1[kw], b1 = g1[3u + kw], c1 = g1[6u + kw];
            const float s0 = a0 + c0, s1 = a1 + c1;
            f0[0][kw] = a0;
            f0[1][kw] = s0 + b0;
            f0[2][kw] = s0 - b0;
            f0[3][kw] = c0;
            f1[0][kw] = a1;
            f1[1][kw] = s1 + b1;
            f1[2][kw] = s1 - b1;
            f1[3][kw] = c1;
        }
#pragma unroll
        for (uint i = 0u; i < 4u; ++i) {
            const float a0 = f0[i][0], b0 = f0[i][1], c0 = f0[i][2];
            const float a1 = f1[i][0], b1 = f1[i][1], c1 = f1[i][2];
            threadgroup float *q = dp + i * 4u * USLICE;
            const float t0 = a0 + c0, t1 = a1 + c1;
            *(threadgroup float2 *)(q) = float2(a0, a1);
            *(threadgroup float2 *)(q + USLICE) = float2(t0 + b0, t1 + b1);
            *(threadgroup float2 *)(q + 2u * USLICE) = float2(t0 - b0, t1 - b1);
            *(threadgroup float2 *)(q + 3u * USLICE) = float2(c0, c1);
        }
    }

    const device half *xg =
        x + ((img * HW + oh0) * HW + 2u * mt) * CI + 4u * cp;

    uint2 win[4][4];
    simdgroup_float8x8 acc[NCOORD];
#pragma unroll
    for (uint i = 0u; i < NCOORD; ++i) acc[i] = simdgroup_float8x8(0.0f);
    simdgroup_float8x8 V, B;

#define LOADWIN(OFF)                                                                 \
    _Pragma("unroll")                                                                \
    for (uint r = 0u; r < 4u; ++r)                                                   \
        _Pragma("unroll")                                                            \
        for (uint s = 0u; s < 4u; ++s)                                               \
            win[r][s] = *(device const uint2 *)(xg + (r * HW + s) * CI + (OFF));

    // 16-byte U read feeds both K-blocks of a channel half; VV0 is the transform of window half
    // x (block pr*2, lane columns .xy), VV1 the transform of window half y (block pr*2+1, .zw)
#define MMA2(I, J, VV0, VV1)                                                         \
    {                                                                                \
        const uint cc = (I) * 4u + (J);                                              \
        const threadgroup float *bp =                                                \
            U + cc * USLICE + pr * 128u + brow * 16u + 2u * bcol;                    \
        const float4 bv = *(threadgroup float4 *)bp;                                 \
        setfrag(V, (VV0));                                                           \
        setfrag(B, bv.xy);                                                           \
        simdgroup_multiply_accumulate(acc[cc], V, B, acc[cc]);                       \
        setfrag(V, (VV1));                                                           \
        setfrag(B, bv.zw);                                                           \
        simdgroup_multiply_accumulate(acc[cc], V, B, acc[cc]);                       \
    }

// T##J holds the row transform of the lane's four channels at ow coordinate J; the column
// transform mixes rows 0..3.  Only a_J = T0-T2, b_J = T1+T2, c_J = T1-T3 are ever formed:
//   V(0,J) = a_J   V(1,J) = b_J   V(3,J) = c_J   V(2,J) = T2-T1 = -(T1-T2)  (one negation
// of the J = 1 row sum, so coordinate 2 costs no extra add)
#define EMITJ(J)                                                                     \
    {                                                                                \
        const float4 a = T##J[0] - T##J[2];                                          \
        const float4 b = T##J[1] + T##J[2];                                          \
        const float4 c = T##J[1] - T##J[3];                                          \
        const float4 g = T##J[2] - T##J[1];                                          \
        MMA2(0u, J, a.xy, a.zw)                                                      \
        MMA2(1u, J, b.xy, b.zw)                                                      \
        MMA2(3u, J, c.xy, c.zw)                                                      \
        MMA2(2u, J, g.xy, g.zw)                                                      \
    }

#define ROWT                                                                         \
    _Pragma("unroll")                                                                \
    for (uint r = 0u; r < 4u; ++r) {                                                 \
        const float4 d0 = float4(as_type<half4>(win[r][0]));                         \
        const float4 d1 = float4(as_type<half4>(win[r][1]));                         \
        const float4 d2 = float4(as_type<half4>(win[r][2]));                         \
        const float4 d3 = float4(as_type<half4>(win[r][3]));                         \
        T0[r] = d0 - d2;                                                             \
        T1[r] = d1 + d2;                                                             \
        T2[r] = d2 - d1;                                                             \
        T3[r] = d1 - d3;                                                             \
    }

// one channel half = two K-blocks (block pair pr) x 16 coordinates
#define BLOCK(PR)                                                                    \
    {                                                                                \
        const uint pr = PR;                                                          \
        float4 T0[4], T1[4], T2[4], T3[4];                                           \
        ROWT                                                                         \
        EMITJ(0)                                                                     \
        EMITJ(1)                                                                     \
        EMITJ(2)                                                                     \
        EMITJ(3)                                                                     \
    }

    LOADWIN(0u)                          // channels 0..15 while U is still being built
    threadgroup_barrier(mem_flags::mem_threadgroup);

    {
        const uint pr = 0u;
        float4 T0[4], T1[4], T2[4], T3[4];
        ROWT
        LOADWIN(16u)                     // channels 16..31 behind block 0's 32 MMAs
        EMITJ(0)
        EMITJ(1)
        EMITJ(2)
        EMITJ(3)
    }
    BLOCK(1u)

    // inverse transform (A^T D) M (A^T D)^T, still in registers; D = diag(1, 1/2, 1/2, 1) is
    // the price of the 2x on G's middle rows, and fma(0.5, t, m) keeps the op count unchanged
    float2 lo[4], hi[4];
#pragma unroll
    for (uint j = 0u; j < 4u; ++j) {
        const float2 a1 = frag2(acc[4u + j]);
        const float2 a2 = frag2(acc[8u + j]);
        lo[j] = fma(0.5f, a1 + a2, frag2(acc[j]));
        hi[j] = fma(0.5f, a1 - a2, -frag2(acc[12u + j]));
    }
    float2 o[4];
    o[0] = fma(0.5f, lo[1] + lo[2], lo[0]);
    o[1] = fma(0.5f, lo[1] - lo[2], -lo[3]);
    o[2] = fma(0.5f, hi[1] + hi[2], hi[0]);
    o[3] = fma(0.5f, hi[1] - hi[2], -hi[3]);

    device half *yp = y + (img * OPIX + oh0 * OW + 2u * mt) * OCLS + oc0 + 2u * cp;
    *(device half2 *)(yp) = half2((half)o[0].x, (half)o[0].y);
    *(device half2 *)(yp + OCLS) = half2((half)o[1].x, (half)o[1].y);
    *(device half2 *)(yp + OW * OCLS) = half2((half)o[2].x, (half)o[2].y);
    *(device half2 *)(yp + OW * OCLS + OCLS) = half2((half)o[3].x, (half)o[3].y);

    uint em = 0u;
#pragma unroll
    for (uint i = 0u; i < 4u; ++i) {
        em = max(em, as_type<uint>(o[i].x) & 0x7F800000u);
        em = max(em, as_type<uint>(o[i].y) & 0x7F800000u);
    }
    if (simd_any(em == 0x7F800000u))
        conv2d_tapslab(x, w, y, tgp.x, tid.x);
}
