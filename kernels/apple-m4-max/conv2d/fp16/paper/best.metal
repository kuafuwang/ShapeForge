// Winograd F(4x4,3x3): 6x6 input patches, 36 transform positions, transforms and matrix
// operands all fp32, one rounding to half at the store.
//
// GEMM in the transform domain: for every position p the fragment product is
//     M_p[tile][oc] = sum_c V_p[tile][c] * U_p[c][oc]
// so the accumulator rows are the output tiles and the columns the output channels. The inverse
// transform Y = A^T M A is linear in M, so it is folded straight into 16 output accumulators and
// the epilogue needs no transpose.
//
// Stage 1 (conv2d_filter) builds U_p[c][oc] = G g G^T in a lane-slot layout: one 16-byte load per
// lane feeds two k-steps of the B operand, and it raises one word when the filter holds an Inf or a
// NaN. Stage 2 (conv2d_kernel) stages V_p for one transform row at a time in threadgroup memory (the row
// transform goes first, so each input row is fetched once per transform row rather than once per
// position), runs 6 positions, folds, and repeats for 6 rows.
//
// Non-finite handling: the reference sums tap by tap in fp32 for such inputs, so a tile block that
// sees an Inf or a NaN is recomputed that way and never goes through the transforms (an infinity
// times a zero transform coefficient would smear a NaN over the tile). The activation check rides on
// the loads the first transform row already makes, so no separate sweep over x is needed, and it is
// decided before the first MMA, so no output is ever half transformed.
/* taic-launch
workspace uwin 1179652 zero
launch conv2d_filter grid=288,1,1 threads=128,1,1 workspace=uwin
launch conv2d_kernel grid=392,2,1 threads=256,1,1 workspace=uwin
*/

// F(4,3): Y = A^T [(G g G^T) . (B^T d B)] A
constant constexpr float ATc[4][6] = {
    {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 0.0f},
    {0.0f, 1.0f, -1.0f, 2.0f, -2.0f, 0.0f},
    {0.0f, 1.0f, 1.0f, 4.0f, 4.0f, 0.0f},
    {0.0f, 1.0f, -1.0f, 8.0f, -8.0f, 1.0f}};

constant constexpr float BTc[6][6] = {
    {4.0f, 0.0f, -5.0f, 0.0f, 1.0f, 0.0f},
    {0.0f, -4.0f, -4.0f, 1.0f, 1.0f, 0.0f},
    {0.0f, 4.0f, -4.0f, -1.0f, 1.0f, 0.0f},
    {0.0f, -2.0f, -1.0f, 2.0f, 1.0f, 0.0f},
    {0.0f, 2.0f, -1.0f, -2.0f, 1.0f, 0.0f},
    {0.0f, 4.0f, 0.0f, -5.0f, 0.0f, 1.0f}};

constant constexpr float GCFc[6][3] = {
    {0.25f, 0.0f, 0.0f},
    {-1.0f / 6.0f, -1.0f / 6.0f, -1.0f / 6.0f},
    {-1.0f / 6.0f, 1.0f / 6.0f, -1.0f / 6.0f},
    {1.0f / 24.0f, 1.0f / 12.0f, 1.0f / 6.0f},
    {1.0f / 24.0f, -1.0f / 12.0f, 1.0f / 6.0f},
    {0.0f, 0.0f, 1.0f}};

constant constexpr uint TPI = 196u;    // 4x4 output tiles per image (14 x 14)
constant constexpr uint TPR = 14u;     // tiles per image row
constant constexpr uint TTG = 16u;     // tiles per threadgroup (two 8-tile fragments)
constant constexpr uint OCG = 64u;     // output channels per threadgroup
constant constexpr uint NTH = 256u;    // threads per threadgroup of the main kernel (8 SIMD groups)
constant constexpr uint SCN = 256u;    // threads that stage V (16 tiles x 16 channel groups)
constant constexpr uint LTE = 128u;    // floats in one [32 slots][4 values] lane block
constant constexpr uint FLAGW = 294912u;  // filter flag, past the 36*128*64 payload
constant constexpr uint TGFLT = 6144u;    // threadgroup floats for V (6 positions x 4 kpairs x 2 frags)

// fp16 pair: exponent all ones means Inf or NaN (an fp32 mask would be wrong here)
static inline bool non_finite_h2(uint b)
{
    return (b & 0x7c00u) == 0x7c00u || (b & 0x7c000000u) == 0x7c000000u;
}

static inline bool non_finite_h8(uint4 v)
{
    return non_finite_h2(v.x) || non_finite_h2(v.y) || non_finite_h2(v.z) || non_finite_h2(v.w);
}

static inline void set_pair(thread simdgroup_float8x8& m, float a, float b)
{
    auto e = m.thread_elements();
    e[0] = a;
    e[1] = b;
    m.thread_elements() = e;
}

static inline void add_scaled(thread float& a0, thread float& a1, thread const simdgroup_float8x8& m, float cf)
{
    const auto e = m.thread_elements();
    a0 = fma(cf, e[0], a0);
    a1 = fma(cf, e[1], a1);
}

static inline void tile_decode(uint tile, thread uint& img, thread uint& tr, thread uint& tc)
{
    img = tile / TPI;
    const uint rem = tile - img * TPI;
    tr = rem / TPR;
    tc = rem - tr * TPR;
}

// Stage 1: U[p][c][oc] = G[pr] g G[pc]^T, written so that one lane-sized 16-byte slot holds the two
// k-steps kk and kk+1 of the B operand. A non-finite filter value raises the flag word.
kernel void conv2d_filter(device const half* w [[buffer(1)]],
                          device float* uwin [[buffer(3)]],
                          uint2 tg [[threadgroup_position_in_grid]],
                          uint lid [[thread_index_in_threadgroup]])
{
    const uint tid = tg.x * 128u + lid;         // 0 .. 36863
    const uint c8 = (tid & 7u) * 8u;            // first of eight channels
    const uint oc = (tid >> 3u) & 127u;
    const uint p = tid >> 10u;                  // 0 .. 35
    const uint pr = p / 6u;
    const uint pc = p - pr * 6u;

    float acc[8];
#pragma clang loop unroll(full)
    for (uint k = 0; k < 8u; ++k)
        acc[k] = 0.0f;
    bool weird = false;
    for (uint i = 0; i < 3u; ++i) {
        const float gi = GCFc[pr][i];
        if (gi == 0.0f)
            continue;
        for (uint j = 0; j < 3u; ++j) {
            const float cf = gi * GCFc[pc][j];
            if (cf == 0.0f)
                continue;
            const uint4 raw = *reinterpret_cast<device const uint4*>(w + oc * K_TOTAL + (i * KW + j) * C + c8);
            weird = weird || non_finite_h8(raw);
            const float4 lo = float4(as_type<half4>(raw.xy));
            const float4 hi = float4(as_type<half4>(raw.zw));
            acc[0] += cf * lo.x;
            acc[1] += cf * lo.y;
            acc[2] += cf * lo.z;
            acc[3] += cf * lo.w;
            acc[4] += cf * hi.x;
            acc[5] += cf * hi.y;
            acc[6] += cf * hi.z;
            acc[7] += cf * hi.w;
        }
    }
    if (weird)
        uwin[FLAGW] = 1.0f;
    // slot = channel-in-8 * 4 + oc pair, inner quadruple = [kk, kk+1] x [oc pair]
    device float* base = uwin + (p * 4u + (c8 >> 4u)) * (16u * LTE) + (oc >> 3u) * LTE
                         + ((oc & 7u) >> 1u) * 4u + ((c8 >> 3u) & 1u) * 2u + (oc & 1u);
#pragma clang loop unroll(full)
    for (uint k = 0; k < 8u; ++k)
        base[k * 16u] = acc[k];
}

// tap-by-tap fp32 sum, the reference's own semantics for non-finite inputs
static inline void tap_by_tap(device const half* x, device const half* w, device half* y, uint tb, uint lid, uint ocy)
{
    for (uint n = lid; n < TTG * 1024u; n += NTH) {
        const uint tl = n >> 10u;
        const uint q = (n >> 6u) & 15u;
        uint img, tr, tc;
        tile_decode(tb + tl, img, tr, tc);
        const uint oh = 4u * tr + (q >> 2u);
        const uint ow = 4u * tc + (q & 3u);
        const uint oc = ocy + (n & 63u);
        if (oh < OH && ow < OW) {
            float sum = 0.0f;
            for (uint kh = 0; kh < KH; ++kh) {
                device const half* xp = x + ((img * H + oh + kh) * W + ow) * C;
                device const half* wp = w + (oc * KH + kh) * KW * C;
                for (uint kw = 0; kw < KW; ++kw) {
#pragma clang loop unroll(full)
                    for (uint c = 0; c < C; ++c)
                        sum = fma(float(xp[kw * C + c]), float(wp[kw * C + c]), sum);
                }
            }
            y[((img * OH + oh) * OW + ow) * OC + oc] = half(sum);
        }
    }
}

[[max_total_threads_per_threadgroup(256)]]
kernel void conv2d_kernel(device const half* x [[buffer(0)]],
                          device const half* w [[buffer(1)]],
                          device half* y [[buffer(2)]],
                          device const float* uwin [[buffer(3)]],
                          uint2 tg [[threadgroup_position_in_grid]],
                          uint lid [[thread_index_in_threadgroup]])
{
    const uint tb = tg.x * TTG;                  // first tile of this threadgroup
    const uint ocy = tg.y * OCG;                 // this threadgroup's output-channel half

    // one lane-sized block per (position, kpair, tile fragment): [row * 4 + colpair][2 k-steps][2 elems]
    threadgroup float sV[TGFLT];                 // 24576 B
    threadgroup uint sweird;                     // raised when the staged x holds Inf or NaN

    // a non-finite filter means the transform domain would smear the special, so this block takes
    // the reference's own semantics up front; non-finite activations are caught while staging V
    if (as_type<uint>(uwin[FLAGW]) != 0u) {
        tap_by_tap(x, w, y, tb, lid, ocy);
        return;
    }
    if (lid == 0u)
        sweird = 0u;

    const uint sg = lid >> 5u;
    const uint lane = lid & 31u;
    const uint lrow = ((lane >> 2u) & 4u) + ((lane >> 1u) & 3u);   // fragment row
    const uint lcp = ((lane >> 2u) & 2u) + (lane & 1u);            // fragment column pair
    const uint slot4 = (lrow * 4u + lcp) * 4u;
    const uint oc0 = ocy + sg * 8u;
    const uint ub0 = (oc0 >> 3u) * LTE;          // compact [p][k-pair][oc-fragment] ordering

    const uint t = lid >> 4u;                    // tile inside the threadgroup (staging)
    const uint cg = lid & 15u;                   // channel group of four (staging)
    const uint c4 = cg * 4u;

    // staging decode for this thread's tile
    uint simg, str, stc;
    tile_decode(tb + t, simg, str, stc);
    const uint stf = t >> 3u;
    const uint srow = t & 7u;
    device const half* spx = x + (simg * H * W) * C + c4;   // + (row * W + col) * C
    const uint sbase = (cg >> 2u) * 2u * LTE + srow * 16u + (cg & 1u) * 8u + ((cg >> 1u) & 1u) * 2u;

    float yacc[16][2][2];
#pragma clang loop unroll(full)
    for (uint q = 0; q < 16u; ++q) {
        yacc[q][0][0] = 0.0f;
        yacc[q][0][1] = 0.0f;
        yacc[q][1][0] = 0.0f;
        yacc[q][1][1] = 0.0f;
    }

#pragma clang loop unroll(full)
    for (uint pr = 0; pr < 6u; ++pr) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lid < SCN) {   // ---- stage V[pr][pc] for all six pc, one transform row at a time ----
            bool weird = false;
            float r[6][4];
#pragma clang loop unroll(full)
            for (uint j = 0; j < 6u; ++j) {
                r[j][0] = 0.0f;
                r[j][1] = 0.0f;
                r[j][2] = 0.0f;
                r[j][3] = 0.0f;
            }
#pragma clang loop unroll(full)
            for (uint i = 0; i < 6u; ++i) {
                const float ci = BTc[pr][i];
                if (ci != 0.0f) {
                    const uint rr = min(4u * str + i, H - 1u);
                    device const half* row = spx + rr * W * C;
#pragma clang loop unroll(full)
                    for (uint j = 0; j < 6u; ++j) {
                        const half4 hd = *reinterpret_cast<device const half4*>(row + min(4u * stc + j, W - 1u) * C);
                        const float4 d = float4(hd);
                        if (pr == 0u) {   // the first row reads every patch value, so one pass decides the block
                            const uint2 raw = as_type<uint2>(hd);
                            weird = weird || ((raw.x & 0x7c007c00u) == 0x7c007c00u) ||
                                    ((raw.y & 0x7c007c00u) == 0x7c007c00u);
                        }
                        r[j][0] = fma(ci, d.x, r[j][0]);
                        r[j][1] = fma(ci, d.y, r[j][1]);
                        r[j][2] = fma(ci, d.z, r[j][2]);
                        r[j][3] = fma(ci, d.w, r[j][3]);
                    }
                }
            }
#pragma clang loop unroll(full)
            for (uint pc = 0; pc < 6u; ++pc) {
                float v[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma clang loop unroll(full)
                for (uint j = 0; j < 6u; ++j) {
                    const float bj = BTc[pc][j];
                    v[0] = fma(bj, r[j][0], v[0]);
                    v[1] = fma(bj, r[j][1], v[1]);
                    v[2] = fma(bj, r[j][2], v[2]);
                    v[3] = fma(bj, r[j][3], v[3]);
                }
                threadgroup float* vb = sV + (pc * 8u + stf) * LTE + sbase;
                *reinterpret_cast<threadgroup float2*>(vb) = float2(v[0], v[1]);
                *reinterpret_cast<threadgroup float2*>(vb + 4u) = float2(v[2], v[3]);
            }
            if (pr == 0u && simd_any(weird))
                sweird = 1u;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (pr == 0u && sweird != 0u) {   // non-finite activation somewhere in this block
            tap_by_tap(x, w, y, tb, lid, ocy);
            return;
        }

        // Keep the inverse transform separable: accumulate the six column positions
        // into four partial rows, then apply the row transform once per pr.
        float rcol[4][2][2];
#pragma clang loop unroll(full)
        for (uint ox = 0; ox < 4u; ++ox)
#pragma clang loop unroll(full)
            for (uint tf = 0; tf < 2u; ++tf)
#pragma clang loop unroll(full)
                for (uint e = 0; e < 2u; ++e)
                    rcol[ox][tf][e] = 0.0f;

#pragma clang loop unroll(full)
        for (uint pc = 0; pc < 6u; ++pc) {
            const uint p = pr * 6u + pc;
            simdgroup_float8x8 acc0 = simdgroup_float8x8(0.0f);
            simdgroup_float8x8 acc1 = simdgroup_float8x8(0.0f);
#pragma clang loop unroll(full)
            for (uint kp = 0; kp < 4u; ++kp) {
                const threadgroup float* vb = sV + (pc * 8u + kp * 2u) * LTE + slot4;
                const device float* ub = uwin + ub0 + (p * 4u + kp) * (16u * LTE) + slot4;
                const float4 ra0 = *reinterpret_cast<threadgroup const float4*>(vb);
                const float4 ra1 = *reinterpret_cast<threadgroup const float4*>(vb + LTE);
                const float4 rb = *reinterpret_cast<device const float4*>(ub);
                simdgroup_float8x8 a00, a01, a10, a11, b0, b1;
                set_pair(a00, ra0.x, ra0.y);     // tile fragment 0, k-step 0
                set_pair(a01, ra0.z, ra0.w);     // tile fragment 0, k-step 1
                set_pair(a10, ra1.x, ra1.y);     // tile fragment 1, k-step 0
                set_pair(a11, ra1.z, ra1.w);     // tile fragment 1, k-step 1
                set_pair(b0, rb.x, rb.y);        // B operand, k-step 0
                set_pair(b1, rb.z, rb.w);        // B operand, k-step 1
                simdgroup_multiply_accumulate(acc0, a00, b0, acc0);
                simdgroup_multiply_accumulate(acc0, a01, b1, acc0);
                simdgroup_multiply_accumulate(acc1, a10, b0, acc1);
                simdgroup_multiply_accumulate(acc1, a11, b1, acc1);
            }
#pragma clang loop unroll(full)
            for (uint ox = 0; ox < 4u; ++ox) {
                const float cf = ATc[ox][pc];
                if (cf != 0.0f) {
                    const auto e0 = acc0.thread_elements();
                    const auto e1 = acc1.thread_elements();
                    rcol[ox][0][0] = fma(cf, e0[0], rcol[ox][0][0]);
                    rcol[ox][0][1] = fma(cf, e0[1], rcol[ox][0][1]);
                    rcol[ox][1][0] = fma(cf, e1[0], rcol[ox][1][0]);
                    rcol[ox][1][1] = fma(cf, e1[1], rcol[ox][1][1]);
                }
            }
        }

#pragma clang loop unroll(full)
        for (uint oy = 0; oy < 4u; ++oy) {
            const float cf = ATc[oy][pr];
            if (cf != 0.0f) {
#pragma clang loop unroll(full)
                for (uint ox = 0; ox < 4u; ++ox) {
#pragma clang loop unroll(full)
                    for (uint tf = 0; tf < 2u; ++tf) {
#pragma clang loop unroll(full)
                        for (uint e = 0; e < 2u; ++e)
                            yacc[oy * 4u + ox][tf][e] = fma(cf, rcol[ox][tf][e], yacc[oy * 4u + ox][tf][e]);
                    }
                }
            }
        }
    }

    // epilogue: one rounding, half2 stores straight to y
    const uint oc = oc0 + lcp * 2u;
#pragma clang loop unroll(full)
    for (uint tf = 0; tf < 2u; ++tf) {
        uint img, tr, tc;
        tile_decode(tb + tf * 8u + lrow, img, tr, tc);
        device half* yp = y + ((img * OH + 4u * tr) * OW + 4u * tc) * OC + oc;
#pragma clang loop unroll(full)
        for (uint oy = 0; oy < 4u; ++oy) {
#pragma clang loop unroll(full)
            for (uint ox = 0; ox < 4u; ++ox) {
                if (4u * tr + oy < OH && 4u * tc + ox < OW)
                    *reinterpret_cast<device half2*>(yp + (oy * OW + ox) * OC) =
                        half2(half(yacc[oy * 4u + ox][tf][0]), half(yacc[oy * 4u + ox][tf][1]));
            }
        }
    }
}
