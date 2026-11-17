// 3x3 convolution, stride 1, no padding, no bias.  NHWC activations, [OC,KH,KW,C] filter, fp16
// operands, fp32 accumulation, exactly one rounding to half where the output is stored.
//
// Winograd F(4x4,3x3).  A 4x4 output tile comes from a 6x6 input window, so the transform domain
// has 36 positions and each position is one small matrix product
//     M_p[tile][oc] += V_p[tile][c] * U_p[c][oc],    V_p = B^T d B,   U_p = G g G^T.
// The inverse transform Y = A^T M A is linear in M, so a whole row of six positions folds as soon
// as it is multiplied: the six column coefficients collapse into four partial output rows, and the
// row coefficient is applied once into the sixteen live outputs a lane owns.
//
// Block geometry: 256 threads, 16 output tiles x 64 output channels per threadgroup.  One warp
// owns eight output channels and two 8-tile fragments (the two halves of the block), so its fold
// holds 16 outputs x 2 halves x 2 lane elements = 64 floats.  That is the largest fold this
// register file carries; anything wider measured 16-32% slower on this chip, so the operand
// traffic below is the best load-to-multiply ratio the budget allows.
//
// Stage 1 (conv2d_filter) builds the transformed filter once for the whole grid, already in the
// shape a consuming lane reads it (see the pack comment at the kernel), and leaves one flag word
// per threadgroup behind when the slice of w it read holds an Inf or a NaN.
//
// Stage 2 (conv2d_main) walks one transform row at a time: it stages that row's six V planes for
// the block's 16 tiles in threadgroup memory, then feeds two tile fragments with one transformed
// filter pack per k-step pair (two device operand loads per four matrix instructions).
//
// Non-finite handling follows the oracle, which sums tap by tap in fp32 when an input is not
// finite: a zero transform coefficient would smear one infinity over a whole tile.  A non-finite
// filter tap changes how the oracle sums every output, so the filter flag is global (one word per
// stage-1 threadgroup, summed by every consumer).  For the activations the whole 6x6 window of
// every tile arrives through the staged V values, and no window element has a zero coefficient in
// all 36 positions, so screening the staged values is complete.  The verdict is voted through
// threadgroup memory and read back after a barrier, so the whole block takes the same branch; the
// tap-by-tap path then rewrites exactly this block's 16 tiles x 64 channels.
//
// The scratch is declared without a zero fill: stage 1 writes every float of the table and every
// flag word before stage 2 reads one, so the harness fill would be paid (inside the timed command
// buffer) for nothing.  A non-const device pointer reflects readwrite on this toolchain whatever
// it does, and an uninitialized-poison producer must reflect write, so stage 1 writes through the
// linear R32Float view of the same bytes: element i is texel (i % 32, i / 32) of a 32 x 9218 view.
// Stage 2 reads the same bytes as a plain read-only device pointer.
/* taic-launch
math fast
scratch uwin fp32 294944
launch conv2d_filter grid=24,1,1 threads=256,1,1 scratch=uwin:write
launch conv2d_main grid=392,2,1 threads=256,1,1 scratch=uwin:read
*/

// F(4,3) on the points 0, 1, -1, 2, -2 with A^T (ATo), B^T (BTo) and G (GCF).
constant constexpr float ATo[4][6] = {
    {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 0.0f},
    {0.0f, 1.0f, -1.0f, 2.0f, -2.0f, 0.0f},
    {0.0f, 1.0f, 1.0f, 4.0f, 4.0f, 0.0f},
    {0.0f, 1.0f, -1.0f, 8.0f, -8.0f, 1.0f}};

constant constexpr float BTo[6][6] = {
    {4.0f, 0.0f, -5.0f, 0.0f, 1.0f, 0.0f},
    {0.0f, -4.0f, -4.0f, 1.0f, 1.0f, 0.0f},
    {0.0f, 4.0f, -4.0f, -1.0f, 1.0f, 0.0f},
    {0.0f, -2.0f, -1.0f, 2.0f, 1.0f, 0.0f},
    {0.0f, 2.0f, -1.0f, -2.0f, 1.0f, 0.0f},
    {0.0f, 4.0f, 0.0f, -5.0f, 0.0f, 1.0f}};

constant constexpr float GCF[6][3] = {
    {0.25f, 0.0f, 0.0f},
    {-1.0f / 6.0f, -1.0f / 6.0f, -1.0f / 6.0f},
    {-1.0f / 6.0f, 1.0f / 6.0f, -1.0f / 6.0f},
    {1.0f / 24.0f, 1.0f / 12.0f, 1.0f / 6.0f},
    {1.0f / 24.0f, -1.0f / 12.0f, 1.0f / 6.0f},
    {0.0f, 0.0f, 1.0f}};

constant constexpr uint TPC = (OW + 3u) / 4u;    // output tile columns per image (14)
constant constexpr uint TPR = (OH + 3u) / 4u;    // output tile rows per image (14)
constant constexpr uint TPI = TPC * TPR;         // output tiles per image (196)
constant constexpr uint NTHR = 256u;             // threads in either stage
constant constexpr uint TTG = 16u;               // output tiles per threadgroup
constant constexpr uint OCG = 64u;               // output channels per threadgroup
constant constexpr uint NFIL = 24u;              // stage-1 threadgroups == filter flag words
constant constexpr uint NQ = C / 4u;             // half4 channel quads of one window element (16)
constant constexpr uint NOC = OC / 8u;           // output-channel octets (16)
constant constexpr uint VSTR = C;                // tile stride of a staged V plane
constant constexpr uint UFLT = 36u * C * OC;     // transformed filter, 36 positions x C x OC
constant constexpr uint USZ = 294944u;           // whole scratch (UFLT + 32 flag words)
constant constexpr uint TEXW = 32u;              // R32Float view width (128-byte rows)

// A half whose five exponent bits are all ones is an Inf or a NaN: adding 0x0400 to the masked
// exponent carries that case into bit 15 and leaves every finite value below it.
static inline bool special_h2(uint bits)
{
    return (((bits & 0x7c007c00u) + 0x04000400u) & 0x80008000u) != 0u;
}

static inline bool special_h8(uint4 raw)
{
    return special_h2(raw.x) || special_h2(raw.y) || special_h2(raw.z) || special_h2(raw.w);
}

// the same carry test for four float lanes, where it lands in bit 31
static inline bool special_f4(float4 v)
{
    const uint m = ((as_type<uint>(v.x) & 0x7f800000u) + 0x00800000u)
                 | ((as_type<uint>(v.y) & 0x7f800000u) + 0x00800000u)
                 | ((as_type<uint>(v.z) & 0x7f800000u) + 0x00800000u)
                 | ((as_type<uint>(v.w) & 0x7f800000u) + 0x00800000u);
    return (m & 0x80000000u) != 0u;
}

static inline void tile_decode(uint tile, thread uint& img, thread uint& tr, thread uint& tc)
{
    img = tile / TPI;
    const uint rem = tile - img * TPI;
    tr = rem / TPC;
    tc = rem - tr * TPC;
}

// Flat index of the transformed-filter float that belongs to (position p, output channel oc,
// channel c).  Within one position the table is [output octet][k-step pair][lane][component], and
// a component is [first/second k step][even/odd output channel], so one 16-byte pack serves both
// k steps of one lane through thread_elements().
static inline uint upack_index(uint p, uint oc, uint c)
{
    const uint kstep = c >> 3;                 // channel octet of the k step
    const uint row = c & 7u;                   // channel row inside the 8x8 operand
    const uint colh = (oc >> 1) & 3u;          // fragment column pair of this output channel
    const uint lane = 16u * (row >> 2) + 8u * (colh >> 1) + 2u * (row & 3u) + (colh & 1u);
    return p * (C * OC) + (oc >> 3) * 512u + (kstep >> 1) * 128u + lane * 4u
         + 2u * (kstep & 1u) + (oc & 1u);
}

// Tap-by-tap fp32 sum: the semantics the oracle uses whenever an input is not finite.  Rewrites
// every element of this block's TTG tiles x OCG channels.  The summation order decides which
// outputs come out NaN and which come out +-Inf, and the harness compares kinds and signs, so the
// chain is one fma per tap and not re-associated.  Kept out of line so the cold path cannot cost
// the fast path any registers.
__attribute__((noinline)) static void conv2d_taps(device const half* x, device const half* w,
                                                  device half* y, uint tile0, uint oc0, uint lid)
{
    for (uint n = lid; n < TTG * 16u * OCG; n += NTHR) {
        const uint oc = oc0 + (n & (OCG - 1u));
        const uint q = (n / OCG) & 15u;         // output position inside the tile
        const uint tl = n / (OCG * 16u);        // tile inside the block
        uint img, tr, tc;
        tile_decode(tile0 + tl, img, tr, tc);
        const uint oh = 4u * tr + (q >> 2u);
        const uint ow = 4u * tc + (q & 3u);
        if (oh < OH && ow < OW) {
            device const half* const xp = x + ((img * H + oh) * W + ow) * C;
            device const half* const wp = w + oc * (KH * KW * C);
            float sum = 0.0f;
            for (uint kh = 0; kh < KH; ++kh)
                for (uint kw = 0; kw < KW; ++kw)
                    for (uint c = 0; c < C; ++c)
                        sum = fma(float(xp[(kh * W + kw) * C + c]),
                                  float(wp[(kh * KW + kw) * C + c]), sum);
            y[((img * OH + oh) * OW + ow) * OC + oc] = half(sum);
        }
    }
}

// Stage 1: U[p][c][oc] = (G g G^T)[p][c][oc] for all 36 positions, in the packed consumer layout
// above, plus this threadgroup's filter flag word.  One threadgroup owns one transform row and 32
// output channels; one thread owns one output channel and one channel octet, reads its nine taps
// once and feeds all six column positions of the row from registers.
kernel void conv2d_filter(device const half* w [[buffer(0)]],
                          texture2d<float, access::write> uwin [[texture(0)]],
                          uint3 tg [[threadgroup_position_in_grid]],
                          uint lid [[thread_index_in_threadgroup]],
                          uint lane [[thread_index_in_simdgroup]])
{
    const uint pr = tg.x >> 2u;                  // transform row
    const uint oc0 = (tg.x & 3u) * (OC / 4u);    // this threadgroup's 32 output channels
    threadgroup uint sbad;

    if (lid == 0u)
        sbad = 0u;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // a warp covers 32 consecutive output channels at one channel octet, so its stores land in
    // whole 32-byte sectors of the packed tiles instead of halves of eight different ones
    const uint oc = oc0 + (lid & 31u);
    const uint c8 = (lid >> 5u) * 8u;
    device const half* const wp = w + oc * (KH * KW * C) + c8;
    uint bad = 0u;
    float tap[9][8];
#pragma clang loop unroll(full)
    for (uint i = 0; i < 3u; ++i) {
#pragma clang loop unroll(full)
        for (uint j = 0; j < 3u; ++j) {
            const uint4 raw = *reinterpret_cast<device const uint4*>(wp + (i * KW + j) * C);
            bad |= special_h8(raw) ? 1u : 0u;
            const uint b = i * 3u + j;
            const float4 lo = float4(as_type<half4>(raw.xy));
            const float4 hi = float4(as_type<half4>(raw.zw));
            tap[b][0] = lo.x;
            tap[b][1] = lo.y;
            tap[b][2] = lo.z;
            tap[b][3] = lo.w;
            tap[b][4] = hi.x;
            tap[b][5] = hi.y;
            tap[b][6] = hi.z;
            tap[b][7] = hi.w;
        }
    }

#pragma clang loop unroll(full)
    for (uint pc = 0; pc < 6u; ++pc) {
        float acc[8];
#pragma clang loop unroll(full)
        for (uint k = 0; k < 8u; ++k)
            acc[k] = 0.0f;
#pragma clang loop unroll(full)
        for (uint i = 0; i < 3u; ++i) {
            const float gi = GCF[pr][i];
            if (gi == 0.0f)
                continue;
#pragma clang loop unroll(full)
            for (uint j = 0; j < 3u; ++j) {
                const float cf = gi * GCF[pc][j];
                if (cf == 0.0f)
                    continue;
                const uint b = i * 3u + j;
#pragma clang loop unroll(full)
                for (uint k = 0; k < 8u; ++k)
                    acc[k] = fma(cf, tap[b][k], acc[k]);
            }
        }
        const uint p = pr * 6u + pc;
#pragma clang loop unroll(full)
        for (uint k = 0; k < 8u; ++k) {
            const uint e = upack_index(p, oc, c8 + k);
            uwin.write(float4(acc[k], 0.0f, 0.0f, 0.0f), uint2(e & (TEXW - 1u), e >> 5u));
        }
    }

    // simd_any first: a short-circuited collective would only be taken by lane 0
    if (simd_any(bad != 0u) && lane == 0u)
        sbad = 1u;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // every float of the declared scratch has to be overwritten by the producer, so the first
    // threadgroup also clears the flag words the 24 real ones do not use
    if (lid == 0u) {
        const uint e = UFLT + tg.x;
        uwin.write(float4(float(sbad), 0.0f, 0.0f, 0.0f), uint2(e & (TEXW - 1u), e >> 5u));
        for (uint q = NFIL; q < USZ - UFLT; ++q)
            uwin.write(float4(0.0f), uint2((UFLT + q) & (TEXW - 1u), (UFLT + q) >> 5u));
    }
}

// Stage the six V planes of one transform row for this thread's (tile, channel quad) and return
// the non-finite bits of the six values it wrote.  Every V value of every row is screened, which
// covers the whole 6x6 window of every tile in the block.
__attribute__((noinline)) static uint stage_row(device const half* x, threadgroup float* sV,
                                                uint pr, uint tile0, uint lid)
{
    const uint tl = lid / NQ;                    // tile inside the block
    const uint cq = lid - tl * NQ;               // channel quad
    uint img, tr, tc;
    tile_decode(tile0 + tl, img, tr, tc);
    // the window of the last tile row/column sticks out of x; clamping the out-of-range rows and
    // columns onto the last real one is exact for every output that is kept, because A^T's
    // coefficient of each discarded position is zero there (those are the outputs past OH / OW)
    const uint rlast = H - 1u - 4u * tr;
    const uint clast = W - 1u - 4u * tc;
    device const half* const base = x + ((img * H + 4u * tr) * W + 4u * tc) * C + 4u * cq;

    // row pass: t[j] = sum_i B^T[pr][i] * d[i][j], all six window columns live as float4
    float4 t[6];
#pragma clang loop unroll(full)
    for (uint j = 0; j < 6u; ++j)
        t[j] = float4(0.0f);
#pragma clang loop unroll(full)
    for (uint i = 0; i < 6u; ++i) {
        const float ci = BTo[pr][i];
        if (ci == 0.0f)
            continue;
        device const half* const rowp = base + min(i, rlast) * (W * C);
#pragma clang loop unroll(full)
        for (uint j = 0; j < 6u; ++j) {
            const float4 d = float4(*reinterpret_cast<device const half4*>(rowp + min(j, clast) * C));
            t[j] = fma(float4(ci), d, t[j]);
        }
    }

    // column pass, factored so the six outputs share the pair sums they are built from
    const float4 p1 = t[1] + t[2];
    const float4 p2 = t[1] - t[2];
    const float4 p3 = t[3] + t[4];
    const float4 p4 = t[4] - t[3];
    const float4 p5 = t[1] - t[3];
    const float4 p6 = t[4] - t[2];
    float4 v[6];
    v[0] = fma(float4(-5.0f), t[2], fma(float4(4.0f), t[0], t[4]));
    v[1] = fma(float4(-4.0f), p1, p3);
    v[2] = fma(float4(4.0f), p2, p4);
    v[3] = fma(float4(-2.0f), p5, p6);
    v[4] = fma(float4(2.0f), p5, p6);
    v[5] = fma(float4(-5.0f), t[3], fma(float4(4.0f), t[1], t[5]));

    threadgroup float* const vb = sV + tl * VSTR + 4u * cq;
    uint bad = 0u;
#pragma clang loop unroll(full)
    for (uint pc = 0; pc < 6u; ++pc) {
        if (special_f4(v[pc]))
            bad = 1u;
        *reinterpret_cast<threadgroup float4*>(vb + pc * (TTG * VSTR)) = v[pc];
    }
    return bad;
}

// One block's worth of work.  A lane owns (drow, dcol) and (drow, dcol + 1) of every accumulator
// fragment: one output tile row and one adjacent output-channel pair, for both tile halves.
static void conv2d_body(device const half* x, device const half* w, device half* y,
                        device const float* uwin, uint tile0, uint oc0,
                        uint lid, uint lane, threadgroup float* sV, threadgroup uint* sflags)
{
    const uint sg = lid >> 5u;
    const uint qid = lane >> 2u;
    const uint drow = (qid & 4u) + ((lane >> 1u) & 3u);
    const uint dcol = (qid & 2u) * 2u + (lane & 1u) * 2u;
    const uint uoc = oc0 + sg * (OCG / 8u);
    const device float* const upos = uwin + (uoc >> 3) * 512u;

    float yacc[4][4][2][2];   // [output row][output column][tile half][channel pair element]
#pragma clang loop unroll(full)
    for (uint q = 0; q < 16u; ++q) {
        yacc[q >> 2u][q & 3u][0][0] = 0.0f;
        yacc[q >> 2u][q & 3u][0][1] = 0.0f;
        yacc[q >> 2u][q & 3u][1][0] = 0.0f;
        yacc[q >> 2u][q & 3u][1][1] = 0.0f;
    }

    uint bad = 0u;
    // unrolled: rolled, the transform coefficients become runtime constant-memory reads and the
    // row costs about 3x (measured), so the pragma is load bearing
#pragma clang loop unroll(full)
    for (uint pr = 0; pr < 6u; ++pr) {
        if (pr != 0u)
            threadgroup_barrier(mem_flags::mem_threadgroup);
        bad |= stage_row(x, sV, pr, tile0, lid);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // the inverse transform is separable: fold this row's six columns into four partial output
        // rows, then apply the row coefficient once
        float col[4][2][2];
#pragma clang loop unroll(full)
        for (uint ox = 0; ox < 4u; ++ox) {
            col[ox][0][0] = 0.0f;
            col[ox][0][1] = 0.0f;
            col[ox][1][0] = 0.0f;
            col[ox][1][1] = 0.0f;
        }
#pragma clang loop unroll(full)
        for (uint pc = 0; pc < 6u; ++pc) {
            const uint p = pr * 6u + pc;
            threadgroup float* const vp = sV + pc * (TTG * VSTR);
            const device float* const up = upos + p * (C * OC);
            simdgroup_float8x8 m0 = simdgroup_float8x8(0.0f);
            simdgroup_float8x8 m1 = simdgroup_float8x8(0.0f);
            simdgroup_float8x8 a0, a1, b0, b1;
            // 8 tiles x 8 channels from threadgroup memory feed two accumulators; one 16-byte
            // device pack builds the operands of two consecutive k steps
#pragma clang loop unroll(full)
            for (uint kp = 0; kp < 4u; ++kp) {
                const float4 pk = *reinterpret_cast<device const float4*>(
                    up + kp * 128u + lane * 4u);
                auto e0 = b0.thread_elements();
                e0[0] = pk.x;
                e0[1] = pk.y;
                b0.thread_elements() = e0;
                auto e1 = b1.thread_elements();
                e1[0] = pk.z;
                e1[1] = pk.w;
                b1.thread_elements() = e1;
                const uint ks = 2u * kp;
                simdgroup_load(a0, vp + ks * 8u, VSTR);
                simdgroup_load(a1, vp + 8u * VSTR + ks * 8u, VSTR);
                simdgroup_multiply_accumulate(m0, a0, b0, m0);
                simdgroup_multiply_accumulate(m1, a1, b0, m1);
                simdgroup_load(a0, vp + (ks + 1u) * 8u, VSTR);
                simdgroup_load(a1, vp + 8u * VSTR + (ks + 1u) * 8u, VSTR);
                simdgroup_multiply_accumulate(m0, a0, b1, m0);
                simdgroup_multiply_accumulate(m1, a1, b1, m1);
            }
            const auto e0 = m0.thread_elements();
            const auto e1 = m1.thread_elements();
#pragma clang loop unroll(full)
            for (uint ox = 0; ox < 4u; ++ox) {
                const float cf = ATo[ox][pc];
                if (cf != 0.0f) {
                    col[ox][0][0] = fma(cf, e0[0], col[ox][0][0]);
                    col[ox][0][1] = fma(cf, e0[1], col[ox][0][1]);
                    col[ox][1][0] = fma(cf, e1[0], col[ox][1][0]);
                    col[ox][1][1] = fma(cf, e1[1], col[ox][1][1]);
                }
            }
        }

#pragma clang loop unroll(full)
        for (uint oy = 0; oy < 4u; ++oy) {
            const float cf = ATo[oy][pr];
            if (cf != 0.0f) {
#pragma clang loop unroll(full)
                for (uint ox = 0; ox < 4u; ++ox) {
                    yacc[oy][ox][0][0] = fma(cf, col[ox][0][0], yacc[oy][ox][0][0]);
                    yacc[oy][ox][0][1] = fma(cf, col[ox][0][1], yacc[oy][ox][0][1]);
                    yacc[oy][ox][1][0] = fma(cf, col[ox][1][0], yacc[oy][ox][1][0]);
                    yacc[oy][ox][1][1] = fma(cf, col[ox][1][1], yacc[oy][ox][1][1]);
                }
            }
        }
    }

    // every staged V value of every row has been screened by now; vote the verdict through
    // threadgroup memory so the whole block takes the same branch
    if (simd_any(bad != 0u) && lane == 0u)
        sflags[sg] = 1u;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint any = 0u;
#pragma clang loop unroll(full)
    for (uint i = 0; i < 8u; ++i)
        any |= sflags[i];
    if (any != 0u) {
        conv2d_taps(x, w, y, tile0, oc0, lid);
        return;
    }

    // epilogue: the single rounding to half, written as half2 straight into y
#pragma clang loop unroll(full)
    for (uint f = 0; f < 2u; ++f) {
        uint img, tr, tc;
        tile_decode(tile0 + 8u * f + drow, img, tr, tc);
        device half* const yp = y + ((img * OH + 4u * tr) * OW + 4u * tc) * OC + uoc + dcol;
#pragma clang loop unroll(full)
        for (uint oy = 0; oy < 4u; ++oy) {
            if (4u * tr + oy < OH) {
#pragma clang loop unroll(full)
                for (uint ox = 0; ox < 4u; ++ox) {
                    if (4u * tc + ox < OW)
                        *reinterpret_cast<device half2*>(yp + (oy * OW + ox) * OC) =
                            half2(half(yacc[oy][ox][f][0]), half(yacc[oy][ox][f][1]));
                }
            }
        }
    }
}

// A non-finite filter tap anywhere in w changes how the oracle sums every output, so every block
// consumes the flag words stage 1 left behind.
static inline bool filter_dirty(device const float* uwin)
{
    float filtnf = 0.0f;
#pragma clang loop unroll(full)
    for (uint i = 0; i < NFIL / 4u; ++i) {
        const float4 v = *reinterpret_cast<device const float4*>(uwin + UFLT + 4u * i);
        filtnf += (v.x + v.y) + (v.z + v.w);
    }
    return filtnf != 0.0f;
}

[[max_total_threads_per_threadgroup(NTHR)]]
kernel void conv2d_main(device const half* x [[buffer(0)]],
                        device const half* w [[buffer(1)]],
                        device half* y [[buffer(2)]],
                        device const float* uwin [[buffer(3)]],
                        uint3 tg [[threadgroup_position_in_grid]],
                        uint lid [[thread_index_in_threadgroup]],
                        uint lane [[thread_index_in_simdgroup]])
{
    const uint tile0 = tg.x * TTG;
    const uint oc0 = tg.y * OCG;

    threadgroup float sV[6u * TTG * VSTR];   // 6 positions x 16 tiles x 64 channels
    threadgroup uint sflags[8];

    if (lid < 8u)
        sflags[lid] = 0u;   // ordered by the first staged row's barrier
    if (filter_dirty(uwin)) {
        conv2d_taps(x, w, y, tile0, oc0, lid);
        return;
    }
    conv2d_body(x, w, y, uwin, tile0, oc0, lid, lane, sV, sflags);
}
