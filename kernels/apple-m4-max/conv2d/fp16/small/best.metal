/* taic-launch
math fast
launch conv2d_kernel grid=64,1,1 threads=128,1,1
*/

// conv2d, NHWC 3x3 stride 1, no pad, no bias.  x[4,18,18,32] w[64,3,3,32] y[4,16,16,64].
//
// Winograd F(2x2,3x3) on fp32 8x8x8 simdgroup matrices, fp32 accumulate, one fp16
// rounding at the very end.
//
// Warp tile: eight output tiles (one output row pair of one image) x eight output
// channels.  The sixteen coordinate fragments M[i][j] are therefore the whole accumulator
// set, and both transforms stay inside the lane -- a lane holds the same (tile, oc pair)
// position in every coordinate fragment, so the fold A^T M A needs no shuffle, no
// threadgroup round trip and no second pass.
//
// 64 threadgroups x 4 warps = 256 warps = 32 tile rows x 8 channel groups.  256 warps is
// the saturation point of this chip's fp32 matrix pipe: measured 11.3 GMMA/s at 256 warps
// but only 6.8 at 128 and 3.7 at 64 (p_mma.py).  That is why a bigger Winograd tile loses:
// F(2x4) would be 12288 MAC-blocks on 128 warps and F(4x4) 9216 on 64, i.e. slower than
// the 16384 here even though it does less arithmetic.
//
// A threadgroup is four warps of one channel group, so the weight transform
// U = G g G^T (16 coords x 32 channels x 8 oc, fp32, row stride 8) is built once per
// threadgroup and all four warps read their B fragments out of it.  16 KB of static
// threadgroup memory keeps two threadgroups resident per core.
//
// The kernel is latency-bound in two places, and both are pipelined here: the window of
// the next k block is fetched before the current block's multiplies (two raw windows in
// flight), and both weight passes' nine taps are fetched before either pass' arithmetic.
// Measured together: 2.31 -> 2.25 us GPU time.
//
// Non-finite input: a factorised transform does not place Inf/NaN where the oracle's
// tap-by-tap fp32 sum places them, so a warp whose folded outputs contain a non-finite
// value recomputes its own 2x16x8 slab with the direct sum.

#define K_HIN 18u
#define K_CH 32u
#define K_OW 16u
#define K_OC 64u
#define K_WR 288u
#define K_NC 16u
#define K_UC 256u

// 256 outputs = 2 rows x 16 cols x 8 channels.  A lane owns one (row, col) output of the
// slab and walks the eight channels; only reached when the transform saw Inf/NaN.
static __attribute__((noinline)) void
k_direct(device const half *x, device const half *w, device half *y,
         uint img, uint ohs, uint oc0, uint lane)
{
    const uint r = lane >> 4u;
    const uint c = lane & 15u;
    const device half *xp = x + ((img * K_HIN + ohs + r) * K_HIN + c) * K_CH;
    for (uint oc = 0u; oc < 8u; ++oc) {
        const device half *wp = w + (oc0 + oc) * K_WR;
        float acc = 0.0f;
        for (uint kh = 0u; kh < 3u; ++kh) {
            for (uint kw = 0u; kw < 3u; ++kw) {
                const device half *xs = xp + (kh * K_HIN + kw) * K_CH;
                const device half *ws = wp + (kh * 3u + kw) * K_CH;
                for (uint c2 = 0u; c2 < K_CH; ++c2) {
                    acc = fma(float(xs[c2]), float(ws[c2]), acc);
                }
            }
        }
        y[((img * 256u + (ohs + r) * K_OW + c) * K_OC) + oc0 + oc] = (half)acc;
    }
}

[[max_total_threads_per_threadgroup(128)]]
kernel void conv2d_kernel(
    device const half* x [[buffer(0)]],
    device const half* w [[buffer(1)]],
    device half* y [[buffer(2)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint3 tid [[thread_position_in_threadgroup]]
)
{
    threadgroup float u[K_NC * K_UC];        // [coord][k-block][channel-in-block][oc]

    const uint lane = tid.x & 31u;
    const uint warp = tid.x >> 5u;
    const uint oc0  = (tgid.x & 7u) * 8u;                 // this threadgroup's 8 channels
    const uint tro  = ((tgid.x >> 3u) << 2u) + warp;      // tile row pair 0..31
    const uint img  = tro >> 3u;
    const uint ohs  = (tro & 7u) * 2u;

    // fragment position: row = tile inside the row pair, column pair = channel pair
    const uint qid = lane >> 2u;
    const uint tr  = ((lane >> 1u) & 3u) | (qid & 4u);
    const uint kp  = (lane & 1u) | (qid & 2u);
    const device half *xg = x + ((img * K_HIN + ohs) * K_HIN + 2u * tr) * K_CH + 2u * kp;

    // ---- weight transform U = G g G^T, one (oc_local, channel) pair per thread pass ----
    // Both passes' taps are fetched up front: 18 scalar loads whose latency then overlaps
    // the first pass' arithmetic instead of splitting the two passes with a stall.
    float gg[2][3][3];
    #pragma unroll
    for (uint pass = 0u; pass < 2u; ++pass) {
        const uint item = tid.x + pass * 128u;
        const device half *wp = w + (oc0 + (item >> 5u)) * K_WR + (item & 31u);
        #pragma unroll
        for (uint kh = 0u; kh < 3u; ++kh) {
            #pragma unroll
            for (uint kw = 0u; kw < 3u; ++kw) {
                gg[pass][kh][kw] = float(wp[(kh * 3u + kw) * K_CH]);
            }
        }
    }
    #pragma unroll
    for (uint pass = 0u; pass < 2u; ++pass) {
        const uint item = tid.x + pass * 128u;
        const uint ocl = item >> 5u;             // 0..7
        const uint ch  = item & 31u;             // 0..31
        float v[4][3];                           // rows of G g
        #pragma unroll
        for (uint kw = 0u; kw < 3u; ++kw) {
            const float a = gg[pass][0][kw], b = gg[pass][1][kw], c = gg[pass][2][kw];
            v[0][kw] = a;
            v[1][kw] = (a + b + c) * 0.5f;
            v[2][kw] = (a - b + c) * 0.5f;
            v[3][kw] = c;
        }
        // channel ch belongs to k block ch/8 at row ch%8 of that block's B fragment
        threadgroup float *dst = u + (ch >> 3u) * 64u + (ch & 7u) * 8u + ocl;
        #pragma unroll
        for (uint i = 0u; i < 4u; ++i) {
            const float a0 = v[i][0], a1 = v[i][1], a2 = v[i][2];
            dst[(i * 4u + 0u) * K_UC] = a0;
            dst[(i * 4u + 1u) * K_UC] = (a0 + a1 + a2) * 0.5f;
            dst[(i * 4u + 2u) * K_UC] = (a0 - a1 + a2) * 0.5f;
            dst[(i * 4u + 3u) * K_UC] = a2;
        }
    }

    // The first two k blocks' 4x4 windows go out before the barrier and before the
    // multiplies: their latency overlaps the staging arithmetetic and then the first
    // block's MMAs.  Two raw windows are 32 registers, which is what fits.
    uint win[4][4], nxt[4][4];
    #pragma unroll
    for (uint r = 0u; r < 4u; ++r) {
        #pragma unroll
        for (uint s = 0u; s < 4u; ++s) {
            win[r][s] = *(device const uint *)(xg + (r * K_HIN + s) * K_CH);
            nxt[r][s] = *(device const uint *)(xg + (r * K_HIN + s) * K_CH + 8u);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    simdgroup_float8x8 acc[K_NC];
    #pragma unroll
    for (uint i = 0u; i < K_NC; ++i) acc[i] = simdgroup_float8x8(0.0f);
    simdgroup_float8x8 A, B;

    // ---- k block 0, window already in registers ---------------------------------------
    {
        float2 d[4][4], t[4][4], m[4][4];
        #pragma unroll
        for (uint r = 0u; r < 4u; ++r) {
            #pragma unroll
            for (uint s = 0u; s < 4u; ++s) {
                d[r][s] = float2(as_type<half2>(win[r][s]));
            }
        }
        #pragma unroll
        for (uint r = 0u; r < 4u; ++r) {
            t[r][0] = d[r][0] - d[r][2];
            t[r][1] = d[r][1] + d[r][2];
            t[r][2] = d[r][2] - d[r][1];
            t[r][3] = d[r][1] - d[r][3];
        }
        #pragma unroll
        for (uint j = 0u; j < 4u; ++j) {
            m[0][j] = t[0][j] - t[2][j];
            m[1][j] = t[1][j] + t[2][j];
            m[2][j] = t[2][j] - t[1][j];
            m[3][j] = t[1][j] - t[3][j];
        }
        #pragma unroll
        for (uint i = 0u; i < K_NC; ++i) {
            *(thread float2 *)&A.thread_elements() = m[i >> 2u][i & 3u];
            simdgroup_load(B, (threadgroup const float *)(u + i * K_UC), 8,
                           ulong2(0, 0), false);
            simdgroup_multiply_accumulate(acc[i], A, B, acc[i]);
        }
    }
    #pragma unroll
    for (uint r = 0u; r < 4u; ++r) {
        #pragma unroll
        for (uint s = 0u; s < 4u; ++s) {
            win[r][s] = nxt[r][s];
        }
    }

    // ---- k blocks 1..3, each with the next block's window already in flight -----------
    #pragma unroll
    for (uint cb = 1u; cb < 4u; ++cb) {
        if (cb < 3u) {
            #pragma unroll
            for (uint r = 0u; r < 4u; ++r) {
                #pragma unroll
                for (uint s = 0u; s < 4u; ++s) {
                    nxt[r][s] = *(device const uint *)(xg + (r * K_HIN + s) * K_CH
                                                       + (cb + 1u) * 8u);
                }
            }
        }
        float2 d[4][4], t[4][4], m[4][4];
        #pragma unroll
        for (uint r = 0u; r < 4u; ++r) {
            #pragma unroll
            for (uint s = 0u; s < 4u; ++s) {
                d[r][s] = float2(as_type<half2>(win[r][s]));
            }
        }
        #pragma unroll
        for (uint r = 0u; r < 4u; ++r) {
            t[r][0] = d[r][0] - d[r][2];
            t[r][1] = d[r][1] + d[r][2];
            t[r][2] = d[r][2] - d[r][1];
            t[r][3] = d[r][1] - d[r][3];
        }
        #pragma unroll
        for (uint j = 0u; j < 4u; ++j) {
            m[0][j] = t[0][j] - t[2][j];
            m[1][j] = t[1][j] + t[2][j];
            m[2][j] = t[2][j] - t[1][j];
            m[3][j] = t[1][j] - t[3][j];
        }
        #pragma unroll
        for (uint i = 0u; i < K_NC; ++i) {
            *(thread float2 *)&A.thread_elements() = m[i >> 2u][i & 3u];
            simdgroup_load(B, (threadgroup const float *)(u + i * K_UC + cb * 64u), 8,
                           ulong2(0, 0), false);
            simdgroup_multiply_accumulate(acc[i], A, B, acc[i]);
        }
        #pragma unroll
        for (uint r = 0u; r < 4u; ++r) {
            #pragma unroll
            for (uint s = 0u; s < 4u; ++s) {
                win[r][s] = nxt[r][s];
            }
        }
    }

    // ---- fold A^T M A straight off the fragments, rounded to fp16 once -----------------
    float2 lo[4], hi[4];
    #pragma unroll
    for (uint j = 0u; j < 4u; ++j) {
        const float2 a0 = *(thread float2 *)&acc[j].thread_elements();
        const float2 a1 = *(thread float2 *)&acc[4u + j].thread_elements();
        const float2 a2 = *(thread float2 *)&acc[8u + j].thread_elements();
        const float2 a3 = *(thread float2 *)&acc[12u + j].thread_elements();
        lo[j] = a0 + a1 + a2;                    // A^T row 0: +
        hi[j] = a1 - a2 - a3;                    // A^T row 1: + - -
    }
    float2 out[4];
    out[0] = lo[0] + lo[1] + lo[2];
    out[1] = lo[1] - lo[2] - lo[3];
    out[2] = hi[0] + hi[1] + hi[2];
    out[3] = hi[1] - hi[2] - hi[3];

    uint bad = 0u;
    #pragma unroll
    for (uint i = 0u; i < 4u; ++i) {
        bad |= ((as_type<uint>(out[i].x) & 0x7F800000u) == 0x7F800000u) ? 1u : 0u;
        bad |= ((as_type<uint>(out[i].y) & 0x7F800000u) == 0x7F800000u) ? 1u : 0u;
    }
    if (simd_any(bad != 0u)) {
        k_direct(x, w, y, img, ohs, oc0, lane);
        return;
    }

    device half *yp = y + (img * 256u + ohs * K_OW + 2u * tr) * K_OC + oc0 + 2u * kp;
    #pragma unroll
    for (uint i = 0u; i < 4u; ++i) {
        *(device half2 *)(yp + ((i >> 1u) * K_OW + (i & 1u)) * K_OC) =
            half2(out[i].x, out[i].y);
    }
}
