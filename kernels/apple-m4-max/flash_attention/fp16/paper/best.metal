/* taic-launch
launch flash_attention_kernel grid=32,16,1 threads=64,1,1
*/
// FlashAttention forward (non-causal), fp16 in / fp16 out.  fp32 scores, fp32 statistics, fp32
// accumulator, output rounded exactly once, P never rounded.
//
// r8 db32: v_guard structure, but K^T is staged as two 4096 B slots of 32 keys each, so one
// threadgroup barrier covers two 16-key softmax blocks (32 barriers total instead of 64).
// Register set, mma stream, softmax scalar work and the guard/redo safety net are unchanged:
// the score fragments of the first 16-key sub-block are reused for the second.
constant constexpr uint LDK = 32u;                    // K^T row stride, halves (= keys per slot)
constant constexpr uint HB  = 32u;                    // keys per staged slot
constant constexpr uint NSB = S / HB;                 // 32-key super-blocks
constant constexpr uint KSL = D * LDK;                // halves per slot (4096 B)
constant constexpr float QSC = SM_SCALE * 1.4426950408889634f;
constant constexpr float LMAX = 1e30f;   // per-lane partial sum: |acc| <= l*65504 stays finite
constant constexpr float AMAX = 1e37f;   // fp32 max is 3.4e38; NaN and Inf both fail this test


kernel void flash_attention_kernel(
    device const half* q [[buffer(0)]],
    device const half* k [[buffer(1)]],
    device const half* v [[buffer(2)]],
    device half*       o [[buffer(3)]],
    uint2 tg  [[threadgroup_position_in_grid]],
    uint  lid [[thread_index_in_threadgroup]],
    uint  sgi [[simdgroup_index_in_threadgroup]])
{
    threadgroup half Kt[2u * KSL] __attribute__((aligned(16)));   // 8192 B

    const ulong obase = (ulong)tg.y * (ulong)(S * D);
    const uint  qrow  = tg.x * 32u + sgi * 16u;

    const device half* qp = q + obase + (ulong)qrow * D;
    const device half* kp = k + obase;
    const device half* vp = v + obase;
    device half*       op = o + obase + (ulong)qrow * D;

    // with 2 SIMD groups each group transposes one 16-key half of the slot over all eight d tiles
    const uint kKey = sgi * 16u;
    // V fragment lane mapping: the lane owns (row, col), (row, col+1)
    const uint lane = lid & 31u;
    const uint qid  = lane >> 2u;
    const uint frow = (qid & 4u) | ((lane >> 1u) & 3u);
    const uint fcol = ((qid & 2u) << 1u) | ((lane & 1u) << 1u);

    simdgroup_half8x8 q0[8], q1[8];
#pragma unroll
    for (uint i = 0; i < 8u; ++i) {
        simdgroup_load(q0[i], qp + i * 8u, D);
        simdgroup_load(q1[i], qp + 8u * D + i * 8u, D);
    }

    simdgroup_float8x8 acc0[8], acc1[8];
#pragma unroll
    for (uint c = 0; c < 8u; ++c) {
        acc0[c] = simdgroup_float8x8(0.0f);
        acc1[c] = simdgroup_float8x8(0.0f);
    }

    float m0 = -INFINITY, l0 = 0.0f;      // reference shift and partial sum, row tile 0
    float m1 = -INFINITY, l1 = 0.0f;      // ... row tile 1

    // prologue: publish super-block 0 into slot 0
#pragma unroll
    for (uint i = 0; i < 8u; ++i) {
        simdgroup_half8x8 f0, f1;
        simdgroup_load(f0, kp + (ulong)kKey * D + (uint64_t)(i * 8u), D);
        simdgroup_load(f1, kp + (ulong)(kKey + 8u) * D + (uint64_t)(i * 8u), D);
        simdgroup_store(f0, Kt + (i * 8u) * LDK + kKey, LDK, ulong2(0, 0), true);
        simdgroup_store(f1, Kt + (i * 8u) * LDK + kKey + 8u, LDK, ulong2(0, 0), true);
    }

    // ---------------- first super-block, peeled: sub-block 0 defines the reference shift ----
    {
        const uint srcN = (1u < NSB) ? 1u : (NSB - 1u);
        threadgroup_barrier(mem_flags::mem_threadgroup);
#pragma unroll
        for (uint i = 0; i < 8u; ++i) {
            simdgroup_half8x8 f0, f1;
            simdgroup_load(f0, kp + (ulong)(srcN * HB + kKey) * D + (uint64_t)(i * 8u), D);
            simdgroup_load(f1, kp + (ulong)(srcN * HB + kKey + 8u) * D + (uint64_t)(i * 8u), D);
            simdgroup_store(f0, Kt + KSL + (i * 8u) * LDK + kKey, LDK, ulong2(0, 0), true);
            simdgroup_store(f1, Kt + KSL + (i * 8u) * LDK + kKey + 8u, LDK, ulong2(0, 0), true);
        }

        // --- sub-block 0 (keys 0..15): exact reference step, fixes m --------------------
        simdgroup_float8x8 sc0a = simdgroup_float8x8(0.0f), sc0b = simdgroup_float8x8(0.0f);
        simdgroup_float8x8 sc1a = simdgroup_float8x8(0.0f), sc1b = simdgroup_float8x8(0.0f);
#pragma unroll
        for (uint i = 0; i < 8u; ++i) {
            simdgroup_half8x8 kb0, kb1;
            simdgroup_load(kb0, Kt + i * 8u * LDK, LDK);
            simdgroup_load(kb1, Kt + i * 8u * LDK + 8u, LDK);
            simdgroup_multiply_accumulate(sc0a, q0[i], kb0, sc0a);
            simdgroup_multiply_accumulate(sc0b, q0[i], kb1, sc0b);
            simdgroup_multiply_accumulate(sc1a, q1[i], kb0, sc1a);
            simdgroup_multiply_accumulate(sc1b, q1[i], kb1, sc1b);
        }

        float2 e0 = *(thread float2*)&sc0a.thread_elements();
        float2 e1 = *(thread float2*)&sc0b.thread_elements();
        float2 e2 = *(thread float2*)&sc1a.thread_elements();
        float2 e3 = *(thread float2*)&sc1b.thread_elements();
        float s0[4] = { e0.x, e0.y, e1.x, e1.y };
        float s1[4] = { e2.x, e2.y, e3.x, e3.y };
        float p0[4], p1[4];

        float mx0 = max(max(s0[0], s0[1]), max(s0[2], s0[3]));
        mx0 = max(mx0, simd_shuffle_xor(mx0, 1u));
        mx0 = max(mx0, simd_shuffle_xor(mx0, 8u));
        const float mn0 = max(m0, mx0 * QSC);
        const float al0 = exp2(m0 - mn0);
        float ps0 = 0.0f;
#pragma unroll
        for (uint j = 0; j < 4u; ++j) {
            p0[j] = exp2(fma(s0[j], QSC, -mn0));
            ps0 += p0[j];
        }
        l0 = l0 * al0 + ps0;
        m0 = mn0;
#pragma unroll
        for (uint c = 0; c < 8u; ++c) {
            float2 v2 = *(thread float2*)&acc0[c].thread_elements();
            v2 *= al0;
            *(thread float2*)&acc0[c].thread_elements() = v2;
        }

        float mx1 = max(max(s1[0], s1[1]), max(s1[2], s1[3]));
        mx1 = max(mx1, simd_shuffle_xor(mx1, 1u));
        mx1 = max(mx1, simd_shuffle_xor(mx1, 8u));
        const float mn1 = max(m1, mx1 * QSC);
        const float al1 = exp2(m1 - mn1);
        float ps1 = 0.0f;
#pragma unroll
        for (uint j = 0; j < 4u; ++j) {
            p1[j] = exp2(fma(s1[j], QSC, -mn1));
            ps1 += p1[j];
        }
        l1 = l1 * al1 + ps1;
        m1 = mn1;
#pragma unroll
        for (uint c = 0; c < 8u; ++c) {
            float2 v2 = *(thread float2*)&acc1[c].thread_elements();
            v2 *= al1;
            *(thread float2*)&acc1[c].thread_elements() = v2;
        }

        *(thread float2*)&sc0a.thread_elements() = float2(p0[0], p0[1]);
        *(thread float2*)&sc0b.thread_elements() = float2(p0[2], p0[3]);
        *(thread float2*)&sc1a.thread_elements() = float2(p1[0], p1[1]);
        *(thread float2*)&sc1b.thread_elements() = float2(p1[2], p1[3]);

#pragma unroll
        for (uint j = 0; j < 2u; ++j) {
            const device half* vrow = vp + (ulong)(j * 8u + frow) * D + fcol;
#pragma unroll
            for (uint c = 0; c < 8u; ++c) {
                simdgroup_float8x8 vt;
                *(thread float2*)&vt.thread_elements() =
                    float2(*(const device half2*)(vrow + c * 8u));
                if (j == 0u) {
                    simdgroup_multiply_accumulate(acc0[c], sc0a, vt, acc0[c]);
                    simdgroup_multiply_accumulate(acc1[c], sc1a, vt, acc1[c]);
                } else {
                    simdgroup_multiply_accumulate(acc0[c], sc0b, vt, acc0[c]);
                    simdgroup_multiply_accumulate(acc1[c], sc1b, vt, acc1[c]);
                }
            }
        }

        // --- sub-block 1 (keys 16..31): fast step against the fixed shift --------------
        sc0a = simdgroup_float8x8(0.0f); sc0b = simdgroup_float8x8(0.0f);
        sc1a = simdgroup_float8x8(0.0f); sc1b = simdgroup_float8x8(0.0f);
#pragma unroll
        for (uint i = 0; i < 8u; ++i) {
            simdgroup_half8x8 kb0, kb1;
            simdgroup_load(kb0, Kt + i * 8u * LDK + 16u, LDK);
            simdgroup_load(kb1, Kt + i * 8u * LDK + 24u, LDK);
            simdgroup_multiply_accumulate(sc0a, q0[i], kb0, sc0a);
            simdgroup_multiply_accumulate(sc0b, q0[i], kb1, sc0b);
            simdgroup_multiply_accumulate(sc1a, q1[i], kb0, sc1a);
            simdgroup_multiply_accumulate(sc1b, q1[i], kb1, sc1b);
        }

        e0 = *(thread float2*)&sc0a.thread_elements();
        e1 = *(thread float2*)&sc0b.thread_elements();
        e2 = *(thread float2*)&sc1a.thread_elements();
        e3 = *(thread float2*)&sc1b.thread_elements();
        s0[0] = e0.x; s0[1] = e0.y; s0[2] = e1.x; s0[3] = e1.y;
        s1[0] = e2.x; s1[1] = e2.y; s1[2] = e3.x; s1[3] = e3.y;

        ps0 = 0.0f; ps1 = 0.0f;
#pragma unroll
        for (uint j = 0; j < 4u; ++j) {
            p0[j] = exp2(fma(s0[j], QSC, -m0));
            ps0 += p0[j];
            p1[j] = exp2(fma(s1[j], QSC, -m1));
            ps1 += p1[j];
        }
        l0 += ps0;
        l1 += ps1;

        *(thread float2*)&sc0a.thread_elements() = float2(p0[0], p0[1]);
        *(thread float2*)&sc0b.thread_elements() = float2(p0[2], p0[3]);
        *(thread float2*)&sc1a.thread_elements() = float2(p1[0], p1[1]);
        *(thread float2*)&sc1b.thread_elements() = float2(p1[2], p1[3]);

#pragma unroll
        for (uint j = 0; j < 2u; ++j) {
            const device half* vrow = vp + (ulong)(16u + j * 8u + frow) * D + fcol;
#pragma unroll
            for (uint c = 0; c < 8u; ++c) {
                simdgroup_float8x8 vt;
                *(thread float2*)&vt.thread_elements() =
                    float2(*(const device half2*)(vrow + c * 8u));
                if (j == 0u) {
                    simdgroup_multiply_accumulate(acc0[c], sc0a, vt, acc0[c]);
                    simdgroup_multiply_accumulate(acc1[c], sc1a, vt, acc1[c]);
                } else {
                    simdgroup_multiply_accumulate(acc0[c], sc0b, vt, acc0[c]);
                    simdgroup_multiply_accumulate(acc1[c], sc1b, vt, acc1[c]);
                }
            }
        }
    }

    // ---------------- super-blocks 1..NSB-1: branch-free, one barrier per 32 keys ----------
    for (uint sb = 1; sb < NSB; ++sb) {
        threadgroup half* Kc = Kt + (sb & 1u) * KSL;
        threadgroup half* Kn = Kt + ((sb + 1u) & 1u) * KSL;
        const device half* vb = vp + (ulong)sb * (HB * D);
        // clamped: on the last super-block the publish target is never read
        const uint srcN = (sb + 1u < NSB) ? (sb + 1u) : (NSB - 1u);

        threadgroup_barrier(mem_flags::mem_threadgroup);
#pragma unroll
        for (uint i = 0; i < 8u; ++i) {
            simdgroup_half8x8 f0, f1;
            simdgroup_load(f0, kp + (ulong)(srcN * HB + kKey) * D + (uint64_t)(i * 8u), D);
            simdgroup_load(f1, kp + (ulong)(srcN * HB + kKey + 8u) * D + (uint64_t)(i * 8u), D);
            simdgroup_store(f0, Kn + (i * 8u) * LDK + kKey, LDK, ulong2(0, 0), true);
            simdgroup_store(f1, Kn + (i * 8u) * LDK + kKey + 8u, LDK, ulong2(0, 0), true);
        }

#pragma unroll
        for (uint sub = 0; sub < 2u; ++sub) {
            threadgroup half* Ks = Kc + sub * 16u;
            simdgroup_float8x8 sc0a = simdgroup_float8x8(0.0f), sc0b = simdgroup_float8x8(0.0f);
            simdgroup_float8x8 sc1a = simdgroup_float8x8(0.0f), sc1b = simdgroup_float8x8(0.0f);
#pragma unroll
            for (uint i = 0; i < 8u; ++i) {
                simdgroup_half8x8 kb0, kb1;
                simdgroup_load(kb0, Ks + i * 8u * LDK, LDK);
                simdgroup_load(kb1, Ks + i * 8u * LDK + 8u, LDK);
                simdgroup_multiply_accumulate(sc0a, q0[i], kb0, sc0a);
                simdgroup_multiply_accumulate(sc0b, q0[i], kb1, sc0b);
                simdgroup_multiply_accumulate(sc1a, q1[i], kb0, sc1a);
                simdgroup_multiply_accumulate(sc1b, q1[i], kb1, sc1b);
            }

            float2 e0 = *(thread float2*)&sc0a.thread_elements();
            float2 e1 = *(thread float2*)&sc0b.thread_elements();
            float2 e2 = *(thread float2*)&sc1a.thread_elements();
            float2 e3 = *(thread float2*)&sc1b.thread_elements();
            float s0[4] = { e0.x, e0.y, e1.x, e1.y };
            float s1[4] = { e2.x, e2.y, e3.x, e3.y };
            float p0[4], p1[4];

            float ps0 = 0.0f, ps1 = 0.0f;
#pragma unroll
            for (uint j = 0; j < 4u; ++j) {
                p0[j] = exp2(fma(s0[j], QSC, -m0));
                ps0 += p0[j];
                p1[j] = exp2(fma(s1[j], QSC, -m1));
                ps1 += p1[j];
            }
            l0 += ps0;
            l1 += ps1;

            *(thread float2*)&sc0a.thread_elements() = float2(p0[0], p0[1]);
            *(thread float2*)&sc0b.thread_elements() = float2(p0[2], p0[3]);
            *(thread float2*)&sc1a.thread_elements() = float2(p1[0], p1[1]);
            *(thread float2*)&sc1b.thread_elements() = float2(p1[2], p1[3]);

#pragma unroll
            for (uint j = 0; j < 2u; ++j) {
                const device half* vrow = vb + (ulong)(sub * 16u + j * 8u + frow) * D + fcol;
#pragma unroll
                for (uint c = 0; c < 8u; ++c) {
                    simdgroup_float8x8 vt;
                    *(thread float2*)&vt.thread_elements() =
                        float2(*(const device half2*)(vrow + c * 8u));
                    if (j == 0u) {
                        simdgroup_multiply_accumulate(acc0[c], sc0a, vt, acc0[c]);
                        simdgroup_multiply_accumulate(acc1[c], sc1a, vt, acc1[c]);
                    } else {
                        simdgroup_multiply_accumulate(acc0[c], sc0b, vt, acc0[c]);
                        simdgroup_multiply_accumulate(acc1[c], sc1b, vt, acc1[c]);
                    }
                }
            }
        }
    }

    // ---------------- is the fixed shift still valid? ----------------
    bool ok = (l0 < LMAX) && (l1 < LMAX);
#pragma unroll
    for (uint c = 0; c < 8u; ++c) {
        float2 e = *(thread float2*)&acc0[c].thread_elements();
        float2 f = *(thread float2*)&acc1[c].thread_elements();
        ok = ok && fabs(e.x) < AMAX && fabs(e.y) < AMAX && fabs(f.x) < AMAX && fabs(f.y) < AMAX;
    }
    if (simd_any(!ok)) {
#pragma unroll
        for (uint c = 0; c < 8u; ++c) {
            acc0[c] = simdgroup_float8x8(0.0f);
            acc1[c] = simdgroup_float8x8(0.0f);
        }
        m0 = -INFINITY; l0 = 0.0f;
        m1 = -INFINITY; l1 = 0.0f;
        for (uint hb = 0; hb < S / 16u; ++hb) {
            const device half* vb = vp + (ulong)hb * (16u * D);

        simdgroup_float8x8 sc0a = simdgroup_float8x8(0.0f), sc0b = simdgroup_float8x8(0.0f);
        simdgroup_float8x8 sc1a = simdgroup_float8x8(0.0f), sc1b = simdgroup_float8x8(0.0f);
        const device half* kbp = kp + (ulong)(hb * 16u) * D;
#pragma unroll
        for (uint i = 0; i < 8u; ++i) {
            simdgroup_half8x8 kb0, kb1;
            simdgroup_load(kb0, kbp + (uint64_t)(i * 8u), D, ulong2(0, 0), true);
            simdgroup_load(kb1, kbp + (uint64_t)(8u * D) + (uint64_t)(i * 8u), D, ulong2(0, 0), true);
            simdgroup_multiply_accumulate(sc0a, q0[i], kb0, sc0a);
            simdgroup_multiply_accumulate(sc0b, q0[i], kb1, sc0b);
            simdgroup_multiply_accumulate(sc1a, q1[i], kb0, sc1a);
            simdgroup_multiply_accumulate(sc1b, q1[i], kb1, sc1b);
        }

        float2 e0 = *(thread float2*)&sc0a.thread_elements();
        float2 e1 = *(thread float2*)&sc0b.thread_elements();
        float2 e2 = *(thread float2*)&sc1a.thread_elements();
        float2 e3 = *(thread float2*)&sc1b.thread_elements();
        float s0[4] = { e0.x, e0.y, e1.x, e1.y };
        float s1[4] = { e2.x, e2.y, e3.x, e3.y };
        float p0[4], p1[4];

        float mx0 = max(max(s0[0], s0[1]), max(s0[2], s0[3]));
        mx0 = max(mx0, simd_shuffle_xor(mx0, 1u));
        mx0 = max(mx0, simd_shuffle_xor(mx0, 8u));
        const float mn0 = max(m0, mx0 * QSC);
        const float al0 = exp2(m0 - mn0);
        float ps0 = 0.0f;
#pragma unroll
        for (uint j = 0; j < 4u; ++j) {
            p0[j] = exp2(fma(s0[j], QSC, -mn0));
            ps0 += p0[j];
        }
        l0 = l0 * al0 + ps0;
        m0 = mn0;
#pragma unroll
        for (uint c = 0; c < 8u; ++c) {
            float2 v2 = *(thread float2*)&acc0[c].thread_elements();
            v2 *= al0;
            *(thread float2*)&acc0[c].thread_elements() = v2;
        }

        float mx1 = max(max(s1[0], s1[1]), max(s1[2], s1[3]));
        mx1 = max(mx1, simd_shuffle_xor(mx1, 1u));
        mx1 = max(mx1, simd_shuffle_xor(mx1, 8u));
        const float mn1 = max(m1, mx1 * QSC);
        const float al1 = exp2(m1 - mn1);
        float ps1 = 0.0f;
#pragma unroll
        for (uint j = 0; j < 4u; ++j) {
            p1[j] = exp2(fma(s1[j], QSC, -mn1));
            ps1 += p1[j];
        }
        l1 = l1 * al1 + ps1;
        m1 = mn1;
#pragma unroll
        for (uint c = 0; c < 8u; ++c) {
            float2 v2 = *(thread float2*)&acc1[c].thread_elements();
            v2 *= al1;
            *(thread float2*)&acc1[c].thread_elements() = v2;
        }

        *(thread float2*)&sc0a.thread_elements() = float2(p0[0], p0[1]);
        *(thread float2*)&sc0b.thread_elements() = float2(p0[2], p0[3]);
        *(thread float2*)&sc1a.thread_elements() = float2(p1[0], p1[1]);
        *(thread float2*)&sc1b.thread_elements() = float2(p1[2], p1[3]);

#pragma unroll
        for (uint j = 0; j < 2u; ++j) {
            const device half* vrow = vb + (ulong)(j * 8u + frow) * D + fcol;
#pragma unroll
            for (uint c = 0; c < 8u; ++c) {
                simdgroup_float8x8 vt;
                *(thread float2*)&vt.thread_elements() =
                    float2(*(const device half2*)(vrow + c * 8u));
                if (j == 0u) {
                    simdgroup_multiply_accumulate(acc0[c], sc0a, vt, acc0[c]);
                    simdgroup_multiply_accumulate(acc1[c], sc1a, vt, acc1[c]);
                } else {
                    simdgroup_multiply_accumulate(acc0[c], sc0b, vt, acc0[c]);
                    simdgroup_multiply_accumulate(acc1[c], sc1b, vt, acc1[c]);
                }
            }
        }

        }
    }

    float L0 = l0 + simd_shuffle_xor(l0, 1u);
    L0 += simd_shuffle_xor(L0, 8u);
    float L1 = l1 + simd_shuffle_xor(l1, 1u);
    L1 += simd_shuffle_xor(L1, 8u);
    const float inv0 = 1.0f / L0;
    const float inv1 = 1.0f / L1;
#pragma unroll
    for (uint c = 0; c < 8u; ++c) {
        float2 e = *(thread float2*)&acc0[c].thread_elements();
        simdgroup_half8x8 oh;
        *(thread half2*)&oh.thread_elements() = half2((half)(e.x * inv0), (half)(e.y * inv0));
        simdgroup_store(oh, op + c * 8u, D);

        float2 f = *(thread float2*)&acc1[c].thread_elements();
        simdgroup_half8x8 og;
        *(thread half2*)&og.thread_elements() = half2((half)(f.x * inv1), (half)(f.y * inv1));
        simdgroup_store(og, op + 8u * D + c * 8u, D);
    }
}
