inline float nan_max(float a, float b) {
    if (isnan(a)) return a;
    if (isnan(b)) return b;
    return max(a, b);
}

inline float2 tile_max_sum(simdgroup_float8x8 C) {
    auto e = C.thread_elements();
    float mx = nan_max(e[0], e[1]);
    mx = nan_max(mx, simd_shuffle_xor(mx, 8));
    mx = nan_max(mx, simd_shuffle_xor(mx, 1));
    float sm = exp2(e[0] - mx) + exp2(e[1] - mx);
    sm += simd_shuffle_xor(sm, 8);
    sm += simd_shuffle_xor(sm, 1);
    return float2(mx, sm);
}

kernel void flash_attention_kernel(
    device const half* q [[buffer(0)]],
    device const half* k [[buffer(1)]],
    device const half* v [[buffer(2)]],
    device half* o [[buffer(3)]],
    uint2 tpg [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]
) {
    constexpr float QK_SCALE = SM_SCALE * 1.4426950408889634f;
    constexpr uint LD = D + 8;

    threadgroup half K_tile[BLOCK_N][LD];

    const uint bh = tpg.y;
    const uint pid_m = tpg.x;
    const uint base = bh * S * D;
    const uint row0 = pid_m * BLOCK_M + sg * 8;

    simdgroup_half8x8 Q[8];
    for (uint kk = 0; kk < 8; ++kk) {
        simdgroup_load(Q[kk], q + base + row0 * D + kk * 8, D);
    }

    simdgroup_float8x8 Acc[8];
    for (uint d = 0; d < 8; ++d) {
        Acc[d] = simdgroup_float8x8(0);
    }

    float m_i = -INFINITY;
    float l_i = 0.0f;

    for (uint start_n = 0; start_n < S; start_n += BLOCK_N) {
        device const half* k_tile_ptr = k + base + start_n * D;

        for (uint c = lid; c < (BLOCK_N * D / 8); c += 128) {
            uint r = c >> 3;
            uint col = (c & 7) << 3;
            uint4 raw = *reinterpret_cast<device const uint4*>(k_tile_ptr + (c << 3));
            *reinterpret_cast<threadgroup uint4*>(&K_tile[r][col]) = raw;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float m_old = m_i;

        simdgroup_float8x8 Scores[4];
        for (uint j = 0; j < 4; ++j) {
            Scores[j] = simdgroup_float8x8(0);
        }

        for (uint kk = 0; kk < 8; ++kk) {
            simdgroup_half8x8 A = Q[kk];
            simdgroup_half8x8 B0;
            simdgroup_half8x8 B1;
            simdgroup_half8x8 B2;
            simdgroup_half8x8 B3;

            simdgroup_load(B0, &K_tile[0][kk * 8], LD, ulong2(0, 0), true);
            simdgroup_load(B1, &K_tile[8][kk * 8], LD, ulong2(0, 0), true);
            simdgroup_load(B2, &K_tile[16][kk * 8], LD, ulong2(0, 0), true);
            simdgroup_load(B3, &K_tile[24][kk * 8], LD, ulong2(0, 0), true);

            simdgroup_multiply_accumulate(Scores[0], A, B0, Scores[0]);
            simdgroup_multiply_accumulate(Scores[1], A, B1, Scores[1]);
            simdgroup_multiply_accumulate(Scores[2], A, B2, Scores[2]);
            simdgroup_multiply_accumulate(Scores[3], A, B3, Scores[3]);
        }

        for (uint j = 0; j < 4; ++j) {
            auto e = Scores[j].thread_elements();
            e[0] *= QK_SCALE;
            e[1] *= QK_SCALE;
            Scores[j].thread_elements() = e;
        }

        for (uint j = 0; j < 4; ++j) {
            float2 ts = tile_max_sum(Scores[j]);
            float m_new = nan_max(m_i, ts.x);
            float alpha_tile = exp2(m_i - m_new);
            l_i = l_i * alpha_tile + ts.y * exp2(ts.x - m_new);
            m_i = m_new;
        }

        float alpha = exp2(m_old - m_i);

        for (uint d = 0; d < 8; ++d) {
            auto e = Acc[d].thread_elements();
            e[0] *= alpha;
            e[1] *= alpha;
            Acc[d].thread_elements() = e;
        }

        simdgroup_half8x8 P[4];
        for (uint j = 0; j < 4; ++j) {
            auto e = Scores[j].thread_elements();
            auto pe = P[j].thread_elements();
            pe[0] = half(exp2(e[0] - m_i));
            pe[1] = half(exp2(e[1] - m_i));
            P[j].thread_elements() = pe;
        }

        for (uint kk = 0; kk < 4; ++kk) {
            simdgroup_half8x8 A = P[kk];
            simdgroup_half8x8 V0;
            simdgroup_half8x8 V1;
            simdgroup_half8x8 V2;
            simdgroup_half8x8 V3;
            simdgroup_half8x8 V4;
            simdgroup_half8x8 V5;
            simdgroup_half8x8 V6;
            simdgroup_half8x8 V7;

            simdgroup_load(V0, v + base + (start_n + kk * 8) * D + 0 * 8, D);
            simdgroup_load(V1, v + base + (start_n + kk * 8) * D + 1 * 8, D);
            simdgroup_load(V2, v + base + (start_n + kk * 8) * D + 2 * 8, D);
            simdgroup_load(V3, v + base + (start_n + kk * 8) * D + 3 * 8, D);
            simdgroup_load(V4, v + base + (start_n + kk * 8) * D + 4 * 8, D);
            simdgroup_load(V5, v + base + (start_n + kk * 8) * D + 5 * 8, D);
            simdgroup_load(V6, v + base + (start_n + kk * 8) * D + 6 * 8, D);
            simdgroup_load(V7, v + base + (start_n + kk * 8) * D + 7 * 8, D);

            simdgroup_multiply_accumulate(Acc[0], A, V0, Acc[0]);
            simdgroup_multiply_accumulate(Acc[1], A, V1, Acc[1]);
            simdgroup_multiply_accumulate(Acc[2], A, V2, Acc[2]);
            simdgroup_multiply_accumulate(Acc[3], A, V3, Acc[3]);
            simdgroup_multiply_accumulate(Acc[4], A, V4, Acc[4]);
            simdgroup_multiply_accumulate(Acc[5], A, V5, Acc[5]);
            simdgroup_multiply_accumulate(Acc[6], A, V6, Acc[6]);
            simdgroup_multiply_accumulate(Acc[7], A, V7, Acc[7]);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (uint d = 0; d < 8; ++d) {
        simdgroup_half8x8 O;
        auto oe = O.thread_elements();
        auto e = Acc[d].thread_elements();
        oe[0] = half(e[0] / l_i);
        oe[1] = half(e[1] / l_i);
        O.thread_elements() = oe;
        simdgroup_store(O, o + base + row0 * D + d * 8, D);
    }
}
