kernel void bitdelta_gemm_kernel(device const {{T}}* x [[buffer(0)]],
                                 device const uint* bits [[buffer(1)]],
                                 constant float& scale [[buffer(2)]],
                                 device {{T}}* y [[buffer(3)]],
                                 uint2 tg [[threadgroup_position_in_grid]],
                                 uint lid [[thread_index_in_threadgroup]]) {
    threadgroup {{T}} As[BLOCK_M][BLOCK_K];
    threadgroup {{T}} Bs[BLOCK_K][BLOCK_N];
    const uint m0 = tg.y * BLOCK_M, n0 = tg.x * BLOCK_N;
    const uint tr = lid / 8, tc = lid % 8;
    float acc[4][8];
    for (uint i = 0; i < 4; ++i) for (uint j = 0; j < 8; ++j) acc[i][j] = 0.0f;
    for (uint kb = 0; kb < K / BLOCK_K; ++kb) {
        for (uint i = lid; i < BLOCK_M * BLOCK_K; i += 128) {
            uint r = i / BLOCK_K, c = i % BLOCK_K;
            As[r][c] = x[(ulong)(m0 + r) * K + kb * BLOCK_K + c];
        }
        for (uint i = lid; i < BLOCK_K * BLOCK_N; i += 128) {
            uint nn = i / BLOCK_K, kk = i % BLOCK_K;
            uint word = bits[(ulong)kb * N + n0 + nn];
            Bs[kk][nn] = (word >> kk) & 1u ? {{T}}(1.0f) : {{T}}(-1.0f);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint kk = 0; kk < BLOCK_K; ++kk) {
            float a[4], b[8];
            for (uint i = 0; i < 4; ++i) a[i] = float(As[tr * 4 + i][kk]);
            for (uint j = 0; j < 8; ++j) b[j] = float(Bs[kk][tc * 8 + j]);
            for (uint i = 0; i < 4; ++i) for (uint j = 0; j < 8; ++j) acc[i][j] += a[i] * b[j];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint i = 0; i < 4; ++i)
        for (uint j = 0; j < 8; ++j)
            y[(ulong)(m0 + tr * 4 + i) * N + n0 + tc * 8 + j] = {{T}}(acc[i][j] * scale);
}
