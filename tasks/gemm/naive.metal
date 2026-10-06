{{EPILOGUE_DEF}}
kernel void {{KERNEL}}(device const {{T}}* A [[buffer(0)]],
                       device const {{T}}* B [[buffer(1)]],
                       device {{T}}* C [[buffer(2)]],
                       uint2 tg [[threadgroup_position_in_grid]],
                       uint lid [[thread_index_in_threadgroup]]) {
    threadgroup {{T}} As[BLOCK_M][BLOCK_K];
    threadgroup {{T}} Bs[BLOCK_K][BLOCK_N];
    const uint m0 = tg.y * BLOCK_M, n0 = tg.x * BLOCK_N;
    const uint tr = lid / {{TPR}}, tc = lid % {{TPR}};            // each thread owns a {{TM}} x 8 output tile
    float acc[{{TM}}][8];
    for (uint i = 0; i < {{TM}}; ++i) for (uint j = 0; j < 8; ++j) acc[i][j] = 0.0f;
    for (uint k0 = 0; k0 < K; k0 += BLOCK_K) {
        for (uint i = lid; i < BLOCK_M * BLOCK_K; i += {{THREADS}}) {
            uint r = i / BLOCK_K, c = i % BLOCK_K;
            As[r][c] = A[(ulong)(m0 + r) * K + k0 + c];
        }
        for (uint i = lid; i < BLOCK_K * BLOCK_N; i += {{THREADS}}) {
            uint r = i / BLOCK_N, c = i % BLOCK_N;
            Bs[r][c] = B[(ulong)(k0 + r) * N + n0 + c];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint kk = 0; kk < BLOCK_K; ++kk) {
            float a[{{TM}}], b[8];
            for (uint i = 0; i < {{TM}}; ++i) a[i] = float(As[tr * {{TM}} + i][kk]);
            for (uint j = 0; j < 8; ++j) b[j] = float(Bs[kk][tc * 8 + j]);
            for (uint i = 0; i < {{TM}}; ++i) for (uint j = 0; j < 8; ++j) acc[i][j] += a[i] * b[j];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint i = 0; i < {{TM}}; ++i)
        for (uint j = 0; j < 8; ++j)
            C[(ulong)(m0 + tr * {{TM}} + i) * N + n0 + tc * 8 + j] = {{T}}({{EPILOGUE}}(acc[i][j]));
}
