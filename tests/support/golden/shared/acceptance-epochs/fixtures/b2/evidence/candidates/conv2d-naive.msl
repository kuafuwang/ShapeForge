kernel void conv2d_kernel(device const {{T}}* x [[buffer(0)]],
                          device const {{T}}* w [[buffer(1)]],
                          device {{T}}* y [[buffer(2)]],
                          uint2 tg [[threadgroup_position_in_grid]],
                          uint lid [[thread_index_in_threadgroup]]) {
    threadgroup {{T}} As[BLOCK_M][BLOCK_K];
    threadgroup {{T}} Bs[BLOCK_K][BLOCK_N];
    const uint m0 = tg.y * BLOCK_M, n0 = tg.x * BLOCK_N;
    const uint tr = lid / 8, tc = lid % 8;
    float acc[4][8];
    for (uint i = 0; i < 4; ++i) for (uint j = 0; j < 8; ++j) acc[i][j] = 0.0f;
    for (uint k0 = 0; k0 < K_TOTAL; k0 += BLOCK_K) {
        const uint khkw = k0 / C, c0 = k0 % C;        // C % BLOCK_K == 0, so a K block stays inside one (kh, kw)
        const uint kh = khkw / KW, kw = khkw % KW;
        for (uint i = lid; i < BLOCK_M * BLOCK_K; i += 128) {
            uint r = i / BLOCK_K, c = i % BLOCK_K;
            uint m = m0 + r, img = m / (OH * OW), rem = m % (OH * OW);
            uint oh = rem / OW, ow = rem % OW;
            As[r][c] = x[(((ulong)img * H + oh + kh) * W + ow + kw) * C + c0 + c];
        }
        for (uint i = lid; i < BLOCK_K * BLOCK_N; i += 128) {
            uint nn = i / BLOCK_K, kk = i % BLOCK_K;
            Bs[kk][nn] = w[(ulong)(n0 + nn) * K_TOTAL + k0 + kk];
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
            y[(ulong)(m0 + tr * 4 + i) * OC + n0 + tc * 8 + j] = {{T}}(acc[i][j]);
}
