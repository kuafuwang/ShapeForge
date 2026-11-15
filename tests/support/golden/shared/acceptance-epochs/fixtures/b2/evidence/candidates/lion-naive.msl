kernel void lion_kernel(device const {{T}}* p [[buffer(0)]],
                        device const {{T}}* g [[buffer(1)]],
                        device const {{T}}* m [[buffer(2)]],
                        device {{T}}* p_out [[buffer(3)]],
                        device {{T}}* m_out [[buffer(4)]],
                        uint tg [[threadgroup_position_in_grid]],
                        uint lid [[thread_index_in_threadgroup]]) {
    uint base = tg * BLOCK_SIZE;
    for (uint i = lid; i < BLOCK_SIZE; i += 128) {
        uint j = base + i;
        if (j < N) {
            float pv = float(p[j]), gv = float(g[j]), mv = float(m[j]);
            float c = BETA1 * mv + (1.0f - BETA1) * gv;
            float u = c > 0.0f ? 1.0f : (c < 0.0f ? -1.0f : 0.0f);
            p_out[j] = {{T}}(pv * (1.0f - LR * WD) - LR * u);
            m_out[j] = {{T}}(BETA2 * mv + (1.0f - BETA2) * gv);
        }
    }
}
