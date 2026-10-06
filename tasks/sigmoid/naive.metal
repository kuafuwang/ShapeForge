inline float sigmoid_f(float v) {
    float e = exp(-fabs(v));
    return v >= 0.0f ? 1.0f / (1.0f + e) : e / (1.0f + e);
}

kernel void sigmoid_kernel(device const {{T}}* x [[buffer(0)]],
                           device {{T}}* y [[buffer(1)]],
                           uint tg [[threadgroup_position_in_grid]],
                           uint lid [[thread_index_in_threadgroup]]) {
    uint base = tg * BLOCK_SIZE;
    for (uint i = lid; i < BLOCK_SIZE; i += 128) {
        uint j = base + i;
        if (j < N) y[j] = {{T}}(sigmoid_f(float(x[j])));
    }
}
