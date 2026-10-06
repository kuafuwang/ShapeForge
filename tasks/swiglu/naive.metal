inline float sigmoid_f(float v) {
    float e = exp(-fabs(v));
    return v >= 0.0f ? 1.0f / (1.0f + e) : e / (1.0f + e);
}

kernel void swiglu_kernel(device const {{T}}* gate [[buffer(0)]],
                          device const {{T}}* up [[buffer(1)]],
                          device {{T}}* y [[buffer(2)]],
                          uint tg [[threadgroup_position_in_grid]],
                          uint lid [[thread_index_in_threadgroup]]) {
    uint base = tg * BLOCK_SIZE;
    for (uint i = lid; i < BLOCK_SIZE; i += 128) {
        uint j = base + i;
        if (j < N) { float g = float(gate[j]); y[j] = {{T}}(g * sigmoid_f(g) * float(up[j])); }
    }
}
