kernel void relu_kernel(device const {{T}}* x [[buffer(0)]],
                        device {{T}}* y [[buffer(1)]],
                        uint tg [[threadgroup_position_in_grid]],
                        uint lid [[thread_index_in_threadgroup]]) {
    uint base = tg * BLOCK_SIZE;
    for (uint i = lid; i < BLOCK_SIZE; i += 128) {
        uint j = base + i;
        if (j < N) { float v = float(x[j]); y[j] = {{T}}(v < 0.0f ? 0.0f : v); }  // v < 0 is false for NaN: NaN propagates, like torch.relu / mx.maximum
    }
}
