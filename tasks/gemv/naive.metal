kernel void gemv_kernel(device const {{T}}* A [[buffer(0)]],
                        device const {{T}}* x [[buffer(1)]],
                        device {{T}}* y [[buffer(2)]],
                        uint tg [[threadgroup_position_in_grid]],
                        uint lane [[thread_index_in_simdgroup]],
                        uint sg [[simdgroup_index_in_threadgroup]]) {
    uint row = tg * BLOCK_M + sg;
    device const {{T}}* a = A + (ulong)row * K;
    float acc = 0.0f;
    for (uint k = lane; k < K; k += 32) acc += float(a[k]) * float(x[k]);
    acc = simd_sum(acc);
    if (lane == 0) y[row] = {{T}}(acc);
}
