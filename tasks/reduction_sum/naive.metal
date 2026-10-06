kernel void reduction_sum_kernel(device const {{T}}* x [[buffer(0)]],
                                 device {{T}}* y [[buffer(1)]],
                                 uint row [[threadgroup_position_in_grid]],
                                 uint lid [[thread_index_in_threadgroup]],
                                 uint lane [[thread_index_in_simdgroup]],
                                 uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float partial[4];
    device const {{T}}* xr = x + (ulong)row * N;
    float s = 0.0f;
    for (uint i = lid; i < N; i += 128) s += float(xr[i]);
    s = simd_sum(s);
    if (lane == 0) partial[sg] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid == 0) y[row] = {{T}}(partial[0] + partial[1] + partial[2] + partial[3]);
}
