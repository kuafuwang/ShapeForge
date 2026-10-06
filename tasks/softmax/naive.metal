kernel void softmax_kernel(device const {{T}}* x [[buffer(0)]],
                           device {{T}}* y [[buffer(1)]],
                           uint row [[threadgroup_position_in_grid]],
                           uint lid [[thread_index_in_threadgroup]],
                           uint lane [[thread_index_in_simdgroup]],
                           uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float partial[8];
    device const {{T}}* xr = x + (ulong)row * N;
    float m = -INFINITY;
    for (uint i = lid; i < N; i += 256) m = max(m, float(xr[i]));
    m = simd_max(m);
    if (lane == 0) partial[sg] = m;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float rowmax = partial[0];
    for (uint s = 1; s < 8; ++s) rowmax = max(rowmax, partial[s]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float sum = 0.0f;
    for (uint i = lid; i < N; i += 256) sum += exp(float(xr[i]) - rowmax);
    sum = simd_sum(sum);
    if (lane == 0) partial[sg] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float tot = 0.0f;
    for (uint s = 0; s < 8; ++s) tot += partial[s];
    float inv = 1.0f / tot;
    for (uint i = lid; i < N; i += 256) y[(ulong)row * N + i] = {{T}}(exp(float(xr[i]) - rowmax) * inv);
}
