kernel void rmsnorm_kernel(device const {{T}}* x [[buffer(0)]],
                           device const {{T}}* w [[buffer(1)]],
                           device {{T}}* y [[buffer(2)]],
                           uint row [[threadgroup_position_in_grid]],
                           uint lid [[thread_index_in_threadgroup]],
                           uint lane [[thread_index_in_simdgroup]],
                           uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float partial[8];
    device const {{T}}* xr = x + (ulong)row * H;
    float ss = 0.0f;
    for (uint i = lid; i < H; i += 256) { float v = float(xr[i]); ss += v * v; }
    ss = simd_sum(ss);
    if (lane == 0) partial[sg] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float tot = 0.0f;
    for (uint s = 0; s < 8; ++s) tot += partial[s];
    float r = rsqrt(tot / float(H) + EPS);
    for (uint i = lid; i < H; i += 256) y[(ulong)row * H + i] = {{T}}(float(xr[i]) * r * float(w[i]));
}
