// One threadgroup = BLOCK_M query rows of one (batch, head); {{TPR}} threads cooperate on one row (D/{{TPR}} dims each).
kernel void flash_attention_kernel(device const {{T}}* q [[buffer(0)]],
                                   device const {{T}}* k [[buffer(1)]],
                                   device const {{T}}* v [[buffer(2)]],
                                   device {{T}}* o [[buffer(3)]],
                                   uint2 tg [[threadgroup_position_in_grid]],
                                   uint lid [[thread_index_in_threadgroup]]) {
    const uint r = lid / {{TPR}}, p = lid % {{TPR}};
    const uint row = tg.x * BLOCK_M + r;
    const ulong base = (ulong)tg.y * S * D;
    constexpr uint DP = D / {{TPR}};
    float qv[DP], acc[DP];
    for (uint i = 0; i < DP; ++i) { qv[i] = float(q[base + (ulong)row * D + p * DP + i]); acc[i] = 0.0f; }
    float m = -3.0e38f, l = 0.0f;
    const float c = SM_SCALE * 1.4426950408889634f;
    for (uint j = 0; j < S; ++j) {
        float s = 0.0f;
        for (uint i = 0; i < DP; ++i) s += qv[i] * float(k[base + (ulong)j * D + p * DP + i]);
{{SHUFFLES}}        s *= c;
        float m_new = max(m, s);
        float alpha = exp2(m - m_new), pj = exp2(s - m_new);
        l = l * alpha + pj;
        for (uint i = 0; i < DP; ++i) acc[i] = acc[i] * alpha + pj * float(v[base + (ulong)j * D + p * DP + i]);
        m = m_new;
    }
    const float inv = 1.0f / l;
    for (uint i = 0; i < DP; ++i) o[base + (ulong)row * D + p * DP + i] = {{T}}(acc[i] * inv);
}
