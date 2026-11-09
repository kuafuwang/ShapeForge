// One threadgroup = one (batch, head, chunk). Thread t owns output row l = t / 2 of the chunk and one half (HEADDIM / 2 columns) of the head dimension.
// Straightforward and slow: the (C[l] . B[s]) products are recomputed for both halves, nothing is staged in threadgroup memory.
kernel void mamba2_chunk_scan_kernel(device const {{T}}* x [[buffer(0)]],
                                     device const {{T}}* dt [[buffer(1)]],
                                     device const {{T}}* dA_cumsum [[buffer(2)]],
                                     device const {{T}}* Bmat [[buffer(3)]],
                                     device const {{T}}* Cmat [[buffer(4)]],
                                     device const {{T}}* Dskip [[buffer(5)]],
                                     device const {{T}}* states [[buffer(6)]],
                                     device {{T}}* y [[buffer(7)]],
                                     uint2 tg [[threadgroup_position_in_grid]],
                                     uint lid [[thread_index_in_threadgroup]]) {
    constexpr uint HP = HEADDIM / 2;
    const uint c = tg.x, b = tg.y / NHEADS, h = tg.y % NHEADS;
    const uint l = lid / 2, p0 = (lid % 2) * HP;
    const uint t0 = b * SEQLEN + c * CHUNK;
    const ulong cs_base = (ulong)((b * NHEADS + h) * NCHUNKS + c) * CHUNK;
    const float cs_l = float(dA_cumsum[cs_base + l]);
    float acc[HP];
    // carried-in state: exp(cs[l]) * (C[l] . state[p])
    const float e = exp(cs_l);
    for (uint p = 0; p < HP; ++p) {
        float s = 0.0f;
        const ulong srow = ((ulong)((b * NCHUNKS + c) * NHEADS + h) * HEADDIM + p0 + p) * DSTATE;
        for (uint n = 0; n < DSTATE; ++n) s += float(Cmat[(ulong)(t0 + l) * DSTATE + n]) * float(states[srow + n]);
        acc[p] = s * e;
    }
    // inside the chunk: sum over s <= l
    for (uint s = 0; s <= l; ++s) {
        float cb = 0.0f;
        for (uint n = 0; n < DSTATE; ++n) cb += float(Cmat[(ulong)(t0 + l) * DSTATE + n]) * float(Bmat[(ulong)(t0 + s) * DSTATE + n]);
        const float w = cb * exp(cs_l - float(dA_cumsum[cs_base + s])) * float(dt[(ulong)(t0 + s) * NHEADS + h]);
        for (uint p = 0; p < HP; ++p) acc[p] += w * float(x[((ulong)(t0 + s) * NHEADS + h) * HEADDIM + p0 + p]);
    }
    const float d = float(Dskip[h]);
    for (uint p = 0; p < HP; ++p) {
        const ulong i = ((ulong)(t0 + l) * NHEADS + h) * HEADDIM + p0 + p;
        y[i] = {{T}}(acc[p] + d * float(x[i]));
    }
}
