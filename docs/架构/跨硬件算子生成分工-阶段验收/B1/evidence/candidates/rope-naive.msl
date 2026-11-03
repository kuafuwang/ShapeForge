kernel void rope_kernel(device const {{T}}* x [[buffer(0)]],
                        device {{T}}* y [[buffer(1)]],
                        uint2 tg [[threadgroup_position_in_grid]],
                        uint lid [[thread_index_in_threadgroup]]) {
    const uint HALF = D / 2;
    const float LOG_BASE = precise::log(BASE);
    uint bh = tg.y;
    for (uint ps = 0; ps < BLOCK_S; ++ps) {
        uint s = tg.x * BLOCK_S + ps;
        ulong row = ((ulong)bh * S + s) * D;
        for (uint i = lid; i < HALF; i += 128) {
            float inv_freq = precise::exp(-(float(i) / float(HALF)) * LOG_BASE);
            float ang = float(s) * inv_freq;
            float c = precise::cos(ang), sn = precise::sin(ang);
            float x1 = float(x[row + i]), x2 = float(x[row + i + HALF]);
            y[row + i] = {{T}}(x1 * c - x2 * sn);
            y[row + i + HALF] = {{T}}(x1 * sn + x2 * c);
        }
    }
}
