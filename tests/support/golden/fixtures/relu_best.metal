kernel void relu_kernel(
    device const half* x [[buffer(0)]],
    device half* y [[buffer(1)]],
    uint tg  [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]])
{
    // One threadgroup per BLOCK_SIZE (1024) halves of the stream.
    const uint base = tg * BLOCK_SIZE;

    // --- read burst: both 8-byte chains issued before any store -----------------
    // chain A covers halves [0, 512)    -> uint2 index lid
    // chain B covers halves [512, 1024) -> uint2 index 256 + lid
    uint2 rawa = *reinterpret_cast<device const uint2*>(x + base + lid * 4);
    uint2 rawb = *reinterpret_cast<device const uint2*>(x + base + 512 + lid * 4);

    half4 a = as_type<half4>(rawa);
    half4 b = as_type<half4>(rawb);

    // ReLU as select(v < 0, 0, v): false for NaN (NaN propagates) and for -0.0 (preserved).
    half4 ra = half4(a.x < 0.0h ? 0.0h : a.x,
                     a.y < 0.0h ? 0.0h : a.y,
                     a.z < 0.0h ? 0.0h : a.z,
                     a.w < 0.0h ? 0.0h : a.w);
    half4 rb = half4(b.x < 0.0h ? 0.0h : b.x,
                     b.y < 0.0h ? 0.0h : b.y,
                     b.z < 0.0h ? 0.0h : b.z,
                     b.w < 0.0h ? 0.0h : b.w);

    // Uniform barrier (no divergent/masked lanes): read phase ends, write phase begins.
    metal::threadgroup_barrier(metal::mem_flags::mem_none);

    // --- write burst ------------------------------------------------------------
    *reinterpret_cast<device uint2*>(y + base + lid * 4)       = as_type<uint2>(ra);
    *reinterpret_cast<device uint2*>(y + base + 512 + lid * 4) = as_type<uint2>(rb);
}
