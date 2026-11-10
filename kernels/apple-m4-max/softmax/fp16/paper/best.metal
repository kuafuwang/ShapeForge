#include <metal_stdlib>
using namespace metal;

// Three low-register phases, with one independent row per SIMD group.  The input
// is staged while computing row max; the next two on-chip passes form the fp32
// denominator and write normalized fp16 output.  Only SIMD synchronization is used.
/* taic-launch
math fast
launch softmax_kernel grid=M,1,1 threads=32,1,1
*/

constant constexpr float SF_LOG2E = 1.4426950408889634f;

kernel void softmax_kernel(
    device const half* x [[buffer(0)]],
    device half* y [[buffer(1)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]
) {
    constexpr uint ROWS = 1;
    threadgroup half tile[N * ROWS];
    const uint row = tgid.x * ROWS + sg;
    const uint col = lane * 64u;
    const uint tcol = sg * N + col;
    device const half* xr = x + (size_t)row * N + col;

    float m = -INFINITY;
    for (uint i = 0; i < 8; ++i) {
        uint4 v = *reinterpret_cast<device const uint4*>(xr + i * 8u);
        *reinterpret_cast<threadgroup uint4*>(tile + tcol + i * 8u) = v;
        float4 a0 = float4(as_type<half4>(v.xy));
        float4 a1 = float4(as_type<half4>(v.zw));
        float4 b = max(a0, a1);
        m = max(m, max(max(b.x, b.y), max(b.z, b.w)));
    }
    const float rmax = simd_max(m);
    simdgroup_barrier(mem_flags::mem_threadgroup);

    float s = 0.0f;
    for (uint i = 0; i < 8; ++i) {
        uint4 v = *reinterpret_cast<threadgroup const uint4*>(tile + tcol + i * 8u);
        float4 a0 = float4(as_type<half4>(v.xy));
        float4 a1 = float4(as_type<half4>(v.zw));
        float4 e0 = exp2((a0 - rmax) * SF_LOG2E);
        float4 e1 = exp2((a1 - rmax) * SF_LOG2E);
        float4 e = e0 + e1;
        s += (e.x + e.y) + (e.z + e.w);
    }
    const float inv = 1.0f / simd_sum(s);

    device half* yr = y + (size_t)row * N + col;
    for (uint i = 0; i < 8; ++i) {
        uint4 v = *reinterpret_cast<threadgroup const uint4*>(tile + tcol + i * 8u);
        float4 a0 = float4(as_type<half4>(v.xy));
        float4 a1 = float4(as_type<half4>(v.zw));
        uint4 o;
        o.xy = as_type<uint2>(half4(exp2((a0 - rmax) * SF_LOG2E) * inv));
        o.zw = as_type<uint2>(half4(exp2((a1 - rmax) * SF_LOG2E) * inv));
        *reinterpret_cast<device uint4*>(yr + i * 8u) = o;
    }
}
