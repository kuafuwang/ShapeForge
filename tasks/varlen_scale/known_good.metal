#include <metal_stdlib>
using namespace metal;

// varlen_scale: y[t] = SCALE * x[t] over the packed varlen batch.
// cu_seqlens is declared for ABI completeness (the elementwise operator does not read it).
/* taic-launch
launch varlen_scale grid=total_tokens*H*D,1,1 threads=32,1,1
*/
kernel void varlen_scale(device const half* x [[buffer(0)]],
                         device half* y [[buffer(1)]],
                         device const int* cu_seqlens [[buffer(2)]],
                         uint i [[thread_position_in_grid]]) {
    if (i < total_tokens * H * D) {
        y[i] = half(SCALE * float(x[i]));
    }
}
