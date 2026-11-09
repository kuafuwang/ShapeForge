// Function: scale_kernel_kernel
#include <metal_stdlib>
#include <metal_simdgroup>
#define TILELANG_PRAGMA_UNROLL _Pragma("clang loop unroll(full)")
using namespace metal;

union __TVMArgUnion {
 int v_int[2];
};

[[kernel, max_total_threads_per_threadgroup(256)]] void scale_kernel_kernel(  const device float* __restrict a [[ buffer(0) ]],
  device float* __restrict b [[ buffer(1) ]],
  uint3 blockIdx [[threadgroup_position_in_grid]],
  uint3 __gridDim [[threadgroups_per_grid]],
  uint3 threadIdx [[thread_position_in_threadgroup]],
  uint __simd_group_id [[simdgroup_index_in_threadgroup]]
) {
  b[((((int)blockIdx.x) * 256) + ((int)threadIdx.x))] = (a[((((int)blockIdx.x) * 256) + ((int)threadIdx.x))] * 2.000000e+00f);
}


