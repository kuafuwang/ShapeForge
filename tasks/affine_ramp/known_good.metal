#include <metal_stdlib>
using namespace metal;

// affine_ramp: read the fp32 ramp the Host Tiling plan wrote into `scale_tab`, multiply in fp32 and
// round to the element type exactly once (at the output).  The workspace is a kernel parameter: it
// is bound by name at its ABI slot and is writable (`access: write`), so it must not be `const`.
kernel void affine_ramp(device const half* x   [[buffer(0)]],
                        device half* y         [[buffer(1)]],
                        device float* scale_tab [[buffer(2)]],
                        uint i [[thread_position_in_grid]]) {
  if (i < N) {
    y[i] = half(scale_tab[i / SEG] * float(x[i]));
  }
}
/* taic-launch
math safe
workspace scale_tab NS*4
launch affine_ramp grid=N,1,1 threads=32,1,1 workspace=scale_tab
*/
