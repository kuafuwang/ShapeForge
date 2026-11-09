kernel void scale_msl(
    device const float* x [[buffer(0)]],
    device float* acc [[buffer(1)]],
    device float* y [[buffer(2)]],
    uint index [[thread_position_in_grid]]) {
  if (index < N) {
    float value = x[index] * 2.0f + 1.0f;
    y[index] = value;
    acc[index] = acc[index] + value;
  }
}
/* taic-launch
math safe
launch scale_msl grid=4,1,1 threads=256,1,1
*/
