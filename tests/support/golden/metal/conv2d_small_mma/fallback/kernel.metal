#include <metal_stdlib>
using namespace metal;

// General F(4x4, 3x3) Winograd for every conv2d point the paper MMA guard misses.
// Finite tiles use the transformed U in `uwin`. A non-finite input tile takes the
// direct fp32 sum. A non-finite U is not repaired from `w`. `y` is never scratch.
constant float WIN_G[18] = {0.25f, 0.0f, 0.0f, -0.1666666666666667f, -0.1666666666666667f, -0.1666666666666667f, -0.1666666666666667f, 0.1666666666666667f, -0.1666666666666667f, 0.04166666666666666f, 0.08333333333333333f, 0.1666666666666667f, 0.04166666666666666f, -0.08333333333333333f, 0.1666666666666667f, 0.0f, 0.0f, 1.0f};
constant float WIN_BT[36] = {4.0f, 0.0f, -5.0f, 0.0f, 1.0f, 0.0f, 0.0f, -4.0f, -4.0f, 1.0f, 1.0f, 0.0f, 0.0f, 4.0f, -4.0f, -1.0f, 1.0f, 0.0f, 0.0f, -2.0f, -1.0f, 2.0f, 1.0f, 0.0f, 0.0f, 2.0f, -1.0f, -2.0f, 1.0f, 0.0f, 0.0f, 4.0f, 0.0f, -5.0f, 0.0f, 1.0f};
constant float WIN_AT[24] = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 0.0f, 0.0f, 1.0f, -1.0f, 2.0f, -2.0f, 0.0f, 0.0f, 1.0f, 1.0f, 4.0f, 4.0f, 0.0f, 0.0f, 1.0f, -1.0f, 8.0f, -8.0f, 1.0f};
constant uint WIN_TILE = 4u;
constant uint WIN_ALPHA = 6u;
constant uint WIN_ELEMS = 36u;

kernel void conv2d_wgrid(
    device const half* w [[buffer(0)]],
    device float* uwin [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
  uint total = C * OC;
  if (tid >= total) return;
  uint c = tid % C;
  uint oc = tid / C;
  float g[9];
  for (uint kh = 0u; kh < 3u; ++kh) {
    for (uint kw = 0u; kw < 3u; ++kw) {
      g[kh * 3u + kw] = float(w[((oc * KH + kh) * KW + kw) * C + c]);
    }
  }
  float tmp[18];
  for (uint i = 0u; i < WIN_ALPHA; ++i) {
    for (uint j = 0u; j < 3u; ++j) {
      float sum = 0.0f;
      for (uint k = 0u; k < 3u; ++k) sum += WIN_G[i * 3u + k] * g[k * 3u + j];
      tmp[i * 3u + j] = sum;
    }
  }
  for (uint e = 0u; e < WIN_ELEMS; ++e) {
    uint i = e / WIN_ALPHA;
    uint j = e % WIN_ALPHA;
    float sum = 0.0f;
    for (uint k = 0u; k < 3u; ++k) sum += tmp[i * 3u + k] * WIN_G[j * 3u + k];
    uwin[(e * C + c) * OC + oc] = sum;
  }
}

kernel void conv2d_kernel(
    device const half* x [[buffer(0)]],
    device const float* uwin [[buffer(1)]],
    device half* y [[buffer(2)]],
    device const half* w [[buffer(3)]],
    uint tid [[thread_position_in_grid]]) {
  uint total = N * OC;
  if (tid >= total) return;
  uint oc = tid % OC;
  uint n = tid / OC;
  bool weights_bad = false;
  for (uint c = 0u; c < C && !weights_bad; ++c) {
    for (uint kh = 0u; kh < KH && !weights_bad; ++kh) {
      for (uint kw = 0u; kw < KW; ++kw) {
        if (!isfinite(float(w[((oc * KH + kh) * KW + kw) * C + c]))) weights_bad = true;
      }
    }
  }
  bool u_finite = true;
  for (uint e = 0u; e < WIN_ELEMS && u_finite; ++e) {
    for (uint c = 0u; c < C; ++c) {
      if (!isfinite(uwin[(e * C + c) * OC + oc])) u_finite = false;
    }
  }
  uint tiles_w = (OW + WIN_TILE - 1u) / WIN_TILE;
  uint tiles_h = (OH + WIN_TILE - 1u) / WIN_TILE;
  for (uint th = 0u; th < tiles_h; ++th) {
    for (uint tw = 0u; tw < tiles_w; ++tw) {
      uint oh0 = th * WIN_TILE;
      uint ow0 = tw * WIN_TILE;
      bool inputs_bad = weights_bad;
      for (uint ih = 0u; ih < WIN_ALPHA && !inputs_bad; ++ih) {
        for (uint iw = 0u; iw < WIN_ALPHA && !inputs_bad; ++iw) {
          uint h = oh0 + ih;
          uint wpos = ow0 + iw;
          if (h >= H || wpos >= W) continue;
          for (uint c = 0u; c < C; ++c) {
            if (!isfinite(float(x[((n * H + h) * W + wpos) * C + c]))) inputs_bad = true;
          }
        }
      }
      float accum[36];
      for (uint e = 0u; e < WIN_ELEMS; ++e) accum[e] = 0.0f;
      bool transform_finite = true;
      for (uint c = 0u; c < C; ++c) {
        float d[36];
        for (uint i = 0u; i < WIN_ALPHA; ++i) {
          for (uint j = 0u; j < WIN_ALPHA; ++j) {
            uint h = oh0 + i;
            uint wpos = ow0 + j;
            float value = 0.0f;
            if (h < H && wpos < W) value = float(x[((n * H + h) * W + wpos) * C + c]);
            d[i * WIN_ALPHA + j] = value;
          }
        }
        float v[36];
        for (uint i = 0u; i < WIN_ALPHA; ++i) {
          for (uint j = 0u; j < WIN_ALPHA; ++j) {
            float sum = 0.0f;
            for (uint k = 0u; k < WIN_ALPHA; ++k) sum += WIN_BT[i * WIN_ALPHA + k] * d[k * WIN_ALPHA + j];
            v[i * WIN_ALPHA + j] = sum;
          }
        }
        float vt[36];
        for (uint i = 0u; i < WIN_ALPHA; ++i) {
          for (uint j = 0u; j < WIN_ALPHA; ++j) {
            float sum = 0.0f;
            for (uint k = 0u; k < WIN_ALPHA; ++k) sum += v[i * WIN_ALPHA + k] * WIN_BT[j * WIN_ALPHA + k];
            vt[i * WIN_ALPHA + j] = sum;
            if (!isfinite(vt[i * WIN_ALPHA + j])) transform_finite = false;
          }
        }
        for (uint e = 0u; e < WIN_ELEMS; ++e) {
          float stored = uwin[(e * C + c) * OC + oc];
          accum[e] += vt[e] * stored;
          if (!isfinite(accum[e])) transform_finite = false;
        }
      }
      float mid[24];
      float tile_out[16];
      for (uint i = 0u; i < WIN_TILE; ++i) {
        for (uint j = 0u; j < WIN_ALPHA; ++j) {
          float sum = 0.0f;
          for (uint k = 0u; k < WIN_ALPHA; ++k) sum += WIN_AT[i * WIN_ALPHA + k] * accum[k * WIN_ALPHA + j];
          mid[i * WIN_ALPHA + j] = sum;
        }
      }
      for (uint i = 0u; i < WIN_TILE; ++i) {
        for (uint j = 0u; j < WIN_TILE; ++j) {
          float sum = 0.0f;
          for (uint k = 0u; k < WIN_ALPHA; ++k) sum += mid[i * WIN_ALPHA + k] * WIN_AT[j * WIN_ALPHA + k];
          tile_out[i * WIN_TILE + j] = sum;
          if (!isfinite(sum)) transform_finite = false;
        }
      }
      bool direct = inputs_bad || (!transform_finite && u_finite);
      for (uint oh = 0u; oh < WIN_TILE; ++oh) {
        for (uint ow = 0u; ow < WIN_TILE; ++ow) {
          uint y_h = oh0 + oh;
          uint y_w = ow0 + ow;
          if (y_h >= OH || y_w >= OW) continue;
          float value = tile_out[oh * WIN_TILE + ow];
          if (direct) {
            value = 0.0f;
            for (uint kh = 0u; kh < KH; ++kh) {
              for (uint kw = 0u; kw < KW; ++kw) {
                float dot = 0.0f;
                for (uint c = 0u; c < C; ++c) {
                  float xv = float(x[((n * H + (y_h + kh)) * W + (y_w + kw)) * C + c]);
                  float wv = float(w[((oc * KH + kh) * KW + kw) * C + c]);
                  dot += xv * wv;
                }
                value += dot;
              }
            }
          }
          y[((n * OH + y_h) * OW + y_w) * OC + oc] = half(value);
        }
      }
    }
  }
}

/* taic-launch
math safe
scratch uwin fp32 36*C*OC zero
launch conv2d_wgrid grid=C*OC/64,1,1 threads=64,1,1 scratch=uwin:write
launch conv2d_kernel grid=N*OC/64,1,1 threads=64,1,1 scratch=uwin:read
*/
