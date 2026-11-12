"""conv2d Winograd F(2x2, 3x3) / F(4x4, 3x3) candidate source.

The same matrices drive the numeric check and the Metal source, so a transcription
drift fails before a GPU launch.  Stage 0 writes ``uwin`` and stage 1 reads it.
A non-finite input tile, or a non-finite transform of finite inputs, falls back to
the direct sum for that tile.  A non-finite ``uwin`` is left as a wrong result:
that is the missed-write signal, not a license to recompute from ``w``.
"""

from __future__ import annotations

import numpy as np

from tcenv.backends.metal.sanitize import GUARD_BYTES
from tcenv.backends.metal.scratch_texture import scratch_texture_layout

# Lavin & Gray.  B^T is stored directly; V = B^T d B.
F2_G = np.array(
    [[1.0, 0.0, 0.0], [0.5, 0.5, 0.5], [0.5, -0.5, 0.5], [0.0, 0.0, 1.0]],
    dtype=np.float64,
)
F2_BT = np.array(
    [[1.0, 0.0, -1.0, 0.0], [0.0, 1.0, 1.0, 0.0], [0.0, -1.0, 1.0, 0.0], [0.0, 1.0, 0.0, -1.0]],
    dtype=np.float64,
)
F2_AT = np.array([[1.0, 1.0, 1.0, 0.0], [0.0, 1.0, -1.0, -1.0]], dtype=np.float64)

F4_G = np.array(
    [
        [1.0 / 4.0, 0.0, 0.0],
        [-1.0 / 6.0, -1.0 / 6.0, -1.0 / 6.0],
        [-1.0 / 6.0, 1.0 / 6.0, -1.0 / 6.0],
        [1.0 / 24.0, 1.0 / 12.0, 1.0 / 6.0],
        [1.0 / 24.0, -1.0 / 12.0, 1.0 / 6.0],
        [0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)
F4_BT = np.array(
    [
        [4.0, 0.0, -5.0, 0.0, 1.0, 0.0],
        [0.0, -4.0, -4.0, 1.0, 1.0, 0.0],
        [0.0, 4.0, -4.0, -1.0, 1.0, 0.0],
        [0.0, -2.0, -1.0, 2.0, 1.0, 0.0],
        [0.0, 2.0, -1.0, -2.0, 1.0, 0.0],
        [0.0, 4.0, 0.0, -5.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)
F4_AT = np.array(
    [
        [1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
        [0.0, 1.0, -1.0, 2.0, -2.0, 0.0],
        [0.0, 1.0, 1.0, 4.0, 4.0, 0.0],
        [0.0, 1.0, -1.0, 8.0, -8.0, 1.0],
    ],
    dtype=np.float64,
)

_MATRICES = {2: (F2_BT, F2_G, F2_AT), 4: (F4_BT, F4_G, F4_AT)}


def matrices(tile: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """@brief 返回 ``(B^T, G, A^T)``。"""
    if tile not in _MATRICES:
        raise ValueError(f"tile must be 2 or 4, got {tile}")
    return _MATRICES[tile]


def _weight_transform(kernel: np.ndarray, transform_g: np.ndarray) -> np.ndarray:
    """@brief ``U = G g G^T``，``kernel`` 是 ``[3, 3]``。"""
    return transform_g @ kernel @ transform_g.T


def _direct_tile(image: np.ndarray, kernel: np.ndarray, tile: int) -> np.ndarray:
    """@brief 一个 tile 的直接卷积。越界输入按 0。``image`` 是 ``[C, H, W]``。"""
    out = np.zeros((tile, tile), dtype=np.float64)
    for oh in range(tile):
        for ow in range(tile):
            acc = 0.0
            for kh in range(3):
                for kw in range(3):
                    ih = oh + kh
                    iw = ow + kw
                    if ih >= image.shape[1] or iw >= image.shape[2]:
                        continue
                    acc += float(np.dot(image[:, ih, iw], kernel[:, kh, kw]))
            out[oh, ow] = acc
    return out


def _winograd_tile(
    image: np.ndarray,
    kernel: np.ndarray,
    stored_u: np.ndarray | None,
    tile: int,
) -> tuple[np.ndarray, bool, bool]:
    """@brief 计算一个输出 tile。

    @return ``(输出, 变换是否全程有限, U 是否全程有限)``。
    """
    transform_b, transform_g, transform_a = matrices(tile)
    alpha = transform_b.shape[0]
    channels = image.shape[0]
    padded = np.zeros((channels, alpha, alpha), dtype=np.float64)
    copy_h = min(alpha, image.shape[1])
    copy_w = min(alpha, image.shape[2])
    padded[:, :copy_h, :copy_w] = image[:, :copy_h, :copy_w]
    u_values = np.empty((alpha * alpha, channels), dtype=np.float64)
    for channel in range(channels):
        transformed = _weight_transform(kernel[channel], transform_g).reshape(-1)
        u_values[:, channel] = transformed if stored_u is None else stored_u[:, channel]
    u_finite = bool(np.isfinite(u_values).all())
    accum = np.zeros(alpha * alpha, dtype=np.float64)
    transform_finite = True
    for channel in range(channels):
        transformed = transform_b @ padded[channel] @ transform_b.T
        flat = transformed.reshape(-1)
        if not np.isfinite(flat).all():
            transform_finite = False
        accum += flat * u_values[:, channel]
    if not np.isfinite(accum).all():
        transform_finite = False
    matrix = accum.reshape(alpha, alpha)
    output = transform_a @ matrix @ transform_a.T
    if not np.isfinite(output).all():
        transform_finite = False
    return output, transform_finite, u_finite


def conv2d_winograd(
    image: np.ndarray,
    weight: np.ndarray,
    *,
    tile: int,
    use_stored_u: bool = True,
    drop_last_oc: bool = False,
) -> np.ndarray:
    """@brief NHWC 直接卷积的 Winograd 复现，含 partial tail 与非有限回退。

    ``image`` 是 ``[N,H,W,C]``，``weight`` 是 ``[OC,3,3,C]``。``drop_last_oc`` 让最后
    一个 OC 的 U 保持 NaN，用来证明漏写不会被直接卷积悄悄补上。
    """
    transform_b, transform_g, _transform_a = matrices(tile)
    alpha = int(transform_b.shape[0])
    count, height, width, channels = image.shape
    outputs, _kh, _kw, weight_channels = weight.shape
    if weight_channels != channels:
        raise ValueError("weight channels do not match the image")
    out_h = height - 2
    out_w = width - 2
    result = np.zeros((count, out_h, out_w, outputs), dtype=np.float64)
    tiles_h = (out_h + tile - 1) // tile
    tiles_w = (out_w + tile - 1) // tile
    for n_index in range(count):
        for oc in range(outputs):
            kernel = np.transpose(weight[oc], (2, 0, 1))  # [C, 3, 3]
            stored = np.empty((alpha * alpha, channels), dtype=np.float64)
            for channel in range(channels):
                stored[:, channel] = _weight_transform(kernel[channel], transform_g).reshape(-1)
            if drop_last_oc and oc == outputs - 1:
                stored[:] = np.nan
            for th in range(tiles_h):
                for tw in range(tiles_w):
                    oh0 = th * tile
                    ow0 = tw * tile
                    ih0 = oh0
                    iw0 = ow0
                    window = image[n_index, ih0 : ih0 + alpha, iw0 : iw0 + alpha, :]
                    spatial = np.transpose(window, (2, 0, 1)) if window.size else np.zeros((channels, 0, 0))
                    inputs_bad = not (
                        np.isfinite(spatial).all() and np.isfinite(kernel).all()
                    )
                    produced, transform_finite, u_finite = _winograd_tile(
                        spatial, kernel, stored if use_stored_u else None, tile
                    )
                    if inputs_bad or (not transform_finite and u_finite):
                        direct_image = np.zeros((channels, alpha, alpha), dtype=np.float64)
                        direct_image[:, : spatial.shape[1], : spatial.shape[2]] = spatial
                        produced = _direct_tile(direct_image, kernel, tile)
                    for oh in range(tile):
                        for ow in range(tile):
                            if oh0 + oh >= out_h or ow0 + ow >= out_w:
                                continue
                            result[n_index, oh0 + oh, ow0 + ow, oc] = produced[oh, ow]
    return result


def _literal(value: float) -> str:
    """@brief Metal 浮点字面量。整数也写成 ``4.0f``，``4f`` 不是合法浮点。"""
    text = f"{float(value):.16g}"
    if "e" not in text and "E" not in text and "." not in text:
        text += ".0"
    return f"{text}f"


def _matrix_literal(name: str, matrix: np.ndarray) -> str:
    values = ", ".join(_literal(float(item)) for item in matrix.reshape(-1))
    return f"constant float {name}[{matrix.size}] = {{{values}}};"


def launch_counts(symbols: dict[str, int], tile: int, *, threads: int = 64) -> tuple[int, int]:
    """@brief paper/small 上能整除的 threadgroup 数。"""
    stage0 = int(symbols["C"]) * int(symbols["OC"])
    tiles_h = (int(symbols["OH"]) + tile - 1) // tile
    tiles_w = (int(symbols["OW"]) + tile - 1) // tile
    stage1 = int(symbols["N"]) * tiles_h * tiles_w * int(symbols["OC"])
    if stage0 % threads or stage1 % threads:
        raise ValueError(f"thread count {threads} does not divide the {tile}x{tile} launch")
    return stage0 // threads, stage1 // threads


def metal_source(
    symbols: dict[str, int],
    tile: int,
    *,
    omit_last_oc: bool = False,
    oob: bool = False,
    zero_last_element: bool = False,
) -> str:
    """@brief 生成两阶段 msl-single 源码。网格按 ``symbols`` 求值后写死，启动块只接受整除表达式。"""
    if sum((omit_last_oc, oob, zero_last_element)) > 1:
        raise ValueError("a negative is exactly one of missed write, partial write, or out-of-bounds")
    transform_b, transform_g, transform_a = matrices(tile)
    alpha = int(transform_b.shape[0])
    elements = alpha * alpha
    stage0, stage1 = launch_counts(symbols, tile)
    width, _height, _row = scratch_texture_layout(
        elements * int(symbols["C"]) * int(symbols["OC"]),
        guard_bytes=GUARD_BYTES,
    )
    skip = "if (oc + 1u == OC) return;\n  " if omit_last_oc else ""
    oob_store = (
        "uwin.write(float4(1.0f), uint2((WIN_ELEMS * C * OC) % UWIN_WIDTH, (WIN_ELEMS * C * OC) / UWIN_WIDTH));\n  "
        if oob
        else ""
    )
    last_element = "if (e + 1u == WIN_ELEMS) sum = 0.0f;\n    " if zero_last_element else ""
    return f"""#include <metal_stdlib>
using namespace metal;

{_matrix_literal("WIN_BT", transform_b)}
{_matrix_literal("WIN_G", transform_g)}
{_matrix_literal("WIN_AT", transform_a)}

constant uint WIN_TILE = {tile}u;
constant uint WIN_ALPHA = {alpha}u;
constant uint WIN_ELEMS = {elements}u;
constant uint UWIN_WIDTH = {width}u;

kernel void conv2d_wgrid(
    device const half* w [[buffer(0)]],
    texture2d<float, access::write> uwin [[texture(0)]],
    uint tid [[thread_position_in_grid]]) {{
  uint total = C * OC;
  if (tid >= total) return;
  uint c = tid % C;
  uint oc = tid / C;
  {skip}float g[9];
  for (uint kh = 0u; kh < 3u; ++kh) {{
    for (uint kw = 0u; kw < 3u; ++kw) {{
      g[kh * 3u + kw] = float(w[((oc * 3u + kh) * 3u + kw) * C + c]);
    }}
  }}
  float tmp[18];
  for (uint i = 0u; i < WIN_ALPHA; ++i) {{
    for (uint j = 0u; j < 3u; ++j) {{
      float sum = 0.0f;
      for (uint k = 0u; k < 3u; ++k) sum += WIN_G[i * 3u + k] * g[k * 3u + j];
      tmp[i * 3u + j] = sum;
    }}
  }}
  for (uint e = 0u; e < WIN_ELEMS; ++e) {{
    uint i = e / WIN_ALPHA;
    uint j = e % WIN_ALPHA;
    float sum = 0.0f;
    for (uint k = 0u; k < 3u; ++k) sum += tmp[i * 3u + k] * WIN_G[j * 3u + k];
    {last_element}uwin.write(float4(sum), uint2(((e * C + c) * OC + oc) % UWIN_WIDTH, ((e * C + c) * OC + oc) / UWIN_WIDTH));
  }}
  {oob_store}}}

kernel void conv2d_kernel(
    device const half* x [[buffer(0)]],
    texture2d<float, access::read> uwin [[texture(0)]],
    device half* y [[buffer(1)]],
    device const half* w [[buffer(2)]],
    uint tid [[thread_position_in_grid]]) {{
  uint tiles_w = (OW + WIN_TILE - 1u) / WIN_TILE;
  uint tiles_h = (OH + WIN_TILE - 1u) / WIN_TILE;
  uint tile_count = N * tiles_h * tiles_w * OC;
  if (tid >= tile_count) return;
  uint oc = tid % OC;
  uint rest = tid / OC;
  uint tw = rest % tiles_w;
  rest /= tiles_w;
  uint th = rest % tiles_h;
  uint n = rest / tiles_h;
  uint oh0 = th * WIN_TILE;
  uint ow0 = tw * WIN_TILE;
  bool inputs_bad = false;
  for (uint c = 0u; c < C && !inputs_bad; ++c) {{
    for (uint kh = 0u; kh < 3u && !inputs_bad; ++kh) {{
      for (uint kw = 0u; kw < 3u; ++kw) {{
        float weight = float(w[((oc * 3u + kh) * 3u + kw) * C + c]);
        if (!isfinite(weight)) inputs_bad = true;
      }}
    }}
  }}
  for (uint ih = 0u; ih < WIN_ALPHA && !inputs_bad; ++ih) {{
    for (uint iw = 0u; iw < WIN_ALPHA && !inputs_bad; ++iw) {{
      uint h = oh0 + ih;
      uint wpos = ow0 + iw;
      if (h >= H || wpos >= W) continue;
      for (uint c = 0u; c < C; ++c) {{
        float value = float(x[((n * H + h) * W + wpos) * C + c]);
        if (!isfinite(value)) inputs_bad = true;
      }}
    }}
  }}
  float tile_out[16];
  bool transform_finite = true;
  bool u_finite = true;
  for (uint e = 0u; e < WIN_ELEMS; ++e) {{
    for (uint c = 0u; c < C; ++c) {{
      float stored = uwin.read(uint2(((e * C + c) * OC + oc) % UWIN_WIDTH, ((e * C + c) * OC + oc) / UWIN_WIDTH)).x;
      if (!isfinite(stored)) u_finite = false;
    }}
  }}
  float accum[36];
  for (uint e = 0u; e < WIN_ELEMS; ++e) accum[e] = 0.0f;
  for (uint c = 0u; c < C; ++c) {{
    float d[36];
    for (uint i = 0u; i < WIN_ALPHA; ++i) {{
      for (uint j = 0u; j < WIN_ALPHA; ++j) {{
        uint h = oh0 + i;
        uint wpos = ow0 + j;
        float value = 0.0f;
        if (h < H && wpos < W) value = float(x[((n * H + h) * W + wpos) * C + c]);
        d[i * WIN_ALPHA + j] = value;
      }}
    }}
    float v[36];
    for (uint i = 0u; i < WIN_ALPHA; ++i) {{
      for (uint j = 0u; j < WIN_ALPHA; ++j) {{
        float sum = 0.0f;
        for (uint k = 0u; k < WIN_ALPHA; ++k) sum += WIN_BT[i * WIN_ALPHA + k] * d[k * WIN_ALPHA + j];
        v[i * WIN_ALPHA + j] = sum;
      }}
    }}
    float vt[36];
    for (uint i = 0u; i < WIN_ALPHA; ++i) {{
      for (uint j = 0u; j < WIN_ALPHA; ++j) {{
        float sum = 0.0f;
        for (uint k = 0u; k < WIN_ALPHA; ++k) sum += v[i * WIN_ALPHA + k] * WIN_BT[j * WIN_ALPHA + k];
        vt[i * WIN_ALPHA + j] = sum;
        if (!isfinite(vt[i * WIN_ALPHA + j])) transform_finite = false;
      }}
    }}
    for (uint e = 0u; e < WIN_ELEMS; ++e) {{
      float stored = uwin.read(uint2(((e * C + c) * OC + oc) % UWIN_WIDTH, ((e * C + c) * OC + oc) / UWIN_WIDTH)).x;
      accum[e] += vt[e] * stored;
      if (!isfinite(accum[e])) transform_finite = false;
    }}
  }}
  float mid[24];
  uint at_cols = WIN_ALPHA;
  for (uint i = 0u; i < WIN_TILE; ++i) {{
    for (uint j = 0u; j < WIN_ALPHA; ++j) {{
      float sum = 0.0f;
      for (uint k = 0u; k < at_cols; ++k) sum += WIN_AT[i * WIN_ALPHA + k] * accum[k * WIN_ALPHA + j];
      mid[i * WIN_ALPHA + j] = sum;
    }}
  }}
  for (uint i = 0u; i < WIN_TILE; ++i) {{
    for (uint j = 0u; j < WIN_TILE; ++j) {{
      float sum = 0.0f;
      for (uint k = 0u; k < WIN_ALPHA; ++k) sum += mid[i * WIN_ALPHA + k] * WIN_AT[j * WIN_ALPHA + k];
      tile_out[i * WIN_TILE + j] = sum;
      if (!isfinite(sum)) transform_finite = false;
    }}
  }}
  bool direct = inputs_bad || (!transform_finite && u_finite);
  if (direct) {{
    for (uint oh = 0u; oh < WIN_TILE; ++oh) {{
      for (uint ow = 0u; ow < WIN_TILE; ++ow) {{
        uint y_h = oh0 + oh;
        uint y_w = ow0 + ow;
        if (y_h >= OH || y_w >= OW) continue;
        float acc = 0.0f;
        for (uint kh = 0u; kh < 3u; ++kh) {{
          for (uint kw = 0u; kw < 3u; ++kw) {{
            float dot = 0.0f;
            for (uint c = 0u; c < C; ++c) {{
              float xv = float(x[((n * H + (y_h + kh)) * W + (y_w + kw)) * C + c]);
              float wv = float(w[((oc * 3u + kh) * 3u + kw) * C + c]);
              dot += xv * wv;
            }}
            acc += dot;
          }}
        }}
        y[((n * OH + y_h) * OW + y_w) * OC + oc] = half(acc);
      }}
    }}
    return;
  }}
  for (uint oh = 0u; oh < WIN_TILE; ++oh) {{
    for (uint ow = 0u; ow < WIN_TILE; ++ow) {{
      uint y_h = oh0 + oh;
      uint y_w = ow0 + ow;
      if (y_h >= OH || y_w >= OW) continue;
      y[((n * OH + y_h) * OW + y_w) * OC + oc] = half(tile_out[oh * WIN_TILE + ow]);
    }}
  }}
}}

/* taic-launch
math safe
scratch uwin fp32 {elements}*C*OC
launch conv2d_wgrid grid={stage0},1,1 threads=64,1,1 scratch=uwin:write
launch conv2d_kernel grid={stage1},1,1 threads=64,1,1 scratch=uwin:read
*/
"""


def metal_source_recompute(symbols: dict[str, int], tile: int) -> str:
    """@brief 单阶段 F(tile) 源码：每个输出 tile 自己重算 U，不声明 scratch。"""
    transform_b, transform_g, transform_a = matrices(tile)
    alpha = int(transform_b.shape[0])
    elements = alpha * alpha
    _stage0, stage1 = launch_counts(symbols, tile)
    return f"""#include <metal_stdlib>
using namespace metal;

{_matrix_literal("WIN_BT", transform_b)}
{_matrix_literal("WIN_G", transform_g)}
{_matrix_literal("WIN_AT", transform_a)}

constant uint WIN_TILE = {tile}u;
constant uint WIN_ALPHA = {alpha}u;
constant uint WIN_ELEMS = {elements}u;

kernel void conv2d_kernel(
    device const half* x [[buffer(0)]],
    device const half* w [[buffer(1)]],
    device half* y [[buffer(2)]],
    uint tid [[thread_position_in_grid]]) {{
  uint tiles_w = (OW + WIN_TILE - 1u) / WIN_TILE;
  uint tiles_h = (OH + WIN_TILE - 1u) / WIN_TILE;
  uint tile_count = N * tiles_h * tiles_w * OC;
  if (tid >= tile_count) return;
  uint oc = tid % OC;
  uint rest = tid / OC;
  uint tw = rest % tiles_w;
  rest /= tiles_w;
  uint th = rest % tiles_h;
  uint n = rest / tiles_h;
  uint oh0 = th * WIN_TILE;
  uint ow0 = tw * WIN_TILE;
  bool inputs_bad = false;
  for (uint c = 0u; c < C && !inputs_bad; ++c) {{
    for (uint kh = 0u; kh < 3u && !inputs_bad; ++kh) {{
      for (uint kw = 0u; kw < 3u; ++kw) {{
        float weight = float(w[((oc * 3u + kh) * 3u + kw) * C + c]);
        if (!isfinite(weight)) inputs_bad = true;
      }}
    }}
  }}
  for (uint ih = 0u; ih < WIN_ALPHA && !inputs_bad; ++ih) {{
    for (uint iw = 0u; iw < WIN_ALPHA && !inputs_bad; ++iw) {{
      uint h = oh0 + ih;
      uint wpos = ow0 + iw;
      if (h >= H || wpos >= W) continue;
      for (uint c = 0u; c < C; ++c) {{
        float value = float(x[((n * H + h) * W + wpos) * C + c]);
        if (!isfinite(value)) inputs_bad = true;
      }}
    }}
  }}
  float tile_out[16];
  bool transform_finite = true;
  bool u_finite = true;
  float accum[36];
  for (uint e = 0u; e < WIN_ELEMS; ++e) accum[e] = 0.0f;
  for (uint c = 0u; c < C; ++c) {{
    float g[9];
    for (uint kh = 0u; kh < 3u; ++kh) {{
      for (uint kw = 0u; kw < 3u; ++kw) {{
        g[kh * 3u + kw] = float(w[((oc * 3u + kh) * 3u + kw) * C + c]);
      }}
    }}
    float tmp[18];
    for (uint i = 0u; i < WIN_ALPHA; ++i) {{
      for (uint j = 0u; j < 3u; ++j) {{
        float sum = 0.0f;
        for (uint k = 0u; k < 3u; ++k) sum += WIN_G[i * 3u + k] * g[k * 3u + j];
        tmp[i * 3u + j] = sum;
      }}
    }}
    float u[36];
    for (uint e = 0u; e < WIN_ELEMS; ++e) {{
      uint i = e / WIN_ALPHA;
      uint j = e % WIN_ALPHA;
      float sum = 0.0f;
      for (uint k = 0u; k < 3u; ++k) sum += tmp[i * 3u + k] * WIN_G[j * 3u + k];
      u[e] = sum;
      if (!isfinite(u[e])) u_finite = false;
    }}
    float d[36];
    for (uint i = 0u; i < WIN_ALPHA; ++i) {{
      for (uint j = 0u; j < WIN_ALPHA; ++j) {{
        uint h = oh0 + i;
        uint wpos = ow0 + j;
        float value = 0.0f;
        if (h < H && wpos < W) value = float(x[((n * H + h) * W + wpos) * C + c]);
        d[i * WIN_ALPHA + j] = value;
      }}
    }}
    float v[36];
    for (uint i = 0u; i < WIN_ALPHA; ++i) {{
      for (uint j = 0u; j < WIN_ALPHA; ++j) {{
        float sum = 0.0f;
        for (uint k = 0u; k < WIN_ALPHA; ++k) sum += WIN_BT[i * WIN_ALPHA + k] * d[k * WIN_ALPHA + j];
        v[i * WIN_ALPHA + j] = sum;
      }}
    }}
    float vt[36];
    for (uint i = 0u; i < WIN_ALPHA; ++i) {{
      for (uint j = 0u; j < WIN_ALPHA; ++j) {{
        float sum = 0.0f;
        for (uint k = 0u; k < WIN_ALPHA; ++k) sum += v[i * WIN_ALPHA + k] * WIN_BT[j * WIN_ALPHA + k];
        vt[i * WIN_ALPHA + j] = sum;
        if (!isfinite(vt[i * WIN_ALPHA + j])) transform_finite = false;
      }}
    }}
    for (uint e = 0u; e < WIN_ELEMS; ++e) {{
      accum[e] += vt[e] * u[e];
      if (!isfinite(accum[e])) transform_finite = false;
    }}
  }}
  float mid[24];
  for (uint i = 0u; i < WIN_TILE; ++i) {{
    for (uint j = 0u; j < WIN_ALPHA; ++j) {{
      float sum = 0.0f;
      for (uint k = 0u; k < WIN_ALPHA; ++k) sum += WIN_AT[i * WIN_ALPHA + k] * accum[k * WIN_ALPHA + j];
      mid[i * WIN_ALPHA + j] = sum;
    }}
  }}
  for (uint i = 0u; i < WIN_TILE; ++i) {{
    for (uint j = 0u; j < WIN_TILE; ++j) {{
      float sum = 0.0f;
      for (uint k = 0u; k < WIN_ALPHA; ++k) sum += mid[i * WIN_ALPHA + k] * WIN_AT[j * WIN_ALPHA + k];
      tile_out[i * WIN_TILE + j] = sum;
      if (!isfinite(sum)) transform_finite = false;
    }}
  }}
  bool direct = inputs_bad || (!transform_finite && u_finite);
  if (direct) {{
    for (uint oh = 0u; oh < WIN_TILE; ++oh) {{
      for (uint ow = 0u; ow < WIN_TILE; ++ow) {{
        uint y_h = oh0 + oh;
        uint y_w = ow0 + ow;
        if (y_h >= OH || y_w >= OW) continue;
        float acc = 0.0f;
        for (uint kh = 0u; kh < 3u; ++kh) {{
          for (uint kw = 0u; kw < 3u; ++kw) {{
            float dot = 0.0f;
            for (uint c = 0u; c < C; ++c) {{
              float xv = float(x[((n * H + (y_h + kh)) * W + (y_w + kw)) * C + c]);
              float wv = float(w[((oc * 3u + kh) * 3u + kw) * C + c]);
              dot += xv * wv;
            }}
            acc += dot;
          }}
        }}
        y[((n * OH + y_h) * OW + y_w) * OC + oc] = half(acc);
      }}
    }}
    return;
  }}
  for (uint oh = 0u; oh < WIN_TILE; ++oh) {{
    for (uint ow = 0u; ow < WIN_TILE; ++ow) {{
      uint y_h = oh0 + oh;
      uint y_w = ow0 + ow;
      if (y_h >= OH || y_w >= OW) continue;
      y[((n * OH + y_h) * OW + y_w) * OC + oc] = half(tile_out[oh * WIN_TILE + ow]);
    }}
  }}
}}

/* taic-launch
math safe
launch conv2d_kernel grid={stage1},1,1 threads=64,1,1
*/
"""
