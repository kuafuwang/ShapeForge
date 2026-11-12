"""CPU proof that the conv2d Winograd matrices match direct convolution.

The GPU file launches the same generator.  This check stays off the GPU: a wrong
matrix or a missed-write fallback that silently recomputes from ``w`` fails here.
"""

from __future__ import annotations

import numpy as np
import torch

from .conv2d_scratch_kernel import (
    _direct_tile,
    _winograd_tile,
    conv2d_winograd,
    launch_counts,
    metal_source,
    metal_source_recompute,
)


def _pair(tile: int, *, seed: int, height: int = 56, channels: int = 4, outputs: int = 3, scale: float = 1.0):
    generator = torch.Generator().manual_seed(seed)
    image = torch.randn(2, height, height, channels, generator=generator) * scale
    weight = torch.randn(outputs, 3, 3, channels, generator=generator) * scale
    got = conv2d_winograd(image.double().numpy(), weight.double().numpy(), tile=tile)
    ref = torch.nn.functional.conv2d(
        image.double().permute(0, 3, 1, 2), weight.double().permute(0, 3, 1, 2)
    ).permute(0, 2, 3, 1).numpy()
    return got, ref


def test_f22_and_f44_match_direct_conv_including_the_partial_tail():
    """@brief F(2x2)/F(4x4) 在 54=13*4+2 的尾图上与直接卷积一致，含 image 0 和最后一张。"""
    for tile in (2, 4):
        got, ref = _pair(tile, seed=11 + tile)
        assert got.shape == ref.shape == (2, 54, 54, 3)
        assert np.max(np.abs(got - ref)) < 1e-8
        assert np.max(np.abs(got[0, 0] - ref[0, 0])) < 1e-8
        assert np.max(np.abs(got[1, 52:54] - ref[1, 52:54])) < 1e-8


def test_nonfinite_inputs_follow_the_direct_sum_and_a_missed_write_does_not():
    """@brief Inf 逐 tile 回退到直接和；漏写最后一个 OC 不得被这次回退补成正确结果。"""
    generator = torch.Generator().manual_seed(19)
    image = torch.randn(1, 18, 18, 3, generator=generator).double().numpy()
    weight = torch.randn(2, 3, 3, 3, generator=generator).double().numpy()
    image[0, 0, 0, 0] = np.inf
    got = conv2d_winograd(image, weight, tile=4)
    x32 = torch.from_numpy(image).float()
    w32 = torch.from_numpy(weight).float()
    out_h = image.shape[1] - 2
    out_w = image.shape[2] - 2
    ref = torch.zeros((1, out_h, out_w, weight.shape[0]))
    for kh in range(3):
        for kw in range(3):
            ref += x32[:, kh : kh + out_h, kw : kw + out_w, :] @ w32[:, kh, kw, :].T
    ref = ref.numpy()
    assert np.array_equal(np.isfinite(got), np.isfinite(ref))
    finite = np.isfinite(ref)
    assert np.max(np.abs(got[finite] - ref[finite])) < 1e-4
    clean = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)
    missed = conv2d_winograd(clean, weight, tile=2, drop_last_oc=True)
    assert not np.isfinite(missed[..., -1]).all()


def test_a_finite_u_with_an_overflowing_transform_uses_direct_conv():
    """@brief 输入和 U 都有限、变换中间溢出时，这个 tile 走直接卷积，结果仍正确。"""
    image = np.zeros((1, 8, 8, 1), dtype=np.float64)
    weight = np.ones((1, 3, 3, 1), dtype=np.float64) * 1e-200
    image[0, 0, 0, 0] = 8e307
    assert np.isfinite(image).all() and np.isfinite(weight).all()
    spatial = np.transpose(image[0, :6, :6, :], (2, 0, 1))
    kernel = np.transpose(weight[0], (2, 0, 1))
    produced, transform_finite, u_finite = _winograd_tile(spatial, kernel, None, 4)
    assert u_finite is True
    assert transform_finite is False
    assert not np.isfinite(produced).all()
    got = conv2d_winograd(image, weight, tile=4)
    direct = _direct_tile(spatial, kernel, 4)
    assert np.isfinite(got).all()
    assert np.allclose(got[0, :4, :4, 0], direct)
    assert np.allclose(got[0, 4:, :, 0], 0.0)
    assert np.allclose(got[0, :, 4:, 0], 0.0)


def test_paper_launch_grid_matches_the_tile_counts():
    """@brief paper 的两组网格能整除，F(4x4) scratch 是 1,179,648 字节。"""
    symbols = {"N": 32, "H": 56, "W": 56, "C": 64, "OC": 128, "OH": 54, "OW": 54, "KH": 3, "KW": 3}
    assert launch_counts(symbols, 4) == (128, 12544)
    assert launch_counts(symbols, 2) == (128, 46656)
    source = metal_source(symbols, 4)
    assert "scratch uwin fp32 36*C*OC" in source
    assert "launch conv2d_wgrid grid=128,1,1 threads=64,1,1 scratch=uwin:write" in source
    assert "launch conv2d_kernel grid=12544,1,1 threads=64,1,1 scratch=uwin:read" in source
    assert "uwin[36u * C * OC]" not in source
    missed = metal_source(symbols, 4, omit_last_oc=True)
    assert "if (oc + 1u == OC) return;" in missed
    oob = metal_source(symbols, 4, oob=True)
    assert "uwin[36u * C * OC]" not in oob
    assert (
        "uwin.write(float4(1.0f), uint2((WIN_ELEMS * C * OC) % UWIN_WIDTH, "
        "(WIN_ELEMS * C * OC) / UWIN_WIDTH));"
    ) in oob
    single = metal_source_recompute(symbols, 2)
    assert "scratch " not in single
    assert "launch conv2d_kernel grid=46656,1,1 threads=64,1,1\n" in single
