"""Mamba-2 (SSD) chunk scan: the reference against an independent sequential-recurrence formulation, input invariants, baselines."""

from __future__ import annotations

import pytest
import torch

from tasks import get_task
from tasks.mamba2_chunk_scan.task import A_HI, CHUNK, DT_HI, DT_LO
from tcenv.evaluate import evaluate
from tcenv.backends.metal.testing import prepare_case


def sequential_scan(inp, B, S, H, P, N):
    """@brief 用逐时间步的状态递推算出 y：h_t = exp(cs_t - cs_{t-1}) h_{t-1} + dt_t x_t B_t^T，y_t = C_t h_t + D x_t，每个 chunk 起点从给定状态重新开始（float64）。

    这一形式与被测参考实现所用的矩阵（SSD）形式无关，两个实现因此可以互相校验。

    @param inp 该用例的输入字典（x、dt、dA_cumsum、Bmat、Cmat、Dskip、states）。
    @param B batch 大小。
    @param S 序列长度。
    @param H 头数。
    @param P 每个头的维度。
    @param N 状态维度。
    @return 形状 [B, S, H, P] 的 float64 输出。
    """
    x, dt, cs, Bm, Cm, D, st = (inp[k].double() for k in ("x", "dt", "dA_cumsum", "Bmat", "Cmat", "Dskip", "states"))
    NC = S // CHUNK
    y = torch.zeros(B, S, H, P, dtype=torch.float64)
    for b in range(B):
        for c in range(NC):
            h = st[b, c].clone()  # [H, P, N]
            prev_cs = torch.zeros(H, dtype=torch.float64)
            for l in range(CHUNK):
                t = c * CHUNK + l
                cur = cs[b, :, c, l]  # [H]
                h = torch.exp(cur - prev_cs)[:, None, None] * h + dt[b, t][:, None, None] * x[b, t][:, :, None] * Bm[b, t][None, None, :]
                y[b, t] = torch.einsum("hpn,n->hp", h, Cm[b, t]) + D[:, None] * x[b, t]
                prev_cs = cur
    return y


def test_paper_flops_count_the_shared_bc_product_once():
    """@brief 验证 paper 规模的 FLOPs 把共享的 B 与 C 乘积只算一次，总数等于手算的 570_425_344。
    """
    task = get_task("mamba2_chunk_scan", "fp16", "paper")
    batch, seq, heads, headdim, dstate = 2, 2048, 8, 64, 64
    chunks = seq // CHUNK
    shared = batch * chunks * (2 * CHUNK * CHUNK * dstate)
    per_head = batch * heads * chunks * (2 * CHUNK * CHUNK * headdim + 2 * CHUNK * headdim * dstate)
    assert task.flops == shared + per_head == 570_425_344


@pytest.mark.parametrize("case", ["normal", "wide_range", "large", "outlier"])
def test_reference_equals_the_sequential_state_recurrence(case):
    """@brief 验证参考实现在各类输入分布下都与独立的逐时间步递推一致（容差相对且随量级缩放）。

    @param case 参数化的输入用例名（normal、wide_range、large、outlier）。
    """
    t = get_task("mamba2_chunk_scan", "fp16", "small")
    d = prepare_case(t, case, 21)
    B, S, H, P = t.contract.arg("x").shape
    N = t.contract.arg("Bmat").shape[-1]
    seq = sequential_scan(d.rounded, B, S, H, P, N)
    ref = d.ref["y"]
    assert torch.allclose(seq, ref, rtol=1e-9, atol=1e-9 * float(ref.abs().max())), float((seq - ref).abs().max())


@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
@pytest.mark.parametrize("case", ["normal", "wide_range", "large", "tiny", "outlier", "ieee"])
def test_inputs_keep_the_ssm_invariants_in_every_case(dtype, case):
    """@brief 验证各用例的输入都满足 SSM 的不变量：dt 在区间内、dA_cumsum 非增且不为正、衰减不会把 exp 冲成 0；ieee 用例只在带进来的状态里放特殊值。

    @param dtype 参数化的数据格式（fp16、bf16）。
    @param case 参数化的输入用例名。
    """
    t = get_task("mamba2_chunk_scan", dtype, "small")
    d = prepare_case(t, case, 5)
    dt, cs = d.rounded["dt"].double(), d.rounded["dA_cumsum"].double()
    assert bool((dt > 0).all()) and float(dt.min()) >= DT_LO * 0.99 and float(dt.max()) <= DT_HI * 1.01
    assert bool(torch.isfinite(cs).all()) and bool((cs[..., 1:] <= cs[..., :-1]).all()) and bool((cs <= 0).all())  # non-increasing: every decay <= 1
    assert float(cs.min()) > -A_HI * DT_HI * CHUNK * 1.01  # exp(cs) is a normal fp32 number, so no kernel can flush a decay to zero
    if case == "ieee":  # specials only in the carried-in states (they enter through a full dot, no masked 0 * Inf)
        assert not torch.isfinite(d.rounded["states"]).all()
        assert all(torch.isfinite(d.rounded[k]).all() for k in ("x", "dt", "dA_cumsum", "Bmat", "Cmat", "Dskip"))
        y = d.ref["y"]
        assert torch.isnan(y).any() and torch.isfinite(y).double().mean() > 0.9


def test_the_chunk_scan_is_causal_inside_a_chunk():
    """@brief 验证 chunk 内的因果性：改一个时间步的输入只影响它自己和它在同一个 chunk 里的后续输出。
    """
    t = get_task("mamba2_chunk_scan", "fp16", "small")
    d = prepare_case(t, "normal", 3)
    inp = {k: v.clone() for k, v in d.rounded.items()}
    base = t.reference(inp)["y"]
    inp["x"][0, 40, 1, :] += 5.0  # a later step of chunk 0 changes ...
    out = t.reference(inp)["y"]
    changed = (out - base).abs().amax(dim=(2, 3))[0]
    assert float(changed[:40].max()) == 0.0  # ... nothing before it,
    assert float(changed[40]) > 0 and float(changed[CHUNK:].max()) == 0.0  # ... itself and the rest of its chunk only


@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
@pytest.mark.gpu
def test_naive_kernel_is_correct_and_the_spec_is_consistent(dtype):
    """@brief 验证 naive 内核能通过评测，且 triton 源码、生成的 MSL 与 contract 的网格彼此一致。

    @param dtype 参数化的数据格式（fp16、bf16）。
    """
    t = get_task("mamba2_chunk_scan", dtype, "small")
    rep = evaluate("mamba2_chunk_scan", dtype, "small", t.naive_msl, mode="quick", seed=4)
    assert rep.status == "ok", rep.message
    assert "tl.dot" in t.triton_source and "{{" not in t.triton_source and "{{" not in t.naive_msl
    assert t.contract.grid == (t.contract.arg("states").shape[1], t.contract.arg("x").shape[0] * t.contract.arg("x").shape[2], 1)


@pytest.mark.gpu
def test_a_kernel_that_ignores_the_carried_in_state_or_the_causal_mask_fails():
    """@brief 验证丢掉带进来的状态、去掉因果掩码、或漏掉 skip 项的变体在内核层面都被判 incorrect。
    """
    t = get_task("mamba2_chunk_scan", "fp16", "small")
    src = t.naive_msl
    no_state = src.replace("acc[p] = s * e;", "acc[p] = 0.0f;")
    no_mask = src.replace("for (uint s = 0; s <= l; ++s) {", "for (uint s = 0; s < CHUNK; ++s) {")
    no_skip = src.replace("y[i] = {{T}}(acc[p] + d * float(x[i]));".replace("{{T}}", "half"), "y[i] = half(acc[p]);")
    for name, bad in (("no_state", no_state), ("no_mask", no_mask), ("no_skip", no_skip)):
        assert bad != src, name
        rep = evaluate("mamba2_chunk_scan", "fp16", "small", bad, mode="quick", seed=4)
        assert rep.status == "incorrect", (name, rep.status, rep.message)
