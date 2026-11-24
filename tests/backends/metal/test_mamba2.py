"""Mamba-2 (SSD) chunk scan: the reference against an independent sequential-recurrence formulation, input invariants, baselines."""

from __future__ import annotations

import pytest
import torch

from tcenv.track import REPO_ROOT


def _registered() -> tuple[float, float, float, int]:
    """@brief 取登记的 ``(DT_LO, DT_HI, A_HI, CHUNK)``。

    取值直接来自当前 v2 reference 模块和 ``opspec.json``，不读取迁移快照。

    @return 步长区间、衰减上界与 chunk 长度。
    """
    import json

    from tasks.mamba2_chunk_scan import reference as current_reference

    dt_lo, dt_hi = current_reference._DT_LO, current_reference._DT_HI
    a_hi = current_reference._DECAY_HI
    opspec = json.loads((REPO_ROOT / "tasks" / "mamba2_chunk_scan" / "opspec.json").read_text(encoding="utf-8"))
    chunk = int(opspec["sizes"]["small"]["CHUNK"])
    return dt_lo, dt_hi, a_hi, chunk


DT_LO, DT_HI, A_HI, CHUNK = _registered()


def _contract(dtype: str = "fp16", size: str = "small"):
    """@brief 装配 v2 的 mamba2 合同（B3 之后不再有 v1 装配）。"""
    from tcenv.taskcontract import load_task_contract

    return load_task_contract(REPO_ROOT / "tasks" / "mamba2_chunk_scan", dtype=dtype, size=size)


def _case(contract, case: str, seed: int):
    """@brief 用 v2 的真实算例生成路径取一个 (case, seed)。"""
    from tcenv.backends.metal.v2_runtime import prepare_v2_case

    return prepare_v2_case(contract, case, seed)



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
    contract = _contract("fp16", "paper")
    batch, seq, heads, headdim, dstate = 2, 2048, 8, 64, 64
    chunks = seq // CHUNK
    shared = batch * chunks * (2 * CHUNK * CHUNK * dstate)
    per_head = batch * heads * chunks * (2 * CHUNK * CHUNK * headdim + 2 * CHUNK * headdim * dstate)
    assert contract.instance.work_accounting.flops == shared + per_head == 570_425_344


@pytest.mark.parametrize("case", ["normal", "wide_range", "large", "outlier"])
def test_reference_equals_the_sequential_state_recurrence(case):
    """@brief 验证参考实现在各类输入分布下都与独立的逐时间步递推一致（容差相对且随量级缩放）。

    @param case 参数化的输入用例名（normal、wide_range、large、outlier）。
    """
    contract = _contract("fp16", "small")
    d = _case(contract, case, 21)
    B, S, H, P = (int(v) for v in contract.instance.tensor("x").shape)
    N = int(contract.instance.tensor("Bmat").shape[-1])
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
    contract = _contract(dtype, "small")
    d = _case(contract, case, 5)
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
    contract = _contract("fp16", "small")
    d = _case(contract, "normal", 3)
    inp = {k: v.clone() for k, v in d.rounded.items()}
    base = contract.reference.execute(contract.instance, inp)["y"]
    inp["x"][0, 40, 1, :] += 5.0  # a later step of chunk 0 changes ...
    out = contract.reference.execute(contract.instance, inp)["y"]
    changed = (out - base).abs().amax(dim=(2, 3))[0]
    assert float(changed[:40].max()) == 0.0  # ... nothing before it,
    assert float(changed[40]) > 0 and float(changed[CHUNK:].max()) == 0.0  # ... itself and the rest of its chunk only
