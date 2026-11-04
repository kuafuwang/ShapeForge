"""Mamba-2 (SSD) chunk scan: the reference against an independent sequential-recurrence formulation, input invariants, baselines."""

from __future__ import annotations

import pytest
import torch

from tasks import get_task
from tcenv.evaluate import evaluate
from tcenv.backends.metal.testing import prepare_case

# The generator's ranges and the chunk length are *registered* facts now (R2): the task module no
# longer carries them as module constants, so the test reads the same source the generator does.
from tcenv.track import REPO_ROOT

_TRACK = REPO_ROOT / "tasks" / "mamba2_chunk_scan" / "track.json"


def _registered() -> tuple[float, float, float, int]:
    """@brief 取登记的 ``(DT_LO, DT_HI, A_HI, CHUNK)``。

    B3 把 mamba2 迁到 v2 标准入口后，``track.json`` 不再存在；这些值现在从 B3 的冻结
    证据（迁移前 v1 的登记参数）和 v2 的 ``opspec.json`` 读取，含义不变。

    @return 步长区间、衰减上界与 chunk 长度。
    """
    import json

    evidence = json.loads(
        (REPO_ROOT / "docs" / "架构" / "跨硬件算子生成分工-阶段验收" / "B3" / "evidence" / "v1-frozen.json").read_text(encoding="utf-8")
    )
    combo = next(combo for combo in evidence["combos"] if combo["task"] == "mamba2_chunk_scan")
    params = combo["implementations"]["gen_inputs"]["params"]
    dt_lo, dt_hi = (float(value) for value in params["dt_range"])
    a_hi = float(params["decay_range"][1])
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


def _evaluate(source: str, dtype: str = "fp16"):
    """@brief 在真实 v2 ``evaluate_task`` 上评测一个 mamba2 候选（small）。

    几何沿用 v1 的 grid=(NCHUNKS, BATCH*NHEADS, 1)：NCHUNKS=4、BATCH*NHEADS=2。
    """
    from tcenv.backends.metal.track_adapter import MetalTrackBuilder, MetalTrackEvaluator, msl_single_geometry_resolver
    from tcenv.taskcontract.locator import TaskLocator
    from tcenv.taskcontract.pack import task_pack_closure
    from tcenv.taskcontract.runner import evaluate_task
    from tcenv.track import REPO_ROOT as TRACK_ROOT

    directory = TRACK_ROOT / "tasks" / "mamba2_chunk_scan"
    contract = _contract(dtype, "small")
    locator = TaskLocator(
        task_id="mamba2_chunk_scan", op_name=contract.op_name, dtype=contract.dtype, size=contract.size,
        pack_digest=contract.pack_digest,
        deployment_pack_digest=task_pack_closure(directory, root=TRACK_ROOT).digest,
        task_dir=directory, code_root=TRACK_ROOT,
    )
    grid = (int(contract.instance.symbols["NCHUNKS"]), int(contract.instance.symbols["BATCH"]) * int(contract.instance.symbols["NHEADS"]), 1)
    launch = (
        "/* taic-launch\nmath fast\n"
        f"launch mamba2_chunk_scan_kernel grid={grid[0]},{grid[1]},{grid[2]} threads=128,1,1\n*/\n"
    )
    return evaluate_task(
        locator,
        track_id="msl-single",
        candidate_source=launch + source,
        builder=MetalTrackBuilder(),
        evaluator=MetalTrackEvaluator(
            mode="quick", seed=4, reps=1,
            cases=("normal", "wide_range", "large", "tiny", "outlier", "ieee"),
        ),
        geometry_resolver=msl_single_geometry_resolver(contract),
        track_registry_path=TRACK_ROOT / "backends" / "metal" / "tracks.json",
    )


def _naive(dtype: str = "fp16") -> str:
    """@brief B3 归档的 naive 候选文本（含 launch 所需的全部常量，v2 合同头提供符号）。"""
    path = REPO_ROOT / "docs" / "架构" / "跨硬件算子生成分工-阶段验收" / "B3" / "evidence" / "candidates" / "mamba2_chunk_scan-naive.msl"
    spelling = "half" if dtype == "fp16" else "bfloat"
    return path.read_text(encoding="utf-8").replace("{{T}}", spelling)




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


@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
@pytest.mark.gpu
def test_naive_kernel_is_correct_and_the_spec_is_consistent(dtype):
    """@brief 验证 naive 内核能通过评测，且 triton 源码、生成的 MSL 与 contract 的网格彼此一致。

    @param dtype 参数化的数据格式（fp16、bf16）。
    """
    result = _evaluate(_naive(dtype), dtype)
    report = result.report
    assert report["status"] == "ok", report.get("message")
    # v2 的题面是 opspec + 独立 reference（没有 Triton spec）；这里改为核对 SSD 算子确实在
    # reference 模块里、候选没有未替换的占位符，以及真实 dispatch 的几何与 v1 一致。
    reference_text = (REPO_ROOT / "tasks" / "mamba2_chunk_scan" / "reference.py").read_text(encoding="utf-8")
    assert "ssd_chunk_scan" in reference_text and "{{" not in _naive(dtype)
    contract = _contract(dtype, "small")
    expected = (
        int(contract.instance.symbols["NCHUNKS"]),
        int(contract.instance.symbols["BATCH"]) * int(contract.instance.symbols["NHEADS"]),
        1,
    )
    assert [(r["kernel"], tuple(r["grid"]), tuple(r["threads"])) for r in result.execution["dispatch"]] == [
        ("mamba2_chunk_scan_kernel", expected, (128, 1, 1))
    ]


@pytest.mark.gpu
def test_a_kernel_that_ignores_the_carried_in_state_or_the_causal_mask_fails():
    """@brief 验证丢掉带进来的状态、去掉因果掩码、或漏掉 skip 项的变体在内核层面都被判 incorrect。
    """
    src = _naive("fp16")
    no_state = src.replace("acc[p] = s * e;", "acc[p] = 0.0f;")
    no_mask = src.replace("for (uint s = 0; s <= l; ++s) {", "for (uint s = 0; s < CHUNK; ++s) {")
    no_skip = src.replace("y[i] = {{T}}(acc[p] + d * float(x[i]));".replace("{{T}}", "half"), "y[i] = half(acc[p]);")
    for name, bad in (("no_state", no_state), ("no_mask", no_mask), ("no_skip", no_skip)):
        assert bad != src, name
        report = _evaluate(bad, "fp16").report
        assert report["status"] == "incorrect", (name, report["status"], report.get("message"))
