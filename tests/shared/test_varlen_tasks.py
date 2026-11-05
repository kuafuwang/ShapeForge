"""``varlen_kda`` / ``varlen_scale``：打包变长（cu_seqlens）标准题的 CPU 侧证据。

Varlen 的语义不在「一个更大的固定张量」里，而在**分段**里：同一道题按不同 ``seq_lens`` 分成
若干档，每档是一个确定的实例。这里在 CPU 上钉住：

* 两道题都被发现为 v2 标准入口，分段边界与派生符号一致（``cu_seqlens`` 首 0、末 total_tokens、
  长度 num_seqs+1）；
* varlen KDA 的 fp64 oracle 与一份**独立手算**的分段递推逐元素一致（不借用 oracle 的中间量）；
* ``gen_inputs`` 给出的 ``cu_seqlens`` 与实例一致，且打包轴长度等于 total_tokens；
* 基线包能加载并闭合，且基线不读候选 workspace / 不读 reference 结果。
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch

from tcenv.dtypes import get_dtype
from tcenv.opspec.v2 import VARLEN_CASE_DOMAIN, instantiate_v2
from tcenv.target import load_target
from tcenv.taskcontract import load_task_contract
from tcenv.taskcontract.baselines import load_baselines
from tcenv.taskcontract.locator import REPO_ROOT, discover_standard_tasks
from tcenv.taskcontract.reference import load_reference
from tcenv.track.abi import derive_tensor_memory_contract

KDA_DIR = REPO_ROOT / "tasks" / "varlen_kda"
SCALE_DIR = REPO_ROOT / "tasks" / "varlen_scale"

KDA_SIZES = {"fixed": [16], "mixed": [4, 8, 2], "uniform": [8, 8]}
SCALE_SIZES = {"mixed": [3, 5, 2], "uniform": [4, 4], "single": [10]}
#: L1 correctness case 域（与 OpSpec ``cases.names`` 一致；``canonical`` 排第一）。
CASES = VARLEN_CASE_DOMAIN


def _discovered() -> dict:
    """@brief 发现标准题入口。

    @return ``{op_name: locator}``。
    """
    return {item.op_name: item for item in discover_standard_tasks(REPO_ROOT / "tasks", code_root=REPO_ROOT)}


@pytest.mark.parametrize("name", ["varlen_kda", "varlen_scale"])
def test_both_varlen_tasks_are_discovered_as_v2_standard_entries(name):
    """@brief 两道 varlen 题都是会被发现的 v2 标准入口。"""
    locators = _discovered()
    assert name in locators
    assert len(locators[name].deployment_pack_digest) == 64


@pytest.mark.parametrize("size,seq_lens", sorted(KDA_SIZES.items()))
def test_every_kda_tier_carries_its_length_distribution(size, seq_lens):
    """@brief 每个 KDA 档的分段边界与派生符号一一对应。"""
    contract = load_task_contract(KDA_DIR, dtype="fp16", size=size)
    instance = contract.instance

    assert instance.cu_seqlens == tuple([0, *torch.tensor(seq_lens).cumsum(0).tolist()])
    assert instance.symbols["total_tokens"] == sum(seq_lens)
    assert instance.symbols["num_seqs"] == len(seq_lens)
    assert instance.symbols["NSP1"] == len(seq_lens) + 1
    assert instance.tensor("q").shape == (sum(seq_lens), instance.symbols["H"], instance.symbols["D"])
    assert instance.tensor("cu_seqlens").shape == (len(seq_lens) + 1,)
    assert instance.tensor("initial_state").shape[0] == len(seq_lens)
    assert instance.tensor("final_state").shape[0] == len(seq_lens)


@pytest.mark.parametrize("size,seq_lens", sorted(SCALE_SIZES.items()))
def test_every_scale_tier_carries_its_length_distribution(size, seq_lens):
    """@brief elementwise varlen 题同样按分段档实例化。"""
    instance = load_task_contract(SCALE_DIR, dtype="fp16", size=size).instance
    assert instance.cu_seqlens == tuple([0, *torch.tensor(seq_lens).cumsum(0).tolist()])
    assert instance.symbols["num_seqs"] == len(seq_lens)


def _hand_recurrence(inputs, instance) -> dict:
    """@brief 一份**独立**的 fp64 分段递推实现（不复用 reference.py 的任何代码）。

    @param inputs 已加宽的输入张量。
    @param instance 语义实例（用来取 H/D/LOWER_BOUND）。
    @return ``{"output": ..., "final_state": ...}``。
    """
    q = inputs["q"].to(torch.float64)
    k = inputs["k"].to(torch.float64)
    v = inputs["v"].to(torch.float64)
    g = inputs["g"].to(torch.float64)
    beta = inputs["beta"].to(torch.float64)
    a_log = inputs["A_log"].to(torch.float64)
    bias = inputs["dt_bias"].to(torch.float64).view(instance.symbols["H"], instance.symbols["D"])
    state = inputs["initial_state"].to(torch.float64).clone()
    scale = float(inputs["scale"].to(torch.float64).item())
    lower = float(instance.resolved_attrs["LOWER_BOUND"])
    cu = [int(value) for value in inputs["cu_seqlens"].tolist()]
    out = torch.zeros_like(q)

    for seq in range(len(cu) - 1):
        current = state[seq]
        for t in range(cu[seq], cu[seq + 1]):
            qt = q[t] / torch.sqrt((q[t] * q[t]).sum(-1, keepdim=True) + 1e-6) * scale
            kt = k[t] / torch.sqrt((k[t] * k[t]).sum(-1, keepdim=True) + 1e-6)
            decay = lower * torch.sigmoid(torch.exp(a_log)[:, None] * (g[t] + bias))
            current = current * torch.exp(decay)[:, None, :]
            delta = (v[t] - (current * kt[:, None, :]).sum(-1)) * torch.sigmoid(beta[t])[:, None]
            current = current + delta[:, :, None] * kt[:, None, :]
            out[t] = (current * qt[:, None, :]).sum(-1)
        state[seq] = current
    return {"output": out, "final_state": state}


@pytest.mark.parametrize("size", sorted(KDA_SIZES))
@pytest.mark.parametrize("case", CASES)
def test_the_oracle_agrees_with_an_independent_segmented_recurrence(size, case):
    """@brief oracle 与独立手算的分段递推在每档、每个 case 上都一致（含逐段终态）。

    手算实现只读实际 ``cu_seqlens``，因此它同时证明了“oracle 确实按本次分段算”，而不是按
    canonical 副本算。
    """
    contract = load_task_contract(KDA_DIR, dtype="fp16", size=size)
    instance = contract.instance
    oracle = load_reference(KDA_DIR, op_name="varlen_kda", instance=instance)
    inputs = oracle.generate(instance, case, seed=7)
    result = oracle.execute(instance, inputs, dict(instance.resolved_attrs))
    expected = _hand_recurrence(inputs, instance)

    assert torch.allclose(result["output"], expected["output"], rtol=0, atol=1e-10)
    assert torch.allclose(result["final_state"], expected["final_state"], rtol=0, atol=1e-10)
    # 每条序列的终态是它自己那段递推的结果，不是整包续传的结果
    if instance.symbols["num_seqs"] > 1 and case not in ("zero_length", "single_sequence"):
        assert not torch.allclose(result["final_state"][0], result["final_state"][-1])


@pytest.mark.parametrize("size", sorted(KDA_SIZES))
@pytest.mark.parametrize("case", CASES)
def test_gen_inputs_keeps_the_tier_shape_and_a_legal_segmentation(size, case):
    """@brief 生成的边界始终合法且保持该档 total/count；只有 canonical 等于声明分布。"""
    contract = load_task_contract(KDA_DIR, dtype="fp16", size=size)
    instance = contract.instance
    oracle = load_reference(KDA_DIR, op_name="varlen_kda", instance=instance)
    inputs = oracle.generate(instance, case, seed=11)

    assert inputs["cu_seqlens"].dtype == torch.int32
    assert inputs["cu_seqlens"].shape == (instance.symbols["num_seqs"] + 1,)
    assert int(inputs["cu_seqlens"][0]) == 0
    assert int(inputs["cu_seqlens"][-1]) == instance.symbols["total_tokens"]
    if case == "canonical":
        assert inputs["cu_seqlens"].tolist() == list(instance.cu_seqlens)
    assert inputs["q"].shape[0] == instance.symbols["total_tokens"]
    assert inputs["initial_state"].shape[0] == instance.symbols["num_seqs"]
    assert inputs["scale"].numel() == 1


def test_the_kda_baseline_package_loads_and_closes():
    """@brief 基线包可加载，且基线的常量与合同一致。"""
    loaded = load_baselines(KDA_DIR, op_name="varlen_kda")
    assert "torch_mps" in loaded.runners
    assert callable(loaded.runners["torch_mps"])
    contract = load_task_contract(KDA_DIR, dtype="fp16", size="mixed")
    assert contract.instance.resolved_attrs["LOWER_BOUND"] == -5


def test_the_kda_baseline_matches_the_oracle_on_a_tiny_distribution():
    """@brief 基线（fp32 逐 token）与 fp64 oracle 在同一容差内一致。"""
    from tasks.varlen_kda import baselines as module

    contract = load_task_contract(KDA_DIR, dtype="fp16", size="mixed")
    instance = contract.instance
    oracle = load_reference(KDA_DIR, op_name="varlen_kda", instance=instance)
    inputs = oracle.generate(instance, "canonical", seed=3)
    args = [inputs[name] for name in ("q", "k", "v", "g", "beta", "A_log", "dt_bias", "initial_state", "cu_seqlens", "scale")]
    got = module.run(*args)
    want = oracle.execute(instance, inputs, dict(instance.resolved_attrs))

    for name in ("output", "final_state"):
        error = (got[name].to(torch.float64) - want[name]).abs()
        scale = want[name].abs().clamp_min(1e-3)
        assert float((error / scale).max()) < 0.05, name


#: V2 声明的每链最大 piece 数（``handoff_states`` 与 ``work_items`` 的容量公式基于它）。
MAX_SPLIT_PER_CHAIN = 3


def _workspace_spans(instance) -> dict[str, int]:
    """@brief 由语义实例重算 V2 声明的 workspace 容量（字节），作为 OpSpec 公式的独立副本。

    @param instance varlen KDA 语义实例。
    @return ``{张量名: 字节数}``。
    """
    symbols = instance.symbols
    chains = symbols["num_seqs"] * symbols["H"]
    total_tokens, heads, dim = symbols["total_tokens"], symbols["H"], symbols["D"]
    return {
        "log_decay_tab": total_tokens * heads * dim * 4,
        "token_stats": total_tokens * heads * 3 * 4,
        "work_items": chains * MAX_SPLIT_PER_CHAIN * 2 * 4,
        "item_counts": chains * MAX_SPLIT_PER_CHAIN * 4,
        "handoff_flags": chains * 4 * 4,
        "handoff_states": chains * (MAX_SPLIT_PER_CHAIN - 1) * dim * dim * 4,
    }


def test_the_kda_workspace_is_a_scratch_buffer_outside_the_reference_contract():
    """@brief workspace 是 scratch，不进 reference 输入/结果；容量公式在 OpSpec 里写明。"""
    contract = load_task_contract(KDA_DIR, dtype="fp16", size="mixed")
    instance = contract.instance
    workspace = [tensor.name for tensor in instance.tensors if tensor.space == "workspace"]
    assert workspace == ["log_decay_tab", "token_stats", "work_items", "item_counts", "handoff_flags", "handoff_states"]
    assert set(workspace).isdisjoint(set(instance.reference_inputs) | set(instance.reference_outputs))

    abi = derive_tensor_memory_contract(instance, target=load_target("apple-m4-max"))
    span = {
        tensor.name: math.prod(int(v) for v in tensor.shape) * get_dtype(tensor.dtype).nbytes
        for tensor in abi.tensors
        if tensor.space == "workspace"
    }
    assert span == _workspace_spans(instance)
    assert all(nbytes % 4 == 0 for nbytes in span.values())
    # V2 的存储上界：handoff_bytes = slots*D*D*4，slots = chains*(MAX_SPLIT_PER_CHAIN-1)。
    assert span["handoff_states"] == instance.symbols["num_seqs"] * instance.symbols["H"] * 2 * instance.symbols["D"] ** 2 * 4


@pytest.mark.parametrize("size", sorted(KDA_SIZES))
def test_every_tier_declares_the_V2_workspace_capacity_formulas(size):
    """@brief 每档的 workspace 容量都能由声明的公式重算（不是某个小档的幸运数字）。"""
    contract = load_task_contract(KDA_DIR, dtype="fp16", size=size)
    instance = contract.instance
    abi = derive_tensor_memory_contract(instance, target=load_target("apple-m4-max"))
    span = {
        tensor.name: math.prod(int(v) for v in tensor.shape) * get_dtype(tensor.dtype).nbytes
        for tensor in abi.tensors
        if tensor.space == "workspace"
    }
    assert span == _workspace_spans(instance)
    # num_ctas = chains*MAX_SPLIT_PER_CHAIN；flags 的公式是 >= num_ctas+1。
    num_ctas = instance.symbols["num_seqs"] * instance.symbols["H"] * MAX_SPLIT_PER_CHAIN
    assert span["handoff_flags"] // 4 >= num_ctas + 1
    assert instance.tensor("work_items").shape == (num_ctas, 2)
    assert instance.tensor("item_counts").shape == (num_ctas,)
    assert instance.tensor("handoff_states").shape == (instance.symbols["num_seqs"] * instance.symbols["H"] * (MAX_SPLIT_PER_CHAIN - 1), instance.symbols["D"], instance.symbols["D"])

    docs = {tensor.name: tensor.doc for tensor in instance.tensors if tensor.space == "workspace"}
    assert "MAX_SPLIT_PER_CHAIN" in docs["work_items"] and "num_ctas" in docs["work_items"]
    assert "MAX_SPLIT_PER_CHAIN" in docs["item_counts"]
    assert "num_ctas+1" in docs["handoff_flags"]
    assert "handoff_bytes = slots*D*D*4" in docs["handoff_states"]
    assert "only when a stage" in docs["log_decay_tab"]


def test_a_workspace_free_scale_task_has_no_workspace_span():
    """@brief 没有 workspace 的 varlen 题 ABI 里就没有 workspace 张量。"""
    instance = load_task_contract(SCALE_DIR, dtype="fp16", size="mixed").instance
    abi = derive_tensor_memory_contract(instance, target=load_target("apple-m4-max"))
    assert [tensor.name for tensor in abi.tensors if tensor.space == "workspace"] == []


def test_the_statement_explains_the_packed_batch():
    """@brief varlen 题面把分段语义说清楚（候选不需要从 oracle 源码反推）。"""
    from tcenv.task_presentation import render_statement

    instance = load_task_contract(KDA_DIR, dtype="fp16", size="mixed").instance
    statement = render_statement(instance, None)

    assert "## Varlen (packed sequence)" in statement
    assert "`[cu_seqlens[s], cu_seqlens[s+1])`" in statement
    assert "runtime data, not compiler constants" in statement
    assert "must never be carried across a boundary" in statement
    assert "must never be assumed to start at zero" in statement
    assert "3 complete sequences" in statement and "total_tokens` (14)" in statement


def test_the_statement_publishes_the_canonical_profile_without_leaking_a_seed():
    """@brief 题面给出 canonical 分布供调度，并明确其他合法分段也会被测；不带任何种子。"""
    from tcenv.task_presentation import render_statement

    instance = load_task_contract(KDA_DIR, dtype="fp16", size="mixed").instance
    statement = render_statement(instance, None)
    section = statement[statement.index("## Varlen") : statement.index("## Cases")]

    assert "Canonical length profile" in section
    assert "seq_lens = [4, 8, 2]" in section
    assert "not the only one" in section
    assert "`cu_seqlens[s] == cu_seqlens[s+1]`" in section
    # master/tune/unseen seed 绝不出现在题面里（候选只能看到公开的 canonical 分布）。
    assert "seed" not in section.lower()
    assert "master" not in section.lower()
    # 题面只给人看的 case 名单，不泄露 fresh 的实际分布。
    assert "[6, 1, 7]" not in statement


def test_an_elementwise_varlen_statement_has_no_state_contract():
    """@brief ``varlen_scale`` 没有 initial/final_state，题面不得打印不存在的状态合同。"""
    from tcenv.task_presentation import render_statement

    instance = load_task_contract(SCALE_DIR, dtype="fp16", size="mixed").instance
    statement = render_statement(instance, None)
    section = statement[statement.index("## Varlen") : statement.index("## Cases")]

    assert "Canonical length profile" in section
    assert "initial_state" not in section
    assert "final_state" not in section
    assert "Per-sequence work" not in section


def test_a_single_sequence_tier_is_the_degenerate_case():
    """@brief 单段档的 canonical profile 就是全长退化情形（题面也说清楚）。"""
    from tcenv.task_presentation import render_statement

    instance = load_task_contract(SCALE_DIR, dtype="fp16", size="single").instance
    section = render_statement(instance, None)
    assert "seq_lens = [10] over 1 sequences and 10 tokens" in section
    assert "single sequence" in section


def test_a_fixed_shape_statement_has_no_varlen_section():
    """@brief 固定 shape 题面不出现 varlen 节（题面逐字不变）。"""
    from tcenv.task_presentation import render_statement

    contract = load_task_contract(REPO_ROOT / "tasks" / "gemm", dtype="fp16", size="paper")
    assert "## Varlen" not in render_statement(contract.instance, None)


def test_the_cli_size_default_follows_the_task_not_a_hardcoded_name():
    """@brief ``eval-v2`` 未给 ``--size`` 时按题声明的档解析，档名不写死。"""
    from tcenv.cli import _size_for

    # varlen 题没有 paper 档：用题自己声明的默认档
    assert _size_for("varlen_kda", None) == "mixed"
    assert _size_for("varlen_scale", None) == "mixed"
    # 有 paper 档的题保持历史行为
    assert _size_for("gemm", None) == "paper"
    # 显式给的永远照用
    assert _size_for("varlen_kda", "uniform") == "uniform"
