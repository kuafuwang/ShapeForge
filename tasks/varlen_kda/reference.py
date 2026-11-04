"""varlen_kda: the independent oracle of the packed-varlen Kimi Delta Attention forward.

Chunked, per-channel-gated delta rule over each sequence of the packed batch, matching the exact
token-by-token recurrence (KDA-for-KDA's ``recurrent_kernel.py``):

* q/k L2 normalization (``rsqrt(sum(x*x) + 1e-6)``, q also scaled by ``scale``);
* per-channel log-decay ``lower_bound * sigmoid(exp(A_log) * (g + dt_bias))``;
* ``state *= exp(log_decay)``;
* delta residual ``(v - state @ k) * sigmoid(beta)``, then ``state += residual ⊗ k``;
* ``output = state @ q``.

The recurrence runs independently inside every ``[cu_seqlens[s], cu_seqlens[s+1])`` segment, with the
fp32 V-first recurrent state carried per sequence and written to ``final_state[s]`` at the segment end.
The oracle stays in fp64 and never reads a symbol or a workspace; segment boundaries come from the
``cu_seqlens`` input, and ``scale``/``lower_bound`` from ``attrs``.
"""

import torch


def reference(inputs, attrs):
    """@brief fp64 oracle of the packed-varlen KDA forward.

    @param inputs 合同输入（浮点张量已按声明 dtype 舍入并加宽成 float32；``cu_seqlens`` 是 int32 偏移）。
    @param attrs 已解析 attr（``scale``、``lower_bound``）。
    @return ``{"output": ..., "final_state": ...}``（fp64）。
    """
    q = inputs["q"].to(torch.float64)
    k = inputs["k"].to(torch.float64)
    v = inputs["v"].to(torch.float64)
    g = inputs["g"].to(torch.float64)
    beta = inputs["beta"].to(torch.float64)
    a_log = inputs["A_log"].to(torch.float64)
    dt_bias = inputs["dt_bias"].to(torch.float64)
    state = inputs["initial_state"].to(torch.float64).clone()
    cu = [int(x) for x in inputs["cu_seqlens"].tolist()]

    total_tokens, heads, dim = q.shape
    scale = float(inputs["scale"].to(torch.float64).item())
    lower_bound = float(attrs["LOWER_BOUND"])
    decay_scale = torch.exp(a_log)  # [H]
    bias = dt_bias.view(heads, dim)  # [H, D]
    output = torch.zeros((total_tokens, heads, dim), dtype=torch.float64)

    for seq in range(len(cu) - 1):
        begin, end = cu[seq], cu[seq + 1]
        for t in range(begin, end):
            qt = q[t]  # [H, D]
            kt = k[t]
            qt = qt * torch.rsqrt((qt * qt).sum(-1, keepdim=True) + 1e-6) * scale
            kt = kt * torch.rsqrt((kt * kt).sum(-1, keepdim=True) + 1e-6)

            gate = g[t] + bias  # [H, D]
            log_decay = lower_bound * torch.sigmoid(decay_scale[:, None] * gate)  # [H, D]
            state[seq] = state[seq] * torch.exp(log_decay)[:, None, :]

            residual = v[t] - (state[seq] * kt[:, None, :]).sum(-1)  # [H, D] (sum over key dim)
            residual = residual * torch.sigmoid(beta[t])[:, None]
            state[seq] = state[seq] + residual[:, :, None] * kt[:, None, :]

            output[t] = (state[seq] * qt[:, None, :]).sum(-1)

    return {"output": output, "final_state": state}


def gen_inputs(case, seed, instance):
    """@brief 生成打包 varlen KDA 输入（含 cu_seqlens 与 per-sequence initial_state）。

    @param case 算例名（未用）。
    @param seed 随机种子。
    @param instance 语义实例（含 cu_seqlens 与 total_tokens/H/D symbols）。
    @return 输入字典。
    """
    generator = torch.Generator().manual_seed(int(seed))
    total_tokens = int(instance.symbols["total_tokens"])
    heads = int(instance.symbols["H"])
    dim = int(instance.symbols["D"])
    num_seqs = int(instance.symbols["num_seqs"])

    def randn(shape, scale=0.5, dtype=torch.float32):
        return (torch.randn(shape, generator=generator) * scale).to(dtype)

    a_log = torch.log(torch.empty(heads).uniform_(1.0, 16.0, generator=generator))
    dt = torch.exp(
        torch.rand(heads * dim, generator=generator) * (torch.log(torch.tensor(0.1)) - torch.log(torch.tensor(0.001)))
        + torch.log(torch.tensor(0.001))
    ).clamp_(min=1e-4)
    dt_bias = dt + torch.log(-torch.expm1(-dt))
    q, k, v, g = (randn((total_tokens, heads, dim)) for _ in range(4))
    beta = randn((total_tokens, heads))
    initial_state = randn((num_seqs, heads, dim, dim), scale=0.25)
    cu = torch.tensor(list(instance.cu_seqlens), dtype=torch.int32)
    scale = torch.tensor([1.0 / (dim ** 0.5)], dtype=torch.float32)
    return {
        "q": q,
        "k": k,
        "v": v,
        "g": g,
        "beta": beta,
        "A_log": a_log,
        "dt_bias": dt_bias,
        "initial_state": initial_state,
        "cu_seqlens": cu,
        "scale": scale,
    }
