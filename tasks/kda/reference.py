"""Fixed-length KDA: independent fp64 oracle and deterministic input generation.

There is deliberately no ``cu_seqlens``: this is the ordinary fixed-shape task, not another varlen
alias.  The single sequence's fp32 V-first state is decayed before each residual and is returned as
``final_state[H,D,D]``.  The oracle reads only inputs/attrs and never reads a workspace or candidate.
"""

import torch


def reference(inputs, attrs):
    q = inputs["q"].to(torch.float64)
    k = inputs["k"].to(torch.float64)
    v = inputs["v"].to(torch.float64)
    g = inputs["g"].to(torch.float64)
    beta = inputs["beta"].to(torch.float64)
    decay_scale = torch.exp(inputs["A_log"].to(torch.float64))
    bias = inputs["dt_bias"].to(torch.float64).view(q.shape[1], q.shape[2])
    state = inputs["initial_state"].to(torch.float64).clone()
    scale = float(inputs["scale"].to(torch.float64).item())
    lower_bound = float(attrs["LOWER_BOUND"])
    output = torch.zeros_like(v)

    for t in range(q.shape[0]):
        qt = q[t]
        kt = k[t]
        qt = qt * torch.rsqrt((qt * qt).sum(-1, keepdim=True) + 1e-6) * scale
        kt = kt * torch.rsqrt((kt * kt).sum(-1, keepdim=True) + 1e-6)
        log_decay = lower_bound * torch.sigmoid(decay_scale[:, None] * (g[t] + bias))
        state = state * torch.exp(log_decay)[:, None, :]
        residual = v[t] - (state * kt[:, None, :]).sum(-1)
        residual = residual * torch.sigmoid(beta[t])[:, None]
        state = state + residual[:, :, None] * kt[:, None, :]
        output[t] = (state * qt[:, None, :]).sum(-1)

    return {"output": output, "final_state": state}


def _sample(case, shape, generator):
    if case == "normal":
        return torch.randn(shape, generator=generator) * 0.5
    if case == "wide_range":
        exponent = torch.rand(shape, generator=generator) * 4.0 - 2.0
        sign = torch.where(torch.rand(shape, generator=generator) < 0.5, -1.0, 1.0)
        return sign * torch.pow(10.0, exponent) * 0.25
    if case == "outlier":
        values = torch.randn(shape, generator=generator) * 0.5
        mask = torch.rand(shape, generator=generator) < 0.01
        return torch.where(mask, values * 5.0, values)
    raise ValueError(f"unknown fixed KDA case {case!r}")


def gen_inputs(case, seed, instance):
    generator = torch.Generator().manual_seed(int(seed))
    tokens = int(instance.symbols["N"])
    heads = int(instance.symbols["H"])
    dim = int(instance.symbols["D"])

    a_log = torch.log(torch.empty(heads).uniform_(1.0, 16.0, generator=generator))
    dt = torch.exp(
        torch.rand(heads * dim, generator=generator)
        * (torch.log(torch.tensor(0.1)) - torch.log(torch.tensor(0.001)))
        + torch.log(torch.tensor(0.001))
    ).clamp_(min=1e-4)
    dt_bias = dt + torch.log(-torch.expm1(-dt))
    q, k, v, g = (_sample(case, (tokens, heads, dim), generator) for _ in range(4))
    beta = _sample(case, (tokens, heads), generator)
    initial_state = torch.randn((heads, dim, dim), generator=generator) * 0.25
    initial_state.diagonal(dim1=-2, dim2=-1).add_(0.5)
    return {
        "q": q,
        "k": k,
        "v": v,
        "g": g,
        "beta": beta,
        "A_log": a_log,
        "dt_bias": dt_bias,
        "initial_state": initial_state,
        "scale": torch.tensor([1.0 / (dim ** 0.5)], dtype=torch.float32),
    }
