"""varlen_kda baselines: one runner, the framework-native token-by-token delta rule.

A baseline is a timing comparison object, never the correctness oracle.  ``BaselineRunner`` passes
exactly ``task.contract.inputs`` as positional arguments, so ``run`` takes the full KDA ABI.  The
recurrence is the same math as the oracle (token-by-token delta rule, fp32 V-first recurrent state
per sequence), evaluated in fp32 and rounded once on output (``numerics.rounding =
single_round_at_output``).
"""

from __future__ import annotations

import torch


def run(q, k, v, g, beta, A_log, dt_bias, initial_state, cu_seqlens, scale):
    """@brief Framework-native token-by-token KDA forward over each packed segment.

    @return ``{"output": ..., "final_state": ...}``（output 按声明 dtype 舍入，final_state 为 fp32）。
    """
    q = q.to(torch.float32)
    k = k.to(torch.float32)
    v = v.to(torch.float32)
    g = g.to(torch.float32)
    beta = beta.to(torch.float32)
    a_log = A_log.to(torch.float32)
    bias = dt_bias.to(torch.float32).view(q.shape[1], q.shape[2])
    state = initial_state.to(torch.float32).clone()
    cu = [int(x) for x in cu_seqlens.tolist()]
    scale_value = float(scale.to(torch.float32).item())
    decay_scale = torch.exp(a_log)
    output = torch.zeros_like(v)

    for seq in range(len(cu) - 1):
        for t in range(cu[seq], cu[seq + 1]):
            qt = q[t]
            kt = k[t]
            qt = qt * torch.rsqrt((qt * qt).sum(-1, keepdim=True) + 1e-6) * scale_value
            kt = kt * torch.rsqrt((kt * kt).sum(-1, keepdim=True) + 1e-6)
            log_decay = -5.0 * torch.sigmoid(decay_scale[:, None] * (g[t] + bias))
            state[seq] = state[seq] * torch.exp(log_decay)[:, None, :]
            residual = v[t] - (state[seq] * kt[:, None, :]).sum(-1)
            residual = residual * torch.sigmoid(beta[t])[:, None]
            state[seq] = state[seq] + residual[:, :, None] * kt[:, None, :]
            output[t] = (state[seq] * qt[:, None, :]).sum(-1)

    return {"output": output.to(v.dtype), "final_state": state}


BASELINES = {"torch_mps": run}
