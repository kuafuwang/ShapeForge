"""Framework-native fixed-length KDA timing baseline; never the correctness oracle."""

import torch


def run(q, k, v, g, beta, A_log, dt_bias, initial_state, scale):
    q = q.to(torch.float32)
    k = k.to(torch.float32)
    v = v.to(torch.float32)
    g = g.to(torch.float32)
    beta = beta.to(torch.float32)
    decay_scale = torch.exp(A_log.to(torch.float32))
    bias = dt_bias.to(torch.float32).view(q.shape[1], q.shape[2])
    state = initial_state.to(torch.float32).clone()
    scale_value = float(scale.to(torch.float32).item())
    output = torch.zeros_like(v)

    for t in range(q.shape[0]):
        qt = q[t]
        kt = k[t]
        qt = qt * torch.rsqrt((qt * qt).sum(-1, keepdim=True) + 1e-6) * scale_value
        kt = kt * torch.rsqrt((kt * kt).sum(-1, keepdim=True) + 1e-6)
        log_decay = -5.0 * torch.sigmoid(decay_scale[:, None] * (g[t] + bias))
        state = state * torch.exp(log_decay)[:, None, :]
        residual = v[t] - (state * kt[:, None, :]).sum(-1)
        residual = residual * torch.sigmoid(beta[t])[:, None]
        state = state + residual[:, :, None] * kt[:, None, :]
        output[t] = (state * qt[:, None, :]).sum(-1)

    return {"output": output.to(v.dtype), "final_state": state}


BASELINES = {"torch_mps": run}
