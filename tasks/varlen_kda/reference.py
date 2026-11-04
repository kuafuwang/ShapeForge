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

L1 correctness cases
--------------------
``sizes.<tier>.seq_lens`` is only the *canonical* profile published for scheduling and benchmarking.
Correctness keeps the tier's ``total_tokens``/``num_seqs`` and reshapes the segmentation:

``canonical``
    the declared profile (also the benchmark/validate workload);
``permuted``
    the same multiset in a different order; when the canonical profile is uniform no permutation
    differs, so the case falls back to a different legal split of the same total/count;
``skewed``
    one long sequence plus ``num_seqs - 1`` two-token fragments;
``all_unaligned_primes``
    every segment prime, so several partial tails sit next to each other (pointer offsets,
    log-decay resets and masks are exercised without an aligned boundary);
``max_handoff_stress``
    one segment of ``total_tokens - (num_seqs - 1)`` and the rest one token each: the longest
    possible recurrent chain next to the smallest fragments;
``chunk_boundary``
    parts planned around a declared chunk size (``BT-1``, ``2BT-1``, ``BT``, ``BT+1``, ``1``,
    ``2BT+1``), with the last part absorbing the tier's remainder;
``zero_length``
    exactly one empty segment (its tokens move to the last sequence);
``single_sequence``
    the degenerate whole-batch sequence — only a tier with ``num_seqs == 1`` can construct it, so
    on a multi-sequence tier it is the canonical profile (the count is never changed to fake it);
``fresh_distribution``
    a seed-derived legal split, never equal to the published canonical profile.

A tier with ``num_seqs == 1`` has exactly one legal segmentation, so every case degenerates to it.
The static profile never contains a zero-length segment (``seq_lens`` requires >= 1); emptiness is a
runtime fact and the harness validates it (``tcenv.taskcontract.varlen``).

The case planner is deliberately local to this task instead of shared with
``tasks/varlen_scale/reference.py``: a reference module may not import this repository (or a relative
sibling), and a repo-level helper would not be part of the task pack.  The shared *contract* is
pinned by ``tests/shared/test_varlen_cases.py``, which runs both tasks' planners through the same
assertions.
"""

import torch

#: Chunk size the ``chunk_boundary`` case plans around.  A candidate owns its real chunk/BT
#: choice; the case only approaches these sizes so partial tails are exercised.
CHUNK = 4
#: ``chunk_boundary`` targets, cycled over the segments: ``BT-1``, ``2BT-1``, ``BT``, ``BT+1``,
#: ``1``, ``2BT+1``.
CHUNK_TARGETS = (CHUNK - 1, 2 * CHUNK - 1, CHUNK, CHUNK + 1, 1, 2 * CHUNK + 1)


def _is_prime(value):
    """@brief 素数判定（只用于 case 生成，范围内足够快）。

    @param value 整数。
    @return 是否素数。
    """
    if value < 2:
        return False
    divisor = 2
    while divisor * divisor <= value:
        if value % divisor == 0:
            return False
        divisor += 1
    return True


def _prime_partition(total_tokens, num_seqs):
    """@brief 找出 ``num_seqs`` 个和为 ``total_tokens`` 的素数（非递增）。

    @param total_tokens 总 token 数。
    @param num_seqs 段数。
    @return 长度元组；该 total/count 不存在全素数分段时为 ``None``。
    """
    largest = total_tokens - 2 * (num_seqs - 1)
    primes = [value for value in range(2, largest + 1) if _is_prime(value)]

    def walk(remaining, slots, ceiling, chosen):
        if slots == 1:
            return chosen + (remaining,) if remaining <= ceiling and _is_prime(remaining) else None
        for prime in reversed(primes):
            if prime > ceiling or prime > remaining - 2 * (slots - 1):
                continue
            found = walk(remaining - prime, slots - 1, prime, chosen + (prime,))
            if found is not None:
                return found
        return None

    return walk(total_tokens, num_seqs, largest, ())


def _seeded_partition(total_tokens, num_seqs, seed):
    """@brief 由 seed 派生的正长度分段（每段 >= 1，和恒为 ``total_tokens``）。

    @param total_tokens 总 token 数。
    @param num_seqs 段数。
    @param seed 输入种子。
    @return 长度元组。
    """
    if num_seqs == 1:
        return (total_tokens,)
    generator = torch.Generator().manual_seed(int(seed))
    cuts = torch.randperm(total_tokens - 1, generator=generator)[: num_seqs - 1].add_(1).tolist()
    bounds = [0, *sorted(int(cut) for cut in cuts), total_tokens]
    return tuple(bounds[index + 1] - bounds[index] for index in range(num_seqs))


def _segment_lengths(case, seed, total_tokens, num_seqs, canonical):
    """@brief 一个 L1 correctness case 的实际分段长度（总长与段数恒不变）。

    @param case 用例名。
    @param seed 输入种子（只有 ``fresh_distribution``/``zero_length`` 依赖它）。
    @param total_tokens 该档的总 token 数。
    @param num_seqs 该档的序列条数。
    @param canonical 该档声明的 canonical 分段长度。
    @return 每段 token 数元组（``zero_length`` 里会出现 0）。
    @raises ValueError 该档不满足 L1 分段的静态前提，或 case 未知。
    """
    if len(canonical) != num_seqs or sum(canonical) != total_tokens or min(canonical) < 1:
        raise ValueError(f"the tier's canonical profile {canonical} is not a legal varlen segmentation")
    if num_seqs == 1:
        # 单段档只有一种合法分段：single_sequence 就是该档本身，其余 case 无法在不改 count 的前提下构造。
        return (total_tokens,)
    if case == "canonical":
        return tuple(canonical)
    if case == "permuted":
        if len(set(canonical)) > 1:
            return tuple(list(canonical[1:]) + [canonical[0]])
        base = canonical[0]
        if base >= 2:
            # 均匀档没有任何不同顺序：换一个同 total/count 的合法切分，而不是假装换了顺序。
            return (base + 1, base - 1, *canonical[2:])
        return tuple(canonical)
    if case == "skewed":
        short = 2 if total_tokens - 2 * (num_seqs - 1) >= 2 else 1
        return (total_tokens - short * (num_seqs - 1), *([short] * (num_seqs - 1)))
    if case == "all_unaligned_primes":
        parts = _prime_partition(total_tokens, num_seqs)
        if parts is None:
            raise ValueError(
                f"total_tokens={total_tokens} / num_seqs={num_seqs} has no all-prime segmentation; the "
                "all_unaligned_primes case must not be replaced by a non-prime layout"
            )
        return parts
    if case == "max_handoff_stress":
        return (total_tokens - (num_seqs - 1), *([1] * (num_seqs - 1)))
    if case == "chunk_boundary":
        parts = []
        remaining, slots = total_tokens, num_seqs
        for index in range(num_seqs):
            if slots == 1:
                parts.append(remaining)
                break
            target = CHUNK_TARGETS[index % len(CHUNK_TARGETS)]
            value = max(1, min(int(target), remaining - (slots - 1)))
            parts.append(value)
            remaining -= value
            slots -= 1
        return tuple(parts)
    if case == "zero_length":
        base = list(_seeded_partition(total_tokens, num_seqs, seed))
        empty = int(seed) % num_seqs
        receiver = num_seqs - 1 if empty != num_seqs - 1 else 0
        base[receiver] += base[empty]
        base[empty] = 0
        return tuple(base)
    if case == "single_sequence":
        # 多段档不能改 count 去凑这个 case；诚实做法是显式退化为该档唯一的公开分布。
        return tuple(canonical)
    if case == "fresh_distribution":
        parts = list(_seeded_partition(total_tokens, num_seqs, seed))
        if tuple(parts) == tuple(canonical):
            # 抽到 canonical 时确定性地挪一个 token（donor 与 receiver 必不相同）。
            donor = max(range(num_seqs), key=lambda index: parts[index])
            if parts[donor] > 1:
                receiver = min((index for index in range(num_seqs) if index != donor), key=lambda index: parts[index])
                parts[donor] -= 1
                parts[receiver] += 1
        return tuple(parts)
    raise ValueError(f"unknown varlen correctness case {case!r}")


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
    """@brief 生成打包 varlen KDA 输入（含实际 ``cu_seqlens`` 与 per-sequence initial_state）。

    ``total_tokens``/``num_seqs`` 与档位一致；只有分段随 case 变化。``initial_state`` 逐序列不同且
    非零（每条序列的对角加一个不同偏置），这样串段、丢初值或从零状态续传都不能靠运气通过。

    @param case L1 correctness case（与 OpSpec ``cases.names`` 一致）。
    @param seed 随机种子。
    @param instance 语义实例（含 canonical ``cu_seqlens`` 与 total_tokens/H/D symbols）。
    @return 输入字典。
    @raises ValueError case 未知或该 total/count 无法构造该 case。
    """
    generator = torch.Generator().manual_seed(int(seed))
    total_tokens = int(instance.symbols["total_tokens"])
    heads = int(instance.symbols["H"])
    dim = int(instance.symbols["D"])
    num_seqs = int(instance.symbols["num_seqs"])
    canonical = tuple(int(value) for value in instance.canonical_seq_lens)
    lengths = _segment_lengths(case, seed, total_tokens, num_seqs, canonical)
    if sum(lengths) != total_tokens or len(lengths) != num_seqs or total_tokens < 1:
        raise ValueError(f"case {case!r} produced {lengths}, which does not keep total_tokens/num_seqs")
    offsets = [0]
    for length in lengths:
        offsets.append(offsets[-1] + length)

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
    for seq in range(num_seqs):
        initial_state[seq].diagonal(dim1=-2, dim2=-1).add_(0.5 + 0.25 * seq)
    cu = torch.tensor(offsets, dtype=torch.int32)
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
