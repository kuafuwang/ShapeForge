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
    log-decay resets and masks are exercised without an aligned boundary); a tier with no
    all-prime segmentation (``total_tokens`` too small for ``num_seqs`` primes) falls back to
    its canonical profile and the harness records ``not_constructible`` — never a non-prime
    layout labelled exact;
``max_handoff_stress``
    one segment of ``total_tokens - (num_seqs - 1)`` and the rest one token each: the longest
    possible recurrent chain next to the smallest fragments;
``chunk_boundary``
    parts planned around a declared chunk size (``BT-1``, ``2BT-1``, ``BT``, ``BT+1``, ``1``,
    ``2BT+1``), with the last part absorbing the tier's remainder;
``zero_length``
    exactly one empty segment (its tokens move to the last sequence);
``single_sequence``
    one non-empty sequence carrying the whole batch and empty remaining sequences (the count is
    never changed to fake it; a tier with ``num_seqs == 1`` is the case itself);
``fresh_distribution``
    a seed-derived legal split, never equal to the published canonical profile.

A tier with ``num_seqs == 1`` has exactly one legal segmentation, so every case degenerates to it.
The static profile never contains a zero-length segment (``seq_lens`` requires >= 1); emptiness is a
runtime fact and the harness validates it (``tcenv.taskcontract.varlen``).

L2 capacity instance (``sizes.capacity``)
-----------------------------------------
This task also declares one **capacity** tier: ``total_tokens=16``/``num_seqs=4`` buffers with a
*runtime* resolved shape, so one compiled ABI serves several batches in a row.  The tier declares four
workload profiles (``a_full``/``b_tiny``/``c_mid``/``d_shrink``) and the harness assigns them cyclically
to the declared cases; the resolved total/count therefore shrinks and grows across one sweep
(16 -> 1 -> 9 -> 3 -> 16 -> ...) instead of replaying one static shape.

For an L2 instance the buffers are capacity-sized and ``cu_seqlens`` is capacity-sized too:

* ``cu_seqlens[0:num_seqs + 1]`` is this call's legal offset prefix (the *same* case planner below is
  applied to the profile's total/count, so every L1 case name keeps its meaning);
* ``cu_seqlens[num_seqs + 1:]`` is the declared sentinel (:data:`CU_SEQLENS_SENTINEL`, not ``0``);
* every element of an input/output tensor outside the logical region is the declared padding poison
  (:data:`PADDING_POISON_BYTE`), so a candidate that reads past the resolved total or writes past it is
  caught by the harness rather than hidden by a prefix comparison.

``gen_inputs`` therefore takes an extra ``resolved=`` keyword for this tier
(:class:`ResolvedShape`); the harness supplies the expectation from the declared profile, so the task
cannot certify its own batch.  The oracle itself reads the *actual* prefix out of ``cu_seqlens``, never
the instance.

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
#: L2 capacity tiers pad ``cu_seqlens[num_seqs + 1:]`` with this sentinel.  It is negative on purpose:
#: a device or planner that walks past the batch reads something that cannot be a legal offset instead
#: of a convenient ``0``.  Must equal the value declared in ``opspec.json`` (the harness reads the
#: declaration; this constant is what the generator writes).
CU_SEQLENS_SENTINEL = -1
#: L2 padding marker: every element outside the logical region of a capacity-sized tensor holds this
#: byte pattern.  Byte-level (not value-level) so fp16/bf16/fp32 padding is checkable without a
#: dtype-specific NaN convention.  Must equal ``sizes.capacity.capacity.padding_poison_byte``.
PADDING_POISON_BYTE = 90


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
    @raises ValueError 该档的 canonical 分布不是合法分段，或 case 未知。
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
        # 只有长段严格长于 2-token 碎片才是真的偏斜：total == 2*num_seqs 时长段也恰好是 2，
        # 那种“偏斜”没有长段可言，退回 1-token 碎片（长段仍严格更长）。
        short = 2 if total_tokens - 2 * (num_seqs - 1) > 2 else 1
        return (total_tokens - short * (num_seqs - 1), *([short] * (num_seqs - 1)))
    if case == "all_unaligned_primes":
        parts = _prime_partition(total_tokens, num_seqs)
        if parts is None:
            # 该 total/count 不存在全素数分段（例如 total=3/count=2）：不能裸中止整次 correctness，
            # 也不能用非质数布局冒充这个 case。退回该档的公开分布——harness 的 realization 会独立
            # 搜索一遍并记 not_constructible，planner 的自述不算数。
            return tuple(canonical)
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
        # 多段档不改 total/count 也能构造这个 case：一条非空序列承载整批，其余序列为空
        # （空序列是合法运行时事实）。这样它是真正的“单序列退化”，不是 canonical 重放。
        return (total_tokens, *([0] * (num_seqs - 1)))
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


def _logical_offsets(cu_seqlens):
    """@brief 从（可能是 capacity 尺寸的）``cu_seqlens`` 取出本次真实的偏移前缀。

    The capacity tier allocates ``cu_seqlens`` at ``capacity.num_seqs + 1`` entries and fills the unused
    tail with :data:`CU_SEQLENS_SENTINEL`.  The oracle segments by the *actual* prefix, so it stops at the
    first sentinel instead of treating the padded tail as further (garbage) sequences.  An L1 tensor has
    no sentinel and is returned unchanged.

    @param cu_seqlens ``cu_seqlens`` 张量。
    @return 偏移列表（首项 0，末项为本次 total_tokens）。
    """
    values = [int(x) for x in cu_seqlens.tolist()]
    for index, value in enumerate(values):
        if value == CU_SEQLENS_SENTINEL:
            return values[:index]
    return values


def _poison_value(dtype):
    """@brief 声明 padding 字节模式在该 dtype 下的**可表示值**。

    The padding contract is a byte pattern (``PADDING_POISON_BYTE`` repeated over one element), not a
    float.  The generator works in the widened logical dtype the harness rounds afterwards, so it must
    write the value whose *storage* encoding is that pattern: for fp16 the two bytes 0x5A5A are
    ``203.25``, which is exactly representable, and for fp32 the four bytes are one normal float.  Writing
    a raw float32 bit pattern into a tensor that will be stored as fp16 would be re-rounded into a
    different pattern, which the harness would (correctly) report as broken padding.

    @param dtype 元素存储 dtype 短名。
    @return 该 dtype 下编码为声明 poison 模式的浮点值。
    @raises ValueError 不支持的 dtype。
    """
    if dtype in ("fp16", "bf16"):
        bits = torch.tensor([int.from_bytes(bytes([PADDING_POISON_BYTE]) * 2, "little")], dtype=torch.int16)
        view = bits.view(torch.float16) if dtype == "fp16" else bits.view(torch.bfloat16)
        return float(view.to(torch.float32).item())
    if dtype == "fp32":
        bits = torch.tensor([int.from_bytes(bytes([PADDING_POISON_BYTE]) * 4, "little")], dtype=torch.int32)
        return float(bits.view(torch.float32).item())
    raise ValueError(f"capacity padding poison is not defined for dtype {dtype!r}")


def _poison_padding(tensor, resolved_terms, total_tokens, num_seqs, value):
    """@brief 把 L2 张量 logical 区域之外的元素逐字节写成声明的 padding poison。

    The generator owns the padding (the harness verifies it independently on both sides of the run), so
    every capacity-sized tensor this module returns carries the declared marker outside the resolved
    total/count.  Axes are named, never guessed: ``total_tokens``/``num_seqs`` shrink, everything else
    keeps its capacity length.

    @param tensor 目标张量（原地写）。
    @param resolved_terms 每个 dim 的语义名（``"total_tokens"``/``"num_seqs"``/``None``）。
    @param total_tokens resolved 总 token 数。
    @param num_seqs resolved 序列条数。
    @param value 该 dtype 下编码为声明 poison 模式的浮点值（见 :func:`_poison_value`）。
    @return 同一个张量。
    """
    resolved = {"total_tokens": int(total_tokens), "num_seqs": int(num_seqs)}
    shape = tuple(int(dim) for dim in tensor.shape)
    logical = []
    axes = []
    for axis, dim in enumerate(shape):
        term = resolved_terms[axis] if axis < len(resolved_terms) else None
        if term in resolved:
            logical.append(slice(0, resolved[term]))
            axes.append((axis, resolved[term], dim))
        else:
            logical.append(slice(0, dim))
    for axis, live, bound in axes:
        if live >= bound:
            continue
        slab = []
        for other in range(len(shape)):
            if other < axis:
                slab.append(logical[other])
            elif other == axis:
                slab.append(slice(live, bound))
            else:
                slab.append(slice(0, shape[other]))
        tensor[tuple(slab)] = value
    return tensor


#: dim 词的语义名。``None`` 表示该轴不随 resolved 形状收缩。
_TOKEN_TERMS = ("total_tokens", None, None)
_SEQ_TERMS = ("num_seqs", None, None, None)


def reference(inputs, attrs):
    """@brief fp64 oracle of the packed-varlen KDA forward.

    For an L2 capacity instance the buffers are capacity-sized: the oracle reads the actual offset
    *prefix* from ``cu_seqlens`` and returns capacity-shaped results whose unused region carries the
    declared padding poison.  The harness compares only ``output[:total_tokens]`` and
    ``final_state[:num_seqs]`` and verifies the padding separately, so the returned padding value is not
    part of the numeric verdict — it is here so the oracle's own contract is the same one the candidate
    must satisfy.

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
    cu = _logical_offsets(inputs["cu_seqlens"])
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

    # ``output``/``state`` are capacity-shaped: the recurrence only ever touched the logical prefix, so
    # the unused rows still hold this generator's padding and are not a result.  The harness compares
    # ``output[:total_tokens]`` and ``final_state[:num_seqs]`` only and verifies the padding separately.
    return {"output": output, "final_state": state}


def gen_inputs(case, seed, instance, resolved=None):
    """@brief 生成打包 varlen KDA 输入（含实际 ``cu_seqlens`` 与 per-sequence initial_state）。

    For an L1 tier ``resolved`` is ``None`` and ``total_tokens``/``num_seqs`` are the tier's own
    total/count, so only the segmentation changes with the case.  For an L2 capacity tier the buffers are
    capacity-sized and ``resolved`` carries this call's total/count (plus the declared workload profile):
    the *same* case planner runs on that profile, the boundary tensor is padded to
    ``NSP1 = capacity.num_seqs + 1`` with :data:`CU_SEQLENS_SENTINEL`, and every element outside the
    logical region is filled with :data:`PADDING_POISON_BYTE`.  ``initial_state`` is per-sequence
    distinct and non-zero (a different diagonal bias per sequence), so crossing a segment boundary,
    dropping the initial state or continuing from a zero state cannot pass by luck.

    @param case correctness case（与 OpSpec ``cases.names`` 一致）。
    @param seed 随机种子。
    @param instance 语义实例（L1 含 canonical ``cu_seqlens``；L2 含 capacity 与 workload profiles）。
    @param resolved L2 本次调用的 ``ResolvedShape``（L1/固定档为 ``None``）。
    @return 输入字典。
    @raises ValueError case 未知、档位类型不支持，或该档的 canonical 分布本身不是合法分段。
    """
    generator = torch.Generator().manual_seed(int(seed))
    heads = int(instance.symbols["H"])
    dim = int(instance.symbols["D"])
    capacity_total = int(instance.symbols["total_tokens"])
    capacity_seqs = int(instance.symbols["num_seqs"])
    if resolved is None:
        if instance.cu_seqlens is None:
            raise ValueError("this tier needs the harness-supplied resolved shape (L2 capacity instance)")
        total_tokens, num_seqs = capacity_total, capacity_seqs
        canonical = tuple(int(value) for value in instance.canonical_seq_lens)
    else:
        total_tokens, num_seqs = int(resolved.total_tokens), int(resolved.num_seqs)
        canonical = tuple(int(value) for value in resolved.canonical_seq_lens)
        if total_tokens > capacity_total or num_seqs > capacity_seqs:
            raise ValueError(
                f"resolved shape {total_tokens}/{num_seqs} exceeds the capacity {capacity_total}/{capacity_seqs}"
            )
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
    # Capacity-sized payload: the logical prefix is the real batch, the tail is the declared poison.  An
    # L1 tier has capacity == resolved, so this is exactly the historical allocation and the historical
    # bytes (no padding region exists).
    q, k, v, g = (randn((capacity_total, heads, dim)) for _ in range(4))
    beta = randn((capacity_total, heads))
    initial_state = randn((capacity_seqs, heads, dim, dim), scale=0.25)
    for seq in range(num_seqs):
        initial_state[seq].diagonal(dim1=-2, dim2=-1).add_(0.5 + 0.25 * seq)
    cu_values = list(offsets) + [CU_SEQLENS_SENTINEL] * (capacity_seqs + 1 - len(offsets))
    cu = torch.tensor(cu_values, dtype=torch.int32)
    scale = torch.tensor([1.0 / (dim ** 0.5)], dtype=torch.float32)
    inputs = {
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
    if total_tokens < capacity_total or num_seqs < capacity_seqs:
        # L2 only: mark every element outside the logical region.  The harness validates this before the
        # run and again after it (the padding must survive the candidate), so "poison the unused input"
        # cannot be skipped by a generator that only writes the prefix.
        for name in ("q", "k", "v", "g", "beta", "output", "log_decay_tab", "token_stats"):
            if name in inputs:
                value = _poison_value(instance.tensor(name).dtype)
                _poison_padding(inputs[name], _TOKEN_TERMS if name != "beta" else _TOKEN_TERMS[:2], total_tokens, num_seqs, value)
        for name in ("initial_state", "final_state"):
            if name in inputs:
                value = _poison_value(instance.tensor(name).dtype)
                _poison_padding(inputs[name], _SEQ_TERMS, total_tokens, num_seqs, value)
    return inputs
