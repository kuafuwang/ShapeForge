"""varlen_scale: the independent oracle of the packed-varlen elementwise scale.

``y[t] = SCALE * x[t]`` over every token of the packed batch.  The operator is elementwise, so the
per-sequence boundaries do not change the math; the oracle still walks the *actual* ``cu_seqlens``
segments instead of reading the instance's canonical copy, so a segmentation that leaves a token
uncovered (or double-covers one) cannot hide behind the elementwise shortcut.  The oracle stays in
fp64 and never reads the workspace or a symbol.

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
    every segment prime, so several partial tails sit next to each other; a tier with no
    all-prime segmentation (``total_tokens`` too small for ``num_seqs`` primes) falls back to
    its canonical profile and the harness records ``not_constructible`` — never a non-prime
    layout labelled exact;
``max_handoff_stress``
    one segment of ``total_tokens - (num_seqs - 1)`` and the rest one token each;
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

The case planner is deliberately local to this task instead of shared with
``tasks/varlen_kda/reference.py``: a reference module may not import this repository (or a relative
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


def reference(inputs, attrs):
    """@brief fp64 oracle of the packed-varlen elementwise scale.

    @param inputs 合同输入（``x`` 已按声明 dtype 舍入并加宽成 float32；``cu_seqlens`` 是分段偏移）。
    @param attrs 已解析的整数 attr（``SCALE``）。
    @return ``{"y": ...}``（fp64）。
    """
    x = inputs["x"].to(torch.float64)
    scale = float(attrs["SCALE"])
    cu = [int(value) for value in inputs["cu_seqlens"].tolist()]
    y = torch.zeros_like(x)
    for seq in range(len(cu) - 1):
        y[cu[seq] : cu[seq + 1]] = scale * x[cu[seq] : cu[seq + 1]]
    return {"y": y}


def gen_inputs(case, seed, instance):
    """@brief 生成打包 varlen 输入：按 case 得到的 cu_seqlens 张量 + 打包 x。

    ``total_tokens``/``num_seqs`` 与档位一致；只有分段随 case 变化。

    @param case L1 correctness case（与 OpSpec ``cases.names`` 一致）。
    @param seed 随机种子。
    @param instance 语义实例（含 canonical ``cu_seqlens`` 与 total_tokens/H/D symbols）。
    @return 输入字典。
    @raises ValueError case 未知，或该档的 canonical 分布本身不是合法分段。
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
    cu = torch.tensor(offsets, dtype=torch.int32)
    x = torch.randn((total_tokens, heads, dim), generator=generator)
    return {"x": x, "cu_seqlens": cu}
