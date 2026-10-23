"""Search decisions that do not call a model, touch a file, or use the GPU.

The live search and a resume replay both call these, so a stored report is judged by the same rule that judged it live.
"""

from __future__ import annotations

import math
import statistics


def candidate_eligible(report_ok: bool, speedup: float | None, *, correctness_only: bool) -> bool:
    """@brief 判断一个已测量的候选能否参与种子比较。

    ``report_ok`` 是报告自己的 ok 位（安全且数值正确）。计时搜索还需要加速比；只查正确性的搜索不需要。

    @param report_ok 报告自身的 ok 标记。
    @param speedup 候选的加速比；未计时时可为 ``None``。
    @param correctness_only 本次搜索是否只查正确性、不比较快慢。
    @return 可参与比较时为 ``True``。
    """
    if not report_ok:
        return False
    if correctness_only:
        return True
    return speedup is not None


def comparison_speedup(eligible: bool, speedup: float | None) -> float:
    """@brief 取比较时使用的加速比。

    不合格的候选按 0 算；合格但缺时间的按 1 算，既不奖励也不惩罚。

    @param eligible 候选是否合格。
    @param speedup 候选的加速比；未计时时为 ``None``。
    @return 用于比较的浮点加速比。
    """
    if not eligible:
        return 0.0
    return 1.0 if speedup is None else float(speedup)


def session_budget_floor(costs: list[float] | tuple[float, ...]) -> float | None:
    """@brief 估算一次 agent 会话的费用，作为「还能不能再开一场」的下限。

    取最近三场已完成会话的中位数，刻意保守：钱不够跑完一场正常会话时，应当在再发一次请求前就停下，
    而不是开一场注定跑不完的会话。

    @param costs 已记录的各会话费用列表（按时间先后）。
    @return 最近至多三场正费用的中位数；没有任何历史时为 ``None``。
    """
    values = []
    for value in costs[-3:]:
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if number > 0:
            values.append(number)
    return statistics.median(values) if values else None


def required_gain(min_improvement: float, noise_z: float, se_a: float, se_b: float) -> float:
    """@brief 算出一次测量要比另一快多少倍才算真的更快。

    取 ``min_improvement`` 与「1 加 ``noise_z`` 倍合并标准误」中的较大者。

    @param min_improvement 搜索要求的最小提速倍数。
    @param noise_z 噪声倍数。
    @param se_a 一方测量的标准误。
    @param se_b 另一方测量的标准误。
    @return 需要的加速比门槛。
    """
    return max(min_improvement, 1.0 + noise_z * math.hypot(se_a, se_b))


def compare_to_best(eligible: bool, speedup: float, best_speedup: float | None, gain: float) -> str:
    """@brief 比较候选与当前种子。

    ``best_speedup`` 为 ``None`` 表示还没有种子。不合格的候选总是 ``keep``；没有种子时 ``gain`` 不起作用。
    相等既不替换也不算平手：候选必须严格更快。

    @param eligible 候选是否合格。
    @param speedup 候选的加速比。
    @param best_speedup 当前种子的加速比；没有种子时为 ``None``。
    @param gain 需要超出的倍数门槛。
    @return 候选成为新种子时为 ``replace``，快但在噪声内时为 ``tie``，其余为 ``keep``。
    """
    if not eligible:
        return "keep"
    if best_speedup is None or speedup > best_speedup * gain:
        return "replace"
    if speedup > best_speedup:
        return "tie"
    return "keep"


def relation_to_best(
    eligible: bool,
    speedup: float,
    best_speedup: float | None,
    se: float,
    best_se: float,
    *,
    min_improvement: float,
    noise_z: float,
) -> str:
    """@brief ``compare_to_best`` 的变体：先用两个标准误算出 ``required_gain``。

    还没有种子时忽略标准误，合格候选直接接替种子。

    @param eligible 候选是否合格。
    @param speedup 候选的加速比。
    @param best_speedup 当前种子的加速比；没有种子时为 ``None``。
    @param se 候选的标准误。
    @param best_se 种子的标准误。
    @param min_improvement 搜索要求的最小提速倍数。
    @param noise_z 噪声倍数。
    @return ``replace``/``tie``/``keep`` 之一。
    """
    if best_speedup is None:
        return compare_to_best(eligible, speedup, None, min_improvement)
    gain = required_gain(min_improvement, noise_z, se, best_se)
    return compare_to_best(eligible, speedup, best_speedup, gain)


def in_confirm_pool(best_speedup: float, other_speedup: float, gain_of_best_over_other: float) -> bool:
    """@brief 判断当前最优是否没比 ``other`` 快出噪声。

    为真时两者都要在确认阶段复测，而不是直接拿当前最优发布。

    @param best_speedup 当前最优的加速比。
    @param other_speedup 另一颗候选的加速比。
    @param gain_of_best_over_other 最优相对另一颗的倍数门槛。
    @return 两者都应进确认池时为 ``True``。
    """
    return best_speedup <= other_speedup * gain_of_best_over_other


# 换一批输入会让同一颗核差 1–2%（taic/llm.py 的 confirm_batches 说明，docs/诚实说明.md）。
# 两颗确认中位数要比，标准误里加上这档的下沿，避免把不到 1% 的输入差当成变快。
# 同一颗核各批极差超过上沿时，公布层不把中位数当成精确值。
CONFIRM_INPUT_FLOOR = 0.01
CONFIRM_SPREAD = 0.02


def confirmation_se(within_rep_se: float, batch_floor: float = CONFIRM_INPUT_FLOOR) -> float:
    """@brief 把 repetitions 内标准误与换输入批次的下沿合并成一个标准误。

    @param within_rep_se 同一批输入重复测量得出的标准误。
    @param batch_floor 换一批输入带来的下沿，缺省为 ``CONFIRM_INPUT_FLOOR``。
    @return 两者平方和开根（负值先截到 0）。
    """
    return math.hypot(max(0.0, float(within_rep_se)), max(0.0, float(batch_floor)))


def batches_agree(values, spread: float = CONFIRM_SPREAD) -> bool:
    """@brief 判断各批输入的结果是否一致。

    @param values 各批测量的时间（微秒）；不是数值的项被忽略。
    @param spread 允许的最快/最慢批极差比例。
    @return 不足两批，或最快与最慢两批落在 ``spread`` 以内时为 ``True``。
    """
    nums = [float(v) for v in values or [] if isinstance(v, (int, float)) and not isinstance(v, bool)]
    if len(nums) < 2:
        return True
    lo, hi = min(nums), max(nums)
    return lo > 0 and hi / lo <= 1.0 + spread


def choose_confirmed(measured: list[dict], *, min_improvement: float, noise_z: float) -> dict | None:
    """@brief 决定确认阶段可以指名哪一颗完整复测过的内核。

    每项带 ``median``、``se``（repetitions 内）与 ``search_best``。除非挑战者在把换输入下沿算进去后
    仍请过 ``required_gain``，否则搜索最优不变。搜索最优没有干净的复测时，最快的挑战者必须同时对每个
    其他挑战者都过这道门槛才能胜出。噪声分不开的一组返回 ``None``。

    @param measured 完整复测过的条目列表。
    @param min_improvement 搜索要求的最小提速倍数。
    @param noise_z 噪声倍数。
    @return 胜出的条目；分不开或列表为空时为 ``None``。
    """
    if not measured:
        return None

    def beats(faster: dict, slower: dict) -> bool:
        gain = required_gain(min_improvement, noise_z, confirmation_se(faster["se"]), confirmation_se(slower["se"]))
        return faster["median"] > slower["median"] * gain

    best = next((item for item in measured if item.get("search_best")), None)
    if best is not None:
        challengers = [item for item in measured if item is not best and beats(item, best)]
        if not challengers:
            return best
        ordered = sorted(challengers, key=lambda item: item["median"], reverse=True)
        top = ordered[0]
        if len(ordered) == 1 or all(beats(top, other) for other in ordered[1:]):
            return top
        return best
    ordered = sorted(measured, key=lambda item: item["median"], reverse=True)
    top = ordered[0]
    if len(ordered) == 1 or all(beats(top, other) for other in ordered[1:]):
        return top
    return None


def next_stagnant(replaced: bool, stagnant: int) -> int:
    """@brief 推进「连续没有新种子」的计数器。

    换掉种子就清零；落在噪声内的平手不会清零。

    @param replaced 本轮是否换掉了种子。
    @param stagnant 当前连续无进展的轮数。
    @return 更新后的计数。
    """
    return 0 if replaced else stagnant + 1


def patience_exhausted(stagnant: int, patience: int) -> bool:
    """@brief 判断连续无新种子的轮数是否已到耐心上限。

    @param stagnant 当前连续无进展的轮数。
    @param patience 允许的耐心轮数。
    @return 到达或超过上限时为 ``True``。
    """
    return stagnant >= patience


def feature_signature(source: str, *, backend_id: str) -> tuple:
    """@brief 提取源码的七个静态特征。

    不用编译器也不用 GPU；只有 Metal 后端实现了这些特征。后端不支持该能力时返回空元组。

    @param source 内核源码。
    @param backend_id 后端标识，决定是否支持特征签名。
    @return 七元组：matrix_unit、acc_bucket、tg_staging、double_buffer、register_prefetch、vector_load、online_softmax。
    """
    try:
        from tcenv.backends.registry import load_backend

        if not load_backend(backend_id).capabilities.supports_feature_signature:
            return ()
    except Exception:
        return ()
    from .hwfeatures import analyze_source

    from tcenv.backends.metal.launch import declared_stage_kernels

    launched = declared_stage_kernels(source)
    ev = analyze_source(source, kernel_names=launched)
    matrix = ev.get("matrix") or {}
    tiles = matrix.get("acc_tiles") or 0
    if not isinstance(tiles, int):
        tiles = 0
    if tiles >= 32:
        bucket = "32+"
    elif tiles >= 16:
        bucket = "16"
    elif tiles >= 4:
        bucket = "4"
    elif tiles >= 1:
        bucket = "1"
    else:
        bucket = "0"
    group = ev.get("threadgroup") or {}
    memory = ev.get("memory") or {}
    load = (memory.get("device_load") or {}).get("max_bytes") or 0
    online = ev.get("online_softmax") or {}
    return (
        bool(matrix.get("present")),
        bucket,
        bool(group.get("staging")),
        group.get("double_buffer") == "explicit",
        bool(group.get("register_prefetch")),
        int(load) >= 8,
        bool(online.get("present")),
    )


def _hamming(left: tuple, right: tuple) -> int:
    """@brief 逐位比较两个签名，返回不同位的个数。

    @param left 左侧特征签名。
    @param right 右侧特征签名。
    @return 不同位的个数。
    """
    return sum(a != b for a, b in zip(left, right))


def select_diverse_parent(candidates: list, best, *, task_name: str = "", max_archive: int = 4, backend_id: str = "metal"):
    """@brief 在归档里挑一颗与 ``best`` 在特征空间距离最远的合格内核。

    ``task_name`` 仅由调用方记录，不参与比较。归档最多保留 ``max_archive`` 个签名，每个签名由其最快的
    内核代表，慢于种子 85% 的候选直接丢掉。

    @param candidates 本轮收集到的候选列表。
    @param best 当前种子候选。
    @param task_name 任务名，仅记录、不参与比较。
    @param max_archive 归档保留的签名数上限。
    @param backend_id 后端标识，决定特征签名是否可用。
    @return 选中的候选；``best`` 为 ``None`` 或归档为空时为 ``None``。
    """
    del task_name
    if best is None:
        return None
    grouped: dict[tuple, object] = {}
    for cand in candidates:
        if not getattr(cand, "eligible", False) or getattr(cand, "recheck_rejected", False):
            continue
        if cand.source == best.source:
            continue
        if cand.speedup < 0.85 * best.speedup:
            continue
        sig = feature_signature(cand.source, backend_id=backend_id)
        kept = grouped.get(sig)
        if kept is None or cand.speedup > kept.speedup:
            grouped[sig] = cand
    archive = sorted(grouped.values(), key=lambda cand: cand.speedup, reverse=True)[:max_archive]
    if not archive:
        return None
    seed_sig = feature_signature(best.source, backend_id=backend_id)
    return max(archive, key=lambda cand: (_hamming(feature_signature(cand.source, backend_id=backend_id), seed_sig), cand.speedup, cand.label))
