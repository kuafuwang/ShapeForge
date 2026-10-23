"""Search decisions that do not call a model, touch a file, or use the GPU.

The live search and a resume replay both call these, so a stored report is judged by the same rule that judged it live.
"""

from __future__ import annotations

import math
import statistics


def candidate_eligible(report_ok: bool, speedup: float | None, *, correctness_only: bool) -> bool:
    """Whether a measured candidate may be compared for the seed.

    ``report_ok`` is the report's own ok flag (safe and numerically correct). A timed search also needs a speedup.
    A correctness-only search does not.
    """
    if not report_ok:
        return False
    if correctness_only:
        return True
    return speedup is not None


def comparison_speedup(eligible: bool, speedup: float | None) -> float:
    """The speedup used in a comparison. An ineligible candidate compares as 0; a missing timing on an eligible one compares as 1."""
    if not eligible:
        return 0.0
    return 1.0 if speedup is None else float(speedup)


def session_budget_floor(costs: list[float] | tuple[float, ...]) -> float | None:
    """The recent cost of one agent session, or None when there is no history.

    The median of the last three completed sessions is deliberately conservative: a search
    that has only enough money for a fraction of its normal session should stop before
    spending another request, rather than starting a session that cannot finish.
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
    """How many times faster a measurement must be before it counts as faster than another.

    At least ``min_improvement``, and also enough to clear ``noise_z`` times the combined standard errors.
    """
    return max(min_improvement, 1.0 + noise_z * math.hypot(se_a, se_b))


def compare_to_best(eligible: bool, speedup: float, best_speedup: float | None, gain: float) -> str:
    """``replace`` when this candidate becomes the seed, ``tie`` when it is faster but inside the noise, else ``keep``.

    ``best_speedup`` None means there is no seed yet. An ineligible candidate is always ``keep``. ``gain`` is ignored
    until a seed exists. Equality does not replace and does not tie: the candidate must be strictly faster.
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
    """``compare_to_best`` after applying ``required_gain`` to the two standard errors.

    When there is no seed yet, the standard errors are ignored and an eligible candidate replaces it.
    """
    if best_speedup is None:
        return compare_to_best(eligible, speedup, None, min_improvement)
    gain = required_gain(min_improvement, noise_z, se, best_se)
    return compare_to_best(eligible, speedup, best_speedup, gain)


def in_confirm_pool(best_speedup: float, other_speedup: float, gain_of_best_over_other: float) -> bool:
    """True when the current best does not beat ``other`` by more than the noise, so both are remeasured."""
    return best_speedup <= other_speedup * gain_of_best_over_other


# 换一批输入会让同一颗核差 1–2%（taic/llm.py 的 confirm_batches 说明，docs/诚实说明.md）。
# 两颗确认中位数要比，标准误里加上这档的下沿，避免把不到 1% 的输入差当成变快。
# 同一颗核各批极差超过上沿时，公布层不把中位数当成精确值。
CONFIRM_INPUT_FLOOR = 0.01
CONFIRM_SPREAD = 0.02


def confirmation_se(within_rep_se: float, batch_floor: float = CONFIRM_INPUT_FLOOR) -> float:
    """Within-repetition standard error combined with the input-batch floor."""
    return math.hypot(max(0.0, float(within_rep_se)), max(0.0, float(batch_floor)))


def batches_agree(values, spread: float = CONFIRM_SPREAD) -> bool:
    """True when one batch, or when the fastest and slowest batches stay inside ``spread``."""
    nums = [float(v) for v in values or [] if isinstance(v, (int, float)) and not isinstance(v, bool)]
    if len(nums) < 2:
        return True
    lo, hi = min(nums), max(nums)
    return lo > 0 and hi / lo <= 1.0 + spread


def choose_confirmed(measured: list[dict], *, min_improvement: float, noise_z: float) -> dict | None:
    """Which fully remeasured kernel the confirmation may name.

    Each item has ``median``, ``se`` (within-repetition) and ``search_best``. The search best stays
    unless a challenger clears ``required_gain`` after the input-batch floor is included. When the
    search best has no clean remeasure, the fastest challenger wins only if it clears that gate
    against every other challenger. A cluster that the noise does not separate returns None.
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
    """Rounds without a new seed. A replacement resets the count; a within-noise tie does not."""
    return 0 if replaced else stagnant + 1


def patience_exhausted(stagnant: int, patience: int) -> bool:
    """True when rounds without a new seed have reached the patience limit."""
    return stagnant >= patience


def feature_signature(source: str) -> tuple:
    """Seven static source features. No compiler and no GPU."""
    from .hwfeatures import analyze_source

    from tcenv.launch import declared_stage_kernels

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
    return sum(a != b for a, b in zip(left, right))


def select_diverse_parent(candidates: list, best, *, task_name: str = "", max_archive: int = 4):
    """The eligible kernel in the archive farthest in feature space from ``best``, or None.

    ``task_name`` is recorded by callers and is not used in the comparison. The archive keeps at most
    ``max_archive`` signatures, each represented by its fastest kernel, and drops anything slower than
    85% of the seed.
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
        sig = feature_signature(cand.source)
        kept = grouped.get(sig)
        if kept is None or cand.speedup > kept.speedup:
            grouped[sig] = cand
    archive = sorted(grouped.values(), key=lambda cand: cand.speedup, reverse=True)[:max_archive]
    if not archive:
        return None
    seed_sig = feature_signature(best.source)
    return max(archive, key=lambda cand: (_hamming(feature_signature(cand.source), seed_sig), cand.speedup, cand.label))
