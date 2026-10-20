"""Search decisions that do not call a model, touch a file, or use the GPU.

The live search and a resume replay both call these, so a stored report is judged by the same rule that judged it live.
"""

from __future__ import annotations

import math


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
