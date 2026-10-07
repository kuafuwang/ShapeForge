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
