"""Extra sentences the search can append to a planner or repair prompt. Every helper returns "" or None when it has nothing to say,
so a switch that is off never changes the text the model sees.

The roofline lower bound counts the direct algorithm's operations. A baseline that is faster than that bound did less work (a 3x3 stride-1
convolution is the case we measured: torch-mps at 777 us against a 932 us direct-convolution bound). Telling the model it is "at the peak"
then points it at a micro-tweak instead of a different algorithm.
"""

from __future__ import annotations

import re
from typing import Any

# A baseline a few percent under the bound is measurement noise (elementwise kernels sit on the bandwidth roof).
# Conv2d's torch-mps baseline is about 17% under the direct-convolution bound, which is a different algorithm.
CLEAR_OF_BOUND = 0.95

_WS = re.compile(r"\s+")
_NUM = re.compile(r"\d+")
_TG = re.compile(r"(\d+)\s*B exceeds the device limit of (\d+)\s*B")


def _report(report: Any) -> dict:
    if report is None:
        return {}
    if isinstance(report, dict):
        return report
    return report.to_json() if hasattr(report, "to_json") else {}


def _roof_and_baselines(report: Any) -> tuple[dict, dict]:
    rep = _report(report)
    roof = rep.get("roofline") or {}
    bases = rep.get("baseline_us") or {}
    return roof if isinstance(roof, dict) else {}, bases if isinstance(bases, dict) else {}


def baseline_faster_than_bound(report: Any) -> tuple[str, float, float] | None:
    """(baseline name, its microseconds, the lower bound) when the fastest baseline beats the roofline bound, else None."""
    roof, bases = _roof_and_baselines(report)
    lb = roof.get("lower_bound_us")
    usable = {k: float(v) for k, v in bases.items() if isinstance(v, (int, float)) and v > 0}
    if not isinstance(lb, (int, float)) or lb <= 0 or not usable:
        return None
    name, us = min(usable.items(), key=lambda kv: kv[1])
    return (name, us, float(lb)) if us < float(lb) * CLEAR_OF_BOUND else None


def bound_beaten_line(report: Any) -> str | None:
    """One sentence for the hardware-feature block. ``target_note`` is the longer version that goes to the planner."""
    beat = baseline_faster_than_bound(report)
    if beat is None:
        return None
    name, us, lb = beat
    return (
        f"The roofline lower bound ({lb:.0f} us) does not hold here: {name} is already at {us:.0f} us, so it uses fewer operations "
        f"(a 3x3 stride-1 convolution is usually Winograd). The reachable target is {us:.0f} us, not the bound."
    )


def target_note(report: Any) -> str | None:
    """The correction for a bound the baseline has already beaten, or None when the bound still holds."""
    beat = baseline_faster_than_bound(report)
    if beat is None:
        return None
    name, us, lb = beat
    return (
        f"The roofline lower bound ({lb:.1f} us) counts the direct algorithm's operations, and it does not hold for this task: "
        f"the fastest baseline ({name}, {us:.1f} us) is faster than that bound, so it uses an algorithm with fewer operations "
        f"(for a 3x3 stride-1 convolution the usual one is Winograd). "
        f"The reachable target is {us:.1f} us, not the lower bound. A kernel near the bound can still be slower than the baseline, "
        f"and closing the gap means a different algorithm, not a tweak of the direct one."
    )


def at_roofline(report: Any, required_gain: float) -> bool:
    """The seed is already inside the noise gate of its own roofline, and no baseline is faster than that bound.

    ``required_gain`` is the factor a replacement must beat the seed by (``Orchestrator._required_gain``). When the seed's
    time/lower-bound is no larger than that, another round cannot clear the noise gate by going faster on the same algorithm.
    A baseline that beats the bound means the bound is the wrong target, so this does not fire.
    """
    if baseline_faster_than_bound(report) is not None:
        return False
    roof, _ = _roof_and_baselines(report)
    gap = roof.get("time_over_lower_bound")
    if not isinstance(gap, (int, float)) or gap <= 0 or not isinstance(required_gain, (int, float)):
        return False
    return float(gap) <= float(required_gain)


def abi_signature(message: str) -> str:
    """The ABI failure with the numbers blanked, so the same limit violation compares equal across repair attempts."""
    return _NUM.sub("#", _WS.sub(" ", message or "").strip())


def threadgroup_limit_note(message: str) -> str | None:
    """The sentence added to a repair prompt when the kernel declared more static threadgroup memory than the device allows."""
    m = _TG.search(message or "")
    if not m:
        return None
    declared, limit = m.group(1), m.group(2)
    return (
        f"Static threadgroup memory must be under {limit} B. This kernel declares {declared} B. "
        f"The next kernel must declare less than {limit} B."
    )


def failure_phrase(status: str, message: str) -> str:
    """One history line for a branch that produced no eligible kernel. Kept to a single short sentence."""
    msg = _WS.sub(" ", message or "").strip()
    msg = msg[:160]
    kind = {"abi_error": "abi", "incorrect": "incorrect", "compile_error": "compile", "unsafe": "unsafe"}.get(status, status or "failed")
    return f"{kind}: {msg}" if msg else kind


def status_lines(*, rnd: int, max_rounds: int, patience_left: int, spent: float, budget: float | None, required_gain: float) -> str:
    """The two lines that tell the model what counts as a win and how much search is left."""
    pct = max(0.0, (float(required_gain) - 1.0) * 100.0)
    money = f"spent ${spent:.2f}" + (f" of ${budget:.2f}" if budget else "")
    return (
        f"A candidate replaces this kernel only when it is at least {pct:.1f}% faster: that is the noise floor of this comparison, and a smaller gain does not count.\n"
        f"Round {rnd}/{max_rounds}, {patience_left} round(s) of patience left, {money}."
    )
