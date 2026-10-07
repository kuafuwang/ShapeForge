"""A/B of the search switches added for cost and for feedback direction. Preset ``base``, $3 per run, so the numbers can sit next to ``fa_flash``.

    uv run python -m experiments.search_ab --which conv2d          # one conv2d run with every switch on
    uv run python -m experiments.search_ab --which fa              # flash_attention, 3 on and 3 off
    uv run python -m experiments.search_ab --which fa --arm on     # only the treatment arm

The switches are the phase-1 and phase-2 set: truncation retry, stop at the roofline, the ABI guard, the target note, rich history and the
status lines. ``hw_feedback`` stays off, so this comparison is not mixed with that one. n=3 on a stochastic search is still a small sample.
"""

from __future__ import annotations

import argparse

from .common import run_one

SWITCHES = dict(
    stop_at_roofline=True,
    abi_guard=True,
    target_note=True,
    rich_history=True,
    status_lines=True,
    truncation_retry=True,
)
# The production default is already on. The off arm has to say so, or it would be the same search as the on arm.
OFF = {k: False for k in SWITCHES}


def _run(name: str, task: str, out, *, on: bool, budget: float) -> None:
    print(f"\n##### {name} switches={'on' if on else 'off'} budget=${budget:.2f}", flush=True)
    run_one(task, "fp16", "paper", out / name, budget=budget, preset="base", **(SWITCHES if on else OFF))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", choices=("conv2d", "fa"), required=True)
    ap.add_argument("--arm", choices=("on", "off", "both"), default="both")
    ap.add_argument("--budget", type=float, default=3.0)
    ap.add_argument("--n", type=int, default=3, help="trials per arm for --which fa")
    ap.add_argument("--out", default="runs/search_ab")
    a = ap.parse_args(argv)
    out = __import__("pathlib").Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.which == "conv2d":
        _run("conv2d_on", "conv2d", out, on=True, budget=a.budget)
        return 0
    arms = ("on", "off") if a.arm == "both" else (a.arm,)
    for arm in arms:
        for i in range(1, a.n + 1):
            _run(f"fa_{arm}_{i}", "flash_attention", out, on=(arm == "on"), budget=a.budget)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
