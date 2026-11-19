#!/usr/bin/env python3
"""One device run that answers the three questions the CPU suite cannot.

Everything the feedback claimed about measurement is a *device* property: the
GPU-timestamp reading, the size of the window's own noise floor, and whether a
v2 trace carries compiler spill events. This script measures all three in one
window and prints a pass/fail table, so the claims can be closed with numbers
instead of with another round of reasoning.

Run it on the machine, from the repository root:

    uv run python scripts/probe_timing_window.py \\
        --task conv2d --dtype fp16 --size small --kernel path/to/kernel.metal

``--kernel`` must be a candidate that already passes ``eval-v2 --quick``: the
two-file ranking this script depends on refuses a file the task's own quick
evaluation rejects.

It consumes ``--timed`` self-check slots (one for the control run). It does not
publish anything and writes no evidence.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _run(args: list[str]) -> tuple[int, str, str]:
    """@brief 在仓库根目录跑一条子命令，返回 (退出码, stdout, stderr)。"""
    proc = subprocess.run(
        [sys.executable, "-m", "tcenv.cli", *args],
        cwd=str(ROOT), text=True, capture_output=True,
    )
    return proc.returncode, proc.stdout, proc.stderr


def _first_json(text: str) -> dict | None:
    """@brief 从可能带尾随说明行的输出里取出第一份 JSON 文档。"""
    decoder = json.JSONDecoder()
    start = text.find("{")
    while start != -1:
        try:
            value, _end = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            start = text.find("{", start + 1)
            continue
        if isinstance(value, dict):
            return value
        start = text.find("{", start + 1)
    return None


def _ab(kernel: Path, copy: Path, task: str, dtype: str, size: str, reps: int) -> dict | None:
    code, out, err = _run([
        "ab", str(kernel), str(copy), "--task", task, "--dtype", dtype, "--size", size,
        "--reps", str(reps), "--json",
    ])
    # ab still prints its document when it refuses (both files unsafe is exit 2 with JSON on
    # stdout and nothing on stderr), so a non-zero code must not be read as "no output".
    document = _first_json(out)
    if code != 0 and document is None:
        print(f"  ab failed (exit {code}): {(err or out).strip()[:500]}")
        return None
    if code != 0:
        print(f"  ab refused (exit {code}): see the rejected list below")
    return document


def _timed_with_control(kernel: Path, task: str, dtype: str, size: str, reps: int, ledger: Path) -> dict | None:
    # The self-check ledger defaults to the kernel's own directory. A probe must not drop a
    # ledger into the repository (or worse, into a golden fixture), so it is redirected.
    code, out, err = _run([
        "eval-v2", task, str(kernel), "--dtype", dtype, "--size", size,
        "--timed", "--control", "--reps", str(reps),
        "--self-check-dir", str(ledger), "--json",
    ])
    document = _first_json(out)
    if code != 0 and document is None:
        print(f"  eval-v2 --timed --control failed (exit {code}): {(err or out).strip()[:500]}")
        return None
    if code != 0:
        print(f"  eval-v2 exited {code}")
    return document


def _profile(kernel: Path, task: str, dtype: str, size: str) -> dict | None:
    code, out, err = _run([
        "profile", task, str(kernel), "--dtype", dtype, "--size", size, "--json",
    ])
    document = _first_json(out)
    if code != 0 and document is None:
        print(f"  profile failed (exit {code}): {(err or out).strip()[:500]}")
        return None
    return document


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", required=True)
    parser.add_argument("--dtype", required=True, choices=("fp16", "bf16"))
    parser.add_argument("--size", required=True)
    parser.add_argument("--kernel", required=True, help="a candidate that already passes eval-v2 --quick")
    parser.add_argument("--reps", type=int, default=16)
    parser.add_argument("--skip-profile", action="store_true", help="skip the xctrace step (it needs xcrun)")
    args = parser.parse_args(argv)

    kernel = Path(args.kernel).resolve()
    if not kernel.is_file():
        print(f"no such kernel: {kernel}", file=sys.stderr)
        return 2

    work = Path(tempfile.mkdtemp(prefix="sf-probe-"))
    copy = work / f"{kernel.stem}_bytecopy{kernel.suffix}"
    shutil.copyfile(kernel, copy)

    checks: list[tuple[str, bool, str]] = []
    # ``ab`` and ``--timed`` measure the same physical quantity (GPU time per launch), so the
    # honest check is that they agree -- not that the median is under some absolute number. The
    # earlier version hard-coded 50 us, which was wrong for any kernel slower than that.
    ab_best: list[float] = [0.0]

    print("=" * 78)
    print("1/3  two files in one window (`tcenv ab`)")
    print("=" * 78)
    ab = _ab(kernel, copy, args.task, args.dtype, args.size, args.reps)
    if ab is None:
        checks.append(("ab measured", False, "the call did not produce a document"))
    else:
        rows = ab.get("rows") or []
        medians = {row.get("label"): float(row.get("median_us") or 0.0) for row in rows}
        best = min(medians.values()) if medians else 0.0
        print(f"  timed_by          : {ab.get('timed_by')}")
        for label, value in medians.items():
            print(f"  median {label:<28}: {value:10.3f} us")
        # The field reports 146-164 us medians for a ~3 us kernel because the old path
        # timed wall clock per launch. A correct GPU-timestamp reading is single digit here.
        checks.append((
            "ab reports GPU timestamps", ab.get("timed_by") == "gpu_timestamp",
            f"timed_by={ab.get('timed_by')}",
        ))
        ab_best[0] = best
        pairs = [row for row in rows if row.get("label") != ab.get("control")]
        for row in pairs:
            ratio = row.get("paired_ratio_to_first")
            if ratio:
                checks.append((
                    f"A/A ratio near 1 ({row.get('label')})", 0.97 <= float(ratio) <= 1.03,
                    f"{float(ratio):.4f}",
                ))
        if not pairs:
            checks.append(("A/A pair measured", False, "only the control row came back"))

    print()
    print("=" * 78)
    print("2/3  same-window control inside --timed (`eval-v2 --control`)")
    print("=" * 78)
    timed = _timed_with_control(kernel, args.task, args.dtype, args.size, args.reps, work)
    resolution = None
    if timed is None:
        checks.append(("--timed --control produced a report", False, "the call did not produce a document"))
    else:
        bench = timed.get("benchmark") if isinstance(timed.get("benchmark"), dict) else {}
        control = bench.get("control") if isinstance(bench.get("control"), dict) else {}
        resolution = control.get("resolvable_relative")
        print(f"  candidate_gpu_us_median : {bench.get('candidate_gpu_us_median')}")
        print(f"  control.median_ratio    : {control.get('median_ratio')}")
        print(f"  control.interval        : {control.get('interval')}")
        print(f"  control.rounds          : {control.get('rounds')}")
        print(f"  resolvable_relative     : {resolution}")
        checks.append((
            "control appears in the report", isinstance(bench.get("control"), dict), "",
        ))
        ratio = control.get("median_ratio")
        checks.append((
            "control ratio near 1.000",
            isinstance(ratio, (int, float)) and 0.97 <= float(ratio) <= 1.03,
            f"median_ratio={ratio}",
        ))
        checks.append((
            "control interval contains 1.0",
            isinstance(control.get("interval"), list)
            and len(control["interval"]) == 2
            and float(control["interval"][0]) <= 1.0 <= float(control["interval"][1]),
            f"interval={control.get('interval')}",
        ))
        checks.append((
            "window resolution is recorded", isinstance(resolution, (int, float)) and float(resolution) > 0,
            f"resolvable_relative={resolution}",
        ))
        timed_median = bench.get("candidate_gpu_us_median")
        if ab_best[0] > 0 and isinstance(timed_median, (int, float)) and timed_median > 0:
            ratio = ab_best[0] / float(timed_median)
            checks.append((
                "ab and --timed agree on the same physical quantity", 0.75 <= ratio <= 1.25,
                f"ab {ab_best[0]:.1f} us vs --timed {float(timed_median):.1f} us (ratio {ratio:.3f}); "
                "the wall-clock path would sit far above the timed median",
            ))

    print()
    print("=" * 78)
    print("3/3  v2 trace spill evidence (`tcenv profile`)")
    print("=" * 78)
    if args.skip_profile:
        print("  skipped by --skip-profile")
    else:
        summ = _profile(kernel, args.task, args.dtype, args.size)
        if summ is None:
            checks.append(("profile recorded a v2 trace", False, "the call did not produce a summary"))
        else:
            tables = summ.get("tables") if isinstance(summ.get("tables"), dict) else {}
            print(f"  dispatch_group   : {summ.get('dispatch_group')}")
            print(f"  spill_events     : {summ.get('spill_events')}")
            print(f"  spilled_bytes    : {summ.get('spilled_bytes')}")
            print(f"  tables.spills    : {tables.get('spills')}")
            checks.append(("profile no longer refuses v2", True, "a summary came back"))
            # 'spills' False means the trace template does not record it: unknown, not zero.
            checks.append((
                "spill table present (unknown is not zero)",
                tables.get("spills") is not False,
                f"tables.spills={tables.get('spills')}",
            ))

    print()
    print("=" * 78)
    print("summary")
    print("=" * 78)
    failures = 0
    for name, ok, detail in checks:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  [{detail}]" if detail else ""))
        failures += 0 if ok else 1
    print()
    if failures == 0:
        print("all device checks passed; the measurement claims can be closed with these numbers")
    else:
        print(f"{failures} check(s) failed — do not close the measurement claims on this run")
    print(f"work directory kept for inspection: {work}")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
