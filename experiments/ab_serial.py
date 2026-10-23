"""A/B the extra strictly serial runner (``candidate_serial``) on real kernels.

This is the measured decision the data-layer review asked for. It does not call a model and does
not run a search; it runs ``tcenv eval --timed`` on already-confirmed kernels, alternating the
serial and no-serial modes, and writes the raw reports plus a median summary.

    uv run python -m experiments.ab_serial --rounds 3
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
KERNELS = (
    ("conv2d", "fp16", "paper", "runs/apple-m4-max/conv2d-feedback-20261021/best.metal"),
    ("flash_attention", "fp16", "paper", "runs/apple-m4-max/flash_attention-20261022-2/best.metal"),
    ("relu", "fp16", "paper", "runs/apple-m4-max/relu-pi-live-20261019/best.metal"),
)


def _run(task: str, dtype: str, size: str, kernel: Path, *, serial: bool, seed: int, reps: int) -> dict:
    """@brief 对已有内核跑一次 ``tcenv eval --timed``，只切换串行模式。

    @param task 任务名，交给 ``tcenv.cli eval``。
    @param dtype 数据类型，例如 ``fp16``。
    @param size 规模档位，例如 ``paper``。
    @param kernel 要评测的金属内核文件路径。
    @param serial 是否启用严格串行 runner；为假时追加 ``--no-serial``。
    @param seed 计时用的随机种子。
    @param reps 计时重复次数。
    @return 报告里的状态、候选耗时、加速比等字段组成的字典；输出不是 JSON 时为 ``parse_error``。
    """
    with tempfile.TemporaryDirectory(prefix="tcenv_ab_check_") as check:
        argv = [
            "uv", "run", "python", "-m", "tcenv.cli", "eval", task, str(kernel),
            "--dtype", dtype, "--size", size, "--timed", "--reps", str(reps),
            "--seed", str(seed), "--target", "apple-m4-max", "--self-check-dir", check, "--json",
        ]
        if not serial:
            argv.append("--no-serial")
        proc = subprocess.run(argv, cwd=ROOT, capture_output=True, text=True, timeout=300)
    try:
        report = json.loads(proc.stdout)
    except ValueError:
        return {"status": "parse_error", "stdout": proc.stdout[-400:], "stderr": proc.stderr[-400:]}
    bench = report.get("benchmark") or {}
    return {
        "status": report.get("status"), "candidate_us": report.get("candidate_us"),
        "speedup": report.get("speedup"), "elapsed_s": report.get("elapsed_s"),
        "noise_cv": bench.get("noise_cv"), "candidate_serial_us": bench.get("candidate_serial_us"),
        "attempts": bench.get("attempts"), "measure_serial": bench.get("measure_serial"),
        "stderr_tail": proc.stderr[-200:],
    }


def main(argv: list[str] | None = None) -> int:
    """@brief 交替跑串行与非串行两种模式，并写出逐次报告与中位数汇总。

    @param argv 命令行参数；为 ``None`` 时取 ``sys.argv``。
    @return 进程退出码，正常结束时为 0。
    """
    parser = argparse.ArgumentParser(prog="python -m experiments.ab_serial")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--reps", type=int, default=15)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--out", default="docs/review/data_layer_20261022/SERIAL_AB.json")
    args = parser.parse_args(argv)
    rows = []
    for task, dtype, size, relative in KERNELS:
        kernel = ROOT / relative
        if not kernel.is_file():
            continue
        for round_index in range(args.rounds):
            order = (True, False) if round_index % 2 == 0 else (False, True)
            for serial in order:
                mode = "serial" if serial else "no-serial"
                result = _run(task, dtype, size, kernel, serial=serial, seed=args.seed, reps=args.reps)
                rows.append({"task": task, "dtype": dtype, "size": size, "kernel": relative, "round": round_index, "mode": mode, **result})
                print(json.dumps(rows[-1], ensure_ascii=False), flush=True)
    summary = []
    for task, dtype, size, relative in KERNELS:
        entry = {"task": task, "kernel": relative}
        for mode in ("serial", "no-serial"):
            values = [row for row in rows if row["task"] == task and row["mode"] == mode and row.get("status") == "ok"]
            entry[mode] = {
                "n": len(values),
                "candidate_us": [round(row["candidate_us"], 3) for row in values],
                "median_candidate_us": round(statistics.median(row["candidate_us"] for row in values), 3) if values else None,
                "speedup": [round(row["speedup"], 4) for row in values],
                "median_speedup": round(statistics.median(row["speedup"] for row in values), 4) if values else None,
                "elapsed_s": [round(row["elapsed_s"], 3) for row in values],
                "median_elapsed_s": round(statistics.median(row["elapsed_s"] for row in values), 3) if values else None,
            }
        if entry.get("serial", {}).get("median_candidate_us") and entry.get("no-serial", {}).get("median_candidate_us"):
            base = entry["serial"]["median_candidate_us"]
            entry["candidate_us_delta_pct"] = round((entry["no-serial"]["median_candidate_us"] - base) / base * 100, 3)
        summary.append(entry)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"rows": rows, "summary": summary}, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"summary": summary, "out": str(out)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
