"""Read-only roofline headroom of OpSpec v2 tasks. No model call and no GPU.

``python -m experiments.task_headroom`` loads every standard v2 task, turns
``work_accounting`` into an arithmetic intensity and a roofline lower bound from
``targets/<id>/calibration.json``, and compares that bound with the fastest
schema-4 ``baseline_us`` already stored under ``runs/``. A confirmed speedup is
read only from ``result.json`` when the confirmation gates pass.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from taic.artifacts import document_state

REPO = Path(__file__).resolve().parents[1]
HIGH_HEADROOM_OPS = frozenset({
    "kda",
    "varlen_kda",
    "mamba2_chunk_scan",
    "flash_attention",
    "bitdelta_gemm",
    "gemm_silu",
})
TIER_RANK = {
    "high_headroom": 0,
    "unmeasured_on_target": 1,
    "algorithm_shift_needed": 2,
    "limited_headroom": 3,
    "saturated_bandwidth": 4,
    "launch_bound_tiny": 5,
}


@dataclass(frozen=True)
class InstanceHeadroom:
    """@brief 一个 v2 档位相对 roofline 和历史基线的余量。"""

    op_name: str
    dtype: str
    size: str
    usage: tuple[str, ...]
    workspace_declared: bool
    variable_length_kind: str
    flops: int
    min_bytes: int
    arithmetic_intensity: float | None
    bound: str
    lower_bound_us: float | None
    best_baseline_us: float | None
    best_confirmed_speedup: float | None
    headroom_vs_bound: float | None
    tier: str

    def document(self) -> dict[str, Any]:
        """@brief JSON 投影。"""
        return {
            "op_name": self.op_name,
            "dtype": self.dtype,
            "size": self.size,
            "usage": list(self.usage),
            "workspace_declared": self.workspace_declared,
            "variable_length_kind": self.variable_length_kind,
            "flops": self.flops,
            "min_bytes": self.min_bytes,
            "arithmetic_intensity": self.arithmetic_intensity,
            "bound": self.bound,
            "lower_bound_us": self.lower_bound_us,
            "best_baseline_us": self.best_baseline_us,
            "best_confirmed_speedup": self.best_confirmed_speedup,
            "headroom_vs_bound": self.headroom_vs_bound,
            "tier": self.tier,
        }


def collect_headroom(
    *,
    target_id: str = "apple-m4-max",
    dtype: str = "fp16",
    runs_root: str | Path = "runs",
    tasks_root: str | Path | None = None,
    include_semantic: bool = False,
) -> list[InstanceHeadroom]:
    """@brief 扫描 v2 标准题并按优化价值排序。

    ``best_baseline_us`` 是同一 ``(op, dtype, size, target)`` 的 schema-4
    ``report.json`` 里，各报告最快基线中位数中的最小值。没有记录时为 None。
    ``best_confirmed_speedup`` 只来自同一键的 ``result.json``：``confirmed_status``
    为 ``ok``、``confirmation_source_sha256`` 是 64 位小写十六进制，且
    ``confirmed_speedup`` 是有限正数。``report.json`` 没有确认加速比这个字段，
    ``best_speedup`` 不能拿来填补。

    @param target_id 标定与报告要匹配的目标。
    @param dtype 只装配这个数据类型。
    @param runs_root 历史运行根；不存在时当作没有实测。
    @param tasks_root 任务根，默认仓库 ``tasks/``。
    @param include_semantic 为真时保留 ``sem_*`` 语义夹具。
    @return 按分诊档和余量排序的行。
    """
    from tcenv.taskcontract import load_task_contract

    root = Path(tasks_root) if tasks_root is not None else REPO / "tasks"
    calibration = _calibration(target_id)
    baselines = _baseline_index(Path(runs_root), target_id)
    rows: list[InstanceHeadroom] = []
    for spec_path in sorted(root.glob("*/opspec.json")):
        task_dir = spec_path.parent
        if not include_semantic and (task_dir.name.startswith("sem_") or task_dir.name.startswith("sem-")):
            continue
        try:
            spec = json.loads(spec_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(spec, dict) or int(spec.get("spec_version") or 0) != 2:
            continue
        if dtype not in _offered_dtypes(spec):
            continue
        for size in spec.get("sizes") or {}:
            contract = load_task_contract(task_dir, dtype=dtype, size=str(size))
            instance = contract.instance
            if not include_semantic and str(instance.op_name).startswith("sem_"):
                continue
            flops = _whole(instance.work_accounting.flops)
            min_bytes = _whole(instance.work_accounting.min_bytes)
            intensity, bound, lower = _roofline(flops, min_bytes, calibration, dtype)
            sample = baselines.get((instance.op_name, dtype, str(size)))
            best_baseline = None if sample is None else sample[0]
            best_speedup = None if sample is None else sample[1]
            headroom = None
            if best_baseline is not None and lower is not None and lower > 0:
                headroom = best_baseline / lower
            row = InstanceHeadroom(
                op_name=str(instance.op_name),
                dtype=dtype,
                size=str(size),
                usage=tuple(str(item) for item in instance.size_usage),
                workspace_declared=any(getattr(tensor, "space", "") == "workspace" for tensor in instance.tensors),
                variable_length_kind=str(instance.variable_length_kind),
                flops=flops,
                min_bytes=min_bytes,
                arithmetic_intensity=intensity,
                bound=bound,
                lower_bound_us=lower,
                best_baseline_us=best_baseline,
                best_confirmed_speedup=best_speedup,
                headroom_vs_bound=headroom,
                tier="",
            )
            rows.append(_with_tier(row))
    rows.sort(key=_sort_key)
    return rows


def format_table(rows: list[InstanceHeadroom]) -> str:
    """@brief 把分诊结果打成 Markdown 表。"""
    header = (
        "| op | size | kind | bound | lower_us | baseline_us | headroom | tier |\n"
        "| --- | --- | --- | --- | ---: | ---: | ---: | --- |"
    )
    body = []
    for row in rows:
        body.append(
            "| {op} | {size} | {kind} | {bound} | {lower} | {base} | {head} | {tier} |".format(
                op=row.op_name,
                size=row.size,
                kind=row.variable_length_kind,
                bound=row.bound,
                lower=_num(row.lower_bound_us),
                base=_num(row.best_baseline_us),
                head=_num(row.headroom_vs_bound),
                tier=row.tier,
            )
        )
    if not body:
        return header + "\n| (none) | | | | | | | |"
    return header + "\n" + "\n".join(body)


def main(argv: list[str] | None = None) -> int:
    """@brief 打印分诊表或 JSON。不启动搜索。

    @param argv 命令行；None 时用进程参数。
    @return 成功为 0。
    """
    parser = argparse.ArgumentParser(description="Read-only v2 roofline headroom. Does not use a GPU or a model.")
    parser.add_argument("--target", default="apple-m4-max")
    parser.add_argument("--dtype", default="fp16")
    parser.add_argument("--runs-root", default="runs")
    parser.add_argument("--tasks-root", default=None)
    parser.add_argument("--include-semantic", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    try:
        rows = collect_headroom(
            target_id=args.target,
            dtype=args.dtype,
            runs_root=args.runs_root,
            tasks_root=args.tasks_root,
            include_semantic=bool(args.include_semantic),
        )
    except Exception as exc:  # noqa: BLE001 - CLI boundary
        print(f"task_headroom: {exc}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps([row.document() for row in rows], indent=2))
    else:
        print(format_table(rows))
    return 0


def _with_tier(row: InstanceHeadroom) -> InstanceHeadroom:
    """@brief 按计划里的顺序贴上分诊档。"""
    tier = _tier(row)
    return InstanceHeadroom(
        op_name=row.op_name,
        dtype=row.dtype,
        size=row.size,
        usage=row.usage,
        workspace_declared=row.workspace_declared,
        variable_length_kind=row.variable_length_kind,
        flops=row.flops,
        min_bytes=row.min_bytes,
        arithmetic_intensity=row.arithmetic_intensity,
        bound=row.bound,
        lower_bound_us=row.lower_bound_us,
        best_baseline_us=row.best_baseline_us,
        best_confirmed_speedup=row.best_confirmed_speedup,
        headroom_vs_bound=row.headroom_vs_bound,
        tier=tier,
    )


def _tier(row: InstanceHeadroom) -> str:
    """@brief 分诊规则，先命中者生效。"""
    if row.lower_bound_us is not None and row.lower_bound_us < 3.0:
        return "launch_bound_tiny"
    if row.headroom_vs_bound is not None and row.headroom_vs_bound < 0.98:
        return "algorithm_shift_needed"
    if (
        row.bound == "bandwidth"
        and row.best_baseline_us is not None
        and row.lower_bound_us is not None
        and row.lower_bound_us > 0
        and row.best_baseline_us <= row.lower_bound_us * 1.15
    ):
        return "saturated_bandwidth"
    if _high_value(row):
        return "high_headroom"
    if row.best_baseline_us is None and row.lower_bound_us is not None and row.lower_bound_us >= 3.0:
        return "unmeasured_on_target"
    return "limited_headroom"


def _high_value(row: InstanceHeadroom) -> bool:
    """@brief 余量大，或题目本身是融合、状态递推、变长。"""
    if row.headroom_vs_bound is not None and row.headroom_vs_bound >= 1.25:
        return True
    if row.op_name in HIGH_HEADROOM_OPS:
        return True
    return row.variable_length_kind in ("l1", "l2")


def _sort_key(row: InstanceHeadroom) -> tuple:
    """@brief 高余量在前；同档里有实测余量的排在前面。"""
    headroom = row.headroom_vs_bound
    lower = row.lower_bound_us
    return (
        TIER_RANK.get(row.tier, 9),
        0 if headroom is not None else 1,
        -(headroom or 0.0),
        -(lower or 0.0),
        row.op_name,
        row.size,
    )


def _roofline(flops: int, min_bytes: int, calibration: Mapping[str, Any], dtype: str) -> tuple[float | None, str, float | None]:
    """@brief 用峰值算算术强度、瓶颈和下界微秒。"""
    peak_gbps = calibration.get("peak_gbps")
    peaks = calibration.get("peak_tflops") or {}
    peak_tflops = peaks.get(dtype) if isinstance(peaks, dict) else None
    if not isinstance(peak_gbps, (int, float)) or isinstance(peak_gbps, bool) or peak_gbps <= 0:
        return (flops / min_bytes) if min_bytes else None, "unknown", None
    if min_bytes <= 0 and flops <= 0:
        return None, "unknown", None
    t_mem = min_bytes / (float(peak_gbps) * 1e9) * 1e6
    if isinstance(peak_tflops, (int, float)) and not isinstance(peak_tflops, bool) and peak_tflops > 0 and flops > 0:
        t_cmp = flops / (float(peak_tflops) * 1e12) * 1e6
    else:
        t_cmp = 0.0
    intensity = (flops / min_bytes) if min_bytes else None
    lower = max(t_mem, t_cmp)
    bound = "compute" if t_cmp > t_mem else "bandwidth"
    return intensity, bound, lower


def _calibration(target_id: str) -> dict[str, Any]:
    """@brief 读目标目录里的 calibration.json。"""
    path = REPO / "targets" / target_id / "calibration.json"
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"cannot read calibration for {target_id}: {exc}") from exc
    if not isinstance(document, dict):
        raise ValueError(f"calibration for {target_id} is not an object")
    return document


_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _baseline_index(runs_root: Path, target_id: str) -> dict[tuple[str, str, str], tuple[float | None, float | None]]:
    """@brief 归集 schema-4 报告的最快基线，以及 result.json 里过门的确认加速比。

    ``report.json`` 只贡献 ``baseline_us``。它的字段里没有确认加速比，文件中即使
    出现 ``confirmed_speedup``、``speedup`` 或 ``best_speedup`` 也不采用。
    ``result.json`` 只在 ``confirmed_status=ok``、确认源摘要合法、``confirmed_speedup``
    为有限正数时贡献加速比，不用 ``best_speedup`` 填补。两边都要核对 task、dtype、size、target。
    """
    found: dict[tuple[str, str, str], list[tuple[float | None, float | None]]] = {}
    if not runs_root.is_dir():
        return {}
    for dirpath, dirnames, filenames in os.walk(runs_root, followlinks=False):
        dirnames[:] = sorted(name for name in dirnames if not name.startswith("."))
        directory = Path(dirpath)
        if "report.json" in filenames:
            sample = _report_baseline(directory / "report.json", target_id)
            if sample is not None:
                key, baseline = sample
                found.setdefault(key, []).append((baseline, None))
        if "result.json" in filenames:
            sample = _result_speedup(directory / "result.json", target_id)
            if sample is not None:
                key, speedup = sample
                found.setdefault(key, []).append((None, speedup))
    index: dict[tuple[str, str, str], tuple[float | None, float | None]] = {}
    for key, samples in found.items():
        baselines = [item[0] for item in samples if item[0] is not None]
        speedups = [item[1] for item in samples if item[1] is not None]
        index[key] = (min(baselines) if baselines else None, max(speedups) if speedups else None)
    return index


def _report_baseline(path: Path, target_id: str) -> tuple[tuple[str, str, str], float] | None:
    """@brief 一份当前 schema 报告的最快基线。身份不符或没有有限正基线时返回 None。"""
    document = _read_object(path)
    if document is None or document_state(document) != "current":
        return None
    key = _identity(document, target_id)
    if key is None:
        return None
    baseline = _fastest_baseline(document.get("baseline_us"))
    if baseline is None:
        return None
    return key, baseline


def _result_speedup(path: Path, target_id: str) -> tuple[tuple[str, str, str], float] | None:
    """@brief 一份结果里过了确认门的有限正加速比。缺字段或用 best_speedup 顶上的都不算。"""
    document = _read_object(path)
    if document is None:
        return None
    key = _identity(document, target_id)
    if key is None or document.get("confirmed_status") != "ok":
        return None
    digest = document.get("confirmation_source_sha256")
    if not isinstance(digest, str) or _SHA256.fullmatch(digest) is None:
        return None
    if "confirmed_speedup" not in document:
        return None
    speedup = _positive(document.get("confirmed_speedup"))
    if speedup is None:
        return None
    return key, speedup


def _identity(document: Mapping[str, Any], target_id: str) -> tuple[str, str, str] | None:
    """@brief 取出并核对 task、dtype、size、target。缺一或目标不同就不是这条记录。"""
    task = document.get("task") or document.get("op_name")
    dtype = document.get("dtype")
    size = document.get("size")
    recorded_target = document.get("target_id")
    if recorded_target is None and isinstance(document.get("target"), dict):
        recorded_target = document["target"].get("id")
    if not all(isinstance(item, str) and item for item in (task, dtype, size, recorded_target)):
        return None
    if recorded_target != target_id:
        return None
    return (str(task), str(dtype), str(size))


def _read_object(path: Path) -> dict | None:
    """@brief 读一份 JSON 对象。缺失、损坏或不是对象时返回 None。"""
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return document if isinstance(document, dict) else None


def _fastest_baseline(value: Any) -> float | None:
    """@brief 一份报告里各基线中位数的最小值。"""
    if not isinstance(value, dict):
        return None
    numbers = [number for number in (_positive(item) for item in value.values()) if number is not None]
    if not numbers:
        return None
    return min(numbers)


def _positive(value: Any) -> float | None:
    """@brief 正的有限数。NaN、inf、负数、0 和布尔值都拒绝。"""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        return None
    return number


def _offered_dtypes(spec: dict) -> set[str]:
    """@brief 规格声明的数据类型。"""
    found: list[str] = []
    for value in (spec.get("type_vars") or {}).values():
        if isinstance(value, list):
            found.extend(str(item) for item in value)
    if not found:
        tolerances = (spec.get("numerics") or {}).get("tolerances") or {}
        if isinstance(tolerances, dict):
            found.extend(str(name) for name in tolerances)
    return set(found)


def _whole(value: Any) -> int:
    """@brief 工作量必须是非负整数。"""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"work accounting value {value!r} is not a number")
    number = float(value)
    if number < 0 or number != int(number):
        raise ValueError(f"work accounting value {value!r} is not a non-negative integer")
    return int(number)


def _num(value: float | None) -> str:
    """@brief 表里的空数写成空白。"""
    if value is None:
        return ""
    return f"{value:.4g}"


if __name__ == "__main__":
    raise SystemExit(main())
