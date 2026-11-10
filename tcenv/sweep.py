"""Rank complete candidate files the model already wrote.

The framework does not pick parameters and does not supply a template. Each file
goes through the same timed evaluation the caller injects. Ranking stays here so
another backend can reuse it; the CLI is what binds a task and calls the evaluator.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .selfcheck import append, budget_reason, earlier_hang

# 任务书和自测日志不是候选。隐藏文件同样不算。
SIDECARS = frozenset({"self_checks.jsonl", "NOTES.md", "FEEDBACK.md", "TASK.md", "HARNESS.md", "report.json"})
_HANG = frozenset({"timeout", "crash"})


class SweepError(Exception):
    """@brief 这一次比较不能开始，或必须在下一份文件之前停下。"""


@dataclass
class SweepObservation:
    """@brief 一份候选的计时结果。没测到的数字是 None。"""

    name: str
    status: str
    median_us: float | None
    cv: float | None
    speedup: float | None
    elapsed_s: float
    speedup_se: float | None = None


@dataclass
class SweepResult:
    """@brief 排好名次的结果。``noise_pairs`` 是差距落在噪声里的文件对，按名字排序。"""

    rows: list[SweepObservation]
    noise_pairs: list[tuple[str, str]]
    stopped: str | None = None


def candidate_files(directory: Path) -> list[Path]:
    """@brief 列出目录里要比较的候选文件，按文件名排序。

    @param directory 候选目录。
    @return 普通文件。跳过目录、隐藏文件和任务书/日志。
    """
    files = []
    for path in sorted(Path(directory).iterdir()):
        if path.name.startswith(".") or not path.is_file():
            continue
        if path.name in SIDECARS:
            continue
        files.append(path)
    return files


def _number(value) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _speedup_se(bench: dict) -> float | None:
    """@brief 从报告的 candidate 与最强基线取出加速比标准误。缺 cv 或重复次数时是 None。

    @param bench 报告里的 benchmark 字典。
    @return 相对标准误；四项统计量不齐时是 None。测到的 0 仍是 0，不算缺失。
    """
    cand = bench.get("candidate")
    bases = bench.get("baselines")
    name = bench.get("best_baseline")
    if not isinstance(cand, dict) or not isinstance(bases, dict) or not isinstance(name, str) or name not in bases:
        return None
    base = bases.get(name)
    if not isinstance(base, dict):
        return None
    values = (cand.get("cv"), cand.get("reps"), base.get("cv"), base.get("reps"))
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in values):
        return None
    from tcenv.bench import approximate_speedup_se

    return float(approximate_speedup_se(*values))


def observation_from_report(name: str, report, elapsed_s: float) -> SweepObservation:
    """@brief 从一份评测报告取出比较要用的数字。

    @param name 文件名。
    @param report ``EvalReport``。
    @param elapsed_s 这次评测的墙钟秒数，记进自测配额。
    @return 一行观察。
    """
    bench = report.benchmark if isinstance(getattr(report, "benchmark", None), dict) else {}
    cand = bench.get("candidate") if isinstance(bench.get("candidate"), dict) else {}
    median = _number(cand.get("median_us"))
    if median is None:
        median = _number(getattr(report, "candidate_us", None))
    return SweepObservation(
        name=name,
        status=str(getattr(report, "status", "") or ""),
        median_us=median,
        cv=_number(cand.get("cv")),
        speedup=_number(getattr(report, "speedup", None)),
        elapsed_s=float(elapsed_s),
        speedup_se=_speedup_se(bench),
    )


def _rank(rows: list[SweepObservation]) -> list[SweepObservation]:
    ranked = []
    rest = []
    for row in rows:
        if row.status == "ok" and _number(row.speedup) is not None:
            ranked.append(row)
        else:
            rest.append(row)
    ranked.sort(key=lambda row: (-float(row.speedup), row.name))
    return ranked + rest


def within_noise(
    a: SweepObservation, b: SweepObservation, noise_z: float = 2.0, min_improvement: float = 1.005,
) -> bool | None:
    """@brief 两份合格加速比的差距是否落在本场噪声门里。

    用加速比和两边的加速比标准误，门槛与搜索的 ``required_gain`` 相同。缺加速比或标准误时返回 None，
    不把缺失当成零误差。

    @param a 一行观察。
    @param b 另一行。
    @param noise_z 噪声倍数，默认 2。
    @param min_improvement 最小提速倍数，默认 1.005。
    @return 落在噪声里为 True，已经分开为 False，统计量不齐为 None。
    """
    if a.status != "ok" or b.status != "ok":
        return False
    sp_a, sp_b = _number(a.speedup), _number(b.speedup)
    if sp_a is None or sp_b is None or sp_a <= 0 or sp_b <= 0:
        return None
    if a.speedup_se is None or b.speedup_se is None:
        return None
    from taic.policy import required_gain

    gain = required_gain(min_improvement, noise_z, a.speedup_se, b.speedup_se)
    faster, slower = (sp_a, sp_b) if sp_a >= sp_b else (sp_b, sp_a)
    if faster == slower:
        return True
    return faster <= slower * gain


def _noise_pairs(rows: list[SweepObservation], noise_z: float, min_improvement: float) -> list[tuple[str, str]]:
    pairs = []
    for i, left in enumerate(rows):
        for right in rows[i + 1:]:
            if within_noise(left, right, noise_z, min_improvement) is True:
                pairs.append(tuple(sorted((left.name, right.name))))
    return pairs


def run_sweep(
    paths, evaluate, *, directory: Path, noise_z: float = 2.0, min_improvement: float = 1.005, limit: int = 16,
) -> SweepResult:
    """@brief 逐份评测并排名。超时或崩溃记下一行后立刻停下。

    ``evaluate(path)`` 返回 ``SweepObservation``。本函数在评测前查自测配额，评测后写一行
    ``self_checks.jsonl``。调用方不要再写第二行。已经评完的行在下一份触发配额时保留，
    ``stopped`` 记为 ``quota``。一份都还没评时仍抛 ``SweepError``。

    @param paths 候选文件。超过 ``limit`` 时一份都不评。
    @param evaluate 注入的评测函数。
    @param directory 自测配额目录。和候选文件所在目录分开；命令行传入本场 ``.selfcheck``。
    @param noise_z 噪声倍数。
    @param min_improvement 最小提速倍数，和本场搜索配置相同。
    @param limit 最多多少个文件，默认 16。
    @return 排名结果。配额中途停下时 ``stopped`` 为 ``quota``，已完成的行仍在。
    @raises SweepError 文件太多，或第一份之前配额已经拒绝计时。
    """
    paths = list(paths)
    if len(paths) > limit:
        raise SweepError(f"sweep accepts at most {limit} candidate files")
    directory = Path(directory)
    rows: list[SweepObservation] = []
    stopped = None
    for path in paths:
        blocked = earlier_hang(directory) or budget_reason(directory)
        if blocked:
            if rows:
                stopped = "quota"
                break
            raise SweepError(blocked)
        observed = evaluate(path)
        if not isinstance(observed, SweepObservation):
            raise SweepError("sweep evaluate() must return a SweepObservation")
        rows.append(observed)
        append(directory, {
            "status": observed.status,
            "candidate_us": observed.median_us,
            "cv": observed.cv,
            "speedup": observed.speedup,
            "elapsed_s": observed.elapsed_s,
            "name": observed.name,
        })
        if observed.status in _HANG:
            stopped = observed.status
            break
    ranked = _rank(rows)
    return SweepResult(ranked, _noise_pairs(ranked, noise_z, min_improvement), stopped)


def format_table(result: SweepResult) -> str:
    """@brief 把排名打成一张表。噪声对写在对应行上。

    @param result ``run_sweep`` 的结果。
    @return 以换行结尾的文本。
    """
    partners: dict[str, list[str]] = {}
    for left, right in result.noise_pairs:
        partners.setdefault(left, []).append(right)
        partners.setdefault(right, []).append(left)
    lines = ["file\tstatus\tmedian_us\tcv\tspeedup\trank\twithin_noise"]
    rank = 0
    for row in result.rows:
        if row.status == "ok" and _number(row.speedup) is not None:
            rank += 1
            rank_cell = str(rank)
        else:
            rank_cell = "-"
        median = "" if row.median_us is None else f"{row.median_us:.2f}"
        cv = "" if row.cv is None else f"{row.cv:.3f}"
        speed = "" if row.speedup is None else f"{row.speedup:.3f}"
        if row.status == "ok" and _number(row.speedup) is not None and row.speedup_se is None:
            noise = "unknown"
        else:
            noise = ",".join(partners.get(row.name, []))
        lines.append("\t".join([row.name, row.status, median, cv, speed, rank_cell, noise]))
    if result.stopped:
        lines.append(f"# stopped after {result.stopped}")
    return "\n".join(lines) + "\n"
