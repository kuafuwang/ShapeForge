"""Committee comparison matrix. The default is a dry run: it prints the arms and the nominal dollars and starts nothing.

    uv run python -m experiments.committee_ab --dry-run --n 3 --budget 3
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .manifest import ensure


ARMS = (
    {"id": "A", "committees": "first and second both deepseek-flash", "plan_effort": None, "patch_effort": None},
    {"id": "B", "committees": "first and second both deepseek-flash", "plan_effort": "high", "patch_effort": "low"},
    {"id": "C", "committees": "first high, second low", "plan_effort": None, "patch_effort": None},
)


def nominal_usd(n: int, budget: float) -> float:
    """@brief 算全部实验组的名义花费上限。

    @param n 每组重复次数。
    @param budget 每次运行的花费上限，单位美元。
    @return ``n × 组数 × budget``。
    """
    return n * len(ARMS) * budget


def declare(out: Path, *, n: int, budget: float, max_total_usd: float | None) -> dict:
    """@brief 为委员会对比写实验清单。

    Write the manifest for a committee comparison. This does not start a search.

    @param out 实验目录，``experiment.json`` 写在这里。
    @param n 每组重复次数。
    @param budget 每次运行的花费上限，单位美元。
    @param max_total_usd 总预算上限；为 ``None`` 时表示不设限。
    @return 写好的实验清单。
    """
    return ensure(
        Path(out),
        kind="committee_ab",
        driver="committee_ab",
        frozen={
            "trials_per_arm": n,
            "budget_usd": budget,
            "max_total_usd": max_total_usd,
            "task": None,
            "dtype": None,
            "size": None,
            "preset": None,
            "model": None,
            "committees": None,
        },
        arms=[
            {
                "arm_id": arm["id"],
                "planned_n": n,
                "delta": {
                    "committees": arm["committees"],
                    "plan_effort": arm["plan_effort"],
                    "patch_effort": arm["patch_effort"],
                },
            }
            for arm in ARMS
        ],
    )


def matrix(*, n: int, budget: float, max_total_usd: float | None) -> dict:
    """@brief 生成实验组矩阵和名义花费，只描述、不启动。

    @param n 每组重复次数。
    @param budget 每次运行的花费上限，单位美元。
    @param max_total_usd 总预算上限；为 ``None`` 时视为放得下。
    @return 含 ``arms``、``nominal_usd``、``fits`` 等字段的字典。
    """
    total = nominal_usd(n, budget)
    return {
        "dry_run": True,
        "arms": list(ARMS),
        "trials_per_arm": n,
        "budget_usd": budget,
        "nominal_usd": total,
        "fits": max_total_usd is None or total <= max_total_usd,
        "note": "This driver does not call a model. A paid run needs a separate approval.",
    }


def main(argv=None) -> int:
    """@brief ``python -m experiments.committee_ab`` 的入口，默认干跑并打印矩阵。

    @param argv 命令行参数；为 ``None`` 时取 ``sys.argv``。
    @return 进程退出码；要求 ``--execute`` 时返回 2 并拒绝执行。
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", default=True)
    ap.add_argument("--execute", action="store_true", help="refused: a paid committee comparison needs a separate approval")
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--budget", type=float, default=3.0)
    ap.add_argument("--max-total-usd", type=float, default=None)
    ap.add_argument("--out", default=None, help="write experiment.json here and still do not start a search")
    a = ap.parse_args(argv)
    if a.execute:
        print("refused: experiments.committee_ab does not start a paid search")
        return 2
    if a.out:
        declare(Path(a.out), n=a.n, budget=a.budget, max_total_usd=a.max_total_usd)
    print(json.dumps(matrix(n=a.n, budget=a.budget, max_total_usd=a.max_total_usd), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
