"""Search the same operator under several launch contracts, one ordinary search per contract per trial.

The menu lives in this driver, not in ``SearchConfig``. Each child run is a normal ``task`` or ``task@variant`` search with its own budget.
Trials are interleaved (every contract once, then every contract again) so a slow provider hour is not spent entirely on the first contract.

    uv run python -m experiments.contract_menu flash_attention --contracts default,32x16t128,64x32t256 --trials 3 --budget 3 --preset base --max-total-usd 27 --out runs/menu/fa-<stamp>

``default`` is the task's built-in launch (for FlashAttention, ``32x32t128``). A menu without it is refused: there is nothing to compare a variant against.
``--preset`` defaults to ``base``. Omitting it must not fall through to the paper preset (15 USD, patience 12). ``--budget`` replaces only ``budget_usd``.

A finished child (``result.json`` present) is skipped. A child whose whole budget would not fit under ``--max-total-usd`` is skipped and marked ``over_cap``;
money a finished child did not spend is not given to a later child. The summary ranks ``published_speedup`` only. Fewer than two published numbers on any
contract, or overlapping ranges, is ``无结论``.
"""

from __future__ import annotations

import argparse
import importlib
import json
import statistics
import sys
from pathlib import Path

from observe.publish import job_state

from taic.control import write_json_atomic

from tasks import PHASE2_TASK_NAMES, TASK_NAMES, get_task

from .common import conclusion_of_run, experiment_commitment, fits_cap, read_result, run_one, write_cap

REPO_ROOT = Path(__file__).resolve().parent.parent
MENU_FILE = "menu.json"
# Keys that define the experiment. A second invocation of the same --out must repeat them; a different budget or preset would mix two searches.
_LOCKED = ("task", "contracts", "trials", "budget_usd", "preset", "dtype", "size", "max_total_usd")


def resolve_menu(base: str, tokens: list[str]) -> list[tuple[str, str]]:
    """``(directory slug, task name)`` in menu order. ``default`` is the bare task name. A missing default is an error."""
    if not tokens:
        raise ValueError("contract menu is empty")
    seen: list[str] = []
    out: list[tuple[str, str]] = []
    for raw in tokens:
        token = raw.strip()
        if not token:
            continue
        if token in ("default", base):
            slug, task = "default", base
        elif token.startswith(f"{base}@"):
            variant = token.split("@", 1)[1]
            slug, task = variant, token
        elif "@" in token:
            raise ValueError(f"contract {token!r} is not a variant of {base}")
        else:
            slug, task = token, f"{base}@{token}"
        if not slug or slug in seen:
            raise ValueError(f"duplicate or empty contract {token!r}")
        seen.append(slug)
        out.append((slug, task))
    if "default" not in seen:
        raise ValueError("contract menu must include default (the task's built-in launch); refusing to start without it")
    return out


def schedule(items: list[tuple[str, str]], trials: int) -> list[tuple[str, str, int]]:
    """Interleave trials: round 1 of every contract, then round 2, and so on."""
    if trials < 1:
        raise ValueError(f"trials must be >= 1, got {trials}")
    return [(slug, task, trial) for trial in range(1, trials + 1) for slug, task in items]


def _require_buildable(base: str, items: list[tuple[str, str]], dtype: str, size: str) -> None:
    """Refuse before any model call when the task has no variants or a variant string does not build."""
    if base not in TASK_NAMES + PHASE2_TASK_NAMES:
        raise ValueError(f"unknown task {base!r}")
    mod = importlib.import_module(f"tasks.{base}.task")
    if not hasattr(mod, "build_variant"):
        raise ValueError(f"task {base!r} has no contract variants")
    for _slug, task in items:
        get_task(task, dtype, size)


def _num(value) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if value == value else None


def judge(rows: list[dict], slugs: list[str]) -> dict:
    """Name a winner only when every contract has at least two published speedups and one range sits strictly above the others.

    ``best_speedup`` is not read. A contract with a single published number is an anecdote (``个例``) and blocks a winner.
    """
    by: dict[str, list[dict]] = {slug: [] for slug in slugs}
    for row in rows:
        if row.get("skipped") or row["slug"] not in by:
            continue
        by[row["slug"]].append(row)
    published: dict[str, list[float]] = {}
    for slug, items in by.items():
        published[slug] = [v for r in items if (v := _num(r.get("published_speedup"))) is not None]
    anecdotes = [slug for slug, nums in published.items() if len(nums) == 1]
    enough = bool(published) and len(published) >= 2 and all(len(nums) >= 2 for nums in published.values())
    leaders: list[str] = []
    if enough:
        leaders = [
            slug for slug, nums in published.items()
            if all(min(nums) > max(other) for other_slug, other in published.items() if other_slug != slug)
        ]
    winner = leaders[0] if len(leaders) == 1 else None
    if winner:
        verdict, reason = "胜出", f"{winner} 的发布数字范围高于其余契约"
    else:
        bits = []
        if len(published) < 2:
            bits.append("只有一套契约，没有对照")
        elif not all(len(nums) >= 2 for nums in published.values()):
            bits.append("有契约的发布数字少于 2 个")
        else:
            bits.append("发布数字的范围重叠")
        if anecdotes:
            bits.append("个例：" + ", ".join(anecdotes))
        verdict, reason = "无结论", "；".join(bits)
    per = {}
    for slug, nums in published.items():
        per[slug] = {
            "published": nums,
            "published_n": len(nums),
            "median": statistics.median(nums) if nums else None,
            "min": min(nums) if nums else None,
            "max": max(nums) if nums else None,
        }
    return {"verdict": verdict, "winner": winner, "reason": reason, "contracts": per}


def _row_from_result(slug: str, task: str, trial: int, run_dir: Path, res: dict) -> dict:
    conclusion = conclusion_of_run(run_dir, res)
    return {
        "slug": slug, "task": task, "trial": trial, "skipped": None,
        "published_speedup": conclusion["speedup"], "publication": conclusion["publication"],
        "best_speedup": res.get("best_speedup"), "spent_usd": res.get("spent_usd"), "stop_reason": res.get("stop_reason") or "",
    }


def _same(saved: dict, fresh: dict) -> bool:
    for key in _LOCKED:
        old, new = saved.get(key), fresh.get(key)
        if isinstance(old, float) or isinstance(new, float):
            if old is None or new is None or abs(float(old) - float(new)) > 1e-9:
                return False
        elif old != new:
            return False
    return True


def collect(out: Path, items: list[tuple[str, str]], trials: int) -> tuple[list[dict], list[dict]]:
    """Rows already on disk, and the schedule entries that have no ``result.json`` yet."""
    rows: list[dict] = []
    pending: list[tuple[str, str, int]] = []
    for slug, task, trial in schedule(items, trials):
        res = read_result(out / slug / f"trial_{trial}")
        if res is None:
            pending.append((slug, task, trial))
        else:
            rows.append(_row_from_result(slug, task, trial, out / slug / f"trial_{trial}", res))
    return rows, pending


def _write_report(out: Path, items: list[tuple[str, str]], rows: list[dict], skipped: list[dict], aborted: str | None) -> dict:
    judged = judge(rows, [slug for slug, _task in items])
    report = {**judged, "skipped": skipped, "aborted": aborted, "rows": rows}
    write_json_atomic(out / "summary.json", report)
    (out / "summary.md").write_text(markdown(report) + "\n", encoding="utf-8")
    return report


def markdown(report: dict) -> str:
    lines = [
        "| contract | published n | median | min | max |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]

    def cell(v) -> str:
        return "-" if not isinstance(v, (int, float)) or isinstance(v, bool) else f"{v:.3f}"

    for slug, info in report.get("contracts", {}).items():
        lines.append(f"| {slug} | {info['published_n']} | {cell(info['median'])} | {cell(info['min'])} | {cell(info['max'])} |")
    lines.append("")
    lines.append(f"Verdict: {report.get('verdict')} ({report.get('reason')})")
    if report.get("winner"):
        lines.append(f"Winner: {report['winner']}")
    for skip in report.get("skipped") or []:
        lines.append(f"skipped {skip['slug']} trial {skip['trial']}: {skip['reason']}")
    if report.get("aborted"):
        lines.append(f"aborted: {report['aborted']}")
    return "\n".join(lines)


def run_menu(
    base: str,
    tokens: list[str],
    out: Path,
    *,
    trials: int = 3,
    budget: float = 3.0,
    preset: str = "base",
    max_total_usd: float | None = None,
    dtype: str = "fp16",
    size: str = "paper",
    run_fn=None,
    log=print,
) -> dict:
    """Validate the menu, then run every missing child whose budget still fits. ``run_fn`` defaults to :func:`experiments.common.run_one`."""
    if budget <= 0:
        raise ValueError(f"budget must be positive, got {budget}")
    items = resolve_menu(base, tokens)
    _require_buildable(base, items, dtype, size)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    fresh = {
        "task": base,
        "contracts": [slug for slug, _task in items],
        "trials": trials,
        "budget_usd": float(budget),
        "preset": preset,
        "dtype": dtype,
        "size": size,
        "max_total_usd": None if max_total_usd is None else float(max_total_usd),
        "schedule": [{"slug": slug, "task": task, "trial": trial} for slug, task, trial in schedule(items, trials)],
    }
    menu_path = out / MENU_FILE
    if menu_path.exists():
        saved = json_loads(menu_path)
        if not isinstance(saved, dict) or not _same(saved, fresh):
            raise ValueError(f"{menu_path} records a different menu; use a new --out")
    else:
        write_json_atomic(menu_path, fresh)
    write_cap(out, max_total_usd)
    rows, pending = collect(out, items, trials)
    skipped: list[dict] = []
    aborted = None
    runner = run_fn or run_one
    for index, (slug, task, trial) in enumerate(pending):
        rest = [{"slug": s, "task": t, "trial": n} for s, t, n in pending[index + 1 :]]
        job_state(out / "state.json", script="contract_menu", current={"slug": slug, "task": task, "trial": trial}, queue=rest, done=rows)
        rd = out / slug / f"trial_{trial}"
        spent = experiment_commitment(out)
        if not fits_cap(spent, budget, max_total_usd):
            log(f"(skip {slug} trial {trial}: its ${budget:.2f} budget does not fit under --max-total-usd ${max_total_usd:.2f}; the experiment has spent ${spent:.2f})")
            skipped.append({"slug": slug, "task": task, "trial": trial, "reason": "over_cap"})
            _write_report(out, items, rows, skipped, aborted)
            continue
        log(f"\n##### {task} trial {trial}/{trials} budget=${budget:.2f} preset={preset}")
        result = runner(task, dtype, size, rd, budget=budget, preset=preset, log=log)
        res = read_result(rd)
        if res is None:
            raise RuntimeError(f"{rd} produced no result.json")
        row = _row_from_result(slug, task, trial, rd, res)
        rows.append(row)
        stop = getattr(result, "stop_reason", None) or row["stop_reason"]
        if str(stop).startswith(("gpu_hang", "repeated_worker_crashes")):
            aborted = str(stop)
            log("== aborted: GPU hang circuit breaker")
            _write_report(out, items, rows, skipped, aborted)
            job_state(out / "state.json", script="contract_menu", current={"slug": slug, "task": task, "trial": trial}, queue=[], done=rows, phase="aborted")
            return _write_report(out, items, rows, skipped, aborted)
        _write_report(out, items, rows, skipped, aborted)
    job_state(out / "state.json", script="contract_menu", current=None, queue=[], done=rows, phase="done")
    return _write_report(out, items, rows, skipped, aborted)


def json_loads(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Search one operator under several launch contracts.")
    ap.add_argument("task", help="base task name, without @variant")
    ap.add_argument("--contracts", required=True, help="comma-separated: default and variant names (32x16t128 or task@32x16t128)")
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--budget", type=float, default=3.0, help="USD budget of every child search (default: 3)")
    ap.add_argument("--preset", default="base", help="config preset (default: base = taic.toml, not the paper preset)")
    ap.add_argument("--max-total-usd", type=float, default=None)
    ap.add_argument("--dtype", default="fp16")
    ap.add_argument("--size", default="paper")
    ap.add_argument("--out", required=True)
    ap.add_argument("--report", action="store_true", help="rewrite summary.md from results already on disk")
    args = ap.parse_args(argv)
    out = REPO_ROOT / args.out if not Path(args.out).is_absolute() else Path(args.out)
    tokens = [part for part in args.contracts.split(",")]
    try:
        if args.report:
            saved = json_loads(out / MENU_FILE)
            items = resolve_menu(saved["task"], saved["contracts"])
            rows, _pending = collect(out, items, int(saved["trials"]))
            report = _write_report(out, items, rows, [], None)
        else:
            report = run_menu(
                args.task, tokens, out, trials=args.trials, budget=args.budget, preset=args.preset,
                max_total_usd=args.max_total_usd, dtype=args.dtype, size=args.size,
            )
    except ValueError as e:
        print(f"contract menu: {e}", file=sys.stderr)
        return 2
    print("\n" + markdown(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
