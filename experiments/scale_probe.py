"""Measure the observe index and retrieval paths on synthetic runs.

The 10^5 extrapolation in the data-layer review had not been measured. This script creates
minimal run directories in a temporary root, indexes them with the real ``observe.index.Index``
and times the request paths. It never touches ``runs/``, the GPU or the model.

    uv run python -m experiments.scale_probe --counts 1000,10000,100000
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
import time
from pathlib import Path

_STATE = json.dumps({
    "task": "relu", "dtype": "fp16", "size": "small", "phase": "done",
    "started_at": 1, "updated_at": 1, "spent_usd": 1.0, "budget_usd": 5.0,
    "model": "model-a", "best_speedup": 1.0,
}).encode()
_RESULT = json.dumps({
    "task": "relu", "dtype": "fp16", "size": "small", "stop_reason": "max_rounds",
    "spent_usd": 1.0, "budget_usd": 5.0, "started_at": 1, "best_speedup": 1.0,
    "confirmed_speedup": 1.0, "confirmed_status": "ok", "publication": "confirmed",
    "published_speedup": 1.0, "protocol_version": "2",
}).encode()


_KERNEL = "kernel void demo() {}\n"
_CONFIRMED = 20


def _confirm(directory: Path, speedup: float) -> None:
    """@brief 给一场合成运行补上能通过发布门的确认材料。

    没有这些文件，``best_confirmed()`` 的 ``best_rows`` 永远是 0，量到的不是定向确认路径。

    @param directory 运行目录。
    @param speedup 确认加速比。
    """
    import hashlib

    from tcenv.schema import PROTOCOL_VERSION

    digest = hashlib.sha256(_KERNEL.encode()).hexdigest()
    (directory / "best.metal").write_text(_KERNEL, encoding="utf-8")
    candidate = directory / "candidates" / "000_r1.p0"
    candidate.mkdir(parents=True)
    (candidate / "kernel.metal").write_text(_KERNEL, encoding="utf-8")
    # result.json 是和模板硬链接的：先摘链接，否则会改写所有 run 共享的 inode。
    (directory / "result.json").unlink()
    (directory / "result.json").write_text(json.dumps({
        "task": "relu", "dtype": "fp16", "size": "small", "stop_reason": "max_rounds",
        "started_at": 1, "spent_usd": 1.0, "target_id": "apple-m4-max",
        "confirmed_speedup": speedup, "confirmed_status": "ok", "confirm_winner_label": "r1.p0",
        "confirmation_source_sha256": digest, "confirmed_values": [speedup],
        "confirmed_worst": speedup, "confirmed_best": speedup,
    }), encoding="utf-8")
    (directory / "confirm_report.json").write_text(json.dumps({
        "status": "ok", "speedup": speedup,
        "benchmark": {"noisy": False, "candidate": {"cv": 0.01, "reps": 15},
                      "baselines": {"mlx": {"cv": 0.01, "reps": 15}}, "best_baseline": "mlx",
                      "confirmation": {"kernel_sha256": digest, "winner_label": "r1.p0",
                                       "confirmed_values": [speedup], "speedup": speedup,
                                       "candidate_us": None}},
        "protocol_version": PROTOCOL_VERSION, "target_id": "apple-m4-max",
    }), encoding="utf-8")


def _make_root(root: Path, count: int) -> None:
    """@brief 在临时根目录下硬链接出 ``count`` 个最小运行目录；前 20 场带真实确认材料。

    @param root 放置运行目录的根。
    @param count 要造的运行个数。
    """
    root.mkdir(parents=True, exist_ok=True)
    state = root / ".state.template"
    result = root / ".result.template"
    state.write_bytes(_STATE)
    result.write_bytes(_RESULT)
    width = max(6, len(str(count)))
    for index in range(count):
        directory = root / f"run_{index:0{width}d}"
        directory.mkdir()
        os.link(state, directory / "state.json")
        os.link(result, directory / "result.json")
        if index < min(_CONFIRMED, count):
            _confirm(directory, 1.0 + index / 1000)
    state.unlink()
    result.unlink()


def _reset_singletons() -> None:
    """@brief 清掉 observe 的进程级单例和缓存，让下一次计时从冷开始。
    """
    import observe.index as index
    import observe.query as query

    index._SINGLE = None
    query._HISTORY["key"] = None
    query._HISTORY["rows"] = []
    query._FEATURE_CACHE.clear()
    query._TICK["at"] = 0.0


def measure(count: int) -> dict:
    """@brief 造 ``count`` 个运行，索引后测量各读取路径的耗时。

    @param count 合成运行的个数。
    @return 各项耗时、返回行数、库体积等字段组成的字典。
    """
    from observe import query
    from observe.budget import summarize
    from observe.index import Index, get_index
    from taic.brief import summarize_run

    root = Path(tempfile.mkdtemp(prefix=f"taic-scale-{count}-"))
    db = root / ".index.sqlite"
    try:
        t0 = time.perf_counter()
        _make_root(root / "runs", count)
        create_s = time.perf_counter() - t0
        os.environ["AI_LOWERLING_RUNS_DIR"] = str(root / "runs")
        os.environ["OBSERVE_DB"] = str(db)
        os.environ["AI_LOWERLING_BOOKS"] = str(root / "none.sqlite")
        os.environ["OBSERVE_TICK_INTERVAL"] = "0"
        _reset_singletons()
        t0 = time.perf_counter()
        index = Index(db)
        index.sync_known()
        sync_s = time.perf_counter() - t0
        rows = index.conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0]

        # 热对账必须与冷建分开报：目录内容没变，但 sync_known 仍要逐 run 比指纹。
        t0 = time.perf_counter()
        index.sync_known()
        sync_warm_s = time.perf_counter() - t0

        # 纯 stat 对账：不读任何文件内容，只量目录与关键文件的 stat 成本。
        t0 = time.perf_counter()
        stat_files = 0
        for directory in (root / "runs").iterdir():
            for name in ("state.json", "result.json"):
                (directory / name).stat()
                stat_files += 1
        stat_only_s = time.perf_counter() - t0

        # 预算横幅：旧实现解码全部 row_json（``summarize(history())``）vs 新窄投影（``budget_report()``）。
        # 打开库的代价不属于“预算读路径”：两次都先取到已打开的库再计时。
        _reset_singletons()
        get_index()
        t0 = time.perf_counter()
        summarize(query.history())
        budget_history_s = time.perf_counter() - t0
        _reset_singletons()
        get_index()
        t0 = time.perf_counter()
        budget = query.budget_report()
        budget_s = time.perf_counter() - t0

        _reset_singletons()
        t0 = time.perf_counter()
        overview = query.overview()
        overview_s = time.perf_counter() - t0

        _reset_singletons()
        t0 = time.perf_counter()
        history = query.history()
        history_s = time.perf_counter() - t0

        _reset_singletons()
        t0 = time.perf_counter()
        index.tick()
        tick_s = time.perf_counter() - t0

        _reset_singletons()
        t0 = time.perf_counter()
        first = query.runs_page(limit=50)
        page_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        second = query.runs_page(limit=50, cursor=first.get("next"))
        page2_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        by_speed = query.runs_page(limit=50, sort="speedup")
        speed_page_s = time.perf_counter() - t0

        # 冷进程模拟：清掉单例与缓存后第一次 /best（确认核对的 stat 阶段从冷开始）。
        _reset_singletons()
        t0 = time.perf_counter()
        best_cold = query.best_confirmed(limit=20)
        best_cold_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        best = query.best_confirmed(limit=20)
        best_s = time.perf_counter() - t0

        read_sync_calls = 0
        watched_scans = 0
        original_sync_known = Index.sync_known
        original_refresh_watched = Index._refresh_watched

        def counted_sync_known(self):
            nonlocal read_sync_calls
            read_sync_calls += 1
            return original_sync_known(self)

        def counted_refresh_watched(self):
            nonlocal watched_scans
            watched_scans += 1
            return original_refresh_watched(self)

        Index.sync_known = counted_sync_known
        Index._refresh_watched = counted_refresh_watched
        timings = {}
        try:
            for name, read in (
                ("experiment_jobs", query.experiment_jobs),
                ("compare_missing", lambda: query.compare("no-such-experiment")),
                ("ledger_report", query.ledger_report),
            ):
                # cold：强制触发 watched mtime 全扫描，模拟 10 秒窗口的第一个请求。
                get_index()._last_full = 0.0
                t0 = time.perf_counter()
                read()
                timings[f"{name}_cold_s"] = time.perf_counter() - t0
                # warm：紧接 cold 的窗口内请求，不含 watched 全扫描。
                get_index()._last_full = time.monotonic()
                t0 = time.perf_counter()
                read()
                timings[f"{name}_warm_s"] = time.perf_counter() - t0
        finally:
            Index.sync_known = original_sync_known
            Index._refresh_watched = original_refresh_watched
        assert watched_scans == 3, f"expected one watched scan per cold read, got {watched_scans}"
        assert read_sync_calls == 0, f"request reads called sync_known {read_sync_calls} times"

        t0 = time.perf_counter()
        brief = summarize_run(root / "runs" / "run_000000")
        brief_s = time.perf_counter() - t0
        return {
            "count": count, "indexed": rows, "create_s": round(create_s, 3),
            "sync_s": round(sync_s, 3), "sync_warm_s": round(sync_warm_s, 3),
            "stat_only_s": round(stat_only_s, 3), "stat_files": stat_files,
            "budget_s": round(budget_s, 3), "budget_history_s": round(budget_history_s, 3),
            "budget_flagged": len(budget.get("flagged") or []), "overview_s": round(overview_s, 3),
            "history_s": round(history_s, 3), "tick_s": round(tick_s, 3),
            "page_s": round(page_s, 3), "page2_s": round(page2_s, 3), "speed_page_s": round(speed_page_s, 3),
            "best_s": round(best_s, 3), "best_cold_s": round(best_cold_s, 3),
            "best_rows": len(best.get("best") or []), "brief_s": round(brief_s, 3),
            **{name: round(value, 3) for name, value in timings.items()},
            # 兼容旧的 warm 字段名；cold 字段单独给出，不能混报。
            "experiment_jobs_s": round(timings["experiment_jobs_warm_s"], 3),
            "compare_missing_s": round(timings["compare_missing_warm_s"], 3),
            "ledger_report_s": round(timings["ledger_report_warm_s"], 3),
            "read_sync_calls": read_sync_calls,
            "cold_watched_scans": watched_scans,
            "history_rows": len(history), "page_rows": len(first.get("runs") or []),
            "page2_rows": len(second.get("runs") or []), "speed_page_rows": len(by_speed.get("runs") or []), "best_rows": len(best.get("best") or []),
            "brief_bytes": len(json.dumps(brief, ensure_ascii=False)),
            "db_bytes": db.stat().st_size if db.exists() else 0,
        }
    finally:
        shutil.rmtree(root, ignore_errors=True)


def main(argv: list[str] | None = None) -> int:
    """@brief ``python -m experiments.scale_probe`` 的入口，按给定规模逐个测。

    @param argv 命令行参数；为 ``None`` 时取 ``sys.argv``。
    @return 进程退出码，正常结束时为 0。
    """
    parser = argparse.ArgumentParser(prog="python -m experiments.scale_probe")
    parser.add_argument("--counts", default="1000,10000,100000")
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)
    counts = [int(item) for item in args.counts.split(",") if item.strip()]
    rows = []
    for count in counts:
        row = measure(count)
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
    if args.out:
        Path(args.out).write_text(json.dumps(rows, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
