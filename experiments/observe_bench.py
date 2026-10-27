"""Print the observe plan's synthetic timings. This is not a test and not a gate.

The hot-cache index build is the number the plan caps at 5 seconds: the files are still in the
page cache because this process just wrote them. A cold file cache is reported only when ``purge``
succeeds. The lines are printed either way. Nothing here starts a kernel or calls a model.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path


def _write_runs(root: Path, count: int) -> None:
    """@brief 造出 ``count`` 个已结束的合成运行目录，供计时用。

    每个运行带 ``state.json``、``result.json`` 与 50 个候选文件。

    @param root 放置运行目录的根。
    @param count 要造的运行个数。
    """
    root.mkdir(parents=True)
    state = {"task": "relu", "dtype": "fp16", "phase": "done", "pid": 1, "model": "bench"}
    result = {"task": "relu", "dtype": "fp16", "stop_reason": "max_rounds", "spent_usd": 0.01, "budget_usd": 1}
    for index in range(count):
        directory = root / f"run_{index:04d}"
        directory.mkdir()
        state["started_at"] = index
        state["updated_at"] = index
        (directory / "state.json").write_text(json.dumps(state))
        (directory / "result.json").write_text(json.dumps(result))
        candidates = directory / "candidates"
        candidates.mkdir()
        for cand in range(50):
            (candidates / f"{cand}.json").write_text("{}")


def _write_wide(root: Path) -> None:
    """@brief 造一个带 400 个候选、带完工锁定标记的宽运行，用来量详情页。

    @param root 放置该运行目录的根。
    """
    directory = root / "wide"
    directory.mkdir()
    (directory / "state.json").write_text(json.dumps({
        "task": "gemm", "dtype": "fp16", "phase": "done", "started_at": 1, "updated_at": 1,
        "committee": "first", "round": 1, "model": "bench",
    }))
    (directory / "result.json").write_text(json.dumps({
        "task": "gemm", "dtype": "fp16", "stop_reason": "max_rounds", "spent_usd": 0.1, "budget_usd": 1,
        "confirm_winner_label": "r1.p0",
    }))
    (directory / "run_config.json").write_text(json.dumps({
        "task": "gemm", "dtype": "fp16", "size": "paper",
        "search": {"plan_ledger": False},
        "committees": [{"name": "first", "members": [{"model": "bench"}]}],
    }))
    candidates = directory / "candidates"
    candidates.mkdir()
    kernel = "kernel void k() {}\n"
    for cand in range(400):
        path = candidates / f"{cand:03d}_r1.p{cand}"
        path.mkdir()
        (path / "kernel.metal").write_text(kernel)
        (path / "report.json").write_text(json.dumps({"speedup": 1.0, "status": "ok"}))


def _write_feedback_run(root: Path) -> None:
    """@brief 造一个 11 候选、80 条反馈记录的运行，用来量反馈密集型详情。

    生产里详情最大的一块是 `feedback`（plan 实测占 68%），只量 400 候选会漏掉它。

    @param root 放置运行目录的根。
    """
    directory = root / "feedback_demo"
    directory.mkdir()
    (directory / "state.json").write_text(json.dumps({
        "task": "gemm", "dtype": "fp16", "phase": "done", "started_at": 1, "updated_at": 1,
        "committee": "first", "round": 1, "model": "bench",
    }))
    (directory / "result.json").write_text(json.dumps({
        "task": "gemm", "dtype": "fp16", "stop_reason": "max_rounds", "spent_usd": 0.1, "budget_usd": 1,
    }))
    (directory / "run_config.json").write_text(json.dumps({
        "task": "gemm", "dtype": "fp16", "size": "paper",
        "search": {"plan_ledger": False},
        "committees": [{"name": "first", "members": [{"model": "bench"}]}],
    }))
    candidates = directory / "candidates"
    candidates.mkdir()
    for cand in range(11):
        path = candidates / f"{cand:03d}_r1.p{cand}"
        path.mkdir()
        (path / "kernel.metal").write_text("kernel void k() {}\n")
        (path / "report.json").write_text(json.dumps({"speedup": 1.0, "status": "ok"}))
    records = []
    for seq in range(80):
        records.append(json.dumps({
            "format": "agent_feedback", "seq": seq + 1, "t": seq,
            "label": f"r{seq % 11}.p0", "round": seq % 11, "committee": "first",
            "items": [{
                "digest": f"d{seq:04d}", "section": "对系统的建议", "audience": "system", "premise": "",
                "text": f"建议 {seq}：" + "x" * 180,
            }],
        }))
    (directory / "feedback.jsonl").write_text("\n".join(records) + "\n", encoding="utf-8")


def _purge() -> str:
    """@brief 尝试清掉系统文件缓存，供冷缓存计时用。

    @return 成功时为 ``purge``；不可用时为带原因的 ``unavailable (...)``。
    """
    try:
        result = subprocess.run(["purge"], capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return f"unavailable ({exc})"
    if result.returncode == 0:
        return "purge"
    detail = (result.stderr or result.stdout or str(result.returncode)).strip()
    return f"unavailable ({detail})"


def _line(name: str, seconds: float, limit: float | None) -> None:
    """@brief 打出一行耗时，并标出是否超出计划上限。

    @param name 这一项的说明。
    @param seconds 实测秒数。
    @param limit 计划上限；为 ``None`` 时不标注。
    """
    mark = ""
    if limit is not None:
        mark = "  within the plan" if seconds < limit else "  over the plan"
    print(f"{name}: {seconds * 1000:.1f} ms{mark}")


def main() -> None:
    """@brief 在临时目录里造数据并打出 observe 各项合成计时。

    这不是测试也不是门禁，无论是否超限都会打印。
    """
    root = Path(tempfile.mkdtemp(prefix="observe-bench-"))
    runs = root / "runs"
    os.environ["AI_LOWERLING_RUNS_DIR"] = str(runs)
    os.environ["OBSERVE_DB"] = str(root / "observe.sqlite")
    os.environ["OBSERVE_CARRYOVER"] = str(root / "missing-carryover.json")
    os.environ["AI_LOWERLING_BOOKS"] = str(root / "missing-books.sqlite")
    os.environ["AI_LOWERLING_OBSERVE_DIR"] = str(root / "observe-cache")
    try:
        started = time.perf_counter()
        _write_runs(runs, 1000)
        print(f"write 1000 runs x 50 candidate files: {time.perf_counter() - started:.3f}s")

        from observe.index import get_index
        from observe.query import live_files, run_detail, runs_page

        index = get_index()
        started = time.perf_counter()
        index.tick()
        _line("hot-cache cold index (1000 finished runs)", time.perf_counter() - started, 5.0)

        started = time.perf_counter()
        index.tick()
        _line("idle frame", time.perf_counter() - started, 0.050)
        print(f"idle frame json_reads={index.stats['json_reads']} dir_lists={index.stats['dir_lists']}")

        started = time.perf_counter()
        page = runs_page()
        _line("/api/v2/runs default page", time.perf_counter() - started, 0.100)
        print(f"default page rows={len(page['runs'])} total={page['total']}")
        print(f"idle live-state files={len(live_files())}")

        _write_wide(runs)
        started = time.perf_counter()
        detail = run_detail("wide")
        _line("detail with 400 candidates and the feature grid", time.perf_counter() - started, 1.0)
        print(f"detail candidates={len(detail['candidates'])} grid={len(detail['feature_grid'])}")

        # 新增的测量档位（plan S0）：反馈密集详情、预算/总览、best 冷热。
        import observe.query as query_mod

        query_mod.detail.gpu_snapshot = lambda: {"holders": [], "waiters": [], "events": []}
        query_mod.detail.live_evals = lambda: []
        _write_feedback_run(runs)
        started = time.perf_counter()
        feedback_detail = run_detail("feedback_demo")
        _line("detail with 80 feedback records", time.perf_counter() - started, 1.0)
        print(f"feedback bytes={len(json.dumps(feedback_detail.get('feedback') or {}))} "
              f"candidates bytes={len(json.dumps(feedback_detail.get('candidates') or []))} "
              f"total bytes={len(json.dumps(feedback_detail, default=str))}")

        started = time.perf_counter()
        budget = query_mod.budget_report()
        _line("/api/v2/budget (1000 runs)", time.perf_counter() - started, 0.200)
        started = time.perf_counter()
        overview = query_mod.overview()
        _line("/api/v2/overview (1000 runs)", time.perf_counter() - started, 0.200)
        live = query_mod.live_overview()
        print(f"budget bytes={len(json.dumps(budget, default=str))} "
              f"overview bytes={len(json.dumps(overview, default=str))} "
              f"live.gpu bytes={len(json.dumps(live.get('gpu') or {}, default=str))} "
              f"flagged={len(budget.get('flagged') or [])}")

        started = time.perf_counter()
        best_cold = query_mod.best_confirmed()
        _line("/api/v2/best cold", time.perf_counter() - started, 0.200)
        started = time.perf_counter()
        query_mod.best_confirmed()
        _line("/api/v2/best warm", time.perf_counter() - started, 0.200)
        print(f"best rows={len(best_cold.get('best') or [])} "
              "(合成目录未过确认门时为 0，量的是这条路径本身)")
        print("note: 1000 个合成目录只有 candidates/*.json，没有 candidates/*/report.json，"
              "索引 candidates 行数为 0（无候选语料）")

        index.conn.close()
        import observe.index as index_mod

        index_mod._SINGLE = None
        for suffix in ("", "-wal", "-shm"):
            path = Path(str(root / "observe.sqlite") + suffix) if suffix else root / "observe.sqlite"
            path.unlink(missing_ok=True)
        how = _purge()
        print(f"cold file cache: {how}")
        if how == "purge":
            started = time.perf_counter()
            get_index().tick()
            print(f"cold-cache index build: {(time.perf_counter() - started) * 1000:.1f} ms  (not part of the 5s promise)")
    finally:
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    main()
