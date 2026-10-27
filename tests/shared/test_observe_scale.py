"""Scale checks for ``/best``: 100k runs, a small non-zero confirmed set, cold/warm/periodic.

Marked ``slow``: building 100k synthetic directories takes tens of seconds. Run explicitly with
``uv run pytest -q -m slow tests/shared/test_observe_scale.py``. No GPU and no model call.
"""

from __future__ import annotations

import time

import pytest

from experiments.scale_probe import _make_root, _reset_singletons
from observe import query
from observe.index import get_index

BUDGET_MS = 50.0


def _use_root(tmp_path, monkeypatch, count: int):
    """@brief 建 100k 合成运行并把观察路径指向临时目录。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 的替换工具。
    @param count 运行个数。
    @return runs 根目录。
    """
    runs = tmp_path / "runs"
    _make_root(runs, count)
    monkeypatch.setenv("AI_LOWERLING_RUNS_DIR", str(runs))
    monkeypatch.setenv("OBSERVE_DB", str(tmp_path / "observe.sqlite"))
    monkeypatch.setenv("AI_LOWERLING_BOOKS", str(tmp_path / "books.sqlite"))
    monkeypatch.setenv("OBSERVE_CARRYOVER", str(tmp_path / "carry.json"))
    monkeypatch.setenv("OBSERVE_TICK_INTERVAL", "0")
    monkeypatch.setenv("OBSERVE_FULL_TICK_INTERVAL", "3600")
    monkeypatch.setenv("AI_LOWERLING_OBSERVE_DIR", str(tmp_path / "cache"))
    return runs


def _best_ms() -> float:
    """@brief 计时一次 ``best_confirmed()``。

    @return 毫秒。
    """
    start = time.perf_counter()
    query.best_confirmed(limit=20)
    return (time.perf_counter() - start) * 1000


@pytest.mark.slow
def test_best_is_targeted_on_100k_runs_with_a_small_confirmed_set(tmp_path, monkeypatch):
    """@brief 100k 场里 20 场真能确认：``/best`` 冷/热/周期对账后都 ≤50ms，且不扫全部 watched。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 的替换工具。
    """
    _use_root(tmp_path, monkeypatch, 100_000)
    index = get_index()
    index.sync_known()
    rows = index.conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
    confirmed = index.conn.execute("SELECT COUNT(*) FROM runs WHERE publication = 'confirmed'").fetchone()[0]
    assert rows == 100_000
    assert confirmed == 20, f"合成 fixture 的确认集必须非零且很小，实际 {confirmed}"

    # 冷进程：清掉单例与缓存，第一次 /best 要自己打开库并现算确认指纹。
    _reset_singletons()
    opened = time.perf_counter()
    get_index()
    open_ms = (time.perf_counter() - opened) * 1000
    cold = _best_ms()
    warm = [_best_ms() for _ in range(5)]

    # 新 run 的发现只 stat “可能装着 run 的父目录”，不做 O(全部 watched) 的周期扫描。
    scans = {"n": 0}
    real_scan = type(get_index())._refresh_watched

    def counting_scan(self):
        scans["n"] += 1
        return real_scan(self)

    monkeypatch.setattr(type(get_index()), "_refresh_watched", counting_scan)
    after_tick = _best_ms()
    assert scans["n"] == 0, "/best 不应做 O(全部 watched) 的周期扫描"

    # 人为把“周期全量扫描”跑一遍，再测 /best（模拟另一个入口刚做过对账）。
    get_index()._last_full = 0.0
    monkeypatch.setenv("OBSERVE_FULL_TICK_INTERVAL", "0")
    get_index().tick()
    after_full = _best_ms()

    payload = query.best_confirmed(limit=20)
    assert len(payload["best"]) == 20
    measured = {"runs": rows, "confirmed": confirmed, "open_ms": round(open_ms, 2),
                "cold_ms": round(cold, 2), "warm_ms": [round(item, 2) for item in warm],
                "after_tick_ms": round(after_tick, 2), "after_full_ms": round(after_full, 2)}
    print("BEST_SCALE", measured)
    for name, value in (("cold", cold), ("warm", max(warm)), ("after-tick", after_tick), ("after-full", after_full)):
        assert value <= BUDGET_MS, f"/best {name} {value:.1f}ms 超过 {BUDGET_MS}ms"


@pytest.mark.slow
def test_the_first_page_and_the_thousandth_page_take_within_two_times(tmp_path, monkeypatch):
    """@brief 100k 场：keyset 首页与第 1000 页的耗时差 ≤2×，且每页 ≤100 行、响应有界。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 的替换工具。
    """
    _use_root(tmp_path, monkeypatch, 100_000)
    index = get_index()
    index.sync_known()
    start = time.perf_counter()
    page = query.runs_page(limit=50)
    first_ms = (time.perf_counter() - start) * 1000
    assert len(page["runs"]) == 50 and page["next"]
    cursor = page["next"]
    last_ms = first_ms
    for _page in range(999):
        start = time.perf_counter()
        page = query.runs_page(limit=50, cursor=cursor)
        last_ms = (time.perf_counter() - start) * 1000
        assert len(page["runs"]) <= 50
        cursor = page["next"]
    assert last_ms <= max(first_ms * 2, 2.0), (first_ms, last_ms)


@pytest.mark.slow
def test_analysis_panels_stay_under_150ms_p95_on_ten_thousand_runs(tmp_path, monkeypatch):
    """@brief 10k 合成 run 下三个分析面板的 p95 ≤150ms（C 目标），用直接插行避免造 10k 目录。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 的替换工具。
    """
    from observe import analysis
    from observe.index import Index

    runs = tmp_path / "runs"
    runs.mkdir(parents=True)
    monkeypatch.setenv("AI_LOWERLING_RUNS_DIR", str(runs))
    monkeypatch.setenv("OBSERVE_DB", str(tmp_path / "observe.sqlite"))
    monkeypatch.setenv("AI_LOWERLING_BOOKS", str(tmp_path / "books.sqlite"))
    monkeypatch.setenv("OBSERVE_CARRYOVER", str(tmp_path / "carry.json"))
    monkeypatch.setenv("OBSERVE_REFRESH_ON_READ", "0")
    monkeypatch.setenv("TAIC_CATALOG_ROOT", str(tmp_path / "kernels"))
    index = Index(tmp_path / "observe.sqlite")
    rows = []
    for number in range(10_000):
        rows.append((
            f"conv2d/{number:05d}", "relu", "fp16", "fp16", "paper", "apple-m4-max",
            "3", "done", "max_rounds", "confirmed" if number < 15 else "in_progress",
            1.1 + (number % 40) / 100 if number < 15 else None, 1.5, 1.0, 0.5,
        ))
    index.conn.executemany(
        "INSERT INTO runs (run_id, task, dtype, status, protocol_version, target_id,"
        " stop_reason, publication, published_speedup, search_best_speedup, spent_usd,"
        " spent_max_usd, row_json) VALUES (?, 'relu', 'fp16', 'done', '3', 'apple-m4-max',"
        " 'max_rounds', ?, ?, ?, ?, 0.5, '{}')",
        [(row[0], row[9], row[10], row[11], row[12]) for row in rows],
    )
    # 真实语料里一场内的候选标签互不相同（r1.p0/r1.p1/…，实测 115 条会话的 candidate_label 109 条
    # 一一 join 到候选）。合成语料若把 10 颗候选都写成同一个 label，``committees`` 的
    # ``sessions LEFT JOIN candidates ON (run_id, label)`` 会 10 倍放大成 10 万行——那是在量一个
    # 线上不存在的语料形状（实测把 p95 从 ~29ms 抬到 ~150ms），不是量面板成本。
    index.conn.executemany(
        "INSERT INTO sessions (run_id, seq, label, candidate_label, attempt, committee, provider,"
        " model, outcome, cost_usd) VALUES (?, ?, ?, ?, 0, ?, 'deepseek', ?, 'evaluated', 0.01)",
        [(f"conv2d/{number:05d}", number, f"r1.p{number % 10}", f"r1.p{number % 10}",
          "first" if number % 2 else "second",
          "deepseek-flash" if number % 3 else "kimi-k3") for number in range(10_000)],
    )
    index.conn.executemany(
        "INSERT INTO candidates (run_id, idx, label, status, stage, speedup, cv, reps, cache_hit)"
        " VALUES (?, ?, ?, 'ok', 'bench', 1.2, ?, 15, ?)",
        [(f"conv2d/{number // 10:05d}", number % 10, f"r1.p{number % 10}",
          0.005 + (number % 50) / 10_000, None if number % 3 else 0) for number in range(100_000)],
    )
    index.conn.commit()

    def p95(callable_):
        timings = []
        for _ in range(15):
            start = time.perf_counter()
            callable_()
            timings.append((time.perf_counter() - start) * 1000)
        timings.sort()
        return timings[int(len(timings) * 0.95) - 1]

    gap_ms = p95(lambda: analysis.gap(protocol_epoch="3"))
    committees_ms = p95(analysis.committees)
    noise_ms = p95(analysis.noise)
    assert gap_ms <= 150, f"gap p95 {gap_ms:.1f}ms"
    assert committees_ms <= 150, f"committees p95 {committees_ms:.1f}ms"
    assert noise_ms <= 150, f"noise p95 {noise_ms:.1f}ms"
