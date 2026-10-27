"""Measure what the observe read paths actually cost, beyond wall-clock time.

六个计数（请求数 / 端点字节 / 文件读取次数 / 读入字节 / stat 次数 / json 解码次数），
外加 `history()` 的 `row_json` 解码行数。所有数据都在临时目录合成，计数器用
monkeypatch 打在 `Path` 与 `json.loads` 上，**不改生产代码、不写 `runs/`、不占 GPU、不调模型**。

    uv run python -m experiments.read_probe --runs 200
    uv run python -m experiments.read_probe --runs 200 --out /tmp/read-probe.json

每个场景输出 cold / warm 两行 JSON：cold 是重置单例与缓存后的第一次调用，
warm 是紧接着的第二次。`row_json_rows` 只在 `history()` 的缓存键真正变化时累加，
所以它量的是"真的解码了多少行"，而不是"调用了几次 history"。
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

from experiments.scale_probe import _make_root, _reset_singletons


class Counters:
    """@brief 一次场景测量的全部计数器。"""

    def __init__(self) -> None:
        self.file_reads = 0
        self.bytes_read = 0
        self.stat_calls = 0
        self.json_decode_calls = 0
        self.row_json_rows = 0
        self.requests = 0
        self.endpoint_bytes = 0
        self.elapsed_ms = 0.0


class _MeteredFile:
    """@brief 包住文件对象，把 read/readline/迭代读到的字节计进计数器。"""

    def __init__(self, handle, counters: Counters) -> None:
        self._handle = handle
        self._counters = counters
        self._counted = False

    def _count(self, data) -> int:
        size = len(data) if isinstance(data, bytes) else len(data.encode("utf-8", errors="replace"))
        self._counters.bytes_read += size
        return size

    def read(self, *args):
        data = self._handle.read(*args)
        self._count(data)
        return data

    def readline(self, *args):
        data = self._handle.readline(*args)
        self._count(data)
        return data

    def __iter__(self):
        for line in self._handle:
            self._count(line)
            yield line

    def __enter__(self):
        self._mark()
        return self

    def __exit__(self, *exc):
        return self._handle.__exit__(*exc)

    def _mark(self):
        if not self._counted:
            self._counters.file_reads += 1
            self._counted = True

    def __getattr__(self, name):
        return getattr(self._handle, name)


def _instrument(counters: Counters):
    """@brief 给 ``Path.open``/``read_text``/``read_bytes``/``stat`` 与 ``json.loads`` 打计数器。

    只包 ``read_text``/``read_bytes`` 会漏掉 PR5 新引入的 ``Path.open`` + 二进制 ``read`` 与
    逐行迭代；这里统一在 ``io.open`` 上计量，文本/二进制、``read``/``readline``/迭代都算。

    @param counters 累加目标。
    @return 无参还原函数。
    """
    import io

    original_open = io.open
    original_stat = Path.stat
    original_loads = json.loads

    def counting_open(file, mode="r", *args, **kwargs):
        handle = original_open(file, mode, *args, **kwargs)
        if "r" not in mode or any(flag in mode for flag in "wa+"):
            return handle
        return _MeteredFile(handle, counters)

    def stat(self, *args, **kwargs):
        counters.stat_calls += 1
        return original_stat(self, *args, **kwargs)

    def loads(value, *args, **kwargs):
        counters.json_decode_calls += 1
        return original_loads(value, *args, **kwargs)

    io.open = counting_open
    Path.stat = stat
    json.loads = loads

    def restore() -> None:
        io.open = original_open
        Path.stat = original_stat
        json.loads = original_loads

    return restore


def _wrap_history(counters: Counters):
    """@brief 统计 ``history()`` 真正解码的 ``row_json`` 行数（按缓存键变化判定）。

    @param counters 累加目标。
    @return 无参还原函数。
    """
    from observe import query

    original = query.history

    def history(*args, **kwargs):
        before = query._HISTORY.get("key")
        rows = original(*args, **kwargs)
        if query._HISTORY.get("key") != before:
            counters.row_json_rows += len(rows)
        return rows

    query.history = history

    def restore() -> None:
        query.history = original

    return restore


def _write_transcript(directory: Path, events: int = 120) -> None:
    """@brief 给回合场景写一份带分组的转录，避免空回合窗口。

    @param directory 运行目录。
    @param events 事件条数。
    """
    lines = []
    for index in range(events):
        label = f"r{index // 6}.p0"
        lines.append(json.dumps({
            "event": "agent", "label": label, "model": "bench",
            "cost_usd": 0.01, "t": index, "outcome": "submitted",
        }))
    (directory / "transcript.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _scenario(name: str, calls, counters: Counters, *, temperature: str, runs: int) -> dict:
    """@brief 依次调用一组读端点，返回一行 JSON 结果。

    @param name 场景名。
    @param calls ``(端点名, 无参调用)`` 序列。
    @param counters 计数器（已被 instrument 包住）。
    @param temperature ``cold`` 或 ``warm``。
    @param runs 合成运行数。
    @return 结果字典。
    """
    started = time.perf_counter()
    for _label, call in calls:
        result = call()
        counters.requests += 1
        counters.endpoint_bytes += len(json.dumps(result, ensure_ascii=False, default=str).encode("utf-8"))
    counters.elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
    return {
        "scenario": name, "temperature": temperature, "synthetic_runs": runs,
        "requests": counters.requests, "endpoint_bytes": counters.endpoint_bytes,
        "file_reads": counters.file_reads, "bytes_read": counters.bytes_read,
        "stat_calls": counters.stat_calls, "json_decode_calls": counters.json_decode_calls,
        "row_json_rows": counters.row_json_rows, "elapsed_ms": counters.elapsed_ms,
    }


def _open_run_calls(saved: dict, temperature: str, query, detail):
    """@brief 详情场景的调用序列；warm 复用 cold 拿到的 stamp，量出戳短路省了多少。

    @param saved cold 阶段的响应。
    @param temperature ``cold`` 或 ``warm``。
    @param query ``observe.query`` 模块。
    @param detail ``observe.detail`` 模块。
    @return ``(端点名, 无参调用)`` 序列。
    """
    if temperature == "cold":
        return [
            ("detail", lambda: saved.setdefault("detail", query.run_detail("wide"))),
            ("sessions", lambda: saved.setdefault("sessions", query.sessions_page("wide"))),
            ("model_stats", lambda: query.model_stats("wide")),
            ("turns", lambda: saved.setdefault("turns", detail.turns_feed("wide"))),
        ]
    return [
        ("detail", lambda: query.run_detail("wide", stamp=(saved["detail"] or {}).get("detail_stamp", ""))),
        ("sessions", lambda: query.sessions_page("wide", stamp=(saved["sessions"] or {}).get("stamp", ""))),
        ("model_stats", lambda: query.model_stats("wide")),
        ("turns", lambda: detail.turns_feed("wide", stamp=(saved["turns"] or {}).get("stamp", ""))),
    ]


def _review_calls(saved: dict, temperature: str, query):
    """@brief 审查页场景的调用序列；warm 用 cold 的 revision 走增量。

    @param saved cold 阶段的响应。
    @param temperature ``cold`` 或 ``warm``。
    @param query ``observe.query`` 模块。
    @return ``(端点名, 无参调用)`` 序列。
    """
    if temperature == "cold":
        return [("feedback_full", lambda: saved.setdefault("feedback", query.feedback_board()))]
    rev = (saved["feedback"] or {}).get("rev")
    return [("feedback_since", lambda: query.feedback_board(since=rev))]


def _git_sha() -> str:
    """@brief 取当前 git 短 SHA；失败时返回 ``unknown``。"""
    try:
        result = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=10)
        return result.stdout.strip() or "unknown"
    except (OSError, subprocess.TimeoutExpired):
        return "unknown"


def measure(runs: int) -> list[dict]:
    """@brief 造合成语料并测量四个场景与 AI 出口字节。

    @param runs 简单合成运行的个数（另加一个 400 候选的宽运行）。
    @return 每个场景 cold/warm 两行，外加一行 AI 出口字节。
    """
    from observe import detail, query
    from observe.index import get_index
    from experiments.observe_bench import _write_wide

    root = Path(tempfile.mkdtemp(prefix="taic-read-probe-"))
    saved_env = {key: os.environ.get(key) for key in (
        "AI_LOWERLING_RUNS_DIR", "OBSERVE_DB", "AI_LOWERLING_BOOKS",
        "AI_LOWERLING_OBSERVE_DIR", "OBSERVE_CARRYOVER",
        "OBSERVE_TICK_INTERVAL", "OBSERVE_FULL_TICK_INTERVAL",
    )}
    saved_gpu = query.detail.gpu_snapshot
    saved_evals = query.detail.live_evals
    try:
        _make_root(root / "runs", runs)
        _write_wide(root / "runs")
        _write_transcript(root / "runs" / "wide")
        os.environ["AI_LOWERLING_RUNS_DIR"] = str(root / "runs")
        os.environ["OBSERVE_DB"] = str(root / "observe.sqlite")
        os.environ["AI_LOWERLING_BOOKS"] = str(root / "none.sqlite")
        os.environ["AI_LOWERLING_OBSERVE_DIR"] = str(root / "observe-cache")
        os.environ["OBSERVE_CARRYOVER"] = str(root / "none-carryover.json")
        os.environ["OBSERVE_TICK_INTERVAL"] = "0"
        os.environ["OBSERVE_FULL_TICK_INTERVAL"] = "3600"
        _reset_singletons()
        get_index().sync_known()
        # 总览里的 GPU/现场文件与本次合成无关，按既有测试的做法替换掉，避免读真机状态。
        query.detail.gpu_snapshot = lambda: {"holders": [], "waiters": [], "events": []}
        query.detail.live_evals = lambda: []

        counters = Counters()
        restore_instrument = _instrument(counters)
        restore_history = _wrap_history(counters)
        rows = []
        try:
            scenarios = {
                "first_screen": [
                    ("runs", lambda: query.runs_page()),
                    ("experiments", query.experiments),
                    ("budget", query.budget_report),
                    ("overview", query.overview),
                ],
                "open_run": [
                    ("detail", lambda: query.run_detail("wide")),
                    ("sessions", lambda: query.sessions_page("wide")),
                    ("model_stats", lambda: query.model_stats("wide")),
                    ("turns", lambda: detail.turns_feed("wide")),
                ],
                "open_reviews": [
                    ("feedback_full", lambda: query.feedback_board()),
                ],
            }
            for name, calls in scenarios.items():
                saved = {}
                for temperature in ("cold", "warm"):
                    if temperature == "cold":
                        _reset_singletons()
                    counters.__init__()
                    if name == "open_run":
                        calls = _open_run_calls(saved, temperature, query, detail)
                    elif name == "open_reviews":
                        calls = _review_calls(saved, temperature, query)
                    rows.append(_scenario(name, calls, counters, temperature=temperature, runs=runs))

            # 打开更早的回合页：先取尾窗，再用它给出的 cursor 往前翻。
            for temperature in ("cold", "warm"):
                if temperature == "cold":
                    _reset_singletons()
                counters.__init__()
                state = {}
                rows.append(_scenario("open_turns_older", [
                    ("turns_tail", lambda: state.setdefault("tail", detail.turns_feed("wide"))),
                    ("turns_older", lambda: detail.turns_feed("wide", before=(state["tail"] or {}).get("cursor"))),
                ], counters, temperature=temperature, runs=runs))

            from taic.brief import summarize_run

            counters.__init__()
            ai = _scenario("ai_bytes", [
                ("runs_page_50", lambda: query.runs_page(limit=50)),
                ("failures_50", lambda: query.failures()),
                ("models", query.model_stats),
                ("brief_wide", lambda: summarize_run(root / "runs" / "wide")),
            ], counters, temperature="cold", runs=runs)
            rows.append(ai)
        finally:
            restore_instrument()
            restore_history()
        return rows
    finally:
        query.detail.gpu_snapshot = saved_gpu
        query.detail.live_evals = saved_evals
        for key, value in saved_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        _reset_singletons()
        shutil.rmtree(root, ignore_errors=True)


def main(argv: list[str] | None = None) -> int:
    """@brief ``python -m experiments.read_probe`` 入口。

    @param argv 命令行参数；为 ``None`` 时取 ``sys.argv``。
    @return 进程退出码。
    """
    parser = argparse.ArgumentParser(prog="python -m experiments.read_probe")
    parser.add_argument("--runs", type=int, default=200)
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)
    rows = measure(args.runs)
    for row in rows:
        print(json.dumps({"git": _git_sha(), **row}, ensure_ascii=False), flush=True)
    if args.out:
        Path(args.out).write_text(json.dumps(rows, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
