"""§6.1-6.3：``taic query`` 的 L0 行与 ``taic brief`` 的 L1 摘要都必须按字节有界。

序列化口径与 plan 一致：``len(json.dumps(row, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))``。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from observe import query
from observe.index import Index
from taic import brief, cli, retrieval

# 真实语料里最长的一条 run id（73 字符）与最长的一条 stop_reason（224 字符，HTTP 402 provider 错误）。
# 两者都来自 ``runs/`` 与真实索引的实测，不是编出来的极值。
WIDE_RUN_ID = "apple-m4-max/20261025-111744_flash_attention_fp16_deepseek-flash_57mix1w_"
LONG_STOP = (
    "llm_error: init2.fix1#t2: HTTP 402: Error code: 402 - {'error': {'message': 'Insufficient "
    "Balance (request_id: d7d4cdef-a7d5-4a37-b0d9-2daf7b507217)', 'type': 'unknown_error', "
    "'param': None, 'code': 'invalid_request_error'}}"
)
# plan §0.3 第 6 条：“字段放不下就砍字段，不改预算”。真实 73 字符 run id 下七字段最坏 276B，
# 所以 L0 只留四个（run_id / 短发布码 / published_speedup / stop_code）；task、spent_usd、
# updated_at 与完整 stop_reason 都在 --wide 的 22 列里。
L0_KEYS = {"run_id", "publication", "published_speedup", "stop_code"}
WIDE_KEYS = {
    "run_id", "experiment_id", "arm_id", "trial", "task", "dtype", "model", "target_id", "status",
    "stop_reason", "publication", "published_speedup", "spent_usd", "budget_usd", "source", "binding",
    "kernel_sha256", "protocol_version", "protocol_epoch", "committees_json", "started_at", "updated_at",
}
L1_KEYS = {
    "schema_version", "protocol_version", "task", "dtype", "size", "committee", "model", "winner_label",
    "winner_provider", "winner_model", "winner_committee", "winner_origin", "phase", "stop_reason",
    "pending_evaluation", "round", "spent_usd", "budget_usd", "published", "publication", "search_best",
    "candidate_us", "baseline_us", "last_failure", "counts", "paths", "stamp",
}
HEAVY_KEYS = {"cands", "model_totals", "agent_failures"}


def _bytes(value) -> int:
    """@brief plan §6.1 的固定序列化口径，按 UTF-8 字节算。

    @param value 任意可 JSON 序列化的对象。
    @return 该对象序列化后的 UTF-8 字节数。
    """
    return len(json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))


def _temp_index(tmp_path, monkeypatch):
    """@brief 把 runs/、索引与账本都指到临时目录，并关掉读时 tick。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 的替换工具。
    @return ``(runs 根, Index)``。
    """
    runs = tmp_path / "runs"
    runs.mkdir()
    monkeypatch.setenv("AI_LOWERLING_RUNS_DIR", str(runs))
    monkeypatch.setenv("OBSERVE_DB", str(tmp_path / "observe.sqlite"))
    monkeypatch.setenv("AI_LOWERLING_BOOKS", str(tmp_path / "books.sqlite"))
    monkeypatch.setenv("OBSERVE_CARRYOVER", str(tmp_path / "carry.json"))
    monkeypatch.setenv("OBSERVE_REFRESH_ON_READ", "0")
    return runs, Index(tmp_path / "observe.sqlite")


def test_the_l0_runs_row_is_the_plan_projection_and_the_page_fits_its_budget(tmp_path, monkeypatch):
    """@brief 单元回归：L0 行是砍过字段的四列，20 行的页 ≤5KB，``--wide`` 仍是旧 22 列。

    单行 ≤200B 是对**真实 63 行**的权威验收，见
    :func:`test_the_real_index_l0_rows_meet_the_authoritative_200_byte_budget`；这里只用合成分布锁字段集合
    与页预算，不用合成 fixture 代替那条验收，也不把单行上限改宽。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 的替换工具。
    """
    _runs, index = _temp_index(tmp_path, monkeypatch)
    publications = ["confirmed", "in_progress", "旧版未绑定源码", "未确认"]
    for number in range(63):
        # 真实语料里大部分 run id 约 40 字符，最长的一条 73 字符；这里保留同样的分布，
        # 否则 20 行全是“最坏行”会把页预算测成一条历史语料不成立的紧约束。
        run_id = WIDE_RUN_ID if number == 0 else (
            f"{WIDE_RUN_ID[:68]}{number:04d}" if number % 10 == 0
            else f"apple-m4-max/2026102{number}-conv2d-fp16-{number:03d}"
        )
        index.conn.execute(
            "INSERT INTO runs (run_id, task, dtype, status, stop_reason, stop_code, publication,"
            " published_speedup, spent_usd, started_at, updated_at, row_json)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                run_id, "flash_attention", "fp16", "done", LONG_STOP, "llm_error",
                publications[number % len(publications)],
                1.0366541502131137, 9.489245735999999, float(number), 1792912751.367266 + number,
                json.dumps({"id": run_id}),
            ),
        )
    index.conn.commit()

    page = query.runs_page(limit=20, compact=True)
    rows = page["runs"]
    assert len(rows) == 20
    assert all(set(row) == L0_KEYS for row in rows)
    assert {row["publication"] for row in rows} <= {"confirmed", "in_progress", "legacy", "unconfirmed"}
    assert all(len(row["stop_code"]) <= 24 for row in rows)
    # 窄行不许带完整 stop_reason、也不许带被砍掉的宽列，更不许解码 row_json。
    assert all("task" not in row and "spent_usd" not in row and "updated_at" not in row for row in rows)
    assert all("stop_reason" not in row for row in rows)
    assert _bytes(page) <= 5120, f"L0 页 {_bytes(page)}B 超过 5KB"
    # 这里不断言单行上限：权威的单行预算只对真实 63 行有意义，合成行会把它测成另一条要求。

    wide = query.runs_page(limit=1, compact=False)
    assert set(wide["runs"][0]) == WIDE_KEYS
    assert _bytes(wide) > 400


def test_the_publication_filter_accepts_the_l0_short_codes(tmp_path, monkeypatch):
    """@brief 回归：``--publication legacy`` 必须真的过滤到 ``旧版未绑定源码`` 那些行。

    L0 输出的是短码（legacy/unconfirmed/uncertain/...），过滤若拿原始列等值比较，短码会静默
    返回 0 行。这里对每个短码断言「过滤结果 = 该短码在 L0 里出现的行」，并锁住 ``不确定`` 也有短码。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 的替换工具。
    """
    _runs, index = _temp_index(tmp_path, monkeypatch)
    raw = ["confirmed", "in_progress", "旧版未绑定源码", "未确认", "不确定", "未确认", "confirmed"]
    for number, publication in enumerate(raw):
        index.conn.execute(
            "INSERT INTO runs (run_id, task, dtype, status, stop_reason, stop_code, publication,"
            " published_speedup, started_at, updated_at, row_json)"
            " VALUES (?, 'relu', 'fp16', 'done', 'max_rounds', 'max_rounds', ?, 1.2, ?, ?, ?)",
            (f"conv2d/run{number}", publication, float(number), float(number), json.dumps({})),
        )
    index.conn.commit()
    everything = query.runs_page(limit=50, compact=True)["runs"]
    by_code: dict[str, set[str]] = {}
    for row in everything:
        by_code.setdefault(str(row["publication"]), set()).add(str(row["run_id"]))
    assert set(by_code) == {"confirmed", "in_progress", "legacy", "unconfirmed", "uncertain"}
    for code, expected in by_code.items():
        filtered = query.runs_page(limit=50, compact=True, publication=code)["runs"]
        assert {row["run_id"] for row in filtered} == expected, code
        assert all(row["publication"] == code for row in filtered), code
    # 两个原始短语都映射到 unconfirmed：过滤短码必须同时命中它们。
    assert by_code["unconfirmed"] == {"conv2d/run3", "conv2d/run5"}
    assert query.runs_page(limit=50, compact=True, publication="旧版未绑定源码")["total"] == 1


def test_the_l0_cursor_pages_without_gaps_or_duplicates(tmp_path, monkeypatch):
    """@brief 评审回归：L0 的 keyset 游标必须带真实的排序键，不能因为窄行没有该列就写成 -1。

    默认 ``sort=started`` 的排序键 ``started_at`` 不在 L0 四列里。旧实现把它当 ``last.get("started_at")``
    取不到值，游标恒为 -1，第二页于是永远为空。这里对四种排序（``spent``/``updated`` 也已不在 L0 里）
    都走完整翻页，断言不重不漏，并断言四列没有被内部排序键污染。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 的替换工具。
    """
    _runs, index = _temp_index(tmp_path, monkeypatch)
    for number in range(7):
        index.conn.execute(
            "INSERT INTO runs (run_id, task, dtype, status, published_speedup, spent_usd,"
            " started_at, updated_at, row_json)"
            " VALUES (?, 'relu', 'fp16', 'done', ?, ?, ?, ?, '{}')",
            (f"r{number}", float(number % 3), float(number), 100.0 + number, 200.0 + number),
        )
    index.conn.commit()

    for sort in ("started", "speedup", "spent", "updated"):
        seen: list[str] = []
        cursor = None
        for _page in range(10):
            page = query.runs_page(limit=3, compact=True, sort=sort, cursor=cursor)
            assert all(set(row) == L0_KEYS for row in page["runs"]), f"{sort} 页里漏了内部排序键"
            seen.extend(row["run_id"] for row in page["runs"])
            cursor = page["next"]
            if not cursor:
                break
        assert sorted(seen) == sorted(f"r{i}" for i in range(7)), f"{sort} 翻页不重不漏"
        assert len(seen) == len(set(seen))


def test_the_real_index_l0_rows_meet_the_authoritative_200_byte_budget():
    """@brief 权威验收（plan §6.1/§12.3）：真实 63 行每行 ≤200 UTF-8B，20 行页 ≤5KB。

    直接只读仓库里的 ``data/observe.sqlite``，用与 ``query runs`` 完全相同的 L0 SELECT 与
    plan 的序列化口径，不经过合成 fixture、不写索引。

    plan §0.3 第 6 条要求“字段放不下就砍字段，不改预算”，所以 L0 从七字段缩到四字段
    （``run_id/publication/published_speedup/stop_code``）。这里断言砍完之后的真实 63 行全部
    ≤200B，并且 ``--limit 20`` 的整页（含真实 envelope 与游标）≤5KB；预算不是 300B，
    也没有用合成行代替真实行。
    """
    import sqlite3

    from observe.query import _RUN_L0_SELECT

    index_path = Path(__file__).resolve().parents[2] / "data" / "observe.sqlite"
    if not index_path.exists():
        pytest.skip("仓库里没有真实索引 data/observe.sqlite，这条验收只能在有真实数据的 checkout 上跑")
    conn = sqlite3.connect(f"file:{index_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        rows = [dict(row) for row in conn.execute(f"SELECT {_RUN_L0_SELECT} FROM runs WHERE status != 'missing'")]
    finally:
        conn.close()
    if len(rows) < 60:
        pytest.skip(f"真实索引里只有 {len(rows)} 行，不是 plan 说的 63 场语料")
    assert all(set(row) == L0_KEYS for row in rows)
    worst = max(_bytes(row) for row in rows)
    assert worst <= 200, f"真实 {len(rows)} 行里最宽一行 {worst}B，超过 plan 的 200B/行"
    page = {"total": len(rows), "limit": 20, "offset": 0, "next": "x" * 160,
            "sort": "started", "order": "desc", "runs": rows[:20]}
    assert _bytes(page) <= 5120, f"20 行的 L0 页（含游标与 envelope）{_bytes(page)}B 超过 5KB"


def test_stop_code_is_written_at_ingest_not_cut_at_read(tmp_path, monkeypatch):
    """@brief ``stop_code`` 是摄入时写的窄列：冒号前第一段、≤24 字符，读路径不截 ``stop_reason``。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 的替换工具。
    """
    runs, index = _temp_index(tmp_path, monkeypatch)
    directory = runs / "conv2d" / "longstop"
    directory.mkdir(parents=True)
    common = {"task": "conv2d", "dtype": "fp16", "started_at": 10.0, "spent_usd": 0.5, "budget_usd": 2.0}
    (directory / "state.json").write_text(json.dumps({**common, "phase": "done", "updated_at": 10.0}))
    (directory / "result.json").write_text(json.dumps({**common, "stop_reason": LONG_STOP}))
    index.sync_known()

    stored = index.conn.execute(
        "SELECT stop_reason, stop_code FROM runs WHERE run_id = 'conv2d/longstop'"
    ).fetchone()
    assert stored["stop_reason"] == LONG_STOP, "完整停止原因仍要留在索引里给 --wide 与详情"
    assert stored["stop_code"] == "llm_error" and len(stored["stop_code"]) <= 24

    row = query.runs_page(limit=1, compact=True)["runs"][0]
    assert row["stop_code"] == "llm_error"
    assert "llm_error:" not in json.dumps(row, ensure_ascii=False)


def test_stop_code_is_one_stable_machine_word_per_stop_class():
    """@brief 停止码要“每类一个词”：轮数不同不能切成两个码，完整文本仍留在 --wide。

    真实语料里 8/63 行是 ``converged (12 rounds without improvement)`` / ``converged (4 rounds ...)``，
    按字符截断会得到两个不同的码，机器筛选就失效了。

    @raises AssertionError 同类停止出现两个码，或完整文本被塞进窄行时抛出。
    """
    from observe.detail import stop_code

    assert stop_code("converged (12 rounds without improvement)") == "converged"
    assert stop_code("converged (4 rounds without improvement)") == "converged"
    assert stop_code("max_rounds") == "max_rounds"
    assert stop_code("budget_exhausted") == "budget_exhausted"
    assert stop_code("llm_error: HTTP 402 provider error") == "llm_error"
    assert stop_code(None) == ""
    assert len(stop_code("x" * 200)) <= 24


def test_the_failure_row_is_redacted_truncated_and_carries_a_signature_and_path(tmp_path, monkeypatch):
    """@brief §6.3：失败 L0 每行 ≤2KB，``message`` 脱敏单行 ≤120 字符，带 ``sig`` 与可复现路径。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 的替换工具。
    """
    runs, index = _temp_index(tmp_path, monkeypatch)
    directory = runs / "conv2d" / "fails"
    directory.mkdir(parents=True)
    common = {"task": "conv2d", "dtype": "fp16", "started_at": 10.0, "spent_usd": 0.5, "budget_usd": 2.0}
    (directory / "state.json").write_text(json.dumps({**common, "phase": "done", "updated_at": 10.0}))
    (directory / "result.json").write_text(json.dumps({**common, "stop_reason": "max_rounds"}))
    message = "compiler said: " + "token sk-abcdefghijklmnop repeated " * 40
    folder = directory / "candidates" / "000_r1.p0"
    folder.mkdir(parents=True)
    (folder / "kernel.metal").write_text("kernel void k() {}\n")
    (folder / "report.json").write_text(json.dumps({
        "schema_version": 4, "status": "compile_error", "stage": "compile", "speedup": None, "message": message,
    }))
    index.sync_known()

    doc = retrieval.failures(limit=1)
    (row,) = doc["failures"]
    assert set(row) == {"run_id", "label", "status", "stage", "message", "sig", "report_path"}
    assert row["run_id"] == "conv2d/fails" and row["label"] == "r1.p0"
    assert row["status"] == "compile_error" and row["stage"] == "compile"
    assert len(row["message"]) <= 120 and "\n" not in row["message"]
    assert "sk-abcdefghijklmnop" not in row["message"], "失败行必须走 taic.redact 的密钥脱敏"
    assert row["sig"] and len(row["sig"]) == 16
    assert row["report_path"].endswith("candidates/000_r1.p0/report.json")
    assert _bytes(row) <= 2048, "一次“这场为何失败”的判断要 ≤2KB"


def test_the_brief_defaults_to_l1_and_expand_restores_the_heavy_sections(tmp_path, monkeypatch, capsys):
    """@brief §6.1/6.2：``brief`` 默认 L1 ≤1,536B，重面板只在 ``--expand`` 时出现且逐字段同值。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 的替换工具。
    @param capsys pytest 的输出捕获器。
    """
    monkeypatch.setenv("OBSERVE_REFRESH_ON_READ", "0")
    run = tmp_path / "run"
    run.mkdir()
    trace = [
        {"label": f"c{index}", "status": "ok", "speedup": 1.0 + index / 100, "within_noise": False,
         "differential_recheck": None}
        for index in range(8)
    ]
    (run / "state.json").write_text(json.dumps({
        "task": "flash_attention", "dtype": "fp16", "size": "paper", "phase": "done", "committee": "second",
        "model": "kimi-k3", "started_at": 10.0, "updated_at": 20.0, "spent_usd": 9.4, "budget_usd": 15.0,
        "best_speedup": 1.03, "best_label": "c7", "trace": trace,
    }), encoding="utf-8")
    (run / "result.json").write_text(json.dumps({
        "task": "flash_attention", "stop_reason": "converged (12 rounds without improvement)",
        "spent_usd": 9.4, "budget_usd": 15.0, "best_speedup": 1.03, "trace": trace,
        "confirmed_speedup": 1.0366541502131137, "confirmed_status": "ok",
        "confirm_winner_label": "c7", "confirmation_source_sha256": "a" * 64,
    }), encoding="utf-8")
    # 真实 schema：candidate_us 是标量，baseline_us 是 {基线名: 微秒} 字典（36/36 份真实确认报告
    # 都是这样；下面这两组数抄自 runs/apple-m4-max/20261024-065432_conv2d_fp16_deepseek-flash）。
    from experiments.common import synthetic_confirmation_documents

    _result, report = synthetic_confirmation_documents(
        task="flash_attention", dtype="fp16", size="paper", target_id="apple-m4-max",
        speedup=1.0366541502131137, label="c7", kernel_sha256="a" * 64,
        candidate_us=448.4129672431771,
    )
    # L1 的两个顶层耗时字段；文档本身是 schema 4，brief 才肯引用它。
    report["candidate_us"] = 448.4129672431771
    report["baseline_us"] = {"mlx": 991.1041389304128, "torch_mps": 779.4236039949789}
    (run / "confirm_report.json").write_text(json.dumps(report), encoding="utf-8")
    rows = [
        {"event": "agent", "label": "c0", "candidate_label": "c0", "provider": "deepseek",
         "model": "deepseek-flash", "committee": "first", "cost_usd": 0.2, "outcome": "provider_failed",
         "error": "Connection error: sk-abcdefghijklmnop timeout after 30s",
         "runtime": {"failure_class": "turn_failed", "failed_turns": 1, "last_terminal_kind": "failed"}},
        {"event": "agent", "label": "c7", "candidate_label": "c7", "provider": "moonshot",
         "model": "kimi-k3", "committee": "second", "cost_usd": 1.5, "outcome": "evaluated",
         "runtime": {"failure_class": "", "failed_turns": 0, "last_terminal_kind": "result"}},
    ]
    (run / "transcript.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    assert cli.main(["brief", str(run)]) == 0
    out = capsys.readouterr().out
    l1 = json.loads(out)
    assert set(l1) == L1_KEYS
    assert HEAVY_KEYS & set(l1) == set()
    assert _bytes(l1) <= 1536, f"L1 {_bytes(l1)}B 超过 1,536B"
    assert l1["candidate_us"] == pytest.approx(448.4129672431771)
    assert l1["baseline_us"] == {"mlx": pytest.approx(991.1041389304128), "torch_mps": pytest.approx(779.4236039949789)}
    assert l1["last_failure"]["failure_class"] == "turn_failed"
    assert l1["last_failure"]["sig"] and len(l1["last_failure"]["sig"]) == 16
    assert len(l1["last_failure"]["error"]) <= 120
    assert "sk-abcdefghijklmnop" not in l1["last_failure"]["error"]

    expanded = brief.summarize_run(run, expand={"cands", "models", "failures"})
    assert HEAVY_KEYS <= set(expanded)
    assert len(expanded["cands"]) == 8
    assert expanded["model_totals"][0]["model"] == "kimi-k3"
    assert len(expanded["agent_failures"]) == 1
    for key, value in l1.items():
        assert expanded[key] == value, f"展开后 {key} 与 L1 不一致"
    # 展开的重面板也不许把错误正文原文放出来。
    assert "sk-abcdefghijklmnop" not in json.dumps(expanded, ensure_ascii=False)


REPO = Path(__file__).resolve().parents[2]
# 四个真实 fixture（plan §6.1 的 brief 预算）：两个当前协议的确认运行、一个旧的 chat 协议运行、
# 一个带 agent 失败的 harness pilot。都在仓库的 runs/ 里，只读；语料不在时整条跳过。
REAL_BRIEF_FIXTURES = (
    "runs/apple-m4-max/20261025-111744_flash_attention_fp16_deepseek-flash_57mix1w_",
    "runs/apple-m4-max/20261025-070409_conv2d_fp16_deepseek-flash_g5yab2vc",
    "runs/apple-m4-max/20261024-065432_conv2d_fp16_deepseek-flash",
    "runs/apple-m4-max/harness-pilot-active-20261022",
)


def _brief_bytes(doc: dict) -> int:
    """@brief CLI 的实际序列化口径（`run_brief` 用 sort_keys）。

    @param doc brief 文档。
    @return UTF-8 字节数。
    """
    return len(json.dumps(doc, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8"))


def test_the_four_real_brief_fixtures_fit_the_l1_budget():
    """@brief 权威验收（plan §6.1）：四个真实 fixture 的 brief 默认 L1 都 ≤1,536B。

    直接读仓库里真实的 run 目录（不复制、不新增 golden），断言与 CLI 相同的序列化口径下每份都
    在预算内，且都恰好是 L1 的键集合（重面板只有 ``--expand`` 才出现）。

    @return None
    """
    roots = [REPO / name for name in REAL_BRIEF_FIXTURES]
    missing = [str(root) for root in roots if not ((root / "state.json").is_file() or (root / "result.json").is_file())]
    if missing:
        pytest.skip(f"仓库里缺少真实 brief fixture：{missing}")
    sizes = {}
    for root in roots:
        doc = brief.summarize_run(root)
        assert set(doc) == L1_KEYS, f"{root.name} 的 brief 默认形状不是 L1"
        sizes[root.name] = _brief_bytes(doc)
        assert sizes[root.name] <= 1536, f"{root.name} 的 L1 {sizes[root.name]}B 超过 1,536B"
    # 至少有一个是“大”的（不是空目录凑数）。
    assert max(sizes.values()) > 700


def _scenario_root(tmp_path: Path, name: str) -> Path:
    """@brief 四种场景夹具的公共外壳：state + 一段 trace。

    @param tmp_path 临时目录。
    @param name 场景名。
    @return run 目录。
    """
    run = tmp_path / name
    run.mkdir(parents=True)
    # trace 行必须绑定到候选自己的当前（schema-4）报告，否则它是不可判的（§8）。
    for index, (label, speed) in enumerate((("init", 1.0), ("r1.p0", 1.25))):
        folder = run / "candidates" / f"{index:03d}_{label}"
        folder.mkdir(parents=True)
        (folder / "kernel.metal").write_text("kernel void k() {}\n", encoding="utf-8")
        (folder / "report.json").write_text(json.dumps({
            "schema_version": 4, "status": "ok", "speedup": speed,
        }), encoding="utf-8")
    trace = [
        {"label": "init", "status": "ok", "speedup": 1.0, "within_noise": False, "differential_recheck": None},
        {"label": "r1.p0", "status": "ok", "speedup": 1.25, "within_noise": False,
         "differential_recheck": {"class": "timeout"}},
    ]
    (run / "state.json").write_text(json.dumps({
        "task": "conv2d", "dtype": "fp16", "size": "paper", "committee": "first", "model": "deepseek-flash",
        "phase": "done", "started_at": 1.0, "updated_at": 2.0, "spent_usd": 0.4, "budget_usd": 2.0,
        "best_speedup": 1.25, "best_label": "r1.p0", "trace": trace,
    }), encoding="utf-8")
    return run


def _agent_line(label: str, *, provider="deepseek", model="deepseek-flash", committee="first",
                cost=0.2, output=10, outcome="evaluated", failure_class=None, error=None) -> dict:
    """@brief 造一条 ``event=agent`` 转录行。

    @param label 会话标签。
    @param provider provider。
    @param model 模型名。
    @param committee 委员会。
    @param cost 花费。
    @param output 输出 token。
    @param outcome 结局。
    @param failure_class 失败类别。
    @param error 错误文本。
    @return 事件字典。
    """
    runtime = {"failure_class": failure_class or "", "failed_turns": 0, "last_terminal_kind": "result"}
    if error is not None:
        runtime["last_error"] = error
    return {
        "event": "agent", "label": label, "candidate_label": label, "attempt": 0,
        "provider": provider, "model": model, "committee": committee, "cost_usd": cost,
        "tokens": {"input": 100, "output": output, "cache_read": 1000, "cache_write": 0},
        "outcome": outcome, "runtime": runtime,
    }


def _expected_totals(events: list[dict]) -> dict:
    """@brief 独立聚合 agent 行的模型费用（不调用 brief 的聚合）。

    @param events transcript 事件列表。
    @return ``{(committee, provider, model): (sessions, cost, output_tokens)}``。
    """
    sums: dict[tuple, list] = {}
    for event in events:
        if event.get("event") != "agent":
            continue
        key = (event.get("committee"), event.get("provider"), event.get("model"))
        item = sums.setdefault(key, [0, 0.0, 0])
        item[0] += 1
        item[1] += float(event.get("cost_usd") or 0.0)
        item[2] += int((event.get("tokens") or {}).get("output") or 0)
    return {key: tuple(value) for key, value in sums.items()}


def test_the_brief_last_failure_signature_is_queryable_through_failures(tmp_path, monkeypatch):
    """@brief §6.1/§6.3 回归：``brief.last_failure.sig`` 必须能被 ``query failures --sig`` 找到。

    brief 的指纹来自会话（整轮没有内核），``failures`` 过去只查 ``candidates.failure_sig``，
    这条承诺是假的。现在 ``failures --sig`` 同时匹配 ``sessions.failure_sig``。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 的替换工具。
    """
    from taic import retrieval

    runs, index = _temp_index(tmp_path, monkeypatch)
    directory = runs / "conv2d" / "no_kernel"
    directory.mkdir(parents=True)
    (directory / "state.json").write_text(json.dumps({
        "task": "conv2d", "dtype": "fp16", "phase": "done", "started_at": 5, "updated_at": 5,
    }), encoding="utf-8")
    (directory / "result.json").write_text(json.dumps({
        "task": "conv2d", "dtype": "fp16", "stop_reason": "agent_failure", "spent_usd": 0.2, "budget_usd": 1.0,
    }), encoding="utf-8")
    failing = _agent_line("r1.p0", outcome="agent_failed", failure_class="turn_failed",
                          error="provider returned 402")
    (directory / "transcript.jsonl").write_text(json.dumps(failing) + "\n", encoding="utf-8")
    index.sync_known()
    import observe.query as query_mod

    monkeypatch.setattr(query_mod, "tick", lambda *_a, **_k: (index.revision(), []))

    doc = brief.summarize_run(directory)
    sig = doc["last_failure"]["sig"]
    assert sig and len(sig) == 16
    # 冷路径：冷建不投影 sessions（§11.4 选 (a)），这里先确认 sessions 还是空的，再**直接**用
    # brief 的指纹查 failures——不能像旧回归那样先 sessions_page 预热，否则测不到要测的东西。
    cold = index.conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
    assert cold == 0, cold
    rows = retrieval.failures(sig=sig)["failures"]
    assert rows, "冷索引上 brief 的 last_failure.sig 在 failures 里查不到"
    assert rows[0]["sig"] == sig and rows[0]["run_id"] == "conv2d/no_kernel"
    assert rows[0]["report_path"] is None and rows[0]["stage"] == "turn_failed"
    assert "402" in (rows[0]["message"] or "")
    # 反查触发的懒投影只落在本 run，指纹与 brief 一致；第二次查询不再重新投影。
    sessions = query_mod.sessions_page("conv2d/no_kernel")["sessions"]
    assert [row["failure_sig"] for row in sessions] == [sig]
    again = query_mod.sessions_page("conv2d/no_kernel")["sessions"]
    assert [row["failure_sig"] for row in again] == [sig]
    assert retrieval.failures(sig=sig)["failures"][0]["sig"] == sig


def test_expand_restores_the_old_default_sections_on_the_four_scenarios(tmp_path):
    """@brief 权威验收（plan §6.1/§6.2）：``--expand`` 与旧默认逐字段一致，四种场景都成立。

    旧默认（PR10 之前）把候选清单、模型费用与完整失败列表直接放进 brief。现在它们只在
    ``expand`` 里，所以这里用**独立构造的期望值**（trace 里的候选、按 agent 行自己累加的费用、
    夹具里注入的失败事件）逐字段对拍，并断言 expand 不改变任何 L1 字段、空场景也照旧给空数组。

    @param tmp_path pytest 提供的临时目录。
    """
    scenarios = {}
    # 正常：agent 行 + chat/retry/truncation + 确认报告。
    normal = _scenario_root(tmp_path, "normal")
    normal_events = [
        _agent_line("init"),
        _agent_line("r1.p0", provider="moonshot", model="kimi-k3", committee="second", cost=1.5, output=40),
        {"event": "chat", "label": "init#t0", "model": "deepseek-flash", "seconds": 1.5, "t": 3.0},
        {"event": "retry", "label": "init#t1", "attempt": 1, "error": "timeout"},
        {"event": "warning", "label": "init#t2", "message": "reply truncated by max_tokens"},
    ]
    (normal / "transcript.jsonl").write_text(
        "\n".join(json.dumps(event) for event in normal_events) + "\n", encoding="utf-8")
    (normal / "result.json").write_text(json.dumps({
        "task": "conv2d", "dtype": "fp16", "stop_reason": "max_rounds", "spent_usd": 0.4,
        "budget_usd": 2.0, "best_speedup": 1.25, "winner_label": "r1.p0",
        "confirm_winner_label": "r1.p0", "confirmation_source_sha256": "a" * 64,
        "confirmed_speedup": 1.25, "confirmed_status": "ok",
        "trace": json.loads((normal / "state.json").read_text())["trace"],
    }), encoding="utf-8")
    from experiments.common import synthetic_confirmation_documents

    _normal_result, normal_report = synthetic_confirmation_documents(
        task="conv2d", dtype="fp16", size="paper", target_id="apple-m4-max",
        speedup=1.25, label="r1.p0", kernel_sha256="a" * 64, candidate_us=448.41,
    )
    normal_report["candidate_us"] = 448.41
    normal_report["baseline_us"] = {"mlx": 991.10, "torch_mps": 779.42}
    (normal / "confirm_report.json").write_text(json.dumps(normal_report), encoding="utf-8")
    scenarios["normal"] = normal

    # cost_incomplete：停止原因就是它，转录里只有一次没有 token 账单的 chat。
    cost = _scenario_root(tmp_path, "cost_incomplete")
    cost_events = [{"event": "chat", "label": "init#t0", "model": "deepseek-flash"}]
    (cost / "transcript.jsonl").write_text(json.dumps(cost_events[0]) + "\n", encoding="utf-8")
    (cost / "result.json").write_text(json.dumps({
        "task": "conv2d", "dtype": "fp16", "stop_reason": "cost_incomplete", "spent_usd": 0.1,
        "trace": json.loads((cost / "state.json").read_text())["trace"],
    }), encoding="utf-8")
    scenarios["cost_incomplete"] = cost

    # agent_failed：注入 5 条无核结果，旧默认只保留最后 4 条。
    failed = _scenario_root(tmp_path, "agent_failed")
    failed_events = [
        _agent_line(f"r{index}.p0", outcome="agent_failed", failure_class="turn_failed",
                    error=f"provider error {index}")
        for index in range(5)
    ]
    (failed / "transcript.jsonl").write_text(
        "\n".join(json.dumps(event) for event in failed_events) + "\n", encoding="utf-8")
    (failed / "result.json").write_text(json.dumps({
        "task": "conv2d", "dtype": "fp16", "stop_reason": "agent_failure", "spent_usd": 0.3,
        "trace": json.loads((failed / "state.json").read_text())["trace"],
    }), encoding="utf-8")
    scenarios["agent_failed"] = failed

    # 无 transcript：计数类字段保持 null，两个重面板仍是空数组。
    empty = _scenario_root(tmp_path, "no_transcript")
    (empty / "result.json").write_text(json.dumps({
        "task": "conv2d", "dtype": "fp16", "stop_reason": "max_rounds", "spent_usd": 0.2,
        "trace": json.loads((empty / "state.json").read_text())["trace"],
    }), encoding="utf-8")
    scenarios["no_transcript"] = empty

    for name, root in scenarios.items():
        l1 = brief.summarize_run(root)
        expanded = brief.summarize_run(root, expand={"cands", "models", "failures"})
        assert set(expanded) == set(l1) | {"cands", "model_totals", "agent_failures"}, name
        for key, value in l1.items():
            assert expanded[key] == value, f"{name} 的 {key} 被 expand 改动了"
        # 旧默认的候选：trace 现场推导（不调用 brief 的映射）。
        trace = json.loads((root / "state.json").read_text())["trace"]
        expected_cands = [
            {
                "label": item["label"], "status": item["status"], "speedup": item["speedup"],
                "within_noise": item["within_noise"],
                "differential_recheck": (item.get("differential_recheck") or {}).get("class")
                if isinstance(item.get("differential_recheck"), dict) else item.get("differential_recheck"),
            }
            for item in trace[-8:]
        ]
        got_cands = [
            {key: row[key] for key in ("label", "status", "speedup", "within_noise", "differential_recheck")}
            for row in expanded["cands"]
        ]
        assert got_cands == expected_cands, name

        transcript = root / "transcript.jsonl"
        events = [json.loads(line) for line in transcript.read_text().splitlines()] if transcript.is_file() else []
        expected_totals = _expected_totals(events)
        got_totals = {
            (row["committee"], row["provider"], row["model"]): (row["sessions"], row["cost_usd"], row["output_tokens"])
            for row in expanded["model_totals"]
        }
        assert set(got_totals) == set(expected_totals), name
        for key, (sessions, cost, output) in expected_totals.items():
            assert got_totals[key][0] == sessions and got_totals[key][2] == output
            assert got_totals[key][1] == pytest.approx(cost)

        failing = [
            event for event in events
            if event.get("event") == "agent"
            and event.get("outcome") in {"agent_failed", "provider_failed", "agent_no_submission"}
        ]
        got_failures = expanded["agent_failures"]
        assert len(got_failures) == min(4, len(failing)), name
        for got, expected in zip(got_failures, failing[-4:]):
            assert got["label"] == expected["label"] and got["outcome"] == expected["outcome"]
            assert got["failure_class"] == expected["runtime"]["failure_class"]
            assert got["failed_turns"] == expected["runtime"]["failed_turns"]
        if name == "no_transcript":
            assert got_failures == [] and expanded["model_totals"] == []
            assert l1["counts"]["chat"] is None and l1["counts"]["retry"] is None


def test_the_kernels_list_is_paged_by_cursor(tmp_path, monkeypatch, capsys):
    """@brief §6.1：``brief --kernels`` 支持 ``--limit/--cursor``，翻页不重不漏。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 的替换工具。
    @param capsys pytest 的输出捕获器。
    """
    from taic import kernel_catalog

    monkeypatch.setenv("TAIC_CATALOG_ROOT", str(tmp_path / "kernels"))
    for task in ("a", "b", "c"):
        base = tmp_path / "kernels" / "apple-m4-max" / task / "fp16" / "paper"
        base.mkdir(parents=True)
        (base / "meta.json").write_text(json.dumps({"task": task, "published_speedup": 1.0}))
        (base / "best.metal").write_text("kernel void k() {}\n")

    assert [row["task"] for row in kernel_catalog.list_kernels(limit=2)] == ["a", "b"]
    first = kernel_catalog.list_kernels(limit=2)
    rest = kernel_catalog.list_kernels(limit=2, cursor=first[-1]["path"])
    assert [row["task"] for row in rest] == ["c"]
    assert len(kernel_catalog.list_kernels()) == 3

    assert cli.main(["brief", "--kernels", "--limit", "1"]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert [row["task"] for row in printed] == ["a"]
