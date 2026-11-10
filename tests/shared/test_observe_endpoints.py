"""PR25 全量端点：显式游标表、响应上限、304 不读文件、LRU 缓存与 suggestions 预读。

每个路由都要有自己的声明（列表上限或单对象），白名单只有页面、reviews 与 SSE 流。
"""

from __future__ import annotations

import json
import os
import time
import urllib.request
from pathlib import Path

import pytest

from observe import query, suggestions
from observe.index import Index, get_index

# Endpoints that deliberately have no ``?stamp=`` cursor: the two pages, the SSE stream, and the
# archived Track-exploration reader, which serves one fixed evidence document per request.
WHITELIST = {"/", "/reviews", "/api/v2/stream", "/api/v2/track-explore"}

# path -> (list key, max items, max response bytes). None 表示单对象端点。
CURSOR_ENDPOINTS = {
    "/api/v2/overview": (None, None, 4096),
    "/api/v2/live": (None, None, 32768),
    "/api/v2/patrol": ("alerts", 10, 1536),
    "/api/v2/feedback": ("upserts", 200, 512 * 1024),
    "/api/v2/suggestions": ("groups", 200, 65536),
    "/api/v2/suggestion": (None, None, 65536),
    "/api/v2/harness/{run_id:path}": (None, None, 65536),
    "/api/v2/runs": ("runs", 100, 131072),
    "/api/v2/experiments": ("experiments", 100, 131072),
    "/api/v2/experiments/{experiment_id:path}": (None, None, 131072),
    "/api/v2/budget": ("flagged", 10, 65536),
    "/api/hmz-web": (None, None, 4096),
    "/api/v2/ledger": ("by_month", 240, 131072),
    "/api/v2/turns/{run_id:path}": ("rows", 400, 131072),
    "/api/v2/failures": ("failures", 100, 131072),
    "/api/v2/publications": ("rows", 100, 131072),
    "/api/v2/search": ("rows", 20, 65536),
    "/api/v2/publications/{sha256}/source": (None, None, 65536),
    "/api/v2/gap": (None, None, 131072),
    "/api/v2/committees": ("buckets", 100, 131072),
    "/api/v2/noise": ("reps", 100, 131072),
    "/api/v2/best": ("best", 100, 131072),
    "/api/v2/failure-clusters": ("clusters", 100, 131072),
    "/api/v2/failure-clusters/{sig}": ("examples", 100, 131072),
    "/api/v2/runs/{run_id:path}/candidates": ("candidates", 50, 131072),
    "/api/v2/runs/{run_id:path}/candidate/{label}": (None, None, 131072),
    "/api/v2/runs/{run_id:path}/feedback/{seq:int}": (None, None, 131072),
    "/api/v2/runs/{run_id:path}/feedback": ("feedback.records", 10, 131072),
    "/api/v2/runs/{run_id:path}/timeline": ("timeline", 200, 131072),
    "/api/v2/runs/{run_id:path}/budget-events": ("budget_events", 100, 65536),
    "/api/v2/runs/{run_id:path}/llm": ("llm", 200, 131072),
    "/api/v2/runs/{run_id:path}/grid": ("feature_grid", 400, 131072),
    "/api/v2/runs/{run_id:path}/plan": ("plan_outcomes", 200, 131072),
    "/api/v2/runs/{run_id:path}/lineage": (None, None, 65536),
    "/api/v2/runs/{run_id:path}/raw": (None, None, 131072),
    "/api/v2/runs/{run_id:path}/detail": ("candidates", 400, 262144),
    "/api/v2/runs/{run_id:path}/sessions": ("sessions", 500, 131072),
    "/api/v2/runs/{run_id:path}/similar": ("items", 200, 131072),
    "/api/v2/model-stats": (None, None, 131072),
    "/api/v2/pi/{run_id:path}": ("lines", 40, 131072),
    "/api/v2/runs/{run_id:path}": (None, None, 65536),
    "/api/overview": (None, None, 4096),
    "/api/budget": ("flagged", 10, 65536),
    "/api/budget/reset": (None, None, 4096),
    "/api/stop": (None, None, 4096),
    "/api/settings/model": (None, None, 4096),
    "/api/records/deletions": ("deletions", 200, 131072),
    "/api/records/delete": (None, None, 4096),
    "/api/runs/{run_id:path}/control": (None, None, 4096),
    "/api/runs/{run_id:path}/budget": (None, None, 4096),
    "/api/runs/{run_id:path}/harness/convert": (None, None, 65536),
    "/api/runs/{run_id:path}/harness/{key}/review": (None, None, 65536),
    "/api/runs/{run_id:path}/harness/{key}/promote": (None, None, 65536),
    "/api/runs/{run_id:path}/feedback/{seq}/claims/{digest}/review": (None, None, 65536),
}


def _routes() -> set[str]:
    """@brief 应用实际注册的路由路径集合。

    @return 路径集合。
    """
    from observe.app import app

    return {route.path for route in app.routes}


def test_the_cursor_table_covers_every_route_except_the_whitelist():
    """@brief ``app.routes`` 与显式表的差集必须正好是 /、/reviews、/api/v2/stream、/api/v2/track-explore。

    @raises AssertionError 新路由没登记或表里有过期路径时抛出。
    """
    routes = _routes()
    assert routes - set(CURSOR_ENDPOINTS) == WHITELIST, routes - set(CURSOR_ENDPOINTS)
    assert set(CURSOR_ENDPOINTS) - routes == set(), set(CURSOR_ENDPOINTS) - routes
    assert "/api/history" not in routes and "/api/runs/{run_id:path}" not in routes, "v1 读端点必须已下线"


def _seed(runs: Path) -> str:
    """@brief 造一场带候选、反馈与实验的合成运行，供各端点做最小请求。

    @param runs ``runs/`` 根。
    @return 运行 id。
    """
    directory = runs / "conv2d" / "run_a"
    directory.mkdir(parents=True)
    (directory / "state.json").write_text(json.dumps({
        "task": "conv2d", "dtype": "fp16", "phase": "done", "started_at": 1, "updated_at": 1,
        "pid": os.getpid(), "best_speedup": 1.2,
    }), encoding="utf-8")
    (directory / "result.json").write_text(json.dumps({
        "task": "conv2d", "dtype": "fp16", "stop_reason": "max_rounds",
        "spent_usd": 0.5, "budget_usd": 2.0, "best_speedup": 1.2,
    }), encoding="utf-8")
    (directory / "run_config.json").write_text(json.dumps({
        "task": "conv2d", "dtype": "fp16", "size": "paper",
        "target": {"id": "apple-m4-max"},
        "committees": [{"name": "first", "members": [{"model": "m"}]}],
    }), encoding="utf-8")
    candidate = directory / "candidates" / "000_r1.p0"
    candidate.mkdir(parents=True)
    (candidate / "kernel.metal").write_text("kernel void k() {}\n", encoding="utf-8")
    (candidate / "report.json").write_text(json.dumps({"schema_version": 4, "status": "ok", "speedup": 1.2}), encoding="utf-8")
    # 一颗失败的候选：失败簇详情端点只有在有成员时才是 200，否则只能测到 404 错误体。
    failing = directory / "candidates" / "001_r1.p1"
    failing.mkdir(parents=True)
    (failing / "kernel.metal").write_text("kernel void k() {}\n", encoding="utf-8")
    (failing / "report.json").write_text(json.dumps({
        "schema_version": 4,
        "status": "incorrect", "stage": "correctness", "speedup": 0.5,
        "message": "worst: out got 1.5e-3",
    }), encoding="utf-8")
    (directory / "feedback.jsonl").write_text(json.dumps({
        "format": "agent_feedback", "seq": 1, "t": 1, "label": "r1.p0", "round": 1, "committee": "first",
        "items": [{"digest": "d1", "section": "对系统的建议", "audience": "system", "premise": "",
                   "text": "建议：把某一步做窄。"}],
    }) + "\n", encoding="utf-8")
    experiment = runs / "exp"
    experiment.mkdir()
    (experiment / "experiment.json").write_text(json.dumps({
        "schema_version": 1, "experiment_id": "exp", "kind": "search_ab", "driver": "search_ab",
        "created_at": 1, "git_commit": "x", "frozen": {"tasks": ["conv2d"]},
        "config_hash": "x", "arms": [{"arm_id": "on", "delta": {}, "planned_n": 1, "runs": []}],
        "budget_extensions": [],
    }), encoding="utf-8")
    return "conv2d/run_a"


def _fill(path: str, run_id: str) -> str:
    """@brief 把路由模板填成最小请求路径。

    @param path 路由模板。
    @param run_id 合成运行 id。
    @return 可请求的路径。
    """
    filled = path
    filled = filled.replace("{run_id:path}", run_id).replace("{experiment_id:path}", "exp")
    from observe import failure_sig

    filled = filled.replace("{sha256}", "0" * 64)
    filled = filled.replace("{sig}", failure_sig.signature("incorrect", "correctness", "worst: out got 1.5e-3"))
    filled = filled.replace("{seq:int}", "1").replace("{seq}", "1")
    filled = filled.replace("{label}", "r1.p0").replace("{key}", "k").replace("{digest}", "d1")
    return filled.replace("{", "").replace("}", "")


def _request(port: int, path: str, *, headers: dict | None = None, body: bytes | None = None) -> tuple[int, bytes, dict]:
    """@brief 发一次最小 HTTP 请求。

    @param port 服务端口。
    @param path 请求路径。
    @param headers 额外请求头。
    @param body 请求体；None 表示 GET。
    @return ``(状态码, 响应体, 响应头)``。
    """
    request = urllib.request.Request("http://127.0.0.1:%d%s" % (port, path), data=body, headers=headers or {})
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return response.status, response.read(), dict(response.headers)
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read(), dict(exc.headers)


def test_every_route_answers_a_minimal_request_within_its_declared_bound(dashboard):  # noqa: F811
    """@brief 每个路由发一次最小请求：状态 <500、响应字节与列表长度都在声明上限内。

    @param dashboard dashboard fixture。
    """
    port, runs = dashboard
    run_id = _seed(runs)
    query.refresh()
    from observe import actions

    token = actions.page_token({})
    payload = b"{}"
    for path, (_key, max_items, max_bytes) in CURSOR_ENDPOINTS.items():
        if path == "/api/v2/stream":
            continue
        target = _fill(path, run_id)
        if path in {"/api/v2/feedback", "/api/v2/runs", "/api/v2/experiments", "/api/records/deletions"}:
            target += "&limit=100" if "?" in target else "?limit=100"
        headers = {actions.TOKEN_HEADER: token, "Content-Type": "application/json"}
        if path.startswith("/api/v2/gap") or path.startswith("/api/v2/committees") or path.startswith("/api/v2/noise"):
            # 分析面板必须带纪元，否则 400 的错误体会让字节上限变成空断言。
            target += "&protocol_epoch=3" if "?" in target else "?protocol_epoch=3"
        status, body, _headers = _request(port, target, headers=headers, body=None if "/budget" not in path or path.endswith("budget-events") else None)
        if status in (405,):
            status, body, _headers = _request(port, target, headers=headers, body=payload)
        assert status < 500, (path, status, body[:200])
        assert len(body) <= max_bytes, (path, len(body), max_bytes)
        if not _key:
            continue
        assert status == 200, (path, status, body[:200])
        doc = json.loads(body)
        assert isinstance(doc, dict), (path, type(doc))
        value = doc
        for part in _key.split("."):
            assert isinstance(value, dict) and part in value, (path, _key, sorted(value) if isinstance(value, dict) else type(value))
            value = value[part]
        if isinstance(value, list):
            assert len(value) <= max_items, (path, len(value), max_items)


def test_a_304_does_not_read_index_files(dashboard):  # noqa: F811
    """@brief ETag 命中 304 时索引不读文件、不列目录。

    @param dashboard dashboard fixture。
    """
    port, runs = dashboard
    _seed(runs)
    query.refresh()
    status, body, headers = _request(port, "/api/v2/runs?limit=10")
    assert status == 200 and body
    etag = headers.get("etag")
    assert etag
    index = get_index()
    index._prepare_frame()
    status, second, _headers = _request(port, "/api/v2/runs?limit=10", headers={"If-None-Match": etag})
    assert status == 304 and second == b""
    assert index.stats["json_reads"] == 0, index.stats
    assert index.stats["dir_lists"] == 0, index.stats


def test_both_lru_caches_evict_and_keep_returning_the_same_rows(tmp_path, monkeypatch):
    """@brief 两个缓存按 256 上限淘汰最旧项；淘汰后同一场的结果逐字段一致。

    @param tmp_path pytest 临时目录。
    @param monkeypatch pytest 替换工具。
    """
    for number in range(query._CACHE_LIMIT + 20):
        query._cache_put(query._FEATURE_CACHE, f"digest-{number}", ("sig",))
    assert len(query._FEATURE_CACHE) == query._CACHE_LIMIT
    assert query._FEATURE_CACHE.get("digest-0") is None
    assert query._FEATURE_CACHE.get(f"digest-{query._CACHE_LIMIT + 19}") == ("sig",)

    runs = tmp_path / "runs"
    monkeypatch.setenv("AI_LOWERLING_RUNS_DIR", str(runs))
    monkeypatch.setenv("OBSERVE_DB", str(tmp_path / "observe.sqlite"))
    monkeypatch.setenv("AI_LOWERLING_BOOKS", str(tmp_path / "none.sqlite"))
    monkeypatch.setenv("OBSERVE_CARRYOVER", str(tmp_path / "carry.json"))
    _seed(runs)
    Index(tmp_path / "observe.sqlite").sync_known(force=True)
    directory = runs / "conv2d" / "run_a"
    result = json.loads((directory / "result.json").read_text(encoding="utf-8"))
    first = query.feature_grid(directory, result)
    assert first and len(query._GRID_CACHE) <= query._CACHE_LIMIT
    for number in range(query._CACHE_LIMIT + 5):
        query._cache_put(query._GRID_CACHE, ("other", number), [])
    assert len(query._GRID_CACHE) == query._CACHE_LIMIT
    again = query.feature_grid(directory, result)
    assert again == first


def test_suggestions_reads_the_harness_tables_once_for_twenty_groups(tmp_path, monkeypatch):
    """@brief 20 组建议共用一个运行目录时，提案/评审两份 JSONL 总共只读 2 次。

    @param tmp_path pytest 临时目录。
    @param monkeypatch pytest 替换工具。
    """
    runs = tmp_path / "runs"
    monkeypatch.setenv("AI_LOWERLING_RUNS_DIR", str(runs))
    monkeypatch.setenv("OBSERVE_DB", str(tmp_path / "observe.sqlite"))
    monkeypatch.setenv("AI_LOWERLING_BOOKS", str(tmp_path / "none.sqlite"))
    monkeypatch.setenv("OBSERVE_CARRYOVER", str(tmp_path / "carry.json"))
    _seed(runs)
    directory = runs / "conv2d" / "run_a"
    index = Index(tmp_path / "observe.sqlite")
    index.sync_known(force=True)
    with index.conn as conn:
        conn.execute("DELETE FROM feedback_claims")
        conn.execute("DELETE FROM feedback_items")
        for number in range(20):
            digest = f"d{number:02d}"
            conn.execute(
                "INSERT OR REPLACE INTO feedback_items (run_id, seq, t, task, dtype, model, label, line_at, has_system)"
                " VALUES (?,?,?,?,?,?,?,?,1)",
                ("conv2d/run_a", number + 1, 100 + number, "conv2d", "fp16", "m", "r1.p0", 0),
            )
            conn.execute(
                "INSERT OR REPLACE INTO feedback_claims (run_id, seq, digest, section, audience, status, rev)"
                " VALUES (?,?,?,?,?,?,?)",
                ("conv2d/run_a", number + 1, digest, "对系统的建议", "system", "pending", 1),
            )
    (directory / "harness_proposals.jsonl").write_text(
        "\n".join(json.dumps({"claim_digest": f"d{number:02d}", "seq": number + 1, "status": "valid",
                              "proposal_key": f"k{number}"}) for number in range(20)) + "\n",
        encoding="utf-8")
    (directory / "harness_reviews.jsonl").write_text(
        "\n".join(json.dumps({"proposal_key": f"k{number}", "useful": True}) for number in range(20)) + "\n",
        encoding="utf-8")
    calls = []
    original = suggestions._read_jsonl

    def counting(path):
        calls.append(str(path))
        return original(path)

    monkeypatch.setattr(suggestions, "_read_jsonl", counting)
    monkeypatch.setattr(suggestions, "get_index", lambda: index)
    monkeypatch.setattr(suggestions, "borrow", lambda: _Borrow(index))
    doc = suggestions.compact(limit=20)
    assert doc["total_groups"] == 20
    assert len(calls) == 2, calls
    assert all(group["converted_key"].startswith("k") for group in doc["groups"])


class _Borrow:
    """@brief 给 suggestions 的 ``borrow()`` 提供固定索引的上下文。"""

    def __init__(self, index: Index) -> None:
        self.index = index

    def __enter__(self) -> Index:
        return self.index

    def __exit__(self, *_exc) -> None:
        return None


def test_the_deletions_page_is_cursor_bounded_even_with_huge_reasons(tmp_path, monkeypatch):
    """@brief 200 条 8KB 理由的删除日志：一页 ≤ 条数/字节上限，游标可翻完，字段被截断。

    @param tmp_path pytest 临时目录。
    @param monkeypatch pytest 替换工具。
    """
    from observe import deletions

    monkeypatch.setenv("OBSERVE_DB", str(tmp_path / "observe.sqlite"))
    path = deletions.log_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for number in range(200):
            handle.write(json.dumps({
                "archive": str(tmp_path / "deleted" / f"{number:04d}"),
                "kind": "run",
                "name": f"conv2d/run_{number:04d}",
                "reason": f"理由 {number} " + "x" * 8000,
                "runs": [f"conv2d/run_{number:04d}"],
                "ip": "127.0.0.1",
                "t": 1_700_000_000 + number,
                "who": "dashboard",
            }, ensure_ascii=False) + "\n")

    page = deletions.list_deletions(limit=100)
    assert page["truncated"] is True and page["next"]
    assert len(page["deletions"]) <= 100
    assert len(json.dumps(page, ensure_ascii=False).encode("utf-8")) <= deletions.DELETIONS_BYTES + 4096
    assert all(len(row["reason"]) <= 512 for row in page["deletions"])

    seen = len(page["deletions"])
    cursor = page["next"]
    pages = 1
    while cursor:
        page = deletions.list_deletions(limit=100, cursor=cursor)
        pages += 1
        assert len(json.dumps(page, ensure_ascii=False).encode("utf-8")) <= deletions.DELETIONS_BYTES + 4096
        assert all(len(row["reason"]) <= 512 for row in page["deletions"])
        seen += len(page["deletions"])
        cursor = page["next"]
        assert pages < 50
    assert seen == 200
    assert page["truncated"] is False


def test_the_legacy_stream_alias_is_removed_and_returns_404(dashboard):  # noqa: F811
    """@brief §11.5 白名单只有 `/api/v2/stream`：旧 `/api/stream` 路由必须真的不存在。

    @param dashboard dashboard fixture。
    """
    port, _runs = dashboard
    assert "/api/stream" not in _routes()
    status, body, _headers = _request(port, "/api/stream")
    assert status == 404, (status, body[:200])


def test_patrol_is_a_real_http_endpoint_with_a_fixed_small_shape(dashboard):  # noqa: F811
    """@brief 首屏三区请求的 `/api/v2/patrol` 必须 200、固定键集合、响应 ≤1536B。

    只测 `query.patrol()` 会漏掉路由未接线的回归；这里走真实 HTTP。

    @param dashboard dashboard fixture。
    """
    port, runs = dashboard
    _seed(runs)
    query.refresh()
    status, body, _headers = _request(port, "/api/v2/patrol")
    assert status == 200, (status, body[:200])
    assert len(body) <= 1536, len(body)
    doc = json.loads(body)
    assert set(doc) == {
        "schema_version", "at", "index_revision", "protocol_epochs", "alerts",
        "live", "failing", "best", "truncated",
    }, set(doc)
    assert set(doc["live"]) == {"runs", "evals", "gpu"}
    assert set(doc["failing"]) == {"total", "clusters"}
    assert set(doc["truncated"]) == {"clusters", "best", "alerts", "epochs"}
