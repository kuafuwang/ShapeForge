"""Local dashboard. Binds to 127.0.0.1 and never starts a kernel.

Reads come from the SQLite index. The idle stream polls about once a second and sends a revision,
not a fresh copy of every run. Changing this file still needs a restart; changing ``static/index.html``
no longer does, because the template is read per request (mtime-keyed) instead of at import.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
from pathlib import Path

from starlette.applications import Starlette
from starlette.middleware.gzip import GZipMiddleware
from starlette.responses import HTMLResponse, JSONResponse, StreamingResponse
from starlette.routing import Route

from observe import actions, budget, hmz_web, query, track_explore as track_explore_reader



class ETagMiddleware:
    """@brief 给缓冲式响应加 ETag，并用 304 回应 ``If-None-Match``；SSE 流不处理。

    原始英文说明：Add an ETag to buffered responses and answer ``If-None-Match`` with 304. SSE is left alone.
    """

    def __init__(self, app) -> None:
        """@brief 记下被包装的 ASGI 应用。

        @param app 内层 ASGI 应用。
        """
        self.app = app

    async def __call__(self, scope, receive, send):
        """@brief 执行一次 ASGI 调用：缓冲响应体、计算 ETag，命中时返回 304。

        @param scope ASGI 连接作用域。
        @param receive ASGI 接收通道。
        @param send ASGI 发送通道。
        """
        if scope.get("type") != "http" or str(scope.get("path") or "").endswith("/stream"):
            await self.app(scope, receive, send)
            return
        start = None
        chunks: list[bytes] = []

        async def capture(message):
            nonlocal start
            if message["type"] == "http.response.start":
                start = message
            elif message["type"] == "http.response.body":
                chunks.append(message.get("body") or b"")

        await self.app(scope, receive, capture)
        if start is None:
            return
        body = b"".join(chunks)
        etag = '"' + hashlib.sha1(body).hexdigest() + '"'
        headers = [(key, value) for key, value in start.get("headers") or [] if key.lower() not in (b"etag", b"cache-control", b"content-length")]
        headers.append((b"etag", etag.encode("ascii")))
        headers.append((b"cache-control", b"no-cache"))
        incoming = {key.lower(): value for key, value in scope.get("headers") or []}
        if incoming.get(b"if-none-match", b"").decode("latin-1").strip() == etag:
            await send({"type": "http.response.start", "status": 304, "headers": headers})
            await send({"type": "http.response.body", "body": b""})
            return
        await send({"type": "http.response.start", "status": start["status"], "headers": headers})
        await send({"type": "http.response.body", "body": body})


_PAGE_PATH = Path(__file__).parent / "static" / "index.html"
_BANNER_SLOT = "<!--BUDGET_BANNER-->"
_TOKEN_SLOT = "<!--OBSERVE_TOKEN-->"
# (路径, mtime_ns, size) -> 模板全文。改文件不用重启进程，下一次请求就重读。
_page_cache: tuple[tuple[Path, int, int], str] | None = None


def _page_template() -> str:
    """@brief 读首页模板，文件一变就重读，所以改 ``static/index.html`` 后刷新即生效。

    以前这份 HTML 在 import 期被读成模块常量，看板进程常驻几天就会一直吐启动那天的界面。
    编辑器原地替换文件的一瞬间可能读到残缺内容或 OSError，此时沿用上一份好的模板。

    @return 页面模板全文。
    """
    global _page_cache
    try:
        stat = _PAGE_PATH.stat()
    except OSError:
        if _page_cache is not None:
            return _page_cache[1]
        raise
    key = (_PAGE_PATH, stat.st_mtime_ns, stat.st_size)
    if _page_cache is None or _page_cache[0] != key:
        try:
            text = _PAGE_PATH.read_text(encoding="utf-8")
        except OSError:
            if _page_cache is not None:
                return _page_cache[1]
            raise
        _page_cache = (key, text)
    return _page_cache[1]


def _json(data: dict | list, status: int = 200) -> JSONResponse:
    """@brief 把数据包成指定状态码的 JSON 响应。

    @param data 响应数据。
    @param status HTTP 状态码，默认为 200。
    @return JSON 响应对象。
    """
    return JSONResponse(data, status_code=status)


def _poll_seconds() -> float:
    """@brief 取空闲流的轮询间隔，来自 ``OBSERVE_POLL_S``，下限 0.05 秒。

    @return 轮询间隔秒数；环境变量非法时返回 1.0。
    """
    try:
        return max(0.05, float(os.environ.get("OBSERVE_POLL_S", "1")))
    except ValueError:
        return 1.0


async def index(request) -> HTMLResponse:
    """@brief 渲染首屏 HTML，把预算横幅与观察 token 填进页面模板。

    索引或运行目录损坏时只丢掉横幅，页面仍可打开。

    @param request Starlette 请求对象。
    @return 首页 HTML 响应（禁用缓存）。
    """
    try:
        await asyncio.to_thread(query.tick)
        banner = budget.render_banner_html(await asyncio.to_thread(query.budget_report))
    except Exception:  # noqa: BLE001  a damaged run directory must never take the page down
        banner = ""
    token = actions.page_token(request.headers)
    template = await asyncio.to_thread(_page_template)
    page = template.replace(_BANNER_SLOT, banner).replace(_TOKEN_SLOT, f'<meta name="observe-token" content="{token}">')
    return HTMLResponse(page, headers={"Cache-Control": "no-store"})


async def api_v2_overview(_request) -> JSONResponse:
    """@brief 返回总览统计。

    @param _request 未使用的请求对象。
    @return 总览 JSON 响应。
    """
    return _json(await asyncio.to_thread(query.overview))


async def api_v2_track_explore(_request) -> JSONResponse:
    """@brief Return the archived three-Track exploration, never a publication authority."""
    evidence = Path(__file__).resolve().parents[1] / "tests/support/golden/shared/多DSL-Track-阶段验收/MT6/evidence/track-explore-r15"
    return _json(await asyncio.to_thread(track_explore_reader.read_track_explore, evidence))


async def api_v2_patrol(_request) -> JSONResponse:
    """@brief 首屏『进行中 / 需要注意 / 最近结果』的巡检读模型（§6.4，响应 ≤1.5KB）。

    @param _request 未使用的请求对象。
    @return 固定键集合的巡检 JSON 响应。
    """
    return _json(await asyncio.to_thread(query.patrol))


async def api_v2_runs(request) -> JSONResponse:
    """@brief 按查询参数分页列出运行；非法 limit/offset 退回 50/0。

    @param request Starlette 请求对象，查询参数含筛选、排序与游标字段。
    @return 运行分页 JSON 响应。
    """
    params = request.query_params
    try:
        limit = int(params.get("limit", "50"))
        offset = int(params.get("offset", "0"))
    except ValueError:
        limit, offset = 50, 0
    page = await asyncio.to_thread(
        query.runs_page,
        experiment_id=params.get("experiment_id") or None,
        task=params.get("task") or None,
        dtype=params.get("dtype") or None,
        model=params.get("model") or None,
        status=params.get("status") or None,
        target=params.get("target") or None,
        publication=params.get("publication") or None,
        min_speedup=params.get("min_speedup") or None,
        since=params.get("since") or None,
        sort=params.get("sort") or "started",
        order=params.get("order") or "desc",
        cursor=params.get("cursor") or None,
        limit=limit,
        offset=offset,
    )
    return _json(page)


async def api_v2_candidates(request) -> JSONResponse:
    """@brief 列出一个运行的候选内核（候选面板：候选行 + 轨迹标记）。

    打开这块之前不读任何 ``report.json``；``stamp`` 相同则只回标记。支持 ``status``/``stage``
    过滤与 ``limit``/``cursor`` 翻页；戳始终覆盖全部候选，过滤不改变新鲜度判断。

    @param request Starlette 请求对象，路径含 ``run_id``，查询参数可含 ``status``/``stage``/``limit``/``cursor``/``stamp``。
    @return 候选面板 JSON 响应；运行不存在时返回 404。
    """
    params = request.query_params
    raw_limit = params.get("limit")
    try:
        limit = int(raw_limit) if raw_limit else query.CANDIDATES_PAGE
    except ValueError:
        return _json({"error": "limit must be an integer"}, status=400)
    doc = await asyncio.to_thread(
        query.candidates_panel,
        request.path_params["run_id"],
        status=params.get("status") or None,
        stage=params.get("stage") or None,
        limit=limit,
        cursor=params.get("cursor") or None,
        stamp=params.get("stamp") or "",
        label=params.get("label") or None,
    )
    if doc is None:
        return _json({"error": "not found"}, status=404)
    return _json(doc)


async def api_v2_sessions(request) -> JSONResponse:
    """@brief 列出一个运行的模型会话；``stamp`` 相同则不回正文。

    @param request Starlette 请求对象，路径含 ``run_id``，查询参数可含 ``limit`` / ``stamp``。
    @return 会话列表 JSON 响应。
    """
    return _json(await asyncio.to_thread(
        query.sessions_page,
        request.path_params["run_id"],
        limit=int(request.query_params.get("limit", "500") or 500),
        stamp=request.query_params.get("stamp") or "",
    ))


async def api_v2_model_stats(request) -> JSONResponse:
    """@brief 返回按模型聚合的统计；可用 ``run`` 限定一个运行。

    @param request Starlette 请求对象，查询参数可含 ``run`` 或 ``run_id``。
    @return 模型统计 JSON 响应。
    """
    run_id = request.query_params.get("run") or request.query_params.get("run_id") or None
    return _json(await asyncio.to_thread(query.model_stats, run_id))


async def api_v2_search(request) -> JSONResponse:
    """@brief §10.3 跨表搜索：runs/candidates 走 FTS5，失败簇走小表。

    @param request Starlette 请求对象，查询参数含 ``q``，可含 ``kind``/``limit``。
    @return 搜索 JSON 响应。
    """
    params = request.query_params
    try:
        limit = int(params.get("limit", "20") or 20)
    except ValueError:
        limit = 20
    return _json(await asyncio.to_thread(
        query.search,
        q=params.get("q") or "",
        kind=params.get("kind") or "runs",
        limit=limit,
    ))


async def api_v2_failures(request) -> JSONResponse:
    """@brief 列出失败的候选。

    @param request Starlette 请求对象，查询参数可含 ``status`` / ``task`` / ``limit``。
    @return 失败列表 JSON 响应。
    """
    params = request.query_params
    return _json(await asyncio.to_thread(
        query.failures,
        status=params.get("status") or None,
        task=params.get("task") or None,
        sig=params.get("sig") or None,
        run=params.get("run") or None,
        label=params.get("label") or None,
        limit=int(params.get("limit", "50") or 50),
    ))


async def api_v2_publications(request) -> JSONResponse:
    """@brief 发布史：目录事实、环比、当前发布判定与来源可用性（§8.1）。

    @param request Starlette 请求对象，查询参数可含 ``task``/``target``/``dtype``/``size``/``stamp``。
    @return 发布史 JSON 响应。
    """
    from observe import publications

    params = request.query_params
    doc = await asyncio.to_thread(
        publications.publications,
        task=params.get("task") or None,
        target=params.get("target") or None,
        dtype=params.get("dtype") or None,
        size=params.get("size") or None,
        stamp=params.get("stamp") or "",
    )
    return _json(doc)


async def api_v2_publications_source(request):
    """@brief 按目录史摘要返回确认内核源码（§9.1：blob 是持久证据，源目录删了也还在）。

    @param request Starlette 请求对象，路径含 ``sha256``。
    @return 纯文本源码；摘要非法或 blob 不存在时返回 404。
    """
    from starlette.responses import PlainTextResponse

    from taic.blobs import read_source

    source = await asyncio.to_thread(read_source, request.path_params["sha256"])
    if source is None:
        return _json({"error": "not found"}, status=404)
    return PlainTextResponse(source, media_type="text/plain; charset=utf-8")


async def api_v2_gap(request) -> JSONResponse:
    """@brief 落差面板：每场已发布运行的 search→published，以及每美元口径。

    @param request Starlette 请求对象，查询参数可含 ``protocol_epoch``/``task``/``target``/``stamp``。
    @return 面板 JSON；跨协议纪元又没指定 epoch 时返回 400。
    """
    from observe import analysis

    params = request.query_params
    try:
        doc = await asyncio.to_thread(
            analysis.gap,
            protocol_epoch=params.get("protocol_epoch") or None,
            task=params.get("task") or None,
            target=params.get("target") or None,
            stamp=params.get("stamp") or "",
        )
    except analysis.EpochRequired as exc:
        return _json({"error": "protocol_epoch required", "epochs": exc.epochs}, status=400)
    return _json(doc)


async def api_v2_committees(request) -> JSONResponse:
    """@brief 委员会有效性面板：单位是 session，按 task + protocol_epoch 分桶。

    @param request Starlette 请求对象，查询参数可含 ``task``/``stamp``。
    @return 面板 JSON 响应。
    """
    from observe import analysis

    params = request.query_params
    try:
        doc = await asyncio.to_thread(
            analysis.committees,
            task=params.get("task") or None,
            protocol_epoch=params.get("protocol_epoch") or None,
            stamp=params.get("stamp") or "",
        )
    except analysis.EpochRequired as exc:
        return _json({"error": "protocol_epoch required", "epochs": exc.epochs}, status=400)
    return _json(doc)


async def api_v2_noise(request) -> JSONResponse:
    """@brief 噪声面板：候选 cv 的分位数、reps 分布与 cache_hit 覆盖率。

    @param request Starlette 请求对象，查询参数可含 ``task``/``stamp``。
    @return 面板 JSON 响应。
    """
    from observe import analysis

    params = request.query_params
    try:
        doc = await asyncio.to_thread(
            analysis.noise,
            task=params.get("task") or None,
            protocol_epoch=params.get("protocol_epoch") or None,
            stamp=params.get("stamp") or "",
        )
    except analysis.EpochRequired as exc:
        return _json({"error": "protocol_epoch required", "epochs": exc.epochs}, status=400)
    return _json(doc)


async def api_v2_failure_clusters(request) -> JSONResponse:
    """@brief 列出失败簇（按签名聚合，从成员表派生）。

    @param request Starlette 请求对象，查询参数可含 ``task`` / ``stage`` / ``limit``。
    @return 失败簇 JSON 响应；``limit`` 非整数时返回 400。
    """
    params = request.query_params
    raw = params.get("limit")
    try:
        limit = int(raw) if raw else 20
    except ValueError:
        return _json({"error": "limit must be an integer"}, status=400)
    return _json(await asyncio.to_thread(
        query.failure_clusters,
        task=params.get("task") or None,
        stage=params.get("stage") or None,
        limit=limit,
    ))


async def api_v2_failure_cluster(request) -> JSONResponse:
    """@brief 返回一个失败簇的证据行（只从索引读）。

    @param request Starlette 请求对象，路径含 ``sig``，查询参数可含 ``limit``。
    @return 失败簇详情 JSON 响应；``limit`` 非整数时返回 400。
    """
    raw = request.query_params.get("limit")
    try:
        limit = int(raw) if raw else 20
    except ValueError:
        return _json({"error": "limit must be an integer"}, status=400)
    doc = await asyncio.to_thread(query.failure_cluster, request.path_params["sig"], limit=limit)
    if doc is None:
        return _json({"error": "not found"}, status=404)
    return _json(doc)


async def api_v2_best(request) -> JSONResponse:
    """@brief 列出已通过确认门的最优结果。

    @param request Starlette 请求对象，查询参数可含 ``task`` / ``target`` / ``limit``。
    @return 最优结果 JSON 响应。
    """
    params = request.query_params
    return _json(await asyncio.to_thread(
        query.best_confirmed,
        task=params.get("task") or None,
        target=params.get("target") or None,
        limit=int(params.get("limit", "20") or 20),
    ))


async def api_v2_run(request) -> JSONResponse:
    """@brief 返回单个运行的 L1 摘要；重面板各自有端点。找不到时返回 404。

    完整详情（旧行为）仍在 ``/api/v2/runs/{id}/detail`` 与 v1 的 ``/api/runs/{id}``。

    @param request Starlette 请求对象，路径含 ``run_id``，查询参数可含 ``stamp``。
    @return 摘要 JSON 响应；运行不存在时返回 404。
    """
    doc = await asyncio.to_thread(
        query.run_summary, request.path_params["run_id"], stamp=request.query_params.get("stamp") or "",
    )
    if doc is None:
        return _json({"error": "not found"}, status=404)
    return _json(doc)


async def _panel(loader, request) -> JSONResponse:
    """@brief 面板端点的共同外壳：带 ``stamp`` 调用并保留 404。

    @param loader 面板查询函数（除 ``run_id`` 外只接受 ``stamp``）。
    @param request Starlette 请求对象，路径含 ``run_id``，查询参数可含 ``stamp``。
    @return 面板 JSON 响应；运行不存在时返回 404。
    """
    doc = await asyncio.to_thread(loader, request.path_params["run_id"], stamp=request.query_params.get("stamp") or "")
    if doc is None:
        return _json({"error": "not found"}, status=404)
    return _json(doc)


async def api_v2_run_detail(request) -> JSONResponse:
    """@brief 一次返回完整详情（摘要 + 全部重面板）；v1 与调试用的旧行为。

    @param request Starlette 请求对象，路径含 ``run_id``，查询参数可含 ``stamp``。
    @return 运行详情 JSON 响应；运行不存在时返回 404。
    """
    doc = await asyncio.to_thread(
        query.run_detail, request.path_params["run_id"], stamp=request.query_params.get("stamp") or "",
    )
    if doc is None:
        return _json({"error": "not found"}, status=404)
    return _json(doc)


async def api_v2_run_candidate(request) -> JSONResponse:
    """@brief 返回单颗候选的完整报告、有界内核片段与相对报告路径。

    @param request Starlette 请求对象，路径含 ``run_id`` 与 ``label``，查询参数可含 ``stamp``。
    @return 候选 JSON 响应；运行或候选不存在时返回 404。
    """
    doc = await asyncio.to_thread(
        query.candidate_panel,
        request.path_params["run_id"],
        request.path_params["label"],
        stamp=request.query_params.get("stamp") or "",
    )
    if doc is None:
        return _json({"error": "not found"}, status=404)
    return _json(doc)


async def api_v2_run_feedback(request) -> JSONResponse:
    """@brief 返回反馈面板：条目与评审状态都在，但不带每条记录的 ``raw`` 全文。

    记录按页返回（缺省 10 条/页），``limit``/``cursor`` 只影响本页，不改变面板戳的覆盖范围。

    @param request Starlette 请求对象，路径含 ``run_id``，查询参数可含 ``stamp``/``limit``/``cursor``。
    @return 反馈面板 JSON 响应；运行不存在时返回 404。
    """
    try:
        limit = int(request.query_params.get("limit") or query.FEEDBACK_PAGE)
        raw_seq = request.query_params.get("seq")
        seq = int(raw_seq) if raw_seq not in (None, "") else None
    except ValueError:
        return _json({"error": "limit/seq must be an integer"}, status=400)
    doc = await asyncio.to_thread(
        query.feedback_panel,
        request.path_params["run_id"],
        stamp=request.query_params.get("stamp") or "",
        limit=limit,
        cursor=request.query_params.get("cursor") or None,
        seq=seq,
    )
    if doc is None:
        return _json({"error": "not found"}, status=404)
    return _json(doc)


async def api_v2_run_feedback_record(request) -> JSONResponse:
    """@brief 返回一条反馈记录的全文（含 ``raw``、claim 正文与校准）。

    @param request Starlette 请求对象，路径含 ``run_id`` 与整数 ``seq``，查询参数可含 ``stamp``。
    @return 单条记录 JSON 响应；运行或序号不存在时返回 404。
    """
    doc = await asyncio.to_thread(
        query.feedback_record,
        request.path_params["run_id"],
        request.path_params["seq"],
        stamp=request.query_params.get("stamp") or "",
    )
    if doc is None:
        return _json({"error": "not found"}, status=404)
    return _json(doc)


async def api_v2_run_timeline(request) -> JSONResponse:
    """@brief 返回时间线面板（GPU 锁泳道与评测/问模型跨度）。

    @param request Starlette 请求对象，路径含 ``run_id``，查询参数可含 ``stamp``。
    @return 时间线面板 JSON 响应；运行不存在时返回 404。
    """
    return await _panel(query.timeline_panel, request)


async def api_v2_run_budget_events(request) -> JSONResponse:
    """@brief 返回预算事件面板。

    @param request Starlette 请求对象，路径含 ``run_id``，查询参数可含 ``stamp``。
    @return 预算事件面板 JSON 响应；运行不存在时返回 404。
    """
    return await _panel(query.budget_events_panel, request)


async def api_v2_run_llm(request) -> JSONResponse:
    """@brief 返回在飞模型请求预览面板。

    @param request Starlette 请求对象，路径含 ``run_id``，查询参数可含 ``stamp``。
    @return ``llm`` 面板 JSON 响应；运行不存在时返回 404。
    """
    return await _panel(query.llm_panel, request)


async def api_v2_run_grid(request) -> JSONResponse:
    """@brief 返回特征格子面板。

    @param request Starlette 请求对象，路径含 ``run_id``，查询参数可含 ``stamp``。
    @return 特征格子面板 JSON 响应；运行不存在时返回 404。
    """
    return await _panel(query.grid_panel, request)


async def api_v2_run_lineage(request) -> JSONResponse:
    """@brief 返回种子链与跨场同源面板（§8.3）。

    @param request Starlette 请求对象，路径含 ``run_id``，查询参数可含 ``stamp``。
    @return 面板 JSON 响应；运行不存在时返回 404。
    """
    from observe import lineage

    return await _panel(lineage.lineage_panel, request)


async def api_v2_run_plan(request) -> JSONResponse:
    """@brief 返回计划结局面板。

    @param request Starlette 请求对象，路径含 ``run_id``，查询参数可含 ``stamp``。
    @return 计划结局面板 JSON 响应；运行不存在时返回 404。
    """
    return await _panel(query.plan_panel, request)


async def api_v2_run_raw(request) -> JSONResponse:
    """@brief 返回原始文件（``?key=state|result|confirm``），显式调试用。

    @param request Starlette 请求对象，路径含 ``run_id``，查询参数含 ``key``，可含 ``stamp``。
    @return 原始文件 JSON 响应；运行不存在或 key 不合法时返回 404。
    """
    doc = await asyncio.to_thread(
        query.raw_panel,
        request.path_params["run_id"],
        request.query_params.get("key") or "",
        stamp=request.query_params.get("stamp") or "",
    )
    if doc is None:
        return _json({"error": "not found"}, status=404)
    return _json(doc)


async def api_v2_turns(request) -> JSONResponse:
    """@brief 返回一段模型轮次窗口；``stamp`` 匹配时返回空行且不查轮次表。

    原始英文说明：One window of model turns. A matching stamp returns no rows and does not read the turns table.

    @param request Starlette 请求对象，路径含 ``run_id``，查询参数可含 ``before`` / ``grp`` / ``all`` / ``limit`` / ``stamp``。
    @return 轮次窗口 JSON 响应；运行不存在时返回 404。
    """
    from observe.detail import TURNS_PAGE, turns_feed

    params = request.query_params
    try:
        limit = int(params.get("limit", str(TURNS_PAGE)))
    except ValueError:
        limit = TURNS_PAGE
    feed = await asyncio.to_thread(
        turns_feed,
        request.path_params["run_id"],
        before=params.get("before"),
        limit=limit,
        stamp=params.get("stamp") or "",
        grp=params.get("grp"),
        everything=params.get("all") in ("1", "true", "yes"),
    )
    if feed is None:
        return _json({"error": "not found"}, status=404)
    return _json(feed)


async def api_v2_pi(request) -> JSONResponse:
    """@brief 返回一个运行的实时 pi 对话；独立轮询，避免重绘整页。

    原始英文说明：The live pi transcript for one run. Polled on its own so the rest of the page is not redrawn.

    @param request Starlette 请求对象，路径含 ``run_id``，查询参数可含 ``stamp``。
    @return 对话 feed JSON 响应；运行不存在时返回 404。
    """
    from observe.detail import pi_feed

    stamp = request.query_params.get("stamp") or ""
    feed = await asyncio.to_thread(pi_feed, request.path_params["run_id"], stamp=stamp)
    if feed is None:
        return _json({"error": "not found"}, status=404)
    return _json(feed)


async def api_v2_live(_request) -> JSONResponse:
    """@brief 按需的活状态总览（GPU、在跑的运行/评估、心跳、任务与实验队列）。

    §11.5：首屏 `/api/v2/overview` 只留反馈计数，这些字段要读机全局文件，必须显式按需取。

    @param _request 未使用的请求对象。
    @return 活状态 JSON 响应。
    """
    return _json(await asyncio.to_thread(query.live_overview))


async def api_v2_experiments(request) -> JSONResponse:
    """@brief 列出实验（keyset 游标）。

    @param request Starlette 请求对象，查询参数可含 ``limit`` / ``cursor``。
    @return 实验分页 JSON 响应。
    """
    params = request.query_params
    try:
        limit = int(params.get("limit", str(query.EXPERIMENTS_PAGE)))
    except ValueError:
        return _json({"error": "limit must be an integer"}, status=400)
    return _json(await asyncio.to_thread(query.experiments, limit=limit, cursor=params.get("cursor") or None))


async def api_v2_experiment(request) -> JSONResponse:
    """@brief 返回一个实验内各 arm 的对比。

    @param request Starlette 请求对象，路径含 ``experiment_id``。
    @return 实验对比 JSON 响应。
    """
    return _json(await asyncio.to_thread(query.compare, request.path_params["experiment_id"]))


async def api_v2_feedback(request) -> JSONResponse:
    """@brief 返回反馈看板；``since`` 不是整数时返回 400。

    @param request Starlette 请求对象，查询参数可含 ``since``。
    @return 反馈看板 JSON 响应。
    """
    params = request.query_params
    raw = params.get("since")
    since = None
    if raw not in (None, ""):
        try:
            since = int(raw)
        except ValueError:
            return _json({"error": "since must be an integer"}, status=400)
    try:
        limit = int(params.get("limit", "200"))
        offset = int(params.get("offset", "0"))
    except ValueError:
        return _json({"error": "limit and offset must be integers"}, status=400)
    return _json(await asyncio.to_thread(
        query.feedback_board, since=since, limit=limit, offset=offset,
        full=params.get("full") == "1",
        similar=params.get("similar") == "1",
    ))


async def api_v2_run_similar(request) -> JSONResponse:
    """@brief 返回一场运行里待审 claim 的语义近邻；展开该 run 时才调用，运行不存在返回 404。

    @param request Starlette 请求对象，路径含 ``run_id``。
    @return ``{run_id, items}`` JSON 响应。
    """
    doc = await asyncio.to_thread(query.feedback_similar, request.path_params["run_id"])
    if doc is None:
        return _json({"error": "not found"}, status=404)
    return _json(doc)


async def api_v2_suggestions(request) -> JSONResponse:
    """@brief 返回按 digest 聚合的系统建议预览；分页参数非整数时返回 400。

    @param request Starlette 请求对象，查询参数可含筛选与 ``limit`` / ``preview`` / ``offset``。
    @return 建议分组 JSON 响应。
    """
    params = request.query_params
    try:
        limit = int(params.get("limit", "20"))
        preview = int(params.get("preview", "120"))
        offset = int(params.get("offset", "0"))
    except ValueError:
        return _json({"error": "limit, preview and offset must be integers"}, status=400)
    return _json(await asyncio.to_thread(
        query.suggestions,
        task=params.get("task") or None,
        target=params.get("target") or None,
        model=params.get("model") or None,
        status=params.get("status") or None,
        limit=limit,
        preview=preview,
        offset=offset,
    ))


async def api_v2_suggestion(request) -> JSONResponse:
    """@brief 返回一条建议及其转换/评审详情；参数缺失返回 400，找不到返回 404。

    @param request Starlette 请求对象，查询参数需含 ``run`` / ``seq`` / ``digest``。
    @return 建议详情 JSON 响应。
    """
    run_id = request.query_params.get("run") or ""
    digest = request.query_params.get("digest") or ""
    try:
        seq = int(request.query_params.get("seq") or "")
    except ValueError:
        return _json({"error": "seq must be an integer"}, status=400)
    if not run_id or not digest:
        return _json({"error": "run and digest are required"}, status=400)
    try:
        doc = await asyncio.to_thread(query.suggestion, run_id, seq, digest)
    except (FileNotFoundError, KeyError):
        return _json({"error": "suggestion not found"}, status=404)
    return _json(doc)


async def api_v2_harness(request) -> JSONResponse:
    """@brief 返回一个运行的 harness 提案与评审看板；找不到时返回 404。

    @param request Starlette 请求对象，路径含 ``run_id``。
    @return harness 看板 JSON 响应。
    """
    doc = await asyncio.to_thread(query.harness_board, request.path_params["run_id"])
    if doc is None:
        return _json({"error": "not found"}, status=404)
    return _json(doc)


async def api_v2_budget(_request) -> JSONResponse:
    """@brief 返回预算报告。

    @param _request 未使用的请求对象。
    @return 预算报告 JSON 响应。
    """
    return _json(await asyncio.to_thread(query.budget_report))


async def api_v2_ledger(_request) -> JSONResponse:
    """@brief 返回花费账本报告。

    @param _request 未使用的请求对象。
    @return 账本报告 JSON 响应。
    """
    return _json(await asyncio.to_thread(query.ledger_report))


def _refused(exc: actions.Refused) -> JSONResponse:
    """@brief 把拒绝异常转成带状态码的错误响应。

    @param exc 动作层抛出的拒绝异常。
    @return 含 ``ok=False`` 与错误信息的 JSON 响应。
    """
    return _json({"ok": False, "error": exc.message}, status=exc.status)


async def api_control(request) -> JSONResponse:
    """@brief 返回一个运行可用的控制提示；非 loopback 请求返回 403。

    @param request Starlette 请求对象，路径含 ``run_id``。
    @return 控制提示 JSON 响应。
    """
    try:
        if not actions.host_is_loopback(request.headers):
            raise actions.Refused(403, "refused: this server only answers on 127.0.0.1 / localhost (Host header check)")
        return _json(await asyncio.to_thread(actions.describe, request.path_params["run_id"]))
    except actions.Refused as exc:
        return _refused(exc)


async def api_promote_harness(request) -> JSONResponse:
    """@brief 将一个 harness 提案提升为正式 skill 或任务文本。

    先校验来源与 token，再交给动作层执行。

    @param request Starlette 请求对象，路径含 ``run_id`` 与提案 ``key``。
    @return 提升结果 JSON 响应。
    """
    try:
        actions.check_origin(request.headers)
        actions.check_token(request.headers)
        promoted = await asyncio.to_thread(
            actions.promote_harness, request.path_params["run_id"], request.path_params["key"],
        )
        return _json(promoted)
    except actions.Refused as exc:
        return _refused(exc)


async def api_convert_harness(request) -> JSONResponse:
    """@brief 把一条系统建议转换成 harness 提案；请求体上限 16384 字节。

    @param request Starlette 请求对象，路径含 ``run_id``，请求体为 JSON。
    @return 转换结果 JSON 响应。
    """
    try:
        actions.check_origin(request.headers)
        actions.check_token(request.headers)
        raw = await request.body()
        if len(raw) > 16384:
            raise actions.Refused(413, "the body is too large")
        body = actions.parse_body(raw)
        try:
            seq = int(body.get("seq"))
        except (TypeError, ValueError) as exc:
            raise actions.Refused(400, "seq must be an integer") from exc
        converted = await asyncio.to_thread(
            actions.convert_harness, request.path_params["run_id"], seq, str(body.get("digest") or ""), body,
        )
        return _json(converted)
    except actions.Refused as exc:
        return _refused(exc)


async def api_review_harness(request) -> JSONResponse:
    """@brief 评审一个 harness 提案；请求体上限 8192 字节。

    @param request Starlette 请求对象，路径含 ``run_id`` 与提案 ``key``，请求体为 JSON。
    @return 评审结果 JSON 响应。
    """
    try:
        actions.check_origin(request.headers)
        actions.check_token(request.headers)
        raw = await request.body()
        if len(raw) > 8192:
            raise actions.Refused(413, "the body is too large")
        body = actions.parse_body(raw)
        reviewed = await asyncio.to_thread(
            actions.review_harness, request.path_params["run_id"], request.path_params["key"], body,
        )
        return _json(reviewed)
    except actions.Refused as exc:
        return _refused(exc)


async def api_review_feedback(request) -> JSONResponse:
    """@brief 评审一条反馈 claim；请求体上限 16384 字节。

    @param request Starlette 请求对象，路径含 ``run_id`` / ``seq`` / ``digest``，请求体为 JSON。
    @return 评审结果 JSON 响应。
    """
    try:
        actions.check_origin(request.headers)
        actions.check_token(request.headers)
        raw = await request.body()
        if len(raw) > 16384:
            raise actions.Refused(413, "the body is too large")
        body = actions.parse_body(raw)
        try:
            seq = int(request.path_params["seq"])
        except (TypeError, ValueError) as exc:
            raise actions.Refused(400, "seq must be an integer") from exc
        reviewed = await asyncio.to_thread(
            actions.review_feedback, request.path_params["run_id"], seq, request.path_params["digest"], body,
        )
        return _json(reviewed)
    except actions.Refused as exc:
        return _refused(exc)


async def api_add_budget(request) -> JSONResponse:
    """@brief 给一个运行追加预算；请求体上限 4096 字节。

    @param request Starlette 请求对象，路径含 ``run_id``，请求体为 JSON。
    @return 追加预算后的结果 JSON 响应。
    """
    try:
        actions.check_origin(request.headers)
        actions.check_token(request.headers)
        raw = await request.body()
        if len(raw) > 4096:
            raise actions.Refused(413, "the body is too large")
        body = actions.parse_body(raw)
        return _json(await asyncio.to_thread(actions.add_budget, request.path_params["run_id"], body))
    except actions.Refused as exc:
        return _refused(exc)


async def api_reset(request) -> JSONResponse:
    """@brief GET 返回重置预览，POST 执行预算重置。

    @param request Starlette 请求对象。
    @return 重置预览或执行结果 JSON 响应。
    """
    try:
        if request.method == "GET":
            if not actions.host_is_loopback(request.headers):
                raise actions.Refused(403, "refused: this server only answers on 127.0.0.1 / localhost (Host header check)")
            return _json(await asyncio.to_thread(actions.reset_preview))
        actions.check_origin(request.headers)
        actions.check_token(request.headers)
        return _json(await asyncio.to_thread(actions.reset_budgets))
    except actions.Refused as exc:
        return _refused(exc)


async def api_stop(request) -> JSONResponse:
    """@brief GET 返回待停进程预览，POST 停止正在跑的搜索。

    @param request Starlette 请求对象。
    @return 停止预览或执行结果 JSON 响应。
    """
    try:
        if request.method == "GET":
            if not actions.host_is_loopback(request.headers):
                raise actions.Refused(403, "refused: this server only answers on 127.0.0.1 / localhost (Host header check)")
            return _json(await asyncio.to_thread(actions.stop_preview))
        actions.check_origin(request.headers)
        actions.check_token(request.headers)
        return _json(await asyncio.to_thread(actions.stop_runs))
    except actions.Refused as exc:
        return _refused(exc)


async def api_model(request) -> JSONResponse:
    """@brief GET 返回可选模型与委员会，POST 写入所选模型（默认模型或两个委员会）。

    @param request Starlette 请求对象，POST 时请求体上限 4096 字节。
    @return 模型选项或写入结果 JSON 响应。
    """
    from observe import settings

    try:
        if request.method == "GET":
            if not actions.host_is_loopback(request.headers):
                raise actions.Refused(403, "refused: this server only answers on 127.0.0.1 / localhost (Host header check)")
            return _json(await asyncio.to_thread(settings.model_choices))
        actions.check_origin(request.headers)
        actions.check_token(request.headers)
        raw = await request.body()
        if len(raw) > 4096:
            raise actions.Refused(413, "the body is too large")
        body = actions.parse_body(raw)
        if not isinstance(body, dict):
            raise actions.Refused(400, "the body must be a JSON object")
        if "committees" in body:
            return _json(await asyncio.to_thread(settings.set_committees, body.get("committees")))
        return _json(await asyncio.to_thread(settings.set_default_model, body.get("provider"), body.get("model")))
    except actions.Refused as exc:
        return _refused(exc)


async def api_deletions(request) -> JSONResponse:
    """@brief 返回删除日志；非 loopback 请求返回 403。

    @param request Starlette 请求对象。
    @return 删除日志 JSON 响应。
    """
    from observe.deletions import list_deletions

    try:
        if not actions.host_is_loopback(request.headers):
            raise actions.Refused(403, "refused: this server only answers on 127.0.0.1 / localhost (Host header check)")
        params = request.query_params
        try:
            limit = int(params.get("limit", "100"))
        except ValueError:
            return _json({"error": "limit must be an integer"}, status=400)
        return _json(await asyncio.to_thread(
            list_deletions, limit=limit, cursor=params.get("cursor") or None,
        ))
    except actions.Refused as exc:
        return _refused(exc)


async def api_delete_records(request) -> JSONResponse:
    """@brief 归档删除实验、operator 或单个搜索；请求体上限 8192 字节。

    @param request Starlette 请求对象，请求体需含 ``kind`` / ``name`` / ``reason``。
    @return 删除结果 JSON 响应。
    """
    from observe.deletions import delete_records

    try:
        actions.check_origin(request.headers)
        actions.check_token(request.headers)
        raw = await request.body()
        if len(raw) > 8192:
            raise actions.Refused(413, "the body is too large")
        body = actions.parse_body(raw)
        if not isinstance(body, dict):
            raise actions.Refused(400, "the body must be a JSON object")
        peer = request.client.host if request.client is not None else ""
        return _json(await asyncio.to_thread(delete_records, body.get("kind"), body.get("name"), body.get("reason"), peer))
    except actions.Refused as exc:
        return _refused(exc)


async def api_hmz_web(request) -> JSONResponse:
    """@brief 查询、启动或停止可选的 ``hmz web`` sidecar。

    GET 只返回状态；URL 只在携带页面 token 时展示。POST 启动（已有则复用），
    DELETE 停止。所有写操作仍走同源和 token 检查，不把 sidecar 当数据源。

    @param request Starlette 请求对象。
    @return sidecar 状态或错误 JSON。
    """
    try:
        if not actions.host_is_loopback(request.headers):
            raise actions.Refused(403, "refused: this server only answers on 127.0.0.1 / localhost (Host header check)")
        if request.method == "GET":
            state = await asyncio.to_thread(hmz_web.manager.status)
            try:
                actions.check_token(request.headers)
            except actions.Refused:
                state = {**state, "url": ""}
            return _json(state)
        actions.check_origin(request.headers)
        actions.check_token(request.headers)
        if request.method == "POST":
            return _json(await asyncio.to_thread(hmz_web.manager.start))
        if request.method == "DELETE":
            return _json(await asyncio.to_thread(hmz_web.manager.stop))
        raise actions.Refused(405, "method not allowed")
    except actions.Refused as exc:
        return _refused(exc)
    except hmz_web.HmzWebError as exc:
        return _json({"available": False, "running": False, "url": "", "reason": str(exc)}, 503)


async def api_budget(_request) -> JSONResponse:
    """@brief v1 兼容接口：返回预算报告。

    @param _request 未使用的请求对象。
    @return 预算报告 JSON 响应。
    """
    return _json(await asyncio.to_thread(query.budget_report))


async def api_stream(request) -> StreamingResponse:
    """@brief 以 SSE 推送流帧；内容未变时不重发。

    @param request Starlette 请求对象，用于检测客户端断开。
    @return ``text/event-stream`` 流式响应。
    """
    async def generate():
        previous = None
        while True:
            if await request.is_disconnected():
                return
            frame = await asyncio.to_thread(query.poll_stream)
            encoded = json.dumps(frame, ensure_ascii=False)
            if encoded != previous:
                yield f"data: {encoded}\n\n"
                previous = encoded
            await asyncio.sleep(_poll_seconds())

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


app = Starlette(
    routes=[
        Route("/", index),
        Route("/reviews", index),
        Route("/api/v2/overview", api_v2_overview),
        Route("/api/v2/track-explore", api_v2_track_explore),
        Route("/api/v2/live", api_v2_live),
        Route("/api/v2/patrol", api_v2_patrol),
        Route("/api/v2/feedback", api_v2_feedback),
        Route("/api/v2/suggestions", api_v2_suggestions),
        Route("/api/v2/suggestion", api_v2_suggestion),
        Route("/api/v2/harness/{run_id:path}", api_v2_harness),
        Route("/api/v2/runs", api_v2_runs),
        Route("/api/v2/experiments", api_v2_experiments),
        Route("/api/v2/experiments/{experiment_id:path}", api_v2_experiment),
        Route("/api/v2/budget", api_v2_budget),
        Route("/api/hmz-web", api_hmz_web, methods=["GET", "POST", "DELETE"]),
        Route("/api/v2/ledger", api_v2_ledger),
        Route("/api/v2/stream", api_stream),
        Route("/api/v2/turns/{run_id:path}", api_v2_turns),
        Route("/api/v2/failures", api_v2_failures),
        Route("/api/v2/publications", api_v2_publications),
        Route("/api/v2/search", api_v2_search),
        Route("/api/v2/publications/{sha256}/source", api_v2_publications_source),
        Route("/api/v2/gap", api_v2_gap),
        Route("/api/v2/committees", api_v2_committees),
        Route("/api/v2/noise", api_v2_noise),
        Route("/api/v2/best", api_v2_best),
        Route("/api/v2/failure-clusters", api_v2_failure_clusters),
        Route("/api/v2/failure-clusters/{sig}", api_v2_failure_cluster),
        Route("/api/v2/runs/{run_id:path}/candidates", api_v2_candidates),
        Route("/api/v2/runs/{run_id:path}/candidate/{label}", api_v2_run_candidate),
        Route("/api/v2/runs/{run_id:path}/feedback/{seq:int}", api_v2_run_feedback_record),
        Route("/api/v2/runs/{run_id:path}/feedback", api_v2_run_feedback),
        Route("/api/v2/runs/{run_id:path}/timeline", api_v2_run_timeline),
        Route("/api/v2/runs/{run_id:path}/budget-events", api_v2_run_budget_events),
        Route("/api/v2/runs/{run_id:path}/llm", api_v2_run_llm),
        Route("/api/v2/runs/{run_id:path}/grid", api_v2_run_grid),
        Route("/api/v2/runs/{run_id:path}/plan", api_v2_run_plan),
        Route("/api/v2/runs/{run_id:path}/lineage", api_v2_run_lineage),
        Route("/api/v2/runs/{run_id:path}/raw", api_v2_run_raw),
        Route("/api/v2/runs/{run_id:path}/detail", api_v2_run_detail),
        Route("/api/v2/runs/{run_id:path}/similar", api_v2_run_similar),
        Route("/api/v2/runs/{run_id:path}/sessions", api_v2_sessions),
        Route("/api/v2/model-stats", api_v2_model_stats),
        Route("/api/v2/pi/{run_id:path}", api_v2_pi),
        Route("/api/v2/runs/{run_id:path}", api_v2_run),
        Route("/api/overview", api_v2_overview),
        Route("/api/budget", api_budget),
        Route("/api/budget/reset", api_reset, methods=["GET", "POST"]),
        Route("/api/stop", api_stop, methods=["GET", "POST"]),
        Route("/api/settings/model", api_model, methods=["GET", "POST"]),
        Route("/api/records/deletions", api_deletions, methods=["GET"]),
        Route("/api/records/delete", api_delete_records, methods=["POST"]),
        Route("/api/runs/{run_id:path}/control", api_control, methods=["GET"]),
        Route("/api/runs/{run_id:path}/budget", api_add_budget, methods=["POST"]),
        Route("/api/runs/{run_id:path}/harness/convert", api_convert_harness, methods=["POST"]),
        Route("/api/runs/{run_id:path}/harness/{key}/review", api_review_harness, methods=["POST"]),
        Route("/api/runs/{run_id:path}/harness/{key}/promote", api_promote_harness, methods=["POST"]),
        Route("/api/runs/{run_id:path}/feedback/{seq}/claims/{digest}/review", api_review_feedback, methods=["POST"]),
    ]
)
app.add_middleware(ETagMiddleware)
app.add_middleware(GZipMiddleware, minimum_size=500)


def main() -> None:
    """@brief 在 127.0.0.1 上启动看板服务，端口取 ``OBSERVE_PORT``（默认 8765）。
    """
    import uvicorn

    from observe import simcache

    # 后台预热句向量/相似结论：请求路径只读持久缓存，不为重计算卡住其它接口。
    # 只在真实服务进程启动；测试里的 TestClient 不启动额外线程。
    simcache.start_warmer()
    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("OBSERVE_PORT", "8765")), log_level="info")
