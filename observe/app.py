"""Local dashboard. Binds to 127.0.0.1 and never starts a kernel.

Reads come from the SQLite index. The idle stream polls about once a second and sends a revision,
not a fresh copy of every run. Restart the process after changing this file or the page; the HTML is read at import.
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

from observe import actions, budget, hmz_web, query



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


_PAGE = (Path(__file__).parent / "static" / "index.html").read_text(encoding="utf-8")
_BANNER_SLOT = "<!--BUDGET_BANNER-->"
_TOKEN_SLOT = "<!--OBSERVE_TOKEN-->"


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
    page = _PAGE.replace(_BANNER_SLOT, banner).replace(_TOKEN_SLOT, f'<meta name="observe-token" content="{token}">')
    return HTMLResponse(page, headers={"Cache-Control": "no-store"})


async def api_v2_overview(_request) -> JSONResponse:
    """@brief 返回总览统计。

    @param _request 未使用的请求对象。
    @return 总览 JSON 响应。
    """
    return _json(await asyncio.to_thread(query.overview))


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
    """@brief 列出一个运行的候选 kernel。

    @param request Starlette 请求对象，路径含 ``run_id``，查询参数可含 ``status`` / ``limit``。
    @return 候选列表 JSON 响应。
    """
    return _json(await asyncio.to_thread(
        query.candidates_page,
        request.path_params["run_id"],
        status=request.query_params.get("status") or None,
        limit=int(request.query_params.get("limit", "200") or 200),
    ))


async def api_v2_sessions(request) -> JSONResponse:
    """@brief 列出一个运行的模型会话。

    @param request Starlette 请求对象，路径含 ``run_id``，查询参数可含 ``limit``。
    @return 会话列表 JSON 响应。
    """
    return _json(await asyncio.to_thread(
        query.sessions_page,
        request.path_params["run_id"],
        limit=int(request.query_params.get("limit", "500") or 500),
    ))


async def api_v2_model_stats(_request) -> JSONResponse:
    """@brief 返回按模型聚合的统计。

    @param _request 未使用的请求对象。
    @return 模型统计 JSON 响应。
    """
    return _json(await asyncio.to_thread(query.model_stats))


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
        limit=int(params.get("limit", "50") or 50),
    ))


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
    """@brief 返回单个运行的详情；找不到时返回 404。

    @param request Starlette 请求对象，路径含 ``run_id``，查询参数可含 ``stamp``。
    @return 运行详情 JSON 响应。
    """
    doc = await asyncio.to_thread(query.run_detail, request.path_params["run_id"], stamp=request.query_params.get("stamp") or "")
    if doc is None:
        return _json({"error": "not found"}, status=404)
    return _json(doc)


async def api_v2_turns(request) -> JSONResponse:
    """@brief 返回一段模型轮次窗口；``stamp`` 匹配时返回空行且不查轮次表。

    原始英文说明：One window of model turns. A matching stamp returns no rows and does not read the turns table.

    @param request Starlette 请求对象，路径含 ``run_id``，查询参数可含 ``before`` / ``limit`` / ``stamp``。
    @return 轮次窗口 JSON 响应；运行不存在时返回 404。
    """
    from observe.detail import turns_feed

    params = request.query_params
    try:
        limit = int(params.get("limit", "40"))
    except ValueError:
        limit = 40
    feed = await asyncio.to_thread(
        turns_feed,
        request.path_params["run_id"],
        before=params.get("before"),
        limit=limit,
        stamp=params.get("stamp") or "",
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


async def api_v2_experiments(_request) -> JSONResponse:
    """@brief 列出实验。

    @param _request 未使用的请求对象。
    @return 实验列表 JSON 响应。
    """
    return _json(await asyncio.to_thread(query.experiments))


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
    raw = request.query_params.get("since")
    since = None
    if raw not in (None, ""):
        try:
            since = int(raw)
        except ValueError:
            return _json({"error": "since must be an integer"}, status=400)
    return _json(await asyncio.to_thread(query.feedback_board, since=since))


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


async def api_v1_history(_request) -> JSONResponse:
    """@brief 返回 v1 历史行列表。

    @param _request 未使用的请求对象。
    @return 历史行 JSON 响应。
    """
    return _json(await asyncio.to_thread(query.history))


async def api_v1_run(request) -> JSONResponse:
    """@brief 返回 v1 运行详情；找不到时返回 404。

    @param request Starlette 请求对象，路径含 ``run_id``。
    @return 运行详情 JSON 响应。
    """
    doc = await asyncio.to_thread(query.run_detail, request.path_params["run_id"])
    if doc is None:
        return _json({"error": "not found"}, status=404)
    return _json(doc)


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
        return _json(await asyncio.to_thread(list_deletions))
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
        Route("/api/v2/best", api_v2_best),
        Route("/api/v2/runs/{run_id:path}/candidates", api_v2_candidates),
        Route("/api/v2/runs/{run_id:path}/sessions", api_v2_sessions),
        Route("/api/v2/model-stats", api_v2_model_stats),
        Route("/api/v2/pi/{run_id:path}", api_v2_pi),
        Route("/api/v2/runs/{run_id:path}", api_v2_run),
        Route("/api/history", api_v1_history),
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
        Route("/api/runs/{run_id:path}", api_v1_run),
        Route("/api/stream", api_stream),
    ]
)
app.add_middleware(ETagMiddleware)
app.add_middleware(GZipMiddleware, minimum_size=500)


def main() -> None:
    """@brief 在 127.0.0.1 上启动看板服务，端口取 ``OBSERVE_PORT``（默认 8765）。
    """
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("OBSERVE_PORT", "8765")), log_level="info")
