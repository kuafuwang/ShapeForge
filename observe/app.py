"""Local read-only dashboard. Binds to 127.0.0.1 and never starts a kernel."""

# 页面上的数来自每次请求调用 scan：scan 只读 runs/，以及 publish 约定的缓存目录。本进程不启动内核。
# 下面的 HTML 在导入时读入，uvicorn 没有热重载。改这个文件、scan、budget 或 static/index.html 之后，必须重启进程，浏览器刷新看到的仍是旧代码。
# 磁盘上新出现的 result.json 不用重启，下一次请求和下一帧 SSE 会重新 scan。

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

from starlette.applications import Starlette
from starlette.responses import HTMLResponse, JSONResponse, StreamingResponse
from starlette.routing import Route

from observe import actions, budget, scan

# 导入时读一次。运行中的进程不会再看磁盘上的 index.html，所以改页面模板一定要重启。
_PAGE = (Path(__file__).parent / "static" / "index.html").read_text(encoding="utf-8")
_BANNER_SLOT = "<!--BUDGET_BANNER-->"
_TOKEN_SLOT = "<!--OBSERVE_TOKEN-->"


# 几个只读接口共用同一种 JSON。页面脚本和 curl 看到的结构都是 scan 刚扫出来的。
def _json(data: dict | list, status: int = 200) -> JSONResponse:
    return JSONResponse(data, status_code=status)


# 首屏外壳是导入时的那份 HTML，只有预算横幅在这次请求里填。横幅读盘失败时仍返回页面，一个坏目录不能让看板打不开。
async def index(request) -> HTMLResponse:
    try:  # the banner is drawn on the server once so the alert is visible at first paint (and without JavaScript); the page script keeps it current
        banner = budget.render_banner_html(scan.budget_report())
    except Exception:  # noqa: BLE001  a damaged run directory must never take the page down
        banner = ""
    token = actions.page_token(request.headers)  # only a loopback Host gets it; the write path checks it on every POST
    page = _PAGE.replace(_BANNER_SLOT, banner).replace(_TOKEN_SLOT, f'<meta name="observe-token" content="{token}">')
    return HTMLResponse(page, headers={"Cache-Control": "no-store"})


# 总览：GPU 锁、仍在跑的搜索、实验队列。每次调用都重新读盘，没有进程内缓存。
async def api_overview(_request) -> JSONResponse:
    return _json(scan.overview())


# 历史表。行里的预算档位是 scan 按当前这份 state/result 算的，刷新即可看到新结果。
async def api_history(_request) -> JSONResponse:
    return _json(scan.history())


# 顶部告警的同一份纯函数结果。首屏 HTML 里的横幅和这个接口用的是同一种判定。
async def api_budget(_request) -> JSONResponse:
    return _json(scan.budget_report())


# 单次运行的详情。resolve 拒绝的路径在这里是 404，仓库外的文件不会被读出来。
async def api_run(request) -> JSONResponse:
    doc = scan.run_detail(request.path_params["run_id"])
    if doc is None:
        return _json({"error": "not found"}, status=404)
    return _json(doc)


# 页面开着时靠这条长连接更新。每 0.1 秒重扫一帧，变的是磁盘上的结果，不是启动时导入的代码。
def _refused(e: actions.Refused) -> JSONResponse:
    return _json({"ok": False, "error": e.message}, status=e.status)


async def api_control(request) -> JSONResponse:
    """What 'add budget' would do for this run (read-only)."""
    try:
        if not actions.host_is_loopback(request.headers):
            raise actions.Refused(403, "refused: this server only answers on 127.0.0.1 / localhost (Host header check)")
        return _json(await asyncio.to_thread(actions.describe, request.path_params["run_id"]))
    except actions.Refused as e:
        return _refused(e)


async def api_add_budget(request) -> JSONResponse:
    """The only write path of the dashboard: add budget to a run (see observe/actions.py for the checks)."""
    try:
        actions.check_origin(request.headers)
        actions.check_token(request.headers)
        raw = await request.body()
        if len(raw) > 4096:
            raise actions.Refused(413, "the body is too large")
        body = actions.parse_body(raw)
        return _json(await asyncio.to_thread(actions.add_budget, request.path_params["run_id"], body))
    except actions.Refused as e:
        return _refused(e)


async def api_reset(request) -> JSONResponse:
    """GET previews the one-click reset. POST adds $15 to every continuable search and lifts its round limit and patience."""
    try:
        if request.method == "GET":
            if not actions.host_is_loopback(request.headers):
                raise actions.Refused(403, "refused: this server only answers on 127.0.0.1 / localhost (Host header check)")
            return _json(await asyncio.to_thread(actions.reset_preview))
        actions.check_origin(request.headers)
        actions.check_token(request.headers)
        return _json(await asyncio.to_thread(actions.reset_budgets))
    except actions.Refused as e:
        return _refused(e)


async def api_stop(request) -> JSONResponse:
    """GET lists the live searches a stop would signal. POST sends SIGTERM to those process groups."""
    try:
        if request.method == "GET":
            if not actions.host_is_loopback(request.headers):
                raise actions.Refused(403, "refused: this server only answers on 127.0.0.1 / localhost (Host header check)")
            return _json(await asyncio.to_thread(actions.stop_preview))
        actions.check_origin(request.headers)
        actions.check_token(request.headers)
        return _json(await asyncio.to_thread(actions.stop_runs))
    except actions.Refused as e:
        return _refused(e)


async def api_model(request) -> JSONResponse:
    """GET lists models. POST writes the chosen provider and model into taic.toml."""
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
        return _json(await asyncio.to_thread(settings.set_default_model, body.get("provider"), body.get("model")))
    except actions.Refused as e:
        return _refused(e)


async def api_stream(request) -> StreamingResponse:
    run_id = request.query_params.get("run") or ""

    async def generate():
        previous = None
        while True:
            if await request.is_disconnected():
                # 浏览器关掉页面就停扫，避免没人看的连接一直读 runs/。
                return
            payload = json.dumps(scan.frame(run_id), ensure_ascii=False, default=str)
            if payload != previous:
                # 整帧逐字相同就不推。runs/ 没变化时页面不会自己重画。
                yield f"data: {payload}\n\n"
                previous = payload
            await asyncio.sleep(0.1)

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# 路由在导入时定死。新加路径或改处理函数，已经跑着的进程不会自己换上，要重启 python -m observe。
app = Starlette(
    routes=[
        Route("/", index),
        Route("/api/overview", api_overview),
        Route("/api/history", api_history),
        Route("/api/budget", api_budget),
        Route("/api/budget/reset", api_reset, methods=["GET", "POST"]),
        Route("/api/stop", api_stop, methods=["GET", "POST"]),
        Route("/api/settings/model", api_model, methods=["GET", "POST"]),
        Route("/api/stream", api_stream),
        Route("/api/runs/{run_id:path}/control", api_control, methods=["GET"]),
        Route("/api/runs/{run_id:path}/budget", api_add_budget, methods=["POST"]),
        Route("/api/runs/{run_id:path}", api_run),
    ]
)


# 没有 reload。这个进程活着的期间，observe 包和 HTML 都是启动时那一版。
def main() -> None:
    import uvicorn

    # 不传 reload。改代码后要停掉这个进程再启动，端口默认 8765，可用 OBSERVE_PORT 换。
    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("OBSERVE_PORT", "8765")), log_level="info")
