"""Local read-only dashboard. Binds to 127.0.0.1 and never starts a kernel."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

from starlette.applications import Starlette
from starlette.responses import HTMLResponse, JSONResponse, StreamingResponse
from starlette.routing import Route

from observe import budget, scan

_PAGE = (Path(__file__).parent / "static" / "index.html").read_text(encoding="utf-8")
_BANNER_SLOT = "<!--BUDGET_BANNER-->"


def _json(data: dict | list, status: int = 200) -> JSONResponse:
    return JSONResponse(data, status_code=status)


async def index(_request) -> HTMLResponse:
    try:  # the banner is drawn on the server once so the alert is visible at first paint (and without JavaScript); the page script keeps it current
        banner = budget.render_banner_html(scan.budget_report())
    except Exception:  # noqa: BLE001  a damaged run directory must never take the page down
        banner = ""
    return HTMLResponse(_PAGE.replace(_BANNER_SLOT, banner), headers={"Cache-Control": "no-store"})


async def api_overview(_request) -> JSONResponse:
    return _json(scan.overview())


async def api_history(_request) -> JSONResponse:
    return _json(scan.history())


async def api_budget(_request) -> JSONResponse:
    return _json(scan.budget_report())


async def api_run(request) -> JSONResponse:
    doc = scan.run_detail(request.path_params["run_id"])
    if doc is None:
        return _json({"error": "not found"}, status=404)
    return _json(doc)


async def api_stream(request) -> StreamingResponse:
    run_id = request.query_params.get("run") or ""

    async def generate():
        previous = None
        while True:
            if await request.is_disconnected():
                return
            payload = json.dumps(scan.frame(run_id), ensure_ascii=False, default=str)
            if payload != previous:
                yield f"data: {payload}\n\n"
                previous = payload
            await asyncio.sleep(0.1)

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


app = Starlette(
    routes=[
        Route("/", index),
        Route("/api/overview", api_overview),
        Route("/api/history", api_history),
        Route("/api/budget", api_budget),
        Route("/api/stream", api_stream),
        Route("/api/runs/{run_id:path}", api_run),
    ]
)


def main() -> None:
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("OBSERVE_PORT", "8765")), log_level="info")
