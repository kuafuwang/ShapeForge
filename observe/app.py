"""Local dashboard. Binds to 127.0.0.1 and never starts a kernel.

Reads come from the SQLite index. The idle stream polls about once a second and sends a revision,
not a fresh copy of every run. Restart the process after changing this file or the page; the HTML is read at import.
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

from starlette.applications import Starlette
from starlette.responses import HTMLResponse, JSONResponse, StreamingResponse
from starlette.routing import Route

from observe import actions, budget, query

_PAGE = (Path(__file__).parent / "static" / "index.html").read_text(encoding="utf-8")
_BANNER_SLOT = "<!--BUDGET_BANNER-->"
_TOKEN_SLOT = "<!--OBSERVE_TOKEN-->"


def _json(data: dict | list, status: int = 200) -> JSONResponse:
    return JSONResponse(data, status_code=status)


def _poll_seconds() -> float:
    try:
        return max(0.05, float(os.environ.get("OBSERVE_POLL_S", "1")))
    except ValueError:
        return 1.0


async def index(request) -> HTMLResponse:
    try:
        banner = budget.render_banner_html(await asyncio.to_thread(query.budget_report))
    except Exception:  # noqa: BLE001  a damaged run directory must never take the page down
        banner = ""
    token = actions.page_token(request.headers)
    page = _PAGE.replace(_BANNER_SLOT, banner).replace(_TOKEN_SLOT, f'<meta name="observe-token" content="{token}">')
    return HTMLResponse(page, headers={"Cache-Control": "no-store"})


async def api_v2_overview(_request) -> JSONResponse:
    return _json(await asyncio.to_thread(query.overview))


async def api_v2_runs(request) -> JSONResponse:
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
        limit=limit,
        offset=offset,
    )
    return _json(page)


async def api_v2_run(request) -> JSONResponse:
    doc = await asyncio.to_thread(query.run_detail, request.path_params["run_id"], stamp=request.query_params.get("stamp") or "")
    if doc is None:
        return _json({"error": "not found"}, status=404)
    return _json(doc)


async def api_v2_turns(request) -> JSONResponse:
    """One window of model turns. A matching stamp returns no rows and does not read the turns table."""
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
    """The live pi transcript for one run. Polled on its own so the rest of the page is not redrawn."""
    from observe.detail import pi_feed

    stamp = request.query_params.get("stamp") or ""
    feed = await asyncio.to_thread(pi_feed, request.path_params["run_id"], stamp=stamp)
    if feed is None:
        return _json({"error": "not found"}, status=404)
    return _json(feed)


async def api_v2_experiments(_request) -> JSONResponse:
    return _json(await asyncio.to_thread(query.experiments))


async def api_v2_experiment(request) -> JSONResponse:
    return _json(await asyncio.to_thread(query.compare, request.path_params["experiment_id"]))


async def api_v2_feedback(request) -> JSONResponse:
    raw = request.query_params.get("since")
    since = None
    if raw not in (None, ""):
        try:
            since = int(raw)
        except ValueError:
            return _json({"error": "since must be an integer"}, status=400)
    return _json(await asyncio.to_thread(query.feedback_board, since=since))


async def api_v2_budget(_request) -> JSONResponse:
    return _json(await asyncio.to_thread(query.budget_report))


async def api_v2_ledger(_request) -> JSONResponse:
    return _json(await asyncio.to_thread(query.ledger_report))


async def api_v1_history(_request) -> JSONResponse:
    return _json(await asyncio.to_thread(query.history))


async def api_v1_run(request) -> JSONResponse:
    doc = await asyncio.to_thread(query.run_detail, request.path_params["run_id"])
    if doc is None:
        return _json({"error": "not found"}, status=404)
    return _json(doc)


def _refused(exc: actions.Refused) -> JSONResponse:
    return _json({"ok": False, "error": exc.message}, status=exc.status)


async def api_control(request) -> JSONResponse:
    try:
        if not actions.host_is_loopback(request.headers):
            raise actions.Refused(403, "refused: this server only answers on 127.0.0.1 / localhost (Host header check)")
        return _json(await asyncio.to_thread(actions.describe, request.path_params["run_id"]))
    except actions.Refused as exc:
        return _refused(exc)


async def api_review_feedback(request) -> JSONResponse:
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
    from observe.deletions import list_deletions

    try:
        if not actions.host_is_loopback(request.headers):
            raise actions.Refused(403, "refused: this server only answers on 127.0.0.1 / localhost (Host header check)")
        return _json(await asyncio.to_thread(list_deletions))
    except actions.Refused as exc:
        return _refused(exc)


async def api_delete_records(request) -> JSONResponse:
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


async def api_budget(_request) -> JSONResponse:
    return _json(await asyncio.to_thread(query.budget_report))


async def api_stream(request) -> StreamingResponse:
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
        Route("/api/v2/runs", api_v2_runs),
        Route("/api/v2/experiments", api_v2_experiments),
        Route("/api/v2/experiments/{experiment_id:path}", api_v2_experiment),
        Route("/api/v2/budget", api_v2_budget),
        Route("/api/v2/ledger", api_v2_ledger),
        Route("/api/v2/stream", api_stream),
        Route("/api/v2/turns/{run_id:path}", api_v2_turns),
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
        Route("/api/runs/{run_id:path}/feedback/{seq}/claims/{digest}/review", api_review_feedback, methods=["POST"]),
        Route("/api/runs/{run_id:path}", api_v1_run),
        Route("/api/stream", api_stream),
    ]
)


def main() -> None:
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("OBSERVE_PORT", "8765")), log_level="info")
