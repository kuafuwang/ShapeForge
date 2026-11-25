"""Read-only real ASGI server for the opt-in Edge browser E2E harness.

Every non-GET request (including HEAD) is refused before it reaches the production app.  The harness never
clicks a mutating control; this is defense in depth, not a replacement for avoiding those routes.
"""

from __future__ import annotations

import json
import os


async def guarded_app(scope, receive, send):
    if scope.get("type") == "http" and str(scope.get("method") or "GET").upper() != "GET":
        body = json.dumps({"error": "browser E2E is read-only"}).encode()
        await send({"type": "http.response.start", "status": 403, "headers": [("content-type", "application/json"), ("content-length", str(len(body))) ]})
        await send({"type": "http.response.body", "body": body})
        return
    from observe.app import app

    await app(scope, receive, send)


def main() -> None:
    import uvicorn

    uvicorn.run(guarded_app, host="127.0.0.1", port=int(os.environ["OBSERVE_PORT"]), log_level="error", access_log=False)


if __name__ == "__main__":
    main()
