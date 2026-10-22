"""Compact read-only queries over the observe index, for a human or an agent.

The dashboard has HTTP endpoints for the same data; this module gives the CLI a stable JSON
surface without opening a browser. It imports ``observe.query`` lazily so a plain ``taic run``
does not pay for the index import.
"""

from __future__ import annotations

from typing import Any

CAPABILITIES = {
    "schema_version": "1",
    "verbs": {
        "runs": "runs_page: filter by task/dtype/model/status/target/publication/min_speedup; sort and cursor",
        "best": "best_confirmed: confirmed kernels, fastest first",
        "failures": "candidates with a non-ok status, optionally by status/task",
        "sessions": "one run's hmz sessions: committee, model, tokens, cost",
        "models": "per-session model totals; unlike runs.model this is not the first committee",
    },
}


def _query():
    from observe import query

    return query


def runs(**kwargs: Any) -> dict:
    return _query().runs_page(**kwargs)


def best(**kwargs: Any) -> dict:
    return _query().best_confirmed(**kwargs)


def failures(**kwargs: Any) -> dict:
    return _query().failures(**kwargs)


def sessions(run_id: str, **kwargs: Any) -> dict:
    return _query().sessions_page(run_id, **kwargs)


def models() -> dict:
    return _query().model_stats()


def capabilities() -> dict:
    return CAPABILITIES
