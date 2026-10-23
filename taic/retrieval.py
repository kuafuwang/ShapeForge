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
        "suggestions": "compact free-text system suggestions grouped by digest; previews only, no full text",
        "suggestion": "one exact free-text suggestion by run/seq/digest",
    },
}


def _query():
    """@brief 懒加载 ``observe.query``，让不查索引的路径不付导入开销。

    @return ``observe.query`` 模块。
    """
    from observe import query

    return query


def runs(**kwargs: Any) -> dict:
    """@brief 分页列出运行，可按任务/数据类型/模型/状态/目标/公布情况/最低加速比过滤。

    @param kwargs 原样传给 ``observe.query.runs_page`` 的过滤、排序与游标参数。
    @return 分页结果字典。
    """
    return _query().runs_page(**kwargs)


def best(**kwargs: Any) -> dict:
    """@brief 列出已确认内核，最快的排在前。

    @param kwargs 原样传给 ``observe.query.best_confirmed`` 的参数。
    @return 已确认内核的列表结果。
    """
    return _query().best_confirmed(**kwargs)


def failures(**kwargs: Any) -> dict:
    """@brief 列出状态不是 ok 的候选，可按状态或任务过滤。

    @param kwargs 原样传给 ``observe.query.failures`` 的过滤参数。
    @return 失败候选的列表结果。
    """
    return _query().failures(**kwargs)


def sessions(run_id: str, **kwargs: Any) -> dict:
    """@brief 列出一场运行的 hmz 会话：委员会、模型、token、花费。

    @param run_id 运行标识。
    @param kwargs 原样传给 ``observe.query.sessions_page`` 的分页参数。
    @return 会话分页结果字典。
    """
    return _query().sessions_page(run_id, **kwargs)


def models() -> dict:
    """@brief 按模型汇总各会话的用量。

    与 ``runs.model`` 不同，这里的统计不只取第一委员会。

    @return 模型统计字典。
    """
    return _query().model_stats()


def suggestions(**kwargs: Any) -> dict:
    """@brief 把自由文本的系统建议按 digest 归组，只给摘要预览。

    @param kwargs 原样传给 ``observe.query.suggestions`` 的过滤参数。
    @return 建议分组结果。
    """
    return _query().suggestions(**kwargs)


def suggestion(run_id: str, seq: int, digest: str) -> dict:
    """@brief 按运行、序号与 digest 取一条自由文本建议的全文。

    @param run_id 运行标识。
    @param seq 该建议在该运行里的会话序号。
    @param digest 建议正文的摘要。
    @return 单条建议字典。
    """
    return _query().suggestion(run_id, seq, digest)


def capabilities() -> dict:
    """@brief 返回本模块支持的查询动词说明。

    @return 含 ``schema_version`` 与 ``verbs`` 的说明字典。
    """
    return CAPABILITIES
