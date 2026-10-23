"""Trust metrics from the observe index. A person grants trust; these numbers only inform it."""

from __future__ import annotations

import math


def wilson_lower(successes: int, total: int, z: float = 1.96) -> float | None:
    """@brief 计算二项比例置信区间的下界；样本为空时返回 ``None``。

    五次评审里 80% 认可并不构成证据；这个下界让小样本看起来明显偏弱。

    @param successes 成功（认可）次数。
    @param total 评审样本总数。
    @param z 正态分位数，默认为 1.96（约 95% 置信）。
    @return 区间下界，取值范围 [0, 1]；``total`` 不为正时返回 ``None``。
    """
    if total <= 0:
        return None
    p = max(0.0, min(1.0, successes / total))
    denom = 1.0 + z * z / total
    centre = p + z * z / (2.0 * total)
    margin = z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * total)) / total)
    return max(0.0, (centre - margin) / denom)


def metrics(conn, *, target_id: str | None = None, task: str | None = None) -> list[dict]:
    """@brief 按（target、task、provider、model）统计评审与证据数量。

    ``reasonable`` 是 adopted/done；``refuted`` 是 rejected 且验证结论为 refuted。
    ``reviewed`` 只含这三种决定，不含 pending 或仅告知的行。

    @param conn 只读的索引数据库连接。
    @param target_id 只统计该 target 的会话，None 表示不限。
    @param task 只统计该任务名，None 表示不限。
    @return 每个分组一项的字典列表，含计数、比例，以及 Wilson 下界与占位符统计等派生字段。
    """
    where = ["1 = 1"]
    args: list[object] = []
    if target_id:
        where.append("r.target_id = ?")
        args.append(target_id)
    if task:
        where.append("i.task = ?")
        args.append(task)
    rows = conn.execute(
        f"""
        SELECT r.target_id AS target_id, i.task AS task, s.provider AS provider, s.model AS model,
               COUNT(DISTINCT s.run_id || '/' || s.seq) AS sessions,
               COUNT(DISTINCT c.run_id || '/' || c.seq || '/' || c.digest) AS claims,
               SUM(CASE WHEN c.status IN ('adopted','done') THEN 1 ELSE 0 END) AS reasonable,
               SUM(CASE WHEN c.status = 'rejected' AND c.verification = 'refuted' THEN 1 ELSE 0 END) AS refuted,
               SUM(CASE WHEN c.status IN ('adopted','rejected','done') THEN 1 ELSE 0 END) AS reviewed,
               SUM(CASE WHEN c.status = 'pending' THEN 1 ELSE 0 END) AS pending,
               SUM(CASE WHEN c.audience = 'system' AND c.status IN ('adopted','rejected','done') THEN 1 ELSE 0 END) AS system_reviewed,
               SUM(CASE WHEN c.audience = 'system' AND c.status IN ('adopted','done') THEN 1 ELSE 0 END) AS system_reasonable,
               SUM(CASE WHEN c.audience = 'system' AND c.status = 'rejected' AND c.verification = 'refuted' THEN 1 ELSE 0 END) AS system_refuted,
               SUM(CASE WHEN c.audience = 'system' AND c.status = 'pending' THEN 1 ELSE 0 END) AS system_pending,
               SUM(CASE WHEN c.premise = 'verified' THEN 1 ELSE 0 END) AS verified_premise,
               SUM(CASE WHEN c.premise IN ('verified','unverified','downgraded') THEN 1 ELSE 0 END) AS premise
        FROM sessions s
        JOIN runs r ON r.run_id = s.run_id
        JOIN feedback_items i ON i.run_id = s.run_id AND i.seq = s.seq
        JOIN feedback_claims c ON c.run_id = s.run_id AND c.seq = s.seq
        WHERE {' AND '.join(where)}
        GROUP BY r.target_id, i.task, s.provider, s.model
        ORDER BY r.target_id, i.task, s.provider, s.model
        """,
        args,
    )
    placeholder_rows = conn.execute(
        f"""
        SELECT r.target_id AS target_id, i.task AS task, s.provider AS provider, s.model AS model,
               SUM(i.placeholder_items) AS n
        FROM sessions s
        JOIN runs r ON r.run_id = s.run_id
        JOIN feedback_items i ON i.run_id = s.run_id AND i.seq = s.seq
        WHERE {' AND '.join(where)}
        GROUP BY r.target_id, i.task, s.provider, s.model
        """,
        args,
    )
    placeholders = {(row["target_id"], row["task"], row["provider"], row["model"]): int(row["n"] or 0) for row in placeholder_rows}
    out = []
    for row in rows:
        item = dict(row)
        key = (item["target_id"], item["task"], item["provider"], item["model"])
        item["placeholder_items"] = placeholders.get(key, 0)
        reviewed = int(item.get("reviewed") or 0)
        reasonable = int(item.get("reasonable") or 0)
        item["reasonable_rate"] = (reasonable / reviewed) if reviewed else None
        item["wilson_lower"] = wilson_lower(reasonable, reviewed)
        system_reviewed = int(item.get("system_reviewed") or 0)
        system_reasonable = int(item.get("system_reasonable") or 0)
        item["system_reasonable_rate"] = (system_reasonable / system_reviewed) if system_reviewed else None
        item["system_wilson_lower"] = wilson_lower(system_reasonable, system_reviewed)
        premise = int(item.get("premise") or 0)
        item["measured_rate"] = (int(item.get("verified_premise") or 0) / premise) if premise else None
        sessions = int(item.get("sessions") or 0)
        placeholders = int(item.get("placeholder_items") or 0)
        item["placeholder_per_session"] = (placeholders / sessions) if sessions else None
        # Five named sections per session; a placeholder can only occur in one of them.
        item["placeholder_rate"] = (placeholders / (sessions * 5)) if sessions else None
        out.append(item)
    return out
