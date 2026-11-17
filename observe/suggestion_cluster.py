"""Read-only clusters of free-text suggestions.

Partition first by backend, simulated measurement, and route intent. Inside one
partition, macOS uses the local NLEmbedding sentence vectors and every other
platform uses ``lexical_overlap``. The command never writes a review, never
calls a model, and never dispatches a kernel.
"""

from __future__ import annotations

import hashlib
import json
import re

from taic.proposal_routing import route_proposal

_CJK = re.compile(r"[\u3400-\u9fff]")
_LEXICAL_LINK = 0.72
_EMBED_ZH = 0.975
_EMBED_EN = 0.65
_EMBED_OVERLAP_ZH = 0.20
_EMBED_OVERLAP_EN = 0.15


_PROPOSAL_KEYS = ("surface", "action", "scope", "evidence", "peer_action", "audience", "needs")


def proposal_document(text: str) -> dict | None:
    """@brief 主张正文若是结构化提案，返回解析后的对象。自由文本返回 None。"""
    raw = str(text or "").strip()
    if not raw.startswith("{"):
        return None
    try:
        document = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(document, dict):
        return None
    if not any(key in document for key in _PROPOSAL_KEYS):
        return None
    return document


def intent_bucket(text: str, section: str) -> str:
    """@brief 路由意图。

    ``需要帮助`` 里只有开发者队列仍算开发者请求，其余记为 ``help``。
    结构化提案把完整对象交给 ``route_proposal``，不用 JSON 外壳当 action。
    自由文本仍只把正文当作 action。

    @param text 建议正文。
    @param section 反馈小节名。
    @return ``developer_queue``、``overlay_candidate``、``review_candidate`` 或 ``help``。
    """
    document = proposal_document(text)
    routed = route_proposal(document if document is not None else {"action": str(text or "")})
    disposition = str(routed.get("disposition") or "review_candidate")
    if section == "需要帮助" and disposition != "developer_queue":
        return "help"
    return disposition


def select_scorer():
    """@brief 选择区内相似度。

    本机句向量可用且探针能嵌入时用 NLEmbedding，否则用词面重叠。

    @return ``(scorer, method)``。scorer 接收两段文本，返回 0 到 1 的分数。
    """
    from observe import similarity

    if similarity.available():
        probe = similarity.embedding("kernel tile")
        if probe:
            return _embedding_score, "nlembedding"
    return similarity.lexical_overlap, "lexical_overlap"


def _embedding_score(left: str, right: str) -> float:
    """@brief 句向量余弦。词面重叠不够时直接返回 0，避免中文高基线把泛泛的句子并到一起。"""
    from observe import similarity

    minimum = _EMBED_OVERLAP_ZH if _CJK.search(left or "") or _CJK.search(right or "") else _EMBED_OVERLAP_EN
    if similarity.lexical_overlap(left, right) < minimum:
        return 0.0
    first = similarity.embedding(left)
    second = similarity.embedding(right)
    if not first or not second:
        return 0.0
    return float(similarity.cosine(first, second))


def _link_threshold(method: str, left: str, right: str, link_at: float | None) -> float:
    if link_at is not None:
        return float(link_at)
    if method == "nlembedding" and (_CJK.search(left or "") or _CJK.search(right or "")):
        return _EMBED_ZH
    if method == "nlembedding":
        return _EMBED_EN
    return _LEXICAL_LINK


def is_near_duplicate(left: str, right: str, *, scorer=None, method: str | None = None, link_at: float | None = None) -> bool:
    """@brief 两段文本在当前平台的聚类阈值下算不算近义重复。

    @param left 第一段文本。
    @param right 第二段文本。
    @param scorer 测试注入的打分函数。None 时按平台选择。
    @param method 打分方法名。注入 scorer 时一并给出；省略则随 ``select_scorer``。
    @param link_at 覆盖默认阈值。
    @return 达到阈值时为 True。
    """
    if scorer is None:
        scorer, method = select_scorer()
    method = method or "lexical_overlap"
    score = float(scorer(left, right))
    return score >= _link_threshold(method, left, right, link_at)


def _preview(text: str, chars: int) -> str:
    collapsed = " ".join(str(text or "").split())
    if len(collapsed) <= chars:
        return collapsed
    return collapsed[: max(0, chars - 1)].rstrip() + "…"


def request_text(text: str) -> str:
    """@brief 聚类用的请求正文。结构化提案只比 ``action``，不用 JSON 外壳。

    外壳上的 ``surface`` / ``scope`` / ``evidence`` 会让句向量把不同 action 连成一团，
    单链接再把这种相似传递成一个大簇。没有 action 的提案外壳不参与连接。

    @param text 主张原文。
    @return 用来比较的正文。自由文本原样返回。
    """
    document = proposal_document(text)
    if document is None:
        return str(text or "").strip()
    action = " ".join(str(document.get("action") or "").split())
    if action:
        return action
    return " ".join(str(document.get("peer_action") or "").split())


def _cluster_id(backend: str, simulated: bool, intent: str, digests: list[str]) -> str:
    payload = json.dumps(
        [backend, bool(simulated), intent, digests],
        ensure_ascii=False, separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _components(nodes: list[dict], scorer, method: str, link_at: float | None) -> list[list[dict]]:
    """@brief 按阈值图做团划分。簇里每对都必须直接连上，桥接节点不能把两端并进去。"""
    count = len(nodes)
    texts = [str(node.get("compare") or "") for node in nodes]
    linked = [[False] * count for _ in range(count)]
    scores = [[0.0] * count for _ in range(count)]
    for left in range(count):
        if not texts[left]:
            continue
        for right in range(left + 1, count):
            if not texts[right]:
                continue
            value = float(scorer(texts[left], texts[right]))
            scores[left][right] = scores[right][left] = value
            threshold = _link_threshold(method, texts[left], texts[right], link_at)
            if value >= threshold:
                linked[left][right] = linked[right][left] = True
    degree = [sum(row) for row in linked]
    order = sorted(range(count), key=lambda index: (-degree[index], index))
    remaining = set(range(count))
    groups: list[list[dict]] = []
    for seed in order:
        if seed not in remaining:
            continue
        group = [seed]
        remaining.remove(seed)
        candidates = sorted(
            (index for index in remaining if linked[seed][index]),
            key=lambda index: (-scores[seed][index], index),
        )
        for index in candidates:
            if all(linked[index][member] for member in group):
                group.append(index)
                remaining.remove(index)
        groups.append([nodes[index] for index in group])
    return groups


def cluster_rows(
    rows: list[dict],
    *,
    scorer=None,
    method: str | None = None,
    link_at: float | None = None,
    limit: int | None = 50,
    offset: int = 0,
    preview: int = 160,
    input_truncated: bool = False,
) -> dict:
    """@brief 先分区再聚类。分区之间即使分数是 1 也不合并。

    每行需要 ``text``、``digest``、``backend``、``simulated``、``section``。
    只读：不写评审、不调用模型、不上设备。

    @param rows 建议行。
    @param scorer 区内打分。None 时按平台选择。
    @param method 打分方法名。
    @param link_at 覆盖默认连接阈值。
    @param limit 返回的簇数上限。
    @param offset 簇页偏移。
    @param preview 预览字符数。
    @param input_truncated 读入的主张是否被上限截断。
    @return 聚类视图。
    """
    if scorer is None:
        scorer, method = select_scorer()
    method = method or "lexical_overlap"
    preview = max(0, min(int(preview if preview is not None else 160), 500))
    offset = max(0, int(offset or 0))
    limit = 50 if limit is None else max(1, min(int(limit), 200))
    partitions: dict[tuple, list[dict]] = {}
    for row in rows:
        text = str(row.get("text") or "").strip()
        digest = str(row.get("digest") or "")
        if not text or not digest:
            continue
        section = str(row.get("section") or "")
        backend = str(row.get("backend") or "")
        simulated = bool(row.get("simulated") is True)
        intent = str(row.get("intent") or "") or intent_bucket(text, section)
        compared = request_text(text)
        occurrences = row.get("occurrences") if isinstance(row.get("occurrences"), list) else []
        node = {
            "digest": digest,
            "text": text,
            "compare": compared,
            "section": section,
            "backend": backend,
            "simulated": simulated,
            "intent": intent,
            "run_id": str(row.get("run_id") or ""),
            "seq": row.get("seq"),
            "claims": int(row.get("claims") or 1),
            "runs": int(row.get("runs") or 1),
            "status": str(row.get("status") or ""),
            "occurrences": [item for item in occurrences if isinstance(item, dict)],
        }
        partitions.setdefault((backend, simulated, intent), []).append(node)
    clusters = []
    for (backend, simulated, intent), nodes in partitions.items():
        # 同一分区里相同 digest 只留一行，主张数相加。
        merged: dict[str, dict] = {}
        for node in nodes:
            current = merged.get(node["digest"])
            if current is None:
                merged[node["digest"]] = dict(node)
                continue
            current["claims"] += node["claims"]
            current["runs"] += node["runs"]
            current["occurrences"] = list(current.get("occurrences") or []) + list(node.get("occurrences") or [])
            if node["status"] and node["status"] not in current["status"].split(","):
                current["status"] = ",".join(part for part in (current["status"], node["status"]) if part)
        ordered_nodes = [merged[key] for key in sorted(merged)]
        for group in _components(ordered_nodes, scorer, method, link_at):
            group.sort(key=lambda item: (-int(item["claims"]), item["digest"]))
            digests = sorted(item["digest"] for item in group)
            representative = group[0]
            coordinates = []
            for item in group:
                found = item.get("occurrences") if isinstance(item.get("occurrences"), list) else []
                if not found:
                    found = [{
                        "run_id": item.get("run_id") or "",
                        "seq": item.get("seq"),
                        "digest": item.get("digest") or "",
                        "status": item.get("status") or "",
                    }]
                for coord in found:
                    if not isinstance(coord, dict):
                        continue
                    status = str(coord.get("status") or "")
                    coordinates.append({
                        "run_id": str(coord.get("run_id") or ""),
                        "seq": coord.get("seq"),
                        "digest": str(coord.get("digest") or item.get("digest") or ""),
                        "status": status,
                        "expected_status": status,
                    })
            clusters.append({
                "cluster_id": _cluster_id(backend, simulated, intent, digests),
                "backend": backend,
                "simulated": simulated,
                "intent": intent,
                "sections": sorted({item["section"] for item in group if item["section"]}),
                "size": len(group),
                "claims": sum(int(item["claims"]) for item in group),
                "representative": {
                    "digest": representative["digest"],
                    "preview": _preview(representative.get("compare") or representative["text"], preview),
                    "run_id": representative["run_id"],
                    "seq": representative["seq"],
                    "section": representative["section"],
                },
                "coordinates": coordinates,
                "members": [
                    {
                        "digest": item["digest"],
                        "preview": _preview(item.get("compare") or item["text"], preview),
                        "section": item["section"],
                        "run_id": item["run_id"],
                        "seq": item["seq"],
                        "claims": item["claims"],
                        "runs": item["runs"],
                        "status": item["status"],
                        "occurrences": list(item.get("occurrences") or []),
                    }
                    for item in sorted(group, key=lambda item: item["digest"])
                ],
            })
    clusters.sort(key=lambda item: (-int(item["size"]), item["cluster_id"]))
    total = len(clusters)
    page = clusters[offset:offset + limit]
    next_offset = offset + len(page)
    truncated = next_offset < total
    return {
        "readonly": True,
        "method": method,
        "total_clusters": total,
        "returned_clusters": len(page),
        "offset": offset,
        "next_offset": next_offset if truncated else None,
        "truncated": truncated,
        "input_rows": sum(len(nodes) for nodes in partitions.values()),
        "input_truncated": bool(input_truncated),
        "clusters": page,
        "hint": "Read-only. Partition is backend, simulated, and route intent. Review one cluster at a time; this command does not close claims.",
    }
