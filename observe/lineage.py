"""§8.3 种子链与跨场同源：两条不同的事实，各自命名。

``trace.parent_label`` 在真实语料里 0 条非空（``diversity_archive`` 默认关），所以“谁编辑自谁”
没有被记录。这里只给两件能证伪的事，且都不读 run 目录（``improved`` 与 ``kernel_sha256`` 是
摄入时的索引窄列）：

1. ``derived_seed``：由 trace 的 ``improved`` 序列**推导**每颗候选当时的种子。根集合先定义：
   ``{首个候选} ∪ {init, init2, init3, …, prior}``。规则：非根候选的种子 = 它之前最近的
   “当前种子”（上一次 improved 的候选，或最近一个根）；``improved`` 的候选成为后续候选的新种子。
   边数 = ``n − |roots|``。它是推导，不是作者声明。
2. ``cross_run_source_match``：``candidates.kernel_sha256`` 相等（含 ``000_prior`` / ``000_init``）。
   源码相同 ≠ 由它衍生。

面板受 §8.5 的 8KB 预算约束：先丢跨场匹配，再丢种子边，最后丢节点，任何丢失都在 ``truncated``
里报出来。
"""

from __future__ import annotations

import hashlib
import json
import math
import re

from observe import detail
from observe.index import borrow

LINEAGE_BYTES = 8192
LINEAGE_LIMIT = 200
MATCH_LIMIT = 200
ROOT_LABEL = re.compile(r"^(?:init\d*|prior)$")

NOTES = {
    "derived_seed": "按轨迹推导的种子关系，不是作者声明",
    "cross_run_source_match": "源码相同 ≠ 由它衍生",
}


def _bytes(document: dict) -> int:
    """@brief 与验收一致的 UTF-8 字节数。

    @param document 面板字典。
    @return 字节数。
    """
    return len(json.dumps(document, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))


def _stamp(parts: list[str]) -> str:
    """@brief 面板戳：同一份索引投影与同一场运行得到同一个值。

    @param parts 源指纹与运行 id。
    @return 十六进制摘要。
    """
    return hashlib.sha256("\x1f".join(["lineage-v1", *parts]).encode("utf-8")).hexdigest()


def _is_root(idx: int, label: str) -> bool:
    """@brief 根：首个候选，或 ``init`` / ``init2`` / … / ``prior``（它们没有父节点）。

    @param idx 候选序号。
    @param label 候选标签。
    @return 是根时为 True。
    """
    return idx == 0 or bool(ROOT_LABEL.match(label))


def _num(value) -> float | None:
    """@brief 只接受有限数值（bool 不算）。

    @param value 原始值。
    @return 有限浮点数，或 None。
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _candidate_rows(run_id: str) -> tuple[list[dict], int]:
    """@brief 一场运行的全部候选（窄列）与索引 revision。

    @param run_id 运行 id。
    @return ``(候选行, revision)``。
    """
    with borrow() as index:
        rows = [
            dict(row) for row in index.conn.execute(
                "SELECT idx, label, status, speedup, improved, kernel_sha256"
                " FROM candidates WHERE run_id = ? ORDER BY idx",
                (run_id,),
            )
        ]
        return rows, index.revision()


def _derive(rows: list[dict]) -> tuple[list[dict], list[dict], list[str]]:
    """@brief 从 ``improved`` 序列推导种子链（节点、边、根）。

    @param rows 按 ``idx`` 排序的候选行。
    @return ``(节点, 边, 根标签)``。
    """
    nodes: list[dict] = []
    edges: list[dict] = []
    roots: list[str] = []
    seed: dict | None = None
    for row in rows:
        idx = int(row["idx"] or 0)
        label = str(row["label"] or f"#{idx}")
        improved = bool(row["improved"])
        root = _is_root(idx, label)
        node = {
            "idx": idx,
            "label": label,
            "status": str(row["status"] or ""),
            "speedup": _num(row["speedup"]),
            "improved": improved,
            "root": root,
            "sha256": str(row["kernel_sha256"] or "")[:12],
        }
        nodes.append(node)
        if root:
            # 根没有父节点，也不许跨根连边；它把“当前种子”重置成自己。
            roots.append(label)
            seed = node
            continue
        if seed is not None:
            edges.append({"from": seed["label"], "to": label, "kind": "derived_seed"})
        if improved:
            seed = node
    return nodes, edges, roots


def _cross_run(rows: list[dict], run_id: str) -> tuple[list[dict], int]:
    """@brief 其它场里源码 sha 相同的候选（``cross_run_source_match``）。

    @param rows 本场候选。
    @param run_id 本场运行 id。
    @return ``(匹配行, 完整匹配数)``。
    """
    by_sha: dict[str, str] = {}
    for row in rows:
        sha = str(row["kernel_sha256"] or "")
        if sha and sha not in by_sha:
            by_sha[sha] = str(row["label"] or "")
    if not by_sha:
        return [], 0
    shas = sorted(by_sha)
    placeholders = ",".join("?" for _ in shas)
    with borrow() as index:
        total = int(index.conn.execute(
            f"SELECT COUNT(*) FROM candidates WHERE run_id != ? AND kernel_sha256 IN ({placeholders})",
            (run_id, *shas),
        ).fetchone()[0] or 0)
        rows_out = [
            dict(row) for row in index.conn.execute(
                f"SELECT run_id, label, kernel_sha256 FROM candidates"
                f" WHERE run_id != ? AND kernel_sha256 IN ({placeholders})"
                " ORDER BY run_id, idx LIMIT ?",
                (run_id, *shas, MATCH_LIMIT),
            )
        ]
    matches = [{
        "run_id": str(row["run_id"]),
        "label": str(row["label"] or ""),
        "from_label": by_sha.get(str(row["kernel_sha256"]), ""),
        "sha256": str(row["kernel_sha256"] or "")[:12],
        "kind": "cross_run_source_match",
    } for row in rows_out]
    return matches, total


def seed_chain(run_id: str) -> dict | None:
    """@brief 一场运行的种子链 + 跨场同源（索引投影，受 8KB 预算约束）。

    @param run_id 运行 id。
    @return 模型字典；运行目录不存在时返回 None。
    """
    if detail.resolve_run(run_id) is None:
        return None
    rows, revision = _candidate_rows(run_id)
    nodes, edges, roots = _derive(rows)
    matches, match_total = _cross_run(rows, run_id)
    edges_total = len(edges)
    truncated = {
        "nodes": len(nodes) > LINEAGE_LIMIT,
        "edges": len(edges) > LINEAGE_LIMIT,
        "roots": len(roots) > LINEAGE_LIMIT,
        "cross_run": match_total > len(matches),
    }
    nodes, edges = nodes[:LINEAGE_LIMIT], edges[:LINEAGE_LIMIT]
    roots_out = roots[:LINEAGE_LIMIT]
    document = {
        "schema_version": "1",
        "run_id": run_id,
        "nodes": nodes,
        "edges": edges,
        "roots": roots_out,
        "roots_total": len(roots),
        "cross_run": matches,
        "cross_run_total": match_total,
        # 完整集统计独立保留：裁边只影响展示，不把真实边数变成 0。
        "edge_counts": {"derived_seed": edges_total},
        "edges_shown": len(edges),
        "notes": dict(NOTES),
        "truncated": truncated,
        "stamp": _stamp([str(revision), run_id, *(
            f"{row['idx']}:{row['label']}:{int(bool(row['improved']))}:{row['kernel_sha256'] or ''}"
            for row in rows
        )]),
    }
    _trim(document)
    return document


def _trim_once(document: dict) -> bool:
    """@brief 按“最不重要先丢”的顺序丢一项：跨场匹配 → 种子边 → 节点 → 根名。

    @param document 面板字典（就地修改）。
    @return 丢掉了至少一项时为 True。
    """
    if document["cross_run"]:
        document["cross_run"].pop()
        document["truncated"]["cross_run"] = True
    elif document["edges"]:
        document["edges"].pop()
        document["truncated"]["edges"] = True
        document["edges_shown"] = len(document["edges"])
    elif document["nodes"]:
        document["nodes"].pop()
        document["truncated"]["nodes"] = True
    elif document["roots"]:
        document["roots"].pop()
        document["truncated"]["roots"] = True
    else:
        return False
    return True


def _trim(document: dict, *, budget: int = LINEAGE_BYTES) -> None:
    """@brief 把内层文档收到字节预算内（跨场匹配 → 边 → 节点 → 根名）。

    @param document 内层文档。
    @param budget 字节上限。
    """
    while _bytes(document) > budget and _trim_once(document):
        pass


def lineage_panel(run_id: str, *, stamp: str = "") -> dict | None:
    """@brief 详情面板：种子链与跨场同源，按需加载。

    @param run_id 运行 id。
    @param stamp 客户端持有的面板戳。
    @return ``{"run_id", "same", "stamp", "lineage"}``；运行不存在时返回 None。
    """
    document = seed_chain(run_id)
    if document is None:
        return None
    if stamp and stamp == document["stamp"]:
        return {"run_id": run_id, "same": True, "stamp": document["stamp"]}
    panel = {"run_id": run_id, "same": False, "stamp": document["stamp"], "lineage": document}
    # 面板/HTTP 响应比内层文档多一层包装；按响应口径再收一次，不能让它超出 8KB。
    while _bytes(panel) > LINEAGE_BYTES and _trim_once(document):
        pass
    return panel
