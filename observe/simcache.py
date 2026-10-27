"""句向量与相似结论的持久缓存：派生、可删重建，与主索引分库。

缓存库跟在 ``OBSERVE_DB`` 旁边（``<stem>.similar.sqlite``），所以测试改 ``OBSERVE_DB`` 就自然隔离，
也不会污染真实看板的缓存。这里只存派生结果，不是第二份事实源：

- ``embeddings``：按（模型 epoch + 语言 + 归一化文本）内容寻址的句向量。句向量是确定性函数，
  没有理由每个进程都重算一遍；Apple NaturalLanguage 模型由系统提供，epoch 变了就换键。
- ``similar``：每条 claim 的语义近邻，带 ``corpus_count`` / ``corpus_rev``。近邻只从**更早**的
  claim 里选，所以语料键不变时结果不可变；一旦出现新的更早候选（或既有候选被改写），键就变，
  读取方必须重算，不能拿旧结果冒充新鲜。

后台预热（:func:`start_warmer`）把这些派生结果提前算好，请求路径只读缓存；预热每次只处理一条
claim 并让出 GIL，并且**不占主索引的写锁**（:func:`observe.index.reader` 的只读连接），避免重计算
挤占看板其它请求。缓存库有行数上限，:func:`prune` 从最旧的行开始删，库不会只增不减。
"""

from __future__ import annotations

import array
import hashlib
import json
import sqlite3
import threading
import time
from pathlib import Path

MODEL_EPOCH = "nl-1"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS embeddings (
    text_sha TEXT PRIMARY KEY,
    language TEXT NOT NULL,
    dim INTEGER NOT NULL,
    vector BLOB NOT NULL,
    at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS similar (
    run_id TEXT NOT NULL,
    seq INTEGER NOT NULL,
    digest TEXT NOT NULL,
    corpus_count INTEGER NOT NULL,
    corpus_rev INTEGER NOT NULL,
    payload TEXT NOT NULL,
    at REAL NOT NULL,
    PRIMARY KEY (run_id, seq, digest)
);
CREATE TABLE IF NOT EXISTS warm_state (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    state TEXT NOT NULL,
    at REAL NOT NULL
);
"""

# 行数上限：缓存是派生物，越界就从最旧的行开始删。similar 一行约 1KB（实测 90 行 107KB），
# embeddings 一行约 2.5KB（363 行 922KB）；两个上限合计把库压在几十 MB 量级。
SIMILAR_MAX_ROWS = 4096
EMBEDDING_MAX_ROWS = 8192

_LOCAL = threading.local()
_WRITE = threading.Lock()
_WARMER: threading.Thread | None = None


def cache_path() -> Path:
    """@brief 返回缓存库路径：主索引路径的同目录兄弟文件。

    @return ``<observe 主索引 stem>.similar.sqlite``。
    """
    from observe.index import db_path

    db = db_path()
    return db.with_name(db.stem + ".similar.sqlite")


def _conn() -> sqlite3.Connection:
    """@brief 返回当前线程、当前缓存库的连接（换 OBSERVE_DB 即换库）。

    @return sqlite3 连接；每线程一把，WAL 模式。
    """
    path = str(cache_path())
    conn = getattr(_LOCAL, "conn", None)
    if conn is not None and getattr(_LOCAL, "path", "") == path:
        return conn
    if conn is not None:
        conn.close()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=30)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.executescript(_SCHEMA)
    _LOCAL.conn = conn
    _LOCAL.path = path
    return conn


def _text_sha(language: str, text: str) -> str:
    """@brief 文本内容寻址键；模型 epoch 与语言不同就是不同的向量。

    @param language 语言标识。
    @param text 已归一化的文本。
    @return sha256 十六进制。
    """
    return hashlib.sha256(f"{MODEL_EPOCH}\0{language}\0{text}".encode("utf-8")).hexdigest()


def get_embedding(language: str, text: str) -> tuple[float, ...] | None:
    """@brief 读持久句向量；没有这条缓存返回 None。

    @param language 语言标识。
    @param text 归一化文本。
    @return 向量元组或 None。
    """
    row = _conn().execute(
        "SELECT vector, dim FROM embeddings WHERE text_sha = ?", (_text_sha(language, text),),
    ).fetchone()
    if row is None:
        return None
    values = array.array("f")
    values.frombytes(row[0])
    if len(values) != int(row[1]):
        return None
    return tuple(float(value) for value in values)


def put_embedding(language: str, text: str, vector: tuple[float, ...]) -> None:
    """@brief 写入一条句向量；已存在就覆盖（同一内容键，值本就确定）。

    @param language 语言标识。
    @param text 归一化文本。
    @param vector 向量。
    """
    blob = array.array("f", vector).tobytes()
    with _WRITE:
        conn = _conn()
        conn.execute(
            "INSERT OR REPLACE INTO embeddings(text_sha, language, dim, vector, at) VALUES (?,?,?,?,?)",
            (_text_sha(language, text), language, len(vector), blob, time.time()),
        )
        conn.commit()


def get_similar(run_id: str, seq: int, digest: str, corpus: tuple[int, int]) -> list[dict] | None:
    """@brief 读一条 claim 的近邻缓存；语料键不同视为未命中。

    @param run_id 该 claim 所属运行。
    @param seq 反馈序号。
    @param digest claim 摘要。
    @param corpus ``(候选条数, 候选 rev)``，由调用方按当前索引现算。
    @return 近邻行列表；未命中或语料已变返回 None。
    """
    row = _conn().execute(
        "SELECT corpus_count, corpus_rev, payload FROM similar WHERE run_id = ? AND seq = ? AND digest = ?",
        (run_id, int(seq), digest),
    ).fetchone()
    if row is None or (int(row[0]), int(row[1])) != (int(corpus[0]), int(corpus[1])):
        return None
    return list(json.loads(row[2]))


def put_similar(run_id: str, seq: int, digest: str, corpus: tuple[int, int], rows: list[dict]) -> None:
    """@brief 写入一条 claim 的近邻结果与当时的语料键。

    @param run_id 该 claim 所属运行。
    @param seq 反馈序号。
    @param digest claim 摘要。
    @param corpus 计算时的语料键。
    @param rows 近邻行（必须是 JSON 可序列化的字典）。
    """
    payload = json.dumps(rows, ensure_ascii=False, default=str)
    with _WRITE:
        conn = _conn()
        conn.execute(
            "INSERT OR REPLACE INTO similar(run_id, seq, digest, corpus_count, corpus_rev, payload, at)"
            " VALUES (?,?,?,?,?,?,?)",
            (run_id, int(seq), digest, int(corpus[0]), int(corpus[1]), payload, time.time()),
        )
        conn.commit()


def cached_keys(run_ids) -> dict[tuple[str, int, str], tuple[int, int]]:
    """@brief 批量读一批运行的缓存键（不读 payload），供预热做廉价的工作发现。

    @param run_ids 运行 id 序列（可重复）。
    @return ``{(run_id, seq, digest): (corpus_count, corpus_rev)}``。
    """
    ids = sorted({str(item) for item in run_ids if item})
    if not ids:
        return {}
    marks = ",".join("?" for _ in ids)
    rows = _conn().execute(
        "SELECT run_id, seq, digest, corpus_count, corpus_rev FROM similar WHERE run_id IN (" + marks + ")",
        ids,
    )
    return {
        (str(row[0]), int(row[1]), str(row[2])): (int(row[3]), int(row[4]))
        for row in rows
    }


def get_warm_state() -> str | None:
    """@brief 读“上次预热收敛时”的状态指纹；没有记录返回 None。

    @return 指纹字符串或 None。
    """
    row = _conn().execute("SELECT state FROM warm_state WHERE id = 1").fetchone()
    return None if row is None else str(row[0])


def set_warm_state(state: str) -> None:
    """@brief 记下预热已收敛的状态指纹，让空闲轮只花几条廉价查询。

    @param state 状态指纹。
    """
    with _WRITE:
        conn = _conn()
        conn.execute(
            "INSERT OR REPLACE INTO warm_state(id, state, at) VALUES (1, ?, ?)", (str(state), time.time()),
        )
        conn.commit()


def stats() -> dict[str, int]:
    """@brief 返回缓存行数与字节数，供运维与测试观察库的增长。

    @return ``{"similar": …, "embeddings": …, "similar_bytes": …, "embedding_bytes": …}``。
    """
    conn = _conn()
    similar, similar_bytes = conn.execute("SELECT COUNT(*), COALESCE(SUM(LENGTH(payload)), 0) FROM similar").fetchone()
    embeddings, embedding_bytes = conn.execute(
        "SELECT COUNT(*), COALESCE(SUM(LENGTH(vector)), 0) FROM embeddings",
    ).fetchone()
    return {
        "similar": int(similar),
        "embeddings": int(embeddings),
        "similar_bytes": int(similar_bytes),
        "embedding_bytes": int(embedding_bytes),
    }


def prune(*, max_similar: int | None = None, max_embeddings: int | None = None) -> dict[str, int]:
    """@brief 把两张表压回上限内：从 ``at`` 最旧的行开始删。

    删除只按行数与写入时间，按“最旧先删”，所以新近的 claim 不会被删掉再让请求路径重算。
    缓存是派生物，删掉的行下一次预热会重新写回来。上限默认取模块常量，调用时才读，
    运维调参（或测试）改常量即生效。

    @param max_similar ``similar`` 表允许的行数上限，None 表示 :data:`SIMILAR_MAX_ROWS`。
    @param max_embeddings ``embeddings`` 表允许的行数上限，None 表示 :data:`EMBEDDING_MAX_ROWS`。
    @return ``{"similar": 删除行数, "embeddings": 删除行数}``。
    """
    if max_similar is None:
        max_similar = SIMILAR_MAX_ROWS
    if max_embeddings is None:
        max_embeddings = EMBEDDING_MAX_ROWS
    removed = {"similar": 0, "embeddings": 0}
    with _WRITE:
        conn = _conn()
        for table, limit in (("similar", int(max_similar)), ("embeddings", int(max_embeddings))):
            count = int(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
            excess = count - max(0, limit)
            if excess <= 0:
                continue
            conn.execute(
                f"DELETE FROM {table} WHERE rowid IN (SELECT rowid FROM {table} ORDER BY at ASC LIMIT ?)",
                (excess,),
            )
            removed[table] = excess
        conn.commit()
    return removed


def start_warmer() -> None:
    """@brief 启动后台预热线程（幂等）；只在真实看板进程里调用，测试不启动。

    每次只推进一条 claim 的相似结论，之间睡眠让出 GIL；没有工作时休眠 2 秒。
    预热失败只等下一轮，不影响看板读取。
    """
    global _WARMER
    if _WARMER is not None and _WARMER.is_alive():
        return

    def loop() -> None:
        from observe import feedback
        from observe.index import get_index

        while True:
            try:
                # 只读连接：预热不占主索引的写锁，请求线程不用排队等它。
                did = feedback.warm_similar_once()
            except Exception:  # 预热是优化，绝不能弄挂看板；库还没建好就等下一次
                try:
                    get_index()
                except Exception:
                    pass
                did = False
            time.sleep(0.05 if did else 2.0)

    _WARMER = threading.Thread(target=loop, name="observe-similar-warmer", daemon=True)
    _WARMER.start()
