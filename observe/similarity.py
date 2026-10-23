"""Local semantic similarity for feedback review.

Uses Apple's NaturalLanguage sentence embeddings through PyObjC.  It is local and
offline; no model call, network request, or GPU dispatch.  If the framework is not
available, callers get an empty list and must not pretend that a claim has no
semantic neighbours.
"""

from __future__ import annotations

import math
import re
from functools import lru_cache

_CJK = re.compile(r"[\u3400-\u9fff]")
_CJK_FULL = re.compile(r"^[\u3400-\u9fff]+$")
_TOKEN = re.compile(r"[\u3400-\u9fff]+|[a-z0-9_]+(?:[._][a-z0-9_]+)*")
_BUNDLE = None
_CLASS = None
_MODELS: dict[str, object] = {}


def _load() -> bool:
    """@brief 惰性加载 NaturalLanguage 框架与 NLEmbedding 类。

    @return 嵌入能力可用时为 True；框架不可用时返回 False。
    """
    global _BUNDLE, _CLASS
    if _CLASS is not None:
        return True
    try:
        import objc

        if _BUNDLE is None:
            _BUNDLE = objc.loadBundle(
                "NaturalLanguage",
                globals(),
                bundle_path="/System/Library/Frameworks/NaturalLanguage.framework",
            )
        _CLASS = objc.lookUpClass("NLEmbedding")
    except Exception:
        _CLASS = False
    return _CLASS is not False


def available() -> bool:
    """@brief 报告本机是否具备本地句向量能力。

    @return 可用时为 True。
    """
    return _load()


def _language(text: str) -> str:
    """@brief 按文本中是否含 CJK 字符选择嵌入语言。

    @param text 待嵌入文本。
    @return ``"zh-Hans"`` 或 ``"en"``。
    """
    return "zh-Hans" if _CJK.search(text or "") else "en"


def _model(language: str):
    """@brief 按语言取（并缓存）句向量模型。

    @param language 语言标识，如 ``"zh-Hans"``。
    @return 模型对象；框架不可用或建模型失败时返回 None。
    """
    if not _load():
        return None
    if language not in _MODELS:
        try:
            _MODELS[language] = _CLASS.sentenceEmbeddingForLanguage_(language)
        except Exception:
            _MODELS[language] = None
    return _MODELS[language]


@lru_cache(maxsize=4096)
def embedding(text: str) -> tuple[float, ...] | None:
    """@brief 返回文本的句向量；本平台无法嵌入时返回 ``None``。

    原始英文说明：The sentence vector, or ``None`` when this platform cannot embed the text.

    @param text 待嵌入文本，会先折叠空白。
    @return 向量元组；空文本、无模型或调用异常时返回 None。
    """
    text = " ".join(str(text or "").split())
    if not text:
        return None
    model = _model(_language(text))
    if model is None:
        return None
    try:
        vector = model.vectorForString_(text)
        if vector is None:
            return None
        return tuple(float(value) for value in vector)
    except Exception:
        return None


def clear_cache() -> None:
    """@brief 清空 ``embedding`` 的 LRU 缓存与已缓存的模型对象。
    """
    embedding.cache_clear()
    _MODELS.clear()


def _tokens(text: str) -> set[str]:
    """@brief 切出 CJK 字 bigram 与代码风格的拉丁/数字 token。

    原始英文说明：CJK character bigrams plus code-like Latin/number tokens.

    @param text 待切分文本。
    @return token 集合（已转小写）。
    """
    out: set[str] = set()
    for match in _TOKEN.finditer(str(text or "").lower()):
        token = match.group(0)
        if _CJK_FULL.fullmatch(token):
            if len(token) == 1:
                out.add(token)
            else:
                out.update(token[index:index + 2] for index in range(len(token) - 1))
        else:
            out.add(token)
    return out


def lexical_overlap(a: str, b: str) -> float:
    """@brief 计算共享 token 的包含度，作为对“听起来很泛”的近邻的廉价拦截。

    原始英文说明：Shared-token containment. A cheap guard against generic-sounding neighbours.

    @param a 第一段文本。
    @param b 第二段文本。
    @return 交集大小除以较小集合大小；任一为空时返回 0.0。
    """
    left, right = _tokens(a), _tokens(b)
    if not left or not right:
        return 0.0
    return len(left & right) / min(len(left), len(right))


def cosine(a: tuple[float, ...] | None, b: tuple[float, ...] | None) -> float:
    """@brief 计算两个向量的余弦相似度。

    @param a 第一个向量，可为 None。
    @param b 第二个向量，可为 None。
    @return 余弦值；任一侧为空、长度不等或模长为零时返回 0.0。
    """
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def find_similar(text: str, candidates: list[dict], *, threshold: float | None = None, limit: int = 3) -> list[dict]:
    """@brief 返回句向量与 ``text`` 足够接近的候选。

    每个候选需带 ``text`` 字段，返回时附加 ``score``。阈值为 0.90 量级时刻意保守：本地嵌入会把
    明显的改写放在 0.95 附近，把无关建议放得低得多，但这仍只是检索提示，不是结论。
    原始英文说明：Each candidate must have ``text`` and is returned with a ``score`` field.

    @param text 查询文本。
    @param candidates 候选字典列表，每项需含 ``text``。
    @param threshold 相似度阈值，None 时按语言取保守默认值。
    @param limit 最多返回条数。
    @return 按分数降序排列的候选副本列表；无法嵌入时返回空列表。
    """
    source = embedding(text)
    if source is None:
        return []
    if threshold is None:
        # Chinese sentence vectors have a high baseline; English vectors are much
        # more spread out. These are deliberately conservative, not a classifier.
        threshold = 0.975 if _language(text) == "zh-Hans" else 0.65
    min_overlap = 0.20 if _language(text) == "zh-Hans" else 0.15
    scored = []
    for candidate in candidates:
        candidate_text = str(candidate.get("text") or "")
        if lexical_overlap(text, candidate_text) < min_overlap:
            continue
        vector = embedding(candidate_text)
        if vector is None:
            continue
        score = cosine(source, vector)
        if score >= threshold:
            scored.append({**candidate, "score": round(score, 4)})
    scored.sort(key=lambda row: (-float(row["score"]), str(row.get("run_id") or ""), int(row.get("seq") or 0)))
    return scored[: max(0, int(limit))]
