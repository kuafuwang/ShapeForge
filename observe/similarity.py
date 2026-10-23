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
    return _load()


def _language(text: str) -> str:
    return "zh-Hans" if _CJK.search(text or "") else "en"


def _model(language: str):
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
    """The sentence vector, or ``None`` when this platform cannot embed the text."""
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
    embedding.cache_clear()
    _MODELS.clear()


def _tokens(text: str) -> set[str]:
    """CJK character bigrams plus code-like Latin/number tokens."""
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
    """Shared-token containment. A cheap guard against generic-sounding neighbours."""
    left, right = _tokens(a), _tokens(b)
    if not left or not right:
        return 0.0
    return len(left & right) / min(len(left), len(right))


def cosine(a: tuple[float, ...] | None, b: tuple[float, ...] | None) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def find_similar(text: str, candidates: list[dict], *, threshold: float | None = None, limit: int = 3) -> list[dict]:
    """Return candidates whose sentence vector is close to ``text``.

    Each candidate must have ``text`` and is returned with a ``score`` field.  A
    threshold of 0.90 is intentionally conservative: the local embeddings put clear
    paraphrases near 0.95 and unrelated suggestions much lower, but this is still a
    retrieval hint, not a verdict.
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
