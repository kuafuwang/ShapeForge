"""Failure fingerprints: one deterministic signature per (status, stage, normalized message).

This module is the single implementation behind ``candidates.failure_sig``,
``sessions.failure_sig`` and the ``failure_members`` table. The order of the rules
matters and is fixed:

1. redact first (``taic.redact``), then normalize -- the other way round would let the
   path/secret shapes be eaten by the later rules;
2. POSIX paths; 3. Windows paths; 4. turn labels; 5. hexadecimal; 6. tensors;
7. quoted text; 8. numbers; 9. whitespace, then the first 160 characters.

The sign of a number is deliberately not part of the number pattern: ``got -#`` and
``got #`` stay two different clusters ("sign error" vs "magnitude error"). Merging them
would collapse two different fixes into one. ``tests/shared/test_observe_clusters.py::test_one_normalized_signature_covers_numbers_labels_and_paths_but_not_the_sign`` pins it.
"""

from __future__ import annotations

import hashlib
import re

from taic.redact import redact_text

# ``(?<![\w./])`` keeps ``a/b`` (a label, not a path) and ``http://x/y`` mostly intact.
_POSIX_PATH = re.compile(r"(?<![\w./])(?:/[\w.\-]+)+/?")
_WINDOWS_PATH = re.compile(r"[A-Za-z]:\\[^\s]+")
_TURN_LABEL = re.compile(r"\br\d+\.p\d+(?:\.(?:retry|fix)\d+)?\b")
_HEX = re.compile(r"0x[0-9a-fA-F]+")
_TENSOR = re.compile(r"\b[A-Za-z_]\w*\[[^\]]*\]")
_QUOTED = re.compile(r"\"[^\"]*\"|'[^']*'")
# Deliberately no leading minus: ``-12`` and ``12`` must not share a signature.
_NUMBER = re.compile(r"\d+(?:\.\d+)?(?:[eE][+-]?\d+)?")
_SPACES = re.compile(r"\s+")

SIGNATURE_CHARS = 16
NORMALIZED_CHARS = 160
MESSAGE_CHARS = 300


def normalize(message) -> str:
    """@brief Redact, then deterministically normalize a failure message.

    @param message Raw message; non-strings count as empty.
    @return At most 160 characters, whitespace-collapsed, credentials removed.
    """
    text = redact_text(message, one_line=True)
    text = _POSIX_PATH.sub("<path>", text)
    text = _WINDOWS_PATH.sub("<path>", text)
    text = _TURN_LABEL.sub("r?.p?", text)
    text = _HEX.sub("0x#", text)
    text = _TENSOR.sub("<tensor>", text)
    text = _QUOTED.sub("…", text)
    text = _NUMBER.sub("#", text)
    text = _SPACES.sub(" ", text).strip()
    return text[:NORMALIZED_CHARS]


def _digest(text: str) -> str:
    """@brief Short hex digest of one canonical signature string.

    @param text Canonical string.
    @return First 16 lowercase hex characters.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:SIGNATURE_CHARS]


def signature(status, stage, message) -> str:
    """@brief The candidate signature: sha256(status \\x1f stage \\x1f normalized)[:16].

    @param status Candidate status (``ok``/``incorrect``/...).
    @param stage Candidate stage (``correctness``/``compile``/...).
    @param message Raw failure message.
    @return Signature.
    """
    return _digest("\x1f".join((str(status or ""), str(stage or ""), normalize(message))))


# 只有这些结局才算“失败”；evaluated/unchanged 即便残留一条 error 也不是失败会话。
SESSION_FAILURE_OUTCOMES = frozenset({"agent_failed", "provider_failed", "agent_no_submission"})


def session_failure_signature(outcome, failure_class, error) -> str | None:
    """@brief A session signature only for a failing outcome; success rows stay NULL.

    A leftover error on an ``evaluated``/``unchanged`` turn must not create a cluster.

    @param outcome Session outcome (``evaluated``/``agent_failed``/...).
    @param failure_class Runtime failure class, may be empty.
    @param error Raw error text, may be empty.
    @return Signature, or None when this is not a failing outcome.
    """
    if str(outcome or "").strip() not in SESSION_FAILURE_OUTCOMES:
        return None
    return session_signature(failure_class, error)


def session_signature(failure_class, error) -> str | None:
    """@brief The session signature, prefixed so session failures never join candidate clusters.

    Without an error text only the failure class takes part; a non-failure outcome has no
    signature.

    @param failure_class Runtime failure class (``turn_failed``/...); empty for no failure.
    @param error Raw error text, may be empty.
    @return Signature, or None when this is not a failure.
    """
    kind = str(failure_class or "").strip()
    text = normalize(error)
    if not kind and not text:
        return None
    return _digest("\x1f".join(("session", kind, text)))


def display_name(status, stage, message, *, limit: int = 60) -> str:
    """@brief The fallback cluster label: ``status/stage`` plus the normalized text prefix.

    @param status Candidate status.
    @param stage Candidate stage.
    @param message Raw failure message.
    @param limit Maximum characters of the normalized prefix.
    @return Human-readable label.
    """
    head = "/".join(part for part in (str(status or ""), str(stage or "")) if part) or "unknown"
    text = normalize(message)[:limit]
    return f"{head} {text}".strip()
