"""Read-only helpers for the frozen ``hw_off.json`` v1 prompt/feedback record.

``tests/support/golden/metal/hw_off.json`` was produced by ``--record`` while the v1 prompts
were still rendered from assembled v1 tasks.  That generator is retired together with the v1
task assembly: the file is now a **historical document**, not a regenerated golden.  This module
keeps only the parts of the old helper that do not need a task — the request-key whitelist and
fixture digests — plus the frozen digest of the document itself, so the record stays an explicit
read-only input instead of an orphan.

The documented producer command remains
``uv run python -m tests.support.hw_golden --record --reason "..."`` in the archive; running it
now is impossible by design, because the v1 prompt path it sampled no longer has a task source.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

GOLDEN_DIR = Path(__file__).resolve().parent / "golden" / "metal"
GOLDEN = GOLDEN_DIR / "hw_off.json"
FIXTURES = GOLDEN_DIR / "fixtures"

#: Keys ``LLM._params`` is allowed to put on the wire.  ``messages`` is added by ``chat`` and is
#: hashed with them.
REQUEST_KEYS = frozenset({"model", "max_tokens", "reasoning_effort", "extra_body", "tools"})

#: The exact bytes of the frozen document (recorded before the v1 prompt path was retired).
GOLDEN_SHA256 = "78ce8c0e2035756aae0e85dca6f3b4a45cd1b44e14c3578d48a15cd3d068b03e"


def sha(text: str) -> str:
    """Return the sha1 digest used by the recorded document.

    @param text 待哈希的文本。
    @return 40 位十六进制摘要。
    """
    return hashlib.sha1(text.encode()).hexdigest()


def fixture_digest(name: str) -> str:
    """Return the combined sha256 of one fixture's Metal source and recorded report.

    @param name fixture 名，对应 ``<name>.metal`` 与 ``<name>.report.json``。
    @return 十六进制摘要。
    """
    metal = (FIXTURES / f"{name}.metal").read_bytes()
    report = (FIXTURES / f"{name}.report.json").read_bytes()
    return hashlib.sha256(metal + b"\0" + report).hexdigest()


def fixture(name: str) -> tuple[dict, str, dict]:
    """Return ``(index entry, source, report)`` for one recorded fixture.

    @param name fixture 名。
    @return 索引条目、Metal 源码与报告字典。
    """
    index = json.loads((FIXTURES / "index.json").read_text(encoding="utf-8"))
    return (
        index[name],
        (FIXTURES / f"{name}.metal").read_text(encoding="utf-8"),
        json.loads((FIXTURES / f"{name}.report.json").read_text(encoding="utf-8")),
    )


def param_keys() -> set[str]:
    """Return the union of parameter keys ``LLM._params`` may emit.

    @return 键集合。
    """
    from taic.llm import CostLedger, LLM, LLMConfig

    keys: set[str] = set()
    for extra in ({}, {"thinking": {"type": "enabled"}}):
        cfg = LLMConfig(base_url="http://x", api_key="k", model="m", reasoning_effort="high", extra_body=dict(extra))
        llm = LLM(cfg, CostLedger(1.0), client=object())
        for tools in (None, "tools"):
            for limit in (None, 1024):
                for effort in ("high", "low"):
                    keys |= set(llm._params(tools, limit, effort))
    return keys


__all__ = ["FIXTURES", "GOLDEN", "GOLDEN_DIR", "GOLDEN_SHA256", "REQUEST_KEYS", "fixture", "fixture_digest", "param_keys", "sha"]
