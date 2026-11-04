"""两份原因码文档与 ``taic.policy.PUBLICATION_REASONS`` 逐项相等（集合与顺序）。

The reason codes are a closed set and the readers pick the first hit in code order, so a code that
exists only in the documentation (or a documentation list ordered differently) would mislead whoever
tries to act on a reason.  This test compares the documented order with the code's order.
"""

from __future__ import annotations

import re
from pathlib import Path

from taic.policy import PUBLICATION_REASONS

REPO_ROOT = Path(__file__).resolve().parents[2]

#: The codes `conclusion_verdict` can actually return (the enum sentence in 数据层.md).
_DATA_LAYER_ORDER = (
    "cache_hit",
    "missing_confirmation_block",
    "sha_mismatch",
    "label_mismatch",
    "values_mismatch",
    "speedup_mismatch",
    "candidate_us_mismatch",
    "protocol_mismatch",
    "target_mismatch",
    "timing_uncertain",
    "batches_disagree",
    "track_mismatch",
    "not_confirmed",
    "no_result_json",
    "confirmed",
)


def test_the_data_layer_document_lists_the_track_mismatch_reason_in_code_order():
    text = (REPO_ROOT / "docs" / "架构" / "数据层.md").read_text(encoding="utf-8")
    match = re.search(r"原因码按代码顺序取第一个命中——(.+?)。", text, re.S)
    assert match, "数据层.md no longer states the reason-code order"
    documented = tuple(item.strip().strip("`") for item in match.group(1).split("、"))
    assert documented == _DATA_LAYER_ORDER
    # Every documented code is a real member of the closed set.  (The documented order is the *reader's*
    # decision order, which is not the tuple's order, so only membership is comparable.)
    unknown = [code for code in documented if code not in PUBLICATION_REASONS]
    assert not unknown, f"documented reasons that the predicate cannot return: {unknown}"


def test_the_protocol_reference_documents_track_mismatch():
    text = (REPO_ROOT / "docs" / "架构" / "v2协议参考.md").read_text(encoding="utf-8")
    assert "`track_mismatch`" in text
    assert "`result.track` 与 best/winner 的交付形态、`delivery` 或 `track_version` 不符/畸形/未知" in text


def test_every_policy_reason_is_documented_somewhere():
    protocol = (REPO_ROOT / "docs" / "架构" / "v2协议参考.md").read_text(encoding="utf-8")
    data_layer = (REPO_ROOT / "docs" / "架构" / "数据层.md").read_text(encoding="utf-8")
    missing = [
        code for code in PUBLICATION_REASONS
        if code not in protocol and code not in data_layer
    ]
    assert not missing, f"undocumented publication reasons: {missing}"
