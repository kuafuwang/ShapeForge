"""两份原因码文档与 ``taic.policy.PUBLICATION_REASONS`` 对得上。

原因码是闭集。``docs/架构/数据层.md`` 那一行列的是**排查顺序**，不是谓词的 first-hit 顺序：
同一个码会在 ``confirmation_publication`` 里多处命中（``not_confirmed`` 就出现三次），first-hit
顺序也不是一条稳定序列。所以这里比较的是集合：文档不能少码、不能多码、不能重复；文档若又写回
「按代码顺序」这句错话，测试同样失败。
"""

from __future__ import annotations

import re
from pathlib import Path

from taic.policy import PUBLICATION_REASONS

REPO_ROOT = Path(__file__).resolve().parents[2]

#: 数据层.md 里那张「排查优先级」列表；列表用「、」分隔，反引号包住每个码。
_TRIAGE = re.compile(r"下面按\*\*排查优先级\*\*排列（不是代码判定顺序）：(.+?)。", re.S)


def test_the_closed_set_has_no_duplicates_and_holds_the_only_passing_value():
    """@brief 闭集里有且只有一个通过值 ``confirmed``，且不重复。"""
    assert isinstance(PUBLICATION_REASONS, tuple)
    assert "confirmed" in PUBLICATION_REASONS
    passing = [code for code in PUBLICATION_REASONS if code == "confirmed"]
    assert len(passing) == 1
    duplicates = sorted({code for code in PUBLICATION_REASONS if PUBLICATION_REASONS.count(code) > 1})
    assert not duplicates, f"duplicated publication reasons: {duplicates}"


def test_the_data_layer_document_lists_every_reason_once():
    """@brief 文档的排查列表 == 代码闭集；少一个、多一个、重复一个都失败。"""
    text = (REPO_ROOT / "docs" / "架构" / "数据层.md").read_text(encoding="utf-8")
    assert "原因码按代码顺序取第一个命中" not in text, (
        "数据层.md 又把它写成了代码顺序：谓词里的 first-hit 顺序不是文档这一列的顺序"
    )
    match = _TRIAGE.search(text)
    assert match, "数据层.md no longer states the triage order of the reason codes"
    documented = tuple(item.strip().strip("`") for item in match.group(1).split("、"))
    duplicates = sorted({code for code in documented if documented.count(code) > 1})
    missing = sorted(set(PUBLICATION_REASONS) - set(documented))
    extra = sorted(set(documented) - set(PUBLICATION_REASONS))
    assert not duplicates, f"duplicated reason codes in 数据层.md: {duplicates}"
    assert not missing and not extra, f"数据层.md reason list drifted: missing={missing} extra={extra}"
    assert len(documented) == len(PUBLICATION_REASONS)


def test_the_data_layer_document_states_the_current_closed_set_size():
    """@brief 散文里写了码的数量，就由代码来钉住它，免得闭集长大而文档停在旧数。"""
    text = (REPO_ROOT / "docs" / "架构" / "数据层.md").read_text(encoding="utf-8")
    assert f"闭集（{len(PUBLICATION_REASONS)} 个" in text


def test_the_data_layer_document_names_the_closed_set_and_the_first_hit_rule():
    """@brief 读者要能查到权威源，并知道一个结果只报第一个命中的码。"""
    text = (REPO_ROOT / "docs" / "架构" / "数据层.md").read_text(encoding="utf-8")
    assert "`taic/policy.py::PUBLICATION_REASONS`" in text
    assert "**第一个命中**" in text


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
