"""Human names for failure clusters: a static signature -> plain-language dict.

No automatic naming and no model call. A signature that is not in the dict falls back to
``status/stage`` plus the normalized message prefix (``failure_sig.display_name``).
"""

from __future__ import annotations

from observe.failure_sig import display_name

# 题注：签名来自 §7.1 的确定性归一化；改了归一化规则就必须重算这里的键。
CLUSTER_NAMES: dict[str, str] = {
    # corpus_disk: 133 条
    "479f61991a0e0edc": "ABI / 启动约定不匹配",
    # 「符号错」与「量级错」刻意分开，见 failure_sig 的注释。
    "e8cef4542baf5c91": "数值超差：符号错",
    "ff7b8bece252eb68": "数值超差：量级错",
    "d7c132970aa6924f": "数值超差：符号错（反向）",
    "5b52576c10a4b539": "Metal 编译失败",
    "ac1f84461cc49aef": "与当前种子相同",
}


def cluster_name(sig: str, status: str = "", stage: str = "", message: str = "") -> str:
    """@brief Look up a human name, falling back to ``status/stage`` plus the text prefix.

    @param sig Failure signature.
    @param status Candidate status for the fallback label.
    @param stage Candidate stage for the fallback label.
    @param message Raw message for the fallback label.
    @return Cluster name.
    """
    key = str(sig or "")
    if key in CLUSTER_NAMES:
        return CLUSTER_NAMES[key]
    short = key[:8]
    for known, name in CLUSTER_NAMES.items():
        if short and known.startswith(short):
            return name
    return display_name(status, stage, message)
