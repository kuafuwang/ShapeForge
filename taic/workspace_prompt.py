"""The transport-only TASK profile for an optional SSH workspace session.

The local contract is rendered by ``taic.kernel_agent.render_contract`` and is not touched
here: its source is part of the trust fingerprint, and the local bytes must not move.  A
session whose workspace is a remote host needs a different prompt surface — it cannot run
the control_host ``tcenv eval`` command, cannot reach the evaluator socket and must not be
told that a probe it writes here proves anything about the device.  That surface is derived
from the local text by dropping the sections that only make sense on the control host, and
is hashed separately by the runner.

Every transformation anchors on a literal from the local renderer and raises if the anchor
is gone.  A future edit to the local contract therefore fails this profile loudly instead
of silently shipping control_host paths to a remote workspace.
"""

from __future__ import annotations

from .kernel_agent import SessionBrief, render_contract

_REMOTE_CHECK = """# 本场没有自测通道

这个工作目录在一台远端主机上，控制机的仓库、评测器和 GPU 都不在这里：不要运行
`uv run ... tcenv eval`、`taic-selfcheck` 或任何控制机路径下的命令，也不要写探针去测这台机器。
本场只写核与笔记；交卷后由外层在控制机上独立评测。你在 NOTES/FEEDBACK 里写的任何数字都只是
文本，不是成绩。

"""

_PROMOTION = """# 晋升标准

外层换种子要同时快过 {gain} 倍，并且超过 {z} 个标准误。这个门槛用的是外层评测的微秒与 cv；
本场看不到自测结果，按已知最好的写法交卷。
"""


def _replace_once(text: str, old: str, new: str) -> str:
    """Replace one anchored block, refusing a contract whose anchor moved."""
    if text.count(old) != 1:
        raise RuntimeError("the local TASK contract no longer holds the anchor this remote profile needs")
    return text.replace(old, new)


def render_remote_contract(brief: SessionBrief, *, max_bytes: int | None = None) -> str:
    """Render the TASK for a workspace the agent can reach but the evaluator cannot.

    @param brief the session contract.
    @param max_bytes optional final size budget; the local renderer already applied its own.
    @return the remote-profile TASK text.
    @raises RuntimeError if the local renderer's anchors are no longer present.
    """
    text = render_contract(brief)
    # The two command sections are exactly the span between these headings.
    start = text.index("\n# 验证命令\n")
    end = text.index("\n# 晋升标准\n")
    text = text[:start] + "\n" + _REMOTE_CHECK + text[end + 1 :]
    # The promotion section still names `--timed`; replace its whole body.
    promotion_start = text.index("# 晋升标准\n")
    promotion_end = text.find("\n# ", promotion_start + 1)
    promotion_end = len(text) if promotion_end < 0 else promotion_end + 1
    text = (
        text[:promotion_start]
        + _PROMOTION.format(gain=brief.min_improvement, z=brief.noise_z)
        + text[promotion_end:]
    )
    text = _replace_once(
        text,
        "固定的是合同签名、容差、围栏和预算。可质疑的是笔记、文档和队友留下的前提。你可以读整个仓库，只能写本目录。",
        "固定的是合同签名、容差、围栏和预算。可质疑的是笔记、文档和队友留下的前提。本场只有这个工作目录可读写；控制机的仓库不在本机。",
    )
    text = _replace_once(
        text,
        "写完跑一次：\n```\n" + _local_feedback_command(text) + "\n```\n",
        "外层会在会话结束后校验 FEEDBACK.md 的格式，你不需要在本机运行校验命令。\n",
    )
    text = _replace_once(
        text,
        f"某版一过 `--quick` 就用 `cp` 复制成",
        "某版写稳了就用 `cp` 复制成",
    )
    # The seed-comparison note is only meaningful where the evaluation command exists, and
    # it carries a control_host path. It is the last appended block.
    marker = "\n\n合同里的外层成绩和你自己跑出来的数字对不上时"
    if marker in text:
        tail = text[text.index(marker) :]
        stop = tail.find("\n\n# ")
        removed = tail if stop < 0 else tail[:stop]
        text = text.replace(removed, "", 1)
    text = text.replace("`--timed`", "外层评测")
    if max_bytes is not None and len(text.encode("utf-8")) > max_bytes:
        raise RuntimeError("the remote TASK profile overran its own byte budget")
    return text


def _local_feedback_command(text: str) -> str:
    """The one feedback-check command line the local renderer printed.

    It is the `uv run --directory ... taic.agent_feedback check ...` line between the
    `写完跑一次：` fence.  Reading it from the text keeps this profile anchored to whatever
    the local renderer actually printed.
    """
    marker = "写完跑一次：\n```\n"
    start = text.index(marker) + len(marker)
    end = text.index("\n```\n", start)
    return text[start:end]


__all__ = ["render_remote_contract"]
