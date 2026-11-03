"""Host-side routing for structured proposals: who can act on this, here?

A proposal may *declare* an audience, but the declaration is only a hint.  The host
re-reads the action and decides three things independently:

- ``recipient``: ``kernel_agent`` (a peer can use it in this session), ``developer``
  (it asks for a system/repository change), or ``unknown``;
- ``requires_repo_change``: whether the requested object is the repository/harness
  rather than the kernel or a measurement in the session workdir;
- ``actionable_within_session``: whether the current kernel session can actually do it
  with the permissions it already has.

Only ``overlay_candidate`` proposals may later become an overlay, and they still have to
pass the existing scope/probe/trust/never-auto gates.  ``review_candidate`` is deliberately
less restrictive: the host cannot prove an in-session use, but also found no repository or
permission request, so a later kernel LLM may judge, rewrite or reject it in TASK.
``developer_queue`` is for actual system/repository requests.  ``acceptance.kind_of``
remains an *evidence* classification and is deliberately not used here.
"""

from __future__ import annotations

import re

# Changing a system object is not the same as using it.  "Run tcenv profile" is a peer
# action; "Add a tcenv.cli subcommand" is a repository change.
_REPO_VERB = re.compile(
    r"\b(?:add|adding|create|creating|implement|implementing|modify|modifying|change|changing|"
    r"update|updating|extend|extending|introduce|introducing|remove|removing|delete|deleting|"
    r"rename|renaming|refactor|refactoring|rewrite|rewriting|patch|patching|support|"
    r"print|document|fix|provide|expose|report)\b"
    r"|新增|添加|增加|修改|改成|改为|实现|删除|去掉|移除|重命名|重构|扩展|引入|补上|支持|打印|输出|记录|文档|修复|加|加上|提供",
    re.I,
)
_SYSTEM_OBJECT = re.compile(
    r"\b(?:tcenv|taic|cli|subcommand|harness|overlay|trust\.toml|trust|grant|evaluator|"
    r"evaluation|contract|task\.md|feedback\.md|template|config|toml|pyproject|workflow|"
    r"pipeline|database|index|server|api|sdk|docker|sandbox|report\.json|reports?|"
    r"tests?/|tasks?/|docs?/|readme|agents\.md|github|ci|mslreference\.md|msl_reference|"
    r"taic-tools|self-check|self_checks|selfcheck|skills?|documentation|dashboard|"
    r"taskcontract|opspec|distributions|experience\.md|device\.md|harness\.md|repository|repo)\b"
    r"|子命令|工具|报告|字段|看板|系统|仓库|配置|文档|测试|评分|评测|协议|信任|权限|预算|容差",
    re.I,
)
_ALWAYS_DEVELOPER = re.compile(
    r"\b(?:permission|permissions|sudo|credential|credentials|secret|secrets|license|"
    r"allowlist|trust\.toml|rtol|atol|tolerance|budget|protocol)\b"
    r"|凭据|密钥|授权|权限|预算|容差|阈值|评分材料|人工决定|人工批准|需要人",
    re.I,
)
_KERNEL_OBJECT = re.compile(
    r"\b(?:kernel|metal|msl|tile|tiles|thread|threads|threadgroup|simd|simdgroup|pragma|"
    r"unroll|guard|staging|accumulator|accumulators|register|registers|buffer|dtype|fp16|"
    r"fp32|gemm|conv2d|softmax|attention|loop|loops|vector|occupancy|launch|grid|"
    r"probe|measure|measurement|timing|interleav\w*|profile|spill|roofline)\b"
    r"|内核|写核|寄存器|线程|线程组|循环|展开|累加器|探针|测量|瓦片|块|驻留|带宽",
    re.I,
)
_SESSION_FILES = re.compile(r"\b(?:NOTES\.md|FEEDBACK\.md)\b", re.I)
_PEER_VERB = re.compile(
    r"\b(?:read|run|use|keep|retain|avoid|prefer|choose|check|inspect|compare|measure|"
    r"profile|remove|drop|lower|raise|write)\b"
    r"|读|查看|跑|运行|使用|保留|避免|优先|选择|检查|对比|测量|不要|先|降|提高|写",
    re.I,
)

_DECLARED_AUDIENCE = {"peer", "developer"}
_DECLARED_NEEDS = {"none", "repo_change", "new_tool", "permission", "decision"}


def route_proposal(proposal: dict) -> dict:
    """@brief 计算一条提案的接收者、仓库改动需求与当前可执行性。

    @param proposal 规范化前后的提案字典，读取 ``action``/``rationale`` 与可选的
        ``audience``/``needs``/``peer_action`` 自报字段。
    @return 路由字典：``recipient``、``requires_repo_change``、
        ``actionable_within_session``、``disposition``（``overlay_candidate``、
        ``review_candidate`` 或 ``developer_queue``）、``reason``、``intent_conflict`` 与 ``declared``。
        本函数只做确定性文本判定，不调用模型，也不授予任何权限。
    """
    action = str(proposal.get("action") or "")
    rationale = str(proposal.get("rationale") or "")
    text = f"{action} {rationale}".strip()
    declared_audience = str(proposal.get("audience") or "").strip()
    declared_needs = str(proposal.get("needs") or "").strip() or "none"
    declared = {
        "audience": declared_audience if declared_audience in _DECLARED_AUDIENCE else "",
        "needs": declared_needs if declared_needs in _DECLARED_NEEDS else "none",
        "peer_action": str(proposal.get("peer_action") or "")[:500],
    }

    # Hard routing comes from the requested action, not from rationale prose.  A rationale
    # saying "no repository or permission change is needed" must not trip the very rule it
    # is trying to explain.
    # NOTES.md and FEEDBACK.md are writable session artifacts, not repository objects.
    # Remove only their names before matching system objects; the action itself remains
    # available for kernel/measurement classification below.
    routing_action = _SESSION_FILES.sub("", action)
    # In "report it in FEEDBACK.md", report is the verb, not a second system object.
    # Other objects (tcenv, trust, evaluator, …) still count normally.
    system_action = re.sub(r"\breports?\b", "", routing_action, flags=re.I) if _SESSION_FILES.search(action) else routing_action
    repo_change = bool(_ALWAYS_DEVELOPER.search(routing_action))
    if _REPO_VERB.search(routing_action) and _SYSTEM_OBJECT.search(system_action):
        repo_change = True
    peer_request = _SESSION_FILES.sub("", declared["peer_action"])
    peer_system = re.sub(r"\breports?\b", "", peer_request, flags=re.I) if _SESSION_FILES.search(declared["peer_action"]) else peer_request
    if peer_request and (_ALWAYS_DEVELOPER.search(peer_request) or (_REPO_VERB.search(peer_request) and _SYSTEM_OBJECT.search(peer_system))):
        repo_change = True
    # A model that itself says a system change is needed must not be re-labelled as a
    # peer tip by a generous host reading.  The reverse (peer declaration on a system
    # request) is an intent conflict and still goes to the developer queue.
    if declared["needs"] != "none" or declared["audience"] == "developer":
        repo_change = True

    kernel_object = bool(_KERNEL_OBJECT.search(text))
    peer_action = bool(_PEER_VERB.search(action))
    if repo_change:
        recipient = "developer"
        actionable = False
        reason = "the request changes the repository, harness, permissions or evaluation material"
    elif declared["audience"] == "peer" and kernel_object:
        recipient = "kernel_agent"
        actionable = True
        reason = "declared peer tip whose action independently names a kernel or measurement object"
    elif kernel_object and peer_action:
        recipient = "kernel_agent"
        actionable = True
        reason = "the action concerns the kernel, a probe or a measurement available in this session"
    elif kernel_object and declared["audience"] == "" and declared["peer_action"] == "":
        # Older proposals predate peer_action.  A kernel/measurement object is enough;
        # the existing evidence gates still decide whether it can activate.
        recipient = "kernel_agent"
        actionable = True
        reason = "kernel or measurement object without a repository-change request"
    else:
        recipient = "unknown"
        actionable = False
        reason = "the host cannot establish an in-session recipient from the action"

    conflict = bool(declared["audience"] == "peer" and recipient != "kernel_agent")
    if recipient == "kernel_agent" and actionable and not repo_change:
        disposition = "overlay_candidate"
    elif repo_change:
        disposition = "developer_queue"
    else:
        # Unclear is not the same as unwanted.  It cannot activate an overlay or enter
        # the experience store, but a later kernel LLM may still find a useful reading.
        disposition = "review_candidate"
    return {
        "recipient": recipient,
        "requires_repo_change": repo_change,
        "actionable_within_session": actionable,
        "disposition": disposition,
        "reason": reason,
        "intent_conflict": conflict,
        "declared": declared,
    }


__all__ = ["route_proposal"]
