"""Security boundaries for the model-facing flows and the P2 code path."""

from __future__ import annotations

import pytest

from taic import code_agent


def test_model_flows_do_not_get_network_or_home_write():
    """@brief 验证写核与审查 flow 只获得本地权限，不能联网或写 home。

    hmz 不可用时跳过；可用时逐一检查 Assistant/Coder/Reviewer 的 Permission 边界。
    """
    pytest.importorskip("hmz")
    from hmz.flows import PermissionKind

    from taic.flows.code_patch import Coder
    from taic.flows.feedback_review import Reviewer
    from taic.flows.kernel_once import Assistant

    for role in (Assistant, Coder, Reviewer):
        permission = role._permission
        assert permission.local == PermissionKind.ALL
        assert permission.user == PermissionKind.READ
        assert permission.system == PermissionKind.READ
        assert permission.online == PermissionKind.NONE


def test_code_agent_cannot_patch_security_critical_paths():
    """@brief 验证 code agent 不能修改测试、评测、提示、信任等安全关键路径。

    同时确认普通实现文件仍可被补丁命中，避免误把整个仓库都封死。
    """
    for path in (
        "tests/shared/test_code_agent.py",
        "tcenv/evaluate.py",
        "tcenv/backends/metal/runtime.py",
        "taic/prompts.py",
        "taic/kernel_agent.py",
        "taic/harness.py",
        "taic/trust.py",
        "experiments/common.py",
        "targets/apple-m4-max/trust.toml",
        ".pi/skills/harness-feedback/SKILL.md",
    ):
        assert code_agent.forbidden_path(path), path
    assert not code_agent.forbidden_path("taic/feedback_extra.py")
