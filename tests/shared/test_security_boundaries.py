"""Security boundaries for the model-facing flows and the P2 code path."""

from __future__ import annotations

import pytest

from taic import code_agent


def test_model_flows_do_not_get_network_or_home_write():
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
