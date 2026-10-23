"""Phase 1 adapter delegates to the existing Metal implementation."""

from __future__ import annotations

from tasks import get_task

from tcenv.backends.registry import load_backend


def test_parent_side_screen_and_launch_parse_delegate():
    backend = load_backend("metal")
    task = get_task("relu", "fp16", "paper")
    assert backend.screen("kernel void k() {}", task) == []
    assert backend.screen("kernel void k() { for (;;) {} }", task)
    assert backend.parse_launch("kernel void k() {}", task.contract) is None
