"""Tests that touch the GPU hold the machine-wide GPU lock (shared), so they never overlap a measurement of a running experiment."""

from __future__ import annotations

import pytest

pytest_plugins = ["tests.handoff"]


@pytest.fixture(autouse=True)
def _isolate_observe_cache(tmp_path, monkeypatch):
    """Dashboard files stay in the test tmp dir, not the real ``~/.cache``."""
    monkeypatch.setenv("AI_LOWERLING_OBSERVE_DIR", str(tmp_path / "observe-cache"))


@pytest.fixture(autouse=True)
def _gpu_tests_share_the_gpu_lock(request):
    if request.node.get_closest_marker("gpu") is None:
        yield
        return
    from tcenv.gpulock import gpu_lock

    with gpu_lock(f"pytest {request.node.nodeid}", exclusive=False):
        yield
