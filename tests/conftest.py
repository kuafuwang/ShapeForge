"""Tests that touch the GPU hold the machine-wide GPU lock (shared), so they never overlap a measurement of a running experiment."""

from __future__ import annotations

import pytest

pytest_plugins = ["tests.handoff"]


@pytest.fixture(autouse=True)
def _isolate_observe_cache(tmp_path, monkeypatch):
    """Dashboard files stay in the test tmp dir, not the real ``~/.cache``."""
    monkeypatch.setenv("AI_LOWERLING_OBSERVE_DIR", str(tmp_path / "observe-cache"))
    monkeypatch.setenv("OBSERVE_DB", str(tmp_path / "observe.sqlite"))
    monkeypatch.setenv("OBSERVE_CARRYOVER", str(tmp_path / "ledger_carryover.json"))
    monkeypatch.setenv("AI_LOWERLING_BOOKS", str(tmp_path / "books.sqlite"))
    monkeypatch.setenv("OBSERVE_TICK_INTERVAL", "0")


@pytest.fixture(autouse=True)
def _isolate_kernel_catalog(tmp_path, monkeypatch):
    """Confirmed-kernel copies stay in the test tmp dir, not ``kernels/`` in the repo."""
    import taic.kernel_catalog as catalog

    monkeypatch.setattr(catalog, "REPO", tmp_path)


@pytest.fixture(autouse=True)
def _gpu_tests_share_the_gpu_lock(request):
    if request.node.get_closest_marker("gpu") is None:
        yield
        return
    from tcenv.gpulock import gpu_lock

    with gpu_lock(f"pytest {request.node.nodeid}", exclusive=False):
        yield
