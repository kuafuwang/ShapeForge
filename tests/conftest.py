"""Tests that touch the GPU hold the machine-wide GPU lock (shared), so they never overlap a measurement of a running experiment."""

from __future__ import annotations

import pytest

pytest_plugins = ["tests.support.run_logs", "tests.support.handoff", "tests.support.dashboard"]


@pytest.fixture(autouse=True)
def _isolate_observe_cache(tmp_path, monkeypatch):
    """Dashboard files stay in the test tmp dir, not the real ``~/.cache``."""
    monkeypatch.setenv("AI_LOWERLING_OBSERVE_DIR", str(tmp_path / "observe-cache"))
    monkeypatch.setenv("OBSERVE_DB", str(tmp_path / "observe.sqlite"))
    monkeypatch.setenv("OBSERVE_CARRYOVER", str(tmp_path / "ledger_carryover.json"))
    monkeypatch.setenv("AI_LOWERLING_BOOKS", str(tmp_path / "books.sqlite"))
    monkeypatch.setenv("OBSERVE_TICK_INTERVAL", "0")
    monkeypatch.setenv("OBSERVE_REFRESH_ON_READ", "1")
    monkeypatch.setenv("OBSERVE_FULL_TICK_INTERVAL", "0")
    import observe.query as query

    query._FEATURE_CACHE.clear()
    query._HISTORY["key"] = None
    query._HISTORY["rows"] = []


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


def pytest_collection_modifyitems(config, items):
    """Keep every ``gpu`` item on one xdist worker with ``--dist loadgroup``.

    A gpu test holds the machine lock shared and upgrades it to exclusive for a measurement. The
    upgrade releases shared first, so parallel gpu tests do not deadlock, but without grouping they
    can queue on each other for up to the lock timeout. Grouping them makes the GPU part serial
    while the CPU tests still spread over the workers.
    """
    try:
        import xdist  # noqa: F401
    except ImportError:  # pytest-xdist is a dependency, but collection must not fail without it
        return
    group = pytest.mark.xdist_group("gpu")
    for item in items:
        if item.get_closest_marker("gpu") is not None:
            item.add_marker(group)
