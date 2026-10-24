"""Tests that touch the GPU hold the machine-wide GPU lock (shared), so they never overlap a measurement of a running experiment."""

from __future__ import annotations

import pytest

pytest_plugins = ["tests.support.run_logs", "tests.support.handoff", "tests.support.dashboard"]


@pytest.fixture(autouse=True)
def _isolate_observe_cache(tmp_path, monkeypatch):
    """@brief 把 dashboard 相关环境变量指向测试临时目录，避免写入真实的 ``~/.cache``。

    @param tmp_path pytest 提供的临时目录，用作缓存、SQLite 与账本的落盘位置。
    @param monkeypatch pytest 的环境变量替换夹具，负责在用例结束后恢复原值。
    """
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
    """@brief 把 kernel 目录的仓库根重定向到临时目录，避免已确认 kernel 的副本写进仓库的 ``kernels/``。

    @param tmp_path pytest 提供的临时目录，用作 ``taic.kernel_catalog.REPO`` 的替代值。
    @param monkeypatch pytest 的属性替换夹具，负责在用例结束后还原 ``REPO``。
    """
    import taic.kernel_catalog as catalog

    monkeypatch.setattr(catalog, "REPO", tmp_path)


@pytest.fixture(autouse=True)
def _gpu_tests_share_the_gpu_lock(request):
    """@brief 让带 ``gpu`` 标记的用例在整个用例期间持有共享的机器级 GPU 锁。

    说明：本文件顶部的模块说明给出的理由是——共享锁保证 GPU 用例不会与正在跑的实验的测量
    重叠；非 gpu 用例不取锁，直接放行。

    @param request pytest 的用例请求对象，用于判断最近的作用域标记与生成锁的描述信息。
    """
    if request.node.get_closest_marker("gpu") is None:
        yield
        return
    from tcenv.gpulock import gpu_lock

    with gpu_lock(f"pytest {request.node.nodeid}", exclusive=False):
        yield


def pytest_collection_modifyitems(config, items):
    """@brief 给所有 ``gpu`` 用例打上同一个 xdist 分组标记，配合 ``--dist loadgroup`` 让它们落在同一个 worker 上。

    gpu 用例先共享持有机器锁，测量时再升级为独占；升级会先释放共享锁，因此并行不会死锁，
    但不分组时会彼此排队，最长排到锁超时。分组后 GPU 部分串行，CPU 用例仍分散到各 worker。

    @param config pytest 配置对象（本函数未使用其内容）。
    @param items 收集到的测试条目列表，就地给 gpu 条目追加分组标记。
    """
    try:
        import xdist  # noqa: F401
    except ImportError:  # pytest-xdist is a dependency, but collection must not fail without it
        return
    group = pytest.mark.xdist_group("gpu")
    for item in items:
        if item.get_closest_marker("gpu") is not None:
            item.add_marker(group)
