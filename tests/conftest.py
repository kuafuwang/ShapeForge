"""Tests that touch the GPU hold the machine-wide GPU lock (shared), so they never overlap a measurement of a running experiment."""

from __future__ import annotations

import pytest

pytest_plugins = ["tests.support.run_logs", "tests.support.handoff", "tests.support.dashboard"]


def _use_recovered_tilelang_frontend_when_present() -> None:
    """Use the explicitly recovered MT6 frontend epoch, if this host has one.

    The original MT0 staging path is intentionally left absent when it no longer matches the
    archived install tree.  Tests that actually lower candidates use the recovered, independently
    verified path instead; archived MT0 evidence is never rewritten to match the recovery.
    """
    import os
    from pathlib import Path

    recovered = Path("/tmp/shapeforge-frontends/tilelang-0.1.15-recovered")
    if recovered.is_dir():
        os.environ.setdefault("SHAPEFORGE_TILELANG_STAGING", str(recovered))


_use_recovered_tilelang_frontend_when_present()


def _route_local_task_pack_to_specimens() -> None:
    """@brief 让 ``local_provenance`` 为归档 specimen 解析出迁移前的 task-pack 摘要。

    ``tcenv.evaluation.local_task_pack_digest`` 按名装配任务来取 pack 登记；v1 活题为零以后，
    已迁题名在生产装配里一律被拒绝。测试侧在解析失败时回退到归档的 v1 登记（同一份 pack
    digest），这样评测者世代与迁移前一致，而 ``tcenv/evaluation.py`` 仍保持未改动——它在
    shape-portfolio 的冻结证据代码闭包里（``code_import_closure``），改一个字节都会让
    ``task_pack_digest`` 与历史验收证据对不上。

    @return None。
    """
    import tcenv.evaluation as evaluation

    original = evaluation.local_task_pack_digest

    def patched(request, **kwargs):
        """@brief 先走生产解析，失败时回退到归档 v1 pack 登记。

        @param request 评测请求。
        @param kwargs 透传参数（例如 ``pack_digest``）。
        @return 十六进制摘要。
        @raises evaluation.EvaluationError 两边都解析不出时。
        """
        try:
            return original(request, **kwargs)
        except evaluation.EvaluationError:
            from tests.support.v1_tasks import v1_pack_digest

            digest = v1_pack_digest(request.task, request.dtype, request.size)
            if not digest:
                raise
            return digest

    evaluation.local_task_pack_digest = patched


_route_local_task_pack_to_specimens()


def _specimen_v1_binding_route() -> None:
    """@brief 给 v1 机制测试提供装配入口：``TAIC_BIND_TASK_ROUTE=specimen``。

    生产 ``tasks.get_task`` 保持原样（迁移测试要验证它拒绝已迁任务）；``bind_task`` 只在
    显式设置该环境变量时才走归档 specimen，所以不会误改生产分流。消费方需要 v1 Task 的
    装配点由下面的 autouse fixture 按用例注入，用完即恢复。

    @return None。
    """
    import os

    os.environ["TAIC_BIND_TASK_ROUTE"] = "specimen"


_specimen_v1_binding_route()


@pytest.fixture(autouse=True)
def _v1_specimen_assembly(monkeypatch):
    """@brief 按用例把消费方的 v1 装配器指向归档 specimen，并在用例结束后还原。

    @param monkeypatch pytest 的属性替换夹具，负责还原。
    @yield 无返回值。
    """
    from tests.support.v1_tasks import v1_task

    for module_name in (
        "tcenv.evaluate", "tcenv.evaluation", "tcenv.triton_runner", "tcenv.xcworker",
        "tcenv.backends.metal.xctrace", "tcenv.backends.metal.triton_baseline",
        "tcenv.remote.task_pack", "taic.hwfeatures", "taic.kernel_catalog",
    ):
        module = __import__(module_name, fromlist=["*"])
        monkeypatch.setattr(module, "get_task", v1_task, raising=False)
    import tasks

    # 留一份生产装配的引用，供"拒绝已迁任务"的断言使用（迁移测试与指纹测试）。
    tasks.__dict__["_production_get_task"] = tasks.__dict__["get_task"]
    monkeypatch.setattr(tasks, "get_task", v1_task)
    yield


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
