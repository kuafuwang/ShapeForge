"""Structural invariants of the tests tree.

The layout is part of the contract, and these are the properties that actually matter:
a test at the root would not use the package layout, a directory without ``__init__.py``
breaks ``tests.support`` imports, and a backend-free test must not reach into a backend.
"""

from __future__ import annotations

import ast
from pathlib import Path

TESTS = Path(__file__).resolve().parent.parent


def _python_dirs() -> list[Path]:
    """@brief 收集 tests 树下所有含 Python 文件的目录（排除 ``__pycache__``）。

    @return 去重并排序后的目录路径列表，每个目录预期都是一个包。
    """
    return sorted({path.parent for path in TESTS.rglob("*.py") if "__pycache__" not in path.parts})


def test_no_test_module_lives_at_the_tests_root():
    """@brief 验证 tests 根目录下不存在任何 ``test_*.py``，保证测试都走包布局。
    """
    assert [path.name for path in TESTS.glob("test_*.py")] == []


def test_every_directory_with_python_is_a_package():
    """@brief 验证每个含 Python 文件的目录都有 ``__init__.py``，否则 ``tests.support`` 导入会断。
    """
    missing = [str(path.relative_to(TESTS)) for path in _python_dirs() if not (path / "__init__.py").is_file()]
    assert missing == []


def test_backend_free_tests_do_not_import_a_backend():
    """@brief 用 AST 扫描 ``tests/shared`` 下所有测试模块，确认它们不导入 ``tcenv.backends``。

    逐个检查 ``ast.Import`` 与 ``ast.ImportFrom`` 的模块名，命中则记录 ``文件:行号`` 并断言为空。
    """
    offenders = []
    for path in sorted((TESTS / "shared").rglob("test_*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            if any(name == "tcenv.backends" or name.startswith("tcenv.backends.") for name in names):
                offenders.append(f"{path.relative_to(TESTS)}:{node.lineno}")
    assert offenders == []


def test_support_holds_no_business_tests():
    """@brief 验证 ``tests/support`` 里不放业务测试，即不存在 ``test_*.py`` 文件。
    """
    assert [path.name for path in (TESTS / "support").rglob("test_*.py")] == []


def test_every_backend_has_a_test_package():
    """@brief 验证每个 backend 都有对应的测试包：目前只有 metal，且带 ``__init__.py``。
    """
    backends = [path.name for path in (TESTS / "backends").iterdir() if path.is_dir() and path.name != "__pycache__"]
    assert backends == ["metal"]
    assert (TESTS / "backends" / "metal" / "__init__.py").is_file()


def test_gpu_items_are_grouped_on_one_xdist_worker():
    """@brief 验证 conftest 钩子给所有 gpu 条目打上同一个 ``xdist_group``，CPU 条目不受影响。

    conftest 的钩子必须把每个 gpu 条目放进同一个 ``xdist_group``；配合 ``--dist loadgroup``，
    这让 GPU 那一半串行，而 CPU 那一半仍旧并行。用例用最小 Item 替身模拟 gpu／cpu 节点，
    直接调用 ``pytest_collection_modifyitems``（config 传 None）后检查标记是否有被添加。
    """
    from tests.conftest import pytest_collection_modifyitems

    class Item:
        def __init__(self, gpu: bool) -> None:
            self.gpu, self.markers = gpu, []

        def get_closest_marker(self, name):
            return object() if name == "gpu" and self.gpu else None

        def add_marker(self, mark) -> None:
            self.markers.append(mark)

    gpu_a, cpu, gpu_b = Item(True), Item(False), Item(True)
    pytest_collection_modifyitems(None, [gpu_a, cpu, gpu_b])

    def grouped(item) -> bool:
        return any(getattr(mark, "name", None) == "xdist_group" for mark in item.markers)

    assert grouped(gpu_a) and grouped(gpu_b) and not grouped(cpu)
