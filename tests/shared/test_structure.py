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
    return sorted({path.parent for path in TESTS.rglob("*.py") if "__pycache__" not in path.parts})


def test_no_test_module_lives_at_the_tests_root():
    assert [path.name for path in TESTS.glob("test_*.py")] == []


def test_every_directory_with_python_is_a_package():
    missing = [str(path.relative_to(TESTS)) for path in _python_dirs() if not (path / "__init__.py").is_file()]
    assert missing == []


def test_backend_free_tests_do_not_import_a_backend():
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
    assert [path.name for path in (TESTS / "support").rglob("test_*.py")] == []


def test_every_backend_has_a_test_package():
    backends = [path.name for path in (TESTS / "backends").iterdir() if path.is_dir() and path.name != "__pycache__"]
    assert backends == ["metal"]
    assert (TESTS / "backends" / "metal" / "__init__.py").is_file()


def test_gpu_items_are_grouped_on_one_xdist_worker():
    """The conftest hook must put every gpu item in the same ``xdist_group``; with
    ``--dist loadgroup`` that serialises the GPU half without serialising the CPU half."""
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
