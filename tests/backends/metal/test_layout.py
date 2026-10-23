"""The Metal suite has its own package; backend-specific tests go here."""

from pathlib import Path


def test_metal_tests_live_under_backends_metal():
    assert Path(__file__).parent.name == "metal"
    assert (Path(__file__).parents[2] / "conftest.py").is_file()
