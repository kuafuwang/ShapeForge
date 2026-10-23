"""The shared test suite is a package, so pytest can collect nested directories."""

from pathlib import Path


def test_shared_tests_live_under_a_package():
    import tests.support

    assert Path(__file__).parent.name == "shared"
    assert (Path(__file__).parents[1] / "__init__.py").is_file()
    assert tests.support.__name__ == "tests.support"
