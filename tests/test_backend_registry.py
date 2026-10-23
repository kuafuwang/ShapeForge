"""Phase 1: backend discovery is a registry, and parent import stays device-free."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from tcenv.backends.registry import load_backend
from tcenv.target import TargetError, load_target


def test_metal_is_loaded_from_backend_toml():
    backend = load_backend("metal")
    assert backend.id == "metal"
    assert backend.version == "1"
    assert backend.language == "msl" and backend.file_ext == ".metal"
    assert backend.capabilities.supports_launch_spec
    assert backend.capabilities.supports_validation
    assert load_target("apple-m4-max").backend == "metal"


def test_an_unknown_backend_is_refused():
    with pytest.raises(TargetError, match="backend"):
        load_backend("does-not-exist")


def test_parent_import_of_the_metal_adapter_does_not_import_metal():
    code = (
        "from tcenv.backends.registry import load_backend; "
        "load_backend('metal'); "
        "assert 'Metal' not in __import__('sys').modules, 'Metal imported in parent'"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
