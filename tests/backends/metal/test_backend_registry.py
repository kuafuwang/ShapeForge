"""Phase 1: backend discovery is a registry, and parent import stays device-free."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from tcenv.backends.registry import load_backend
from tcenv.target import TargetError, load_target


def test_metal_is_loaded_from_backend_toml():
    """@brief 验证 metal backend 由 ``backend.toml`` 注册表加载，且 target 指向它。

    校验 id／version／语言为 msl／扩展名 ``.metal``，支持 launch spec 与 validation，
    并确认 ``apple-m4-max`` 这个 target 的 backend 是 metal。
    """
    backend = load_backend("metal")
    assert backend.id == "metal"
    assert backend.version == "1"
    assert backend.language == "msl" and backend.file_ext == ".metal"
    assert backend.capabilities.supports_launch_spec
    assert backend.capabilities.supports_validation
    assert load_target("apple-m4-max").backend == "metal"


def test_an_unknown_backend_is_refused():
    """@brief 验证加载不存在的 backend 会以提及 backend 的 ``TargetError`` 拒绝。

    @raises TargetError 加载未注册的 backend id 时抛出。
    """
    with pytest.raises(TargetError, match="backend"):
        load_backend("does-not-exist")


def test_parent_import_of_the_metal_adapter_does_not_import_metal():
    """@brief 在子进程里加载 metal adapter 并调用两个无设备方法，确认父进程未导入 ``Metal`` 框架。

    这保证 backend 发现与父进程导入仍是设备无关的。
    """
    code = (
        "from tcenv.backends.registry import load_backend; "
        "b = load_backend('metal'); "
        "b.validation_env({}); b.is_instrumented_size_only([]); "
        "assert 'Metal' not in __import__('sys').modules, 'Metal imported in parent'"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).resolve().parents[3],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr


def test_the_profile_command_uses_the_backend_module():
    """@brief 确认 profile 命令使用 backend 的 xctrace 模块，不引用已退役的 Triton 路径。"""
    import inspect

    import tcenv.cli as c
    from tcenv.backends.metal import xctrace

    src = inspect.getsource(c)
    assert "tcenv.xctrace" not in src and "tcenv.triton_baseline" not in src
    assert callable(xctrace.profile_kernel)
