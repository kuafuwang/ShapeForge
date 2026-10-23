"""Load a backend from ``backends/<id>/backend.toml``.

Only repository modules under ``tcenv.backends.`` are accepted.  Importing a
backend is allowed in the parent process; the module itself must be lazy about
the device runtime.
"""

from __future__ import annotations

import importlib
import re
import tomllib
from functools import lru_cache
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
BACKENDS = REPO / "backends"
_NAME = re.compile(r"^[A-Za-z0-9_-]+$")
_PREFIX = "tcenv.backends."


def _error(message: str) -> Exception:
    """@brief 构造一个后端加载失败异常，延迟导入以避免循环依赖。

    @param message 面向调用方的错误说明。
    @return 一个 ``TargetError`` 实例。
    """
    from tcenv.target import TargetError

    return TargetError(message)


@lru_cache(maxsize=16)
def load_backend(name: str):
    """@brief 按名称加载并校验一个后端。

    Load and validate one backend.  Raises ``TargetError`` for unknown names.

    @param name 后端 id，对应 ``backends/<id>/backend.toml``。
    @return 后端实例，其 id 与版本和 toml 一致。
    @raises TargetError 名称非法、配置缺失或损坏、入口不在 ``tcenv.backends.`` 下时抛出。
    """
    name = str(name or "").strip()
    if not name or not _NAME.fullmatch(name):
        raise _error(f"invalid backend name {name!r}")
    path = BACKENDS / name / "backend.toml"
    if not path.is_file():
        raise _error(f"backend {name} has no backend.toml")
    try:
        doc = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise _error(f"backend {name} cannot be read: {exc}") from exc
    if int(doc.get("schema_version") or 0) != 1:
        raise _error(f"backend {name} has unsupported schema_version {doc.get('schema_version')!r}")
    if str(doc.get("id") or "") != name:
        raise _error(f"backend.toml id {doc.get('id')!r} is not directory {name}")
    entrypoint = str(doc.get("entrypoint") or "")
    if not entrypoint.startswith(_PREFIX) or entrypoint.count(":") != 1:
        raise _error(f"backend {name} entrypoint must be {_PREFIX}* module:Class")
    module_name, class_name = entrypoint.split(":", 1)
    if ".." in module_name or not module_name.startswith(_PREFIX):
        raise _error(f"backend {name} entrypoint is outside {_PREFIX}")
    try:
        module = importlib.import_module(module_name)
        cls = getattr(module, class_name)
        backend = cls()
    except (ImportError, AttributeError, TypeError) as exc:
        raise _error(f"backend {name} cannot load {entrypoint}: {exc}") from exc
    version = str(doc.get("version") or "")
    if not version:
        raise _error(f"backend {name} has no version")
    if str(getattr(backend, "id", "")) != name:
        raise _error(f"backend class id {getattr(backend, 'id', '')!r} is not {name}")
    if str(getattr(backend, "version", "")) != version:
        raise _error(f"backend {name} code version {getattr(backend, 'version', '')!r} is not toml {version!r}")
    return backend
