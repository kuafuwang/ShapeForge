"""Optional v2 performance baselines (``baselines.json`` + ``baselines.py``).

A baseline is a vendor/library implementation of the same computation, used only as the
timing comparison object.  It is not a correctness oracle: the reference oracle is
``reference.py``, and no baseline result can decide pass/fail.

No ``baselines.json`` means ``Task.baselines`` is empty and the task derives
``effective_correctness_only``: correctness is published, speedup is not.  A
``baselines.json`` without its runner (or a runner for an unregistered name) is an
assembly error, not a silently empty baseline set.
"""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from ..identity import IdentityError, load_strict_json
from ..opspec.validator import SchemaFailure, validate
from .errors import AssemblyError, TaskContractError

BASELINES_FILE = "baselines.json"
BASELINES_RUNNER_FILE = "baselines.py"
BASELINES_SCHEMA_VERSION = 1
BASELINES_SCHEMA_PATH = Path(__file__).resolve().parent / "schema" / "baselines.v1.schema.json"

#: A baseline runner may not import the oracle/renderer/known-good: a comparison object must
#: not be able to become the answer.
_FORBIDDEN_RUNNER_SEGMENTS = frozenset({"known_good", "reference", "renderer"})


@dataclass(frozen=True)
class BaselineRegistration:
    """@brief 一个已登记的 baseline 身份。"""

    name: str
    framework: str
    implementation_id: str
    implementation_version: int
    params: Mapping[str, Any]

    def document(self) -> dict[str, Any]:
        """@brief 身份文档（进入任务身份）。

        @return 字段字典。
        """
        return {
            "name": self.name,
            "framework": self.framework,
            "implementation_id": self.implementation_id,
            "implementation_version": int(self.implementation_version),
            "params": dict(self.params),
        }


@dataclass(frozen=True)
class BaselinePackage:
    """@brief 一个任务的 baseline 登记与 runner 闭包。"""

    registrations: tuple[BaselineRegistration, ...]
    runners: Mapping[str, Callable[..., Any]]
    source_sha256: str = ""

    @property
    def correctness_only(self) -> bool:
        """@brief 没有 baseline 时必须走 ``effective_correctness_only``。

        @return 登记为空时为 True。
        """
        return not self.registrations

    def document(self) -> dict[str, Any]:
        """@brief baseline 包身份文档。

        @return 字段字典。
        """
        return {
            "correctness_only": self.correctness_only,
            "baselines": [entry.document() for entry in self.registrations],
            "source_sha256": self.source_sha256,
        }


def _empty() -> BaselinePackage:
    """@brief 空 baseline 包。

    @return 空包。
    """
    return BaselinePackage(registrations=(), runners={}, source_sha256="")


def _scan_runner_source(path: Path) -> None:
    """@brief 拒绝 runner 直接 import oracle/renderer/known-good。

    @param path runner 路径。
    @raises AssemblyError 越界 import。
    """
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except SyntaxError as exc:
        raise AssemblyError(f"{path}: baseline runner does not parse: {exc}") from exc
    for node in ast.walk(tree):
        names: list[str] = []
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            names = [node.module]
        for module in names:
            bad = sorted(set(module.split(".")) & _FORBIDDEN_RUNNER_SEGMENTS)
            if bad:
                raise AssemblyError(
                    f"{path}: a baseline runner may not import {module!r} ({bad}); the comparison object must "
                    "not be able to become the correctness oracle"
                )


def _load_runner(path: Path, digest: str) -> Any:
    """@brief 加载 ``baselines.py`` 并返回模块。

    @param path runner 路径。
    @param digest 源摘要。
    @return 模块。
    @raises AssemblyError 加载失败。
    """
    _scan_runner_source(path)
    module_name = f"_taic_baselines_{digest[:16]}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:  # pragma: no cover
        raise AssemblyError(f"{path}: cannot be loaded as a module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        sys.modules.pop(module_name, None)
        raise AssemblyError(f"{path}: baseline module failed to import: {exc}") from exc
    return module


def load_baselines(task_dir: str | Path, *, op_name: str) -> BaselinePackage:
    """@brief 读取可选 baseline 登记与 runner。

    @param task_dir 任务目录。
    @param op_name 算子名（诊断）。
    @return baseline 包；没有 ``baselines.json`` 时为空包（→ correctness_only）。
    @raises AssemblyError 登记/runner 不匹配或文档非法。
    """
    root = Path(task_dir)
    manifest_path = root / BASELINES_FILE
    runner_path = root / BASELINES_RUNNER_FILE
    if not manifest_path.is_file():
        if runner_path.exists():
            raise AssemblyError(
                f"{runner_path}: a baseline runner without {BASELINES_FILE} is a second, unregistered source; "
                "either register the baselines or delete the runner"
            )
        return _empty()
    if not runner_path.is_file():
        raise AssemblyError(f"{manifest_path}: {BASELINES_RUNNER_FILE} is required when baselines are registered")
    try:
        document = load_strict_json(manifest_path)
        schema = load_strict_json(BASELINES_SCHEMA_PATH)
    except IdentityError as exc:
        raise TaskContractError(f"{manifest_path}: {exc}") from exc
    try:
        validate(document, schema)
    except SchemaFailure as exc:
        raise TaskContractError(f"{manifest_path}: {exc}") from exc

    registrations: list[BaselineRegistration] = []
    seen: set[str] = set()
    for entry in document["baselines"]:
        name = str(entry["name"])
        if name in seen:
            raise AssemblyError(f"{manifest_path}: duplicate baseline name {name!r}")
        seen.add(name)
        registrations.append(
            BaselineRegistration(
                name=name,
                framework=str(entry["framework"]),
                implementation_id=str(entry["implementation_id"]),
                implementation_version=int(entry["implementation_version"]),
                params=dict(entry["params"]),
            )
        )

    digest = hashlib.sha256(runner_path.read_bytes()).hexdigest()
    module = _load_runner(runner_path, digest)
    table = getattr(module, "BASELINES", None)
    if not isinstance(table, Mapping):
        raise AssemblyError(f"{runner_path}: must define BASELINES as a name -> callable mapping")
    registered = {entry.name for entry in registrations}
    missing = sorted(registered - set(table))
    extra = sorted(set(table) - registered)
    if missing or extra:
        raise AssemblyError(
            f"{runner_path}: BASELINES does not match {BASELINES_FILE}; missing={missing} extra={extra}"
        )
    runners: dict[str, Callable[..., Any]] = {}
    for name, function in table.items():
        if not callable(function):
            raise AssemblyError(f"{runner_path}: BASELINES[{name!r}] is not callable")
        runners[str(name)] = function
    return BaselinePackage(registrations=tuple(registrations), runners=runners, source_sha256=digest)


__all__ = [
    "BASELINES_FILE",
    "BASELINES_RUNNER_FILE",
    "BASELINES_SCHEMA_PATH",
    "BASELINES_SCHEMA_VERSION",
    "BaselinePackage",
    "BaselineRegistration",
    "load_baselines",
]
