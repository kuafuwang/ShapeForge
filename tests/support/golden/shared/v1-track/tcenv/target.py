"""Backend-neutral device profiles. No import of ``taic``."""

from __future__ import annotations

import hashlib
import json
import os
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

REPO = Path(__file__).resolve().parent.parent
TARGETS = REPO / "targets"
BACKENDS = REPO / "backends"
DEFAULT_TARGET_ID = "apple-m4-max"


class TargetError(RuntimeError):
    """@brief 请求的目标档案不可用或无法加载。

    The requested target cannot be used.
    """


@dataclass(frozen=True)
class Limits:
    """Metal compatibility view of four target limits.

    Generic target loading and evaluation use ``TargetProfile.raw_limits`` and
    backend hooks instead.  A future non-MSL target need not define these keys.
    """

    max_buffer_index: int
    max_threads_per_threadgroup: int
    max_threadgroup_memory_bytes: int
    counter_sample_us: int


@dataclass(frozen=True)
class TargetProfile:
    """@brief 一个目标设备档案：后端身份、匹配规则、限制与配套文档路径。"""

    id: str
    backend: str
    status: str
    display_name: str
    # Metal compatibility fields; generic matching is Backend.probe/device_matches.
    match_name_contains: str
    match_family: str
    calibration_device_name: str
    raw_limits: Mapping[str, Any]
    root: Path
    language: str
    file_ext: str
    backend_version: str = ""

    @property
    def limits(self) -> Limits:
        """Return the legacy Metal limits view, validating only when a Metal caller asks.

        B0.5 keeps this property for apple-m4-max and Metal profiling modules.
        Generic code must not access it: missing/non-integer keys are a
        backend-specific target error, not a target-file schema requirement.
        """
        try:
            return Limits(
                max_buffer_index=int(self.raw_limits["max_buffer_index"]),
                max_threads_per_threadgroup=int(self.raw_limits["max_threads_per_threadgroup"]),
                max_threadgroup_memory_bytes=int(self.raw_limits["max_threadgroup_memory_bytes"]),
                counter_sample_us=int(self.raw_limits["counter_sample_us"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise TargetError(f"target {self.id} has invalid Metal limits: {exc}") from exc

    @property
    def device_doc(self) -> Path:
        """@brief 目标目录下的设备文档路径。

        @return ``device.md`` 的路径。
        """
        return self.root / "device.md"

    @property
    def experience(self) -> Path:
        """@brief 目标目录下的经验文档路径。

        @return ``experience.md`` 的路径。
        """
        return self.root / "experience.md"

    @property
    def skills(self) -> Path:
        """@brief 该后端技能目录的路径。

        @return ``backends/<backend>/skills`` 的路径。
        """
        return BACKENDS / self.backend / "skills"

    @property
    def wiki(self) -> Path:
        """@brief 该后端 wiki 目录的路径。

        @return ``backends/<backend>/wiki`` 的路径。
        """
        return BACKENDS / self.backend / "wiki"


def load_target(target_id: str | None = None) -> TargetProfile:
    """@brief 加载一个目标档案。

    Load one profile. ``reserved`` profiles load, but :func:`require_active` refuses to run them.

    @param target_id 目标 id；为 None 时用默认目标。
    @return 解析后的 ``TargetProfile``。
    @raises TargetError 档案缺失、id 不匹配或缺少配套文档时抛出。
    """
    name = target_id or DEFAULT_TARGET_ID
    path = TARGETS / name / "target.toml"
    if not path.is_file():
        raise TargetError(f"target {name} has no target.toml")
    doc = tomllib.loads(path.read_text(encoding="utf-8"))
    if str(doc.get("id") or "") != name:
        raise TargetError(f"target.toml id {doc.get('id')!r} is not the directory {name}")
    backend = str(doc.get("backend") or "")
    from .backends.registry import load_backend

    backend_obj = load_backend(backend)
    backend_doc = {
        "language": backend_obj.language,
        "file_ext": backend_obj.file_ext,
    }
    device = doc.get("device") or {}
    limits = doc.get("limits") or {}
    root = path.parent
    for asset in ("device.md", "experience.md"):
        if not (root / asset).is_file():
            raise TargetError(f"target {name} is missing {asset}")
    return TargetProfile(
        id=str(doc["id"]),
        backend=backend,
        status=str(doc.get("status") or "reserved"),
        display_name=str(doc.get("display_name") or ""),
        match_name_contains=str(device.get("match_name_contains") or ""),
        match_family=str(device.get("match_family") or ""),
        calibration_device_name=str(device.get("calibration_device_name") or ""),
        raw_limits=dict(limits),
        root=root,
        language=str(backend_doc.get("language") or ""),
        file_ext=str(backend_doc.get("file_ext") or ""),
        backend_version=str(backend_obj.version or ""),
    )


def require_active(target_id: str | None = None) -> TargetProfile:
    """@brief 加载目标并确认它处于 ``active`` 状态。

    @param target_id 目标 id；为 None 时用默认目标。
    @return 可运行的目标档案。
    @raises TargetError 目标不可加载或其状态不是 active 时抛出。
    """
    profile = load_target(target_id)
    if profile.status != "active":
        raise TargetError(f"target {profile.id} is {profile.status}; it cannot run")
    return profile


def device_matches(profile: TargetProfile, info: dict) -> bool:
    """Compatibility entry point; matching semantics belong to the selected backend."""
    from .backends.registry import load_backend

    return load_backend(profile.backend).device_matches(profile, info)


def limit_notes(profile: TargetProfile, info: dict) -> list[str]:
    """Compatibility entry point; limit comparisons belong to the selected backend."""
    from .backends.registry import load_backend

    return load_backend(profile.backend).limit_notes(profile, info)


def calibration_cache_path(target_id: str | None = None) -> Path:
    """@brief 新标定结果的写入路径（在源码树之外）。

    Where a new measurement is written. This is outside the source tree.

    @param target_id 目标 id；为 None 时用默认目标。
    @return 用户缓存目录下的标定文件路径。
    """
    profile = load_target(target_id)
    return Path.home() / ".cache" / "ai_lowerling" / "calibration" / f"{profile.id}.json"


def calibration_file(target_id: str | None = None) -> tuple[Path, str]:
    """@brief 选定标定文件及其来源。

    ``(path, source)``. ``TCENV_CALIBRATION`` wins, then the cache, then a legacy file already in the target directory.

    A measurement is not written back into the target directory. ``get_calibration`` writes the cache path.

    @param target_id 目标 id；为 None 时用默认目标。
    @return ``(路径, 来源)``，来源为 ``"env"``、``"cache"`` 或 ``"target"``。
    """
    override = os.environ.get("TCENV_CALIBRATION")
    if override:
        return Path(override), "env"
    cache = calibration_cache_path(target_id)
    if cache.is_file():
        return cache, "cache"
    legacy = load_target(target_id).root / "calibration.json"
    if legacy.is_file():
        return legacy, "target"
    return cache, "cache"


def read_calibration(target_id: str | None = None) -> tuple[dict, Path, str]:
    """@brief 读入标定数据。

    @param target_id 目标 id；为 None 时用默认目标。
    @return ``(标定字典, 文件路径, 来源)``。
    @raises TargetError 文件内容不是 JSON 对象时抛出。
    """
    path, source = calibration_file(target_id)
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise TargetError(f"calibration {path} is not an object")
    return data, path, source


def sha256_file(path: Path) -> str:
    """@brief 计算文件的 sha256 摘要。

    @param path 文件路径。
    @return 十六进制摘要字符串。
    """
    return hashlib.sha256(path.read_bytes()).hexdigest()
