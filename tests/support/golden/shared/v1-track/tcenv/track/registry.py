"""``tracks.json`` v1: the formal TrackRegistry (delivery mechanisms, not algorithms).

A track is a *delivery mechanism*.  ``msl-single`` is one MSL source file plus a declarative
launch; ``msl-bundle`` is a candidate bundle with candidate-owned Host Tiling and device
kernel(s).  The registry therefore contains no ``op_name``, no shape, no tile, no grid, no
threads and no per-task lowering profile — the closed schema refuses them, and the semantic
checks below refuse op-specific evidence text.  Track selection is by formal
``track_id``/``track_version``; an unknown or unverified track is refused, never silently
replaced.

A1 ships both Metal tracks as ``interface_only``.  That is deliberate: the build adapter,
ABI derivation and real Mac evidence are A2/A4.  ``require_available`` fails until a track
carries real evidence, so no simulated or source-only track can be presented as available.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from ..identity import IdentityError, load_strict_json
from ..opspec.validator import SchemaFailure, validate
from ..contract_errors import TaskContractError, UnsupportedContractCapability

TRACKS_SCHEMA_VERSION = 1
TRACKS_SCHEMA_PATH = Path(__file__).resolve().parent / "schema" / "tracks.v1.schema.json"
#: Where the shipped Metal registry lives (backend metadata belongs under ``backends/<id>/``).
METAL_TRACKS_PATH = Path(__file__).resolve().parent.parent.parent / "backends" / "metal" / "tracks.json"

STATUSES = frozenset({"interface_only", "available", "unavailable"})
DELIVERIES = frozenset({"single_source", "candidate_bundle"})
CANDIDATE_ROLES = frozenset({"device_kernel", "host_tiling", "schedule", "launch"})
CAPABILITIES = frozenset(
    {"single_file_msl", "candidate_host_tiling", "candidate_device_kernel", "workspace", "multi_stage"}
)


@dataclass(frozen=True)
class TrackImplementation:
    """@brief 一个已登记的 track 实现身份。"""

    implementation_id: str
    implementation_version: int

    def document(self) -> dict[str, Any]:
        """@brief 身份文档。

        @return 字段字典。
        """
        return {
            "implementation_id": self.implementation_id,
            "implementation_version": int(self.implementation_version),
        }


@dataclass(frozen=True)
class TrackSpec:
    """@brief 一条通过 schema 与语义校验的 track 登记。"""

    track_id: str
    track_version: int
    delivery: str
    status: str
    candidate_roles: tuple[str, ...]
    capabilities: tuple[str, ...]
    renderer: TrackImplementation
    validator: TrackImplementation
    build_adapter: TrackImplementation
    evidence: tuple[str, ...] = ()

    @property
    def task_agnostic(self) -> bool:
        """@brief Track 是否只描述交付机制（schema 保证无 op/shape/tile 字段）。

        @return 恒为 True。
        """
        return True

    def supports(self, capability: str) -> bool:
        """@brief 是否声明了某项能力。

        @param capability 能力名。
        @return 已声明时为 True。
        """
        return capability in self.capabilities

    def require_available(self) -> None:
        """@brief 要求该 track 有真实证据且可用。

        @raises UnsupportedContractCapability 仍是接口骨架或不可用。
        """
        if self.status != "available":
            raise UnsupportedContractCapability(
                f"track {self.track_id}/{self.track_version} is {self.status!r}; it has no real evidence and "
                "cannot be selected or published"
            )

    def document(self) -> dict[str, Any]:
        """@brief 该 track 的登记文档。

        @return 字段字典。
        """
        document: dict[str, Any] = {
            "track_id": self.track_id,
            "track_version": int(self.track_version),
            "delivery": self.delivery,
            "status": self.status,
            "candidate_roles": list(self.candidate_roles),
            "capabilities": list(self.capabilities),
            "renderer": self.renderer.document(),
            "validator": self.validator.document(),
            "build_adapter": self.build_adapter.document(),
        }
        if self.evidence:
            document["evidence"] = list(self.evidence)
        return document


def _implementation(block: Mapping[str, Any], where: str) -> TrackImplementation:
    """@brief 校验一个实现登记块。

    @param block 登记块。
    @param where 诊断位置。
    @return 实现身份。
    @raises TaskContractError id 为空或版本非法。
    """
    implementation_id = str(block.get("implementation_id", ""))
    if not implementation_id.strip():
        raise TaskContractError(f"{where}: implementation_id must be non-empty")
    version = block.get("implementation_version")
    if isinstance(version, bool) or not isinstance(version, int) or version < 1:
        raise TaskContractError(f"{where}: implementation_version must be an integer >= 1")
    return TrackImplementation(implementation_id=implementation_id, implementation_version=version)


def _track_from_document(document: Mapping[str, Any], *, backend_id: str, where: str) -> TrackSpec:
    """@brief 由一个已通过 schema 的 track 文档构造并做语义校验。

    @param document track 文档。
    @param backend_id registry 的后端 id。
    @param where 诊断位置。
    @return track。
    @raises TaskContractError 语义非法。
    """
    roles = tuple(str(role) for role in document["candidate_roles"])
    if len(set(roles)) != len(roles):
        raise TaskContractError(f"{where}.candidate_roles: must not repeat a role")
    unknown_roles = sorted(set(roles) - CANDIDATE_ROLES)
    if unknown_roles:
        raise TaskContractError(f"{where}.candidate_roles: unknown roles {unknown_roles}")
    capabilities = tuple(str(capability) for capability in document["capabilities"])
    if len(set(capabilities)) != len(capabilities):
        raise TaskContractError(f"{where}.capabilities: must not repeat a capability")
    unknown_capabilities = sorted(set(capabilities) - CAPABILITIES)
    if unknown_capabilities:
        raise TaskContractError(f"{where}.capabilities: unknown capabilities {unknown_capabilities}")

    delivery = str(document["delivery"])
    if delivery == "single_source":
        if roles != ("device_kernel",):
            raise TaskContractError(
                f"{where}: a single-source track delivers exactly one device_kernel role, got {list(roles)}"
            )
        if "host_tiling" in capabilities:
            raise TaskContractError(
                f"{where}: Host Tiling belongs to a candidate bundle, never to a single-source track"
            )
    elif delivery == "candidate_bundle":
        if "device_kernel" not in roles or "host_tiling" not in roles:
            raise TaskContractError(
                f"{where}: a candidate bundle needs candidate-owned host_tiling and device_kernel roles, got {list(roles)}"
            )
    else:  # pragma: no cover - schema enum
        raise TaskContractError(f"{where}: unknown delivery {delivery!r}")

    status = str(document["status"])
    evidence = tuple(str(item) for item in document.get("evidence", ()))
    if status == "available" and not evidence:
        raise TaskContractError(
            f"{where}: an available track must name the real evidence that verifies it; source-only or "
            "simulated work cannot make a track available"
        )
    if status == "interface_only" and evidence:
        raise TaskContractError(f"{where}: an interface_only track must not claim evidence")

    return TrackSpec(
        track_id=str(document["track_id"]),
        track_version=int(document["track_version"]),
        delivery=delivery,
        status=status,
        candidate_roles=roles,
        capabilities=capabilities,
        renderer=_implementation(document["renderer"], f"{where}.renderer"),
        validator=_implementation(document["validator"], f"{where}.validator"),
        build_adapter=_implementation(document["build_adapter"], f"{where}.build_adapter"),
        evidence=evidence,
    )


class TrackRegistry:
    """@brief 只读的正式 TrackRegistry。"""

    def __init__(self, *, backend_id: str, tracks: tuple[TrackSpec, ...]) -> None:
        """@brief 由已校验的 track 列表构造注册表。

        @param backend_id 后端 id。
        @param tracks 已校验的 track。
        """
        self._backend_id = backend_id
        self._tracks = {track.track_id: track for track in tracks}

    @property
    def backend_id(self) -> str:
        """@brief registry 的后端 id。

        @return 后端 id。
        """
        return self._backend_id

    @property
    def tracks(self) -> tuple[TrackSpec, ...]:
        """@brief 全部 track（注册顺序）。

        @return track 元组。
        """
        return tuple(self._tracks.values())

    def get(self, track_id: str, *, version: int | None = None) -> TrackSpec:
        """@brief 按正式 id（和可选版本）取 track。

        @param track_id 正式 track id。
        @param version 要求的版本；``None`` 表示接受已登记版本。
        @return track。
        @raises UnsupportedContractCapability 未知 id 或版本不符。
        """
        track = self._tracks.get(str(track_id))
        if track is None:
            raise UnsupportedContractCapability(
                f"unknown track {track_id!r}; registered tracks are {sorted(self._tracks)}"
            )
        if version is not None and int(version) != track.track_version:
            raise UnsupportedContractCapability(
                f"track {track_id!r} is registered at version {track.track_version}, requested {version}"
            )
        return track

    def available(self) -> tuple[TrackSpec, ...]:
        """@brief 有真实证据、可被选择的 track。

        @return track 元组。
        """
        return tuple(track for track in self._tracks.values() if track.status == "available")

    def require_available(self, track_id: str, *, version: int | None = None) -> TrackSpec:
        """@brief 取 track 并要求其可用。

        @param track_id 正式 track id。
        @param version 要求的版本。
        @return track。
        @raises UnsupportedContractCapability 未知或未验证。
        """
        track = self.get(track_id, version=version)
        track.require_available()
        return track

    def document(self) -> dict[str, Any]:
        """@brief registry 的登记文档。

        @return 字段字典。
        """
        return {
            "schema_version": TRACKS_SCHEMA_VERSION,
            "backend_id": self._backend_id,
            "tracks": [track.document() for track in self._tracks.values()],
        }


def registry_from_document(document: Mapping[str, Any]) -> TrackRegistry:
    """@brief 校验一份 ``tracks.json`` 文档并构造注册表。

    @param document JSON 文档。
    @return 注册表。
    @raises TaskContractError 结构/语义非法。
    """
    try:
        schema = load_strict_json(TRACKS_SCHEMA_PATH)
    except IdentityError as exc:  # pragma: no cover - schema is shipped
        raise TaskContractError(f"registered tracks schema is unreadable: {exc}") from exc
    try:
        validate(document, schema)
    except SchemaFailure as exc:
        raise TaskContractError(f"tracks.json: {exc}") from exc
    backend_id = str(document["backend_id"])
    tracks: list[TrackSpec] = []
    seen: set[str] = set()
    for index, item in enumerate(document["tracks"]):
        track = _track_from_document(item, backend_id=backend_id, where=f"tracks.json.tracks[{index}]")
        if track.track_id in seen:
            raise TaskContractError(f"tracks.json: duplicate track_id {track.track_id!r}")
        seen.add(track.track_id)
        tracks.append(track)
    return TrackRegistry(backend_id=backend_id, tracks=tuple(tracks))


def load_track_registry(path: str | Path = METAL_TRACKS_PATH) -> TrackRegistry:
    """@brief 读取并校验一份 ``tracks.json``。

    @param path registry 路径。
    @return 注册表。
    @raises TaskContractError 文件/结构/语义非法。
    """
    resolved = Path(path)
    try:
        document = load_strict_json(resolved)
    except IdentityError as exc:
        raise TaskContractError(f"{resolved}: {exc}") from exc
    try:
        return registry_from_document(document)
    except TaskContractError as exc:
        raise TaskContractError(f"{resolved}: {exc}") from exc


__all__ = [
    "CANDIDATE_ROLES",
    "CAPABILITIES",
    "DELIVERIES",
    "METAL_TRACKS_PATH",
    "STATUSES",
    "TRACKS_SCHEMA_PATH",
    "TRACKS_SCHEMA_VERSION",
    "TrackImplementation",
    "TrackRegistry",
    "TrackSpec",
    "load_track_registry",
    "registry_from_document",
]
