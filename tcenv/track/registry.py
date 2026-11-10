"""``tracks.json``: the formal TrackRegistry (delivery mechanisms, not algorithms).

A track is a *delivery mechanism*.  ``msl-single`` is one MSL source file plus a declarative
launch; ``msl-bundle`` is a candidate bundle with candidate-owned Host Tiling and device
kernel(s); ``tilelang-pipeline`` is a Python DSL source built by a pinned frontend.  The
registry therefore contains no ``op_name``, no shape, no tile, no grid, no threads and no
per-task lowering profile — the closed schema refuses them, and the semantic checks below
refuse op-specific evidence text.  Track selection is by formal
``(track_id, track_version)``; an unknown id/version or an unverified track is refused, never
silently replaced.

MT1b names the epoch this module reads.  ``schema_version: 2`` is the live registry:

* tracks are keyed by ``(track_id, track_version)`` so versions coexist.  A new run selects the
  *highest available* version of an id (``available()``/``get()``); a frozen reconstruction,
  resume or replay selects the *recorded* version explicitly and a missing version is a typed
  refusal, never a silent upgrade.
* a DSL track declares ``delivery: dsl_source_pipeline``, its candidate source (path, media
  type, role) and its frozen frontend identity, instead of disguising a Python DSL as
  ``kernel.metal``.  Such a track is ``interface_only`` until real Metal evidence exists: the
  document may not claim ``metal_dispatch`` while ``interface_only``, and
  :meth:`TrackSpec.require_available` refuses to select one that does not.

``schema_version: 1`` documents are **replay-only**.  A frozen run/v2-worker payload keeps the
registry document it was created against; :func:`registry_from_document` reconstructs that
exact document (same bytes, therefore the same registry digest), while
:func:`load_track_registry` — the live/new-run path — refuses a schema-1 file.  This is a
deliberate two-reader split for frozen identity, not a silent dual read.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

from ..identity import IdentityError, load_strict_json
from ..opspec.validator import SchemaFailure, validate
from ..contract_errors import TaskContractError, UnsupportedContractCapability

TRACKS_SCHEMA_VERSION = 2
TRACKS_SCHEMA_PATH = Path(__file__).resolve().parent / "schema" / "tracks.v2.schema.json"
#: The frozen MT1a/A4 epoch.  Only :func:`registry_from_document` reads it, and only to
#: reconstruct a document a frozen run already carries.
LEGACY_TRACKS_SCHEMA_VERSION = 1
LEGACY_TRACKS_SCHEMA_PATH = Path(__file__).resolve().parent / "schema" / "tracks.v1.schema.json"
#: Where the shipped Metal registry lives (backend metadata belongs under ``backends/<id>/``).
METAL_TRACKS_PATH = Path(__file__).resolve().parent.parent.parent / "backends" / "metal" / "tracks.json"

STATUSES = frozenset({"interface_only", "available", "unavailable"})
DELIVERIES = frozenset({"single_source", "candidate_bundle", "dsl_source_pipeline"})
#: The two schema-1 deliveries; a legacy document may not use anything newer.
LEGACY_DELIVERIES = frozenset({"single_source", "candidate_bundle"})
CANDIDATE_ROLES = frozenset({"device_kernel", "host_tiling", "schedule", "launch", "dsl_source"})
MSL_CANDIDATE_ROLES = frozenset({"device_kernel", "host_tiling", "launch"})
CAPABILITIES = frozenset(
    {
        "single_file_msl",
        "candidate_host_tiling",
        "candidate_device_kernel",
        "workspace",
        "multi_stage",
        "dsl_python_source",
        "source_only_build",
        "kernel_pipeline_geometry",
        "device_independent_build_identity",
        "metal_dispatch",
    }
)
#: Capabilities that only describe MSL candidate delivery; a DSL Track may not borrow them.
MSL_CAPABILITIES = frozenset(
    {"single_file_msl", "candidate_host_tiling", "candidate_device_kernel", "workspace", "multi_stage"}
)
#: Capabilities whose only honest evidence is a real device dispatch.
DEVICE_CAPABILITIES = frozenset({"metal_dispatch"})
#: A DSL Track must declare a kernel-pipeline execution plan: this epoch does not freeze a graph.
REQUIRED_DSL_CAPABILITIES = frozenset({"kernel_pipeline_geometry"})
#: Media types that are MSL source/library bytes.  A DSL candidate source must never use one:
#: a Python (or other) DSL is not ``kernel.metal``.
MSL_MEDIA_TYPES = frozenset({"text/x-metal-source", "text/x-msl", "application/x-metallib", "application/x-metal-air"})
MSL_SOURCE_SUFFIXES = (".metal", ".metallib", ".air")


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
class CandidateSourceSpec:
    """@brief DSL Track 声明的候选源（路径/媒体类型/角色）。

    This is what replaces "disguise the DSL as ``kernel.metal``": the Track *declares* the exact
    source contract, :meth:`tcenv.track.plane.CandidateArtifact.validate_for` enforces it, and the
    backend adapter cross-audits it against its frozen contract module.
    """

    path: str
    media_type: str
    role: str = "dsl_source"

    def __post_init__(self) -> None:
        """@brief 校验相对路径/媒体类型/角色。

        @raises TaskContractError 非法。
        """
        _relative_candidate_path(self.path, "candidate_source.path")
        if not self.media_type or "/" not in self.media_type:
            raise TaskContractError("candidate_source.media_type must look like type/subtype")
        if self.media_type.casefold() in MSL_MEDIA_TYPES:
            raise TaskContractError(
                f"candidate_source.media_type {self.media_type!r} is MSL; a DSL Track must not disguise its source "
                "as kernel.metal"
            )
        if self.path.casefold().endswith(MSL_SOURCE_SUFFIXES):
            raise TaskContractError(
                f"candidate_source.path {self.path!r} is an MSL source path; a DSL Track's candidate source is not a "
                "device kernel file"
            )
        if self.role != "dsl_source":
            raise TaskContractError(f"candidate_source.role must be 'dsl_source', got {self.role!r}")

    def document(self) -> dict[str, Any]:
        """@brief 该声明的 JSON 文档。

        @return 字段字典。
        """
        return {"path": self.path, "media_type": self.media_type, "role": self.role}


@dataclass(frozen=True)
class FrontendSpec:
    """@brief DSL Track 的前端/编译器身份（版本 + 入口 + 冻结 options 摘要）。

    ``options_sha256`` is the digest of the device-independent build options that the backend
    contract owns; the registry pins the identity, not a second copy of the options.  A frontend
    version bump therefore moves build identity even when the candidate source is unchanged.
    """

    frontend_id: str
    frontend_version: str
    entry: str
    options_sha256: str

    def __post_init__(self) -> None:
        """@brief 校验身份字段。

        @raises TaskContractError 非法。
        """
        for name in ("frontend_id", "frontend_version", "entry"):
            if not str(getattr(self, name)).strip():
                raise TaskContractError(f"frontend.{name} must be non-empty")
        _digest_text(self.options_sha256, "frontend.options_sha256")

    def document(self) -> dict[str, Any]:
        """@brief 该身份的 JSON 文档。

        @return 字段字典。
        """
        return {
            "frontend_id": self.frontend_id,
            "frontend_version": self.frontend_version,
            "entry": self.entry,
            "options_sha256": self.options_sha256,
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
    #: Present exactly for ``dsl_source_pipeline`` tracks (schema/语义保证).
    candidate_source: CandidateSourceSpec | None = None
    #: Present exactly for ``dsl_source_pipeline`` tracks (schema/语义保证).
    frontend: FrontendSpec | None = None

    @property
    def task_agnostic(self) -> bool:
        """@brief Track 是否只描述交付机制（schema 保证无 op/shape/tile 字段）。

        @return 恒为 True。
        """
        return True

    @property
    def is_dsl(self) -> bool:
        """@brief 是否为 DSL 源 Track。

        @return ``delivery == "dsl_source_pipeline"``。
        """
        return self.delivery == "dsl_source_pipeline"

    @property
    def key(self) -> tuple[str, int]:
        """@brief 版本化选择键。

        @return ``(track_id, track_version)``。
        """
        return (self.track_id, int(self.track_version))

    def supports(self, capability: str) -> bool:
        """@brief 是否声明了某项能力。

        @param capability 能力名。
        @return 已声明时为 True。
        """
        return capability in self.capabilities

    @property
    def selectable(self) -> bool:
        """@brief 新 run 是否可以选中该版本。

        A DSL Track that never dispatched a real device kernel cannot be selected even if a
        document says ``available``: source-only build evidence is not execution evidence.
        """
        if self.status != "available":
            return False
        if self.is_dsl and not self.supports("metal_dispatch"):
            return False
        return True

    def require_available(self) -> None:
        """@brief 要求该 track 有真实证据且可用。

        @raises UnsupportedContractCapability 仍是接口骨架、不可用，或 DSL Track 只有 source-only 证据。
        """
        if self.status != "available":
            raise UnsupportedContractCapability(
                f"track {self.track_id}/{self.track_version} is {self.status!r}; it has no real evidence and "
                "cannot be selected or published"
            )
        if self.is_dsl and not self.supports("metal_dispatch"):
            raise UnsupportedContractCapability(
                f"track {self.track_id}/{self.track_version} declares no metal_dispatch capability; a source-only "
                "DSL build is not real Metal evidence and the Track is not selectable"
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
        if self.candidate_source is not None:
            document["candidate_source"] = self.candidate_source.document()
        if self.frontend is not None:
            document["frontend"] = self.frontend.document()
        if self.evidence:
            document["evidence"] = list(self.evidence)
        return document


def _digest_text(value: Any, where: str) -> str:
    """@brief 断言 64 位小写十六进制摘要。

    @param value 候选值。
    @param where 诊断位置。
    @return 原值。
    @raises TaskContractError 不是摘要。
    """
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise TaskContractError(f"{where} must be a 64-character lowercase hex digest")
    return value


def _relative_candidate_path(value: str, where: str) -> str:
    """@brief 校验候选源声明的是规范相对 POSIX 路径。

    @param value 路径。
    @param where 诊断位置。
    @return 原路径。
    @raises TaskContractError 非法。
    """
    if not isinstance(value, str) or not value or value.startswith("/") or "\\" in value:
        raise TaskContractError(f"{where}: {value!r} must be a relative POSIX path")
    path = PurePosixPath(value)
    if any(part in (".", "..") for part in path.parts) or str(path) != value:
        raise TaskContractError(f"{where}: {value!r} must be a normalized relative path")
    return value


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


def _check_status_and_evidence(status: str, evidence: tuple[str, ...], where: str) -> None:
    """@brief 状态与证据必须一致。

    @param status 登记状态。
    @param evidence 证据路径。
    @param where 诊断位置。
    @raises TaskContractError 声明的证据与状态不符。
    """
    if status == "available" and not evidence:
        raise TaskContractError(
            f"{where}: an available track must name the real evidence that verifies it; source-only or "
            "simulated work cannot make a track available"
        )
    if status == "interface_only" and evidence:
        raise TaskContractError(f"{where}: an interface_only track must not claim evidence")


def _check_dsl_track(
    *,
    document: Mapping[str, Any],
    delivery: str,
    roles: tuple[str, ...],
    capabilities: tuple[str, ...],
    status: str,
    where: str,
) -> tuple[CandidateSourceSpec | None, FrontendSpec | None]:
    """@brief DSL Track 的版本化闭集语义检查。

    A DSL Track declares a source contract and a frontend identity, never MSL roles/capabilities,
    and never a graph/RPC hole.  ``metal_dispatch`` is the only capability that means "real device
    evidence"; the document may not claim it while ``interface_only``.

    @param document 原始登记文档。
    @param delivery 交付机制。
    @param roles 角色。
    @param capabilities 能力。
    @param status 状态。
    @param where 诊断位置。
    @return ``(candidate_source, frontend)``；非 DSL Track 返回 ``(None, None)``。
    @raises TaskContractError 非法。
    """
    if delivery != "dsl_source_pipeline":
        for name in ("candidate_source", "frontend"):
            if name in document:
                raise TaskContractError(
                    f"{where}: {name} is only meaningful for a dsl_source_pipeline track; an MSL delivery mechanism "
                    "does not declare a DSL frontend"
                )
        if "dsl_source" in roles:
            raise TaskContractError(f"{where}: the dsl_source role belongs to a dsl_source_pipeline track")
        borrowed = sorted(set(capabilities) & (CAPABILITIES - MSL_CAPABILITIES))
        if borrowed:
            raise TaskContractError(
                f"{where}: {borrowed} are DSL capabilities; an MSL delivery mechanism may not claim them"
            )
        return None, None

    if "dsl_source" not in roles:
        raise TaskContractError(f"{where}: a dsl_source_pipeline track needs the dsl_source role, got {list(roles)}")
    borrowed = sorted(set(roles) & MSL_CANDIDATE_ROLES)
    if borrowed:
        raise TaskContractError(
            f"{where}: roles {borrowed} are MSL candidate roles; a DSL Track owns its device kernels and never "
            "submits MSL sources or an MSL Host Tiling plan"
        )
    extra = sorted(set(capabilities) & MSL_CAPABILITIES)
    if extra:
        raise TaskContractError(
            f"{where}: capabilities {extra} describe MSL candidate delivery; a DSL Track keeps its own source "
            "contract instead of borrowing one"
        )
    missing = sorted(REQUIRED_DSL_CAPABILITIES - set(capabilities))
    if missing:
        raise TaskContractError(
            f"{where}: a DSL Track must declare {missing}; this epoch freezes kernel_pipeline execution only"
        )
    source_block = document.get("candidate_source")
    if not isinstance(source_block, Mapping):
        raise TaskContractError(f"{where}: a dsl_source_pipeline track must declare candidate_source")
    unknown = sorted(set(source_block) - {"path", "media_type", "role"})
    if unknown:
        raise TaskContractError(f"{where}.candidate_source: unregistered fields {unknown}")
    source = CandidateSourceSpec(
        path=str(source_block.get("path", "")),
        media_type=str(source_block.get("media_type", "")),
        role=str(source_block.get("role", "")),
    )
    frontend_block = document.get("frontend")
    if not isinstance(frontend_block, Mapping):
        raise TaskContractError(f"{where}: a dsl_source_pipeline track must declare its frontend identity")
    unknown = sorted(set(frontend_block) - {"frontend_id", "frontend_version", "entry", "options_sha256"})
    if unknown:
        raise TaskContractError(f"{where}.frontend: unregistered fields {unknown}")
    frontend = FrontendSpec(
        frontend_id=str(frontend_block.get("frontend_id", "")),
        frontend_version=str(frontend_block.get("frontend_version", "")),
        entry=str(frontend_block.get("entry", "")),
        options_sha256=str(frontend_block.get("options_sha256", "")),
    )
    if status == "interface_only" and "metal_dispatch" in capabilities:
        raise TaskContractError(
            f"{where}: an interface_only DSL track must not claim metal_dispatch; source-only build evidence is not "
            "real device evidence"
        )
    return source, frontend


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
    elif delivery == "dsl_source_pipeline":
        pass  # checked below with the DSL-specific closed set
    else:  # pragma: no cover - schema enum
        raise TaskContractError(f"{where}: unknown delivery {delivery!r}")

    status = str(document["status"])
    evidence = tuple(str(item) for item in document.get("evidence", ()))
    _check_status_and_evidence(status, evidence, where)
    candidate_source, frontend = _check_dsl_track(
        document=document,
        delivery=delivery,
        roles=roles,
        capabilities=capabilities,
        status=status,
        where=where,
    )

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
        candidate_source=candidate_source,
        frontend=frontend,
    )


class TrackRegistry:
    """@brief 只读的正式 TrackRegistry（``(track_id, track_version)`` 多版本共存）。"""

    def __init__(
        self,
        *,
        backend_id: str,
        tracks: tuple[TrackSpec, ...],
        schema_version: int = TRACKS_SCHEMA_VERSION,
    ) -> None:
        """@brief 由已校验的 track 列表构造注册表。

        @param backend_id 后端 id。
        @param tracks 已校验的 track（注册顺序）。
        @param schema_version 该文档来自的 registry schema 版本（1 = replay-only）。
        """
        self._backend_id = backend_id
        self._schema_version = int(schema_version)
        ordered: list[tuple[str, int]] = []
        versioned: dict[tuple[str, int], TrackSpec] = {}
        for track in tracks:
            if track.key in versioned:
                raise TaskContractError(
                    f"tracks.json: duplicate track {track.track_id}/{track.track_version}; a version is registered "
                    "once and a new contract is a new version, not a second row"
                )
            versioned[track.key] = track
            ordered.append(track.key)
        self._versioned = versioned
        self._order = tuple(ordered)
        self._ids: tuple[str, ...] = tuple(dict.fromkeys(track_id for track_id, _ in self._order))

    @property
    def backend_id(self) -> str:
        """@brief registry 的后端 id。

        @return 后端 id。
        """
        return self._backend_id

    @property
    def schema_version(self) -> int:
        """@brief 该 registry 文档的 schema 版本。

        @return 1 或 2。
        """
        return self._schema_version

    @property
    def tracks(self) -> tuple[TrackSpec, ...]:
        """@brief 全部 ``(id, version)`` 登记（注册顺序）。

        @return track 元组。
        """
        return tuple(self._versioned[key] for key in self._order)

    @property
    def track_ids(self) -> tuple[str, ...]:
        """@brief 去重后的 track id（首次登记顺序）。

        @return id 元组。
        """
        return self._ids

    def versions(self, track_id: str) -> tuple[int, ...]:
        """@brief 某个 id 已登记的版本（升序）。

        @param track_id 正式 track id。
        @return 版本元组。
        @raises UnsupportedContractCapability 未知 id。
        """
        found = sorted(version for name, version in self._order if name == str(track_id))
        if not found:
            raise UnsupportedContractCapability(
                f"unknown track {track_id!r}; registered tracks are {sorted(self._ids)}"
            )
        return tuple(found)

    def _highest(self, track_id: str, *, available_only: bool) -> TrackSpec | None:
        """@brief 取某个 id 的最高（可选可用）版本。

        @param track_id 正式 track id。
        @param available_only 只考虑新 run 可选择的版本。
        @return track 或 None。
        """
        candidates = [
            self._versioned[(name, version)]
            for name, version in self._order
            if name == str(track_id) and (not available_only or self._versioned[(name, version)].selectable)
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda track: track.track_version)

    def get(self, track_id: str, *, version: int | None = None) -> TrackSpec:
        """@brief 按正式 id（和可选版本）取 track。

        Version resolution is frozen here and nowhere else:

        * with ``version``: that exact registered version, or a typed refusal;
        * without ``version``: the highest **selectable** version, falling back to the highest
          registered version only so the caller receives the precise status/capability error
          instead of "unknown track".  A frozen reconstruction always passes ``version``.

        @param track_id 正式 track id。
        @param version 要求的版本；``None`` 表示按上面的规则解析。
        @return track。
        @raises UnsupportedContractCapability 未知 id 或版本不符。
        """
        if version is not None:
            if isinstance(version, bool) or not isinstance(version, int):
                raise UnsupportedContractCapability(f"track {track_id!r}: version must be an integer, got {version!r}")
            track = self._versioned.get((str(track_id), int(version)))
            if track is None:
                registered = sorted(version for name, version in self._order if name == str(track_id))
                if not registered:
                    raise UnsupportedContractCapability(
                        f"unknown track {track_id!r}; registered tracks are {sorted(self._ids)}"
                    )
                raise UnsupportedContractCapability(
                    f"track {track_id!r} is not registered at version {version}; registered versions are "
                    f"{registered}.  A recorded/frozen version never silently moves to another version"
                )
            return track
        chosen = self._highest(track_id, available_only=True) or self._highest(track_id, available_only=False)
        if chosen is None:
            raise UnsupportedContractCapability(
                f"unknown track {track_id!r}; registered tracks are {sorted(self._ids)}"
            )
        return chosen

    def available(self) -> tuple[TrackSpec, ...]:
        """@brief 有真实证据、可被新 run 选择的最高版本。

        @return track 元组（每个 id 至多一条，首次登记顺序）。
        """
        found = []
        for track_id in self._ids:
            chosen = self._highest(track_id, available_only=True)
            if chosen is not None:
                found.append(chosen)
        return tuple(found)

    def require_available(self, track_id: str, *, version: int | None = None) -> TrackSpec:
        """@brief 取 track 并要求其可用。

        @param track_id 正式 track id。
        @param version 要求的版本。
        @return track。
        @raises UnsupportedContractCapability 未知、版本不符或未验证。
        """
        track = self.get(track_id, version=version)
        track.require_available()
        return track

    def document(self) -> dict[str, Any]:
        """@brief registry 的登记文档（保留来源 schema 版本，冻结文档因此可重算同一摘要）。

        @return 字段字典。
        """
        return {
            "schema_version": self._schema_version,
            "backend_id": self._backend_id,
            "tracks": [track.document() for track in self.tracks],
        }


def _legacy_schema() -> Mapping[str, Any]:
    """@brief 读取冻结的 schema-1 文档校验器。

    @return schema。
    @raises TaskContractError 不可读。
    """
    try:
        return load_strict_json(LEGACY_TRACKS_SCHEMA_PATH)
    except IdentityError as exc:  # pragma: no cover - schema is shipped
        raise TaskContractError(f"frozen tracks v1 schema is unreadable: {exc}") from exc


def _live_schema() -> Mapping[str, Any]:
    """@brief 读取 schema-2 文档校验器。

    @return schema。
    @raises TaskContractError 不可读。
    """
    try:
        return load_strict_json(TRACKS_SCHEMA_PATH)
    except IdentityError as exc:  # pragma: no cover - schema is shipped
        raise TaskContractError(f"registered tracks schema is unreadable: {exc}") from exc


def _legacy_track_from_document(document: Mapping[str, Any], *, where: str) -> TrackSpec:
    """@brief 由冻结的 schema-1 文档构造 track（只用于重建旧身份）。

    @param document track 文档。
    @param where 诊断位置。
    @return track。
    @raises TaskContractError 文档不是 schema-1 的 MSL 交付机制。
    """
    for name in ("candidate_source", "frontend"):
        if name in document:
            raise TaskContractError(
                f"{where}: {name} cannot appear in a schema-1 registry document; the frozen epoch had no DSL Track"
            )
    delivery = str(document["delivery"])
    if delivery not in LEGACY_DELIVERIES:
        raise TaskContractError(f"{where}: schema-1 delivery {delivery!r} is not one of {sorted(LEGACY_DELIVERIES)}")
    roles = tuple(str(role) for role in document["candidate_roles"])
    if "dsl_source" in roles:
        raise TaskContractError(f"{where}: the dsl_source role does not exist in a schema-1 registry document")
    return _track_from_document(document, backend_id="", where=where)


def registry_from_document(document: Mapping[str, Any]) -> TrackRegistry:
    """@brief 校验一份 ``tracks.json`` 文档并构造注册表（v2 live，v1 replay-only）。

    @param document JSON 文档。
    @return 注册表。
    @raises TaskContractError 结构/语义非法。
    """
    if not isinstance(document, Mapping):
        raise TaskContractError("a track registry document must be an object")
    declared = document.get("schema_version")
    if declared == LEGACY_TRACKS_SCHEMA_VERSION:
        schema = _legacy_schema()
    elif declared == TRACKS_SCHEMA_VERSION:
        schema = _live_schema()
    else:
        raise TaskContractError(
            f"registered tracks schema_version must be {TRACKS_SCHEMA_VERSION} "
            f"(or {LEGACY_TRACKS_SCHEMA_VERSION} for a frozen reconstruction), got {declared!r}"
        )
    try:
        validate(document, schema)
    except SchemaFailure as exc:
        raise TaskContractError(f"tracks.json: {exc}") from exc
    backend_id = str(document["backend_id"])
    tracks: list[TrackSpec] = []
    seen: set[tuple[str, int]] = set()
    for index, item in enumerate(document["tracks"]):
        where = f"tracks.json.tracks[{index}]"
        if declared == LEGACY_TRACKS_SCHEMA_VERSION:
            track = _legacy_track_from_document(item, where=where)
        else:
            track = _track_from_document(item, backend_id=backend_id, where=where)
        if track.key in seen:
            raise TaskContractError(
                f"tracks.json: duplicate track {track.track_id}/{track.track_version}; versions coexist, a repeated "
                "version does not"
            )
        seen.add(track.key)
        tracks.append(track)
    return TrackRegistry(
        backend_id=backend_id,
        tracks=tuple(tracks),
        schema_version=int(declared),
    )


def load_track_registry(path: str | Path = METAL_TRACKS_PATH) -> TrackRegistry:
    """@brief 读取并校验一份 live（schema 2）``tracks.json``。

    @param path registry 路径。
    @return 注册表。
    @raises TaskContractError 文件/结构/语义非法，或文档是 replay-only 的 schema 1。
    """
    resolved = Path(path)
    try:
        document = load_strict_json(resolved)
    except IdentityError as exc:
        raise TaskContractError(f"{resolved}: {exc}") from exc
    if isinstance(document, Mapping) and document.get("schema_version") != TRACKS_SCHEMA_VERSION:
        raise TaskContractError(
            f"{resolved}: a live registry must be schema_version {TRACKS_SCHEMA_VERSION}; "
            f"{document.get('schema_version')!r} documents are frozen replay inputs only and are never selected "
            "from by a new run"
        )
    try:
        return registry_from_document(document)
    except TaskContractError as exc:
        raise TaskContractError(f"{resolved}: {exc}") from exc


def dsl_candidate_sources(registry: TrackRegistry) -> tuple[CandidateSourceSpec, ...]:
    """@brief 已登记 DSL Track 声明的候选源契约。

    The portfolio/search plane uses this to recognise DSL content *without* naming a DSL: a
    candidate that matches a registered DSL source declaration is refused by the MSL projection
    instead of being relabelled ``msl-single``/``msl-bundle``.

    @param registry 注册表。
    @return 候选源声明元组。
    """
    return tuple(
        track.candidate_source for track in registry.tracks if track.candidate_source is not None
    )


def is_dsl_track_id(track_id: str, registry: TrackRegistry | None = None) -> bool:
    """@brief 某个 id 是否登记为 DSL Track。

    @param track_id track id。
    @param registry 可选注册表；缺省读 live registry。
    @return 已登记为 DSL Track 时为 True。
    """
    chosen = registry if registry is not None else load_track_registry()
    try:
        track = chosen.get(str(track_id))
    except UnsupportedContractCapability:
        return False
    return track.is_dsl


def require_msl_track(
    track_id: str, *, version: int | None = None, registry: TrackRegistry | None = None
) -> TrackSpec:
    """@brief 解析一个 **MSL** Track（MSL 专有投影/预检/确认入口的统一闸门）。

    The portfolio projection, the Metal preflight and the Metal confirmation rebuild are MSL
    mechanisms.  They must never be entered with a registered DSL Track: the failure has to be a
    typed capability refusal at the Track boundary, not an MSL parse/compile error later, and it
    must never depend on "has a geometry" to decide the family.

    @param track_id 正式 track id。
    @param version 冻结的选择版本；None 表示按 registry 默认（新 run 的最高可选版本）。
    @param registry 可选注册表；缺省读 live registry。
    @return 已登记的 MSL track。
    @raises UnsupportedContractCapability 未知 track 或已登记的 DSL Track。
    """
    chosen = registry if registry is not None else load_track_registry()
    track = chosen.get(str(track_id), version=version)
    if track.is_dsl:
        raise UnsupportedContractCapability(
            f"track {track.track_id}/{track.track_version} is a {track.delivery} Track (status {track.status}); the "
            "MSL portfolio projection/preflight/confirmation plane does not handle it. Its own Track "
            "validator/build adapter and an independent expected identity own that path, and an interface_only "
            "Track is not selectable by a new run"
        )
    return track


__all__ = [
    "CANDIDATE_ROLES",
    "CAPABILITIES",
    "DELIVERIES",
    "DEVICE_CAPABILITIES",
    "LEGACY_DELIVERIES",
    "LEGACY_TRACKS_SCHEMA_PATH",
    "LEGACY_TRACKS_SCHEMA_VERSION",
    "METAL_TRACKS_PATH",
    "MSL_CANDIDATE_ROLES",
    "MSL_CAPABILITIES",
    "MSL_MEDIA_TYPES",
    "MSL_SOURCE_SUFFIXES",
    "REQUIRED_DSL_CAPABILITIES",
    "STATUSES",
    "TRACKS_SCHEMA_PATH",
    "TRACKS_SCHEMA_VERSION",
    "CandidateSourceSpec",
    "FrontendSpec",
    "TrackImplementation",
    "TrackRegistry",
    "TrackSpec",
    "dsl_candidate_sources",
    "is_dsl_track_id",
    "load_track_registry",
    "registry_from_document",
    "require_msl_track",
]
