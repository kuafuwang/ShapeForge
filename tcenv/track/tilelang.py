"""The frozen TileLang Track contract (MT1b).

This module is the *candidate-source / frontend / build-identity* contract of the TileLang
``kernel_pipeline`` Track.  It is deliberately hardware-neutral and importable on the control
host: it never imports ``tilelang``, ``torch``, ``hmz`` or any Metal runtime, and it never runs
a build.  The build itself belongs to the bounded v2 worker reusing the existing lower-level
hmz Profile (`tcenv.worker_isolation`), not to a second sandbox in this module.

What is frozen here (task-agnostic; no op name, shape, algorithm or tiling):

* the formal ``track_id``/``track_version`` and the candidate source contract — one
  ``kernel.py``/``text/x-python`` entry with role ``dsl_source``.  A Python DSL is **not**
  ``kernel.metal``; the MSL text the pinned frontend emits is a *build artifact* with its own
  declared media type, never a candidate entry.
* the frontend/compiler identity and the device-independent build options that enter build
  identity (frontend version, entry, options digest).  A frontend bump therefore moves build
  identity even when the candidate source is unchanged.
* the boundary between a ``source_only`` build (what MT0/the pinned probe has actually done:
  MSL source, device IR, kernel params, host source; **zero dispatch**) and a ``metal_build``
  (device library + dispatch; not implemented in this epoch).  Only ``source_only`` is
  implemented; asking for ``metal_build`` is a typed capability refusal, never a fake build.
* the isolation facts a DSL build request must declare (offline, empty host allowlist, declared
  write roots only) and the closed source-only artifact set with its canonical digest rule.
* a CPU-only replay of archived source-only artifacts: it re-derives the artifact set digest,
  the frontend identity and the no-dispatch facts from raw bytes.  It is explicitly *not* a
  Metal correctness, performance or availability claim.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from ..contract_errors import TaskContractError, UnsupportedContractCapability
from ..identity import fingerprint

TILELANG_TRACK_ID = "tilelang-pipeline"
#: MT1b 的 source-only 版本必须永久保留给 frozen replay；MT2 的 Metal 能力是新版本。
TILELANG_TRACK_VERSION = 1
TILELANG_METAL_TRACK_VERSION = 2
SUPPORTED_TRACK_VERSIONS = (TILELANG_TRACK_VERSION, TILELANG_METAL_TRACK_VERSION)
TILELANG_DELIVERY = "dsl_source_pipeline"

TRACK_CONTRACT_SCHEMA_ID = "tcenv.track.tilelang.contract.1"
BUILD_IDENTITY_SCHEMA_ID = "tcenv.track.tilelang.build-identity.2"
SOURCE_ONLY_REPLAY_SCHEMA_ID = "tcenv.track.tilelang.source-only-replay.1"

#: The candidate source contract.  The DSL is Python; it is never an MSL kernel file.
SOURCE_PATH = "kernel.py"
SOURCE_MEDIA_TYPE = "text/x-python"
SOURCE_ROLE = "dsl_source"
#: Candidate-owned execution geometry (kernel_pipeline).  Same closed document the existing
#: kernel-pipeline plane parses; it is *not* an MSL Host Tiling plan and needs no lifecycle.
GEOMETRY_ROLE = "schedule"
GEOMETRY_PATH = "host/execution_geometry.json"
GEOMETRY_MEDIA_TYPE = "application/json"

#: The pinned frontend (MT0 source-only probe, wheel identity re-derived by the MT0 package).
FRONTEND_ID = "tilelang"
FRONTEND_VERSION = "0.1.15"
FRONTEND_ENTRY = "tilelang.engine.lower"
FRONTEND_UPSTREAM_COMMIT = "a35f8ddf45eba16c21211ec8822d56ce5363036f"
FRONTEND_WHEEL_SHA256 = "dcffc9aa8d4cd086dd93090f5afeeaa38aa621d2bb80998c4e5f5ee07be116aa"
#: The installed-package identity MT0 re-derived from the staged wheel (r5).  A well-formed
#: digest of *different* bytes is drift, not a valid observation.
FRONTEND_PACKAGE_MANIFEST_SHA256 = "06f3b0adee7abce77af1111cb36e704b4b23c3cbe38d095dd62cce86509c11cc"
FRONTEND_DIST_INFO_METADATA_SHA256 = "380d3486286aeb708dc4846cfdc4b438985c84199a160b21238cadf3e4e3fa19"
FRONTEND_DIST_INFO_RECORD_SHA256 = "6ba83c2ed267154eedc9dec6d916e315c553a8e515744f0feb7ca05dfa388c34"
FRONTEND_PACKAGE_FILES = 3651
FRONTEND_PACKAGE_BYTES = 134219728
#: The package-manifest formatting rule is part of the identity (MT0 probe rule).
FRONTEND_MANIFEST_RULE = "sha256(join('\n', f'{rel}  {bytes}  {sha256}' for sorted non-pycache regular files))"
#: Environment facts are observed, device-independent and closed.
FRONTEND_ENVIRONMENT_KEYS = frozenset({"python_version", "python_implementation", "platform", "machine"})

SOURCE_ONLY_BUILD_MODE = "source_only"
METAL_BUILD_MODE = "metal_build"
BUILD_MODES = (SOURCE_ONLY_BUILD_MODE, METAL_BUILD_MODE)
#: /1 只实现 source-only；/2 在保留 source-only replay 的同时实现真实 Metal build。
IMPLEMENTED_BUILD_MODES = (SOURCE_ONLY_BUILD_MODE,)
IMPLEMENTED_BUILD_MODES_BY_VERSION = {
    TILELANG_TRACK_VERSION: (SOURCE_ONLY_BUILD_MODE,),
    TILELANG_METAL_TRACK_VERSION: (SOURCE_ONLY_BUILD_MODE, METAL_BUILD_MODE),
}

#: Device-independent build options.  ``enable_device_compile`` is False because no Metal
#: program is created here; a Metal build is a *different* mode, not a silent option flip.
SOURCE_ONLY_OPTIONS: Mapping[str, Any] = {
    "frontend_entry": FRONTEND_ENTRY,
    "target_kind": "metal",
    "target_keys": ["metal", "gpu"],
    "target_host": "c",
    "enable_host_codegen": True,
    "enable_device_compile": False,
    "host_artifact_format": "c-source",
}
METAL_BUILD_OPTIONS: Mapping[str, Any] = {
    **SOURCE_ONLY_OPTIONS,
    "enable_device_compile": True,
}
#: The isolation route already implemented by ``tcenv.worker_isolation`` (MT0-isolation-profile).
#: The TileLang adapter reuses it; it does not render a second Seatbelt profile.
ISOLATION_ROUTE = "hmz.coganchor.fence.proxy.Proxy + hmz.coganchor.darwin.seatbelt.Profile"

#: Closed environment keys for a source-only identity: no device fact may be claimed.
ENVIRONMENT_KEYS = frozenset(
    {"python_version", "python_implementation", "platform", "machine"}
)

#: Closed source-only artifact set, in the pinned order the digest rule uses.
SOURCE_ONLY_ARTIFACTS: tuple[tuple[str, str], ...] = (
    ("frontend_identity.json", "application/json"),
    ("kernel.metal", "text/x-metal-source"),
    ("device_ir.json", "application/json"),
    ("kernel_params.json", "application/json"),
    ("host_module.txt", "text/x-c"),
)
SOURCE_ONLY_ARTIFACT_PATHS = tuple(path for path, _ in SOURCE_ONLY_ARTIFACTS)
SOURCE_ONLY_ARTIFACT_MEDIA_TYPES = dict(SOURCE_ONLY_ARTIFACTS)

#: Isolation keys a DSL build request must declare; unknown keys are refused.
ISOLATION_KEYS = frozenset(
    {"route", "online", "hosts", "external_reads", "external_writes", "declared_write_roots", "internal_fence_or_wrap"}
)
_MISSING = object()


class TileLangContractError(TaskContractError):
    """@brief TileLang Track 合同/交付形状非法。"""

    code = "tilelang_track_contract_invalid"


class TileLangIdentityDrift(TileLangContractError):
    """@brief 前端/编译器/options 身份漂移：同一候选不得沿用旧 build 身份。"""

    code = "track_frontend_identity_drift"


class TileLangFrontendUnavailable(UnsupportedContractCapability):
    """@brief 本机缺已登记前端（能力缺失，不是候选源码错误，也不得无隔离回退）。"""

    code = "backend_capability_unavailable"


class TileLangBuildModeUnavailable(UnsupportedContractCapability):
    """@brief 本 epoch 只实现 source-only；Metal build 属于 MT2，不能伪装成完成。"""

    code = "unsupported_contract_capability"


class TileLangCodegenError(TileLangContractError):
    """@brief 前端 codegen 超时/非零/缺工件：只允许 partial，不升级为 full build。"""

    code = "codegen_failed"


def _require_mapping(value: Any, where: str) -> Mapping[str, Any]:
    """@brief 断言文档是对象。

    @param value 候选值。
    @param where 诊断位置。
    @return 原值。
    @raises TileLangContractError 不是对象。
    """
    if not isinstance(value, Mapping):
        raise TileLangContractError(f"{where}: expected an object")
    return value


def _require_keys(value: Mapping[str, Any], keys: Iterable[str], where: str) -> None:
    """@brief 断言闭集键完全一致。

    @param value 文档。
    @param keys 期望键。
    @param where 诊断位置。
    @raises TileLangContractError 缺键或有多余键。
    """
    expected = set(keys)
    found = set(value)
    missing = sorted(expected - found)
    unknown = sorted(found - expected)
    if missing or unknown:
        raise TileLangContractError(f"{where}: missing {missing}, unregistered {unknown}")


def _sha256_bytes(data: bytes) -> str:
    """@brief 字节摘要。

    @param data 字节。
    @return 十六进制 sha256。
    """
    return hashlib.sha256(bytes(data)).hexdigest()


def _require_digest(value: Any, where: str) -> str:
    """@brief 断言 64 位小写十六进制摘要。

    @param value 候选值。
    @param where 诊断位置。
    @return 原值。
    @raises TileLangContractError 不是摘要。
    """
    if not isinstance(value, str) or len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise TileLangContractError(f"{where} must be a 64-character lowercase hex digest")
    return value


def options_document(mode: str = SOURCE_ONLY_BUILD_MODE) -> dict[str, Any]:
    """@brief 指定构建模式下的冻结 options 文档。

    @param mode ``source_only`` 或 ``metal_build``。
    @return options 文档。
    @raises TileLangBuildModeUnavailable 未知模式。
    """
    if mode == SOURCE_ONLY_BUILD_MODE:
        return dict(SOURCE_ONLY_OPTIONS)
    if mode == METAL_BUILD_MODE:
        return dict(METAL_BUILD_OPTIONS)
    raise TileLangBuildModeUnavailable(f"unknown TileLang build mode {mode!r}; known {list(BUILD_MODES)}")


def options_sha256(mode: str = SOURCE_ONLY_BUILD_MODE) -> str:
    """@brief 冻结 options 的摘要（登记进 registry 的 ``frontend.options_sha256``）。

    @param mode 构建模式。
    @return 十六进制 sha256。
    """
    return fingerprint(options_document(mode=mode))


def source_spec() -> dict[str, str]:
    """@brief 候选源契约（与 registry 的 ``candidate_source`` 逐字段对应）。

    @return ``{path, media_type, role}``。
    """
    return {"path": SOURCE_PATH, "media_type": SOURCE_MEDIA_TYPE, "role": SOURCE_ROLE}


def frontend_identity() -> dict[str, str]:
    """@brief 前端/编译器身份的登记投影。

    @return ``{frontend_id, frontend_version, entry, options_sha256}``。
    """
    return {
        "frontend_id": FRONTEND_ID,
        "frontend_version": FRONTEND_VERSION,
        "entry": FRONTEND_ENTRY,
        "options_sha256": options_sha256(),
    }


def frozen_contract(*, track_version: int = TILELANG_TRACK_VERSION) -> dict[str, Any]:
    """@brief 指定版本的完整冻结合同（/1 默认文档与 MT1b 保持一致）。

    @param track_version 1=source-only，2=MT2 固定 kernel-pipeline Metal 能力。
    @return 字段字典。
    @raises TileLangContractError 未登记版本。
    """
    if track_version not in SUPPORTED_TRACK_VERSIONS:
        raise TileLangContractError(f"unsupported TileLang Track contract version {track_version!r}")
    implemented = IMPLEMENTED_BUILD_MODES_BY_VERSION[track_version]
    return {
        "schema_id": TRACK_CONTRACT_SCHEMA_ID,
        "track_id": TILELANG_TRACK_ID,
        "track_version": int(track_version),
        "delivery": TILELANG_DELIVERY,
        "execution_kind": "kernel_pipeline",
        "candidate_source": source_spec(),
        "geometry": {"role": GEOMETRY_ROLE, "path": GEOMETRY_PATH, "media_type": GEOMETRY_MEDIA_TYPE},
        "frontend": {
            "frontend_id": FRONTEND_ID,
            "frontend_version": FRONTEND_VERSION,
            "entry": FRONTEND_ENTRY,
            "upstream_commit": FRONTEND_UPSTREAM_COMMIT,
            "wheel_sha256": FRONTEND_WHEEL_SHA256,
            "options_sha256": options_sha256(),
        },
        "build": {
            "modes": list(BUILD_MODES),
            "implemented_modes": list(implemented),
            "source_only_artifact_set": [
                {"path": path, "media_type": media_type} for path, media_type in SOURCE_ONLY_ARTIFACTS
            ],
            "source_only_options": options_document(mode=SOURCE_ONLY_BUILD_MODE),
            "metal_build_options": options_document(mode=METAL_BUILD_MODE),
            "artifact_set_digest_rule": (
                "sha256(RFC8785-canonical JSON of [{path,bytes,sha256}] in the declared artifact order)"
            ),
        },
        "isolation": {
            "route": ISOLATION_ROUTE,
            "reuses": "tcenv.worker_isolation (lower-level hmz Profile + empty-host Proxy)",
            "online": False,
            "hosts": [],
        },
        "evidence_boundary": {
            "source_only": "MT0 pinned source-only probe: MSL source + device IR + params + host source, zero dispatch",
            "metal_build": (
                "not implemented in this epoch (MT2); no Metal correctness, dispatch, timing or speedup"
                if track_version == TILELANG_TRACK_VERSION
                else "MT2 fixed candidate-owned kernel-pipeline build/dispatch/correctness/fresh-rotation only"
            ),
            "availability": (
                "interface_only until a real Metal slice is evaluated in MT2"
                if track_version == TILELANG_TRACK_VERSION
                else "available only with the registered MT2 real-Metal evidence; no graph or LLM-generation claim"
            ),
        },
    }


def audit_registration(track: Any) -> dict[str, Any]:
    """@brief 交叉核对 registry 声明与冻结合同（registry 是权威选择单位，合同是身份事实）。

    @param track 已登记的 ``TrackSpec``。
    @return 核对通过的合同文档摘要（供证据记录）。
    @raises TileLangContractError 声明与合同不一致。
    @raises UnsupportedContractCapability 不是 DSL Track 或没有前端身份。
    """
    track_id = str(getattr(track, "track_id", ""))
    if track_id != TILELANG_TRACK_ID:
        raise TileLangContractError(f"track {track_id!r} is not the frozen TileLang Track {TILELANG_TRACK_ID!r}")
    track_version = int(getattr(track, "track_version", 0))
    if track_version not in SUPPORTED_TRACK_VERSIONS:
        raise TileLangContractError(
            f"{TILELANG_TRACK_ID}: registry registers version {getattr(track, 'track_version', None)}, the frozen "
            f"contract versions are {list(SUPPORTED_TRACK_VERSIONS)}"
        )
    if track_version == TILELANG_METAL_TRACK_VERSION and int(
        getattr(getattr(track, "build_adapter", None), "implementation_version", 0)
    ) != TILELANG_METAL_TRACK_VERSION:
        raise TileLangContractError(
            "frozen contract is version 1 for this source-only TrackSpec; tilelang-pipeline/2 requires "
            "build_adapter implementation_version=2"
        )
    delivery = str(getattr(track, "delivery", ""))
    if delivery != TILELANG_DELIVERY:
        raise TileLangContractError(f"{TILELANG_TRACK_ID}: delivery {delivery!r} is not {TILELANG_DELIVERY!r}")
    source = getattr(track, "candidate_source", None)
    if source is None:
        raise UnsupportedContractCapability(f"{TILELANG_TRACK_ID}: the registered Track declares no candidate source")
    if source.document() != source_spec():
        raise TileLangContractError(
            f"{TILELANG_TRACK_ID}: registered candidate_source {source.document()} does not match the frozen source "
            f"contract {source_spec()}"
        )
    frontend = getattr(track, "frontend", None)
    if frontend is None:
        raise UnsupportedContractCapability(f"{TILELANG_TRACK_ID}: the registered Track declares no frontend identity")
    expected = frontend_identity()
    found = frontend.document()
    for name, value in expected.items():
        if found.get(name) != value:
            raise TileLangIdentityDrift(
                f"{TILELANG_TRACK_ID}: registered frontend.{name} is {found.get(name)!r}, the frozen frontend is "
                f"{value!r}; a frontend/options drift is a new Track version, not a silent reuse"
            )
    roles = set(getattr(track, "candidate_roles", ()))
    if SOURCE_ROLE not in roles or roles - {SOURCE_ROLE, GEOMETRY_ROLE}:
        raise TileLangContractError(
            f"{TILELANG_TRACK_ID}: candidate roles {sorted(roles)} are not the frozen {{{SOURCE_ROLE!r}, "
            f"{GEOMETRY_ROLE!r}}} contract"
        )
    if GEOMETRY_ROLE in roles and "kernel_pipeline_geometry" not in set(getattr(track, "capabilities", ())):
        raise TileLangContractError(
            f"{TILELANG_TRACK_ID}: the registered Track accepts a geometry entry without the kernel_pipeline_geometry "
            "capability"
        )
    for role, suffix in (("renderer", "renderer"), ("validator", "validator"), ("build_adapter", "build")):
        spec = getattr(track, role, None)
        identity = (getattr(spec, "implementation_id", ""), int(getattr(spec, "implementation_version", 0)))
        expected_version = track_version if role == "build_adapter" else 1
        if identity != (f"metal.track.{TILELANG_TRACK_ID}.{suffix}", expected_version):
            raise TileLangIdentityDrift(
                f"{TILELANG_TRACK_ID}/{track_version}: {role} identity {identity!r} is not the frozen "
                f"('metal.track.{TILELANG_TRACK_ID}.{suffix}', {expected_version})"
            )
    return {
        "track": f"{TILELANG_TRACK_ID}/{track_version}",
        "contract_digest": fingerprint(frozen_contract(track_version=track_version)),
    }


def validate_candidate(candidate: Any, track: Any) -> str:
    """@brief 严校验一个 TileLang 候选的交付形状（不使用 MSL 合同）。

    @param candidate ``CandidateArtifact``。
    @param track 已登记的 TileLang ``TrackSpec``（先做注册⇄合同核对）。
    @return ``"accepted"``。
    @raises TileLangContractError 路径/媒体类型/角色/条目集非法。
    @raises UnsupportedContractCapability track 不是冻结的 TileLang Track。
    """
    audit_registration(track)
    verdict = candidate.validate_for(track)  # registry-declared roles/paths/media types
    entries = tuple(candidate.entries)
    sources = [entry for entry in entries if entry.role == SOURCE_ROLE]
    if len(sources) != 1:
        raise TileLangContractError(
            f"{TILELANG_TRACK_ID}: a candidate delivers exactly one {SOURCE_ROLE!r} entry, got "
            f"{[(entry.path, entry.role) for entry in entries]}"
        )
    source = sources[0]
    if (source.path, source.media_type) != (SOURCE_PATH, SOURCE_MEDIA_TYPE):
        raise TileLangContractError(
            f"{TILELANG_TRACK_ID}: the DSL source must be {SOURCE_PATH!r}/{SOURCE_MEDIA_TYPE!r}, got "
            f"{source.path!r}/{source.media_type!r}; a Python DSL is not kernel.metal"
        )
    schedules = [entry for entry in entries if entry.role == GEOMETRY_ROLE]
    if len(schedules) > 1:
        raise TileLangContractError(f"{TILELANG_TRACK_ID}: a candidate ships at most one execution geometry")
    for entry in schedules:
        if (entry.path, entry.media_type) != (GEOMETRY_PATH, GEOMETRY_MEDIA_TYPE):
            raise TileLangContractError(
                f"{TILELANG_TRACK_ID}: the execution geometry must be {GEOMETRY_PATH!r}/{GEOMETRY_MEDIA_TYPE!r}, got "
                f"{entry.path!r}/{entry.media_type!r}"
            )
    allowed_paths = {SOURCE_PATH, GEOMETRY_PATH}
    stray = sorted(entry.path for entry in entries if entry.path not in allowed_paths)
    if stray:
        raise TileLangContractError(
            f"{TILELANG_TRACK_ID}: undeclared candidate entries {stray}; the frozen source contract declares only "
            f"{sorted(allowed_paths)}"
        )
    return verdict


def verify_frontend_observation(observed: Mapping[str, Any]) -> dict[str, Any]:
    """@brief 校验一次真实前端观测的完整身份（缺前端/漂移即 fail closed）。

    The observation is what a bounded worker reports after hashing the installed package and
    importing the frontend.  This function is pure: it does not import the frontend.

    @param observed 观测文档。
    @return 规范化后的身份文档。
    @raises TileLangFrontendUnavailable 文档缺失或没有前端。
    @raises TileLangIdentityDrift 版本/入口/包摘要/options 与冻结身份不符。
    """
    if not isinstance(observed, Mapping):
        raise TileLangFrontendUnavailable("no frontend observation was supplied; a source-only build needs the pinned one")
    required = (
        "frontend_id",
        "frontend_version",
        "entry",
        "options_sha256",
        "package_manifest_sha256",
        "dist_info_metadata_sha256",
        "dist_info_record_sha256",
        "package_files",
        "package_bytes",
    )
    missing = [name for name in required if not observed.get(name)]
    if missing:
        raise TileLangFrontendUnavailable(
            f"the frontend observation is incomplete ({missing}); a missing frontend is a typed capability gap, "
            "never a build with a guessed version"
        )
    if str(observed["frontend_id"]) != FRONTEND_ID:
        raise TileLangIdentityDrift(
            f"frontend id {observed['frontend_id']!r} is not the frozen {FRONTEND_ID!r}"
        )
    if str(observed["frontend_version"]) != FRONTEND_VERSION:
        raise TileLangIdentityDrift(
            f"frontend version {observed['frontend_version']!r} is not the frozen {FRONTEND_VERSION!r}; the candidate "
            "identity would silently change meaning"
        )
    if str(observed["entry"]) != FRONTEND_ENTRY:
        raise TileLangIdentityDrift(
            f"frontend entry {observed['entry']!r} is not the frozen {FRONTEND_ENTRY!r}"
        )
    if str(observed["options_sha256"]) != options_sha256():
        raise TileLangIdentityDrift(
            f"frontend options digest {observed['options_sha256']!r} is not the frozen {options_sha256()!r}"
        )
    for name in ("package_manifest_sha256", "dist_info_metadata_sha256", "dist_info_record_sha256"):
        _require_digest(observed[name], f"frontend observation {name}")
    # A digest is only meaningful if it is the *frozen* digest: a well-formed hash of other bytes
    # is exactly the "silent frontend drift" this check exists to refuse.
    for name, expected in (
        ("package_manifest_sha256", FRONTEND_PACKAGE_MANIFEST_SHA256),
        ("dist_info_metadata_sha256", FRONTEND_DIST_INFO_METADATA_SHA256),
        ("dist_info_record_sha256", FRONTEND_DIST_INFO_RECORD_SHA256),
    ):
        if str(observed[name]) != expected:
            raise TileLangIdentityDrift(
                f"frontend observation {name} is {observed[name]!r}, the frozen install identity is {expected!r}; "
                "the installed package is not the pinned one"
            )
    for name, expected, label in (
        ("package_files", FRONTEND_PACKAGE_FILES, "file count"),
        ("package_bytes", FRONTEND_PACKAGE_BYTES, "byte count"),
    ):
        value = observed[name]
        if isinstance(value, bool) or not isinstance(value, int) or value != expected:
            raise TileLangIdentityDrift(
                f"frontend observation {name} is {value!r}, the pinned {label} is {expected}"
            )
    document = {name: observed[name] for name in required}
    for name in ("frontend_id", "frontend_version", "entry", "options_sha256"):
        document[name] = str(document[name])
    return document


def verify_isolation(isolation: Mapping[str, Any]) -> dict[str, Any]:
    """@brief 校验一次 DSL 构建请求声明的隔离事实（离线、无外部读写、无第二套 Seatbelt）。

    @param isolation 隔离声明。
    @return 规范化声明。
    @raises TileLangContractError 在线、有 host、外部读写或另写一套 Seatbelt。
    """
    document = _require_mapping(isolation, "isolation")
    _require_keys(document, ISOLATION_KEYS, "isolation")
    if document["route"] != ISOLATION_ROUTE:
        raise TileLangContractError(
            f"isolation.route {document['route']!r} is not the existing lower-level hmz Profile route "
            f"{ISOLATION_ROUTE!r}; the TileLang Track does not maintain a second Seatbelt renderer"
        )
    if document["online"] is not False:
        raise TileLangContractError(
            f"isolation.online is {document['online']!r}; an untrusted DSL build is offline (no online fallback)"
        )
    hosts = document["hosts"]
    if not isinstance(hosts, (list, tuple)) or hosts:
        raise TileLangContractError(
            f"isolation.hosts is {hosts!r}; the build runs with an empty host allowlist"
        )
    for name in ("external_reads", "external_writes"):
        value = document[name]
        if not isinstance(value, (list, tuple)):
            raise TileLangContractError(f"isolation.{name} must be a list")
        if value:
            raise TileLangContractError(
                f"isolation.{name} declares {list(value)!r}; a DSL build reads only its declared inputs and writes "
                "only its declared artifact/temporary roots"
            )
    roots = document["declared_write_roots"]
    if not isinstance(roots, (list, tuple)) or not roots:
        raise TileLangContractError("isolation.declared_write_roots must be a non-empty list of attempt-owned roots")
    for root in roots:
        if not isinstance(root, str) or not root.startswith("/"):
            raise TileLangContractError(
                f"isolation.declared_write_roots entry {root!r} is not an absolute attempt root"
            )
    if document["internal_fence_or_wrap"] is not False:
        raise TileLangContractError(
            "isolation.internal_fence_or_wrap must be False; the retired internal-fence wrapper is not a second route"
        )
    return {
        "route": ISOLATION_ROUTE,
        "online": False,
        "hosts": [],
        "external_reads": [],
        "external_writes": [],
        "declared_write_roots": [str(root) for root in roots],
        "internal_fence_or_wrap": False,
    }


@dataclass(frozen=True)
class SourceOnlyBuildIdentity:
    """@brief 一次真实 source-only 构建的完整身份（候选 manifest + 前端/options + 观测 + 工件）。"""

    candidate_sha256: str
    dsl_source_sha256: str
    candidate_entries: tuple[Mapping[str, Any], ...]
    frontend: Mapping[str, Any]
    environment: Mapping[str, str]
    attempt: Mapping[str, Any]
    artifacts: tuple[Mapping[str, Any], ...]
    artifact_set_sha256_value: str
    cache_key_value: str
    full_build_value: bool
    track_version: int = TILELANG_TRACK_VERSION

    @property
    def artifact_set_sha256(self) -> str:
        """@brief source-only 工件集摘要（与 MT0 探针规则一致）。

        @return 十六进制 sha256。
        """
        return self.artifact_set_sha256_value

    @property
    def full_build(self) -> bool:
        """@brief 是否是一次通过全部核验的 full source-only 构建。

        Derived from the attempt terminal state and artifact verification, never a caller flag.

        @return True。
        """
        return self.full_build_value

    def document(self) -> dict[str, Any]:
        """@brief 身份文档。

        @return 字段字典。
        """
        return {
            "schema_id": BUILD_IDENTITY_SCHEMA_ID,
            "track": {"track_id": TILELANG_TRACK_ID, "track_version": int(self.track_version)},
            "build_mode": SOURCE_ONLY_BUILD_MODE,
            "device_independent": True,
            "candidate_sha256": self.candidate_sha256,
            "dsl_source_sha256": self.dsl_source_sha256,
            "candidate_entries": [dict(row) for row in self.candidate_entries],
            "frontend": dict(self.frontend),
            "options": options_document(mode=SOURCE_ONLY_BUILD_MODE),
            "options_sha256": options_sha256(),
            "environment": dict(self.environment),
            "attempt": dict(self.attempt),
            "artifacts": [dict(row) for row in self.artifacts],
            "artifact_set_sha256": self.artifact_set_sha256_value,
            "full_build": self.full_build_value,
            "device_dispatch_calls": 0,
            "device_execution": {
                "device_library_created": False,
                "pipeline_state_created": False,
                "buffer_allocated": False,
                "dispatch_calls": 0,
            },
            "isolation": {
                "route": ISOLATION_ROUTE,
                "online": False,
                "hosts": [],
                "internal_fence_or_wrap": False,
            },
        }

    @property
    def digest(self) -> str:
        """@brief 构建身份摘要。

        @return 十六进制 sha256。
        """
        return fingerprint(self.document())

    @property
    def cache_key(self) -> str:
        """@brief 构建缓存键（候选 manifest + DSL 源 + 前端/options + 环境，不含结果工件）。

        @return 十六进制 sha256。
        """
        return self.cache_key_value


#: The terminal facts a real attempt must report before a source-only build may be called full.
ATTEMPT_KEYS = frozenset(
    {
        "status",
        "returncode",
        "timed_out",
        "online",
        "hosts",
        "route",
        "request_sha256",
        "hmz_version",
        "policy_sha256",
    }
)


def verify_attempt(attempt: Mapping[str, Any]) -> dict[str, Any]:
    """@brief 校验一次真实尝试的终态（完成、0 退出、未超时、离线、既有 lower-level 路线）。

    @param attempt 尝试记录（由 ``HmzFenceExecutor`` 的结果派生，不是候选自报）。
    @return 规范化记录。
    @raises TileLangCodegenError 超时/非零/未完成。
    @raises TileLangContractError 离线或隔离路线不符。
    """
    document = _require_mapping(attempt, "attempt")
    _require_keys(document, ATTEMPT_KEYS, "attempt")
    if document["online"] is not False or list(document["hosts"] or ()):
        raise TileLangContractError("the attempt was not an offline build with an empty host allowlist")
    if document["route"] != ISOLATION_ROUTE:
        raise TileLangContractError(
            f"attempt.route {document['route']!r} is not the lower-level hmz Profile route {ISOLATION_ROUTE!r}"
        )
    _require_digest(document["request_sha256"], "attempt.request_sha256")
    _require_digest(document["policy_sha256"], "attempt.policy_sha256")
    if not str(document["hmz_version"]).strip():
        raise TileLangContractError("attempt.hmz_version must name the driver installation that ran the fence")
    returncode = document["returncode"]
    if isinstance(returncode, bool) or not isinstance(returncode, int):
        raise TileLangContractError("attempt.returncode must be an integer")
    require_codegen_success(
        returncode=returncode,
        timed_out=document["timed_out"] is not False,
        artifacts_present=True,
    )
    if document["status"] != "completed":
        raise TileLangCodegenError(
            f"the attempt status is {document['status']!r}; only a completed attempt is a full build"
        )
    return {
        "status": "completed",
        "returncode": 0,
        "timed_out": False,
        "online": False,
        "hosts": [],
        "route": ISOLATION_ROUTE,
        "request_sha256": str(document["request_sha256"]),
        "hmz_version": str(document["hmz_version"]),
        "policy_sha256": str(document["policy_sha256"]),
    }


def _require_text(artifacts: Mapping[str, bytes], path: str) -> str:
    """@brief 以 UTF-8 文本读取一个工件。

    @param artifacts 工件字节。
    @param path 工件路径。
    @return 文本。
    @raises TileLangContractError 非法 UTF-8 或空。
    """
    data = artifacts[path]
    try:
        text = bytes(data).decode("utf-8")
    except UnicodeDecodeError as exc:
        raise TileLangContractError(f"artifact {path!r} is not UTF-8 text: {exc}") from exc
    if not text.strip():
        raise TileLangContractError(f"artifact {path!r} is empty")
    return text


def verify_source_only_artifacts(
    artifacts: Mapping[str, bytes],
    *,
    dsl_source_sha256: str,
    expected_kernel_names: Iterable[str] = (),
) -> list[dict[str, Any]]:
    """@brief 核验 source-only 工件的内容契约（不是“有字节就算 full build”）。

    @param artifacts 闭集工件字节。
    @param dsl_source_sha256 候选 ``kernel.py`` 摘要（前端身份里必须回指它）。
    @param expected_kernel_names 候选几何声明的内核名（生成的 MSL 必须包含它们）。
    @return 工件行。
    @raises TileLangContractError 内容/身份/闭集不符。
    """
    rows = _artifact_rows(artifacts)
    identity = json.loads(_require_text(artifacts, "frontend_identity.json"))
    if not isinstance(identity, Mapping):
        raise TileLangContractError("frontend_identity.json must be an object")
    for name in ("verified", "build_options", "device_execution", "candidate", "python"):
        if name not in identity:
            raise TileLangContractError(f"frontend_identity.json is missing {name!r}")
    verified = _require_mapping(identity["verified"], "frontend_identity.verified")
    observed = verify_frontend_observation(
        {
            "frontend_id": FRONTEND_ID,
            "frontend_version": verified.get("tilelang_version"),
            "entry": FRONTEND_ENTRY,
            "options_sha256": options_sha256(),
            "package_manifest_sha256": verified.get("package_manifest_sha256"),
            "dist_info_metadata_sha256": verified.get("dist_info_metadata_sha256"),
            "dist_info_record_sha256": verified.get("dist_info_record_sha256"),
            "package_files": verified.get("package_files"),
            "package_bytes": verified.get("package_bytes"),
        }
    )
    if str(observed["frontend_version"]) != FRONTEND_VERSION:
        raise TileLangIdentityDrift("the in-fence identity does not report the pinned tilelang version")
    options = _require_mapping(identity["build_options"], "frontend_identity.build_options")
    for name, value in (
        ("frontend_entry", FRONTEND_ENTRY),
        ("enable_host_codegen", True),
        ("enable_device_compile", False),
        ("host_artifact_format", "c-source"),
    ):
        if options.get(name) != value:
            raise TileLangIdentityDrift(
                f"frontend_identity.build_options.{name} is {options.get(name)!r}, the frozen option is {value!r}"
            )
    target = _require_mapping(options.get("target"), "frontend_identity.build_options.target")
    for name, value in (("kind", SOURCE_ONLY_OPTIONS["target_kind"]), ("target_host", SOURCE_ONLY_OPTIONS["target_host"])):
        if target.get(name) != value:
            raise TileLangIdentityDrift(f"frontend_identity target.{name} is {target.get(name)!r}, expected {value!r}")
    if [str(key) for key in target.get("keys") or ()] != list(SOURCE_ONLY_OPTIONS["target_keys"]):
        raise TileLangIdentityDrift("frontend_identity target keys drifted from the frozen options")
    device = _require_mapping(identity["device_execution"], "frontend_identity.device_execution")
    for name in ("device_library_created", "pipeline_state_created", "buffer_allocated"):
        if device.get(name) is not False:
            raise TileLangContractError(f"frontend_identity reports {name}={device.get(name)!r}")
    dispatch = device.get("dispatch_calls", -1)
    if isinstance(dispatch, bool) or dispatch != 0:
        raise TileLangContractError("frontend_identity reports device dispatch calls")
    candidate = _require_mapping(identity["candidate"], "frontend_identity.candidate")
    if str(candidate.get("source_sha256")) != dsl_source_sha256:
        raise TileLangIdentityDrift(
            "frontend_identity.candidate.source_sha256 is not the projected DSL source; the report belongs to another "
            "candidate"
        )
    kernel = _require_text(artifacts, "kernel.metal")
    if "metal_stdlib" not in kernel and "using namespace metal" not in kernel:
        raise TileLangContractError("kernel.metal is not Metal Shading Language source")
    symbols = _kernel_symbols(kernel)
    if not symbols:
        raise TileLangContractError("kernel.metal declares no [[kernel]] entry point")
    missing = [
        name
        for name in expected_kernel_names
        if name and not any(_same_kernel(name, symbol) for symbol in symbols)
    ]
    if missing:
        raise TileLangContractError(
            f"kernel.metal omits the candidate-declared kernel(s) {missing} (emitted {sorted(symbols)}); the build does "
            "not match the frozen geometry"
        )
    device_ir = json.loads(_require_text(artifacts, "device_ir.json"))
    if not isinstance(device_ir, (Mapping, list)) or not device_ir:
        raise TileLangContractError("device_ir.json is not a non-empty lowered device module")
    params = json.loads(_require_text(artifacts, "kernel_params.json"))
    if not params:
        raise TileLangContractError("kernel_params.json is empty")
    host = _require_text(artifacts, "host_module.txt")
    if "#include" not in host and not host.lstrip().startswith("{"):
        raise TileLangContractError("host_module.txt is not generated host source")
    return rows


_KERNEL_ENTRY = re.compile(r"\[\[\s*kernel[^\]]*\]\]\s*void\s+([A-Za-z_][A-Za-z0-9_]*)|\bkernel\s+void\s+([A-Za-z_][A-Za-z0-9_]*)")


def _kernel_symbols(kernel_text: str) -> set[str]:
    """@brief 从 MSL 源里取出声明的内核符号。

    @param kernel_text MSL 源。
    @return 符号集合。
    """
    return {match.group(1) or match.group(2) for match in _KERNEL_ENTRY.finditer(kernel_text)}


def _same_kernel(declared: str, emitted: str) -> bool:
    """@brief 候选声明的 stage 内核与前端发出的符号是否是同一颗核。

    TileLang 会给 prim_func 派生后缀（``foo`` → ``foo_kernel``），候选也可以直接写发出名。
    这里只接受 ``相等`` 或一侧是另一侧加 ``_kernel``，不引入新的命名约定，也不接受无关符号。

    @param declared 几何声明的内核名。
    @param emitted MSL 里声明的符号。
    @return 是否同一颗核。
    """
    return emitted in {declared, declared + "_kernel"} or declared in {emitted, emitted + "_kernel"}


def _artifact_rows(artifacts: Mapping[str, bytes]) -> list[dict[str, Any]]:
    """@brief 按冻结顺序构造工件行。

    @param artifacts 路径 → 字节。
    @return 行列表。
    @raises TileLangContractError 缺失/多余/空工件。
    """
    if not isinstance(artifacts, Mapping):
        raise TileLangContractError("artifacts must be a mapping of path -> bytes")
    unknown = sorted(set(artifacts) - set(SOURCE_ONLY_ARTIFACT_PATHS))
    missing = [path for path in SOURCE_ONLY_ARTIFACT_PATHS if path not in artifacts]
    if unknown or missing:
        raise TileLangContractError(
            f"the source-only artifact set is closed: missing {missing}, undeclared {unknown}"
        )
    rows: list[dict[str, Any]] = []
    for path in SOURCE_ONLY_ARTIFACT_PATHS:
        data = artifacts[path]
        if not isinstance(data, (bytes, bytearray)) or not data:
            raise TileLangContractError(f"artifact {path!r} must be non-empty bytes")
        rows.append({"path": path, "bytes": len(data), "sha256": _sha256_bytes(data)})
    return rows


def artifact_set_sha256(rows: Iterable[Mapping[str, Any]]) -> str:
    """@brief 冻结的工件集摘要规则（与 MT0 探针逐字节相同）。

    @param rows 工件行（保持声明顺序）。
    @return 十六进制 sha256。
    """
    canonical = json.dumps([dict(row) for row in rows], sort_keys=True, separators=(",", ":")).encode("utf-8")
    return _sha256_bytes(canonical)


def _candidate_manifest(candidate: Any) -> tuple[str, str, tuple[Mapping[str, Any], ...]]:
    """@brief 取候选 manifest 身份与 DSL 源摘要。

    @param candidate ``CandidateArtifact``（鸭子类型）。
    @return ``(candidate_sha256, dsl_source_sha256, entries)``。
    @raises TileLangContractError 不是 DSL 候选。
    """
    sources = [entry for entry in candidate.entries if entry.role == SOURCE_ROLE]
    if len(sources) != 1:
        raise TileLangContractError("the candidate must carry exactly one dsl_source entry")
    manifest = candidate.manifest()
    entries = tuple(dict(row) for row in manifest["entries"])
    return str(candidate.candidate_sha256), _sha256_bytes(bytes(sources[0].data)), entries


def expected_build_subject(
    *, candidate: Any, environment: Mapping[str, Any], track_version: int = TILELANG_TRACK_VERSION
) -> SourceOnlyBuildIdentity:
    """@brief 由独立可信输入重建“期望构建主语”（候选 manifest + 冻结前端/options + 环境）。

    This is reproducible *before* a build: it binds every candidate entry (therefore the
    geometry) and the pinned frontend, so two candidates that differ only in their execution
    geometry have different subjects.  Artifacts/attempt facts are results and are not part of
    the cache key.

    @param candidate 投影候选（``CandidateArtifact``）。
    @param environment 设备无关环境事实（闭集）。
    @return 主语身份（``full_build`` 为 False，因为还没有尝试结果）。
    @raises TileLangContractError 环境/候选非法。
    """
    if track_version not in SUPPORTED_TRACK_VERSIONS:
        raise TileLangContractError(f"unsupported TileLang Track contract version {track_version!r}")
    candidate_sha256, dsl_source_sha256, entries = _candidate_manifest(candidate)
    frontend = verify_frontend_observation(
        {
            "frontend_id": FRONTEND_ID,
            "frontend_version": FRONTEND_VERSION,
            "entry": FRONTEND_ENTRY,
            "options_sha256": options_sha256(),
            "package_manifest_sha256": FRONTEND_PACKAGE_MANIFEST_SHA256,
            "dist_info_metadata_sha256": FRONTEND_DIST_INFO_METADATA_SHA256,
            "dist_info_record_sha256": FRONTEND_DIST_INFO_RECORD_SHA256,
            "package_files": FRONTEND_PACKAGE_FILES,
            "package_bytes": FRONTEND_PACKAGE_BYTES,
        }
    )
    env = _require_mapping(environment, "environment")
    _require_keys(env, FRONTEND_ENVIRONMENT_KEYS, "environment")
    for name, value in env.items():
        if not isinstance(value, str) or not value:
            raise TileLangContractError(f"environment.{name} must be a non-empty string")
    cache_key = fingerprint(
        {
            "schema_id": BUILD_IDENTITY_SCHEMA_ID,
            "track": {"track_id": TILELANG_TRACK_ID, "track_version": int(track_version)},
            "build_mode": SOURCE_ONLY_BUILD_MODE,
            "candidate_sha256": candidate_sha256,
            "dsl_source_sha256": dsl_source_sha256,
            "candidate_entries": [dict(row) for row in entries],
            "frontend": frontend,
            "options_sha256": options_sha256(),
            "environment": dict(env),
        }
    )
    return SourceOnlyBuildIdentity(
        candidate_sha256=candidate_sha256,
        dsl_source_sha256=dsl_source_sha256,
        candidate_entries=entries,
        frontend=frontend,
        environment=dict(env),
        attempt={},
        artifacts=(),
        artifact_set_sha256_value="",
        cache_key_value=cache_key,
        full_build_value=False,
        track_version=int(track_version),
    )


def source_only_build_identity(
    *,
    candidate: Any,
    observed_frontend: Mapping[str, Any],
    environment: Mapping[str, Any],
    artifacts: Mapping[str, bytes],
    attempt: Mapping[str, Any],
    expected_kernel_names: Iterable[str] = (),
    build_mode: str = SOURCE_ONLY_BUILD_MODE,
    track_version: int = TILELANG_TRACK_VERSION,
) -> SourceOnlyBuildIdentity:
    """@brief 由一次真实 source-only 尝试构造并校验 build 身份。

    Every input is checked: the candidate manifest (all entries, therefore the geometry), the
    exact pinned frontend install, the device-independent environment, the attempt terminal state
    and every artifact's content.  ``full_build`` is derived, never supplied.

    @param candidate 投影候选（``CandidateArtifact``）。
    @param observed_frontend 前端观测（必须等于冻结安装身份）。
    @param environment 设备无关环境事实（闭集）。
    @param artifacts 尝试产出的闭集工件字节（控制机重新读取）。
    @param attempt 尝试终态。
    @param expected_kernel_names 候选几何声明的内核名。
    @param build_mode 构建模式；本 epoch 只有 ``source_only``。
    @return build 身份。
    @raises TileLangBuildModeUnavailable 请求 Metal build。
    @raises TileLangContractError/ TileLangIdentityDrift/ TileLangCodegenError 任一核验失败。
    """
    if build_mode not in BUILD_MODES:
        raise TileLangBuildModeUnavailable(f"unknown TileLang build mode {build_mode!r}; known {list(BUILD_MODES)}")
    if track_version not in SUPPORTED_TRACK_VERSIONS:
        raise TileLangContractError(f"unsupported TileLang Track contract version {track_version!r}")
    if build_mode != SOURCE_ONLY_BUILD_MODE:
        if track_version == TILELANG_TRACK_VERSION:
            raise TileLangBuildModeUnavailable(
                f"TileLang build mode {build_mode!r} is not implemented in this epoch; only "
                f"{list(IMPLEMENTED_BUILD_MODES)} exists in /1"
            )
        raise TileLangBuildModeUnavailable(
            f"source_only_build_identity describes only {SOURCE_ONLY_BUILD_MODE!r}; Metal identity is produced by "
            "the /2 Metal builder after real device compilation"
        )
    candidate_sha256, dsl_source_sha256, entries = _candidate_manifest(candidate)
    frontend = verify_frontend_observation(observed_frontend)
    env = _require_mapping(environment, "environment")
    _require_keys(env, FRONTEND_ENVIRONMENT_KEYS, "environment")
    for name, value in env.items():
        if not isinstance(value, str) or not value:
            raise TileLangContractError(f"environment.{name} must be a non-empty string")
    terminal = verify_attempt(attempt)
    rows = verify_source_only_artifacts(
        artifacts,
        dsl_source_sha256=dsl_source_sha256,
        expected_kernel_names=tuple(expected_kernel_names),
    )
    artifact_digest = artifact_set_sha256(rows)
    cache_key = fingerprint(
        {
            "schema_id": BUILD_IDENTITY_SCHEMA_ID,
            "track": {"track_id": TILELANG_TRACK_ID, "track_version": int(track_version)},
            "build_mode": build_mode,
            "candidate_sha256": candidate_sha256,
            "dsl_source_sha256": dsl_source_sha256,
            "candidate_entries": [dict(row) for row in entries],
            "frontend": frontend,
            "options_sha256": options_sha256(mode=build_mode),
            "environment": dict(env),
        }
    )
    return SourceOnlyBuildIdentity(
        candidate_sha256=candidate_sha256,
        dsl_source_sha256=dsl_source_sha256,
        candidate_entries=entries,
        frontend=frontend,
        environment=dict(env),
        attempt=terminal,
        artifacts=tuple(rows),
        artifact_set_sha256_value=artifact_digest,
        cache_key_value=cache_key,
        full_build_value=True,
        track_version=int(track_version),
    )


def require_codegen_success(*, returncode: int, timed_out: bool, artifacts_present: bool) -> None:
    """@brief codegen 终态门：超时/非零/缺工件都不是构建。

    @param returncode 前端退出码。
    @param timed_out 是否被看门狗终止。
    @param artifacts_present 闭集工件是否齐全（由围栏收集器判定）。
    @raises TileLangCodegenError 任一失败。
    """
    if timed_out:
        raise TileLangCodegenError("the frontend codegen timed out; a timed-out build is not a build")
    if isinstance(returncode, bool) or not isinstance(returncode, int):
        raise TileLangCodegenError("the frontend returncode must be an integer")
    if returncode != 0:
        raise TileLangCodegenError(f"the frontend codegen exited {returncode}; a failed codegen is not a build")
    if not artifacts_present:
        raise TileLangCodegenError("the frontend produced no closed artifact set (incomplete)")


def replay_source_only(
    *,
    result: Mapping[str, Any],
    artifacts: Mapping[str, bytes],
    candidate_source: bytes,
) -> dict[str, Any]:
    """@brief 只用归档字节重建一次 MT0 source-only 构建的身份（CPU-only，不执行前端）。

    This re-derives the archived facts; it does not re-run TileLang, Clang or Metal and it is
    **not** a Metal correctness/performance claim.

    @param result 归档的 ``probe-result-r5.json``。
    @param artifacts 归档的六个工件字节（含 ``build_identity.json``）。
    @param candidate_source 归档的候选 ``kernel.py`` 字节。
    @return 重建的身份文档。
    @raises TileLangContractError 归档内部不一致或含设备执行声明。
    @raises TileLangIdentityDrift 前端/options 与冻结身份不符。
    """
    document = _require_mapping(result, "archived result")
    if document.get("full_build") is not True:
        raise TileLangContractError("the archived result is not a full source-only build")
    if document.get("gpu_npu_executed") is not False:
        raise TileLangContractError("the archived result claims GPU/NPU execution; it is not a source-only probe")
    if int(document.get("device_dispatch_calls", -1)) != 0:
        raise TileLangContractError("the archived result reports device dispatches; the source-only boundary is broken")
    build = _require_mapping(document.get("build"), "archived result.build")
    rows = build.get("artifact_rows")
    if not isinstance(rows, (list, tuple)):
        raise TileLangContractError("the archived result has no artifact rows")
    declared_paths = [str(row.get("path")) for row in rows]
    if declared_paths != list(SOURCE_ONLY_ARTIFACT_PATHS):
        raise TileLangContractError(
            f"the archived artifact rows {declared_paths} are not the frozen set {list(SOURCE_ONLY_ARTIFACT_PATHS)}"
        )
    verified_rows = []
    for row in rows:
        path = str(row["path"])
        data = artifacts.get(path)
        if not isinstance(data, (bytes, bytearray)):
            raise TileLangContractError(f"archived artifact {path!r} is missing")
        if len(data) != int(row["bytes"]) or _sha256_bytes(data) != str(row["sha256"]):
            raise TileLangContractError(f"archived artifact {path!r} bytes do not match its declared row")
        verified_rows.append({"path": path, "bytes": len(data), "sha256": _sha256_bytes(data)})
    recomputed = artifact_set_sha256(verified_rows)
    if recomputed != str(build.get("artifact_set_sha256")):
        raise TileLangContractError(
            "the archived artifact_set_sha256 does not match the bytes; the probe identity is not reproducible"
        )
    options = _require_mapping(build.get("build_options"), "archived result.build.build_options")
    for name, expected in (
        ("frontend_entry", FRONTEND_ENTRY),
        ("enable_device_compile", False),
        ("enable_host_codegen", True),
        ("host_artifact_format", "c-source"),
    ):
        if options.get(name) != expected:
            raise TileLangIdentityDrift(
                f"archived build option {name} is {options.get(name)!r}, the frozen option is {expected!r}"
            )
    target = _require_mapping(options.get("target"), "archived build option target")
    for name, expected in (("kind", "metal"), ("target_host", "c")):
        if target.get(name) != expected:
            raise TileLangIdentityDrift(f"archived target {name} is {target.get(name)!r}, expected {expected!r}")
    if [str(key) for key in target.get("keys") or ()] != list(SOURCE_ONLY_OPTIONS["target_keys"]):
        raise TileLangIdentityDrift("archived target keys drifted from the frozen options")
    device = _require_mapping(build.get("device_execution"), "archived result.build.device_execution")
    for name in ("device_library_created", "pipeline_state_created", "buffer_allocated"):
        if device.get(name) is not False:
            raise TileLangContractError(f"archived source-only build reports {name}={device.get(name)!r}")
    if int(device.get("dispatch_calls", -1)) != 0:
        raise TileLangContractError("archived source-only build reports dispatch calls")
    frontend = _require_mapping(document.get("frontend"), "archived result.frontend")
    if str(frontend.get("version")) != FRONTEND_VERSION:
        raise TileLangIdentityDrift(
            f"archived frontend version {frontend.get('version')!r} is not the frozen {FRONTEND_VERSION!r}"
        )
    if str(frontend.get("upstream_commit")) != FRONTEND_UPSTREAM_COMMIT:
        raise TileLangIdentityDrift("archived frontend upstream commit drifted from the frozen identity")
    if str(frontend.get("wheel_sha256")) != FRONTEND_WHEEL_SHA256:
        raise TileLangIdentityDrift("archived frontend wheel sha256 drifted from the frozen identity")
    isolation = _require_mapping(document.get("isolation"), "archived result.isolation")
    if isolation.get("status") != "completed" or int(isolation.get("returncode", -1)) != 0:
        raise TileLangContractError("the archived attempt did not complete successfully")
    if isolation.get("policy_online") is not False or list(isolation.get("policy_hosts") or ()):
        raise TileLangContractError("the archived attempt was not an offline build with an empty host allowlist")
    if isolation.get("internal_fence_or_wrap_used") is not False:
        raise TileLangContractError("the archived attempt used a second fence/wrap instead of the lower-level route")
    manifest = _sha256_bytes(bytes(candidate_source))
    return {
        "schema_id": SOURCE_ONLY_REPLAY_SCHEMA_ID,
        "track": {"track_id": TILELANG_TRACK_ID, "track_version": TILELANG_TRACK_VERSION},
        "build_mode": SOURCE_ONLY_BUILD_MODE,
        "candidate_sha256": manifest,
        "artifact_set_sha256": recomputed,
        "artifacts": verified_rows,
        "frontend": {
            "frontend_id": FRONTEND_ID,
            "frontend_version": FRONTEND_VERSION,
            "entry": FRONTEND_ENTRY,
            "wheel_sha256": FRONTEND_WHEEL_SHA256,
            "upstream_commit": FRONTEND_UPSTREAM_COMMIT,
        },
        "device_dispatch_calls": 0,
        "gpu_npu_executed": False,
        "metal_correctness_or_performance_claim": False,
        "production_ready": False,
        "publication_allowed": False,
        "note": (
            "Replay of the archived MT0 source-only artifacts under the frozen Track contract identity. "
            "It re-derives bytes/digests only and is not a Metal build, correctness, timing or availability claim."
        ),
    }


__all__ = [
    "BUILD_IDENTITY_SCHEMA_ID",
    "BUILD_MODES",
    "ENVIRONMENT_KEYS",
    "FRONTEND_ENTRY",
    "FRONTEND_ID",
    "FRONTEND_UPSTREAM_COMMIT",
    "FRONTEND_VERSION",
    "FRONTEND_WHEEL_SHA256",
    "GEOMETRY_MEDIA_TYPE",
    "GEOMETRY_PATH",
    "GEOMETRY_ROLE",
    "IMPLEMENTED_BUILD_MODES",
    "ISOLATION_KEYS",
    "ISOLATION_ROUTE",
    "METAL_BUILD_MODE",
    "METAL_BUILD_OPTIONS",
    "SOURCE_MEDIA_TYPE",
    "SOURCE_ONLY_ARTIFACT_MEDIA_TYPES",
    "SOURCE_ONLY_ARTIFACT_PATHS",
    "SOURCE_ONLY_ARTIFACTS",
    "SOURCE_ONLY_BUILD_MODE",
    "SOURCE_ONLY_OPTIONS",
    "SOURCE_ONLY_REPLAY_SCHEMA_ID",
    "SUPPORTED_TRACK_VERSIONS",
    "SOURCE_PATH",
    "SOURCE_ROLE",
    "SourceOnlyBuildIdentity",
    "TILELANG_DELIVERY",
    "TILELANG_METAL_TRACK_VERSION",
    "TILELANG_TRACK_ID",
    "TILELANG_TRACK_VERSION",
    "TRACK_CONTRACT_SCHEMA_ID",
    "TileLangBuildModeUnavailable",
    "TileLangCodegenError",
    "TileLangContractError",
    "TileLangFrontendUnavailable",
    "TileLangIdentityDrift",
    "artifact_set_sha256",
    "audit_registration",
    "expected_build_subject",
    "frozen_contract",
    "frontend_identity",
    "options_document",
    "options_sha256",
    "replay_source_only",
    "require_codegen_success",
    "source_only_build_identity",
    "source_spec",
    "validate_candidate",
    "verify_frontend_observation",
    "verify_isolation",
]
