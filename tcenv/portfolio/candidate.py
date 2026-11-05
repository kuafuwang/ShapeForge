"""Candidate portfolio envelope and the mechanical projection to a CandidateArtifact (P0).

The outer portfolio is a *candidate bundle* whose fixed entry ``coverage_plan/portfolio.json``
declares coverage buckets, variants (guard + entries + one geometry declaration) and one domain
fallback.  It is **not** a Track deliverable: it is never handed to
``CandidateArtifact.validate_for`` or to a TrackRegistry.  Instead every variant and fallback is
projected, mechanically and without touching candidate bytes, into an ordinary
``CandidateArtifact``:

* a variant/fallback with an explicit ``geometry_entry`` becomes ``msl-bundle``: exactly one
  ``host_tiling`` entry, at least one device source, the declared geometry mapped to the fixed
  ``host/execution_geometry.json`` ``schedule`` entry, and a test OpSpec that explicitly declares
  workspace;
* a variant/fallback without ``geometry_entry`` must be exactly one self-contained device entry
  whose source keeps its own ``taic-launch`` block, and becomes ``msl-single``;
* anything else — multi-source without the bundle conditions, two host entries, a missing or
  duplicated geometry declaration, a role no registered track delivers — is refused.  The
  projection never invents a host plan, a workspace tensor or a launch block.

Shared sources are listed by every provider that compiles them and are copied into each
projection byte-for-byte; the projection never keeps only a primary source file.  Two providers
may list the same shared entry (that is how a common dependency is declared), but every declared
entry is validated against the parent bundle and two providers cannot share (or duplicate) a
geometry entry.  The child's ``candidate_sha256`` is the projected candidate-entry manifest digest
and is never the parent portfolio's digest: an ``EvaluationBinding`` records both the parent
identity and the original geometry entry path and content SHA-256.

The domain fallback is a *correctness provider* with its own identity and its own geometry
declaration.  Its registered role is ``domain_fallback`` and is deliberately distinct from the
timing-baseline role: a timing baseline is not a correctness oracle, and a fallback success or
timing is never credited to a generated variant (the dispatch layer records
``provider=fallback``).
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence

from ..backends.metal.launch import LaunchRejected, unique_launch_block
from ..contract_errors import GeometryError, TaskContractError
from ..identity import IdentityError, canonical_bytes, fingerprint, is_hex64, load_strict_json
from ..opspec.model import OpSpec
from ..opspec.v2 import SemanticInstanceV2
from ..target import TargetProfile
from ..schema import SchemaError, loads_strict
from ..opspec.validator import SchemaFailure, validate
from ..track.delivery import CANDIDATE_GEOMETRY_PATH, parse_candidate_geometry
from ..track.plane import (
    CANDIDATE_MANIFEST_SCHEMA_ID,
    CandidateArtifact,
    CandidateEntry,
    ExecutionGeometry,
)
from ..track.registry import TrackRegistry, load_track_registry
from .domain import (
    DomainAuthority,
    Guard,
    SemanticDomainPlan,
    guard_ast_document,
    parse_guard,
    semantic_instance_digest,
)

PORTFOLIO_SCHEMA_ID = "tcenv.candidate-portfolio.1"
SCHEMA_DIR = Path(__file__).resolve().parent / "schema"
PORTFOLIO_SCHEMA_PATH = SCHEMA_DIR / "candidate-portfolio.v1.json"

#: The one fixed outer entry.  The portfolio parser — not a Track — reads it.  Its role is
#: ``launch``: a registered candidate-role name that no TrackSpec accepts for a source, so the
#: outer envelope can never pass ``CandidateArtifact.validate_for`` as if it were a deliverable.
PORTFOLIO_ENTRY_PATH = "coverage_plan/portfolio.json"
PORTFOLIO_ENTRY_ROLE = "launch"
PORTFOLIO_ENTRY_MEDIA_TYPE = "application/json"
#: Where a declared variant/fallback geometry is projected.  The same fixed path the existing
#: ``candidate_geometry`` reader owns; the parent's original path is kept in the binding.
PROJECTED_GEOMETRY_PATH = CANDIDATE_GEOMETRY_PATH
PROJECTED_GEOMETRY_ROLE = "schedule"
#: A declared geometry entry in the parent is candidate Host Tiling data (JSON), not an MSL source.
GEOMETRY_ENTRY_ROLE = "host_tiling"
GEOMETRY_MEDIA_TYPE = "application/json"
#: The two ordinary delivery mechanisms a projection may use.  No portfolio-specific Track exists.
SINGLE_TRACK_ID = "msl-single"
BUNDLE_TRACK_ID = "msl-bundle"
#: Roles a projected child may carry.  ``launch`` is registered as a candidate role but no
#: registered Track delivers it: a launch block lives in the MSL source, not in a second entry.
PROJECTABLE_ROLES = frozenset({"device_kernel", "host_tiling"})
LAUNCH_MARKER = "taic-launch"
#: A quoted ``#include`` is resolved inside the delivered entry set.  The projection preserves
#: bytes, so the existing A4 include/host-path screen still applies; this check additionally
#: refuses a projection that dropped the included dependency (or that includes outside the bundle).
_INCLUDE_LINE = re.compile(r"^\s*#\s*include\s*(.*)$", re.MULTILINE)
_QUOTED_TARGET = re.compile(r'^"([^"]*)"\s*$')
#: The A4/v2 screen allows only ``<metal_...>`` headers.  The projection boundary uses the same
#: closed set, so an angle-bracket host path (``</etc/passwd>``, ``<stdio.h>``) is refused here
#: rather than at the first build, while a real Metal standard header still passes.
_METAL_STANDARD_TARGET = re.compile(r"^<metal_[a-z_0-9]+>\s*$")
FALLBACK_KINDS = frozenset({"registered_implementation", "refuse"})
#: Reasons the *runner* may derive from an execution stage.  A candidate declares which reasons it
#: supports; it never supplies the reason for a given request.
FALLBACK_REASONS = frozenset({"outside_candidate_coverage", "resource_unavailable", "variant_unavailable"})
DOMAIN_FALLBACK_ROLE = "domain_fallback"
TIMING_BASELINE_ROLE = "timing_baseline"
EVIDENCE_STATUSES = frozenset({"validated", "unknown"})


def _schema() -> Mapping[str, Any]:
    """@brief 读取登记 portfolio schema（闭集子集校验器）。

    @return schema 文档。
    @raises TaskContractError schema 不可读。
    """
    try:
        document = load_strict_json(PORTFOLIO_SCHEMA_PATH)
    except IdentityError as exc:
        raise TaskContractError(f"{PORTFOLIO_SCHEMA_PATH}: the portfolio schema is not strict JSON: {exc}") from exc
    if not isinstance(document, Mapping):
        raise TaskContractError(f"{PORTFOLIO_SCHEMA_PATH}: the portfolio schema must be an object")
    return document


def _relative_path(text: Any, *, where: str) -> str:
    """@brief 校验一个规范化相对 POSIX 路径。

    @param text 路径文本。
    @param where 诊断位置。
    @return 原文本。
    @raises TaskContractError 绝对路径、``.``/``..``、反斜杠或非规范化。
    """
    if not isinstance(text, str) or not text:
        raise TaskContractError(f"{where}: expected a non-empty relative POSIX path")
    if text.startswith("/") or "\\" in text:
        raise TaskContractError(f"{where}: {text!r} must be a relative POSIX path")
    path = PurePosixPath(text)
    if str(path) != text or any(part in (".", "..") for part in path.parts):
        raise TaskContractError(f"{where}: {text!r} must be a normalized relative path without '.'/'..' segments")
    return text


def _require_digest(value: Any, *, where: str) -> str:
    """@brief 校验 64 位小写十六进制摘要。

    @param value 值。
    @param where 诊断位置。
    @return 原值。
    @raises TaskContractError 非法。
    """
    if not is_hex64(value):
        raise TaskContractError(f"{where} must be a 64-character lowercase hex digest, got {value!r}")
    return str(value)


def _hash_bytes(data: bytes) -> str:
    """@brief 字节 sha256。

    @param data 字节。
    @return 十六进制摘要。
    """
    return hashlib.sha256(bytes(data)).hexdigest()


@dataclass(frozen=True)
class EntryReference:
    """@brief 一个候选 entry 的清单引用（路径/角色/媒体类型/字节摘要）。"""

    path: str
    role: str
    media_type: str
    size_bytes: int
    sha256: str

    @classmethod
    def from_entry(cls, entry: CandidateEntry) -> "EntryReference":
        """@brief 由 entry 构造引用。

        @param entry 候选 entry。
        @return 引用。
        """
        return cls(
            path=entry.path,
            role=entry.role,
            media_type=entry.media_type,
            size_bytes=len(entry.data),
            sha256=entry.sha256,
        )

    def __post_init__(self) -> None:
        """@brief 校验引用形状。

        @raises TaskContractError 非法。
        """
        _relative_path(self.path, where="EntryReference.path")
        if self.role not in PROJECTABLE_ROLES | {PROJECTED_GEOMETRY_ROLE}:
            raise TaskContractError(f"{self.path}: unknown projected role {self.role!r}")
        if not self.media_type or "/" not in self.media_type:
            raise TaskContractError(f"{self.path}: media_type must look like type/subtype")
        if not isinstance(self.size_bytes, int) or isinstance(self.size_bytes, bool) or self.size_bytes < 1:
            raise TaskContractError(f"{self.path}: size_bytes must be a positive integer")
        _require_digest(self.sha256, where=f"{self.path}.sha256")

    def document(self) -> dict[str, Any]:
        """@brief 引用的 JSON 文档。

        @return 字段字典。
        """
        return {
            "path": self.path,
            "role": self.role,
            "media_type": self.media_type,
            "size_bytes": int(self.size_bytes),
            "sha256": self.sha256,
        }

    def matches(self, entry: CandidateEntry) -> bool:
        """@brief 该引用是否与某个 entry 完全一致（路径/角色/媒体类型/字节）。

        @param entry 候选 entry。
        @return 一致时为 True。
        """
        return self.document() == EntryReference.from_entry(entry).document()


@dataclass(frozen=True)
class PlannerIdentity:
    """@brief 候选 planner 身份（id + version，不含作者/会话）。"""

    planner_id: str
    planner_version: int

    def __post_init__(self) -> None:
        """@brief 校验身份。

        @raises TaskContractError 非法。
        """
        if not isinstance(self.planner_id, str) or not self.planner_id.strip():
            raise TaskContractError("planner.id must be a non-empty string")
        if isinstance(self.planner_version, bool) or not isinstance(self.planner_version, int) or self.planner_version < 1:
            raise TaskContractError("planner.version must be an integer >= 1")

    def document(self) -> dict[str, Any]:
        """@brief 身份的 JSON 文档。

        @return 字段字典。
        """
        return {"id": self.planner_id, "version": int(self.planner_version)}


@dataclass(frozen=True)
class PortfolioBucket:
    """@brief 一个 coverage bucket：名字 + 它声明的 guard 摘要。"""

    bucket_key: str
    guard_digest: str

    def __post_init__(self) -> None:
        """@brief 校验。

        @raises TaskContractError 非法。
        """
        if not isinstance(self.bucket_key, str) or not self.bucket_key.strip():
            raise TaskContractError("a coverage bucket needs a non-empty bucket_key")
        _require_digest(self.guard_digest, where=f"bucket {self.bucket_key}.guard_digest")

    def document(self) -> dict[str, Any]:
        """@brief bucket 的 JSON 文档。

        @return 字段字典。
        """
        return {"bucket_key": self.bucket_key, "guard_digest": self.guard_digest}


@dataclass(frozen=True)
class PortfolioVariant:
    """@brief 一个候选特化 variant：guard + 自己的 entries + 几何声明。"""

    variant_key: str
    bucket_key: str
    guard: Guard
    entries: tuple[str, ...]
    geometry_entry: str | None
    geometry: ExecutionGeometry | None
    geometry_sha256: str | None
    delivered: tuple[EntryReference, ...]
    domain_plan_digest: str
    planner: PlannerIdentity

    def __post_init__(self) -> None:
        """@brief 校验 variant 自身的一致性与身份。

        @raises TaskContractError 非法。
        """
        if not isinstance(self.variant_key, str) or not self.variant_key.strip():
            raise TaskContractError("a variant needs a non-empty diagnostic variant_key")
        if not isinstance(self.bucket_key, str) or not self.bucket_key.strip():
            raise TaskContractError(f"{self.variant_key}: a variant needs a bucket_key")
        if not isinstance(self.entries, tuple) or not self.entries:
            raise TaskContractError(f"{self.variant_key}: a variant must list at least one entry")
        if len(set(self.entries)) != len(self.entries):
            raise TaskContractError(f"{self.variant_key}: a variant repeats an entry path")
        for path in self.entries:
            _relative_path(path, where=f"{self.variant_key}.entries")
            if path == PORTFOLIO_ENTRY_PATH:
                raise TaskContractError(
                    f"{self.variant_key}: {path!r} is the parent portfolio manifest, not a variant entry"
                )
        if (self.geometry_entry is None) != (self.geometry is None) or (self.geometry_entry is None) != (
            self.geometry_sha256 is None
        ):
            raise TaskContractError(
                f"{self.variant_key}: geometry_entry, the typed geometry and its SHA-256 must be all present or all "
                "absent; a half-declared geometry would be guessed by a later reader"
            )
        if self.geometry_entry is not None:
            _relative_path(self.geometry_entry, where=f"{self.variant_key}.geometry_entry")
            _require_digest(self.geometry_sha256, where=f"{self.variant_key}.geometry_entry sha256")
            if self.geometry_entry in self.entries:
                raise TaskContractError(
                    f"{self.variant_key}: geometry_entry {self.geometry_entry!r} must not also be listed in entries; "
                    "the projection maps it to the fixed execution-geometry path exactly once"
                )
        if not isinstance(self.delivered, tuple) or not self.delivered:
            raise TaskContractError(f"{self.variant_key}: a variant needs a delivered entry manifest")
        paths = [item.path for item in self.delivered]
        if paths != sorted(paths) or len(set(paths)) != len(paths):
            raise TaskContractError(f"{self.variant_key}: the delivered entry manifest must be sorted and unique")
        _require_digest(self.domain_plan_digest, where=f"{self.variant_key}.domain_plan_digest")
        if not isinstance(self.planner, PlannerIdentity):
            raise TaskContractError(f"{self.variant_key}: a variant needs the planner identity")

    def identity_document(self) -> dict[str, Any]:
        """@brief variant_id 的唯一输入（稳定、author-neutral、无实例解析几何）。

        @return 字段字典。
        """
        return {
            "fingerprint_kind": "portfolio-variant",
            "fingerprint_version": 1,
            "domain_plan_digest": self.domain_plan_digest,
            "planner": self.planner.document(),
            "guard": {"ast": guard_ast_document(self.guard.node), "digest": self.guard.digest},
            "entries": [item.document() for item in self.delivered],
            "geometry": None
            if self.geometry_entry is None
            else {"origin_path": self.geometry_entry, "origin_sha256": self.geometry_sha256},
        }

    @property
    def variant_id(self) -> str:
        """@brief 稳定 variant 身份（不含 variant_key、测量结果或实例解析几何）。

        @return 十六进制 sha256。
        """
        return fingerprint(self.identity_document())

    @property
    def requires_geometry(self) -> bool:
        """@brief 该 variant 是否声明了自己的执行几何（即必须走 msl-bundle）。

        @return 有几何声明时为 True。
        """
        return self.geometry_entry is not None

    def document(self) -> dict[str, Any]:
        """@brief variant 的诊断文档（不含实例解析几何）。

        @return 字段字典。
        """
        return {
            "variant_key": self.variant_key,
            "bucket_key": self.bucket_key,
            "variant_id": self.variant_id,
            "guard": self.guard.document(),
            "entries": list(self.entries),
            "geometry_entry": self.geometry_entry,
        }


@dataclass(frozen=True)
class DomainFallback:
    """@brief 候选 Bundle 内的通用正确实现（``registered_implementation``）或显式拒绝。"""

    kind: str
    implementation_id: str | None
    implementation_version: int | None
    entries: tuple[str, ...]
    geometry_entry: str | None
    geometry: ExecutionGeometry | None
    geometry_sha256: str | None
    reasons: tuple[str, ...]
    delivered: tuple[EntryReference, ...]
    domain_plan_digest: str

    @property
    def role(self) -> str:
        """@brief 该 provider 的注册角色（永远不是计时 baseline）。

        @return ``"domain_fallback"``。
        """
        return DOMAIN_FALLBACK_ROLE

    def __post_init__(self) -> None:
        """@brief 校验 fallback 的种类、身份与原因。

        @raises TaskContractError 非法。
        """
        if self.kind not in FALLBACK_KINDS:
            raise TaskContractError(f"unknown fallback kind {self.kind!r}; known {sorted(FALLBACK_KINDS)}")
        if not isinstance(self.reasons, tuple) or not self.reasons:
            raise TaskContractError("a domain fallback must declare at least one registered reason")
        if len(set(self.reasons)) != len(self.reasons):
            raise TaskContractError("a domain fallback repeats a reason")
        unknown = sorted(set(self.reasons) - FALLBACK_REASONS)
        if unknown:
            raise TaskContractError(f"unknown fallback reasons {unknown}; known {sorted(FALLBACK_REASONS)}")
        if (self.geometry_entry is None) != (self.geometry is None) or (self.geometry_entry is None) != (
            self.geometry_sha256 is None
        ):
            raise TaskContractError("the fallback geometry declaration must be all present or all absent")
        if self.kind == "refuse":
            if self.entries or self.geometry_entry is not None:
                raise TaskContractError("a refusing fallback must not carry entries or a geometry declaration")
            if self.implementation_id is not None or self.implementation_version is not None:
                raise TaskContractError("a refusing fallback must not carry an implementation identity")
        else:
            if not isinstance(self.implementation_id, str) or not self.implementation_id.strip():
                raise TaskContractError("a registered_implementation fallback needs a non-empty implementation_id")
            if (
                isinstance(self.implementation_version, bool)
                or not isinstance(self.implementation_version, int)
                or self.implementation_version < 1
            ):
                raise TaskContractError("a registered_implementation fallback needs implementation_version >= 1")
            if not self.entries:
                raise TaskContractError("a registered_implementation fallback must list at least one entry")
            if len(set(self.entries)) != len(self.entries):
                raise TaskContractError("a domain fallback repeats an entry path")
            for path in self.entries:
                _relative_path(path, where="fallback.entries")
                if path == PORTFOLIO_ENTRY_PATH:
                    raise TaskContractError(
                        f"fallback entry {path!r} is the parent portfolio manifest, not a projection source"
                    )
            if self.geometry_entry is not None:
                _relative_path(self.geometry_entry, where="fallback.geometry_entry")
                _require_digest(self.geometry_sha256, where="fallback.geometry_entry sha256")
                if self.geometry_entry in self.entries:
                    raise TaskContractError(
                        f"fallback: geometry_entry {self.geometry_entry!r} must not also be listed in entries; the "
                        "projection maps it to the fixed execution-geometry path exactly once"
                    )
        if not isinstance(self.delivered, tuple) or (self.kind == "registered_implementation" and not self.delivered):
            raise TaskContractError("a registered_implementation fallback needs a delivered entry manifest")
        paths = [item.path for item in self.delivered]
        if paths != sorted(paths) or len(set(paths)) != len(paths):
            raise TaskContractError("the fallback delivered entry manifest must be sorted and unique")
        _require_digest(self.domain_plan_digest, where="fallback.domain_plan_digest")

    def identity_document(self) -> dict[str, Any]:
        """@brief fallback_id 的唯一输入。

        @return 字段字典。
        """
        return {
            "fingerprint_kind": "portfolio-domain-fallback",
            "fingerprint_version": 1,
            "domain_plan_digest": self.domain_plan_digest,
            "role": DOMAIN_FALLBACK_ROLE,
            "kind": self.kind,
            "implementation": None
            if self.implementation_id is None
            else {"id": self.implementation_id, "version": int(self.implementation_version)},
            "entries": [item.document() for item in self.delivered],
            "geometry": None
            if self.geometry_entry is None
            else {"origin_path": self.geometry_entry, "origin_sha256": self.geometry_sha256},
            "reasons": list(self.reasons),
        }

    @property
    def fallback_id(self) -> str:
        """@brief 独立 fallback 身份。

        @return 十六进制 sha256。
        """
        return fingerprint(self.identity_document())

    def document(self) -> dict[str, Any]:
        """@brief fallback 的诊断文档。

        @return 字段字典。
        """
        return {
            "role": DOMAIN_FALLBACK_ROLE,
            "kind": self.kind,
            "implementation_id": self.implementation_id,
            "implementation_version": self.implementation_version,
            "entries": list(self.entries),
            "geometry_entry": self.geometry_entry,
            "reasons": list(self.reasons),
            "fallback_id": self.fallback_id,
        }


@dataclass(frozen=True)
class PortfolioCandidate:
    """@brief 外层 portfolio envelope + 未改动的父 bundle entries。"""

    entries: tuple[CandidateEntry, ...]
    manifest_entry: CandidateEntry
    manifest: Mapping[str, Any]
    domain_plan_digest: str
    planner: PlannerIdentity
    buckets: tuple[PortfolioBucket, ...]
    variants: tuple[PortfolioVariant, ...]
    fallback: DomainFallback | None
    schema_id: str = PORTFOLIO_SCHEMA_ID

    def __post_init__(self) -> None:
        """@brief 交叉校验 bucket/variant/fallback/entries 的闭集一致性。

        @raises TaskContractError 非法组合。
        """
        if self.schema_id != PORTFOLIO_SCHEMA_ID:
            raise TaskContractError(f"unsupported portfolio schema {self.schema_id!r}")
        _require_digest(self.domain_plan_digest, where="portfolio.domain_plan_digest")
        if not isinstance(self.manifest_entry, CandidateEntry):
            raise TaskContractError("a portfolio needs its fixed manifest entry")
        if self.manifest_entry.path != PORTFOLIO_ENTRY_PATH:
            raise TaskContractError(f"the portfolio manifest must live at {PORTFOLIO_ENTRY_PATH!r}")
        if self.manifest_entry.role != PORTFOLIO_ENTRY_ROLE:
            raise TaskContractError(
                f"the portfolio manifest entry has role {self.manifest_entry.role!r}; the outer envelope is "
                f"candidate control policy and must declare {PORTFOLIO_ENTRY_ROLE!r} (no registered Track delivers "
                "that role for a source, so the parent can never masquerade as a Track candidate)"
            )
        if self.manifest_entry.media_type != PORTFOLIO_ENTRY_MEDIA_TYPE:
            raise TaskContractError(f"the portfolio manifest media type must be {PORTFOLIO_ENTRY_MEDIA_TYPE!r}")

        by_path: dict[str, CandidateEntry] = {}
        folded: set[str] = set()
        for entry in self.entries:
            if entry.path.casefold() in folded:
                raise TaskContractError(f"the parent portfolio repeats the entry path {entry.path!r}")
            folded.add(entry.path.casefold())
            by_path[entry.path] = entry
        if PORTFOLIO_ENTRY_PATH not in by_path:
            raise TaskContractError("the parent portfolio entries do not contain the manifest entry")
        if by_path[PORTFOLIO_ENTRY_PATH] is not self.manifest_entry:
            raise TaskContractError("the portfolio manifest entry is not the parent bundle's entry")
        for path, entry in by_path.items():
            if path == PORTFOLIO_ENTRY_PATH:
                continue
            if entry.role not in PROJECTABLE_ROLES:
                raise TaskContractError(
                    f"parent entry {entry.path!r} has role {entry.role!r}; a portfolio may carry only "
                    f"{sorted(PROJECTABLE_ROLES)} sources plus its manifest.  A candidate-provided schedule/launch "
                    "entry would be a second geometry declaration"
                )

        bucket_keys: dict[str, PortfolioBucket] = {}
        for bucket in self.buckets:
            if bucket.bucket_key in bucket_keys:
                raise TaskContractError(f"the coverage plan repeats the bucket {bucket.bucket_key!r}")
            bucket_keys[bucket.bucket_key] = bucket
        if not bucket_keys:
            raise TaskContractError("a coverage plan needs at least one bucket")

        variant_keys: set[str] = set()
        geometry_paths: dict[str, str] = {}
        for variant in self.variants:
            if variant.variant_key in variant_keys:
                raise TaskContractError(f"the portfolio repeats the diagnostic variant_key {variant.variant_key!r}")
            variant_keys.add(variant.variant_key)
            bucket = bucket_keys.get(variant.bucket_key)
            if bucket is None:
                raise TaskContractError(f"{variant.variant_key}: unknown bucket {variant.bucket_key!r}")
            if bucket.guard_digest != variant.guard.digest:
                raise TaskContractError(
                    f"{variant.variant_key}: the bucket {bucket.bucket_key!r} declares guard_digest "
                    f"{bucket.guard_digest}, but the variant's canonical guard digest is {variant.guard.digest}"
                )
            for path in variant.entries:
                if path not in by_path:
                    raise TaskContractError(f"{variant.variant_key}: entry {path!r} does not exist in the bundle")
            if variant.geometry_entry is not None:
                _require_geometry_entry(by_path, variant.geometry_entry, where=f"{variant.variant_key}.geometry_entry")
                _require_unique_geometry(
                    geometry_paths, variant.geometry_entry, f"variant {variant.variant_key}", "geometry"
                )
            _require_delivered(variant.delivered, by_path, where=variant.variant_key)
            if {reference.path for reference in variant.delivered} != set(variant.entries):
                raise TaskContractError(
                    f"{variant.variant_key}: the delivered entry manifest must cover exactly its declared entries; "
                    "an omitted entry would not be part of variant_id"
                )
            if variant.domain_plan_digest != self.domain_plan_digest:
                raise TaskContractError(f"{variant.variant_key}: the variant is bound to a different semantic domain")
        if not self.variants:
            raise TaskContractError("a portfolio needs at least one variant")

        if self.fallback is not None:
            if self.fallback.domain_plan_digest != self.domain_plan_digest:
                raise TaskContractError("the domain fallback is bound to a different semantic domain")
            if self.fallback.kind == "registered_implementation":
                for path in self.fallback.entries:
                    if path not in by_path:
                        raise TaskContractError(f"fallback entry {path!r} does not exist in the bundle")
            if self.fallback.geometry_entry is not None:
                _require_geometry_entry(by_path, self.fallback.geometry_entry, where="fallback.geometry_entry")
                _require_unique_geometry(
                    geometry_paths, self.fallback.geometry_entry, "the domain fallback", "geometry"
                )
            _require_delivered(self.fallback.delivered, by_path, where="fallback")
            if {reference.path for reference in self.fallback.delivered} != set(self.fallback.entries):
                raise TaskContractError(
                    "the fallback delivered entry manifest must cover exactly its declared entries"
                )

        # A declared geometry is mapped to host/execution_geometry.json, never copied as a source.
        # This is checked across providers too: one provider may not smuggle another provider's
        # geometry entry into its own entries (it would be delivered as an ordinary Host Tiling).
        declared_geometry = set(geometry_paths)
        for variant in self.variants:
            leaked = sorted(declared_geometry & set(variant.entries))
            if leaked:
                raise TaskContractError(
                    f"{variant.variant_key}: the declared geometry path(s) {leaked} are also listed as source "
                    "entries; a geometry entry is mapped by the projection, not delivered as Host Tiling"
                )
        if self.fallback is not None:
            leaked = sorted(declared_geometry & set(self.fallback.entries))
            if leaked:
                raise TaskContractError(
                    f"the domain fallback lists the declared geometry path(s) {leaked} as source entries; a geometry "
                    "entry is mapped by the projection, not delivered as Host Tiling"
                )

    @property
    def manifest_document(self) -> dict[str, Any]:
        """@brief ``coverage_plan/portfolio.json`` 的原文（已闭集校验）。

        @return 字段字典。
        """
        return dict(self.manifest)

    @property
    def parent_sha256(self) -> str:
        """@brief 父 portfolio bundle 的字节身份（全部父 entries 的 canonical manifest 摘要）。

        This is the parent envelope identity.  It is never the projected candidate identity: the
        projection refuses to reuse it for a variant/fallback.
        """
        manifest = {
            "schema_id": CANDIDATE_MANIFEST_SCHEMA_ID,
            "entries": [entry.document() for entry in sorted(self.entries, key=lambda item: item.path)],
        }
        return hashlib.sha256(canonical_bytes(manifest)).hexdigest()

    def identity_document(self) -> dict[str, Any]:
        """@brief portfolio 语义身份（规范化 envelope，不含父字节摘要）。

        @return 字段字典。
        """
        return {
            "fingerprint_kind": "portfolio-candidate",
            "fingerprint_version": 1,
            "schema_id": self.schema_id,
            "domain_plan_digest": self.domain_plan_digest,
            "planner": self.planner.document(),
            "buckets": [bucket.document() for bucket in self.buckets],
            "variants": [
                {
                    "variant_key": variant.variant_key,
                    "bucket_key": variant.bucket_key,
                    "variant_id": variant.variant_id,
                    "entries": list(variant.entries),
                    "geometry_entry": variant.geometry_entry,
                }
                for variant in self.variants
            ],
            "fallback": None if self.fallback is None else self.fallback.identity_document(),
        }

    @property
    def digest(self) -> str:
        """@brief portfolio 语义摘要。

        @return 十六进制 sha256。
        """
        return fingerprint(self.identity_document())

    @property
    def variant_ids(self) -> tuple[str, ...]:
        """@brief 全部 variant 的稳定身份（声明顺序）。

        @return 身份元组。
        """
        return tuple(variant.variant_id for variant in self.variants)

    @property
    def fallback_id(self) -> str | None:
        """@brief fallback 身份（没有 fallback 时为 None）。

        @return 十六进制 sha256 或 None。
        """
        return None if self.fallback is None else self.fallback.fallback_id

    def variant_for_provider(self, identity: str) -> PortfolioVariant | None:
        """@brief 按稳定身份找 variant。

        @param identity variant_id。
        @return variant 或 None。
        """
        for variant in self.variants:
            if variant.variant_id == identity:
                return variant
        return None

    def entry(self, path: str) -> CandidateEntry:
        """@brief 取父 bundle 的 entry。

        @param path 相对路径。
        @return entry。
        @raises KeyError 不存在。
        """
        for entry in self.entries:
            if entry.path == path:
                return entry
        raise KeyError(path)

    def document(self) -> dict[str, Any]:
        """@brief portfolio 的证据投影（含父字节身份与全部声明身份）。

        @return 字段字典。
        """
        return {
            "schema_id": self.schema_id,
            "domain_plan_digest": self.domain_plan_digest,
            "planner": self.planner.document(),
            "buckets": [bucket.document() for bucket in self.buckets],
            "variants": [variant.document() for variant in self.variants],
            "fallback": None if self.fallback is None else self.fallback.document(),
            "parent_sha256": self.parent_sha256,
            "portfolio_digest": self.digest,
        }

    def registration_report(
        self,
        domain_plan: SemanticDomainPlan,
        instances: Iterable[SemanticInstanceV2],
        *,
        spec: OpSpec,
        target: TargetProfile,
        max_buffer_length: int | None = None,
    ) -> "RegistrationReport":
        """@brief 有限域注册期检查：枚举全部域点，报告 overlap/空 variant/不可判定 guard。

        This is a **finite point-set check** only.  It never claims static mutual exclusion for a
        non-finite domain and it never replaces runtime guard evaluation: a candidate can still be
        resolved at runtime only through :class:`ExactDispatchResolver`.

        @param domain_plan 该 portfolio 绑定的语义域计划。
        @param instances 显式命名的实例（必须与域计划逐点一致）。
        @param spec 权威 OpSpec（用于重算域计划，拒绝伪造 opspec/ABI 身份）。
        @param target 权威目标档案。
        @param max_buffer_length 设备缓冲长度。
        @return 注册报告。
        @raises TaskContractError 域计划/实例不是权威派生。
        """
        if domain_plan.digest != self.domain_plan_digest:
            raise TaskContractError(
                "the portfolio is bound to a different semantic domain plan; a registration check may not borrow "
                "another domain"
            )
        instances = tuple(instances)
        resolved: dict[tuple[str, str], SemanticInstanceV2] = {}
        for instance in instances:
            if not isinstance(instance, SemanticInstanceV2):
                raise TaskContractError("a registration check needs v2 semantic instances")
            key = (instance.dtype, instance.size)
            if key in resolved:
                raise TaskContractError(f"the registration check repeats the instance {key}")
            resolved[key] = instance
        points: list[RegistrationPoint] = []
        overlaps: list[str] = []
        unknown: list[str] = []
        matched_ids: set[str] = set()
        for plan_instance in domain_plan.instances:
            instance = resolved.get((plan_instance.dtype, plan_instance.size))
            if instance is None:
                raise TaskContractError(
                    f"{plan_instance.instance_key}: the registration check is missing the named instance"
                )
            if semantic_instance_digest(instance) != plan_instance.semantic_instance_digest:
                raise TaskContractError(
                    f"{plan_instance.instance_key}: the supplied instance does not match the domain plan's semantics"
                )
            names = instance_names(instance, where=plan_instance.instance_key)
            results: list[tuple[str, str]] = []
            matched: list[str] = []
            for variant in self.variants:
                try:
                    hit = variant.guard.evaluate(names, where=variant.variant_key)
                except GeometryError as exc:
                    results.append((variant.variant_id, "unknown"))
                    unknown.append(f"{plan_instance.instance_key}:{variant.variant_key}: {exc}")
                    continue
                results.append((variant.variant_id, "true" if hit else "false"))
                if hit:
                    matched.append(variant.variant_id)
                    matched_ids.add(variant.variant_id)
            if len(matched) > 1:
                overlaps.append(plan_instance.instance_key)
            points.append(
                RegistrationPoint(
                    instance_key=plan_instance.instance_key,
                    matched=tuple(matched),
                    unknown=tuple(identity for identity, outcome in results if outcome == "unknown"),
                )
            )
        empty = tuple(variant.variant_id for variant in self.variants if variant.variant_id not in matched_ids)
        from .domain import require_authoritative_domain_plan

        require_authoritative_domain_plan(
            domain_plan,
            instances,
            spec=spec,
            target=target,
            max_buffer_length=max_buffer_length,
            where="registration check",
        )
        return RegistrationReport(
            domain_plan_digest=domain_plan.digest,
            portfolio_digest=self.digest,
            points=tuple(points),
            overlaps=tuple(overlaps),
            empty_variants=empty,
            unknown_guards=tuple(unknown),
        )


def _require_unique_geometry(seen: dict[str, str], path: str, who: str, what: str) -> None:
    """@brief 保证同一 geometry entry 只被一个 provider 声明。

    @param seen 已声明表。
    @param path 路径。
    @param who 声明者。
    @param what 诊断类别。
    @raises TaskContractError 重复/伪造 geometry 声明。
    """
    if path in seen:
        raise TaskContractError(
            f"duplicate {what} declaration: {path!r} is declared by {seen[path]} and {who}; each provider needs its "
            "own execution geometry"
        )
    seen[path] = who


def _require_geometry_entry(by_path: Mapping[str, CandidateEntry], path: str, *, where: str) -> None:
    """@brief 校验几何声明 entry 存在且是 Host Tiling JSON。

    @param by_path 父 entries。
    @param path 几何路径。
    @param where 诊断位置。
    @raises TaskContractError 缺失或角色/媒体类型非法。
    """
    entry = by_path.get(path)
    if entry is None:
        raise TaskContractError(f"{where}: geometry entry {path!r} does not exist in the bundle")
    if entry.role != GEOMETRY_ENTRY_ROLE:
        raise TaskContractError(
            f"{where}: geometry entry {path!r} has role {entry.role!r}; a declared geometry is candidate Host Tiling "
            f"and must use {GEOMETRY_ENTRY_ROLE!r}"
        )
    if entry.media_type != GEOMETRY_MEDIA_TYPE:
        raise TaskContractError(
            f"{where}: geometry entry {path!r} has media type {entry.media_type!r}; expected {GEOMETRY_MEDIA_TYPE!r}"
        )


def _require_delivered(delivered: Sequence[EntryReference], by_path: Mapping[str, CandidateEntry], *, where: str) -> None:
    """@brief 校验 delivered 清单就是父 bundle 的同一字节。

    @param delivered 声明清单。
    @param by_path 父 entries。
    @param where 诊断位置。
    @raises TaskContractError 漂移。
    """
    for reference in delivered:
        entry = by_path.get(reference.path)
        if entry is None:
            raise TaskContractError(f"{where}: delivered entry {reference.path!r} does not exist in the bundle")
        if not reference.matches(entry):
            raise TaskContractError(
                f"{where}: delivered entry {reference.path!r} does not carry the bundle's path/role/media_type/bytes"
            )


def _geometry_from_entry(entry: CandidateEntry, *, where: str) -> tuple[ExecutionGeometry, str]:
    """@brief 读取一个几何声明 entry 为 typed ``ExecutionGeometry``。

    @param entry 几何 entry。
    @param where 诊断位置。
    @return ``(geometry, sha256)``。
    @raises GeometryError 文档不是候选执行的几何。
    """
    try:
        text = bytes(entry.data).decode("utf-8")
    except UnicodeDecodeError as exc:
        raise GeometryError(f"{where}: the geometry entry is not UTF-8 JSON: {exc}") from exc
    try:
        document = loads_strict(text, what=where)
    except SchemaError as exc:
        raise GeometryError(f"{where}: the geometry entry is not strict JSON: {exc}") from exc
    geometry = parse_candidate_geometry(document, where=where)
    return geometry, _hash_bytes(entry.data)


def _entry_references(paths: Sequence[str], by_path: Mapping[str, CandidateEntry], *, where: str) -> tuple[EntryReference, ...]:
    """@brief 由路径清单构造排序后的 entry 引用。

    @param paths 路径。
    @param by_path 父 entries。
    @param where 诊断位置。
    @return 引用元组。
    @raises TaskContractError 路径不存在。
    """
    references = []
    for path in paths:
        entry = by_path.get(path)
        if entry is None:
            raise TaskContractError(f"{where}: entry {path!r} does not exist in the parent bundle")
        references.append(EntryReference.from_entry(entry))
    return tuple(sorted(references, key=lambda item: item.path))


def _require_role_for_projection(entry: CandidateEntry, *, where: str) -> None:
    """@brief 投影源 entry 只能是 device_kernel/host_tiling。

    @param entry entry。
    @param where 诊断位置。
    @raises TaskContractError 非法角色。
    """
    if entry.role not in PROJECTABLE_ROLES:
        raise TaskContractError(
            f"{where}: entry {entry.path!r} has role {entry.role!r}; a portfolio source is device_kernel or "
            "host_tiling, never a second schedule/launch declaration"
        )


def parse_portfolio(
    entries: Iterable[CandidateEntry],
    *,
    domain_plan: SemanticDomainPlan | None = None,
    where: str = "coverage_plan/portfolio.json",
) -> PortfolioCandidate:
    """@brief 解析外层 portfolio envelope（在任何 Track 校验之前）。

    @param entries 父 bundle 的 entries（字节是权威）。
    @param domain_plan 可选：要求 envelope 绑定的域计划摘要与它一致。
    @param where 诊断位置。
    @return 已交叉校验的 portfolio。
    @raises TaskContractError 文档/字段/引用非法。
    @raises GeometryError 几何声明非法。
    """
    entry_list = tuple(entries)
    if not entry_list:
        raise TaskContractError("a portfolio needs at least one entry")
    by_path: dict[str, CandidateEntry] = {}
    folded: set[str] = set()
    for entry in entry_list:
        if not isinstance(entry, CandidateEntry):
            raise TaskContractError("parse_portfolio needs CandidateEntry values")
        if entry.path.casefold() in folded:
            raise TaskContractError(f"the parent bundle repeats the entry path {entry.path!r}")
        folded.add(entry.path.casefold())
        by_path[entry.path] = entry
    manifest_entry = by_path.get(PORTFOLIO_ENTRY_PATH)
    if manifest_entry is None:
        raise TaskContractError(f"a portfolio requires the fixed entry {PORTFOLIO_ENTRY_PATH!r}")
    try:
        text = bytes(manifest_entry.data).decode("utf-8")
    except UnicodeDecodeError as exc:
        raise TaskContractError(f"{where}: the portfolio manifest is not UTF-8 JSON: {exc}") from exc
    try:
        document = loads_strict(text, what=where)
    except SchemaError as exc:
        raise TaskContractError(f"{where}: the portfolio manifest is not strict JSON: {exc}") from exc
    try:
        validate(document, _schema(), path=where)
    except SchemaFailure as exc:
        raise TaskContractError(f"{where}: {exc}") from exc

    planner = PlannerIdentity(
        planner_id=str(document["planner"]["id"]), planner_version=int(document["planner"]["version"])
    )
    domain_plan_digest = _require_digest(document["domain_plan_digest"], where=f"{where}.domain_plan_digest")
    if domain_plan is not None and domain_plan.digest != domain_plan_digest:
        raise TaskContractError(
            f"{where}: the portfolio is bound to the domain plan {domain_plan_digest}, not the supplied "
            f"{domain_plan.digest}"
        )
    coverage = document["coverage_plan"]
    buckets = tuple(
        PortfolioBucket(bucket_key=str(item["bucket_key"]), guard_digest=str(item["guard_digest"]))
        for item in coverage["buckets"]
    )
    variants: list[PortfolioVariant] = []
    for index, item in enumerate(document["variants"]):
        variant_where = f"{where}.variants[{index}]"
        guard = parse_guard(item["guard"], where=f"{variant_where}.guard")
        own = tuple(_relative_path(path, where=f"{variant_where}.entries") for path in item["entries"])
        geometry_entry = item.get("geometry_entry")
        geometry: ExecutionGeometry | None = None
        geometry_sha256: str | None = None
        if geometry_entry is not None:
            geometry_entry = _relative_path(geometry_entry, where=f"{variant_where}.geometry_entry")
            entry = by_path.get(geometry_entry)
            if entry is None:
                raise TaskContractError(f"{variant_where}.geometry_entry: {geometry_entry!r} does not exist")
            _require_geometry_entry(by_path, geometry_entry, where=f"{variant_where}.geometry_entry")
            geometry, geometry_sha256 = _geometry_from_entry(entry, where=f"{variant_where}.geometry_entry")
        for path in own:
            entry = by_path.get(path)
            if entry is None:
                raise TaskContractError(f"{variant_where}.entries: {path!r} does not exist in the parent bundle")
            _require_role_for_projection(entry, where=f"{variant_where}.entries")
        delivered = _entry_references(own, by_path, where=variant_where)
        variants.append(
            PortfolioVariant(
                variant_key=str(item["variant_key"]),
                bucket_key=str(item["bucket_key"]),
                guard=guard,
                entries=own,
                geometry_entry=geometry_entry,
                geometry=geometry,
                geometry_sha256=geometry_sha256,
                delivered=delivered,
                domain_plan_digest=domain_plan_digest,
                planner=planner,
            )
        )

    fallback: DomainFallback | None = None
    fallback_item = document.get("fallback")
    if fallback_item is not None:
        kind = str(fallback_item["kind"])
        own = tuple(_relative_path(path, where=f"{where}.fallback.entries") for path in fallback_item["entries"])
        geometry_entry = fallback_item.get("geometry_entry")
        geometry = None
        geometry_sha256 = None
        if geometry_entry is not None:
            geometry_entry = _relative_path(geometry_entry, where=f"{where}.fallback.geometry_entry")
            entry = by_path.get(geometry_entry)
            if entry is None:
                raise TaskContractError(f"{where}.fallback.geometry_entry: {geometry_entry!r} does not exist")
            _require_geometry_entry(by_path, geometry_entry, where=f"{where}.fallback.geometry_entry")
            geometry, geometry_sha256 = _geometry_from_entry(entry, where=f"{where}.fallback.geometry_entry")
        for path in own:
            entry = by_path.get(path)
            if entry is None:
                raise TaskContractError(f"{where}.fallback.entries: {path!r} does not exist in the parent bundle")
            _require_role_for_projection(entry, where=f"{where}.fallback.entries")
        delivered = _entry_references(own, by_path, where=f"{where}.fallback")
        fallback = DomainFallback(
            kind=kind,
            implementation_id=fallback_item.get("implementation_id"),
            implementation_version=fallback_item.get("implementation_version"),
            entries=own,
            geometry_entry=geometry_entry,
            geometry=geometry,
            geometry_sha256=geometry_sha256,
            reasons=tuple(str(reason) for reason in fallback_item["reasons"]),
            delivered=delivered,
            domain_plan_digest=domain_plan_digest,
        )

    return PortfolioCandidate(
        entries=entry_list,
        manifest_entry=manifest_entry,
        manifest=document,
        domain_plan_digest=domain_plan_digest,
        planner=planner,
        buckets=buckets,
        variants=tuple(variants),
        fallback=fallback,
        schema_id=str(document["schema_id"]),
    )


@dataclass(frozen=True)
class RegistrationPoint:
    """@brief 一个有限域点的注册期检查结果。"""

    instance_key: str
    matched: tuple[str, ...]
    unknown: tuple[str, ...]

    def document(self) -> dict[str, Any]:
        """@brief 该点的 JSON 文档。

        @return 字段字典。
        """
        return {"instance_key": self.instance_key, "matched": list(self.matched), "unknown": list(self.unknown)}


@dataclass(frozen=True)
class RegistrationReport:
    """@brief 有限点集注册报告（不是全域互斥证明，也不产生 active 证据）。"""

    domain_plan_digest: str
    portfolio_digest: str
    points: tuple[RegistrationPoint, ...]
    overlaps: tuple[str, ...]
    empty_variants: tuple[str, ...]
    unknown_guards: tuple[str, ...]

    @property
    def ok(self) -> bool:
        """@brief 有限域点上无 overlap、无空 variant、无不可判定 guard。

        @return 检查通过时为 True。
        """
        return not self.overlaps and not self.empty_variants and not self.unknown_guards

    def document(self) -> dict[str, Any]:
        """@brief 报告的 JSON 文档。

        @return 字段字典。
        """
        return {
            "domain_plan_digest": self.domain_plan_digest,
            "portfolio_digest": self.portfolio_digest,
            "points": [point.document() for point in self.points],
            "overlaps": list(self.overlaps),
            "empty_variants": list(self.empty_variants),
            "unknown_guards": list(self.unknown_guards),
            "ok": self.ok,
            "coverage_kind": "exhaustive_finite_point_check",
        }


@dataclass(frozen=True)
class EvaluationBinding:
    """@brief 投影候选的构建/评测前身份（含原几何路径与内容摘要）。

    P0 can bind the provider identity, the projected candidate identity, the parent identity and
    the geometry declaration.  Bundle/artifact/native/evaluation identities are added by the
    runner (P1) and the confirmation sidecars (P2); none of them is invented here.
    """

    provider: str
    variant_id: str | None
    fallback_id: str | None
    domain_plan_digest: str
    portfolio_digest: str
    parent_portfolio_sha256: str
    candidate_sha256: str
    track_id: str
    geometry_origin_path: str | None
    geometry_origin_sha256: str | None
    geometry_path: str | None
    entries: tuple[EntryReference, ...]

    def __post_init__(self) -> None:
        """@brief 校验身份完整性与父 SHA 不复用。

        @raises TaskContractError 非法。
        """
        if self.provider not in ("candidate", "fallback"):
            raise TaskContractError(f"unknown provider {self.provider!r}")
        for name in ("domain_plan_digest", "portfolio_digest", "parent_portfolio_sha256", "candidate_sha256"):
            _require_digest(getattr(self, name), where=f"binding.{name}")
        if (self.variant_id is None) == (self.fallback_id is None):
            raise TaskContractError("a binding names exactly one of variant_id/fallback_id")
        if self.provider == "candidate" and self.variant_id is None:
            raise TaskContractError("a candidate binding needs the variant identity")
        if self.provider == "fallback" and self.fallback_id is None:
            raise TaskContractError("a fallback binding needs the fallback identity")
        if self.candidate_sha256 == self.parent_portfolio_sha256:
            raise TaskContractError(
                "the projected candidate identity is the parent portfolio identity; a projection must not reuse the "
                "parent SHA"
            )
        if self.track_id not in (SINGLE_TRACK_ID, BUNDLE_TRACK_ID):
            raise TaskContractError(f"unknown projected track {self.track_id!r}")
        if (self.geometry_path is None) != (self.geometry_origin_path is None) or (self.geometry_path is None) != (
            self.geometry_origin_sha256 is None
        ):
            raise TaskContractError("the geometry origin path/SHA and the projected path must be all present or all absent")
        if self.geometry_path is not None:
            if self.geometry_path != PROJECTED_GEOMETRY_PATH:
                raise TaskContractError(
                    f"the projected geometry path is {self.geometry_path!r}; the fixed path is "
                    f"{PROJECTED_GEOMETRY_PATH!r}"
                )
            _require_digest(self.geometry_origin_sha256, where="binding.geometry_origin_sha256")
        if not isinstance(self.entries, tuple) or not self.entries:
            raise TaskContractError("a binding needs the projected entry manifest")

    def document(self) -> dict[str, Any]:
        """@brief 绑定的 JSON 文档。

        @return 字段字典。
        """
        return {
            "provider": self.provider,
            "variant_id": self.variant_id,
            "fallback_id": self.fallback_id,
            "domain_plan_digest": self.domain_plan_digest,
            "portfolio_digest": self.portfolio_digest,
            "parent_portfolio_sha256": self.parent_portfolio_sha256,
            "candidate_sha256": self.candidate_sha256,
            "track_id": self.track_id,
            "geometry_origin_path": self.geometry_origin_path,
            "geometry_origin_sha256": self.geometry_origin_sha256,
            "geometry_path": self.geometry_path,
            "entries": [item.document() for item in self.entries],
        }

    @property
    def identity(self) -> str:
        """@brief 绑定身份。

        @return 十六进制 sha256。
        """
        return fingerprint(self.document())


@dataclass(frozen=True)
class CandidateProjection:
    """@brief 一个已投影的普通 CandidateArtifact（可交给唯一 ``evaluate_task``）。"""

    provider: str
    label: str
    identity: str
    track_id: str
    candidate: CandidateArtifact
    geometry: ExecutionGeometry | None
    binding: EvaluationBinding

    def __post_init__(self) -> None:
        """@brief 校验投影与绑定一致。

        @raises TaskContractError 非法。
        """
        if self.candidate.track_id != self.track_id:
            raise TaskContractError("the projected candidate and the projection track disagree")
        if self.binding.candidate_sha256 != self.candidate.candidate_sha256:
            raise TaskContractError("the binding does not name the projected candidate identity")
        if self.binding.provider != self.provider:
            raise TaskContractError("the binding and the projection disagree on the provider")

    def document(self) -> dict[str, Any]:
        """@brief 投影的证据投影。

        @return 字段字典。
        """
        return {
            "provider": self.provider,
            "label": self.label,
            "identity": self.identity,
            "track_id": self.track_id,
            "candidate_sha256": self.candidate.candidate_sha256,
            "candidate_manifest": self.candidate.manifest(),
            "binding": self.binding.document(),
        }


def _require_single_device(entries: Sequence[CandidateEntry], *, where: str) -> CandidateEntry:
    """@brief 单文件交付必须正好一个 device entry。

    @param entries 投影 entries（不含几何）。
    @param where 诊断位置。
    @return 该 device entry。
    @raises TaskContractError 非法。
    """
    if len(entries) != 1 or entries[0].role != "device_kernel":
        raise TaskContractError(
            f"{where}: a variant/fallback without Host Tiling delivers exactly one self-contained device entry; got "
            f"{[(entry.path, entry.role) for entry in entries]}.  A multi-source candidate needs a legal Host Tiling "
            "plan and a declared workspace, not a second entry crammed into msl-single"
        )
    return entries[0]


def _require_bundle_shape(entries: Sequence[CandidateEntry], *, where: str, workspace_declared: bool) -> None:
    """@brief msl-bundle 的交付形状：一个 host entry、至少一个 device、声明 workspace。

    @param entries 投影 entries（含映射后几何）。
    @param where 诊断位置。
    @param workspace_declared 权威 OpSpec 是否显式声明 workspace（由 :class:`DomainAuthority` 派生）。
    @raises TaskContractError 非法。
    """
    host = [entry for entry in entries if entry.role == "host_tiling"]
    device = [entry for entry in entries if entry.role == "device_kernel"]
    if len(host) != 1:
        raise TaskContractError(
            f"{where}: an msl-bundle projection needs exactly one host_tiling entry, got {len(host)}; Host Tiling is "
            "candidate-owned work, not a second geometry file"
        )
    if not device:
        raise TaskContractError(f"{where}: an msl-bundle projection needs at least one device_kernel entry")
    if not workspace_declared:
        raise TaskContractError(
            f"{where}: the variant/fallback declares Host Tiling but the OpSpec declares no workspace tensor; an "
            "msl-bundle is refused instead of inventing a workspace or dropping Host Tiling"
        )


def _require_includes_resolved(entries: Sequence[CandidateEntry], *, where: str) -> None:
    """@brief 每个 ``#include "x"`` 必须是投影内的相对 entry（不得逃逸或漏拷）。

    A candidate's MSL sources keep their bytes, so the existing v2 source screen applies
    unchanged.  That screen allows only ``<metal_...>`` headers: a quoted include and an
    angle-bracket host path are both refused here, at the projection boundary, instead of
    producing a candidate that the existing screen would reject at the first build.

    @param entries 投影 entries。
    @param where 诊断位置。
    @raises TaskContractError include 非法或依赖不在投影内。
    """
    paths = {entry.path for entry in entries}
    for entry in entries:
        if entry.role != "device_kernel":
            continue
        try:
            text = bytes(entry.data).decode("utf-8")
        except UnicodeDecodeError as exc:
            raise TaskContractError(f"{where}: device source {entry.path!r} is not UTF-8 text: {exc}") from exc
        for match in _INCLUDE_LINE.finditer(text):
            target = match.group(1).strip()
            quoted = _QUOTED_TARGET.match(target)
            if quoted is not None:
                raise TaskContractError(
                    f"{where}: device source {entry.path!r} has a quoted #include {quoted.group(1)!r}; the v2 "
                    "source screen allows only Metal standard headers `<metal_...>`, so a quoted include is "
                    "refused at the projection boundary"
                )
            if _METAL_STANDARD_TARGET.match(target) is not None:
                continue
            raise TaskContractError(
                f"{where}: device source {entry.path!r} has `#include {target}`, which is not a Metal standard "
                "header `<metal_...>`; host paths and non-Metal headers are refused at the projection boundary"
            )


def _require_launch_retained(entries: Sequence[CandidateEntry], *, single: bool, where: str) -> None:
    """@brief 投影必须保留候选自己的唯一 ``/* taic-launch */`` 块（单文件就在该源里）。

    The structural check is the backend's own launch scan (``unique_launch_block``), not a
    substring search: a line comment, a string literal or a second block is rejected here, so a
    projector cannot turn a fake marker into a launch.

    @param entries 投影 entries（不含几何）。
    @param single 是否 msl-single。
    @param where 诊断位置。
    @raises TaskContractError 缺失。
    """
    device = [entry for entry in entries if entry.role == "device_kernel"]
    carriers: list[CandidateEntry] = []
    for entry in device:
        try:
            text = bytes(entry.data).decode("utf-8")
        except UnicodeDecodeError as exc:
            raise TaskContractError(f"{where}: device source {entry.path!r} is not UTF-8 text: {exc}") from exc
        try:
            block = unique_launch_block(text)
        except LaunchRejected as exc:
            raise TaskContractError(
                f"{where}: device source {entry.path!r} has an invalid {LAUNCH_MARKER!r} marker: {exc}"
            ) from exc
        if block is not None:
            carriers.append(entry)
    if single:
        if device[0].path not in {entry.path for entry in carriers}:
            raise TaskContractError(
                f"{where}: the msl-single source has no {LAUNCH_MARKER!r} block; the harness does not invent a launch"
            )
        return
    if not carriers:
        raise TaskContractError(
            f"{where}: none of the projected device sources keeps a {LAUNCH_MARKER!r} block; the Metal executor "
            "cross-checks workspace/launch from it"
        )


def project_provider(
    portfolio: PortfolioCandidate,
    provider: PortfolioVariant | DomainFallback,
    *,
    authority: DomainAuthority,
    track_registry: TrackRegistry | None = None,
) -> CandidateProjection:
    """@brief 把一个 variant/fallback 机械投影成普通 ``CandidateArtifact``。

    The projection copies candidate bytes verbatim, maps exactly one declared geometry entry to
    ``host/execution_geometry.json`` and keeps every other entry's path/role/media_type/bytes.  It
    chooses ``msl-single``/``msl-bundle`` from the delivered content only, then validates the child
    against the registered TrackSpec (the *child* is a normal candidate; the parent never is).

    @param portfolio 父 portfolio。
    @param provider variant 或 domain fallback。
    @param authority 与 portfolio/domain 绑定的权威 OpSpec/target/instances；workspace 由它
        自行判定，调用方不能再用一个布尔声明来伪造 ``msl-bundle`` 的前提。
    @param track_registry 可选 registry；缺省用已登记的默认 registry（只读）。
    @return 投影。
    @raises TaskContractError/GeometryError/UnsupportedContractCapability 投影不合法。
    """
    if not isinstance(authority, DomainAuthority):
        raise TaskContractError("project_provider needs the DomainAuthority of this portfolio's domain")
    if portfolio.domain_plan_digest != authority.plan.digest:
        raise TaskContractError(
            "the projection authority names a different semantic domain plan than the portfolio; a projection may "
            "not borrow another domain's OpSpec/workspace facts"
        )
    if isinstance(provider, PortfolioVariant):
        if portfolio.variant_for_provider(provider.variant_id) is None:
            raise TaskContractError(
                f"{provider.variant_key}: the variant does not belong to this portfolio; a projection may not borrow "
                "another portfolio's provider"
            )
        provider_kind = "candidate"
        label = provider.variant_key
        identity = provider.variant_id
        variant_id: str | None = provider.variant_id
        fallback_id: str | None = None
        own = provider.entries
        geometry_entry = provider.geometry_entry
        geometry = provider.geometry
        geometry_sha256 = provider.geometry_sha256
    elif isinstance(provider, DomainFallback):
        if portfolio.fallback is None or portfolio.fallback.fallback_id != provider.fallback_id:
            raise TaskContractError(
                "the domain fallback does not belong to this portfolio; a projection may not borrow another "
                "portfolio's fallback"
            )
        if provider.kind == "refuse":
            raise TaskContractError(
                "a refusing domain fallback is not projected: it is the declared absence of a provider, so the "
                "resolver reports dispatch_unmatched instead"
            )
        provider_kind = "fallback"
        label = str(provider.implementation_id)
        identity = provider.fallback_id
        variant_id = None
        fallback_id = provider.fallback_id
        own = provider.entries
        geometry_entry = provider.geometry_entry
        geometry = provider.geometry
        geometry_sha256 = provider.geometry_sha256
    else:
        raise TaskContractError("project_provider needs a PortfolioVariant or a DomainFallback")

    projected: list[CandidateEntry] = []
    paths: set[str] = set()
    for path in own:
        entry = portfolio.entry(path)
        if entry.path in paths:  # pragma: no cover - parse_portfolio closes this
            raise TaskContractError(f"{label}: the projection repeats the entry {entry.path!r}")
        paths.add(entry.path)
        projected.append(entry)
    if geometry_entry is not None:
        assert geometry is not None and geometry_sha256 is not None
        source = portfolio.entry(geometry_entry)
        if PROJECTED_GEOMETRY_PATH in paths:
            raise TaskContractError(
                f"{label}: the projection would overwrite {PROJECTED_GEOMETRY_PATH!r}; the fixed geometry path is "
                "written by the projection only"
            )
        paths.add(PROJECTED_GEOMETRY_PATH)
        projected.append(
            CandidateEntry(
                path=PROJECTED_GEOMETRY_PATH,
                role=PROJECTED_GEOMETRY_ROLE,
                media_type=GEOMETRY_MEDIA_TYPE,
                data=bytes(source.data),
            )
        )
    projected.sort(key=lambda entry: entry.path)
    if not projected:  # pragma: no cover - parse closes this
        raise TaskContractError(f"{label}: the projection produced no entries")

    if geometry_entry is None:
        _require_single_device(projected, where=label)
        _require_launch_retained(projected, single=True, where=label)
        track_id = SINGLE_TRACK_ID
    else:
        _require_bundle_shape(projected, where=label, workspace_declared=authority.workspace_declared)
        _require_launch_retained(projected, single=False, where=label)
        track_id = BUNDLE_TRACK_ID
    _require_includes_resolved(projected, where=label)

    candidate = CandidateArtifact(track_id=track_id, entries=tuple(projected))
    if track_id == BUNDLE_TRACK_ID:
        # A host_tiling role is not enough: the closed parser for the declared representation must
        # accept the plan.  JSON linear uses the exact-fields parser; a restricted C99 program uses
        # the neutral static precheck (the real Clang call happens in the bounded worker's build).
        from ..hostprogram.contract import host_contract_spec
        from ..hostprogram.delivery import host_tiling_entry, preflight_host_program

        if authority.contract is None:
            raise TaskContractError(
                f"{label}: an msl-bundle projection needs the authoritative TaskContract to validate Host Tiling"
            )
        entry = host_tiling_entry(candidate)
        if entry is None:  # pragma: no cover
            raise TaskContractError(f"{label}: the projected bundle has no Host Tiling plan")
        if entry.kind == "program":
            spec = host_contract_spec(instance=authority.contract.instance)
            declared = {tensor.name for tensor in authority.contract.instance.tensors if tensor.space == "workspace"}
            problems = preflight_host_program(
                entry.source, spec=spec, declared_workspaces=declared, where=entry.path
            )
            if problems:
                raise TaskContractError(f"{label}: Host Program preflight failed: " + "; ".join(problems[:4]))
        else:
            from ..backends.metal.v2_runtime import HostTilingPlan

            if HostTilingPlan.from_candidate(candidate, authority.contract) is None:  # pragma: no cover
                raise TaskContractError(f"{label}: the projected bundle has no Host Tiling plan")
    if candidate.candidate_sha256 == portfolio.parent_sha256:
        raise TaskContractError(
            f"{label}: the projection produced the parent portfolio identity; a child candidate is a new subject"
        )
    registry = load_track_registry() if track_registry is None else track_registry
    candidate.validate_for(registry.get(track_id))

    binding = EvaluationBinding(
        provider=provider_kind,
        variant_id=variant_id,
        fallback_id=fallback_id,
        domain_plan_digest=portfolio.domain_plan_digest,
        portfolio_digest=portfolio.digest,
        parent_portfolio_sha256=portfolio.parent_sha256,
        candidate_sha256=candidate.candidate_sha256,
        track_id=track_id,
        geometry_origin_path=geometry_entry,
        geometry_origin_sha256=geometry_sha256,
        geometry_path=None if geometry_entry is None else PROJECTED_GEOMETRY_PATH,
        entries=tuple(EntryReference.from_entry(entry) for entry in projected),
    )
    return CandidateProjection(
        provider=provider_kind,
        label=label,
        identity=identity,
        track_id=track_id,
        candidate=candidate,
        geometry=geometry,
        binding=binding,
    )


def instance_names(instance: SemanticInstanceV2, *, where: str) -> dict[str, int]:
    """@brief guard 的名字空间：符号 + 整数 attr（与 A2 几何求值同名）。

    A name may not be both a symbol and an attr: silently letting one overwrite the other would
    make the same guard mean two things.  This is the one helper the registration check and the
    resolver both use, so the name space cannot drift between them.

    @param instance 语义实例。
    @param where 诊断位置。
    @return 名字到整数。
    @raises TaskContractError 符号与属性重名。
    """
    names: dict[str, int] = {}
    for name, value in instance.symbols.items():
        names[str(name)] = int(value)
    for name, value in instance.resolved_attrs.items():
        if name in names:
            raise TaskContractError(f"{where}: name {name!r} is both a symbol and an attr; a guard cannot resolve it")
        if isinstance(value, bool) or not isinstance(value, int):
            continue
        names[str(name)] = int(value)
    if not names:
        raise TaskContractError(f"{where}: the instance declares no symbol or integer attr; no guard can be evaluated")
    return names


__all__ = [
    "BUNDLE_TRACK_ID",
    "CandidateProjection",
    "DOMAIN_FALLBACK_ROLE",
    "DomainFallback",
    "EVIDENCE_STATUSES",
    "EntryReference",
    "EvaluationBinding",
    "FALLBACK_KINDS",
    "FALLBACK_REASONS",
    "GEOMETRY_ENTRY_ROLE",
    "GEOMETRY_MEDIA_TYPE",
    "LAUNCH_MARKER",
    "PORTFOLIO_ENTRY_MEDIA_TYPE",
    "PORTFOLIO_ENTRY_PATH",
    "PORTFOLIO_ENTRY_ROLE",
    "PORTFOLIO_SCHEMA_ID",
    "PORTFOLIO_SCHEMA_PATH",
    "PROJECTABLE_ROLES",
    "PROJECTED_GEOMETRY_PATH",
    "PROJECTED_GEOMETRY_ROLE",
    "PlannerIdentity",
    "PortfolioBucket",
    "PortfolioCandidate",
    "PortfolioVariant",
    "RegistrationPoint",
    "RegistrationReport",
    "SINGLE_TRACK_ID",
    "TIMING_BASELINE_ROLE",
    "instance_names",
    "parse_portfolio",
    "project_provider",
]
