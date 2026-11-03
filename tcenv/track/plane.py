"""Frozen A1 interfaces of the candidate/track delivery plane.

A1 freezes four types and one renderer contract; it does **not** derive the target ABI,
parse/evaluate geometry or build a candidate.  Those are A2 (``TensorMemoryContract``
derivation, geometry validation, build adapters) and A3 (evidence/digest integration).

The boundary is the plan's division of responsibilities:

* :class:`TensorMemoryContract` locks ABI, memory and numeric constraints only.  It has no
  tile/parallel constant and no fallback geometry.
* :class:`ExecutionGeometry` is candidate-owned.  ``source`` is the interface-level
  distinction between the candidate geometry and the independent ``fallback`` geometry.
  Expressions are frozen here as JSON terms; A2 owns typed parsing/validation/evaluation
  and the rule that a fallback is never embedded in a candidate's geometry.
* :class:`CandidateArtifact` is a pure byte manifest.  It has no author, committee or
  session field, and its identity is the single-file SHA-256 (``msl-single``, matching the
  existing ``candidate_sha256``) or the canonical candidate-entry manifest digest
  (``msl-bundle``).  Host Tiling is a candidate entry, never a track property.
* :class:`TrackRenderer` formats declared facts.  It never receives a tiling algorithm and
  refuses to render without an ABI contract, so an A1-only checkout cannot pretend to have
  A2's ABI.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any, Mapping

from ..dtypes import get_dtype
from ..identity import IdentityError
from ..task_presentation import TaskPresentation, render_statement
from ..contract_errors import TaskContractError, UnsupportedContractCapability
from .registry import TrackSpec

TENSOR_MEMORY_SCHEMA_VERSION = 1
EXECUTION_GEOMETRY_SCHEMA_VERSION = 1
CANDIDATE_ARTIFACT_SCHEMA_VERSION = 1
CANDIDATE_MANIFEST_SCHEMA_ID = "tcenv.candidate.artifact-manifest.1"
CANDIDATE_ROLES = frozenset({"device_kernel", "host_tiling", "schedule", "launch"})
TENSOR_SPACES = frozenset({"global", "workspace"})


def _json_term(value: Any, where: str) -> Any:
    """@brief 校验一个几何/ABI 术语是可 canonical 化的 JSON 值。

    @param value 术语。
    @param where 诊断位置。
    @return 原值。
    @raises TaskContractError 非法类型。
    """
    if isinstance(value, bool):
        raise TaskContractError(f"{where}: a geometry/ABI term must not be a boolean")
    if isinstance(value, int):
        if value < 0:
            raise TaskContractError(f"{where}: a geometry/ABI term must not be negative")
        return value
    if isinstance(value, float):
        if not math.isfinite(value) or value < 0:
            raise TaskContractError(f"{where}: a geometry/ABI term must be a finite non-negative number")
        return value
    if isinstance(value, str):
        if not value:
            raise TaskContractError(f"{where}: a geometry/ABI term string must be non-empty")
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_term(item, f"{where}.{key}") for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_term(item, f"{where}[{index}]") for index, item in enumerate(value)]
    raise TaskContractError(f"{where}: unsupported geometry/ABI term type {type(value).__name__}")


@dataclass(frozen=True, kw_only=True)
class TensorMemoryArg:
    """@brief TensorMemoryContract 的一个张量绑定（符号形状，不含 tiling）。"""

    name: str
    role: str
    dtype: str
    space: str
    access: str
    shape: tuple[Any, ...]
    strides: tuple[Any, ...]
    doc: str = ""

    def __post_init__(self) -> None:
        """@brief 校验角色/读写/空间/形状长度。

        @raises TaskContractError 非法。
        """
        if not self.name:
            raise TaskContractError("TensorMemoryArg.name must be non-empty")
        try:
            get_dtype(self.dtype)
        except Exception as exc:  # noqa: BLE001 - get_dtype raises KeyError
            raise TaskContractError(f"{self.name}: unknown dtype {self.dtype!r}") from exc
        if self.role not in ("input", "output"):
            raise TaskContractError(f"{self.name}: role must be input/output, got {self.role!r}")
        if self.access not in ("read", "write", "read_write"):
            raise TaskContractError(f"{self.name}: access must be read/write/read_write, got {self.access!r}")
        if self.role == "input" and self.access not in ("read", "read_write"):
            raise TaskContractError(f"{self.name}: an input needs read/read_write")
        if self.role == "output" and self.access not in ("write", "read_write"):
            raise TaskContractError(f"{self.name}: an output needs write/read_write")
        if self.space not in TENSOR_SPACES:
            raise TaskContractError(f"{self.name}: space must be one of {sorted(TENSOR_SPACES)}, got {self.space!r}")
        if len(self.shape) != len(self.strides):
            raise TaskContractError(f"{self.name}: shape and strides need one term per dimension")
        for index, term in enumerate(self.shape):
            _json_term(term, f"{self.name}.shape[{index}]")
        for index, term in enumerate(self.strides):
            _json_term(term, f"{self.name}.strides[{index}]")

    def document(self) -> dict[str, Any]:
        """@brief 该张量绑定的 JSON 文档。

        @return 字段字典。
        """
        return {
            "name": self.name,
            "role": self.role,
            "dtype": self.dtype,
            "space": self.space,
            "access": self.access,
            "shape": [_json_term(term, f"{self.name}.shape") for term in self.shape],
            "strides": [_json_term(term, f"{self.name}.strides") for term in self.strides],
            "doc": self.doc,
        }


@dataclass(frozen=True, kw_only=True)
class TensorMemoryContract:
    """@brief 目标 ABI/内存/数值约束（由 A2 的 TargetAbiDeriver 派生；A1 只冻结接口）。"""

    abi_id: str
    backend_id: str
    target_id: str
    tensors: tuple[TensorMemoryArg, ...]
    acc_dtype: str
    rtol: float
    atol_rel: float
    exact: bool
    schema_version: int = TENSOR_MEMORY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        """@brief 校验身份、唯一张量与数值约束。

        @raises TaskContractError 非法。
        """
        if self.schema_version != TENSOR_MEMORY_SCHEMA_VERSION:
            raise TaskContractError(f"unsupported TensorMemoryContract schema {self.schema_version}")
        if not self.abi_id or not self.backend_id or not self.target_id:
            raise TaskContractError("TensorMemoryContract needs a non-empty abi_id/backend_id/target_id")
        try:
            get_dtype(self.acc_dtype)
        except Exception as exc:  # noqa: BLE001
            raise TaskContractError(f"unknown acc_dtype {self.acc_dtype!r}") from exc
        if not self.tensors:
            raise TaskContractError("TensorMemoryContract needs at least one tensor")
        names = [tensor.name for tensor in self.tensors]
        if len(set(names)) != len(names):
            raise TaskContractError(f"TensorMemoryContract has duplicate tensor names {names}")
        for value, label in ((self.rtol, "rtol"), (self.atol_rel, "atol_rel")):
            if not math.isfinite(float(value)) or float(value) < 0:
                raise TaskContractError(f"TensorMemoryContract.{label} must be a finite non-negative number")

    def tensor(self, name: str) -> TensorMemoryArg:
        """@brief 按名字取张量。

        @param name 张量名。
        @return 绑定。
        @raises KeyError 没有该张量。
        """
        for tensor in self.tensors:
            if tensor.name == name:
                return tensor
        raise KeyError(name)

    def document(self) -> dict[str, Any]:
        """@brief 该 ABI 合同的 JSON 文档（题面展示的唯一 ABI 片段）。

        @return 字段字典。
        """
        return {
            "schema_version": self.schema_version,
            "abi_id": self.abi_id,
            "backend_id": self.backend_id,
            "target_id": self.target_id,
            "tensors": [tensor.document() for tensor in self.tensors],
            "acc_dtype": self.acc_dtype,
            "tolerances": {"rtol": float(self.rtol), "atol_rel": float(self.atol_rel), "exact": bool(self.exact)},
        }


@dataclass(frozen=True, kw_only=True)
class GeometryStage:
    """@brief 一个 stage 的候选几何声明（A1 只冻结字段；解析/校验/求值属 A2）。

    The fields are the interface A2's typed geometry will consume.  A1 does not parse the
    expressions, check kernel names, validate stage arity or define an execution identity.
    """

    kernel: str
    grid: tuple[Any, Any, Any]
    threads: tuple[Any, Any, Any]
    threadgroup_bytes: Any = 0
    workspace: tuple[Any, ...] = ()
    when: Any = None
    derived_scalars: tuple[tuple[str, Any], ...] = ()

    def document(self) -> dict[str, Any]:
        """@brief 该 stage 的 JSON 文档（字段投影，不做语义校验）。

        @return 字段字典。
        """
        return {
            "kernel": self.kernel,
            "grid": list(self.grid),
            "threads": list(self.threads),
            "threadgroup_bytes": self.threadgroup_bytes,
            "workspace": list(self.workspace),
            "when": self.when,
            "derived_scalars": [{"name": name, "term": term} for name, term in self.derived_scalars],
        }


@dataclass(frozen=True, kw_only=True)
class ExecutionGeometry:
    """@brief 候选或 fallback 的执行几何接口（A1 不定义执行身份或几何校验）。

    ``source`` is the interface-level distinction between candidate-owned geometry and the
    independent fallback.  A2/A3 own typed parsing, validation, pairing and the execution
    identity that covers both documents.
    """

    source: str
    stages: tuple[GeometryStage, ...]
    schema_version: int = EXECUTION_GEOMETRY_SCHEMA_VERSION

    @property
    def is_fallback(self) -> bool:
        """@brief source 是否为 fallback。

        @return ``source == "fallback"``。
        """
        return self.source == "fallback"

    def document(self) -> dict[str, Any]:
        """@brief 该几何的字段投影（没有 A1 执行身份）。

        @return 字段字典。
        """
        return {
            "schema_version": self.schema_version,
            "source": self.source,
            "stages": [stage.document() for stage in self.stages],
        }


def execution_geometry_from_document(document: Mapping[str, Any]) -> ExecutionGeometry:
    """@brief 从 ``ExecutionGeometry.document`` 还原同一份未求值几何。

    暂存 worker 必须执行控制机已经选定的几何，不能按 launch 文本再猜一遍。派生标量、
    ``when`` 和 workspace 都在这份文档里。

    @param document ``ExecutionGeometry.document()`` 的结果。
    @return 对应的几何。
    @raises TaskContractError 文档不是这份几何。
    """
    if not isinstance(document, Mapping) or "stages" not in document or "source" not in document:
        raise TaskContractError("an execution geometry document needs source and stages")
    stages = []
    for index, item in enumerate(document["stages"]):
        if not isinstance(item, Mapping):
            raise TaskContractError(f"execution geometry stage {index} is not an object")
        derived = tuple(
            (str(pair["name"]), pair["term"])
            for pair in item.get("derived_scalars") or ()
        )
        stages.append(
            GeometryStage(
                kernel=str(item["kernel"]),
                grid=tuple(item["grid"]),
                threads=tuple(item["threads"]),
                threadgroup_bytes=item.get("threadgroup_bytes", 0),
                workspace=tuple(item.get("workspace") or ()),
                when=item.get("when"),
                derived_scalars=derived,
            )
        )
    return ExecutionGeometry(
        source=str(document["source"]),
        stages=tuple(stages),
        schema_version=int(document.get("schema_version", EXECUTION_GEOMETRY_SCHEMA_VERSION)),
    )


def _candidate_path(value: str, where: str) -> str:
    """@brief 校验候选 entry 的相对 POSIX 路径。

    @param value 路径。
    @param where 诊断位置。
    @return 原路径。
    @raises TaskContractError 非法。
    """
    if not value or value.startswith("/") or "\\" in value:
        raise TaskContractError(f"{where}: {value!r} must be a relative POSIX path")
    path = PurePosixPath(value)
    # ``PurePosixPath`` collapses ``//`` and a trailing slash, but it deliberately keeps
    # ``.``/``..`` components.  Check every component explicitly: a path such as
    # ``a/../../b.metal`` is not normalized and escapes the candidate root even though it
    # does not start with ``../``.
    if any(part in (".", "..") for part in path.parts):
        raise TaskContractError(f"{where}: {value!r} must not contain '.' or '..' path segments")
    normalized = str(path)
    if normalized != value or normalized in (".", ".."):
        raise TaskContractError(f"{where}: {value!r} must be a normalized relative path")
    return value


@dataclass(frozen=True, kw_only=True)
class CandidateEntry:
    """@brief 一个候选提交源文件（字节是权威，路径/角色只是清单）。"""

    path: str
    role: str
    media_type: str
    data: bytes

    def __post_init__(self) -> None:
        """@brief 校验路径/角色/媒体类型/字节。

        @raises TaskContractError 非法。
        """
        _candidate_path(self.path, "CandidateEntry.path")
        if self.role not in CANDIDATE_ROLES:
            raise TaskContractError(f"{self.path}: unknown candidate role {self.role!r}; known {sorted(CANDIDATE_ROLES)}")
        if not self.media_type or "/" not in self.media_type:
            raise TaskContractError(f"{self.path}: media_type must look like type/subtype")
        if not isinstance(self.data, (bytes, bytearray)) or not self.data:
            raise TaskContractError(f"{self.path}: candidate entry bytes must be non-empty")

    @property
    def sha256(self) -> str:
        """@brief 该 entry 的字节摘要。

        @return 十六进制 sha256。
        """
        import hashlib

        return hashlib.sha256(bytes(self.data)).hexdigest()

    def document(self) -> dict[str, Any]:
        """@brief 清单条目（不含 bytes）。

        @return 字段字典。
        """
        return {
            "path": self.path,
            "role": self.role,
            "media_type": self.media_type,
            "size_bytes": len(self.data),
            "sha256": self.sha256,
        }


@dataclass(frozen=True, kw_only=True)
class CandidateArtifact:
    """@brief 候选提交的纯字节工件（没有作者/委员会/会话字段）。"""

    track_id: str
    entries: tuple[CandidateEntry, ...]
    schema_version: int = CANDIDATE_ARTIFACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        """@brief 校验 track id、entry 非空与路径唯一。

        @raises TaskContractError 非法。
        """
        if self.schema_version != CANDIDATE_ARTIFACT_SCHEMA_VERSION:
            raise TaskContractError(f"unsupported CandidateArtifact schema {self.schema_version}")
        if not self.track_id:
            raise TaskContractError("CandidateArtifact.track_id must be non-empty")
        if not self.entries:
            raise TaskContractError("CandidateArtifact needs at least one entry")
        seen: dict[str, str] = {}
        for entry in self.entries:
            folded = entry.path.casefold()
            if folded in seen:
                raise TaskContractError(f"candidate paths collide: {entry.path!r} and {seen[folded]!r}")
            seen[folded] = entry.path

    @property
    def single_source(self) -> bool:
        """@brief 是否为单文件提交（``msl-single`` 身份 = 文件字节 SHA-256）。

        @return 单 device_kernel 条目时为 True。
        """
        return len(self.entries) == 1 and self.entries[0].role == "device_kernel"

    def manifest(self) -> dict[str, Any]:
        """@brief 仅候选提交源 entries 的 canonical 清单。

        @return 清单文档。
        """
        return {
            "schema_id": CANDIDATE_MANIFEST_SCHEMA_ID,
            "entries": [entry.document() for entry in sorted(self.entries, key=lambda entry: entry.path)],
        }

    @property
    def manifest_bytes(self) -> bytes:
        """@brief 仅候选源 entries 的 canonical manifest 字节。

        This is the exact byte string whose SHA-256 is the bundle's ``candidate_sha256``.  A3
        carries it as the build's ``SOURCE`` artifact, so the frozen schema-4 projection can
        check ``SOURCE.sha256 == candidate_sha256`` for a multi-file candidate without inventing
        a second identity.

        @return RFC 8785 canonical JSON bytes of :meth:`manifest`。
        """
        from ..identity import canonical_bytes

        return canonical_bytes(self.manifest())

    @property
    def candidate_sha256(self) -> str:
        """@brief 提交身份：单文件用字节 SHA-256，束用 canonical manifest digest。

        The single-file spelling is byte-for-byte the existing ``candidate_sha256`` used by
        ``EvaluationRequest``; no second hash algorithm is introduced for ``msl-single``.  A
        bundle's identity is the SHA-256 of :attr:`manifest_bytes`, i.e. exactly the manifest
        digest the plan names — never the hash of the transport envelope.

        @return 十六进制 sha256。
        """
        if self.single_source:
            return self.entries[0].sha256
        import hashlib

        try:
            return hashlib.sha256(self.manifest_bytes).hexdigest()
        except IdentityError as exc:  # pragma: no cover - entries are canonicalised
            raise TaskContractError(f"candidate manifest is not canonical: {exc}") from exc

    def verdict_key(self) -> str:
        """@brief 与作者/LLM/committee/会话无关的判决键。

        @return ``candidate_sha256``。
        """
        return self.candidate_sha256

    def validate_for(self, track: TrackSpec) -> str:
        """@brief 结构交付判决（同一候选字节 + 同一 track 必得同一结论）。

        @param track 目标 track。
        @return ``"accepted"``。
        @raises TaskContractError 角色与 delivery 不匹配。
        """
        if track.track_id != self.track_id:
            raise TaskContractError(
                f"candidate is for track {self.track_id!r}, not {track.track_id!r}; evidence must not be reused"
            )
        roles = {entry.role for entry in self.entries}
        unknown = sorted(roles - set(track.candidate_roles))
        if unknown:
            raise TaskContractError(
                f"candidate roles {unknown} are not registered for track {track.track_id}/"
                f"{track.track_version}; registered roles are {list(track.candidate_roles)}"
            )
        if track.delivery == "single_source":
            if not self.single_source:
                raise TaskContractError(
                    f"track {track.track_id} is single-source but the candidate has {len(self.entries)} entries "
                    f"with roles {sorted(roles)}"
                )
        elif track.delivery == "candidate_bundle":
            missing = sorted({"device_kernel", "host_tiling"} - roles)
            if missing:
                raise TaskContractError(
                    f"track {track.track_id} is a candidate bundle and needs {missing}; Host Tiling belongs to "
                    "the candidate bundle, not to the track"
                )
        else:  # pragma: no cover - registry enum
            raise TaskContractError(f"unknown track delivery {track.delivery!r}")
        return "accepted"


@dataclass(frozen=True)
class TrackRenderer:
    """@brief 冻结的 TrackRenderer：只格式化已声明事实，绝不提供 tiling/ABI 派生。"""

    renderer_id: str
    renderer_version: int

    @classmethod
    def for_track(cls, track: TrackSpec) -> "TrackRenderer":
        """@brief 由 track 登记构造 renderer。

        @param track track。
        @return renderer。
        """
        return cls(renderer_id=track.renderer.implementation_id, renderer_version=track.renderer.implementation_version)

    def render(
        self,
        *,
        backend_id: str,
        instance: Any,
        reference_source: str,
        contract: TensorMemoryContract | None,
        display: Any,
        track: TrackSpec,
    ) -> TaskPresentation:
        """@brief 渲染统一题面（OpSpec + 只读 reference + TensorMemoryContract）。

        @param backend_id 目标后端 id。
        @param instance v2 语义实例。
        @param reference_source 只读 reference 源。
        @param contract A2 派生的 ABI 合同；A1 没有派生器，``None`` 时拒绝。
        @param display 表示覆盖。
        @param track 目标 track。
        @return 确定性展示。
        @raises UnsupportedContractCapability renderer 未登记或 ABI 合同缺失。
        """
        if (self.renderer_id, self.renderer_version) != (
            track.renderer.implementation_id,
            track.renderer.implementation_version,
        ):
            raise UnsupportedContractCapability(
                f"renderer {self.renderer_id}/{self.renderer_version} is not the renderer registered for "
                f"track {track.track_id}/{track.track_version}"
            )
        if contract is None:
            raise UnsupportedContractCapability(
                f"track {track.track_id}: rendering needs the A2-derived TensorMemoryContract; A1 does not invent "
                "an ABI or a launch geometry"
            )
        if not isinstance(contract, TensorMemoryContract):
            raise TaskContractError("contract must be a TensorMemoryContract")
        if contract.backend_id != backend_id:
            raise TaskContractError(
                f"TensorMemoryContract.backend_id {contract.backend_id!r} differs from the target backend {backend_id!r}"
            )
        return TaskPresentation(
            track_id=track.track_id,
            track_version=track.track_version,
            statement=render_statement(instance, display),
            reference_source=reference_source,
            abi_document=contract.document(),
        )


__all__ = [
    "CANDIDATE_ARTIFACT_SCHEMA_VERSION",
    "CANDIDATE_MANIFEST_SCHEMA_ID",
    "CANDIDATE_ROLES",
    "EXECUTION_GEOMETRY_SCHEMA_VERSION",
    "TENSOR_MEMORY_SCHEMA_VERSION",
    "CandidateArtifact",
    "CandidateEntry",
    "ExecutionGeometry",
    "GeometryStage",
    "TensorMemoryArg",
    "TensorMemoryContract",
    "TrackRenderer",
]
