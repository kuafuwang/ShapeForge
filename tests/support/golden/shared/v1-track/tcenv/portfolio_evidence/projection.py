"""Typed, deterministic evidence projection for the shape-portfolio P2-0 prototype.

This module deliberately separates three things that the P1 archive interleaved:

* :class:`FrozenPortfolioInputs` — self-contained candidate/OpSpec/domain/request bytes and facts;
* :class:`RawPortfolioObservation` — a small, self-reported observation from an actual runner;
* the derived evidence view — produced only by :func:`project_document` from those two inputs.

The complete JSON document is a *cache* of the projection.  :func:`verify_document` parses the
small inputs, checks the freeze digest supplied by the caller (an external trust root), recomputes
the entire view and compares the complete document with :func:`strict_json_equal`.  Derived fields
therefore need no per-field comparison code.

An offline observation is **not** inherently trustworthy.  In particular ``elapsed_s`` is volatile
and no hash in this module can prove that a GPU produced it.  ``stable_core`` exists for a live
runner to compare stable observations and the derived view; fresh execution remains a separate
trust root for P2.  No active manifest, performance claim, unseen shard or resume state is created.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from ..contract_errors import TaskContractError
from ..identity import (
    MAX_SAFE_INTEGER,
    ContractError,
    IdentityError,
    fingerprint,
    is_hex64,
    load_strict_json,
    loads_strict_json,
)
from ..opspec.model import OpSpec, OpSpecError
from ..opspec.v2 import instantiate_v2, opspec_v2_from_document
from ..target import TargetError, TargetProfile, load_target
from ..taskcontract.loader import TaskContract
from ..taskcontract.portfolio_runner import EVALUATED_OUTCOMES
from ..track.plane import CandidateEntry
from ..portfolio.candidate import (
    EVIDENCE_STATUSES,
    DomainFallback,
    PortfolioCandidate,
    PortfolioVariant,
    parse_portfolio,
    project_provider,
)
from ..portfolio.dispatch import DispatchRequest, ExactDispatchResolver
from ..portfolio.domain import DomainAuthority, SemanticDomainPlan, parse_semantic_domain_plan

FREEZE_SCHEMA_ID = "tcenv.portfolio.evidence-freeze.1"
OBSERVATION_SCHEMA_ID = "tcenv.portfolio.observation.1"
VIEW_SCHEMA_ID = "tcenv.portfolio.dispatch-view.1"
EVIDENCE_SCHEMA_ID = "tcenv.portfolio.typed-evidence.1"
ORIGIN_INVENTORY_VERSION = 1

STRICT_PARSE = "strict_parse"
FREEZE_SELF_CONSISTENCY = "freeze_self_consistency"
EXTERNAL_PIN = "external_pin"
SEMANTIC_REPROJECTION = "semantic_reprojection"
DERIVED_VIEW = "derived_view"
REJECTION_STAGES = frozenset(
    {STRICT_PARSE, FREEZE_SELF_CONSISTENCY, EXTERNAL_PIN, SEMANTIC_REPROJECTION, DERIVED_VIEW}
)


class EvidenceRejection(TaskContractError):
    """@brief 一个带机器可读验收阶段的合同拒绝，不靠消息文本猜测归因。"""

    code = "portfolio_evidence_rejected"

    def __init__(self, message: str, *, stage: str, rejected_by: str) -> None:
        if stage not in REJECTION_STAGES or not rejected_by:
            raise ValueError("an evidence rejection needs a registered stage and rejected_by")
        super().__init__(message)
        self.stage = stage
        self.rejected_by = rejected_by


FROZEN_INPUT = "frozen_input"
DERIVED = "derived"
OBSERVED_STABLE = "observed_stable"
OBSERVED_VOLATILE = "observed_volatile"
EXTERNAL_SECRET = "external_secret"
FIELD_ORIGINS = frozenset(
    {FROZEN_INPUT, DERIVED, OBSERVED_STABLE, OBSERVED_VOLATILE, EXTERNAL_SECRET}
)

#: Origin is part of the evidence contract, not a comment.  A new top-level or observation field
#: cannot enter the document without declaring how it is proved.  ``freeze`` and ``view`` cover
#: their complete subtrees; observation fields differ, so they are listed individually.
EVIDENCE_FIELD_ORIGINS: dict[str, str] = {
    "schema_id": FROZEN_INPUT,
    "freeze": FROZEN_INPUT,
    "freeze_digest": DERIVED,
    "observation.schema_id": FROZEN_INPUT,
    "observation.provider": OBSERVED_STABLE,
    "observation.track_id": OBSERVED_STABLE,
    "observation.candidate_sha256": OBSERVED_STABLE,
    "observation.instance_key": OBSERVED_STABLE,
    "observation.elapsed_s": OBSERVED_VOLATILE,
    "view": DERIVED,
    "view_digest": DERIVED,
    "root_digest": DERIVED,
}

#: A fixed semantic pin, not a digest recomputed from whichever labels happen to be in the dict.
#: It covers ORIGIN_INVENTORY_VERSION and every exact field -> origin assignment.  Changing a label
#: (for example view -> observed_volatile) without deliberately revising this contract fails closed.
EVIDENCE_FIELD_ORIGIN_INVENTORY_SHA256 = "9e87eae12ea81c09ffa8ab3833b54cfca0d3d2afbb27631b1609ab20f41e3ff3"

FREEZE_KEYS = frozenset(
    {
        "schema_id",
        "protocol_epoch",
        "opspec",
        "opspec_tensor_order",
        "domain_plan",
        "candidate_entries",
        "evidence",
        "target_id",
        "max_buffer_length",
        "request",
    }
)
ENTRY_KEYS = frozenset({"path", "role", "media_type", "data_base64"})
OBSERVATION_KEYS = frozenset(
    {"schema_id", "provider", "track_id", "candidate_sha256", "instance_key", "elapsed_s"}
)
EVIDENCE_KEYS = frozenset(
    {"schema_id", "freeze", "freeze_digest", "observation", "view", "view_digest", "root_digest"}
)


JSON_MAX_DEPTH = 32


def _json_closure(value: Any, *, where: str, _depth: int = 0) -> None:
    """@brief 递归拒绝 JSON 之外的容器/标量、非字符串 key、非有限/越界数与过深嵌套。"""
    if _depth > JSON_MAX_DEPTH:
        raise TaskContractError(f"{where} exceeds the maximum JSON nesting depth {JSON_MAX_DEPTH}")
    if value is None or type(value) in (bool, str):
        return
    if type(value) is int:
        if not -MAX_SAFE_INTEGER <= value <= MAX_SAFE_INTEGER:
            raise TaskContractError(f"{where} integer must be within the IEEE-754 safe range")
        return
    if type(value) is float:
        if not math.isfinite(value):
            raise TaskContractError(f"{where} must be a finite JSON number")
        return
    if type(value) is list:
        for index, item in enumerate(value):
            _json_closure(item, where=f"{where}[{index}]", _depth=_depth + 1)
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if type(key) is not str:
                raise TaskContractError(f"{where} has a non-string JSON object key")
            _json_closure(item, where=f"{where}.{key}", _depth=_depth + 1)
        return
    raise TaskContractError(
        f"{where} must use only the JSON type closure; got {type(value).__name__}"
    )


def _mapping(value: Any, *, where: str) -> Mapping[str, Any]:
    """@brief 要求 JSON object，不把空 list 当成空 object，也不接受非字符串 key。"""
    if not isinstance(value, Mapping):
        raise TaskContractError(f"{where} must be a JSON object, got {value!r}")
    if any(type(key) is not str for key in value):
        raise TaskContractError(f"{where} has a non-string JSON object key")
    return value


def _keys(value: Any, expected: frozenset[str], *, where: str) -> Mapping[str, Any]:
    """@brief 要求闭集 object，未知字段和缺字段都拒绝。"""
    document = _mapping(value, where=where)
    if set(document) != expected:
        raise TaskContractError(
            f"{where} has fields {sorted(document)}; expected exactly {sorted(expected)}"
        )
    return document


def _string(value: Any, *, where: str) -> str:
    """@brief 要求非空字符串。"""
    if not isinstance(value, str) or not value:
        raise TaskContractError(f"{where} must be a non-empty string, got {value!r}")
    return value


def _integer(value: Any, *, where: str, minimum: int = 0) -> int:
    """@brief 要求非 bool、IEEE-754 安全整数，拒绝 integral-float 与任意精度越界值。"""
    if type(value) is not int or not minimum <= value <= MAX_SAFE_INTEGER:
        raise TaskContractError(
            f"{where} must be an integer in [{minimum}, {MAX_SAFE_INTEGER}]"
        )
    return value


def _duration(value: Any, *, where: str) -> int | float:
    """@brief 原始时长只证明是闭集内有限、非负数值；整数绝不先转 float。"""
    if type(value) is int:
        if 0 <= value <= MAX_SAFE_INTEGER:
            return value
    elif type(value) is float:
        if math.isfinite(value) and value >= 0.0:
            return value
    raise TaskContractError(
        f"{where} must be a finite non-negative number within the JSON safe range"
    )


def _digest(value: Any, *, where: str) -> str:
    """@brief 要求规范 SHA-256 文本。"""
    if not is_hex64(value):
        raise TaskContractError(f"{where} must be a 64-character lowercase hex digest, got {value!r}")
    return str(value)


def strict_json_equal(left: Any, right: Any) -> bool:
    """@brief JSON 整树严格相等；非法 JSON 值、-0.0/+0.0 混淆均为 false。

    Unlike :func:`tcenv.identity.canonical_value`, this comparator never turns a tuple into an
    array.  Object key order remains insignificant, while every key must be an actual string.
    """
    if left is None or right is None:
        return left is None and right is None
    if type(left) is bool or type(right) is bool:
        return type(left) is bool and type(right) is bool and left is right
    if type(left) is int or type(right) is int:
        return (
            type(left) is int
            and type(right) is int
            and -MAX_SAFE_INTEGER <= left <= MAX_SAFE_INTEGER
            and -MAX_SAFE_INTEGER <= right <= MAX_SAFE_INTEGER
            and left == right
        )
    if type(left) is float or type(right) is float:
        if type(left) is not float or type(right) is not float:
            return False
        if not math.isfinite(left) or not math.isfinite(right):
            return False
        if left == 0.0 and right == 0.0:
            return math.copysign(1.0, left) == math.copysign(1.0, right)
        return left == right
    if type(left) is str or type(right) is str:
        return type(left) is str and type(right) is str and left == right
    if isinstance(left, Mapping) or isinstance(right, Mapping):
        if not isinstance(left, Mapping) or not isinstance(right, Mapping):
            return False
        if any(type(key) is not str for key in left) or any(type(key) is not str for key in right):
            return False
        return set(left) == set(right) and all(strict_json_equal(left[key], right[key]) for key in left)
    if type(left) is list or type(right) is list:
        return (
            type(left) is list
            and type(right) is list
            and len(left) == len(right)
            and all(strict_json_equal(a, b) for a, b in zip(left, right))
        )
    return False


def _origin_inventory_digest() -> str:
    payload = json.dumps(
        {"version": ORIGIN_INVENTORY_VERSION, "origins": EVIDENCE_FIELD_ORIGINS},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _verify_origin_inventory() -> None:
    expected_keys = (EVIDENCE_KEYS - {"observation"}) | {
        f"observation.{key}" for key in OBSERVATION_KEYS
    }
    if set(EVIDENCE_FIELD_ORIGINS) != expected_keys:
        raise TaskContractError("the evidence field-origin inventory does not cover the exact schema")
    if _origin_inventory_digest() != EVIDENCE_FIELD_ORIGIN_INVENTORY_SHA256:
        raise TaskContractError("the evidence field-origin inventory has incorrect semantic classifications")


def _ordered_opspec_document(
    document: Mapping[str, Any], tensor_order: tuple[str, ...]
) -> dict[str, Any]:
    """@brief 只按 freeze 的显式顺序权威重建 tensors；绝不排序或依赖 object key 顺序。"""
    tensors = _mapping(document.get("tensors"), where="freeze.opspec.tensors")
    if len(tensor_order) != len(set(tensor_order)) or set(tensor_order) != set(tensors):
        raise TaskContractError(
            "freeze.opspec_tensor_order must contain every opspec.tensors name exactly once"
        )
    ordered = dict(document)
    ordered["tensors"] = {name: tensors[name] for name in tensor_order}
    return ordered


def _entry_from_document(value: Any, *, where: str) -> CandidateEntry:
    """@brief 从 canonical base64 严格恢复一个候选 entry。"""
    document = _keys(value, ENTRY_KEYS, where=where)
    encoded = _string(document["data_base64"], where=f"{where}.data_base64")
    try:
        data = base64.b64decode(encoded.encode("ascii"), validate=True)
    except (UnicodeEncodeError, binascii.Error) as exc:
        raise TaskContractError(f"{where}.data_base64 is not canonical base64: {exc}") from exc
    if base64.b64encode(data).decode("ascii") != encoded:
        raise TaskContractError(f"{where}.data_base64 is not the canonical encoding of its bytes")
    return CandidateEntry(
        path=_string(document["path"], where=f"{where}.path"),
        role=_string(document["role"], where=f"{where}.role"),
        media_type=_string(document["media_type"], where=f"{where}.media_type"),
        data=data,
    )


def _entry_document(entry: CandidateEntry) -> dict[str, Any]:
    """@brief 候选 entry 的自包含 JSON 表示（字节而非另一个自述摘要）。"""
    return {
        "path": entry.path,
        "role": entry.role,
        "media_type": entry.media_type,
        "data_base64": base64.b64encode(bytes(entry.data)).decode("ascii"),
    }


@dataclass(frozen=True)
class FrozenPortfolioInputs:
    """@brief 一次 CPU dispatch/projection 所需的自包含冻结输入。"""

    protocol_epoch: int
    opspec_document: Mapping[str, Any]
    opspec_tensor_order: tuple[str, ...]
    domain_plan_document: Mapping[str, Any]
    candidate_entries: tuple[CandidateEntry, ...]
    evidence: Mapping[str, str]
    target_id: str
    max_buffer_length: int | None
    request: DispatchRequest

    def __post_init__(self) -> None:
        _integer(self.protocol_epoch, where="freeze.protocol_epoch", minimum=1)
        _json_closure(self.opspec_document, where="freeze.opspec")
        _mapping(self.opspec_document, where="freeze.opspec")
        if not isinstance(self.opspec_tensor_order, tuple) or any(
            type(name) is not str or not name for name in self.opspec_tensor_order
        ):
            raise TaskContractError("freeze.opspec_tensor_order must be a tuple of non-empty strings")
        ordered_opspec = _ordered_opspec_document(self.opspec_document, self.opspec_tensor_order)
        object.__setattr__(self, "opspec_document", ordered_opspec)
        _json_closure(self.domain_plan_document, where="freeze.domain_plan")
        _mapping(self.domain_plan_document, where="freeze.domain_plan")
        if not isinstance(self.candidate_entries, tuple) or not self.candidate_entries:
            raise TaskContractError("freeze.candidate_entries must be a non-empty tuple")
        if any(not isinstance(entry, CandidateEntry) for entry in self.candidate_entries):
            raise TaskContractError("freeze.candidate_entries must contain CandidateEntry values")
        _json_closure(self.evidence, where="freeze.evidence")
        evidence = _mapping(self.evidence, where="freeze.evidence")
        for provider, status in evidence.items():
            _string(provider, where="freeze.evidence provider")
            if not isinstance(status, str) or status not in EVIDENCE_STATUSES:
                raise TaskContractError(
                    f"freeze.evidence.{provider} must be one of {sorted(EVIDENCE_STATUSES)}, got {status!r}"
                )
        _string(self.target_id, where="freeze.target_id")
        if self.max_buffer_length is not None:
            _integer(self.max_buffer_length, where="freeze.max_buffer_length", minimum=1)
        if not isinstance(self.request, DispatchRequest):
            raise TaskContractError("freeze.request must be a DispatchRequest")
        _json_closure(self.request.document(), where="freeze.request")
        # Parse and cross-check the complete frozen semantics at the constructor boundary.  The
        # projector never receives a syntactically valid but semantically unrelated pack.
        spec, plan, portfolio, instances, target = self._semantic_parts()
        expected_providers = set(portfolio.variant_ids)
        if portfolio.fallback_id is not None:
            expected_providers.add(portfolio.fallback_id)
        if set(self.evidence) != expected_providers:
            raise TaskContractError(
                "freeze.evidence must name exactly the portfolio's providers, got "
                f"{sorted(set(self.evidence) ^ expected_providers)} as the symmetric difference"
            )
        DomainAuthority(
            plan=plan,
            spec=spec,
            target=target,
            instances=instances,
            max_buffer_length=self.max_buffer_length,
        )

    @classmethod
    def from_document(cls, value: Any) -> "FrozenPortfolioInputs":
        _json_closure(value, where="freeze")
        document = _keys(value, FREEZE_KEYS, where="freeze")
        if document["schema_id"] != FREEZE_SCHEMA_ID:
            raise TaskContractError(f"freeze.schema_id must be {FREEZE_SCHEMA_ID!r}")
        entries_value = document["candidate_entries"]
        if not isinstance(entries_value, list) or not entries_value:
            raise TaskContractError("freeze.candidate_entries must be a non-empty array")
        entries = tuple(
            _entry_from_document(item, where=f"freeze.candidate_entries[{index}]")
            for index, item in enumerate(entries_value)
        )
        order_value = document["opspec_tensor_order"]
        if not isinstance(order_value, list) or any(
            type(name) is not str or not name for name in order_value
        ):
            raise TaskContractError("freeze.opspec_tensor_order must be an array of non-empty strings")
        return cls(
            protocol_epoch=_integer(document["protocol_epoch"], where="freeze.protocol_epoch", minimum=1),
            opspec_document=_mapping(document["opspec"], where="freeze.opspec"),
            opspec_tensor_order=tuple(order_value),
            domain_plan_document=_mapping(document["domain_plan"], where="freeze.domain_plan"),
            candidate_entries=entries,
            evidence=_mapping(document["evidence"], where="freeze.evidence"),
            target_id=_string(document["target_id"], where="freeze.target_id"),
            max_buffer_length=(
                None
                if document["max_buffer_length"] is None
                else _integer(document["max_buffer_length"], where="freeze.max_buffer_length", minimum=1)
            ),
            request=DispatchRequest.parse(document["request"]),
        )

    def document(self) -> dict[str, Any]:
        """@brief 唯一 canonical 输入视图；entry 顺序不成为第二个身份来源。"""
        return {
            "schema_id": FREEZE_SCHEMA_ID,
            "protocol_epoch": self.protocol_epoch,
            "opspec": _ordered_opspec_document(self.opspec_document, self.opspec_tensor_order),
            "opspec_tensor_order": list(self.opspec_tensor_order),
            "domain_plan": dict(self.domain_plan_document),
            "candidate_entries": [
                _entry_document(entry)
                for entry in sorted(self.candidate_entries, key=lambda item: item.path)
            ],
            "evidence": {str(key): self.evidence[key] for key in sorted(self.evidence)},
            "target_id": self.target_id,
            "max_buffer_length": self.max_buffer_length,
            "request": self.request.document(),
        }

    @property
    def digest(self) -> str:
        """@brief 冻结输入的 RFC 8785 身份；调用方必须在外部钉住这个值。"""
        return fingerprint(self.document())

    def _semantic_parts(
        self,
    ) -> tuple[OpSpec, SemanticDomainPlan, PortfolioCandidate, tuple[Any, ...], TargetProfile]:
        spec = opspec_v2_from_document(
            _ordered_opspec_document(self.opspec_document, self.opspec_tensor_order),
            source="freeze.opspec",
        )
        plan = parse_semantic_domain_plan(self.domain_plan_document, where="freeze.domain_plan")
        portfolio = parse_portfolio(self.candidate_entries, domain_plan=plan)
        named_instances = list(dict.fromkeys((point.dtype, point.size) for point in plan.instances))
        instances = tuple(instantiate_v2(spec, dtype, size) for dtype, size in named_instances)
        target = load_target(self.target_id)
        return spec, plan, portfolio, instances, target

    def authority(self, *, contract: TaskContract | None = None) -> DomainAuthority:
        """@brief 从冻结 OpSpec/target/instances 重建权威，可选绑定同包 TaskContract。"""
        spec, plan, _portfolio, instances, target = self._semantic_parts()
        return DomainAuthority(
            plan=plan,
            spec=spec,
            target=target,
            instances=instances,
            contract=contract,
            max_buffer_length=self.max_buffer_length,
        )

    def resolve(self):
        """@brief 用唯一生产 ExactDispatchResolver 重放冻结请求。"""
        spec, plan, portfolio, instances, target = self._semantic_parts()
        resolver = ExactDispatchResolver(
            domain_plan=plan,
            portfolio=portfolio,
            evidence=self.evidence,
            spec=spec,
            target=target,
            instances=instances,
            max_buffer_length=self.max_buffer_length,
        )
        return portfolio, resolver.resolve(self.request)


@dataclass(frozen=True)
class RawPortfolioObservation:
    """@brief  Runner 对一次实际投影的小观察；它不是离线真实性证明。"""

    provider: str | None
    track_id: str | None
    candidate_sha256: str | None
    instance_key: str
    elapsed_s: int | float

    def __post_init__(self) -> None:
        _string(self.instance_key, where="observation.instance_key")
        _duration(self.elapsed_s, where="observation.elapsed_s")
        if self.provider is None:
            if self.track_id is not None or self.candidate_sha256 is not None:
                raise TaskContractError("a build-free observation cannot name a track or candidate")
            return
        if self.provider not in ("candidate", "fallback"):
            raise TaskContractError(f"observation.provider must be candidate/fallback/null, got {self.provider!r}")
        _string(self.track_id, where="observation.track_id")
        _digest(self.candidate_sha256, where="observation.candidate_sha256")

    @classmethod
    def from_document(cls, value: Any) -> "RawPortfolioObservation":
        _json_closure(value, where="observation")
        document = _keys(value, OBSERVATION_KEYS, where="observation")
        if document["schema_id"] != OBSERVATION_SCHEMA_ID:
            raise TaskContractError(f"observation.schema_id must be {OBSERVATION_SCHEMA_ID!r}")
        return cls(
            provider=document["provider"],
            track_id=document["track_id"],
            candidate_sha256=document["candidate_sha256"],
            instance_key=_string(document["instance_key"], where="observation.instance_key"),
            elapsed_s=_duration(document["elapsed_s"], where="observation.elapsed_s"),
        )

    def document(self) -> dict[str, Any]:
        return {
            "schema_id": OBSERVATION_SCHEMA_ID,
            "provider": self.provider,
            "track_id": self.track_id,
            "candidate_sha256": self.candidate_sha256,
            "instance_key": self.instance_key,
            "elapsed_s": self.elapsed_s,
        }

    def stable_document(self) -> dict[str, Any]:
        """@brief live 可比较的观察；volatile 墙钟不属于稳定核心。"""
        document = self.document()
        del document["elapsed_s"]
        return document


def _selected_provider(
    portfolio: PortfolioCandidate, record: Any
) -> PortfolioVariant | DomainFallback | None:
    """@brief 按重放的 record 选择唯一 provider；拒绝路径没有投影。"""
    if record.outcome not in EVALUATED_OUTCOMES:
        return None
    if record.outcome == "resolved":
        provider = portfolio.variant_for_provider(str(record.variant_id))
        if provider is None:  # pragma: no cover - production resolver closes this
            raise TaskContractError("the resolved record names a variant outside its frozen portfolio")
        return provider
    provider = portfolio.fallback
    if provider is None or provider.fallback_id != record.fallback_id:  # pragma: no cover
        raise TaskContractError("the fallback record names a fallback outside its frozen portfolio")
    return provider


def project_document(
    freeze: FrozenPortfolioInputs,
    observation: RawPortfolioObservation,
    *,
    contract: TaskContract | None = None,
) -> dict[str, Any]:
    """@brief 从冻结输入 + 小观察确定性地生成完整证据文档。

    The observation is checked against the independently replayed dispatch/projection.  It cannot
    choose another provider, Track or candidate.  A bundle projection needs the authoritative
    TaskContract from the same frozen pack; the P2-0 single-source fixture does not need one.
    """
    if not isinstance(freeze, FrozenPortfolioInputs):
        raise TaskContractError("project_document needs FrozenPortfolioInputs")
    if not isinstance(observation, RawPortfolioObservation):
        raise TaskContractError("project_document needs a RawPortfolioObservation")
    portfolio, record = freeze.resolve()
    if observation.instance_key != freeze.request.instance_key:
        raise TaskContractError(
            f"observation.instance_key {observation.instance_key!r} is not the frozen request's "
            f"{freeze.request.instance_key!r}"
        )
    provider = _selected_provider(portfolio, record)
    projection = None
    if provider is not None:
        projection = project_provider(portfolio, provider, authority=freeze.authority(contract=contract))
        expected_observation = {
            "provider": projection.provider,
            "track_id": projection.track_id,
            "candidate_sha256": projection.candidate.candidate_sha256,
            "instance_key": freeze.request.instance_key,
        }
        actual_observation = observation.stable_document()
        actual_observation.pop("schema_id")
        if actual_observation != expected_observation:
            raise TaskContractError(
                f"the raw observation {actual_observation!r} is not the replayed projection "
                f"{expected_observation!r}"
            )
    elif observation.provider is not None:
        raise TaskContractError(
            f"a {record.outcome} refusal cannot carry an executed provider observation"
        )
    view = {
        "schema_id": VIEW_SCHEMA_ID,
        "protocol_epoch": freeze.protocol_epoch,
        "record": record.document(),
        "projection": None if projection is None else projection.document(),
        # A fallback may execute correctly, but it never earns the generated variant's credit.
        "credits_variant": record.outcome == "resolved" and record.provider == "candidate",
    }
    view_digest = fingerprint(view)
    freeze_digest = freeze.digest
    root_digest = fingerprint(
        {
            "schema_id": EVIDENCE_SCHEMA_ID,
            "freeze_digest": freeze_digest,
            "observation": observation.document(),
            "view_digest": view_digest,
        }
    )
    return {
        "schema_id": EVIDENCE_SCHEMA_ID,
        "freeze": freeze.document(),
        "freeze_digest": freeze_digest,
        "observation": observation.document(),
        "view": view,
        "view_digest": view_digest,
        "root_digest": root_digest,
    }


def stable_core(document: Mapping[str, Any]) -> dict[str, Any]:
    """@brief live 比较核心：冻结身份 + 稳定观察 + 完整派生视图，不含 volatile/root digest。"""
    _json_closure(document, where="evidence")
    evidence = _keys(document, EVIDENCE_KEYS, where="evidence")
    observation = RawPortfolioObservation.from_document(evidence["observation"])
    return {
        "schema_id": EVIDENCE_SCHEMA_ID,
        "freeze_digest": evidence["freeze_digest"],
        "observation": observation.stable_document(),
        "view": evidence["view"],
    }


_CONTRACT_ERRORS = (TaskContractError, IdentityError, OpSpecError, TargetError, ContractError, RecursionError)


def verify_document(
    value: Any,
    *,
    expected_freeze_digest: str,
    contract: TaskContract | None = None,
) -> dict[str, Any]:
    """@brief 严格解析、外部 freeze 绑定、全量重投影并整树比较。

    ``expected_freeze_digest`` must come from the caller's protected freeze record, never from the
    archive.  This in-memory Mapping API cannot recover duplicate keys that a previous permissive
    parser already discarded; production file/text input must use :func:`verify_json` or
    :func:`loads_document` instead.  Every expected refusal carries ``stage`` and ``rejected_by``.
    """
    try:
        _json_closure(value, where="evidence")
        _verify_origin_inventory()
        document = _keys(value, EVIDENCE_KEYS, where="evidence")
        if document["schema_id"] != EVIDENCE_SCHEMA_ID:
            raise TaskContractError(f"evidence.schema_id must be {EVIDENCE_SCHEMA_ID!r}")
        _mapping(document["freeze"], where="evidence.freeze")
        expected_digest = _digest(expected_freeze_digest, where="expected_freeze_digest")
        archived_digest = _digest(document["freeze_digest"], where="evidence.freeze_digest")
    except EvidenceRejection:
        raise
    except _CONTRACT_ERRORS as exc:
        raise EvidenceRejection(
            f"strict evidence parsing failed: {exc}",
            stage=STRICT_PARSE,
            rejected_by="json_type_closure_and_schema",
        ) from exc

    # Pin the archived canonical value before semantic replay.  A different declaration order must
    # therefore be rejected by the external trust root itself, not hidden by a downstream ABI error.
    try:
        actual_freeze_digest = fingerprint(document["freeze"])
    except _CONTRACT_ERRORS as exc:
        raise EvidenceRejection(
            f"the archived freeze is not canonical: {exc}",
            stage=FREEZE_SELF_CONSISTENCY,
            rejected_by="freeze_fingerprint",
        ) from exc
    if actual_freeze_digest != archived_digest:
        raise EvidenceRejection(
            "evidence.freeze_digest is not the digest of the archived freeze bytes",
            stage=FREEZE_SELF_CONSISTENCY,
            rejected_by="freeze_digest",
        )
    if actual_freeze_digest != expected_digest:
        raise EvidenceRejection(
            "the archived freeze is not the externally pinned freeze: "
            f"{actual_freeze_digest} != {expected_digest}",
            stage=EXTERNAL_PIN,
            rejected_by="expected_freeze_digest",
        )

    try:
        freeze = FrozenPortfolioInputs.from_document(document["freeze"])
        observation = RawPortfolioObservation.from_document(document["observation"])
        expected = project_document(freeze, observation, contract=contract)
    except EvidenceRejection:
        raise
    except _CONTRACT_ERRORS as exc:
        raise EvidenceRejection(
            f"semantic evidence reprojection failed: {exc}",
            stage=SEMANTIC_REPROJECTION,
            rejected_by="frozen_inputs_observation_projector",
        ) from exc
    if not strict_json_equal(document, expected):
        raise EvidenceRejection(
            "the archived evidence is not the deterministic projection of its frozen inputs and observation",
            stage=DERIVED_VIEW,
            rejected_by="strict_json_equal",
        )
    return stable_core(expected)


def verify_json(
    path: str | Path,
    *,
    expected_freeze_digest: str,
    contract: TaskContract | None = None,
) -> dict[str, Any]:
    """@brief 唯一生产文件入口：先经 ``load_strict_json`` 拒绝重复 key，再验收。"""
    try:
        value = load_strict_json(path)
    except (IdentityError, OSError, RecursionError) as exc:
        raise EvidenceRejection(
            f"strict evidence file loading failed: {exc}",
            stage=STRICT_PARSE,
            rejected_by="load_strict_json",
        ) from exc
    return verify_document(value, expected_freeze_digest=expected_freeze_digest, contract=contract)


def loads_document(
    text: str,
    *,
    expected_freeze_digest: str,
    contract: TaskContract | None = None,
) -> dict[str, Any]:
    """@brief 唯一生产文本入口：重复 key 在进入 Mapping 之前拒绝。"""
    try:
        value = loads_strict_json(text, origin="<portfolio-evidence>")
    except (IdentityError, RecursionError) as exc:
        raise EvidenceRejection(
            f"strict evidence text loading failed: {exc}",
            stage=STRICT_PARSE,
            rejected_by="loads_strict_json",
        ) from exc
    return verify_document(value, expected_freeze_digest=expected_freeze_digest, contract=contract)


__all__ = [
    "DERIVED",
    "DERIVED_VIEW",
    "EVIDENCE_FIELD_ORIGINS",
    "EVIDENCE_FIELD_ORIGIN_INVENTORY_SHA256",
    "EVIDENCE_SCHEMA_ID",
    "EXTERNAL_PIN",
    "EXTERNAL_SECRET",
    "EvidenceRejection",
    "FIELD_ORIGINS",
    "FREEZE_SCHEMA_ID",
    "FREEZE_SELF_CONSISTENCY",
    "FROZEN_INPUT",
    "FrozenPortfolioInputs",
    "OBSERVED_STABLE",
    "OBSERVED_VOLATILE",
    "OBSERVATION_SCHEMA_ID",
    "REJECTION_STAGES",
    "RawPortfolioObservation",
    "SEMANTIC_REPROJECTION",
    "STRICT_PARSE",
    "VIEW_SCHEMA_ID",
    "loads_document",
    "project_document",
    "stable_core",
    "strict_json_equal",
    "verify_document",
    "verify_json",
]
