"""Offline exact dispatch: one request, one provider, or a typed refusal (shape portfolio P0).

The resolver never guesses.  Its order is fixed and every step has a distinct, recorded outcome:

1. an unparseable request is ``semantic_unresolved``;
2. a request outside the finite semantic domain is ``unsupported_contract_capability`` and never
   falls back — the fallback is a *correctness* provider inside the domain, not an escape hatch
   for illegal semantics;
3. a guard that references an undeclared field, fails to evaluate (for example ``rem(N, 0)``) or a
   matched provider whose independent evidence is not validated is ``dispatch_unknown``;
4. more than one match is ``dispatch_overlap``; the resolver does not rank, prioritise or guess;
5. exactly one match selects the candidate variant (after the runner-owned geometry preflight);
6. zero matches select a declared, applicable, validated domain fallback or ``dispatch_unmatched``.

The finite registration check lives in :mod:`tcenv.portfolio.candidate`; runtime dispatch always
evaluates **all** guards, so a point-set precheck never becomes a priority order.

Fallback reasons are derived from the execution stage, never self-reported: all guards false is
``outside_candidate_coverage``, and a geometry preflight that exceeds :class:`TargetAbiLimits` is
``resource_unavailable``.  A fallback may only handle a reason it declared.  A fallback's own
resource failure is a failure, never a second fallback.  Fallback results carry
``provider=fallback``: they are not the generated variant's correctness or speedup.

The composition record is :class:`PortfolioDispatchRecord`, a deliberately distinct type from the
single-stage ``v2_runtime.DispatchRecord``.  It is a decision record, not an active manifest and
not performance evidence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping

from ..contract_errors import GeometryError, TaskContractError
from ..identity import IdentityError, fingerprint, is_hex64, load_strict_json
from ..opspec.model import OpSpec
from ..opspec.v2 import SemanticInstanceV2
from ..target import TargetProfile
from ..track.abi import TargetAbiLimits
from ..track.geometry import ResolvedGeometry, resolve_geometry
from ..track.plane import ExecutionGeometry, TensorMemoryContract
from .candidate import EVIDENCE_STATUSES, FALLBACK_REASONS, PortfolioCandidate, PortfolioVariant, instance_names
from .domain import (
    INSTANCE_USAGES,
    SemanticDomainInstance,
    SemanticDomainPlan,
    instance_key,
    require_authoritative_domain_plan,
    semantic_instance_digest,
)

#: Field caps from the registered dispatch-record schema (``$.request.*`` and ``$.detail``).
#: A request that cannot be carried by a legal record is refused at parse time, not recorded.
MAX_REQUEST_OP_NAME = 128
MAX_REQUEST_DTYPE = 32
MAX_REQUEST_SIZE = 64
MAX_RECORD_DETAIL = 4000

DISPATCH_RECORD_SCHEMA_ID = "tcenv.portfolio.dispatch-record.1"
DISPATCH_RECORD_VERSION = 1
DISPATCH_RULES_VERSION = 1
SCHEMA_DIR = Path(__file__).resolve().parent / "schema"
DISPATCH_RECORD_SCHEMA_PATH = SCHEMA_DIR / "portfolio-dispatch-record.v1.json"

#: Closed request document.  ``instance_key`` is derived from ``(dtype, size, usage)`` and is
#: carried (and re-checked) so a replay document is a complete, self-describing record.
REQUEST_FIELDS = frozenset(
    {
        "op_name",
        "dtype",
        "size",
        "usage",
        "instance_key",
        "symbols",
        "attrs",
        "semantic_instance_digest",
        "target_abi_digest",
    }
)
#: Fixed outcome vocabulary.  ``variant_unavailable`` is the "exactly one variant matched but it
#: cannot run" outcome (resource overflow without a declaring fallback, an invalid geometry, or a
#: falling-back implementation that is itself unavailable).  It is not a fallback chain.
OUTCOMES = frozenset(
    {
        "resolved",
        "fallback",
        "dispatch_overlap",
        "dispatch_unknown",
        "dispatch_unmatched",
        "semantic_unresolved",
        "unsupported_contract_capability",
        "variant_unavailable",
    }
)
PROVIDERS = frozenset({"candidate", "fallback"})
PREFLIGHT_STATUSES = frozenset({"ok", "resource_unavailable", "invalid", "not_required"})
#: The A2 resolver's own wording for a limit violation ("... exceeds the target limit ...").  The
#: preflight classifies only these as ``resource_unavailable``; every other geometry failure is an
#: invalid candidate, so a broken launch cannot be rescued by a fallback.
RESOURCE_LIMIT_MARKERS = ("exceeds the target limit",)


def _schema() -> Mapping[str, Any]:
    """@brief 读取登记 dispatch record schema。

    @return schema 文档。
    @raises TaskContractError schema 不可读。
    """
    try:
        document = load_strict_json(DISPATCH_RECORD_SCHEMA_PATH)
    except IdentityError as exc:
        raise TaskContractError(f"{DISPATCH_RECORD_SCHEMA_PATH}: the dispatch schema is not strict JSON: {exc}") from exc
    if not isinstance(document, Mapping):
        raise TaskContractError(f"{DISPATCH_RECORD_SCHEMA_PATH}: the dispatch schema must be an object")
    return document


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


def _clamp_detail(detail: str | None) -> str | None:
    """@brief 把 detail 截到登记 schema 的 maxLength，保证记录始终合法。

    Request-controlled text (a long symbol name, a forged instance) can reach a refusal detail.
    The record is still a legal closed document: the detail is truncated with an explicit marker
    instead of producing a record that fails ``$.detail``.

    @param detail 原始 detail。
    @return 截断后的 detail。
    """
    if detail is None or len(detail) <= MAX_RECORD_DETAIL:
        return detail
    suffix = "… (truncated)"
    return detail[: MAX_RECORD_DETAIL - len(suffix)] + suffix


def _int_map(value: Any, *, where: str) -> dict[str, int]:
    """@brief 校验一个字符串→整数的映射（bool 不算整数）。

    @param value 值。
    @param where 诊断位置。
    @return 新字典。
    @raises TaskContractError 非法。
    """
    if not isinstance(value, Mapping):
        raise TaskContractError(f"{where} must be an object of integer fields")
    out: dict[str, int] = {}
    for name, item in value.items():
        if not isinstance(name, str) or not name:
            raise TaskContractError(f"{where}: field names must be non-empty strings")
        if isinstance(item, bool) or not isinstance(item, int):
            raise TaskContractError(f"{where}.{name}: a guard-visible field must be an integer, got {item!r}")
        out[name] = int(item)
    return out


@dataclass(frozen=True)
class DispatchRequest:
    """@brief 一个规范化的语义请求（dispatch 的唯一输入）。"""

    op_name: str
    dtype: str
    size: str
    usage: str
    symbols: Mapping[str, int]
    attrs: Mapping[str, int]
    semantic_instance_digest: str
    target_abi_digest: str

    def __post_init__(self) -> None:
        """@brief 校验请求字段（严格类型、无静默强转、名字空间不重叠）。

        @raises TaskContractError 非法。
        """
        for name, limit in (
            ("op_name", MAX_REQUEST_OP_NAME),
            ("dtype", MAX_REQUEST_DTYPE),
            ("size", MAX_REQUEST_SIZE),
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise TaskContractError(
                    f"DispatchRequest.{name} must be a non-empty string, got {type(value).__name__} {value!r}"
                )
            if len(value) > limit:
                raise TaskContractError(
                    f"DispatchRequest.{name} is longer than {limit} characters; a request that the registered "
                    "dispatch-record schema cannot carry is refused instead of recorded"
                )
        if not isinstance(self.usage, str) or self.usage not in INSTANCE_USAGES:
            raise TaskContractError(f"DispatchRequest.usage must be one of {sorted(INSTANCE_USAGES)}, got {self.usage!r}")
        symbols = _int_map(self.symbols, where="DispatchRequest.symbols")
        attrs = _int_map(self.attrs, where="DispatchRequest.attrs")
        overlap = sorted(set(symbols) & set(attrs))
        if overlap:
            raise TaskContractError(
                f"DispatchRequest names {overlap} as both a symbol and an attr; a guard name must mean one value"
            )
        _require_digest(self.semantic_instance_digest, where="DispatchRequest.semantic_instance_digest")
        _require_digest(self.target_abi_digest, where="DispatchRequest.target_abi_digest")
        instance_key(self.dtype, self.size, self.usage)

    @property
    def instance_key(self) -> str:
        """@brief 该请求的有限域键。

        @return ``"<dtype>/<size>/<usage>"``。
        """
        return instance_key(self.dtype, self.size, self.usage)

    @property
    def names(self) -> dict[str, int]:
        """@brief guard 可见的名字空间（符号 + 整数 attrs）。

        ``__post_init__`` refuses a name that is both a symbol and an attr, so this merge can
        never let one silently overwrite the other.

        @return 名字到整数。
        """
        return {**{str(name): int(value) for name, value in self.symbols.items()},
                **{str(name): int(value) for name, value in self.attrs.items()}}

    def document(self) -> dict[str, Any]:
        """@brief 请求的 JSON 文档（可重放）。

        @return 字段字典。
        """
        return {
            "op_name": self.op_name,
            "dtype": self.dtype,
            "size": self.size,
            "usage": self.usage,
            "instance_key": self.instance_key,
            "symbols": {name: int(value) for name, value in sorted(self.symbols.items())},
            "attrs": {name: int(value) for name, value in sorted(self.attrs.items())},
            "semantic_instance_digest": self.semantic_instance_digest,
            "target_abi_digest": self.target_abi_digest,
        }

    @classmethod
    def from_instance(
        cls,
        instance: SemanticInstanceV2,
        *,
        target_abi_digest: str,
        usage: str | None = None,
    ) -> "DispatchRequest":
        """@brief 由真实语义实例构造请求（摘要由本函数计算，调用方不能伪造）。

        @param instance v2 语义实例。
        @param target_abi_digest 该实例的派生 ABI 摘要。
        @param usage 用途；省略时仅当该 size 只声明一个用途才可用。
        @return 请求。
        @raises TaskContractError 实例/用途非法。
        """
        from .domain import semantic_instance_digest

        if not isinstance(instance, SemanticInstanceV2):
            raise TaskContractError("DispatchRequest.from_instance needs a SemanticInstanceV2")
        declared = tuple(instance.size_usage)
        if usage is None:
            if len(declared) != 1:
                raise TaskContractError(
                    f"{instance.op_name}.{instance.dtype}.{instance.size} declares usages {list(declared)}; pass one "
                    "explicitly instead of letting dispatch guess golden versus benchmark"
                )
            usage = declared[0]
        if usage not in declared:
            raise TaskContractError(
                f"{instance.op_name}.{instance.dtype}.{instance.size} does not declare usage {usage!r}; declared "
                f"{list(declared)}"
            )
        attrs = {
            str(name): int(value)
            for name, value in instance.resolved_attrs.items()
            if isinstance(value, int) and not isinstance(value, bool)
        }
        overlap = sorted(set(instance.symbols) & set(attrs))
        if overlap:
            raise TaskContractError(
                f"{instance.op_name}.{instance.dtype}.{instance.size}: {overlap} is both a symbol and an integer "
                "attr; the guard name space is ambiguous"
            )
        return cls(
            op_name=instance.op_name,
            dtype=instance.dtype,
            size=instance.size,
            usage=usage,
            symbols={str(name): int(value) for name, value in instance.symbols.items()},
            attrs=attrs,
            semantic_instance_digest=semantic_instance_digest(instance),
            target_abi_digest=target_abi_digest,
        )

    @classmethod
    def parse(cls, document: Any) -> "DispatchRequest":
        """@brief 闭集解析一个请求文档。

        @param document 请求文档。
        @return 请求。
        @raises TaskContractError 字段缺失、未知或类型非法（调用方转为 semantic_unresolved）。
        """
        if not isinstance(document, Mapping):
            raise TaskContractError("a dispatch request must be an object")
        keys = set(document)
        if keys != REQUEST_FIELDS:
            raise TaskContractError(
                f"a dispatch request has fields {sorted(keys)}; expected exactly {sorted(REQUEST_FIELDS)}"
            )
        declared_key = document["instance_key"]
        if not isinstance(declared_key, str) or not declared_key:
            raise TaskContractError("request.instance_key must be a non-empty string")
        request = cls(
            op_name=document["op_name"],
            dtype=document["dtype"],
            size=document["size"],
            usage=document["usage"],
            symbols=document["symbols"],
            attrs=document["attrs"],
            semantic_instance_digest=document["semantic_instance_digest"],
            target_abi_digest=document["target_abi_digest"],
        )
        if declared_key != request.instance_key:
            raise TaskContractError(
                f"request.instance_key {declared_key!r} is not the key derived from dtype/size/usage "
                f"({request.instance_key!r})"
            )
        return request


@dataclass(frozen=True)
class GeometryPreflight:
    """@brief runner 在 ``evaluate_task`` 之前对选定几何做的纯 CPU 预检结果。"""

    status: str
    detail: str | None = None
    resolved: ResolvedGeometry | None = None

    def __post_init__(self) -> None:
        """@brief 校验状态词与结果一致性。

        @raises TaskContractError 非法。
        """
        if self.status not in PREFLIGHT_STATUSES:
            raise TaskContractError(f"unknown preflight status {self.status!r}; known {sorted(PREFLIGHT_STATUSES)}")
        if self.status == "ok" and self.resolved is None:
            raise TaskContractError("an ok preflight must carry the resolved geometry")
        if self.status != "ok" and self.resolved is not None:
            raise TaskContractError(f"a {self.status} preflight must not carry a resolved geometry")

    def document(self) -> dict[str, Any]:
        """@brief 预检的 JSON 文档（不含已求值的几何）。

        @return 字段字典。
        """
        return {"status": self.status, "detail": self.detail}


#: A runner-owned preflight: it resolves a projected provider's declared geometry against the
#: real instance/ABI/limits.  Dispatch only consumes the decision.
#: A runner-supplied CPU preflight.  It is called as ``hook(geometry)`` for a provider whose
#: geometry the portfolio freezes, and — when the resolver opts in with ``source_preflight`` —
#: as ``hook(geometry_or_none, provider=<key>)`` so a provider that owns its launch in the source
#: can still be preflighted before any build.
PreflightHook = Callable[..., GeometryPreflight]


def preflight_geometry(
    geometry: ExecutionGeometry,
    *,
    instance: SemanticInstanceV2,
    contract: TensorMemoryContract,
    limits: TargetAbiLimits,
    where: str = "portfolio_preflight",
) -> GeometryPreflight:
    """@brief ``resolve_geometry`` 的纯 CPU 预检，并把资源超限与普通非法几何分开。

    This is the runner's pre-``evaluate_task`` gate: a variant that cannot fit
    :class:`TargetAbiLimits` must be diverted *before* a build, and only a declared
    ``resource_unavailable`` fallback may take over.  A malformed geometry is an invalid
    candidate, not a resource switch.

    @param geometry 候选/fallback 声明的几何。
    @param instance 语义实例。
    @param contract 派生 ABI。
    @param limits 目标上限。
    @param where 诊断前缀。
    @return 预检结果。
    """
    try:
        resolved = resolve_geometry(geometry, instance=instance, contract=contract, limits=limits, where=where)
    except GeometryError as exc:
        text = str(exc)
        status = "resource_unavailable" if any(marker in text for marker in RESOURCE_LIMIT_MARKERS) else "invalid"
        return GeometryPreflight(status=status, detail=text)
    return GeometryPreflight(status="ok", resolved=resolved)


@dataclass(frozen=True)
class GuardOutcome:
    """@brief 一个 variant guard 的求值记录。"""

    variant_id: str
    guard_digest: str
    result: str
    detail: str | None = None

    def __post_init__(self) -> None:
        """@brief 校验结果词。

        @raises TaskContractError 非法。
        """
        if self.result not in ("true", "false", "unknown"):
            raise TaskContractError(f"unknown guard result {self.result!r}")
        _require_digest(self.variant_id, where="guard outcome variant_id")
        _require_digest(self.guard_digest, where="guard outcome guard_digest")
        if self.detail is not None and len(self.detail) > MAX_RECORD_DETAIL:
            raise TaskContractError(
                f"guard outcome detail is longer than {MAX_RECORD_DETAIL} characters; the record schema cannot carry it"
            )

    def document(self) -> dict[str, Any]:
        """@brief 记录的 JSON 文档。

        @return 字段字典。
        """
        return {
            "variant_id": self.variant_id,
            "guard_digest": self.guard_digest,
            "result": self.result,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class PortfolioDispatchRecord:
    """@brief 组合层一次 exact dispatch 的记录（与单 stage 的 runtime 记录不同）。"""

    domain_plan_digest: str
    portfolio_digest: str
    request: Mapping[str, Any] | None
    outcome: str
    provider: str | None = None
    variant_id: str | None = None
    fallback_id: str | None = None
    fallback_reason: str | None = None
    detail: str | None = None
    guards: tuple[GuardOutcome, ...] = ()
    evidence: Mapping[str, str] = field(default_factory=dict)
    preflight: GeometryPreflight = field(default_factory=lambda: GeometryPreflight("not_required"))
    schema_id: str = DISPATCH_RECORD_SCHEMA_ID
    record_version: int = DISPATCH_RECORD_VERSION
    dispatch_rules_version: int = DISPATCH_RULES_VERSION

    def __post_init__(self) -> None:
        """@brief 校验记录的一致性。

        @raises TaskContractError 非法。
        """
        if self.schema_id != DISPATCH_RECORD_SCHEMA_ID:
            raise TaskContractError(f"unsupported dispatch record schema {self.schema_id!r}")
        if self.record_version != DISPATCH_RECORD_VERSION:
            raise TaskContractError(f"unsupported dispatch record version {self.record_version!r}")
        if self.dispatch_rules_version != DISPATCH_RULES_VERSION:
            raise TaskContractError(f"unsupported dispatch rules version {self.dispatch_rules_version!r}")
        _require_digest(self.domain_plan_digest, where="dispatch record domain_plan_digest")
        _require_digest(self.portfolio_digest, where="dispatch record portfolio_digest")
        for label, value in (
            ("detail", self.detail),
            ("preflight detail", self.preflight.detail),
        ):
            if value is not None and len(value) > MAX_RECORD_DETAIL:
                raise TaskContractError(
                    f"dispatch record {label} is longer than {MAX_RECORD_DETAIL} characters; the registered schema "
                    "cannot carry it"
                )
        for guard in self.guards:
            if guard.detail is not None and len(guard.detail) > MAX_RECORD_DETAIL:
                raise TaskContractError("a guard detail is longer than the registered record schema allows")
        if self.outcome not in OUTCOMES:
            raise TaskContractError(f"unknown dispatch outcome {self.outcome!r}")
        if self.provider is not None and self.provider not in PROVIDERS:
            raise TaskContractError(f"unknown dispatch provider {self.provider!r}")
        if self.fallback_reason is not None and self.fallback_reason not in FALLBACK_REASONS:
            raise TaskContractError(f"unknown fallback reason {self.fallback_reason!r}")
        for name in ("variant_id", "fallback_id"):
            value = getattr(self, name)
            if value is not None:
                _require_digest(value, where=f"dispatch record {name}")
        if self.provider == "candidate":
            if self.outcome != "resolved":
                raise TaskContractError(f"provider=candidate must be a resolved outcome, got {self.outcome!r}")
            if self.variant_id is None or self.fallback_id is not None:
                raise TaskContractError("a candidate outcome records the variant and no fallback")
        elif self.provider == "fallback":
            if self.outcome != "fallback":
                raise TaskContractError(f"provider=fallback must be a fallback outcome, got {self.outcome!r}")
            if self.fallback_id is None:
                raise TaskContractError("a fallback outcome records the fallback identity")
            if self.variant_id is not None and self.fallback_reason is None:
                raise TaskContractError("a variant-triggered fallback records the derived reason")
        else:
            if self.outcome in ("resolved", "fallback"):
                raise TaskContractError(f"a {self.outcome} outcome needs a provider")
        if self.request is not None and not isinstance(self.request, Mapping):
            raise TaskContractError("a dispatch record request is a mapping or None")
        for key, status in self.evidence.items():
            _require_digest(key, where="dispatch record evidence key")
            if status not in EVIDENCE_STATUSES:
                raise TaskContractError(f"unknown evidence status {status!r}")
        if not isinstance(self.preflight, GeometryPreflight):
            raise TaskContractError("a dispatch record needs a GeometryPreflight")

    def document(self) -> dict[str, Any]:
        """@brief 记录的 JSON 文档。

        @return 字段字典。
        """
        return {
            "schema_id": self.schema_id,
            "record_version": self.record_version,
            "dispatch_rules_version": self.dispatch_rules_version,
            "domain_plan_digest": self.domain_plan_digest,
            "portfolio_digest": self.portfolio_digest,
            "request": None if self.request is None else dict(self.request),
            "outcome": self.outcome,
            "provider": self.provider,
            "variant_id": self.variant_id,
            "fallback_id": self.fallback_id,
            "fallback_reason": self.fallback_reason,
            "detail": self.detail,
            "guards": [guard.document() for guard in self.guards],
            "evidence": {key: self.evidence[key] for key in sorted(self.evidence)},
            "preflight": self.preflight.document(),
        }

    @property
    def identity(self) -> str:
        """@brief 记录身份。

        @return 十六进制 sha256。
        """
        return fingerprint(self.document())


@dataclass(frozen=True)
class _BoundRequest:
    """@brief 一个已与权威实例绑定的请求。"""

    point: SemanticDomainInstance
    instance: SemanticInstanceV2
    names: Mapping[str, int]


@dataclass(frozen=True)
class ExactDispatchResolver:
    """@brief 有限语义域上的 exact resolver（顺序固定，不做优先级或 SMT 推理）。"""

    domain_plan: SemanticDomainPlan
    portfolio: PortfolioCandidate
    evidence: Mapping[str, str]
    #: The authoritative OpSpec, target profile and the instances derived from them.  The plan
    #: must be exactly the plan normalized from these; dispatch never trusts a plan's declared
    #: opspec/ABI/target-generation identity any more than a request's declared digests.
    spec: OpSpec
    target: TargetProfile
    instances: tuple[SemanticInstanceV2, ...]
    max_buffer_length: int | None = None
    preflight: PreflightHook | None = None
    #: Whether the preflight hook also covers providers that do **not** freeze a geometry entry
    #: (``msl-single`` sources own their launch).  Off by default, which is exactly P0's
    #: behaviour: without a source-aware hook such a provider records ``not_required``.
    source_preflight: bool = False

    def __post_init__(self) -> None:
        """@brief 校验域绑定、权威实例、证据状态与证据键。

        @raises TaskContractError 非法。
        """
        if self.portfolio.domain_plan_digest != self.domain_plan.digest:
            raise TaskContractError(
                "the portfolio and the semantic domain plan do not share a domain; dispatch may not borrow another "
                "domain's points"
            )
        if not isinstance(self.instances, tuple) or not self.instances:
            raise TaskContractError(
                "an exact resolver needs the authoritative SemanticInstance for every domain point; a request may not "
                "be accepted from its self-declared op/symbols/attrs/digests alone"
            )
        by_size: dict[tuple[str, str], SemanticInstanceV2] = {}
        for instance in self.instances:
            if not isinstance(instance, SemanticInstanceV2):
                raise TaskContractError("the resolver's authoritative instances must be SemanticInstanceV2 values")
            key = (instance.dtype, instance.size)
            if key in by_size:
                raise TaskContractError(f"the resolver repeats the authoritative instance {key}")
            by_size[key] = instance
        expected = {(point.dtype, point.size) for point in self.domain_plan.instances}
        if set(by_size) != expected:
            raise TaskContractError(
                "the authoritative instance set must be exactly the domain plan's (dtype, size) points; "
                f"missing={sorted(expected - set(by_size))} extra={sorted(set(by_size) - expected)}"
            )
        for point in self.domain_plan.instances:
            instance = by_size[(point.dtype, point.size)]
            if semantic_instance_digest(instance) != point.semantic_instance_digest:
                raise TaskContractError(
                    f"{point.instance_key}: the supplied authoritative instance does not hash to the domain point's "
                    "semantic_instance_digest"
                )
        require_authoritative_domain_plan(
            self.domain_plan,
            self.instances,
            spec=self.spec,
            target=self.target,
            max_buffer_length=self.max_buffer_length,
            where="exact resolver",
        )
        # The guard name space is validated once here, so resolve() can never fail on an
        # ambiguous symbol/attr name; a collision is a broken instance, not a dispatch outcome.
        for instance in self.instances:
            instance_names(instance, where=f"{instance.dtype}/{instance.size}")
        known = set(self.portfolio.variant_ids)
        if self.portfolio.fallback_id is not None:
            known.add(self.portfolio.fallback_id)
        for key, status in self.evidence.items():
            _require_digest(key, where="dispatch evidence key")
            if status not in EVIDENCE_STATUSES:
                raise TaskContractError(f"unknown evidence status {status!r} for {key}")
            if key not in known:
                raise TaskContractError(
                    f"dispatch evidence names {key}, which is not a variant or fallback of this portfolio; stale or "
                    "borrowed evidence is refused"
                )

    def resolve(
        self,
        request: DispatchRequest | Mapping[str, Any],
        *,
        preflight: PreflightHook | None = None,
    ) -> PortfolioDispatchRecord:
        """@brief 解析一个请求。

        @param request 规范化请求或其闭集文档。
        @param preflight 本次运行的几何预检钩子；缺省用 resolver 自身的钩子。
        @return 决策记录。
        """
        hook = self.preflight if preflight is None else preflight
        parsed: DispatchRequest | None
        if isinstance(request, DispatchRequest):
            parsed = request
        else:
            try:
                parsed = DispatchRequest.parse(request)
            except TaskContractError as exc:
                return self._record(
                    outcome="semantic_unresolved",
                    request=None,
                    detail=f"the request cannot be parsed into a semantic instance: {exc}",
                )
        assert parsed is not None
        bound = self._bind_request(parsed)
        if isinstance(bound, PortfolioDispatchRecord):
            return bound
        names = dict(bound.names)

        guards: list[GuardOutcome] = []
        unknown: list[str] = []
        for variant in self.portfolio.variants:
            try:
                hit = variant.guard.evaluate(names, where=f"variant {variant.variant_key}")
            except GeometryError as exc:
                guards.append(
                    GuardOutcome(
                        variant_id=variant.variant_id,
                        guard_digest=variant.guard.digest,
                        result="unknown",
                        detail=str(exc),
                    )
                )
                unknown.append(f"{variant.variant_key}: {exc}")
                continue
            guards.append(
                GuardOutcome(
                    variant_id=variant.variant_id,
                    guard_digest=variant.guard.digest,
                    result="true" if hit else "false",
                )
            )
        if unknown:
            return self._record(
                outcome="dispatch_unknown",
                request=parsed.document(),
                guards=tuple(guards),
                detail="a guard is not decidable for this request: " + "; ".join(unknown),
            )

        matches = [
            variant
            for variant, outcome in zip(self.portfolio.variants, guards)
            if outcome.result == "true"
        ]
        unvalidated = [variant for variant in matches if self.evidence.get(variant.variant_id) != "validated"]
        if unvalidated:
            return self._record(
                outcome="dispatch_unknown",
                request=parsed.document(),
                guards=tuple(guards),
                detail=(
                    "the matched variant(s) "
                    + ", ".join(sorted(variant.variant_key for variant in unvalidated))
                    + " have no validated evidence; a declared guard is not a verified variant"
                ),
            )
        if len(matches) > 1:
            return self._record(
                outcome="dispatch_overlap",
                request=parsed.document(),
                guards=tuple(guards),
                detail=(
                    "multiple variants match this request ("
                    + ", ".join(sorted(variant.variant_key for variant in matches))
                    + "); exact dispatch refuses to rank them"
                ),
            )
        if len(matches) == 1:
            return self._resolve_variant(matches[0], parsed, tuple(guards), hook)
        return self._resolve_fallback(
            reason="outside_candidate_coverage",
            parsed=parsed,
            guards=tuple(guards),
            hook=hook,
            detail="every variant guard is false for this instance",
            matched=None,
            preflight=GeometryPreflight("not_required"),
        )

    # -- internals ---------------------------------------------------------------------------

    def _bind_request(self, request: DispatchRequest) -> "_BoundRequest | PortfolioDispatchRecord":
        """@brief 把请求绑定到有限域点与权威实例；任何矛盾都拒绝（永不 fallback）。

        A request names a point and claims an op/shape/attrs/digest identity.  Dispatch does not
        trust that claim: it looks the point up in the finite plan, takes the authoritative
        ``SemanticInstanceV2`` for that point and requires every declared field to agree.  A
        contradiction means the document cannot be resolved to a semantic instance
        (``semantic_unresolved``); a coherent point that is not named by the plan is
        ``unsupported_contract_capability`` and never reaches the fallback.

        @param request 已通过闭集解析的请求。
        @return 绑定结果或拒绝记录。
        """
        point = self.domain_plan.find(request.dtype, request.size, request.usage)
        if point is None:
            return self._record(
                outcome="unsupported_contract_capability",
                request=request.document(),
                detail=(
                    f"{request.instance_key} is not a named point of the finite semantic domain "
                    f"({list(self.domain_plan.keys)}); the domain is not extended by proximity"
                ),
            )
        instance = next(
            item for item in self.instances if (item.dtype, item.size) == (request.dtype, request.size)
        )
        names = instance_names(instance, where=point.instance_key)
        authoritative_symbols = {str(name): int(value) for name, value in instance.symbols.items()}
        authoritative_attrs = {
            str(name): int(value)
            for name, value in instance.resolved_attrs.items()
            if isinstance(value, int) and not isinstance(value, bool)
        }
        problems: list[str] = []
        if request.op_name != instance.op_name:
            problems.append(f"op_name {request.op_name!r} is not the instance's {instance.op_name!r}")
        if dict(request.symbols) != authoritative_symbols:
            problems.append(
                f"symbols {dict(request.symbols)} are not the instance's {authoritative_symbols}"
            )
        if dict(request.attrs) != authoritative_attrs:
            problems.append(f"attrs {dict(request.attrs)} are not the instance's integer attrs {authoritative_attrs}")
        if request.semantic_instance_digest != point.semantic_instance_digest:
            problems.append("semantic_instance_digest is not the domain point's semantic instance digest")
        if request.target_abi_digest != point.target_abi_digest:
            problems.append("target_abi_digest is not the domain point's derived ABI digest")
        if problems:
            return self._record(
                outcome="semantic_unresolved",
                request=request.document(),
                detail=(
                    "the request contradicts the authoritative semantic instance for "
                    f"{point.instance_key}: " + "; ".join(problems)
                ),
            )
        return _BoundRequest(point=point, instance=instance, names=names)

    def _run_preflight(
        self,
        geometry: ExecutionGeometry | None,
        hook: PreflightHook | None,
        *,
        where: str,
        provider: str,
    ) -> GeometryPreflight:
        """@brief 对选中 provider 的几何运行 runner 预检。

        A provider that freezes its geometry in the portfolio is preflighted with that geometry.
        A provider that owns its launch in the source (``msl-single``) has no geometry here, so the
        default answer stays ``not_required``; a resolver built with ``source_preflight`` instead
        asks the hook to resolve **that provider's** source launch geometry for the requested
        instance, which makes a source-owned launch over the target limits a resource reason
        rather than a late build failure.

        @param geometry 声明几何（可能为 None）。
        @param hook 预检钩子。
        @param where 诊断位置。
        @param provider provider 的稳定键（variant key 或 fallback id）。
        @return 预检结果。
        @raises TaskContractError 需要预检却没有钩子。
        """
        if geometry is None:
            if not self.source_preflight:
                return GeometryPreflight(
                    "not_required", detail=f"{where} has no explicit geometry; the source owns its launch"
                )
            if hook is None:
                raise TaskContractError(
                    f"{where}: a source-owned launch geometry must be preflighted with the backend resolver before "
                    "evaluate_task; dispatch refuses to select it without the CPU preflight"
                )
            return hook(None, provider=provider)
        if hook is None:
            raise TaskContractError(
                f"{where}: a provider with a declared geometry must be preflighted with resolve_geometry before "
                "evaluate_task; dispatch refuses to select it without the CPU preflight"
            )
        return hook(geometry, provider=provider) if self.source_preflight else hook(geometry)

    def _resolve_variant(
        self,
        variant: PortfolioVariant,
        request: DispatchRequest,
        guards: tuple[GuardOutcome, ...],
        hook: PreflightHook | None,
    ) -> PortfolioDispatchRecord:
        """@brief 恰好一个 variant 命中：预检后选择，或按声明的原因转 fallback。

        @param variant 命中的 variant。
        @param request 请求。
        @param guards guard 记录。
        @param hook 预检钩子。
        @return 记录。
        """
        preflight = self._run_preflight(
            variant.geometry, hook, where=f"variant {variant.variant_key}", provider=variant.variant_key
        )
        if preflight.status in ("ok", "not_required"):
            return self._record(
                outcome="resolved",
                request=request.document(),
                provider="candidate",
                variant_id=variant.variant_id,
                guards=guards,
                preflight=preflight,
            )
        if preflight.status == "resource_unavailable":
            if self._fallback_declares("resource_unavailable"):
                return self._resolve_fallback(
                    reason="resource_unavailable",
                    parsed=request,
                    guards=guards,
                    hook=hook,
                    detail=(
                        f"variant {variant.variant_key} exceeds the target limits: {preflight.detail}; the request is "
                        "outside what this variant can run"
                    ),
                    matched=variant,
                    preflight=preflight,
                )
            return self._record(
                outcome="variant_unavailable",
                request=request.document(),
                variant_id=variant.variant_id,
                guards=guards,
                preflight=preflight,
                detail=(
                    f"variant {variant.variant_key} exceeds the target limits ({preflight.detail}) and the domain "
                    "fallback does not declare resource_unavailable; the variant fails instead of running a fallback "
                    "it did not register"
                ),
            )
        return self._record(
            outcome="variant_unavailable",
            request=request.document(),
            variant_id=variant.variant_id,
            guards=guards,
            preflight=preflight,
            detail=(
                f"variant {variant.variant_key} declares an invalid geometry ({preflight.detail}); an invalid launch "
                "is a candidate defect, not a resource switch"
            ),
        )

    def _fallback_declares(self, reason: str) -> bool:
        """@brief fallback 是否声明了该触发原因。

        @param reason 原因。
        @return 声明时为 True。
        """
        fallback = self.portfolio.fallback
        return fallback is not None and reason in fallback.reasons

    def _resolve_fallback(
        self,
        *,
        reason: str,
        parsed: DispatchRequest,
        guards: tuple[GuardOutcome, ...],
        hook: PreflightHook | None,
        detail: str,
        matched: PortfolioVariant | None,
        preflight: GeometryPreflight,
    ) -> PortfolioDispatchRecord:
        """@brief 零匹配或资源不可用时的 fallback 选择（原因由执行阶段派生）。

        @param reason 派生出的原因。
        @param parsed 请求。
        @param guards guard 记录。
        @param hook 预检钩子。
        @param detail 人类可读说明。
        @param matched 之前命中的 variant（资源原因时）。
        @param preflight 该 variant 的预检。
        @return 记录。
        """
        fallback = self.portfolio.fallback
        if fallback is None or reason not in fallback.reasons:
            return self._record(
                outcome="dispatch_unmatched",
                request=parsed.document(),
                variant_id=None if matched is None else matched.variant_id,
                guards=guards,
                preflight=preflight,
                detail=(
                    f"{detail}; no domain fallback declares {reason!r}"
                    + ("" if fallback is None else f" (declared: {list(fallback.reasons)})")
                ),
            )
        if fallback.kind == "refuse":
            return self._record(
                outcome="dispatch_unmatched",
                request=parsed.document(),
                variant_id=None if matched is None else matched.variant_id,
                fallback_reason=reason,
                guards=guards,
                preflight=preflight,
                detail=f"{detail}; the candidate explicitly refuses outside its coverage",
            )
        if self.evidence.get(fallback.fallback_id) != "validated":
            return self._record(
                outcome="dispatch_unknown",
                request=parsed.document(),
                variant_id=None if matched is None else matched.variant_id,
                fallback_id=fallback.fallback_id,
                fallback_reason=reason,
                guards=guards,
                preflight=preflight,
                detail=(
                    f"{detail}; the domain fallback's independent evidence is not validated, so its correctness is "
                    "not decidable"
                ),
            )
        fallback_preflight = self._run_preflight(
            fallback.geometry, hook, where="the domain fallback", provider=fallback.fallback_id
        )
        if fallback_preflight.status not in ("ok", "not_required"):
            return self._record(
                outcome="variant_unavailable",
                request=parsed.document(),
                variant_id=None if matched is None else matched.variant_id,
                fallback_id=fallback.fallback_id,
                fallback_reason=reason,
                guards=guards,
                preflight=fallback_preflight,
                detail=(
                    f"{detail}; the domain fallback is itself unavailable ({fallback_preflight.detail}); there is no "
                    "chained fallback"
                ),
            )
        return self._record(
            outcome="fallback",
            request=parsed.document(),
            provider="fallback",
            variant_id=None if matched is None else matched.variant_id,
            fallback_id=fallback.fallback_id,
            fallback_reason=reason,
            guards=guards,
            preflight=fallback_preflight,
            detail=detail,
        )

    def _record(self, **fields: Any) -> PortfolioDispatchRecord:
        """@brief 构造记录并带上组合身份。

        @param fields 记录字段。
        @return 记录。
        """
        evidence = {key: str(self.evidence[key]) for key in sorted(self.evidence)}
        if "detail" in fields:
            fields["detail"] = _clamp_detail(fields["detail"])
        guards = fields.get("guards")
        if guards is not None:
            fields["guards"] = tuple(
                GuardOutcome(
                    variant_id=guard.variant_id,
                    guard_digest=guard.guard_digest,
                    result=guard.result,
                    detail=_clamp_detail(guard.detail),
                )
                for guard in guards
            )
        preflight = fields.get("preflight")
        if isinstance(preflight, GeometryPreflight):
            fields["preflight"] = GeometryPreflight(
                status=preflight.status,
                detail=_clamp_detail(preflight.detail),
                resolved=preflight.resolved,
            )
        return PortfolioDispatchRecord(
            domain_plan_digest=self.domain_plan.digest,
            portfolio_digest=self.portfolio.digest,
            evidence=evidence,
            **fields,
        )


__all__ = [
    "DISPATCH_RECORD_SCHEMA_ID",
    "DISPATCH_RECORD_SCHEMA_PATH",
    "DISPATCH_RECORD_VERSION",
    "DISPATCH_RULES_VERSION",
    "MAX_RECORD_DETAIL",
    "MAX_REQUEST_DTYPE",
    "MAX_REQUEST_OP_NAME",
    "MAX_REQUEST_SIZE",
    "DispatchRequest",
    "ExactDispatchResolver",
    "GeometryPreflight",
    "GuardOutcome",
    "OUTCOMES",
    "PREFLIGHT_STATUSES",
    "PROVIDERS",
    "PortfolioDispatchRecord",
    "PreflightHook",
    "REQUEST_FIELDS",
    "RESOURCE_LIMIT_MARKERS",
    "preflight_geometry",
]
