"""Semantic domain plan and the one guard AST (shape portfolio P0).

Three domains are kept apart by the shape-portfolio plan: the **semantic domain** (what the
OpSpec accepts), the **candidate coverage** (what a candidate's guards claim) and the **validated
coverage** (what independent evidence proves).  This module owns the first one and the guard
language both later domains are expressed in:

* :class:`SemanticDomainPlan` is a *finite, explicitly named* set of ``(dtype, size, usage)``
  points of one OpSpec, each with the full semantic-instance and target-ABI identity.  There is
  no shape family, no numeric proximity and no implicit default: a shape that is not a declared
  point of the plan is outside the domain.  Only ``domain_kind=finite`` is activated; a
  ``non_finite`` document is refused instead of being treated as a sampled finite one
  (``validation_kind`` is a later phase concern and never inferred here).
* Guards use the existing :mod:`tcenv.track.expr` functional/JSON AST.  There is **no** second
  infix parser: :func:`canonical_ast` rearranges an already parsed ``expr.Node`` and
  :func:`guard_digest` hashes that canonical AST, never the author's string.  Canonicalization is
  *purely structural*: ``gt(a,b)`` becomes ``lt(b,a)``, ``ge(a,b)`` becomes ``le(b,a)``,
  ``and``/``or`` are flattened and their operands are sorted, and the operands of **every**
  registered commutative function (``eq``, ``ne``, ``+``, ``*``, ``min``, ``max``) are sorted by
  canonical bytes.  Nothing is simplified: ``a-b < 0`` stays ``lt(-(a,b), 0)`` and never becomes
  ``lt(a,b)``, terms are not cancelled, and no SMT/termination claim is made.  The plan is never
  self-certified: :func:`require_authoritative_domain_plan` recomputes it from the authoritative
  OpSpec/target/instances, so a document cannot relabel another OpSpec or a forged ABI digest as
  this domain.

The domain plan reads no candidate tile, no measurement and no hidden shard: it is derived from
the OpSpec instance and the target generation only.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Mapping

from ..contract_errors import GeometryError, TaskContractError, UnsupportedContractCapability
from ..identity import IdentityError, canonical_bytes, fingerprint, is_hex64, load_strict_json
from ..opspec.model import OpSpec, OpSpecError
from ..opspec.v2 import SemanticInstanceV2, instantiate_v2
from ..opspec.validator import SchemaFailure, validate
from ..target import TargetProfile
from ..taskcontract import TaskContract
from ..taskcontract.runtime_shape import resolved_shape_from_inputs, workload_profile_for_case
from ..track.abi import derive_tensor_memory_contract, derived_abi_digest
from ..track.expr import BOOL, Call, Const, Node, Var, evaluate, kind_of, parse_term

SEMANTIC_DOMAIN_PLAN_SCHEMA_ID = "tcenv.semantic-domain-plan.1"
SEMANTIC_DOMAIN_PLAN_NORMALIZATION_VERSION = 1
SCHEMA_DIR = Path(__file__).resolve().parent / "schema"
SEMANTIC_DOMAIN_PLAN_SCHEMA_PATH = SCHEMA_DIR / "semantic-domain-plan.v1.json"

#: The registered domain vocabulary.  Only ``finite`` is activated by this phase.
DOMAIN_KINDS = frozenset({"finite", "non_finite"})
ACTIVE_DOMAIN_KINDS = frozenset({"finite"})
INSTANCE_USAGES = frozenset({"golden", "benchmark"})
GUARD_FIELDS = frozenset({"expr", "digest"})
#: A guard is declared together with the digest of its canonical AST.  The declared value is
#: mandatory and must be reproducible, so a digest of the author's spelling cannot pass as one.
GUARD_DIGEST_KIND = "portfolio-guard"


def _schema() -> Mapping[str, Any]:
    """@brief 读取登记 JSON Schema（闭集子集校验器唯一读取面）。

    @return schema 文档。
    @raises TaskContractError schema 不可读/不可解析。
    """
    return _load_schema(SEMANTIC_DOMAIN_PLAN_SCHEMA_PATH)


@lru_cache(maxsize=8)
def _load_schema(path: Path) -> Mapping[str, Any]:
    """@brief 严格读取一份登记 schema（按路径缓存）。

    @param path schema 路径。
    @return schema 文档。
    @raises TaskContractError 不可读或不是对象。
    """
    try:
        document = load_strict_json(path)
    except IdentityError as exc:
        raise TaskContractError(f"{path}: the registered portfolio schema is not strict JSON: {exc}") from exc
    if not isinstance(document, Mapping):
        raise TaskContractError(f"{path}: a registered portfolio schema must be an object")
    return document


def _require_schema(document: Any, schema: Mapping[str, Any], *, where: str) -> None:
    """@brief 用登记 schema 校验一份文档（未知字段一律拒绝）。

    @param document 文档。
    @param schema 登记 schema。
    @param where 诊断位置。
    @raises TaskContractError 不符合 schema。
    """
    try:
        validate(document, schema, path=where)
    except SchemaFailure as exc:
        raise TaskContractError(f"{where}: {exc}") from exc


def _require_digest(value: Any, *, where: str) -> str:
    """@brief 校验一个 64 位小写十六进制摘要。

    @param value 值。
    @param where 诊断位置。
    @return 原值。
    @raises TaskContractError 不是规范摘要。
    """
    if not is_hex64(value):
        raise TaskContractError(f"{where} must be a 64-character lowercase hex digest, got {value!r}")
    return str(value)


def semantic_instance_digest(instance: SemanticInstanceV2) -> str:
    """@brief 一个 v2 语义实例的完整身份（含 opspec/dtype/size/symbols/tensors/attrs）。

    @param instance 语义实例。
    @return 十六进制 sha256。
    @raises TaskContractError 传入的不是 v2 语义实例。
    """
    if not isinstance(instance, SemanticInstanceV2):
        raise TaskContractError("semantic_instance_digest needs a SemanticInstanceV2")
    return fingerprint(
        {
            "fingerprint_kind": "semantic-instance",
            "fingerprint_version": 2,
            "instance": instance.fingerprint_document(),
        }
    )


def workload_profile_digest(instance: SemanticInstanceV2) -> str:
    """@brief 一个 L2 实例已声明的 workload-profile 集合身份（不绑定单次调用）。

    The semantic instance digest of an L2 tier binds the capacity ABI and the declared profile set; this
    digest names that set on its own so an evaluation record can say *which declared benchmark
    distribution family* it ran without re-deriving the instance.  Per-call resolved values stay out of
    it on purpose: they belong to ``evaluation_input_digest``/launch evidence.

    @param instance 语义实例。
    @return 十六进制 sha256。
    @raises TaskContractError 传入的不是变长实例。
    """
    if not isinstance(instance, SemanticInstanceV2) or not instance.is_capacity:
        raise TaskContractError("workload_profile_digest needs an L2 (capacity) SemanticInstanceV2")
    return fingerprint(
        {
            "fingerprint_kind": "workload-profile-set",
            "fingerprint_version": 1,
            "semantic_instance_digest": semantic_instance_digest(instance),
            "profiles": [
                profile.document() for profile in sorted(instance.workload_profiles, key=lambda item: item.id)
            ],
        }
    )


def evaluation_input_digest(
    instance: SemanticInstanceV2,
    case: str,
    seed: int,
    inputs: Mapping[str, Any],
) -> str:
    """@brief 一次评测 trial 的输入身份：声明的 workload profile + 实际边界字节 + 张量形态。

    Three identities are kept apart (§2.1 of the varlen plan):

    * ``semantic_instance_digest`` binds the ABI and, for an L1 instance, the *canonical*
      ``cu_seqlens`` — that is the domain point;
    * ``workload_profile_digest`` binds the declared benchmark profile set (L2);
    * this digest binds **what a trial actually ran**: the case name, the input seed, the
      identity (name/dtype/shape) of every reference input and the *actual* ``cu_seqlens``
      bytes — plus, for L2, the declared workload profile of that case and the resolved
      total/count read back from those bytes.

    Permuted/skewed/fresh segmentations are evaluation cases of one domain point, not new
    domain points.  They are auditable and replayable through this digest instead of through a
    forged instance identity: regenerating ``(instance, case, seed)`` reproduces the same
    digest, while changing the boundaries of an otherwise identical input set changes it.

    **Scope, stated exactly:** this is a *boundary + identity* digest, not a byte-complete hash of
    every input.  Only ``cu_seqlens`` bytes are hashed; the packed payloads of a real-size batch are
    hundreds of megabytes and are fully determined by ``(instance, case, seed)``, so hashing them would
    cost a device copy without adding identity.  A payload change that keeps the same shape and
    boundaries is therefore *not* visible in this digest; it is caught by the value comparison and the
    padding poison check instead.  An L2 fingerprint says so explicitly (``identity_scope``) so no
    evidence reader can mistake the digest for a complete content hash; an L1 fingerprint keeps its
    historical bytes (adding a key there would silently re-identify already-recorded V1 evidence).

    For an L2 capacity instance the declared workload profile is bound **and checked**: the case's
    profile fixes the expected ``total_tokens``/``num_seqs``, the actual boundaries have to derive the
    same total/count (a generator cannot quietly run a different batch under this case name), and both
    the profile id and its canonical ``seq_lens`` enter the digest.  The resolved values themselves are
    still read from the bytes, never from the profile.

    @param instance 语义实例。
    @param case 评测用例名。
    @param seed 本次输入种子。
    @param inputs 本次实际输入映射（须与该实例的 reference 初值合同闭合）。
    @return 十六进制 sha256。
    @raises TaskContractError 形态非法或引用了未声明 case/输入。
    @raises VarlenContractError L2 的实际 total/count 与声明的 workload profile 不一致。
    """
    if not isinstance(instance, SemanticInstanceV2):
        raise TaskContractError("evaluation_input_digest needs a SemanticInstanceV2")
    if not isinstance(case, str) or not case:
        raise TaskContractError(f"evaluation_input_digest.case must be a non-empty string, got {case!r}")
    if case not in instance.case_domain.cases:
        raise TaskContractError(
            f"evaluation_input_digest: case {case!r} is not declared; declared cases are {list(instance.case_domain.cases)}"
        )
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TaskContractError(f"evaluation_input_digest.seed must be an integer, got {seed!r}")
    if not isinstance(inputs, Mapping):
        raise TaskContractError("evaluation_input_digest.inputs must be a mapping of tensor name -> tensor")
    expected = set(instance.reference_inputs)
    missing = sorted(expected - set(inputs))
    extra = sorted(set(inputs) - expected)
    if missing or extra:
        raise TaskContractError(
            f"evaluation_input_digest: inputs do not match the reference input contract; missing={missing} extra={extra}"
        )
    tensors: list[dict[str, Any]] = []
    for name in sorted(expected):
        value = inputs[name]
        tensors.append(
            {
                "name": name,
                "dtype": str(getattr(value, "dtype", "")),
                "shape": [int(dim) for dim in getattr(value, "shape", ())],
            }
        )
    boundaries: dict[str, Any] | None = None
    cu = inputs.get("cu_seqlens")
    if cu is not None:
        boundaries = {
            "dtype": str(getattr(cu, "dtype", "")),
            "count": int(cu.numel()),
            "sha256": _tensor_bytes_sha256(cu),
        }
    document: dict[str, Any] = {
        "fingerprint_kind": "evaluation-input",
        "fingerprint_version": 1,
        "semantic_instance_digest": semantic_instance_digest(instance),
        "case": case,
        "seed": int(seed),
        "tensors": tensors,
        "cu_seqlens": boundaries,
    }
    if instance.is_capacity:
        # L2: the semantic identity only binds the capacity ABI, so the *declared workload profile* of
        # this case and the resolved shape of this call have to enter the input identity explicitly.
        # Without them two different runtime batches of one capacity instance could share a digest.
        profile = workload_profile_for_case(instance, case)
        resolved = resolved_shape_from_inputs(instance, inputs)
        if (resolved.total_tokens, resolved.num_seqs) != (profile.total_tokens, profile.num_seqs):
            raise TaskContractError(
                f"evaluation_input_digest: case {case!r} declares the workload profile {profile.id!r} "
                f"[{profile.total_tokens} tokens / {profile.num_seqs} sequences], but the actual cu_seqlens "
                f"describe [{resolved.total_tokens} tokens / {resolved.num_seqs} sequences]; the batch is read "
                "from the bytes and must be the declared one"
            )
        # The digest covers the boundary bytes, the shape metadata and the declared profile; the packed
        # payload is checked by the value comparison and the padding contract, not hashed here.  Stating
        # that scope is part of the identity, so no evidence reader can mistake it for a byte-complete
        # content hash.  L1 keeps its historical fingerprint bytes (no new key).
        document["identity_scope"] = "boundary-bytes+shape-metadata+declared-profile"
        document["workload_profile_digest"] = workload_profile_digest(instance)
        # The declared profile (id + canonical seq_lens) is bound explicitly; the resolved total/count are
        # the byte-derived values, so a replay cannot relabel a different batch as this profile.
        document["workload_profile_id"] = profile.id
        document["declared_profile"] = profile.document()
        document["resolved"] = {
            "total_tokens": int(resolved.total_tokens),
            "num_seqs": int(resolved.num_seqs),
            "cu_seqlens_prefix_sha256": boundaries["sha256"] if boundaries is not None else None,
        }
    return fingerprint(document)


def _tensor_bytes_sha256(value: Any) -> str:
    """@brief 张量内容（精确位型）的 SHA-256。

    @param value torch 张量（或带 ``numpy()`` 的数组）。
    @return 十六进制 sha256。
    """
    raw = value.detach().to("cpu").contiguous().numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def target_generation_digest(target: TargetProfile) -> str:
    """@brief 目标代际身份（设备档案 + 后端 + ABI 上限，不含测量结果）。

    @param target 目标档案。
    @return 十六进制 sha256。
    """
    if not isinstance(target, TargetProfile):
        raise TaskContractError("target_generation_digest needs a TargetProfile")
    return fingerprint(
        {
            "fingerprint_kind": "target-generation",
            "fingerprint_version": 1,
            "target_id": target.id,
            "backend_id": target.backend,
            "backend_version": target.backend_version,
            "limits": {str(key): value for key, value in dict(target.raw_limits).items()},
        }
    )


def instance_key(dtype: str, size: str, usage: str) -> str:
    """@brief 一个有限域点的稳定命名（dtype/size/usage）。

    @param dtype dtype。
    @param size size 档位名。
    @param usage ``golden``/``benchmark``。
    @return ``"<dtype>/<size>/<usage>"``。
    @raises TaskContractError 任一分量非法。
    """
    for value, label in ((dtype, "dtype"), (size, "size"), (usage, "usage")):
        if not isinstance(value, str) or not value or "/" in value:
            raise TaskContractError(f"instance_key.{label} must be a non-empty string without '/', got {value!r}")
    if usage not in INSTANCE_USAGES:
        raise TaskContractError(f"instance_key usage must be one of {sorted(INSTANCE_USAGES)}, got {usage!r}")
    return f"{dtype}/{size}/{usage}"


@dataclass(frozen=True)
class SemanticDomainInstance:
    """@brief 一个有限语义域点：命名实例 + 语义身份 + ABI 身份。"""

    instance_key: str
    dtype: str
    size: str
    usage: str
    semantic_instance_digest: str
    target_abi_digest: str

    def __post_init__(self) -> None:
        """@brief 校验命名与摘要。

        @raises TaskContractError 非法。
        """
        expected = instance_key(self.dtype, self.size, self.usage)
        if self.instance_key != expected:
            raise TaskContractError(
                f"SemanticDomainInstance.instance_key {self.instance_key!r} is not the canonical key {expected!r}"
            )
        if self.usage not in INSTANCE_USAGES:
            raise TaskContractError(f"{self.instance_key}: unknown usage {self.usage!r}")
        _require_digest(self.semantic_instance_digest, where=f"{self.instance_key}.semantic_instance_digest")
        _require_digest(self.target_abi_digest, where=f"{self.instance_key}.target_abi_digest")

    def document(self) -> dict[str, Any]:
        """@brief 该域点的 JSON 文档。

        @return 字段字典。
        """
        return {
            "instance_key": self.instance_key,
            "dtype": self.dtype,
            "size": self.size,
            "usage": self.usage,
            "semantic_instance_digest": self.semantic_instance_digest,
            "target_abi_digest": self.target_abi_digest,
        }


@dataclass(frozen=True)
class SemanticDomainPlan:
    """@brief 一个 OpSpec 的有限语义域（不含候选 tile，也不含事后隐藏分片）。"""

    opspec_digest: str
    target_generation_digest: str
    domain_kind: str
    instances: tuple[SemanticDomainInstance, ...]
    normalization_version: int = SEMANTIC_DOMAIN_PLAN_NORMALIZATION_VERSION
    schema_id: str = SEMANTIC_DOMAIN_PLAN_SCHEMA_ID

    def __post_init__(self) -> None:
        """@brief 校验域种类、摘要与域点唯一性；只激活有限域。

        @raises UnsupportedContractCapability ``domain_kind`` 不是有限域。
        @raises TaskContractError 非法文档。
        """
        if self.schema_id != SEMANTIC_DOMAIN_PLAN_SCHEMA_ID:
            raise TaskContractError(f"unsupported semantic domain plan schema {self.schema_id!r}")
        if self.normalization_version != SEMANTIC_DOMAIN_PLAN_NORMALIZATION_VERSION:
            raise TaskContractError(
                f"unsupported semantic domain normalization version {self.normalization_version!r}"
            )
        if self.domain_kind not in DOMAIN_KINDS:
            raise TaskContractError(
                f"unknown domain_kind {self.domain_kind!r}; registered kinds are {sorted(DOMAIN_KINDS)}"
            )
        if self.domain_kind not in ACTIVE_DOMAIN_KINDS:
            raise UnsupportedContractCapability(
                f"domain_kind={self.domain_kind!r} is not activated: only a finite, explicitly named OpSpec "
                "instance pool may route a portfolio; sampled/non-finite coverage stays refused"
            )
        _require_digest(self.opspec_digest, where="semantic domain plan opspec_digest")
        _require_digest(self.target_generation_digest, where="semantic domain plan target_generation_digest")
        if not isinstance(self.instances, tuple) or not self.instances:
            raise TaskContractError("a semantic domain plan needs at least one named instance")
        ordered = tuple(sorted(self.instances, key=lambda item: (item.dtype, item.size, item.usage)))
        if any(not isinstance(item, SemanticDomainInstance) for item in ordered):
            raise TaskContractError("semantic domain plan instances must be SemanticDomainInstance values")
        seen: dict[str, SemanticDomainInstance] = {}
        for item in ordered:
            if item.instance_key in seen:
                raise TaskContractError(f"semantic domain plan repeats the instance {item.instance_key!r}")
            seen[item.instance_key] = item
        # Canonical order is part of normalization: the digest must not depend on author order.
        object.__setattr__(self, "instances", ordered)

    def document(self) -> dict[str, Any]:
        """@brief 规范化后的域计划文档。

        @return 字段字典。
        """
        return {
            "schema_id": self.schema_id,
            "normalization_version": self.normalization_version,
            "opspec_digest": self.opspec_digest,
            "target_generation_digest": self.target_generation_digest,
            "domain_kind": self.domain_kind,
            "instances": [item.document() for item in self.instances],
        }

    @property
    def digest(self) -> str:
        """@brief 域计划身份。

        @return 十六进制 sha256。
        """
        return fingerprint(
            {
                "fingerprint_kind": "semantic-domain-plan",
                "fingerprint_version": 1,
                "plan": self.document(),
            }
        )

    @property
    def keys(self) -> tuple[str, ...]:
        """@brief 全部有限域点的键（规范顺序）。

        @return 键元组。
        """
        return tuple(item.instance_key for item in self.instances)

    def find(self, dtype: str, size: str, usage: str) -> SemanticDomainInstance | None:
        """@brief 按 ``(dtype, size, usage)`` 找域点。

        @param dtype dtype。
        @param size size 档位名。
        @param usage 用途。
        @return 域点或 None。
        """
        wanted = (dtype, size, usage)
        for item in self.instances:
            if (item.dtype, item.size, item.usage) == wanted:
                return item
        return None

    def instance(self, key: str) -> SemanticDomainInstance:
        """@brief 按键取域点。

        @param key 域点键。
        @return 域点。
        @raises KeyError 该键不在有限域内。
        """
        for item in self.instances:
            if item.instance_key == key:
                return item
        raise KeyError(key)


def normalize_semantic_domain_plan(
    spec: OpSpec,
    instances: Iterable[SemanticInstanceV2],
    *,
    target: TargetProfile,
    domain_kind: str = "finite",
    max_buffer_length: int | None = None,
) -> SemanticDomainPlan:
    """@brief 把显式命名的 OpSpec 实例规范化为一个有限语义域计划。

    Every declared ``usage`` of an instance's size becomes its own finite point, so
    ``golden`` and ``benchmark`` are never conflated.  The instance must come from *this* OpSpec:
    a dtype/size outside the document, or an instance assembled from another OpSpec, is
    ``unsupported_contract_capability`` and is never absorbed into the domain by numeric
    proximity.

    @param spec OpSpec。
    @param instances 显式命名的 v2 实例。
    @param target 目标档案（派生 ABI 与代际身份）。
    @param domain_kind 域种类；只接受 ``finite``。
    @param max_buffer_length 设备缓冲长度；CPU 派生时省略。
    @return 规范化域计划。
    @raises UnsupportedContractCapability 域种类未激活或实例超出 OpSpec。
    @raises TaskContractError 重复/非法实例。
    """
    if domain_kind not in DOMAIN_KINDS:
        raise TaskContractError(f"unknown domain_kind {domain_kind!r}; registered kinds are {sorted(DOMAIN_KINDS)}")
    if domain_kind not in ACTIVE_DOMAIN_KINDS:
        raise UnsupportedContractCapability(
            f"domain_kind={domain_kind!r} is not activated; only finite, explicitly named OpSpec instances "
            "are in the semantic domain of this phase"
        )
    if not isinstance(spec, OpSpec):
        raise TaskContractError("normalize_semantic_domain_plan needs an OpSpec")
    type_vars = spec.document.get("type_vars") or {}
    declared_dtypes = {str(name) for values in type_vars.values() for name in values}
    declared_sizes = set(spec.sizes)
    points: list[SemanticDomainInstance] = []
    seen: set[str] = set()
    for instance in instances:
        if not isinstance(instance, SemanticInstanceV2):
            raise TaskContractError("normalize_semantic_domain_plan needs v2 semantic instances")
        if instance.opspec_digest != spec.digest or instance.op_name != spec.op_name:
            raise UnsupportedContractCapability(
                f"{instance.op_name}.{instance.dtype}.{instance.size} was assembled from a different OpSpec "
                f"({instance.opspec_digest}) than the domain plan's ({spec.digest}); the semantic domain has one "
                "specification"
            )
        if instance.dtype not in declared_dtypes or instance.size not in declared_sizes:
            raise UnsupportedContractCapability(
                f"{instance.op_name}.{instance.dtype}.{instance.size} is outside the OpSpec: declared dtypes "
                f"{sorted(declared_dtypes)}, declared sizes {sorted(declared_sizes)}"
            )
        declared_usage = tuple(instance.size_usage)
        if not declared_usage:
            raise UnsupportedContractCapability(f"{instance.size}: the OpSpec declares no usage for this size")
        abi = derive_tensor_memory_contract(instance, target=target, max_buffer_length=max_buffer_length)
        abi_digest = derived_abi_digest(abi)
        semantic_digest = semantic_instance_digest(instance)
        for usage in declared_usage:
            key = instance_key(instance.dtype, instance.size, usage)
            if key in seen:
                raise TaskContractError(f"the semantic domain plan repeats the instance {key!r}")
            seen.add(key)
            points.append(
                SemanticDomainInstance(
                    instance_key=key,
                    dtype=instance.dtype,
                    size=instance.size,
                    usage=usage,
                    semantic_instance_digest=semantic_digest,
                    target_abi_digest=abi_digest,
                )
            )
    if not points:
        raise TaskContractError("a semantic domain plan needs at least one named instance")
    return SemanticDomainPlan(
        opspec_digest=spec.digest,
        target_generation_digest=target_generation_digest(target),
        domain_kind=domain_kind,
        instances=tuple(points),
    )


def require_authoritative_domain_plan(
    domain_plan: SemanticDomainPlan,
    instances: Iterable[SemanticInstanceV2],
    *,
    spec: OpSpec,
    target: TargetProfile,
    max_buffer_length: int | None = None,
    where: str = "semantic domain plan",
) -> SemanticDomainPlan:
    """@brief 用权威 OpSpec/target/instances 重算域计划，并要求传入计划逐字节一致。

    A plan document can be self-consistent and still forged: rewriting ``opspec_digest`` (or a
    per-point ``target_abi_digest``) and re-binding the portfolio to the forged plan digest would
    otherwise let a request for another OpSpec/ABI be dispatched as this domain.  The only
    authority is the OpSpec, the target profile and the instances derived from them, so this
    helper recomputes the normalized plan and refuses anything that is not that plan.  Each
    instance must also be exactly the one :func:`instantiate_v2` derives from the OpSpec.

    @param domain_plan 待验证的域计划。
    @param instances 该计划的显式命名实例。
    @param spec 权威 OpSpec。
    @param target 权威目标档案。
    @param max_buffer_length 设备缓冲长度（影响派生 ABI）。
    @param where 诊断位置。
    @return 权威规范化计划。
    @raises TaskContractError 计划身份、Opspec/ABI 摘要或实例不是权威派生。
    """
    if not isinstance(domain_plan, SemanticDomainPlan):
        raise TaskContractError(f"{where}: needs a SemanticDomainPlan")
    if not isinstance(spec, OpSpec):
        raise TaskContractError(f"{where}: needs the authoritative OpSpec")
    named = tuple(instances)
    for instance in named:
        if not isinstance(instance, SemanticInstanceV2):
            raise TaskContractError(f"{where}: needs v2 semantic instances")
        try:
            derived = instantiate_v2(spec, instance.dtype, instance.size)
        except OpSpecError as exc:
            raise TaskContractError(
                f"{where}: {instance.dtype}/{instance.size} cannot be derived from the authoritative OpSpec: {exc}"
            ) from exc
        if derived != instance:
            raise TaskContractError(
                f"{where}: the supplied {instance.dtype}/{instance.size} instance is not the one the authoritative "
                "OpSpec derives; a hand-built instance may not stand in for the OpSpec's semantics"
            )
    authoritative = normalize_semantic_domain_plan(
        spec, named, target=target, max_buffer_length=max_buffer_length
    )
    if authoritative.digest != domain_plan.digest:
        raise TaskContractError(
            f"{where}: the plan is not the finite plan normalized from the authoritative OpSpec/target/instances; "
            "opspec_digest, target_generation_digest and every point's semantic and derived ABI digest must be "
            "derived, not declared"
        )
    return authoritative


@dataclass(frozen=True)
class DomainAuthority:
    """@brief 与 portfolio/domain 绑定的权威来源：规范化计划 + OpSpec + target + instances。

    Modules that must decide something about the domain (dispatch, projection) take this object
    instead of caller-supplied booleans: the plan is re-normalized from the authority in
    ``__post_init__``, so neither the plan's declared digests nor a "workspace declared" flag can
    be forged.
    """

    plan: SemanticDomainPlan
    spec: OpSpec
    target: TargetProfile
    instances: tuple[SemanticInstanceV2, ...]
    #: Optional full task contract (spec + default instance + reference).  Projection uses it for
    #: the existing HostTilingPlan validation; when present it must be this OpSpec's contract.
    contract: TaskContract | None = None
    max_buffer_length: int | None = None

    def __post_init__(self) -> None:
        """@brief 校验并重算权威计划。

        @raises TaskContractError 计划不是权威派生。
        """
        if not isinstance(self.instances, tuple):
            raise TaskContractError("DomainAuthority needs a tuple of authoritative instances")
        object.__setattr__(
            self,
            "plan",
            require_authoritative_domain_plan(
                self.plan,
                self.instances,
                spec=self.spec,
                target=self.target,
                max_buffer_length=self.max_buffer_length,
                where="domain authority",
            ),
        )
        if self.contract is not None:
            if not isinstance(self.contract, TaskContract):
                raise TaskContractError("DomainAuthority.contract must be a TaskContract")
            if self.contract.spec.digest != self.spec.digest:
                raise TaskContractError(
                    "DomainAuthority.contract belongs to a different OpSpec; the host-tiling authority may not be borrowed"
                )
            digest = semantic_instance_digest(self.contract.instance)
            if not any(point.semantic_instance_digest == digest for point in self.plan.instances):
                raise TaskContractError(
                    "DomainAuthority.contract's instance is not one of the domain plan's points"
                )

    @property
    def workspace_declared(self) -> bool:
        """@brief 权威 OpSpec 是否显式声明 ``space == "workspace"`` 张量。

        This is derived from the OpSpec itself, not from a candidate or caller flag.  A
        workspace tensor is a spec-level fact (dims may depend on ``N``, its existence does not).

        @return 是否存在 workspace 张量。
        """
        tensors = self.spec.document.get("tensors") or {}
        return any(str(tensor.get("space")) == "workspace" for tensor in tensors.values())


def parse_semantic_domain_plan(document: Any, *, where: str = "semantic_domain_plan") -> SemanticDomainPlan:
    """@brief 解析并校验一份语义域计划文档（闭集，未知字段拒绝）。

    @param document 文档。
    @param where 诊断位置。
    @return 域计划。
    @raises UnsupportedContractCapability ``domain_kind`` 未激活。
    @raises TaskContractError 文档非法。
    """
    _require_schema(document, _schema(), where=where)
    instances = tuple(
        SemanticDomainInstance(
            instance_key=str(item["instance_key"]),
            dtype=str(item["dtype"]),
            size=str(item["size"]),
            usage=str(item["usage"]),
            semantic_instance_digest=str(item["semantic_instance_digest"]),
            target_abi_digest=str(item["target_abi_digest"]),
        )
        for item in document["instances"]
    )
    return SemanticDomainPlan(
        opspec_digest=str(document["opspec_digest"]),
        target_generation_digest=str(document["target_generation_digest"]),
        domain_kind=str(document["domain_kind"]),
        instances=instances,
        normalization_version=int(document["normalization_version"]),
        schema_id=str(document["schema_id"]),
    )


def load_semantic_domain_plan(path: str | Path) -> SemanticDomainPlan:
    """@brief 从严格 JSON 文件读取语义域计划。

    @param path 文件路径。
    @return 域计划。
    @raises TaskContractError 文件不可读/非法。
    """
    try:
        document = load_strict_json(path)
    except IdentityError as exc:
        raise TaskContractError(f"{path}: the semantic domain plan is not strict JSON: {exc}") from exc
    return parse_semantic_domain_plan(document, where=str(path))


# ---------------------------------------------------------------------------------------------
# Guard AST: one closed expression tree, canonicalized structurally
# ---------------------------------------------------------------------------------------------


def guard_ast_document(node: Node) -> Any:
    """@brief 表达式树的闭集 JSON 投影（digest 的唯一输入）。

    @param node 表达式树。
    @return ``{"const": n}`` / ``{"var": name}`` / ``{"call": f, "args": [...]}``。
    @raises GeometryError 节点不是合法表达式树。
    """
    if isinstance(node, Const):
        return {"const": int(node.value)}
    if isinstance(node, Var):
        return {"var": node.name}
    if isinstance(node, Call):
        return {"call": node.func, "args": [guard_ast_document(arg) for arg in node.args]}
    raise GeometryError(f"unreachable guard AST node {type(node).__name__}")  # pragma: no cover


#: Operations whose operands may be reordered without changing meaning.  Sorting is purely
#: structural: operands are neither folded nor cancelled, and non-commutative operations (``-``,
#: ``/``, ``trunc_div``, ``rem``, comparisons) keep the author's order.
COMMUTATIVE_FUNCTIONS: frozenset[str] = frozenset({"eq", "ne", "+", "*", "min", "max"})


def _sort_key(node: Node) -> bytes:
    """@brief 可交换操作数的 canonical 排序键。

    @param node 表达式树（已规范化）。
    @return canonical JSON 字节。
    """
    return canonical_bytes({"ast": guard_ast_document(node)})


def canonical_ast(node: Node) -> Node:
    """@brief 纯结构规范化：``gt``/``ge`` 换向、``and``/``or`` 扁平化排序、可交换项排序。

    The transformation is deliberately limited: it never cancels, expands, distributes or
    rewrites arithmetic.  ``a-b<0`` and ``a<b`` therefore stay *different* guards.  Evaluation
    order of a flattened ``and``/``or`` changes only inside a total boolean function.

    @param node 表达式树。
    @return 规范化表达式树。
    @raises GeometryError 节点非法。
    """
    if isinstance(node, (Const, Var)):
        return node
    if not isinstance(node, Call):
        raise GeometryError(f"unreachable guard AST node {type(node).__name__}")  # pragma: no cover
    args = tuple(canonical_ast(arg) for arg in node.args)
    func = node.func
    if func == "gt":
        return Call("lt", (args[1], args[0]))
    if func == "ge":
        return Call("le", (args[1], args[0]))
    if func in ("and", "or"):
        flattened: list[Node] = []
        for arg in args:
            if isinstance(arg, Call) and arg.func == func:
                flattened.extend(arg.args)
            else:
                flattened.append(arg)
        flattened.sort(key=_sort_key)
        return Call(func, tuple(flattened))
    if func in COMMUTATIVE_FUNCTIONS:
        return Call(func, tuple(sorted(args, key=_sort_key)))
    return Call(func, args)


def _canonical_digest(canonical: Node) -> str:
    """@brief 规范化 AST 的摘要。

    @param canonical 已规范化的表达式树。
    @return 十六进制 sha256。
    """
    return fingerprint(
        {
            "fingerprint_kind": GUARD_DIGEST_KIND,
            "fingerprint_version": 1,
            "ast": guard_ast_document(canonical),
        }
    )


def guard_digest(term: Any, *, where: str = "guard") -> str:
    """@brief 规范化 guard 的 AST 摘要（不哈希原始字符串）。

    @param term JSON/text guard 术语。
    @param where 诊断位置。
    @return 十六进制 sha256。
    @raises GeometryError 术语不可解析、不是谓词或调用非法函数。
    """
    node = parse_term(term, where)
    if kind_of(node) != BOOL:
        raise GeometryError(f"{where}: a guard must be a predicate expression, got {kind_of(node)}")
    return _canonical_digest(canonical_ast(node))


@dataclass(frozen=True)
class Guard:
    """@brief 一个已规范化且已核对摘要的 guard。"""

    node: Node
    digest: str

    def __post_init__(self) -> None:
        """@brief 校验摘要形状与谓词类型。

        @raises TaskContractError/GeometryError 非法。
        """
        _require_digest(self.digest, where="guard digest")
        if kind_of(self.node) != BOOL:
            raise GeometryError("a guard must be a predicate expression")

    def document(self) -> dict[str, Any]:
        """@brief 规范化 guard 的文档（canonical AST + 摘要）。

        @return 字段字典。
        """
        return {"expr": guard_ast_document(self.node), "digest": self.digest}

    def evaluate(self, names: Mapping[str, int], *, where: str = "guard") -> bool:
        """@brief 在该请求的名字空间上求值。

        @param names 整数名字映射。
        @param where 诊断位置。
        @return 谓词结果。
        @raises GeometryError 未知名字或求值失败（调用方转为 dispatch_unknown）。
        """
        value = evaluate(self.node, names)
        if not isinstance(value, bool):  # pragma: no cover - kind_of closes this
            raise GeometryError(f"{where}: a guard evaluated to a non-predicate value")
        return value


def canonical_guard(term: Any, *, where: str = "guard") -> Guard:
    """@brief 解析一个 guard 术语并返回规范 AST + 摘要。

    @param term JSON/text guard 术语。
    @param where 诊断位置。
    @return guard。
    @raises GeometryError 术语非法。
    """
    node = parse_term(term, where)
    if kind_of(node) != BOOL:
        raise GeometryError(f"{where}: a guard must be a predicate expression, got {kind_of(node)}")
    canonical = canonical_ast(node)
    return Guard(node=canonical, digest=_canonical_digest(canonical))


def parse_guard(document: Any, *, where: str = "guard") -> Guard:
    """@brief 解析 ``{"expr", "digest"}`` 并核对声明的摘要就是规范 AST 摘要。

    @param document guard 文档。
    @param where 诊断位置。
    @return guard。
    @raises TaskContractError 字段集/摘要不合法或声明的摘要不是规范摘要。
    @raises GeometryError 表达式非法。
    """
    if not isinstance(document, Mapping):
        raise TaskContractError(f"{where}: a guard must be an object")
    keys = set(document)
    if keys != GUARD_FIELDS:
        raise TaskContractError(f"{where}: a guard has fields {sorted(keys)}; expected exactly {sorted(GUARD_FIELDS)}")
    declared = _require_digest(document["digest"], where=f"{where}.digest")
    guard = canonical_guard(document["expr"], where=f"{where}.expr")
    if declared != guard.digest:
        raise TaskContractError(
            f"{where}: the declared guard digest {declared} is not the canonical AST digest {guard.digest}; a digest "
            "of the author's spelling (or any other spelling) is not a guard identity"
        )
    return guard


__all__ = [
    "ACTIVE_DOMAIN_KINDS",
    "COMMUTATIVE_FUNCTIONS",
    "DOMAIN_KINDS",
    "DomainAuthority",
    "GUARD_FIELDS",
    "INSTANCE_USAGES",
    "SEMANTIC_DOMAIN_PLAN_NORMALIZATION_VERSION",
    "SEMANTIC_DOMAIN_PLAN_SCHEMA_ID",
    "SEMANTIC_DOMAIN_PLAN_SCHEMA_PATH",
    "Guard",
    "SemanticDomainInstance",
    "SemanticDomainPlan",
    "canonical_ast",
    "canonical_guard",
    "evaluation_input_digest",
    "guard_ast_document",
    "guard_digest",
    "instance_key",
    "load_semantic_domain_plan",
    "normalize_semantic_domain_plan",
    "parse_guard",
    "parse_semantic_domain_plan",
    "require_authoritative_domain_plan",
    "semantic_instance_digest",
    "target_generation_digest",
    "workload_profile_digest",
]
