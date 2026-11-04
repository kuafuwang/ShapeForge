"""P2-A: authoritative portfolio freezing and two-stage validation-shard derivation.

The candidate sees only :class:`CandidateTuneView`.  The full :class:`ValidationPlan` — including
which instances are held out and every fresh input seed — belongs to the harness.  ``master_seed``
is accepted only as an HMAC key for derivation and is never serialized, returned or included in an
error message.  Required boundary/tail/resource points must be *unseen*, never tuned.
"""

from __future__ import annotations

import hashlib
import hmac
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from ..contract_errors import TaskContractError
from ..identity import (
    MAX_SAFE_INTEGER,
    IdentityError,
    canonical_bytes,
    fingerprint,
    load_strict_json,
    loads_strict_json,
)
from ..portfolio.candidate import PortfolioCandidate, parse_portfolio
from ..portfolio.domain import DomainAuthority, SemanticDomainPlan, parse_semantic_domain_plan
from ..remote.contracts import ContractError
from ..track.plane import CandidateEntry
from ..portfolio_evidence.projection import (
    _digest,
    _entry_document,
    _entry_from_document,
    _integer,
    _json_closure,
    _keys,
    _mapping,
    _string,
)

PORTFOLIO_FREEZE_SCHEMA_ID = "tcenv.portfolio.freeze.1"
VALIDATION_PLAN_SCHEMA_ID = "tcenv.portfolio.validation-plan.1"
CANDIDATE_TUNE_VIEW_SCHEMA_ID = "tcenv.portfolio.candidate-tune-view.1"
VALIDATION_DERIVATION_VERSION = 1
SELECTION_MODE = "hmac-sha256-master-seed-v1"

FREEZE_KEYS = frozenset(
    {"schema_id", "protocol_epoch", "domain_plan", "candidate_entries", "task_pack_digest", "guidance_digest"}
)
PLAN_KEYS = frozenset(
    {
        "schema_id",
        "derivation_version",
        "selection_mode",
        "portfolio_freeze_digest",
        "opspec_digest",
        "tune_shards",
        "unseen_instance_keys",
        "required_instance_keys",
        "fresh_seeds",
    }
)
TUNE_VIEW_KEYS = frozenset({"schema_id", "portfolio_freeze_digest", "tune_shards"})


def _string_list(value: Any, *, where: str, nonempty: bool = True) -> tuple[str, ...]:
    """@brief 非空、去重、排序的字符串键列表。"""
    if not isinstance(value, list) or (nonempty and not value):
        raise TaskContractError(f"{where} must be a non-empty array of instance keys")
    items = tuple(_string(item, where=f"{where} item") for item in value)
    if len(set(items)) != len(items):
        raise TaskContractError(f"{where} contains duplicate instance keys")
    return tuple(sorted(items))


def _master_key(master_seed: Any) -> bytes:
    """@brief 把外部 master seed 变成 HMAC key；int/bytes 命名空间不碰撞，绝不回显。"""
    if type(master_seed) is int and 0 <= master_seed <= MAX_SAFE_INTEGER:
        return f"portfolio-master-seed-int-v1:{master_seed}".encode("ascii")
    if isinstance(master_seed, bytes) and len(master_seed) >= 16:
        return b"portfolio-master-seed-bytes-v1\x00" + bytes(master_seed)
    raise TaskContractError("master_seed must be a safe non-negative integer or at least 16 bytes of secret key material")


def _selection_digest(master: bytes, *, purpose: str, opspec_digest: str, freeze_digest: str, instance: str) -> bytes:
    """@brief HMAC 派生选择/输入种子；消息只含公开身份与 instance key。"""
    message = canonical_bytes(
        {
            "derivation_version": VALIDATION_DERIVATION_VERSION,
            "purpose": purpose,
            "opspec_digest": opspec_digest,
            "portfolio_freeze_digest": freeze_digest,
            "instance_key": instance,
        }
    )
    return hmac.new(master, message, hashlib.sha256).digest()


def _fresh_seed(master: bytes, *, opspec_digest: str, freeze_digest: str, instance: str) -> int:
    return int.from_bytes(
        _selection_digest(
            master,
            purpose="fresh-input-seed",
            opspec_digest=opspec_digest,
            freeze_digest=freeze_digest,
            instance=instance,
        )[:8],
        "big",
    ) & MAX_SAFE_INTEGER


@dataclass(frozen=True)
class PortfolioFreeze:
    """@brief 权威 domain + 候选字节 + 外部 pack/guidance 身份的只读组合。

    ``authority`` is required so a self-consistent but non-authoritative domain plan cannot be
    frozen.  Task-pack/guidance digests remain external trust inputs: the caller must derive them
    from the real pack/guidance bytes before constructing this object.
    """

    protocol_epoch: int
    authority: DomainAuthority = field(compare=False, repr=False)
    candidate_entries: tuple[CandidateEntry, ...]
    task_pack_digest: str
    guidance_digest: str
    _digest: str = field(init=False, compare=False, repr=False)

    def __post_init__(self) -> None:
        _integer(self.protocol_epoch, where="freeze.protocol_epoch", minimum=1)
        if not isinstance(self.authority, DomainAuthority):
            raise TaskContractError("freeze.authority must be an authoritative DomainAuthority")
        if not isinstance(self.candidate_entries, tuple) or not self.candidate_entries:
            raise TaskContractError("freeze.candidate_entries must be a non-empty tuple")
        # CandidateEntry accepts bytearray; copy into immutable bytes so a caller cannot mutate the
        # freeze after verification.  Also normalize the tuple itself.
        if any(not isinstance(entry, CandidateEntry) for entry in self.candidate_entries):
            raise TaskContractError("freeze.candidate_entries must contain CandidateEntry values")
        entries = tuple(
            CandidateEntry(path=entry.path, role=entry.role, media_type=entry.media_type, data=bytes(entry.data))
            for entry in self.candidate_entries
        )
        object.__setattr__(self, "candidate_entries", entries)
        _digest(self.task_pack_digest, where="freeze.task_pack_digest")
        _digest(self.guidance_digest, where="freeze.guidance_digest")
        self.portfolio()
        try:
            digest = fingerprint(self.document())
        except (ContractError, IdentityError) as exc:
            raise TaskContractError(f"portfolio freeze is not canonicalizable: {exc}") from exc
        object.__setattr__(self, "_digest", digest)

    @classmethod
    def from_document(cls, value: Any, *, authority: DomainAuthority) -> "PortfolioFreeze":
        _json_closure(value, where="portfolio freeze")
        document = _keys(value, FREEZE_KEYS, where="portfolio freeze")
        if document["schema_id"] != PORTFOLIO_FREEZE_SCHEMA_ID:
            raise TaskContractError(f"freeze.schema_id must be {PORTFOLIO_FREEZE_SCHEMA_ID!r}")
        parsed_plan = parse_semantic_domain_plan(document["domain_plan"], where="freeze.domain_plan")
        if parsed_plan.digest != authority.plan.digest:
            raise TaskContractError("freeze.domain_plan is not the authoritative domain plan")
        entries_value = document["candidate_entries"]
        if not isinstance(entries_value, list) or not entries_value:
            raise TaskContractError("freeze.candidate_entries must be a non-empty array")
        return cls(
            protocol_epoch=_integer(document["protocol_epoch"], where="freeze.protocol_epoch", minimum=1),
            authority=authority,
            candidate_entries=tuple(
                _entry_from_document(item, where=f"freeze.candidate_entries[{index}]")
                for index, item in enumerate(entries_value)
            ),
            task_pack_digest=_digest(document["task_pack_digest"], where="freeze.task_pack_digest"),
            guidance_digest=_digest(document["guidance_digest"], where="freeze.guidance_digest"),
        )

    @property
    def domain_plan(self) -> SemanticDomainPlan:
        return self.authority.plan

    def portfolio(self) -> PortfolioCandidate:
        return parse_portfolio(self.candidate_entries, domain_plan=self.domain_plan)

    def document(self) -> dict[str, Any]:
        return {
            "schema_id": PORTFOLIO_FREEZE_SCHEMA_ID,
            "protocol_epoch": self.protocol_epoch,
            "domain_plan": self.domain_plan.document(),
            "candidate_entries": [
                _entry_document(entry) for entry in sorted(self.candidate_entries, key=lambda item: item.path)
            ],
            "task_pack_digest": self.task_pack_digest,
            "guidance_digest": self.guidance_digest,
        }

    @property
    def digest(self) -> str:
        return self._digest

    @property
    def opspec_digest(self) -> str:
        return self.domain_plan.opspec_digest

    def instance_keys(self) -> tuple[str, ...]:
        return tuple(sorted(point.instance_key for point in self.domain_plan.instances))


@dataclass(frozen=True)
class CandidateTuneView:
    """@brief 候选可读的视图：只有 freeze 身份与 tune keys，绝无 unseen/seed/master。"""

    portfolio_freeze_digest: str
    tune_shards: tuple[str, ...]

    def __post_init__(self) -> None:
        _digest(self.portfolio_freeze_digest, where="candidate_view.portfolio_freeze_digest")
        tune = _string_list(list(self.tune_shards), where="candidate_view.tune_shards")
        if tune != tuple(self.tune_shards):
            raise TaskContractError("candidate_view.tune_shards must be sorted and unique")
        object.__setattr__(self, "tune_shards", tune)

    def document(self) -> dict[str, Any]:
        return {
            "schema_id": CANDIDATE_TUNE_VIEW_SCHEMA_ID,
            "portfolio_freeze_digest": self.portfolio_freeze_digest,
            "tune_shards": list(self.tune_shards),
        }

    @classmethod
    def from_document(cls, value: Any) -> "CandidateTuneView":
        _json_closure(value, where="candidate tune view")
        document = _keys(value, TUNE_VIEW_KEYS, where="candidate tune view")
        if document["schema_id"] != CANDIDATE_TUNE_VIEW_SCHEMA_ID:
            raise TaskContractError("candidate tune view has the wrong schema")
        return cls(
            portfolio_freeze_digest=_digest(
                document["portfolio_freeze_digest"], where="candidate_view.portfolio_freeze_digest"
            ),
            tune_shards=_string_list(document["tune_shards"], where="candidate_view.tune_shards"),
        )


@dataclass(frozen=True)
class ValidationPlan:
    """@brief harness 持有的完整两阶段计划；候选只能拿到 :meth:`candidate_view`。"""

    portfolio_freeze_digest: str
    opspec_digest: str
    tune_shards: tuple[str, ...]
    unseen_instance_keys: tuple[str, ...]
    required_instance_keys: tuple[str, ...]
    fresh_seeds: Mapping[str, int]
    derivation_version: int = VALIDATION_DERIVATION_VERSION
    selection_mode: str = SELECTION_MODE

    def __post_init__(self) -> None:
        _integer(self.derivation_version, where="validation.derivation_version", minimum=1)
        if self.derivation_version != VALIDATION_DERIVATION_VERSION:
            raise TaskContractError("unsupported validation derivation version")
        if self.selection_mode != SELECTION_MODE:
            raise TaskContractError("validation plan must use the registered master-seed selection mode")
        _digest(self.portfolio_freeze_digest, where="validation.portfolio_freeze_digest")
        _digest(self.opspec_digest, where="validation.opspec_digest")
        tune = _string_list(list(self.tune_shards), where="validation.tune_shards")
        unseen = _string_list(list(self.unseen_instance_keys), where="validation.unseen_instance_keys")
        required = _string_list(list(self.required_instance_keys), where="validation.required_instance_keys")
        if tune != tuple(self.tune_shards) or unseen != tuple(self.unseen_instance_keys) or required != tuple(self.required_instance_keys):
            raise TaskContractError("validation tune/unseen/required keys must be sorted and unique")
        if set(tune) & set(unseen):
            raise TaskContractError("validation tune and unseen shards overlap")
        if set(required) & set(tune):
            raise TaskContractError("required boundary/tail/resource instances must be unseen, not tuned")
        if not set(required) <= set(unseen):
            raise TaskContractError("every required boundary/tail/resource instance must be in unseen")
        seeds = dict(_mapping(self.fresh_seeds, where="validation.fresh_seeds"))
        if set(seeds) != set(unseen):
            raise TaskContractError("validation fresh_seeds must cover exactly the unseen instance keys")
        for key, value in seeds.items():
            _string(key, where="validation.fresh_seeds key")
            _integer(value, where=f"validation.fresh_seeds.{key}", minimum=0)
        object.__setattr__(self, "tune_shards", tune)
        object.__setattr__(self, "unseen_instance_keys", unseen)
        object.__setattr__(self, "required_instance_keys", required)
        object.__setattr__(self, "fresh_seeds", MappingProxyType(seeds))

    @classmethod
    def derive(
        cls,
        freeze: PortfolioFreeze,
        *,
        tune_shards: tuple[str, ...] | list[str],
        unseen_count: int,
        master_seed: Any,
        required_instance_keys: tuple[str, ...] | list[str],
    ) -> "ValidationPlan":
        """@brief 用外部 master seed 选择未见 keys 并派生 fresh seeds。

        Required points are mandatory *unseen* instances and therefore cannot also be tune shards.
        HMAC only fills the remaining slots after those required points have been included.
        """
        if not isinstance(freeze, PortfolioFreeze):
            raise TaskContractError("ValidationPlan.derive needs a PortfolioFreeze")
        tune = _string_list(list(tune_shards), where="validation.tune_shards")
        required = _string_list(list(required_instance_keys), where="validation.required_instance_keys")
        available = set(freeze.instance_keys())
        if set(required) - available:
            raise TaskContractError("required validation coverage contains an instance outside the frozen domain")
        if set(required) & set(tune):
            raise TaskContractError("required boundary/tail/resource instances must be unseen, not tuned")
        eligible = sorted(available - set(tune))
        _integer(unseen_count, where="validation.unseen_count", minimum=1)
        if unseen_count < len(required):
            raise TaskContractError(
                f"validation unseen_count {unseen_count} does not cover {len(required)} required unseen instances"
            )
        if unseen_count > len(eligible):
            raise TaskContractError(
                f"validation requests {unseen_count} unseen instances but only {len(eligible)} are available"
            )
        master = _master_key(master_seed)
        ranked = sorted(
            (key for key in eligible if key not in required),
            key=lambda key: _selection_digest(
                master,
                purpose="select-unseen-instance",
                opspec_digest=freeze.opspec_digest,
                freeze_digest=freeze.digest,
                instance=key,
            ),
        )
        unseen = tuple(sorted([*required, *ranked[: unseen_count - len(required)]]))
        seeds = {
            key: _fresh_seed(
                master,
                opspec_digest=freeze.opspec_digest,
                freeze_digest=freeze.digest,
                instance=key,
            )
            for key in unseen
        }
        plan = cls(
            portfolio_freeze_digest=freeze.digest,
            opspec_digest=freeze.opspec_digest,
            tune_shards=tune,
            unseen_instance_keys=unseen,
            required_instance_keys=required,
            fresh_seeds=seeds,
        )
        plan.validate_for(freeze)
        return plan

    @classmethod
    def from_document(cls, value: Any) -> "ValidationPlan":
        _json_closure(value, where="validation plan")
        document = _keys(value, PLAN_KEYS, where="validation plan")
        if document["schema_id"] != VALIDATION_PLAN_SCHEMA_ID:
            raise TaskContractError(f"validation.schema_id must be {VALIDATION_PLAN_SCHEMA_ID!r}")
        return cls(
            derivation_version=_integer(document["derivation_version"], where="validation.derivation_version", minimum=1),
            selection_mode=_string(document["selection_mode"], where="validation.selection_mode"),
            portfolio_freeze_digest=_digest(
                document["portfolio_freeze_digest"], where="validation.portfolio_freeze_digest"
            ),
            opspec_digest=_digest(document["opspec_digest"], where="validation.opspec_digest"),
            tune_shards=_string_list(document["tune_shards"], where="validation.tune_shards"),
            unseen_instance_keys=_string_list(document["unseen_instance_keys"], where="validation.unseen_instance_keys"),
            required_instance_keys=_string_list(document["required_instance_keys"], where="validation.required_instance_keys"),
            fresh_seeds=dict(_mapping(document["fresh_seeds"], where="validation.fresh_seeds")),
        )

    def validate_for(self, freeze: PortfolioFreeze) -> None:
        """@brief 绑定权威 freeze/domain 与 required 覆盖，不允许事后改 plan 或 opspec。"""
        if self.portfolio_freeze_digest != freeze.digest or self.opspec_digest != freeze.opspec_digest:
            raise TaskContractError("validation plan belongs to a different portfolio freeze or OpSpec")
        available = set(freeze.instance_keys())
        tune = _string_list(list(self.tune_shards), where="validation.tune_shards")
        unseen = _string_list(list(self.unseen_instance_keys), where="validation.unseen_instance_keys")
        required = _string_list(list(self.required_instance_keys), where="validation.required_instance_keys")
        if tune != self.tune_shards or unseen != self.unseen_instance_keys or required != self.required_instance_keys:
            raise TaskContractError("validation tune/unseen/required keys must remain sorted and unique")
        if set(tune) & set(unseen) or set(required) & set(tune) or not set(required) <= set(unseen):
            raise TaskContractError("validation selection violates tune/unseen/required separation")
        selected = set(tune) | set(unseen)
        if selected - available or set(required) - available:
            raise TaskContractError("validation names instances outside the frozen domain")

    def verify_fresh_seeds(
        self,
        freeze: PortfolioFreeze,
        *,
        master_seed: Any,
        expected_plan_digest: str,
    ) -> None:
        """@brief resume 必须同时钉住 plan 与 freeze，再重算所有 fresh seeds。"""
        _digest(expected_plan_digest, where="expected_plan_digest")
        self.validate_for(freeze)
        if self.digest != expected_plan_digest:
            raise TaskContractError("validation plan is not the externally pinned plan")
        master = _master_key(master_seed)
        expected = {
            key: _fresh_seed(
                master,
                opspec_digest=freeze.opspec_digest,
                freeze_digest=freeze.digest,
                instance=key,
            )
            for key in self.unseen_instance_keys
        }
        if dict(self.fresh_seeds) != expected:
            raise TaskContractError("validation fresh seeds do not match this master seed and portfolio freeze")

    def candidate_view(self, freeze: PortfolioFreeze) -> CandidateTuneView:
        """@brief 生成候选可读投影，完整 plan/unseen/seeds 不穿过这个边界。"""
        self.validate_for(freeze)
        return CandidateTuneView(
            portfolio_freeze_digest=self.portfolio_freeze_digest,
            tune_shards=tuple(self.tune_shards),
        )

    def document(self) -> dict[str, Any]:
        return {
            "schema_id": VALIDATION_PLAN_SCHEMA_ID,
            "derivation_version": self.derivation_version,
            "selection_mode": self.selection_mode,
            "portfolio_freeze_digest": self.portfolio_freeze_digest,
            "opspec_digest": self.opspec_digest,
            "tune_shards": list(self.tune_shards),
            "unseen_instance_keys": list(self.unseen_instance_keys),
            "required_instance_keys": list(self.required_instance_keys),
            "fresh_seeds": {key: int(self.fresh_seeds[key]) for key in sorted(self.fresh_seeds)},
        }

    @property
    def digest(self) -> str:
        return fingerprint(self.document())


def load_portfolio_freeze(path: str | Path, *, authority: DomainAuthority) -> PortfolioFreeze:
    """@brief freeze 的唯一生产文件入口：重复 JSON key 在 Mapping 之前拒绝。"""
    try:
        value = load_strict_json(path)
    except (IdentityError, OSError) as exc:
        raise TaskContractError(f"cannot load portfolio freeze: {exc}") from exc
    return PortfolioFreeze.from_document(value, authority=authority)


def loads_portfolio_freeze(text: str, *, authority: DomainAuthority) -> PortfolioFreeze:
    """@brief freeze 的唯一生产文本入口：重复 JSON key 在 Mapping 之前拒绝。"""
    try:
        value = loads_strict_json(text, origin="<portfolio-freeze>")
    except IdentityError as exc:
        raise TaskContractError(f"cannot parse portfolio freeze: {exc}") from exc
    return PortfolioFreeze.from_document(value, authority=authority)


def load_validation_plan(path: str | Path) -> ValidationPlan:
    """@brief 唯一生产文件入口：重复 JSON key 在 Mapping 之前拒绝。"""
    try:
        value = load_strict_json(path)
    except (IdentityError, OSError) as exc:
        raise TaskContractError(f"cannot load validation plan: {exc}") from exc
    return ValidationPlan.from_document(value)


def loads_validation_plan(text: str) -> ValidationPlan:
    """@brief 唯一生产文本入口：重复 JSON key 在 Mapping 之前拒绝。"""
    try:
        value = loads_strict_json(text, origin="<validation-plan>")
    except IdentityError as exc:
        raise TaskContractError(f"cannot parse validation plan: {exc}") from exc
    return ValidationPlan.from_document(value)


__all__ = [
    "CANDIDATE_TUNE_VIEW_SCHEMA_ID",
    "CandidateTuneView",
    "PORTFOLIO_FREEZE_SCHEMA_ID",
    "PortfolioFreeze",
    "VALIDATION_PLAN_SCHEMA_ID",
    "ValidationPlan",
    "load_validation_plan",
    "loads_portfolio_freeze",
    "load_portfolio_freeze",
    "loads_validation_plan",
]
