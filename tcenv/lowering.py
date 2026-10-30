"""Backend lowering: registered LoweringProfiles, family lowerers and Specialization.

Two-stage derivation (plan §4.2):

``instantiate(opspec, dtype, size) -> SemanticInstance``   (``tcenv.opspec``)
``lower(instance, profile) -> Specialization``            (this module)

A ``LoweringProfile`` is a *registered, static* statement of one backend's contract
defaults: kernel name, default grid/threads, the constant target types, the start form,
variant parameters and the target name shown in the contract.  It is not build
evidence, it never contains probe results, and it never goes on the wire: the remote
evaluator derives the same ``Specialization`` from the same registered profile
(deployment prerequisite: same host/evaluator version), and the resulting
``task_fingerprint`` covers the disagreement.

A lowerer is registered by id *and* version; a variant task registers a variant factory
by id and version as well.  ``lowering_profile_digest`` is the §3.1 canonical envelope
over the fully resolved profile plus those identities, so changing the threads (F-2) or
bumping the lowerer version (F-25) invalidates the lowering identity and everything
downstream while leaving ``opspec_digest`` untouched.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

from .contract import AbiArg, Contract
from .identity import IdentityError, fingerprint
from .opspec.model import SemanticInstance
from .terms import TermError, eval_term, normalize_term, term_document

#: Version of the registered profile document shape.
LOWERING_PROFILE_SCHEMA_VERSION = 1
#: Version of the ``lowering_profile_digest`` canonical envelope.
LOWERING_FINGERPRINT_VERSION = 1

_MSL_INT_TYPES = frozenset({"uint", "ulong", "int", "long", "ushort", "short"})


class LoweringError(ValueError):
    """@brief Lowering 配置或派生非法（未登记 profile/lowerer、目标不支持、字段不自洽）。"""


def _resolve_source_term(
    term: Any,
    *,
    symbols: Mapping[str, int],
    attrs: Mapping[str, Any],
    where: str,
) -> Any:
    """@brief 解析 ``symbol:NAME`` / ``attr:NAME`` / ``literal:VALUE`` 形式的来源词。

    @param term 来源词字符串。
    @param symbols 已解析符号值。
    @param attrs 已物化 attr 值。
    @param where 诊断位置。
    @return 原始值（未按 msl_type 转换）。
    @raises LoweringError 形式未知或名字未声明。
    """
    if not isinstance(term, str) or ":" not in term:
        raise LoweringError(f"{where}: source term {term!r} must look like kind:name")
    kind, _, name = term.partition(":")
    if kind == "symbol":
        if name not in symbols:
            raise LoweringError(f"{where}: symbol {name!r} is not in the semantic instance")
        return int(symbols[name])
    if kind == "attr":
        if name not in attrs:
            raise LoweringError(f"{where}: attr {name!r} is not resolved on the semantic instance")
        return attrs[name]
    if kind == "literal":
        try:
            return int(name)
        except ValueError as exc:
            raise LoweringError(f"{where}: literal {name!r} is not an integer") from exc
    if kind == "product":
        names = [part for part in name.split(",") if part]
        if not names:
            raise LoweringError(f"{where}: product source needs at least one symbol name")
        try:
            return eval_term({"product": names}, symbols, where=where)
        except TermError as exc:
            raise LoweringError(str(exc)) from exc
    raise LoweringError(f"{where}: unknown source kind {kind!r}")


def _cast_msl(value: Any, msl_type: str, where: str) -> Any:
    """@brief 按登记的 MSL 类型把值物化为唯一 JSON 类型。

    @param value 原始值。
    @param msl_type MSL 类型名。
    @param where 诊断位置。
    @return ``uint``/``ulong`` 等为 int，``float``/``half``/``bfloat`` 为 float。
    @raises LoweringError 值不能按该类型物化。
    """
    if msl_type in _MSL_INT_TYPES:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise LoweringError(f"{where}: {value!r} cannot be materialised as {msl_type}")
        if isinstance(value, float) and not float(value).is_integer():
            raise LoweringError(f"{where}: {value!r} is not an integer constant for {msl_type}")
        return int(value)
    if msl_type in ("float", "half", "bfloat"):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise LoweringError(f"{where}: {value!r} cannot be materialised as {msl_type}")
        return float(value)
    raise LoweringError(f"{where}: unsupported constant target type {msl_type!r}")


@dataclass(frozen=True)
class ConstantSpec:
    """@brief 一个契约 ``constant constexpr``：名字、目标类型与来源词。"""

    name: str
    msl_type: str
    source: str

    def to_dict(self) -> dict[str, Any]:
        """@brief 完全物化的登记表示。

        @return 含 name/msl_type/source 的字典。
        """
        return {"name": self.name, "msl_type": self.msl_type, "source": self.source}


@dataclass(frozen=True)
class ScalarSpec:
    """@brief 一个契约标量参数（非 buffer）。

    ``index`` pins the ABI slot when the scalar is not appended after every buffer (BitDelta's
    ``scale`` sits between ``bits`` and ``y``); ``None`` means "append after the tensors".
    """

    name: str
    msl_type: str
    source: str
    doc: str = ""
    index: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """@brief 完全物化的登记表示。

        @return 含 name/msl_type/source/doc/index 的字典。
        """
        return {
            "name": self.name,
            "msl_type": self.msl_type,
            "source": self.source,
            "doc": self.doc,
            "index": self.index,
        }


@dataclass(frozen=True)
class LoweringProfile:
    """@brief 一个后端登记、静态的 lowering 配置（不是构建证据）。"""

    schema_version: int
    name: str
    backend_id: str
    target_id: str
    display_name: str
    lowerer_id: str
    kernel_name: str
    grid: tuple[int | Mapping[str, Any], ...]
    threads: tuple[int, int, int]
    constants: tuple[ConstantSpec, ...] = ()
    scalars: tuple[ScalarSpec, ...] = ()
    #: ``backend_params``: fully resolved launch/tiling substitutions for track templates
    #: (``{{BLOCK_SIZE}}``, ``{{NUM_WARPS}}`` …).  They are launch facts of this profile, so
    #: they are part of ``lowering_profile_digest``.  A variant factory that derives a profile
    #: also sets ``variant_factory_id``; a plain profile does not need one.
    variant_params: Mapping[str, Any] = field(default_factory=dict)
    variant_factory_id: str | None = None

    def __post_init__(self) -> None:
        """@brief 校验 profile 形状，避免未解析的 alias 进入身份。

        @raises LoweringError 字段非法。
        """
        if self.schema_version != LOWERING_PROFILE_SCHEMA_VERSION:
            raise LoweringError(f"profile {self.name!r}: unsupported schema_version {self.schema_version}")
        if len(self.grid) != 3 or len(self.threads) != 3:
            raise LoweringError(f"profile {self.name!r}: grid and threads are 3-tuples")
        if any(not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in self.threads):
            raise LoweringError(f"profile {self.name!r}: threads must be non-negative integers")
        try:
            for term in self.grid:
                normalize_term(term, where=f"profile {self.name!r}.grid")
        except TermError as exc:
            raise LoweringError(str(exc)) from exc

    def to_dict(self) -> dict[str, Any]:
        """@brief 完全解析后的 profile 登记文档（进入 lowering 摘要）。

        @return 只含 JSON 值的闭集字典。
        """
        return {
            "schema_version": int(self.schema_version),
            "name": self.name,
            "backend_id": self.backend_id,
            "target_id": self.target_id,
            "display_name": self.display_name,
            "lowerer_id": self.lowerer_id,
            "kernel_name": self.kernel_name,
            "grid": [term_document(term) for term in self.grid],
            "threads": [int(value) for value in self.threads],
            "constants": [constant.to_dict() for constant in self.constants],
            "scalars": [scalar.to_dict() for scalar in self.scalars],
            "variant_factory_id": self.variant_factory_id,
            "variant_params": dict(self.variant_params),
        }


@dataclass(frozen=True)
class Specialization:
    """@brief ``lower`` 的唯一产物：派生 Contract 与两侧 identity。"""

    opspec_digest: str
    lowering_profile_digest: str
    spec_params: Mapping[str, str]
    backend_params: Mapping[str, str]
    contract: Contract

    def require_matches(self, instance: SemanticInstance) -> None:
        """@brief 断言本 Specialization 与语义实例同源（不得靠 lower 回写对齐）。

        @param instance 语义实例。
        @raises LoweringError ``opspec_digest`` 或 ``spec_params`` 不一致。
        """
        if self.opspec_digest != instance.opspec_digest:
            raise LoweringError("specialization and semantic instance disagree on opspec_digest")
        if dict(self.spec_params) != dict(instance.spec_params):
            raise LoweringError("specialization and semantic instance disagree on spec_params")

    def fingerprint_document(self) -> dict[str, Any]:
        """@brief §3.1 冻结的 Specialization 投影（闭集）。

        @return 只含 JSON 值、字段闭集的派生文档。
        """
        return {
            "opspec_digest": self.opspec_digest,
            "lowering_profile_digest": self.lowering_profile_digest,
            "spec_params": dict(self.spec_params),
            "backend_params": dict(self.backend_params),
            "contract": contract_document(self.contract),
        }


def contract_document(contract: Contract) -> dict[str, Any]:
    """@brief Contract 的全字段派生投影（args/constants 保持登记顺序）。

    @param contract 契约。
    @return 含 kernel_name/grid/threads/gpu/args/constants 的字典。
    """
    return {
        "kernel_name": contract.kernel_name,
        "grid": [int(value) for value in contract.grid],
        "threads": [int(value) for value in contract.threads],
        "gpu": contract.gpu,
        "args": [
            {
                "name": arg.name,
                "index": int(arg.index),
                "role": arg.role,
                "dtype": arg.dtype,
                "shape": [int(dim) for dim in arg.shape],
                "value": arg.value,
                "doc": arg.doc,
            }
            for arg in contract.args
        ],
        "constants": [
            {"name": name, "msl_type": msl_type, "value": value} for name, msl_type, value in contract.constants
        ],
    }


@dataclass(frozen=True)
class _Lowerer:
    """@brief 一个登记 family lowerer：实现 id、版本与派生函数。"""

    lowerer_id: str
    version: str
    build: Callable[[SemanticInstance, LoweringProfile], tuple[Contract, Mapping[str, str]]]


_PROFILES: dict[tuple[str, str], LoweringProfile] = {}
_LOWERERS: dict[str, _Lowerer] = {}
_LOADED_BACKENDS: set[str] = set()


def register_profile(profile: LoweringProfile) -> LoweringProfile:
    """@brief 登记一份 LoweringProfile（同一 ``(backend, name)`` 只能登记一次）。

    @param profile 待登记的 profile。
    @return 同一 profile。
    @raises LoweringError 同名 profile 已以不同内容登记。
    """
    key = (profile.backend_id, profile.name)
    existing = _PROFILES.get(key)
    if existing is not None and existing != profile:
        raise LoweringError(f"profile {profile.name!r} is already registered with different content")
    _PROFILES[key] = profile
    return profile


def register_lowerer(
    lowerer_id: str,
    version: str,
    build: Callable[[SemanticInstance, LoweringProfile], tuple[Contract, Mapping[str, str]]],
) -> None:
    """@brief 登记一个 family lowerer。

    @param lowerer_id 实现 id，如 ``metal.rmsnorm``。
    @param version 实现版本，进入 ``lowering_profile_digest``。
    @param build 派生函数。
    @raises LoweringError 重复登记不同的实现或版本。
    """
    existing = _LOWERERS.get(lowerer_id)
    if existing is not None and (existing.version != version or existing.build is not build):
        raise LoweringError(f"lowerer {lowerer_id!r} is already registered with a different version/implementation")
    _LOWERERS[lowerer_id] = _Lowerer(lowerer_id, version, build)


def ensure_lowerings(backend_id: str) -> None:
    """@brief 确保某个 backend 的 lowering 模块已导入（登记副作用只发生一次）。

    @param backend_id 后端 id。
    @raises LoweringError 该 backend 有 lowering 模块但导入失败。
    """
    if backend_id in _LOADED_BACKENDS:
        return
    module = f"tcenv.backends.{backend_id}.lowering"
    try:
        importlib.import_module(module)
    except ModuleNotFoundError as exc:
        if exc.name not in (module, f"tcenv.backends.{backend_id}"):
            raise LoweringError(f"{module} failed to import: {exc}") from exc
        _LOADED_BACKENDS.add(backend_id)
        return
    _LOADED_BACKENDS.add(backend_id)


def profile_for(backend_id: str, name: str) -> LoweringProfile:
    """@brief 取一份已登记 profile。

    @param backend_id 后端 id。
    @param name 登记名。
    @return 对应 profile。
    @raises LoweringError 没有登记。
    """
    ensure_lowerings(backend_id)
    profile = _PROFILES.get((backend_id, name))
    if profile is None:
        raise LoweringError(f"no registered LoweringProfile {name!r} for backend {backend_id!r}")
    return profile


def registered_profiles(backend_id: str) -> tuple[LoweringProfile, ...]:
    """@brief 列出某个 backend 已登记的 profile。

    @param backend_id 后端 id。
    @return 按登记名排序的 profile 元组。
    """
    ensure_lowerings(backend_id)
    return tuple(sorted((p for (b, _), p in _PROFILES.items() if b == backend_id), key=lambda p: p.name))


def registered_targets(backend_id: str) -> frozenset[str]:
    """@brief 列出该 backend 已登记 profile 覆盖的 target id。

    @param backend_id 后端 id。
    @return target id 集合。
    """
    return frozenset(profile.target_id for profile in registered_profiles(backend_id))


def require_target_profile(backend_id: str, target_id: str) -> tuple[LoweringProfile, ...]:
    """@brief 要求 ``target_id`` 至少有一份登记 profile；没有任何登记 profile 的 backend 不受此限。

    @param backend_id 后端 id。
    @param target_id 目标设备 id。
    @return 覆盖该 target 的全部 profile（按名字排序）；该 backend 尚未登记任何 profile 时为空元组。
    @raises LoweringError 该 backend 有登记 profile 但目标不在其中（不得回落到 default profile）。
    """
    profiles = registered_profiles(backend_id)
    if not profiles:
        return ()
    matching = tuple(profile for profile in profiles if profile.target_id == target_id)
    if matching:
        return matching
    raise LoweringError(
        f"target {target_id!r} has no registered LoweringProfile for backend {backend_id!r}; "
        f"registered targets are {sorted(registered_targets(backend_id))}"
    )


def lowerer_version(lowerer_id: str) -> str:
    """@brief 取已登记 lowerer / variant factory 的版本。

    @param lowerer_id 实现 id。
    @return 版本字符串。
    @raises LoweringError 未登记。
    """
    lowerer = _LOWERERS.get(lowerer_id)
    if lowerer is None:
        raise LoweringError(f"lowerer {lowerer_id!r} is not registered")
    return lowerer.version


def lowering_profile_digest(profile: LoweringProfile) -> str:
    """@brief 计算 §3.1 的 ``lowering_profile_digest``。

    @param profile 已登记的 profile。
    @return 64 位小写十六进制摘要。
    @raises LoweringError lowerer/variant factory 未登记，或 profile 不能 canonical 化。
    """
    variant_factory = None
    if profile.variant_factory_id is not None:
        factory = _VARIANT_FACTORIES.get(profile.variant_factory_id)
        if factory is None:
            raise LoweringError(f"variant factory {profile.variant_factory_id!r} is not registered")
        variant_factory = {"id": factory.factory_id, "version": factory.version}
    document = {
        "fingerprint_kind": "lowering_profile",
        "fingerprint_version": LOWERING_FINGERPRINT_VERSION,
        "lowerer": {"id": profile.lowerer_id, "version": lowerer_version(profile.lowerer_id)},
        "variant_factory": variant_factory,
        "profile": profile.to_dict(),
    }
    try:
        return fingerprint(document)
    except IdentityError as exc:
        raise LoweringError(f"profile {profile.name!r} is not canonical: {exc}") from exc


def _resolve_grid(profile: LoweringProfile, instance: SemanticInstance) -> tuple[int, int, int]:
    """@brief 把登记的 grid 词解析成具体三元组。

    @param profile profile。
    @param instance 语义实例。
    @return ``(gx, gy, gz)``。
    @raises LoweringError 引用了未声明符号或除法非法。
    """
    out = [eval_term(term, instance.symbols, where=f"profile {profile.name!r}.grid") for term in profile.grid]
    return out[0], out[1], out[2]


def _scalar_dtype(msl_type: str) -> str:
    """@brief 把 MSL 标量类型映射回 ``tcenv.dtypes`` 短名。

    @param msl_type MSL 类型名。
    @return dtype 短名。
    @raises LoweringError 不支持的类型。
    """
    mapping = {"uint": "u32", "int": "i32", "float": "fp32", "half": "fp16", "bfloat": "bf16"}
    if msl_type not in mapping:
        raise LoweringError(f"unsupported scalar target type {msl_type!r}")
    return mapping[msl_type]


def build_contract(
    instance: SemanticInstance,
    profile: LoweringProfile,
) -> tuple[Contract, Mapping[str, str]]:
    """@brief 通用 family lower：语义张量→buffer、登记常量/标量、登记 grid/threads。

    @param instance 语义实例。
    @param profile 登记的 profile。
    @return ``(contract, backend_params)``；identity 由 :func:`lower` 统一填写。
    @raises LoweringError 常量/标量来源非法或类型不匹配。
    """
    ordered: list[tuple[int, AbiArg]] = []
    # A scalar may pin its ABI slot (BitDelta's ``scale`` sits between two buffers, the integer
    # program's ``K`` between ``y`` and ``out``).  Pinned slots are reserved first; buffers then
    # take the remaining slots in declaration order, and unpinned scalars are appended after the
    # buffers.  Without a pinned scalar this is exactly the historical 0..n-1 assignment.
    total = len(instance.tensors) + len(profile.scalars)
    pinned = {int(scalar.index) for scalar in profile.scalars if scalar.index is not None}
    if any(slot < 0 or slot >= total for slot in pinned) or len(pinned) != sum(
        1 for scalar in profile.scalars if scalar.index is not None
    ):
        raise LoweringError(f"profile {profile.name!r}: scalar slots must be unique and within 0..{total - 1}")
    free_slots = iter(slot for slot in range(total) if slot not in pinned)
    buffer_slots = [next(free_slots) for _ in instance.tensors]
    for slot, binding in zip(buffer_slots, instance.tensors):
        ordered.append(
            (
                slot,
                AbiArg(
                    name=binding.name,
                    index=slot,
                    role=binding.role,
                    dtype=binding.dtype,
                    shape=tuple(binding.shape),
                    doc=binding.doc,
                ),
            )
        )
    for scalar in profile.scalars:
        slot = int(scalar.index) if scalar.index is not None else next(free_slots)
        raw = _resolve_source_term(
            scalar.source, symbols=instance.symbols, attrs=instance.resolved_attrs, where=f"scalar {scalar.name}"
        )
        ordered.append(
            (
                slot,
                AbiArg(
                    name=scalar.name,
                    index=slot,
                    role="scalar",
                    dtype=_scalar_dtype(scalar.msl_type),
                    value=_cast_msl(raw, scalar.msl_type, f"scalar {scalar.name}"),
                    doc=scalar.doc,
                ),
            )
        )
    if len({slot for slot, _ in ordered}) != len(ordered):
        raise LoweringError(f"profile {profile.name!r}: argument slots collide: {[slot for slot, _ in ordered]}")
    args = [arg for _slot, arg in sorted(ordered, key=lambda item: item[0])]
    if [arg.index for arg in args] != list(range(len(args))):
        raise LoweringError(
            f"profile {profile.name!r}: argument slots must be contiguous from 0, got {[arg.index for arg in args]}"
        )
    constants: list[tuple[str, str, Any]] = []
    for constant in profile.constants:
        raw = _resolve_source_term(
            constant.source,
            symbols=instance.symbols,
            attrs=instance.resolved_attrs,
            where=f"constant {constant.name}",
        )
        constants.append(
            (constant.name, constant.msl_type, _cast_msl(raw, constant.msl_type, f"constant {constant.name}"))
        )
    contract = Contract(
        kernel_name=profile.kernel_name,
        grid=_resolve_grid(profile, instance),
        threads=(int(profile.threads[0]), int(profile.threads[1]), int(profile.threads[2])),
        args=tuple(args),
        constants=tuple(constants),
        gpu=profile.display_name,
    )
    return contract, resolve_backend_params(profile, instance)


def resolve_backend_params(profile: LoweringProfile, instance: SemanticInstance) -> dict[str, str]:
    """@brief 解析 profile 登记的后端替换值（track 模板的 ``backend_params`` 来源）。

    @param profile profile。
    @param instance 语义实例。
    @return 名字到字符串值的映射。
    """
    out: dict[str, str] = {}
    for name, term in profile.variant_params.items():
        if isinstance(term, Mapping):
            try:
                raw = eval_term(term, instance.symbols, where=f"backend_params.{name}")
            except TermError as exc:
                raise LoweringError(str(exc)) from exc
        elif isinstance(term, str) and ":" in term:
            raw = _resolve_source_term(
                term, symbols=instance.symbols, attrs=instance.resolved_attrs, where=f"backend_params.{name}"
            )
        else:
            raw = term
        out[str(name)] = str(raw)
    return out


@dataclass(frozen=True)
class _VariantFactory:
    """@brief 一个登记的动态启动族：文法 + 由语义实例产生具体 profile 的函数。"""

    factory_id: str
    version: str
    build: Callable[[SemanticInstance, str], "LoweringProfile"]


_VARIANT_FACTORIES: dict[str, _VariantFactory] = {}


def register_variant_factory(
    factory_id: str,
    version: str,
    build: Callable[[SemanticInstance, str], "LoweringProfile"],
) -> None:
    """@brief 登记一个 variant factory（文法与默认 alias 只在后端 lowering 层）。

    @param factory_id 实现 id，如 ``metal.gemm.variant``。
    @param version 实现版本，进入 ``lowering_profile_digest``。
    @param build ``(instance, variant_text) -> LoweringProfile``。
    @raises LoweringError 重复登记不同实现/版本。
    """
    existing = _VARIANT_FACTORIES.get(factory_id)
    if existing is not None and (existing.version != version or existing.build is not build):
        raise LoweringError(f"variant factory {factory_id!r} is already registered differently")
    _VARIANT_FACTORIES[factory_id] = _VariantFactory(factory_id, version, build)


def variant_factory(factory_id: str) -> _VariantFactory:
    """@brief 取一个已登记的 variant factory。

    @param factory_id 实现 id。
    @return 登记的 factory。
    @raises LoweringError 未登记。
    """
    factory = _VARIANT_FACTORIES.get(factory_id)
    if factory is None:
        raise LoweringError(f"variant factory {factory_id!r} is not registered")
    return factory


def lower_variant(
    instance: SemanticInstance, *, backend_id: str, factory_id: str, variant: str
) -> Specialization:
    """@brief 由登记 factory 按 variant 文本产生具体 profile 并派生 Specialization。

    @param instance 语义实例。
    @param backend_id 后端 id（factory 必须属于它，见各 backend lowering 模块的登记）。
    @param factory_id 登记 factory id。
    @param variant variant 文法文本。
    @return :class:`Specialization`。
    @raises LoweringError factory 未登记、variant 非法或派生失败。
    """
    ensure_lowerings(backend_id)
    factory = variant_factory(factory_id)
    profile = factory.build(instance, variant)
    if profile.backend_id != backend_id:
        raise LoweringError(f"variant factory {factory_id!r} produced a profile for backend {profile.backend_id!r}")
    if profile.variant_factory_id != factory_id:
        raise LoweringError(f"variant factory {factory_id!r} produced a profile without its own variant_factory_id")
    return lower(instance, profile)


def lower(instance: SemanticInstance, profile: LoweringProfile) -> Specialization:
    """@brief 唯一的派生入口：语义实例 + 登记 profile → Specialization。

    @param instance 语义实例。
    @param profile 已登记 profile。
    @return :class:`Specialization`，两个 identity 都由本次派生计算。
    @raises LoweringError lowerer 未登记，或 lowerer 与实例不自洽。
    """
    lowerer = _LOWERERS.get(profile.lowerer_id)
    if lowerer is None:
        raise LoweringError(f"profile {profile.name!r}: lowerer {profile.lowerer_id!r} is not registered")
    contract, backend_params = lowerer.build(instance, profile)
    specialization = Specialization(
        opspec_digest=instance.opspec_digest,
        lowering_profile_digest=lowering_profile_digest(profile),
        spec_params=dict(instance.spec_params),
        backend_params=dict(backend_params),
        contract=contract,
    )
    specialization.require_matches(instance)
    return specialization


def derive_specialization(instance: SemanticInstance, *, backend_id: str, profile_name: str) -> Specialization:
    """@brief 按后端 id + 登记名派生 Specialization（``lower`` 的唯一生产调用面）。

    @param instance 语义实例。
    @param backend_id 后端 id。
    @param profile_name 登记 profile 名。
    @return :class:`Specialization`。
    """
    return lower(instance, profile_for(backend_id, profile_name))


__all__ = [
    "LOWERING_FINGERPRINT_VERSION",
    "LOWERING_PROFILE_SCHEMA_VERSION",
    "ConstantSpec",
    "LoweringError",
    "LoweringProfile",
    "ScalarSpec",
    "Specialization",
    "build_contract",
    "contract_document",
    "derive_specialization",
    "ensure_lowerings",
    "eval_term",
    "normalize_term",
    "term_document",
    "lower",
    "lower_variant",
    "lowerer_version",
    "lowering_profile_digest",
    "profile_for",
    "register_lowerer",
    "register_profile",
    "register_variant_factory",
    "registered_profiles",
    "registered_targets",
    "require_target_profile",
    "resolve_backend_params",
    "variant_factory",
]
