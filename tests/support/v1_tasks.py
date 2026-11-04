"""Specimen tasks for v1-mechanism tests after every v1 task has migrated to v2 (B5).

There is no live v1 task left to assemble: B1–B4 moved them all to the v2 standard entry.
``tasks.get_task`` therefore refuses every name, and the old ``V1_TASK in V1_TASKS`` import-time
assertion made more than a thousand tests fail during collection.

What a v1-mechanism test actually needs is a **v1-shaped ``tcenv.task.Task``**, not a live task:
the object the registry / resume / publication / schema-4 / remote / hw-feature code consumes.
``v1_task`` builds exactly that object from two stable sources:

* the **v2 contract** (``load_task_contract``) for the instance, the reference oracle and the
  registered baselines;
* the **archived launch geometry** (``v1_specimen``) for the kernel name, grid and threads,
  which v2 deliberately leaves to the candidate rather than to a task registry.

``metal_contract`` then synthesises the v1 ``Contract`` from the v2 ABI plus that geometry, and
the rest of the ``Task`` is filled from the v2 instance.  The result is API-compatible with the
old v1 assembly, so mechanism tests keep their assertions; it is not a second task registry and
it never claims to be a production evaluation path (v1's production assembly is retired).
"""

from __future__ import annotations

import dataclasses
import functools
from pathlib import Path
from collections.abc import Mapping
from typing import Any

from tcenv.task import Task
from tcenv.contract import Contract
from tcenv.task import DIAGNOSTIC_META_KEYS
from tcenv.lowering import Specialization
from tests.support.v1_specimen import _BUILDERS, _identity

ROOT = Path(__file__).resolve().parents[2]

#: The default specimen for a v1-mechanism test.  It is an *archived* specimen, not a live task:
#: the test is about the mechanism, not about this task's mathematics.  sem_attention_odd_scale
#: is still the simplest complete example (a real OpSpec, a real reference, vendor baselines).
V1_TASK = "sem_attention_odd_scale"

#: Specimens that have an archived v1 contract and a v2 contract to draw from.  A mechanism test
#: must name one of these instead of assuming that "any v1 task" still exists.
V1_TASKS: tuple[str, ...] = ("sem_attention_odd_scale", "sem_int_5", "sem_int_3")
assert V1_TASK in V1_TASKS, "V1_TASK must have an archived specimen"

#: Representative seeds of the dynamic integer family (now frozen into fixed v2 directories).
#: sem_int_007 is spelled sem_int_7 as an op_name; the directory mapping is handled in v1_task.
V1_DYNAMIC: tuple[str, ...] = ("sem_int_5", "sem_int_3", "sem_int_007")


@dataclasses.dataclass(frozen=True)
class _ArchivedText(Contract):
    """@brief 用归档的 v1 原替换 ``describe()`` / ``abi_stub()`` 的 Contract。

    v2 的 ABI + 归档几何能重建参数槽位与常量，但 v1 的这两段文本还包含描述性文字（线程组的
    SIMD 拆分、参数形状注记）。提示词与反馈逐字节回归靠的就是原文，所以这里直接返回归档文本。
    """

    describe_text: str = ""
    abi_stub_text: str = ""

    def describe(self) -> str:
        """@brief 归档的 v1 契约描述。

        @return 描述文本；没有归档时回退到父类。
        """
        return self.describe_text or super().describe()

    def abi_stub(self) -> str:
        """@brief 归档的 v1 签名骨架。

        @return 骨架文本；没有归档时回退到父类。
        """
        return self.abi_stub_text or super().abi_stub()


def _archived_contract(document: dict) -> Contract:
    """@brief 从 v1 装配记录的 Contract 投影重建一个 ``tcenv.contract.Contract``。

    ``metal_contract`` 由 v2 ABI 派生参数，会丢掉 v1 的 ``doc`` 注记（例如 ``[B, H, S, D]``）与
    标量参数（sem_int 的 ``K`` 是 buffer 而不是编译常量）。归档投影含全部字段，直接用它重建，
    参数顺序、形状、常量与 ``gpu`` 都与迁移前一致。

    @param document ``contract_document()`` 的归档输出。
    @return 重建的 Contract。
    """
    from tcenv.contract import AbiArg

    return Contract(
        kernel_name=document["kernel_name"],
        grid=tuple(document["grid"]),
        threads=tuple(document["threads"]),
        args=tuple(
            AbiArg(
                name=arg["name"], index=arg["index"], role=arg["role"], dtype=arg["dtype"],
                shape=tuple(arg["shape"]), value=arg["value"], doc=arg["doc"],
            )
            for arg in document["args"]
        ),
        constants=tuple((item["name"], item["msl_type"], item["value"]) for item in document["constants"]),
        gpu=document["gpu"],
    )


@dataclasses.dataclass(frozen=True)
class _SpecimenSpecialization(Specialization):
    """@brief 带归档投影的 ``Specialization``。

    ``fingerprint_document`` 用 v1 装配记录的原文（含 Contract 投影与 backend_params），所以
    ``task_fingerprint`` 与迁移前逐位一致；其余字段就是构造时的实值。
    """

    document: dict | None = None

    def fingerprint_document(self) -> dict:
        """@brief 归档的 Specialization 投影。

        @return 投影文档；没有记录时用父类实现。
        """
        if self.document:
            return dict(self.document)
        return super().fingerprint_document()


@dataclasses.dataclass(frozen=True)
class _SpecimenInstance:
    """@brief 包装真实 v2 ``SemanticInstanceV2``，但按 v1 身份投影。

    ``spec_params``/``opspec_digest`` 用 v1 装配的冻结值，其余字段（形状、张量、容差、算例）
    直接委托给真实实例——它们已由 B1–B4 的逐字节比对证明与 v1 相同。``fingerprint_document``
    只覆盖这两处身份字段，不伪造其它内容。

    做成 frozen dataclass 是为了让 ``dataclasses.replace`` 也能作用在投影上（机制测试会用
    ``replace(instance, tolerances=...)`` 构造"分叉"来验证 Task 拒绝它）。

    @ivar inner 真实 v2 实例。
    @ivar identity v1 身份记录。
    @ivar tolerances v1 的容差值；缺省时按归档记录投影。
    """

    inner: Any
    identity: dict = dataclasses.field(default_factory=dict)
    tolerances: Any = None

    def __post_init__(self) -> None:
        """@brief 归档记录里的容差覆盖真实实例的同名字段。

        @raises AttributeError 归档容差含未知字段（说明记录与实例形状不同步）。
        """
        if self.tolerances is None:
            record = (self.identity or {}).get("instance", {}).get("tolerances", {})
            if record:
                names = {f.name for f in dataclasses.fields(self.inner.tolerances)}
                object.__setattr__(self, "tolerances", dataclasses.replace(
                    self.inner.tolerances, **{k: v for k, v in record.items() if k in names},
                ))
            else:
                object.__setattr__(self, "tolerances", self.inner.tolerances)

    @property
    def _inner(self) -> Any:
        """@brief 兼容旧字段名（真实实例）。

        @return 真实 v2 实例。
        """
        return self.inner

    @property
    def _identity(self) -> dict:
        """@brief 兼容旧字段名（v1 身份记录）。

        @return 身份字典。
        """
        return self.identity or {}

    def __getattr__(self, name: str) -> Any:
        """@brief 未覆盖的字段交给真实实例。

        @param name 属性名。
        @return 属性值。
        """
        return getattr(self._inner, name)

    @property
    def spec_params(self) -> dict:
        """@brief v1 的 spec_params（与 v1 Specialization 一致）。

        @return 参数字典。
        """
        return dict(self._identity.get("spec_params", self._inner.spec_params))

    @property
    def case_domain(self) -> Any:
        """@brief v1 的算例域（病例名与 IEEE 注入计划）。

        @return 真实实例的 ``case_domain``。
        """
        return self._inner.case_domain

    @property
    def work_accounting(self) -> Any:
        """@brief 记账字段（flops / min_bytes）。

        @return 记账对象；归档值覆盖真实实例的对应字段。
        """
        record = self._identity.get("instance", {}).get("work_accounting", {})
        if not record:
            return self._inner.work_accounting
        return dataclasses.replace(self._inner.work_accounting, **{
            key: value for key, value in record.items()
            if key in {f.name for f in dataclasses.fields(self._inner.work_accounting)}
        })

    @staticmethod
    def _thaw(value: Any) -> Any:
        """@brief 把只读 meta 视图还原成可比较的普通值（tuple → list）。

        @param value 任意 meta 值。
        @return 普通 JSON 值。
        """
        if isinstance(value, Mapping):
            return {str(key): _SpecimenInstance._thaw(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [_SpecimenInstance._thaw(item) for item in value]
        return value

    def evaluation_meta(self) -> dict:
        """@brief v1 的评测域视图（domain_case / ieee_* / 程序行数）。

        @return 语义键字典。
        """
        # 语义键（domain_case/ieee_*）以真实实例为准；归档只补 v1 的诊断键（如 program_lines），
        # 它们在 v2 里不作为语义域登记。
        # ``Task.__post_init__`` 会把自己的 meta 冻结成 tuple 视图后再比较，所以这里返回的
        # 语义键也必须是解冻后的普通 JSON 值，list 与 tuple 不能混用。
        meta = {key: self._thaw(value) for key, value in dict(self._inner.evaluation_meta()).items()}
        from tcenv.task import DIAGNOSTIC_META_KEYS, SEMANTIC_META_KEYS

        for key, value in self._identity.get("meta", {}).items():
            if key in SEMANTIC_META_KEYS:
                meta[key] = self._thaw(value)  # 语义键以 v1 归档为准（Task.meta 必须与之相等）
        # ``dataclasses.replace(task, ...)`` 会重新跑 ``__post_init__``：那时 Task.meta 已被冻结成
        # tuple 视图，实例的投影必须给出同样的形态，否则 list/tuple 不相等而被误判为分叉。
        from tcenv.task import _frozen_meta

        return {key: _frozen_meta(value) for key, value in meta.items()}

    @property
    def opspec_digest(self) -> str:
        """@brief v1 的 opspec_digest。

        @return 摘要；没有归档时用真实实例的值。
        """
        return self._identity.get("instance_opspec_digest", self._identity.get("opspec_digest", self._inner.opspec_digest))

    def fingerprint_document(self) -> dict:
        """@brief 直接用 v1 装配记录的实例投影。

        v1 与 v2 的投影形状不同（v1 是 ``shapes``，v2 展开成 ``tensors``），混用会让任务
        指纹在迁移前后变成两个值，所以这里用记录的原文而不是现场改写。
        @return 投影文档。
        """
        recorded = self._identity.get("instance")
        if recorded:
            return dict(recorded)
        return self._inner.fingerprint_document()


def v1_kernel_name(name: str = V1_TASK, dtype: str = "fp16", size: str = "small") -> str:
    """@brief 归档样本在 v1 装配下的内核函数名。

    @param name 任务名（``V1_TASKS`` 或 ``V1_DYNAMIC`` 之一）。
    @param dtype 元素类型短名。
    @param size 问题规模名。
    @return 契约声明的内核函数名。
    """
    return v1_task(name, dtype, size).contract.kernel_name


@functools.lru_cache(maxsize=None)
def v1_task(name: str = V1_TASK, dtype: str = "fp16", size: str = "paper") -> Task:
    """@brief 用 v2 的合同 + 归档的启动几何构造一个**与 v1 完全兼容**的 Task。

    ``sem_attention_odd_scale`` 与 ``flash_attention`` 共享 FA profile 的几何；sem_int 动态族的
    每个 seed 的 N 与 threads 相同（``program.BLOCK``/64），所以它们共用 sem_int_5 的几何。
    ``sem_int_007`` 的 op_name 是 ``sem_int_7``，目录名按 int(seed) 取。

    @param name 任务名（``V1_TASKS`` 或 ``V1_DYNAMIC`` 之一）。
    @param dtype 元素类型短名。
    @param size 问题规模名。
    @return 与 v1 装配一致的 ``tcenv.task.Task``。
    @raises KeyError 没有归档样本的题名。
    """
    from tcenv.taskcontract import load_task_contract
    from tcenv.track import REPO_ROOT

    if name not in _BUILDERS:
        raise KeyError(f"{name!r} has no archived v1 specimen (known: {sorted(_BUILDERS)})")
    directory_name = f"sem_int_{int(name.rsplit('_', 1)[1])}" if name.startswith("sem_int_") else name
    directory = REPO_ROOT / "tasks" / directory_name
    dtype = "i32" if name.startswith("sem_int_") else dtype  # sem_int 的 v2 只声明 i32；fp16/bf16 是 v1 的别名
    contract = load_task_contract(directory, dtype=dtype, size=size)
    instance = contract.instance
    # sem_int 的 v2 目录只声明 i32，v1 的 fp16/bf16 是别名；归档的 v1 合同/题面对两个别名
    # 完全相同，所以形状也用同一个 i32 记录，不受调用方写的别名影响。
    shape = _BUILDERS[name]("i32" if name.startswith("sem_int_") else dtype, size)
    # 身份来自归档记录（builder 只负责启动几何；lion/conv2d 这类早期样本的 builder 不带身份）。
    ident = shape.identity or _identity(name, dtype, size)
    # 合同优先**直接从归档投影重建**：v1 的 Contract 带着它的 ``doc`` 注记、标量槽位与常量，
    # 由 v2 ABI 重新派生会丢掉这些（sem_int 的 K 是标量参数）。直接重建还让本函数不 import
    # 任何 Metal 后端模块——后端中立测试要求通用路径不触碰 Metal。
    archived_contract = ident.get("specialization", {}).get("contract")
    if archived_contract:
        metal = _archived_contract(archived_contract)
    else:
        from tcenv.backends.metal.track_adapter import metal_contract
        from tcenv.target import load_target
        from tcenv.track.abi import derive_tensor_memory_contract
        from tcenv.track.geometry import ResolvedGeometry, ResolvedStage

        abi = derive_tensor_memory_contract(instance, target=load_target("apple-m4-max"))
        stage = ResolvedStage(
            index=0, kernel=shape.contract.kernel_name, grid=shape.contract.grid,
            threads=shape.contract.threads, threadgroup_bytes=0, workspace=(),
            active=True, when=None, derived_scalars=(),
        )
        metal = metal_contract(
            task_contract=contract, abi=abi,
            geometry=ResolvedGeometry(source="candidate", stages=(stage,)), target_id="apple-m4-max",
        )
        archived = tuple(shape.contract.constants)
        seen = {item[0] for item in archived}
        metal = dataclasses.replace(metal, constants=archived + tuple(
            item for item in metal.constants if item[0] not in seen
        ))
    if shape.contract.describe_text or shape.contract.abi_stub_text:
        metal = _ArchivedText(
            kernel_name=metal.kernel_name, grid=metal.grid, threads=metal.threads, args=metal.args,
            constants=metal.constants, gpu=metal.gpu,
            describe_text=shape.contract.describe_text, abi_stub_text=shape.contract.abi_stub_text,
        )
    # 身份一律用 v1 装配的冻结值：opspec_digest 与 lowering_profile_digest 在迁移前是什么样，
    # 现在还是什么样，历史 run 的指纹/评测缓存/远端 epoch 才不会被判成另一个问题。
    specialization = _SpecimenSpecialization(
        opspec_digest=ident.get("opspec_digest", contract.spec.digest),
        lowering_profile_digest=ident.get("lowering_profile_digest", contract.fingerprint()),
        spec_params=ident.get("spec_params", instance.spec_params),
        backend_params=ident.get("backend_params", {}),
        contract=metal,
        document=ident.get("specialization"),
    )
    # v1 的 Task.name 保留 seed 原名（sem_int_007），即使 v2 的 op_name 是 int(seed) 后的写法
    # （sem_int_7）；名字进指纹，所以这里用调用方给的题名，不用合同的 op_name。
    semantic = _SpecimenInstance(instance, ident)
    return Task(
        name=name,
        dtype=instance.dtype,
        size=instance.size,
        contract=metal,
        naive_msl=ident.get("naive", shape.naive_msl),
        triton_source=shape.triton_source or ident.get("triton_source") or contract.statement,
        description=ident.get("description") or shape.description or contract.statement,
        cases=tuple(ident.get("cases", instance.case_domain.cases)),
        gen_inputs=lambda case, seed: contract.reference.generate(instance, case, seed),
        reference=lambda inputs: contract.reference.execute(instance, inputs),
        baselines=contract.task_baselines(),
        flops=semantic.work_accounting.flops,
        min_bytes=semantic.work_accounting.min_bytes,
        rtol=semantic.tolerances.rtol,
        atol_rel=semantic.tolerances.atol_rel,
        exact=semantic.tolerances.exact,
        error_scale=(
            (lambda inputs, refs: contract.reference.error_scale(instance, inputs, refs))
            if contract.reference.error_scale_kind not in ("none", "")
            else None
        ),
        semantic_instance=semantic,
        specialization=specialization,
        # 诊断键（program_lines 等）不在 ``evaluation_meta`` 里，单独并进 Task.meta。
        meta={**semantic.evaluation_meta(), **{
            key: value for key, value in ident.get("meta", {}).items() if key in DIAGNOSTIC_META_KEYS
        }},
        evaluation_implementations=dict(ident.get("evaluation_implementations", {})),
    )


def bind_specimen(name: str, dtype: str, size: str, **kwargs):
    """@brief 测试专用的 v1 绑定：任何题名都交给 ``v1_task`` 装配。

    生产 ``bind_task`` 按目录的 ``spec_version`` 分流，v1 活题为零以后所有题名都进 v2 分支。
    机制测试要的是 v1 的绑定路径本身，所以显式走这条入口，而不是让生产代码为了测试去改变
    分流规则。v2 的装配路径有 ``bind_task`` 与 ``test_search_v2_orchestrator`` 覆盖。

    @param name 题名。
    @param dtype 元素类型。
    @param size 尺寸档位。
    @param kwargs 传给 ``bind_task`` 的其余参数（如 target_id）。
    @return v1 模式的 ``TaskBinding``。
    """
    from taic.task_binding import bind_task

    return bind_task(name, dtype, size, v1_loader=v1_task, v1_only=True, **kwargs)


def v1_pack_digest(name: str, dtype: str = "fp16", size: str = "paper") -> str:
    """@brief 归档 v1 任务登记的 task-pack 摘要（``local_provenance`` 的评测者世代输入）。

    @param name 题名。
    @param dtype 元素类型。
    @param size 尺寸档位。
    @return 摘要；没有登记时返回空串（由调用方决定如何报错）。
    """
    task = v1_task(name, dtype, size)
    pack = (task.evaluation_implementations or {}).get("task_pack")
    return str(pack.get("digest") or "") if isinstance(pack, dict) else ""


def production_get_task(name: str, dtype: str = "fp16", size: str = "paper"):
    """@brief 真实生产 v1 装配（**不受测试夹具影响**），供"拒绝已迁任务"的断言使用。

    ``tests.conftest`` 的 autouse fixture 会把 ``tasks.get_task`` 指向 specimen；需要验证生产拒绝
    路径的测试必须显式调用本函数，它从注册表保存的原始引用取装配器。

    @param name 题名。
    @param dtype 元素类型。
    @param size 尺寸档位。
    @return 生产装配的 Task（已迁题会抛 ``TaskContractError``）。
    """
    import tasks

    original = tasks.__dict__.get("_production_get_task") or tasks.__dict__["get_task"]
    return original(name, dtype, size)
