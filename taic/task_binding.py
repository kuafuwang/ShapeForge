"""搜索的任务装配分派：v1 保留题走 ``tasks.get_task``，v2 标准目录走 locator + 合同。

这是搜索侧唯一的装题入口。v1 与 v2 之间**没有 fallback**：目录的 ``spec_version`` 决定读哪条路，
``@variant`` 在 v2 上是装配错误（几何是候选自己的声明，不是第二份合同）。v2 的非默认实例要重建
locator——worker 会按 locator 重建任务，父进程的 ABI/几何必须和它是同一个 ``(dtype, size)``。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tcenv.contract_errors import TaskContractError
from tcenv.identity import fingerprint
from tcenv.target import DEFAULT_TARGET_ID
from tcenv.taskcontract import discover_standard_tasks, load_task_contract
from tcenv.taskcontract.loader import TaskContract
from tcenv.taskcontract.locator import TaskLocator
from tcenv.taskcontract.pack import task_pack_closure
from tcenv.track import REPO_ROOT
from tcenv.track.abi import derive_tensor_memory_contract
from tcenv.track.plane import TrackRenderer


class TaskBindingError(ValueError):
    """@brief 题名/档位在两套装配里都不合法，或跨代混用。"""


@dataclass(frozen=True)
class TaskBinding:
    """@brief 一次搜索绑定的那道题（v1 ``Task`` 或 v2 ``TaskContract``+locator）。"""

    mode: str  # "v1" | "v2"
    name: str
    dtype: str
    size: str
    target_id: str
    task: Any = None  # v1 tcenv.task.Task
    locator: TaskLocator | None = None
    contract: TaskContract | None = None
    track_id: str = "msl-single"
    track_version: int | None = None

    @property
    def track(self):
        """Resolve the frozen ``(track_id, track_version)`` selection for control-host rendering.

        ``None`` means "this binding predates version selection" and resolves the registry default
        (the highest selectable version).  A resumed run always carries the recorded version, so
        publishing a new version never silently upgrades an old run's renderer/evaluator path.
        """
        from tcenv.track.registry import load_track_registry

        return load_track_registry().get(self.track_id, version=self.track_version)

    @property
    def is_v2(self) -> bool:
        """@brief 是否 v2 标准目录。

        @return True/False。
        """
        return self.mode == "v2"

    @property
    def kernel_name(self) -> str:
        """@brief v1 的契约核名；v2 没有契约核名（候选自己声明），返回空串。

        @return 核名或 ``""``。
        """
        return "" if self.is_v2 else str(self.task.contract.kernel_name)

    @property
    def task_pack_digest(self) -> str | None:
        """@brief v2 的部署包摘要；v1 为 None（v1 用 track 登记的 pack）。

        @return 摘要或 None。
        """
        return self.locator.deployment_pack_digest if self.locator is not None else None

    @property
    def contract_fingerprint(self) -> str:
        """@brief 本运行写进请求/配置的任务身份。

        v1 用 ``task_fingerprint``（SemanticInstance+Specialization+Triton 源），v2 用
        ``TaskContract.fingerprint()``（OpSpec+reference+baseline+display+pack）。两者语义不同，
        由 ``mode`` 区分，绝不互相冒充。

        @return 十六进制摘要。
        """
        if self.is_v2:
            return self.contract.fingerprint()
        from tcenv.task import task_fingerprint
        from tcenv.target import load_target

        return task_fingerprint(self.task, load_target(self.target_id).backend)

    def tolerances(self) -> dict[str, Any]:
        """@brief 失败记录里兜底的容差（报告里没带 tolerance 时才用）。

        @return ``{"rtol","atol_rel","exact"}``。
        """
        if not self.is_v2:
            return {"rtol": self.task.rtol, "atol_rel": self.task.atol_rel, "exact": self.task.exact}
        tolerances = self.contract.instance.tolerances
        return {
            "rtol": float(tolerances.rtol),
            "atol_rel": float(tolerances.atol_rel),
            "exact": bool(tolerances.exact),
        }

    def presentation(self) -> Any:
        """@brief v2 的统一题面（TrackRenderer，与 evaluate_task 用的是同一个渲染器）。

        @return ``TaskPresentation``。
        """
        if not self.is_v2:
            raise TaskBindingError("only a v2 task has a TrackRenderer presentation")
        from tcenv.target import load_target

        profile = load_target(self.target_id)
        track = self.track
        track.require_available()
        abi = derive_tensor_memory_contract(self.contract.instance, target=profile)
        rendered = TrackRenderer.for_track(track).render(
            backend_id=profile.backend,
            instance=self.contract.instance,
            reference_source=self.contract.reference.source,
            contract=abi,
            display=self.contract.display,
            track=track,
        )
        return rendered

    def signature_stub(self) -> str:
        """@brief v2 的签名骨架（ABI 声明顺序的 buffer 参数 + 常用线程索引）。

        符号/attrs 不作为函数参数：它们由编译头以常量形式提供（``metal_contract``）。

        交付机制决定 workspace 槽位：``msl-single`` 里 workspace 不占函数参数（运行时按
        ``taic-launch`` 的 workspace 声明分配），所以 single 的输出逐字节不变；``msl-bundle`` 里
        workspace **是**按名字绑定的 kernel 参数（``stage_binding`` 要求本 stage 声明它，且可写性
        与 ``access`` 一致），漏掉它就会被 ABI 拒。方法签名不变、读的是 ``self.track_id``，
        调用方不需要（也不许）另外传一个 track。

        @return MSL 签名文本。
        """
        if not self.is_v2:
            raise TaskBindingError("only a v2 task derives its signature from the TensorMemoryContract")
        dtypes = {
            "fp16": "half", "bf16": "bfloat", "fp32": "float", "fp64": "double",
            "int32": "int", "uint32": "uint", "int16": "short", "uint16": "ushort",
            "int8": "char", "uint8": "uchar",
            "i32": "int", "u32": "uint", "i16": "short", "u16": "ushort",
            "i8": "char", "u8": "uchar",
        }
        from .track_prompt import prompt_style_for_track

        style = prompt_style_for_track(self.track)
        if style.style_id == "tilelang-pipeline":
            tilelang_dtypes = {
                "fp16": "float16", "bf16": "bfloat16", "fp32": "float32", "fp64": "float64",
                "int32": "int32", "uint32": "uint32", "int16": "int16", "uint16": "uint16",
                "int8": "int8", "uint8": "uint8",
                "i32": "int32", "u32": "uint32", "i16": "int16", "u16": "uint16",
                "i8": "int8", "u8": "uint8",
            }
            parameters = []
            for tensor in self.presentation().abi_document["tensors"]:
                dimensions = ", ".join(str(term) for term in tensor["shape"])
                parameters.append(
                    f"    {tensor['name']}: T.Tensor(({dimensions}), \"{tilelang_dtypes[tensor['dtype']]}\"),"
                    f"  # {tensor['space']}/{tensor['access']}"
                )
            name = f"{self.contract.op_name}_kernel"
            return (
                "import tilelang\nimport tilelang.metal.language as T\n\n"
                "@T.prim_func\n"
                f"def {name}(\n" + "\n".join(parameters) + "\n):\n"
                "    with T.Kernel(...):\n"
                "        ...\n\n"
                f"KERNEL = {name}  # use KERNELS = (stage0, stage1, ...) for a candidate-owned work-list"
            )
        bundle = style.style_id == "msl-bundle"
        params: list[str] = []
        for index, tensor in enumerate(self.presentation().abi_document["tensors"]):
            if tensor["space"] == "workspace" and not bundle:
                continue
            const = "const " if tensor["access"] == "read" else ""
            params.append(
                f"    device {const}{dtypes[tensor['dtype']]}* {tensor['name']} [[buffer({index})]]"
            )
        params.append("    uint3 tgid [[threadgroup_position_in_grid]]")
        params.append("    uint3 tid [[thread_position_in_threadgroup]]")
        return f"kernel void {self.contract.op_name}_kernel(\n" + ",\n".join(params) + "\n)"

    def presentation_sha256(self) -> str:
        """@brief 题面渲染身份（与 runner 的 canonical 拼写一致）。

        @return 十六进制摘要。
        """
        p = self.presentation()
        return fingerprint({
            "track_id": p.track_id,
            "track_version": p.track_version,
            "statement": p.statement,
            "reference_source": p.reference_source,
            "abi_document": p.abi_document,
        })


def _v2_spec_version(tasks_root: Path, base: str) -> int | None:
    """@brief 读目录的 spec_version（目录不存在或文件不可读时返回 None）。

    @param tasks_root 任务根。
    @param base 不含 variant 的题名。
    @return 1/2 或 None。
    """
    import json

    spec = tasks_root / base / "opspec.json"
    try:
        if not spec.is_file():
            return None
        return json.loads(spec.read_text(encoding="utf-8")).get("spec_version")
    except (OSError, ValueError):
        return None


def bind_task(
    name: str,
    dtype: str,
    size: str,
    *,
    target_id: str = DEFAULT_TARGET_ID,
    tasks_root: str | Path | None = None,
    code_root: str | Path | None = None,
    v1_loader: Callable[[str, str, str], Any] | None = None,
    v1_only: bool = False,
) -> TaskBinding:
    """@brief 按目录版本装题：v1 走 ``get_task``，v2 走 locator+合同，互不 fallback。

    ``v1_loader`` 是给 v1 **机制**测试留的注入口：所有题都迁到 v2 之后，生产装配不再有任何
    v1 活题，但机制测试（提示词、编排、resume、发布门）仍需要一个 v1 形状的 Task 作为输入。
    调用方可以传 ``tests.support.v1_tasks.v1_task``，由归档合同构造同一个对象。生产不传，
    因此不存在的 v1 题仍然照常被拒绝，不会因为测试夹具而"又能装了"。
    该 loader 只在 v1 分支被调用，永远不会改变 v2 的发现与装配。

    @param name 题名（可带 ``@variant``；v2 拒绝 variant）。
    @param dtype 数据类型。
    @param size 尺寸档位。
    @param target_id 目标设备 id。
    @param tasks_root 任务根（默认仓库 ``tasks/``）。
    @param code_root 代码闭包根（默认仓库根）。
    @return 绑定。
    @param v1_loader 可选的 v1 装配函数（机制测试用），签名与 ``tasks.get_task`` 相同。
    @param v1_only 显式要求 v1 绑定（机制测试用），跳过目录版本分流。
    @raises TaskBindingError v2 带 variant、v2 未发现该题、或实例装配失败。
    @raises KeyError/ValueError v1 题名或档位非法（``get_task`` 原样抛出）。
    """
    root = Path(tasks_root) if tasks_root is not None else REPO_ROOT / "tasks"
    code = Path(code_root) if code_root is not None else REPO_ROOT
    base, _, variant = name.partition("@")
    # 版本分流只有一个来源：默认读目录的 spec_version。测试可以安装一个分流函数
    # （``tests.conftest`` 把 ``TAIC_BIND_TASK_ROUTE`` 设为 "specimen"），让 v1 机制测试显式走 v1
    # 分支；生产不设置该变量，分流完全由目录决定。
    route = os.environ.get("TAIC_BIND_TASK_ROUTE", "")
    if v1_only:
        route = "v1"
    if route == "specimen":
        # specimen 分流只对**有归档 v1 合同**的题名生效。v2 的搜索/portfolio 测试用 bind_task
        # 装配的是 v2 任务（不在归档里），它们必须照常走目录版本分流，否则 v2 合同会被一份
        # v1 specimen 顶掉，确认种子与身份随之错位。
        from tests.support.v1_specimen import _BUILDERS
        from tests.support.v1_tasks import v1_task

        if base in _BUILDERS:
            v1_only, v1_loader = True, v1_task
    version = 1 if v1_only else _v2_spec_version(root, base)
    if version != 2:
        if v1_loader is None:
            from tasks import get_task

            v1_loader = get_task
        task = v1_loader(name, dtype, size)
        return TaskBinding(mode="v1", name=name, dtype=dtype, size=size, target_id=target_id, task=task)
    if variant:
        raise TaskBindingError(
            f"task {name!r}: v2 has no @variant assembly; a variant is a candidate-owned launch choice, "
            "not a second task contract"
        )
    found = [item for item in discover_standard_tasks(root, code_root=code) if item.op_name == base]
    if not found:
        raise TaskBindingError(f"task {name!r}: the v2 standard entry did not discover this task under {root}")
    if len(found) > 1:
        raise TaskBindingError(f"task {name!r}: more than one v2 task declares op_name {base!r}")
    locator = found[0]
    if (locator.dtype, locator.size) != (dtype, size):
        contract = load_task_contract(locator.task_dir, dtype=dtype, size=size)
        locator = TaskLocator(
            task_id=locator.task_id,
            op_name=contract.op_name,
            dtype=contract.dtype,
            size=contract.size,
            pack_digest=contract.pack_digest,
            deployment_pack_digest=task_pack_closure(locator.task_dir, root=code).digest,
            task_dir=locator.task_dir,
            code_root=code,
        )
    contract = locator.load()
    if (contract.dtype, contract.size) != (dtype, size):
        raise TaskBindingError(
            f"task {name!r}: assembled {contract.dtype}/{contract.size} instead of the requested {dtype}/{size}"
        )
    return TaskBinding(
        mode="v2", name=name, dtype=dtype, size=size, target_id=target_id,
        locator=locator, contract=contract,
    )


__all__ = ["TaskBinding", "TaskBindingError", "bind_task"]
