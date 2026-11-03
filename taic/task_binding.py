"""搜索的任务装配分派：v1 保留题走 ``tasks.get_task``，v2 标准目录走 locator + 合同。

这是搜索侧唯一的装题入口。v1 与 v2 之间**没有 fallback**：目录的 ``spec_version`` 决定读哪条路，
``@variant`` 在 v2 上是装配错误（几何是候选自己的声明，不是第二份合同）。v2 的非默认实例要重建
locator——worker 会按 locator 重建任务，父进程的 ABI/几何必须和它是同一个 ``(dtype, size)``。
"""

from __future__ import annotations

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
from tcenv.track.registry import load_track_registry


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
        track = load_track_registry().get(self.track_id)
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
        workspace 张量不占函数参数（运行时按 ``taic-launch`` 的 workspace 声明分配）。

        @return MSL 签名文本。
        """
        if not self.is_v2:
            raise TaskBindingError("only a v2 task derives its signature from the TensorMemoryContract")
        dtypes = {
            "fp16": "half", "bf16": "bfloat", "fp32": "float", "fp64": "double",
            "int32": "int", "uint32": "uint", "int16": "short", "uint16": "ushort",
            "int8": "char", "uint8": "uchar",
        }
        params: list[str] = []
        for index, tensor in enumerate(self.presentation().abi_document["tensors"]):
            if tensor["space"] == "workspace":
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
) -> TaskBinding:
    """@brief 按目录版本装题：v1 走 ``get_task``，v2 走 locator+合同，互不 fallback。

    @param name 题名（可带 ``@variant``；v2 拒绝 variant）。
    @param dtype 数据类型。
    @param size 尺寸档位。
    @param target_id 目标设备 id。
    @param tasks_root 任务根（默认仓库 ``tasks/``）。
    @param code_root 代码闭包根（默认仓库根）。
    @return 绑定。
    @raises TaskBindingError v2 带 variant、v2 未发现该题、或实例装配失败。
    @raises KeyError/ValueError v1 题名或档位非法（``get_task`` 原样抛出）。
    """
    root = Path(tasks_root) if tasks_root is not None else REPO_ROOT / "tasks"
    code = Path(code_root) if code_root is not None else REPO_ROOT
    base, _, variant = name.partition("@")
    version = _v2_spec_version(root, base)
    if version != 2:
        from tasks import get_task

        task = get_task(name, dtype, size)
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
