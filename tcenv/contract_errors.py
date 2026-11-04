"""Typed failures shared by the task-contract and track planes.

This module has no package imports on purpose: both :mod:`tcenv.taskcontract` and
:mod:`tcenv.track` need the error types, and putting them in either package would create an
import cycle between the two planes.  ``tcenv.taskcontract.errors`` re-exports them.
"""

from __future__ import annotations


class TaskContractError(ValueError):
    """@brief v2 任务合同/交付平面的基类错误（带稳定 ``code``）。"""

    code = "task_contract_invalid"


class AssemblyError(TaskContractError):
    """@brief 目录/文件/接口的结构装配错误（含 v1/v2 混合目录）。"""

    code = "assembly_error"


class SemanticUnresolved(TaskContractError):
    """@brief OpSpec/reference/表示冲突，必须人工或已知答案仲裁，不能自动选边。"""

    code = "semantic_unresolved"


class UnsupportedContractCapability(TaskContractError):
    """@brief 超出本合同版本的能力，禁止回退到 v1 或伪造实现。"""

    code = "unsupported_contract_capability"


class ReferenceBoundaryError(TaskContractError):
    """@brief reference 越界：import 候选/renderer/known-good、改写输入或返回非法结果。"""

    code = "reference_boundary_violation"


class GeometryError(TaskContractError):
    """@brief 执行几何/派生标量/参数绑定的 typed 错误（A2）。"""

    code = "execution_geometry_invalid"


class VarlenContractError(TaskContractError):
    """@brief varlen 运行时张量不满足 L1 分段合同（边界形状/单调/端点/i32 域）。"""

    code = "varlen_contract_invalid"


__all__ = [
    "AssemblyError",
    "GeometryError",
    "ReferenceBoundaryError",
    "SemanticUnresolved",
    "TaskContractError",
    "UnsupportedContractCapability",
    "VarlenContractError",
]
