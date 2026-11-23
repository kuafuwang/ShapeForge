"""Read-only access to the frozen v1 specimen records.

This module is a **historical input reader**, not a task assembly path.  It never builds a
``tcenv.task.Task``, never touches the retired v1 registry and is never imported by production
code.  Two kinds of bytes are exposed:

* the archived candidate sources under
  ``tests/support/golden/shared/acceptance-epochs/fixtures/<batch>/evidence/candidates/``;
* the archived v1 contract text in ``tests/support/golden/shared/v1-contract-text.json``.

On top of those, :func:`contract_view` renders the v1 LoweringProfile constant header from an
explicit table.  The tables below are the recorded v1 constants of three specimens that migrated
to v2; they are kept here so hardware-analysis tests can feed the *same* header the archived
evidence was produced with.  The current OpSpec constants are a different contract and must not
be substituted for them.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
_FIXTURES = ROOT / "tests/support/golden/shared/acceptance-epochs/fixtures"
_CONTRACT_TEXT = ROOT / "tests/support/golden/shared/v1-contract-text.json"

#: Archived v1 constants of the elementwise ``lion`` specimen (paper instance, fp16/bf16).
LION_CONSTANTS: tuple[tuple[str, str, Any], ...] = (
    ("N", "uint", 8388608),
    ("BLOCK_SIZE", "uint", 1024),
    ("LR", "float", 0.001),
    ("WD", "float", 0.1),
    ("BETA1", "float", 0.9),
    ("BETA2", "float", 0.99),
)

#: Archived v1 constants of the compute ``bitdelta_gemm`` specimen (paper instance).
BITDELTA_GEMM_CONSTANTS: tuple[tuple[str, str, Any], ...] = (
    ("M", "uint", 512),
    ("N", "uint", 4096),
    ("K", "uint", 4096),
    ("KB", "uint", 128),
    ("BLOCK_M", "uint", 64),
    ("BLOCK_N", "uint", 64),
    ("BLOCK_K", "uint", 32),
)

#: Archived v1 constants of the attention specimens (paper instance; both share the FA profile).
ATTENTION_CONSTANTS: tuple[tuple[str, str, Any], ...] = (
    ("B", "uint", 2),
    ("H", "uint", 8),
    ("S", "uint", 1024),
    ("D", "uint", 64),
    ("BLOCK_M", "uint", 32),
    ("BLOCK_N", "uint", 32),
    ("SM_SCALE", "float", 0.125),
)

#: Archived v1 constants of the ``conv2d`` specimen (paper instance).
CONV2D_CONSTANTS: tuple[tuple[str, str, Any], ...] = (
    ("N", "uint", 32),
    ("H", "uint", 56),
    ("W", "uint", 56),
    ("C", "uint", 64),
    ("OC", "uint", 128),
    ("KH", "uint", 3),
    ("KW", "uint", 3),
    ("OH", "uint", 54),
    ("OW", "uint", 54),
    ("M_TOTAL", "uint", 93312),
    ("K_TOTAL", "uint", 576),
    ("BLOCK_M", "uint", 64),
    ("BLOCK_N", "uint", 64),
    ("BLOCK_K", "uint", 32),
)


def candidate(task: str, dtype: str = "fp16", *, batch: str) -> str:
    """Return one archived candidate source with its dtype placeholder rendered.

    @param task 题名。
    @param dtype 元素类型短名。
    @param batch 归档批次（``b1``–``b4``）。
    @return MSL 源文本。
    """
    path = _FIXTURES / batch / "evidence" / "candidates" / f"{task}-naive.msl"
    spelling = "half" if dtype == "fp16" else "bfloat"
    return path.read_text(encoding="utf-8").replace("{{T}}", spelling)


def contract_text(task: str, dtype: str = "fp16", size: str = "paper") -> dict:
    """Return the archived v1 ``describe``/``abi_stub`` text for one specimen.

    @param task 题名。
    @param dtype 元素类型短名。
    @param size 问题规模名。
    @return 含 ``describe``/``abi_stub`` 的字典。
    """
    data = json.loads(_CONTRACT_TEXT.read_text(encoding="utf-8"))
    return data.get(f"{task}|{dtype}|{size}", data.get(f"{task}|{dtype}", data[task]))


def render_header(constants: tuple[tuple[str, str, Any], ...]) -> str:
    """Render the v1 ``Contract.render_header`` text from a constant table.

    @param constants ``(名字, MSL 类型, 数值)`` 三元组。
    @return 常量声明头。
    """
    prefix = "#include <metal_stdlib>\nusing namespace metal;\n"
    suffix = {"uint": "u", "float": "f"}
    return prefix + "".join(
        f"constant constexpr {msl_type} {name} = {value}{suffix.get(msl_type, '')};\n"
        for name, msl_type, value in constants
    )


def contract_view(
    kernel_name: str,
    constants: tuple[tuple[str, str, Any], ...],
    *,
    describe_text: str = "",
    abi_stub_text: str = "",
    arg_dtypes: tuple[str, ...] = (),
) -> SimpleNamespace:
    """Return the ``contract`` shape ``taic.hwfeatures.build_report`` reads.

    Only ``constants``, ``kernel_name``, ``render_header()``, ``describe()``, ``abi_stub()`` and
    the argument storage types are provided; there is no task identity, no evaluation path and no
    registry entry.

    @param kernel_name 归档的核名。
    @param constants 归档的常量表。
    @param describe_text 归档的 v1 合同描述。
    @param abi_stub_text 归档的 v1 签名骨架。
    @param arg_dtypes 归档的参数存储类型（仅供提示词骨架使用）。
    @return 只读合同视图。
    """
    header = render_header(constants)

    def describe() -> str:
        if describe_text:
            return describe_text
        lines = [f"kernel {kernel_name}", "constants:"]
        lines.extend(f"  {name} ({msl_type}) = {value}" for name, msl_type, value in constants)
        return "\n".join(lines)

    def abi_stub() -> str:
        return abi_stub_text or f"kernel void {kernel_name}(...);"

    return SimpleNamespace(
        kernel_name=kernel_name, constants=constants, render_header=lambda: header,
        describe=describe, abi_stub=abi_stub,
        args=tuple(SimpleNamespace(name=f"a{index}", dtype=dtype) for index, dtype in enumerate(arg_dtypes)),
    )


__all__ = [
    "ATTENTION_CONSTANTS",
    "BITDELTA_GEMM_CONSTANTS",
    "CONV2D_CONSTANTS",
    "LION_CONSTANTS",
    "candidate",
    "contract_text",
    "contract_view",
    "render_header",
]
