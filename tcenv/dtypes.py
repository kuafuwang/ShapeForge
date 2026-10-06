"""Element types shared by contracts, the Metal runtime and the test harness.

Torch is imported lazily so that light-weight users (prompt building, CLI help)
do not pay the import cost.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DType:
    name: str  # short name used in task/contract definitions, e.g. "fp16"
    msl: str  # Metal Shading Language type, e.g. "half"
    nbytes: int
    kind: str  # "float" | "int" | "uint"
    mtl_datatype: str  # attribute name on the Metal module, for reflection checks
    torch_name: str  # attribute name on torch
    raw_torch_name: str  # integer torch dtype with the same width, used for bit casts

    @property
    def is_float(self) -> bool:
        return self.kind == "float"

    @property
    def torch(self):
        import torch

        return getattr(torch, self.torch_name)

    @property
    def raw_torch(self):
        import torch

        return getattr(torch, self.raw_torch_name)

    @property
    def mtl_code(self) -> int:
        import Metal

        return getattr(Metal, self.mtl_datatype)


DTYPES: dict[str, DType] = {
    d.name: d
    for d in (
        DType("fp16", "half", 2, "float", "MTLDataTypeHalf", "float16", "int16"),
        DType("bf16", "bfloat", 2, "float", "MTLDataTypeBFloat", "bfloat16", "int16"),
        DType("fp32", "float", 4, "float", "MTLDataTypeFloat", "float32", "int32"),
        DType("i32", "int", 4, "int", "MTLDataTypeInt", "int32", "int32"),
        DType("u32", "uint", 4, "uint", "MTLDataTypeUInt", "int32", "int32"),
        DType("i8", "char", 1, "int", "MTLDataTypeChar", "int8", "int8"),
        DType("u8", "uchar", 1, "uint", "MTLDataTypeUChar", "uint8", "uint8"),
    )
}

# Largest finite value per float dtype (used by adversarial input generators).
FLOAT_MAX = {"fp16": 65504.0, "bf16": 3.3895313892515355e38, "fp32": 3.4028234663852886e38}
# Smallest positive subnormal / normal value.
FLOAT_TINY_SUBNORMAL = {"fp16": 2.0**-24, "bf16": 2.0**-133, "fp32": 2.0**-149}
FLOAT_TINY_NORMAL = {"fp16": 2.0**-14, "bf16": 2.0**-126, "fp32": 2.0**-126}
# Machine epsilon (spacing at 1.0).
FLOAT_EPS = {"fp16": 2.0**-10, "bf16": 2.0**-7, "fp32": 2.0**-23}


def get_dtype(name: str) -> DType:
    try:
        return DTYPES[name]
    except KeyError as e:
        raise KeyError(f"unknown dtype {name!r}; known: {sorted(DTYPES)}") from e
