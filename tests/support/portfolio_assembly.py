"""Source trees for portfolio-assembly tests.

These helpers only write files.  They do not pack, check, or touch a device.
"""

from __future__ import annotations

import json
from pathlib import Path

from tcenv.opspec.v2 import instantiate_v2
from tests.support.kda_fixed_bundle import bundle_entries

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "tests/support/golden/metal/fixtures"

SMALL_GUARD = ["and", ["eq", "M", 256], ["and", ["eq", "N", 256], ["eq", "K", 128]]]
PAPER_GUARD = ["and", ["eq", "M", 2048], ["and", ["eq", "N", 2048], ["eq", "K", 2048]]]
KDA_LOW_GUARD = ["lt", "N", 100]
KDA_HIGH_GUARD = ["ge", "N", 100]


def gemm_variant_source(contract, size: str) -> str:
    """@brief 带 ``taic-launch`` 的 gemm 变体源码。

    网格来自该档的 ``M``/``N``。正文用登记过的 steel fixture。

    @param contract gemm 任务合同。
    @param size ``small`` 或 ``paper``。
    @return MSL 源码。
    """
    instance = instantiate_v2(contract.spec, contract.dtype, size)
    height, width, _depth = (int(instance.symbols[name]) for name in ("M", "N", "K"))
    fixture = "gemm_small_steel.metal" if size == "small" else "gemm_paper_best.metal"
    body = (FIXTURES / fixture).read_text(encoding="utf-8")
    thread_shape = "32,2,2" if size == "small" else "128,1,1"
    return (
        "constant constexpr uint BLOCK_M = 64;\n"
        "constant constexpr uint BLOCK_N = 64;\n"
        "constant constexpr uint BLOCK_K = 32;\n"
        "/* taic-launch\nmath fast\n"
        f"launch gemm_kernel grid={width // 64},{height // 64},1 threads={thread_shape}\n*/\n"
        + body
    )


def write_gemm_tree(root: Path, contract) -> None:
    """@brief 写出 gemm 的两个 single 变体和一份覆盖全部两点的 ``coverage.json``。

    @param root 源目录。
    @param contract gemm 任务合同。
    """
    root = Path(root)
    for size in ("small", "paper"):
        destination = root / "variants" / size / "kernel.metal"
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(gemm_variant_source(contract, size), encoding="utf-8")
    _write_coverage(
        root,
        [
            {"variant_key": "small", "guard": SMALL_GUARD},
            {"variant_key": "paper", "guard": PAPER_GUARD},
        ],
        {"kind": "refuse", "reasons": ["outside_candidate_coverage"]},
        planner_id="gemm-fixture",
    )


def write_kda_tree(root: Path) -> None:
    """@brief 写出 kda 的两个 bundle 变体。

    ``a`` 接 ``N < 100``（small、tail），``b`` 接 ``N >= 100``（paper）。文件字节来自
    ``bundle_entries()``，路径落在 ``variants/<key>/`` 下。

    @param root 源目录。
    """
    root = Path(root)
    grouped = (
        ("a", KDA_LOW_GUARD),
        ("b", KDA_HIGH_GUARD),
    )
    for key, _guard in grouped:
        base = root / "variants" / key
        for entry in bundle_entries():
            destination = base / entry.path
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(entry.data)
    _write_coverage(
        root,
        [
            {"variant_key": "a", "guard": KDA_LOW_GUARD},
            {"variant_key": "b", "guard": KDA_HIGH_GUARD},
        ],
        {"kind": "refuse", "reasons": ["outside_candidate_coverage"]},
        planner_id="kda-fixture",
    )


def _write_coverage(root: Path, variants: list[dict], fallback: dict, *, planner_id: str) -> None:
    """@brief 写 ``coverage.json``。

    @param root 源目录。
    @param variants variant 条目。
    @param fallback fallback 对象。
    @param planner_id planner 名字。
    """
    document = {
        "schema_id": "tcenv.portfolio-assembly.coverage.1",
        "planner": {"id": planner_id, "version": 1},
        "variants": variants,
        "fallback": fallback,
    }
    (root / "coverage.json").write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
