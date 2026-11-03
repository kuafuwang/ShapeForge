#!/usr/bin/env python3
"""B1: prove the migrated v2 contracts reproduce every frozen v1 byte.

For each of the 8 ``(task, dtype, size)`` combinations and each of the 18 ``(case, seed)``
records, this replays the v2 contract exactly the way the v2 evaluator consumes it:

    logical = contract.reference.generate(instance, case, seed)
    rounded = round_to(logical[name], dtype)
    outputs = contract.reference.execute(instance, rounded)
    scale   = default RMS scale (both tasks registered ``error_scale: None`` in v1)

and compares sha256 of the raw input bytes, the fp64 reference bytes and the scale bytes with
``evidence/v1-frozen.json``.  It also checks the semantic document survived the mapping:
cases, tolerances, sizes, work accounting, tensor order, attrs and the baseline identity.

Exit code 0 only when every record matches.  Any mismatch is a migration defect, not a
tolerance question: these are the same mathematical oracle and the same seed stream.

Run:  uv run python docs/架构/跨硬件算子生成分工-阶段验收/B1/compare.py [--task gemv]
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

B1 = Path(__file__).resolve().parent
ROOT = B1.parents[3]
EVIDENCE = B1 / "evidence"


def sha256(raw: bytes) -> str:
    """@brief 字节串的 sha256。

    @param raw 原始字节。
    @return 十六进制摘要。
    """
    return hashlib.sha256(raw).hexdigest()


def raw64(value: Any) -> bytes:
    """@brief 张量的 float64 连续字节（与冻结脚本同规则）。

    @param value 张量。
    @return 小端 float64 字节。
    """
    return value.detach().to("cpu").double().contiguous().numpy().tobytes()


def default_scale(reference_outputs: dict[str, Any], *, finite_only: bool) -> dict[str, Any]:
    """@brief v1 默认误差尺度：RMS；``ieee`` 算例只取有限部分（v1 ``prepare_case`` 的规则）。

    @param reference_outputs reference 输出。
    @param finite_only 是否把 NaN/Inf 置零后再算 RMS。
    @return 每个输出的标量 RMS。
    """
    import torch

    scale = {}
    for name, value in reference_outputs.items():
        if finite_only and value.is_floating_point():
            value = torch.where(torch.isfinite(value), value, torch.zeros_like(value))
        scale[name] = torch.sqrt(torch.mean(value.to(torch.float64) ** 2))
    return scale


def compare_semantics(contract: Any, combo: dict[str, Any]) -> list[str]:
    """@brief 语义文档在 v1→v2 映射后保持一致。

    @param contract v2 合同。
    @param combo 冻结组合。
    @return 通过的检查项描述。
    """
    instance = contract.instance
    spec = combo["opspec"]
    where = f"{combo['task']}/{combo['dtype']}/{combo['size']}"
    v2_document = json.loads((ROOT / "tasks" / combo["task"] / "opspec.json").read_text(encoding="utf-8"))
    failures: list[str] = []
    if list(instance.case_domain.cases) != list(combo["cases"]):
        failures.append(f"{where}: cases {list(instance.case_domain.cases)} != {combo['cases']}")
    # 语义字段必须原样跨过 v1→v2 映射：只有 spec_version/uses/discovery 允许变化。
    for key in ("numerics", "sizes", "work_accounting", "attrs", "tensors", "symbols", "spec_params", "formula", "op_name", "category"):
        if v2_document.get(key) != spec.get(key):
            failures.append(f"{where}: opspec field {key} changed during migration")
    if "uses" in v2_document or v2_document.get("spec_version") != 2:
        failures.append(f"{where}: v2 document still carries v1 markers")
    expected_inputs = [name for name, tensor in spec["tensors"].items() if tensor["role"] == "input"]
    if list(instance.reference_inputs) != expected_inputs:
        failures.append(f"{where}: input order {list(instance.reference_inputs)} != {expected_inputs}")
    checks = [
        f"cases:{len(combo['cases'])}",
        f"inputs:{len(instance.reference_inputs)}",
        f"outputs:{len(instance.reference_outputs)}",
        "opspec:semantics-preserved",
    ]
    if failures:
        raise AssertionError("; ".join(failures))
    return checks


def compare_combo(combo: dict[str, Any]) -> dict[str, Any]:
    """@brief 对照一个组合的全部算例。

    @param combo 冻结组合。
    @return 结果摘要（记录数与逐字节命中数）。
    """
    import torch  # noqa: F401  保证与冻结脚本同一后端
    from tcenv.taskcontract import load_task_contract
    from tcenv.tensors import round_to, to_raw

    task_dir = ROOT / "tasks" / combo["task"]
    contract = load_task_contract(task_dir, dtype=combo["dtype"], size=combo["size"])
    instance = contract.instance
    where = f"{combo['task']}/{combo['dtype']}/{combo['size']}"
    checks = compare_semantics(contract, combo)
    exact_input = exact_reference = exact_scale = 0
    for record in combo["records"]:
        label = f"{where}/{record['case']}/{record['seed']}"
        logical = contract.reference.generate(instance, record["case"], record["seed"])
        if set(logical) != set(instance.reference_inputs):
            raise AssertionError(f"{label}: gen_inputs keys {sorted(logical)} != {list(instance.reference_inputs)}")
        rounded = {}
        for name in instance.reference_inputs:
            shape = tuple(int(v) for v in instance.tensor(name).shape)
            rounded[name] = round_to(logical[name].reshape(shape), instance.tensor(name).dtype)
            raw = to_raw(rounded[name], instance.tensor(name).dtype)
            expected = record["inputs"][name]
            if len(raw) != expected["nbytes"] or sha256(raw) != expected["sha256"]:
                raise AssertionError(f"{label}: input {name} bytes differ from frozen v1")
            exact_input += 1
        outputs = contract.reference.execute(instance, rounded)
        if set(outputs) != set(instance.reference_outputs):
            raise AssertionError(f"{label}: reference keys {sorted(outputs)}")
        for name, value in outputs.items():
            piece = raw64(value)
            expected = record["reference"][name]
            if len(piece) != expected["nbytes"] or sha256(piece) != expected["sha256"]:
                raise AssertionError(f"{label}: reference {name} bytes differ from frozen v1")
            exact_reference += 1
        scale = default_scale(outputs, finite_only=record["case"] == "ieee")
        for name, value in scale.items():
            piece = raw64(value)
            expected = record["scale"][name]
            if sha256(piece) != expected["sha256"]:
                raise AssertionError(f"{label}: scale {name} differs from frozen v1")
            exact_scale += 1
    return {
        "combo": where,
        "records": len(combo["records"]),
        "exact_input_bytes": exact_input,
        "exact_reference_bytes": exact_reference,
        "exact_scale_bytes": exact_scale,
        "checks": checks,
    }


def main() -> None:
    """@brief 跑全部组合并写比对报告。

    @return 无。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=["gemv", "rope"], default=None)
    args = parser.parse_args()
    frozen = json.loads((EVIDENCE / "v1-frozen.json").read_text(encoding="utf-8"))
    results = []
    for combo in frozen["combos"]:
        if args.task and combo["task"] != args.task:
            continue
        result = compare_combo(combo)
        results.append(result)
        print(
            f"OK {result['combo']}: {result['records']} records, "
            f"input={result['exact_input_bytes']} reference={result['exact_reference_bytes']} scale={result['exact_scale_bytes']}",
            flush=True,
        )
    report = {"schema_version": 1, "phase": "B1", "results": results, "ok": True}
    out = EVIDENCE / "compare-report.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"wrote {out}: {len(results)} combos, all byte-exact")


if __name__ == "__main__":
    main()
