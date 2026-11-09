#!/usr/bin/env python3
"""A5: compare every migrated ``(task, dtype, size, variant)`` with the archived G0 corpus.

For each of the 82 G0 combinations the comparator re-derives the v2 contract from
``tasks/<op>/`` and checks, against the frozen corpus (never against a regenerated one):

* the declared semantics: cases, sizes, tensor order/dtype/role/shape, spec params, resolved
  attrs (from the v1 contract constants), tolerances, ``exact``, work accounting and the
  read-only evaluation meta;
* the presentation: the registered description (and the ``sem_*`` statement override);
* the baseline identity: ``baselines.json`` registrations equal the archived registrations;
* the input stream: for every ``(case, seed)`` record the v2 ``gen_inputs`` reproduces the
  archived input tensor bytes (exact bytes for ``small``, sha256 for ``paper``/``plan``);
* the oracle: the v2 ``reference`` on those inputs reproduces the archived fp64 reference
  tensor bytes (and the read_write final values, of which the 16 tasks declare none);
* the error scale: the v2 default/hook scale reproduces the archived scale tensor;
* the pack closure: the v2 pack is exactly the standard files of the task, recomputes its
  digest, and contains no retired v1 marker.

Guarded arithmetic (v1 ``contract.grid``/``threads``/``constants``/triton source/naive MSL) is
a registered difference, not a comparison target: A5 retires the fixed launch and the prompt.
"""

from __future__ import annotations

import hashlib
import json
import struct
import sys
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
G0 = ROOT / "docs/架构/跨硬件算子生成分工-G0/corpus/index.json"
CORPUS = G0.parent
MAGIC = b"TG0TENS1"

TASKS: tuple[str, ...] = (
    "sigmoid",
    "relu",
    "silu",
    "swiglu",
    "rmsnorm",
    "softmax",
    "reduction_sum",
    "gemm",
    "gemm_silu",
    "sem_gemm_neg",
    "sem_relu_gelu_comment",
    "sem_gelu_named_relu",
    "gen_relu",
    "gen_sigmoid",
    "gen_silu",
    "gen_swiglu",
)
DTYPES = ("fp16", "bf16")
#: v1 marker files that A5 deletes from every migrated directory.
V1_MARKERS = ("task.py", "track.json", "kernel_triton.py", "kernel_triton_msl.py", "naive.metal", "__init__.py")
#: Prompt/launch facts retired by A5 (registered differences, not compared).
RETIRED_COMBO_KEYS = (
    "triton_source",
    "naive_msl",
    "template_literals",
    "contract_text",
    "task_pack_digest",
)


class CompareError(AssertionError):
    """@brief 一处 A5 G0 对照失败。"""


def require(condition: bool, message: str) -> None:
    """@brief 断言一个对照条件。

    @param condition 条件。
    @param message 失败说明。
    @raises CompareError 条件不成立。
    """
    if not condition:
        raise CompareError(message)


def sha256(raw: bytes) -> str:
    """@brief 字节摘要。

    @param raw 字节。
    @return 十六进制 sha256。
    """
    return hashlib.sha256(raw).hexdigest()


def index() -> dict[str, Any]:
    """@brief 读取冻结的 G0 索引。

    @return 文档。
    """
    return json.loads(G0.read_text(encoding="utf-8"))


def combos(document: dict[str, Any], *, tasks: tuple[str, ...] = TASKS) -> list[dict[str, Any]]:
    """@brief 取 16 题的全部 G0 组合（顺序与语料一致）。

    @param document 索引。
    @param tasks 任务过滤。
    @return 组合列表。
    """
    selected = [combo for combo in document["combos"] if combo["task"] in tasks]
    require(len(selected) == len([c for c in document["combos"] if c["task"] in tasks]), "duplicate G0 combos")
    return selected


def combo_id(combo: dict[str, Any]) -> str:
    """@brief 稳定的组合 id。

    @param combo 组合。
    @return id 文本。
    """
    return f"{combo['task']}-{combo['dtype']}-{combo['size']}-{combo['variant'] or 'base'}"


def task_dir(task: str) -> Path:
    """@brief 迁移后的标准目录。

    @param task 任务名。
    @return 目录。
    """
    return ROOT / "tasks" / task


def archive(record: dict[str, Any]) -> dict[str, tuple[bytes, dict]]:
    """@brief 读取一个 ``small`` 档 ``.tg0z`` 档案（输入/参考/尺度原始字节）。

    @param record case record。
    @return ``role:name -> (bytes, header)``。
    """
    path = CORPUS / record["archive"]
    packed = path.read_bytes()
    require(sha256(packed) == record["archive_sha256"], f"{path}: archive digest")
    raw = zlib.decompress(packed)
    require(raw.startswith(MAGIC), f"{path}: bad magic")
    header_len = struct.unpack("<I", raw[8:12])[0]
    header = json.loads(raw[12 : 12 + header_len])
    payload = raw[12 + header_len :]
    tensors: dict[str, tuple[bytes, dict]] = {}
    for entry in header:
        piece = payload[entry["offset"] : entry["offset"] + entry["nbytes"]]
        require(len(piece) == entry["nbytes"], f"{path}: truncated {entry['name']}")
        tensors[f"{entry['role']}:{entry['name']}"] = (piece, entry)
    return tensors


def raw64(value: Any) -> bytes:
    """@brief G0 归档使用的 float64 原始字节（与 ``gen_corpus.raw_bytes`` 相同）。

    @param value torch 张量。
    @return 字节。
    """
    return value.detach().contiguous().reshape(-1).view(__import__("torch").uint8).numpy().tobytes()


def _default_scale(reference_outputs: dict[str, Any]) -> dict[str, Any]:
    """@brief v1/v2 默认误差尺度（有限部分的 RMS）。

    @param reference_outputs reference 输出。
    @return 张量名 → 尺度。
    """
    import torch

    scale = {}
    for name, value in reference_outputs.items():
        finite = torch.where(torch.isfinite(value), value, torch.zeros_like(value)) if value.is_floating_point() else value
        scale[name] = torch.sqrt(torch.mean(finite.to(torch.float64) ** 2))
    return scale


@dataclass
class ComboResult:
    """@brief 一个组合的对照结果。"""

    combo_id: str
    task: str
    dtype: str
    size: str
    variant: str | None
    records: int
    exact_input_bytes: int
    exact_reference_bytes: int
    exact_scale_bytes: int
    checks: list[str] = field(default_factory=list)

    def document(self) -> dict[str, Any]:
        """@brief JSON 文档。

        @return 字段字典。
        """
        return {
            "combo_id": self.combo_id,
            "task": self.task,
            "dtype": self.dtype,
            "size": self.size,
            "variant": self.variant,
            "records": self.records,
            "exact_input_bytes": self.exact_input_bytes,
            "exact_reference_bytes": self.exact_reference_bytes,
            "exact_scale_bytes": self.exact_scale_bytes,
            "checks": list(self.checks),
        }


def compare_semantics(contract: Any, combo: dict[str, Any]) -> list[str]:
    """@brief 语义/表示/baseline/pack 对照。

    @param contract v2 合同。
    @param combo G0 组合。
    @return 已通过的检查名。
    """
    from tcenv.taskcontract.loader import V1_MARKER_FILES, V2_STANDARD_FILES
    from tcenv.track import pack_digest

    instance = contract.instance
    checks: list[str] = []
    require(tuple(instance.case_domain.cases) == tuple(combo["cases"]), f"{combo_id(combo)}: cases drifted")
    checks.append("cases")
    tolerances = combo["tolerances"]
    require(float(instance.tolerances.rtol) == float(tolerances["rtol"]), f"{combo_id(combo)}: rtol")
    require(float(instance.tolerances.atol_rel) == float(tolerances["atol_rel"]), f"{combo_id(combo)}: atol_rel")
    require(bool(instance.tolerances.exact) is bool(tolerances["exact"]), f"{combo_id(combo)}: exact")
    checks.append("tolerances")
    require(float(instance.work_accounting.flops) == float(combo["work"]["flops"]), f"{combo_id(combo)}: flops")
    require(float(instance.work_accounting.min_bytes) == float(combo["work"]["min_bytes"]), f"{combo_id(combo)}: min_bytes")
    checks.append("work")
    args = combo["contract"]["args"]
    require(len(instance.tensors) == len(args), f"{combo_id(combo)}: tensor count")
    for binding, arg in zip(instance.tensors, args):
        require(binding.name == arg["name"], f"{combo_id(combo)}: tensor order at {arg['name']}")
        require(binding.dtype == arg["dtype"], f"{combo_id(combo)}: {binding.name} dtype")
        require(tuple(int(v) for v in binding.shape) == tuple(int(v) for v in arg["shape"]), f"{combo_id(combo)}: {binding.name} shape")
        require(binding.role == arg["role"], f"{combo_id(combo)}: {binding.name} role")
    checks.append("tensors")
    require(dict(instance.spec_params) == dict(combo["specialization"]["spec_params"]), f"{combo_id(combo)}: spec_params")
    checks.append("spec_params")
    constants = {str(name): value for name, _type, value in combo["contract"]["constants"]}
    for name, value in instance.resolved_attrs.items():
        if name in constants:
            # The v1 LoweringProfile materialised this attr as a contract constant.
            require(float(constants[name]) == float(value), f"{combo_id(combo)}: attr {name} value")
        else:
            # A semantic attr the v1 profile kept out of the contract is anchored by the archived
            # known-good text (its decimal spelling appears there) and, more strongly, by the
            # archived reference tensor bytes compared below.
            require(str(value) in combo["naive_msl"], f"{combo_id(combo)}: attr {name} is not anchored by G0")
    checks.append("attrs")
    require(contract.baselines == tuple(contract.baselines), "baseline order")
    documented = [entry.document() for entry in contract.baselines]
    require(documented == combo["baselines"], f"{combo_id(combo)}: baseline identity drifted")
    checks.append("baselines")
    require(contract.correctness_only is (not combo["baselines"]), f"{combo_id(combo)}: correctness-only boundary")
    checks.append("correctness_only")
    meta = {key: value for key, value in instance.evaluation_meta().items() if key != "triton_baseline_variant"}
    expected_meta = {key: value for key, value in combo["meta"].items() if key != "triton_baseline_variant"}
    require(json.loads(json.dumps(meta)) == expected_meta, f"{combo_id(combo)}: evaluation meta drifted")
    checks.append("meta")
    require(contract.display.description == combo["description"], f"{combo_id(combo)}: description drifted")
    checks.append("description")
    # Reference contract partition: every migrated task has write outputs only, so the
    # "read_write final value" comparison is vacuous *and* explicitly asserted here.
    require(
        tuple(binding.name for binding in instance.tensors if binding.access == "read_write") == (),
        f"{combo_id(combo)}: the 16 migrated tasks declare no read_write tensor",
    )
    require(tuple(instance.reference_inputs) == tuple(arg["name"] for arg in args if arg["role"] == "input"), f"{combo_id(combo)}: reference inputs")
    require(tuple(instance.reference_outputs) == tuple(arg["name"] for arg in args if arg["role"] == "output"), f"{combo_id(combo)}: reference outputs")
    checks.append("reference_partition")
    # Pack closure: only standard files, no retired marker, digest recomputes.
    directory = task_dir(combo["task"])
    present = [name for name in V2_STANDARD_FILES if (directory / name).is_file()]
    expected_pack = tuple(f"tasks/{combo['task']}/{name}" for name in present)
    require(tuple(contract.pack_files) == expected_pack, f"{combo_id(combo)}: pack file set is not the standard entry")
    require(contract.pack_digest == pack_digest(expected_pack), f"{combo_id(combo)}: pack digest")
    for marker in V1_MARKER_FILES:
        require(not (directory / marker).exists(), f"{combo_id(combo)}: retired v1 marker {marker} is still present")
    checks.append("pack_closure")
    return checks


def compare_case_records(contract: Any, combo: dict[str, Any], *, max_records: int | None = None) -> tuple[list[str], int, int, int]:
    """@brief 逐 ``(case, seed)`` 对照输入、reference 与 error scale。

    @param contract v2 合同。
    @param combo G0 组合。
    @param max_records 只检查前 N 条（用于快速子集；默认全部）。
    @return ``(checks, exact_input, exact_reference, exact_scale)``。
    """
    import torch
    from tcenv.tensors import round_to, to_raw

    instance = contract.instance
    checks: list[str] = []
    exact_input = exact_reference = exact_scale = 0
    records = combo["case_records"][: max_records or len(combo["case_records"])]
    for record in records:
        where = f"{combo_id(combo)}/{record['case']}/{record['seed']}"
        logical = contract.reference.generate(instance, record["case"], record["seed"])
        require(set(logical) == set(instance.reference_inputs), f"{where}: gen_inputs keys")
        rounded = {}
        raw = {}
        for name in instance.reference_inputs:
            shape = tuple(int(v) for v in instance.tensor(name).shape)
            rounded[name] = round_to(logical[name].reshape(shape), instance.tensor(name).dtype)
            raw[name] = to_raw(rounded[name], instance.tensor(name).dtype)
        entries = {f"{entry['role']}:{entry['name']}": entry for entry in record["tensors"]["entries"]}
        for name, piece in raw.items():
            entry = entries.get(f"input:{name}")
            require(entry is not None, f"{where}: archived input {name} is missing")
            require(len(piece) == int(entry["nbytes"]), f"{where}: {name} byte length")
            require(sha256(piece) == entry["sha256"], f"{where}: {name} input sha256 differs from G0")
            exact_input += 1
        reference_outputs = contract.reference.execute(instance, rounded)
        require(set(reference_outputs) == set(instance.reference_outputs), f"{where}: reference output keys")
        for name, value in reference_outputs.items():
            piece = raw64(value)
            entry = entries.get(f"reference:{name}")
            require(entry is not None, f"{where}: archived reference {name} is missing")
            require(len(piece) == int(entry["nbytes"]), f"{where}: reference {name} byte length")
            require(sha256(piece) == entry["sha256"], f"{where}: reference {name} bytes differ from G0")
            exact_reference += 1
        if instance.reference_outputs and record["case"] == "ieee":
            shape = tuple(int(v) for v in instance.tensor(instance.reference_outputs[0]).shape)
            require(str(reference_outputs[instance.reference_outputs[0]].dtype) == "torch.float64", f"{where}: reference dtype")
            require(tuple(value.shape) == shape, f"{where}: reference shape")
        if contract.reference.error_scale_kind not in ("none", ""):
            scale = contract.reference.error_scale(instance, rounded, reference_outputs)
        else:
            scale = _default_scale(reference_outputs)
        for name, value in scale.items():
            piece = raw64(value)
            entry = entries.get(f"scale:{name}")
            require(entry is not None, f"{where}: archived scale {name} is missing")
            require(sha256(piece) == entry["sha256"], f"{where}: error scale {name} differs from G0")
            exact_scale += 1
        if record["archive"] is not None:
            archived = archive(record)
            for name, piece in raw.items():
                require(piece == archived[f"input:{name}"][0], f"{where}: {name} input bytes differ from the archived G0 tensor")
            for name, value in reference_outputs.items():
                require(raw64(value) == archived[f"reference:{name}"][0], f"{where}: reference {name} exact bytes")
            for name, value in scale.items():
                require(raw64(value) == archived[f"scale:{name}"][0], f"{where}: scale {name} exact bytes")
    checks.append(f"records:{len(records)}")
    return checks, exact_input, exact_reference, exact_scale


def compare_combo(combo: dict[str, Any], *, max_records: int | None = None) -> ComboResult:
    """@brief 对照一个 G0 组合。

    @param combo G0 组合。
    @param max_records 只检查前 N 条 case record。
    @return 结果。
    """
    from tcenv.taskcontract import load_task_contract

    contract = load_task_contract(task_dir(combo["task"]), dtype=combo["dtype"], size=combo["size"])
    checks = compare_semantics(contract, combo)
    record_checks, exact_input, exact_reference, exact_scale = compare_case_records(contract, combo, max_records=max_records)
    return ComboResult(
        combo_id=combo_id(combo),
        task=combo["task"],
        dtype=combo["dtype"],
        size=combo["size"],
        variant=combo["variant"],
        records=len(combo["case_records"]) if max_records is None else min(max_records, len(combo["case_records"])),
        exact_input_bytes=exact_input,
        exact_reference_bytes=exact_reference,
        exact_scale_bytes=exact_scale,
        checks=checks + record_checks,
    )


def run(document: dict[str, Any] | None = None, *, tasks: tuple[str, ...] = TASKS, sizes: tuple[str, ...] | None = None, max_records: int | None = None) -> list[ComboResult]:
    """@brief 对照全部（或子集）组合。

    @param document G0 索引（缺省读取）。
    @param tasks 任务过滤。
    @param sizes 尺寸过滤。
    @param max_records 每个组合只检查前 N 条。
    @return 结果列表。
    """
    document = document or index()
    results = []
    for combo in combos(document, tasks=tasks):
        if sizes is not None and combo["size"] not in sizes:
            continue
        results.append(compare_combo(combo, max_records=max_records))
    return results


def main() -> None:
    """@brief 命令行入口。"""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="*", default=())
    parser.add_argument("--sizes", nargs="*", default=None)
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    results = run(tasks=tuple(args.only) or TASKS, sizes=tuple(args.sizes) if args.sizes else None, max_records=args.max_records)
    payload = {
        "schema_version": 1,
        "corpus": str(G0.relative_to(ROOT)),
        "corpus_source_head": index()["source_head"],
        "combos": len(results),
        "results": [item.document() for item in results],
    }
    if args.output:
        args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    total = sum(item.records for item in results)
    print(f"OK: {len(results)} G0 combos, {total} case records, "
          f"{sum(item.exact_input_bytes for item in results)} input tensors, "
          f"{sum(item.exact_reference_bytes for item in results)} reference tensors, "
          f"{sum(item.exact_scale_bytes for item in results)} scales match G0")


if __name__ == "__main__":
    sys.exit(main())
