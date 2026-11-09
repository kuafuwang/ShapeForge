#!/usr/bin/env python3
"""B2: freeze the v1 evidence for conv2d and lion *before* their v1 assembly is deleted.

There is no G0 corpus for these two tasks (they were ``preserved_v1_out_of_scope``), so the
equivalence oracle for B1 is generated here, once, from the live v1 path:

    tasks.get_task(task, dtype, size) + tcenv.backends.metal.testing.prepare_case

For every ``(task, dtype, size)`` combination and every declared case we record the same 18
``(case, seed)`` records G0 used (``seed = 7919 * (case_index + 1) + {0, 7, 1234}``): the raw
input bytes' sha256, the fp64 reference bytes' sha256 and the error-scale bytes' sha256, plus
the semantic document, baseline identity and the naive candidate text.  ``compare.py`` later
replays the v2 contract against these hashes; the hashes are the frozen evidence, so the v1
code can be deleted without losing the ability to prove equivalence.

Run:  uv run python docs/架构/跨硬件算子生成分工-阶段验收/B2/freeze_v1.py
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

B2 = Path(__file__).resolve().parent
ROOT = B2.parents[3]
EVIDENCE = B2 / "evidence"
TASKS = ("conv2d", "lion")
DTYPES = ("fp16", "bf16")
SIZES = ("paper", "small")
SEED_OFFSETS = (0, 7, 1234)
SEED_STRIDE = 7919


def sha256(raw: bytes) -> str:
    """@brief 字节串的 sha256。

    @param raw 原始字节。
    @return 十六进制摘要。
    """
    return hashlib.sha256(raw).hexdigest()


def raw64(value: Any) -> bytes:
    """@brief 张量的 float64 连续字节（与 A5 ``g0_compare.raw64`` 同规则）。

    @param value 张量。
    @return 小端 float64 字节。
    """
    return value.detach().to("cpu").double().contiguous().numpy().tobytes()


def describe(tensor_bytes: bytes, tensor: Any) -> dict[str, Any]:
    """@brief 记录一段字节的身份：长度、摘要、dtype 与形状。

    @param tensor_bytes 原始字节。
    @param tensor 来源张量。
    @return 可 JSON 化的描述。
    """
    return {
        "sha256": sha256(tensor_bytes),
        "nbytes": len(tensor_bytes),
        "dtype": str(tensor.dtype),
        "shape": list(tensor.shape),
    }


def freeze_combo(task_name: str, dtype: str, size: str) -> dict[str, Any]:
    """@brief 用 v1 装配冻结一个组合的全部算例。

    @param task_name 任务名。
    @param dtype 元素类型。
    @param size 尺寸档位。
    @return 组合证据。
    """
    from tasks import get_task
    from tcenv.backends.metal.testing import prepare_case
    from tcenv.tensors import to_raw

    task = get_task(task_name, dtype, size)
    instance = task.instance if hasattr(task, "instance") else None
    opspec = json.loads((ROOT / "tasks" / task_name / "opspec.json").read_text(encoding="utf-8"))
    track = json.loads((ROOT / "tasks" / task_name / "track.json").read_text(encoding="utf-8"))
    cases = list(opspec["cases"]["names"])
    records: list[dict[str, Any]] = []
    for index, case in enumerate(cases):
        for offset in SEED_OFFSETS:
            seed = SEED_STRIDE * (index + 1) + offset
            data = prepare_case(task, case, seed)
            records.append(
                {
                    "case": case,
                    "seed": seed,
                    "inputs": {
                        name: describe(to_raw(tensor, task.contract.arg(name).dtype), tensor)
                        for name, tensor in data.rounded.items()
                    },
                    "reference": {name: describe(raw64(tensor), tensor) for name, tensor in data.ref.items()},
                    "scale": {name: describe(raw64(tensor), tensor) for name, tensor in data.scale.items()},
                }
            )
    return {
        "task": task_name,
        "dtype": dtype,
        "size": size,
        "cases": cases,
        "opspec": opspec,
        "description": track.get("description", ""),
        "baselines": track.get("baselines", []),
        "implementations": track.get("implementations", {}),
        "lowering_profile": track.get("lowering_profile"),
        "naive_msl": (ROOT / "tasks" / task_name / "naive.metal").read_text(encoding="utf-8"),
        "work": opspec.get("work_accounting"),
        "tolerances": opspec.get("numerics", {}).get("tolerances"),
        "evaluation_domain": json.loads(json.dumps(task.evaluation_domain, default=lambda o: dict(o))),
        "records": records,
    }


def main() -> None:
    """@brief 冻结 B1 两题的全部组合并写证据文件。

    @return 无。
    """
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    combos = []
    for task_name in TASKS:
        for dtype in DTYPES:
            for size in SIZES:
                combo = freeze_combo(task_name, dtype, size)
                combos.append(combo)
                print(f"frozen {task_name}/{dtype}/{size}: {len(combo['records'])} records", flush=True)
    document = {
        "schema_version": 1,
        "phase": "B2",
        "generator": "v1 tasks.get_task + tcenv.backends.metal.testing.prepare_case",
        "seed_formula": "7919 * (case_index + 1) + {0, 7, 1234}",
        "note": "B2 的等价性 oracle：G0 语料不含这两题，故在删除 v1 装配前现场冻结。",
        "combos": combos,
    }
    out = EVIDENCE / "v1-frozen.json"
    out.write_text(json.dumps(document, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"wrote {out} ({out.stat().st_size} bytes, {len(combos)} combos)")


if __name__ == "__main__":
    main()
