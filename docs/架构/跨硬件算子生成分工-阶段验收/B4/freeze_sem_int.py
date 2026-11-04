#!/usr/bin/env python3
"""B4 (part 2): freeze the v1 evidence for the sem_int dynamic family before it is retired.

The dynamic family is not a static task directory: ``sem_int_<seed>`` is materialised by the
registered generator (``tasks.sem_int.document.opspec_document``).  To keep the migration honest,
this freezes, for every seed the test suite and the experiment arms actually use, the real v1
``prepare_case`` output: inputs, the int64-then-wrap-int32 reference, and the RMS error scale.

* 6 seeds (sem_int_5, sem_int_3, sem_int_007, sem_int_1, sem_int_2, sem_int_4)
* 2 requested dtypes (fp16/bf16, both aliased to i32) x 2 sizes (paper/small)
* 3 declared cases (random/small/edge) x 3 seeds = 9 records per combination, 216 in total.

Run:  uv run python docs/架构/跨硬件算子生成分工-阶段验收/B4/freeze_sem_int.py
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

B4 = Path(__file__).resolve().parent
ROOT = B4.parents[3]
EVIDENCE = B4 / "evidence"
SEEDS = ("sem_int_5", "sem_int_3", "sem_int_007", "sem_int_1", "sem_int_2", "sem_int_4")
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
    """@brief 张量的 float64 连续字节（与 B1–B3 同规则）。

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


def main() -> None:
    """@brief 冻结 sem_int 全部组合并写证据文件。

    @return 无。
    """
    from tasks import get_task
    from tasks.sem_int import document as opspec_document
    from tcenv.backends.metal.testing import prepare_case
    from tcenv.tensors import to_raw

    EVIDENCE.mkdir(parents=True, exist_ok=True)
    track = json.loads((ROOT / "tasks" / "sem_int" / "track.json").read_text(encoding="utf-8"))
    combos = []
    for seed in SEEDS:
        seed_int = int(seed.rsplit("_", 1)[1])
        for dtype in DTYPES:
            for size in SIZES:
                task = get_task(seed, dtype, size)
                opspec = opspec_document.opspec_document(seed_int, size)
                cases = list(opspec["cases"]["names"])
                records = []
                for index, case in enumerate(cases):
                    for offset in SEED_OFFSETS:
                        case_seed = SEED_STRIDE * (index + 1) + offset
                        data = prepare_case(task, case, case_seed)
                        records.append(
                            {
                                "case": case,
                                "seed": case_seed,
                                "inputs": {
                                    name: describe(to_raw(tensor, task.contract.arg(name).dtype), tensor)
                                    for name, tensor in data.rounded.items()
                                },
                                "reference": {name: describe(raw64(tensor), tensor) for name, tensor in data.ref.items()},
                                "scale": {name: describe(raw64(tensor), tensor) for name, tensor in data.scale.items()},
                            }
                        )
                combos.append(
                    {
                        "task": seed,
                        "dtype": dtype,
                        "size": size,
                        "cases": cases,
                        "opspec": opspec,
                        "description": track.get("description", ""),
                        "baselines": track.get("baselines", []),
                        "implementations": track.get("implementations", {}),
                        "naive_msl": task.naive_msl,
                        "records": records,
                    }
                )
                print(f"frozen {seed}/{dtype}/{size}: {len(records)} records", flush=True)
    document = {
        "schema_version": 1,
        "phase": "B4",
        "generator": "v1 tasks.get_task + tcenv.backends.metal.testing.prepare_case",
        "seed_formula": "7919 * (case_index + 1) + {0, 7, 1234}",
        "combos": combos,
    }
    out = EVIDENCE / "sem_int-frozen.json"
    out.write_text(json.dumps(document, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"wrote {out} ({out.stat().st_size} bytes, {len(combos)} combos)")


if __name__ == "__main__":
    main()
