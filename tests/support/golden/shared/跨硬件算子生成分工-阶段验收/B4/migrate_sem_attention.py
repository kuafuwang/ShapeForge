#!/usr/bin/env python3
"""B4 (part 1): migrate sem_attention_odd_scale to the v2 standard entry.

It shares the FlashAttention representation (q/k/v/o, fp32 online softmax) but not its scale:
the semantic-recovery task forms the qk scale as a **sum** ``SM_SCALE + log2(e)`` instead of the
usual product.  The v1 ``effective_scale`` helper evaluates that to ``(SM_SCALE + LOG2E) * ln(2)``
as the *natural-log* scale handed to the framework attention, and both baselines use the same
mode.  Everything else (seed offsets, ieee-injects-q-only, two baselines) matches flash_attention.

Run:  uv run python docs/架构/跨硬件算子生成分工-阶段验收/B4/migrate_sem_attention.py [--check]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
from pathlib import Path
from typing import Any

B4 = Path(__file__).resolve().parent
ROOT = B4.parents[3]
TASKS = ROOT / "tasks"
TASK = "sem_attention_odd_scale"
V1_MARKERS = ("task.py", "track.json", "kernel_triton.py", "kernel_triton_msl.py", "naive.metal", "__init__.py")

LOG2E = 1.4426950408889634
LN2 = 0.6931471805599453


def _a5() -> Any:
    """@brief 载入 A5 迁移模块，复用冻结采样器文本与 OpSpec 映射规则。

    @return A5 migrate 模块。
    """
    spec = importlib.util.spec_from_file_location("_b4_a5_migrate", B4.parent / "A5" / "migrate.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def reference_source(combo: dict[str, Any]) -> str:
    """@brief 生成 sem_attention_odd_scale 的 reference.py（sum_with_log2e 标度）。

    @param combo 冻结组合。
    @return 源文本。
    """
    gen = combo["implementations"]["gen_inputs"]["params"]
    constants = [
        f"_MAGS = {json.dumps(gen['mags'], sort_keys=True)}",
        f"_DEFAULT_MAG = {float(gen['default_mag'])!r}",
        f"_LARGE = {float(gen['large'])!r}",
        f"_SPIKE = {float(gen['spike'])!r}",
        f"_LOG2E = {LOG2E!r}",
        f"_LN2 = {LN2!r}",
    ] + [f"_SEED_OFFSET_{name} = {int(offset)}" for name, offset in gen["seed_offsets"].items()]
    body = [
        '    base = "normal" if case == "ieee" else case',
        '    mag = _MAGS.get(base, _DEFAULT_MAG)',
        '    shape = tuple(instance.tensor("q").shape)',
        '    out = {}',
        '    for name in ("q", "k", "v"):',
        "        offset = {'q': _SEED_OFFSET_q, 'k': _SEED_OFFSET_k, 'v': _SEED_OFFSET_v}[name]",
        "        out[name] = _sample(base, shape, seed + offset, instance.dtype, mag=mag, large=_LARGE, spike=_SPIKE)",
        '    if case == "ieee":',
        "        out[\"q\"] = _inject(out[\"q\"], seed + _IEEE_SEED_STRIDE)",
        "    return out",
    ]
    reference = '''    sm_scale = float(attrs["SM_SCALE"])
    # The semantic-recovery scale mode is a SUM, not the usual product:
    # exp2(qk * (SM_SCALE + log2 e)) == exp(qk * (SM_SCALE + log2 e) * ln 2), so the framework's
    # natural-log scale is (SM_SCALE + LOG2E) * ln(2) — exactly v1's effective_scale helper.
    scale = (sm_scale + _LOG2E) * _LN2
    o = F.scaled_dot_product_attention(inputs["q"].float(), inputs["k"].float(), inputs["v"].float(), scale=scale)
    return {"o": o.double()}'''
    return "\n".join(
        [
            f'"""{TASK}: independent fp64 reference and the frozen v1 input stream.',
            "",
            "The module is the whole oracle of the task.  It may not import the harness (``tasks``/",
            "``tcenv``) or any candidate, so the v1 sampler and the harness IEEE injection are carried",
            "as the frozen block below (byte-identical to the A5-migrated references).",
            "The scale mode is ``sum_with_log2e``, not the usual product.",
            '"""',
            "",
            "import torch",
            "import torch.nn.functional as F",
            _a5().SAMPLER_BLOCK,
            "",
            "\n".join(constants),
            "",
            "",
            "def gen_inputs(case, seed, instance):",
            '    """The v1 case/seed stream (q/k/v at seed + 0/31/62; ieee injects q only)."""',
            *body,
            "",
            "",
            "def reference(inputs, attrs):",
            '    """F.scaled_dot_product_attention with the sum_with_log2e scale, fp64 output."""',
            reference,
            "",
        ]
    )


def baselines_source() -> str:
    """@brief 生成 baselines.py（两条基线都用 sum_with_log2e 标度）。

    @return 源文本。
    """
    return f'''"""Performance baselines for {TASK}: the vendor attention calls with the SUM scale mode.

A baseline is a timing comparison object, never the correctness oracle.  The scale is the same
sum_with_log2e form the v1 track registered for both baselines.
"""

from __future__ import annotations

_LOG2E = {LOG2E!r}
_LN2 = {LN2!r}
_SM_SCALE = 0.125  # the frozen OpSpec attr default (1/sqrt(D) with D = 64)
_SCALE = (_SM_SCALE + _LOG2E) * _LN2


def mlx(q, k, v):
    """v1 baseline mlx (``mx.fast.scaled_dot_product_attention``, sum_with_log2e)."""
    import mlx.core as mx
    return mx.fast.scaled_dot_product_attention(q, k, v, scale=_SCALE)


def torch_mps(q, k, v):
    """v1 baseline torch_mps (``F.scaled_dot_product_attention``, sum_with_log2e)."""
    import torch.nn.functional as F
    return F.scaled_dot_product_attention(q, k, v, scale=_SCALE)


BASELINES = {{"mlx": mlx, "torch_mps": torch_mps}}
'''


def main() -> None:
    """@brief 迁移 sem_attention_odd_scale 并做 v2 装配冒烟。

    @return 无。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    frozen = json.loads((B4 / "evidence" / "v1-frozen.json").read_text(encoding="utf-8"))
    combo = next(combo for combo in frozen["combos"] if combo["task"] == TASK)
    directory = TASKS / TASK
    v2_document = _a5().v2_opspec(combo["opspec"], TASK)
    files = {
        "opspec.json": json.dumps(v2_document, ensure_ascii=False, indent=2) + "\n",
        "reference.py": reference_source(combo),
        "reference.json": json.dumps({"schema_version": 1, "reference_version": 1, "error_scale": {"kind": "none"}}, indent=2) + "\n",
        "baselines.json": json.dumps({"schema_version": 1, "baselines": combo["baselines"]}, ensure_ascii=False, indent=2) + "\n",
        "baselines.py": baselines_source(),
        "display.json": json.dumps({"schema_version": 1, "description": combo["description"]}, ensure_ascii=False, indent=2) + "\n",
    }
    if args.check:
        for name in files:
            print(f"would write {directory / name}")
        for name in V1_MARKERS:
            if (directory / name).exists():
                print(f"would delete {directory / name}")
        return
    candidates = B4 / "evidence" / "candidates"
    candidates.mkdir(parents=True, exist_ok=True)
    naive = directory / "naive.metal"
    if naive.is_file():
        (candidates / f"{TASK}-naive.msl").write_text(naive.read_text(encoding="utf-8"), encoding="utf-8")
    for name, text in files.items():
        (directory / name).write_text(text, encoding="utf-8")
        print(f"wrote {directory / name}")
    for name in V1_MARKERS:
        path = directory / name
        if path.is_file():
            path.unlink()
            print(f"deleted {path}")
    cache = directory / "__pycache__"
    if cache.is_dir():
        shutil.rmtree(cache)
    from tcenv.taskcontract import load_task_contract

    contract = load_task_contract(directory, dtype="fp16", size="small")
    print(f"loaded v2 contract {contract.op_name}: dtype={contract.dtype} size={contract.size}")


if __name__ == "__main__":
    main()
