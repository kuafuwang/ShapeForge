#!/usr/bin/env python3
"""B1: migrate gemv and rope to the v2 standard task directory, atomically per task.

The approach is A5's, reused rather than reinvented: the frozen G0 sampler block is copied
verbatim into each ``reference.py`` (a reference may not import the harness), the v1 OpSpec
document is mapped to ``spec_version 2`` by dropping ``uses`` and adding the discovery block,
and the baselines are the exact vendor calls the v1 track registered.  What B1 adds is the
frozen evidence: G0 has no corpus for these two tasks, so ``freeze_v1.py`` ran first and
``compare.py`` proves the v2 contract reproduces every frozen byte.

Written per task:  opspec.json (spec_version 2), reference.py, reference.json,
                   baselines.json + baselines.py, display.json
Deleted per task:  task.py, track.json, kernel_triton.py, kernel_triton_msl.py, naive.metal,
                   __init__.py, __pycache__
Preserved:         the naive candidate text, under evidence/candidates/ (it is the hand-written
                   candidate used by the B1 GPU slice, not a v1 assembly marker).

Run:  uv run python docs/架构/跨硬件算子生成分工-阶段验收/B1/migrate.py [--check]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
from pathlib import Path
from typing import Any

B1 = Path(__file__).resolve().parent
ROOT = B1.parents[3]
TASKS = ROOT / "tasks"
A5 = B1.parent / "A5"
V1_MARKERS = ("task.py", "track.json", "kernel_triton.py", "kernel_triton_msl.py", "naive.metal", "__init__.py")


def _a5() -> Any:
    """@brief 载入 A5 的迁移模块，只为复用冻结采样器文本与 OpSpec 映射规则。

    @return A5 migrate 模块。
    """
    spec = importlib.util.spec_from_file_location("_b1_a5_migrate", A5 / "migrate.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def reference_source(task: str, combo: dict[str, Any]) -> str:
    """@brief 生成一题的标准 ``reference.py``。

    @param task 任务名。
    @param combo 该题冻结证据中的任一组合（提供 v1 登记参数）。
    @return 源文本。
    """
    a5 = _a5()
    gen = combo["implementations"]["gen_inputs"]["params"]
    mags = json.dumps(gen["mags"], sort_keys=True)
    default_mag = float(gen["default_mag"])
    large = float(gen["large"])
    spike = float(gen.get("spike", 100.0))
    constants = [
        f"_MAGS = {mags}",
        f"_DEFAULT_MAG = {default_mag!r}",
        f"_LARGE = {large!r}",
        f"_SPIKE = {spike!r}",
    ]
    if task == "gemv":
        offsets = gen["seed_offsets"]
        for name, offset in offsets.items():
            constants.append(f"_SEED_OFFSET_{name} = {int(offset)}")
        body = [
            '    base = "normal" if case == "ieee" else case',
            '    mag = _MAGS.get(base, _DEFAULT_MAG)',
            '    a_shape = tuple(instance.tensor("A").shape)',
            '    x_shape = tuple(instance.tensor("x").shape)',
            '    A = _sample(base, a_shape, seed + _SEED_OFFSET_A, instance.dtype, mag=mag, large=_LARGE, spike=_SPIKE)',
            '    x = _sample(base, x_shape, seed + _SEED_OFFSET_x, instance.dtype, mag=mag, large=_LARGE, spike=_SPIKE)',
            '    if case == "ieee":',
            "        A = _inject(A, seed + _IEEE_SEED_STRIDE)",
            '    return {"A": A, "x": x}',
        ]
        reference_body = 'return {"y": inputs["A"].double() @ inputs["x"].double()}'
    elif task == "rope":
        body = [
            '    base = "normal" if case == "ieee" else case',
            '    shape = tuple(instance.tensor("x").shape)',
            '    x = _sample(base, shape, seed, instance.dtype, mag=_MAGS.get(base, _DEFAULT_MAG), large=_LARGE, spike=_SPIKE)',
            '    if case == "ieee":',
            "        x = _inject(x, seed + _IEEE_SEED_STRIDE)",
            '    return {"x": x}',
        ]
        reference_body = (
            'x = inputs["x"].to(torch.float64)\n'
            '    base = float(attrs["BASE"])\n'
            '    half = x.shape[-1] // 2\n'
            '    seq = x.shape[-2]\n'
            '    inv_freq = base ** (-torch.arange(half, dtype=torch.float64) / half)\n'
            '    ang = torch.arange(seq, dtype=torch.float64)[:, None] * inv_freq[None, :]\n'
            '    c, s = torch.cos(ang), torch.sin(ang)\n'
            '    x1, x2 = x[..., :half], x[..., half:]\n'
            '    return {"y": torch.cat([x1 * c - x2 * s, x1 * s + x2 * c], dim=-1)}'
        )
    else:
        raise SystemExit(f"{task}: B1 does not migrate this task")

    text = [
        f'"""{task}: independent fp64 reference and the frozen v1 input stream.',
        "",
        "The module is the whole oracle of the task.  It may not import the harness (``tasks``/",
        "``tcenv``) or any candidate, so the v1 sampler and the harness IEEE injection are carried",
        "as the frozen block below (byte-identical to the A5-migrated references).  The case/seed",
        "stream, the magnitudes and the fp64 math are the ones the v1 track registered; B1's",
        "compare.py proves the byte-level equivalence against evidence/v1-frozen.json.",
        '"""',
        "",
        "import torch",
        a5.SAMPLER_BLOCK,
        "",
        "\n".join(constants),
        "",
        "",
        "def gen_inputs(case, seed, instance):",
        '    """The v1 case/seed stream (the ``ieee`` case is the normal stream plus specials)."""',
        *body,
        "",
        "",
        "def reference(inputs, attrs):",
        '    """The independent fp64 oracle (one rounding at the output)."""',
        f"    {reference_body}",
        "",
    ]
    return "\n".join(text)


def baselines_source(task: str) -> str:
    """@brief 生成一题的 ``baselines.py``（与 v1 登记的供应商调用逐字对应）。

    @param task 任务名。
    @return 源文本。
    """
    header = [
        f'"""Performance baselines for {task}: the vendor/library calls the v1 track registered.',
        "",
        "A baseline is a timing comparison object, never the correctness oracle.  Each function",
        "receives the framework-native arrays BaselineRunner prepared (mlx array or torch MPS",
        "tensor) and returns the framework-native result.  The call is the one the v1 track",
        "registered, so a v1/v2 timing comparison measures the same comparison object.",
        '"""',
        "",
        "from __future__ import annotations",
        "",
    ]
    if task == "gemv":
        body = [
            "def mlx(a, x):",
            '    """v1 baseline mlx (``matmul``)."""',
            "    return a @ x",
            "",
            "",
            "def torch_mps(a, x):",
            '    """v1 baseline torch_mps (``torch.mv``)."""',
            "    import torch",
            "    return torch.mv(a, x)",
            "",
            'BASELINES = {"mlx": mlx, "torch_mps": torch_mps}',
            "",
        ]
    elif task == "rope":
        body = [
            "_BASE = 10000.0  # the frozen OpSpec attr default (BASE has no per-instance override)",
            "",
            "",
            "def mlx(x):",
            '    """v1 baseline mlx (``mx.fast.rope`` with the registered traditional/scale/offset)."""',
            "    import mlx.core as mx",
            "    return mx.fast.rope(x, x.shape[-1], traditional=False, base=_BASE, scale=1.0, offset=0)",
            "",
            'BASELINES = {"mlx": mlx}',
            "",
        ]
    else:
        raise SystemExit(f"{task}: B1 does not migrate this task")
    return "\n".join(header + body)


def migrate_task(task: str, combo: dict[str, Any], *, check: bool) -> list[str]:
    """@brief 迁移一个任务目录：写 v2 标准入口，删 v1 装配标记。

    @param task 任务名。
    @param combo 冻结证据中的组合（提供 v1 文档与登记参数）。
    @param check 只检查不写。
    @return 动作日志。
    """
    a5 = _a5()
    directory = TASKS / task
    if (directory / "opspec.json").exists():
        document = json.loads((directory / "opspec.json").read_text(encoding="utf-8"))
        if int(document.get("spec_version", 0)) == 2:
            return [f"{task}: already v2, nothing to do"]
    v2_document = a5.v2_opspec(combo["opspec"], task)
    files = {
        "opspec.json": json.dumps(v2_document, ensure_ascii=False, indent=2) + "\n",
        "reference.py": reference_source(task, combo),
        "reference.json": json.dumps({"schema_version": 1, "reference_version": 1, "error_scale": {"kind": "none"}}, indent=2) + "\n",
        "baselines.json": json.dumps({"schema_version": 1, "baselines": combo["baselines"]}, ensure_ascii=False, indent=2) + "\n",
        "baselines.py": baselines_source(task),
        "display.json": json.dumps({"schema_version": 1, "description": combo["description"]}, ensure_ascii=False, indent=2) + "\n",
    }
    actions: list[str] = []
    if check:
        for name in files:
            actions.append(f"would write {directory / name}")
        for name in V1_MARKERS:
            if (directory / name).exists():
                actions.append(f"would delete {directory / name}")
        return actions
    candidates = B1 / "evidence" / "candidates"
    candidates.mkdir(parents=True, exist_ok=True)
    naive = directory / "naive.metal"
    if naive.is_file():
        target = candidates / f"{task}-naive.msl"
        target.write_text(naive.read_text(encoding="utf-8"), encoding="utf-8")
        actions.append(f"preserved candidate {target.relative_to(ROOT)}")
    for name, text in files.items():
        (directory / name).write_text(text, encoding="utf-8")
        actions.append(f"wrote {directory / name}")
    for name in V1_MARKERS:
        path = directory / name
        if path.is_file():
            path.unlink()
            actions.append(f"deleted {path}")
    cache = directory / "__pycache__"
    if cache.is_dir():
        shutil.rmtree(cache)
    return actions


def main() -> None:
    """@brief 迁移 B1 的两题，并以 v2 合同装配结果作为立即冒烟检查。

    @return 无。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="只列出将要发生的写入与删除，不改文件")
    args = parser.parse_args()
    frozen = json.loads((B1 / "evidence" / "v1-frozen.json").read_text(encoding="utf-8"))
    combos = {}
    for combo in frozen["combos"]:
        combos.setdefault(combo["task"], combo)
    for task, combo in combos.items():
        for line in migrate_task(task, combo, check=args.check):
            print(line)
    if not args.check:
        from tcenv.taskcontract import discover_standard_tasks, load_task_contract

        for task in combos:
            contract = load_task_contract(TASKS / task, dtype="fp16", size="small")
            print(f"loaded v2 contract {contract.op_name}: dtype={contract.dtype} size={contract.size}")
        names = [locator.task_id for locator in discover_standard_tasks()]
        print(f"v2 discovery now finds {len(names)} tasks: {names}")


if __name__ == "__main__":
    main()
