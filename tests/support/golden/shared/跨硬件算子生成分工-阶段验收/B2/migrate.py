#!/usr/bin/env python3
"""B2: migrate conv2d and lion to the v2 standard task directory.

Same rule as B1: the frozen G0 sampler block is copied verbatim into each reference.py, the
v1 OpSpec is mapped to spec_version 2, and the v1 assembly + LoweringProfile are retired in the
same change.  ``freeze_v1.py`` ran first; ``compare.py`` proves the byte-level equivalence.

B2 adds three things B1 did not have:

* **conv2d** declares a *sized* IEEE injection cap (``cases.inputs.ieee._max``), which the v1
  instance evaluates from the shape symbols (``max(8, N*OH*OW // 100)`` for x, ``4`` for w).
  The v2 ``gen_inputs`` recomputes it from ``instance.symbols`` instead of freezing a number,
  so paper and small stay correct.
* **conv2d**'s oracle has two paths: finite inputs go through ``F.conv2d``; non-finite inputs
  are summed tap by tap, because a Winograd transform would smear one Inf into NaN tiles.
* **lion** has a *module* ``error_scale`` (the sign-discontinuity allowance) and three
  baselines, including the ``mx.compile`` / ``torch.compile`` fused variants.  The v2 hook
  receives no case argument, so it applies the finite-value filtering itself — that is what
  the v1 evaluator did on the ``ieee`` case before calling the hook, and it is a no-op on the
  finite cases.

Run:  uv run python docs/架构/跨硬件算子生成分工-阶段验收/B2/migrate.py [--check]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
from pathlib import Path
from typing import Any

B2 = Path(__file__).resolve().parent
ROOT = B2.parents[3]
TASKS = ROOT / "tasks"
V1_MARKERS = ("task.py", "track.json", "kernel_triton.py", "kernel_triton_msl.py", "naive.metal", "__init__.py")


def _a5() -> Any:
    """@brief 载入 A5 迁移模块，复用冻结采样器文本与 OpSpec 映射规则。

    @return A5 migrate 模块。
    """
    spec = importlib.util.spec_from_file_location("_b2_a5_migrate", B2.parent / "A5" / "migrate.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _header(task: str, extra: str) -> list[str]:
    """@brief reference.py 的头部与采样器块。

    @param task 任务名。
    @param extra 说明该题特有的语义注意点。
    @return 文本行列表。
    """
    return [
        f'"""{task}: independent fp64 reference and the frozen v1 input stream.',
        "",
        "The module is the whole oracle of the task.  It may not import the harness (``tasks``/",
        "``tcenv``) or any candidate, so the v1 sampler and the harness IEEE injection are carried",
        "as the frozen block below (byte-identical to the A5-migrated references).",
        extra,
        '"""',
        "",
        "import torch",
        "import torch.nn.functional as F",
        _a5().SAMPLER_BLOCK,
        "",
    ]


def _conv2d(combo: dict[str, Any]) -> str:
    """@brief 生成 conv2d 的 reference.py。

    @param combo 冻结组合（提供 v1 登记参数）。
    @return 源文本。
    """
    gen = combo["implementations"]["gen_inputs"]["params"]
    constants = [
        f"_MAGS = {json.dumps(gen['mags'], sort_keys=True)}",
        f"_DEFAULT_MAG = {float(gen['default_mag'])!r}",
        f"_LARGE = {float(gen['large'])!r}",
        f"_SPIKE = {float(gen['spike'])!r}",
        f"_SEED_OFFSET_x = {int(gen['seed_offsets']['x'])}",
        f"_SEED_OFFSET_w = {int(gen['seed_offsets']['w'])}",
    ]
    body = [
        '    base = "normal" if case == "ieee" else case',
        '    mag = _MAGS.get(base, _DEFAULT_MAG)',
        '    x = _sample(base, tuple(instance.tensor("x").shape), seed + _SEED_OFFSET_x, instance.dtype, mag=mag, large=_LARGE, spike=_SPIKE)',
        '    w = _sample(base, tuple(instance.tensor("w").shape), seed + _SEED_OFFSET_w, instance.dtype, mag=mag, large=_LARGE, spike=_SPIKE)',
        '    if case == "ieee":',
        "        # v1 opspec cases.inputs.ieee._max: x -> max(8, N*OH*OW // 100), w -> 4.",
        "        # It is a *sized* cap, so it is recomputed from the instance symbols rather than frozen.",
        '        symbols = instance.symbols',
        '        max_x = max(8, (int(symbols["N"]) * int(symbols["OH"]) * int(symbols["OW"])) // 100)',
        "        x = _inject(x, seed + _IEEE_SEED_STRIDE, max_x)",
        "        w = _inject(w, seed + 2 * _IEEE_SEED_STRIDE, 4)",
        '    return {"x": x, "w": w}',
    ]
    reference = '''    x, w = inputs["x"], inputs["w"]
    if not (torch.isfinite(x).all() and torch.isfinite(w).all()):
        # CPU conv2d may use a Winograd transform, which smears one Inf over a whole tile as NaN;
        # IEEE semantics are those of the sum of products, so take the same fp32 sum tap by tap.
        x32, w32 = x.float(), w.float()
        n = x32.shape[0]
        oh = x32.shape[1] - w32.shape[1] + 1
        ow = x32.shape[2] - w32.shape[2] + 1
        oc = w32.shape[0]
        y = torch.zeros((n, oh, ow, oc), dtype=torch.float32)
        for i in range(w32.shape[1]):
            for j in range(w32.shape[2]):
                y += x32[:, i:i + oh, j:j + ow, :] @ w32[:, i, j, :].T
        return {"y": y.double()}
    x4 = x.permute(0, 3, 1, 2).contiguous().float()
    w4 = w.permute(0, 3, 1, 2).contiguous().float()
    return {"y": F.conv2d(x4, w4).permute(0, 2, 3, 1).contiguous().double()}'''
    return "\n".join(
        _header("conv2d", "IEEE 注入上限随 shape 变化（见 gen_inputs）；参考实现在非有限输入时逐 tap 求和。")
        + ["\n".join(constants), "", "", "def gen_inputs(case, seed, instance):",
           '    """The v1 case/seed stream, including the sized IEEE caps."""']
        + body
        + ["", "", "def reference(inputs, attrs):", '    """Finite: F.conv2d.  Non-finite: the tap-by-tap fp32 sum."""', reference, ""]
    )


def _lion(combo: dict[str, Any]) -> str:
    """@brief 生成 lion 的 reference.py（含模块级 error_scale）。

    @param combo 冻结组合。
    @return 源文本。
    """
    gen = combo["implementations"]["gen_inputs"]["params"]
    scale = combo["implementations"]["error_scale"]["params"]
    constants = [
        f"_MAG = {float(gen['mag'])!r}",
        f"_LARGE = {float(gen['large'])!r}",
        f"_SEED_OFFSET = {int(gen['seed_offset'])}",
        f"_TINY_NON_FP16_SCALE = {float(gen['tiny_non_fp16_scale'])!r}",
        f"_DISCONTINUITY_REL = {float(scale['discontinuity_rel'])!r}",
        f"_ATOL_STEPS = {float(scale['atol_steps'])!r}",
    ]
    body = [
        '    base = "normal" if case == "ieee" else case',
        '    shape = tuple(instance.tensor("p").shape)',
        "    out = {}",
        '    for index, name in enumerate(("p", "g", "m")):',
        "        out[name] = _sample(base, shape, seed + _SEED_OFFSET * index, instance.dtype, mag=_MAG, large=_LARGE)",
        '    if case == "tiny" and instance.dtype != "fp16":',
        "        # fp32 subnormals are flushed by the GPU: keep (1-beta)*x in the normal range, otherwise",
        "        # sign(beta1*m + (1-beta1)*g) legitimately flips for every implementation.",
        "        out = {key: value * _TINY_NON_FP16_SCALE for key, value in out.items()}",
        '    if case == "ieee":',
        '        for index, name in enumerate(("p", "g", "m")):',
        "            out[name] = _inject(out[name], seed + _IEEE_SEED_STRIDE * (index + 1))",
        "    return out",
    ]
    reference = '''    lr = float(attrs["LR"])
    wd = float(attrs["WD"])
    beta1 = float(attrs["BETA1"])
    beta2 = float(attrs["BETA2"])
    p, g, m = (inputs[k].to(torch.float64) for k in ("p", "g", "m"))
    u = torch.sign(beta1 * m + (1 - beta1) * g)
    return {"p_out": p * (1 - lr * wd) - lr * u, "m_out": beta2 * m + (1 - beta2) * g}'''
    error_scale = '''    """The v1 sign-discontinuity allowance.

    The v1 evaluator filtered non-finite values on the ``ieee`` case *before* calling this hook;
    the v2 hook receives no case argument, so it filters here instead — a no-op on finite cases.
    """
    attrs = instance.resolved_attrs
    lr = float(attrs["LR"])
    beta1 = float(attrs["BETA1"])
    beta2 = float(attrs["BETA2"])
    atol_rel = float(tolerances.atol_rel)

    def finite(t):
        return torch.where(torch.isfinite(t), t, torch.zeros_like(t)) if t.is_floating_point() else t

    p, g, m = (finite(inputs[k]).to(torch.float64) for k in ("p", "g", "m"))
    ref = {key: finite(value) for key, value in reference_outputs.items()}
    a, b = beta1 * m, (1 - beta1) * g
    amb = (a + b).abs() <= _DISCONTINUITY_REL * (a.abs() + b.abs())
    base = torch.sqrt(torch.mean(ref["p_out"] ** 2)).expand_as(ref["p_out"]).clone()
    base[amb] = _ATOL_STEPS * lr / atol_rel
    return {"p_out": base, "m_out": torch.sqrt(torch.mean(ref["m_out"] ** 2))}'''
    return "\n".join(
        _header("lion", "tiny 的非 fp16 缩放与 sign 间断点的模块级 error scale 都按 v1 登记保留。")
        + ["\n".join(constants), "", "", "def gen_inputs(case, seed, instance):",
           '    """The v1 case/seed stream (p/g/m at seed + 7*index; tiny scaled off fp16)."""']
        + body
        + ["", "", "def reference(inputs, attrs):", '    """The independent fp64 Lion step."""', reference,
           "", "", "def error_scale(inputs, reference_outputs, instance, tolerances):", error_scale, ""]
    )


def _baselines(task: str) -> str:
    """@brief 生成一题的 baselines.py（与 v1 登记的供应商调用一致）。

    @param task 任务名。
    @return 源文本。
    """
    header = [
        f'"""Performance baselines for {task}: the vendor/library calls the v1 track registered.',
        "",
        "A baseline is a timing comparison object, never the correctness oracle.  Functions receive",
        "the framework-native arrays BaselineRunner prepared and return the framework-native result.",
        '"""',
        "",
        "from __future__ import annotations",
        "",
    ]
    if task == "conv2d":
        return "\n".join(header + [
            "def mlx(x, w):",
            '    """v1 baseline mlx (``mx.conv2d``)."""',
            "    import mlx.core as mx",
            "    return mx.conv2d(x, w)",
            "",
            "",
            "def torch_mps(x, w):",
            '    """v1 baseline torch_mps (NHWC -> NCHW -> conv2d -> NHWC)."""',
            "    import torch.nn.functional as F",
            "    return F.conv2d(x.permute(0, 3, 1, 2), w.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)",
            "",
            'BASELINES = {"mlx": mlx, "torch_mps": torch_mps}',
            "",
        ])
    if task == "lion":
        return "\n".join(header + [
            "# Frozen OpSpec attr defaults (LR/WD/BETA1/BETA2 have no per-instance override).",
            "_LR, _WD, _BETA1, _BETA2 = 0.001, 0.1, 0.9, 0.99",
            "_COMPILED: dict = {}",
            "",
            "",
            "def _compiled(key, builder):",
            '    """@brief 每个 key 只编译一次（与 v1 ``mlx_compiled``/``torch_compiled`` 相同）。"""',
            "    fn = _COMPILED.get(key)",
            "    if fn is None:",
            "        fn = _COMPILED[key] = builder()",
            "    return fn",
            "",
            "",
            "def _mlx_step(mx):",
            "    def step(p, g, m):",
            "        u = mx.sign(_BETA1 * m + (1 - _BETA1) * g)",
            "        return p * (1 - _LR * _WD) - _LR * u, _BETA2 * m + (1 - _BETA2) * g",
            "    return step",
            "",
            "",
            "def mlx(p, g, m):",
            '    """v1 baseline mlx: the whole step fused with ``mx.compile``."""',
            "    import mlx.core as mx",
            "    return _compiled(\"lion-mlx\", lambda: mx.compile(_mlx_step(mx)))(p, g, m)",
            "",
            "",
            "def _torch_step(p, g, m):",
            "    import torch",
            "    u = torch.sign(_BETA1 * m + (1 - _BETA1) * g)",
            "    return p * (1 - _LR * _WD) - _LR * u, _BETA2 * m + (1 - _BETA2) * g",
            "",
            "",
            "def torch_mps(p, g, m):",
            '    """v1 baseline torch_mps: eager ops."""',
            "    return _torch_step(p, g, m)",
            "",
            "",
            "def torch_compile(p, g, m):",
            '    """v1 baseline torch_compile: Inductor fuses the whole step into one kernel."""',
            "    import torch",
            "    return _compiled(\"lion-torch\", lambda: torch.compile(_torch_step))(p, g, m)",
            "",
            'BASELINES = {"mlx": mlx, "torch_mps": torch_mps, "torch_compile": torch_compile}',
            "",
        ])
    raise SystemExit(f"{task}: B2 does not migrate this task")


def migrate_task(task: str, combo: dict[str, Any], *, check: bool) -> list[str]:
    """@brief 迁移一个任务目录。

    @param task 任务名。
    @param combo 冻结组合。
    @param check 只检查不写。
    @return 动作日志。
    """
    directory = TASKS / task
    document = json.loads((directory / "opspec.json").read_text(encoding="utf-8"))
    if int(document.get("spec_version", 0)) == 2:
        return [f"{task}: already v2, nothing to do"]
    v2_document = _a5().v2_opspec(combo["opspec"], task)
    reference = _conv2d(combo) if task == "conv2d" else _lion(combo)
    error_kind = "module" if task == "lion" else "none"
    files = {
        "opspec.json": json.dumps(v2_document, ensure_ascii=False, indent=2) + "\n",
        "reference.py": reference,
        "reference.json": json.dumps({"schema_version": 1, "reference_version": 1, "error_scale": {"kind": error_kind}}, indent=2) + "\n",
        "baselines.json": json.dumps({"schema_version": 1, "baselines": combo["baselines"]}, ensure_ascii=False, indent=2) + "\n",
        "baselines.py": _baselines(task),
        "display.json": json.dumps({"schema_version": 1, "description": combo["description"]}, ensure_ascii=False, indent=2) + "\n",
    }
    if check:
        return [f"would write {directory / name}" for name in files] + [
            f"would delete {directory / name}" for name in V1_MARKERS if (directory / name).exists()
        ]
    actions: list[str] = []
    candidates = B2 / "evidence" / "candidates"
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
    """@brief 迁移 B2 两题并做 v2 装配冒烟。

    @return 无。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    frozen = json.loads((B2 / "evidence" / "v1-frozen.json").read_text(encoding="utf-8"))
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
        print(f"v2 discovery now finds {len(names)} tasks")


if __name__ == "__main__":
    main()
