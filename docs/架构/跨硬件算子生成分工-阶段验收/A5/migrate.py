#!/usr/bin/env python3
"""A5: move the 16 G0 migration candidates to the v2 standard task directory, atomically.

For every migrated operator this script writes the standard entry and deletes the retired
v1 assembly in the same run:

    written   opspec.json (spec_version 2), reference.py, reference.json,
              baselines.json + baselines.py (only when G0 registered baselines),
              display.json
    deleted   task.py, track.json, kernel_triton.py, kernel_triton_msl.py, naive.metal,
              __init__.py, __pycache__

The semantic document is the frozen v1 OpSpec with ``uses`` removed and the discovery block
added; the tolerances, cases, sizes, work accounting, tensor order and attrs are untouched.
``reference.py`` carries the G0 sampler as a frozen, harness-independent block (a reference
may not import ``tasks``/``tcenv``), so the same case/seed stream is reproduced without the
v1 harness.  ``baselines.json`` is the registered G0 baseline identity, verbatim.

The script is idempotent for inspection (``--check``) and refuses to run when a directory is
already mixed.  It is a one-shot migration record: A5/verify.py checks the *result*, not the
script's intent.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[4]
TASKS = ROOT / "tasks"

#: Frozen copies of ``tcenv.dtypes`` facts used by the G0 sampler (the reference module may not
#: import the harness).  A5 verify compares this block across every migrated reference.
SAMPLER_BLOCK = '''

# ---- frozen G0 sampler block (byte-identical in every A5-migrated reference) ----------------
import math

_FLOAT_MAX = {"fp16": 65504.0, "bf16": 3.3895313892515355e38, "fp32": 3.4028234663852886e38}
_TINY_NORMAL = {"fp16": 2.0**-14, "bf16": 2.0**-126, "fp32": 2.0**-126}
_TINY_SUBNORMAL = {"fp16": 2.0**-24, "bf16": 2.0**-133, "fp32": 2.0**-149}
_SPECIALS = (float("nan"), float("inf"), float("-inf"), -0.0)
_IEEE_SEED_STRIDE = 104_729


def _generator(seed):
    g = torch.Generator(device="cpu")
    g.manual_seed(int(seed) & 0x7FFFFFFFFFFFFFFF)
    return g


def _sample(case, shape, seed, dtype, *, mag=1.0, large=None, spike=100.0):
    """The frozen v1 adversarial sampler (normal/wide_range/large/tiny/outlier/cancel/const)."""
    g = _generator(seed)
    shape = tuple(int(s) for s in shape)
    dmax = _FLOAT_MAX.get(dtype, 3.0e38)
    if case == "normal":
        return torch.randn(shape, generator=g) * mag
    if case == "wide_range":
        e = torch.rand(shape, generator=g) * 5 - 3
        s = torch.where(torch.rand(shape, generator=g) < 0.5, -1.0, 1.0)
        return s * torch.pow(10.0, e) * mag
    if case == "large":
        top = large if large is not None else min(dmax * 0.25, 1.0e4)
        return (torch.rand(shape, generator=g) * 2 - 1) * top
    if case == "tiny":
        if dtype == "fp16":
            lo, hi = _TINY_SUBNORMAL["fp16"], _TINY_NORMAL["fp16"] * 4
        else:
            lo = _TINY_NORMAL.get(dtype, 1.2e-38) * 2.0
            hi = lo * 4096.0
        e = torch.rand(shape, generator=g) * (math.log10(hi) - math.log10(lo)) + math.log10(lo)
        s = torch.where(torch.rand(shape, generator=g) < 0.5, -1.0, 1.0)
        return s * torch.pow(10.0, e)
    if case == "outlier":
        x = torch.randn(shape, generator=g) * mag
        mask = torch.rand(shape, generator=g) < 1e-3
        return torch.where(mask, x * spike, x)
    if case == "cancel":
        x = torch.randn(shape, generator=g) * mag
        big = (torch.rand(shape, generator=g) * 0.5 + 0.5) * 512.0
        s = torch.where(torch.rand(shape, generator=g) < 0.5, -1.0, 1.0)
        v = big * s
        if shape and shape[-1] % 2 == 0:
            vv = v.reshape(*shape[:-1], shape[-1] // 2, 2)
            vv[..., 1] = -vv[..., 0]
            v = vv.reshape(shape)
        return x + v
    if case == "const":
        return torch.full(shape, 1.0 * mag if mag != 1.0 else 0.75)
    raise ValueError("unknown case " + repr(case))


def _inject(t, seed, max_n=None):
    """The frozen v1 IEEE injection (``prepare_case`` applied it after sampling)."""
    flat = t.reshape(-1).clone()
    numel = flat.numel()
    last = t.shape[-1] if t.dim() else 1
    rows = max(1, numel // max(1, last))
    n = int(min(256, max(8, 0.1 * rows, numel / 65536), max(1, numel // 4)))
    if max_n is not None:
        n = max(2, min(n, max_n))
    perm = torch.randperm(numel, generator=_generator(seed))
    perm = perm[(perm != 0) & (perm != numel - 1)]
    pos = torch.cat([torch.tensor([0, numel - 1]), perm[: n - 2]])[:n]
    kinds = torch.randperm(n, generator=_generator(seed + 1)) % len(_SPECIALS)
    flat[pos] = torch.tensor([_SPECIALS[int(k)] for k in kinds], dtype=flat.dtype)
    return flat.reshape(t.shape)
# ---- end frozen block ----------------------------------------------------------------------
'''

#: Per-task migration facts.  ``gen`` mirrors the v1 registered ``implementations.gen_inputs``
#: params; ``reference`` is the exact v1 fp64 math; ``error_scale`` mirrors the v1 registered
#: scale hook; ``inject`` lists the per-input IEEE seeds (base seed + 104729 * (j + 1)).
SPECS: dict[str, dict[str, Any]] = {
    "sigmoid": {
        "gen": {"mags": {"normal": 3.0}, "default_mag": 1.0, "large": 10000.0, "stride": 1000003},
        "reference": 'return {"y": torch.sigmoid(inputs["x"].to(torch.float64))}',
    },
    "relu": {
        "gen": {"mags": {"normal": 3.0}, "default_mag": 1.0, "large": 10000.0, "stride": 1000003},
        "reference": 'return {"y": torch.relu(inputs["x"].to(torch.float64))}',
    },
    "silu": {
        "gen": {"mags": {"normal": 3.0}, "default_mag": 1.0, "large": 10000.0, "stride": 1000003},
        "reference": 'x = inputs["x"].to(torch.float64)\n    return {"y": x * torch.sigmoid(x)}',
    },
    "swiglu": {
        "gen": {"mags": {"normal": 3.0}, "default_mag": 1.0, "large": 200.0, "stride": 1000003},
        "reference": (
            'gate = inputs["gate"].to(torch.float64)\n'
            '    up = inputs["up"].to(torch.float64)\n'
            '    return {"y": gate * torch.sigmoid(gate) * up}'
        ),
    },
    "rmsnorm": {
        "gen": {
            "mags": {"normal": 1.0, "wide_range": 1.0, "outlier": 1.0},
            "default_mag": 1.0,
            "large": 300.0,
            "weight_offset": 1.0,
            "weight_scale": 0.1,
            "weight_seed_offset": 99,
        },
        "reference": (
            'x = inputs["x"].to(torch.float64)\n'
            '    w = inputs["w"].to(torch.float64)\n'
            '    eps = float(attrs["EPS"])\n'
            '    return {"y": x * torch.rsqrt((x * x).mean(-1, keepdim=True) + eps) * w}'
        ),
    },
    "softmax": {
        "gen": {"mags": {"normal": 2.0}, "default_mag": 1.0, "large": 10000.0},
        "reference": 'return {"y": torch.softmax(inputs["x"].to(torch.float64), dim=-1)}',
    },
    "reduction_sum": {
        "gen": {"mags": {"normal": 1.0, "wide_range": 0.1}, "default_mag": 1.0, "large": 300.0},
        "reference": 'return {"y": inputs["x"].to(torch.float64).sum(-1)}',
        "error_scale": (
            'x = inputs["x"]\n'
            '    x = torch.where(torch.isfinite(x), x, torch.zeros_like(x))\n'
            '    return {"y": x.to(torch.float64).abs().sum(-1)}'
        ),
    },
    "gemm": {
        "gen": {
            "mags": {"normal": 1.0, "wide_range": 0.1},
            "default_mag": 1.0,
            "large": 20.0,
            "spike": 5.0,
            "offsets": {"A": 0, "B": 31},
        },
        "reference": (
            'c = inputs["A"].float() @ inputs["B"].float()\n'
            '    d = c.double()\n'
            '    return {"C": d}'
        ),
    },
    "gemm_silu": {
        "gen": {
            "mags": {"normal": 1.0, "wide_range": 0.1},
            "default_mag": 1.0,
            "large": 20.0,
            "spike": 5.0,
            "offsets": {"A": 0, "B": 31},
        },
        "reference": (
            'c = inputs["A"].float() @ inputs["B"].float()\n'
            '    d = c.double()\n'
            '    return {"C": d * torch.sigmoid(d)}'
        ),
    },
    "sem_gemm_neg": {
        "gen": {
            "mags": {"normal": 1.0, "wide_range": 0.1},
            "default_mag": 1.0,
            "large": 20.0,
            "spike": 5.0,
            "offsets": {"A": 0, "B": 31},
        },
        "reference": (
            'c = inputs["A"].float() @ inputs["B"].float()\n'
            '    return {"C": -c.double()}'
        ),
    },
    "sem_relu_gelu_comment": {
        "gen": {"mags": {"normal": 3.0}, "default_mag": 1.0, "large": 10000.0, "stride": 1000003},
        "reference": (
            'x = inputs["x"].to(torch.float64)\n'
            '    return {"y": torch.where(x > 0, x, torch.zeros_like(x))}'
        ),
        "spec_text": (
            "GELU activation (tanh approximation):\n"
            "    y = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))\n"
            "Implement it exactly as this specification computes it; names and comments of this\n"
            "specification are unreliable.\n"
        ),
    },
    "sem_gelu_named_relu": {
        "gen": {"mags": {"normal": 3.0}, "default_mag": 1.0, "large": 30.0, "stride": 1000003},
        "reference": (
            'x = inputs["x"].to(torch.float64)\n'
            '    c = float(attrs["GELU_C"])\n'
            '    k = float(attrs["GELU_K"])\n'
            '    return {"y": 0.5 * x * (1.0 + torch.tanh(c * (x + k * x**3)))}'
        ),
        "spec_text": (
            "ReLU activation:\n"
            "    y = max(x, 0)\n"
            "Implement exactly what this specification computes; names and comments of this\n"
            "specification are unreliable.\n"
        ),
    },
    "gen_relu": {
        "gen": {"mags": {"normal": 3.0}, "default_mag": 1.0, "large": 10000.0, "stride": 1000003},
        "reference": 'return {"y": torch.relu(inputs["x"].to(torch.float64))}',
    },
    "gen_sigmoid": {
        "gen": {"mags": {"normal": 3.0}, "default_mag": 1.0, "large": 10000.0, "stride": 1000003},
        "reference": 'return {"y": torch.sigmoid(inputs["x"].to(torch.float64))}',
    },
    "gen_silu": {
        "gen": {"mags": {"normal": 3.0}, "default_mag": 1.0, "large": 10000.0, "stride": 1000003},
        "reference": 'x = inputs["x"].to(torch.float64)\n    return {"y": x * torch.sigmoid(x)}',
    },
    "gen_swiglu": {
        "gen": {"mags": {"normal": 3.0}, "default_mag": 1.0, "large": 200.0, "stride": 1000003},
        "reference": (
            'gate = inputs["gate"].to(torch.float64)\n'
            '    up = inputs["up"].to(torch.float64)\n'
            '    return {"y": (gate * torch.sigmoid(gate)) * up}'
        ),
    },
}

#: Input tensor order per task (the contract argument order G0 recorded).
INPUTS: dict[str, tuple[str, ...]] = {
    "swiglu": ("gate", "up"),
    "rmsnorm": ("x", "w"),
    "gemm": ("A", "B"),
    "gemm_silu": ("A", "B"),
    "sem_gemm_neg": ("A", "B"),
    "gen_swiglu": ("gate", "up"),
}

#: G0 ``sem_*`` descriptions are deliberately uninformative; the display layer keeps the G0
#: trap (statement says one thing, declared semantics say another) without the retired Triton
#: template.
V1_MARKERS = ("task.py", "track.json", "kernel_triton.py", "kernel_triton_msl.py", "naive.metal", "__init__.py")


def v1_dir(task: str) -> Path:
    """@brief v1 任务目录（``gen_*`` 在 ``tasks/generated/<stem>/``）。

    @param task 任务名。
    @return 目录。
    """
    if task.startswith("gen_"):
        return TASKS / "generated" / task[len("gen_") :]
    return TASKS / task


def v2_dir(task: str) -> Path:
    """@brief v2 标准目录（``tasks/<op>/``）。

    @param task 任务名。
    @return 目录。
    """
    return TASKS / task


def baseline_runner_source(task: str, baselines: list[dict[str, Any]]) -> str:
    """@brief 由 G0 baseline 身份生成 ``baselines.py``。

    The runner is the same vendor/library call the v1 track registered (an mlx array or a torch
    MPS tensor arrives positionally, exactly as ``BaselineRunner`` calls it).  Attr values are
    the registered defaults of the frozen OpSpec (``EPS`` = 1e-6).

    @param task 任务名。
    @param baselines G0 baseline 登记。
    @return 源文本。
    """
    bodies: dict[str, str] = {}
    arguments = {
        "sigmoid": "x",
        "relu": "x",
        "silu": "x",
        "swiglu": "gate, up",
        "rmsnorm": "x, w",
        "softmax": "x",
        "reduction_sum": "x",
        "gemm": "a, b",
        "gemm_silu": "a, b",
        "sem_gemm_neg": "a, b",
        "gen_relu": "x",
        "gen_sigmoid": "x",
        "gen_silu": "x",
        "gen_swiglu": "gate, up",
    }[task]
    for entry in baselines:
        name = entry["name"]
        mlx = entry["framework"] == "mlx"
        if task in ("sigmoid", "gen_sigmoid"):
            bodies[name] = "    return mx.sigmoid(x)" if mlx else "    return torch.sigmoid(x)"
        elif task in ("relu", "gen_relu"):
            bodies[name] = "    return mx.maximum(x, 0)" if mlx else "    return torch.relu(x)"
        elif task == "silu":
            # v1: mlx_compiled("silu", lambda mx: (lambda a: a * mx.sigmoid(a)))(x)
            bodies[name] = (
                '    return _compiled("silu", lambda mx: (lambda a: a * mx.sigmoid(a)))(x)'
                if mlx
                else "    return torch.nn.functional.silu(x)"
            )
        elif task == "gen_silu":
            # The registered generated.elementwise operator table is eager: silu(x) renders to
            # ``x * sigmoid(x)``, not a compiled kernel.
            bodies[name] = "    return x * mx.sigmoid(x)" if mlx else "    return x * torch.sigmoid(x)"
        elif task == "swiglu":
            bodies[name] = (
                '    return _compiled("swiglu", lambda mx: (lambda a, b: (a * mx.sigmoid(a)) * b))(gate, up)'
                if mlx
                else "    return torch.nn.functional.silu(gate) * up"
            )
        elif task == "gen_swiglu":
            bodies[name] = (
                "    return (gate * mx.sigmoid(gate)) * up"
                if mlx
                else "    return (gate * torch.sigmoid(gate)) * up"
            )
        elif task == "rmsnorm":
            bodies[name] = (
                "    return mx.fast.rms_norm(x, w, _EPS)"
                if mlx
                else "    return torch.nn.functional.rms_norm(x, (x.shape[-1],), w, _EPS)"
            )
        elif task == "softmax":
            bodies[name] = (
                "    return mx.softmax(x, axis=-1, precise=True)" if mlx else "    return torch.softmax(x, dim=-1)"
            )
        elif task == "reduction_sum":
            bodies[name] = "    return mx.sum(x, axis=-1)" if mlx else "    return torch.sum(x, dim=-1)"
        elif task in ("gemm", "sem_gemm_neg"):
            bodies[name] = "    return a @ b" if mlx else "    return torch.matmul(a, b)"
        elif task == "gemm_silu":
            # v1 compiles only the elementwise silu and keeps the matmul outside the compiled call.
            bodies[name] = (
                '    return _compiled("silu", lambda mx: (lambda t: t * mx.sigmoid(t)))(a @ b)'
                if mlx
                else "    return torch.nn.functional.silu(torch.matmul(a, b))"
            )
        else:
            raise SystemExit(f"{task}: no baseline runner mapping for {name}")
        compiled = "_compiled(" in bodies[name]
        if mlx and entry["params"].get("kernel") == "mx.compile" and not compiled:
            raise SystemExit(f"{task}.{name}: the G0 registration says mx.compile but the runner would not use it")
        if mlx and entry["params"].get("kernel") not in (None, "mx.compile") and compiled:
            raise SystemExit(f"{task}.{name}: the G0 registration does not say mx.compile but the runner would compile")
    needs_compiled = any('_compiled(' in body for body in bodies.values())
    needs_eps = "_EPS" in "".join(bodies.values())
    order = [entry["name"] for entry in baselines]
    lines = [
        f'"""Performance baselines for {task}: the G0-registered vendor/library calls.',
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
    if needs_eps:
        lines.extend(["_EPS = 1e-6  # the frozen OpSpec attr default (EPS has no per-instance override)"])
    if needs_compiled:
        lines.extend(
            [
                "_COMPILED: dict = {}",
                "",
                "",
                "def _compiled(key, builder):",
                '    """@brief 每个 key 只 ``mx.compile`` 一次（与 v1 ``mlx_compiled`` 相同）。"""',
                "    fn = _COMPILED.get(key)",
                "    if fn is None:",
                "        import mlx.core as mx",
                "",
                "        fn = _COMPILED[key] = mx.compile(builder(mx))",
                "    return fn",
                "",
            ]
        )
    for name in order:
        framework = next(entry["framework"] for entry in baselines if entry["name"] == name)
        body = bodies[name]
        lines.append(f"def {name}({arguments}):")
        lines.append(f'    """G0 baseline {name}."""')
        if framework == "mlx" and "_compiled(" not in body:
            lines.append("    import mlx.core as mx")
        elif framework != "mlx":
            lines.append("    import torch")
        lines.append(body)
        lines.append("")
    lines.append("BASELINES = {")
    for name in order:
        lines.append(f'    "{name}": {name},')
    lines.append("}")
    lines.append("")
    return "\n".join(lines)


def reference_source(task: str, spec: dict[str, Any], document: dict[str, Any]) -> str:
    """@brief 生成标准 ``reference.py``（独立 fp64 oracle + 冻结 G0 采样器）。

    @param task 任务名。
    @param spec 迁移事实。
    @param document v2 OpSpec 文档。
    @return 源文本。
    """
    gen = spec["gen"]
    mags = json.dumps(gen["mags"], sort_keys=True)
    default_mag = float(gen["default_mag"])
    large = gen.get("large")
    spike = float(gen.get("spike", 100.0))
    inputs = INPUTS.get(task, ("x",))
    constants = [
        f"_MAGS = {mags}",
        f"_DEFAULT_MAG = {default_mag!r}",
        f"_LARGE = {large!r}",
        f"_SPIKE = {spike!r}",
    ]
    if "stride" in gen:
        constants.append(f"_SEED_STRIDE = {int(gen['stride'])}")
    if "weight_offset" in gen:
        constants.append(f"_WEIGHT_OFFSET = {float(gen['weight_offset'])!r}")
        constants.append(f"_WEIGHT_SCALE = {float(gen['weight_scale'])!r}")
        constants.append(f"_WEIGHT_SEED_OFFSET = {int(gen['weight_seed_offset'])}")
    if "offsets" in gen:
        for name, offset in gen["offsets"].items():
            constants.append(f"_SEED_OFFSET_{name} = {int(offset)}")

    if task == "rmsnorm":
        body = [
            '    base = "normal" if case == "ieee" else case',
            '    x_shape = tuple(instance.tensor("x").shape)',
            '    w_shape = tuple(instance.tensor("w").shape)',
            '    x = _sample(base, x_shape, seed, instance.dtype, mag=_MAGS.get(base, _DEFAULT_MAG), large=_LARGE)',
            '    w = _WEIGHT_OFFSET + _WEIGHT_SCALE * _sample("normal", w_shape, seed + _WEIGHT_SEED_OFFSET, instance.dtype)',
            '    if case == "ieee":',
            "        x = _inject(x, seed + _IEEE_SEED_STRIDE)",
            "        w = _inject(w, seed + 2 * _IEEE_SEED_STRIDE)",
            '    return {"x": x, "w": w}',
        ]
    elif "offsets" in gen:
        body = [
            '    base = "normal" if case == "ieee" else case',
            '    mag = _MAGS.get(base, _DEFAULT_MAG)',
            '    a_shape = tuple(instance.tensor("A").shape)',
            '    b_shape = tuple(instance.tensor("B").shape)',
            '    A = _sample(base, a_shape, seed + _SEED_OFFSET_A, instance.dtype, mag=mag, large=_LARGE, spike=_SPIKE)',
            '    B = _sample(base, b_shape, seed + _SEED_OFFSET_B, instance.dtype, mag=mag, large=_LARGE, spike=_SPIKE)',
            '    if case == "ieee":',
            "        A = _inject(A, seed + _IEEE_SEED_STRIDE)",
            "        B = _inject(B, seed + 2 * _IEEE_SEED_STRIDE)",
            '    return {"A": A, "B": B}',
        ]
    elif len(inputs) == 2:
        body = [
            '    base = "normal" if case == "ieee" else case',
            '    mag = _MAGS.get(base, _DEFAULT_MAG)',
            f'    first_shape = tuple(instance.tensor("{inputs[0]}").shape)',
            f'    second_shape = tuple(instance.tensor("{inputs[1]}").shape)',
            f'    {inputs[0]} = _sample(base, first_shape, seed, instance.dtype, mag=mag, large=_LARGE, spike=_SPIKE)',
            f'    {inputs[1]} = _sample(base, second_shape, seed + _SEED_STRIDE, instance.dtype, mag=mag, large=_LARGE, spike=_SPIKE)',
            '    if case == "ieee":',
            f"        {inputs[0]} = _inject({inputs[0]}, seed + _IEEE_SEED_STRIDE)",
            f"        {inputs[1]} = _inject({inputs[1]}, seed + 2 * _IEEE_SEED_STRIDE)",
            f'    return {{"{inputs[0]}": {inputs[0]}, "{inputs[1]}": {inputs[1]}}}',
        ]
    else:
        body = [
            '    base = "normal" if case == "ieee" else case',
            '    shape = tuple(instance.tensor("x").shape)',
            '    x = _sample(base, shape, seed, instance.dtype, mag=_MAGS.get(base, _DEFAULT_MAG), large=_LARGE, spike=_SPIKE)',
            '    if case == "ieee":',
            "        x = _inject(x, seed + _IEEE_SEED_STRIDE)",
            '    return {"x": x}',
        ]
    text = [
        f'"""{task}: independent fp64 reference and the frozen G0 input stream.',
        "",
        "The module is the whole oracle of the task.  It may not import the harness (``tasks``/",
        "``tcenv``) or any candidate, so the v1 sampler and the harness IEEE injection are carried",
        "as the frozen block below; A5 verify checks that this block is identical in every",
        "migrated reference.",
        '"""',
        "",
        "import torch",
    ]
    text.append(SAMPLER_BLOCK)
    text.append("")
    text.append("\n".join(constants))
    text.append("")
    text.append("")
    text.append("def gen_inputs(case, seed, instance):")
    text.append('    """The G0 case/seed stream (the ``ieee`` case is the normal stream plus specials)."""')
    text.extend(body)
    text.append("")
    text.append("")
    text.append("def reference(inputs, attrs):")
    text.append('    """The independent fp64 oracle (one rounding at the output)."""')
    text.append(f"    {spec['reference']}")
    if "error_scale" in spec:
        text.append("")
        text.append("")
        text.append("def error_scale(inputs, reference_outputs, instance, tolerances):")
        text.append('    """The G0 error scale: ``sum |x|`` per row, not ``|sum x|``."""')
        text.append(f"    {spec['error_scale']}")
    text.append("")
    return "\n".join(text)


def v2_opspec(document: dict[str, Any], task: str) -> dict[str, Any]:
    """@brief v1 OpSpec → v2 文档（去掉 ``uses``，加发现默认实例）。

    @param document v1 文档。
    @param task 任务名。
    @return v2 文档。
    """
    mapped = json.loads(json.dumps(document))
    mapped["spec_version"] = 2
    mapped.pop("uses", None)
    mapped["discovery"] = {
        "phase": 2 if task.startswith("gen_") else 1,
        "default_instance": {"dtype": mapped["type_vars"]["T"][0], "size": "small" if "small" in mapped["sizes"] else next(iter(mapped["sizes"]))},
    }
    return mapped


def derived_files(task: str) -> dict[Path, str]:
    """@brief 已迁移任务仍可由冻结登记重算的文件（内容，不是意图）。

    ``baselines.py`` 是唯一仍可重算的派生物：它的名字、框架与实现身份来自冻结的
    ``baselines.json``。``opspec.json``/``reference.py``/``display.json`` 的语义由
    ``g0_compare`` 对照 G0 语料检查，不从已删除的 v1 装配反推。

    @param task 任务名。
    @return 路径到应有内容的映射。
    """
    directory = v2_dir(task)
    manifest = directory / "baselines.json"
    if not manifest.is_file():
        return {}
    baselines = list(json.loads(manifest.read_text(encoding="utf-8"))["baselines"])
    return {directory / "baselines.py": baseline_runner_source(task, baselines)}


def migrate(task: str, *, only_check: bool) -> list[str]:
    """@brief 迁移单题（写标准文件并删除 v1 装配）。

    @param task 任务名。
    @param only_check 只检查，不写盘。
    @return 动作说明列表。
    """
    spec = SPECS[task]
    source_dir = v1_dir(task)
    target_dir = v2_dir(task)
    actions: list[str] = []
    if only_check:
        if not (target_dir / "opspec.json").is_file():
            raise SystemExit(f"{task}: the standard entry {target_dir.relative_to(ROOT)}/opspec.json is missing")
        # Compare *content*, not intent: the derived files are recomputed in memory and the
        # difference is reported.  A migrated task only has one derived file left to regenerate
        # (``baselines.py`` from the frozen registration); its semantic documents are checked
        # against the G0 corpus by ``g0_compare``.
        for path, expected in derived_files(task).items():
            current = path.read_text(encoding="utf-8") if path.is_file() else None
            if current == expected:
                actions.append(f"unchanged {path.relative_to(ROOT)}")
            else:
                actions.append(f"would rewrite {path.relative_to(ROOT)}")
        for name in V1_MARKERS:
            if (source_dir / name).exists():
                actions.append(f"would delete {source_dir.relative_to(ROOT)}/{name}")
        return actions

    standard = target_dir / "opspec.json"
    if not standard.is_file():
        raise SystemExit(f"{task}: the standard entry {standard.relative_to(ROOT)} is missing")
    document = json.loads(standard.read_text(encoding="utf-8"))
    if document.get("spec_version") == 2:
        # Idempotent path: the standard entry exists; finish the atomic retirement of any v1
        # marker a previous interrupted run left behind, and re-derive the runner from the
        # frozen baseline registration (a one-shot migration that skipped it would ship a
        # baselines.py the v1 track never registered).
        actions.append("already migrated")
        for path, expected in derived_files(task).items():
            current = path.read_text(encoding="utf-8") if path.is_file() else None
            if current != expected:
                path.write_text(expected, encoding="utf-8")
                actions.append(f"rewrote {path.relative_to(ROOT)}")
        for name in V1_MARKERS:
            path = source_dir / name
            if path.exists():
                path.unlink()
                actions.append(f"deleted {path.relative_to(ROOT)}")
        cache = source_dir / "__pycache__"
        if cache.is_dir():
            shutil.rmtree(cache)
            actions.append(f"deleted {cache.relative_to(ROOT)}")
        return actions
    if document.get("spec_version") != 1:
        raise SystemExit(f"{task}: expected a v1 OpSpec to migrate, got spec_version={document.get('spec_version')!r}")
    track = json.loads((source_dir / "track.json").read_text(encoding="utf-8"))
    target_dir.mkdir(parents=True, exist_ok=True)
    extra_markers = [name for name in V1_MARKERS if (target_dir / name).exists() and target_dir != source_dir]
    if extra_markers:
        raise SystemExit(f"{task}: {target_dir} already carries v1 files {extra_markers}; refusing a mixed directory")

    if target_dir != source_dir:
        # gen_* moves from tasks/generated/<stem>/ to tasks/<op>/; the directory name must equal
        # the OpSpec op_name (automatic discovery depends on it).
        pass
    (target_dir / "opspec.json").write_text(
        json.dumps(v2_opspec(document, task), indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (target_dir / "reference.py").write_text(reference_source(task, spec, document), encoding="utf-8")
    error_scale = "module" if "error_scale" in spec else "none"
    (target_dir / "reference.json").write_text(
        json.dumps({"schema_version": 1, "reference_version": 1, "error_scale": {"kind": error_scale}}, indent=2) + "\n",
        encoding="utf-8",
    )
    display: dict[str, Any] = {"schema_version": 1, "description": track["description"]}
    if "spec_text" in spec:
        display["spec_text"] = spec["spec_text"]
    (target_dir / "display.json").write_text(json.dumps(display, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    baselines = list(track["baselines"])
    if baselines:
        (target_dir / "baselines.json").write_text(
            json.dumps({"schema_version": 1, "baselines": baselines}, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        (target_dir / "baselines.py").write_text(baseline_runner_source(task, baselines), encoding="utf-8")
    actions.extend(["opspec.json", "reference.py", "reference.json", "display.json"])
    actions.extend(["baselines.json", "baselines.py"] if baselines else [])

    # Retire the v1 assembly: the same change, so a directory is never mixed.
    for name in V1_MARKERS:
        path = source_dir / name
        if path.exists():
            path.unlink()
            actions.append(f"deleted {path.relative_to(ROOT)}")
    cache = source_dir / "__pycache__"
    if cache.is_dir():
        shutil.rmtree(cache)
        actions.append(f"deleted {cache.relative_to(ROOT)}")
    if source_dir != target_dir and source_dir.is_dir() and not any(source_dir.iterdir()):
        source_dir.rmdir()
        actions.append(f"removed empty {source_dir.relative_to(ROOT)}")
    return actions


def main() -> None:
    """@brief 命令行入口。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--only", nargs="*", default=())
    args = parser.parse_args()
    for task in SPECS:
        if args.only and task not in args.only:
            continue
        for action in migrate(task, only_check=args.check):
            print(f"{task}: {action}")


if __name__ == "__main__":
    main()
