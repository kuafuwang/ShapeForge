#!/usr/bin/env python3
"""B3: migrate flash_attention, bitdelta_gemm and mamba2_chunk_scan to the v2 standard entry.

Same rule as B1/B2: freeze first (216 records across 12 combinations), then write the v2 entry,
then prove byte-level equivalence.  B3 is the batch where the oracle helpers can no longer be
shared through the harness: a v2 ``reference.py`` may not import ``tasks``/``tcenv``, so the
BitDelta unpacker and the whole Mamba-2 ``ssd_chunk_scan`` operator set are *inlined* into the
reference and baseline modules.  Duplication is deliberate — an oracle that imports the thing it
judges is not an oracle.

IEEE details that differ per task and are preserved exactly:
* flash_attention injects specials into ``q`` only (``cases.inputs.ieee._inject = ["q"]``).
* bitdelta uses ``random`` as its ieee base distribution (mapped back to ``normal`` for x), and
  its packed int32 words are never injected (v1 only injects float tensors).
* mamba2 injects into ``states`` only, at ``seed + 104729 * 7`` (its position in the input list).

Run:  uv run python docs/架构/跨硬件算子生成分工-阶段验收/B3/migrate.py [--check]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
from pathlib import Path
from typing import Any

B3 = Path(__file__).resolve().parent
ROOT = B3.parents[3]
TASKS = ROOT / "tasks"
V1_MARKERS = ("task.py", "track.json", "kernel_triton.py", "kernel_triton_msl.py", "naive.metal", "__init__.py")


def _a5() -> Any:
    """@brief 载入 A5 迁移模块，复用冻结采样器文本与 OpSpec 映射规则。

    @return A5 migrate 模块。
    """
    spec = importlib.util.spec_from_file_location("_b3_a5_migrate", B3.parent / "A5" / "migrate.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _header(task: str, extra: str) -> list[str]:
    """@brief reference.py 的头部与采样器块。

    @param task 任务名。
    @param extra 该题特有的语义注意点。
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


def _flash_attention(combo: dict[str, Any]) -> str:
    """@brief 生成 flash_attention 的 reference.py。

    @param combo 冻结组合。
    @return 源文本。
    """
    gen = combo["implementations"]["gen_inputs"]["params"]
    constants = [
        f"_MAGS = {json.dumps(gen['mags'], sort_keys=True)}",
        f"_DEFAULT_MAG = {float(gen['default_mag'])!r}",
        f"_LARGE = {float(gen['large'])!r}",
        f"_SPIKE = {float(gen['spike'])!r}",
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
        '        # v1 declares cases.inputs.ieee._inject = ["q"]: only q receives specials.',
        "        out[\"q\"] = _inject(out[\"q\"], seed + _IEEE_SEED_STRIDE)",
        "    return out",
    ]
    reference = '''    scale = float(attrs["SM_SCALE"])
    o = F.scaled_dot_product_attention(
        inputs["q"].float(), inputs["k"].float(), inputs["v"].float(), scale=scale
    )
    return {"o": o.double()}'''
    return "\n".join(
        _header("flash_attention", "IEEE 特值只注入 q；有效标度是 eff_scale（直接用 SM_SCALE）。")
        + ["\n".join(constants), "", "", "def gen_inputs(case, seed, instance):",
           '    """The v1 case/seed stream (q/k/v at seed + 0/31/62; ieee injects q only)."""']
        + body
        + ["", "", "def reference(inputs, attrs):", '    """F.scaled_dot_product_attention in fp32, rounded once to fp64."""', reference, ""]
    )


def _bitdelta(combo: dict[str, Any]) -> str:
    """@brief 生成 bitdelta_gemm 的 reference.py（含内联解包器）。

    @param combo 冻结组合。
    @return 源文本。
    """
    gen = combo["implementations"]["gen_inputs"]["params"]
    constants = [
        f"_SEED_MASK = {int(gen['seed_mask'])}",
        f"_EDGE_VALUES = torch.tensor({[int(v) for v in gen['edge_words']]}, dtype=torch.int64)",
        f"_BITS_CASE = {gen['bits_case']!r}",
        f"_X_CASE_MAP = {json.dumps(gen['x_case_map'], sort_keys=True)}",
        f"_X_SEED_OFFSET = {int(gen['x_seed_offset'])}",
        f"_X_MAG = {float(gen['x_mag'])!r}",
        f"_X_LARGE = {float(gen['x_large'])!r}",
        f"_X_SPIKE = {float(gen['x_spike'])!r}",
    ]
    unpack = '''

def _unpack_pm1(words):
    """[K//32, N] int32 words -> [K, N] float32 of +-1 (bit j of word kb is row kb*32 + j)."""
    w = words.to(torch.int64)
    j = torch.arange(32, dtype=torch.int64).reshape(1, 32, 1)
    bit = (w[:, None, :] >> j) & 1
    return (bit.to(torch.float32) * 2 - 1).reshape(-1, words.shape[-1])

'''
    body = [
        '    # v1 uses "random" as the ieee base; x_case_map sends random (and edge_bits) to normal.',
        '    base = "random" if case == "ieee" else case',
        '    symbols = instance.symbols',
        '    m, n, k, kb = (int(symbols[name]) for name in ("M", "N", "K", "KB"))',
        "    generator = torch.Generator().manual_seed(int(seed) & _SEED_MASK)",
        "    if base == _BITS_CASE:",
        "        words = _EDGE_VALUES[torch.randint(0, len(_EDGE_VALUES), (kb, n), generator=generator)]",
        "    else:",
        "        words = torch.randint(-(1 << 31), 1 << 31, (kb, n), generator=generator, dtype=torch.int64)",
        "    xcase = _X_CASE_MAP.get(base, base)",
        '    x = _sample(xcase, (m, k), seed + _X_SEED_OFFSET, instance.dtype, mag=_X_MAG, large=_X_LARGE, spike=_X_SPIKE)',
        '    if case == "ieee":',
        "        # bits is int32: v1 only injects float tensors, so only x is injected.",
        "        x = _inject(x, seed + _IEEE_SEED_STRIDE)",
        '    return {"x": x, "bits": words}',
    ]
    reference = '''    w = _unpack_pm1(inputs["bits"])
    return {"y": (inputs["x"].float() @ w).double() * float(attrs["SCALE"])}'''
    return "\n".join(
        _header("bitdelta_gemm", "打包符号字的解包器内联在此；ieee 以 random 为基分布，bits 不注入。")
        + ["\n".join(constants), unpack, "", "def gen_inputs(case, seed, instance):",
           '    """The v1 case/seed stream (packed sign words plus the sampled x)."""']
        + body
        + ["", "", "def reference(inputs, attrs):", '    """Unpack to +-1, fp32 matmul, fp64 output scaled by SCALE."""', reference, ""]
    )


#: The whole Mamba-2 operator, inlined into both reference.py and baselines.py.  This is the v1
#: formulation verbatim (paper listing 1 for one B/C group); the ops classes differ only in
#: whether f32 widens to fp32 (baselines) or fp64 (the reference oracle).
MAMBA_OPS = '''

def ssd_chunk_scan(x, dt, cs, bm, cm, d, st, *, ops, batch, seq, heads, headdim, dstate, chunk, half=False):
    """Mamba-2 chunk scan for one B/C group (the paper's listing 1)."""
    nc = seq // chunk
    f = ops.f32
    mm = (lambda a: a) if half else f
    x5 = x.reshape(batch, nc, chunk, heads, headdim)
    dt4 = ops.permute(f(dt).reshape(batch, nc, chunk, heads), (0, 3, 1, 2))
    cs4 = f(cs)
    br = bm.reshape(batch, nc, chunk, dstate)
    cr = cm.reshape(batch, nc, chunk, dstate)
    cb = f(ops.einsum("bcln,bcsn->bcls", mm(cr), mm(br)))
    decay = ops.exp(cs4[..., :, None] - cs4[..., None, :])
    w = ops.where(ops.causal(chunk), cb[:, None] * decay * dt4[..., None, :], 0.0)
    w = ops.like(w, x) if half else w
    y = f(ops.einsum("bhcls,bcshp->bclhp", w, mm(x5)))
    y = y + f(ops.einsum("bcln,bchpn->bclhp", mm(cr), mm(st))) * ops.permute(ops.exp(cs4), (0, 2, 3, 1))[..., None]
    y = y + f(d)[:, None] * f(x5)
    return y.reshape(batch, seq, heads, headdim)


class _MlxOps:
    """mlx operator set (fp32 decay and sums)."""

    @staticmethod
    def f32(a):
        import mlx.core as mx
        return a.astype(mx.float32)

    @staticmethod
    def permute(a, axes):
        return a.transpose(*axes)

    @staticmethod
    def like(a, ref):
        return a.astype(ref.dtype)

    @staticmethod
    def einsum(spec, *a):
        import mlx.core as mx
        return mx.einsum(spec, *a)

    @staticmethod
    def exp(a):
        import mlx.core as mx
        return mx.exp(a)

    @staticmethod
    def where(m, a, b):
        import mlx.core as mx
        return mx.where(m, a, b)

    @staticmethod
    def causal(n):
        import mlx.core as mx
        return mx.tril(mx.ones((n, n))) > 0


class _TorchOps:
    """torch operator set (fp32 decay and sums)."""

    @staticmethod
    def f32(a):
        return a.float()

    @staticmethod
    def permute(a, axes):
        return a.permute(*axes)

    @staticmethod
    def like(a, ref):
        return a.to(ref.dtype)

    einsum = staticmethod(torch.einsum)
    exp = staticmethod(torch.exp)

    @staticmethod
    def where(m, a, b):
        return torch.where(m, a, torch.as_tensor(b, dtype=a.dtype, device=a.device))

    @staticmethod
    def causal(n):
        return torch.tril(torch.ones((n, n), dtype=torch.bool, device="mps"))


class _RefOps(_TorchOps):
    """The oracle's operator set: f32 widens to fp64 and the causal mask is built on CPU."""

    f32 = staticmethod(lambda a: a.double())
    where = staticmethod(lambda m, a, b: torch.where(m, a, torch.zeros((), dtype=a.dtype)))
    causal = staticmethod(lambda n: torch.tril(torch.ones((n, n), dtype=torch.bool)))

'''


def _mamba2(combo: dict[str, Any]) -> str:
    """@brief 生成 mamba2_chunk_scan 的 reference.py（内联整个 SSD 算子）。

    @param combo 冻结组合。
    @return 源文本。
    """
    gen = combo["implementations"]["gen_inputs"]["params"]
    constants = [
        f"_MAGS = {json.dumps(gen['mags'], sort_keys=True)}",
        f"_DEFAULT_MAG = {float(gen['default_mag'])!r}",
        f"_LARGE = {float(gen['large'])!r}",
        f"_SPIKE = {float(gen['spike'])!r}",
        f"_OFFSETS = {json.dumps(gen['seed_offsets'], sort_keys=True)}",
        f"_DT_LO, _DT_HI = {float(gen['dt_range'][0])!r}, {float(gen['dt_range'][1])!r}",
        f"_DECAY_LO, _DECAY_HI = {float(gen['decay_range'][0])!r}, {float(gen['decay_range'][1])!r}",
        '_TORCH_DTYPE = {"fp16": torch.float16, "bf16": torch.bfloat16}',
    ]
    round_to = '''

def _round_to(t, dtype):
    """The v1 round_to: round to the storage dtype, then widen back to float32."""
    return t.to(_TORCH_DTYPE[dtype]).to(torch.float32)

'''
    body = [
        '    base = "normal" if case == "ieee" else case',
        '    mag = _MAGS.get(base, _DEFAULT_MAG)',
        '    symbols = instance.symbols',
        '    B = int(symbols["BATCH"])',
        '    S = int(symbols["SEQLEN"])',
        '    H = int(symbols["NHEADS"])',
        '    P = int(symbols["HEADDIM"])',
        '    N = int(symbols["DSTATE"])',
        '    chunk = int(symbols["CHUNK"])',
        "    nc = S // chunk",
        "    dt = _round_to(_DT_LO + (_DT_HI - _DT_LO) * torch.sigmoid(_sample(\"normal\", (B, S, H), seed + _OFFSETS[\"dt\"], instance.dtype)), instance.dtype)",
        "    a = -(_DECAY_LO + (_DECAY_HI - _DECAY_LO) * torch.sigmoid(_sample(\"normal\", (H,), seed + _OFFSETS[\"decay\"], instance.dtype)))",
        "    cs = (dt.float().reshape(B, nc, chunk, H) * a).cumsum(dim=2).permute(0, 3, 1, 2).contiguous()",
        "    out = {",
        '        "x": _sample(base, (B, S, H, P), seed + _OFFSETS["x"], instance.dtype, mag=mag, large=_LARGE, spike=_SPIKE),',
        '        "dt": dt,',
        '        "dA_cumsum": cs,',
        '        "Bmat": _sample(base, (B, S, N), seed + _OFFSETS["Bmat"], instance.dtype, mag=mag, large=_LARGE, spike=_SPIKE),',
        '        "Cmat": _sample(base, (B, S, N), seed + _OFFSETS["Cmat"], instance.dtype, mag=mag, large=_LARGE, spike=_SPIKE),',
        '        "Dskip": _sample(base, (H,), seed + _OFFSETS["Dskip"], instance.dtype, mag=mag, large=_LARGE, spike=_SPIKE),',
        '        "states": _sample(base, (B, nc, H, P, N), seed + _OFFSETS["states"], instance.dtype, mag=mag, large=_LARGE, spike=_SPIKE),',
        "    }",
        '    if case == "ieee":',
        '        # v1 declares cases.inputs.ieee._inject = ["states"], at seed + 104729 * 7.',
        '        out["states"] = _inject(out["states"], seed + 7 * _IEEE_SEED_STRIDE)',
        "    return out",
    ]
    reference = '''    x = inputs["x"]
    batch, seq, heads, headdim = (int(v) for v in x.shape)
    kw = dict(
        batch=batch, seq=seq, heads=heads, headdim=headdim,
        dstate=int(inputs["states"].shape[-1]), chunk=int(inputs["dA_cumsum"].shape[-1]),
    )
    y = ssd_chunk_scan(
        *(inputs[key] for key in ("x", "dt", "dA_cumsum", "Bmat", "Cmat", "Dskip", "states")),
        ops=_RefOps, **kw,
    )
    return {"y": y}'''
    return "\n".join(
        _header("mamba2_chunk_scan", "整个 SSD 算子内联；参考实现的 f32 升到 fp64，ieee 只注入 states。")
        + ["\n".join(constants), round_to, MAMBA_OPS, "", "def gen_inputs(case, seed, instance):",
           '    """The v1 case/seed stream, including the dt/decay construction."""']
        + body
        + ["", "", "def reference(inputs, attrs):", '    """The paper listing-1 scan, evaluated in fp64."""', reference, ""]
    )


def _baselines(task: str) -> str:
    """@brief 生成一题的 baselines.py。

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
    if task == "flash_attention":
        return "\n".join(header + [
            "_SM_SCALE = 0.125  # the frozen OpSpec attr default (1/sqrt(D) with D = 64)",
            "",
            "",
            "def mlx(q, k, v):",
            '    """v1 baseline mlx (``mx.fast.scaled_dot_product_attention``)."""',
            "    import mlx.core as mx",
            "    return mx.fast.scaled_dot_product_attention(q, k, v, scale=_SM_SCALE)",
            "",
            "",
            "def torch_mps(q, k, v):",
            '    """v1 baseline torch_mps (``F.scaled_dot_product_attention``)."""',
            "    import torch.nn.functional as F",
            "    return F.scaled_dot_product_attention(q, k, v, scale=_SM_SCALE)",
            "",
            'BASELINES = {"mlx": mlx, "torch_mps": torch_mps}',
            "",
        ])
    raise SystemExit(f"{task}: use _bitdelta_baselines() or _mamba2_baselines() for this task")


def _bitdelta_baselines() -> str:
    """@brief bitdelta 的四条基线（K/N 在调用时由张量形状给出）。

    @return 源文本。
    """
    return '''"""Performance baselines for bitdelta_gemm: separate or compiled decode + dense matmul.

A baseline is a timing comparison object, never the correctness oracle.  The packed words arrive
as the framework-native integer tensor; the decode follows the v1 registration (bit j of word kb
is row kb*32 + j), and K/N come from the operand shapes, exactly as the v1 closures captured them.
"""

from __future__ import annotations

_SCALE = 0.02  # the frozen OpSpec attr default
_COMPILED: dict = {}


def _compiled(key, builder):
    """@brief 每个 key 只编译一次（与 v1 的编译缓存一致）。"""
    fn = _COMPILED.get(key)
    if fn is None:
        fn = _COMPILED[key] = builder()
    return fn


def _mlx_decode(x, bits):
    import mlx.core as mx
    k, n = x.shape[1], bits.shape[1]
    j = mx.arange(32, dtype=mx.uint32).reshape(1, 32, 1)
    b = (bits.reshape(k // 32, 1, n) >> j) & 1
    w = (b.astype(x.dtype) * 2 - 1).reshape(k, n)
    return (x @ w) * _SCALE


def _torch_decode(x, bits):
    import torch
    k, n = x.shape[1], bits.shape[1]
    j = torch.arange(32, dtype=torch.int32, device=x.device).reshape(1, 32, 1)
    b = (bits.reshape(k // 32, 1, n) >> j) & 1
    w = (b.to(x.dtype) * 2 - 1).reshape(k, n)
    return (x @ w) * _SCALE


def mlx(x, bits):
    """v1 baseline mlx: separate unpack + dense matmul."""
    return _mlx_decode(x, bits)


def mlx_compile(x, bits):
    """v1 baseline mlx_compile: the decode is compiled once per (K, N)."""
    import mlx.core as mx
    return _compiled(f"bitdelta_{x.shape[1]}_{bits.shape[1]}", lambda: mx.compile(_mlx_decode))(x, bits)


def torch_mps(x, bits):
    """v1 baseline torch_mps: separate unpack + dense matmul."""
    return _torch_decode(x, bits)


def torch_compile(x, bits):
    """v1 baseline torch_compile: Inductor fuses the decode (the strong baseline)."""
    import torch
    return _compiled(f"bitdelta_torch_{x.shape[1]}_{bits.shape[1]}", lambda: torch.compile(_torch_decode))(x, bits)


BASELINES = {"mlx": mlx, "mlx_compile": mlx_compile, "torch_mps": torch_mps, "torch_compile": torch_compile}
'''


def _mamba2_baselines(combo: dict[str, Any]) -> str:
    """@brief mamba2 的五条基线（内联 SSD 算子与两套 ops）。

    @param combo 冻结组合。
    @return 源文本。
    """
    return (
        '"""Performance baselines for mamba2_chunk_scan: the SSD formulation on mlx and torch.\n\n'
        "A baseline is a timing comparison object, never the correctness oracle.  The operator is\n"
        "inlined for the same reason as in reference.py: these modules may not import the harness.\n"
        '"""\n\n'
        "from __future__ import annotations\n\n"
        "import torch\n"
        + MAMBA_OPS
        + '''


def _kwargs():
    """@brief small/paper 的实例符号在调用处给出；这里只是缓存编译键的辅助。

    @return 空字典（真正的形状在 _run 里计算）。
    """
    return {}


def _shape_kwargs(x, states, chunk):
    """@brief 由输入形状推出 ssd_chunk_scan 的关键字（与 v1 的实例符号一致）。

    @param x 输入张量。
    @param states 每块开始时的 state。
    @param chunk 每块时间步数。
    @return 形状关键字字典。
    """
    batch, seq, heads, headdim = (int(v) for v in x.shape)
    return dict(batch=batch, seq=seq, heads=heads, headdim=headdim, dstate=int(states.shape[-1]), chunk=int(chunk))


def _run(args, ops, half):
    """@brief 按 v1 的调用形式执行一次 scan，并转回输入 dtype。

    @param args 位置参数（x/dt/dA_cumsum/Bmat/Cmat/Dskip/states）。
    @param ops 算子集。
    @param half 矩阵乘操作数是否保持输入 dtype。
    @return 输出张量。
    """
    x, dt, cs, bm, cm, d, st = args
    chunk = int(cs.shape[-1])
    kw = _shape_kwargs(x, st, chunk)
    return ssd_chunk_scan(*args, ops=ops, half=half, **kw).to(x.dtype) if ops is _TorchOps or ops is _RefOps else ssd_chunk_scan(*args, ops=ops, half=half, **kw).astype(x.dtype)


def mlx(*args):
    """v1 baseline mlx: fp32 formulation, eager."""
    return _run(args, _MlxOps, False)


def mlx_compile(*args):
    """v1 baseline mlx_compile: the whole formulation compiled once."""
    import mlx.core as mx
    return _compiled_mlx("mamba2_full", False)(*args)


def mlx_half_compile(*args):
    """v1 baseline mlx_half_compile: matmul operands stay in the input dtype."""
    import mlx.core as mx
    return _compiled_mlx("mamba2_half", True)(*args)


def _compiled_mlx(key, half):
    """@brief 编译并缓存一条 mlx scan（与 v1 的 mlx_compiled 一致）。

    @param key 缓存键。
    @param half 操作数是否保持输入 dtype。
    @return 编译后的闭包。
    """
    import mlx.core as mx
    cache = _compiled_mlx.cache
    if key not in cache:
        cache[key] = mx.compile(lambda *a: _run(a, _MlxOps, half))
    return cache[key]


_compiled_mlx.cache = {}


def torch_mps(*args):
    """v1 baseline torch_mps: fp32 formulation, eager."""
    return _run(args, _TorchOps, False)


def torch_half(*args):
    """v1 baseline torch_half: matmul operands stay in the input dtype."""
    return _run(args, _TorchOps, True)


BASELINES = {
    "mlx": mlx,
    "mlx_compile": mlx_compile,
    "mlx_half_compile": mlx_half_compile,
    "torch_mps": torch_mps,
    "torch_half": torch_half,
}
'''
    )


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
    if task == "flash_attention":
        reference = _flash_attention(combo)
        baselines = _baselines("flash_attention")
    elif task == "bitdelta_gemm":
        reference = _bitdelta(combo)
        baselines = _bitdelta_baselines()
    elif task == "mamba2_chunk_scan":
        reference = _mamba2(combo)
        baselines = _mamba2_baselines(combo)
    else:
        raise SystemExit(f"{task}: B3 does not migrate this task")
    files = {
        "opspec.json": json.dumps(v2_document, ensure_ascii=False, indent=2) + "\n",
        "reference.py": reference,
        "reference.json": json.dumps({"schema_version": 1, "reference_version": 1, "error_scale": {"kind": "none"}}, indent=2) + "\n",
        "baselines.json": json.dumps({"schema_version": 1, "baselines": combo["baselines"]}, ensure_ascii=False, indent=2) + "\n",
        "baselines.py": baselines,
        "display.json": json.dumps({"schema_version": 1, "description": combo["description"]}, ensure_ascii=False, indent=2) + "\n",
    }
    if check:
        return [f"would write {directory / name}" for name in files] + [
            f"would delete {directory / name}" for name in V1_MARKERS if (directory / name).exists()
        ]
    actions: list[str] = []
    candidates = B3 / "evidence" / "candidates"
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
    """@brief 迁移 B3 三题并做 v2 装配冒烟。

    @return 无。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    frozen = json.loads((B3 / "evidence" / "v1-frozen.json").read_text(encoding="utf-8"))
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
