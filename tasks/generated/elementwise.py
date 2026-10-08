"""Build a pointwise task from one expression in ``elementwise.toml``.

The expression is parsed with ``ast`` and evaluated by a fixed table of operators. There is no ``eval``. A name that is
not an input, a numeric constant, arithmetic, or a listed operator is rejected. ``gen_`` tasks do not replace the
hand-written ones; a field-by-field comparison has to pass before that is even discussed.
"""

from __future__ import annotations

import ast
import hashlib
import tomllib
from pathlib import Path

import torch

from tcenv.task import BaselineImpl
from tasks._elementwise import BLOCK_SIZE, NUM_WARPS, build_elementwise

MANIFEST = Path(__file__).with_name("elementwise.toml")
_CMP = {ast.Lt: "<", ast.Gt: ">", ast.LtE: "<=", ast.GtE: ">="}
_BIN = {ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/"}


def _relu(x):
    return torch.relu(x)


def _sigmoid(x):
    return torch.sigmoid(x)


def _silu(x):
    return x * torch.sigmoid(x)


def _mlx_relu(x):
    import mlx.core as mx

    return mx.maximum(x, 0)


def _mlx_sigmoid(x):
    import mlx.core as mx

    return mx.sigmoid(x)


def _mlx_silu(x):
    import mlx.core as mx

    return x * mx.sigmoid(x)


def _mlx_exp(x):
    import mlx.core as mx

    return mx.exp(x)


def _mlx_maximum(a, b):
    import mlx.core as mx

    return mx.maximum(a, b)


def _mlx_where(cond, a, b):
    import mlx.core as mx

    return mx.where(cond, a, b)


# 每一项同时给数值求值和三种文本。Metal 的 relu 用比较而不是 max，因为 Metal 的 max 会丢掉 NaN。
OPS = {
    "relu": {
        "torch": _relu,
        "mlx": _mlx_relu,
        "torch_mps": _relu,
        "tl": lambda a: f"tl.where({a} < 0, 0.0, {a})",
        "msl": lambda a: f"({a} < 0.0f ? 0.0f : {a})",
    },
    "sigmoid": {
        "torch": _sigmoid,
        "mlx": _mlx_sigmoid,
        "torch_mps": _sigmoid,
        "tl": lambda a: f"(1.0 / (1.0 + tl.exp(-({a}).to(tl.float32))))",
        "msl": lambda a: f"sigmoid_f({a})",
    },
    "silu": {
        "torch": _silu,
        "mlx": _mlx_silu,
        "torch_mps": _silu,
        "tl": lambda a: f"(({a}).to(tl.float32) * (1.0 / (1.0 + tl.exp(-({a}).to(tl.float32)))))",
        "msl": lambda a: f"(({a}) * sigmoid_f({a}))",
    },
    "exp": {
        "torch": torch.exp,
        "mlx": _mlx_exp,
        "torch_mps": torch.exp,
        "tl": lambda a: f"tl.exp(({a}).to(tl.float32))",
        "msl": lambda a: f"exp({a})",
    },
    "maximum": {
        "torch": torch.maximum,
        "mlx": _mlx_maximum,
        "torch_mps": torch.maximum,
        "tl": lambda a, b: f"tl.maximum({a}, {b})",
        "msl": lambda a, b: f"(({a} >= {b} || isnan({a})) ? {a} : {b})",
    },
    "where": {
        "torch": torch.where,
        "mlx": _mlx_where,
        "torch_mps": torch.where,
        "tl": lambda c, a, b: f"tl.where({c}, {a}, {b})",
        "msl": lambda c, a, b: f"({c} ? {a} : {b})",
    },
}

_ALLOWED = (
    ast.Expression, ast.Call, ast.Name, ast.Load, ast.Constant,
    ast.UnaryOp, ast.USub, ast.BinOp, ast.Add, ast.Sub, ast.Mult, ast.Div,
    ast.Compare, ast.Lt, ast.Gt, ast.LtE, ast.GtE,
)


def parse_expr(expr: str, inputs: list[str]) -> ast.Expression:
    """The expression, or ``ValueError`` when it uses anything outside the operator table."""
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as e:
        raise ValueError(f"cannot parse {expr!r}: {e}") from e
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED):
            raise ValueError(f"{expr!r} uses {type(node).__name__}, which is not allowed")
        if isinstance(node, ast.Call):
            if node.keywords or not isinstance(node.func, ast.Name) or node.func.id not in OPS:
                raise ValueError(f"{expr!r} calls something outside the operator table")
        if isinstance(node, ast.Name) and not isinstance(node.ctx, ast.Load):
            raise ValueError(f"{expr!r} assigns to {node.id}")
        if isinstance(node, ast.Constant) and (isinstance(node.value, bool) or not isinstance(node.value, (int, float))):
            raise ValueError(f"{expr!r} has a non-numeric constant")
        if isinstance(node, ast.Compare) and len(node.ops) != 1:
            raise ValueError(f"{expr!r} chains comparisons")
    _check_names(tree, inputs)
    return tree


def _check_names(tree: ast.AST, inputs: list[str]) -> None:
    """A name is either an input or the callee of an operator. Anything else is rejected."""
    callees = {id(node.func) for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)}
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and id(node) not in callees and node.id not in inputs:
            raise ValueError(f"unknown name {node.id!r}")


def evaluate(tree: ast.Expression, env: dict, backend: str):
    """Run the expression. ``backend`` is ``torch`` or ``mlx``."""
    return _eval(tree.body, env, backend)


def _eval(node, env, backend):
    if isinstance(node, ast.Name):
        if node.id not in env:
            raise ValueError(f"unknown name {node.id!r}")
        return env[node.id]
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -_eval(node.operand, env, backend)
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN:
        import operator
        fn = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv}[type(node.op)]
        return fn(_eval(node.left, env, backend), _eval(node.right, env, backend))
    if isinstance(node, ast.Compare):
        left = _eval(node.left, env, backend)
        right = _eval(node.comparators[0], env, backend)
        return {ast.Lt: left.__lt__, ast.Gt: left.__gt__, ast.LtE: left.__le__, ast.GtE: left.__ge__}[type(node.ops[0])](right)
    if isinstance(node, ast.Call):
        args = [_eval(a, env, backend) for a in node.args]
        return OPS[node.func.id][backend](*args)
    raise ValueError(f"cannot evaluate {type(node).__name__}")


def render(tree: ast.Expression, lang: str, inputs: list[str]) -> str:
    """``lang`` is ``tl`` or ``msl``. Input names are the loaded locals."""
    _check_names(tree, inputs)
    prefix = "v_" if lang == "msl" else ""
    return _render(tree.body, lang, prefix)


def _render(node, lang: str, prefix: str) -> str:
    if isinstance(node, ast.Name):
        return prefix + node.id
    if isinstance(node, ast.Constant):
        number = float(node.value)
        return f"{number}f" if lang == "msl" else repr(number)
    if isinstance(node, ast.UnaryOp):
        return f"(-{_render(node.operand, lang, prefix)})"
    if isinstance(node, ast.BinOp):
        return f"({_render(node.left, lang, prefix)} {_BIN[type(node.op)]} {_render(node.right, lang, prefix)})"
    if isinstance(node, ast.Compare):
        sym = _CMP[type(node.ops[0])]
        return f"({_render(node.left, lang, prefix)} {sym} {_render(node.comparators[0], lang, prefix)})"
    if isinstance(node, ast.Call):
        args = [_render(a, lang, prefix) for a in node.args]
        return OPS[node.func.id][lang](*args)
    raise ValueError(f"cannot render {type(node).__name__}")


def _validate(entry: dict) -> None:
    inputs = list(entry["inputs"])
    if not inputs or any(not str(n).isidentifier() for n in inputs):
        raise ValueError(f"bad inputs on {entry.get('task')!r}")
    if entry.get("domain_case") and len(inputs) != 1:
        raise ValueError(f"{entry.get('task')!r} declares domain_case with {len(inputs)} inputs")
    parse_expr(entry["expr"], inputs)


def load_manifest(path: Path | None = None) -> tuple[list[dict], str]:
    path = Path(path or MANIFEST)
    text = path.read_text(encoding="utf-8")
    doc = tomllib.loads(text)
    entries = doc.get("kernel") or []
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"{path} has no [[kernel]] entries")
    seen = set()
    for entry in entries:
        if entry["task"] in seen:
            raise ValueError(f"duplicate generated task {entry['task']!r}")
        seen.add(entry["task"])
        _validate(entry)
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]
    return entries, f"{path.as_posix().split('tasks/generated/')[-1]}#{digest}" if "tasks/generated/" in path.as_posix() else f"{path.name}#{digest}"


def _source_tag(path: Path, digest12: str) -> str:
    return f"tasks/generated/elementwise.toml#{digest12}"


def _triton(kernel: str, inputs: list[str], expr: str, n: int) -> str:
    ptrs = ", ".join(f"{name}_ptr" for name in inputs)
    loads = "\n    ".join(f"{name} = tl.load({name}_ptr + offsets, mask=mask)" for name in inputs)
    body = render(parse_expr(expr, inputs), "tl", inputs)
    return (
        f"# Triton specification: {kernel}, element type tl.float16\n"
        f"# Launch: grid = (N // BLOCK_SIZE,), num_warps = {NUM_WARPS}, N = {n}, BLOCK_SIZE = {BLOCK_SIZE}\n"
        "import triton\nimport triton.language as tl\n\n\n"
        "@triton.jit\n"
        f"def {kernel}_kernel({ptrs}, y_ptr, N: tl.constexpr, BLOCK_SIZE: tl.constexpr):\n"
        "    pid = tl.program_id(axis=0)\n"
        "    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)\n"
        "    mask = offsets < N\n"
        f"    {loads}\n"
        f"    y = {body}\n"
        "    tl.store(y_ptr + offsets, y.to(tl.float16), mask=mask)\n"
    )


_SIGMOID_MSL = """inline float sigmoid_f(float v) {
    float e = exp(-fabs(v));
    return v >= 0.0f ? 1.0f / (1.0f + e) : e / (1.0f + e);
}

"""


def _metal(kernel: str, inputs: list[str], expr: str, dtype_msl: str) -> str:
    tree = parse_expr(expr, inputs)
    body = render(tree, "msl", inputs)
    prelude = _SIGMOID_MSL if "sigmoid_f(" in body else ""
    args = ", ".join(f"device const {dtype_msl}* {name} [[buffer({i})]]" for i, name in enumerate(inputs))
    args += f", device {dtype_msl}* y [[buffer({len(inputs)})]]"
    locals_ = " ".join(f"float v_{name} = float({name}[j]);" for name in inputs)
    return (
        prelude
        + f"kernel void {kernel}_kernel({args}, uint tg [[threadgroup_position_in_grid]], uint lid [[thread_index_in_threadgroup]]) {{\n"
        + "    uint base = tg * BLOCK_SIZE;\n"
        + "    for (uint i = lid; i < BLOCK_SIZE; i += 128) {\n"
        + "        uint j = base + i;\n"
        + f"        if (j < N) {{ {locals_} y[j] = {dtype_msl}({body}); }}\n"
        + "    }\n}\n"
    )


def _baselines(tree: ast.Expression, inputs: list[str]) -> tuple[BaselineImpl, ...]:
    def torch_fn(*args):
        return evaluate(tree, dict(zip(inputs, args)), "torch")

    def mlx_fn(*args):
        return evaluate(tree, dict(zip(inputs, args)), "mlx")

    return (BaselineImpl("mlx", "mlx", mlx_fn), BaselineImpl("torch_mps", "torch", torch_fn))


def build(task_name: str, dtype: str, size: str, *, manifest: Path | None = None):
    path = Path(manifest or MANIFEST)
    text = path.read_text(encoding="utf-8")
    entries, _tag = load_manifest(path)
    digest12 = hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]
    entry = next((e for e in entries if e["task"] == task_name), None)
    if entry is None:
        known = ", ".join(e["task"] for e in entries)
        raise KeyError(f"unknown generated task {task_name!r}; known: {known}")
    inputs = list(entry["inputs"])
    tree = parse_expr(entry["expr"], inputs)
    _check_names(tree, inputs)
    from tcenv.dtypes import get_dtype

    from tasks._common import TL_DTYPES, check_dtype
    from tasks._elementwise import SIZES

    check_dtype(dtype)
    n = SIZES[size]
    # 清单文本写成 fp16。bf16 只替换题面里的元素类型，公式不变。
    triton = _triton(entry["kernel"], inputs, entry["expr"], n).replace("tl.float16", TL_DTYPES[dtype])
    naive = _metal(entry["kernel"], inputs, entry["expr"], get_dtype(dtype).msl)

    def ref_fn(*args):
        return evaluate(tree, dict(zip(inputs, args)), "torch")

    task = build_elementwise(
        entry["kernel"], dtype, size,
        input_names=tuple(inputs),
        ref_fn=ref_fn,
        baselines=_baselines(tree, inputs),
        flops_per_elem=float(entry["flops_per_elem"]),
        description=entry["description"],
        large=float(entry.get("large", 1.0e4)),
        normal_mag=float(entry.get("normal_mag", 3.0)),
        triton_source=triton,
        naive_msl=naive,
        task_name=task_name,
        domain_case=bool(entry.get("domain_case")) and dtype == "fp16",
    )
    task.meta = {**task.meta, "source": _source_tag(path, digest12)}
    return task


def mismatches(hand, generated) -> list[str]:
    """Fields that must agree. The manifest path in ``meta['source']`` is allowed to differ."""
    bad = []
    hc, gc = hand.contract, generated.contract
    for field in ("kernel_name", "grid", "threads", "constants"):
        if getattr(hc, field) != getattr(gc, field):
            bad.append(field)
    if len(hc.args) != len(gc.args):
        bad.append("args")
    else:
        for a, b in zip(hc.args, gc.args):
            if (a.name, a.index, a.role, a.dtype, a.shape) != (b.name, b.index, b.role, b.dtype, b.shape):
                bad.append(f"arg {a.name}")
    for field in ("cases", "rtol", "atol_rel", "exact", "flops", "min_bytes"):
        if getattr(hand, field) != getattr(generated, field):
            bad.append(field)
    for key in ("ieee_base", "ieee_inputs", "ieee_max", "domain_case"):
        if (hand.meta or {}).get(key) != (generated.meta or {}).get(key):
            bad.append(f"meta.{key}")
    return bad
