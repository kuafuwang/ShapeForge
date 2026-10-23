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
    """@brief torch 侧的 ReLU 基线。

    @param x 输入张量。
    @return ``torch.relu(x)``。
    """
    return torch.relu(x)


def _sigmoid(x):
    """@brief torch 侧的 sigmoid 基线。

    @param x 输入张量。
    @return ``torch.sigmoid(x)``。
    """
    return torch.sigmoid(x)


def _silu(x):
    """@brief torch 侧的 SiLU 基线。

    @param x 输入张量。
    @return ``x * sigmoid(x)``。
    """
    return x * torch.sigmoid(x)


def _mlx_relu(x):
    """@brief mlx 侧的 ReLU 基线。

    @param x mlx 数组。
    @return 把小于 0 的项截成 0 的结果。
    """
    import mlx.core as mx

    return mx.maximum(x, 0)


def _mlx_sigmoid(x):
    """@brief mlx 侧的 sigmoid 基线。

    @param x mlx 数组。
    @return ``mx.sigmoid(x)``。
    """
    import mlx.core as mx

    return mx.sigmoid(x)


def _mlx_silu(x):
    """@brief mlx 侧的 SiLU 基线。

    @param x mlx 数组。
    @return ``x * mx.sigmoid(x)``。
    """
    import mlx.core as mx

    return x * mx.sigmoid(x)


def _mlx_exp(x):
    """@brief mlx 侧的 exp 基线。

    @param x mlx 数组。
    @return ``mx.exp(x)``。
    """
    import mlx.core as mx

    return mx.exp(x)


def _mlx_maximum(a, b):
    """@brief mlx 侧的逐元素最大值基线。

    @param a 第一个 mlx 数组。
    @param b 第二个 mlx 数组。
    @return ``mx.maximum(a, b)``。
    """
    import mlx.core as mx

    return mx.maximum(a, b)


def _mlx_where(cond, a, b):
    """@brief mlx 侧的选择基线。

    @param cond 条件数组。
    @param a 条件为真时取的值。
    @param b 条件为假时取的值。
    @return ``mx.where(cond, a, b)``。
    """
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
    """@brief 校验并解析清单里的表达式。

    The expression, or ``ValueError`` when it uses anything outside the operator table.

    @param expr 清单里的表达式文本。
    @param inputs 允许作为名字出现的输入名。
    @return 解析后的 ``ast.Expression``。
    @raises ValueError 表达式解析失败、用到运算符表以外的节点，或名字不合法时抛出。
    """
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
    """@brief 检查每个名字要么是输入，要么是运算符被调用者。

    A name is either an input or the callee of an operator. Anything else is rejected.

    @param tree 已解析的语法树。
    @param inputs 允许作为名字出现的输入名。
    @raises ValueError 出现未知名字时抛出。
    """
    callees = {id(node.func) for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)}
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and id(node) not in callees and node.id not in inputs:
            raise ValueError(f"unknown name {node.id!r}")


def evaluate(tree: ast.Expression, env: dict, backend: str):
    """@brief 在一个后端上把表达式算出来。

    Run the expression. ``backend`` is ``torch`` or ``mlx``.

    @param tree 已解析的语法树。
    @param env 变量名到取值的映射。
    @param backend ``torch`` 或 ``mlx``。
    @return 计算结果。
    """
    return _eval(tree.body, env, backend)


def _eval(node, env, backend):
    """@brief 递归求值一个语法树节点。

    @param node 语法树节点。
    @param env 变量名到取值的映射。
    @param backend ``torch`` 或 ``mlx``。
    @return 该节点的结果。
    @raises ValueError 名字未知或节点类型不支持时抛出。
    """
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
    """@brief 把表达式渲染成 Triton 或 MSL 源文本。

    ``lang`` is ``tl`` or ``msl``. Input names are the loaded locals.

    @param tree 已解析的语法树。
    @param lang ``tl`` 或 ``msl``。
    @param inputs 输入名，按局部变量名渲染。
    @return 渲染出的表达式文本。
    """
    _check_names(tree, inputs)
    prefix = "v_" if lang == "msl" else ""
    return _render(tree.body, lang, prefix)


def _render(node, lang: str, prefix: str) -> str:
    """@brief 递归渲染一个语法树节点。

    @param node 语法树节点。
    @param lang ``tl`` 或 ``msl``。
    @param prefix 局部变量名前缀，MSL 下是 ``v_``。
    @return 该节点的源文本。
    @raises ValueError 节点类型不支持时抛出。
    """
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
    """@brief 检查清单的一条 ``[[kernel]]`` 条目。

    @param entry 清单里的一项。
    @raises ValueError 输入名非法、一元题以外的题声明了 ``domain_case``，或表达式不合法时抛出。
    """
    inputs = list(entry["inputs"])
    if not inputs or any(not str(n).isidentifier() for n in inputs):
        raise ValueError(f"bad inputs on {entry.get('task')!r}")
    if entry.get("domain_case") and len(inputs) != 1:
        raise ValueError(f"{entry.get('task')!r} declares domain_case with {len(inputs)} inputs")
    parse_expr(entry["expr"], inputs)


def load_manifest(path: Path | None = None) -> tuple[list[dict], str]:
    """@brief 读并校验生成式任务的清单文件。

    @param path 清单路径；为 None 时用 ``MANIFEST``。
    @return ``(条目列表, 带内容摘要的来源标记)``。
    @raises ValueError 清单里没有 ``[[kernel]]`` 条目、题目名重复或条目校验不通过时抛出。
    """
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
    """@brief 拼出清单来源标记。

    @param path 清单路径。
    @param digest12 清单文本的 sha256 前 12 位。
    @return 写进 ``meta['source']`` 的标记文本。
    """
    return f"tasks/generated/elementwise.toml#{digest12}"


def _triton(kernel: str, inputs: list[str], expr: str, n: int) -> str:
    """@brief 生成该条目的 Triton 题面正文。

    @param kernel 算子名，用来拼核名。
    @param inputs 输入名，按契约槽位顺序。
    @param expr 清单里的表达式。
    @param n 元素总数，写进头注释。
    @return 完整的 Triton 源码文本。
    """
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
    """@brief 生成该条目的朴素 Metal 正文。

    @param kernel 算子名，用来拼核名。
    @param inputs 输入名，按契约槽位顺序。
    @param expr 清单里的表达式。
    @param dtype_msl Metal 元素类型名，例如 ``half``。
    @return 完整的 Metal 源码文本。
    """
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
    """@brief 造出该条目的 mlx 与 torch 两条基线。

    @param tree 已解析的语法树。
    @param inputs 输入名，按位置绑定到参数。
    @return 基线实现元组。
    """
    def torch_fn(*args):
        return evaluate(tree, dict(zip(inputs, args)), "torch")

    def mlx_fn(*args):
        return evaluate(tree, dict(zip(inputs, args)), "mlx")

    return (BaselineImpl("mlx", "mlx", mlx_fn), BaselineImpl("torch_mps", "torch", torch_fn))


def build(task_name: str, dtype: str, size: str, *, manifest: Path | None = None):
    """@brief 按清单里的一条条目装配出一个 ``gen_`` 任务。

    @param task_name 题目名，对应清单的 ``task`` 字段。
    @param dtype 元素类型，``fp16`` 或 ``bf16``。
    @param size 尺寸档位，``paper`` 或 ``small``。
    @param manifest 清单路径；为 None 时用 ``MANIFEST``。
    @return 装配好的 ``Task``。
    @raises KeyError 清单里没有该题目名时抛出。
    """
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
    """@brief 比较生成式任务与手写任务的字段，列出不一致项。

    Fields that must agree. The manifest path in ``meta['source']`` is allowed to differ.

    @param hand 手写题目的 ``Task``。
    @param generated 生成式题目的 ``Task``。
    @return 不一致的字段名列表，完全一致时为空列表。
    """
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
