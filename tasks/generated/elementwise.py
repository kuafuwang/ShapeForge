"""The registered generator behind the ``gen_*`` family: closed expression grammar, no TOML.

Each ``gen_<op>`` task owns ``tasks/generated/<op>/opspec.json``.  The expression *is* the
document's ``formula`` and is parsed by the fixed operator table below (no ``eval``, no free
text: an input name, a numeric constant, arithmetic, a comparison or a listed operator, and
nothing else).  The three views -- Triton prompt, naive Metal, reference/baselines -- are
rendered from that document, and the launch facts (kernel name, ``BLOCK_SIZE``/``N``, warp
count, thread count, dtype spellings) come from the registered LoweringProfile and the track.
``elementwise.toml`` and the manifest-driven ``build()`` are gone (R2).

"""

from __future__ import annotations

import ast

import torch

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


# ---------------------------------------------------------------------------------------------
# Registered generator (R2): the same renderers, driven by an OpSpec document instead of the
# retired ``elementwise.toml``.  The expression is the document's ``formula`` and is parsed by
# the closed grammar above; launch facts come from the registered LoweringProfile.

#: Metal helper for the ``sigmoid`` operator (shared by ``sigmoid``/``silu``/``swiglu``).
_SIGMOID_MSL = """inline float sigmoid_f(float v) {
    float e = exp(-fabs(v));
    return v >= 0.0f ? 1.0f / (1.0f + e) : e / (1.0f + e);
}

"""

#: Generator identity (recorded in ``track.json``'s ``spec_text``/``known_good`` blocks).
GENERATOR_ID = "generated.elementwise"
GENERATOR_VERSION = 1
#: Version of the expression grammar/operator table above (enters the generator params).
GRAMMAR_VERSION = 1


def document_expression(document: dict) -> tuple[ast.Expression, list[str]]:
    """@brief 从 OpSpec 文档取表达式与输入名并校验闭集文法。

    @param document OpSpec 文档。
    @return ``(语法树, 输入名列表)``。
    @raises ValueError 表达式超出算子表，或输入不是声明的输入张量。
    """
    inputs = [str(name) for name, tensor in document["tensors"].items() if tensor["role"] == "input"]
    if not inputs:
        raise ValueError("generated.elementwise: the document declares no input tensor")
    return parse_expr(str(document["formula"]), inputs), inputs


def triton_source(
    document: dict, *, kernel_name: str, n: int, num_warps: int, block_size: int, tl_dtype: str
) -> str:
    """@brief 视图一：由 OpSpec 文档渲染的 Triton 题面。

    @param document OpSpec 文档。
    @param kernel_name 契约核名（``<stem>_kernel``，来自登记 profile）。
    @param n 元素总数。
    @param num_warps 登记的 warp 数。
    @param block_size 登记的 ``BLOCK_SIZE`` 常量。
    @param tl_dtype Triton 元素类型拼写（dtype 拼写表）。
    @return 完整题面文本。
    """
    tree, inputs = document_expression(document)
    stem = str(kernel_name).removesuffix("_kernel")
    ptrs = ", ".join(f"{name}_ptr" for name in inputs)
    loads = "\n    ".join(f"{name} = tl.load({name}_ptr + offsets, mask=mask)" for name in inputs)
    body = render(tree, "tl", inputs)
    return (
        f"# Triton specification: {stem}, element type {tl_dtype}\n"
        f"# Launch: grid = (N // BLOCK_SIZE,), num_warps = {num_warps}, N = {n}, BLOCK_SIZE = {block_size}\n"
        "import triton\nimport triton.language as tl\n\n\n"
        "@triton.jit\n"
        f"def {kernel_name}({ptrs}, y_ptr, N: tl.constexpr, BLOCK_SIZE: tl.constexpr):\n"
        "    pid = tl.program_id(axis=0)\n"
        "    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)\n"
        "    mask = offsets < N\n"
        f"    {loads}\n"
        f"    y = {body}\n"
        f"    tl.store(y_ptr + offsets, y.to({tl_dtype}), mask=mask)\n"
    )


def _constant_names(constants) -> tuple[str, str]:
    """@brief 从契约常量里取 ``N``/``BLOCK_SIZE`` 的名字（不写死标识符）。

    @param constants 契约的 ``constants``（``(name, msl_type, value)`` 序列）。
    @return ``(n_name, block_name)``。
    @raises ValueError 常量不是登记的两项 ``uint``。
    """
    names = [str(name) for name, msl_type, _value in constants]
    if len(names) != 2 or names[0] != "N" or names[1] != "BLOCK_SIZE":
        raise ValueError(f"generated.elementwise: expected the registered N/BLOCK_SIZE constants, got {names}")
    for name, msl_type, _value in constants:
        if str(msl_type) != "uint":
            raise ValueError(f"generated.elementwise: constant {name} must be uint, got {msl_type}")
    return names[0], names[1]


def msl(document: dict, *, kernel_name: str, constants, dtype_msl: str, threads: int) -> str:
    """@brief 视图二：由 OpSpec 文档渲染的朴素 Metal。

    ``N``/``BLOCK_SIZE`` stay as identifiers: the contract header declares them as
    ``constant constexpr`` when the source is assembled, so the text names the same constants
    the contract binds instead of copying their values.

    @param document OpSpec 文档。
    @param kernel_name 契约核名（``<stem>_kernel``）。
    @param constants 契约的 ``constants``（名字/类型/值）。
    @param dtype_msl Metal 元素类型拼写。
    @param threads 每个 threadgroup 的线程数（登记 profile）。
    @return 完整 known-good 文本。
    """
    tree, inputs = document_expression(document)
    n_name, block_name = _constant_names(constants)
    body = render(tree, "msl", inputs)
    prelude = _SIGMOID_MSL if "sigmoid_f(" in body else ""
    args = ", ".join(f"device const {dtype_msl}* {name} [[buffer({i})]]" for i, name in enumerate(inputs))
    args += f", device {dtype_msl}* y [[buffer({len(inputs)})]]"
    locals_ = " ".join(f"float v_{name} = float({name}[j]);" for name in inputs)
    return (
        prelude
        + f"kernel void {kernel_name}({args}, uint tg [[threadgroup_position_in_grid]], uint lid [[thread_index_in_threadgroup]]) {{\n"
        + f"    uint base = tg * {block_name};\n"
        + f"    for (uint i = lid; i < {block_name}; i += {threads}) {{\n"
        + "        uint j = base + i;\n"
        + f"        if (j < {n_name}) {{ {locals_} y[j] = {dtype_msl}({body}); }}\n"
        + "    }\n}\n"
    )


def reference_for(document: dict):
    """@brief 视图三：由 OpSpec 文档渲染的标准答案（float64 求值，单次舍入）。

    @param document OpSpec 文档。
    @return ``(*inputs) -> 输出张量``。
    """
    tree, inputs = document_expression(document)

    def reference(*args):
        return evaluate(tree, dict(zip(inputs, args)), "torch")

    reference.input_names = tuple(inputs)
    return reference


def baseline_for(document: dict, backend: str):
    """@brief 由 OpSpec 文档渲染的基线（``torch``/``mlx`` 走同一算子表）。

    @param document OpSpec 文档。
    @param backend ``torch`` 或 ``mlx``。
    @return ``(*inputs) -> 输出``。
    @raises ValueError backend 未登记。
    """
    if backend not in ("torch", "mlx"):
        raise ValueError(f"generated.elementwise: unregistered baseline backend {backend!r}")
    tree, inputs = document_expression(document)

    def baseline(*args):
        return evaluate(tree, dict(zip(inputs, args)), backend)

    baseline.input_names = tuple(inputs)
    return baseline


def build_generated(task_dir: str, dtype: str, size: str):
    """@brief 装配一道生成式逐元素题：OpSpec 文档 → 登记生成器 → 登记 profile → ``Task``。

    @param task_dir ``tasks/`` 下的目录名（如 ``generated/relu``）。
    @param dtype 元素类型，``fp16`` 或 ``bf16``。
    @param size 尺寸档位，``paper`` 或 ``small``。
    @return 装配好的 ``Task``。
    @raises ValueError track 登记与文档不符（表达式、基线后端或启动事实）。
    """
    import torch

    from tcenv.lowering import derive_specialization
    from tcenv.opspec.model import instantiate, load_opspec
    from tcenv.track import REPO_ROOT, load_track
    from tasks._assembly import GeneratedText, build_task
    from tasks._pointwise import gen_inputs_from_params

    track = load_track(REPO_ROOT / "tasks" / task_dir / "track.json")
    spec = load_opspec(track.opspec_path)
    instance = instantiate(spec, dtype, size)
    specialization = derive_specialization(
        instance, backend_id=track.backend_id, profile_name=track.lowering_profile
    )
    document = dict(spec.document)
    tree, names = document_expression(document)
    contract = specialization.contract
    n = int(instance.symbols["N"])
    num_warps = int(specialization.backend_params["NUM_WARPS"])
    block_size = next(int(value) for name, _type, value in contract.constants if name == "BLOCK_SIZE")
    spelling = track.dtype_spelling[dtype]
    source = triton_source(
        document,
        kernel_name=contract.kernel_name,
        n=n,
        num_warps=num_warps,
        block_size=block_size,
        tl_dtype=str(spelling["TL"]),
    )
    known_good = msl(
        document,
        kernel_name=contract.kernel_name,
        constants=contract.constants,
        dtype_msl=str(spelling["T"]),
        threads=int(contract.threads[0]),
    )

    ref = reference_for(document)
    ref_params = dict(track.implementations["reference"]["params"])
    if ref_params.get("expression") != document["formula"] or ref_params.get("accumulate") != "float64":
        raise ValueError(f"{task_dir}: the registered reference does not match the document expression {document['formula']!r}")

    def reference(inp: dict) -> dict:
        return {"y": ref(*[inp[name].to(torch.float64) for name in names])}

    baseline_fns = {}
    for entry in track.baselines:
        params = dict(entry["params"])
        backend = str(params["backend"])
        if params.get("expression") != document["formula"]:
            raise ValueError(f"{task_dir}: baseline {entry['name']!r} registers a different expression")
        baseline_fns[str(entry["name"])] = baseline_for(document, backend)

    shapes = {name: instance.tensor(name).shape for name in names}
    return build_task(
        task_dir,
        dtype,
        size,
        implementations={
            "gen_inputs": gen_inputs_from_params(dict(track.implementations["gen_inputs"]["params"]), names, shapes, dtype),
            "reference": reference,
            "error_scale": None,
        },
        baseline_fns=baseline_fns,
        derived=(track, instance, specialization),
        generated=GeneratedText(spec_text=source, known_good=known_good),
    )
