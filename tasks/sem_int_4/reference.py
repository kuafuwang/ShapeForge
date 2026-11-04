"""sem_int_4: independent int64-then-wrap-int32 reference and the frozen v1 input stream.

The evaluator (Stmt/wrap/block_of/statements_of/_rules/_implemented_rules/evaluate) is
extracted verbatim from the registered sem_int generator; the program document of this
seed is embedded below.  A v2 reference may not import the harness, so nothing here
imports tasks/tcenv.  The arithmetic is 32-bit two's complement with wraparound after
exactly the kinds the rule table marks as wrapping.
"""

import torch
import json
from dataclasses import dataclass

M32 = 1 << 32

RULE_VERSION = 1

ARITHMETIC = "int32_twos_complement"

_INT32_MASK = (1 << 31) - 1

WRAP_KINDS = ("bin", "sbin", "shl", "neg", "hash", "loop", "sum")

PLAIN_KINDS = ("shr", "mod", "where", "max", "min")

ALL_KINDS = frozenset(WRAP_KINDS) | frozenset(PLAIN_KINDS)

@dataclass
class Stmt:
    """@brief 整数 DSL 的一条语句，字段按 kind 取用。"""

    kind: str  # bin | shl | shr | mod | where | neg | hash | loop | sum | max | min | sbin
    dst: str  # temp name (block) or scalar name (reductions)
    a: str = ""
    b: str = ""  # operand name, or "" when const is used
    c: str = ""
    d: str = ""
    op: str = ""
    const: int = 0
    rounds: int = 0

def wrap(t: torch.Tensor) -> torch.Tensor:
    """@brief 折回 int32 补码，结果仍放在 int64 里。

    Wrap an int64 tensor to int32 two's complement (kept in int64).

    @param t int64 张量。
    @return 回绕后的 int64 张量。
    """
    return ((t + (1 << 31)) % M32) - (1 << 31)

def block_of(document: dict) -> int:
    """@brief 文档的归约块大小。

    @param document ``program`` 文档。
    @return ``block``。
    """
    return int(document["block"])

def statements_of(document: dict) -> list[Stmt]:
    """@brief 从程序文档还原语句对象。

    @param document ``program`` 文档。
    @return ``Stmt`` 列表。
    """
    return [
        Stmt(
            kind=str(entry["kind"]),
            dst=str(entry["dst"]),
            a=str(entry["a"]),
            b=str(entry.get("b", "")),
            c=str(entry.get("c", "")),
            d=str(entry.get("d", "")),
            op=str(entry.get("op", "")),
            const=int(entry.get("const", 0)),
            rounds=int(entry.get("rounds", 0)),
        )
        for entry in document["statements"]
    ]

def _rules(document: dict, *, view: str) -> dict:
    """@brief 读取并校验文档的规则表（三份视图共用，越域一律 fail closed）。

    Every view goes through this gate: a rule table the view cannot implement must be refused
    rather than silently ignored, otherwise one stale view could render code that disagrees
    with the declared semantics (the reference reading the rule, the renderer not).

    @param document ``program`` 文档。
    @param view 视图名（诊断用：``triton``/``msl``/``reference``）。
    @return 规则字典。
    @raises ValueError 规则表版本/算术域/数值域/包装分区不在已实现闭集内。
    """
    rules = dict(document["rules"])
    where = f"sem_int.{view}.rules"
    if int(rules["version"]) != RULE_VERSION:
        raise ValueError(f"{where}.version: unregistered rule version {rules['version']!r} (registered: {RULE_VERSION})")
    if str(rules["arithmetic"]) != ARITHMETIC:
        raise ValueError(f"{where}.arithmetic: this view only implements {ARITHMETIC!r}, got {rules['arithmetic']!r}")
    for field in ("hash_shift", "loop_shift"):
        if not 0 <= int(rules[field]) <= 31:
            raise ValueError(f"{where}.{field}: must be within 0..31, got {rules[field]!r}")
    for kind in ("shl", "shr"):
        low, high = (int(value) for value in rules[f"{kind}_range"])
        if not 0 <= low <= high <= 31:
            raise ValueError(f"{where}.{kind}_range: must be a defined int32 shift domain, got {[low, high]}")
    if not 0 <= int(rules["mod_mask"]) <= _INT32_MASK:
        raise ValueError(f"{where}.mod_mask: must be a non-negative int32 mask, got {rules['mod_mask']!r}")
    if int(rules["index_mask"]) < 0:
        raise ValueError(f"{where}.index_mask: must be non-negative, got {rules['index_mask']!r}")
    wrap_kinds = set(rules["wrap_kinds"])
    plain_kinds = set(rules["plain_kinds"])
    if wrap_kinds & plain_kinds:
        raise ValueError(f"{where}: a kind cannot be both wrapping and plain: {sorted(wrap_kinds & plain_kinds)}")
    if wrap_kinds | plain_kinds != set(ALL_KINDS):
        raise ValueError(
            f"{where}: the wrap/plain partition must cover every kind; "
            f"unknown={sorted((wrap_kinds | plain_kinds) - set(ALL_KINDS))} "
            f"missing={sorted(set(ALL_KINDS) - (wrap_kinds | plain_kinds))}"
        )
    return rules

def _implemented_rules(document: dict, *, view: str) -> dict:
    """@brief 取本视图能实现的规则表：wrap 归属必须是登记集合，否则拒绝。

    All three views share one limit, and it is not a rendering detail: a ``loop`` statement wraps
    the product of *every* round internally (that is the registered int32 arithmetic, identical in
    Triton and MSL), so a document that declares ``loop`` plain cannot be honoured by the reference
    either — an int64 evaluation would silently keep the registered behaviour for that kind.  A
    document that moves a kind between ``wrap_kinds`` and ``plain_kinds`` is therefore refused by
    ``triton``, ``msl`` and ``reference`` alike, instead of one view pretending to implement a
    partition it cannot express.

    @param document ``program`` 文档。
    @param view 视图名（``triton``/``msl``/``reference``）。
    @return 规则字典。
    @raises ValueError 声明分区不是登记分区。
    """
    rules = _rules(document, view=view)
    wrap_kinds = set(rules["wrap_kinds"])
    if wrap_kinds != set(WRAP_KINDS):
        raise ValueError(
            f"sem_int.{view}.rules.wrap_kinds: this view implements exactly the registered "
            f"wrapping set {sorted(WRAP_KINDS)} (``loop`` wraps every round internally), "
            f"got {sorted(wrap_kinds)}"
        )
    return rules

def evaluate(document: dict, x: torch.Tensor, y: torch.Tensor, k: int) -> torch.Tensor:
    """@brief 视图二：标准答案（由文档的语句与规则表逐步求值）。

    @param document ``program`` 文档。
    @param x 输入 x，展平成 P*block 个元素。
    @param y 输入 y，形状同 x。
    @param k 运行时标量参数 K。
    @return int32 结果，形状同 x。
    """
    rules = _implemented_rules(document, view="reference")
    block = block_of(document)
    hash_shift = int(rules["hash_shift"])
    loop_shift = int(rules["loop_shift"])
    index_mask = int(rules["index_mask"])
    mod_mask = int(rules["mod_mask"])
    wrap_kinds = set(rules["wrap_kinds"])
    P = x.numel() // block
    env: dict[str, torch.Tensor] = {
        "xv": x.reshape(P, block).to(torch.int64),
        "yv": y.reshape(P, block).to(torch.int64),
        "idx": torch.arange(P * block, dtype=torch.int64).reshape(P, block),
        "K": torch.full((P, 1), k, dtype=torch.int64),
    }
    env["idx"] = wrap(env["idx"])
    cmp = {"<": torch.lt, ">": torch.gt, "<=": torch.le, ">=": torch.ge, "==": torch.eq, "!=": torch.ne}

    def apply_wrap(kind: str, value: torch.Tensor) -> torch.Tensor:
        return wrap(value) if kind in wrap_kinds else value

    for statement in statements_of(document):
        name = statement.dst
        argv = env.__getitem__
        if statement.kind in ("bin", "sbin"):
            a = argv(statement.a)
            b = argv(statement.b) if statement.b else torch.tensor(statement.const, dtype=torch.int64)
            if statement.op == "+":
                r = a + b
            elif statement.op == "-":
                r = a - b
            elif statement.op == "*":
                r = a * b  # |a|,|b| < 2^31 so the product fits int64 exactly
            elif statement.op == "^":
                r = a ^ b
            elif statement.op == "&":
                r = a & b
            else:
                r = a | b
            env[name] = apply_wrap(statement.kind, r)
        elif statement.kind == "shl":
            env[name] = apply_wrap("shl", argv(statement.a) << statement.const)
        elif statement.kind == "shr":
            env[name] = apply_wrap("shr", argv(statement.a) >> statement.const)
        elif statement.kind == "mod":
            env[name] = apply_wrap("mod", (argv(statement.a) & mod_mask) % statement.const)
        elif statement.kind == "where":
            env[name] = apply_wrap(
                "where",
                torch.where(
                    cmp[statement.op](argv(statement.a), argv(statement.b)), argv(statement.c), argv(statement.d)
                ),
            )
        elif statement.kind == "neg":
            env[name] = apply_wrap("neg", -argv(statement.a))
        elif statement.kind == "hash":
            a = argv(statement.a)
            env[name] = apply_wrap("hash", (a ^ (a >> hash_shift)) * statement.const)
        elif statement.kind == "loop":
            acc = argv(statement.a)
            for _ in range(statement.rounds):
                # Registered int32 arithmetic: the product of *each* round wraps, which is what
                # both renderers get from the target's int32 arithmetic.  This is why a document
                # declaring ``loop`` plain is refused above rather than evaluated without it.
                acc = wrap(wrap(acc * statement.const) + argv(statement.b)) ^ (acc >> loop_shift)
            env[name] = apply_wrap("loop", acc)
        elif statement.kind == "sum":
            env[name] = apply_wrap("sum", argv(statement.a).sum(dim=1, keepdim=True))
        elif statement.kind == "max":
            env[name] = apply_wrap("max", argv(statement.a).max(dim=1, keepdim=True).values)
        elif statement.kind == "min":
            env[name] = apply_wrap("min", argv(statement.a).min(dim=1, keepdim=True).values)
    res = env[document["final"][0]]
    for n in document["final"][1:]:
        res = res ^ env[n]
    # The final projection lands in an int32 output buffer (Triton stores ``out_ptr + offs``,
    # MSL casts to ``int``), so it wraps regardless of the per-kind partition.
    res = wrap(res + (env["idx"] & index_mask))
    return res.reshape(-1).to(torch.int32)

_PROGRAM = json.loads(r'''{"kind": "int32-block-program", "block": 256, "rules": {"version": 1, "arithmetic": "int32_twos_complement", "mod_mask": 2147483647, "hash_shift": 15, "loop_shift": 7, "index_mask": 7, "shl_range": [0, 31], "shr_range": [0, 31], "wrap_kinds": ["bin", "sbin", "shl", "neg", "hash", "loop", "sum"], "plain_kinds": ["shr", "mod", "where", "max", "min"]}, "statements": [{"kind": "bin", "dst": "t1", "a": "xv", "b": "yv", "op": "+"}, {"kind": "mod", "dst": "t2", "a": "xv", "const": 230}, {"kind": "mod", "dst": "t3", "a": "yv", "const": 849}, {"kind": "bin", "dst": "t4", "a": "t3", "b": "idx", "op": "-"}, {"kind": "bin", "dst": "t5", "a": "t4", "op": "*", "const": 1822202737}, {"kind": "bin", "dst": "t6", "a": "t4", "b": "t3", "op": "^"}, {"kind": "mod", "dst": "t7", "a": "t4", "const": 289}, {"kind": "bin", "dst": "t8", "a": "t4", "b": "t6", "op": "&"}, {"kind": "bin", "dst": "t9", "a": "t4", "b": "t5", "op": "^"}, {"kind": "hash", "dst": "t10", "a": "t4", "const": 993575711}, {"kind": "where", "dst": "t11", "a": "t9", "b": "t10", "c": "t10", "d": "t8", "op": ">"}, {"kind": "where", "dst": "t12", "a": "t7", "b": "t11", "c": "t11", "d": "t8", "op": "<"}, {"kind": "loop", "dst": "t13", "a": "t2", "b": "t7", "const": 1851343613, "rounds": 2}, {"kind": "bin", "dst": "t14", "a": "t10", "op": "-", "const": 1497422917}, {"kind": "bin", "dst": "t15", "a": "xv", "b": "t11", "op": "*"}, {"kind": "bin", "dst": "t16", "a": "t11", "b": "idx", "op": "^"}, {"kind": "bin", "dst": "t17", "a": "t13", "b": "t12", "op": "-"}, {"kind": "shl", "dst": "t18", "a": "t12", "const": 8}, {"kind": "bin", "dst": "t19", "a": "t16", "b": "t11", "op": "*"}, {"kind": "sum", "dst": "s1", "a": "t14"}, {"kind": "neg", "dst": "t20", "a": "t15"}, {"kind": "where", "dst": "t21", "a": "t19", "b": "t15", "c": "t19", "d": "t18", "op": "!="}, {"kind": "bin", "dst": "t22", "a": "t17", "b": "t19", "op": "^"}, {"kind": "bin", "dst": "t23", "a": "t20", "b": "t13", "op": "^"}, {"kind": "bin", "dst": "t24", "a": "t19", "op": "+", "const": 1658535117}, {"kind": "bin", "dst": "t25", "a": "t21", "op": "*", "const": 1214067433}, {"kind": "bin", "dst": "t26", "a": "t18", "b": "t20", "op": "&"}, {"kind": "neg", "dst": "t27", "a": "t21"}, {"kind": "shr", "dst": "t28", "a": "t16", "const": 3}, {"kind": "shl", "dst": "t29", "a": "t28", "const": 6}, {"kind": "hash", "dst": "t30", "a": "t29", "const": 1720987551}, {"kind": "where", "dst": "t31", "a": "t28", "b": "t2", "c": "t15", "d": "t10", "op": "<"}, {"kind": "shr", "dst": "t32", "a": "t28", "const": 11}, {"kind": "loop", "dst": "t33", "a": "t27", "b": "t32", "const": 866995261, "rounds": 3}, {"kind": "bin", "dst": "t34", "a": "t32", "op": "+", "const": 1882105321}, {"kind": "bin", "dst": "t35", "a": "t12", "b": "t30", "op": "^"}, {"kind": "bin", "dst": "t36", "a": "t28", "op": "&", "const": 1965451871}, {"kind": "mod", "dst": "t37", "a": "t31", "const": 715}, {"kind": "where", "dst": "t38", "a": "t34", "b": "t33", "c": "t32", "d": "t31", "op": "<"}, {"kind": "bin", "dst": "t39", "a": "t37", "b": "t34", "op": "*"}, {"kind": "hash", "dst": "t40", "a": "t35", "const": 1629320907}, {"kind": "shr", "dst": "t41", "a": "t4", "const": 2}, {"kind": "bin", "dst": "t42", "a": "t39", "b": "t41", "op": "&"}, {"kind": "hash", "dst": "t43", "a": "t38", "const": 1367811659}, {"kind": "shl", "dst": "t44", "a": "t25", "const": 2}, {"kind": "hash", "dst": "t45", "a": "t40", "const": 378710769}, {"kind": "where", "dst": "t46", "a": "t42", "b": "t40", "c": "xv", "d": "t40", "op": "=="}, {"kind": "bin", "dst": "t47", "a": "t45", "b": "t43", "op": "|"}, {"kind": "shl", "dst": "t48", "a": "t38", "const": 1}, {"kind": "where", "dst": "t49", "a": "t45", "b": "t46", "c": "t34", "d": "t47", "op": ">="}, {"kind": "where", "dst": "t50", "a": "t45", "b": "t48", "c": "t49", "d": "t47", "op": "<"}, {"kind": "where", "dst": "t51", "a": "t49", "b": "t47", "c": "t47", "d": "t45", "op": "<"}, {"kind": "shr", "dst": "t52", "a": "t9", "const": 6}, {"kind": "where", "dst": "t53", "a": "t49", "b": "t36", "c": "t47", "d": "t49", "op": "<"}, {"kind": "bin", "dst": "t54", "a": "t48", "b": "t5", "op": "&"}, {"kind": "bin", "dst": "t55", "a": "t37", "b": "t40", "op": "^"}, {"kind": "shl", "dst": "t56", "a": "t23", "const": 10}, {"kind": "bin", "dst": "t57", "a": "t52", "b": "t32", "op": "^"}, {"kind": "where", "dst": "t58", "a": "t55", "b": "t30", "c": "t54", "d": "t55", "op": "=="}, {"kind": "bin", "dst": "t59", "a": "t57", "b": "t55", "op": "-"}, {"kind": "shr", "dst": "t60", "a": "t54", "const": 3}, {"kind": "mod", "dst": "t61", "a": "t60", "const": 945}, {"kind": "shl", "dst": "t62", "a": "t9", "const": 5}, {"kind": "shr", "dst": "t63", "a": "t60", "const": 10}, {"kind": "hash", "dst": "t64", "a": "t59", "const": 148345737}, {"kind": "bin", "dst": "t65", "a": "t31", "op": "+", "const": 39787026}, {"kind": "mod", "dst": "t66", "a": "t60", "const": 213}, {"kind": "bin", "dst": "t67", "a": "t66", "op": "*", "const": 620744157}, {"kind": "where", "dst": "t68", "a": "t63", "b": "t51", "c": "t8", "d": "t67", "op": "<"}, {"kind": "neg", "dst": "t69", "a": "t67"}, {"kind": "shr", "dst": "t70", "a": "t66", "const": 8}, {"kind": "mod", "dst": "t71", "a": "t66", "const": 898}, {"kind": "min", "dst": "s2", "a": "t71"}, {"kind": "shl", "dst": "t72", "a": "t67", "const": 7}, {"kind": "bin", "dst": "t73", "a": "t69", "op": "+", "const": 2061787069}, {"kind": "hash", "dst": "t74", "a": "t68", "const": 1822330679}, {"kind": "where", "dst": "t75", "a": "t70", "b": "t72", "c": "t66", "d": "t72", "op": "!="}, {"kind": "where", "dst": "t76", "a": "t46", "b": "t71", "c": "t73", "d": "t73", "op": "=="}, {"kind": "max", "dst": "s3", "a": "t76"}, {"kind": "loop", "dst": "t77", "a": "t76", "b": "t75", "const": 1842463329, "rounds": 4}, {"kind": "where", "dst": "t78", "a": "t59", "b": "t74", "c": "t8", "d": "t38", "op": "<"}, {"kind": "where", "dst": "t79", "a": "t77", "b": "t76", "c": "t43", "d": "t20", "op": "!="}, {"kind": "bin", "dst": "t80", "a": "t75", "b": "t79", "op": "&"}, {"kind": "bin", "dst": "t81", "a": "t75", "b": "t79", "op": "&"}, {"kind": "loop", "dst": "t82", "a": "t77", "b": "t80", "const": 2049691547, "rounds": 2}, {"kind": "sum", "dst": "s4", "a": "t82"}, {"kind": "bin", "dst": "t83", "a": "t42", "op": "*", "const": 1752168489}, {"kind": "bin", "dst": "t84", "a": "t78", "b": "t81", "op": "^"}, {"kind": "bin", "dst": "t85", "a": "t63", "b": "t84", "op": "^"}, {"kind": "bin", "dst": "t86", "a": "t81", "b": "t83", "op": "-"}, {"kind": "max", "dst": "s5", "a": "t84"}, {"kind": "sbin", "dst": "t87", "a": "t50", "b": "s5", "op": "^"}, {"kind": "bin", "dst": "t88", "a": "t86", "op": "&", "const": 616643087}, {"kind": "where", "dst": "t89", "a": "t87", "b": "t9", "c": "t19", "d": "t88", "op": ">"}, {"kind": "sbin", "dst": "t90", "a": "t85", "b": "s3", "op": "+"}, {"kind": "where", "dst": "t91", "a": "t89", "b": "t19", "c": "t89", "d": "t12", "op": "<="}, {"kind": "bin", "dst": "t92", "a": "t91", "b": "t87", "op": "&"}, {"kind": "min", "dst": "s6", "a": "t88"}, {"kind": "where", "dst": "t93", "a": "t91", "b": "t89", "c": "t75", "d": "t18", "op": ">"}, {"kind": "neg", "dst": "t94", "a": "t88"}, {"kind": "sbin", "dst": "t95", "a": "t89", "b": "s5", "op": "*"}, {"kind": "bin", "dst": "t96", "a": "t94", "b": "t93", "op": "-"}, {"kind": "neg", "dst": "t97", "a": "t95"}, {"kind": "sum", "dst": "s7", "a": "t93"}, {"kind": "bin", "dst": "t98", "a": "t96", "b": "t95", "op": "*"}], "final": ["t95", "t96", "t97", "t98"]}''')

_SEED_MASK = 2147483647
_LO, _HI = -2147483648, 2147483648
_SMALL_LO, _SMALL_HI = -50, 50
_EDGE_VALUES = torch.tensor([0, 1, -1, 2147483647, -2147483648, 2, -2, 255, -256, 1431655765, -1431655766], dtype=torch.int64)


def gen_inputs(case, seed, instance):
    """The v1 case/seed stream (random/small/edge, int64 values)."""
    generator = torch.Generator().manual_seed(int(seed) & _SEED_MASK)
    n = int(instance.symbols["N"])
    if case == "random":
        make = lambda: torch.randint(_LO, _HI, (n,), generator=generator, dtype=torch.int64)
    elif case == "small":
        make = lambda: torch.randint(_SMALL_LO, _SMALL_HI, (n,), generator=generator, dtype=torch.int64)
    else:  # edge values
        make = lambda: _EDGE_VALUES[torch.randint(0, len(_EDGE_VALUES), (n,), generator=generator)]
    return {"x": make(), "y": make()}


def reference(inputs, attrs):
    """Evaluate this seed's program document (int64, then wrap to int32)."""
    return {"out": evaluate(_PROGRAM, inputs["x"], inputs["y"], int(attrs["SCALAR_K"]))}
