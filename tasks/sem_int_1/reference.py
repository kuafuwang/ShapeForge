"""sem_int_1: independent int64-then-wrap-int32 reference and the frozen v1 input stream.

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

_PROGRAM = json.loads(r'''{"kind": "int32-block-program", "block": 256, "rules": {"version": 1, "arithmetic": "int32_twos_complement", "mod_mask": 2147483647, "hash_shift": 15, "loop_shift": 7, "index_mask": 7, "shl_range": [0, 31], "shr_range": [0, 31], "wrap_kinds": ["bin", "sbin", "shl", "neg", "hash", "loop", "sum"], "plain_kinds": ["shr", "mod", "where", "max", "min"]}, "statements": [{"kind": "bin", "dst": "t1", "a": "yv", "op": "+", "const": 1014138929}, {"kind": "where", "dst": "t2", "a": "xv", "b": "t1", "c": "xv", "d": "idx", "op": "!="}, {"kind": "hash", "dst": "t3", "a": "xv", "const": 1935800733}, {"kind": "shl", "dst": "t4", "a": "t3", "const": 9}, {"kind": "bin", "dst": "t5", "a": "t3", "b": "idx", "op": "^"}, {"kind": "hash", "dst": "t6", "a": "t1", "const": 742385985}, {"kind": "bin", "dst": "t7", "a": "xv", "b": "t5", "op": "-"}, {"kind": "bin", "dst": "t8", "a": "t7", "b": "t2", "op": "^"}, {"kind": "bin", "dst": "t9", "a": "t6", "b": "t4", "op": "*"}, {"kind": "mod", "dst": "t10", "a": "t1", "const": 313}, {"kind": "bin", "dst": "t11", "a": "t9", "b": "t5", "op": "^"}, {"kind": "bin", "dst": "t12", "a": "t10", "b": "t8", "op": "^"}, {"kind": "bin", "dst": "t13", "a": "t6", "b": "t3", "op": "|"}, {"kind": "shr", "dst": "t14", "a": "t8", "const": 5}, {"kind": "where", "dst": "t15", "a": "t10", "b": "t9", "c": "t5", "d": "t11", "op": "=="}, {"kind": "shl", "dst": "t16", "a": "t15", "const": 10}, {"kind": "bin", "dst": "t17", "a": "t16", "b": "t12", "op": "^"}, {"kind": "mod", "dst": "t18", "a": "t15", "const": 975}, {"kind": "bin", "dst": "t19", "a": "t17", "b": "t15", "op": "*"}, {"kind": "shr", "dst": "t20", "a": "t18", "const": 10}, {"kind": "hash", "dst": "t21", "a": "t19", "const": 60075811}, {"kind": "hash", "dst": "t22", "a": "t20", "const": 1255120169}, {"kind": "bin", "dst": "t23", "a": "t6", "b": "t22", "op": "+"}, {"kind": "bin", "dst": "t24", "a": "t6", "b": "t18", "op": "+"}, {"kind": "hash", "dst": "t25", "a": "t21", "const": 149283141}, {"kind": "bin", "dst": "t26", "a": "t22", "b": "t20", "op": "*"}, {"kind": "bin", "dst": "t27", "a": "t6", "op": "^", "const": 233563962}, {"kind": "bin", "dst": "t28", "a": "t26", "b": "t22", "op": "|"}, {"kind": "bin", "dst": "t29", "a": "t3", "op": "^", "const": 957065848}, {"kind": "where", "dst": "t30", "a": "t28", "b": "t26", "c": "t29", "d": "t29", "op": "=="}, {"kind": "hash", "dst": "t31", "a": "t28", "const": 126240069}, {"kind": "neg", "dst": "t32", "a": "t27"}, {"kind": "loop", "dst": "t33", "a": "t27", "b": "t29", "const": 1597388789, "rounds": 2}, {"kind": "shr", "dst": "t34", "a": "t28", "const": 9}, {"kind": "loop", "dst": "t35", "a": "t30", "b": "t34", "const": 989673197, "rounds": 2}, {"kind": "loop", "dst": "t36", "a": "t30", "b": "t31", "const": 745028411, "rounds": 2}, {"kind": "bin", "dst": "t37", "a": "t34", "b": "t36", "op": "|"}, {"kind": "shr", "dst": "t38", "a": "t32", "const": 6}, {"kind": "where", "dst": "t39", "a": "t35", "b": "t34", "c": "t34", "d": "t19", "op": ">="}, {"kind": "bin", "dst": "t40", "a": "t38", "op": "|", "const": 738411855}, {"kind": "min", "dst": "s1", "a": "t39"}, {"kind": "bin", "dst": "t41", "a": "t36", "op": "|", "const": 1956331373}, {"kind": "mod", "dst": "t42", "a": "t38", "const": 617}, {"kind": "mod", "dst": "t43", "a": "t39", "const": 351}, {"kind": "bin", "dst": "t44", "a": "t43", "b": "t6", "op": "-"}, {"kind": "mod", "dst": "t45", "a": "t18", "const": 43}, {"kind": "shr", "dst": "t46", "a": "t41", "const": 3}, {"kind": "shl", "dst": "t47", "a": "t44", "const": 2}, {"kind": "mod", "dst": "t48", "a": "t42", "const": 978}, {"kind": "bin", "dst": "t49", "a": "t27", "b": "t4", "op": "*"}, {"kind": "hash", "dst": "t50", "a": "xv", "const": 1317861155}, {"kind": "where", "dst": "t51", "a": "t45", "b": "t48", "c": "t46", "d": "t35", "op": ">="}, {"kind": "bin", "dst": "t52", "a": "t51", "op": "^", "const": 1814664284}, {"kind": "bin", "dst": "t53", "a": "t49", "b": "t52", "op": "^"}, {"kind": "shr", "dst": "t54", "a": "t53", "const": 6}, {"kind": "bin", "dst": "t55", "a": "t44", "b": "t52", "op": "+"}, {"kind": "shr", "dst": "t56", "a": "t50", "const": 6}, {"kind": "sbin", "dst": "t57", "a": "t5", "b": "s1", "op": "-"}, {"kind": "bin", "dst": "t58", "a": "t54", "op": "&", "const": 174738088}, {"kind": "hash", "dst": "t59", "a": "t56", "const": 194322443}, {"kind": "where", "dst": "t60", "a": "t55", "b": "t56", "c": "t55", "d": "t58", "op": "<="}, {"kind": "bin", "dst": "t61", "a": "t59", "b": "t56", "op": "+"}, {"kind": "bin", "dst": "t62", "a": "t56", "b": "idx", "op": "-"}, {"kind": "where", "dst": "t63", "a": "t59", "b": "t58", "c": "t59", "d": "t62", "op": ">"}, {"kind": "bin", "dst": "t64", "a": "t38", "b": "t63", "op": "-"}, {"kind": "mod", "dst": "t65", "a": "t35", "const": 132}, {"kind": "sum", "dst": "s2", "a": "t65"}, {"kind": "bin", "dst": "t66", "a": "t65", "b": "t24", "op": "*"}, {"kind": "bin", "dst": "t67", "a": "t66", "b": "t61", "op": "^"}, {"kind": "where", "dst": "t68", "a": "t65", "b": "t66", "c": "t66", "d": "t65", "op": "<="}, {"kind": "bin", "dst": "t69", "a": "t66", "op": "^", "const": 2096614056}, {"kind": "mod", "dst": "t70", "a": "t66", "const": 596}, {"kind": "bin", "dst": "t71", "a": "t33", "op": "-", "const": 854263224}, {"kind": "mod", "dst": "t72", "a": "t66", "const": 242}, {"kind": "shr", "dst": "t73", "a": "t69", "const": 9}, {"kind": "max", "dst": "s3", "a": "t26"}, {"kind": "bin", "dst": "t74", "a": "t50", "b": "t72", "op": "^"}, {"kind": "min", "dst": "s4", "a": "t74"}, {"kind": "bin", "dst": "t75", "a": "t71", "b": "t70", "op": "+"}, {"kind": "shl", "dst": "t76", "a": "t74", "const": 6}, {"kind": "bin", "dst": "t77", "a": "t71", "b": "t63", "op": "|"}, {"kind": "mod", "dst": "t78", "a": "t74", "const": 439}, {"kind": "bin", "dst": "t79", "a": "t61", "b": "t78", "op": "&"}, {"kind": "where", "dst": "t80", "a": "t75", "b": "t74", "c": "t74", "d": "t78", "op": "<"}, {"kind": "bin", "dst": "t81", "a": "t79", "b": "t8", "op": "+"}, {"kind": "shr", "dst": "t82", "a": "t46", "const": 7}, {"kind": "shr", "dst": "t83", "a": "t54", "const": 3}, {"kind": "where", "dst": "t84", "a": "t79", "b": "t82", "c": "t78", "d": "t80", "op": ">"}, {"kind": "shl", "dst": "t85", "a": "t80", "const": 9}, {"kind": "shr", "dst": "t86", "a": "t85", "const": 10}, {"kind": "bin", "dst": "t87", "a": "t85", "op": "*", "const": 430457199}, {"kind": "bin", "dst": "t88", "a": "t55", "b": "t19", "op": "&"}, {"kind": "mod", "dst": "t89", "a": "t83", "const": 790}, {"kind": "bin", "dst": "t90", "a": "t1", "op": "^", "const": 253556680}, {"kind": "mod", "dst": "t91", "a": "t87", "const": 991}, {"kind": "where", "dst": "t92", "a": "t81", "b": "t90", "c": "t88", "d": "t91", "op": "<="}, {"kind": "neg", "dst": "t93", "a": "t87"}, {"kind": "shr", "dst": "t94", "a": "t90", "const": 9}, {"kind": "shr", "dst": "t95", "a": "t85", "const": 10}, {"kind": "shr", "dst": "t96", "a": "t93", "const": 7}, {"kind": "bin", "dst": "t97", "a": "t96", "b": "t91", "op": "+"}, {"kind": "mod", "dst": "t98", "a": "t57", "const": 618}, {"kind": "loop", "dst": "t99", "a": "t98", "b": "t37", "const": 1509448871, "rounds": 2}, {"kind": "shr", "dst": "t100", "a": "t95", "const": 6}, {"kind": "mod", "dst": "t101", "a": "t99", "const": 439}], "final": ["t98", "t99", "t100", "t101"]}''')

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
