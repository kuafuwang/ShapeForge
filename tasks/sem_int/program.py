"""The registered integer-program generator, its rules and its three document-driven views.

The *program document* (``program`` key of the OpSpec) is semantic data and the single
authority for the generated task:

* it carries the closed statement list **and** the numeric rule table (wrap behaviour,
  hash/loop shift amounts, the mod mask, the index mask, the shift domains).  The three
  views below read those values from the document, so editing a rule changes the digest,
  the rendered Triton prompt and the reference together (a validator checks the domains);
* the per-block reduction size ``block`` is semantic (a reduction spans one block), so it
  is part of the document and drives all three views; the thread mapping
  (``threads``/``num_warps``/``per_thread``) is a launch fact that comes from the registered
  LoweringProfile and is only used to render the naive MSL.

Vocabulary: negative modulo, division, and shifts outside ``0..31`` are never generated and
are refused by the OpSpec validator.  Arithmetic is 32-bit two's complement with wraparound
after exactly the kinds the rule table marks as wrapping.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

import torch

#: Default block size of the generated family (the OpSpec document carries its value; the
#: registered Metal profile must declare the same ``BLOCK`` constant).
BLOCK = 256
#: Registered runtime scalar argument ``K`` (the OpSpec attr ``SCALAR_K`` owns the value).
SCALAR_K = 12345
M32 = 1 << 32

#: Registered numeric rule table.  Every value is consumed by the views or enforced by the
#: validator, and all of it enters ``opspec_digest`` through the document.
RULE_VERSION = 1
#: Registered arithmetic domain (the views only implement this one).
ARITHMETIC = "int32_twos_complement"
_INT32_MASK = (1 << 31) - 1
#: Kinds whose result is wrapped back into int32 after the operation.
WRAP_KINDS = ("bin", "sbin", "shl", "neg", "hash", "loop", "sum")
#: Kinds that are already in range (arithmetic shift of a wrapped value, masked modulo,
#: comparison/select, max/min of in-range values).
PLAIN_KINDS = ("shr", "mod", "where", "max", "min")
ALL_KINDS = frozenset(WRAP_KINDS) | frozenset(PLAIN_KINDS)


def rules_document() -> dict:
    """@brief 物化登记数值规则表（进入程序文档与摘要）。

    @return 规则字典。
    """
    return {
        "version": RULE_VERSION,
        "arithmetic": ARITHMETIC,
        "mod_mask": _INT32_MASK,
        "hash_shift": 15,
        "loop_shift": 7,
        "index_mask": 7,
        "shl_range": [0, 31],
        "shr_range": [0, 31],
        "wrap_kinds": list(WRAP_KINDS),
        "plain_kinds": list(PLAIN_KINDS),
    }


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


# ------------------------------------------------------------------------------------- generator


def generate_program(seed: int, n_stmts: int = 105) -> tuple[list[Stmt], list[str]]:
    """@brief 按种子抽出一份约 n_stmts 条语句的整数程序（固定种子可确定性复现）。

    sum、max、min 至少各一条。生成器避开负取模、除法，以及移位量大于等于 32 的语句。

    @param seed 生成用的种子，同时决定题目名里的数字。
    @param n_stmts 抽取的语句条数。
    @return ``(statements, final)``。
    """
    rng = random.Random(seed)
    stmts: list[Stmt] = []
    blocks = ["xv", "yv", "idx"]
    scalars = ["K"]
    counter = {"t": 0, "s": 0}

    def pick(pool, recent=6, avoid=()):
        for _ in range(20):  # avoid degenerate expressions such as t ^ t or t >= t
            v = rng.choice(pool[-recent:]) if rng.random() < 0.7 else rng.choice(pool)
            if v not in avoid:
                return v
        return v

    def new_t():
        counter["t"] += 1
        return f"t{counter['t']}"

    def new_s():
        counter["s"] += 1
        return f"s{counter['s']}"

    def k31():
        return rng.randrange(1, (1 << 31) - 1)

    for _ in range(n_stmts):
        r = rng.random()
        if r < 0.30:
            op = rng.choice(["+", "-", "*", "^", "&", "|"])
            if rng.random() < 0.35:
                st = Stmt("bin", new_t(), a=pick(blocks), b="", op=op, const=(k31() | 1) if op == "*" else k31())
            else:
                a_ = pick(blocks)
                st = Stmt("bin", new_t(), a=a_, b=pick(blocks, avoid=(a_,)), op=op)
        elif r < 0.38:
            st = Stmt("shl", new_t(), a=pick(blocks), const=rng.randrange(1, 13))
        elif r < 0.50:
            st = Stmt("shr", new_t(), a=pick(blocks), const=rng.randrange(1, 13))
        elif r < 0.58:
            st = Stmt("mod", new_t(), a=pick(blocks), const=rng.randrange(3, 1001))
        elif r < 0.72:
            a_ = pick(blocks)
            st = Stmt(
                "where",
                new_t(),
                a=a_,
                b=pick(blocks, avoid=(a_,)),
                c=pick(blocks),
                d=pick(blocks),
                op=rng.choice(["<", ">", "<=", ">=", "==", "!="]),
            )
        elif r < 0.76:
            st = Stmt("neg", new_t(), a=pick(blocks))
        elif r < 0.82:
            st = Stmt("hash", new_t(), a=pick(blocks), const=k31() | 1)
        elif r < 0.88:
            a_ = pick(blocks)
            st = Stmt("loop", new_t(), a=a_, b=pick(blocks, avoid=(a_,)), const=k31() | 1, rounds=rng.randrange(2, 5))
        elif r < 0.92 and len(blocks) > 8:
            st = Stmt(rng.choice(["sum", "max", "min"]), new_s(), a=pick(blocks))
            scalars.append(st.dst)
            stmts.append(st)
            continue
        elif r < 0.97 and len(scalars) > 1:
            st = Stmt("sbin", new_t(), a=pick(blocks), b=pick(scalars, 3), op=rng.choice(["+", "^", "-", "*"]))
        else:
            a_ = pick(blocks)
            st = Stmt("bin", new_t(), a=a_, b=pick(blocks, avoid=(a_,)), op="^")
        stmts.append(st)
        blocks.append(st.dst)
    # make sure at least one of each reduction exists
    have = {s.kind for s in stmts}
    for kind in ("sum", "max", "min"):
        if kind not in have:
            s = Stmt(kind, new_s(), a=blocks[-1])
            stmts.append(s)
            scalars.append(s.dst)
            t = Stmt("sbin", new_t(), a=blocks[-1], b=s.dst, op="^")
            stmts.append(t)
            blocks.append(t.dst)
    return stmts, blocks[-4:]


def program_document(seed: int, n_stmts: int = 105, block: int = BLOCK) -> dict:
    """@brief 由种子确定性地物化程序文档（OpSpec 的 ``program`` 键）。

    @param seed 生成种子（名字里的整数）。
    @param n_stmts 语句条数（登记在生成器参数里）。
    @param block 每个 threadgroup 处理的元素数（也是归约域）。
    @return ``{kind, block, rules, statements, final}``。
    """
    statements, final = generate_program(seed, n_stmts)
    entries = []
    for statement in statements:
        entry = {"kind": statement.kind, "dst": statement.dst, "a": statement.a}
        for name in ("b", "c", "d", "op"):
            value = getattr(statement, name)
            if value:
                entry[name] = value
        if statement.const:
            entry["const"] = statement.const
        if statement.rounds:
            entry["rounds"] = statement.rounds
        entries.append(entry)
    return {
        "kind": "int32-block-program",
        "block": block,
        "rules": rules_document(),
        "statements": entries,
        "final": list(final),
    }


# ---------------------------------------------------------------------------------- document views


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


def _hex(value: int) -> str:
    """@brief 把整数常量渲染成十六进制字面量。

    @param value 常量值。
    @return 带 ``0x`` 前缀的文本。
    """
    return f"{int(value):#x}"


def triton_source(document: dict, n: int, num_warps: int, scalar_k: int) -> str:
    """@brief 视图一：题面（由文档的语句与规则表渲染）。

    @param document ``program`` 文档。
    @param n 元素总数。
    @param num_warps 启动的 warp 数（登记 profile 的启动事实）。
    @param scalar_k 运行时标量参数值（OpSpec attr ``SCALAR_K``）。
    @return 完整的 Triton 源码文本。
    """
    rules = _implemented_rules(document, view="triton")
    block = block_of(document)
    if n % block:
        raise ValueError(f"sem_int: N={n} must be a multiple of the document block {block}")
    hash_shift = int(rules["hash_shift"])
    loop_shift = int(rules["loop_shift"])
    index_mask = int(rules["index_mask"])
    mod_mask = int(rules["mod_mask"])
    lines = [
        "# Triton specification: a synthetic integer kernel (no recognizable algorithm). All arithmetic is int32 with wraparound.",
        f"# Launch: grid = (N // BLOCK,), num_warps = {num_warps}, N = {n}, BLOCK = {block}, K = {scalar_k} (runtime scalar argument)",
        "import triton",
        "import triton.language as tl",
        "",
        "",
        "@triton.jit",
        "def sem_int_kernel(x_ptr, y_ptr, K, out_ptr, N: tl.constexpr, BLOCK: tl.constexpr):",
        "    pid = tl.program_id(axis=0)",
        "    offs = pid * BLOCK + tl.arange(0, BLOCK)",
        "    offs = tl.max_contiguous(tl.multiple_of(offs, BLOCK), BLOCK)",
        "    xv = tl.load(x_ptr + offs)",
        "    yv = tl.load(y_ptr + offs)",
        "    idx = offs",
    ]
    for statement in statements_of(document):
        lines.extend(
            "    " + line
            for line in _triton_stmt(
                statement, block=block, hash_shift=hash_shift, loop_shift=loop_shift, mod_mask=mod_mask
            )
        )
    fin = " ^ ".join(str(name) for name in document["final"])
    lines += [f"    res = ({fin}) + (idx & {index_mask})", "    tl.store(out_ptr + offs, res)"]
    return "\n".join(lines) + "\n"


def _triton_stmt(statement: Stmt, *, block: int, hash_shift: int, loop_shift: int, mod_mask: int) -> list[str]:
    """@brief 把一条语句渲染成 Triton 源码的若干行。

    @param statement 语句。
    @param block 归约块大小（未直接使用，保留给逐 kind 的渲染规则）。
    @param hash_shift ``hash`` 的移位量（来自规则表）。
    @param loop_shift ``loop`` 的移位量（来自规则表）。
    @param mod_mask ``mod`` 的取模掩码（来自规则表，不得写死）。
    @return 该语句的源码行列表。
    """
    del block
    d = statement.dst
    if statement.kind == "bin":
        rhs = statement.b if statement.b else _hex(statement.const)
        return [f"{d} = {statement.a} {statement.op} {rhs}"]
    if statement.kind == "sbin":
        return [f"{d} = {statement.a} {statement.op} {statement.b}"]
    if statement.kind == "shl":
        return [f"{d} = {statement.a} << {statement.const}"]
    if statement.kind == "shr":
        return [f"{d} = {statement.a} >> {statement.const}"]
    if statement.kind == "mod":
        return [f"{d} = (({statement.a}) & 0x{mod_mask:X}) % {statement.const}"]
    if statement.kind == "where":
        return [f"{d} = tl.where({statement.a} {statement.op} {statement.b}, {statement.c}, {statement.d})"]
    if statement.kind == "neg":
        return [f"{d} = -{statement.a}"]
    if statement.kind == "hash":
        return [f"{d} = ({statement.a} ^ ({statement.a} >> {hash_shift})) * {_hex(statement.const)}"]
    if statement.kind == "loop":
        return [
            f"{d} = {statement.a}",
            f"for r in tl.static_range({statement.rounds}):",
            f"    {d} = ({d} * {_hex(statement.const)} + {statement.b}) ^ ({d} >> {loop_shift})",
        ]
    red = {"sum": "tl.sum", "max": "tl.max", "min": "tl.min"}[statement.kind]
    return [f"{d} = {red}({statement.a}, axis=0)"]


def msl(document: dict, *, threads: int, num_warps: int, per_thread: int) -> str:
    """@brief 视图三：朴素 MSL（由文档的语句、规则表与登记启动参数渲染）。

    @param document ``program`` 文档。
    @param threads 每个 threadgroup 的线程数（登记 profile 的启动事实）。
    @param num_warps warp 数（登记 profile 的启动事实）。
    @param per_thread 每个线程负责的元素数（``block // threads``）。
    @return 完整的 Metal 源码文本。
    """
    rules = _implemented_rules(document, view="msl")
    block = block_of(document)
    if block != threads * per_thread:
        raise ValueError(f"sem_int: block {block} != threads {threads} * per_thread {per_thread}")
    hash_shift = int(rules["hash_shift"])
    loop_shift = int(rules["loop_shift"])
    index_mask = int(rules["index_mask"])
    mod_mask = int(rules["mod_mask"])
    stmts = statements_of(document)
    names_b = {"xv", "yv", "idx"} | {s.dst for s in stmts if s.kind not in ("sum", "max", "min")}
    lines = ["inline uint sh(int v, uint k) { return uint(v >> k); }", ""]
    lines += [
        "kernel void sem_int_kernel(device const int* x [[buffer(0)]], device const int* y [[buffer(1)]],",
        "                           constant int& K [[buffer(2)]], device int* out [[buffer(3)]],",
        "                           uint pid [[threadgroup_position_in_grid]], uint lid [[thread_index_in_threadgroup]],",
        "                           uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {",
        f"    const uint E = {per_thread};",
    ]
    for name in sorted(names_b):
        lines.append(f"    uint {name}[E];")
    lines.append("    uint Ku = uint(K);")
    scalar_names = [s.dst for s in stmts if s.kind in ("sum", "max", "min")]
    for name in scalar_names:
        lines.append(f"    uint {name};")
    lines += [
        f"    threadgroup uint pu[{num_warps}];",
        f"    threadgroup int pi[{num_warps}];",
        "    for (uint e = 0; e < E; ++e) {",
        f"        uint o = pid * {block} + lid + e * {threads};",
        "        xv[e] = uint(x[o]); yv[e] = uint(y[o]); idx[e] = o;",
        "    }",
    ]
    for statement in stmts:
        d = statement.dst
        if statement.kind in ("sum", "max", "min"):
            if statement.kind == "sum":
                lines += [
                    f"    {{ uint p = 0; for (uint e = 0; e < E; ++e) p += {statement.a}[e];",
                    "      p = simd_sum(p); if (lane == 0) pu[sg] = p; threadgroup_barrier(mem_flags::mem_threadgroup);",
                    f"      {d} = 0; for (uint w = 0; w < {num_warps}; ++w) {d} += pu[w]; threadgroup_barrier(mem_flags::mem_threadgroup); }}",
                ]
            else:
                red, init, simd = ("max", "INT_MIN", "simd_max") if statement.kind == "max" else ("min", "INT_MAX", "simd_min")
                lines += [
                    f"    {{ int p = {init}; for (uint e = 0; e < E; ++e) p = {red}(p, int({statement.a}[e]));",
                    f"      p = {simd}(p); if (lane == 0) pi[sg] = p; threadgroup_barrier(mem_flags::mem_threadgroup);",
                    f"      int r = pi[0]; for (uint w = 1; w < {num_warps}; ++w) r = {red}(r, pi[w]); {d} = uint(r); threadgroup_barrier(mem_flags::mem_threadgroup); }}",
                ]
            continue
        body = []
        if statement.kind in ("bin", "sbin"):
            a = f"{statement.a}[e]"
            if statement.kind == "sbin":
                b = statement.b if statement.b in scalar_names else ("Ku" if statement.b == "K" else f"{statement.b}[e]")
            else:
                b = f"{statement.b}[e]" if statement.b else f"{statement.const}u"
            body.append(f"{d}[e] = {a} {statement.op} {b};")
        elif statement.kind == "shl":
            body.append(f"{d}[e] = {statement.a}[e] << {statement.const};")
        elif statement.kind == "shr":
            body.append(f"{d}[e] = sh(int({statement.a}[e]), {statement.const});")
        elif statement.kind == "mod":
            body.append(f"{d}[e] = ({statement.a}[e] & 0x{mod_mask:X}u) % {statement.const}u;")
        elif statement.kind == "where":
            body.append(
                f"{d}[e] = (int({statement.a}[e]) {statement.op} int({statement.b}[e])) ? {statement.c}[e] : {statement.d}[e];"
            )
        elif statement.kind == "neg":
            body.append(f"{d}[e] = 0u - {statement.a}[e];")
        elif statement.kind == "hash":
            body.append(f"{d}[e] = ({statement.a}[e] ^ sh(int({statement.a}[e]), {hash_shift})) * {statement.const}u;")
        elif statement.kind == "loop":
            body.append(f"uint acc = {statement.a}[e];")
            body.append(
                f"for (uint r = 0; r < {statement.rounds}; ++r) acc = (acc * {statement.const}u + {statement.b}[e]) ^ sh(int(acc), {loop_shift});"
            )
            body.append(f"{d}[e] = acc;")
        lines.append("    for (uint e = 0; e < E; ++e) { " + " ".join(body) + " }")
    fin = " ^ ".join(f"{name}[e]" for name in document["final"])
    lines += [
        "    for (uint e = 0; e < E; ++e) {",
        f"        out[pid * {block} + lid + e * {threads}] = int(({fin}) + (idx[e] & {index_mask}u));",
        "    }",
        "}",
    ]
    return "\n".join(lines) + "\n"


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


__all__ = [
    "ALL_KINDS",
    "BLOCK",
    "M32",
    "PLAIN_KINDS",
    "RULE_VERSION",
    "SCALAR_K",
    "Stmt",
    "WRAP_KINDS",
    "block_of",
    "evaluate",
    "generate_program",
    "msl",
    "program_document",
    "rules_document",
    "statements_of",
    "triton_source",
    "wrap",
]
