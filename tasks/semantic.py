"""Semantic-recovery tasks (paper Section 4.1): does the compiler follow the *code*, not names, comments or familiar patterns?

  * ``sem_int_<seed>``     random ~150-line integer Triton kernels that implement no recognizable algorithm
                           (bitwise/modular/hash-like arithmetic, where, tl.sum/tl.max/tl.min block reductions,
                           tl.static_range loops, tl.multiple_of / tl.max_contiguous hints). Integers avoid any
                           floating-point discrepancy, so results are compared bit-exactly.
  * ``sem_gemm_neg``       a matmul whose accumulation was changed to ``acc -= a @ b`` (the kernel is still called gemm)
  * ``sem_relu_gelu_comment`` a kernel named/commented as GELU that actually computes ReLU
  * ``sem_gelu_named_relu``   a kernel named/commented as ReLU that actually computes GELU
  * ``sem_attention_odd_scale`` FlashAttention whose scale is ``SM_SCALE + log2(e)`` (a sum) instead of the usual product

The integer programs come from a tiny DSL that renders three consistent views of the same program: the Triton source
(the specification), a torch evaluation (the exact reference) and a naive MSL translation (used to validate the harness).
All arithmetic is 32-bit two's complement with wraparound; operations whose Triton/C semantics could differ from torch's
(negative modulo, division, shifts by >= 32, constants outside int32) are never generated.
"""

# 语义恢复：名字和注释故意写错，看模型跟的是标签还是正文里的计算。get_task 见到 sem_ 前缀就调用 build。
# sem_gemm_neg 把 gemm 题面的 acc = tl.dot(a, b, acc) 改成 acc -= tl.dot(a, b)，参考用 epilogue_ref 对 C 取负，baselines 为空。
# sem_relu_gelu_comment 的注释写成 GELU，正文是 tl.where(x > 0, x, 0.0)。NaN 不大于 0，写成 0。标准答案按这句比较，不用 torch.relu。sem_gelu_named_relu 的注释写成 ReLU，正文是 _gelu_ref 的 tanh 近似。
# 上面两题复用逐元素骨架，kernel_name 是 relu_kernel，baselines 为空。模型不能改的是那份元素级契约（核名、缓冲 x/y、grid、threads、N、BLOCK_SIZE）。
# sem_attention_odd_scale 走 flash_attention.build_attention(odd_scale=True)：契约常量 SM_SCALE 仍是 1/sqrt(D)，题面把 qk 标度写成 SM_SCALE + log2(e)。
# sem_int_<种子> 由 IntProgram 从同一份语句生成 Triton 题面、evaluate 标准答案和朴素 MSL；exact=True，按 int32 逐位比较，baselines 为空。
from __future__ import annotations

import random
import re
from dataclasses import dataclass, field

import torch

from tcenv.contract import AbiArg, Contract
from tcenv.task import Task

INT_PREFIX = "sem_int_"
# 四个固定错位题。sem_int_<种子> 不在这里，由 _build_int 按名字里的种子现生成。
FIXED_NAMES = ("sem_gemm_neg", "sem_relu_gelu_comment", "sem_gelu_named_relu", "sem_attention_odd_scale")
BLOCK = 256
NUM_WARPS = 2
THREADS = NUM_WARPS * 32
PER_THREAD = BLOCK // THREADS  # 4 elements per thread in the naive MSL
SCALAR_K = 12345
M32 = 1 << 32


def wrap(t: torch.Tensor) -> torch.Tensor:
    # 折回 int32 补码，结果仍放在 int64 里。整数题多数运算做完都会过它。
    """Wrap an int64 tensor to int32 two's complement (kept in int64)."""
    return ((t + (1 << 31)) % M32) - (1 << 31)


# ------------------------------------------------------------------------------------------------------------
# integer program DSL
# ------------------------------------------------------------------------------------------------------------


@dataclass
class Stmt:
    kind: str  # bin | shl | shr | mod | where | neg | hash | loop | sum | max | min | sbin
    dst: str  # temp name (block) or scalar name (reductions)
    a: str = ""
    b: str = ""  # operand name, or "" when const is used
    c: str = ""
    d: str = ""
    op: str = ""
    const: int = 0
    rounds: int = 0


@dataclass
class IntProgram:
    # 一份程序三份视图：triton_source 是题面，evaluate 是标准答案，msl 是用来核对评测环境的朴素 Metal。
    seed: int
    stmts: list[Stmt] = field(default_factory=list)
    final: list[str] = field(default_factory=list)

    # -- generation ---------------------------------------------------------------------------------
    @staticmethod
    def generate(seed: int, n_stmts: int = 105) -> "IntProgram":
        # 按种子抽约 n_stmts 条语句。sum、max、min 至少各一条。生成器避开负取模、除法，以及移位量大于等于 32 的语句。
        rng = random.Random(seed)
        prog = IntProgram(seed)
        blocks = ["xv", "yv", "idx"]  # block-valued names available as operands
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
                st = Stmt("where", new_t(), a=a_, b=pick(blocks, avoid=(a_,)), c=pick(blocks), d=pick(blocks), op=rng.choice(["<", ">", "<=", ">=", "==", "!="]))
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
                prog.stmts.append(st)
                continue
            elif r < 0.97 and len(scalars) > 1:
                st = Stmt("sbin", new_t(), a=pick(blocks), b=pick(scalars, 3), op=rng.choice(["+", "^", "-", "*"]))
            else:
                a_ = pick(blocks)
                st = Stmt("bin", new_t(), a=a_, b=pick(blocks, avoid=(a_,)), op="^")
            prog.stmts.append(st)
            blocks.append(st.dst)
        # make sure at least one of each reduction exists
        have = {s.kind for s in prog.stmts}
        for kind in ("sum", "max", "min"):
            if kind not in have:
                s = Stmt(kind, new_s(), a=blocks[-1])
                prog.stmts.append(s)
                scalars.append(s.dst)
                t = Stmt("sbin", new_t(), a=blocks[-1], b=s.dst, op="^")
                prog.stmts.append(t)
                blocks.append(t.dst)
        prog.final = blocks[-4:]
        return prog

    # -- views ---------------------------------------------------------------------------------------
    def triton_source(self, n: int) -> str:
        # 视图一，发给模型的 Triton 正文。结果是最后四个块变量相异或，再加上 idx & 7。
        L = [
            "# Triton specification: a synthetic integer kernel (no recognizable algorithm). All arithmetic is int32 with wraparound.",
            f"# Launch: grid = (N // BLOCK,), num_warps = {NUM_WARPS}, N = {n}, BLOCK = {BLOCK}, K = {SCALAR_K} (runtime scalar argument)",
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
        for s in self.stmts:
            L.extend("    " + line for line in self._triton_stmt(s))
        fin = " ^ ".join(self.final)
        L += [f"    res = ({fin}) + (idx & 7)", "    tl.store(out_ptr + offs, res)"]
        return "\n".join(L) + "\n"

    @staticmethod
    def _c(v: int) -> str:
        return f"{v:#x}"

    def _triton_stmt(self, s: Stmt) -> list[str]:
        d = s.dst
        if s.kind == "bin":
            rhs = s.b if s.b else self._c(s.const)
            return [f"{d} = {s.a} {s.op} {rhs}"]
        if s.kind == "sbin":
            return [f"{d} = {s.a} {s.op} {s.b}"]
        if s.kind == "shl":
            return [f"{d} = {s.a} << {s.const}"]
        if s.kind == "shr":
            return [f"{d} = {s.a} >> {s.const}"]
        if s.kind == "mod":
            return [f"{d} = (({s.a}) & 0x7FFFFFFF) % {s.const}"]
        if s.kind == "where":
            return [f"{d} = tl.where({s.a} {s.op} {s.b}, {s.c}, {s.d})"]
        if s.kind == "neg":
            return [f"{d} = -{s.a}"]
        if s.kind == "hash":
            return [f"{d} = ({s.a} ^ ({s.a} >> 15)) * {self._c(s.const)}"]
        if s.kind == "loop":
            return [f"{d} = {s.a}", f"for r in tl.static_range({s.rounds}):", f"    {d} = ({d} * {self._c(s.const)} + {s.b}) ^ ({d} >> 7)"]
        red = {"sum": "tl.sum", "max": "tl.max", "min": "tl.min"}[s.kind]
        return [f"{d} = {red}({s.a}, axis=0)"]

    def evaluate(self, x: torch.Tensor, y: torch.Tensor, k: int = SCALAR_K) -> torch.Tensor:
        # 视图二，标准答案，和 Triton 语句逐步对应。右移沿用已折回的值做算术右移；bin、sbin、左移、取负、hash、循环和 sum 的结果再经 wrap。
        """Exact reference: int64 arithmetic wrapped to int32 after every operation. x, y: [P*BLOCK] int."""
        P = x.numel() // BLOCK
        env: dict[str, torch.Tensor] = {
            "xv": x.reshape(P, BLOCK).to(torch.int64),
            "yv": y.reshape(P, BLOCK).to(torch.int64),
            "idx": torch.arange(P * BLOCK, dtype=torch.int64).reshape(P, BLOCK),
            "K": torch.full((P, 1), k, dtype=torch.int64),
        }
        env["idx"] = wrap(env["idx"])
        cmp = {"<": torch.lt, ">": torch.gt, "<=": torch.le, ">=": torch.ge, "==": torch.eq, "!=": torch.ne}
        for s in self.stmts:
            g = lambda n: env[n]
            if s.kind in ("bin", "sbin"):
                a = g(s.a)
                b = g(s.b) if s.b else torch.tensor(s.const, dtype=torch.int64)
                if s.op == "+":
                    r = a + b
                elif s.op == "-":
                    r = a - b
                elif s.op == "*":
                    r = a * b  # |a|,|b| < 2^31 so the product fits int64 exactly
                elif s.op == "^":
                    r = a ^ b
                elif s.op == "&":
                    r = a & b
                else:
                    r = a | b
                env[s.dst] = wrap(r)
            elif s.kind == "shl":
                env[s.dst] = wrap(g(s.a) << s.const)
            elif s.kind == "shr":
                env[s.dst] = g(s.a) >> s.const  # arithmetic shift of the wrapped value
            elif s.kind == "mod":
                env[s.dst] = (g(s.a) & 0x7FFFFFFF) % s.const
            elif s.kind == "where":
                env[s.dst] = torch.where(cmp[s.op](g(s.a), g(s.b)), g(s.c), g(s.d))
            elif s.kind == "neg":
                env[s.dst] = wrap(-g(s.a))
            elif s.kind == "hash":
                a = g(s.a)
                env[s.dst] = wrap((a ^ (a >> 15)) * s.const)
            elif s.kind == "loop":
                acc = g(s.a)
                for _ in range(s.rounds):
                    acc = wrap(wrap(acc * s.const) + g(s.b)) ^ (acc >> 7)
                env[s.dst] = wrap(acc)
            elif s.kind == "sum":
                env[s.dst] = wrap(g(s.a).sum(dim=1, keepdim=True))
            elif s.kind == "max":
                env[s.dst] = g(s.a).max(dim=1, keepdim=True).values
            elif s.kind == "min":
                env[s.dst] = g(s.a).min(dim=1, keepdim=True).values
        res = env[self.final[0]]
        for n in self.final[1:]:
            res = res ^ env[n]
        res = wrap(res + (env["idx"] & 7))
        return res.reshape(-1).to(torch.int32)

    def msl(self) -> str:
        # 视图三，朴素 MSL，用 uint 做回绕。归约先在 simdgroup 内做，再合并 NUM_WARPS 个槽。评测拿它检查环境。
        """Naive MSL translation (uint arithmetic for guaranteed wraparound)."""
        names_b = {"xv", "yv", "idx"} | {s.dst for s in self.stmts if s.kind not in ("sum", "max", "min")}
        L = ["inline uint sh(int v, uint k) { return uint(v >> k); }", ""]
        L += [
            "kernel void sem_int_kernel(device const int* x [[buffer(0)]], device const int* y [[buffer(1)]],",
            "                           constant int& K [[buffer(2)]], device int* out [[buffer(3)]],",
            "                           uint pid [[threadgroup_position_in_grid]], uint lid [[thread_index_in_threadgroup]],",
            "                           uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {",
            f"    const uint E = {PER_THREAD};",
        ]
        for n in sorted(names_b):
            L.append(f"    uint {n}[E];")
        L.append("    uint Ku = uint(K);")
        sc = [s.dst for s in self.stmts if s.kind in ("sum", "max", "min")]
        for n in sc:
            L.append(f"    uint {n};")
        L += [
            f"    threadgroup uint pu[{NUM_WARPS}];",
            f"    threadgroup int pi[{NUM_WARPS}];",
            "    for (uint e = 0; e < E; ++e) {",
            f"        uint o = pid * {BLOCK} + lid + e * {THREADS};",
            "        xv[e] = uint(x[o]); yv[e] = uint(y[o]); idx[e] = o;",
            "    }",
        ]
        cmpop = lambda op: op
        for s in self.stmts:
            d = s.dst
            if s.kind in ("sum", "max", "min"):
                if s.kind == "sum":
                    L += [f"    {{ uint p = 0; for (uint e = 0; e < E; ++e) p += {s.a}[e];", "      p = simd_sum(p); if (lane == 0) pu[sg] = p; threadgroup_barrier(mem_flags::mem_threadgroup);",
                          f"      {d} = 0; for (uint w = 0; w < {NUM_WARPS}; ++w) {d} += pu[w]; threadgroup_barrier(mem_flags::mem_threadgroup); }}"]
                else:
                    red, init, simd = ("max", "INT_MIN", "simd_max") if s.kind == "max" else ("min", "INT_MAX", "simd_min")
                    L += [f"    {{ int p = {init}; for (uint e = 0; e < E; ++e) p = {red}(p, int({s.a}[e]));", f"      p = {simd}(p); if (lane == 0) pi[sg] = p; threadgroup_barrier(mem_flags::mem_threadgroup);",
                          f"      int r = pi[0]; for (uint w = 1; w < {NUM_WARPS}; ++w) r = {red}(r, pi[w]); {d} = uint(r); threadgroup_barrier(mem_flags::mem_threadgroup); }}"]
                continue
            body = []
            if s.kind in ("bin", "sbin"):
                a = f"{s.a}[e]"
                if s.kind == "sbin":
                    b = s.b if s.b in sc else ("Ku" if s.b == "K" else f"{s.b}[e]")
                else:
                    b = f"{s.b}[e]" if s.b else f"{s.const}u"
                body.append(f"{d}[e] = {a} {s.op} {b};")
            elif s.kind == "shl":
                body.append(f"{d}[e] = {s.a}[e] << {s.const};")
            elif s.kind == "shr":
                body.append(f"{d}[e] = sh(int({s.a}[e]), {s.const});")
            elif s.kind == "mod":
                body.append(f"{d}[e] = ({s.a}[e] & 0x7FFFFFFFu) % {s.const}u;")
            elif s.kind == "where":
                body.append(f"{d}[e] = (int({s.a}[e]) {s.op} int({s.b}[e])) ? {s.c}[e] : {s.d}[e];")
            elif s.kind == "neg":
                body.append(f"{d}[e] = 0u - {s.a}[e];")
            elif s.kind == "hash":
                body.append(f"{d}[e] = ({s.a}[e] ^ sh(int({s.a}[e]), 15)) * {s.const}u;")
            elif s.kind == "loop":
                body.append(f"uint acc = {s.a}[e];")
                body.append(f"for (uint r = 0; r < {s.rounds}; ++r) acc = (acc * {s.const}u + {s.b}[e]) ^ sh(int(acc), 7);")
                body.append(f"{d}[e] = acc;")
            L.append("    for (uint e = 0; e < E; ++e) { " + " ".join(body) + " }")
        fin = " ^ ".join(f"{n}[e]" for n in self.final)
        L += ["    for (uint e = 0; e < E; ++e) {", f"        out[pid * {BLOCK} + lid + e * {THREADS}] = int(({fin}) + (idx[e] & 7u));", "    }", "}"]
        return "\n".join(L) + "\n"


# ------------------------------------------------------------------------------------------------------------
# task builders
# ------------------------------------------------------------------------------------------------------------

INT_N = {"paper": 1 << 16, "small": 1 << 12}


def _build_int(name: str, size: str) -> Task:
    # 整数题的契约：核名 sem_int_kernel，grid=(n/BLOCK, 1, 1)，threads=(THREADS, 1, 1)，缓冲 x、y、标量 K、输出 out。模型不能改这些。
    seed = int(name[len(INT_PREFIX) :])
    prog = IntProgram.generate(seed)
    n = INT_N[size]
    contract = Contract(
        kernel_name="sem_int_kernel",
        grid=(n // BLOCK, 1, 1),
        threads=(THREADS, 1, 1),
        args=(
            AbiArg("x", 0, "input", "i32", (n,)),
            AbiArg("y", 1, "input", "i32", (n,)),
            AbiArg("K", 2, "scalar", "i32", value=SCALAR_K, doc="runtime scalar argument"),
            AbiArg("out", 3, "output", "i32", (n,)),
        ),
        constants=(("N", "uint", n), ("BLOCK", "uint", BLOCK)),
    )

    def gen_inputs(case: str, s: int) -> dict[str, torch.Tensor]:
        g = torch.Generator().manual_seed(s & 0x7FFFFFFF)
        if case == "random":
            mk = lambda: torch.randint(-(1 << 31), 1 << 31, (n,), generator=g, dtype=torch.int64)
        elif case == "small":
            mk = lambda: torch.randint(-50, 50, (n,), generator=g, dtype=torch.int64)
        else:  # edge values
            vals = torch.tensor([0, 1, -1, (1 << 31) - 1, -(1 << 31), 2, -2, 255, -256, 0x55555555, -0x55555556], dtype=torch.int64)
            mk = lambda: vals[torch.randint(0, len(vals), (n,), generator=g)]
        return {"x": mk(), "y": mk()}

    def reference(inp: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {"out": prog.evaluate(inp["x"], inp["y"], SCALAR_K)}

    return Task(
        name=name, dtype="i32", size=size, contract=contract,
        triton_source=prog.triton_source(n),
        description="A synthetic integer kernel with no recognizable algorithm. Implement exactly what the code computes (32-bit two's-complement wraparound, arithmetic right shifts, per-block reductions).",
        cases=("random", "small", "edge"),
        gen_inputs=gen_inputs, reference=reference, baselines=(), flops=0.0, min_bytes=float(3 * n * 4),
        rtol=0.0, atol_rel=0.0, exact=True, naive_msl=prog.msl(), meta={"program_lines": len(prog.triton_source(n).splitlines())},
    )


def _relu_as_written(x: torch.Tensor) -> torch.Tensor:
    """``x > 0 ? x : 0``. NaN and negative zero become +0. ``torch.relu`` keeps both, so it is not this specification."""
    return torch.where(x > 0, x, torch.zeros_like(x))


def _gelu_ref(x: torch.Tensor) -> torch.Tensor:
    # sem_gelu_named_relu 的标准答案。系数 0.7978845608028654 和 0.044715 与下面题面正文里的常数相同。
    return 0.5 * x * (1.0 + torch.tanh(0.7978845608028654 * (x + 0.044715 * x**3)))


# 题面字符串：头注释和 docstring 写成 GELU，函数名是 relu_kernel，真正执行的是 y = tl.where(x > 0, x, 0.0)。
RELU_COMMENT_GELU = '''# Triton specification: GELU activation (tanh approximation), element type {{TL}}
# Launch: grid = (N // BLOCK_SIZE,), num_warps = {{NUM_WARPS}}, N = {{N}}, BLOCK_SIZE = {{BLOCK_SIZE}}
import triton
import triton.language as tl


@triton.jit
def relu_kernel(x_ptr, y_ptr, N: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    """GELU: y = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))"""
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    x = tl.load(x_ptr + offsets, mask=mask)
    # apply the GELU nonlinearity
    y = tl.where(x > 0, x, 0.0)
    tl.store(y_ptr + offsets, y.to({{TL}}), mask=mask)
'''

# 题面字符串：头注释和 docstring 写成 ReLU，函数名仍是 relu_kernel，正文是 tanh 近似 GELU。
GELU_NAMED_RELU = '''# Triton specification: ReLU activation, element type {{TL}}
# Launch: grid = (N // BLOCK_SIZE,), num_warps = {{NUM_WARPS}}, N = {{N}}, BLOCK_SIZE = {{BLOCK_SIZE}}
import triton
import triton.language as tl


@triton.jit
def relu_kernel(x_ptr, y_ptr, N: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    """ReLU: y = max(x, 0)"""
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    x = tl.load(x_ptr + offsets, mask=mask).to(tl.float32)
    # rectified linear unit
    t = 0.7978845608028654 * (x + 0.044715 * x * x * x)
    e = tl.exp(-2.0 * tl.abs(t))
    th = tl.where(t >= 0, (1.0 - e) / (1.0 + e), (e - 1.0) / (1.0 + e))
    y = 0.5 * x * (1.0 + th)
    tl.store(y_ptr + offsets, y.to({{TL}}), mask=mask)
'''

# 朴素 Metal。sem_relu_gelu_comment 用 v > 0，NaN 变成 0，与题面一致，不借用会保留 NaN 的 relu/naive.metal。
RELU_BODY_NAIVE = """kernel void relu_kernel(device const {{T}}* x [[buffer(0)]],
                        device {{T}}* y [[buffer(1)]],
                        uint tg [[threadgroup_position_in_grid]],
                        uint lid [[thread_index_in_threadgroup]]) {
    uint base = tg * BLOCK_SIZE;
    for (uint i = lid; i < BLOCK_SIZE; i += 128) {
        uint j = base + i;
        if (j < N) {
            float v = float(x[j]);
            y[j] = {{T}}(v > 0.0f ? v : 0.0f);
        }
    }
}
"""


# 朴素 Metal：按真正的 tanh 近似 GELU 写，给 sem_gelu_named_relu 核对环境。
GELU_NAIVE = """kernel void relu_kernel(device const {{T}}* x [[buffer(0)]],
                        device {{T}}* y [[buffer(1)]],
                        uint tg [[threadgroup_position_in_grid]],
                        uint lid [[thread_index_in_threadgroup]]) {
    uint base = tg * BLOCK_SIZE;
    for (uint i = lid; i < BLOCK_SIZE; i += 128) {
        uint j = base + i;
        if (j < N) {
            float v = float(x[j]);
            float t = clamp(0.7978845608028654f * (v + 0.044715f * v * v * v), -20.0f, 20.0f);
            float e = exp(-2.0f * fabs(t));
            float th = t >= 0.0f ? (1.0f - e) / (1.0f + e) : (e - 1.0f) / (1.0f + e);
            y[j] = {{T}}(0.5f * v * (1.0f + th));
        }
    }
}
"""


def build(name: str, dtype: str, size: str) -> Task:
    # sem_ 的总装。整数题走 _build_int；两个名字错位题换掉题面字符串；注意力走 odd_scale；gemm 只改题面和朴素 Metal 的累加，contract 用 gemm 的那一份。
    from ._common import check_dtype, load_text
    from ._elementwise import BLOCK_SIZE, NUM_WARPS, SIZES, build_elementwise

    if name.startswith(INT_PREFIX):
        return _build_int(name, size)
    check_dtype(dtype)
    if name in ("sem_relu_gelu_comment", "sem_gelu_named_relu"):
        n = SIZES[size]
        subs = {"N": str(n), "BLOCK_SIZE": str(BLOCK_SIZE), "NUM_WARPS": str(NUM_WARPS)}
        from ._common import TL_DTYPES
        from tcenv.dtypes import get_dtype

        def fill(t: str) -> str:
            t = t.replace("{{TL}}", TL_DTYPES[dtype]).replace("{{T}}", get_dtype(dtype).msl)
            for k, v in subs.items():
                t = t.replace("{{" + k + "}}", v)
            return t

        relu_like = name == "sem_relu_gelu_comment"
        naive = fill(RELU_BODY_NAIVE) if relu_like else fill(GELU_NAIVE)
        return build_elementwise(
            "relu", dtype, size, input_names=("x",),
            ref_fn=(_relu_as_written if relu_like else _gelu_ref),
            baselines=(), flops_per_elem=1 if relu_like else 10,
            description="Implement exactly what the kernel body computes; names and comments of this specification are unreliable.",
            kernel_name="relu_kernel", task_name=name,
            triton_source=fill(RELU_COMMENT_GELU if relu_like else GELU_NAMED_RELU), naive_msl=naive,
            large=30.0 if not relu_like else 1.0e4,
        )
    if name == "sem_attention_odd_scale":
        from .flash_attention.task import build_attention

        return build_attention(
            name, dtype, size, odd_scale=True,
            description="FlashAttention forward as written in the specification (read how qk_scale is formed before using the standard 1/sqrt(D) scale).",
        )
    if name == "sem_gemm_neg":
        from .gemm.task import build_gemm

        t = build_gemm(
            "gemm", dtype, size,
            epilogue_ref=lambda c: -c, epilogue_msl="", epilogue_def="", baselines=(),
            description="Matrix multiplication as described in the specification's code (read the accumulation carefully).", extra_flops=0.0,
        )
        # 题面从 gemm 的 kernel_triton.py 读入后，把累加那一行改成减法，说明文字加上 standard GEMM，核名保持 gemm_kernel。
        src = t.triton_source
        assert "acc = tl.dot(a, b, acc)" in src
        src = src.replace("acc = tl.dot(a, b, acc)", "acc -= tl.dot(a, b)").replace("matrix multiplication C = A @ B", "matrix multiplication C = A @ B (standard GEMM)")
        t.triton_source = src
        # 朴素 Metal 里的 += 同步改成 -=，与上面的 epilogue_ref（对 C 取负）是同一条计算。
        t.naive_msl = t.naive_msl.replace("acc[i][j] += a[i] * b[j]", "acc[i][j] -= a[i] * b[j]")
        t.name = name
        return t
    raise KeyError(name)
