"""Prompt construction. One system prompt serves every request type so that provider-side prefix caching applies."""

# 本文件只负责把每轮要说的话拼出来，不跑模型、也不评测。所有请求共用一份 system_prompt，服务商才能吃到前缀缓存；真正区分轮次的是后面的用户消息。
# 阅读顺序：HARNESS_RULES 与 system_prompt（契约和签名不许改）→ initial_user（初始生成）→ repair_user（修补）→ plan_user（诊断加规划）→ patch_user（按一份方案打补丁）。提示词正文一个字都不能改，注释都在字符串外面。

from __future__ import annotations

from pathlib import Path

from tcenv.task import Task

from .tools import SKILLS_DIR, SkillIndex

# 每条请求都带着的系统约束。内核名、buffer 槽位、参数名和访问模式必须与契约一致；启动配置和编译期常量不许重定义，也不许再 #include。模型改了签名，评测的 ABI 检查会拒绝，这颗内核不能成为下一轮种子。
HARNESS_RULES = """\
You are the code-generation backend of a compiler. You translate ONE Triton kernel directly into ONE Metal Shading
Language (MSL) kernel for the Apple GPU described below. There is no intermediate IR and no existing compiler in the loop:
you choose instructions, thread mapping, data movement, synchronization and scheduling yourself.

What "valid" means (all of it is enforced mechanically by the evaluation environment):
1. The kernel must have exactly the name, buffer slots, argument names and access modes of the contract. Inputs are
   `device const T*`, outputs `device T*`, scalars `constant T&`. You may add thread-position attribute parameters
   ([[thread_position_in_grid]] etc., see the MSL reference for the dimension rule) but no other buffers, no dynamic
   `threadgroup` parameters, no textures.
2. The launch configuration (threadgroups per grid and threads per threadgroup) and the compile-time constants are
   fixed by the contract and already defined in a header that the harness prepends. Do not redefine them and do not
   `#include` anything (the header already includes metal_stdlib, metal_simdgroup and metal_simdgroup_matrix).
3. Helpers (`inline` functions, structs, `constant constexpr` values of your own) may precede the kernel. Return the
   complete source text of everything you define.
4. Results must match a float64/float32 CPU reference on adversarial random inputs (overflow-, underflow-, cancellation- and
   precision-stressing distributions) within the stated tolerance. Accumulate in float; round to the element type once.
5. Safety: every output element must be written, no out-of-bounds access (guard bands + Metal shader validation), inputs
   must not be modified, results must be bit-identical across launches, and the result must not depend on the initial
   contents of the output buffer (it is NaN-filled). No unbounded loops: a runaway GPU kernel can freeze the machine's
   display, so `while (true)` / `for (;;)` are refused and any single launch that runs for seconds is killed.
6. Performance is measured against the fastest vendor implementation of the same operator (MLX, PyTorch-MPS and, for composite
   operators, their compiled/fused variants); only a
   candidate that is correct, safe AND faster than the current best is kept.

Reply format: first at most a few sentences of reasoning about the design (what each thread/SIMD group/threadgroup does and why),
then EXACTLY ONE fenced code block tagged ```metal containing the complete kernel source, and nothing after it.
"""


# 把 skills 目录里的一篇参考整篇读进系统提示。文件缺失时组提示失败，这一轮请求发不出去。
def _skill(name: str) -> str:
    return (SKILLS_DIR / f"{name}.md").read_text()


# 唯一的系统提示，初始生成、修补、诊断规划和补丁都用它，服务商才能做前缀缓存。contract.describe() 与 abi_stub() 把契约和签名骨架钉死，用户消息改不了函数签名。
def system_prompt(task: Task, *, use_tools: bool, profiler_evidence: bool = False) -> str:
    c = task.contract
    parts = [HARNESS_RULES, "# Target hardware and task\n"]
    parts.append(task.description)
    parts.append("\nCompilation contract:\n" + c.describe())
    parts.append("\nRequired signature skeleton:\n```metal\n" + c.abi_stub() + "\n```")
    parts.append(
        f"\nElement type: {task.dtype} (storage `{c.args[0].dtype}` buffers). Tolerance: |out - ref| <= {task.rtol} * |ref| + {task.atol_rel} * scale "
        "(scale = RMS of the reference or a task-specific condition number), relaxed to at most 2x the vendor baseline's own error."
    )
    if "ieee" in task.cases:
        parts.append(
            "\nIEEE-754 special values: one of the test cases (`ieee`) plants NaN, +Inf, -Inf and -0.0 in the inputs. The result must match the "
            "reference exactly there: NaN exactly where the reference is NaN, infinities of the same sign where it is infinite. So NaN must propagate "
            "(a comparison such as `v > 0 ? v : 0` or `max()` that silently drops a NaN is wrong), and reductions must not skip or mask non-finite values."
        )
    parts.append("\n# Skill 1: Apple GPU optimization principles\n" + _skill("apple_gpu_optimization"))
    parts.append("\n# Skill 2: interpreting the evaluation feedback (profiling)\n" + _skill("metal_profiling"))
    # 只有反馈可能带上 Instruments 痕迹时才多这一段。其它运行必须保持提示字节不变，否则前缀缓存会失效。
    if profiler_evidence:  # only when the feedback can carry an Instruments trace; every other run keeps the byte-identical prompt
        parts.append("\n# Skill 2b: reading the profiler evidence (Instruments trace and GPU counters)\n" + _skill("profiler_evidence"))
    # 开工具时只给目录，细节靠 lookup_msl，避免模型不查就编；关工具时没有这个工具，MSL 参考必须全文在场。
    if use_tools:
        idx = SkillIndex("msl_reference")
        parts.append(
            "\n# Skill 3: MSL reference (retrieval)\nCall the `lookup_msl` tool for details. Sections available:\n" + idx.outline()
            + "\nThe tool `quick_check` compiles and correctness-tests a candidate without benchmarking; use it to debug."
        )
    else:
        parts.append("\n# Skill 3: MSL reference\n" + SkillIndex("msl_reference").full_text())
    return "\n".join(parts)


# 初始生成。Triton 源码是语义规格，不是名字或注释；program 到 threadgroup 的启动映射保持不变。这一步只求正确且速度合理，结构级的性能改动留给后面的 patch_user。
def initial_user(task: Task) -> str:
    return (
        "Translate this Triton kernel into MSL for the contract above. It is the *specification*: follow its actual semantics, not its name or comments, "
        "and keep the same launch mapping of programs to threadgroups.\n\n```python\n"
        + task.triton_source.rstrip()
        + "\n```\n\nWrite a first correct, reasonably fast implementation."
    )


# 修补轮。评测在编译、ABI、安全或正确性上失败后，把反馈塞回，并要求整颗内核放进一个 metal 块。次数用尽仍不合格，这颗候选作废，当前种子不动。
def repair_user(feedback: str, attempt: int, max_tries: int) -> str:
    return (
        f"The evaluation of your kernel failed (repair attempt {attempt}/{max_tries}).\n\n{feedback}\n\n"
        "Fix the problem and reply with the complete corrected kernel in one ```metal block."
    )


# plan_mix 且分支数至少为 2 时使用：前面几份是可独立落地的结构改动，最后一份必须保住内核结构，只做指令或参数级的小改。
MIXED_PLANS = (
    "Plans 1 to {m} are concrete structural changes (not micro-tweaks) that could be implemented independently of the others, ordered from most to least promising. "
    "Plan {k} is the opposite: a LOCAL, low-risk change that keeps the kernel's structure, at instruction or parameter level (vector width, unroll or tile factor, loop order, "
    "hoisting loop-invariant work, fewer barriers or live registers, fast-math intrinsics, address arithmetic, reading operands once). "
)


# 一轮的诊断加规划：从 roofline、流水线统计和源码指出主导瓶颈，再要 K 份互不相同的方案（JSON）。抽不出 JSON 时不在这里收场，编排器会重试或改用 FALLBACK_PLANS。
def plan_user(task: Task, best_source: str, best_feedback: str, history: list[str], k: int, mixed: bool = False) -> str:
    # 只带最近 24 条尝试。更早的失败假设不会出现，模型有可能把它们再提一遍。
    hist = "\n".join(f"- {h}" for h in history[-24:]) or "- (none yet)"
    # 混合规划时最后一份是低风险局部改动；否则 K 份都要求是结构性的，避免分支挤在同一种微调上。
    if mixed and k >= 2:
        kinds = f"propose {k} DIFFERENT optimization plans. " + MIXED_PLANS.format(m=k - 1, k=k).rstrip() + "\n"
    else:
        kinds = (
            f"propose {k} DIFFERENT optimization plans, each a concrete structural change "
            "(not a micro-tweak) that could be implemented independently of the others, ordered from most to least promising.\n"
        )
    return (
        f"Current best kernel for `{task.name}` ({task.dtype}):\n```metal\n{best_source.rstrip()}\n```\n\n"
        f"Evaluation feedback of this kernel:\n{best_feedback}\n\n"
        f"Hypotheses already tried and their outcomes (do not repeat failed ones):\n{hist}\n\n"
        f"Diagnose the dominant bottleneck from the roofline, pipeline statistics and the code, then {kinds}"
        'Reply with JSON only, in a ```json block: {"bottleneck": "<one sentence>", "analysis": "<2-4 sentences>", '
        '"plans": [{"title": "<short>", "change": "<precise description of what to change in the kernel and why it helps>", "expected_gain": "<estimate>"}]}'
    )


# 一路分支的补丁。只落实这一份 plan，已经能工作的部分保持不动，因此不许借改写改掉契约或签名。回复仍是完整内核的一个 metal 块。
def patch_user(task: Task, best_source: str, best_feedback: str, plan: dict) -> str:
    return (
        f"Current best kernel for `{task.name}` ({task.dtype}):\n```metal\n{best_source.rstrip()}\n```\n\n"
        f"Evaluation feedback of this kernel:\n{best_feedback}\n\n"
        f"Implement exactly this optimization plan and nothing else:\n**{plan['title']}**\n{plan['change']}\n\n"
        "Keep everything that already works. Reply with the complete new kernel in one ```metal block."
    )
