"""Prompt construction. Initial, repair and patch requests share one system prompt so that provider-side prefix caching applies."""

# 本文件只负责把每轮要说的话拼出来，不跑模型、也不评测。记下来的 system_prompt 整次搜索不变，初始、修补和补丁请求都用它，服务商才能吃到前缀缓存。
# 规划是例外：报告里的 roofline.bound 已知时，这一次请求的系统提示拿掉 memory 与 compute 两节，对应的一节放进 plan_user。bound 未知时规划请求也用原文。
# 阅读顺序：HARNESS_RULES 与 system_prompt（契约和签名不许改）→ initial_user（初始生成）→ repair_user（修补）→ plan_user（诊断加规划）→ patch_user（按一份方案打补丁）。提示词正文一个字都不能改，注释都在字符串外面。

from __future__ import annotations

from pathlib import Path

from tcenv.task import Task

from .symptom import section_for
from .target import skill_text
from .tools import SkillIndex

WIKI_DOCS = ("experience",)


def _skill(name: str, target_id: str | None = None) -> str:
    return skill_text(name, target_id)
HARNESS_RULES = """\
You are the code-generation backend of a compiler. You translate ONE Triton kernel directly into ONE Metal Shading
Language (MSL) kernel for the target device named in the contract. There is no intermediate IR and no existing compiler in the loop:
you choose instructions, thread mapping, data movement, synchronization and scheduling yourself.

What "valid" means (all of it is enforced mechanically by the evaluation environment):
1. Without a `/* taic-launch */` block, the kernel must have exactly the name, buffer slots, argument names and access
   modes of the contract. Inputs are `device const T*`, outputs `device T*`, scalars `constant T&`. You may add
   thread-position attribute parameters ([[thread_position_in_grid]] etc., see the MSL reference for the dimension rule)
   but no other buffers, no dynamic `threadgroup` parameters, no textures.
   With one `/* taic-launch */` block, that block is the only launch description: several `kernel` functions, each with
   its own grid, threads, optional dynamic threadgroup bytes and optional workspace buffers. Contract inputs stay
   `device const` in every stage. An output may be `device const` in a stage that only reads it, but some stage must
   write it. No textures or samplers. Do not add scalar arguments the contract does not list.
2. Without that block, the launch configuration (threadgroups per grid and threads per threadgroup) and the compile-time
   constants are fixed by the contract and already defined in a header that the harness prepends. With the block, each
   stage's grid and threads come from the block. Write the line as `launch NAME grid=16,1,1 threads=128,1,1`.
   Spaces and parentheses in those numbers are the same launch: `grid=(16, 1, 1)` and `grid=16, 1, 1` both mean 16 by 1 by 1.
   Sizes may be integer expressions over the contract constants (`M_TOTAL / BLOCK_M`). A division must come out even.
   A name that is not a contract constant is rejected before compilation. One `math fast` or `math safe` line only.
   At most 32 stages: encoding sits outside the timer, and the cap is what keeps a long
   pipeline from hiding its setup there. One stage may launch at most 2^24 threadgroups (65536 is allowed; a larger
   grid is rejected). Do not redefine the constants and do not `#include` anything (the header already includes
   metal_stdlib, metal_simdgroup and metal_simdgroup_matrix). `math safe` in the block turns off fast-math for that candidate.
   Every workspace is filled with zeros inside the timed command buffer, on every iteration, before your stages.
   That fill is part of your measured time. A smaller workspace pays less of it.
3. Helpers (`inline` functions, structs, `constant constexpr` values of your own) may precede the kernel. Return the
   complete source text of everything you define.
4. Results must match a float64/float32 CPU reference on adversarial random inputs (overflow-, underflow-, cancellation- and
   precision-stressing distributions) within the stated tolerance. Accumulate in float; round to the element type once.
5. Safety: every output element must be written, no out-of-bounds access (guard bands + Metal shader validation), inputs
   must not be modified, results must be bit-identical across launches, and the result must not depend on the initial
   contents of the output buffer (it is NaN-filled). No unbounded loops: a runaway GPU kernel can freeze the machine's
   display, so `while (true)` / `for (;;)` are refused. One kernel's command buffer is killed after 2 seconds. A pipeline's
   one buffer is allowed 2 seconds per stage, up to 8 seconds. Past that the search treats it as a hang and stops.
6. Performance is measured against the fastest vendor implementation of the same operator (MLX, PyTorch-MPS and, for composite
   operators, their compiled/fused variants); only a
   candidate that is correct, safe AND faster than the current best is kept.

Reply format: first at most a few sentences of reasoning about the design (what each thread/SIMD group/threadgroup does and why),
then EXACTLY ONE fenced code block tagged ```metal containing the complete kernel source, and nothing after it.
"""


def system_prompt(task: Task, *, use_tools: bool, profiler_evidence: bool = False, hw_feedback: bool = False, target_id: str | None = None) -> str:
    from tcenv.backends.registry import load_backend
    from tcenv.target import load_target

    backend = load_backend(load_target(target_id).backend)
    c = task.contract
    parts = [HARNESS_RULES, "# Target hardware and task\n", skill_text("device", target_id)]
    parts.append(task.description)
    parts.append("\nCompilation contract:\n" + c.describe())
    parts.append(f"\nRequired signature skeleton:\n```{backend.source_fence}\n" + c.abi_stub() + "\n```")
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
    parts.append("\n# Skill 1: Apple GPU optimization principles\n" + _skill(backend.skill_names[0], target_id))
    parts.append("\n# Skill 2: interpreting the evaluation feedback (profiling)\n" + _skill(backend.skill_names[1], target_id))
    # 只有反馈可能带上 Instruments 痕迹时才多这一段。其它运行必须保持提示字节不变，否则前缀缓存会失效。
    if profiler_evidence:  # only when the feedback can carry an Instruments trace; every other run keeps the byte-identical prompt
        parts.append("\n# Skill 2b: reading the profiler evidence (Instruments trace and GPU counters)\n" + _skill(backend.skill_names[3], target_id))
    if hw_feedback:  # only when search.hw_feedback appends the hardware-feature analysis to the feedback; off = the byte-identical prompt
        parts.append("\n# Skill 2c: reading the hardware-feature analysis in the feedback\n" + _skill(backend.skill_names[4], target_id))
    idx = SkillIndex(backend.reference_skill, extra=WIKI_DOCS, target_id=target_id)
    if use_tools:
        parts.append(
            "\n# Skill 3: MSL reference (retrieval)\nCall the `lookup_msl` tool for details. Sections available:\n" + idx.outline()
        )
    else:
        parts.append("\n# Skill 3: MSL reference\n" + idx.full_text())
    return "\n".join(parts)


# 初始生成。Triton 源码是语义规格，不是名字或注释；program 到 threadgroup 的启动映射保持不变。这一步只求正确且速度合理，结构级的性能改动留给后面的 patch_user。
def initial_user(task: Task) -> str:
    return (
        "Translate this Triton kernel into MSL for the contract above. It is the *specification*: follow its actual semantics, not its name or comments. "
        "Without a taic-launch block, keep the contract's launch mapping of programs to threadgroups. "
        "A taic-launch block may split the work into stages and choose each stage's grid and threads.\n\n```python\n"
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


# bound 已知时，从这一次规划用的系统提示里拿掉两节优化章节。拿不干净就原样返回，避免只删掉一半。
def planning_system(system: str, bound: str | None) -> str:
    if bound not in ("memory", "compute"):
        return system
    chunks = [section_for(name) for name in ("memory", "compute")]
    if any(not chunk or ("\n\n" + chunk) not in system for chunk in chunks):
        return system
    out = system
    for chunk in chunks:
        out = out.replace("\n\n" + chunk, "", 1)
    return out


def _roofline_notes(bound: str | None) -> str:
    section = section_for(bound)
    if not section:
        return ""
    return f"Measured roofline class: {bound}. Matching section of the Apple GPU optimization notes:\n\n{section}"


# 一轮的诊断加规划：从 roofline、流水线统计和源码指出主导瓶颈，再要 K 份互不相同的方案（JSON）。抽不出 JSON 时不在这里收场，编排器会重试或改用 FALLBACK_PLANS。
# bound 为 memory 或 compute 时，只在这条用户消息里附上对应的那一节。其它值不附，也不改原句。
def plan_user(task: Task, best_source: str, best_feedback: str, history: list[str], k: int, mixed: bool = False, status: str = "", bound: str | None = None, history_block: str | None = None) -> str:
    # 只带最近 24 条尝试。history_block 非空时用它替换这一段；为空时字节与原来一致。
    hist = history_block if history_block is not None else ("\n".join(f"- {h}" for h in history[-24:]) or "- (none yet)")
    # 混合规划时最后一份是低风险局部改动；否则 K 份都要求是结构性的，避免分支挤在同一种微调上。
    if mixed and k >= 2:
        kinds = f"propose {k} DIFFERENT optimization plans. " + MIXED_PLANS.format(m=k - 1, k=k).rstrip() + "\n"
    else:
        kinds = (
            f"propose {k} DIFFERENT optimization plans, each a concrete structural change "
            "(not a micro-tweak) that could be implemented independently of the others, ordered from most to least promising.\n"
        )
    text = (
        f"Current best kernel for `{task.name}` ({task.dtype}):\n```metal\n{best_source.rstrip()}\n```\n\n"
        f"Evaluation feedback of this kernel:\n{best_feedback}\n\n"
        f"Hypotheses already tried and their outcomes (do not repeat failed ones):\n{hist}\n\n"
        f"Diagnose the dominant bottleneck from the roofline, pipeline statistics and the code, then {kinds}"
        'Reply with JSON only, in a ```json block: {"bottleneck": "<one sentence>", "analysis": "<2-4 sentences>", '
        '"plans": [{"title": "<short>", "change": "<precise description of what to change in the kernel and why it helps>", "expected_gain": "<estimate>"}]}'
    )
    extra = []
    if status:
        extra.append(status.rstrip())
    notes = _roofline_notes(bound)
    if notes:
        extra.append(notes)
    if not extra:
        return text
    return text.replace("\n\nDiagnose the dominant bottleneck", "\n\n" + "\n\n".join(extra) + "\n\nDiagnose the dominant bottleneck", 1)


# 一路分支的补丁。只落实这一份 plan。上一颗种子已经保存；这一支交回的文件和种子相同，搜索记为 unchanged，不算合格。回复仍是完整内核的一个 metal 块。
def patch_user(task: Task, best_source: str, best_feedback: str, plan: dict, status: str = "") -> str:
    text = (
        f"Current best kernel for `{task.name}` ({task.dtype}):\n```metal\n{best_source.rstrip()}\n```\n\n"
        f"Evaluation feedback of this kernel:\n{best_feedback}\n\n"
        f"Implement exactly this optimization plan and nothing else:\n**{plan['title']}**\n{plan['change']}\n\n"
        "The current kernel is already saved as the search seed. This branch is scored only on the kernel you hand in.\n"
        "If this plan fails a check, leave that failing kernel in the file. Do not paste the current kernel back to make the check pass.\n"
        "A file identical to the current kernel is recorded as unchanged and does not count.\n"
        "Reply with the complete new kernel in one ```metal block."
    )
    if not status:
        return text
    return text.replace("\n\nImplement exactly this optimization plan", "\n\n" + status.rstrip() + "\n\nImplement exactly this optimization plan", 1)
