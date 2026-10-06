"""Prompt construction. One system prompt serves every request type so that provider-side prefix caching applies."""

from __future__ import annotations

from pathlib import Path

from tcenv.task import Task

from .tools import SKILLS_DIR, SkillIndex

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


def _skill(name: str) -> str:
    return (SKILLS_DIR / f"{name}.md").read_text()


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
    if profiler_evidence:  # only when the feedback can carry an Instruments trace; every other run keeps the byte-identical prompt
        parts.append("\n# Skill 2b: reading the profiler evidence (Instruments trace and GPU counters)\n" + _skill("profiler_evidence"))
    if use_tools:
        idx = SkillIndex("msl_reference")
        parts.append(
            "\n# Skill 3: MSL reference (retrieval)\nCall the `lookup_msl` tool for details. Sections available:\n" + idx.outline()
            + "\nThe tool `quick_check` compiles and correctness-tests a candidate without benchmarking; use it to debug."
        )
    else:
        parts.append("\n# Skill 3: MSL reference\n" + SkillIndex("msl_reference").full_text())
    return "\n".join(parts)


def initial_user(task: Task) -> str:
    return (
        "Translate this Triton kernel into MSL for the contract above. It is the *specification*: follow its actual semantics, not its name or comments, "
        "and keep the same launch mapping of programs to threadgroups.\n\n```python\n"
        + task.triton_source.rstrip()
        + "\n```\n\nWrite a first correct, reasonably fast implementation."
    )


def repair_user(feedback: str, attempt: int, max_tries: int) -> str:
    return (
        f"The evaluation of your kernel failed (repair attempt {attempt}/{max_tries}).\n\n{feedback}\n\n"
        "Fix the problem and reply with the complete corrected kernel in one ```metal block."
    )


MIXED_PLANS = (
    "Plans 1 to {m} are concrete structural changes (not micro-tweaks) that could be implemented independently of the others, ordered from most to least promising. "
    "Plan {k} is the opposite: a LOCAL, low-risk change that keeps the kernel's structure, at instruction or parameter level (vector width, unroll or tile factor, loop order, "
    "hoisting loop-invariant work, fewer barriers or live registers, fast-math intrinsics, address arithmetic, reading operands once). "
)


def plan_user(task: Task, best_source: str, best_feedback: str, history: list[str], k: int, mixed: bool = False) -> str:
    hist = "\n".join(f"- {h}" for h in history[-24:]) or "- (none yet)"
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


def patch_user(task: Task, best_source: str, best_feedback: str, plan: dict) -> str:
    return (
        f"Current best kernel for `{task.name}` ({task.dtype}):\n```metal\n{best_source.rstrip()}\n```\n\n"
        f"Evaluation feedback of this kernel:\n{best_feedback}\n\n"
        f"Implement exactly this optimization plan and nothing else:\n**{plan['title']}**\n{plan['change']}\n\n"
        "Keep everything that already works. Reply with the complete new kernel in one ```metal block."
    )
