"""Does the model use GPU-counter evidence, and does teaching it how to read the counters change its diagnosis?

One real Instruments trace (recorded with a counter template, see README "Hardware counters") of a kernel is turned into profiler evidence exactly as the
orchestrator does it, and the planner prompt of the search is sent to the real model under four conditions (``--samples`` replies each):

  none               no evidence at all
  trace_only         evidence without the counters (spills, shader size, GPU-active fraction)
  counters_untaught  evidence with counters, system prompt without the profiler skill
  counters_taught    evidence with counters, system prompt with Skill 2b (the setting of ``search.profile_poor = true``)

Only the planner request is sent (no patches, no GPU work apart from exporting the trace tables), so a run costs a few cents.
Usage: ``python -m experiments.counter_teaching --trace <capture.trace> --task gemm --size small --out runs/apple-m4-max/counter_teaching``
"""

# 和语义测试是同一类检查的另一面：模型有没有读到眼前的证据，还是在复述提示里点过的名字。
# 这里不改内核。同一条轨迹拆成四种提示，看规划回复是否依据计数器读数，而不是因为技能文本里写了 occupancy、limiter 就这么诊断。

from __future__ import annotations

import argparse
import asyncio
import json
import re
from pathlib import Path

from taic import prompts
from taic.llm import LLM, CostLedger, Transcript, load_config
from taic.orchestrator import parse_plans
from tasks import get_task
from tcenv import xctrace

INTRO = (
    "\n\nProfiler evidence for this kernel (Instruments Metal System Trace; taken because the kernel is slower than the strongest baseline). "
    "Read it with the \"Instruments trace\" section of Skill 2. Name the finding or counter that your diagnosis rests on (spills, the saturated limiter, low occupancy, ...) "
    "and make the changes address it; if the evidence says nothing about your hypothesis, say so instead of inventing numbers:\n"
)
OLD_INTRO = "\n\nProfiler evidence for this kernel (Instruments Metal System Trace; taken because the kernel is slower than the strongest baseline). Use it to choose and implement changes:\n"
BASE_FEEDBACK = "STATUS: ok  (stage reached: bench)\n(offline diagnosis test: baseline timings are not attached to this feedback)"

# 事后在回复全文里数这些词。百分数那一条用来看它有没有引用一个读数，而不是只复述技能里的术语。
MARKERS = {
    "occupancy": r"occupanc",
    "limiter": r"limiter",
    "latency": r"latency|dependen(t|cy)|stall",
    "grid/launch": r"grid|few threadgroups|small (problem|grid)|launch contract|fixed (by|grid)",
    "ILP": r"instruction-level|\bILP\b|independent (accumulat|work|loads)|several accumulators|multiple accumulators",
    "counter cited": r"\d+(\.\d+)?\s?%",
}


# 返回的是（系统提示是否带 Skill 2b，贴在反馈后的证据）。none 两样都空。
# trace_only 去掉计数器；untaught 保留读数但用旧导语；taught 才换上点名 limiter / occupancy 的导语，并打开技能。
def conditions(summ: xctrace.TraceSummary) -> dict[str, tuple[bool, str]]:
    """name -> (system prompt carries Skill 2b, evidence text appended to the feedback)"""
    # 拷一份再把计数器清空，这样格式化出来的正文里没有那些读数。
    no_counters = xctrace.TraceSummary(**{**summ.__dict__, "counters": {}})
    return {
        "none": (False, ""),
        "trace_only": (False, OLD_INTRO + xctrace.format_for_llm(no_counters)),
        "counters_untaught": (False, OLD_INTRO + xctrace.format_for_llm(summ)),
        # 这一档的导语自己会点出 limiter、occupancy。和 untaught 比的时候，多出来的词可能来自导语或技能，不一定来自读数。
        "counters_taught": (True, INTRO + xctrace.format_for_llm(summ)),
    }


# 只发规划那一次请求。标记是对回复文本的正则，用来统计它说出了哪些词，内核不会被编译或再测。
async def ask(llm: LLM, system: str, task, source: str, feedback: str, k: int, label: str) -> dict:
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": prompts.plan_user(task, source, feedback, [], k, source_fence="metal")}]
    reply = await llm.chat(msgs, label=label)
    analysis, plans = parse_plans(reply.text(), k)
    text = reply.text()
    return {"label": label, "analysis": analysis, "plans": plans, "markers": {n: bool(re.search(p, text, re.I)) for n, p in MARKERS.items()}, "raw": text}


# 四种条件各采样多次，看的是出现频率。内核源码每次相同，变的只有证据正文和要不要把读计数器的技能放进系统提示。
async def main_async(a) -> None:
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "export").mkdir(exist_ok=True)
    summ = xctrace.analyze(Path(a.trace), out / "export")
    (out / "summary.json").write_text(json.dumps(summ.to_json(), indent=1))
    task = get_task(a.task, a.dtype, a.size)
    source = Path(a.kernel).read_text() if a.kernel else task.naive_msl
    cfg, search = load_config()
    ledger = CostLedger(cfg, a.budget)
    llm = LLM(cfg, ledger, Transcript(out / "transcript.jsonl"), max_concurrency=4)
    jobs, names = [], []
    for name, (taught, evidence) in conditions(summ).items():
        # taught 决定技能在不在系统提示里；evidence 另附在反馈后面。两边独立，才能分开「会背提示里的词」和「看见了轨迹里的数」。
        system = prompts.system_prompt(task, use_tools=cfg.use_tools, profiler_evidence=taught)
        for i in range(a.samples):
            jobs.append(ask(llm, system, task, source, BASE_FEEDBACK + evidence, a.k, f"{name}.{i}"))
            names.append(name)
    try:
        results = await asyncio.gather(*jobs)
    finally:
        await llm.aclose()
    by: dict[str, list[dict]] = {}
    for n, r in zip(names, results):
        by.setdefault(n, []).append(r)
    (out / "results.json").write_text(json.dumps(by, indent=1))
    lines = [f"# Counter teaching check: {a.task} {a.dtype} {a.size}, model {cfg.model}, {a.samples} replies per condition, spend ${ledger.total_usd:.3f}\n"]
    lines.append("| condition | " + " | ".join(MARKERS) + " |\n|---|" + "---|" * len(MARKERS))
    for n, rs in by.items():
        lines.append(f"| {n} | " + " | ".join(f"{sum(r['markers'][m] for r in rs)}/{len(rs)}" for m in MARKERS) + " |")
    for n, rs in by.items():
        lines.append(f"\n## {n}")
        for r in rs:
            lines.append(f"- analysis: {r['analysis']}\n  plans: " + "; ".join(p["title"] for p in r["plans"]))
    (out / "REPORT.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


# 必须给一条用计数器模板录的 .trace。默认 $1 只够这些规划请求，不会按论文的 $15 去搜内核。
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--trace", required=True, help="a .trace bundle recorded with a counter template")
    ap.add_argument("--task", default="gemm")
    ap.add_argument("--dtype", default="fp16")
    ap.add_argument("--size", default="small")
    ap.add_argument("--kernel", help="the MSL source that was traced (default: the task's reference kernel)")
    ap.add_argument("--samples", type=int, default=3)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--budget", type=float, default=1.0)
    ap.add_argument("--out", default="runs/apple-m4-max/counter_teaching")
    asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    main()
