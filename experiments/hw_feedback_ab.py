"""Did the hardware-feature feedback change what the model proposed and wrote? (read-only analysis of run directories; no GPU, no LLM)

For every round of a run it takes the seed the planner saw and the list of features the feedback block named as missing. In a run with ``search.hw_feedback`` that list is
parsed from the block in the planner request; in a run without it (the control, e.g. ``runs/apple-m4-max/p1/fa_flash``) the block that WOULD have been shown is computed from the same
seed with ``taic.hwfeedback``. Then it counts

* plans of the round whose title or description mention one of the listed features (keyword match: a rough measure of "the model talks about it"),
* patch candidates of the round (the final kernel of each branch) that GAIN a listed feature the seed lacked (the static detector decides: ``taic.hwfeatures``),

so the two kinds of run can be compared with the same yardstick. It measures influence on proposals, not on speed; one run of each is an anecdote.

    uv run python -m experiments.hw_feedback_ab runs/apple-m4-max/p1/fa_flash_hw runs/apple-m4-max/p1/fa_flash [--json out.json]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
HEADER = "Hardware-feature analysis of this kernel"

# rough keyword map: does a plan's text talk about the feature? (case-insensitive)
KEYWORDS: dict[str, str] = {
    "matrix_unit": r"simdgroup_matrix|simdgroup_multiply|\bmma\b|matrix unit",
    "float_accumulate": r"float accumul|simdgroup_float8x8|float32 accumul",
    "register_blocking": r"fragment|register.block|accumulator tile|more accumulators|2x2|4x4",
    "tg_staging": r"threadgroup (memory|staging|tile)|stag(e|ing) .{0,30}threadgroup|shared.memory",
    "tg_padding": r"\bpad|bank.conflict|odd stride",
    "pipelining": r"double.?buffer|prefetch|pipelin|overlap|ring buffer|multi.stage",
    "epilogue_on_fragments": r"epilogue",
    "online_softmax": r"online softmax|running max|rescal|m_old|m_new",
    "simd_reduce": r"simd_sum|simd_max|simd.group reduc",
    "simd_shuffle": r"simd_shuffle|shuffle",
    "operand_in_registers": r"thread_elements|operand fragment|fragments? .{0,40}register|register.{0,40}fragment",
    "bit_packing": r"bit.?pack|bitwise|shift.{0,10}mask|decode",
    "vector_access": r"16.byte|vector(ized)? (load|access|store)|uint4|half4|wide (load|access)",
    "single_pass": r"single.pass|one pass|second pass|online recurrence",
    "few_barriers": r"barrier",
    "fast_math": r"fast::|exp2|fast.math|log2",
}


def _read(p: Path):
    """@brief 严格读一个 JSON 文件（拒绝重复键），读不出来时返回 ``None``。

    Plan §6.4: report/state/result reads go through the strict loader — duplicate keys are refused
    instead of silently keeping the last value.  Currency is judged separately
    (:func:`taic.artifacts.document_state`); this helper only parses.

    @param p JSON 文件路径。
    @return 解析结果；文件缺失、损坏或含重复键时为 ``None``。
    """
    from tcenv.schema import loads_strict

    try:
        return loads_strict(p.read_text(encoding="utf-8"), what=str(p))
    except (OSError, ValueError):
        return None


def _state(report: dict | None) -> str:
    """@brief 一份候选报告的当前性；没有报告时是 ``missing``。

    @param report 已解析的报告，可为 None。
    @return ``current`` / ``old_format`` / ``unreadable`` / ``missing``。
    """
    from taic.artifacts import document_state

    return "missing" if report is None else document_state(report)


def _current(report: dict | None) -> bool:
    """@brief 这份报告是不是当前证据（schema 4，类型严格）。

    @param report 已解析的报告，可为 None。
    @return 是当前格式时为 ``True``。
    """
    return report is not None and _state(report) == "current"


def listed_features(block: str) -> list[str]:
    """@brief 从反馈块的``缺失列表``里取出特征 id。

    Feature ids named in the 'not found' list of a block (``- matrix unit: ...`` lines).

    @param block 硬件特征反馈正文。
    @return 归一成下划线形式的特征 id 列表。
    """
    return [m.group(1).replace(" ", "_") for m in re.finditer(r"^- ([a-z][a-z ]+?): ", block, re.M)]


def _chats(run_dir: Path) -> list[dict]:
    """@brief 读出 transcript 里所有 ``event=chat`` 的行。

    @param run_dir 运行目录。
    @return chat 事件列表；文件读不到时为空列表。
    """
    from taic.reads import iter_jsonl

    # Same line reader as the rest of the tree: bad lines are skipped, not fatal.
    return [e for e in iter_jsonl(run_dir / "transcript.jsonl") if isinstance(e, dict) and e.get("event") == "chat"]


def _plans_of(chats: list[dict], rnd: int) -> list[dict]:
    """@brief 取某一轮规划器最终返回的计划列表。

    The plans the planner returned in round ``rnd`` (last parsable JSON reply of its chat turns).

    @param chats ``_chats`` 读出的 chat 事件。
    @param rnd 轮次号。
    @return 计划字典列表；没解析出来时为空列表。
    """
    plans: list[dict] = []
    for e in chats:
        if not re.match(rf"^r{rnd}\.plan(?:\.retry)?(?:#t\d+)?$", str(e.get("label", ""))):
            continue
        text = (e.get("reply") or {}).get("content") or ""
        m = re.search(r"```json\s*(\{.*\})\s*```", text, re.S) or re.search(r"(\{\s*\"bottleneck\".*\})", text, re.S)
        if not m:
            continue
        try:
            doc = json.loads(m.group(1))
        except ValueError:
            continue
        if isinstance(doc.get("plans"), list) and doc["plans"]:
            plans = [p for p in doc["plans"] if isinstance(p, dict)]
    return plans


def analyze(run_dir: Path, *, would_be: bool | None = None) -> dict:
    """@brief 逐轮统计硬件特征反馈有没有影响计划与补丁。

    Per-round and total counts for one run directory. ``would_be``: compute the block from the seed instead of reading it (default: when the run has none).

    @param run_dir 运行目录。
    @param would_be 为真时用同一个种子现算反馈块；为 ``None`` 时只在运行里没有该块时现算。
    @return 含 ``feedback_in_run``、``total`` 与逐轮明细的字典。
    """
    sys.path.insert(0, str(REPO))
    from taic import hwfeatures as hw, hwfeedback as hf
    from tcenv.evaluate import EvalReport

    run_dir = Path(run_dir)
    doc = _read(run_dir / "result.json") or _read(run_dir / "state.json") or {}
    task_name, dtype, size = doc.get("task"), doc.get("dtype") or "fp16", doc.get("size") or "paper"
    backend_id = doc.get("backend") or "metal"
    task = hw._get_task(task_name, dtype, size)
    chats = _chats(run_dir)
    by_sha: dict[str, tuple[str, dict | None]] = {}
    cands: dict[str, tuple[str, dict | None]] = {}
    for d in sorted((run_dir / "candidates").glob("*")):
        km = d / "kernel.metal"
        if km.exists():
            src = km.read_text()
            rep = _read(d / "report.json")
            by_sha[hashlib.sha1(src.rstrip().encode()).hexdigest()] = (src, rep)
            cands[d.name.split("_", 1)[1]] = (src, rep)
    rounds = []
    tot = {"rounds": 0, "rounds_with_list": 0, "plans": 0, "plans_mentioning": 0, "branches": 0, "branches_gaining": 0, "branches_correct": 0, "branches_correct_gaining": 0, "branches_undecidable": 0}
    plan_reqs = {}
    for e in chats:
        m = re.match(r"^r(\d+)\.plan(?:#t0)?$", str(e.get("label", "")))
        if m and int(m.group(1)) not in plan_reqs:
            plan_reqs[int(m.group(1))] = e
    for rnd in sorted(plan_reqs):
        user = plan_reqs[rnd]["messages"][-1]["content"]
        sm = re.search(r"```metal\n(.*?)\n```\n\nEvaluation feedback of this kernel:", user, re.S)
        seed_src, seed_rep = by_sha.get(hashlib.sha1(sm.group(1).rstrip().encode()).hexdigest(), (None, None)) if sm else (None, None)
        bm = re.search(re.escape(HEADER) + r".*?decides\.", user, re.S)
        block, source_of_list = (bm.group(0), "recorded in the planner request") if bm else (None, "")
        if (block is None) and (would_be is not False) and seed_src and seed_rep and seed_rep.get("status") == "ok" and _current(seed_rep):
            # ``seed_rep`` is already the parsed document (``_read`` → ``loads_strict``): hand it to
            # ``from_json`` directly.  Re-parsing it as report *text* would raise SchemaError, since
            # the strict report reader takes a string, not a dict.
            block = hf.hw_feedback(task, seed_src, EvalReport.from_json(seed_rep), backend_id=backend_id)
            source_of_list = "computed from the seed (not shown to the model)"
        listed = listed_features(block) if block else []
        plans = _plans_of(chats, rnd)
        mentioning = [p for p in plans if listed and any(re.search(KEYWORDS.get(f, r"(?!x)x"), f"{p.get('title', '')} {p.get('change', '')}", re.I) for f in listed)]
        seed_feats = hw.build_report(seed_src, label="seed", task=task.name, dtype=dtype, size=size, compile_evidence=False, task_obj=task).features if seed_src else None
        branches = []
        for i in range(8):
            labels = sorted((l for l in cands if re.match(rf"^r{rnd}\.p{i}(\.fix\d+)?$", l)), key=lambda l: (l.count(".fix"), l))
            if not labels:
                continue
            last = labels[-1]
            src, rep = cands[last]
            # §8: an old/damaged candidate report is not evidence.  It is listed (so a reader sees
            # it happened) but it cannot be counted as correct, carry a speedup, or enter the A/B
            # totals — a plain ``status == "ok"`` check would let a schema-3 report do exactly that.
            current = rep is not None and _current(rep)
            ok = bool(current and rep.get("status") == "ok")
            gained: list[str] = []
            if seed_feats is not None and listed:
                f = hw.build_report(src, label=last, task=task.name, dtype=dtype, size=size, compile_evidence=False, task_obj=task).features
                gained = [x for x in listed if x in f and f[x]["present"] and not seed_feats[x]["present"]]
            branches.append({
                "label": last, "correct": ok, "speedup": rep.get("speedup") if current else None,
                "gained_listed": gained, "report_state": _state(rep),
            })
        rounds.append({"round": rnd, "seed_found": seed_src is not None, "list": listed, "list_source": source_of_list, "plans": [p.get("title") for p in plans], "plans_mentioning_listed": [p.get("title") for p in mentioning], "branches": branches})
        tot["rounds"] += 1
        if listed:
            tot["rounds_with_list"] += 1
            tot["plans"] += len(plans)
            tot["plans_mentioning"] += len(mentioning)
            counted = [b for b in branches if b["report_state"] == "current"]
            tot["branches"] += len(counted)
            tot["branches_undecidable"] += len(branches) - len(counted)
            tot["branches_gaining"] += sum(1 for b in counted if b["gained_listed"])
            tot["branches_correct"] += sum(1 for b in counted if b["correct"])
            tot["branches_correct_gaining"] += sum(1 for b in counted if b["correct"] and b["gained_listed"])
    return {"run": str(run_dir), "task": task_name, "feedback_in_run": any(HEADER in (e["messages"][-1]["content"] if e.get("messages") else "") for e in plan_reqs.values()), "total": tot, "rounds": rounds}


def markdown(results: list[dict]) -> str:
    """@brief 把多个运行的影响统计拼成一张 markdown 表加逐轮明细。

    @param results ``analyze`` 返回的字典列表。
    @return 报告文本。
    """
    L = ["| run | hw block in the run | rounds with a missing list | plans | plans mentioning a listed feature | branches | branches gaining a listed feature | correct branches gaining one | undecidable branches (not counted) |", "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for r in results:
        t = r["total"]
        L.append(f"| {Path(r['run']).name} | {'yes' if r['feedback_in_run'] else 'no (list computed from the seed)'} | {t['rounds_with_list']} / {t['rounds']} | {t['plans']} | {t['plans_mentioning']} | {t['branches']} | {t['branches_gaining']} | {t['branches_correct_gaining']} / {t['branches_correct']} | {t['branches_undecidable']} |")
    L.append("")
    for r in results:
        L += [f"**{Path(r['run']).name}**", ""]
        for rd in r["rounds"]:
            L.append(f"* round {rd['round']}: list {rd['list'] or '-'}; plans mentioning a listed feature: {len(rd['plans_mentioning_listed'])} of {len(rd['plans'])}; branches gaining one: "
                     + (", ".join(f"{b['label']}{'' if b['correct'] else ' (not correct)'}: {b['gained_listed']}" for b in rd["branches"] if b["gained_listed"]) or "none"))
        L.append("")
    return "\n".join(L)


def main(argv: list[str] | None = None) -> int:
    """@brief ``python -m experiments.hw_feedback_ab`` 的入口，只读分析、不调模型不跑 GPU。

    @param argv 命令行参数；为 ``None`` 时取 ``sys.argv``。
    @return 进程退出码，正常结束时为 0。
    """
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--json", default=None)
    a = ap.parse_args(argv)
    results = [analyze(Path(p)) for p in a.runs]
    print(markdown(results))
    if a.json:
        Path(a.json).write_text(json.dumps(results, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
