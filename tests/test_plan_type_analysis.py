"""experiments/plan_type_analysis.py on hand-built run directories: no GPU, no LLM, no network, no tcenv/taic import.

The fixtures write tiny runs in the on-disk layout of a real search (``candidates/NNN_<label>/report.json`` + ``kernel.metal``,
``transcript.jsonl`` chat events with ``finish_reason`` and ``usage``, ``result.json``) so that the classification of plans and of the
first attempt of every patch request, the cause of a reply without code, the statistics and the report text are checked against known answers.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import pytest

from experiments import plan_type_analysis as pta

CAP = 48000
MIXED_PLAN_PROMPT = (
    "Diagnose the dominant bottleneck, then propose 3 DIFFERENT optimization plans. Plans 1 to 2 are concrete structural changes (not micro-tweaks). "
    "Plan 3 is the opposite: a LOCAL, low-risk change that keeps the kernel's structure."
)
STRUCT_PLAN_PROMPT = "Diagnose the dominant bottleneck, then propose 3 DIFFERENT optimization plans, each a concrete structural change (not a micro-tweak)."
SEED = "kernel void k(device half* a [[buffer(0)]]) {\n  a[0] = 1;\n}"


def patch_prompt(title: str, change: str, seed: str = SEED) -> str:
    return (
        f"Current best kernel for `flash_attention` (fp16):\n```metal\n{seed}\n```\n\nEvaluation feedback of this kernel:\nok\n\n"
        f"Implement exactly this optimization plan and nothing else:\n**{title}**\n{change}\n\nKeep everything that already works. Reply with the complete new kernel in one ```metal block."
    )


class Writer:
    """Builds one run directory; the tests describe the run with calls, the files are real."""

    def __init__(self, root: Path, name: str, task: str = "flash_attention", model: str = "deepseek-flash"):
        self.dir = root / name
        (self.dir / "candidates").mkdir(parents=True)
        self.task, self.model, self.n = task, model, 0
        self.transcript: list[str] = []
        self.usd = 0.0

    def cand(self, label: str, status: str = "ok", speedup: float | None = None, abi: str | None = None, source: str = SEED) -> None:
        d = self.dir / "candidates" / f"{self.n:03d}_{label}"
        d.mkdir()
        self.n += 1
        rep: dict = {"status": status, "task": self.task, "dtype": "fp16", "speedup": speedup if status == "ok" else None, "message": status, "roofline": {}}
        if status == "ok":
            rep["speedup"] = speedup if speedup is not None else 1.0
        if abi:
            rep["abi"] = {"violations": [abi]}
        (d / "report.json").write_text(json.dumps(rep))
        (d / "kernel.metal").write_text(source)

    def chat(self, label: str, *, finish: str = "stop", completion: int = 3000, reasoning: int = 500, prompt: int = 9000, content: str = "", last_user: str = "", tool_calls: int = 0, cost: float = 0.01) -> None:
        self.usd += cost
        reply = {"role": "assistant", "content": content, "reasoning_content": "think ```x``` think ```y```" if reasoning else ""}
        if tool_calls:
            reply["tool_calls"] = [{"id": "t", "type": "function", "function": {"name": "lookup_msl", "arguments": "{}"}}] * tool_calls
        ev = {
            "event": "chat", "label": label, "model": self.model, "seconds": 1.0, "messages": [{"role": "system", "content": "s"}, {"role": "user", "content": last_user or "u"}],
            "reply": reply, "finish_reason": finish,
            "usage": {"prompt_tokens": prompt, "completion_tokens": completion, "completion_tokens_details": {"reasoning_tokens": reasoning}},
            "cost_usd": cost, "total_usd": self.usd,
        }
        self.transcript.append(json.dumps(ev))

    def code_reply(self, label: str, *, last_user: str = "", reasoning: int = 500, completion: int = 3000) -> None:
        self.chat(label, content="here\n```metal\n" + SEED + "\n```", last_user=last_user, reasoning=reasoning, completion=completion)

    def truncated(self, label: str, *, last_user: str = "") -> None:
        self.chat(label, finish="length", completion=CAP, reasoning=CAP, content="", last_user=last_user, cost=0.058)

    def finish(self, rounds: int = 1, stop: str = "max_rounds", partial_tail: str | None = None) -> Path:
        text = "\n".join(self.transcript) + "\n"
        if partial_tail is not None:
            text += partial_tail  # a line being written right now
        (self.dir / "transcript.jsonl").write_text(text)
        if stop:
            (self.dir / "result.json").write_text(json.dumps({"task": self.task, "dtype": "fp16", "model": self.model, "rounds": rounds, "stop_reason": stop, "spent_usd": self.usd, "budget_usd": 3.0}))
        return self.dir


def build_fa_run(root: Path, name: str = "fa_run", mixed: bool = False) -> Path:
    """init ok; round 1 = three plans: p0 data movement (ok), p1 thread mapping (reply cut by max_tokens, then ok after a repair turn), p2 local (abi_error, then ok)."""
    w = Writer(root, name)
    w.chat("init#t0", content="```metal\n" + SEED + "\n```")
    w.cand("init", "ok", 1.0)
    w.chat("r1.plan", content="{}", last_user=(MIXED_PLAN_PROMPT if mixed else STRUCT_PLAN_PROMPT))
    w.code_reply("r1.p0#t0", last_user=patch_prompt("Double-buffer K/V staging with one barrier per iteration", "Stage K and V in threadgroup memory, ping-pong, one barrier per iteration."))
    w.cand("r1.p0", "ok", 1.30)
    w.truncated("r1.p1#t0", last_user=patch_prompt("Split the key range across SIMD groups", "Each SIMD group handles a quarter of the keys, then merge the partial softmax states."))
    w.code_reply("r1.p1.fix1#t0")
    w.cand("r1.p1.fix1", "ok", 1.10)
    w.code_reply("r1.p2#t0", last_user=patch_prompt("Hoist address arithmetic out of the loop (local, structure-preserving)", "Local, structure-preserving: hoist the loop-invariant index math and strength-reduce the pointers."))
    w.cand("r1.p2", "abi_error", abi="static threadgroup memory 40000 B exceeds the device limit of 32768 B")
    w.code_reply("r1.p2.fix1#t0")
    w.cand("r1.p2.fix1", "ok", 1.05)
    return w.finish()


# ----------------------------------------------------------------------------------------------------------------------
# plan classification
# ----------------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    "title,change,kind",
    [
        ("Split the key range across SIMD groups", "Each SIMD group takes a quarter of the keys", "thread_mapping"),
        ("Producer/consumer warp split with 16-row M tiles", "", "thread_mapping"),
        ("Double-buffered K/V staging", "Stage the tiles in threadgroup memory and prefetch the next one", "data_movement"),
        ("Hoist loop-invariant address arithmetic", "strength-reduce the pointers", "local"),
        ("Cheaper activation: exp2 + reciprocal-multiply", "", "local"),
        ("2-D Winograd F(2x2,3x3)", "implicit GEMM", "algorithm"),
        ("Some tweak", "Local, structure-preserving: change the staging layout", "local"),  # declared local beats the content words
        ("Lagged uniform shift", "stop rescaling the accumulator on most tiles", "algorithm"),
        ("zzz", "qqq", "unclassified"),
    ],
)
def test_classify_plan_kinds(title, change, kind):
    r = pta.classify_plan(title, change)
    assert r["kind"] == kind
    assert r["scope"] == ("local" if kind == "local" else "unclassified" if kind == "unclassified" else "structural")


def test_fallback_plans_are_generic_whatever_their_words():
    r = pta.classify_plan("use simdgroup_matrix for matrix products", "stage in threadgroup memory", fallback=True)
    assert r["kind"] == "generic" and r["scope"] == "generic"
    assert all(t in pta.FALLBACK_TITLES for t in ("vectorized 16-byte memory access", "single pass over global memory with SIMD-group reductions"))


def test_title_outweighs_a_long_description():
    # three structural words in the description must not outvote a local headline plus a local description
    r = pta.classify_plan("Hoist the loop-invariant index math", "staging staging staging ring buffer smem")
    assert r["kind"] == "local"
    assert pta.classify_plan("", "")["kind"] == "unclassified"  # empty title and description: no exception


def test_task_groups():
    assert pta.task_group("flash_attention@64x32t256") == "flash_attention"
    assert pta.task_group("conv2d") == pta.task_group("gemm") == "matrix (gemm/conv2d/mamba2)"
    assert pta.task_group("relu").startswith("memory-bound")


# ----------------------------------------------------------------------------------------------------------------------
# statistics helpers
# ----------------------------------------------------------------------------------------------------------------------
def test_fisher_exact_known_values():
    assert pta.fisher_exact(1, 9, 11, 3) == pytest.approx(0.002759, abs=1e-5)  # the classic tea-tasting style table
    assert pta.fisher_exact(5, 5, 5, 5) == pytest.approx(1.0)
    assert pta.fisher_exact(0, 0, 0, 0) == 1.0
    assert 0.0 <= pta.fisher_exact(3, 0, 0, 3) <= 0.11


def test_wilson_interval_and_small_n_flag():
    lo, hi = pta.wilson(8, 10)
    assert lo == pytest.approx(0.490, abs=2e-3) and hi == pytest.approx(0.943, abs=2e-3)
    assert pta.wilson(0, 0) == (None, None)
    assert pta.rate_cell(3, 9).endswith("†") and not pta.rate_cell(30, 90).endswith("†")
    assert pta.rate_cell(0, 0) == "0/0"


def test_mann_whitney_detects_a_shift_and_not_noise():
    assert pta.mann_whitney_p([10, 11, 12, 13, 14, 15], [1, 2, 3, 4, 5, 6]) < 0.01
    assert pta.mann_whitney_p([1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6]) > 0.9
    assert pta.mann_whitney_p([], [1]) is None


def test_cmh_pools_strata_and_skips_uninformative_ones():
    # two strata in which A (ok) is clearly better than B: common odds ratio > 1
    res = pta.cmh([(8, 2, 3, 7), (9, 1, 4, 6), (5, 0, 0, 5)])
    assert res["strata"] == 3 and res["odds_ratio"] > 1 and res["p"] < 0.05
    assert pta.cmh([(3, 0, 0, 0)]) == {"strata": 0, "odds_ratio": None, "p": None}


def test_within_run_ranks_remove_the_differences_between_runs():
    recs = [{"run": "a", "x": 1}, {"run": "a", "x": 2}, {"run": "a", "x": 3}, {"run": "b", "x": 1000}, {"run": "b", "x": 2000}, {"run": "b", "x": None}]
    rk = pta.within_run_ranks(recs, "x")
    assert [round(rk[id(r)], 3) for r in recs[:3]] == [round(1 / 6, 3), 0.5, round(5 / 6, 3)]
    assert rk[id(recs[3])] == pytest.approx(0.25) and rk[id(recs[4])] == pytest.approx(0.75)
    assert id(recs[5]) not in rk


# ----------------------------------------------------------------------------------------------------------------------
# transcript turns and the cause of a reply without code
# ----------------------------------------------------------------------------------------------------------------------
def test_turn_row_reads_tokens_and_how_the_reply_ended():
    ev = {"label": "r2.p1#t3", "finish_reason": "length", "usage": {"prompt_tokens": 100, "completion_tokens": CAP, "completion_tokens_details": {"reasoning_tokens": CAP}},
          "reply": {"content": "", "reasoning_content": "a ```x``` b ```y``` c"}, "cost_usd": 0.058}
    row = pta.turn_row(ev, "r2.p1")
    assert (row["turn"], row["finish"], row["prompt"], row["completion"], row["reasoning"], row["content_chars"]) == (3, "length", 100, CAP, CAP, 0)
    assert row["reasoning_fences"] == 2 and not row["has_fence"] and row["tool_calls"] == 0
    odd = pta.turn_row({"label": "r1.plan.retry", "usage": "CompletionUsage(...)", "reply": None}, "r1.plan.retry")  # usage may be a plain string
    assert odd["turn"] == 0 and odd["completion"] is None and odd["finish"] is None


def test_nocode_cause_from_the_last_turn(tmp_path):
    w = Writer(tmp_path, "causes")
    w.truncated("a#t0")  # reasoning exhausted the cap
    w.chat("b#t0", finish="length", completion=CAP, reasoning=1000, content="x" * 1000)  # a long visible reply that did not fit
    w.chat("c#t0", finish="tool_calls", tool_calls=1)  # the run ended (budget) before the tool result was answered
    w.chat("d#t0", content="I could not do it")  # a normal reply without a code block
    w.chat("e#t0", content="```python\nprint(1)\n```")  # a code block that is not the kernel
    w.finish()
    tr = pta.read_transcript(w.dir / "transcript.jsonl")
    assert [pta.nocode_cause(tr, t) for t in "abcde"] == ["reasoning_cap", "output_cap", "run_ended", "no_block", "block_without_kernel"]
    assert pta.nocode_cause(tr, "zzz") == "unknown"


# ----------------------------------------------------------------------------------------------------------------------
# one hand-built run: branch records
# ----------------------------------------------------------------------------------------------------------------------
@pytest.fixture()
def fa_run(tmp_path):
    return build_fa_run(tmp_path / "runs")


def test_branches_of_a_hand_built_run(fa_run):
    run = pta.load_run(fa_run, "fa_run", "runs")
    assert run.finished and run.k == 3 and run.plan_mix is False
    branches, attempts = pta.build_branches(run, pta.DEFAULT_SIZE_EDGES)
    by = {b["base"]: b for b in branches}
    assert set(by) == {"init", "r1.p0", "r1.p1", "r1.p2"}
    assert (by["r1.p0"]["first_status"], by["r1.p0"]["plan_kind"], by["r1.p0"]["accepted"]) == ("ok", "data_movement", True)
    p1 = by["r1.p1"]
    assert (p1["first_status"], p1["first_cause"], p1["plan_kind"], p1["finish_first"]) == ("no_code", "reasoning_cap", "thread_mapping", "length")
    assert p1["needed_repair"] and p1["repaired"] and p1["ok_final"] and p1["truncated_turns"] == 1 and p1["first_nocode_cost_usd"] == pytest.approx(0.058)
    p2 = by["r1.p2"]
    assert (p2["first_status"], p2["plan_kind"], p2["plan_scope"], p2["ok_final"]) == ("abi_error", "local", "local", True)
    assert p2["prompt_tokens_first"] == 9000 and p2["plan_chars"] > 20
    st = {a["label"]: a for a in attempts}
    assert st["r1.p1"]["status"] == "no_code" and st["r1.p1"]["cause"] == "reasoning_cap" and st["r1.p1.fix1"]["status"] == "ok"
    assert st["r1.p2"]["status"] == "abi_error" and st["r1.p2"]["detail"].startswith("static threadgroup memory N B exceeds")


def test_a_request_cut_by_the_budget_stop_is_not_blamed_on_the_model(tmp_path):
    w = Writer(tmp_path, "cut")
    w.chat("init#t0", content="```metal\n" + SEED + "\n```")
    w.cand("init", "ok", 1.0)
    w.chat("r1.plan", content="{}", last_user=STRUCT_PLAN_PROMPT)
    w.code_reply("r1.p0#t0", last_user=patch_prompt("Double-buffer K/V staging", "stage in threadgroup memory"))
    w.cand("r1.p0", "ok", 1.2)
    w.chat("r1.p1#t0", finish="tool_calls", tool_calls=1, last_user=patch_prompt("Split keys across SIMD groups", "split across simd groups"))  # never answered
    w.finish(stop="budget_exhausted")
    run = pta.load_run(w.dir, "cut", "runs")
    branches, _ = pta.build_branches(run, pta.DEFAULT_SIZE_EDGES)
    p1 = next(b for b in branches if b["base"] == "r1.p1")
    assert (p1["first_status"], p1["first_cause"]) == ("no_code", "run_ended")


def test_partial_run_and_half_written_files_do_not_crash_and_are_reported_partial(tmp_path):
    w = Writer(tmp_path, "live")
    w.chat("init#t0", content="```metal\n" + SEED + "\n```")
    w.cand("init", "ok", 1.0)
    w.chat("r1.plan", content="{}", last_user=STRUCT_PLAN_PROMPT)
    w.code_reply("r1.p0#t0", last_user=patch_prompt("Double-buffer K/V staging", "stage in threadgroup memory"))  # no candidate yet: still being evaluated
    d = w.finish(stop="", partial_tail='{"event": "chat", "label": "r1.p1#t0", "mess')
    (d / "candidates" / "009_half").mkdir()
    (d / "candidates" / "009_half" / "report.json").write_text("")  # being written
    (d / "state.json").write_text("{not json")
    data = pta.analyse_paths([str(tmp_path)], now=0.0)
    assert data["meta"]["n_runs"] == 1 and data["runs"][0]["finished"] is False
    assert data["runs"][0]["unreadable_candidates"] == 1  # the empty report.json of the half-written candidate
    pend = [b for b in data["branches"] if b["base"] == "r1.p0"]
    assert pend and pend[0]["first_status"] == "pending"  # an unevaluated request of a live run is not counted as a failure
    assert data["meta"]["pending_branches_excluded"] == 1
    assert "partial" in pta.render_report(data)


# ----------------------------------------------------------------------------------------------------------------------
# tables and report over a small collection of runs
# ----------------------------------------------------------------------------------------------------------------------
@pytest.fixture()
def data(tmp_path):
    root = tmp_path / "runs"
    root.mkdir()
    build_fa_run(root, "fa_a")
    build_fa_run(root, "fa_b")
    # a plan_mix run in which the third plan is the designated local one and the classifier must agree
    w = Writer(root, "fa_mix")
    w.chat("init#t0", content="```metal\n" + SEED + "\n```")
    w.cand("init", "ok", 1.0)
    w.chat("r1.plan", content="{}", last_user=MIXED_PLAN_PROMPT)
    for i, (title, change, status) in enumerate([
        ("Ping-pong staged K/V with a double buffer", "staging in threadgroup memory, double buffer", "ok"),
        ("Split keys across SIMD groups", "split across simd groups", "abi_error"),
        ("Local instruction diet", "Local, structure-preserving: hoist invariant index math", "ok"),
    ]):
        w.code_reply(f"r1.p{i}#t0", last_user=patch_prompt(title, change))
        w.cand(f"r1.p{i}", status, 1.1 if status == "ok" else None, abi="static threadgroup memory 40000 B exceeds the device limit of 32768 B" if status == "abi_error" else None)
    w.finish()
    # a memory-bound run where nothing is cut
    m = Writer(root, "relu_run", task="relu")
    m.chat("init#t0", content="```metal\n" + SEED + "\n```")
    m.cand("init", "ok", 1.0)
    m.chat("r1.plan", content="{}", last_user=STRUCT_PLAN_PROMPT)
    for i, title in enumerate(["Two independent 16-byte vectors in flight per thread", "Cheaper activation: exp2 + reciprocal", "Hoist the address math"]):
        m.code_reply(f"r1.p{i}#t0", last_user=patch_prompt(title, title))
        m.cand(f"r1.p{i}", "ok", 1.0 + 0.01 * i)
    m.finish()
    return pta.analyse_paths([str(root)])


def test_first_attempt_outcome_by_task_and_scope(data):
    k = data["kinds"]
    fa = k["by_task_scope"]["flash_attention"]
    # fa_a and fa_b: p0 ok (structural) / p1 no_code (structural) / p2 abi (local);  fa_mix: p0 ok, p1 abi (structural) / p2 ok (local)
    assert fa["structural"]["n"] == 6 and fa["structural"]["ok"] == 3 and fa["structural"]["no_code"] == 2 and fa["structural"]["abi_error"] == 1
    assert fa["local"]["n"] == 3 and fa["local"]["ok"] == 1 and fa["local"]["abi_error"] == 2 and fa["local"]["no_code"] == 0
    assert fa["structural"]["ok_final"] == 5  # both cut requests are repaired, the abi request of fa_mix is not
    assert sum(k["by_task_scope"]["relu"][s]["n"] for s in k["by_task_scope"]["relu"]) == 3
    assert k["by_task_scope"]["relu"]["local"]["n"] >= 2  # 'Cheaper activation', 'Hoist ...'
    assert k["by_task_kind"]["flash_attention"]["thread_mapping"]["no_code"] == 2


def test_classifier_is_checked_against_the_plan_mix_designation(data):
    v = data["kinds"]["validation"]
    assert v["n"] == 3 and v["agree"] == 3 and v["confusion"] == {"local->local": 1, "structural->structural": 2}
    assert v["disagreements"] == []


def test_default_prompt_yields_local_plans_anyway_on_memory_bound_ops(data):
    dp = data["kinds"]["default_prompt"]
    assert dp["memory-bound (elementwise/reduction/gemv/...)"]["scope"]["local"] == 2
    assert dp["flash_attention"]["n"] == 6  # fa_a + fa_b (fa_mix is a plan_mix run)


def test_local_vs_structural_tests_and_abi_by_kind(data):
    t = data["kinds"]["tests_fa"]
    assert t["ok"]["local"] == [1, 3] and t["ok"]["structural"] == [3, 6]
    assert t["ok given code"]["structural"] == [3, 4]  # the two cut requests have no kernel
    assert t["no_code"]["structural"] == [2, 6] and t["no_code"]["local"] == [0, 3]
    ab = data["kinds"]["fa_abi_by_kind"]
    assert ab["local"] == {"abi": 2, "threadgroup_memory": 2} and ab["thread_mapping"] == {"abi": 1, "threadgroup_memory": 1}  # the fixtures' abi message is 'static threadgroup memory ...'
    assert sum(v["abi"] for v in ab.values()) == 3


def test_nocode_tables_cause_cost_and_mechanism(data):
    n = data["nocode"]
    assert n["cap_tokens"] == CAP and n["length_turns"] == 2
    assert n["length_cost"] == pytest.approx(0.116)
    mech = n["mechanism"]
    assert mech["truncated"]["n"] == 2 and mech["truncated"]["pure_thinking"] == 2 and mech["truncated"]["reasoning_share_median"] == 1.0
    assert mech["code_replies"]["visible_tokens"]["p50"] == 2500 and mech["code_replies"]["visible_tokens"]["max"] == 2500  # completion 3000 - reasoning 500
    runs = {r["run"]: r for r in n["per_run"]}
    assert runs["runs/fa_a"]["no_code_first"] == 1 and runs["runs/fa_a"]["causes"] == {"reasoning_cap": 1} and runs["runs/fa_a"]["length_turns"] == 1
    assert runs["runs/relu_run"]["length_turns"] == 0
    rec = n["recovery_fa"]
    assert rec["n"] == 2 and rec["ok_final"] == 2
    flash = [d for d in n["first_turn_reasoning"] if d["group"] == "flash_attention" and d["tg"] == "patch"][0]
    assert flash["length"] == 2 and flash["n"] >= 6


def test_predictor_tables_run_on_small_pools(data):
    pr = data["nocode"]["predictors"]["flash_attention (deepseek-flash)"]
    assert pr["truncated"] == 2 and set(pr["features"]) >= {"plan text (characters)", "round"}
    assert pr["by_round"]["rounds 1-2"][0] == 2


def test_report_has_the_answer_first_and_all_sections(data):
    md = pta.render_report(data)
    heads = [ln for ln in md.splitlines() if ln.startswith("## ")]
    assert heads[0] == "## Answer" and any(h.startswith("## 1. Plan kind") for h in heads) and any(h.startswith("## 2. Replies without code") for h in heads)
    assert [h.split(".")[0] for h in heads[1:]] == ["## 0", "## 1", "## 2", "## 3", "## 4", "## 5"]
    for needle in ("reasoning_cap", "finish_reason=length", "Root cause", "no_code", "ok given code", "thread_mapping", "Proposal, nothing implemented"):
        assert needle in md
    answer = md.split("## 0.")[0]
    # the answer states the numbers it is computed from: 2 of the 9 FlashAttention patch requests had no kernel at the first attempt
    assert "2 of 9 FlashAttention patch requests (22%)" in answer
    assert md.count("#") > 20 and not re.search(r"\bnan\b", md, re.I) and "None" not in md  # no undefined number leaked into the text


def test_data_json_is_serialisable_and_has_no_nan(data):
    text = json.dumps(data, default=str)
    assert "NaN" not in text and json.loads(text)["meta"]["n_runs"] == 4


# ----------------------------------------------------------------------------------------------------------------------
# log cross-check, CLI, read-only guarantee
# ----------------------------------------------------------------------------------------------------------------------
def test_log_cross_check_matches_the_no_code_lines(tmp_path):
    root = tmp_path / "runs"
    root.mkdir()
    build_fa_run(root, "fa_run")
    log = tmp_path / "exp.log"
    log.write_text(
        "##### fa_run: flash_attention fp16\n== flash_attention fp16 paper | model deepseek-flash | budget $3.00 | branches 3 | run fa_run\n"
        "  [init] ok            speedup   1.000x\n  [r1.p1] no code block in reply\n  [r1.p9] no code block in reply\n"
    )
    d = pta.analyse_paths([str(root)], logs=[str(log)])
    (chk,) = d["log_check"]
    assert chk["run"] == "runs/fa_run" and chk["matched"] == ["r1.p1"] and chk["only_in_log"] == ["r1.p9"] and chk["only_in_transcript"] == []
    md = pta.render_report(d)
    assert "Cross-check" in md and "agree with this classification (1 of 2 lines matched" in md.split("## 0.")[0]
    assert pta.analyse_paths([str(root)])["log_check"] == []


def test_cli_writes_report_and_data_and_never_touches_the_runs(tmp_path, capsys):
    root = tmp_path / "runs"
    root.mkdir()
    build_fa_run(root, "fa_run")

    def snapshot():
        return {str(p.relative_to(root)): (p.stat().st_size, p.stat().st_mtime_ns) for p in sorted(root.rglob("*")) if p.is_file()}

    before = snapshot()
    out = tmp_path / "out"
    assert pta.main([str(root), "--out", str(out), "--note", "inputs are a frozen copy"]) == 0
    assert snapshot() == before
    assert (out / "REPORT.md").read_text().startswith("# Plan-type evidence report") and "_inputs are a frozen copy_" in (out / "REPORT.md").read_text() and json.loads((out / "data.json").read_text())["meta"]["n_runs"] == 1
    assert "1 runs" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        pta.main([str(root), "--size-edges", "0.5,0.2"])


def test_roots_may_be_globs_and_nested_and_missing_paths_are_ignored(tmp_path):
    root = tmp_path / "runs"
    (root / "sub").mkdir(parents=True)
    build_fa_run(root / "sub", "fa_run")
    found = pta.discover([str(tmp_path / "runs" / "s*"), str(tmp_path / "does_not_exist")])
    assert [label for _, label, _ in found] == ["sub/fa_run"]


def test_nothing_to_analyse_still_renders(tmp_path):
    d = pta.analyse_paths([str(tmp_path)])
    md = pta.render_report(d)
    assert d["meta"]["n_runs"] == 0 and "## Answer" in md
    assert math.isfinite(d["nocode"]["length_cost"]) and d["nocode"]["cap_tokens"] is None
