"""Experiment 3 (paper Section 4.1): does the compiler follow the *code*, not names, comments or familiar patterns?

  * ``sem_int_<seed>``: random ~130-line integer Triton kernels implementing no recognizable algorithm; bit-exact comparison.
  * four misleading-cue kernels: a matmul accumulating with ``-=``, a kernel named/commented GELU that computes ReLU, one
    named/commented ReLU that computes GELU, and FlashAttention with qk_scale = SM_SCALE + log2(e).

Each test is a pure correctness judgement (quick evaluation, no benchmark): *first try* = the very first generated kernel is
correct; *within repairs* = correct after at most ``--repairs`` compile/test feedback turns.

    uv run python -m experiments.semantic_tests --n-random 10
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from observe.publish import job_state

from .common import read_result, run_one

REPO_ROOT = Path(__file__).resolve().parent.parent
MISLEADING = ("sem_gemm_neg", "sem_relu_gelu_comment", "sem_gelu_named_relu", "sem_attention_odd_scale")


def run_semantic(names: list[str], out_dir: Path, *, dtype="fp16", size="small", repairs=3, budget=0.15, model=None, resume=True, client_factory=None, evaluate_fn=None, log=print, **overrides) -> list[dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    names = list(names)
    for i, name in enumerate(names):
        job_state(out_dir / "state.json", script="semantic_tests", current={"name": name}, queue=[{"name": n} for n in names[i + 1 :]], done=rows)
        rd = out_dir / name
        res = read_result(rd) if resume else None
        if res is None:
            log(f"\n##### {name}")
            run_one(name, dtype, size, rd, budget=budget, max_rounds=0, model=model, client=client_factory() if client_factory else None, evaluate_fn=evaluate_fn, log=log, correctness_only=True, max_repair_tries=repairs, branches=1, **overrides)
            res = read_result(rd)
        cands = res["candidates"]
        first = cands[0]["status"] if cands else "none"
        ok_idx = next((i for i, c in enumerate(cands) if c["status"] == "ok"), None)
        rows.append(
            {
                "name": name, "first_try_ok": first == "ok", "first_status": first, "ok_within_repairs": ok_idx is not None,
                "attempts_until_ok": None if ok_idx is None else ok_idx + 1, "statuses": [c["status"] for c in cands], "spent_usd": res["spent_usd"],
                "messages": [c["message"] for c in cands[:2] if c["status"] != "ok"],
            }
        )
        (out_dir / "semantic.json").write_text(json.dumps(rows, indent=1))
    job_state(out_dir / "state.json", script="semantic_tests", current=None, queue=[], done=rows, phase="done")
    return rows


def markdown(rows: list[dict]) -> str:
    L = ["| test | first try | within repairs | attempts | statuses | USD |", "| --- | :---: | :---: | ---: | --- | ---: |"]
    for r in rows:
        L.append(f"| {r['name']} | {'yes' if r['first_try_ok'] else 'no'} | {'yes' if r['ok_within_repairs'] else 'no'} | {r['attempts_until_ok'] or '-'} | {', '.join(r['statuses'])} | {r['spent_usd']:.3f} |")
    n = len(rows)
    L.append(f"\nFirst try: {sum(r['first_try_ok'] for r in rows)}/{n}; within repair budget: {sum(r['ok_within_repairs'] for r in rows)}/{n}; total ${sum(r['spent_usd'] for r in rows):.3f}")
    return "\n".join(L)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-random", type=int, default=10)
    ap.add_argument("--seed0", type=int, default=1)
    ap.add_argument("--no-misleading", action="store_true")
    ap.add_argument("--repairs", type=int, default=3)
    ap.add_argument("--budget", type=float, default=0.15)
    ap.add_argument("--dtype", default="fp16")
    ap.add_argument("--model", default=None)
    ap.add_argument("--out-dir", default=None)
    a = ap.parse_args(argv)
    names = [f"sem_int_{a.seed0 + i}" for i in range(a.n_random)] + ([] if a.no_misleading else list(MISLEADING))
    out = Path(a.out_dir) if a.out_dir else REPO_ROOT / "runs" / f"semantic_{time.strftime('%Y%m%d-%H%M%S')}"
    rows = run_semantic(names, out, dtype=a.dtype, repairs=a.repairs, budget=a.budget, model=a.model)
    md = markdown(rows)
    (out / "semantic.md").write_text(md + "\n")
    print("\n" + md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
