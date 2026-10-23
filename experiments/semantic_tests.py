"""Experiment 3 (paper Section 4.1): does the compiler follow the *code*, not names, comments or familiar patterns?

  * ``sem_int_<seed>``: random ~130-line integer Triton kernels implementing no recognizable algorithm; bit-exact comparison.
  * four misleading-cue kernels: a matmul accumulating with ``-=``, a kernel named/commented GELU that computes ReLU, one
    named/commented ReLU that computes GELU, and FlashAttention with qk_scale = SM_SCALE + log2(e).

Each test is a pure correctness judgement (quick evaluation, no benchmark): *first try* = the very first generated kernel is
correct; *within repairs* = correct after at most ``--repairs`` compile/test feedback turns.

    uv run python -m experiments.semantic_tests --n-random 10
"""

# 论文第 4.1 节：模型有没有按计算步骤翻译，还是看见函数名、注释里的 GELU / ReLU / GEMM 就套熟悉公式。
# 随机整数核没有可认出的算法；四道误导核的名字和注释与真正的算式相反。这里记录的是对错，不是加速比。

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from tcenv.events import install_file_log, job_state

from .common import read_result, run_one

REPO_ROOT = Path(__file__).resolve().parent.parent
# 名字和注释故意不可信的四道固定题。随机整数核由 --n-random 另生成，不在这个元组里。
MISLEADING = ("sem_gemm_neg", "sem_relu_gelu_comment", "sem_gelu_named_relu", "sem_attention_odd_scale")


# 判的是正确性。max_rounds=0 时改进轮的循环是空的，只留第一次生成和随后最多 --repairs 次编译反馈。
# correctness_only 走快速评测、不做基准，所以「第一次就对」只看首个候选的 status 是不是 ok。
def run_semantic(names: list[str], out_dir: Path, *, dtype="fp16", size="small", repairs=3, budget=0.15, model=None, resume=True, client_factory=None, evaluate_fn=None, log=print, **overrides) -> list[dict]:
    """@brief 逐道跑语义题，只判对错、不测速，返回每题的结果行。

    @param names 任务名列表，含随机整数核与误导核。
    @param out_dir 本次实验的输出目录。
    @param dtype 数据类型，默认 ``fp16``。
    @param size 规模档位，默认 ``small``。
    @param repairs 允许的编译/测错反馈次数上限。
    @param budget 每道题的花费上限，单位美元。
    @param model 模型名；为 ``None`` 时用默认。
    @param resume 为真时复用已有 ``result.json``，不重跑。
    @param client_factory 造 LLM 客户端的可调用对象；为 ``None`` 时不显式传入。
    @param evaluate_fn 评测函数；为 ``None`` 时用默认实现。
    @param log 进度输出函数。
    @param overrides 其余透传给 ``run_one`` 的覆盖项。
    @return 每道题的``首次就对``、``修复预算内就对``、各候选状态与花费。
    """
    install_file_log()
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    names = list(names)
    for i, name in enumerate(names):
        job_state(out_dir / "state.json", script="semantic_tests", current={"name": name}, queue=[{"name": n} for n in names[i + 1 :]], done=rows)
        rd = out_dir / name
        res = read_result(rd) if resume else None
        if res is None:
            log(f"\n##### {name}")
            # 0.15 美元只覆盖对错判断。max_rounds=0 时改进轮不会发生，候选里剩下初次生成和它的修复。
            run_one(name, dtype, size, rd, budget=budget, max_rounds=0, model=model, client=client_factory() if client_factory else None, evaluate_fn=evaluate_fn, log=log, correctness_only=True, max_repair_tries=repairs, branches=1, **overrides)
            res = read_result(rd)
        cands = res["candidates"]
        first = cands[0]["status"] if cands else "none"
        # 修复预算内算对：任一候选 ok 即可。attempts 从 1 起算，对上「第几次才对」。
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


# 两列「第一次」和「修复预算内」分开，是为了看出模型是一次读对了算式，还是靠编译报错才改回来。
def markdown(rows: list[dict]) -> str:
    """@brief 把语义题结果拍成 markdown 表，并附两列汇总。

    @param rows ``run_semantic`` 返回的结果行。
    @return 表格文本。
    """
    L = ["| test | first try | within repairs | attempts | statuses | USD |", "| --- | :---: | :---: | ---: | --- | ---: |"]
    for r in rows:
        L.append(f"| {r['name']} | {'yes' if r['first_try_ok'] else 'no'} | {'yes' if r['ok_within_repairs'] else 'no'} | {r['attempts_until_ok'] or '-'} | {', '.join(r['statuses'])} | {r['spent_usd']:.3f} |")
    n = len(rows)
    L.append(f"\nFirst try: {sum(r['first_try_ok'] for r in rows)}/{n}; within repair budget: {sum(r['ok_within_repairs'] for r in rows)}/{n}; total ${sum(r['spent_usd'] for r in rows):.3f}")
    return "\n".join(L)


# 随机核的名字是 sem_int_加种子，每颗种子一份没有可认算法的整数程序。--no-misleading 时不跑那四道名字和注释骗人的题。
def main(argv=None) -> int:
    """@brief ``python -m experiments.semantic_tests`` 的入口，跑完写 ``semantic.json`` 与 ``semantic.md``。

    @param argv 命令行参数；为 ``None`` 时取 ``sys.argv``。
    @return 进程退出码，正常结束时为 0。
    """
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
    # 种子从 --seed0 起连续取。误导题的算式在任务定义里和名字相反：负号累加、注释写 GELU 实际是 ReLU、名字写 ReLU 实际是 GELU、注意力缩放写成相加。
    names = [f"sem_int_{a.seed0 + i}" for i in range(a.n_random)] + ([] if a.no_misleading else list(MISLEADING))
    out = Path(a.out_dir) if a.out_dir else REPO_ROOT / "runs" / f"semantic_{time.strftime('%Y%m%d-%H%M%S')}"
    rows = run_semantic(names, out, dtype=a.dtype, repairs=a.repairs, budget=a.budget, model=a.model)
    md = markdown(rows)
    (out / "semantic.md").write_text(md + "\n")
    print("\n" + md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
