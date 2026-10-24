"""Reusable resume/CLI helpers shared by tests."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from .fakes import make_response, make_usage
from .orchestrator import code


def ns(run_dir: Path, **kw) -> argparse.Namespace:
    """@brief 造一个覆盖 ``taic resume`` 全部命令行参数的命名空间，默认值取“什么都不改”。

    @param run_dir 运行目录。
    @param kw 需要覆盖的字段，例如 ``add_budget`` 或 ``max_rounds``。
    @return 填好默认值再叠加 ``kw`` 的 ``argparse.Namespace``。
    """
    base = dict(run_dir=str(run_dir), add_budget=None, budget=None, dry_run=False, who="cli", config=None, preset=None, model=None, branches=None, max_rounds=None,
                effort=None, no_tools=False, profile_best=False, profile_poor=False, plan_mix=False, init_samples=None, patience=None,
                lift_count_limits=False, accept_cost_gap=False)
    base.update(kw)
    return argparse.Namespace(**base)

def script_for(counter: dict | None = None, usage=None):
    """@brief 造一个内容驱动的脚本化模型：答复只取决于提示词，与调用顺序无关。

    @param counter 可选的计数器字典；置 ``kill_at`` 后在第 N 次调用里抛出 ``SystemExit`` 模拟进程被杀。
    @param usage 可选的用量对象，未给时用 ``make_usage()``。
    @return 接收消息列表的 ``route`` 函数，供假客户端调用。
    """
    pat = {"A": ["status=compile_error", "speed=1.4"], "B": ["speed=1.2"], "C": ["status=incorrect", "status=incorrect", "status=incorrect"]}

    def route(messages, **kw):
        if counter is not None:
            counter["n"] = counter.get("n", 0) + 1
            if counter.get("kill_at") == counter["n"]:
                raise SystemExit("killed")
        u = usage or make_usage()
        last = messages[-1]["content"]
        if "Translate this Triton kernel" in last:
            return make_response(code("speed=1.0"), usage=u)
        if "Diagnose the dominant bottleneck" in last:
            rnd = max([int(m) for m in re.findall(r"^- round (\d+):", last, re.M)] or [0]) + 1  # distinct plan titles in every round
            plans = [{"title": f"{t}{rnd}", "change": f"do {t}"} for t in "ABC"]
            return make_response("```json\n" + json.dumps({"bottleneck": "m", "analysis": "a", "plans": plans}) + "\n```", usage=u)
        if "Implement exactly this optimization plan" in last:
            title = re.search(r"\*\*(.+?)\*\*", last).group(1)
            return make_response(code(f"{pat[title[0]][0]} // {title}"), usage=u)
        if "evaluation of your kernel failed" in last:
            title = re.search(r"\*\*(\w\d+)\*\*", messages[1]["content"]).group(1)
            n = int(re.search(r"repair attempt (\d+)/", last).group(1))
            seq = pat[title[0]]
            return make_response(code(f"{seq[min(n, len(seq) - 1)]} // {title}"), usage=u)
        raise AssertionError(last[:80])

    return route

def mark_dead(run_dir: Path) -> None:
    """@brief 让测试进程扮演被杀的进程：改掉 ``state.json`` 里的 pid，使下一个进程看到它已死。

    @param run_dir 运行目录，其中的 ``state.json`` 会被重写。
    """
    st = json.loads((run_dir / "state.json").read_text())
    st["pid"] = 2**22 + 4242
    (run_dir / "state.json").write_text(json.dumps(st))
