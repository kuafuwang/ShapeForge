"""Derived evaluation seeds. The golden vector is recomputed in this file, not copied from the implementation. No GPU."""

from __future__ import annotations

import asyncio
import hashlib
import json

import pytest

from taic import cli
from taic.llm import LLMConfig, SearchConfig
from taic.orchestrator import Orchestrator, RunResult, RunStartError, eval_seed
from tcenv.evaluate import EvalReport

from ..support.fakes import FakeClient


SOURCE = "kernel void k() {}\n"
GOLDEN_SHA = "ae97b4b6b017756328f7bbf1c39c8a871cf9f4857b583abcc236fce00e90d6e9"
GOLDEN_SEED = 96049487906312


def _recompute(run_seed: int, phase: str, label: str, source: str) -> int:
    """@brief 在本文件内独立重算 v1 评测种子，不调用被测实现。

    对 ``["ai_lowerling/eval_seed/v1", run_seed, phase, label, source_sha256]`` 的紧凑 JSON
    取 sha256，再取前 6 字节转大端整数。

    @param run_seed 本次运行的种子。
    @param phase 阶段名，如 search 或 confirm。
    @param label 候选标签，如 r1.p0。
    @param source 内核源码。
    @return 重算出的整数种子。
    """
    source_sha256 = hashlib.sha256(source.encode("utf-8")).hexdigest()
    payload = json.dumps(["ai_lowerling/eval_seed/v1", run_seed, phase, label, source_sha256], ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:6], "big")


def test_the_v1_golden_vector_matches_an_independent_recomputation():
    """@brief 验证 v1 金标准向量与文件内独立重算一致，且 confirm 阶段落在不同种子上。"""
    assert hashlib.sha256(SOURCE.encode("utf-8")).hexdigest() == GOLDEN_SHA
    assert _recompute(1, "search", "init", SOURCE) == GOLDEN_SEED
    assert eval_seed(1, "search", "init", SOURCE) == GOLDEN_SEED
    assert eval_seed(1, "confirm", "init", SOURCE) != GOLDEN_SEED
    assert eval_seed(1, "search", "init", SOURCE) == eval_seed(1, "search", "init", SOURCE)


def _cfg():
    """@brief 构造本文件测试用的固定模型配置与搜索配置。

    @return (LLMConfig, SearchConfig) 二元组。
    """
    cfg = LLMConfig(base_url="http://x", api_key="sk-test-0123456789abcdef", model="fake", reasoning_effort="off", use_tools=False)
    search = SearchConfig(budget_usd=5.0, branches=1, max_rounds=1, eval_reps=3, max_hangs=3, stop_at_roofline=False, abi_guard=False, target_note=False, rich_history=False, status_lines=False)
    return cfg, search


def _orch(tmp_path, name, *, seed=1, evaluate_fn, resuming=False):
    """@brief 在给定子目录下创建一个无脚本化回复的 Orchestrator，评测函数可注入。

    @param tmp_path pytest 提供的临时目录。
    @param name 运行子目录名。
    @param seed 运行种子。
    @param evaluate_fn 替换真实评测的注入函数。
    @param resuming 是否为续跑模式。
    @return 构造好的 Orchestrator。
    """
    cfg, search = _cfg()
    return Orchestrator("relu", "fp16", "small", cfg, search, run_dir=tmp_path / name, client=FakeClient([]), evaluate_fn=evaluate_fn, log=lambda _s: None, seed=seed, resuming=resuming)


def _ok(task, dtype, size, source, *, seed, mode):
    """@brief 造一份状态为 ok、含速度与基准数据的 EvalReport，供种子相关用例使用。

    @param task 任务名。
    @param dtype 数据类型。
    @param size 规模名。
    @param source 内核源码（仅用于签名兼容，未写入报告）。
    @param seed 评测种子，None 时记为 0。
    @param mode 评测模式。
    @return 填好速度、候选耗时与基准数据的 EvalReport。
    """
    report = EvalReport(task=task, dtype=dtype, size=size, mode=mode, seed=0 if seed is None else int(seed), status="ok", stage="done", speedup=1.2)
    report.speedups, report.candidate_us, report.baseline_us = {"mlx": 1.2}, 10.0, {"mlx": 12.0}
    report.benchmark = {"candidate": {"median_us": 10.0, "cv": 0.01, "reps": 3}, "baselines": {"mlx": {"median_us": 12.0, "cv": 0.01, "reps": 3}}, "best_baseline": "mlx"}
    return report


def test_tag_seeds_do_not_depend_on_which_branch_finishes_first(tmp_path):
    """@brief 验证按标签派生的种子与分支完成顺序无关，乱序评测得到同样的种子。

    @param tmp_path pytest 提供的临时目录。
    """
    pairs = [("r1.p0", "kernel void k() { /* A */ }\n"), ("r1.p1", "kernel void k() { /* B */ }\n")]

    async def collect(order, name):
        seen = []

        def ev(task, dtype, size, source, *, mode="full", seed=None, reps=15, **_kw):
            seen.append((source, seed, mode))
            return _ok(task, dtype, size, source, seed=seed, mode=mode)

        orch = _orch(tmp_path, name, evaluate_fn=ev)
        for label, source in order:
            await orch._eval(source, label)
        return {row["label"]: row["eval_seed"] for row in orch.trace}, seen

    forward, seen_f = asyncio.run(collect(pairs, "forward"))
    backward, seen_b = asyncio.run(collect(list(reversed(pairs)), "backward"))
    assert forward == backward
    assert forward["r1.p0"] == eval_seed(1, "search", "r1.p0", pairs[0][1])
    assert [seed for _src, seed, _mode in seen_f] == [seed for _src, seed, _mode in reversed(seen_b)]


def test_an_identical_patch_does_not_count_as_another_kernel(tmp_path):
    """@brief 验证与当前种子完全相同的补丁不触发评测、不产生新种子，并在轨迹里标记为未变。

    @param tmp_path pytest 提供的临时目录。
    """
    source = "kernel void k() { /* same */ }\n"
    calls = []

    def ev(task, dtype, size, src, *, mode="full", seed=None, reps=15, **_kw):
        calls.append({"seed": seed, "mode": mode, "reps": reps})
        return _ok(task, dtype, size, src, seed=seed, mode=mode)

    orch = _orch(tmp_path, "run", evaluate_fn=ev)

    async def go():
        return await orch._eval(source, "r1.p0"), await orch._eval(source, "r1.p1")

    first, second = asyncio.run(go())
    assert first.report.seed == eval_seed(1, "search", "r1.p0", source)
    assert orch.trace[1]["eval_seed_from"] == "unchanged"
    assert orch.trace[1]["eval_seed"] == 0 and orch.trace[1]["status"] == "unchanged"
    assert len([c for c in calls if c["mode"] == "full" and c["reps"] == 3]) == 1


def test_confirm_uses_a_different_seed_and_a_cache_hit_keeps_the_old_one(tmp_path):
    """@brief 验证确认阶段使用与搜索不同的种子，且非种子的缓存命中仍复用旧报告与旧种子。

    @param tmp_path pytest 提供的临时目录。
    """
    seed_source = "kernel void k() { /* seed */ }\n"
    other = "kernel void k() { /* other, evaluated once */ }\n"
    calls = []

    def ev(task, dtype, size, src, *, mode="full", seed=None, reps=15, **_kw):
        calls.append({"seed": seed, "mode": mode, "reps": reps})
        return _ok(task, dtype, size, src, seed=seed, mode=mode)

    orch = _orch(tmp_path, "run", evaluate_fn=ev)

    async def go():
        first = await orch._eval(seed_source, "r1.p0")
        await orch._eval(other, "r1.p1")       # 评一次，进缓存；它不是当前种子
        second = await orch._eval(other, "r1.p2")  # 命中缓存
        res = RunResult(run_dir=str(orch.run_dir), task="relu", dtype="fp16", size="small", model="fake", budget_usd=5.0)
        await orch._confirm(res)
        return first, second, res

    first, second, res = asyncio.run(go())
    assert first.report.seed == eval_seed(1, "search", "r1.p0", seed_source)
    assert orch.trace[2]["eval_seed_from"] == "duplicate"
    assert orch.trace[2]["eval_seed"] == 0
    assert len([c for c in calls if c["mode"] == "full" and c["reps"] == 3]) == 2
    assert res.confirmations[0]["eval_seed_from"] == "confirm"
    assert res.confirmations[0]["eval_seed"] == eval_seed(1, "confirm", "b1", "")
    assert res.confirmations[0]["eval_seed"] != orch.trace[0]["eval_seed"]
    assert second.label == "r1.p2"  # 第三次请求复用第一次的评测结论


def test_the_same_label_and_source_keep_the_seed_across_a_retry(tmp_path):
    """@brief 验证同一标签同一源码在重试时使用相同种子，保证重试结果可比。

    @param tmp_path pytest 提供的临时目录。
    """
    source = "kernel void k() { /* retry */ }\n"
    seeds = []

    def ev(task, dtype, size, src, *, mode="full", seed=None, reps=15, **_kw):
        seeds.append(seed)
        report = EvalReport(task=task, dtype=dtype, size=size, mode=mode, seed=0 if seed is None else int(seed), status="timeout", stage="timeout", message="slow")
        return report

    orch = _orch(tmp_path, "retry", evaluate_fn=ev)

    async def go():
        await orch._eval(source, "r2.p1.fix1")
        await orch._eval(source, "r2.p1.fix1")

    asyncio.run(go())
    assert seeds == [eval_seed(1, "search", "r2.p1.fix1", source)] * 2


def test_quick_check_stays_random_and_an_old_run_keeps_seed_none(tmp_path):
    """@brief 验证续跑旧运行时保持 run_config 中的 seed 为 None，不会补画种子。

    @param tmp_path pytest 提供的临时目录。
    """
    seen = []

    def ev(task, dtype, size, src, *, mode="full", seed=None, reps=15, **_kw):
        seen.append(seed)
        report = EvalReport(task=task, dtype=dtype, size=size, mode=mode, seed=12345, status="ok", stage="done", message="ok")
        return report

    old = tmp_path / "old"
    old.mkdir()
    (old / "marker.txt").write_text("stay")
    (old / "run_config.json").write_text(json.dumps({"schema_version": 1, "seed": None, "task": "relu"}))
    resumed = _orch(tmp_path, "old", evaluate_fn=ev, resuming=True, seed=99)
    assert resumed.run_seed is None
    assert json.loads((old / "run_config.json").read_text())["seed"] is None


def test_a_new_run_records_the_given_seed_and_resume_does_not_draw_another(tmp_path, monkeypatch):
    """@brief 验证新运行记录抽取的种子、续跑不重新抽取，并拒绝越界种子。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 补丁工具，用来固定 secrets.randbits 的返回值。
    @raises RunStartError 传入的 seed 超出 u64 范围时抛出。
    @raises SystemExit CLI 收到负数 --seed 时退出。
    """
    monkeypatch.setattr("taic.orchestrator.secrets.randbits", lambda n: 42)
    cfg, search = _cfg()

    def ev(*_a, **_k):
        raise AssertionError("no evaluation")

    fresh = Orchestrator("relu", "fp16", "small", cfg, search, run_dir=tmp_path / "fresh", client=FakeClient([]), evaluate_fn=ev, log=lambda _s: None)
    assert json.loads((fresh.run_dir / "run_config.json").read_text())["run_seed"] == 42
    again = Orchestrator("relu", "fp16", "small", cfg, search, run_dir=fresh.run_dir, client=FakeClient([]), evaluate_fn=ev, log=lambda _s: None, resuming=True)
    assert again.run_seed == 42
    with pytest.raises(RunStartError):
        Orchestrator("relu", "fp16", "small", cfg, search, run_dir=tmp_path / "bad", client=FakeClient([]), evaluate_fn=ev, log=lambda _s: None, seed=2**64)
    with pytest.raises(SystemExit):
        cli.main(["run", "relu", "--seed", "-1"])
    assert cli._u64("0") == 0 and cli._u64(str(2**64 - 1)) == 2**64 - 1
    with pytest.raises(Exception):
        cli._u64(str(2**64))
