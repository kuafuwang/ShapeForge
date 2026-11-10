"""Direction ledger and plateau text. CPU fixtures only; no search is started."""

from __future__ import annotations

from taic.kernel_agent import SessionBrief, render_contract
from taic.llm import SearchConfig
from taic.plan_ledger import classify_outcome, clip_ledger_text, format_ledger_for_planner, nodes_from_candidates
from tests.support.orchestrator import make_orch


class _Report:
    def __init__(self, status, speedup, message=""):
        self.status = status
        self.speedup = speedup
        self.message = message


class _Cand:
    def __init__(self, label, status, speedup, *, plan=None, within_noise=False, recheck_rejected=False, recheck_pending=False, message=""):
        self.label = label
        self.plan = plan
        self.report = _Report(status, speedup, message)
        self.within_noise = within_noise
        self.recheck_rejected = recheck_rejected
        self.recheck_pending = recheck_pending


def _brief(**kw) -> SessionBrief:
    base = dict(
        repo="/repo", task_name="conv2d", dtype="fp16", size="paper", kernel_name="conv2d_kernel",
        triton_source="def conv():\n    pass\n", description="a convolution", contract_text="grid 1,1,1",
        abi_stub="kernel void conv2d_kernel() {}", rtol=1e-3, atol_rel=1e-3, seedless=False,
        seed_source="kernel void conv2d_kernel() {}\n", reps=15, min_improvement=1.005, noise_z=2.0,
        work_dir="/work/r1",
    )
    base.update(kw)
    return SessionBrief(**base)


def test_within_noise_is_a_validated_tie():
    """@brief ``status=ok``、没换种子、``within_noise=True`` 归为 ``validated_tie``。"""
    kind, hint = classify_outcome("ok", "within noise", improved=False, within_noise=True, speedup=1.01, parent_speedup=1.0)
    assert kind == "validated_tie" and hint == ""
    nodes = nodes_from_candidates([
        _Cand("init", "ok", 1.0, plan="seed family"),
        _Cand("r1.p0", "ok", 1.01, plan="nearby", within_noise=True),
    ])
    assert nodes[1].outcome_kind == "validated_tie"
    assert nodes[1].title == "nearby"


def test_recheck_rejection_is_not_a_timed_loss_and_a_missing_direction_is_named():
    """@brief 第二次种子没过的 ok 仍是未计时；没有方向的标题是未声明。"""
    nodes = nodes_from_candidates([
        _Cand("init", "ok", 1.0),
        _Cand("r2.p0", "ok", 1.4, recheck_rejected=True),
    ])
    assert nodes[0].title == "未声明"
    assert nodes[1].outcome_kind == "untested_impl_fail"
    assert nodes[1].constraint_hint == "a second seed failed correctness, so this direction was not timed as a loss"
    text = format_ledger_for_planner(nodes, limit=12)
    assert text.endswith(
        "Do not repeat a direction marked falsified. A direction marked untested_impl_fail was not measured; it may be retried with the hint."
    )


def test_task_ledger_replaces_the_five_history_lines_and_keeps_the_retry_sentence():
    """@brief 处理臂用账本替换最近 5 行；对照臂只有那 5 行。重试句与格式化函数的末句相同。"""
    nodes = nodes_from_candidates([
        _Cand("r1.p0", "ok", 1.2, plan="vector loads"),
        _Cand("r2.p0", "ok", 0.9, plan="scalar fallback"),
    ])
    ledger = format_ledger_for_planner(nodes, limit=12)
    history = "\n".join(f"round {i}: raw-history-{i} -> not better" for i in range(1, 6))
    control = render_contract(_brief(history=history, use_task_ledger=False))
    treatment = render_contract(_brief(history=history, use_task_ledger=True, ledger_text=ledger))
    assert "raw-history-1" in control and "validated_win" not in control
    assert "raw-history-1" not in treatment
    assert "vector loads -> validated_win" in treatment
    assert ledger.splitlines()[-1] in treatment


def test_plateau_adds_only_the_hypothesis_paragraph():
    """@brief ``plateau`` 为真时多一段假设；为假时这段不出现。"""
    on = render_contract(_brief(plateau=True, history="round 1: first -> 1.000x"))
    off = render_contract(_brief(plateau=False, history="round 1: first -> 1.000x"))
    assert "不在当前结构上微调" in on
    assert "假设是什么" in on and "依赖哪几条记录" in on and "什么观察会否定它" in on
    assert "算法族或启动几何" in on
    assert "不在当前结构上微调" not in off


def test_new_switches_stay_off_and_do_not_open_diversity_or_patience():
    """@brief 新开关默认关。``diversity_archive`` 仍关，``patience`` 仍是 6。"""
    search = SearchConfig()
    assert search.init_samples == 1
    assert search.task_ledger is False
    assert search.plateau_after == 0
    assert search.sweep_tool is False
    assert search.diversity_archive is False
    assert search.patience == 6


def test_plateau_flag_follows_stagnant_rounds(tmp_path):
    """@brief 只有 ``plateau_after > 0`` 且停滞轮数达到它时，下一场有种子的简报才带平台期。"""
    orch = make_orch(tmp_path, lambda messages, **kw: None, [], max_rounds=1)
    orch.search.plateau_after = 2
    orch._stagnant = 1
    assert orch._session_brief(False).plateau is False
    orch._stagnant = 2
    assert orch._session_brief(False).plateau is True
    orch.search.plateau_after = 0
    assert orch._session_brief(False).plateau is False
    assert orch._session_brief(True).plateau is False


def test_a_pending_recheck_does_not_become_a_win_or_move_the_parent():
    """@brief 1.0、未决 1.4、随后 1.2：未决保持未计时，1.2 仍相对 1.0 记成赢。"""
    nodes = nodes_from_candidates([
        _Cand("init", "ok", 1.0, plan="seed"),
        _Cand("r1.p0", "ok", 1.4, plan="pending family", recheck_pending=True),
        _Cand("r2.p0", "ok", 1.2, plan="later family"),
    ])
    assert [node.outcome_kind for node in nodes] == ["validated_win", "untested_impl_fail", "validated_win"]
    assert nodes[1].parent_speedup == 1.0
    assert nodes[2].parent_speedup == 1.0
    assert "did not finish" in nodes[1].constraint_hint
    assert nodes[2].outcome_kind != "falsified"


def test_a_long_ledger_keeps_the_newest_nodes_and_the_retry_footer():
    """@brief 12 条长方向在 900 字节内仍保留最后一轮和结尾重试句，不把一行截断。"""
    nodes = [
        _Cand(f"r{i}.p0", "ok", 1.0 + i / 100, plan=f"family-{i}-" + ("direction " * 12))
        for i in range(1, 13)
    ]
    full = format_ledger_for_planner(nodes_from_candidates(nodes), limit=12)
    clipped = clip_ledger_text(full, 900)
    assert len(clipped.encode("utf-8")) <= 900
    assert "round 12" in clipped
    assert "round 1:" not in clipped
    assert clipped.endswith(
        "Do not repeat a direction marked falsified. A direction marked untested_impl_fail was not measured; it may be retried with the hint."
    )
    for line in clipped.splitlines():
        assert line.startswith("- round ") or line.startswith("Do not repeat ")
    rendered = render_contract(_brief(use_task_ledger=True, ledger_text=full))
    assert "round 12" in rendered
    assert "may be retried with the hint." in rendered
    assert "本段已截断" not in rendered.split("最近的历史", 1)[-1]
