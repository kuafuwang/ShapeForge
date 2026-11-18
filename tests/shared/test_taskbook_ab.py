"""The task book names the in-window timing tool, and the two renderers agree.

Field feedback asked for one session to compare several variants inside a single GPU
window. ``tcenv ab`` does that, read-only, and ``eval-v2 --timed --control`` reports how
much difference that window can resolve. Both were implemented but never named in
``TASK.md``, so sessions kept hand-copying a probe. These tests pin the wording, the two
gates that decide whether the section may appear at all (``sweep_tool``, and a dtype the
CLI actually accepts), the byte equality of the renderer's copy and the delivery
receipt's copy, and that the prompt fingerprint really covers the prose. The ranking
itself is covered by the Metal tests.
"""

from __future__ import annotations

import contextlib
import io
import re
from pathlib import Path

import pytest

from taic.kernel_agent import AB_DTYPES, SessionBrief, _ab_command, render_contract
from taic.task_delivery import _sweep_suffix, build_receipt

ROOT = Path(__file__).resolve().parents[2]


def _brief(tmp_path: Path, *, sweep: bool = True, v2: bool = True, dtype: str = "fp16") -> SessionBrief:
    work = tmp_path / "session"
    return SessionBrief(
        repo=tmp_path, task_name="gemm", dtype=dtype, size="small", kernel_name="gemm_kernel",
        triton_source="", description="gemm", contract_text="grid 1,1,1",
        abi_stub="kernel void gemm_kernel() {}", rtol=1e-3, atol_rel=1e-3,
        reps=7, work_dir=str(work), v2=v2, sweep_tool=sweep,
    )


def _cli_dtype_choices(argv: list[str]) -> set[str]:
    """The dtype set a real ``tcenv`` subcommand parses, read from its own help text."""
    from tcenv.cli import main

    buffer = io.StringIO()
    with pytest.raises(SystemExit), contextlib.redirect_stdout(buffer):
        main(argv)
    found = re.search(r"--dtype \{([a-z0-9,]+)\}", buffer.getvalue())
    assert found, buffer.getvalue()[:400]
    return set(found.group(1).split(","))


def test_the_task_book_names_ab_with_its_readonly_boundary(tmp_path):
    """@brief ``sweep_tool`` 打开时任务书给出 ``ab`` 命令，并说清同窗口、只读和不占配额。"""
    brief = _brief(tmp_path, dtype="fp16")
    task = render_contract(brief)

    assert "# 同一段窗口里比较多个变体" in task
    assert _ab_command(brief) in task
    assert "tcenv.cli ab" in task and "--task" in task and "gemm" in task
    # 同一段窗口、只读、不占 --timed 配额，三条都不能只靠上下文猜。
    assert "同一段 GPU 窗口" in task
    assert "**只读**" in task and "不占 `--timed` 的 20 次/120 秒配额" in task
    assert "不是发布证据" in task
    # 正确性口径：算错的仍计时但不排序，不安全的根本不进计时循环。
    assert "标成不可排序、不进名次" in task
    assert "不安全的文件不进计时循环" in task
    # --control 的目的：同一段窗口的分辨率。
    assert "resolvable_relative" in task
    assert "同一份字节本该读 1.000" in task


def test_the_sweep_section_points_at_ab_instead_of_only_denying_one_process(tmp_path):
    """@brief 旧句子「这一版不在同一进程里交错计时」不再单独出现，改成指向 ``ab``。"""
    task = render_contract(_brief(tmp_path, dtype="fp16"))
    assert "这一版不在同一进程里交错计时。" not in task
    assert "这一版不把多个候选放进同一个进程里交错计时；要在一段窗口里排先后，用下面那条 `ab`。" in task
    # sweep-v2 那一段本身仍在，不能被 ab 段顶掉。
    assert "# 一条命令比较多个变体" in task
    assert "tcenv.cli sweep-v2" in task
    assert task.index("# 一条命令比较多个变体") < task.index("# 同一段窗口里比较多个变体")


def test_control_is_the_timed_flag_and_not_an_ab_argument(tmp_path):
    """@brief ``--control`` 写清属于 ``eval-v2 ... --timed``，不能被读成 ``ab`` 的参数。"""
    brief = _brief(tmp_path, dtype="fp16")
    task = render_contract(brief)
    assert "`--control` 不是 `ab` 的参数" in task
    assert "`eval-v2 ... --timed --control`" in task
    assert "--control" not in _ab_command(brief)
    # ab 段里出现 --control 的行必须同时点明 eval-v2，否则就是一条无效命令的引导。
    lines = [line for line in task.splitlines() if "--control" in line]
    assert lines and all("eval-v2" in line for line in lines)


@pytest.mark.parametrize(
    "v2,dtype", [(True, "fp16"), (True, "bf16"), (True, "i32"), (False, "fp16"), (False, "i32")],
)
def test_both_renderers_hold_the_same_static_sweep_text(tmp_path, v2, dtype):
    """@brief 两份静态文字逐字节相等（含 ab 段出现的四种组合），账单对得上；命令仍只有一份定义。"""
    brief = _brief(tmp_path, v2=v2, dtype=dtype)
    task = render_contract(brief)
    section = _sweep_suffix(brief)

    assert section
    assert task.count(section) == 1
    receipt = build_receipt(brief, task)
    assert receipt["complete"] is True
    assert receipt["accounting_error"] == ""
    assert receipt["static"]["sweep_bytes"] == len(section.encode("utf-8"))

    from taic import kernel_agent as ka

    # 两份文字各自独立，但命令都来自同一对构造函数，不会分叉。
    assert ka._sweep_command(brief) in section and ka._sweep_command(brief) in task
    if v2 and dtype in AB_DTYPES:
        assert ka._ab_command(brief) in section and ka._ab_command(brief) in task
    else:
        assert ka._ab_command(brief) not in section and ka._ab_command(brief) not in task


def test_the_sweep_prose_is_inside_the_hashed_prompt_symbol(tmp_path):
    """@brief 段落文字必须在 ``render_contract`` 体内，否则改任务书不会动 ``prompt_sha1``。"""
    import inspect

    from taic.kernel_agent import render_contract as renderer
    from taic.trust import prompt_fingerprint_parts

    source = inspect.getsource(renderer)
    assert "同一段窗口里比较多个变体" in source, (
        "the ab section prose must stay inside render_contract, not a helper"
    )
    blob = b"".join(data for _label, data in prompt_fingerprint_parts())
    for needle in ("同一段窗口里比较多个变体", "不占 `--timed` 的 20 次/120 秒配额", "一条命令比较多个变体"):
        assert needle.encode("utf-8") in blob, needle


def test_switching_sweep_tool_off_removes_the_whole_section(tmp_path):
    """@brief ``sweep_tool`` 关闭时既没有 ``sweep-v2`` 也没有 ``ab`` 段。"""
    task = render_contract(_brief(tmp_path, sweep=False))
    assert "tcenv.cli ab" not in task
    assert "# 同一段窗口里比较多个变体" not in task
    assert "# 一条命令比较多个变体" not in task
    assert _sweep_suffix(_brief(tmp_path, sweep=False)) == ""


def test_a_v1_task_book_keeps_sweep_v2_and_adds_no_ab(tmp_path):
    """@brief ``ab`` 只绑 v2 标准题，v1 任务书不写一条必然被拒的命令。"""
    task = render_contract(_brief(tmp_path, v2=False))
    assert "tcenv.cli sweep-v2" in task
    assert "tcenv.cli ab" not in task
    assert "# 同一段窗口里比较多个变体" not in task
    # 没有下面那一节时不能留一个指向它的句子。
    assert "下面那条 `ab`" not in task
    assert "这一版不把多个候选放进同一个进程里交错计时。" in task


def test_an_integer_task_book_gets_no_ab_command(tmp_path):
    """@brief i32 题（``tasks/sem_int_*``）的 ``--dtype`` 不在 CLI 的取值里，不写 ``ab``。"""
    brief = _brief(tmp_path, dtype="i32")
    task = render_contract(brief)
    assert "tcenv.cli sweep-v2" in task
    assert "tcenv.cli ab" not in task
    assert "# 同一段窗口里比较多个变体" not in task
    assert "--control" not in task
    # 指针跟着那一节一起消失，剩下的是一句独立的陈述。
    assert "下面那条 `ab`" not in task
    assert "这一版不把多个候选放进同一个进程里交错计时。" in task
    assert "tcenv.cli ab" not in _sweep_suffix(brief)
    assert "下面那条 `ab`" not in _sweep_suffix(brief)


def test_the_ab_dtype_gate_is_the_dtype_set_the_cli_parses():
    """@brief 任务书只在 CLI 真收的 dtype 上写命令；两处取值由真实 parser 钉住。"""
    assert _cli_dtype_choices(["ab", "--help"]) == set(AB_DTYPES)
    # 计时那条命令用的是同一套取值，所以 ``ab`` 的门与 ``--timed`` 一致，没有各说一套。
    assert _cli_dtype_choices(["eval-v2", "--help"]) == set(AB_DTYPES)


def test_the_task_book_has_no_hint_for_a_command_without_a_caller(tmp_path):
    """@brief 没有调用点的两核配对命令拼法已删除，任务书也不提 ``paired-compare``。"""
    from taic import kernel_agent as ka

    assert not hasattr(ka, "_paired_command")
    mentions = [
        str(path.relative_to(ROOT))
        for path in sorted((ROOT / "taic").rglob("*.py"))
        if "_paired_command" in path.read_text(encoding="utf-8")
    ]
    assert mentions == []
    assert "paired-compare" not in render_contract(_brief(tmp_path, dtype="fp16"))


@pytest.mark.parametrize("bridge", [False, True])
def test_the_sweep_section_survives_the_remote_renderer(tmp_path, bridge):
    """@brief 远端渲染与本地渲染用同一段静态文字，账单才对得上。"""
    if bridge:
        brief = _brief(tmp_path, dtype="fp16")
        brief.selfcheck_socket = str(tmp_path / "selfcheck.sock")
        brief.selfcheck_token_file = str(tmp_path / "selfcheck.token")
        from taic.workspace_prompt import render_remote_contract

        task = render_remote_contract(brief)
        receipt = build_receipt(brief, task, renderer="remote")
        assert "这里不提供 `sweep-v2`" in task
        assert "tcenv.cli ab" not in task
        assert receipt["complete"] is True
        assert receipt["static"]["sweep_bytes"] > 0
        return
    brief = _brief(tmp_path, dtype="fp16")
    assert _sweep_suffix(brief) in render_contract(brief)
