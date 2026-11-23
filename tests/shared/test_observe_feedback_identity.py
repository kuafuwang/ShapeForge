"""Review identity is ``run_id#seq#digest``: the tree that authorizes a claim is the tree written.

A bare token resolves through the index first, so a same-named directory in the working directory
cannot redirect a review.  An explicit directory is accepted only when it resolves to the canonical
path of an indexed run; a directory outside ``runs/`` cannot borrow a same-named run's claim.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from observe.actions import Refused, feedback_run, review_feedback_batch


def _record(seq: int = 1) -> dict:
    """@brief 一条带 system claim 的反馈记录。

    @param seq 记录序号。
    @return ``feedback.jsonl`` 记录。
    """
    return {
        "format": "agent_feedback", "v": 1, "seq": seq, "t": 10.0, "task": "relu", "dtype": "fp16",
        "size": "small", "label": "r1.p0", "round": 1, "committee": "first", "provider": "p",
        "model": "m", "effort": "high", "outcome": "evaluated", "exit": "normal", "best_speedup": 1.0,
        "kinds": "tool", "work_dir": "agent_work/r1.p0",
        "sections": {"需要帮助": "- 缺一把探针\n"},
        "items": [{"digest": "d1", "section": "需要帮助", "audience": "system", "premise": "",
                   "text": "缺一把探针"}],
        "unverified_premises": 0, "raw": "# 需要帮助\n- 缺一把探针\n",
    }


def _run(root: Path, name: str = "conv") -> Path:
    """@brief 造一个含快照与一条反馈的已结束运行。

    @param root ``runs/`` 根。
    @param name run id。
    @return 运行目录。
    """
    directory = root / name
    directory.mkdir(parents=True)
    (directory / "state.json").write_text(json.dumps({
        "task": "relu", "dtype": "fp16", "phase": "done", "started_at": 1, "updated_at": 1,
    }), encoding="utf-8")
    (directory / "feedback.jsonl").write_text(json.dumps(_record(), ensure_ascii=False) + "\n",
                                              encoding="utf-8")
    return directory


def _env(tmp_path: Path, monkeypatch) -> tuple[Path, Path]:
    """@brief 建临时 runs/ 与索引，返回 runs 根与外部目录。

    @param tmp_path pytest 临时目录。
    @param monkeypatch pytest 环境变量夹具。
    @return ``(runs 根, 外部目录)``。
    """
    runs = tmp_path / "runs"
    runs.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    monkeypatch.setenv("AI_LOWERLING_RUNS_DIR", str(runs))
    monkeypatch.setenv("OBSERVE_DB", str(tmp_path / "observe.sqlite"))
    monkeypatch.setenv("AI_LOWERLING_BOOKS", str(tmp_path / "none.sqlite"))
    monkeypatch.setenv("TAIC_REVIEW_EVENTS", str(tmp_path / "review_events.jsonl"))
    import observe.index as index_mod

    index_mod._SINGLE = None
    return runs, outside


def _refresh() -> None:
    """@brief 把当前 ``runs/`` 同步进索引（显式目录测试需要 run 已在索引里）。

    """
    from observe.index import get_index

    get_index().sync_known(force=True)


_ROWS = [{"seq": 1, "digest": "d1", "status": "info", "verification": "unchecked", "expected_status": "pending"}]


def test_a_bare_id_resolves_the_indexed_run_not_a_same_named_local_directory(tmp_path, monkeypatch):
    """@brief 工作目录里同名目录不能顶替索引 run：写入落在 ``runs/conv``。

    旧实现先看 ``Path(token).is_dir()``，于是 cwd 的 ``conv`` 成为写入目录，而 claim 仍按索引
    ``conv`` 校验——授权树与写入树分离。

    @param tmp_path pytest 临时目录。
    @param monkeypatch pytest 替换工具。
    """
    runs, outside = _env(tmp_path, monkeypatch)
    directory = _run(runs)
    decoy = _run(outside, "conv")
    _refresh()
    monkeypatch.chdir(outside)
    assert feedback_run("conv") == ("conv", directory.resolve())
    stored = review_feedback_batch("conv", _ROWS, method="human")
    assert [row["status"] for row in stored["items"]] == ["info"]
    assert (directory / "feedback_review.json").is_file()
    assert not (decoy / "feedback_review.json").exists()


def test_an_explicit_indexed_run_directory_is_still_accepted(tmp_path, monkeypatch):
    """@brief 显式目录路径只要解析到已索引 run 的 canonical 目录就照常工作（含尾随分隔符）。

    @param tmp_path pytest 临时目录。
    @param monkeypatch pytest 替换工具。
    """
    runs, _outside = _env(tmp_path, monkeypatch)
    directory = _run(runs)
    _refresh()
    assert feedback_run(str(directory)) == ("conv", directory.resolve())
    assert feedback_run(str(directory) + "/") == ("conv", directory.resolve())
    stored = review_feedback_batch(str(directory), _ROWS, method="human")
    assert [row["status"] for row in stored["items"]] == ["info"]
    assert (directory / "feedback_review.json").is_file()


def test_an_in_root_symlink_alias_keeps_the_canonical_identity(tmp_path, monkeypatch):
    """@brief 根内目录别名指向已索引 run 时，按 canonical id 接受。

    @param tmp_path pytest 临时目录。
    @param monkeypatch pytest 替换工具。
    """
    runs, _outside = _env(tmp_path, monkeypatch)
    directory = _run(runs)
    _refresh()
    (runs / "alias").symlink_to(directory, target_is_directory=True)
    assert feedback_run(str(runs / "alias")) == ("conv", directory.resolve())


def test_a_directory_outside_runs_cannot_borrow_a_same_named_run_claim(tmp_path, monkeypatch):
    """@brief ``runs/`` 外的同名目录即使含快照也不能借用索引 claim，更不能被写入。

    @param tmp_path pytest 临时目录。
    @param monkeypatch pytest 替换工具。
    """
    runs, outside = _env(tmp_path, monkeypatch)
    _run(runs)
    _refresh()
    decoy = _run(outside, "conv")
    with pytest.raises(Refused):
        feedback_run(str(decoy))
    with pytest.raises(Refused):
        review_feedback_batch(str(decoy), _ROWS, method="human")
    assert not (decoy / "feedback_review.json").exists()


def test_a_bare_id_that_is_not_indexed_is_refused(tmp_path, monkeypatch):
    """@brief 未入索引的同名本地目录不再被当成运行。

    @param tmp_path pytest 临时目录。
    @param monkeypatch pytest 替换工具。
    """
    _runs, outside = _env(tmp_path, monkeypatch)
    _run(outside, "ghost")
    monkeypatch.chdir(outside)
    with pytest.raises(Refused):
        feedback_run("ghost")
