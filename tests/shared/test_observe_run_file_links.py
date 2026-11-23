"""Linked run files must not be ingested, and a manifest cannot borrow another tree's identity.

A run directory is the only source of its facts.  A file leaf that is a symbolic link — or a
directory below ``runs/`` that is one — is refused: ``result.json -> /outside/secret.json`` must
not become a row, a candidate, a confirmation or a brief path.  The experiment id is the canonical
path under ``runs/``; two trees that both declare one id cannot overwrite each other's frozen
conditions or arms.
"""

from __future__ import annotations

import io
import json
import os
from pathlib import Path

import pytest

MARKER = "SECRET-F1-OUTSIDE-DO-NOT-INGEST"


def _write_json(path: Path, document: dict) -> None:
    """@brief 写一个 JSON 文件，父目录不存在时创建。

    @param path 目标路径。
    @param document 要写的字典。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document), encoding="utf-8")


def _feedback_record() -> dict:
    """@brief 一条最小但合法的 ``agent_feedback`` 记录。

    @return 反馈记录。
    """
    return {
        "format": "agent_feedback", "v": 1, "seq": 1, "t": 10, "task": "relu", "dtype": "fp16",
        "size": "small", "label": "r1.p0", "round": 1, "committee": "first", "provider": "p",
        "model": "m", "effort": "high", "outcome": "evaluated", "exit": "normal",
        "best_speedup": 1.0, "kinds": "tool", "work_dir": "agent_work/r1.p0",
        "sections": {"需要帮助": "- 缺一把探针\n"},
        "items": [{"digest": "d1", "section": "需要帮助", "audience": "system", "premise": "",
                   "text": "缺一把探针"}],
        "unverified_premises": 0, "raw": "# 需要帮助\n- 缺一把探针\n",
    }


def _run(root: Path, name: str = "run_00") -> Path:
    """@brief 造一个各项都正常的已结束运行（状态、结果、配置、候选、转录、反馈、确认、best）。

    @param root ``runs/`` 根。
    @param name 运行目录名。
    @return 运行目录。
    """
    directory = root / name
    _write_json(directory / "state.json", {
        "task": "relu", "dtype": "fp16", "phase": "done", "started_at": 1, "updated_at": 1,
    })
    _write_json(directory / "result.json", {
        "task": "relu", "dtype": "fp16", "stop_reason": "max_rounds", "spent_usd": 0.1, "budget_usd": 1,
    })
    _write_json(directory / "run_config.json", {
        "target": {"id": "apple-m4-max", "backend": "metal", "backend_version": "v1"},
        "search": {"kernel_agent": "chat"},
    })
    _write_json(directory / "candidates" / "000_r1.p0" / "report.json", {
        "schema_version": 4, "status": "ok", "stage": "bench", "speedup": 1.1,
    })
    (directory / "candidates" / "000_r1.p0" / "kernel.metal").write_text("kernel void k() {}\n", encoding="utf-8")
    (directory / "transcript.jsonl").write_text(
        json.dumps({"event": "chat", "label": "init", "model": "m"}) + "\n", encoding="utf-8")
    (directory / "feedback.jsonl").write_text(json.dumps(_feedback_record(), ensure_ascii=False) + "\n",
                                              encoding="utf-8")
    _write_json(directory / "feedback_review.json", {"v": 1, "claims": {}})
    _write_json(directory / "confirm_report.json", {"schema_version": 4, "status": "ok"})
    (directory / "best.metal").write_text("kernel void best() {}\n", encoding="utf-8")
    _write_json(directory / "evaluation_outbox" / "req-1.json", {"request_id": "req-1", "status": "pending"})
    return directory


def _env(tmp_path: Path, monkeypatch) -> tuple[Path, Path]:
    """@brief 指向临时 ``runs/`` 与数据库，返回根与外部目录。

    @param tmp_path pytest 临时目录。
    @param monkeypatch pytest 环境变量夹具。
    @return ``(runs 根, runs 之外的目录)``。
    """
    runs = tmp_path / "runs"
    runs.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    monkeypatch.setenv("AI_LOWERLING_RUNS_DIR", str(runs))
    monkeypatch.setenv("OBSERVE_DB", str(tmp_path / "observe.sqlite"))
    monkeypatch.setenv("AI_LOWERLING_BOOKS", str(tmp_path / "none.sqlite"))
    monkeypatch.setenv("OBSERVE_CARRYOVER", str(tmp_path / "carry.json"))
    import observe.index as index_mod

    index_mod._SINGLE = None
    return runs, outside


def _secret(outside: Path) -> Path:
    """@brief 写一份带标记的根外文件，任何字节被摄入都能在库里搜到。

    @param outside ``runs/`` 之外的目录。
    @return 文件路径。
    """
    path = outside / "secret.json"
    path.write_text(json.dumps({
        "task": MARKER, "dtype": "fp16", "phase": "done", "started_at": 1, "updated_at": 1,
        "stop_reason": MARKER, "spent_usd": 9.5, "budget_usd": 10,
        "target": {"id": MARKER}, "search": {"kernel_agent": MARKER},
        "status": MARKER, "speedup": 99.0,
        "items": [{"digest": "d1", "section": "需要帮助", "text": MARKER}],
        "request_id": MARKER, "stop_code": MARKER, "message": MARKER,
    }), encoding="utf-8")
    return path


@pytest.fixture
def opened(monkeypatch) -> tuple[list[str], list[str]]:
    """@brief 记录所有源文件打开的路径，以及其中经过符号链接的路径。

    ``io.open`` 与 ``safe_read.open_file`` 两处都记；``linked`` 用 ``realpath != abspath`` 判断
    “这次打开是否穿了链接”（叶子或任一父目录）。读内容本身已由读取层决定，测试只关心它有没有
    穿过链接。

    @param monkeypatch pytest 替换工具。
    @return ``(打开过的路径, 穿过链接的路径)``。
    """
    try:
        from observe import safe_read
    except ImportError:  # 旧实现没有这一层；记录仍照常统计 io.open
        safe_read = None

    seen: list[str] = []
    linked: list[str] = []
    real_io = io.open
    real_safe = getattr(safe_read, "open_file", None) if safe_read is not None else None

    def note(file) -> None:
        try:
            text = os.fspath(file)
        except TypeError:
            return
        seen.append(text)
        try:
            if os.path.realpath(text) != os.path.abspath(text):
                linked.append(text)
        except OSError:
            pass

    def counting_io(file, mode="r", *args, **kwargs):
        handle = real_io(file, mode, *args, **kwargs)
        if "r" in mode:
            note(file)
        return handle

    def counting_safe(path, *args, **kwargs):
        descriptor = real_safe(path, *args, **kwargs)
        note(path)
        return descriptor

    monkeypatch.setattr(io, "open", counting_io)
    if safe_read is not None and real_safe is not None:
        monkeypatch.setattr(safe_read, "open_file", counting_safe)
    return seen, linked


def _dump(index) -> str:
    """@brief 整库转储，用来搜索根外内容是否被写进索引。

    @param index 索引实例。
    @return 全部 SQL 文本。
    """
    return "\n".join(index.conn.iterdump())


_LINKED = {
    "state.json": ("state.json",),
    "result.json": ("result.json",),
    "run_config.json": ("run_config.json",),
    "transcript.jsonl": ("transcript.jsonl",),
    "confirm_report.json": ("confirm_report.json",),
    "best.metal": ("best.metal",),
    "candidate_report.json": ("candidates", "000_r1.p0", "report.json"),
    "candidate_kernel.metal": ("candidates", "000_r1.p0", "kernel.metal"),
    "feedback.jsonl": ("feedback.jsonl",),
    "feedback_review.json": ("feedback_review.json",),
    "outbox_request.json": ("evaluation_outbox", "req-1.json"),
}


@pytest.mark.parametrize("kind", sorted(_LINKED))
def test_no_run_file_is_read_through_a_symlink(tmp_path, monkeypatch, opened, kind):
    """@brief 每个摄入入口的文件叶子换成符号链接后都不读根外字节，也不把标记写进索引。

    旧实现直接 ``read_text``/``read_bytes``/``stat`` 跟随链接：``result.json`` 的内容会成为 run
    行，候选报告、内核摘要、反馈与确认也都会从根外读取。

    @param tmp_path pytest 临时目录。
    @param monkeypatch pytest 替换工具。
    @param opened 打开路径记录器。
    @param kind 被链接的文件种类。
    """
    runs, outside = _env(tmp_path, monkeypatch)
    directory = _run(runs)
    secret = _secret(outside)
    target = directory.joinpath(*_LINKED[kind])
    target.unlink()
    target.symlink_to(secret)

    from observe.index import Index

    index = Index(tmp_path / "observe.sqlite")
    index.sync_known(force=True)

    seen, linked = opened
    assert linked == [], linked
    assert not any(os.fspath(outside) in path for path in seen), seen
    dump = _dump(index)
    assert MARKER not in dump
    # 正常快照仍在：拒绝的是那个链接文件，不是整场运行。
    assert index.conn.execute("SELECT 1 FROM runs WHERE run_id = 'run_00'").fetchone() is not None


def test_a_normal_run_files_are_still_ingested(tmp_path, monkeypatch):
    """@brief 没有链接时同一套文件照常入库（拒绝逻辑没有把正常路径一起关掉）。

    @param tmp_path pytest 临时目录。
    @param monkeypatch pytest 替换工具。
    """
    runs, _outside = _env(tmp_path, monkeypatch)
    _run(runs)
    from observe.index import Index

    index = Index(tmp_path / "observe.sqlite")
    index.sync_known(force=True)
    row = index.conn.execute("SELECT status, task, spent_usd FROM runs WHERE run_id = 'run_00'").fetchone()
    assert row["status"] == "done" and row["task"] == "relu" and row["spent_usd"] == pytest.approx(0.1)
    assert index.conn.execute("SELECT COUNT(*) FROM candidates WHERE run_id = 'run_00'").fetchone()[0] == 1
    assert index.conn.execute("SELECT COUNT(*) FROM feedback_claims WHERE run_id = 'run_00'").fetchone()[0] == 1
    assert index.conn.execute("SELECT COUNT(*) FROM run_pending_eval WHERE run_id = 'run_00'").fetchone()[0] == 1


def test_a_linked_candidates_directory_is_not_walked(tmp_path, monkeypatch, opened):
    """@brief ``candidates -> outside``：父目录链接同样拒绝，候选与内核都不从根外读。

    @param tmp_path pytest 临时目录。
    @param monkeypatch pytest 替换工具。
    @param opened 打开路径记录器。
    """
    runs, outside = _env(tmp_path, monkeypatch)
    directory = _run(runs)
    external = outside / "candidates" / "000_r1.p0"
    _write_json(external / "report.json", {"schema_version": 4, "status": "ok", "speedup": 42.0,
                                            "message": MARKER})
    (external / "kernel.metal").write_text(f"// {MARKER}\n", encoding="utf-8")
    import shutil

    shutil.rmtree(directory / "candidates")
    (directory / "candidates").symlink_to(outside / "candidates", target_is_directory=True)

    from observe.index import Index

    index = Index(tmp_path / "observe.sqlite")
    index.sync_known(force=True)
    seen, linked = opened
    assert linked == [], linked
    assert not any(os.fspath(outside) in path for path in seen), seen
    assert MARKER not in _dump(index)
    assert index.conn.execute("SELECT COUNT(*) FROM candidates WHERE run_id = 'run_00'").fetchone()[0] == 0


def test_the_brief_reports_no_outside_path_and_no_linked_bytes(tmp_path, monkeypatch):
    """@brief ``result.json`` 链接到根外时，brief 不给出口径外的绝对路径，也不读它的内容。

    旧实现用 ``path.resolve().relative_to(_REPO)``，失败就打印根外绝对路径；同时
    ``read_result``/``conclusion_of_run`` 会把链接目标当成本场结果读。

    @param tmp_path pytest 临时目录。
    @param monkeypatch pytest 替换工具。
    """
    runs, outside = _env(tmp_path, monkeypatch)
    directory = _run(runs)
    secret = _secret(outside)
    (directory / "result.json").unlink()
    (directory / "result.json").symlink_to(secret)

    from taic.brief import summarize_run

    document = summarize_run(directory)
    rendered = json.dumps(document, ensure_ascii=False)
    assert document["paths"]["result"] is None
    assert not document["stop_reason"] and not document["spent_usd"]
    assert str(outside) not in rendered
    assert MARKER not in rendered


def test_an_outside_run_dir_brief_uses_a_run_relative_path(tmp_path, monkeypatch):
    """@brief 仓库外的运行目录（显式传入）不把外部绝对路径写进摘要。

    @param tmp_path pytest 临时目录。
    @param monkeypatch pytest 替换工具。
    """
    runs, _outside = _env(tmp_path, monkeypatch)
    directory = _run(runs)
    from taic.brief import summarize_run

    document = summarize_run(directory)
    assert document["paths"]["result"] == "result.json"
    assert document["paths"]["transcript"] == "transcript.jsonl"
    assert not document["paths"]["result"].startswith("/")


def _valid_experiment(runs: Path) -> None:
    """@brief 合法实验：自述 id 等于 canonical 路径，冻结条件与 arm 都应入库。

    @param runs ``runs/`` 根。
    """
    root = runs / "search_ab"
    root.mkdir()
    _write_json(root / "experiment.json", {
        "schema_version": 1, "experiment_id": "search_ab", "kind": "search_ab", "driver": "search_ab",
        "created_at": 1, "frozen": {"tasks": ["relu"]},
        "arms": [{"arm_id": "on", "delta": {}, "planned_n": 1, "runs": ["trial_1"]}],
    })
    _run(root, "trial_1")


def _manifest(runs: Path, name: str, declared: str, frozen: dict, arm: str) -> None:
    """@brief 一棵自述 ``declared`` 的实验目录。

    @param runs ``runs/`` 根。
    @param name 目录名（canonical id）。
    @param declared 文档自述的 experiment id。
    @param frozen 冻结条件。
    @param arm arm 名。
    """
    root = runs / name
    root.mkdir()
    _write_json(root / "experiment.json", {
        "schema_version": 1, "experiment_id": declared, "kind": "search_ab", "driver": "search_ab",
        "created_at": 1, "frozen": frozen,
        "arms": [{"arm_id": arm, "delta": {}, "planned_n": 1, "runs": ["trial_1"]}],
    })
    _run(root, "trial_1")


def test_a_valid_experiment_keeps_its_manifest_rows_and_arm_binding(tmp_path, monkeypatch):
    """@brief canonical id 相符时，manifest 的 frozen、arm 与 run 归属照旧。

    @param tmp_path pytest 临时目录。
    @param monkeypatch pytest 替换工具。
    """
    runs, _outside = _env(tmp_path, monkeypatch)
    _valid_experiment(runs)
    from observe.index import Index

    index = Index(tmp_path / "observe.sqlite")
    index.sync_known(force=True)
    row = index.conn.execute("SELECT status, frozen_json FROM experiments WHERE experiment_id = 'search_ab'").fetchone()
    assert row["status"] == "recorded" and json.loads(row["frozen_json"]) == {"tasks": ["relu"]}
    assert index.conn.execute("SELECT COUNT(*) FROM arms WHERE experiment_id = 'search_ab' AND arm_id = 'on'").fetchone()[0] == 1
    run = index.conn.execute("SELECT source, experiment_id, arm_id FROM runs WHERE run_id = 'search_ab/trial_1'").fetchone()
    assert (run["source"], run["experiment_id"], run["arm_id"]) == ("manifest", "search_ab", "on")


def test_a_self_declared_id_cannot_overwrite_another_tree_s_frozen_rows(tmp_path, monkeypatch):
    """@brief 两棵树自述同一 id：合法那棵保留自己的 frozen/arm，另一棵只标错，不覆盖。

    旧实现用文档里的 ``experiment_id`` 当主键并 ``ON CONFLICT DO UPDATE``，第二棵树会把第一棵的
    frozen 与 arm 换成自己的。

    @param tmp_path pytest 临时目录。
    @param monkeypatch pytest 替换工具。
    """
    runs, _outside = _env(tmp_path, monkeypatch)
    _manifest(runs, "tree_a", "tree_a", {"tasks": ["a"]}, "on_a")
    _manifest(runs, "tree_b", "tree_a", {"tasks": ["b"]}, "on_b")
    from observe.index import Index

    index = Index(tmp_path / "observe.sqlite")
    index.sync_known(force=True)

    kept = index.conn.execute("SELECT status, frozen_json FROM experiments WHERE experiment_id = 'tree_a'").fetchone()
    assert kept["status"] == "recorded"
    assert json.loads(kept["frozen_json"]) == {"tasks": ["a"]}
    assert index.conn.execute("SELECT COUNT(*) FROM arms WHERE experiment_id = 'tree_a' AND arm_id = 'on_a'").fetchone()[0] == 1
    assert index.conn.execute("SELECT COUNT(*) FROM arms WHERE arm_id = 'on_b'").fetchone()[0] == 0

    marked = index.conn.execute("SELECT status, frozen_json FROM experiments WHERE experiment_id = 'tree_b'").fetchone()
    assert marked["status"] == "experiment_id_mismatch" and json.loads(marked["frozen_json"]) == {}
    run = index.conn.execute("SELECT experiment_id, arm_id, binding FROM runs WHERE run_id = 'tree_b/trial_1'").fetchone()
    assert run["experiment_id"] == "tree_b" and run["arm_id"] == "" and run["binding"] == "drift"
    # 合法那棵的 run 归属没被第二棵树影响。
    first = index.conn.execute("SELECT experiment_id, arm_id FROM runs WHERE run_id = 'tree_a/trial_1'").fetchone()
    assert (first["experiment_id"], first["arm_id"]) == ("tree_a", "on_a")


def test_two_trees_with_one_bogus_id_are_both_marked_with_no_arms(tmp_path, monkeypatch):
    """@brief 两棵树都自述同一个不存在的 id：都按 canonical 路径标错，谁也不写 frozen/arms。

    @param tmp_path pytest 临时目录。
    @param monkeypatch pytest 替换工具。
    """
    runs, _outside = _env(tmp_path, monkeypatch)
    _manifest(runs, "tree_c", "shared", {"tasks": ["c"]}, "on_c")
    _manifest(runs, "tree_d", "shared", {"tasks": ["d"]}, "on_d")
    from observe.index import Index

    index = Index(tmp_path / "observe.sqlite")
    index.sync_known(force=True)
    rows = {row["experiment_id"]: dict(row) for row in index.conn.execute("SELECT experiment_id, status, frozen_json FROM experiments")}
    assert set(rows) == {"tree_c", "tree_d"}
    assert all(row["status"] == "experiment_id_mismatch" and json.loads(row["frozen_json"]) == {} for row in rows.values())
    assert index.conn.execute("SELECT COUNT(*) FROM arms").fetchone()[0] == 0


def test_a_brief_on_an_in_root_directory_alias_reads_the_canonical_run(tmp_path, monkeypatch):
    """@brief ``runs/alias -> runs/run_00`` 是根内别名：brief 按 canonical 目录读取，不会全空。

    @param tmp_path pytest 临时目录。
    @param monkeypatch pytest 替换工具。
    """
    runs, _outside = _env(tmp_path, monkeypatch)
    directory = _run(runs)
    (runs / "alias").symlink_to(directory, target_is_directory=True)
    from taic.brief import summarize_run

    document = summarize_run(runs / "alias")
    assert document["paths"]["result"] == "result.json"
    assert document["task"] == "relu"


def test_a_run_outside_runs_still_refuses_a_linked_leaf(tmp_path, monkeypatch):
    """@brief 显式传入的根外运行目录：链接叶子仍拒绝（检查起点是最近的含快照祖先）。

    @param tmp_path pytest 临时目录。
    @param monkeypatch pytest 替换工具。
    """
    _runs, outside = _env(tmp_path, monkeypatch)
    directory = _run(outside, "copy")
    secret = _secret(outside)
    (directory / "result.json").unlink()
    (directory / "result.json").symlink_to(secret)
    from taic.brief import summarize_run

    document = summarize_run(directory)
    rendered = json.dumps(document, ensure_ascii=False)
    assert document["paths"]["result"] is None
    assert MARKER not in rendered


def test_the_brief_never_quotes_a_linked_candidate_report_or_outbox_request(tmp_path, monkeypatch, opened):
    """@brief brief 的候选摘要与待评测计数也不从链接读取。

    @param tmp_path pytest 临时目录。
    @param monkeypatch pytest 替换工具。
    @param opened 打开路径记录器。
    """
    runs, outside = _env(tmp_path, monkeypatch)
    directory = _run(runs)
    secret = _secret(outside)
    report = directory / "candidates" / "000_r1.p0" / "report.json"
    report.unlink()
    report.symlink_to(secret)
    request = directory / "evaluation_outbox" / "req-1.json"
    request.unlink()
    request.symlink_to(secret)
    from taic.brief import summarize_run

    document = summarize_run(directory, expand=frozenset({"cands"}))
    rendered = json.dumps(document, ensure_ascii=False)
    assert document["pending_evaluation"]["count"] == 0
    assert [row["label"] for row in document.get("cands") or []] == ["r1.p0"]
    assert (document.get("cands") or [{}])[0].get("speedup") is None
    assert MARKER not in rendered
    seen, linked = opened
    assert linked == [], linked
    assert not any(os.fspath(outside) in path for path in seen), seen


def test_a_non_string_declared_id_is_marked_not_treated_as_canonical(tmp_path, monkeypatch):
    """@brief 自述 id 不是字符串（例如 7）时同样是错标，不当作合法 canonical id。

    @param tmp_path pytest 临时目录。
    @param monkeypatch pytest 替换工具。
    """
    runs, _outside = _env(tmp_path, monkeypatch)
    root = runs / "tree_e"
    root.mkdir()
    _write_json(root / "experiment.json", {
        "schema_version": 1, "experiment_id": 7, "kind": "search_ab", "driver": "search_ab",
        "created_at": 1, "frozen": {"tasks": ["e"]},
        "arms": [{"arm_id": "on_e", "delta": {}, "planned_n": 1, "runs": []}],
    })
    _write_json(root / "state.json", {"task": "relu", "phase": "done", "started_at": 1, "updated_at": 1})
    from observe.index import Index

    index = Index(tmp_path / "observe.sqlite")
    index.sync_known(force=True)
    row = index.conn.execute("SELECT status, frozen_json FROM experiments WHERE experiment_id = 'tree_e'").fetchone()
    assert row["status"] == "experiment_id_mismatch" and json.loads(row["frozen_json"]) == {}
    assert index.conn.execute("SELECT COUNT(*) FROM arms").fetchone()[0] == 0
