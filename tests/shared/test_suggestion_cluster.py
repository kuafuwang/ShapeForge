"""Read-only suggestion clusters. No GPU and no model calls."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from observe.similarity import lexical_overlap
from observe.suggestion_cluster import cluster_rows, intent_bucket, select_scorer
from taic.proposal_routing import route_proposal


def _row(text: str, *, digest: str, backend: str = "metal", simulated: bool = False, section: str = "对系统的建议", run_id: str = "run", seq: int = 1) -> dict:
    return {
        "text": text,
        "digest": digest,
        "backend": backend,
        "simulated": simulated,
        "section": section,
        "run_id": run_id,
        "seq": seq,
        "claims": 1,
        "runs": 1,
        "status": "pending",
    }


def test_near_duplicates_cluster_and_unrelated_sentences_stay_apart():
    """@brief 词面接近的句子合成一簇，不相干的句子分开。"""
    rows = [
        _row("add paired compare subcommand to tcenv now", digest="a"),
        _row("add paired compare subcommand to tcenv today", digest="b"),
        _row("change the tolerance threshold for evaluation", digest="c"),
    ]
    view = cluster_rows(rows, scorer=lexical_overlap, method="lexical_overlap", limit=20)
    assert view["readonly"] is True and view["method"] == "lexical_overlap"
    sizes = sorted(cluster["size"] for cluster in view["clusters"])
    assert sizes == [1, 2]
    merged = next(cluster for cluster in view["clusters"] if cluster["size"] == 2)
    assert {item["digest"] for item in merged["members"]} == {"a", "b"}
    assert merged["intent"] == "developer_queue"
    assert "text" not in merged["members"][0]


def test_partition_is_stronger_than_a_perfect_score():
    """@brief 后端、模拟态、路由意图不同时，即使相似度为 1 也不并簇。"""
    text = "add paired compare subcommand to tcenv now"
    rows = [
        _row(text, digest="real", backend="metal", simulated=False),
        _row(text, digest="real", backend="npu", simulated=False),
        _row(text, digest="real", backend="metal", simulated=True),
        _row("measure the tile occupancy before the next launch", digest="tile", backend="metal", simulated=False),
        _row("缺一把探针", digest="help", backend="metal", simulated=False, section="需要帮助"),
    ]
    view = cluster_rows(rows, scorer=lambda left, right: 1.0, method="lexical_overlap", link_at=0.0)
    keys = {(cluster["backend"], cluster["simulated"], cluster["intent"]) for cluster in view["clusters"]}
    assert ("metal", False, "developer_queue") in keys
    assert ("npu", False, "developer_queue") in keys
    assert ("metal", True, "developer_queue") in keys
    assert ("metal", False, "overlay_candidate") in keys
    assert ("metal", False, "help") in keys
    for cluster in view["clusters"]:
        assert len({(item["digest"], cluster["backend"], cluster["simulated"], cluster["intent"]) for item in cluster["members"]}) == cluster["size"]
        assert len({cluster["backend"], cluster["simulated"], cluster["intent"]}) == 3


def test_help_and_system_sections_both_cluster_and_keep_their_intent():
    """@brief 两个小节都进视图；帮助请求与开发者请求不因词面接近而合并。"""
    assert intent_bucket("缺一把探针", "需要帮助") == "help"
    assert intent_bucket("修改容差阈值", "需要帮助") == "developer_queue"
    rows = [
        _row("缺一把探针", digest="h1", section="需要帮助"),
        _row("还是缺一把探针", digest="h2", section="需要帮助"),
        _row("修改容差阈值", digest="d1", section="需要帮助"),
    ]
    view = cluster_rows(rows, scorer=lexical_overlap, method="lexical_overlap")
    by_intent = {cluster["intent"]: cluster for cluster in view["clusters"]}
    assert by_intent["help"]["size"] == 2
    assert set(by_intent["help"]["sections"]) == {"需要帮助"}
    assert by_intent["developer_queue"]["size"] == 1
    assert {item["digest"] for item in by_intent["help"]["members"]} == {"h1", "h2"}


def test_structured_proposals_cluster_by_action_not_by_json_shell():
    """@brief 两份外壳相同、action 不同的提案不能进同一簇；action 的近义改写仍然可以。"""
    shared = " ".join(
        ["The session probe measurement stayed on the assembled kernel cost and the same micro probe family."] * 12
    )
    action_a = "Give micro-probes a data-dependent loop trip count otherwise constant-operand loops are folded"
    action_b = "Add to the v2 correctness report a breakdown of failing elements by output axis parity and oc-octet"
    action_near = "Give micro-probes a data-dependent loop trip count otherwise constant operand loops are folded"
    def proposal(action: str) -> str:
        return json.dumps({
            "surface": "tip",
            "scope": {"target_id": "apple-m4-max", "task": "conv2d", "dtype": "fp16"},
            "action": action,
            "audience": "developer",
            "needs": "repo_change",
            "evidence": {"probe": "p7/fma.metal", "measurement": shared},
            "rationale": shared,
        }, indent=2)
    left, right, near = proposal(action_a), proposal(action_b), proposal(action_near)
    assert lexical_overlap(left, right) >= 0.72
    assert lexical_overlap(action_a, action_b) < 0.72
    for body in (left, right, near):
        parsed = json.loads(body)
        assert route_proposal(parsed)["disposition"] == "developer_queue"
        assert intent_bucket(body, "对系统的建议") == route_proposal(parsed)["disposition"]
    assert route_proposal({"action": left})["disposition"] == "overlay_candidate"
    assert route_proposal({"action": near})["disposition"] == "overlay_candidate"
    rows = [
        _row(left, digest="0aba4cecef56cac9", seq=4),
        _row(right, digest="24e4069aede36412", seq=4),
        _row(near, digest="nearactiondigest1", seq=5),
    ]
    view = cluster_rows(rows, scorer=lexical_overlap, method="lexical_overlap", preview=60)
    assert {cluster["intent"] for cluster in view["clusters"]} == {"developer_queue"}
    clusters = view["clusters"]
    owners = {}
    for cluster in clusters:
        for member in cluster["members"]:
            owners[member["digest"]] = cluster["cluster_id"]
    assert owners["0aba4cecef56cac9"] != owners["24e4069aede36412"]
    assert owners["0aba4cecef56cac9"] == owners["nearactiondigest1"]
    probe = next(cluster for cluster in clusters if cluster["cluster_id"] == owners["0aba4cecef56cac9"])
    assert "micro-probes" in probe["representative"]["preview"]
    assert "surface" not in probe["representative"]["preview"]


def test_nlembedding_bridge_does_not_pull_unrelated_actions_into_one_cluster():
    """@brief 默认嵌入阈值下，桥接节点不能把两个不相干的 action 收进同一簇。"""
    probe = "Give micro-probes a data-dependent loop trip count otherwise constant-operand loops are folded"
    report = "Add to the v2 correctness report a breakdown of failing elements by output axis parity and oc-octet"
    bridge = "Measure register spill and Winograd timing before the next kernel launch"
    direct = {
        frozenset((probe, bridge)): 0.90,
        frozenset((report, bridge)): 0.88,
        frozenset((probe, report)): 0.40,
    }

    def scorer(left: str, right: str) -> float:
        return direct.get(frozenset((left, right)), 0.0)

    rows = [
        _row(probe, digest="0aba4cecef56cac9"),
        _row(report, digest="24e4069aede36412"),
        _row(bridge, digest="bridge"),
    ]
    for row in rows:
        row["intent"] = "developer_queue"
    view = cluster_rows(rows, scorer=scorer, method="nlembedding")
    assert view["method"] == "nlembedding"
    owners = {}
    for cluster in view["clusters"]:
        for member in cluster["members"]:
            owners[member["digest"]] = cluster["cluster_id"]
    assert owners["0aba4cecef56cac9"] != owners["24e4069aede36412"]
    close = "add paired compare subcommand to tcenv now"
    closer = "add paired compare subcommand to tcenv today"
    closest = "add paired compare subcommand to tcenv soon"
    together = cluster_rows(
        [
            _row(close, digest="p1"),
            _row(closer, digest="p2"),
            _row(closest, digest="p3"),
        ],
        scorer=lambda left, right: 0.9,
        method="nlembedding",
    )
    assert [cluster["size"] for cluster in together["clusters"]] == [3]


def test_selector_uses_embedding_only_when_a_probe_vector_exists(monkeypatch):
    """@brief 句向量探针失败时退回词面重叠，不把空结果说成语义聚类。"""
    from observe import similarity

    monkeypatch.setattr(similarity, "available", lambda: False)
    _scorer, method = select_scorer()
    assert method == "lexical_overlap"
    monkeypatch.setattr(similarity, "available", lambda: True)
    monkeypatch.setattr(similarity, "embedding", lambda text: None if not text else (1.0, 0.0))
    _scorer, method = select_scorer()
    assert method == "nlembedding"
    monkeypatch.setattr(similarity, "embedding", lambda text: None)
    _scorer, method = select_scorer()
    assert method == "lexical_overlap"


def test_embedding_score_does_not_merge_texts_without_shared_tokens(monkeypatch):
    """@brief 嵌入路径仍要求词面重叠，避免中文高基线把无关建议并成一簇。"""
    from observe.suggestion_cluster import _embedding_score

    monkeypatch.setattr("observe.similarity.lexical_overlap", lambda left, right: 0.0)
    monkeypatch.setattr("observe.similarity.embedding", lambda text: (1.0, 0.0))
    monkeypatch.setattr("observe.similarity.cosine", lambda left, right: 0.99)
    assert _embedding_score("甲", "乙") == 0.0


def test_cluster_view_is_read_only_and_covers_both_sections(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys):
    """@brief 索引上的聚类覆盖两个小节，并且不改反馈文件。"""
    from observe.suggestions import cluster_view, compact
    from taic.agent_feedback import parse_feedback
    from taic.cli import main
    from tests.shared.test_observe_feedback import _open, _record, _run

    runs, index = _open(tmp_path, monkeypatch)
    parsed = parse_feedback(
        "# 对系统的建议\n- add paired compare subcommand to tcenv now\n"
        "- add paired compare subcommand to tcenv today\n"
        "# 需要帮助\n- 缺一把探针\n"
    )
    record = _record(parsed, t=20)
    directory = _run(runs, "metal-one", record, started_at=20)
    state = json.loads((directory / "state.json").read_text(encoding="utf-8"))
    state["backend"] = "metal"
    state["target_id"] = "apple-m4-max"
    (directory / "state.json").write_text(json.dumps(state), encoding="utf-8")
    (directory / "result.json").write_text(json.dumps({"simulated": False, "task": record["task"]}), encoding="utf-8")
    before = (directory / "feedback.jsonl").read_bytes()
    index.tick()
    view = cluster_view(scorer=lexical_overlap, method="lexical_overlap", limit=20)
    assert view["readonly"] is True
    sections = {section for cluster in view["clusters"] for section in cluster["sections"]}
    assert sections == {"对系统的建议", "需要帮助"}
    assert (directory / "feedback.jsonl").read_bytes() == before
    assert not (directory / "feedback_review.json").exists()
    compact_doc = compact(limit=20)
    assert "groups" in compact_doc and "clusters" not in compact_doc
    code = main(["query", "suggestions", "--cluster", "--limit", "20"])
    assert code == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["readonly"] is True and printed["clusters"]
    assert (directory / "feedback.jsonl").read_bytes() == before


def test_cluster_runs_count_distinct_runs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """@brief 同一 run 的两个 seq 只算一个 run。两场三条 claim 是 claims=3、runs=2。"""
    from observe.suggestions import cluster_view
    from taic.agent_feedback import parse_feedback
    from tests.shared.test_observe_feedback import _open, _record, _run

    monkeypatch.setenv("OBSERVE_FULL_TICK_INTERVAL", "0")
    runs, index = _open(tmp_path, monkeypatch)
    text = "add paired compare subcommand to tcenv now"
    parsed = parse_feedback(f"# 对系统的建议\n- {text}\n")
    first = _run(runs, "metal-a", _record(parsed, seq=1, t=10), started_at=10)
    extra = _record(parsed, seq=2, t=11)
    with (first / "feedback.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(extra, ensure_ascii=False) + "\n")
    second = _run(runs, "metal-b", _record(parsed, seq=1, t=12), started_at=12)
    for directory, record in ((first, extra), (second, _record(parsed, seq=1, t=12))):
        state = json.loads((directory / "state.json").read_text(encoding="utf-8"))
        state["backend"] = "metal"
        state["target_id"] = "apple-m4-max"
        (directory / "state.json").write_text(json.dumps(state), encoding="utf-8")
        (directory / "result.json").write_text(json.dumps({"simulated": False, "task": record["task"]}), encoding="utf-8")
    index.tick()
    view = cluster_view(scorer=lexical_overlap, method="lexical_overlap", limit=20)
    member = next(item for cluster in view["clusters"] for item in cluster["members"])
    assert member["claims"] == 3
    assert member["runs"] == 2
    coordinates = [cluster["coordinates"] for cluster in view["clusters"] if any(item["digest"] == member["digest"] for item in cluster["members"])][0]
    assert {(item["run_id"], item["seq"], item["digest"], item["status"]) for item in coordinates} == {
        ("metal-a", 1, member["digest"], "pending"),
        ("metal-a", 2, member["digest"], "pending"),
        ("metal-b", 1, member["digest"], "pending"),
    }
    assert all(item["expected_status"] == item["status"] for item in coordinates)


def test_cluster_coordinates_keep_mixed_status(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """@brief 同一 digest 的 pending 和 done 都留在坐标里，旧状态不会被最新一行盖掉。"""
    from observe.suggestions import cluster_view
    from taic.agent_feedback import parse_feedback
    from tests.shared.test_observe_feedback import _open, _record, _run

    monkeypatch.setenv("OBSERVE_FULL_TICK_INTERVAL", "0")
    runs, index = _open(tmp_path, monkeypatch)
    text = "keep every coordinate when one claim is already done"
    parsed = parse_feedback(f"# 对系统的建议\n- {text}\n")
    digest = next(item["digest"] for item in parsed["items"] if item["text"] == text)
    first = _run(runs, "metal-a", _record(parsed, seq=1, t=10), started_at=10)
    extra = _record(parsed, seq=2, t=11)
    with (first / "feedback.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(extra, ensure_ascii=False) + "\n")
    second = _run(runs, "metal-b", _record(parsed, seq=1, t=12), started_at=12)
    (first / "feedback_review.json").write_text(json.dumps({
        "v": 1,
        "claims": {f"1:{digest}": {"status": "done", "verification": "confirmed", "change_ref": "commit:abc"}},
    }), encoding="utf-8")
    for directory in (first, second):
        (directory / "result.json").write_text(json.dumps({"simulated": False}), encoding="utf-8")
    index.tick()
    view = cluster_view(scorer=lexical_overlap, method="lexical_overlap", limit=20)
    coordinates = next(cluster["coordinates"] for cluster in view["clusters"] if any(item["digest"] == digest for item in cluster["members"]))
    assert {(item["run_id"], item["seq"], item["status"], item["expected_status"]) for item in coordinates} == {
        ("metal-a", 1, "done", "done"),
        ("metal-a", 2, "pending", "pending"),
        ("metal-b", 1, "pending", "pending"),
    }
    assert view["incomplete"] is False and view["warning"] == ""


def test_cluster_refuses_input_past_the_cap_and_jsonl_keeps_the_warning(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys):
    """@brief 2001 条不能被截成一个看起来完整的簇，jsonl 也要打出警告。"""
    from observe.suggestions import cluster_view
    from taic.agent_feedback import _digest
    from taic.cli import main
    from tests.shared.test_observe_feedback import _open

    monkeypatch.setenv("OBSERVE_FULL_TICK_INTERVAL", "0")
    runs, index = _open(tmp_path, monkeypatch)
    directory = runs / "cap"
    directory.mkdir()
    (directory / "state.json").write_text(json.dumps({"task": "gemm", "phase": "done", "backend": "metal"}), encoding="utf-8")
    (directory / "result.json").write_text(json.dumps({"simulated": False, "task": "gemm"}), encoding="utf-8")
    lines = []
    for seq in range(1, 2002):
        text = f"cluster cap row {seq}"
        lines.append(json.dumps({
            "format": "agent_feedback", "v": 1,
            "seq": seq, "t": seq, "task": "gemm", "label": "r1",
            "items": [{"section": "对系统的建议", "digest": _digest("对系统的建议", text), "text": text}],
        }, ensure_ascii=False))
    (directory / "feedback.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    index.tick()
    view = cluster_view(scorer=lexical_overlap, method="lexical_overlap", limit=20)
    assert view["matched"] == 2001
    assert view["read"] == 0
    assert view["clusters"] == []
    assert view["input_truncated"] is True
    assert view["incomplete"] is True
    assert view["warning"]
    code = main(["query", "suggestions", "--cluster", "--jsonl", "--limit", "5"])
    assert code == 0
    printed = json.loads(capsys.readouterr().out.splitlines()[0])
    assert printed["incomplete"] is True and printed["warning"] and printed["input_truncated"] is True
    assert printed["matched"] == 2001


def test_unreadable_claim_text_is_incomplete_and_not_a_silent_zero(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys):
    """@brief 索引里有主张但正文读不到时，不能显示成 total_clusters=0 且完整。"""
    from observe.suggestions import cluster_view
    from taic.agent_feedback import parse_feedback
    from taic.cli import main
    from tests.shared.test_observe_feedback import _open, _record, _run

    monkeypatch.setenv("OBSERVE_FULL_TICK_INTERVAL", "0")
    runs, index = _open(tmp_path, monkeypatch)
    text = "this claim text will not be readable"
    parsed = parse_feedback(f"# 对系统的建议\n- {text}\n")
    digest = next(item["digest"] for item in parsed["items"] if item["text"] == text)
    _run(runs, "metal-unreadable", _record(parsed, seq=1, t=10), started_at=10)
    index.tick()
    monkeypatch.setattr("observe.suggestions._claim_source", lambda *args, **kwargs: ("", None))
    view = cluster_view(scorer=lexical_overlap, method="lexical_overlap", limit=20)
    assert view["total_clusters"] == 0
    assert view["input_truncated"] is False
    assert view["incomplete"] is True
    assert view["matched"] == 1 and view["read"] == 0
    assert view["source_errors"][0]["digest"] == digest
    assert view["warning"]
    code = main(["query", "suggestions", "--cluster", "--jsonl"])
    assert code == 0
    printed = json.loads(capsys.readouterr().out.splitlines()[0])
    assert printed["incomplete"] is True and printed["warning"]
    assert printed["source_errors"]


def test_simulated_comes_from_the_same_feedback_record(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """@brief 只有反馈记录标明模拟时，聚类和 entries 用同一条记录；测量文件优先。"""
    from observe.feedback import entries
    from observe.suggestions import cluster_view
    from taic.agent_feedback import parse_feedback
    from tests.shared.test_observe_feedback import _open, _record, _run

    monkeypatch.setenv("OBSERVE_FULL_TICK_INTERVAL", "0")
    runs, index = _open(tmp_path, monkeypatch)
    text = "record only simulated timing"
    parsed = parse_feedback(f"# 对系统的建议\n- {text}\n")
    record = _record(parsed, seq=1, t=10)
    record["simulated"] = True
    record_only = _run(runs, "record-only", record, started_at=10)
    legacy_record = _record(parsed, seq=1, t=11)
    legacy = _run(runs, "legacy-file", legacy_record, started_at=11)
    (legacy / "result.json").write_text(json.dumps({"simulated": True, "task": "gemm"}), encoding="utf-8")
    measured = _record(parsed, seq=1, t=12)
    measured["simulated"] = True
    file_wins = _run(runs, "file-wins", measured, started_at=12)
    (file_wins / "result.json").write_text(json.dumps({"simulated": False, "task": "gemm"}), encoding="utf-8")
    index.tick()
    view = cluster_view(scorer=lexical_overlap, method="lexical_overlap", limit=20)
    flags = {
        coord["run_id"]: cluster["simulated"]
        for cluster in view["clusters"]
        for coord in cluster["coordinates"]
    }
    assert entries(record_only)["records"][0]["claims"][0]["simulated"] is True
    assert flags["record-only"] is True
    assert flags["legacy-file"] is True
    assert flags["file-wins"] is False
    kind_and_type = _record(parse_feedback("# 对系统的建议\n- kind and type both set\n"), seq=1, t=13)
    kind_and_type["measurement_kind"] = "measured"
    kind_and_type["measurement_type"] = "simulated"
    kind_run = _run(runs, "kind-and-type", kind_and_type, started_at=13)
    level_only = _record(parse_feedback("# 对系统的建议\n- evidence level only\n"), seq=1, t=14)
    level_only["evidence_level"] = "simulated"
    level_run = _run(runs, "evidence-level", level_only, started_at=14)
    provenance_only = _record(parse_feedback("# 对系统的建议\n- provenance only\n"), seq=1, t=15)
    provenance_only["evidence_provenance"] = {"level": "simulated", "simulated": True}
    provenance_run = _run(runs, "evidence-provenance", provenance_only, started_at=15)
    file_record = _record(parse_feedback("# 对系统的建议\n- file declares both kinds\n"), seq=1, t=16)
    file_both = _run(runs, "file-both-kinds", file_record, started_at=16)
    (file_both / "result.json").write_text(json.dumps({
        "measurement_kind": "measured", "measurement_type": "simulated", "task": "gemm",
    }), encoding="utf-8")
    index.tick()
    view = cluster_view(scorer=lexical_overlap, method="lexical_overlap", limit=20)
    flags = {
        coord["run_id"]: cluster["simulated"]
        for cluster in view["clusters"]
        for coord in cluster["coordinates"]
    }
    assert flags["kind-and-type"] is True
    assert flags["evidence-level"] is True
    assert flags["evidence-provenance"] is True
    assert flags["file-both-kinds"] is True
    assert flags["file-wins"] is False
    assert entries(kind_run)["records"][0]["claims"][0]["simulated"] is True
    assert entries(level_run)["records"][0]["claims"][0]["simulated"] is True
    assert entries(provenance_run)["records"][0]["claims"][0]["simulated"] is True
    assert entries(file_both)["records"][0]["claims"][0]["simulated"] is True
    assert entries(file_wins)["records"][0]["claims"][0]["simulated"] is False
