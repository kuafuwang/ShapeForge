"""G12 read-only experience scope audit.

Every check runs on synthetic repositories under ``tmp_path`` plus one stability
pass over the shipped ``targets/apple-m4-max/experience.md``.  The shipped file's
entry count is not pinned: experience entries are expected to change, and a test
that fails on every legitimate addition would be noise.  What is pinned is the
shape of the audit, the honest ``unknown`` result when a path proves nothing, and
that the audit never writes.

No backend import, no GPU, no model call.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from taic import experience_scope
from taic.experience_scope import (
    audit_experience_file,
    audit_repo,
    inspect_pointer,
    main,
    render_summary,
)

REPO = Path(__file__).resolve().parents[2]
SHIPPED = REPO / "targets" / "apple-m4-max" / "experience.md"

WARNING_CODES = {
    "multiple_host_pointers",
    "host_pointer_invalid",
    "host_pointer_missing",
    "source_pointer_invalid",
    "source_pointer_missing",
    "body_pointer_invalid",
    "body_pointer_missing",
    "verified_without_pointer",
    "landing_not_chip_layer",
    "target_landing_mismatch",
    "target_file_mismatch",
    "landing_file_mismatch",
    "evidence_task_mismatch",
    "evidence_target_mismatch",
}


def _repo(tmp_path: Path) -> Path:
    """@brief 造一个最小仓库：两个题、两个 target、几条 run 证据。

    @param tmp_path pytest 临时目录。
    @return 仓库根。
    """
    files = {
        "tasks/gemm/opspec.json": "{}",
        "tasks/conv2d/opspec.json": "{}",
        "targets/apple-m4-max/target.toml": "schema_version = 1\n",
        "targets/other-chip/target.toml": "schema_version = 1\n",
        "runs/host-rerun-gemm-20261120/stdout.txt": "measured\n",
        "runs/apple-m4-max/host-rerun-gemm-20261120/stdout.txt": "measured\n",
        "runs/host-rerun-gemm_silu-reg-small-20261119/stdout.txt": "measured\n",
        "runs/host-rerun-conv2d-20261120/stdout.txt": "measured\n",
        "runs/host-rerun-20261120/stdout.txt": "measured\n",
        "runs/other-chip/run1/result.json": "{}",
    }
    for name, text in files.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return tmp_path


def _section(
    title: str,
    *,
    confidence: str = "verified",
    task: str = "",
    target: str = "",
    landing: str = "",
    source: str = "",
    boundary: str = "",
    observation: str = "",
    pointer: str | None = None,
) -> str:
    """@brief 拼一节经验库 Markdown。空参数不写行，用来模拟旧条目。"""
    lines = [f"## {title}", "", "Keywords: audit fixture", f"Confidence: {confidence}"]
    if task:
        lines.append(f"题目: {task}")
    if target:
        lines.append(f"芯片: {target}")
    if landing:
        lines.append(f"落点: {landing}")
    lines.append(f"观察: {observation or title}")
    if source:
        lines.append(f"来源: {source}")
    lines.append("本机实测: 1.0 µs")
    if boundary:
        lines.append(f"边界: {boundary}")
    lines.append("反例: 换题后不成立。")
    if pointer is not None:
        lines.append(f"宿主重跑: {pointer}")
    return "\n".join(lines) + "\n"


def _experience(repo: Path, *sections: str) -> Path:
    """@brief 把若干节写成 ``targets/apple-m4-max/experience.md``。"""
    path = repo / "targets" / "apple-m4-max" / "experience.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("# 经验条目\n\n前言。\n\n" + "".join(sections), encoding="utf-8")
    return path


def test_valid_chip_section_is_clean_and_a_promotion_input(tmp_path):
    """@brief 合格芯片层条目：无告警、作用域 match、边界三项明确、只是评审输入。"""
    repo = _repo(tmp_path)
    path = _experience(repo, _section(
        "有效条目",
        task="gemm",
        target="apple-m4-max",
        landing="targets/apple-m4-max/experience.md",
        source="`runs/apple-m4-max/host-rerun-gemm-20261120/stdout.txt`",
        boundary="gemm fp16 small",
        pointer="runs/apple-m4-max/host-rerun-gemm-20261120/stdout.txt",
    ))
    doc = audit_experience_file(path)
    section = doc["sections"][0]
    assert section["warnings"] == []
    assert section["legacy"] is False and section["legacy_missing_fields"] == []
    assert section["scope"]["task"]["verdict"] == "match"
    assert section["scope"]["target"]["verdict"] == "match"
    assert section["boundary"] == {
        "present": True, "declares_task": True, "declares_dtype": True, "declares_size": True,
    }
    assert section["promotion_review"] == {"eligible": True, "blockers": []}
    # 来源 与 宿主重跑 指同一份证据时只记一次，不把一条证据算成两条。
    assert [pointer["text"] for pointer in section["pointers"]] == [
        "runs/apple-m4-max/host-rerun-gemm-20261120/stdout.txt"
    ]
    assert doc["summary"]["pointers"] == {"total": 1, "valid": 1, "invalid": 0, "existing": 1, "missing": 0}


def test_verified_new_section_without_pointer_stays_incomplete(tmp_path):
    """@brief 新格式 verified 小节没有宿主重跑指针：报缺失，不编造指针。"""
    repo = _repo(tmp_path)
    path = _experience(repo, _section(
        "缺指针",
        task="gemm",
        target="apple-m4-max",
        landing="targets/apple-m4-max/experience.md",
        source="`runs/host-rerun-gemm-20261120/stdout.txt`",
    ))
    doc = audit_experience_file(path)
    section = doc["sections"][0]
    assert section["host_rerun"] is None
    assert [warning["code"] for warning in section["warnings"]] == ["verified_without_pointer"]
    assert section["promotion_review"]["eligible"] is False
    assert "no_host_pointer" in section["promotion_review"]["blockers"]
    # 来源 里的指针本身是对的，所以只有“没写宿主重跑”这一件事。
    assert doc["summary"]["pointers"]["valid"] == 1


@pytest.mark.parametrize(
    "pointer, reason",
    [
        ("runs/../etc/passwd.json", "dot_or_empty_segment"),
        ("runs/./x.json", "dot_or_empty_segment"),
        ("https://example.com/x.json", "url"),
        ("runs/a b/c.json", "whitespace"),
        ("docs/x.json", "not_under_runs"),
        ("runs/x", "too_short"),
        ("", "empty"),
    ],
)
def test_pointer_escapes_urls_whitespace_and_short_paths_are_invalid(tmp_path, pointer, reason):
    """@brief 越界、URL、空白、不在 runs/ 与过短指针一律无效，并给出原因。"""
    repo = _repo(tmp_path)
    path = _experience(repo, _section(
        "坏指针",
        task="gemm",
        target="apple-m4-max",
        landing="targets/apple-m4-max/experience.md",
        source="`runs/host-rerun-gemm-20261120/stdout.txt`",
        pointer=pointer,
    ))
    section = audit_experience_file(path)["sections"][0]
    assert section["host_rerun"]["valid"] is False
    assert section["host_rerun"]["reason"] == reason
    assert "host_pointer_invalid" in {warning["code"] for warning in section["warnings"]}
    assert section["promotion_review"]["eligible"] is False


def test_pointer_path_that_does_not_exist_is_missing_not_valid(tmp_path):
    """@brief 格式合格但路径不存在：valid 与 exists 分开，汇总记 missing。"""
    repo = _repo(tmp_path)
    path = _experience(repo, _section(
        "指针不存在",
        task="gemm",
        target="apple-m4-max",
        landing="targets/apple-m4-max/experience.md",
        source="`runs/host-rerun-gemm-20261120/stdout.txt`",
        pointer="runs/host-rerun-gone/x.json",
    ))
    doc = audit_experience_file(path)
    section = doc["sections"][0]
    assert section["host_rerun"]["valid"] is True and section["host_rerun"]["exists"] is False
    assert section["host_rerun"]["reason"] == "not_found"
    assert "host_pointer_missing" in {warning["code"] for warning in section["warnings"]}
    # 两条指针都记：来源 指向存在的那份，宿主重跑 指向不存在的路径。
    assert doc["summary"]["pointers"]["missing"] == 1 and doc["summary"]["pointers"]["existing"] == 1
    assert "missing_host_pointer" in section["promotion_review"]["blockers"]


def test_a_pointer_cited_in_the_body_is_validated_too(tmp_path):
    """@brief 正文（观察/本机实测等）里反引号写出的 runs/ 路径也要查存在，不只查来源。"""
    repo = _repo(tmp_path)
    path = _experience(repo, _section(
        "正文引用",
        task="gemm",
        target="apple-m4-max",
        landing="targets/apple-m4-max/experience.md",
        source="`runs/host-rerun-gemm-20261120/stdout.txt`",
        observation="证据放在 `runs/` 下，原始输出在 `runs/host-rerun-gone/x.json`。",
        pointer="runs/host-rerun-gemm-20261120/stdout.txt",
    ))
    doc = audit_experience_file(path)
    section = doc["sections"][0]
    assert "body_pointer_missing" in {warning["code"] for warning in section["warnings"]}
    assert [pointer["field"] for pointer in section["pointers"]] == ["宿主重跑", "正文"]
    # 裸 runs/ 是说明文字，不是指针。
    assert all(pointer["text"] != "runs/" for pointer in section["pointers"])


def test_inspect_pointer_accepts_files_and_directories_only_under_runs(tmp_path):
    """@brief 指针检查复用 valid_host_pointer：文件与目录都算存在，其它目录不算。"""
    repo = _repo(tmp_path)
    (repo / "runs" / "batch" / "20261120").mkdir(parents=True)
    assert inspect_pointer("runs/host-rerun-gemm-20261120/stdout.txt", repo=repo, field="宿主重跑")["kind"] == "file"
    directory = inspect_pointer("runs/batch/20261120", repo=repo, field="宿主重跑")
    assert directory["valid"] is True and directory["exists"] is True and directory["kind"] == "directory"
    assert inspect_pointer("tasks/gemm/opspec.json", repo=repo, field="来源")["valid"] is False


def test_target_and_landing_disagreement_is_a_warning_not_a_forgery_claim(tmp_path):
    """@brief 芯片与落点、落点与文件位置不一致时给 warning，不升级成“伪造”。"""
    repo = _repo(tmp_path)
    path = _experience(repo, _section(
        "落点错",
        task="gemm",
        target="apple-m4-max",
        landing="targets/other-chip/experience.md",
        source="`runs/host-rerun-gemm-20261120/stdout.txt`",
        pointer="runs/host-rerun-gemm-20261120/stdout.txt",
    ), _section(
        "芯片错",
        task="gemm",
        target="other-chip",
        landing="targets/other-chip/experience.md",
        source="`runs/other-chip/run1/result.json`",
        pointer="runs/other-chip/run1/result.json",
    ))
    doc = audit_experience_file(path)
    first, second = doc["sections"]
    codes = {warning["code"] for warning in first["warnings"]}
    assert {"target_landing_mismatch", "landing_file_mismatch"} <= codes
    assert "target_file_mismatch" not in codes  # 芯片 与文件位置一致，只错在落点
    second_codes = {warning["code"] for warning in second["warnings"]}
    assert "target_file_mismatch" in second_codes
    # 芯片和证据路径都指向 other-chip：作用域是 match（是落点错了），不是 mismatch。
    assert second["scope"]["target"] == {
        "value": "other-chip", "evidence": ["other-chip"], "verdict": "match",
        "reason": "the target appears in the evidence",
    }
    assert doc["summary"]["warnings"]["by_code"]["target_landing_mismatch"] == 1


def test_a_bare_source_path_is_a_pointer_and_is_checked(tmp_path):
    """@brief 来源里不带反引号的 runs/ 路径也要过 valid_host_pointer 与存在性检查。"""
    repo = _repo(tmp_path)
    path = _experience(repo, _section(
        "裸写来源",
        task="gemm",
        target="apple-m4-max",
        landing="targets/apple-m4-max/experience.md",
        source="原始输出在 runs/host-rerun-gemm_silu-reg-small-20261119/stdout.txt，未改动",
        pointer="runs/host-rerun-gemm-20261120/stdout.txt",
    ), _section(
        "裸写来源不存在",
        task="gemm",
        target="apple-m4-max",
        landing="targets/apple-m4-max/experience.md",
        source="原始输出在 runs/host-rerun-gone/x.json，未改动",
        pointer="runs/host-rerun-gemm-20261120/stdout.txt",
    ))
    doc = audit_experience_file(path)
    present, missing = doc["sections"]
    bare = [pointer for pointer in present["pointers"] if pointer["field"] == "来源"]
    assert [pointer["text"] for pointer in bare] == ["runs/host-rerun-gemm_silu-reg-small-20261119/stdout.txt"]
    assert bare[0]["valid"] is True and bare[0]["exists"] is True
    assert present["warnings"] == []
    missing_source = [pointer for pointer in missing["pointers"] if pointer["field"] == "来源"]
    assert missing_source[0]["valid"] is True and missing_source[0]["exists"] is False
    assert "source_pointer_missing" in {warning["code"] for warning in missing["warnings"]}
    assert doc["summary"]["pointers"]["missing"] == 1


def test_task_names_in_source_prose_are_not_mismatch_proof(tmp_path):
    """@brief 来源说明文字提到别的题名不算证据；只有路径或 ``tasks/<id>/`` 引用才证明错配。"""
    repo = _repo(tmp_path)
    path = _experience(repo, _section(
        "背景提及",
        task="gemm",
        target="apple-m4-max",
        landing="targets/apple-m4-max/experience.md",
        source="不要与 conv2d 混淆；原始输出 `runs/host-rerun-20261120/stdout.txt`",
        pointer="runs/host-rerun-20261120/stdout.txt",
    ), _section(
        "结构化引用",
        task="gemm",
        target="apple-m4-max",
        landing="targets/apple-m4-max/experience.md",
        source="`tasks/conv2d/` 的题面",
        pointer="runs/host-rerun-20261120/stdout.txt",
    ))
    doc = audit_experience_file(path)
    prose, citation = doc["sections"]
    assert prose["scope"]["task"]["evidence"] == []
    assert prose["scope"]["task"]["verdict"] == "unknown"
    assert "evidence_task_mismatch" not in {warning["code"] for warning in prose["warnings"]}
    assert "unknown_task" in prose["promotion_review"]["blockers"]
    # ``tasks/<id>/`` 是明确的任务引用，仍然算错配证据。
    assert citation["scope"]["task"]["verdict"] == "mismatch"
    assert citation["scope"]["task"]["evidence"] == ["conv2d"]
    assert "evidence_task_mismatch" in {warning["code"] for warning in citation["warnings"]}


def test_evidence_task_mismatch_requires_a_path_that_names_the_other_task(tmp_path):
    """@brief 证据路径明确写了另一个题才是 mismatch；路径看不出题就是 unknown。"""
    repo = _repo(tmp_path)
    path = _experience(repo, _section(
        "明确错配",
        task="gemm",
        target="apple-m4-max",
        landing="targets/apple-m4-max/experience.md",
        source="`runs/host-rerun-conv2d-20261120/stdout.txt`",
        pointer="runs/host-rerun-conv2d-20261120/stdout.txt",
    ), _section(
        "conv2d 的旧发现",
        task="gemm",
        target="apple-m4-max",
        landing="targets/apple-m4-max/experience.md",
        source="`runs/host-rerun-20261120/stdout.txt`",
        pointer="runs/host-rerun-20261120/stdout.txt",
    ))
    doc = audit_experience_file(path)
    proved, unknown = doc["sections"]
    assert proved["scope"]["task"]["verdict"] == "mismatch"
    assert proved["scope"]["task"]["evidence"] == ["conv2d"]
    assert "evidence_task_mismatch" in {warning["code"] for warning in proved["warnings"]}
    assert "task_mismatch" in proved["promotion_review"]["blockers"]
    # 标题里有 conv2d，但路径看不出题；不靠标题臆测。
    assert unknown["scope"]["task"]["verdict"] == "unknown"
    assert unknown["warnings"] == []
    assert "unknown_task" in unknown["promotion_review"]["blockers"]


def test_metal_terms_and_npu_simulation_are_marked_and_never_promotable(tmp_path):
    """@brief Metal/NPU 词要标出来；模拟声明把条目排除在真实硬件晋升之外。"""
    repo = _repo(tmp_path)
    path = _experience(repo, _section(
        "threadgroup 上限",
        task="gemm",
        target="apple-m4-max",
        landing="targets/apple-m4-max/experience.md",
        source="`runs/host-rerun-gemm-20261120/stdout.txt`",
        observation="Metal 的 simdgroup 与 threadgroup 静态占用。",
        boundary="fp16 small",
        pointer="runs/host-rerun-gemm-20261120/stdout.txt",
    ), _section(
        "NPU 模拟条目",
        task="gemm",
        target="apple-m4-max",
        landing="targets/apple-m4-max/experience.md",
        source="`runs/host-rerun-gemm-20261120/stdout.txt`",
        observation="Qingwei NPU simulation，simulated=true：这是模拟数据，不是真实硬件。",
        pointer="runs/host-rerun-gemm-20261120/stdout.txt",
    ))
    doc = audit_experience_file(path)
    metal, simulated = doc["sections"]
    assert metal["device_terms"] == {"metal": ["metal", "simdgroup", "threadgroup"], "npu": []}
    assert metal["simulation"] == {"declared": False, "terms": []}
    assert metal["promotion_review"]["eligible"] is True
    assert simulated["simulation"]["declared"] is True
    assert {"simulated", "simulation", "模拟"} <= set(simulated["simulation"]["terms"])
    assert "npu" in simulated["device_terms"]["npu"] and "qingwei" in simulated["device_terms"]["npu"]
    assert simulated["promotion_review"]["eligible"] is False
    assert "simulated_evidence" in simulated["promotion_review"]["blockers"]
    assert doc["summary"]["simulated_sections"] == 1
    assert doc["summary"]["device_specific_sections"] == 2


def test_legacy_section_is_incomplete_not_forged(tmp_path):
    """@brief 缺 题目/芯片/落点 的旧条目标 legacy，不补 target，也不报成伪造。"""
    repo = _repo(tmp_path)
    path = _experience(repo, _section(
        "旧条目",
        source="`runs/host-rerun-gemm-20261120/stdout.txt`",
        boundary="fp16 paper",
        pointer="runs/host-rerun-gemm-20261120/stdout.txt",
    ))
    doc = audit_experience_file(path)
    section = doc["sections"][0]
    assert section["legacy"] is True
    assert section["legacy_missing_fields"] == ["题目", "芯片", "落点"]
    assert section["target"] is None
    assert section["warnings"] == []
    assert section["scope"]["task"]["verdict"] == "unknown"
    assert section["promotion_review"]["eligible"] is False
    assert set(section["promotion_review"]["blockers"]) == {"legacy_incomplete", "unknown_task"}
    assert doc["summary"]["legacy_sections"] == 1
    assert doc["summary"]["warnings"]["total"] == 0


def test_cli_prints_json_and_summary_and_rejects_a_missing_target(tmp_path, capsys):
    """@brief CLI 两种输出都只读；不存在的 target 返回 2，不伪造空成功。"""
    repo = _repo(tmp_path)
    _experience(repo, _section(
        "有效条目",
        task="gemm",
        target="apple-m4-max",
        landing="targets/apple-m4-max/experience.md",
        source="`runs/host-rerun-gemm-20261120/stdout.txt`",
        pointer="runs/host-rerun-gemm-20261120/stdout.txt",
    ))
    assert main(["--repo", str(repo), "--json"]) == 0
    document = json.loads(capsys.readouterr().out)
    assert document["kind"] == "experience-scope-audit"
    assert document["read_only"] is True and document["notice"] == experience_scope.NOTICE
    assert [item["path"] for item in document["files"]] == ["targets/apple-m4-max/experience.md"]

    assert main(["--repo", str(repo), "--target", "apple-m4-max"]) == 0
    text = capsys.readouterr().out
    assert "经验作用域审计" in text and "汇总:" in text
    assert "Metal:" in text and "模拟: no" in text

    assert main(["--repo", str(repo), "--target", "nope"]) == 2
    assert "no experience file" in capsys.readouterr().err


def test_the_shipped_experience_file_audits_without_writing(tmp_path):
    """@brief 真实经验库可稳定审计：结构、只读、内部自洽；不钉今天条目数。"""
    before_bytes = SHIPPED.read_bytes()
    before_stat = SHIPPED.stat()
    # 只看这一个 target 目录：并行跑测试时仓库别处会写日志与缓存，那不是本审计的写。
    target_dir = SHIPPED.parent
    before_files = sorted(str(path.relative_to(target_dir)) for path in target_dir.rglob("*"))
    doc = audit_experience_file(SHIPPED)
    json.dumps(doc, ensure_ascii=False)  # 机器可读，且可序列化

    assert doc["path"] == "targets/apple-m4-max/experience.md"
    assert doc["target_id"] == "apple-m4-max"
    sections = doc["sections"]
    assert sections, "the shipped experience file must contain sections"
    for section in sections:
        assert set(section) >= {
            "title", "confidence", "target", "declared", "host_rerun", "pointers",
            "boundary", "device_terms", "simulation", "scope", "legacy",
            "legacy_missing_fields", "warnings", "promotion_review",
        }
        assert section["title"]
        assert set(section["scope"]) == {"task", "target", "landing_target"}
        for pointer in section["pointers"]:
            assert pointer["text"].startswith("runs/")
        for warning in section["warnings"]:
            assert warning["code"] in WARNING_CODES
        if section["promotion_review"]["eligible"]:
            # 只读审计给出的“可进入人工评审”必须自证：verified、非模拟、非 legacy、有存在的指针。
            assert section["confidence"] == "verified"
            assert section["simulation"]["declared"] is False
            assert section["legacy"] is False
            assert section["host_rerun"]["valid"] is True
            assert section["host_rerun"]["exists"] is True

    summary = doc["summary"]
    assert summary["sections"] == len(sections)
    assert sum(summary["confidence"].values()) == len(sections)
    assert summary["pointers"]["total"] == sum(len(section["pointers"]) for section in sections)
    assert summary["pointers"]["valid"] + summary["pointers"]["invalid"] == summary["pointers"]["total"]
    assert summary["pointers"]["existing"] + summary["pointers"]["missing"] == summary["pointers"]["valid"]
    assert summary["warnings"]["total"] == sum(len(section["warnings"]) for section in sections)
    assert summary["legacy_sections"] == sum(int(section["legacy"]) for section in sections)

    repo_doc = audit_repo(REPO)
    assert "targets/apple-m4-max/experience.md" in [item["path"] for item in repo_doc["files"]]
    assert repo_doc["summary"]["sections"] >= len(sections)
    assert "汇总:" in render_summary(repo_doc)

    # 只读：文件字节、mtime 与 target 目录里的文件集合都不变。
    after_stat = SHIPPED.stat()
    assert SHIPPED.read_bytes() == before_bytes
    assert (after_stat.st_mtime_ns, after_stat.st_size) == (before_stat.st_mtime_ns, before_stat.st_size)
    assert sorted(str(path.relative_to(target_dir)) for path in target_dir.rglob("*")) == before_files
