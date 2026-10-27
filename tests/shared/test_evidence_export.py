"""§9.2 证据导出与校验：白名单清单、净化、截断、两遍复核与 ``--against``。"""

from __future__ import annotations

import gzip
import io
import json
import tarfile
from pathlib import Path

import pytest

from taic import cli, evidence

_SOURCE = "kernel void evidence_fixture() {}\n"


def _run(tmp_path, monkeypatch) -> Path:
    """@brief 一场带排除项、绝对路径与候选文件的运行目录。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 的替换工具。
    @return run 目录。
    """
    runs = tmp_path / "runs"
    directory = runs / "conv2d" / "one"
    directory.mkdir(parents=True)
    monkeypatch.setenv("AI_LOWERLING_RUNS_DIR", str(runs))
    monkeypatch.setenv("OBSERVE_DB", str(tmp_path / "absent.sqlite"))
    monkeypatch.setenv("OBSERVE_REFRESH_ON_READ", "0")
    monkeypatch.setenv("TAIC_BLOB_ROOT", str(tmp_path / "blobs"))
    catalog = tmp_path / "kernels"
    folder = catalog / "apple-m4-max" / "relu" / "fp16" / "small"
    folder.mkdir(parents=True)
    (folder / "history.jsonl").write_text(json.dumps({
        "run_id": "one", "winner_label": "r1.p0", "confirmed_speedup": 1.5,
        "updated_at": 10.0, "source_sha256": "a" * 64,
    }) + "\n", encoding="utf-8")
    monkeypatch.setenv("TAIC_CATALOG_ROOT", str(catalog))
    (directory / "state.json").write_text(json.dumps({
        "task": "relu", "dtype": "fp16", "phase": "done", "started_at": 1.0,
        "run_dir": str(directory), "note": f"workspace {Path.home()}/work",
    }), encoding="utf-8")
    (directory / "result.json").write_text(json.dumps({
        "task": "relu", "dtype": "fp16", "size": "small", "stop_reason": "max_rounds",
        "protocol_version": "3", "best_speedup": 1.5, "run_dir": str(directory),
        "target_id": "apple-m4-max",
    }), encoding="utf-8")
    # 确认报告刻意不写：它必须进 manifest.missing，而不是悄悄消失。
    candidate = directory / "candidates" / "000_r1.p0"
    candidate.mkdir(parents=True)
    (candidate / "kernel.metal").write_text(_SOURCE, encoding="utf-8")
    (candidate / "report.json").write_text(json.dumps({"status": "ok", "stage": "bench"}), encoding="utf-8")
    (candidate / "NOTES.md").write_text("notes\n", encoding="utf-8")
    (directory / "feedback.jsonl").write_text(json.dumps({"seq": 1, "note": "ok"}) + "\n", encoding="utf-8")
    for name in ("session-1", "session-2"):
        work = directory / "agent_work" / name
        work.mkdir(parents=True)
        (work / "transcript.jsonl").write_text("x" * 64, encoding="utf-8")
    (directory / "transcript.jsonl").write_text("not whitelisted\n", encoding="utf-8")
    return directory


def _members(package: Path) -> tuple[dict, dict[str, bytes]]:
    """@brief 读出包的 manifest 与成员字节。

    @param package 包路径。
    @return ``(manifest, {名字: 字节})``。
    """
    with tarfile.open(package, "r:gz") as handle:
        manifest = json.loads(handle.extractfile("MANIFEST.json").read().decode("utf-8"))
        members = {
            item.name: handle.extractfile(item).read()
            for item in handle.getmembers()
            if item.isfile() and item.name != "MANIFEST.json"
        }
    return manifest, members


def _repack(source: Path, target: Path, mutate) -> None:
    """@brief 按包内成员重打一个包，便于构造各种篡改。

    @param source 原包。
    @param target 新包。
    @param mutate 接收 ``(manifest, members)`` 的回调。
    """
    manifest, members = _members(source)
    mutate(manifest, members)
    buffer = io.BytesIO()
    with gzip.GzipFile(fileobj=buffer, mode="wb", mtime=0) as gz:
        with tarfile.open(fileobj=gz, mode="w") as handle:
            items = [("MANIFEST.json", json.dumps(manifest, ensure_ascii=False, indent=1).encode("utf-8"))]
            items += sorted(members.items())
            for name, data in items:
                info = tarfile.TarInfo(name)
                info.size = len(data)
                info.mtime = 0
                info.mode = 0o644
                info.uid = info.gid = 0
                info.uname = info.gname = ""
                handle.addfile(info, io.BytesIO(data))
    target.write_bytes(buffer.getvalue())


def _repack_consistent(source: Path, target: Path, mutate) -> None:
    """@brief 重打一个哈希完全自洽的包（攻击者能重算全部哈希），只让语义检查兜底。

    @param source 原包。
    @param target 新包。
    @param mutate 接收 ``(manifest, members)`` 的回调，返回被改动的成员路径集合。
    """
    manifest, members = _members(source)
    changed = set(mutate(manifest, members) or ())
    for row in manifest["files"]:
        data = members.get(row["path"])
        if data is None:
            continue
        digest = evidence._sha256(data)
        row["bytes"] = len(data)
        row["sha256"] = digest
        row["source_sha256"] = digest
        row["source_bytes"] = len(data)
        row["sanitized"] = False
        row.pop("truncated", None)
    manifest["truncated"] = [row for row in manifest.get("truncated", []) if row.get("path") not in changed]
    entries = evidence._inventory_entries(manifest["files"], manifest["excluded"], manifest["missing"])
    manifest["inventory_sha256"] = evidence.inventory_sha256(entries)
    buffer = io.BytesIO()
    with gzip.GzipFile(fileobj=buffer, mode="wb", mtime=0) as gz:
        with tarfile.open(fileobj=gz, mode="w") as handle:
            items = [("MANIFEST.json", json.dumps(manifest, ensure_ascii=False, indent=1).encode("utf-8"))]
            items += sorted(members.items())
            for name, data in items:
                info = tarfile.TarInfo(name)
                info.size = len(data)
                info.mtime = 0
                info.mode = 0o644
                info.uid = info.gid = 0
                info.uname = info.gname = ""
                handle.addfile(info, io.BytesIO(data))
    target.write_bytes(buffer.getvalue())


def test_the_export_is_whitelisted_bounded_and_deterministic(tmp_path, monkeypatch):
    """@brief 验收①/⑤：白名单、排除项逐条枚举、missing 明写、两次导出 inventory 相同。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 的替换工具。
    """
    directory = _run(tmp_path, monkeypatch)
    first = tmp_path / "first.tar.gz"
    second = tmp_path / "second.tar.gz"
    manifest = evidence.export(directory, first)
    again = evidence.export(directory, second)

    assert manifest["run_id"] == "conv2d/one"
    paths = [row["path"] for row in manifest["files"]]
    assert paths == [
        "candidates/000_r1.p0/kernel.metal", "candidates/000_r1.p0/report.json",
        "feedback.jsonl", "publications.json", "result.json", "state.json",
    ]
    derived = next(row for row in manifest["files"] if row["path"] == "publications.json")
    assert derived["derived"] is True and derived["bytes"] <= 2048
    assert all(row["path"] != "transcript.jsonl" for row in manifest["files"])
    excluded = {row["path"]: row for row in manifest["excluded"]}
    assert excluded["agent_work/session-1/"]["files"] == 1 and excluded["agent_work/session-1/"]["bytes"] == 64
    assert excluded["agent_work/session-2/"]["files"] == 1
    assert excluded["transcript.jsonl"]["files"] == 1
    assert excluded["candidates/000_r1.p0/NOTES.md"]["bytes"] == len("notes\n")
    missing = {row["path"]: row["reason"] for row in manifest["missing"]}
    assert missing["confirm_report.json"] == "not produced"

    assert sum(int(row["bytes"]) for row in manifest["files"]) <= 2 * 1024 * 1024
    assert first.stat().st_size <= 500 * 1024
    assert manifest["inventory_sha256"] == again["inventory_sha256"]
    assert manifest["generated_at"] != 0
    verified = evidence.verify(first)
    assert verified["ok"] is True and verified["files"] == 6
    _manifest, members = _members(first)
    shard = json.loads(members["publications.json"])
    assert shard["target_id"] == "apple-m4-max" and shard["task"] == "relu"
    assert shard["catalog_history"][0]["confirmed_speedup"] == 1.5
    assert shard["current_publication"] and shard["current_publication"]["reason"]


def test_the_real_run_package_stays_inside_the_documented_size_budget(tmp_path, monkeypatch):
    """@brief 验收①的真实语料版本：那场 11 候选 / 84 claim 的运行，包 ≤2MB、gz ≤500KB。

    fixture 只有几个 KB，量不出“白名单漏进大文件”；这条只读真实 ``runs/``，输出到临时目录。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 的替换工具。
    """
    repo = Path(__file__).resolve().parents[2]
    directory = repo / "runs" / "apple-m4-max" / "20261025-070409_conv2d_fp16_deepseek-flash_g5yab2vc"
    if not directory.is_dir():
        pytest.skip("真实语料不在这台机器上")
    monkeypatch.setenv("AI_LOWERLING_RUNS_DIR", str(repo / "runs"))
    monkeypatch.setenv("OBSERVE_DB", str(tmp_path / "absent.sqlite"))
    monkeypatch.setenv("OBSERVE_REFRESH_ON_READ", "0")
    monkeypatch.setenv("TAIC_BLOB_ROOT", str(tmp_path / "blobs"))
    monkeypatch.setenv("TAIC_CATALOG_ROOT", str(repo / "kernels"))
    package = tmp_path / "real.tar.gz"
    manifest = evidence.export(directory, package)
    total = sum(int(row["bytes"]) for row in manifest["files"])
    assert len(manifest["files"]) == 27 and total <= 2 * 1024 * 1024, (len(manifest["files"]), total)
    assert package.stat().st_size <= 500 * 1024, package.stat().st_size
    assert evidence.verify(package)["ok"] is True


def test_verify_passes_and_names_every_tampering_case(tmp_path, monkeypatch):
    """@brief 验收②：完整包通过；删成员/改字节/改 inventory 都非零且点名。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 的替换工具。
    """
    directory = _run(tmp_path, monkeypatch)
    package = tmp_path / "good.tar.gz"
    evidence.export(directory, package)
    assert evidence.verify(package)["ok"] is True

    missing = tmp_path / "missing.tar.gz"
    _repack(package, missing, lambda manifest, members: members.pop("state.json"))
    result = evidence.verify(missing)
    assert result["ok"] is False and any("missing member: state.json" in item for item in result["failures"])

    changed = tmp_path / "changed.tar.gz"
    _repack(package, changed, lambda manifest, members: members.__setitem__(
        "result.json", members["result.json"].replace(b"best_speedup", b"best_speeduX")))
    result = evidence.verify(changed)
    assert result["ok"] is False and any("result.json" in item for item in result["failures"])

    inventory = tmp_path / "inventory.tar.gz"
    _repack(package, inventory, lambda manifest, members: manifest.__setitem__("inventory_sha256", "0" * 64))
    assert any("inventory_sha256" in item for item in evidence.verify(inventory)["failures"])


def test_verify_runs_without_the_index_or_runs_dir_and_against_rechecks_the_source(tmp_path, monkeypatch):
    """@brief 验收③：无索引/无 runs 的目录里能验；``--against`` 对源重算并点名变化。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 的替换工具。
    """
    directory = _run(tmp_path, monkeypatch)
    package = tmp_path / "run.tar.gz"
    evidence.export(directory, package)
    moved = tmp_path / "isolated"
    moved.mkdir()
    isolated = moved / "run.tar.gz"
    isolated.write_bytes(package.read_bytes())
    monkeypatch.setenv("AI_LOWERLING_RUNS_DIR", str(moved / "no-runs-here"))
    monkeypatch.setenv("OBSERVE_DB", str(moved / "no-index.sqlite"))
    assert evidence.verify(isolated)["ok"] is True

    assert evidence.verify(package, against=directory)["ok"] is True
    body = json.loads((directory / "result.json").read_text())
    body["best_speedup"] = 9.9
    (directory / "result.json").write_text(json.dumps(body), encoding="utf-8")
    changed = evidence.verify(package, against=directory)
    assert changed["ok"] is False
    assert any("result.json" in item for item in changed["failures"])


def test_the_package_carries_no_absolute_paths_and_the_run_dir_becomes_the_run_id(tmp_path, monkeypatch):
    """@brief 验收④：包内 grep 绝对路径为空；``state.json`` 的 ``run_dir`` 等于 run id。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 的替换工具。
    """
    directory = _run(tmp_path, monkeypatch)
    package = tmp_path / "clean.tar.gz"
    manifest = evidence.export(directory, package)
    _manifest, members = _members(package)
    manifest_bytes = json.dumps(manifest, ensure_ascii=False).encode("utf-8")
    for name, data in [("MANIFEST.json", manifest_bytes), *members.items()]:
        assert b"/Users/" not in data, name
        assert b"/home/" not in data, name
    state = json.loads(members["state.json"])
    result = json.loads(members["result.json"])
    assert state["run_dir"] == "conv2d/one" and result["run_dir"] == "conv2d/one"
    assert state["note"] == "workspace $HOME/work"
    row = next(item for item in manifest["files"] if item["path"] == "state.json")
    assert row["sanitized"] is True and row["source_sha256"] != row["sha256"]


def test_the_json_format_export_verifies_and_the_cli_wires_both_actions(tmp_path, monkeypatch, capsys):
    """@brief ``--format json`` 可校验；CLI 的 export/verify 返回码与篡改一致。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 的替换工具。
    @param capsys pytest 的 stdout/stderr 捕获。
    """
    directory = _run(tmp_path, monkeypatch)
    document = tmp_path / "evidence.json"
    assert cli.main(["evidence", "export", str(directory), "--out", str(document), "--format", "json"]) == 0
    assert json.loads(capsys.readouterr().out)["format"] == "json"
    assert evidence.verify(document)["ok"] is True

    package = tmp_path / "evidence.tar.gz"
    assert cli.main(["evidence", "export", str(directory), "--out", str(package)]) == 0
    capsys.readouterr()
    assert cli.main(["evidence", "verify", str(package), "--against", str(directory)]) == 0
    assert json.loads(capsys.readouterr().out)["ok"] is True

    broken = tmp_path / "broken.tar.gz"
    _repack(package, broken, lambda manifest, members: members.pop("feedback.jsonl"))
    assert cli.main(["evidence", "verify", str(broken)]) == 5
    assert json.loads(capsys.readouterr().out)["ok"] is False
    assert cli.main(["evidence", "export", str(tmp_path / "nope"), "--out", str(tmp_path / "x.tar.gz")]) == 5


def test_a_truncated_file_is_only_verified_as_a_prefix(tmp_path, monkeypatch):
    """@brief 截断项必须明写 full_sha256=null，verify 只承诺前缀一致。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 的替换工具。
    """
    directory = _run(tmp_path, monkeypatch)
    monkeypatch.setattr(evidence, "MAX_FILE_BYTES", 16)
    (directory / "feedback.jsonl").write_text("y" * 100, encoding="utf-8")
    package = tmp_path / "truncated.tar.gz"
    manifest = evidence.export(directory, package)
    row = next(item for item in manifest["files"] if item["path"] == "feedback.jsonl")
    assert row["truncated"] is True and row["bytes"] == 16 and row["source_bytes"] == 100
    item = next(entry for entry in manifest["truncated"] if entry["path"] == "feedback.jsonl")
    assert item["full_bytes"] == 100 and item["prefix_bytes"] == 16 and item["full_sha256"] is None
    result = evidence.verify(package)
    assert result["ok"] is True
    assert result["verified"] == "prefix"
    assert "feedback.jsonl" in result["prefix_verified"]


def test_a_source_rewrite_between_the_two_passes_fails_the_export(tmp_path, monkeypatch):
    """@brief §9.2 两遍流式：第二遍必须重开源文件；中途改写源要让导出失败且不留包。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 的替换工具。
    """
    directory = _run(tmp_path, monkeypatch)
    real_build = evidence.build

    def rewriting_build(root):
        manifest = real_build(root)
        body = json.loads((Path(root) / "result.json").read_text())
        body["best_speedup"] = 9.9
        (Path(root) / "result.json").write_text(json.dumps(body), encoding="utf-8")
        return manifest

    monkeypatch.setattr(evidence, "build", rewriting_build)
    package = tmp_path / "changed.tar.gz"
    with pytest.raises(RuntimeError, match="changed between passes"):
        evidence.export(directory, package)
    assert not package.exists()
    assert not list(tmp_path.glob(".*changed.tar.gz.*.tmp"))


def test_a_source_publications_file_is_part_of_the_inventory_and_changes_fail_against(tmp_path, monkeypatch):
    """@brief 源里真有 ``publications.json`` 时，它必须进清单；改写后 ``--against`` 必须失败。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 的替换工具。
    """
    directory = _run(tmp_path, monkeypatch)
    (directory / "publications.json").write_text("first", encoding="utf-8")
    package = tmp_path / "source-publications.tar.gz"
    manifest = evidence.export(directory, package)
    assert "publications.json" in {row["path"] for row in manifest["files"]}
    assert "publications.json" not in {row["path"] for row in manifest["excluded"]}
    assert evidence.verify(package, against=directory)["ok"] is True

    (directory / "publications.json").write_text("second and longer", encoding="utf-8")
    result = evidence.verify(package, against=directory)
    assert result["ok"] is False
    assert any("publications.json" in item for item in result["failures"])


def test_an_external_catalog_append_does_not_invalidate_against_a_unchanged_run_dir(tmp_path, monkeypatch):
    """@brief 派生的目录史分片不属于源清单：外部追加不能让未变的 RUN_DIR 在 ``--against`` 里失效。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 的替换工具。
    """
    directory = _run(tmp_path, monkeypatch)
    package = tmp_path / "external.tar.gz"
    evidence.export(directory, package)
    assert evidence.verify(package, against=directory)["ok"] is True
    history = tmp_path / "kernels" / "apple-m4-max" / "relu" / "fp16" / "small" / "history.jsonl"
    with history.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({
            "run_id": "later", "winner_label": "r2.p0", "confirmed_speedup": 1.7,
            "updated_at": 20.0, "source_sha256": "b" * 64,
        }) + "\n")
    assert evidence.verify(package, against=directory)["ok"] is True


def test_a_symlinked_candidates_directory_is_excluded_and_never_packaged(tmp_path, monkeypatch):
    """@brief 评审回归：``candidates`` 是符号链接时整条排除，绝不把外部内核打进包。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 的替换工具。
    """
    import shutil

    directory = _run(tmp_path, monkeypatch)
    shutil.rmtree(directory / "candidates")
    external = tmp_path / "external" / "candidates" / "000_r1.p0"
    external.mkdir(parents=True)
    (external / "kernel.metal").write_text("TOP SECRET EXTERNAL KERNEL\n", encoding="utf-8")
    (external / "report.json").write_text("{\"status\":\"ok\"}\n", encoding="utf-8")
    (directory / "candidates").symlink_to(external.parent, target_is_directory=True)

    package = tmp_path / "symlink.tar.gz"
    manifest = evidence.export(directory, package)
    assert [row for row in manifest["excluded"] if row["path"] == "candidates"] == [{
        "path": "candidates", "reason": "symlink", "bytes": 0, "files": 0,
    }]
    _manifest, members = _members(package)
    assert not any(name.startswith("candidates/") for name in members)
    assert b"TOP SECRET" not in b"".join(members.values())
    assert evidence.verify(package)["ok"] is True


def test_a_file_that_disappears_between_inventory_and_read_is_marked_and_fails_verify(tmp_path, monkeypatch):
    """@brief 评审回归：读失败不能变成“空文件”并通过校验。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 的替换工具。
    """
    directory = _run(tmp_path, monkeypatch)
    package = tmp_path / "vanished.tar.gz"
    real_entries = evidence.source_entries

    def delete_after_inventory(source):
        entries = real_entries(source)
        (Path(source) / "result.json").unlink(missing_ok=True)
        return entries

    monkeypatch.setattr(evidence, "source_entries", delete_after_inventory)
    manifest = evidence.export(directory, package)
    row = next(row for row in manifest["files"] if row["path"] == "result.json")
    assert row.get("unreadable")
    _manifest, members = _members(package)
    assert "result.json" not in members
    result = evidence.verify(package)
    assert result["ok"] is False
    assert any("unreadable source: result.json" in item for item in result["failures"])


def test_a_truncated_prefix_is_sanitized_too(tmp_path, monkeypatch):
    """@brief 评审回归：截断分支也必须净化；否则导出成功但自己的 verify 会拒绝。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 的替换工具。
    """
    directory = _run(tmp_path, monkeypatch)
    monkeypatch.setattr(evidence, "MAX_FILE_BYTES", 64)
    (directory / "feedback.jsonl").write_bytes(b"/Users/alice/private\n" + b"x" * 200)
    package = tmp_path / "truncated-absolute.tar.gz"
    manifest = evidence.export(directory, package)
    row = next(row for row in manifest["files"] if row["path"] == "feedback.jsonl")
    assert row["truncated"] is True
    _manifest, members = _members(package)
    assert b"/Users/" not in members["feedback.jsonl"]
    assert b"$ABS" in members["feedback.jsonl"]
    result = evidence.verify(package)
    assert result["ok"] is True and "feedback.jsonl" in result["prefix_verified"]


def test_a_relative_run_dir_does_not_corrupt_json(tmp_path, monkeypatch):
    """@brief 评审回归：``RUN_DIR=.`` 只能用 ``resolve()`` 后的绝对路径做替换，不能逐字符替换。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 的替换工具。
    """
    import os

    directory = _run(tmp_path, monkeypatch)
    package = tmp_path / "relative-dot.tar.gz"
    old_cwd = Path.cwd()
    try:
        os.chdir(directory)
        evidence.export(Path("."), package)
    finally:
        os.chdir(old_cwd)
    _manifest, members = _members(package)
    state = json.loads(members["state.json"])
    assert state["run_dir"] == "conv2d/one"
    assert state["started_at"] == 1.0
    assert evidence.verify(package)["ok"] is True


def test_a_malformed_manifest_is_rejected(tmp_path):
    """@brief 评审回归：``files`` 里没有可用行时不能因为“空清单”而被当成通过。

    @param tmp_path pytest 提供的临时目录。
    """
    package = tmp_path / "bad-schema.json"
    package.write_text(json.dumps({
        "manifest": {
            "run_id": "bad", "files": ["not-a-row"], "excluded": [], "missing": [],
            "inventory_sha256": evidence.inventory_sha256([]),
        },
        "files": {},
    }), encoding="utf-8")
    result = evidence.verify(package)
    assert result["ok"] is False
    assert any("non-object" in item or "schema" in item for item in result["failures"])


def test_absolute_paths_in_the_manifest_itself_are_rejected(tmp_path, monkeypatch):
    """@brief 评审回归：``MANIFEST.json`` 也要扫绝对路径，不能只扫文件成员。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 的替换工具。
    """
    directory = _run(tmp_path, monkeypatch)
    package = tmp_path / "manifest-absolute.json"
    evidence.export(directory, package, fmt="json")
    document = json.loads(package.read_text(encoding="utf-8"))
    document["manifest"]["taic_commit"] = "/Users/alice/private"
    package.write_text(json.dumps(document), encoding="utf-8")
    result = evidence.verify(package)
    assert result["ok"] is False
    assert any("MANIFEST.json" in item for item in result["failures"])


def test_the_publications_shard_stays_under_two_kilobytes_after_pretty_printing(tmp_path, monkeypatch):
    """@brief 评审回归：2KB 上限必须按最终（缩进）文本算，不是紧凑序列化。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 的替换工具。
    """
    directory = _run(tmp_path, monkeypatch)
    history = tmp_path / "kernels" / "apple-m4-max" / "relu" / "fp16" / "small" / "history.jsonl"
    with history.open("a", encoding="utf-8") as handle:
        for number in range(20):
            handle.write(json.dumps({
                "run_id": f"run-{number:02d}-" + "x" * 20, "winner_label": f"r{number}.p0",
                "confirmed_speedup": 1.2, "updated_at": float(number), "source_sha256": "a" * 64,
            }) + "\n")
    package = tmp_path / "publication-size.tar.gz"
    manifest = evidence.export(directory, package)
    row = next(row for row in manifest["files"] if row["path"] == "publications.json")
    assert row["bytes"] <= 2048, row["bytes"]
    _manifest, members = _members(package)
    assert len(members["publications.json"]) <= 2048
    assert evidence.verify(package)["ok"] is True


def test_a_wrong_type_candidates_path_is_listed_in_the_inventory(tmp_path, monkeypatch):
    """@brief 评审回归：``candidates`` 不是目录时也要进 ``excluded``，后续改写要被发现。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 的替换工具。
    """
    import shutil

    directory = _run(tmp_path, monkeypatch)
    shutil.rmtree(directory / "candidates")
    (directory / "candidates").write_text("first", encoding="utf-8")
    package = tmp_path / "wrong-type.tar.gz"
    manifest = evidence.export(directory, package)
    assert any(row["path"] == "candidates" for row in manifest["excluded"])
    assert evidence.verify(package, against=directory)["ok"] is True
    (directory / "candidates").write_text("second and longer", encoding="utf-8")
    assert evidence.verify(package, against=directory)["ok"] is False


def test_other_absolute_namespaces_are_sanitized_and_verified(tmp_path, monkeypatch):
    """@brief 评审回归：``/tmp``、``/private`` 等绝对路径同样不能留在包里。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 的替换工具。
    """
    directory = _run(tmp_path, monkeypatch)
    (directory / "feedback.jsonl").write_text('{"note":"/tmp/private evidence"}\n', encoding="utf-8")
    package = tmp_path / "tmp-absolute.tar.gz"
    evidence.export(directory, package)
    _manifest, members = _members(package)
    assert b"/tmp/" not in members["feedback.jsonl"]
    assert b"$ABS" in members["feedback.jsonl"]
    assert evidence.verify(package)["ok"] is True


def test_an_excluded_source_change_between_the_two_passes_fails_the_export(tmp_path, monkeypatch):
    """@brief 评审回归：排除项（agent_work）在导出中途变化也要让导出失败。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 的替换工具。
    """
    directory = _run(tmp_path, monkeypatch)
    real_build = evidence.build

    def rewrite_excluded(root):
        manifest = real_build(root)
        (Path(root) / "agent_work" / "session-1" / "transcript.jsonl").write_text(
            "second and longer", encoding="utf-8")
        return manifest

    monkeypatch.setattr(evidence, "build", rewrite_excluded)
    package = tmp_path / "excluded-race.tar.gz"
    with pytest.raises(RuntimeError, match="source inventory changed"):
        evidence.export(directory, package)
    assert not package.exists()


def test_unsafe_member_names_are_rejected(tmp_path, monkeypatch):
    """@brief 评审回归：包内成员不能是绝对路径或含 ``..``。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 的替换工具。
    """
    directory = _run(tmp_path, monkeypatch)
    package = tmp_path / "good.tar.gz"
    evidence.export(directory, package)
    evil = tmp_path / "evil.tar.gz"
    _repack(package, evil, lambda manifest, members: members.__setitem__("../escape", b"x"))
    result = evidence.verify(evil)
    assert result["ok"] is False
    assert any("unreadable package" in item or "unsafe member" in item for item in result["failures"])


def test_a_symlinked_candidate_file_is_registered_exactly_once(tmp_path, monkeypatch):
    """@brief 评审回归：候选内单个文件是符号链接时只进 ``excluded``，不能同时进 ``missing``。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 的替换工具。
    """
    directory = _run(tmp_path, monkeypatch)
    kernel = directory / "candidates" / "000_r1.p0" / "kernel.metal"
    external = tmp_path / "external-kernel.metal"
    external.write_text("TOP SECRET EXTERNAL\n", encoding="utf-8")
    kernel.unlink()
    kernel.symlink_to(external)

    package = tmp_path / "file-symlink.tar.gz"
    manifest = evidence.export(directory, package)
    path = "candidates/000_r1.p0/kernel.metal"
    excluded_paths = [row["path"] for row in manifest["excluded"]]
    missing_paths = [row["path"] for row in manifest["missing"]]
    assert excluded_paths.count(path) == 1
    assert path not in missing_paths
    _manifest, members = _members(package)
    assert path not in members
    assert b"TOP SECRET" not in b"".join(members.values())
    assert evidence.verify(package)["ok"] is True


def test_absolute_paths_outside_the_known_prefixes_are_sanitized_and_rejected(tmp_path, monkeypatch):
    """@brief 评审回归：``/etc``、``/root``、``/opt`` 等通用绝对路径不能原样入包。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 的替换工具。
    """
    directory = _run(tmp_path, monkeypatch)
    (directory / "feedback.jsonl").write_text(
        '{"a":"/etc/passwd","b":"/root/secret","c":"/opt/private","d":"a/b"}\n', encoding="utf-8")
    package = tmp_path / "generic-absolute.tar.gz"
    evidence.export(directory, package)
    _manifest, members = _members(package)
    text = members["feedback.jsonl"].decode("utf-8")
    assert "/etc/passwd" not in text and "/root/secret" not in text and "/opt/private" not in text
    assert "$ABS" in text and "a/b" in text
    assert evidence.verify(package)["ok"] is True

    tampered = tmp_path / "generic-tampered.tar.gz"
    _repack(package, tampered, lambda manifest, loaded: loaded.__setitem__(
        "feedback.jsonl", b'{"a":"/opt/private"}\n'))
    result = evidence.verify(tampered)
    assert result["ok"] is False
    assert any("absolute path in member" in item for item in result["failures"])


def test_an_append_to_an_unsanitized_file_between_passes_fails_the_export(tmp_path, monkeypatch):
    """@brief 评审回归：清单只记录旧 bytes；第二遍必须核对当前大小并多读一字节，追加也要失败。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 的替换工具。
    """
    directory = _run(tmp_path, monkeypatch)
    report = directory / "candidates" / "000_r1.p0" / "report.json"
    real_build = evidence.build

    def append_after_build(root):
        manifest = real_build(root)
        with report.open("a", encoding="utf-8") as handle:
            handle.write(" ")
        return manifest

    monkeypatch.setattr(evidence, "build", append_after_build)
    package = tmp_path / "append-race.tar.gz"
    # 明确指出是写前大小核对，而不是 `_sources_unchanged` 的库存变化。
    with pytest.raises(RuntimeError, match="source size changed between passes"):
        evidence.export(directory, package)
    assert not package.exists()
    assert not list(tmp_path.glob(".append-race.tar.gz*"))


def test_a_symlinked_top_level_file_is_excluded_not_missing(tmp_path, monkeypatch):
    """@brief 顶层文件是符号链接时也只进 ``excluded``，不跟随、不记成 missing。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 的替换工具。
    """
    directory = _run(tmp_path, monkeypatch)
    result = directory / "result.json"
    external = tmp_path / "external-result.json"
    external.write_text('{"secret": "EXTERNAL"}\n', encoding="utf-8")
    result.unlink()
    result.symlink_to(external)
    package = tmp_path / "top-symlink.tar.gz"
    manifest = evidence.export(directory, package)
    excluded = [row["path"] for row in manifest["excluded"]]
    missing = [row["path"] for row in manifest["missing"]]
    assert excluded.count("result.json") == 1
    assert "result.json" not in missing
    _manifest, members = _members(package)
    assert "result.json" not in members
    assert b"EXTERNAL" not in b"".join(members.values())
    assert evidence.verify(package)["ok"] is True


def test_the_second_pass_trailing_byte_check_rejects_growth_after_the_stat(tmp_path, monkeypatch):
    """@brief 写前大小核对通过（注入旧大小）后，写后多读的一字节仍必须发现追加。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 的替换工具。
    """
    directory = _run(tmp_path, monkeypatch)
    report = directory / "candidates" / "000_r1.p0" / "report.json"
    real_build = evidence.build
    real_size = evidence._source_size

    def append_after_build(root):
        manifest = real_build(root)
        with report.open("a", encoding="utf-8") as handle:
            handle.write(" ")
        return manifest

    def stale_size(path):
        # 模拟“写前 stat 还是旧大小、读完之后文件已变长”的窗口。
        if Path(path) == report:
            return len(report.read_bytes()) - 1
        return real_size(path)

    monkeypatch.setattr(evidence, "build", append_after_build)
    monkeypatch.setattr(evidence, "_source_size", stale_size)
    package = tmp_path / "trailing-race.tar.gz"
    with pytest.raises(RuntimeError, match="bytes changed between passes"):
        evidence.export(directory, package)
    assert not package.exists()
    assert not list(tmp_path.glob(".trailing-race.tar.gz*"))


def test_a_wrong_type_file_inside_a_candidate_is_registered_exactly_once(tmp_path, monkeypatch):
    """@brief 候选内 `report.json` 是目录（错类型）时只进 excluded 一次，不跟随、不记 missing。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 的替换工具。
    """
    directory = _run(tmp_path, monkeypatch)
    path = directory / "candidates" / "000_r1.p0" / "report.json"
    path.unlink()
    path.mkdir()
    (path / "nested.txt").write_text("must not be packaged\n", encoding="utf-8")
    package = tmp_path / "wrong-type-file.tar.gz"
    manifest = evidence.export(directory, package)
    relative = "candidates/000_r1.p0/report.json"
    excluded = [row["path"] for row in manifest["excluded"]]
    missing = [row["path"] for row in manifest["missing"]]
    files = [row["path"] for row in manifest["files"]]
    assert sum(1 for path in excluded if path in (relative, relative + "/")) == 1
    assert relative not in missing and relative not in files
    _manifest, members = _members(package)
    assert relative not in members
    assert b"must not be packaged" not in b"".join(members.values())
    assert evidence.verify(package)["ok"] is True


def test_division_is_not_an_absolute_path_and_double_slash_paths_are_rejected(tmp_path, monkeypatch):
    """@brief 除法操作数不能被净化成 `$ABS`；`//etc`、`///opt` 必须整段净化并被 verify 拒绝。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 的替换工具。
    """
    for probe in ("//etc/passwd", "///opt/private"):
        match = evidence.ABS_PATH_RE.search(probe)
        assert match and match.group(0) == probe, probe
    for probe in ("return (a + b)/b;", "x = v[i]/n;", "return a / b;", "a/b", "1/2"):
        assert evidence.ABS_PATH_RE.search(probe) is None, probe

    directory = _run(tmp_path, monkeypatch)
    kernel = directory / "candidates" / "000_r1.p0" / "kernel.metal"
    code = "kernel void f() { float x = (a + b)/b; x = v[i]/n; }\n"
    kernel.write_text(code, encoding="utf-8")
    (directory / "feedback.jsonl").write_text(json.dumps({
        "double": "//etc/passwd",
        "triple": "///opt/private",
        "div": "return (a + b)/b;",
        "index": "x = v[i]/n;",
    }) + "\n", encoding="utf-8")
    package = tmp_path / "division-and-double-slash.tar.gz"
    evidence.export(directory, package)
    _manifest, members = _members(package)
    assert members["candidates/000_r1.p0/kernel.metal"] == code.encode("utf-8")
    feedback = json.loads(members["feedback.jsonl"].decode("utf-8"))
    assert feedback["double"] == "$ABS" and feedback["triple"] == "$ABS"
    assert feedback["div"] == "return (a + b)/b;" and feedback["index"] == "x = v[i]/n;"
    assert evidence.verify(package)["ok"] is True

    for name, payload in (
        ("etc-passwd.tar.gz", b'{"p":"/etc/passwd"}\n'),
        ("root-secret.tar.gz", b'{"p":"/root/secret"}\n'),
        ("double-slash.tar.gz", b'{"p":"//etc/passwd"}\n'),
        ("triple-slash.tar.gz", b'{"p":"///opt/private"}\n'),
    ):
        tampered = tmp_path / name
        _repack_consistent(
            package, tampered,
            lambda manifest, loaded, body=payload: loaded.__setitem__("feedback.jsonl", body) or {"feedback.jsonl"},
        )
        result = evidence.verify(tampered)
        assert result["ok"] is False, name
        assert any("absolute path in member" in item for item in result["failures"]), (name, result["failures"])


def test_embedded_colon_paths_are_sanitized_and_rejected(tmp_path, monkeypatch):
    """@brief `path:/etc/passwd`、`C:/Users/...` 这类冒号后的绝对路径也要净化；URL 不动。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 的替换工具。
    """
    for probe in ("path:/etc/passwd", "x:/root/secret", "C:/Users/example"):
        assert evidence.ABS_PATH_RE.search(probe), probe
    for probe in ("http://example.com/a/b", "https://x/y", "a/b", "10:30/pm"):
        assert evidence.ABS_PATH_RE.search(probe) is None, probe

    directory = _run(tmp_path, monkeypatch)
    (directory / "feedback.jsonl").write_text(json.dumps({
        "note": "path:/etc/passwd",
        "drive": "C:/Users/example",
        "url": "http://example.com/a/b",
        "secure": "scheme:https://x/y",
    }) + "\n", encoding="utf-8")
    package = tmp_path / "colon-absolute.tar.gz"
    evidence.export(directory, package)
    _manifest, members = _members(package)
    text = members["feedback.jsonl"].decode("utf-8")
    assert "/etc/passwd" not in text and "C:/Users/example" not in text
    assert "$ABS" in text
    assert "http://example.com/a/b" in text and "scheme:https://x/y" in text
    assert evidence.verify(package)["ok"] is True

    tampered = tmp_path / "colon-tampered.tar.gz"
    _repack(package, tampered, lambda manifest, loaded: loaded.__setitem__(
        "feedback.jsonl", b'{"p":"path:/root/secret"}\n'))
    result = evidence.verify(tampered)
    assert result["ok"] is False
    assert any("absolute path in member" in item for item in result["failures"])


def test_numeric_and_unicode_absolute_paths_are_sanitized_and_verified(tmp_path, monkeypatch):
    """@brief `/123`、`/opt/私货` 这类合法 POSIX 路径也要整段替换，URL/相对路径保持原样。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 的替换工具。
    """
    for probe in ("/123", "/opt/私货", "/v1/2", "path:/123", r"/opt/\u79c1\u8d27"):
        assert evidence.ABS_PATH_RE.search(probe), probe
    for probe in ("http://example.com/a/b", "https://x/y/123", "a/b", "1/2", "10/26/2024", "10:30/pm"):
        assert evidence.ABS_PATH_RE.search(probe) is None, probe

    directory = _run(tmp_path, monkeypatch)
    state_path = directory / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["numeric_path"] = "/123"
    state["unicode_path"] = "/opt/私货"
    state["url"] = "http://example.com/a/b"
    state["relative"] = "a/b"
    state_path.write_text(json.dumps(state), encoding="utf-8")

    package = tmp_path / "numeric-unicode.tar.gz"
    evidence.export(directory, package)
    _manifest, members = _members(package)
    text = members["state.json"].decode("utf-8")
    assert "/123" not in text and "/opt/私货" not in text and "私货" not in text
    assert "$ABS" in text
    assert "http://example.com/a/b" in text and "a/b" in text
    assert evidence.verify(package)["ok"] is True

    for name, payload in (
        ("numeric-member.tar.gz", b'{"p":"/123"}\n'),
        ("unicode-member.tar.gz", '{"p":"/opt/私货"}\n'.encode()),
    ):
        tampered = tmp_path / name
        _repack(package, tampered, lambda manifest, loaded, body=payload: loaded.__setitem__("state.json", body))
        result = evidence.verify(tampered)
        assert result["ok"] is False
        assert any("absolute path in member" in item for item in result["failures"])


def test_publications_symlink_fifo_and_directory_are_excluded_and_never_derived(tmp_path, monkeypatch):
    """@brief publications 为 symlink/FIFO/目录时只进 excluded 一次；不得再派生同名 member。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 的替换工具。
    """
    import os
    import shutil

    for kind in ("symlink", "fifo", "dir"):
        directory = _run(tmp_path / kind, monkeypatch)
        source = directory / "publications.json"
        if kind == "symlink":
            external = tmp_path / kind / "external.json"
            external.write_text('{"external": true}\n', encoding="utf-8")
            source.symlink_to(external)
        elif kind == "fifo":
            os.mkfifo(source)
        else:
            source.mkdir()
            (source / "nested.txt").write_text("not a member\n", encoding="utf-8")
        package = tmp_path / f"publications-{kind}.tar.gz"
        manifest = evidence.export(directory, package)
        paths = [row["path"] for row in manifest["files"]]
        excluded = [row["path"] for row in manifest["excluded"]]
        assert "publications.json" not in paths
        assert excluded.count("publications.json") + excluded.count("publications.json/") == 1
        result = evidence.verify(package)
        assert result["ok"] is True, result["failures"]
        assert not any("duplicate" in item for item in result["failures"])
        shutil.rmtree(directory)


def test_an_empty_top_level_directory_is_registered_in_the_inventory(tmp_path, monkeypatch):
    """@brief 空目录不能从清单里消失：它自己要有 excluded 行，verify 仍覆盖源清单。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 的替换工具。
    """
    directory = _run(tmp_path, monkeypatch)
    empty = directory / "empty_dir"
    empty.mkdir()
    package = tmp_path / "empty-dir.tar.gz"
    manifest = evidence.export(directory, package)
    rows = [row for row in manifest["excluded"] if row["path"] == "empty_dir/"]
    assert len(rows) == 1 and rows[0]["bytes"] == 0 and rows[0]["files"] == 0
    assert evidence.verify(package)["ok"] is True
    assert evidence.verify(package, against=directory)["ok"] is True


def test_escaped_slashes_in_a_self_consistent_package_are_rejected(tmp_path, monkeypatch):
    """@brief `\u002f` 与 `\/` 形式的绝对路径即使重算全部哈希，verify 也必须拒绝。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 的替换工具。
    """
    directory = _run(tmp_path, monkeypatch)
    package = tmp_path / "escaped.tar.gz"
    evidence.export(directory, package)

    for name, body in (
        ("unicode-slash.tar.gz", b'{"p":"\\u002fetc\\u002fpasswd"}\n'),
        ("solidus.tar.gz", b'{"p":"\\/etc\\/passwd"}\n'),
        ("unicode-segment.tar.gz", '{"p":"/opt/\\u79c1\\u8d27"}\n'.encode("utf-8")),
    ):
        attack = tmp_path / name
        _repack_consistent(
            package, attack,
            lambda manifest, members, payload=body: members.__setitem__("feedback.jsonl", payload) or {"feedback.jsonl"},
        )
        result = evidence.verify(attack)
        assert result["ok"] is False, name
        assert any("absolute path in member" in item for item in result["failures"]), result["failures"]


def test_a_unicode_run_dir_sanitizes_the_escaped_run_dir_to_the_run_id(tmp_path, monkeypatch):
    """@brief Unicode RUN_DIR 经默认 json.dumps（转义）后，run_dir 必须仍是 run id，导出后 verify 通过。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 的替换工具。
    """
    runs = tmp_path / "runs"
    directory = runs / "中文" / "one"
    (directory / "candidates" / "000_r1.p0").mkdir(parents=True)
    monkeypatch.setenv("AI_LOWERLING_RUNS_DIR", str(runs))
    monkeypatch.setenv("OBSERVE_DB", str(tmp_path / "absent.sqlite"))
    monkeypatch.setenv("OBSERVE_REFRESH_ON_READ", "0")
    monkeypatch.setenv("TAIC_CATALOG_ROOT", str(tmp_path / "kernels"))
    (directory / "state.json").write_text(json.dumps({
        "task": "relu", "dtype": "fp16", "run_dir": str(directory),
    }), encoding="utf-8")
    (directory / "result.json").write_text(json.dumps({
        "task": "relu", "dtype": "fp16", "target_id": "apple-m4-max", "run_dir": str(directory),
    }), encoding="utf-8")
    (directory / "candidates" / "000_r1.p0" / "kernel.metal").write_text(_SOURCE, encoding="utf-8")
    (directory / "candidates" / "000_r1.p0" / "report.json").write_text('{"status": "ok"}', encoding="utf-8")
    (directory / "feedback.jsonl").write_text(
        json.dumps({"seq": 1, "note": "http://example.com/中文/one"}) + "\n", encoding="utf-8")

    package = tmp_path / "unicode-run.tar.gz"
    evidence.export(directory, package)
    _manifest, members = _members(package)
    state = json.loads(members["state.json"].decode("utf-8"))
    assert state["run_dir"] == "中文/one"
    feedback = json.loads(members["feedback.jsonl"].decode("utf-8"))
    assert feedback["note"] == "http://example.com/中文/one"
    result = evidence.verify(package)
    assert result["ok"] is True, result["failures"]


def test_non_utf8_evidence_fails_closed_on_export_and_verify(tmp_path, monkeypatch):
    """@brief 不可解码的输入在导出与校验间同一策略：导出失败、包不留；包内遇到则 verify 失败。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 的替换工具。
    """
    directory = _run(tmp_path, monkeypatch)
    (directory / "feedback.jsonl").write_bytes(b"\xff /etc/passwd\n")
    package = tmp_path / "binary.tar.gz"
    with pytest.raises(RuntimeError, match="not UTF-8"):
        evidence.export(directory, package)
    assert not package.exists()
    assert not list(tmp_path.glob(".binary.tar.gz*"))

    # 从干净源做一个包，再把成员改成非 UTF-8 并重算全部哈希：verify 仍必须拒绝。
    (directory / "feedback.jsonl").write_text(json.dumps({"seq": 1}) + "\n", encoding="utf-8")
    clean = tmp_path / "clean.tar.gz"
    evidence.export(directory, clean)
    attack = tmp_path / "binary-attack.tar.gz"
    _repack_consistent(
        clean, attack,
        lambda manifest, members: members.__setitem__("feedback.jsonl", b"\xff /etc/passwd\n") or {"feedback.jsonl"},
    )
    result = evidence.verify(attack)
    assert result["ok"] is False
    assert any("not UTF-8" in item for item in result["failures"]), result["failures"]

    # 源目录不可解码时 --against 也失败，而不是抛未捕获异常。
    (directory / "feedback.jsonl").write_bytes(b"\xff /etc/passwd\n")
    against = evidence.verify(clean, against=directory)
    assert against["ok"] is False
    assert any("source unreadable" in item and "not UTF-8" in item for item in against["failures"])


def test_derived_publications_never_follow_symlinked_run_metadata(tmp_path, monkeypatch):
    """@brief result/候选 kernel 是 symlink 时，派生 publications 不能读外部 JSON/哈希。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 的替换工具。
    """
    directory = _run(tmp_path, monkeypatch)
    external = tmp_path / "external-result.json"
    external.write_text(json.dumps({
        "task": "EXTERNAL_TASK", "dtype": "EXTERNAL_DTYPE", "size": "EXTERNAL_SIZE",
        "target_id": "EXTERNAL_TARGET", "confirm_winner_label": "r1.p0",
    }), encoding="utf-8")
    (directory / "result.json").unlink()
    (directory / "result.json").symlink_to(external)
    external_kernel = tmp_path / "external-kernel.metal"
    external_kernel.write_text("TOP SECRET EXTERNAL KERNEL\n", encoding="utf-8")
    kernel = directory / "candidates" / "000_r1.p0" / "kernel.metal"
    kernel.unlink()
    kernel.symlink_to(external_kernel)

    package = tmp_path / "derived-symlink.tar.gz"
    evidence.export(directory, package)
    _manifest, members = _members(package)
    derived = members["publications.json"].decode("utf-8")
    assert "EXTERNAL_TASK" not in derived and "EXTERNAL_TARGET" not in derived
    assert "TOP SECRET" not in b"".join(members.values()).decode("utf-8", errors="replace")
    document = json.loads(derived)
    assert document["current_publication"] is None
    result = evidence.verify(package)
    assert result["ok"] is True, result["failures"]


def test_a_duplicate_tar_member_is_rejected_before_scanning_its_content(tmp_path, monkeypatch):
    """@brief 同名 member 不能后写覆盖前写，否则恶意第一份能绕过绝对路径扫描。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 的替换工具。
    """
    directory = _run(tmp_path, monkeypatch)
    package = tmp_path / "clean.tar.gz"
    evidence.export(directory, package)
    manifest, members = _members(package)
    buffer = io.BytesIO()
    with gzip.GzipFile(fileobj=buffer, mode="wb", mtime=0) as gz:
        with tarfile.open(fileobj=gz, mode="w") as handle:
            items = [("MANIFEST.json", json.dumps(manifest, ensure_ascii=False, indent=1).encode("utf-8"))]
            items += sorted(members.items())
            # 在合法成员前塞一份同名恶意内容；旧实现会被后面的合法内容覆盖。
            items.insert(1, ("feedback.jsonl", b'{"p":"file:///etc/passwd"}\n'))
            for name, data in items:
                info = tarfile.TarInfo(name)
                info.size = len(data)
                info.mtime = 0
                info.mode = 0o644
                info.uid = info.gid = 0
                info.uname = info.gname = ""
                handle.addfile(info, io.BytesIO(data))
    duplicate = tmp_path / "duplicate.tar.gz"
    duplicate.write_bytes(buffer.getvalue())
    result = evidence.verify(duplicate)
    assert result["ok"] is False
    assert any("duplicate member" in item for item in result["failures"]), result["failures"]


def test_file_urls_are_sanitized_and_verified(tmp_path, monkeypatch):
    """@brief `file:///etc/passwd` 三斜杠也要净化；verify 对未净化成员拒绝。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 的替换工具。
    """
    assert evidence.ABS_PATH_RE.search("file:///etc/passwd")
    assert evidence.ABS_PATH_RE.search("http://example.com/a/b") is None
    directory = _run(tmp_path, monkeypatch)
    (directory / "feedback.jsonl").write_text(
        json.dumps({"note": "file:///etc/passwd", "url": "http://example.com/a/b"}) + "\n", encoding="utf-8")
    package = tmp_path / "file-url.tar.gz"
    evidence.export(directory, package)
    _manifest, members = _members(package)
    text = members["feedback.jsonl"].decode("utf-8")
    assert "/etc/passwd" not in text and "http://example.com/a/b" in text
    assert evidence.verify(package)["ok"] is True
    tampered = tmp_path / "file-url-tampered.tar.gz"
    _repack_consistent(
        package, tampered,
        lambda manifest, loaded: loaded.__setitem__("feedback.jsonl", b'{"p":"file:///etc/passwd"}\n') or {"feedback.jsonl"},
    )
    result = evidence.verify(tampered)
    assert result["ok"] is False
    assert any("absolute path in member" in item for item in result["failures"])


def test_candidate_extra_directories_are_summed_and_rewrites_break_against(tmp_path, monkeypatch):
    """@brief 候选内真实子目录要递归汇总；两遍之间或 after 导出后追加都必须被发现。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 的替换工具。
    """
    directory = _run(tmp_path, monkeypatch)
    extra = directory / "candidates" / "000_r1.p0" / "extra-dir"
    extra.mkdir()
    nested = extra / "nested.txt"
    nested.write_text("one\n", encoding="utf-8")
    package = tmp_path / "extra-dir.tar.gz"
    manifest = evidence.export(directory, package)
    rows = [row for row in manifest["excluded"] if row["path"] == "candidates/000_r1.p0/extra-dir/"]
    assert len(rows) == 1 and rows[0]["files"] == 1 and rows[0]["bytes"] == 4
    assert len({row["path"] for row in manifest["excluded"]}) == len(manifest["excluded"])
    assert evidence.verify(package, against=directory)["ok"] is True

    nested.write_text("one\nmore content\n", encoding="utf-8")
    result = evidence.verify(package, against=directory)
    assert result["ok"] is False

    # 两遍之间追加：build 之后改排除目录，export 必须在写包前失败。
    real_build = evidence.build

    def append_after_build(root):
        manifest_now = real_build(root)
        nested.write_text("one\nmore content\nagain\n", encoding="utf-8")
        return manifest_now

    monkeypatch.setattr(evidence, "build", append_after_build)
    race = tmp_path / "extra-dir-race.tar.gz"
    with pytest.raises(RuntimeError, match="source inventory changed"):
        evidence.export(directory, race)
    assert not race.exists()


def test_a_wrong_type_top_level_directory_is_recursively_summed(tmp_path, monkeypatch):
    """@brief result.json 是目录时递归汇总；子文件追加让 against 与两遍检查都失败。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 的替换工具。
    """
    directory = _run(tmp_path, monkeypatch)
    result = directory / "result.json"
    result.unlink()
    result.mkdir()
    nested = result / "nested.txt"
    nested.write_text("a\n", encoding="utf-8")
    package = tmp_path / "wrong-type-dir.tar.gz"
    manifest = evidence.export(directory, package)
    rows = [row for row in manifest["excluded"] if row["path"] == "result.json/"]
    assert len(rows) == 1 and rows[0]["files"] == 1 and rows[0]["bytes"] == 2
    assert "result.json" not in [row["path"] for row in manifest["files"]]
    assert evidence.verify(package, against=directory)["ok"] is True

    nested.write_text("a\nb\n", encoding="utf-8")
    assert evidence.verify(package, against=directory)["ok"] is False

    real_build = evidence.build

    def append_after_build(root):
        manifest_now = real_build(root)
        nested.write_text("a\nb\nc\n", encoding="utf-8")
        return manifest_now

    monkeypatch.setattr(evidence, "build", append_after_build)
    race = tmp_path / "wrong-type-dir-race.tar.gz"
    with pytest.raises(RuntimeError, match="source inventory changed"):
        evidence.export(directory, race)
    assert not race.exists()


@pytest.mark.parametrize("fmt", ["tar.gz", "json"])
def test_an_append_to_a_truncated_file_during_the_second_pass_fails_without_a_package(
    tmp_path, monkeypatch, fmt
):
    """@brief 截断分支读完 8MiB 前缀后追加一个空格必须失败；tar/json 都不留包。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 的替换工具。
    @param fmt 导出格式。
    """
    directory = _run(tmp_path, monkeypatch)
    monkeypatch.setattr(evidence, "MAX_FILE_BYTES", 16)
    feedback = directory / "feedback.jsonl"
    feedback.write_text("y" * 100, encoding="utf-8")
    real_pack = evidence._packed_bytes
    real_build = evidence.build
    second_pass = {"on": False}

    def build_then_arm(root):
        manifest_now = real_build(root)
        second_pass["on"] = True
        return manifest_now

    def append_before_sanitize(data, root, run_id, path=None):
        if second_pass["on"] and Path(path) == feedback:
            with feedback.open("ab") as handle:
                handle.write(b" ")
        return real_pack(data, root, run_id, path)

    monkeypatch.setattr(evidence, "build", build_then_arm)
    monkeypatch.setattr(evidence, "_packed_bytes", append_before_sanitize)
    suffix = ".json" if fmt == "json" else ".tar.gz"
    package = tmp_path / f"truncated-race{suffix}"
    with pytest.raises(RuntimeError, match="source size changed between passes"):
        evidence.export(directory, package, fmt=fmt)
    assert not package.exists()
    assert not list(tmp_path.glob(f".truncated-race{suffix}*"))
    assert feedback.stat().st_size == 101


@pytest.mark.parametrize("fmt", ["tar.gz", "json"])
def test_an_append_to_a_sanitized_file_during_the_second_pass_fails_without_a_package(tmp_path, monkeypatch, fmt):
    """@brief 普通净化文件（非截断）也要写前核大小、读后查追加；tar/json 都不留包。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 的替换工具。
    @param fmt 导出格式。
    """
    directory = _run(tmp_path, monkeypatch)
    feedback = directory / "feedback.jsonl"
    feedback.write_text(json.dumps({"seq": 1, "note": "/opt/private"}) + "\n", encoding="utf-8")
    real_pack = evidence._packed_bytes
    real_build = evidence.build
    second_pass = {"on": False}

    def build_then_arm(root):
        manifest_now = real_build(root)
        second_pass["on"] = True
        return manifest_now

    def append_before_sanitize(data, root, run_id, path=None):
        if second_pass["on"] and Path(path) == feedback:
            with feedback.open("ab") as handle:
                handle.write(b" ")
        return real_pack(data, root, run_id, path)

    monkeypatch.setattr(evidence, "build", build_then_arm)
    monkeypatch.setattr(evidence, "_packed_bytes", append_before_sanitize)
    package = tmp_path / f"sanitized-race.{fmt.replace('.', '_')}"
    with pytest.raises(RuntimeError, match="source size changed between passes"):
        evidence.export(directory, package, fmt=fmt)
    assert not package.exists()
    assert not list(tmp_path.glob(f".{package.name}*"))
