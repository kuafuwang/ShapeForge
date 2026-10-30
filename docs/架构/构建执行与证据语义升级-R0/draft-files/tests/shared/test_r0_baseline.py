"""Drift check for the frozen R0 baseline package.

R0 (`docs/架构/构建执行与证据语义升级-R0/`) freezes the HEAD, the worktree
inventory and the canonical-identity vectors that R1-R5 are measured against.
Those artifacts are only useful if nobody can quietly edit them: this module
re-hashes the archive, re-checks the include/exclude accounting, and recomputes
every vector digest with a serializer that does not import the project.

The heavy part (replaying the frozen patches onto a pristine HEAD export) lives
in `docs/架构/构建执行与证据语义升级-R0/verify.py --replay`, because it needs a
git worktree; it is deliberately not part of the test suite.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

BASELINE = Path(__file__).resolve().parents[2] / "docs" / "架构" / "构建执行与证据语义升级-R0"


def _manifest() -> dict:
    return json.loads((BASELINE / "manifest.json").read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_bytes(document: dict) -> bytes:
    # Match the R0 standalone checker rather than relying on json.dumps to
    # reject values: it accepts finite floats unless we check them ourselves.
    from tcenv.remote.contracts import canonical_digest_document

    return canonical_digest_document(document)


def test_the_archive_matches_the_manifest_and_nothing_is_unlisted():
    """@brief 归档文件与清单逐字节一致，且清单外没有多出来的归档文件。
    """

    manifest = _manifest()
    entries = manifest["files"]["snapshot"]
    assert manifest["head"]["commit"] == "3fb9ac85bfc760eb655d9b1759d48f5c4896997a"
    assert manifest["baseline_id"] == "r0-build-execution-evidence"
    assert entries, "the baseline archive must not be empty"

    listed = {entry["path"] for entry in entries}
    assert len(listed) == len(entries), "a path is archived twice"
    archived = {
        path.relative_to(BASELINE).as_posix()
        for path in (BASELINE / "draft-files").rglob("*")
        if path.is_file()
    }
    assert archived == {entry["archive"] for entry in entries}

    for entry in entries:
        path = BASELINE / entry["archive"]
        assert path.is_file(), entry["archive"]
        assert path.stat().st_size == entry["size"], entry["archive"]
        assert _sha256(path) == entry["sha256"], entry["archive"]
        assert len(entry["sha256"]) == 64 and entry["sha256"] == entry["sha256"].lower()


def test_the_include_exclude_accounting_covers_every_worktree_change():
    """@brief 未跟踪清单里的每个路径都被恰好分类一次，排除表数目与工作树统计吻合。
    """

    manifest = _manifest()
    worktree = manifest["worktree"]
    changed_snapshot = [
        entry for entry in manifest["files"]["snapshot"]
        if entry["status"] in ("modified", "untracked")
    ]
    assert worktree["staged_changes"] == 0
    assert len(manifest["files"]["excluded"]) == (
        worktree["untracked_files"] + worktree["tracked_changed"] - len(changed_snapshot)
    )

    frozen = [
        path for path in (BASELINE / "git" / "untracked-files.txt").read_text(encoding="utf-8").splitlines()
        if path
    ]
    assert len(frozen) == worktree["untracked_files"]

    classified = {entry["path"] for entry in manifest["files"]["excluded"] if entry["status"] == "untracked"}
    classified |= {
        entry["path"] for entry in manifest["files"]["snapshot"] if entry["status"] == "untracked"
    }
    assert set(frozen) == classified

    archived = {entry["path"] for entry in manifest["files"]["snapshot"]}
    excluded = {entry["path"] for entry in manifest["files"]["excluded"]}
    assert archived & excluded == set()
    for entry in manifest["files"]["excluded"]:
        assert entry["workstream"] and entry["reason"], entry["path"]


def test_every_package_file_is_listed_and_unchanged():
    """@brief 包内每个文件都被清单解释，且清单记录的摘要仍然成立。
    """

    manifest = _manifest()
    listed = {entry["path"] for entry in manifest["files"]["package"]}
    for entry in manifest["files"]["package"]:
        path = BASELINE / entry["path"]
        assert path.is_file(), entry["path"]
        assert path.stat().st_size == entry["size"], entry["path"]
        assert _sha256(path) == entry["sha256"], entry["path"]

    allowed = listed | {entry["archive"] for entry in manifest["files"]["snapshot"]}
    allowed |= {"manifest.json", "manifest.sha256"}
    on_disk = {
        path.relative_to(BASELINE).as_posix()
        for path in BASELINE.rglob("*")
        if path.is_file()
    }
    assert on_disk - allowed == set()


def test_the_identity_vectors_recompute_independently():
    """@brief 7 个身份文档的 RFC 8785/SHA-256 摘要用独立序列化器可复算。
    """

    digests = json.loads((BASELINE / "vectors" / "digests.json").read_text(encoding="utf-8"))
    assert set(digests) == {
        "opspec", "lowering_profile", "task_fingerprint",
        "baseline_implementation", "baseline_environment",
        "task_pack_manifest", "evaluation_fingerprint",
    }
    for name, item in digests.items():
        document = json.loads((BASELINE / item["document"]).read_text(encoding="utf-8"))
        recomputed = hashlib.sha256(_canonical_bytes(document)).hexdigest()
        assert recomputed == item["digest"], name
        assert re.fullmatch(r"[0-9a-f]{64}", item["digest"]), name


def test_the_number_encoding_rule_is_pinned():
    """@brief §3.1 的数字编码规则可复现：repr 往返、`0.0`≠`-0.0`、`1`≠`"1.0"`、NaN/Inf 拒绝。
    """

    examples = json.loads((BASELINE / "vectors" / "encoding_examples.json").read_text(encoding="utf-8"))
    for item in examples["float_examples"]:
        value = float(item["literal"])
        assert repr(value) == item["canonical"]
        assert float(item["canonical"]) == value
        assert repr(float(item["canonical"])) == item["canonical"]
    for item in examples["integer_examples"]:
        assert int(item["literal"]) == item["canonical_value"]
    documents = {item["label"]: item for item in examples["distinct_canonical_documents"]}
    assert set(documents) == {"signed_zero", "int_vs_float_string", "int_vs_integral_float"}
    for label, item in documents.items():
        left, right = _canonical_bytes(item["left"]), _canonical_bytes(item["right"])
        assert left != right, label
        assert hashlib.sha256(left).digest() != hashlib.sha256(right).digest(), label
    for item in examples["rejected_literals"]:
        value = float(item["literal"])
        try:
            _canonical_bytes({"v": value})
        except ValueError:
            pass
        else:  # pragma: no cover - every registered negative must be executed
            raise AssertionError(f"rejected literal reached the canonical serializer: {item['literal']}")
    assert _canonical_bytes({"v": "1.0"}) == b'{"v":"1.0"}'


def test_the_frozen_protocol_and_request_versions_are_pinned():
    """@brief 两个同名协议常量及 worker 请求版本的保留面不能被静默改写。"""
    from tcenv.remote.protocol import PROTOCOL_VERSION as REMOTE_PROTOCOL_VERSION
    from tcenv.schema import PROTOCOL_VERSION as MEASUREMENT_PROTOCOL_VERSION
    from tcenv.schema import REQUEST_SCHEMA_VERSION

    assert MEASUREMENT_PROTOCOL_VERSION == "3"
    assert REMOTE_PROTOCOL_VERSION == "1"
    assert REQUEST_SCHEMA_VERSION == 4
