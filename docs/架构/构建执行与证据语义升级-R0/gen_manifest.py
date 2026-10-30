#!/usr/bin/env python3
"""Freeze the R0 baseline package (run once, from the repository root).

The package is authoritative for: the exact HEAD, the exact worktree state, the
per-file owner/decision tables, the frozen patches and the canonical-identity
vectors.  It writes ``git/``, ``drafts/``, ``draft-files/`` and ``manifest.json``;
``README.md`` and ``vectors/`` are authored elsewhere and only hashed here.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
PKG_PREFIX = "docs/架构/构建执行与证据语义升级-R0/"

# --- what is this plan's own material ---------------------------------------
BUILD_PATH = "build-path-draft"
SPEC = "spec-docs"
SECURITY = "security-hardening"
INDEX = "observe-index-refresh"
DOCS = "docs-concurrent"
RUNS = "kernel-run-artifacts"
CONFIG = "committee-config"
SESSIONS = "harness-session-artifacts"
REFS = "reference-material"
DATA = "data-layer-observe"
R0 = "r0-baseline"

SNAPSHOT = {
    # tracked build-path draft (partly adopted, partly discarded)
    "tcenv/backends/base.py": (BUILD_PATH, "adopted-r3-input"),
    "tcenv/backends/metal/backend.py": (BUILD_PATH, "adopted-r3-input"),
    "tcenv/backends/metal/source_builder.py": (BUILD_PATH, "adopted-r3-input"),
    "tcenv/backends/metal/probes.py": (BUILD_PATH, "discarded"),
    "tcenv/backends/metal/profile.py": (BUILD_PATH, "discarded"),
    "tcenv/evaluate.py": (BUILD_PATH, "adopted-r3-input"),
    "tcenv/xcworker.py": (BUILD_PATH, "adopted-r3-input"),
    "tcenv/remote/artifact_builder.py": (BUILD_PATH, "adopted-r3-input"),
    "tests/backends/metal/test_source_builder.py": (BUILD_PATH, "adopted-r3-input"),
    "tests/support/npu_sim/fake_backend.py": (BUILD_PATH, "adopted-r3-input"),
    "docs/架构/构建执行与证据语义升级-plan.md": (SPEC, "reference-spec"),
    # unmodified files frozen because plan §4.3 uses them as R1's independent oracle
    "tasks/rmsnorm/task.py": (SPEC, "r1-comparison-oracle"),
    "tasks/rmsnorm/kernel_triton.py": (SPEC, "r1-comparison-oracle"),
    "tasks/rmsnorm/naive.metal": (SPEC, "r1-comparison-oracle"),
    "tasks/rmsnorm/__init__.py": (SPEC, "r1-comparison-oracle"),
    "tasks/_common.py": (SPEC, "r1-comparison-oracle"),
    # untracked
    "tests/shared/test_single_build_path.py": (BUILD_PATH, "discarded"),
    "docs/架构/构建执行与证据语义升级-plan-附录A-OpSpec与specialize.md": (SPEC, "reference-spec"),
    "docs/架构/构建执行与证据语义升级-决策与对比记录.md": (SPEC, "reference-spec"),
    # R0's own drift check for this package
    "tests/shared/test_r0_baseline.py": (R0, "r0-check"),
}

#: tracked changes that belong to other, concurrent workstreams
EXCLUDED_TRACKED = {
    "tcenv/remote/artifact_bundle.py": (SECURITY, "materialize() now requires a trusted anchor; unrelated to the build path"),
    "tcenv/remote/blob_channel.py": (SECURITY, "descriptor-relative blob cleanup/read; unrelated to the build path"),
    "tests/shared/test_remote_blob_channel.py": (SECURITY, "symlinked-chunks cases for the blob hardening"),
    "tests/shared/test_remote_npu_bundle_contract.py": (SECURITY, "materialize(anchor=...) call sites for the bundle hardening"),
    "tests/shared/test_evaluation_outbox.py": (SECURITY, "revoked external HMAC key case; unrelated to the build path"),
    "tests/support/blob_sim/faults.py": (SECURITY, "fault injection for the descriptor-relative blob IO"),
    "tests/support/vendor_sim/vendor_b.py": (SECURITY, "passes the trusted state anchor to materialize()"),
    "tests/support/vendor_sim/worker_c.py": (SECURITY, "passes the trusted state anchor to materialize()"),
    "docs/security.md": (DOCS, "P2 code agent / remote evaluation security documentation"),
    "AGENTS.md": (DOCS, "execution-runtime documentation pointer"),
    "docs/README.md": (DOCS, "documentation index"),
    "docs/架构/backend-phase0-baseline.json": (INDEX, "observe index schema 12 baseline refresh"),
    "tests/backends/metal/test_backend_phase0.py": (INDEX, "asserts INDEX_SCHEMA_VERSION == 12 (HEAD code is already 12)"),
    "kernels/apple-m4-max/flash_attention/fp16/paper/best.metal": (RUNS, "run output of another session"),
    "kernels/apple-m4-max/flash_attention/fp16/paper/history.jsonl": (RUNS, "run output of another session"),
    "kernels/apple-m4-max/flash_attention/fp16/paper/meta.json": (RUNS, "run output of another session"),
    "taic.toml": (CONFIG, "committee model swap (kimi-k3 -> k3.1-pro)"),
    "docs/架构/hmz.md": (DOCS, "unrelated concurrent doc edit"),
    "docs/架构/搜索.md": (DOCS, "unrelated concurrent doc edit"),
    "docs/架构/数据层.md": (DOCS, "unrelated concurrent doc edit"),
    "docs/架构/评测.md": (DOCS, "unrelated concurrent doc edit"),
    "docs/架构/远程评测模拟验收-plan.md": (DOCS, "unrelated concurrent doc edit"),
    "docs/架构说明.md": (DOCS, "unrelated concurrent doc edit"),
    "docs/代理反馈.md": (DOCS, "unrelated concurrent doc edit"),
    "docs/代理理论.md": (DOCS, "unrelated concurrent doc edit"),
    "docs/算子工程师边界.md": (DOCS, "unrelated concurrent doc edit"),
    "docs/维护决策.md": (DOCS, "unrelated concurrent doc edit"),
    "docs/可信harness/human-in-loop_rsi_feedback_20f84da7.plan.md": (DOCS, "trusted-harness plan edit"),
    "docs/可信harness/实现交接.md": (DOCS, "trusted-harness doc edit"),
    "docs/可信harness/架构与理论.md": (DOCS, "trusted-harness doc edit"),
    "docs/可信harness/试点方案.md": (DOCS, "trusted-harness doc edit"),
    "docs/跑掉四五千万人民币等值 Token 之后，两位英伟达工程师重新理解了 Agent：代码不是护城河，时间才是.pdf": (DOCS, "reference PDF removed by another workstream"),
}

UNTRACKED_RULES = [
    ("docs/paper/", REFS, "reference material (papers/PDFs), not part of this plan"),
    ("docs/可信harness/", REFS, "trusted-harness material of another workstream"),
    ("docs/tsm/", REFS, "reference material of another workstream"),
    ("docs/一个公众号的论文解读/", REFS, "reference material of another workstream"),
    (".pi/subagent-schedules/", SESSIONS, "agent session schedule state"),
    (".pi/subagents/", SESSIONS, "agent session state"),
    ("kernels/", RUNS, "kernel run output (best.metal/history/meta) of another session"),
    ("docs/架构/tasks现状：用途、价值与不足.md", DOCS, "unrelated concurrent doc"),
    ("docs/架构/工业使用：AI算子生成、Kernel CI-CD与受控发布.md", DOCS, "unrelated concurrent doc"),
    ("tests/shared/test_read_probe.py", DATA, "data-layer read probe test of another workstream"),
    ("tests/shared/test_session_kpi.py", DATA, "model-cost KPI test of another workstream"),
]


def git(*args: str, check: bool = True) -> str:
    result = subprocess.run(
        ["git", "-C", str(REPO), "-c", "core.quotePath=false", *args],
        check=check, capture_output=True, text=True,
    )
    return result.stdout


def porcelain_records() -> list[tuple[str, str]]:
    """``git status --porcelain -z`` as (status, path) pairs, never C-quoted."""

    result = subprocess.run(
        ["git", "-C", str(REPO), "-c", "core.quotePath=false", "status", "--porcelain", "-z", "-uall"],
        check=True, capture_output=True,
    )
    fields = result.stdout.split(b"\0")
    records: list[tuple[str, str]] = []
    index = 0
    while index < len(fields):
        entry = fields[index]
        index += 1
        if not entry:
            continue
        status = entry[:2].decode("utf-8")
        path = entry[3:].decode("utf-8")
        if status[:1] in ("R", "C"):
            index += 1  # the original path follows the destination
        records.append((status, path))
    return records


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def worktree_sha(path: str) -> tuple[str, int] | tuple[None, None]:
    target = REPO / path
    if not target.is_file():
        return None, None
    data = target.read_bytes()
    return sha256_bytes(data), len(data)


def head_sha(path: str) -> str | None:
    result = subprocess.run(
        ["git", "-C", str(REPO), "show", f"HEAD:{path}"],
        check=False, capture_output=True,
    )
    if result.returncode != 0:
        return None
    return sha256_bytes(result.stdout)


def main() -> int:
    frozen_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    head = git("rev-parse", "HEAD").strip()
    head_short = git("rev-parse", "--short", "HEAD").strip()

    (HERE / "git").mkdir(exist_ok=True)
    (HERE / "git" / "status-short.txt").write_text(git("status", "--short"), encoding="utf-8")
    (HERE / "git" / "status-short-uall.txt").write_text(git("status", "--short", "-uall"), encoding="utf-8")
    (HERE / "git" / "diff-staged.patch").write_text(git("diff", "--cached"), encoding="utf-8")
    (HERE / "git" / "diff-unstaged.patch").write_text(git("diff"), encoding="utf-8")

    records = porcelain_records()
    package_paths = [path for _status, path in records if path.startswith(PKG_PREFIX)]
    porcelain = [(status, path) for status, path in records if not path.startswith(PKG_PREFIX)]
    untracked = [path for status, path in porcelain if status == "??"]
    (HERE / "git" / "untracked-files.txt").write_text("\n".join(untracked) + "\n", encoding="utf-8")
    (HERE / "git" / "package-files.txt").write_text(
        "\n".join(sorted(package_paths)) + "\n", encoding="utf-8",
    )

    # --- archived snapshot + patches ---------------------------------------
    (HERE / "draft-files").mkdir(exist_ok=True)
    snapshot_entries = []
    changed_paths = {path for _status, path in porcelain}
    tracked_snapshot = [
        p for p in sorted(SNAPSHOT) if p not in untracked and p in changed_paths
    ]
    untracked_snapshot = sorted(p for p in SNAPSHOT if p in untracked)
    for path in sorted(SNAPSHOT):
        workstream, decision = SNAPSHOT[path]
        digest, size = worktree_sha(path)
        if digest is None:
            raise SystemExit(f"snapshot file is missing: {path}")
        archive = HERE / "draft-files" / path
        archive.parent.mkdir(parents=True, exist_ok=True)
        archive.write_bytes((REPO / path).read_bytes())
        if path in untracked:
            status = "untracked"
        elif path in changed_paths:
            status = "modified"
        else:
            status = "unmodified-at-head"
        snapshot_entries.append({
            "path": path,
            "archive": f"draft-files/{path}",
            "status": status,
            "workstream": workstream,
            "decision": decision,
            "size": size,
            "sha256": digest,
            "head_sha256": None if path in untracked else head_sha(path),
            "head_blob_equals_worktree": status != "untracked" and head_sha(path) == digest,
        })

    (HERE / "drafts").mkdir(exist_ok=True)
    (HERE / "drafts" / "tracked-worktree.patch").write_text(
        git("diff", "--", *tracked_snapshot), encoding="utf-8",
    )
    untracked_patch = []
    for path in untracked_snapshot:
        result = subprocess.run(
            ["git", "-C", str(REPO), "-c", "core.quotePath=false",
             "diff", "--no-index", "--binary", "/dev/null", path],
            capture_output=True, text=True,
        )
        if result.returncode not in (0, 1):
            raise SystemExit(f"git diff --no-index failed for {path}: {result.stderr}")
        untracked_patch.append(result.stdout)
    (HERE / "drafts" / "untracked-draft.patch").write_text("".join(untracked_patch), encoding="utf-8")

    # --- excluded inventory -------------------------------------------------
    excluded = []
    for status, path in porcelain:
        if path in SNAPSHOT:
            continue
        if path in untracked:
            workstream, reason = next(
                ((ws, why) for prefix, ws, why in UNTRACKED_RULES if path.startswith(prefix)),
                ("unattributed", "untracked file not owned by this plan"),
            )
            digest, size = worktree_sha(path)
            excluded.append({
                "path": path, "status": "untracked", "workstream": workstream,
                "reason": reason, "size": size, "sha256": digest,
            })
            continue
        workstream, reason = EXCLUDED_TRACKED.get(
            path, ("unattributed", "tracked change not owned by this plan"),
        )
        digest, size = worktree_sha(path)
        excluded.append({
            "path": path,
            "status": "deleted" if status.strip() == "D" else "modified",
            "workstream": workstream,
            "reason": reason,
            "size": size,
            "sha256": digest,
            "head_sha256": head_sha(path),
        })

    decisions = json.loads((HERE / "decisions.json").read_text(encoding="utf-8"))
    package_files = []
    for path in sorted(HERE.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(HERE).as_posix()
        if relative in ("manifest.json", "manifest.sha256") or relative.startswith("draft-files/"):
            continue
        package_files.append({
            "path": relative,
            "size": path.stat().st_size,
            "sha256": sha256_bytes(path.read_bytes()),
        })
    manifest = {
        "baseline_id": "r0-build-execution-evidence",
        "frozen_at_utc": frozen_at,
        "head": {
            "commit": head,
            "short": head_short,
            "subject": git("log", "-1", "--pretty=%s").strip(),
        },
        "scope": {
            "stage": "R0 (freeze baseline, register delete/keep, decide on the worktree build-path draft)",
            "not_in_scope": [
                "R1-R5 implementation", "any git commit or git stage", "GPU/NPU/real-device runs",
                "taic run / taic resume", "paid model calls or search",
            ],
            "authoritative_docs": [
                "docs/架构/构建执行与证据语义升级-plan.md",
                "docs/架构/构建执行与证据语义升级-plan-附录A-OpSpec与specialize.md",
                "docs/架构/构建执行与证据语义升级-决策与对比记录.md",
            ],
        },
        "capture_commands": [
            "git rev-parse HEAD",
            "git -c core.quotePath=false status --short",
            "git -c core.quotePath=false status --short -uall",
            "git -c core.quotePath=false status --porcelain -z -uall",
            "git -c core.quotePath=false diff --cached",
            "git -c core.quotePath=false diff",
            "git diff -- <snapshot tracked paths>",
            "git diff --no-index --binary /dev/null <snapshot untracked path>",
        ],
        "worktree": {
            "staged_changes": 0,
            "tracked_changed": len(porcelain) - len(untracked),
            "untracked_files": len(untracked),
            "porcelain_lines": len(porcelain) + len(package_paths),
            "r0_package": {
                "prefix": PKG_PREFIX,
                "note": "R0's own output lives under this prefix; it is enumerated by files.package and files.snapshot and is excluded from the concurrent counts above.",
            },
        },
        "files": {"snapshot": snapshot_entries, "excluded": excluded, "package": package_files},
        "decisions": decisions["decisions"],
        "defects": decisions["defects"],
        "post_freeze_r0_outputs": [
            {
                "path": "docs/架构/构建执行与证据语义升级-R0/",
                "note": "R0's own package; enumerated by files.snapshot + files.package + manifest.json/manifest.sha256.",
            },
            {
                "path": "tests/shared/test_r0_baseline.py",
                "note": "backend-free drift check for this package; it is archived and patched like the other snapshot files (workstream r0-baseline).",
            },
        ],
        "verification": json.loads((HERE / "verification.json").read_text(encoding="utf-8")),
        "vectors": json.loads((HERE / "vectors" / "digests.json").read_text(encoding="utf-8")),
    }
    manifest_path = HERE / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=False) + "\n", encoding="utf-8",
    )
    (HERE / "manifest.sha256").write_text(
        f"{sha256_bytes(manifest_path.read_bytes())}  manifest.json\n", encoding="utf-8",
    )
    print(f"frozen {len(snapshot_entries)} archived files, {len(excluded)} excluded entries")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
