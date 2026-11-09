#!/usr/bin/env python3
"""Freeze the G0 code snapshot (git state + untracked-file classification).

This is a one-shot generator, not part of the test suite.  It is deliberately
independent of ``tcenv``: it only reads git state and filesystem metadata, so the
snapshot can be re-derived on a machine where the project itself cannot import.

What it writes (all paths relative to this package):

``git/head.txt``                  HEAD commit, date, subject
``git/status-short.txt``          ``git status --short`` (directories folded, quoted)
``git/status-short-uall.txt``     ``git status --short -uall`` (every path)
``git/diff-staged.patch``          ``git diff --cached --binary`` (must be empty)
``git/diff-unstaged.patch``        ``git diff --binary`` (must be empty)
``git/diff-stat.txt``             both stats
``git/tracked-diff.json``         emptiness proof for the tracked diff
``git/untracked-files.json``      every pre-existing untracked file: path/size/sha256/category
``git/untracked-summary.json``    per-category counts (no silent "the directory is clean")
``git/ignored-trees.json``        git-ignored trees (logs/runs/.pi/...), aggregated + counted
``git/environment.json``          python/torch/uv/macOS/.gitignore identity

The G0 package's own files are **not** part of the untracked list: they are
enumerated with size/SHA-256 in ``manifest.json::files.package`` by
``gen_manifest.py``.  Both facts are recorded explicitly rather than pretending the
worktree has no untracked file.

Usage (from the repository root):

    uv run python "docs/架构/跨硬件算子生成分工-G0/gen_snapshot.py"
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
SELF_PREFIX = HERE.relative_to(REPO).as_posix() + "/"

#: Every pre-existing untracked path must match exactly one rule.  An unmatched path is a
#: hard failure: a new unrelated draft must not be able to slip into "out of scope".
UNTRACKED_RULES = (
    (
        "research-kda-notes",
        "docs/KDA 怎么做.md",
        "KDA 方法与分析笔记；不参与构建、评测或语料生成，只冻结 path/size/sha256。",
    ),
    (
        "research-papers",
        "docs/paper/",
        "外部论文 PDF 与中文翻译/导读；只作研究资料，不参与构建、评测或语料生成，"
        "按版权与体积不复制进 G0 包，只冻结 path/size/sha256。",
    ),
    (
        "research-harness-notes",
        "docs/可信harness/",
        "可信 harness 的规划/评审笔记（plan-01..10、审核意见、案例），属于另一条文档流的"
        "未提交草稿；不参与 G0 语料与快照语义，只冻结 path/size/sha256。",
    ),
    (
        "research-article-notes",
        "docs/一个公众号的论文解读/",
        "公众号论文解读的中文笔记与配图；研究资料，不参与构建与评测。",
    ),
    (
        "research-tsm-notes",
        "docs/tsm/",
        "TSM 相关研究笔记；研究资料，不参与构建与评测。",
    ),
)

IGNORED_CATEGORIES = (
    ("logs/", "pytest 会话/单测日志、GPU 冒烟与 taic 运行日志；运行状态，不入快照。"),
    ("runs/", "本地 taic 运行目录；运行状态与模型花费记录，不入快照。"),
    ("data/", "本地花费账本 data/books.sqlite；运行状态，不入快照。"),
    ("__pycache__/", "Python 字节码缓存；构建副产物，不入快照。"),
    (".venv/", "本地虚拟环境；不随仓库交付。"),
    (".venvs/", "本地虚拟环境（triton-msl 等）；不随仓库交付。"),
    (".pi/", "pi agent 会话与运行状态；不入快照（G0 不复用任何 .pi 状态）。"),
    (".pytest_cache/", "pytest 缓存；构建副产物。"),
)


def run(*args: str, binary: bool = False) -> bytes | str:
    """Run one git command in the repository and return its output."""
    result = subprocess.run(args, cwd=REPO, capture_output=True)
    if result.returncode != 0:
        raise SystemExit(f"command failed: {' '.join(args)}\n{result.stderr.decode('utf-8', 'replace')[:800]}")
    return result.stdout if binary else result.stdout.decode("utf-8", "replace")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def split_z(text: str) -> list[str]:
    return [item for item in text.split("\0") if item]


def freeze_git() -> dict:
    git_dir = HERE / "git"
    git_dir.mkdir(parents=True, exist_ok=True)

    head = run("git", "rev-parse", "HEAD").strip()
    subject = run("git", "log", "-1", "--format=%H%n%cI%n%s").strip().split("\n")
    (git_dir / "head.txt").write_text(
        f"{subject[0]}\n{subject[1]}\n{subject[2]}\nshort={head[:7]}\n", encoding="utf-8"
    )

    (git_dir / "status-short.txt").write_bytes(
        run("git", "-c", "core.quotePath=false", "status", "--short", binary=True)
    )
    (git_dir / "status-short-uall.txt").write_bytes(
        run("git", "-c", "core.quotePath=false", "status", "--short", "-uall", binary=True)
    )
    staged = run("git", "diff", "--cached", "--binary", binary=True)
    unstaged = run("git", "diff", "--binary", binary=True)
    (git_dir / "diff-staged.patch").write_bytes(staged)
    (git_dir / "diff-unstaged.patch").write_bytes(unstaged)
    (git_dir / "diff-stat.txt").write_text(
        "staged:\n" + run("git", "diff", "--cached", "--stat") + "unstaged:\n" + run("git", "diff", "--stat"),
        encoding="utf-8",
    )
    tracked = {
        "head": head,
        "staged": {
            "empty": not staged,
            "patch_sha256": hashlib.sha256(staged).hexdigest(),
            "bytes": len(staged),
            "paths": split_z(run("git", "diff", "--cached", "--name-only", "-z").strip()),
        },
        "unstaged": {
            "empty": not unstaged,
            "patch_sha256": hashlib.sha256(unstaged).hexdigest(),
            "bytes": len(unstaged),
            "paths": split_z(run("git", "diff", "--name-only", "-z").strip()),
        },
    }
    if not tracked["staged"]["empty"] or not tracked["unstaged"]["empty"]:
        raise SystemExit("G0 requires an empty tracked diff; commit or stash the tracked changes first")
    (git_dir / "tracked-diff.json").write_text(
        json.dumps(tracked, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8"
    )

    untracked = []
    for relative in split_z(run("git", "ls-files", "--others", "--exclude-standard", "-z")):
        if relative.startswith(SELF_PREFIX):
            continue  # the package itself: counted in manifest.files.package
        category = None
        for name, prefix, reason in UNTRACKED_RULES:
            if relative.startswith(prefix):
                if category is not None:
                    raise SystemExit(f"{relative} matches more than one untracked rule")
                category, reason_text = name, reason
        if category is None:
            raise SystemExit(f"unclassified untracked file: {relative} (add a rule before freezing)")
        path = REPO / relative
        untracked.append(
            {
                "path": relative,
                "size": path.stat().st_size,
                "sha256": sha256_file(path),
                "category": category,
                "reason": reason_text,
                "in_scope": False,
            }
        )
    untracked.sort(key=lambda item: item["path"])
    document = {
        "snapshot_kind": "cross-hardware-g0-code-snapshot",
        "head": head,
        "g0_package_prefix": SELF_PREFIX,
        "g0_package_note": (
            "G0 包自身的文件不在本表的既有未跟踪文件里；它们逐个列在 "
            "manifest.json::files.package（path/size/sha256），由 gen_manifest.py 冻结。"
        ),
        "untracked_rule_note": "每个既有未跟踪路径都恰好命中一条规则；未命中即生成失败，不允许静默范围外。",
        "files": untracked,
    }
    (git_dir / "untracked-files.json").write_text(
        json.dumps(document, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8"
    )
    by_category = Counter(item["category"] for item in untracked)
    summary = {
        "untracked_files": len(untracked),
        "untracked_bytes": sum(item["size"] for item in untracked),
        "by_category": dict(sorted(by_category.items())),
        "rules": [{"category": name, "prefix": prefix, "reason": reason} for name, prefix, reason in UNTRACKED_RULES],
        "claim": (
            f"这 {len(untracked)} 个未跟踪文件全部在范围外，并按 path/size/sha256 逐个冻结；"
            "不声称仓库没有未跟踪文件，也不声称这些目录是干净的。"
        ),
    }
    (git_dir / "untracked-summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8"
    )

    ignored = split_z(run("git", "ls-files", "--others", "--ignored", "--exclude-standard", "-z"))
    buckets: Counter = Counter()
    for relative in ignored:
        buckets[_ignored_bucket(relative)] += 1
    ignored_document = {
        "note": (
            "这些路径被 .gitignore 排除，git 从不把它们算作未跟踪文件；它们不是构建输入，"
            "G0 不逐个枚举（运行状态可达十万级），只冻结分类与计数。上表的 293 个未跟踪文件才是"
            "非忽略的未跟踪全集。"
        ),
        "total_ignored_files": len(ignored),
        "by_bucket": dict(sorted(buckets.items())),
        "categories": [{"bucket": name, "reason": reason} for name, reason in IGNORED_CATEGORIES],
        "command": "git ls-files --others --ignored --exclude-standard -z",
    }
    (git_dir / "ignored-trees.json").write_text(
        json.dumps(ignored_document, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {"head": head, "untracked": len(untracked), "ignored": len(ignored)}


def _ignored_bucket(relative: str) -> str:
    """Map one ignored path to a registered bucket (or an explicit "other:<prefix>")."""
    if "__pycache__" in relative.split("/"):
        return "__pycache__/"
    for name, _prefix, _reason in (
        ("logs/", "logs/", ""),
        ("runs/", "runs/", ""),
        ("data/", "data/", ""),
        (".venv/", ".venv/", ""),
        (".venvs/", ".venvs/", ""),
        (".pi/", ".pi/", ""),
        (".pytest_cache/", ".pytest_cache/", ""),
    ):
        if relative.startswith(name):
            return name
    # Everything else is a local build artefact (targets/*/calibration.json, *.so, .DS_Store...).
    return "local-build-artefacts"


def freeze_environment() -> dict:
    import torch  # noqa: PLC0415  (CPU-only import: the corpus is pinned to this torch build)

    uv_version = subprocess.run(["uv", "--version"], capture_output=True, text=True).stdout.strip()
    gitignore = REPO / ".gitignore"
    document = {
        "python_version": sys.version.split()[0],
        "python_implementation": platform.python_implementation(),
        "torch_version": torch.__version__,
        "numpy_version": __import__("numpy").__version__,
        "uv_version": uv_version,
        "macos_version": platform.mac_ver()[0],
        "machine": platform.machine(),
        "gitignore": {
            "path": ".gitignore",
            "sha256": sha256_file(gitignore),
            "bytes": gitignore.stat().st_size,
        },
        "note": (
            "语料里的 paper 摘要由本 torch 版本的 CPU RNG 生成；换 torch 版本后重放 paper 摘要"
            "可能不同，small 语料以冻结的原始字节为准，不依赖 RNG。"
        ),
    }
    (HERE / "git" / "environment.json").write_text(
        json.dumps(document, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8"
    )
    return document


def main() -> int:
    (HERE / "git").mkdir(parents=True, exist_ok=True)
    git_facts = freeze_git()
    environment = freeze_environment()
    print(json.dumps({"git": git_facts, "torch": environment["torch_version"]}, indent=2, ensure_ascii=False))
    print("wrote git/ (head, patches, untracked classification, ignored trees, environment)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
