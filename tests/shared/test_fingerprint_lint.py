"""Lint: the current prompt fingerprint must only appear in allowed files.

A test that hardcodes the current fingerprint will break the next time it changes,
and nobody will know why because it is not in the "active tests pin fingerprint" list.

This lint checks that the current fingerprint only appears in:

* ``tests/support/live_anchors.json`` (the authoritative source)
* ``tests/support/golden/shared/acceptance-epochs/`` (historical epoch records)
* seven historical epoch/baseline assertion modules that already pin recorded values

Any other ``tests/**/*.py`` file that contains the current fingerprint is a bug:
请读 live_anchors.json.
"""

from __future__ import annotations

from pathlib import Path

from tests.support.live_anchors import load_live_anchors

REPO_ROOT = Path(__file__).resolve().parents[2]

ALLOWED_FILES = {
    "tests/support/live_anchors.json",
    # 历史 epoch/baseline 断言允许含旧指纹（它们钉的是历史值，不是现行值）
    "tests/shared/test_scratch_spec.py",
    "tests/backends/metal/test_candidate_scratch_cpu.py",
    "tests/shared/test_scratch_prompt.py",
    "tests/shared/test_seq_handoff_decision.py",
    "tests/shared/test_bitdelta_unpack_decision.py",
    "tests/shared/test_gemm_silu_scratch_decision.py",
    "tests/shared/test_experience_compact_overlay.py",
}

ALLOWED_DIRS = {
    "tests/support/golden/shared/acceptance-epochs",
}


def _is_allowed(path: Path, root: Path = REPO_ROOT) -> bool:
    """Check if a file is allowed to contain the current fingerprint."""
    rel = path.relative_to(root).as_posix()
    if rel in ALLOWED_FILES:
        return True
    return any(rel == directory or rel.startswith(directory + "/") for directory in ALLOWED_DIRS)


def collect_fingerprint_problems(root: Path, current: str) -> list[str]:
    """Return lint failures for ``tests/**/*.py`` under ``root`` that hardcode ``current``."""
    problems: list[str] = []
    for path in root.glob("tests/**/*.py"):
        if path.name == "test_fingerprint_lint.py":
            continue
        if _is_allowed(path, root):
            continue
        text = path.read_text(encoding="utf-8")
        if current in text:
            rel = path.relative_to(root).as_posix()
            problems.append(
                f"{rel}: hardcoded current fingerprint {current[:8]}…; "
                "read live_anchors.json instead. 请读 live_anchors.json"
            )
    return sorted(problems)


def test_fingerprint_only_in_allowed_files():
    """The current prompt fingerprint must only appear in live_anchors.json and epoch files."""
    missing = [rel for rel in sorted(ALLOWED_FILES) if not (REPO_ROOT / rel).is_file()]
    assert missing == []
    current = load_live_anchors()["prompt_sha1"]
    problems = collect_fingerprint_problems(REPO_ROOT, current)
    assert not problems, "\n".join(problems)


def test_detects_hardcoded_fingerprint(tmp_path: Path):
    """A temporary test that pastes the current fingerprint is reported."""
    current = load_live_anchors()["prompt_sha1"]
    tests = tmp_path / "tests"
    tests.mkdir()
    (tests / "test_bad.py").write_text(
        f'def test_prompt_fingerprint():\n    assert current_prompt_sha1() == "{current}"\n',
        encoding="utf-8",
    )
    problems = collect_fingerprint_problems(tmp_path, current)
    assert len(problems) == 1
    assert problems[0].startswith("tests/test_bad.py:")
    assert "请读 live_anchors.json" in problems[0]
    assert "read live_anchors.json instead" in problems[0]


def test_allows_live_anchors_json():
    """``live_anchors.json`` is the source tests must read, so it may contain the fingerprint."""
    path = REPO_ROOT / "tests/support/live_anchors.json"
    current = load_live_anchors()["prompt_sha1"]
    assert path.is_file()
    assert _is_allowed(path)
    assert current in path.read_text(encoding="utf-8")


def test_allows_epoch_files(tmp_path: Path):
    """Historical acceptance-epoch records may contain the current fingerprint."""
    epoch_dir = REPO_ROOT / "tests/support/golden/shared/acceptance-epochs"
    epoch_files = sorted(epoch_dir.glob("*.json"))
    assert epoch_files
    assert all(_is_allowed(path) for path in epoch_files)

    current = load_live_anchors()["prompt_sha1"]
    fake = tmp_path / "tests/support/golden/shared/acceptance-epochs"
    fake.mkdir(parents=True)
    (fake / "epoch.json").write_text(current, encoding="utf-8")
    (fake / "test_historical.py").write_text(f'PIN = "{current}"\n', encoding="utf-8")
    assert collect_fingerprint_problems(tmp_path, current) == []


def test_ignores_old_fingerprints(tmp_path: Path):
    """A historical fingerprint that is not the live value is not a lint failure."""
    current = load_live_anchors()["prompt_sha1"]
    old = "cfb02704b803e4f54b379ba6e52e9870487d898e78240bc72229dd3924018ec6"
    assert old != current
    tests = tmp_path / "tests"
    tests.mkdir()
    (tests / "test_old.py").write_text(
        f'def test_old():\n    assert recorded == "{old}"\n',
        encoding="utf-8",
    )
    assert collect_fingerprint_problems(tmp_path, current) == []
