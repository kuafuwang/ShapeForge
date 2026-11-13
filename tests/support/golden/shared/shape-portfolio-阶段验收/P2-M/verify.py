#!/usr/bin/env python3
"""P2-M acceptance: ScratchSpec is correctly exposed in the hashed prompt surface."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[6]
P2M = Path(__file__).resolve().parent
MANIFEST = P2M / "manifest.sha256"
EPOCH = P2M / "prompt-epoch.json"
CPU_TESTS = (
    "tests/shared/test_scratch_prompt.py",
    "tests/shared/test_search_v2_bundle_prompt.py",
    "tests/shared/test_scratch_spec.py",
    "tests/shared/test_track_mt1a.py",
    "tests/backends/metal/test_candidate_scratch_cpu.py",
    "tests/backends/metal/test_backend_phase0.py",
    "tests/backends/metal/test_hw_feedback.py",
)
DOC_TESTS = ("tests/shared/test_structure.py", "tests/shared/test_docs_integrity.py")
RECOVERY_TESTS = (
    "tests/backends/metal/test_tilelang_mt2_contract.py::test_mt2_multi_function_module_has_one_preamble_and_two_real_kernels",
    "tests/backends/metal/test_tilelang_mt2_contract.py::test_mt2_repeated_source_only_artifacts_are_attempt_path_independent",
    "tests/backends/metal/test_track_mt1b_build_seam.py::test_a_real_attempt_is_composed_and_launched_through_the_hmz_profile",
    "tests/backends/metal/test_track_mt1b_seams.py::test_the_source_only_builder_fails_closed_at_every_stage",
    "tests/backends/metal/test_track_mt1b_build_seam.py::test_the_staged_frontend_is_observed_byte_exactly_or_reported_as_a_capability_gap",
)
DELIVERABLES = (
    "taic/prompts.py",
    "backends/metal/skills/msl_reference.md",
    "taic/flows/kernel_once/skills/taic-tools/SKILL.md",
    "docs/agent-skills.md",
    "docs/写核代理读什么.md",
    "docs/架构/scratch-prompt-epoch.md",
    "tests/shared/test_scratch_spec.py",
    "tests/backends/metal/test_candidate_scratch_cpu.py",
    "tests/shared/test_track_mt1a.py",
    "tests/backends/metal/test_backend_phase0.py",
    "tests/shared/test_search_v2_bundle_prompt.py",
    "tests/shared/test_scratch_prompt.py",
    "tests/backends/metal/test_hw_feedback.py",
    "tests/support/golden/metal/hw_off.json",
    "tests/support/golden/shared/shape-portfolio-阶段验收/P2-M/hw_off-previous.json",
    "tests/support/golden/shared/shape-portfolio-阶段验收/P2-M/M.md",
    "tests/support/golden/shared/shape-portfolio-阶段验收/P2-M/prompt-epoch.json",
    "tests/support/golden/shared/shape-portfolio-阶段验收/P2-M/pack-closure.json",
    "tests/support/golden/shared/shape-portfolio-阶段验收/P2-M/rendered-prompts.json",
    "tests/support/golden/shared/shape-portfolio-阶段验收/P2-M/frontend-recovery-note.json",
    "tests/support/golden/shared/shape-portfolio-阶段验收/P2-M/verify.py",
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_manifest() -> None:
    MANIFEST.write_text("\n".join(f"{sha256(ROOT / name)}  {name}" for name in sorted(DELIVERABLES)) + "\n", encoding="utf-8")
    print(f"OK: wrote P2-M manifest ({len(DELIVERABLES)} files)")


def verify_manifest() -> None:
    entries = {}
    for line in MANIFEST.read_text(encoding="utf-8").splitlines():
        digest, name = line.split("  ", 1)
        entries[name] = digest
    require(set(entries) == set(DELIVERABLES), "P2-M manifest file set drifted")
    for name, digest in entries.items():
        require(sha256(ROOT / name) == digest, f"P2-M manifest mismatch: {name}")
    print(f"OK: P2-M manifest ({len(entries)} files)")


def verify_baseline() -> None:
    from taic.trust import current_prompt_sha1
    from tcenv.taskcontract.pack import code_import_closure

    document = json.loads(EPOCH.read_text(encoding="utf-8"))
    require(document["epoch"] == "p2-m-scratch-prompt-1", "wrong P2-M epoch")
    require(document["parent_head"] == "2383bae8a3f56c5dec317e819b46c38b755aa737", "P2-M parent HEAD drifted")
    require(document["previous_prompt_sha1"] == "27e2bd2c2dfe3c009b065e1e83c9e2051bed9381fce79aa36b92618c2ac0fe3c", "P2-M previous prompt drifted")
    require(document["current_prompt_sha1"] == current_prompt_sha1(), "P2-M prompt SHA is not the live fingerprint")
    require(document["current_prompt_sha1"] != document["previous_prompt_sha1"], "P2-M did not advance the prompt epoch")
    require(document["publication_allowed"] is False, "P2-M must not publish")
    closure = json.loads((P2M / "pack-closure.json").read_text(encoding="utf-8"))
    files = code_import_closure(ROOT)
    digest = hashlib.sha256(b"".join((ROOT / name).read_bytes() for name in files)).hexdigest()
    require(list(closure["files"]) == list(files), "P2-M task-pack closure drifted")
    require(closure["sha256"] == digest == document["pack_closure_sha256"], "P2-M task-pack digest drifted")
    require(len(files) == document["pack_closure_files"] == 79, "P2-M pack file count drifted")
    print(f"OK: P2-M baseline and prompt epoch (79 pack files, prompt={document['current_prompt_sha1'][:12]}…)")


def verify_history() -> None:
    for phase, count in (("P2-I", 71), ("P2-J", 22), ("P2-K", 13), ("P2-L", 12)):
        path = ROOT / f"tests/support/golden/shared/shape-portfolio-阶段验收/{phase}/manifest.sha256"
        entries = [line.split("  ", 1) for line in path.read_text(encoding="utf-8").splitlines()]
        require(len(entries) == count, f"{phase} historical manifest count drifted")
        for digest, name in entries:
            require(sha256(ROOT / name) == digest, f"{phase} historical file changed: {name}")
        print(f"OK: {phase} historical manifest remains byte-identical ({count} files)")


def verify_rendered_prompts() -> None:
    from taic.prompts import system_prompt_v2
    from tests.shared.test_track_mt1a import _brief

    recorded = json.loads((P2M / "rendered-prompts.json").read_text(encoding="utf-8"))
    for track_id in ("msl-single", "msl-bundle"):
        binding, _brief_value = _brief(track_id)
        prompt = system_prompt_v2(binding, use_tools=True, target_id="apple-m4-max")
        row = recorded["tracks"][track_id]
        require(hashlib.sha256(prompt.encode()).hexdigest() == row["system_prompt_sha256"], f"{track_id}: rendered prompt digest drifted")
        require(len(prompt.encode()) == row["bytes"], f"{track_id}: rendered prompt length drifted")
        require("Candidate-owned ScratchSpec" in prompt, f"{track_id}: ScratchSpec is missing")
    print("OK: P2-M single/bundle rendered prompt identities and ScratchSpec content")


def pytest_nodes(arguments: list[str], *, where: str, expected: int) -> None:
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-n", "0", "-p", "no:cacheprovider", *arguments],
        cwd=ROOT,
        text=True,
        capture_output=True,
        env=dict(os.environ),
    )
    output = result.stdout + result.stderr
    require(result.returncode == 0, f"{where} failed:\n{output}")
    require(f"{expected} passed" in output, f"{where}: expected {expected} passed:\n{output}")
    require(" skipped" not in output and "xfail" not in output and "deselected" not in output, f"{where}: skip/xfail/deselect is not accepted:\n{output}")
    print(f"OK: {where} ({expected} passed)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write-manifest", action="store_true")
    args = parser.parse_args()
    if args.write_manifest:
        write_manifest()
        return
    verify_manifest()
    verify_baseline()
    verify_history()
    verify_rendered_prompts()
    pytest_nodes(list(CPU_TESTS), where="P2-M prompt/scratch/legacy anchors", expected=88)
    pytest_nodes(list(RECOVERY_TESTS), where="P2-M recovered TileLang frontend", expected=5)
    pytest_nodes(list(DOC_TESTS), where="P2-M structure and docs", expected=10)
    print("OK: P2-M acceptance (103 CPU; prompt epoch only, no model/search/publication)")


if __name__ == "__main__":
    main()
