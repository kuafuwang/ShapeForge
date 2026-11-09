#!/usr/bin/env python3
"""MT1a before/after identity, golden-output, regression and manifest verifier."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
NEW_NODES = tuple(
    json.loads((HERE / "nodes.json").read_text())["new"]
    + json.loads((HERE / "nodes.json").read_text())["negative"]
)
REGRESSION_NODES = (
    "tests/shared/test_search_v2_bundle_prompt.py",
    "tests/shared/test_search_v2_bundle_track.py",
    "tests/shared/test_search_v2_prompt.py",
    "tests/shared/test_search_task_binding.py",
    "tests/shared/test_trust.py",
    "tests/backends/metal/test_search_v2_bundle_session.py",
    "tests/backends/metal/test_track_registry.py",
    "tests/backends/metal/test_track_a4.py",
)


def _load(name):
    return json.loads((HERE / name).read_text(encoding="utf-8"))


def _brief(track_id):
    from taic.kernel_agent import SessionBrief
    from taic.task_binding import bind_task
    from tcenv.backends.registry import load_backend
    from tcenv.target import load_target

    binding = replace(bind_task("gemm", "fp16", "small"), track_id=track_id)
    presentation = binding.presentation()
    backend = load_backend("metal")
    target = load_target("apple-m4-max")
    return binding, SessionBrief(
        repo="/repo", task_name="gemm", dtype="fp16", size="small", kernel_name="gemm_kernel", triton_source="",
        description=presentation.statement,
        contract_text=json.dumps(presentation.abi_document, indent=2, sort_keys=True),
        abi_stub=binding.signature_stub(), rtol=binding.tolerances()["rtol"], atol_rel=binding.tolerances()["atol_rel"],
        v2=True, reference_source=presentation.reference_source, seedless=True, label="init", work_dir="/work/init",
        target_id="apple-m4-max", track_id=track_id, file_ext=target.file_ext,
        language_label=backend.display_name, source_fence=backend.source_fence,
        reference_skill=backend.reference_skill,
        source_name_pattern=r"\bkernel\s+void\s+[A-Za-z_][A-Za-z0-9_]*\b",
    )


def recompute_surface():
    from taic.kernel_agent import _check_command, render_contract
    from taic.prompts import system_prompt_v2
    from taic.task_binding import TaskBinding
    from tcenv.identity import fingerprint
    from tcenv.taskcontract.locator import discover_standard_tasks
    from tcenv.track.registry import METAL_TRACKS_PATH, load_track_registry

    registry = load_track_registry()
    packs, presentations = {}, {}
    for locator in discover_standard_tasks(ROOT / "tasks"):
        contract = locator.load()
        packs[locator.task_id] = {
            "deployment_pack_digest": locator.deployment_pack_digest,
            "locator_identity": locator.identity,
        }
        base = TaskBinding(
            mode="v2", name=locator.op_name, dtype=locator.dtype, size=locator.size,
            target_id="apple-m4-max", locator=locator, contract=contract,
        )
        for track in registry.tracks:
            binding = replace(base, track_id=track.track_id)
            presentation = binding.presentation()
            row = {
                "presentation_sha256": binding.presentation_sha256(),
                "statement_sha256": hashlib.sha256(presentation.statement.encode()).hexdigest(),
            }
            try:
                row["signature_stub_sha256"] = hashlib.sha256(binding.signature_stub().encode()).hexdigest()
            except Exception as exc:
                row["signature_stub_error"] = f"{type(exc).__name__}: {exc}"
            presentations[f"{locator.task_id}:{locator.dtype}:{locator.size}:{track.track_id}"] = row
    tasks = {}
    for track_id in ("msl-single", "msl-bundle"):
        binding, brief = _brief(track_id)
        task = render_contract(brief)
        tasks[track_id] = {
            "task_sha256": hashlib.sha256(task.encode()).hexdigest(),
            "quick_command": _check_command(brief, "quick"),
            "timed_command": _check_command(brief, "timed"),
            "system_prompt_sha256": hashlib.sha256(system_prompt_v2(binding, use_tools=True, target_id="apple-m4-max").encode()).hexdigest(),
        }
    return {
        "registry": {
            "path_sha256": hashlib.sha256(Path(METAL_TRACKS_PATH).read_bytes()).hexdigest(),
            "digest": fingerprint(registry.document()),
            "tracks": {track.track_id: fingerprint(track.document()) for track in registry.tracks},
        },
        "task_packs": packs,
        "presentations": presentations,
        "task_goldens": tasks,
    }


def verify_surface(before: dict, after: dict) -> None:
    for key in ("registry", "task_packs", "presentations", "task_goldens"):
        if before[key] != after[key]:
            raise ValueError(f"MT1a compatibility surface drifted: {key}")
    if before["current_prompt_sha1"] == after["current_prompt_sha1"]:
        raise ValueError("editing registered prompt functions must move the source fingerprint")
    if after["changes_from_baseline"]["old_current_prompt_sha1"] != before["current_prompt_sha1"]:
        raise ValueError("the old prompt fingerprint is not honestly recorded")
    if after["changes_from_baseline"]["new_current_prompt_sha1"] != after["current_prompt_sha1"]:
        raise ValueError("the new prompt fingerprint record is inconsistent")
    changes = after["changes_from_baseline"]
    for key in ("task_pack_changes", "presentation_changes", "task_golden_changes"):
        if changes[key]:
            raise ValueError(f"unrecorded compatibility change: {key}={changes[key]}")
    if changes["registry_document_changed"] is not False:
        raise ValueError("registry identity changed in MT1a")
    historical = "docs/架构/backend-phase0-baseline.json"
    if hashlib.sha256((ROOT / historical).read_bytes()).hexdigest() != after["historical_evidence"][historical]:
        raise ValueError("historical phase-0 evidence was rewritten")
    baseline = json.loads((ROOT / historical).read_text())
    if baseline["prompt_sha1"] != before["current_prompt_sha1"]:
        raise ValueError("historical prompt identity was backfilled")


def verify_manifest() -> None:
    for line in (HERE / "manifest.sha256").read_text().splitlines():
        if not line.strip():
            continue
        expected, relative = line.split("  ", 1)
        path = HERE / relative
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"manifest mismatch: {relative}")


def self_test(before: dict, after: dict) -> None:
    import copy

    pack = copy.deepcopy(after)
    pack["task_packs"] = dict(pack["task_packs"])
    pack["task_packs"]["gemm"] = dict(pack["task_packs"]["gemm"])
    pack["task_packs"]["gemm"]["deployment_pack_digest"] = "0" * 64
    prompt = copy.deepcopy(after)
    prompt["current_prompt_sha1"] = before["current_prompt_sha1"]
    registry = copy.deepcopy(after)
    registry["registry"] = dict(registry["registry"])
    registry["registry"]["digest"] = "0" * 64
    for index, changed in enumerate((pack, prompt, registry)):
        try:
            verify_surface(before, changed)
        except ValueError:
            continue
        raise AssertionError(f"MT1a verifier self-test mutation {index} was accepted")


def _pytest(nodes: tuple[str, ...], expected: int) -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-n", "0", "-m", "not gpu and not llm and not slow", *nodes],
        cwd=ROOT, text=True, capture_output=True, check=False,
    )
    output = completed.stdout + completed.stderr
    if completed.returncode != 0:
        print(output, end="")
        raise ValueError(f"pytest failed ({completed.returncode}); read logs/pytest/triage/triage_summary.md")
    match = re.search(r"(\d+) passed", output)
    if match is None or int(match.group(1)) != expected or re.search(r"[1-9]\d* (?:skipped|xfailed)", output):
        raise ValueError(f"unexpected pytest count for {nodes}: expected {expected}, output was {output[-1000:]}")


def main() -> int:
    before, after = _load("baseline.json"), _load("after.json")
    verify_surface(before, after)
    from taic.trust import current_prompt_sha1

    if current_prompt_sha1() != after["current_prompt_sha1"]:
        raise ValueError("live prompt source identity differs from after.json")
    current = recompute_surface()
    for key in ("registry", "task_packs", "presentations", "task_goldens"):
        if before[key] != current[key]:
            raise ValueError(f"live MT1a compatibility surface drifted: {key}")
        if current[key] != after[key]:
            raise ValueError(f"after.json is not the independently recomputed current surface: {key}")
    self_test(before, after)
    if "--check" in sys.argv:
        verify_manifest()
        _pytest(REGRESSION_NODES, 86)
        _pytest(NEW_NODES, 23)
    print(json.dumps({
        "done": True,
        "stage": "MT1a",
        "scope": "DSL-neutral CPU/protocol acceptance; no real Metal run",
        "task_packs_unchanged": len(current["task_packs"]),
        "presentations_unchanged": len(current["presentations"]),
        "prompt_source_epoch": [before["current_prompt_sha1"], after["current_prompt_sha1"]],
        "regression_passed": 86,
        "new_passed": 23 if "--check" in sys.argv else None,
        "manifest": "--check" in sys.argv,
    }, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
