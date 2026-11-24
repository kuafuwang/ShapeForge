"""Current structural invariants after the old task assembly was removed.

No archived task, pre-migration registry or pre-A6 source is opened.  These checks cover the live
v2 discovery/contract/track boundaries and the single-owner rules that remain relevant today.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from tcenv.contract_errors import TaskContractError

ROOT = Path(__file__).resolve().parents[2]


def _walk(root: Path):
    for path in sorted(root.rglob("*.py")):
        if "__pycache__" not in path.parts:
            yield path


def _source(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _definitions(path: Path, kinds=(ast.ClassDef, ast.FunctionDef)) -> set[str]:
    tree = ast.parse(_source(path))
    return {node.name for node in tree.body if isinstance(node, kinds) and not node.name.startswith("_")}


def test_current_task_directories_are_v2_and_carry_no_launch_facts():
    from tcenv.taskcontract import discover_standard_tasks, load_task_contract

    locators = discover_standard_tasks()
    assert locators
    retired = ("task.py", "track.json", "kernel_triton.py", "kernel_triton_msl.py", "naive.metal")
    for locator in locators:
        directory = locator.task_dir
        for name in retired:
            assert not (directory / name).exists(), (directory, name)
        document = locator.load().spec.document
        assert document["spec_version"] == 2 and document["op_name"] == locator.op_name
        keys: list[str] = []

        def collect(value):
            if isinstance(value, dict):
                keys.extend(str(key).lower() for key in value)
                for item in value.values():
                    collect(item)
            elif isinstance(value, list):
                for item in value:
                    collect(item)

        collect(document)
        assert not ({"grid", "threads", "num_warps", "tile", "block_size", "lowering_profile"} & set(keys))
        with pytest.raises(TaskContractError):
            load_task_contract(directory, variant="128x64x32t256")


def test_the_track_registry_stays_task_agnostic():
    from tcenv.track.registry import METAL_TRACKS_PATH, load_track_registry

    document = json.loads(METAL_TRACKS_PATH.read_text(encoding="utf-8"))
    assert document["schema_version"] == 2
    keys: set[str] = set()

    def collect(value):
        if isinstance(value, dict):
            keys.update(str(key).lower() for key in value)
            for item in value.values():
                collect(item)
        elif isinstance(value, list):
            for item in value:
                collect(item)

    collect(document)
    assert not ({"op_name", "task", "shape", "tile", "grid", "threads", "lowering_profile"} & keys)
    registry = load_track_registry(METAL_TRACKS_PATH)
    assert registry.require_available("msl-single").track_id == "msl-single"
    assert registry.require_available("msl-bundle").track_id == "msl-bundle"
    with pytest.raises(TaskContractError):
        registry.require_available("no-such-track")


def test_each_current_configuration_class_has_one_reader():
    loader = ROOT / "tcenv/taskcontract/loader.py"
    assert {"load_task_contract", "TaskContract"} <= _definitions(loader)
    for path in _walk(ROOT / "tcenv"):
        if path != loader:
            assert "load_task_contract" not in _definitions(path), path

    registry = ROOT / "tcenv/track/registry.py"
    assert "TrackRegistry" in _definitions(registry)
    for path in _walk(ROOT / "tcenv"):
        if path != registry:
            assert "TrackRegistry" not in _definitions(path), path

    assert {"profile_for", "lowering_profile_digest", "register_profile"} <= _definitions(ROOT / "tcenv/lowering.py")
    metal_lowering = _source(ROOT / "tcenv/backends/metal/lowering.py")
    assert "def register_profile(" not in metal_lowering and "_PROFILES" not in metal_lowering

    artifact_builder = ROOT / "tcenv/remote/artifact_builder.py"
    assert {"ArtifactBuilder", "ArtifactBuilderRegistry"} <= _definitions(artifact_builder)
    for path in _walk(ROOT / "tcenv"):
        if path != artifact_builder:
            names = _definitions(path)
            assert "ArtifactBuilderRegistry" not in names and "ArtifactBuilder" not in names, path


def test_no_second_pattern_registry_or_numeric_verdict():
    verdict_owners = [path for path in _walk(ROOT / "tcenv") if "def scaled_error(" in _source(path)]
    assert [path.relative_to(ROOT).as_posix() for path in verdict_owners] == ["tcenv/backends/metal/testing.py"]
    runtime = _source(ROOT / "tcenv/backends/metal/v2_runtime.py")
    assert "from .testing import" in runtime and "scaled_error" in runtime
    forbidden = {"StagePartitioner", "TileConnector", "LoweringRuleRegistry", "PatternRegistry"}
    for root in ("tcenv", "taic"):
        for path in _walk(ROOT / root):
            assert not (_definitions(path) & forbidden), path


def test_a_noisy_measurement_is_uncertain_and_never_a_successful_confirmation():
    from tcenv.bench import QUIET_CV, timing_is_uncertain

    def report(candidate_cv: float, baseline_cv: float = 0.02, *, reps: int = 5, noisy: bool = False):
        return {
            "benchmark": {
                "candidate": {"cv": candidate_cv, "reps": reps},
                "baselines": {"mlx": {"cv": baseline_cv, "reps": reps}},
                "best_baseline": "mlx", "noisy": noisy,
            }
        }

    assert QUIET_CV == 0.05
    assert timing_is_uncertain(report(0.02)) is False
    assert timing_is_uncertain(report(QUIET_CV + 0.01)) is True
    assert timing_is_uncertain(report(0.02, QUIET_CV + 0.01)) is True
    assert timing_is_uncertain(report(0.02, noisy=True)) is True
    assert timing_is_uncertain(report(0.02, reps=1)) is True
    assert timing_is_uncertain({"benchmark": {}}) is True
    assert timing_is_uncertain(report(float("nan"))) is True
