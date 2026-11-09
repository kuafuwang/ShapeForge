#!/usr/bin/env python3
"""MT5 deterministic fixture, schema, selection-policy and manifest verifier."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
TEST_NODE = "tests/shared/test_track_explore.py"
TEST_NODES = tuple(
    json.loads((HERE / "nodes.json").read_text())["positive"]
    + json.loads((HERE / "nodes.json").read_text())["negative"]
)


def _sanitize(value, root: Path):
    if isinstance(value, str):
        return value.replace(str(root), "<fixture>")
    if isinstance(value, list):
        return [_sanitize(item, root) for item in value]
    if isinstance(value, dict):
        return {key: _sanitize(item, root) for key, item in value.items()}
    return value


def _fixture(root: Path):
    from tests.support.track_explore import FixedTrackExploreFixture

    return FixedTrackExploreFixture(root)


def _run(fixture, **kwargs):
    from taic.track_explore import explore_tracks

    return explore_tracks(
        fixture.config, runner=fixture, adapter_factory=fixture.adapter_factory,
        metadata_provider=fixture.metadata, **kwargs,
    )


def verify_example() -> None:
    with tempfile.TemporaryDirectory() as directory:
        fixture = _fixture(Path(directory))
        summary = _run(fixture)
        expected = json.loads((HERE / "evidence/example-summary.json").read_text())
        if _sanitize(summary, Path(directory)) != expected:
            raise ValueError("the full-success fixture does not reproduce example-summary.json")
        if summary["winner"]["track_id"] != "msl-bundle":
            raise ValueError("deterministic winner drifted")
        if summary["same_fresh_input"] is not False or summary["formal_publication"] is not False:
            raise ValueError("scheme A is being misrepresented as same-input or formal publication")
        single, bundle = summary["tracks"]
        if single["instances"][0]["actual_confirmation"]["confirmation_seed"] == bundle["instances"][0]["actual_confirmation"]["confirmation_seed"]:
            raise ValueError("the scheme-A fixture must retain distinct per-Track fresh seeds")
        before = fixture.config.output_path.read_bytes()
        calls = len(fixture.calls)
        if _run(fixture) != summary or fixture.config.output_path.read_bytes() != before or len(fixture.calls) != calls:
            raise ValueError("idempotent replay restarted work or changed the summary")


def verify_policy_and_schema() -> None:
    from taic.track_explore import selection_policy_digest, validate_track_explore

    if selection_policy_digest() != "22e4283765d35cba939699a10cfac70592f8f6147d2ec8100fb0ac06cd3ca5f8":
        raise ValueError("selection policy digest drifted")
    with tempfile.TemporaryDirectory() as directory:
        fixture = _fixture(Path(directory))
        for index, mutate in enumerate((
            lambda doc: doc.update(master_seed="0" * 64),
            lambda doc: doc.update(comparability_digest="0" * 64),
            lambda doc: doc.update(generation_policy="parallel"),
            lambda doc: doc["participating_tracks"][1].update(run_dir="runs/msl-single"),
        )):
            changed = copy.deepcopy(fixture.config.document)
            mutate(changed)
            try:
                validate_track_explore(changed)
            except ValueError:
                continue
            raise AssertionError(f"schema self-test mutation {index} was accepted")


def verify_manifest() -> None:
    for line in (HERE / "manifest.sha256").read_text().splitlines():
        if not line.strip():
            continue
        expected, relative = line.split("  ", 1)
        path = HERE / relative
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"manifest mismatch: {relative}")


def _pytest() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-n", "0", *TEST_NODES],
        cwd=ROOT, text=True, capture_output=True, check=False,
    )
    output = completed.stdout + completed.stderr
    if completed.returncode != 0:
        print(output, end="")
        raise ValueError("MT5 fixed pytest nodes failed; read logs/pytest/triage/triage_summary.md")
    match = re.search(r"(\d+) passed", output)
    if match is None or int(match.group(1)) != 45 or re.search(r"[1-9]\d* (?:skipped|xfailed)", output):
        raise ValueError(f"expected exactly 45 passed without skips, got: {output[-1000:]}")


def main() -> int:
    verify_policy_and_schema()
    verify_example()
    if "--check" in sys.argv:
        verify_manifest()
        _pytest()
    print(json.dumps({
        "done": True,
        "stage": "MT5",
        "scope": "deterministic fake agent and CPU fixtures only",
        "fixed_nodes": {TEST_NODE: 45},
        "schema_negative_mutations": 4,
        "winner": "msl-bundle (fixture only)",
        "same_fresh_input": False,
        "formal_publication": False,
        "real_taic_run_resume_model_gpu_npu": False,
    }, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
