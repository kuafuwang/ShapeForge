"""MT1a DSL-neutral registry, lazy adapter, prompt and TASK compatibility tests."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from dataclasses import replace
from pathlib import Path

import pytest

from taic.cli import _resume_track_choices, _run_track_choices
from taic.llm import LLMConfig, SearchConfig
from taic.orchestrator import Orchestrator
from taic.kernel_agent import SessionBrief, _check_command, render_contract
from taic.prompts import system_prompt_v2
from taic.task_binding import bind_task
from taic.track_prompt import prompt_style_for_track
from taic.trust import current_prompt_sha1
from types import SimpleNamespace
from tcenv.contract_errors import UnsupportedContractCapability
from tcenv.evaluate import EvalReport
from tcenv.identity import fingerprint
from tcenv.target import load_target
from tcenv.track.adapter import LazyTrackAdapter
from tcenv.track.plane import TrackRenderer
from tcenv.track.registry import TrackImplementation, registry_from_document
from tests.support.fakes import FakeClient
from tests.support.v1_tasks import V1_TASK

ROOT = Path(__file__).resolve().parents[2]
ACCEPTANCE = ROOT / "tests/support/golden/shared/多DSL-Track-阶段验收/MT1a"


def test_dynamic_cli_choices_read_available_and_registered_tracks(monkeypatch):
    from tcenv.track.registry import load_track_registry

    document = copy.deepcopy(load_track_registry().document())
    available = copy.deepcopy(document["tracks"][0])
    available["track_id"] = "fixture-available"
    available["evidence"] = ["fixture-only"]
    interface = copy.deepcopy(document["tracks"][0])
    interface["track_id"] = "fixture-interface-only"
    interface["status"] = "interface_only"
    interface.pop("evidence", None)
    document["tracks"].extend([available, interface])
    registry = registry_from_document(document)
    monkeypatch.setattr("tcenv.track.registry.load_track_registry", lambda *args, **kwargs: registry)
    assert _run_track_choices() == ["msl-single", "msl-bundle", "tilelang-pipeline", "fixture-available"]
    # Resume parses every registered id.  TileLang /1 remains interface_only, while MT2 /2
    # is available only for the fixed manual evaluation scope.
    assert _resume_track_choices() == [
        "msl-single",
        "msl-bundle",
        "tilelang-pipeline",
        "fixture-available",
        "fixture-interface-only",
    ]


def test_lazy_adapter_constructs_on_demand_and_needs_no_default_registry(monkeypatch):
    from tcenv.track.registry import load_track_registry

    spec = load_track_registry().get("msl-single")
    calls = []
    adapter = LazyTrackAdapter(spec, {
        "renderer": lambda: calls.append("renderer") or TrackRenderer.for_track(spec),
    })
    assert calls == []
    monkeypatch.setattr("tcenv.track.registry.load_track_registry", lambda: (_ for _ in ()).throw(AssertionError("default registry read")))
    first = adapter.resolve("renderer")
    assert first.renderer_id == spec.renderer.implementation_id
    assert adapter.resolve("renderer") is first
    assert calls == ["renderer"]
    with pytest.raises(UnsupportedContractCapability, match="no registered 'builder'"):
        adapter.resolve("builder")


def test_unknown_renderer_identity_fails_closed_instead_of_getting_msl_prompt_text():
    from tcenv.track.registry import load_track_registry

    spec = load_track_registry().get("msl-single")
    changed = replace(spec, renderer=TrackImplementation("unregistered.frontend", 1))
    with pytest.raises(UnsupportedContractCapability, match="no registered prompt adapter"):
        prompt_style_for_track(changed)


def _brief(track_id: str):
    binding = replace(bind_task("gemm", "fp16", "small"), track_id=track_id)
    presentation = binding.presentation()
    backend = SimpleNamespace(display_name="Metal", source_fence="metal", reference_skill="msl_reference")
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


@pytest.mark.parametrize("track_id", ["msl-single", "msl-bundle"])
def test_v2_prompt_task_and_commands_remain_byte_identical_to_the_mt1a_goldens(track_id):
    before = json.loads((ACCEPTANCE / "baseline.json").read_text())["task_goldens"][track_id]
    binding, brief = _brief(track_id)
    task = render_contract(brief)
    assert _check_command(brief, "quick") == before["quick_command"]
    assert _check_command(brief, "timed") == before["timed_command"]
    prompt = system_prompt_v2(binding, use_tools=True, target_id="apple-m4-max")
    if track_id == "msl-single":
        assert hashlib.sha256(task.encode()).hexdigest() == before["task_sha256"]
        assert task == (ACCEPTANCE / "golden" / f"gemm-{track_id}-TASK.md").read_text()
        assert hashlib.sha256(prompt.encode()).hexdigest() == before["system_prompt_sha256"]
    else:
        # The archived MT1a golden stays unchanged; the live msl-bundle prompt/TASK surface
        # now documents restricted-C99 host/tiling.c, tcenv.cli host-probe and per-stage
        # contiguous [[buffer(0..n-1)]] slots alongside host/tiling.json.
        assert hashlib.sha256(task.encode()).hexdigest() != before["task_sha256"]
        assert hashlib.sha256(prompt.encode()).hexdigest() != before["system_prompt_sha256"]
        assert "host/tiling.c" in task and "host/tiling.json" in task and "host-probe" in task
        assert "host/tiling.c" in prompt and "host/tiling.json" in prompt and "host-probe" in prompt


def test_prompt_source_fingerprint_moves_to_a_new_epoch_without_rewriting_the_old_baseline():
    before = json.loads((ACCEPTANCE / "baseline.json").read_text())
    after = json.loads((ROOT / "tests/support/golden/shared/多DSL-Track-阶段验收/MT4/prompt-epoch.json").read_text())
    assert before["current_prompt_sha1"] == "5fb6aa1ff55e03b72d10c5dca160aacc1e95691d5245ba46f9114d9fc05d515c"
    # MT4 records its own epoch and is not rewritten.  Later prompt text is a new live fingerprint.
    assert after["current_prompt_sha1"] == "cfb02704b803e4f54b379ba6e52e9870487d898e78240bc72229dd3924018ec6"
    assert current_prompt_sha1() == "f27dc0d14ae81ab29c42accc38d6e83a578902811fb4c3f707b3feab5186d7ed"
    assert current_prompt_sha1() != after["current_prompt_sha1"]
    assert current_prompt_sha1() != before["current_prompt_sha1"]
    historical = json.loads((ROOT / "docs/架构/backend-phase0-baseline.json").read_text())
    assert historical["prompt_sha1"] == before["current_prompt_sha1"]


def test_v1_run_config_does_not_gain_the_v2_confirmation_guidance_field(tmp_path):
    cfg = LLMConfig(base_url="http://x", api_key="sk-test-0123456789abcdef", model="fake", reasoning_effort="off", use_tools=False)
    search = SearchConfig(
        budget_usd=1.0, branches=1, max_rounds=0, eval_reps=3,
        stop_at_roofline=False, abi_guard=False, target_note=False, rich_history=False, status_lines=False,
    )

    def evaluate(*_args, **_kwargs):
        return EvalReport(task=V1_TASK, dtype="fp16", size="small", mode="full", status="ok")

    orch = Orchestrator(
        V1_TASK, "fp16", "small", cfg, search, run_dir=tmp_path / "run",
        client=FakeClient([]), evaluate_fn=evaluate, log=lambda _s: None, seed=1,
    )
    config = json.loads((orch.run_dir / "run_config.json").read_text())
    assert "confirmation_guidance_digest" not in config


def _verifier_module():
    spec = importlib.util.spec_from_file_location("mt1a_acceptance_verify", ACCEPTANCE / "verify.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    from tests.support.boundaries import rebind_acceptance_root

    rebind_acceptance_root(module)
    return module


@pytest.mark.parametrize("mutation", ["pack", "prompt", "registry"])
def test_verifier_rejects_each_fixed_negative_mutation(mutation):
    before = json.loads((ACCEPTANCE / "baseline.json").read_text())
    after = json.loads((ACCEPTANCE / "after.json").read_text())
    if mutation == "pack":
        after["task_packs"]["gemm"]["deployment_pack_digest"] = "0" * 64
    elif mutation == "prompt":
        after["current_prompt_sha1"] = before["current_prompt_sha1"]
    else:
        after["registry"]["digest"] = "0" * 64
    with pytest.raises(ValueError):
        _verifier_module().verify_surface(before, after)


def test_registry_document_and_shipped_track_identities_stay_unchanged():
    """@brief MT1a 的冻结 registry 字节仍可重建；live registry 只在 MT1b 新 epoch 前进。

    MT1b is required to register a new delivery mechanism, so the *live* registry necessarily
    moves.  The MT1a epoch itself is not backfilled: its exact schema-1 bytes are archived in the
    MT1b package and still reconstruct the recorded registry digest, and the two MSL Track
    documents are byte-identical (same per-Track digests).
    """
    before = json.loads((ACCEPTANCE / "baseline.json").read_text())["registry"]
    from tcenv.track.registry import LEGACY_TRACKS_SCHEMA_VERSION, METAL_TRACKS_PATH, load_track_registry

    frozen_path = ROOT / "tests/support/golden/shared/多DSL-Track-阶段验收/MT1b/evidence/tracks-v1-epoch.json"
    assert hashlib.sha256(frozen_path.read_bytes()).hexdigest() == before["path_sha256"]
    frozen = registry_from_document(json.loads(frozen_path.read_text(encoding="utf-8")))
    assert frozen.schema_version == LEGACY_TRACKS_SCHEMA_VERSION
    assert fingerprint(frozen.document()) == before["digest"]
    assert {track.track_id: fingerprint(track.document()) for track in frozen.tracks} == before["tracks"]

    registry = load_track_registry()
    assert registry.schema_version == 2
    # The MSL Track documents did not change; only the registry as a whole gained a Track.
    assert {
        track.track_id: fingerprint(track.document())
        for track in registry.tracks
        if track.track_id.startswith("msl-")
    } == before["tracks"]
    assert Path(METAL_TRACKS_PATH).is_file()
    assert registry.get("tilelang-pipeline", version=1).status == "interface_only"
    assert registry.get("tilelang-pipeline", version=2).status == "available"
