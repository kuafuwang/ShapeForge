"""MT1b: TileLang adapter construction and the MSL seam gates (CPU only).

The point of these tests is negative control: the MSL portfolio projection, preflight and
confirmation rebuild must never swallow a registered DSL Track, and an ``interface_only`` Track
must never look selectable, buildable, confirmable or publishable.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from types import SimpleNamespace

import pytest

from tcenv.backends.metal import search_candidate as search
from tcenv.backends.metal.tilelang_adapter import (
    TileLangSourceOnlyBuilder,
    TileLangTrackValidator,
    expected_build_identity,
)
from tcenv.backends.metal.track_factory import adapter_for_track
from tcenv.contract_errors import TaskContractError, UnsupportedContractCapability
from tcenv.portfolio.candidate import project_provider
from tcenv.portfolio.domain import DomainAuthority, normalize_semantic_domain_plan
from tcenv.target import load_target
from tcenv.taskcontract.locator import discover_standard_tasks
from tcenv.track import REPO_ROOT, CandidateArtifact, CandidateEntry, load_track_registry
from tcenv.track import tilelang as contract

from tests.support.shape_portfolio_metal import load_p1_task, p1_instances
from tests.backends.metal.test_track_mt1b_build_seam import _candidate as source_only_candidate

ROOT = Path(REPO_ROOT)
TARGET = load_target("apple-m4-max")
TILELANG = "tilelang-pipeline"

GOOD_FRONTEND = {
    "frontend_id": contract.FRONTEND_ID,
    "frontend_version": contract.FRONTEND_VERSION,
    "entry": contract.FRONTEND_ENTRY,
    "options_sha256": contract.options_sha256(),
    "package_manifest_sha256": "a" * 64,
    "dist_info_metadata_sha256": "b" * 64,
    "dist_info_record_sha256": "c" * 64,
}


def _spec():
    # MT1b's source-only/interface-only behavior is frozen on /1; MT2 adds a separate /2.
    return load_track_registry().get(TILELANG, version=1)


def _candidate() -> CandidateArtifact:
    return CandidateArtifact(
        track_id=TILELANG,
        entries=(
            CandidateEntry(
                path=contract.SOURCE_PATH,
                role=contract.SOURCE_ROLE,
                media_type=contract.SOURCE_MEDIA_TYPE,
                data=b"from __future__ import annotations\nimport tilelang\n",
            ),
            CandidateEntry(
                path=contract.GEOMETRY_PATH,
                role=contract.GEOMETRY_ROLE,
                media_type=contract.GEOMETRY_MEDIA_TYPE,
                data=b'{"schema_version": 1, "source": "candidate", "stages": [{"kernel": "k", "grid": [1, 1, 1], "threads": [1, 1, 1]}]}',
            ),
        ),
    )


def _request(candidate: CandidateArtifact | None = None, spec=None):
    spec = spec or _spec()
    return SimpleNamespace(
        track_selection=SimpleNamespace(track=spec),
        candidate=source_only_candidate() if candidate is None else candidate,
        selection=SimpleNamespace(
            geometry=SimpleNamespace(
                source="candidate", stages=(SimpleNamespace(kernel="scale_kernel"),)
            )
        ),
    )


def _p1(tmp_path):
    root = tmp_path / "tasks"
    load_p1_task(root)
    locator = discover_standard_tasks(root, code_root=ROOT)[0]
    contract_ = locator.load()
    instances = p1_instances(contract_)
    plan = normalize_semantic_domain_plan(contract_.spec, instances, target=TARGET)
    authority = DomainAuthority(
        plan=plan, spec=contract_.spec, target=TARGET, instances=instances, contract=contract_
    )
    return locator, contract_, plan, authority


def test_the_factory_constructs_the_tilelang_adapter_without_an_msl_substitution():
    """@brief 工厂按登记身份构造 TileLang 服务；绝不回落到 MSL 构建/几何解析器。"""
    spec = _spec()
    adapter = adapter_for_track(spec)
    assert adapter.spec is spec
    validator = adapter.resolve("validator")
    assert isinstance(validator, TileLangTrackValidator)
    assert isinstance(adapter.resolve("builder"), TileLangSourceOnlyBuilder)
    from tcenv.track.delivery import candidate_geometry

    assert adapter.resolve("geometry_resolver") is candidate_geometry
    renderer = adapter.resolve("renderer")
    assert renderer.renderer_id == spec.renderer.implementation_id
    for service in ("evaluator", "confirmation"):
        with pytest.raises(UnsupportedContractCapability, match="interface_only"):
            adapter.resolve(service)


def test_the_factory_does_not_read_a_default_registry(monkeypatch):
    """@brief 工厂只消费调用方选定的 TrackSpec（worker 不读默认 registry）。"""
    spec = _spec()
    monkeypatch.setattr(
        "tcenv.track.registry.load_track_registry",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("default registry read")),
    )
    adapter = adapter_for_track(spec)
    assert adapter.resolve("validator")(_candidate()) == "accepted"


def test_an_unknown_tilelang_adapter_identity_is_refused_not_routed_to_msl():
    """@brief build adapter 版本漂移 → 无适配器，不落到 MSL 构建。"""
    drifted = dataclasses.replace(
        _spec(), build_adapter=dataclasses.replace(_spec().build_adapter, implementation_version=2)
    )
    with pytest.raises(UnsupportedContractCapability, match="no Metal adapter implements"):
        adapter_for_track(drifted)


def test_the_validator_accepts_a_dsl_candidate_and_refuses_msl_bytes():
    """@brief validator 是 TileLang 源合同，不是 MSL 解析器。"""
    validator = adapter_for_track(_spec()).resolve("validator")
    assert validator(_candidate()) == "accepted"
    msl = CandidateArtifact(
        track_id=TILELANG,
        entries=(CandidateEntry(path="kernel.metal", role="device_kernel", media_type="text/plain", data=b"kernel void k() {}"),),
    )
    with pytest.raises(TaskContractError, match="not registered"):
        validator(msl)


def test_the_source_only_builder_fails_closed_at_every_stage(tmp_path):
    """@brief prepare 检查 Track/交付；真实 build 只有“成功”或 typed 能力缺口，Metal build 拒绝。"""
    spec = _spec()
    builder = TileLangSourceOnlyBuilder(spec, attempt_base=tmp_path)
    builder.prepare(_request())

    wrong_track = dataclasses.replace(spec, track_version=2)
    with pytest.raises(TaskContractError, match="frozen contract is version 1"):
        TileLangSourceOnlyBuilder(wrong_track, attempt_base=tmp_path)

    with pytest.raises(TaskContractError, match="does not belong to the TileLang Track"):
        builder.prepare(
            _request(
                CandidateArtifact(
                    track_id="msl-single",
                    entries=(CandidateEntry(path="kernel.metal", role="device_kernel", media_type="text/plain", data=b"k"),),
                )
            )
        )

    # The production build either completes (and then it is a verified source-only build) or it is
    # a typed capability gap.  There is no third outcome and no unfenced fallback.
    try:
        outcome = builder.build(_request())
    except contract.TileLangFrontendUnavailable as gap:
        assert gap.code == "backend_capability_unavailable"
    else:
        assert outcome.identity.full_build is True
        assert outcome.document()["device_dispatch_calls"] == 0

    with pytest.raises(contract.TileLangBuildModeUnavailable) as unimplemented:
        builder.build_metal(_request())
    assert unimplemented.value.code == "unsupported_contract_capability"


def test_expected_build_identity_is_independent_and_binds_the_projected_bytes():
    """@brief expected 只从投影候选 + 冻结前端/options + 环境重建，不接收任何报告/工件。"""
    candidate = _candidate()
    environment = {
        "python_version": "3.13.1", "python_implementation": "CPython", "platform": "macOS", "machine": "arm64",
    }
    identity = expected_build_identity(candidate, environment=environment)
    assert identity.candidate_sha256 == candidate.candidate_sha256
    assert identity.full_build is False and identity.artifacts == () and identity.attempt == {}
    assert identity.frontend["package_manifest_sha256"] == contract.FRONTEND_PACKAGE_MANIFEST_SHA256

    edited = CandidateArtifact(
        track_id=TILELANG,
        entries=tuple(
            CandidateEntry(path=entry.path, role=entry.role, media_type=entry.media_type, data=entry.data + b"\n# edit")
            if entry.path == contract.SOURCE_PATH
            else entry
            for entry in candidate.entries
        ),
    )
    assert expected_build_identity(edited, environment=environment).cache_key != identity.cache_key
    # The same source with a different geometry is a different build subject.
    from tests.backends.metal.test_track_mt1b_build_seam import _candidate as _grid_candidate

    assert expected_build_identity(_grid_candidate(grid=1), environment=environment).cache_key != (
        expected_build_identity(_grid_candidate(grid=999), environment=environment).cache_key
    )
    # A device fact is not an environment fact.
    with pytest.raises(TaskContractError, match="environment"):
        expected_build_identity(candidate, environment={**environment, "device_name": "Apple M4 Max"})


def test_the_msl_preflight_refuses_a_dsl_track_before_any_msl_population(tmp_path, monkeypatch):
    """@brief preflight 的 track 关卡：typed code、停在本关、不进入 MSL 域/投影。"""
    _, contract_, _, _ = _p1(tmp_path)
    monkeypatch.setattr(
        search,
        "domain_for",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("MSL domain/projection was reached")),
    )
    outcome = search.preflight_submission(
        contract_, "import tilelang\n", target=TARGET, track_id=TILELANG, track_version=1
    )
    assert outcome.ok is False
    assert outcome.stage == "track"
    assert outcome.code == "unsupported_contract_capability"
    assert "interface_only" in outcome.rejection

    # An unknown track id no longer falls through to the single-file MSL branch.
    unknown = search.preflight_submission(contract_, "kernel void k() {}", target=TARGET, track_id="msl-future")
    assert unknown.ok is False and unknown.stage == "track"
    assert unknown.code == "unsupported_contract_capability"


def test_the_msl_single_file_preflight_still_works(tmp_path, monkeypatch):
    """@brief 现有 MSL 单文件预检不受新 track 增影响；冻结版本也进入预检内部的投影。"""
    _, contract_, _, _ = _p1(tmp_path)
    from tests.backends.metal.test_search_v2_bundle_preflight import SINGLE_SOURCE

    outcome = search.preflight_submission(
        contract_, SINGLE_SOURCE, target=TARGET, track_id="msl-single", track_version=1
    )
    assert outcome.ok is True, outcome.rejection
    assert outcome.projection.track_version == 1

    # A *selectable* second version must not capture a run frozen to version 1: the projection
    # inside the preflight is version-pinned, not "whatever the registry now prefers".  Without
    # the pin the second assertion below observes 2 (the run would silently change implementation).
    import tcenv.portfolio.candidate as candidate_module
    import tcenv.track.registry as registry_module
    from tcenv.track.registry import TrackRegistry

    live = registry_module.load_track_registry()
    upgraded = TrackRegistry(
        backend_id="metal",
        tracks=(
            live.get("msl-single"),
            dataclasses.replace(live.get("msl-single"), track_version=2),
            live.get("msl-bundle"),
            live.get(TILELANG, version=1),
        ),
    )
    monkeypatch.setattr(registry_module, "load_track_registry", lambda path=None: upgraded)
    monkeypatch.setattr(candidate_module, "load_track_registry", lambda path=None: upgraded)
    assert upgraded.require_available("msl-single").track_version == 2
    frozen = search.preflight_submission(
        contract_, SINGLE_SOURCE, target=TARGET, track_id="msl-single", track_version=1
    )
    assert frozen.ok is True, frozen.rejection
    assert frozen.projection.track_version == 1


def test_portfolio_for_submission_never_relabels_a_dsl_source(tmp_path):
    """@brief MT4 的显式 DSL 包装保留候选身份；裸 Python 仍拒绝，MSL 默认投影仍隔离。"""
    from tcenv.evaluation import CandidateDeliveryError
    from tcenv.track import encode_envelope

    with pytest.raises(CandidateDeliveryError, match="not raw Python"):
        search.portfolio_for_submission("import tilelang\n", None, authority=None, track_id=TILELANG)
    _, _, plan, authority = _p1(tmp_path)
    submitted = _candidate()
    candidate, _portfolio = search.portfolio_for_submission(
        encode_envelope(submitted), plan, authority=authority, track_id=TILELANG, track_version=2
    )
    assert candidate.candidate_sha256 == submitted.candidate_sha256
    assert candidate.entries[0].path == "host/execution_geometry.json"
    assert candidate.entries[1].path == "kernel.py" and candidate.entries[1].role == "dsl_source"


def test_the_msl_portfolio_projection_refuses_dsl_content(tmp_path):
    """@brief MSL 投影不按 geometry 有无选路：已登记 DSL 源声明被 typed 拒绝。"""
    _, contract_, plan, authority = _p1(tmp_path)
    from tcenv.track.registry import dsl_candidate_sources

    registered = dsl_candidate_sources(load_track_registry())[0]
    dsl_entry = CandidateEntry(
        path=registered.path,
        role="device_kernel",
        media_type=registered.media_type,
        data=b"import tilelang\n",
    )
    fake = SimpleNamespace(entries=(dsl_entry,))
    with pytest.raises(UnsupportedContractCapability, match="registered 'dsl_source' Track"):
        project_provider(fake, None, authority=authority)
    # A parent that declares the dsl_source role is refused even earlier by the portfolio parser,
    # and the projection gate covers the role explicitly.
    from tcenv.portfolio.candidate import _refuse_non_msl_projection_content

    role_entry = CandidateEntry(path="kernel.py", role="dsl_source", media_type="text/x-python", data=b"x")
    with pytest.raises(UnsupportedContractCapability, match="dsl_source role"):
        _refuse_non_msl_projection_content(
            SimpleNamespace(entries=(role_entry,)), load_track_registry(), where="fixture"
        )


def test_the_msl_confirmation_rebuild_refuses_a_dsl_subject(monkeypatch):
    """@brief confirmation expected 不调用 MSL source builder/probe 去判 DSL 构建。"""
    from tcenv.backends.metal import portfolio_confirmation as confirmation

    monkeypatch.setattr(
        confirmation,
        "combined_candidate_source",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("MSL source concatenation was reached")),
    )
    monkeypatch.setattr(
        confirmation,
        "MetalSourceBuilder",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("MSL probe was reached")),
    )
    projection = SimpleNamespace(track_id=TILELANG, track_version=1)
    with pytest.raises(UnsupportedContractCapability, match="MSL portfolio"):
        confirmation._resolved(SimpleNamespace(), projection, TARGET)
    with pytest.raises(UnsupportedContractCapability, match="MSL portfolio"):
        confirmation.confirmation_expected(
            SimpleNamespace(), projection, SimpleNamespace(), locator=SimpleNamespace(), target_id="apple-m4-max"
        )


def test_the_prompt_style_never_hands_msl_bundle_text_to_a_dsl_track():
    """@brief 未登记 renderer 的 prompt 适配器 → typed 拒绝（尤其不能给 MSL bundle 说明）。"""
    from taic.track_prompt import prompt_style_for_track

    with pytest.raises(UnsupportedContractCapability, match="no registered prompt adapter"):
        prompt_style_for_track(_spec())  # /1 remains source-only and has no generation adapter
    from taic.track_prompt import prompt_style_for_id

    style = prompt_style_for_id(TILELANG)
    assert style.style_id == "tilelang-pipeline" and style.envelope and style.explicit_track


def test_evaluate_candidate_and_evaluate_task_stop_at_the_track_boundary(tmp_path, monkeypatch):
    """@brief 搜索评测入口：interface_only Track 在 require_available 处 typed 拒绝。"""
    from tcenv.taskcontract.runner import evaluate_task

    locator, contract_, _, _ = _p1(tmp_path)
    monkeypatch.setattr(
        search,
        "domain_for",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("MSL domain/projection was reached")),
    )
    with pytest.raises(UnsupportedContractCapability, match="interface_only"):
        search.evaluate_candidate(locator, "import tilelang\n", target=TARGET, track_id=TILELANG, track_version=1)
    with pytest.raises(UnsupportedContractCapability, match="interface_only"):
        evaluate_task(
            locator,
            track_id=TILELANG,
            track_version=1,
            candidate_source="import tilelang\n",
            builder=SimpleNamespace(),
            evaluator=SimpleNamespace(),
        )


def test_confirmation_input_refuses_a_dsl_track(tmp_path):
    """@brief MT5 scheme-A confirmation 派生同样在 portfolio 选路前拒绝未开放确认的 DSL Track（/1、/2）。"""
    from tcenv.backends.metal.search_candidate import confirmation_input

    locator, _, _, _ = _p1(tmp_path)
    for version in (1, 2):
        with pytest.raises(UnsupportedContractCapability, match="MSL portfolio"):
            confirmation_input(
                locator,
                "import tilelang\n",
                target=TARGET,
                guidance_digest="0" * 64,
                master_seed=b"0123456789abcdef",
                track_id=TILELANG,
                track_version=version,
            )


def test_a_forged_resume_of_the_interface_only_dsl_track_is_refused(tmp_path):
    """@brief resume 按记录版本精确恢复，但新进程仍拒绝 interface_only 的 DSL Track。"""
    import json

    from taic.cli import ResumeRefused, _recorded_track
    from taic.orchestrator import _resolve_run_track
    from taic.task_binding import bind_task

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "run_config.json").write_text(
        json.dumps(
            {
                "task_mode": "v2",
                "track": {"track_id": TILELANG, "track_version": 1, "delivery": "dsl_source_pipeline"},
            }
        ),
        encoding="utf-8",
    )
    # The record is restored by its exact version (frozen reconstruction), not silently upgraded.
    assert _recorded_track(run_dir) == TILELANG
    with pytest.raises(ResumeRefused, match="does not match"):
        _recorded_track(run_dir, requested="msl-single")
    # A recorded version the registry does not carry is a typed refusal, never a silent upgrade.
    (run_dir / "run_config.json").write_text(
        json.dumps(
            {
                "task_mode": "v2",
                "track": {"track_id": TILELANG, "track_version": 4, "delivery": "dsl_source_pipeline"},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ResumeRefused, match="not registered at version 4"):
        _recorded_track(run_dir)
    # Availability is still enforced before any session starts.
    from taic.orchestrator import RunStartError

    with pytest.raises(RunStartError, match="interface_only"):
        _resolve_run_track(bind_task("gemm", "fp16", "small"), TILELANG, track_version=1)


# --------------------------------------------------------------------- frozen confirmation chain


def _two_version_registry():
    """@brief 登记的 ``msl-single`` 同时有可选的 1 与 2（默认解析确实是 2）。"""
    from tcenv.track.registry import TrackRegistry

    live = load_track_registry()
    upgraded = TrackRegistry(
        backend_id="metal",
        tracks=(
            live.get("msl-single"),
            dataclasses.replace(live.get("msl-single"), track_version=2),
            live.get("msl-bundle"),
            live.get(TILELANG, version=1),
        ),
    )
    assert upgraded.require_available("msl-single").track_version == 2
    return upgraded


def _p1_freeze(tmp_path):
    """@brief P2 激活链需要的 (locator, freeze, portfolio, validation)，全部 CPU。"""
    from tcenv.identity import fingerprint
    from tcenv.portfolio.candidate import parse_portfolio
    from tcenv.portfolio_validation.validation import PortfolioFreeze, ValidationPlan
    from tests.support.shape_portfolio_metal import parent_entries, portfolio_document

    locator, _contract, plan, authority = _p1(tmp_path)
    portfolio = parse_portfolio(parent_entries(portfolio_document(plan.digest)), domain_plan=plan)
    freeze = PortfolioFreeze(
        protocol_epoch=1,
        authority=authority,
        candidate_entries=portfolio.entries,
        task_pack_digest=locator.deployment_pack_digest,
        guidance_digest=fingerprint({"guidance": "MT1b frozen confirmation chain"}),
    )
    required = tuple(
        f"fp16/{size}/{usage}"
        for size in ("below", "threshold", "above", "tail", "hole", "boundary")
        for usage in ("golden", "benchmark")
    )
    validation = ValidationPlan.derive(
        freeze,
        tune_shards=("fp16/large/golden", "fp16/large/benchmark"),
        unseen_count=len(required),
        master_seed=b"0123456789abcdef",
        required_instance_keys=required,
    )
    return locator, freeze, portfolio, validation


class _RecordedCall(Exception):
    """@brief 让被 spy 的生产调用在记录 kwargs 后立刻停下。"""


def test_a_v1_run_confirms_and_activates_through_the_same_frozen_version(tmp_path, monkeypatch):
    """@brief v1 记录 + 可选 v2 已登记：fresh confirmation 与 activation 全程钉在 1。

    这条链此前只在 resume 启动与局部 expected 重建上被覆盖；``confirm_candidate``、
    ``confirm_subject`` 的实际执行、激活重投影都能从 live registry 重选，于是一个冻结在 1 的旧 run
    会在最终确认/激活时静默换到 2。这里用 spy 逐跳断言收到的版本，并证明默认解析确实是 2。
    """
    import asyncio
    import json

    import taic.portfolio_confirmation as document_module
    import taic.portfolio_manifest as manifest_module
    import tcenv.portfolio.candidate as candidate_module
    import tcenv.track.registry as registry_module
    from taic.cli import _recorded_track_selection
    from taic.llm import LLMConfig, SearchConfig
    from taic.orchestrator import Orchestrator
    from tcenv.backends.metal import portfolio_confirmation as confirmation_module

    upgraded = _two_version_registry()
    monkeypatch.setattr(registry_module, "load_track_registry", lambda path=None: upgraded)
    monkeypatch.setattr(candidate_module, "load_track_registry", lambda path=None: upgraded)

    # A recorded v1 selection is restored by its exact version, not by the registry default (2).
    recorded_run = tmp_path / "recorded-run"
    recorded_run.mkdir()
    (recorded_run / "run_config.json").write_text(
        json.dumps(
            {
                "task_mode": "v2",
                "track": {"track_id": "msl-single", "track_version": 1, "delivery": "single_source"},
            }
        ),
        encoding="utf-8",
    )
    recorded_id, recorded_version = _recorded_track_selection(recorded_run)
    assert (recorded_id, recorded_version) == ("msl-single", 1)

    llm = LLMConfig(
        base_url="http://x", api_key="sk-test-0123456789abcdef", model="fake",
        reasoning_effort="off", use_tools=False, truncation_retry=False,
    )
    off = dict(
        stop_at_roofline=False, abi_guard=False, target_note=False, rich_history=False,
        status_lines=False, plan_ledger=False,
    )
    search_cfg = SearchConfig(
        budget_usd=5.0, branches=1, max_rounds=0, eval_reps=3, kernel_agent="pi",
        correctness_only=False, **off,
    )
    orchestrator = Orchestrator(
        "gemm", "fp16", "small", llm, search_cfg,
        run_dir=tmp_path / "run",
        log=lambda _message: None,
        track_id=recorded_id,
        track_version=recorded_version,
    )
    assert (orchestrator.binding.track_id, orchestrator.binding.track_version) == ("msl-single", 1)
    assert orchestrator.track.track_version == 1

    # (1) The final v2 confirmation hands the frozen selection to ``confirm_candidate``.
    confirm_kwargs: dict[str, object] = {}

    def fake_confirm(_locator, _source, **kwargs):
        confirm_kwargs.update(kwargs)
        return {
            "instance_key": "fp16/small/benchmark",
            "status": "rejected",
            "reason": "not_evaluated",
            "confirmation_report": {},
            "identity": {},
        }

    real_confirm = search.confirm_candidate
    monkeypatch.setattr(search, "confirm_candidate", fake_confirm)
    orchestrator.best = SimpleNamespace(label="c-1", speedup=1.2, source="kernel void k() {}")
    result = SimpleNamespace(
        confirmations=[], confirmed_status=None, confirmed_speedup=None, confirmed_values=[],
        candidate_us=None, baseline_us={},
    )
    asyncio.run(orchestrator._confirm_v2(result))
    assert confirm_kwargs["track_id"] == "msl-single"
    assert confirm_kwargs["track_version"] == 1
    assert result.confirmations and result.confirmations[0]["status"] == "not_evaluated"

    # (2) The real wrapper threads it into the two real executions.
    run_kwargs: dict[str, object] = {}

    def fake_run(**kwargs):
        run_kwargs.update(kwargs)
        raise _RecordedCall

    monkeypatch.setattr(confirmation_module, "run_portfolio_task", fake_run)
    with pytest.raises(_RecordedCall):
        real_confirm(
            orchestrator.locator,
            orchestrator.best.source,
            target=orchestrator.target,
            guidance_digest=orchestrator._confirmation_guidance_digest,
            master_seed=b"k" * 32,
            track_id=orchestrator.binding.track_id,
            track_version=orchestrator.binding.track_version,
        )
    assert run_kwargs["track_version"] == 1
    assert run_kwargs["target_id"] == "apple-m4-max"

    # (3) The activation expectation is rebuilt with the same frozen version, and the *unpinned*
    #     call really resolves the registry default (2) — so the assertion above is not vacuous.
    locator, freeze, _portfolio, validation = _p1_freeze(tmp_path)
    document = {"instance_key": "fp16/above/benchmark"}
    projection_kwargs: dict[str, object] = {}
    hook_kwargs: dict[str, object] = {}
    real_project = confirmation_module.project_provider
    real_hook = confirmation_module._preflight_hook

    def spy_project(portfolio, provider, **kwargs):
        projection_kwargs.update(kwargs)
        return real_project(portfolio, provider, **kwargs)

    def spy_hook(*args, **kwargs):
        hook_kwargs.update(kwargs)
        return real_hook(*args, **kwargs)

    monkeypatch.setattr(confirmation_module, "project_provider", spy_project)
    monkeypatch.setattr(confirmation_module, "_preflight_hook", spy_hook)
    record, expected = confirmation_module.activation_expectation(
        locator=locator, freeze=freeze, validation=validation, document=document, track_version=1,
    )
    assert projection_kwargs["track_version"] == 1
    assert hook_kwargs["track_version"] == 1
    projection_kwargs.clear()
    hook_kwargs.clear()
    confirmation_module.activation_expectation(
        locator=locator, freeze=freeze, validation=validation, document=document,
    )
    assert projection_kwargs["track_version"] is None
    assert hook_kwargs["track_version"] is None

    # (4) Manifest activation threads it into that rebuild.
    activation_kwargs: dict[str, object] = {}

    def fake_activation(**kwargs):
        activation_kwargs.update(kwargs)
        raise _RecordedCall

    monkeypatch.setattr(confirmation_module, "activation_expectation", fake_activation)
    with pytest.raises(_RecordedCall):
        manifest_module.DispatchManifest.from_confirmations(
            freeze,
            validation,
            [document],
            {document["instance_key"]: (record, expected)},
            locator=locator,
            track_version=1,
        )
    assert activation_kwargs["track_version"] == 1

    # (5) ... and into the activation document's own re-projection.
    verified = {
        "provider": record.provider,
        "variant_id": record.variant_id,
        "fallback_id": record.fallback_id,
        "credits_variant": record.provider == "candidate",
        "instance_key": document["instance_key"],
        "confirmation_seed": document_module.confirmation_seed_for(validation, document["instance_key"]),
        "expected": {"target_id": TARGET.id},
        "search_report": {"target_id": TARGET.id},
        "confirmation_report": {"target_id": TARGET.id},
        "candidate_sha256": "0" * 64,
    }
    document_projection: dict[str, object] = {}

    def stop_project(portfolio, provider, **kwargs):
        document_projection.update(kwargs)
        raise _RecordedCall

    monkeypatch.setattr(document_module, "verify_subject_document", lambda _document: verified)
    monkeypatch.setattr(document_module, "project_provider", stop_project)
    with pytest.raises(_RecordedCall):
        document_module.validate_confirmation_for_activation(
            {},
            freeze,
            validation,
            expected_record=record,
            expected={},
            track_version=1,
        )
    assert document_projection["track_version"] == 1

    # (6) The MT5 scheme-A seed deriver must derive from the registered version too: deriving
    #     with the registry default would change an old run's confirmation seeds.
    derive_input_kwargs: dict[str, object] = {}

    def fake_confirmation_input(_locator, _source, **kwargs):
        derive_input_kwargs.update(kwargs)
        raise _RecordedCall

    monkeypatch.setattr(search, "confirmation_input", fake_confirmation_input)
    derive = adapter_for_track(upgraded.get("msl-single", version=1)).resolve("confirmation")
    with pytest.raises(_RecordedCall):
        derive(
            orchestrator.locator,
            "kernel void k() {}",
            target=TARGET,
            guidance_digest="a" * 64,
            master_seed=b"k" * 32,
            instance_key="fp16/small/benchmark",
        )
    assert derive_input_kwargs["track_id"] == "msl-single"
    assert derive_input_kwargs["track_version"] == 1
