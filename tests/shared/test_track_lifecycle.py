"""Hardware-neutral prepare/run lifecycle: host/device split, timing and identity (A2, CPU).

The lifecycle adapts the upstream ``prepare``/``run`` idiom without importing any backend: the
host side may only produce geometry/resources/launch arguments, an input-dependent host step is
billed to the measured run, a reusable step is bound to a computed cache identity (candidate +
target + instance + step), and the device side is exactly the active stages of the resolved
geometry.  The description is versioned and hashed, so prepare/run behaviour enters the build
subject instead of living in prose.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tcenv.contract_errors import GeometryError
from tcenv.target import DEFAULT_TARGET_ID, load_target
from tcenv.taskcontract import load_task_contract
from tcenv.track import (
    CandidateArtifact,
    CandidateEntry,
    CandidateLifecycle,
    ExecutionGeometry,
    GeometryStage,
    HostStep,
    PrepareDeclaration,
    TargetAbiLimits,
    cache_key,
    derive_tensor_memory_contract,
    resolve_geometry,
)

from ..support.v2_tasks import write_task

LIMITS = TargetAbiLimits(max_buffer_index=30, max_threads_per_threadgroup=1024, max_threadgroup_memory_bytes=32768)
CANDIDATE_SHA = "a" * 64


def _geometry(tmp_path: Path, *, source: str = "candidate"):
    """@brief 构造 ``(contract, resolved geometry)``。

    @param tmp_path 临时目录。
    @param source 几何来源。
    @return ``(contract, resolved geometry)``。
    """
    task = load_task_contract(write_task(tmp_path / source))
    abi = derive_tensor_memory_contract(task.instance, target=load_target(DEFAULT_TARGET_ID))
    geometry = resolve_geometry(
        ExecutionGeometry(
            source=source,
            stages=(
                GeometryStage(kernel="fill", grid=({"ceil_div": ["M*H", 256]}, 1, 1), threads=(256, 1, 1)),
                GeometryStage(
                    kernel="reduce",
                    grid=(1, 1, 1),
                    threads=(64, 1, 1),
                    when=("gt", "M", 10**9),
                ),
            ),
        ),
        instance=task.instance,
        contract=abi,
        limits=LIMITS,
    )
    return task, geometry


def _lifecycle(
    tmp_path: Path,
    *,
    steps=None,
    candidate_sha: str = CANDIDATE_SHA,
    instance_key: str | None = None,
    track_id: str = "msl-bundle",
):
    """@brief 构造一个生命周期。

    @param tmp_path 临时目录。
    @param steps host 步骤；省略时为两段（harness + host_tiling）。
    @param candidate_sha 候选身份。
    @param instance_key 实例身份；省略时用任务合同摘要。
    @param track_id track。
    @return ``(lifecycle, contract)``。
    """
    task, geometry = _geometry(tmp_path)
    host_steps = steps
    if host_steps is None:
        host_steps = (
            HostStep(PrepareDeclaration(name="compile", input_dependent=False), role="harness"),
            HostStep(PrepareDeclaration(name="tiling", input_dependent=False), role="host_tiling"),
        )
    lifecycle = CandidateLifecycle(
        track_id=track_id,
        geometry=geometry,
        host_steps=tuple(host_steps),
        candidate_sha256=candidate_sha,
        target_id=DEFAULT_TARGET_ID,
        instance_key=instance_key or task.fingerprint(),
    )
    return lifecycle, task


def _candidate(*, host_tiling: bool = True) -> CandidateArtifact:
    """@brief 构造一个 ``msl-bundle`` 候选清单（含或不含 Host Tiling entry）。

    @param host_tiling 是否带 Host Tiling entry。
    @return 候选。
    """
    entries = [CandidateEntry(path="kernel.metal", role="device_kernel", media_type="text/plain", data=b"k\n")]
    if host_tiling:
        entries.append(CandidateEntry(path="host/tiling.metal", role="host_tiling", media_type="text/plain", data=b"t\n"))
    return CandidateArtifact(track_id="msl-bundle", entries=tuple(entries))


def test_the_lifecycle_splits_host_prepare_from_device_run(tmp_path):
    """@brief prepare 全在 host；run 恰好是已求值几何的激活 stage，按顺序。"""
    lifecycle, _ = _lifecycle(tmp_path)
    document = lifecycle.document()
    assert {step["role"] for step in document["prepare"]} == {"harness", "host_tiling"}
    assert [stage["kernel"] for stage in document["run"]] == ["fill"]
    assert [stage.index for stage in lifecycle.device_stages] == [0]
    assert document["run"][0]["grid"] == list(lifecycle.geometry.stages[0].grid)
    assert document["geometry"]["source"] == "candidate"
    assert document["used_fallback"] is False
    with pytest.raises(GeometryError, match="host roles are"):
        HostStep(PrepareDeclaration(name="kernel", input_dependent=False), role="device_kernel")
    with pytest.raises(GeometryError, match="duplicate host prepare step"):
        CandidateLifecycle(
            track_id="msl-bundle",
            geometry=lifecycle.geometry,
            host_steps=(
                HostStep(PrepareDeclaration(name="same", input_dependent=False)),
                HostStep(PrepareDeclaration(name="same", input_dependent=False)),
            ),
            candidate_sha256=CANDIDATE_SHA,
            target_id=DEFAULT_TARGET_ID,
            instance_key="instance",
        )


def test_a_reusable_step_gets_a_computed_cache_identity_and_a_hand_written_one_is_refused(tmp_path):
    """@brief 输入无关可复用步骤绑定 candidate+target+instance+step 的缓存身份，不手写。"""
    cached = HostStep(
        PrepareDeclaration(name="compile", input_dependent=False, cached=True),
        role="harness",
    )
    lifecycle, _ = _lifecycle(tmp_path, steps=(cached,))
    document = lifecycle.document()
    expected = cache_key(lifecycle.cache_scope, "compile")
    assert document["prepare"][0]["cache_identity"] == expected
    assert lifecycle.cache_scope == {
        "candidate_sha256": CANDIDATE_SHA,
        "target_id": DEFAULT_TARGET_ID,
        "instance_key": lifecycle.instance_key,
    }
    # The identity is scope-sensitive: each of the three parts changes the key.
    assert cache_key({**lifecycle.cache_scope, "candidate_sha256": "b" * 64}, "compile") != expected
    assert cache_key({**lifecycle.cache_scope, "target_id": "other"}, "compile") != expected
    assert cache_key({**lifecycle.cache_scope, "instance_key": "other"}, "compile") != expected
    assert cache_key(lifecycle.cache_scope, "other") != expected

    uncached = _lifecycle(
        tmp_path / "uncached", steps=(HostStep(PrepareDeclaration(name="compile", input_dependent=False)),)
    )[0]
    assert uncached.document()["prepare"][0]["cache_identity"] is None

    with pytest.raises(GeometryError, match="hand-written cache_identity"):
        HostStep(
            PrepareDeclaration(name="compile", input_dependent=False, cached=True, cache_identity="my-own-key"),
            role="harness",
        )


def test_input_dependent_prepare_must_be_timed_inside_the_lifecycle(tmp_path):
    """@brief 输入相关 prepare 必须计时；计时外一律拒绝（包括带缓存的）。"""
    timed = HostStep(
        PrepareDeclaration(name="bind", input_dependent=True, inside_timing=True),
        role="harness",
    )
    lifecycle, _ = _lifecycle(tmp_path, steps=(timed,))
    assert lifecycle.document()["prepare"][0]["inside_timing"] is True

    for declaration in (
        PrepareDeclaration(name="hidden", input_dependent=True),
        # Caching does not make input-dependent work input-independent: the lifecycle computes
        # the identity, and the timing rule still refuses the step.
        PrepareDeclaration(name="hidden_cache", input_dependent=True, cached=True),
        PrepareDeclaration(name="preprocess", input_dependent=True, preprocesses_inputs=True),
        PrepareDeclaration(name="oracle", input_dependent=False, computes_reference=True),
    ):
        with pytest.raises(GeometryError):
            _lifecycle(tmp_path / declaration.name, steps=(HostStep(declaration, role="harness"),))


def test_the_lifecycle_and_the_candidate_must_agree_on_host_work(tmp_path):
    """@brief 候选自带 Host Tiling 就必须执行；生命周期声明 host role 就必须有对应 entry。"""
    artifact = _candidate()
    lifecycle, _ = _lifecycle(tmp_path, candidate_sha=artifact.candidate_sha256)
    assert lifecycle.validate_for(artifact) == "accepted"

    no_step, _ = _lifecycle(
        tmp_path / "no_step",
        steps=(HostStep(PrepareDeclaration(name="compile", input_dependent=False)),),
        candidate_sha=artifact.candidate_sha256,
    )
    with pytest.raises(GeometryError, match="no 'host_tiling' prepare step"):
        no_step.validate_for(artifact)

    device_only = _candidate(host_tiling=False)
    device_lifecycle, _ = _lifecycle(tmp_path / "device_only", candidate_sha=device_only.candidate_sha256)
    with pytest.raises(GeometryError, match="no 'host_tiling' entry"):
        device_lifecycle.validate_for(device_only)

    with pytest.raises(GeometryError, match="track"):
        lifecycle.validate_for(CandidateArtifact(track_id="msl-single", entries=device_only.entries))
    with pytest.raises(GeometryError, match="candidate_sha256"):
        lifecycle.validate_for(_candidate(host_tiling=False))


def test_the_lifecycle_identity_covers_prepare_run_and_identity_fields(tmp_path):
    """@brief 同输入同身份；prepare/run/身份字段变化都改变生命周期摘要。"""
    base, _ = _lifecycle(tmp_path / "base")
    again, _ = _lifecycle(tmp_path / "again")
    assert base.identity == again.identity

    changed_prepare, _ = _lifecycle(
        tmp_path / "prepare",
        steps=(
            HostStep(PrepareDeclaration(name="compile", input_dependent=False), role="harness"),
            HostStep(PrepareDeclaration(name="tiling", input_dependent=False, inside_timing=True), role="host_tiling"),
        ),
    )
    assert changed_prepare.identity != base.identity

    changed_candidate, _ = _lifecycle(tmp_path / "candidate", candidate_sha="b" * 64)
    assert changed_candidate.identity != base.identity
    changed_instance, _ = _lifecycle(tmp_path / "instance", instance_key="another-instance")
    assert changed_instance.identity != base.identity

    task, geometry = _geometry(tmp_path / "fallback", source="fallback")
    fallback_lifecycle = CandidateLifecycle(
        track_id="msl-bundle",
        geometry=geometry,
        host_steps=base.host_steps,
        candidate_sha256=CANDIDATE_SHA,
        target_id=DEFAULT_TARGET_ID,
        instance_key=base.instance_key,
    )
    assert fallback_lifecycle.used_fallback is True
    assert fallback_lifecycle.identity != base.identity
    assert fallback_lifecycle.document()["geometry"]["source"] == "fallback"
    assert task.op_name == "rmsnorm"
