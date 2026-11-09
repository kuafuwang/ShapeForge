"""A5: the 16 G0 migration candidates on the v2 standard entry.

The CPU half is the G0 comparison (every ``(task, dtype, size, variant)`` combination the frozen
corpus recorded: inputs, reference outputs, error scales, cases/sizes/tolerances/attrs/work,
baseline identity and pack closure) plus the retirement/dispatch negatives.  The GPU half runs
real ``evaluate_task`` slices for a representative set of migrated operators, including the
tiled GEMM and the task that owns a module ``error_scale``.

This file is fixed evidence for A5: ``A5/verify.py`` runs these node ids.  It never calls a
model and never uses a candidate that a model wrote: the candidate body is the G0 archived
``naive_msl`` text.
"""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

import pytest
# 生产装配（拒绝已迁任务）必须绕过测试的 specimen 夹具。
from tests.support.v1_tasks import production_get_task as get_task

ROOT = Path(__file__).resolve().parents[3]
A5 = ROOT / "tests/support/golden/shared/跨硬件算子生成分工-阶段验收/A5"
gpu = pytest.mark.gpu

TASKS: tuple[str, ...] = (
    "sigmoid",
    "relu",
    "silu",
    "swiglu",
    "rmsnorm",
    "softmax",
    "reduction_sum",
    "gemm",
    "gemm_silu",
    "sem_gemm_neg",
    "sem_relu_gelu_comment",
    "sem_gelu_named_relu",
    "gen_relu",
    "gen_sigmoid",
    "gen_silu",
    "gen_swiglu",
)


def _load(name: str):
    """@brief 载入 A5 验收工具模块（不是生产模块）。

    @param name 模块文件基名。
    @return 模块对象。
    """
    import sys

    path = A5 / f"{name}.py"
    module_name = f"_a5_{name}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    from tests.support.boundaries import rebind_acceptance_root

    rebind_acceptance_root(module)
    return module


G0 = _load("g0_compare")
PERF = _load("perf")
INDEX = G0.index()
COMBOS = G0.combos(INDEX)
COMBO_IDS = [G0.combo_id(combo) for combo in COMBOS]


def _record(name: str, payload: dict) -> None:
    """@brief A5 verify sets ``A5_EVIDENCE_DIR``; a real slice archives one record."""
    directory = os.environ.get("A5_EVIDENCE_DIR")
    if not directory:
        return
    Path(directory).mkdir(parents=True, exist_ok=True)
    (Path(directory) / name).write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _locator(task: str, *, dtype: str = "fp16", size: str = "small"):
    """@brief 为指定实例构造 locator（非默认实例也必须是同一份实例）。"""
    from tcenv.taskcontract import load_task_contract
    from tcenv.taskcontract.locator import TaskLocator
    from tcenv.taskcontract.pack import task_pack_closure
    from tcenv.track import REPO_ROOT

    directory = ROOT / "tasks" / task
    contract = load_task_contract(directory, dtype=dtype, size=size)
    return (
        TaskLocator(
            task_id=task,
            op_name=contract.op_name,
            dtype=contract.dtype,
            size=contract.size,
            pack_digest=contract.pack_digest,
            deployment_pack_digest=task_pack_closure(directory, root=REPO_ROOT).digest,
            task_dir=directory,
            code_root=REPO_ROOT,
        ),
        contract,
    )


def _frozen_candidate(task: str, *, dtype: str = "fp16", size: str = "small") -> tuple[str, dict]:
    """@brief G0 归档的 ``naive_msl`` 正文 + 候选自有的启动常量 + ``taic-launch``。"""
    from tcenv.taskcontract import load_task_contract

    combo = PERF.combo_of(INDEX, task, dtype=dtype, size=size)
    contract = load_task_contract(ROOT / "tasks" / task, dtype=dtype, size=size)
    return PERF.frozen_candidate(combo, phase="v2", contract=contract), combo


def _observe(result) -> list[tuple]:
    """@brief runtime 记录的真实 dispatch（不是候选源码重解析）。"""
    assert result.execution is not None
    seen: list[tuple] = []
    for record in result.execution["dispatch"]:
        item = (
            record["kernel"],
            tuple(record["grid"]),
            tuple(record["threads"]),
            int(record["threadgroup_bytes"]),
            tuple(record["workspace"]),
        )
        if item not in seen:
            seen.append(item)
    return seen


# ---------------------------------------------------------------------------------------------
# CPU: the G0 comparison for every recorded combination
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("combo", COMBOS, ids=COMBO_IDS)
def test_a5_g0_combo_matches_the_frozen_corpus(combo):
    """@brief 每个 G0 组合的语义/输入/reference/scale/baseline/pack 与冻结语料一致。"""
    result = G0.compare_combo(combo)
    assert result.records == len(combo["case_records"])
    assert result.exact_input_bytes >= result.records
    assert result.exact_reference_bytes == result.records
    assert result.exact_scale_bytes == result.records


def test_a5_covers_exactly_the_frozen_inventory():
    """@brief 覆盖 16 题/82 组合，且每题的 dtype/size/variant 集合与 G0 相同。"""
    assert len(COMBOS) == 82
    assert {combo["task"] for combo in COMBOS} == set(TASKS)
    seen = {(combo["task"], combo["dtype"], combo["size"], str(combo["variant"])) for combo in COMBOS}
    assert len(seen) == 82
    gemm_variants = {combo["variant"] for combo in COMBOS if combo["task"] == "gemm"}
    assert gemm_variants == {None, "128x64x32t256", "64x64x32t256", "32x32x16t128", "128x128x32t512"}


def test_a5_the_shipped_standard_dirs_are_v2_and_carry_no_v1_assembly():
    """@brief 16 个迁移目录只有标准 entry，v1 装配文件/pycache 不再存在。"""
    from tcenv.taskcontract import inspect_standard_entry

    for task in TASKS:
        directory = ROOT / "tasks" / task
        entry = inspect_standard_entry(directory)
        assert entry.is_v2, task
        assert entry.files[:2] == ("opspec.json", "reference.py"), task
        for marker in ("task.py", "track.json", "kernel_triton.py", "kernel_triton_msl.py", "naive.metal", "__init__.py"):
            assert not (directory / marker).exists(), f"{task}: {marker}"


def test_a5_the_out_of_scope_v1_tasks_keep_their_v1_assembly():
    """@brief 当前仍保留的 v1 题仍是 v1（含 track.json/task.py），不被迁移批次触碰。

    G0 的 ``preserved_v1_out_of_scope`` 是冻结证据，B1 又迁走了 gemv/rope；这里读的是由
    各批次边界派生的**当前**保留集，而不是那份历史名单。
    """
    from tcenv.taskcontract import inspect_standard_entry
    from tests.support.boundaries import preserved_v1

    preserved = [name for name in preserved_v1() if not name.startswith("sem_int ")]

    for task in preserved:
        entry = inspect_standard_entry(ROOT / "tasks" / task)
        assert not entry.is_v2 and entry.mode == "v1", task
        for name in ("track.json", "task.py", "kernel_triton.py"):
            assert (ROOT / "tasks" / task / name).is_file(), f"{task}: {name}"
        assert get_task(task, "fp16", "small").name == task
    # B4 已把最后的静态题与 sem_int 动态族全部迁到 v2：保留集为空，v1 装配不再有任何活题。
    # 若后续还有保留题，上面的循环会逐题校验其 v1 装配；sem_int_5 此刻必须被 v1 注册表拒绝。
    if not preserved:
        from tcenv.contract_errors import TaskContractError

        with pytest.raises(TaskContractError):
            get_task("sem_int_5", "fp16", "small")


def test_a5_the_v1_registry_refuses_a_v2_task_instead_of_falling_back():
    """@brief get_task 在 import 前按 spec_version 拒绝 v2，错误可读、不回退。"""
    from tcenv.contract_errors import TaskContractError

    for name in TASKS + ("gemm@128x64x32t256",):
        with pytest.raises(TaskContractError) as error:
            get_task(name, "fp16", "small")
        assert "v2 standard entry" in str(error.value)


def test_a5_a_mixed_directory_is_an_assembly_error(tmp_path):
    """@brief v2 目录里残留任一 v1 标记 → 装配失败（不做“能读就先读”）。"""
    import shutil

    from tcenv.contract_errors import AssemblyError
    from tcenv.taskcontract import inspect_standard_entry

    source = ROOT / "tasks" / "sigmoid"
    for marker in ("track.json", "task.py", "kernel_triton.py", "naive.metal"):
        directory = tmp_path / marker.replace(".", "_")
        directory.mkdir()
        for name in ("opspec.json", "reference.py"):
            shutil.copy(source / name, directory / name)
        (directory / marker).write_text("{}\n" if marker.endswith(".json") else "# retired\n")
        with pytest.raises(AssemblyError) as error:
            inspect_standard_entry(directory)
        assert "mixed v1/v2" in str(error.value)


def test_a5_a_v2_task_without_a_reference_is_refused(tmp_path):
    """@brief v2 目录缺 reference.py → 装配失败（v2 不回退 v1）。"""
    import shutil

    from tcenv.contract_errors import AssemblyError
    from tcenv.taskcontract import inspect_standard_entry

    directory = tmp_path / "sigmoid"
    directory.mkdir()
    shutil.copy(ROOT / "tasks" / "sigmoid" / "opspec.json", directory / "opspec.json")
    with pytest.raises(AssemblyError) as error:
        inspect_standard_entry(directory)
    assert "requires reference.py" in str(error.value)


def test_a5_a_v2_task_refuses_a_launch_variant():
    """@brief v2 没有 @variant 装配：登记变体文本被拒绝，不是“同名不同 launch”的第二份合同。"""
    import pytest as _pytest

    from tcenv.contract_errors import UnsupportedContractCapability
    from tcenv.taskcontract import load_task_contract

    with _pytest.raises(UnsupportedContractCapability) as error:
        load_task_contract(ROOT / "tasks" / "gemm", dtype="fp16", size="small", variant="128x64x32t256")
    assert "@variant" in str(error.value)
    # The five G0 gemm combinations differ only by the retired launch, so the v2 contract of the
    # two instances compared here is one contract, not five.
    one = load_task_contract(ROOT / "tasks" / "gemm", dtype="fp16", size="small").fingerprint()
    again = load_task_contract(ROOT / "tasks" / "gemm", dtype="fp16", size="small").fingerprint()
    assert one == again


def test_a5_the_reference_boundary_holds_for_the_migrated_tasks():
    """@brief 迁移题的 reference 不 import harness/候选，也不带文件/进程原语。"""
    from tcenv.taskcontract import load_task_contract

    for task in TASKS:
        contract = load_task_contract(ROOT / "tasks" / task, dtype="fp16", size="small")
        source = contract.reference.display_source()
        assert "pytest" not in source and "importlib" not in source
        assert contract.reference.error_scale_kind == (
            "module" if task == "reduction_sum" else "none"
        ), task


def test_a5_the_sampler_block_is_identical_in_every_migrated_reference():
    """@brief 冻结的 G0 采样块在 16 个 reference 中逐字节相同（单一口径，无逐题漂移）。"""
    marker = "# ---- frozen G0 sampler block (byte-identical in every A5-migrated reference) ----------------"
    end = "# ---- end frozen block ----------------------------------------------------------------------"
    blocks = set()
    for task in TASKS:
        source = (ROOT / "tasks" / task / "reference.py").read_text(encoding="utf-8")
        start = source.index(marker)
        stop = source.index(end) + len(end)
        blocks.add(source[start:stop])
    assert len(blocks) == 1


def test_a5_the_frozen_candidate_geometry_is_the_g0_contract_not_a_fallback():
    """@brief 候选声明 G0 的 grid/threads，选择后的几何与实际一致；fallback/错 grid 可见。"""
    from tcenv.backends.metal.track_adapter import msl_single_geometry_resolver
    from tcenv.track import (
        CandidateArtifact,
        CandidateEntry,
        abi_limits_from_target,
        candidate_geometry,
        dispatch_parity,
        geometry_pair,
        select_geometry,
        selection_parity,
    )
    from tcenv.track.abi import derive_tensor_memory_contract
    from tcenv.target import DEFAULT_TARGET_ID, load_target
    from tcenv.taskcontract import load_task_contract

    contract = load_task_contract(ROOT / "tasks" / "gemm", dtype="fp16", size="paper")
    source, combo = _frozen_candidate("gemm", size="paper")
    candidate = CandidateArtifact(
        track_id="msl-single",
        entries=(CandidateEntry(path="kernel.metal", role="device_kernel", media_type="text/plain", data=source.encode()),),
    )
    geometry = msl_single_geometry_resolver(contract)(candidate)
    abi = derive_tensor_memory_contract(contract.instance, target=load_target(DEFAULT_TARGET_ID))
    pair = geometry_pair(
        geometry, None, instance=contract.instance, contract=abi, limits=abi_limits_from_target(load_target(DEFAULT_TARGET_ID))
    )
    chosen = select_geometry(pair)
    observed = [
        (stage.kernel, tuple(stage.grid), tuple(stage.threads), int(stage.threadgroup_bytes), tuple())
        for stage in chosen.geometry.stages
    ]
    expected_grid = tuple(int(v) for v in combo["contract"]["grid"])
    expected_threads = tuple(int(v) for v in combo["contract"]["threads"])
    assert observed[0][1] == expected_grid, observed
    assert observed[0][2] == expected_threads, observed
    assert dispatch_parity(chosen.geometry, observed) == []
    assert selection_parity(chosen, provider="candidate", observed=observed) == []
    assert dispatch_parity(chosen.geometry, [("gemm_kernel", (1, 1, 1), expected_threads, 0, ())]) != []
    assert selection_parity(chosen, provider="fallback", observed=observed) != []
    # The body is the G0 text; only the candidate-owned launch constants precede it.
    assert source.endswith(combo["naive_msl"])


def test_a5_the_baseline_identity_is_the_frozen_g0_registration(tmp_path):
    """@brief 迁移后的 baseline 身份 = G0 登记；缺 runner 的登记被拒绝。"""
    import shutil

    from tcenv.contract_errors import AssemblyError
    from tcenv.taskcontract import load_task_contract

    for task in TASKS:
        contract = load_task_contract(ROOT / "tasks" / task, dtype="fp16", size="small")
        combo = next(item for item in COMBOS if item["task"] == task and item["dtype"] == "fp16" and item["size"] == "small")
        assert [entry.document() for entry in contract.baselines] == combo["baselines"], task
        assert contract.correctness_only is (not combo["baselines"]), task
    # A registered baseline without its runner is an assembly error, not a silent empty set.
    directory = tmp_path / "sigmoid"
    shutil.copytree(ROOT / "tasks" / "sigmoid", directory)
    (directory / "baselines.py").unlink()
    with pytest.raises(AssemblyError) as error:
        load_task_contract(directory)
    assert "baselines.py is required" in str(error.value)


def test_a5_the_performance_gate_is_recomputed_not_trusted(monkeypatch):
    """@brief 性能门由记录重算：paper+small 都必须 <=1.10，且实例/任务集合固定。"""
    # summarize_gate also rewrites perf_gate.json. The recorded paths are part of the
    # frozen bytes; a directory move must not be written back into that file.
    archived = PERF.EVIDENCE / "perf_gate.json"
    original_write = Path.write_text

    def keep_archived(self, text, *args, **kwargs):
        if self.resolve() == archived.resolve():
            return len(text)
        return original_write(self, text, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", keep_archived)
    gate = PERF.summarize_gate()
    assert gate["approved_max_ratio"] == 1.10
    assert [item["size"] for item in gate["gate_instances"]] == ["paper", "small"]
    assert len(gate["gate_ratios"]) == 13 and len(gate["small_ratios"]) == 13
    assert len(gate["comparable_tasks"]) == 13
    assert gate["correctness_only_tasks"] == ["sem_gemm_neg", "sem_relu_gelu_comment", "sem_gelu_named_relu"]
    assert gate["gate_pass"] is True
    assert gate["gate_geomean"] <= 1.10 and gate["small_geomean"] <= 1.10
    for instance in gate["gate_instances"]:
        assert instance["pass"] is True, instance
        assert instance["rounds"] >= 3
        assert set(instance["round_geomeans"]) == {str(round_index) for round_index in range(1, instance["rounds"] + 1)}
        assert instance["round_pass"] is True, instance["round_geomeans"]
        assert len(instance["ratios"]) == 13
        assert set(instance["measured_baselines"]) == set(gate["comparable_tasks"])
        for task, baselines in instance["measured_baselines"].items():
            assert set(baselines) == {"mlx", "torch_mps"}, (task, baselines)
            for value in baselines.values():
                assert value["v1_us"] > 0 and value["v2_us"] > 0
        for task, item in gate["comparable_tasks"].items():
            # Full source bytes differ by the v2-owned constants; the frozen G0 body is one.
            assert item["g0_body_sha256"], task
            assert all(item["candidate_sha256"].values()), task
            assert item["gpu_family"] and item["device_name"], task
    assert gate["measurement"]["protocol_digest"] == gate["v2_phase"]["source"]["protocol_digest"]
    assert gate["v1_phase"]["source"]["protocol_digest"] == gate["v2_phase"]["source"]["protocol_digest"]
    # A drifted (20% slower) v2 record must fail the same metric on both instances.
    import math

    for key in ("gate_ratios", "small_ratios"):
        ratios = {task: value * 1.20 for task, value in gate[key].items()}
        drifted = math.exp(sum(math.log(value) for value in ratios.values()) / len(ratios))
        assert drifted > 1.10


def test_a5_a_missing_registered_baseline_is_an_explicit_evidence_error():
    """@brief 缺 baseline 的证据是明确拒绝；不能被折叠、跳过或变成 KeyError。"""
    import copy

    document = json.loads((A5 / "evidence/perf_v2_paper.json").read_text())
    record = next(item for item in document["records"] if item["baselines"])
    PERF.validate_record(record, phase="v2", size="paper", where="fixture")
    missing = copy.deepcopy(record)
    missing["baselines"].pop("mlx")
    with pytest.raises(PERF.EvidenceError) as error:
        PERF.validate_record(missing, phase="v2", size="paper", where="fixture")
    assert "mlx" in str(error.value) and "missing" in str(error.value)
    stale = copy.deepcopy(record)
    stale.pop("baseline_registrations")
    with pytest.raises(PERF.EvidenceError) as error:
        PERF.validate_record(stale, phase="v2", size="paper", where="fixture")
    assert "pre-interleaving" in str(error.value)
    zero = copy.deepcopy(record)
    zero["baselines"]["mlx"]["median_us"] = 0.0
    with pytest.raises(PERF.EvidenceError):
        PERF.validate_record(zero, phase="v2", size="paper", where="fixture")


def test_a5_an_invalid_task_name_is_a_lookup_miss_not_a_filesystem_error():
    """@brief 非法/超长题名不得被当成 tasks/ 下的路径去 stat（observe 长 run_id 回归）。

    A run id may carry an arbitrary task string.  ``get_task`` is the v1 *lookup*: a name that
    cannot be a single directory under ``tasks/`` (separator, ``.``/``..``, over ``NAME_MAX``)
    must answer "unknown task" instead of raising ``OSError(ENAMETOOLONG)`` or probing outside
    the tasks root.  The observe budget fixture that exposed this is
    ``tests/shared/test_observe_analysis.py::test_the_panels_keep_the_byte_budget_with_long_text_and_many_epochs``.
    """
    from tcenv.contract_errors import TaskContractError

    too_long = "task-" + "x" * 494
    for name in (too_long, f"gen_{too_long}", "../escape", "/etc/passwd", "..", ""):
        try:
            get_task(name, "fp16", "small")
        except (KeyError, TaskContractError):
            continue
        raise AssertionError(f"invalid task name {name!r} was assembled")
    # The specific failure that was reported is a clean lookup miss, not ENAMETOOLONG.
    with pytest.raises(KeyError):
        get_task(too_long, "fp16", "small")


def test_a5_the_frozen_cross_language_vectors_do_not_depend_on_a_live_task(tmp_path):
    """@brief 四个 R4/R6 冻结向量由自带身份重建，不经过已迁移题的 v1/v2 装配。"""
    import shutil

    from tcenv.task import Task
    from tests.support import remote_contract as rc

    # The report identity is frozen wire data, not a task object and not a second registry.
    assert not isinstance(rc.frozen_golden_task(), Task)
    assert json.loads(json.dumps(rc.build_blob_golden())) == rc.load_blob_golden()
    for name, builder in (
        ("npu_bundle_v4.json", rc.build_bundle_golden),
        ("npu_build_v5.json", rc.build_build_golden),
    ):
        frozen = json.loads((rc.SUPPORT / "golden" / "remote" / name).read_text(encoding="utf-8"))
        assert json.loads(json.dumps(builder())) == frozen, name
    # The R4 rebuild helper must reproduce the legacy vector, not a new one.
    target = tmp_path / "evaluator_v1.json"
    shutil.copy(rc.GOLDEN, target)
    rc.write_evaluator_golden(target)
    assert json.loads(target.read_text(encoding="utf-8")) == rc.load_evaluator_golden()


# ---------------------------------------------------------------------------------------------
# GPU: real migrated-path slices
# ---------------------------------------------------------------------------------------------


def _evaluate(task: str, *, size: str = "small", mode: str = "timed", seed: int = 7, reps: int = 3):
    """@brief 用冻结候选在 v2 真实路径上评测一个迁移题。"""
    from tcenv.backends.metal.track_adapter import MetalTrackBuilder, MetalTrackEvaluator, msl_single_geometry_resolver
    from tcenv.taskcontract.runner import evaluate_task

    locator, contract = _locator(task, size=size)
    source, combo = _frozen_candidate(task, size=size)
    result = evaluate_task(
        locator,
        track_id="msl-single",
        candidate_source=source,
        builder=MetalTrackBuilder(),
        evaluator=MetalTrackEvaluator(mode=mode, seed=seed, reps=reps, cases=("normal", "wide_range", "outlier")),
        geometry_resolver=msl_single_geometry_resolver(contract),
        track_registry_path=ROOT / "backends/metal/tracks.json",
    )
    return result, combo


def _assert_slice(task: str, result, combo: dict) -> dict:
    """@brief 断言真实切片：构建/ABI/正确性/计时/真实 dispatch 与 G0 契约一致。"""
    report = result.report
    assert report["status"] == "ok", report.get("message")
    assert report["abi"]["ok"] is True
    assert report["correctness"]["ok"] is True
    assert report["benchmark"]["candidate"]["median_us"] > 0
    assert result.effective_correctness_only is (not combo["baselines"])
    observed = _observe(result)
    assert observed == [
        (
            combo["contract"]["kernel_name"],
            tuple(int(v) for v in combo["contract"]["grid"]),
            tuple(int(v) for v in combo["contract"]["threads"]),
            0,
            (),
        )
    ], observed
    assert result.execution["provider"] == "candidate"
    record = {
        "task": task,
        "dtype": "fp16",
        "size": "small",
        "candidate_sha256": result.candidate_sha256,
        "geometry_identity": result.geometry_identity,
        "bundle_digest": result.bundle_digest,
        "artifact_set_digest": result.artifact_set_digest,
        "native_digest": result.native_digest,
        "track": result.track,
        "worker": result.execution["worker"],
        "dispatch": _observe(result),
        "provider": result.execution["provider"],
        "status": report["status"],
        "abi_ok": report["abi"]["ok"],
        "correctness_ok": report["correctness"]["ok"],
        "median_us": report["benchmark"]["candidate"]["median_us"],
        "presentation_sha256": result.presentation_sha256,
        "effective_correctness_only": result.effective_correctness_only,
        "g0_grid": combo["contract"]["grid"],
        "g0_threads": combo["contract"]["threads"],
    }
    _record(f"{task}.json", record)
    return record


@gpu
def test_a5_gpu_sigmoid_elementwise_real_slice():
    """@brief 逐元素 sigmoid：G0 候选正文 + 真实 v2 构建/正确性/计时/dispatch。"""
    result, combo = _evaluate("sigmoid")
    _assert_slice("sigmoid", result, combo)


@gpu
def test_a5_gpu_rmsnorm_two_inputs_and_an_attr_real_slice():
    """@brief RMSNorm：两输入 + EPS attr 的 v2 ABI 与真实切片。"""
    result, combo = _evaluate("rmsnorm")
    _assert_slice("rmsnorm", result, combo)


@gpu
def test_a5_gpu_gemm_tiled_real_slice():
    """@brief GEMM：候选自报 G0 的 tile/threads，真实 tiled 内核与 ABI。"""
    result, combo = _evaluate("gemm")
    _assert_slice("gemm", result, combo)


@gpu
def test_a5_gpu_reduction_sum_module_error_scale_real_slice():
    """@brief 行求和：模块 error_scale（sum|x|）是 v2 reference 钩子，真实切片通过。"""
    result, combo = _evaluate("reduction_sum")
    record = _assert_slice("reduction_sum", result, combo)
    from tcenv.taskcontract import load_task_contract

    contract = load_task_contract(ROOT / "tasks/reduction_sum", dtype="fp16", size="small")
    assert contract.reference.error_scale_kind == "module"
    assert record["status"] == "ok"


@gpu
def test_a5_gpu_gen_swiglu_two_inputs_real_slice():
    """@brief 生成族迁移题 gen_swiglu：两输入、baseline 身份来自 generated.elementwise。"""
    result, combo = _evaluate("gen_swiglu")
    record = _assert_slice("gen_swiglu", result, combo)
    from tcenv.taskcontract import load_task_contract

    contract = load_task_contract(ROOT / "tasks/gen_swiglu", dtype="fp16", size="small")
    identities = {entry.implementation_id for entry in contract.baselines}
    assert identities == {"generated.elementwise.baseline.mlx", "generated.elementwise.baseline.torch_mps"}
    assert record["effective_correctness_only"] is False


SIMDGROUP_GEMM_VARIANTS: tuple[str, ...] = ("32x32x32t128", "128x64x32t256", "64x64x32t64")


def _simdgroup_gemm_source(contract, variant: str) -> str:
    """@brief 三个手写 simdgroup GEMM 候选的 v2 源码：候选自有 tile 常量 + 自己的 launch。

    The v1 assembly supplied ``BLOCK_M/N/K`` and the launch through ``gemm@<variant>``.  On the
    v2 standard entry the *candidate* owns both: the constants are prepended here and the
    ``taic-launch`` block declares the same grid/threads the retired variant contract used
    (grid = N/BN, M/BM; threads = TH).  Nothing is re-derived from the task.

    @param contract v2 GEMM 合同（提供 M/N）。
    @param variant ``<BM>x<BN>x<BK>t<threads>``。
    @return 完整候选源码。
    """
    from experiments.contract_sensitivity import gemm_source, parse

    bm, bn, bk, threads = parse(variant)
    m, n = int(contract.instance.symbols["M"]), int(contract.instance.symbols["N"])
    assert m % bm == 0 and n % bn == 0, (variant, m, n)
    constants = "\n".join((
        f"constant constexpr uint BLOCK_M = {bm};",
        f"constant constexpr uint BLOCK_N = {bn};",
        f"constant constexpr uint BLOCK_K = {bk};",
    )) + "\n"
    launch = (
        "/* taic-launch\n"
        "math fast\n"
        f"launch gemm_kernel grid={n // bn},{m // bm},1 threads={threads},1,1\n"
        "*/\n"
    )
    return constants + launch + gemm_source(variant)


@gpu
@pytest.mark.parametrize("variant", SIMDGROUP_GEMM_VARIANTS)
def test_a5_gpu_hand_written_simdgroup_gemm_candidate(variant):
    """@brief 三个手写 simdgroup GEMM 候选在 v2 真实路径上 correctness/ABI/status 与自有 dispatch。

    This is the replacement for the deleted v1 body of
    ``tests/backends/metal/test_contracts.py::test_the_hand_written_simdgroup_gemm_is_correct_and_fits_every_listed_contract``.
    The candidate is not the G0 ``naive_msl`` body: it is the hand-written
    ``experiments.contract_sensitivity.gemm_source`` kernel, so it proves the migrated path still
    accepts a candidate-owned tile/launch instead of only the frozen archive candidate.
    """
    from experiments.contract_sensitivity import parse
    from tcenv.backends.metal.track_adapter import (
        MetalTrackBuilder, MetalTrackEvaluator, msl_single_geometry_resolver,
    )
    from tcenv.taskcontract.runner import evaluate_task

    bm, bn, bk, threads = parse(variant)
    locator, contract = _locator("gemm")
    m, n = int(contract.instance.symbols["M"]), int(contract.instance.symbols["N"])
    source = _simdgroup_gemm_source(contract, variant)
    result = evaluate_task(
        locator,
        track_id="msl-single",
        candidate_source=source,
        builder=MetalTrackBuilder(),
        evaluator=MetalTrackEvaluator(
            mode="timed", seed=7, reps=3, cases=tuple(contract.instance.case_domain.cases),
        ),
        geometry_resolver=msl_single_geometry_resolver(contract),
        track_registry_path=ROOT / "backends/metal/tracks.json",
    )
    report = result.report
    assert report["status"] == "ok", (variant, report.get("message"))
    assert report["abi"]["ok"] is True, variant
    assert report["correctness"]["ok"] is True, variant
    assert report["benchmark"]["candidate"]["median_us"] > 0, variant
    assert result.effective_correctness_only is False, variant
    assert result.execution["provider"] == "candidate", variant
    assert (result.track or {}).get("track_id") == "msl-single", variant
    assert _observe(result) == [
        ("gemm_kernel", (n // bn, m // bm, 1), (threads, 1, 1), 0, ()),
    ], (variant, _observe(result))
    # BLOCK_M/N/K are the candidate's, not the contract's: the archived G0 contract declares
    # none of them (only M/N/K), and a different tile still compiles and dispatches.
    assert "BLOCK_M" not in dict(contract.instance.symbols), variant
    assert "BLOCK_M" not in dict(contract.instance.resolved_attrs), variant
    assert f"constant constexpr uint BLOCK_M = {bm};" in source, variant


@gpu
def test_a5_gpu_sem_gelu_named_relu_is_correctness_only_real_slice():
    """@brief 无 baseline 的语义题：真实切片正确、effective_correctness_only，无 speedup 声明。"""
    result, combo = _evaluate("sem_gelu_named_relu")
    record = _assert_slice("sem_gelu_named_relu", result, combo)
    assert record["effective_correctness_only"] is True
    assert result.report.get("speedup") in (None, 0.0)


def test_a5_the_evidence_directory_matches_the_archived_records():
    """@brief ``A5_EVIDENCE_DIR`` 的 live 记录与归档逐字段相等（由 verify 设置）。"""
    directory = os.environ.get("A5_EVIDENCE_DIR")
    if not directory:
        pytest.skip("A5_EVIDENCE_DIR is only set by A5/verify.py")
    archived = A5 / "evidence" / "slices"
    assert archived.is_dir(), archived
    seen = sorted(path.name for path in archived.glob("*.json"))
    assert seen, "no archived A5 slices"
    for name in seen:
        live = json.loads((Path(directory) / name).read_text())
        expected = json.loads((archived / name).read_text())
        for key in ("candidate_sha256", "geometry_identity", "bundle_digest", "artifact_set_digest", "native_digest", "presentation_sha256"):
            assert live[key] == expected[key], f"{name}.{key}"
        assert live["dispatch"] == expected["dispatch"], name
        assert live["status"] == expected["status"] == "ok"
        # Timing is a fresh measurement of a 64 Ki-element slice: it must be the same regime, not an
        # exact repeat (the launch-bound small instance swings by 2x on a tiled kernel).  The
        # performance gate is `perf_gate.json` on the paper instance, not this slice.
        assert 0.2 <= live["median_us"] / expected["median_us"] <= 5.0, (name, live["median_us"], expected["median_us"])
