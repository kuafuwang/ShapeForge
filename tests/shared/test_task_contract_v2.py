"""The v2 task contract plane: standard entry, reference boundary, display and baselines.

CPU only.  Every fixture is written to ``tmp_path``; the shipped v1 tasks are read (never
modified) only to prove the dispatch and identity behaviour.
"""

from __future__ import annotations

import tempfile
import dataclasses
import json
import shutil
from pathlib import Path

import torch

import pytest

from tcenv.contract_errors import (
    AssemblyError,
    ReferenceBoundaryError,
    SemanticUnresolved,
    TaskContractError,
    UnsupportedContractCapability,
)
from tcenv.opspec import OpSpecError, load_opspec
from tcenv.taskcontract.display import display_from_document
from tcenv.taskcontract import (
    LegacyStandardTask,
    ReferenceRegistry,
    TaskContract,
    inspect_standard_entry,
    load_standard_task,
    load_task_contract,
    spec_text_override_allowed,
)

from ..support.v2_tasks import (
    RMSNORM_READ_WRITE_REFERENCE,
    RMSNORM_REFERENCE,
    SEM_REFERENCE,
    rmsnorm_opspec,
    rmsnorm_read_write_opspec,
    rmsnorm_workspace_opspec,
    sem_opspec,
    write_task,
)

ROOT = Path(__file__).resolve().parents[2]
# B4 之后仓库里没有 v1 任务目录了：v1 的 ``sem_attention_odd_scale``（含 opspec v1/track.json/
# kernel_triton.py）整体归档在 ``tests/support/golden/shared/v1-track``，本文件的分派测试以它为样本。
V1_TASK = Path(__file__).resolve().parents[1] / "support" / "golden" / "shared" / "v1-track" / "tasks" / "sem_attention_odd_scale"


def _task(tmp_path: Path, **kwargs) -> Path:
    """@brief 写一个默认 v2 RMSNorm 任务。

    @param tmp_path 临时目录。
    @param kwargs 传给 :func:`write_task` 的覆盖项。
    @return 任务目录。
    """
    return write_task(tmp_path, **kwargs)


def test_a_v1_task_dispatches_to_the_legacy_reader_without_reading_v2():
    """@brief v1 目录（spec_version 1 + track.json）经显式分派返回旧绑定，不被当成 v2。

    B4/B5 之后仓库里已经没有 v1 任务目录：分派逻辑仍按 ``spec_version`` 判定，所以这里用
    归档树里那份真实的 v1 目录，把它拷进一个临时 tasks 根后再装配；相对路径与 ``pack_files``
    都在该临时根内解析，不再依赖真实仓库里已删除的 v1 标记文件。
    """
    with tempfile.TemporaryDirectory() as td:
        tasks_root = Path(td) / "tasks"
        task_dir = tasks_root / "sem_attention_odd_scale"
        shutil.copytree(V1_TASK, task_dir, ignore=shutil.ignore_patterns("__pycache__"))
        # ``pack_files`` 是仓库相对路径；把它们在临时根里补齐，装配才能完成（真实仓库不再有它们）。
        for relative in json.loads((task_dir / "track.json").read_text(encoding="utf-8"))["pack_files"]:
            target = Path(td) / relative
            if target.exists():
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            source = V1_TASK.parents[2] / relative
            if source.is_file():
                shutil.copy2(source, target)
            else:
                target.write_text("", encoding="utf-8")
        monkeypatch = pytest.MonkeyPatch()
        try:
            monkeypatch.setattr("tcenv.track.REPO_ROOT", Path(td))
            result = load_standard_task(task_dir)
            assert isinstance(result, LegacyStandardTask)
            assert result.mode == "v1"
            assert result.spec.spec_version == 1
            assert result.entry.files[0] == "opspec.json"
            assert "track.json" in result.entry.files
            with pytest.raises(AssemblyError, match="v1"):
                load_task_contract(task_dir)
        finally:
            monkeypatch.undo()


def test_a_v2_task_has_one_contract_with_reference_display_and_pack(tmp_path):
    """@brief v2 标准题装配出语义/ reference /表示 / pack 的单一合同。"""
    task_dir = _task(tmp_path)
    contract = load_task_contract(task_dir)
    assert isinstance(contract, TaskContract)
    assert (contract.op_name, contract.dtype, contract.size) == ("rmsnorm", "fp16", "small")
    assert contract.spec.spec_version == 2
    assert contract.reference_outputs == ("y",)
    assert contract.reference_source == RMSNORM_REFERENCE
    assert contract.baselines == ()
    assert contract.correctness_only is True
    assert contract.pack_files == ("opspec.json", "reference.py")
    assert len(contract.pack_digest) == 64
    assert len(contract.fingerprint()) == 64
    assert contract.statement.startswith("# Task: rmsnorm")
    assert "grid" not in contract.statement and "threads" not in contract.statement


def test_default_dtype_and_size_are_applied_only_for_none(tmp_path):
    """@brief 只有 None 才取 discovery 默认值；空字符串进入既有域校验并拒绝。"""
    task_dir = _task(tmp_path)
    default = load_task_contract(task_dir)
    assert (default.dtype, default.size) == ("fp16", "small")
    assert (load_task_contract(task_dir, dtype="fp16", size="paper").size) == "paper"

    for dtype, size in (("", None), (None, ""), ("", "")):
        with pytest.raises(OpSpecError, match="dtype|size"):
            load_task_contract(task_dir, dtype=dtype, size=size)


def test_the_standard_optional_files_are_the_only_recognized_files(tmp_path):
    """@brief reference.json/display.json/baselines/known_good 被识别并进入 pack。"""
    baselines = json.dumps(
        {
            "schema_version": 1,
            "baselines": [
                {
                    "name": "torch_ref",
                    "framework": "torch",
                    "implementation_id": "rmsnorm.baseline.torch",
                    "implementation_version": 1,
                    "params": {"kernel": "torch.nn.functional.rms_norm"},
                }
            ],
        }
    )
    task_dir = _task(
        tmp_path,
        extra={
            "reference.json": json.dumps(
                {"schema_version": 1, "reference_version": 2, "sampling": {"default": {"kind": "normal", "mean": 0, "std": 1}}}
            ),
            "display.json": json.dumps({"schema_version": 1, "description": "RMSNorm standard entry."}),
            "baselines.json": baselines,
            "baselines.py": "import torch\n\ndef run(inputs):\n    return inputs\n\nBASELINES = {'torch_ref': run}\n",
            "known_good.metal": "// harness self-check only\n",
        },
    )
    contract = load_task_contract(task_dir)
    assert contract.pack_files == (
        "opspec.json",
        "reference.py",
        "reference.json",
        "baselines.json",
        "baselines.py",
        "display.json",
        "known_good.metal",
    )
    assert contract.known_good.startswith("// harness")
    assert [entry.name for entry in contract.baselines] == ["torch_ref"]
    assert contract.correctness_only is False
    assert contract.reference.reference_version == 2
    assert contract.statement.startswith("RMSNorm standard entry.")


@pytest.mark.parametrize("marker", ["track.json", "task.py", "kernel_triton.py", "naive.metal"])
def test_a_v2_directory_with_a_v1_marker_is_a_mixed_assembly_error(tmp_path, marker):
    """@brief v2 目录里出现任何 v1 装配文件都是混合目录错误。"""
    task_dir = _task(tmp_path)
    (task_dir / marker).write_text("{}\n" if marker.endswith(".json") else "pass\n", encoding="utf-8")
    with pytest.raises(AssemblyError, match="mixed v1/v2"):
        inspect_standard_entry(task_dir)


def test_a_v1_directory_with_a_v2_marker_is_a_mixed_assembly_error(tmp_path):
    """@brief v1 目录里出现 reference.py/display.json 等 v2 文件也是混合目录错误。"""
    task_dir = tmp_path / "rmsnorm"
    shutil.copytree(V1_TASK, task_dir, ignore=shutil.ignore_patterns("__pycache__"))
    (task_dir / "reference.py").write_text("def reference(inputs, attrs):\n    return {}\n", encoding="utf-8")
    with pytest.raises(AssemblyError, match="mixed v1/v2"):
        inspect_standard_entry(task_dir)


def test_v2_without_reference_does_not_fall_back_to_v1(tmp_path):
    """@brief 缺 reference.py 的 v2 题直接装配失败；不存在 v1 回退。"""
    task_dir = _task(tmp_path, reference=None)
    with pytest.raises(AssemblyError, match="never falls back"):
        inspect_standard_entry(task_dir)


def test_the_v2_directory_name_is_the_op_identity(tmp_path):
    """@brief 标准入口是 tasks/<op>/：目录名与 op_name 不一致时拒绝（自动发现依赖它）。"""
    task_dir = _task(tmp_path, op_name="rmsnorm")
    renamed = tmp_path / "other"
    renamed.mkdir()
    for name in ("opspec.json", "reference.py"):
        shutil.copy(task_dir / name, renamed / name)
    with pytest.raises(AssemblyError, match="differs from opspec op_name"):
        inspect_standard_entry(renamed)


def test_no_baselines_means_empty_task_baselines_and_correctness_only(tmp_path):
    """@brief 没有 baselines.json 时 ``baselines`` 为空且派生 correctness_only。"""
    contract = load_task_contract(_task(tmp_path))
    assert contract.baselines == ()
    assert contract.task_baselines() == ()
    assert contract.correctness_only is True
    assert contract.baseline_runners == {}


def test_baselines_require_their_runner_and_matching_names(tmp_path):
    """@brief baselines.json 与 baselines.py 是成对的，名字集合必须一致。"""
    manifest = json.dumps(
        {
            "schema_version": 1,
            "baselines": [
                {
                    "name": "b1",
                    "framework": "torch",
                    "implementation_id": "b1",
                    "implementation_version": 1,
                    "params": {},
                }
            ],
        }
    )
    task_dir = _task(tmp_path, extra={"baselines.json": manifest})
    with pytest.raises(AssemblyError, match="baselines.py is required"):
        load_task_contract(task_dir)

    (task_dir / "baselines.py").write_text("BASELINES = {'other': lambda inputs: inputs}\n", encoding="utf-8")
    with pytest.raises(AssemblyError, match="does not match"):
        load_task_contract(task_dir)

    (task_dir / "baselines.py").write_text("BASELINES = {'b1': lambda inputs: inputs}\n", encoding="utf-8")
    registered = load_task_contract(task_dir)
    assert registered.correctness_only is False
    task_baselines = registered.task_baselines()
    assert len(task_baselines) == 1
    assert (task_baselines[0].name, task_baselines[0].framework) == ("b1", "torch")
    assert task_baselines[0].implementation_id == "b1"


def test_a_baseline_runner_without_a_manifest_is_a_second_source(tmp_path):
    """@brief 只有 runner 没有登记时拒绝，避免第二套 baseline 来源。"""
    task_dir = _task(tmp_path, extra={"baselines.py": "BASELINES = {}\n"})
    with pytest.raises(AssemblyError, match="without baselines.json"):
        load_task_contract(task_dir)


def test_a_baseline_runner_may_not_import_the_oracle(tmp_path):
    """@brief baseline 是计时比较对象，不允许 import reference/known_good/renderer。"""
    manifest = json.dumps(
        {
            "schema_version": 1,
            "baselines": [
                {
                    "name": "b1",
                    "framework": "torch",
                    "implementation_id": "b1",
                    "implementation_version": 1,
                    "params": {},
                }
            ],
        }
    )
    task_dir = _task(
        tmp_path,
        extra={
            "baselines.json": manifest,
            "baselines.py": "import reference\nBASELINES = {'b1': lambda inputs: inputs}\n",
        },
    )
    with pytest.raises(AssemblyError, match="may not import"):
        load_task_contract(task_dir)


def test_display_spec_text_is_limited_to_the_g0_sem_allowlist(tmp_path):
    """@brief 普通算子不能覆盖题面；G0 冻结的 sem_* 夹具可以。"""
    assert spec_text_override_allowed("sem_gemm_neg")
    assert spec_text_override_allowed("sem_int_7")
    assert spec_text_override_allowed("sem_int_007")
    assert spec_text_override_allowed("sem_int_" + "0" * 17 + "7")
    assert spec_text_override_allowed("sem_int_" + "9" * 18)
    assert not spec_text_override_allowed("rmsnorm")

    task_dir = _task(tmp_path, extra={"display.json": json.dumps({"schema_version": 1, "spec_text": "Do anything."})})
    with pytest.raises(SemanticUnresolved, match="may not override"):
        load_task_contract(task_dir)

    sem_dir = write_task(
        tmp_path / "sem",
        op_name="sem_gemm_neg",
        opspec=sem_opspec("sem_gemm_neg"),
        reference=SEM_REFERENCE,
        extra={"display.json": json.dumps({"schema_version": 1, "spec_text": "Implement the code, not the name."})},
    )
    assert load_task_contract(sem_dir).statement == "Implement the code, not the name."


def test_display_spec_text_rejects_unregistered_sem_int_suffixes(tmp_path):
    """@brief sem_int_ 后缀必须是至多 18 位的 ASCII 十进制种子；任意替换不能借用该族。"""
    invalid = (
        "sem_int_not_a_seed",
        "sem_int_",
        "sem_int_-1",
        "sem_int_1e3",
        "sem_int_ 1",
        "sem_int_\u0661",          # ARABIC-INDIC ONE
        "sem_int_\uff11",          # FULLWIDTH DIGIT ONE
        "sem_int_1\u0661",         # mixed ASCII + Unicode digit
        "sem_int_" + "0" * 18 + "1",  # 19 digits, even with leading zeroes
    )
    for op_name in invalid:
        assert not spec_text_override_allowed(op_name), op_name
        with pytest.raises(SemanticUnresolved, match="may not override"):
            display_from_document(
                {"schema_version": 1, "spec_text": "replacement"},
                op_name=op_name,
            )

    # Full assembly for schema-valid spellings that are not valid seed suffixes: the display
    # gate, not an unrelated OpSpec error, must reject the replacement.
    for op_name in ("sem_int_not_a_seed", "sem_int_", "sem_int_1e3", "sem_int_" + "0" * 18 + "1"):
        task_dir = write_task(
            tmp_path / "invalid_suffix",
            op_name=op_name,
            opspec=sem_opspec(op_name),
            reference=SEM_REFERENCE,
            extra={"display.json": json.dumps({"schema_version": 1, "spec_text": "replacement"})},
        )
        with pytest.raises(SemanticUnresolved, match="may not override"):
            load_task_contract(task_dir)


def test_display_formula_must_agree_with_the_opspec_and_carries_no_semantics(tmp_path):
    """@brief 冗余公式不一致判 semantic_unresolved；display 不能携带容差/启动事实。"""
    task_dir = _task(
        tmp_path,
        extra={"display.json": json.dumps({"schema_version": 1, "formula": "y = x + 1"})},
    )
    with pytest.raises(SemanticUnresolved, match="disagrees"):
        load_task_contract(task_dir)

    bad = _task(
        tmp_path / "bad",
        extra={"display.json": json.dumps({"schema_version": 1, "tolerances": {"fp16": 1.0}})},
    )
    with pytest.raises(TaskContractError, match="unknown field"):
        load_task_contract(bad)


def test_display_description_changes_only_the_statement(tmp_path):
    """@brief description 只改表示，不改 opspec_digest 或语义实例。"""
    plain = load_task_contract(_task(tmp_path / "plain"))
    described = load_task_contract(
        _task(tmp_path / "described", extra={"display.json": json.dumps({"schema_version": 1, "description": "My task."})})
    )
    assert plain.spec.digest == described.spec.digest
    assert plain.instance.fingerprint_document() == described.instance.fingerprint_document()
    assert described.statement.startswith("My task.")
    assert plain.statement != described.statement


@pytest.mark.parametrize(
    "reference_source",
    [
        "import tasks.rmsnorm.opspec\n\ndef reference(inputs, attrs):\n    return {}\n",
        "import tcenv.track\n\ndef reference(inputs, attrs):\n    return {}\n",
        "import known_good\n\ndef reference(inputs, attrs):\n    return {}\n",
        "from . import helper\n\ndef reference(inputs, attrs):\n    return {}\n",
        "def reference(inputs, attrs):\n    open('/etc/passwd').read()\n    return {}\n",
        "import os\n\ndef reference(inputs, attrs):\n    os.system('true')\n    return {}\n",
    ],
)
def test_the_reference_module_cannot_depend_on_the_harness_or_a_candidate(tmp_path, reference_source):
    """@brief reference 静态边界：不 import 仓库/候选/known-good，不用文件/进程原语。"""
    task_dir = _task(tmp_path, reference=reference_source)
    with pytest.raises(ReferenceBoundaryError):
        load_task_contract(task_dir)


@pytest.mark.parametrize(
    "source",
    [
        "def reference(inputs):\n    return {}\n",
        "def reference(inputs, attrs, extra):\n    return {}\n",
        "def reference(*args):\n    return {}\n",
        "def reference(inputs, attrs=None):\n    return {}\n",
        "def not_a_reference(inputs, attrs):\n    return {}\n",
    ],
)
def test_the_reference_signature_is_fixed(tmp_path, source):
    """@brief ``reference(inputs, attrs)`` 是固定接口。"""
    task_dir = _task(tmp_path, reference=source)
    with pytest.raises(AssemblyError):
        load_task_contract(task_dir)


def test_optional_hooks_must_use_their_fixed_signatures(tmp_path):
    """@brief gen_inputs/error_scale 是可选项，但出现时必须使用固定签名。"""
    bad_gen = _task(
        tmp_path / "gen",
        reference=RMSNORM_REFERENCE + "\n\ndef gen_inputs(case, seed):\n    return {}\n",
    )
    with pytest.raises(AssemblyError, match="gen_inputs"):
        load_task_contract(bad_gen)

    bad_scale = _task(
        tmp_path / "scale",
        reference=RMSNORM_REFERENCE + "\n\ndef error_scale(inputs, outputs):\n    return {}\n",
    )
    with pytest.raises(AssemblyError, match="error_scale"):
        load_task_contract(bad_scale)


def test_the_default_sampler_is_deterministic_and_leaves_the_global_rng_alone(tmp_path):
    """@brief 默认采样用私有 generator：同种子同值，不扰动进程全局 RNG。"""
    import torch

    contract = load_task_contract(_task(tmp_path))
    torch.manual_seed(1234)
    before = torch.random.get_rng_state().clone()
    first = contract.reference.generate(contract.instance, "normal", 7919)
    after = torch.random.get_rng_state()
    assert torch.equal(before, after)
    second = contract.reference.generate(contract.instance, "normal", 7919)
    third = contract.reference.generate(contract.instance, "normal", 7920)
    assert torch.equal(first["x"], second["x"])
    assert not torch.equal(first["x"], third["x"])
    assert first["x"].dtype == torch.float16 and first["w"].dtype == torch.float16
    with pytest.raises(TaskContractError, match="not declared"):
        contract.reference.generate(contract.instance, "missing", 1)


def test_a_reference_gen_inputs_hook_is_used_and_validated(tmp_path):
    """@brief 有 ``gen_inputs`` 时优先使用，且键/形状必须与 OpSpec 一致。"""
    marker = (
        RMSNORM_REFERENCE
        + """

def gen_inputs(case, seed, instance):
    import torch
    return {
        "x": torch.full(tuple(instance.tensor("x").shape), float(seed), dtype=torch.float16),
        "w": torch.ones(tuple(instance.tensor("w").shape), dtype=torch.float16),
    }
"""
    )
    contract = load_task_contract(_task(tmp_path, reference=marker))
    inputs = contract.reference.generate(contract.instance, "normal", 7)
    assert float(inputs["x"].flatten()[0]) == 7.0

    broken = (
        RMSNORM_REFERENCE
        + """

def gen_inputs(case, seed, instance):
    return {"x": instance.tensor("x")}
"""
    )
    broken_contract = load_task_contract(_task(tmp_path / "broken", reference=broken))
    with pytest.raises(SemanticUnresolved, match="tensor keys"):
        broken_contract.reference.generate(broken_contract.instance, "normal", 7)


def test_read_write_final_values_are_required_in_the_reference_result(tmp_path):
    """@brief read_write 终值缺失判 semantic_unresolved；提供时进入差分合同。"""
    missing = _task(
        tmp_path / "missing",
        op_name="rmsnorm_rw",
        opspec=rmsnorm_read_write_opspec(),
        reference="""import torch\n\ndef reference(inputs, attrs):\n    return {"y": inputs["x"]}\n""",
    )
    contract = load_task_contract(missing)
    with pytest.raises(SemanticUnresolved, match="missing=\\['acc'\\]"):
        contract.reference.execute(contract.instance, contract.reference.generate(contract.instance, "normal", 1))

    provided = _task(
        tmp_path / "provided",
        op_name="rmsnorm_rw",
        opspec=rmsnorm_read_write_opspec(),
        reference=RMSNORM_READ_WRITE_REFERENCE,
    )
    contract = load_task_contract(provided)
    inputs = contract.reference.generate(contract.instance, "normal", 7919)
    assert contract.reference_inputs == contract.instance.reference_inputs == ("x", "w", "y", "acc")
    assert set(inputs) == set(contract.reference_inputs)
    outputs = contract.reference.execute(contract.instance, inputs)
    assert set(outputs) == set(contract.instance.reference_outputs) == {"y", "acc"}
    assert outputs["acc"].shape == inputs["acc"].shape

    eps = float(contract.instance.resolved_attrs["EPS"])
    x = inputs["x"].to(torch.float64)
    w = inputs["w"].to(torch.float64)
    y0 = inputs["y"].to(torch.float64)
    expected_y = (y0 + x * torch.rsqrt((x * x).mean(dim=-1, keepdim=True) + eps) * w).to(torch.float32)
    expected_acc = (inputs["acc"].to(torch.float64) + y0).to(torch.float32)
    assert torch.equal(outputs["y"], expected_y)
    assert torch.equal(outputs["acc"], expected_acc)

    # An output/read_write tensor's initial value is not optional: dropping it must fail the
    # input contract before the reference is allowed to read a KeyError.
    missing_y = dict(inputs)
    missing_y.pop("y")
    with pytest.raises(SemanticUnresolved, match=r"missing=\['y'\]"):
        contract.reference.execute(contract.instance, missing_y)


def test_error_scale_gets_working_copies_and_cannot_mutate_inputs(tmp_path):
    """@brief error_scale 与 reference 一样只拿工作副本；原地改写报错且调用方不变。"""
    reference = RMSNORM_REFERENCE + '''

def error_scale(inputs, outputs, instance, tolerances):
    inputs["x"].mul_(0)
    outputs["y"].mul_(0)
    return {name: torch.ones((), dtype=torch.float64) for name in outputs}
'''
    contract = load_task_contract(_task(tmp_path, reference=reference))
    inputs = contract.reference.generate(contract.instance, "normal", 7)
    outputs = contract.reference.execute(contract.instance, inputs)
    before_inputs = {name: tensor.clone() for name, tensor in inputs.items()}
    before_outputs = {name: tensor.clone() for name, tensor in outputs.items()}
    input_versions = {name: tensor._version for name, tensor in inputs.items()}
    output_versions = {name: tensor._version for name, tensor in outputs.items()}
    with pytest.raises(ReferenceBoundaryError, match="error_scale mutated"):
        contract.reference.error_scale(contract.instance, inputs, outputs)
    for name, tensor in inputs.items():
        assert torch.equal(tensor, before_inputs[name])
        assert tensor._version == input_versions[name]
    for name, tensor in outputs.items():
        assert torch.equal(tensor, before_outputs[name])
        assert tensor._version == output_versions[name]


@pytest.mark.parametrize("channel", ["data", "numpy"])
def test_reference_mutation_through_data_or_numpy_is_rejected(tmp_path, channel):
    """@brief .data / NumPy 视图不改 _version，也必须被内容快照识别。"""
    mutation = 'inputs["x"].data.mul_(2)' if channel == "data" else 'inputs["x"].numpy().__imul__(2)'
    contract = load_task_contract(_task(
        tmp_path,
        reference=RMSNORM_REFERENCE
        + f"""

def reference(inputs, attrs):
    {mutation}
    return {{"y": inputs["x"]}}
""",
    ))
    inputs = contract.reference.generate(contract.instance, "normal", 1)
    before = inputs["x"].clone()
    with pytest.raises(ReferenceBoundaryError, match="mutated"):
        contract.reference.execute(contract.instance, inputs)
    assert torch.equal(inputs["x"], before)


@pytest.mark.parametrize("channel", ["data", "numpy"])
def test_error_scale_mutation_through_data_or_numpy_is_rejected(tmp_path, channel):
    """@brief error_scale 的 .data / NumPy 视图写入也必须拒绝。"""
    mutation = 'inputs["x"].data.mul_(2)' if channel == "data" else 'inputs["x"].numpy().__imul__(2)'
    reference = RMSNORM_REFERENCE + f'''

def error_scale(inputs, outputs, instance, tolerances):
    {mutation}
    return {{name: torch.ones((), dtype=torch.float64) for name in outputs}}
'''
    contract = load_task_contract(_task(tmp_path, reference=reference))
    inputs = contract.reference.generate(contract.instance, "normal", 7)
    outputs = contract.reference.execute(contract.instance, inputs)
    before = inputs["x"].clone()
    with pytest.raises(ReferenceBoundaryError, match="error_scale mutated"):
        contract.reference.error_scale(contract.instance, inputs, outputs)
    assert torch.equal(inputs["x"], before)


def test_the_reference_cannot_mutate_its_inputs_or_write_through_the_mapping(tmp_path):
    """@brief 原地改写输入与映射赋值都触发只读 reference 边界。"""
    inplace = _task(
        tmp_path / "inplace",
        reference=RMSNORM_REFERENCE
        + """

def reference(inputs, attrs):
    inputs["x"].mul_(2)
    return {"y": inputs["x"]}
""",
    )
    contract = load_task_contract(inplace)
    inputs = contract.reference.generate(contract.instance, "normal", 1)
    before = {name: tensor.clone() for name, tensor in inputs.items()}
    versions = {name: tensor._version for name, tensor in inputs.items()}
    with pytest.raises(ReferenceBoundaryError, match="mutated"):
        contract.reference.execute(contract.instance, inputs)
    for name, tensor in inputs.items():
        assert torch.equal(tensor, before[name]), f"caller input {name!r} was mutated despite the boundary error"
        assert tensor._version == versions[name]

    rebound = _task(
        tmp_path / "rebound",
        reference="""def reference(inputs, attrs):\n    inputs["x"] = None\n    return {"y": inputs["x"]}\n""",
    )
    contract = load_task_contract(rebound)
    with pytest.raises(ReferenceBoundaryError, match="read-only"):
        contract.reference.execute(contract.instance, contract.reference.generate(contract.instance, "normal", 1))


def test_reference_result_shape_and_keys_are_part_of_the_semantic_contract(tmp_path):
    """@brief 结果形状不符或多余键都判 semantic_unresolved。"""
    wrong_shape = _task(
        tmp_path / "shape",
        reference="""import torch\n\ndef reference(inputs, attrs):\n    return {"y": torch.zeros((1,), dtype=torch.float32)}\n""",
    )
    contract = load_task_contract(wrong_shape)
    with pytest.raises(SemanticUnresolved, match="shape"):
        contract.reference.execute(contract.instance, contract.reference.generate(contract.instance, "normal", 1))

    extra = _task(
        tmp_path / "extra",
        reference=RMSNORM_REFERENCE + "\n\ndef reference(inputs, attrs):\n    import torch\n    return {'y': inputs['x'], 'z': inputs['x']}\n",
    )
    contract = load_task_contract(extra)
    with pytest.raises(SemanticUnresolved, match="extra=\\['z'\\]"):
        contract.reference.execute(contract.instance, contract.reference.generate(contract.instance, "normal", 1))


def _int_add_opspec() -> dict:
    """@brief 一个纯整数 v2 OpSpec（默认采样必须拒绝，显式 gen_inputs 必须可用）。

    @return v2 文档。
    """
    return {
        "spec_version": 2,
        "op_name": "int_add",
        "category": "elementwise",
        "formula": "y = x + 1",
        "type_vars": {"T": ["i32"]},
        "symbols": {"N": {"min": 1}},
        "tensors": {
            "x": {"role": "input", "dims": ["N"], "dtype": "T", "layout": "contiguous", "strides": ["1"], "space": "global", "access": "read"},
            "y": {"role": "output", "dims": ["N"], "dtype": "T", "layout": "contiguous", "strides": ["1"], "space": "global", "access": "write"},
        },
        "numerics": {"acc_dtype": "i32", "rounding": "exact_integer", "tolerances": {"i32": {"rtol": 0, "atol_rel": 0}}, "exact": True},
        "attrs": {},
        "cases": {"names": ["normal"]},
        "sizes": {"small": {"N": 8, "usage": ["golden"]}},
        "spec_params": ["N"],
        "work_accounting": {"flops": {"terms": [{"coefficient": 1, "symbols": ["N"]}]}, "min_bytes": {"terms": [{"tensor": "x"}, {"tensor": "y"}]}},
        "discovery": {"phase": 1, "default_instance": {"dtype": "i32", "size": "small"}},
    }


def test_an_integer_task_with_only_opspec_and_reference_uses_its_gen_inputs(tmp_path):
    """@brief 仅两个必需文件的整数题：自定义 gen_inputs 不受默认采样器限制。"""
    reference = '''"""Integer reference with its own generator."""

import torch


def gen_inputs(case, seed, instance):
    return {"x": torch.arange(instance.tensor("x").shape[0], dtype=torch.int32)}


def reference(inputs, attrs):
    return {"y": inputs["x"] + 1}
'''
    contract = load_task_contract(_task(tmp_path, op_name="int_add", opspec=_int_add_opspec(), reference=reference))
    assert contract.reference.sampling is None
    inputs = contract.reference.generate(contract.instance, "normal", 7919)
    assert inputs["x"].dtype == torch.int32
    assert contract.reference.execute(contract.instance, inputs)["y"].tolist() == list(range(1, 9))


def test_gen_inputs_and_a_sampling_profile_are_mutually_exclusive(tmp_path):
    """@brief reference.py 的 gen_inputs 与 reference.json 的 sampling 是两个输入源，不能并存。"""
    reference = RMSNORM_REFERENCE + "\n\ndef gen_inputs(case, seed, instance):\n    return {}\n"
    task_dir = _task(
        tmp_path,
        reference=reference,
        extra={"reference.json": json.dumps({"schema_version": 1, "reference_version": 1, "sampling": {"default": {"kind": "normal", "mean": 0, "std": 1}}})},
    )
    with pytest.raises(SemanticUnresolved, match="one input source"):
        load_task_contract(task_dir)


def test_the_reference_registry_is_one_authoritative_oracle_per_op(tmp_path):
    """@brief ReferenceRegistry 按 op 唯一；同摘要可重复登记，不同摘要拒绝。"""
    first = load_task_contract(_task(tmp_path / "first"))
    second = load_task_contract(_task(tmp_path / "second"))
    registry = ReferenceRegistry()
    registry.register(first.reference)
    registry.register(second.reference)
    assert registry.names() == ("rmsnorm",)
    assert registry.get("rmsnorm").source_sha256 == first.reference.source_sha256
    assert "rmsnorm" in registry
    changed_dir = _task(tmp_path / "changed", reference=RMSNORM_REFERENCE + "\n# different bytes\n")
    with pytest.raises(AssemblyError, match="one authoritative oracle"):
        registry.register(load_task_contract(changed_dir).reference)

    versioned = _task(
        tmp_path / "versioned",
        extra={"reference.json": json.dumps({"schema_version": 1, "reference_version": 2})},
    )
    with pytest.raises(AssemblyError, match="one authoritative oracle"):
        registry.register(load_task_contract(versioned).reference)

    resampled = _task(
        tmp_path / "resampled",
        extra={"reference.json": json.dumps({"schema_version": 1, "reference_version": 1, "sampling": {"default": {"kind": "uniform", "low": 0, "high": 1}}})},
    )
    with pytest.raises(AssemblyError, match="one authoritative oracle"):
        registry.register(load_task_contract(resampled).reference)


def test_the_reference_source_is_read_only(tmp_path):
    """@brief ReferenceModule 冻结；展示文本只读。"""
    contract = load_task_contract(_task(tmp_path))
    assert contract.reference.display_source() == RMSNORM_REFERENCE
    with pytest.raises(dataclasses.FrozenInstanceError):
        contract.reference.source = "changed"


def test_a_workspace_tensor_enters_the_a2_abi_and_stays_out_of_the_reference_contract(tmp_path):
    """@brief A2 提供 workspace ABI：暂存张量进 ABI，但不是 reference 初值/结果。"""
    task_dir = write_task(tmp_path / "ws", op_name="rmsnorm_ws", opspec=rmsnorm_workspace_opspec())
    contract = load_task_contract(task_dir)
    instance = contract.instance
    assert "scratch" not in instance.reference_inputs
    assert "scratch" not in instance.reference_outputs
    assert [tensor.name for tensor in instance.tensors if tensor.space == "workspace"] == ["scratch"]

    from tcenv.target import DEFAULT_TARGET_ID, load_target
    from tcenv.track import TargetAbiDeriver, derive_tensor_memory_contract

    deriver = TargetAbiDeriver.for_target(load_target(DEFAULT_TARGET_ID))
    abi = deriver.derive(instance)
    assert abi.tensor("scratch").space == "workspace"
    assert derive_tensor_memory_contract(instance, target=load_target(DEFAULT_TARGET_ID)) == abi

    # A task whose only output is scratch has no result tensor: that is still an assembly error,
    # because a workspace buffer is not a result the reference can diff.
    document = rmsnorm_opspec()
    document["tensors"]["y"]["space"] = "workspace"
    with pytest.raises(AssemblyError, match="result tensor"):
        load_task_contract(_task(tmp_path / "ws_only", opspec=document))


def test_v2_variants_are_refused_at_assembly(tmp_path):
    """@brief v2 的 @variant 在装配期拒绝（无论来自调用方还是 discovery）。"""
    task_dir = _task(tmp_path)
    with pytest.raises(UnsupportedContractCapability, match="variant"):
        load_task_contract(task_dir, variant="128x64x32t256")
    # The interface says "not None", not "truthy": an empty @variant is still an assembly
    # request and must not silently select the default instance.
    with pytest.raises(UnsupportedContractCapability, match="variant"):
        load_task_contract(task_dir, variant="")

    document = rmsnorm_opspec()
    document["discovery"]["variant_acceptance"] = {"factory_id": "f", "implementation_version": 1}
    with pytest.raises(UnsupportedContractCapability, match="variant_acceptance"):
        load_task_contract(_task(tmp_path / "va", opspec=document))


def test_the_reference_profile_is_closed_and_error_scale_must_match(tmp_path):
    """@brief reference.json 不能是第二份语义；error_scale 声明必须与模块一致。"""
    bad = _task(tmp_path / "bad", extra={"reference.json": json.dumps({"schema_version": 1, "reference_version": 1, "params": {}})})
    with pytest.raises(TaskContractError, match="unknown field 'params'"):
        load_task_contract(bad)

    declared_none = _task(
        tmp_path / "none",
        reference=RMSNORM_REFERENCE + "\n\ndef error_scale(inputs, outputs, instance, tolerances):\n    return {}\n",
        extra={"reference.json": json.dumps({"schema_version": 1, "reference_version": 1, "error_scale": {"kind": "none"}})},
    )
    with pytest.raises(SemanticUnresolved, match="error_scale"):
        load_task_contract(declared_none)

    declared_module = _task(
        tmp_path / "module",
        extra={"reference.json": json.dumps({"schema_version": 1, "reference_version": 1, "error_scale": {"kind": "module"}})},
    )
    with pytest.raises(SemanticUnresolved, match="error_scale"):
        load_task_contract(declared_module)


def test_sampling_profile_precedence_and_domain(tmp_path):
    """@brief reference.json 采样：tensor > case > default，且名字必须在声明域内。"""
    reference_json = {
        "schema_version": 1,
        "reference_version": 1,
        "sampling": {
            "default": {"kind": "normal", "mean": 0.0, "std": 1.0},
            "cases": {"large": {"kind": "uniform", "low": -2.0, "high": 2.0}},
            "tensors": {"w": {"kind": "uniform", "low": 3.0, "high": 3.0}},
        },
    }
    contract = load_task_contract(_task(tmp_path, extra={"reference.json": json.dumps(reference_json)}))
    inputs = contract.reference.generate(contract.instance, "large", 7919)
    assert float(inputs["w"].min()) == 3.0 and float(inputs["w"].max()) == 3.0
    assert float(inputs["x"].min()) >= -2.0 and float(inputs["x"].max()) <= 2.0

    bad = _task(
        tmp_path / "bad",
        extra={"reference.json": json.dumps({"schema_version": 1, "reference_version": 1, "sampling": {"default": {"kind": "normal", "mean": 0, "std": 1}, "cases": {"missing": {"kind": "normal", "mean": 0, "std": 1}}}})},
    )
    with pytest.raises(TaskContractError, match="unknown cases"):
        load_task_contract(bad)


def test_integer_case_override_is_checked_with_the_effective_profile(tmp_path):
    """@brief 整数校验必须按 tensor > case > default 的有效 profile 在装配期判定。"""
    reference = """# Integer add reference without a custom generator.

import torch


def reference(inputs, attrs):
    return {"y": inputs["x"] + 1}
"""
    legal = {
        "schema_version": 1,
        "reference_version": 1,
        "sampling": {
            "default": {"kind": "normal", "mean": 0.0, "std": 1.0},
            "cases": {"normal": {"kind": "uniform", "low": 0, "high": 3}},
        },
    }
    contract = load_task_contract(
        _task(
            tmp_path / "case_uniform",
            op_name="int_add",
            opspec=_int_add_opspec(),
            reference=reference,
            extra={"reference.json": json.dumps(legal)},
        )
    )
    inputs = contract.reference.generate(contract.instance, "normal", 7)
    assert inputs["x"].dtype == torch.int32
    assert int(inputs["x"].min()) >= 0 and int(inputs["x"].max()) <= 3

    illegal = {
        "schema_version": 1,
        "reference_version": 1,
        "sampling": {
            "default": {"kind": "uniform", "low": 0, "high": 3},
            "cases": {"normal": {"kind": "normal", "mean": 0.0, "std": 1.0}},
        },
    }
    with pytest.raises(TaskContractError, match="case 'normal'"):
        load_task_contract(
            _task(
                tmp_path / "case_normal",
                op_name="int_add",
                opspec=_int_add_opspec(),
                reference=reference,
                extra={"reference.json": json.dumps(illegal)},
            )
        )


def test_integer_tensors_need_an_explicit_uniform_sampling_profile(tmp_path):
    """@brief 整数输入的默认采样不猜分布；显式 uniform 后才可生成。"""
    document = {
        "spec_version": 2,
        "op_name": "int_add",
        "category": "elementwise",
        "formula": "y = x + 1",
        "type_vars": {"T": ["i32"]},
        "symbols": {"N": {"min": 1}},
        "tensors": {
            "x": {"role": "input", "dims": ["N"], "dtype": "T", "layout": "contiguous", "strides": ["1"], "space": "global", "access": "read"},
            "y": {"role": "output", "dims": ["N"], "dtype": "T", "layout": "contiguous", "strides": ["1"], "space": "global", "access": "write"},
        },
        "numerics": {"acc_dtype": "i32", "rounding": "exact_integer", "tolerances": {"i32": {"rtol": 0, "atol_rel": 0}}, "exact": True},
        "attrs": {},
        "cases": {"names": ["normal"]},
        "sizes": {"small": {"N": 8, "usage": ["golden"]}},
        "spec_params": ["N"],
        "work_accounting": {"flops": {"terms": [{"coefficient": 1, "symbols": ["N"]}]}, "min_bytes": {"terms": [{"tensor": "x"}, {"tensor": "y"}]}},
        "discovery": {"phase": 1, "default_instance": {"dtype": "i32", "size": "small"}},
    }
    reference = "def reference(inputs, attrs):\n    return {\"y\": inputs[\"x\"] + 1}\n"
    with pytest.raises(TaskContractError, match="integer input"):
        load_task_contract(_task(tmp_path / "none", op_name="int_add", opspec=document, reference=reference))

    profile = {"schema_version": 1, "reference_version": 1, "sampling": {"default": {"kind": "uniform", "low": 0, "high": 4}}}
    contract = load_task_contract(
        _task(
            tmp_path / "uniform",
            op_name="int_add",
            opspec=document,
            reference=reference,
            extra={"reference.json": json.dumps(profile)},
        )
    )
    inputs = contract.reference.generate(contract.instance, "normal", 1)
    assert inputs["x"].dtype.__str__() in ("torch.int32",) or str(inputs["x"].dtype) == "torch.int32"
    assert int(inputs["x"].min()) >= 0 and int(inputs["x"].max()) <= 4


def test_task_pack_and_fingerprint_exclude_guidance_and_bind_reference_bytes(tmp_path):
    """@brief 知识文件不进 task_pack_digest；reference/display/baseline 变化进入合同身份。"""
    task_dir = _task(tmp_path)
    before = load_task_contract(task_dir)
    (task_dir / "guidance.md").write_text("prefer more warps\n", encoding="utf-8")
    (task_dir / "skills").mkdir()
    (task_dir / "skills" / "author.md").write_text("secret trick\n", encoding="utf-8")
    after = load_task_contract(task_dir)
    assert after.pack_files == before.pack_files
    assert after.pack_digest == before.pack_digest
    assert after.fingerprint() == before.fingerprint()

    (task_dir / "reference.py").write_text(RMSNORM_REFERENCE + "\n# a comment changes the oracle bytes\n", encoding="utf-8")
    changed = load_task_contract(task_dir)
    assert changed.pack_digest != before.pack_digest
    assert changed.fingerprint() != before.fingerprint()

    display_dir = _task(tmp_path / "display", extra={"display.json": json.dumps({"schema_version": 1, "description": "d"})})
    display_contract = load_task_contract(display_dir)
    assert display_contract.pack_digest != before.pack_digest
    assert display_contract.fingerprint() != before.fingerprint()


def test_loading_v2_does_not_change_the_v1_task_identity(tmp_path):
    """@brief 读/装配 v2 不触碰 v1 题：v1 opspec_digest 与 task fingerprint 前后一致。"""
    from tasks import get_task
    from tcenv.task import task_fingerprint

    v1_before = get_task(V1_TASK.name, "fp16", "small")
    digest_before = v1_before.semantic_instance.opspec_digest
    fingerprint_before = task_fingerprint(v1_before)
    load_task_contract(_task(tmp_path))
    v1_after = get_task(V1_TASK.name, "fp16", "small")
    assert v1_after.semantic_instance.opspec_digest == digest_before == load_opspec(V1_TASK / "opspec.json").digest
    assert task_fingerprint(v1_after) == fingerprint_before


def test_the_contract_has_no_second_verdict_or_diagnostics_field(tmp_path):
    """@brief A1 合同不引入第二套 pass/fail 或 ULP/precision 判决字段。"""
    names = {field.name for field in dataclasses.fields(TaskContract)}
    for forbidden in ("ulp", "precision", "verdict", "diagnostics", "cake"):
        assert forbidden not in names
    contract = load_task_contract(_task(tmp_path))
    document = contract.fingerprint_document()
    serialized = json.dumps(document)
    for forbidden in ("ulp", "precision", "cake"):
        assert forbidden not in serialized.lower()
