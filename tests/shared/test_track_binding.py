"""Track binding: three placeholder sources, the closed key set and the pack file set.

CPU only.  These tests pin plan §4.4's rendering rules: every placeholder declares exactly
one of ``spec_params``/``dtype_spelling``/``backend_params``, a second provider or a
surviving ``{{...}}`` is refused (F-22), and the task-pack file set is relative, existing,
unique and symlink-free (the R4 ``task_pack_digest`` depends on it).
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from tasks import get_task
from tasks._common import load_text
from tcenv.track import REPO_ROOT, TrackBinding, TrackError, load_track
from tests.support.v1_tasks import V1_TASK as _DEFAULT_V1_TASK  # noqa: F401

# B2 之后 conv2d 也不再有 v1 track.json；本文件改以 bitdelta_gemm 为样本（保留题，含
# backend_params 与多个 baseline），TrackBinding 机制的断言不变。
V1_TASK = "bitdelta_gemm"

# A5 retired the migrated rmsnorm v1 track; this file pins the *v1 TrackBinding mechanism*, so it
# runs on the preserved v1 bitdelta_gemm task (B2 moved conv2d/lion to v2).  The v2 track plane has its own tests (tests/shared/test_track_*).
TRACK_PATH = REPO_ROOT / "tasks" / V1_TASK / "track.json"


def _document() -> dict:
    """@brief 读取真实 track 文档的可变异副本。

    @return 深拷贝字典。
    """
    return json.loads(TRACK_PATH.read_text(encoding="utf-8"))


def _write(tmp_path: Path, document: dict) -> Path:
    """@brief 把一份 track 文档写到临时目录。

    @param tmp_path 临时目录。
    @param document track 文档。
    @return 写出的路径。
    """
    path = tmp_path / "track.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def test_the_shipped_binding_renders_exactly_what_the_legacy_loader_rendered():
    """@brief 三类来源渲染出的题面/known-good 与旧 ``tasks/_common.py::load_text`` 逐字节一致。"""
    track = load_track(TRACK_PATH)
    for dtype, size in (("fp16", "paper"), ("bf16", "small")):
        task = get_task(V1_TASK, dtype, size)
        extra = dict(task.semantic_instance.spec_params) | dict(task.specialization.backend_params)
        assert task.triton_source == load_text(V1_TASK, "kernel_triton.py", dtype, extra)
        assert task.naive_msl == load_text(V1_TASK, "naive.metal", dtype)


def test_spec_params_and_dtype_spelling_come_from_their_declared_sources():
    """@brief ``{{M}}``/``{{H}}`` 只来自 spec_params，``{{TL}}``/``{{T}}`` 只来自拼写表。"""
    track = load_track(TRACK_PATH)
    assert track.spec_text.placeholders == {
        "M": "spec_params", "N": "spec_params", "K": "spec_params", "TL": "dtype_spelling",
        "NW": "backend_params", "BM": "backend_params", "BN": "backend_params", "BK": "backend_params",
    }
    assert track.known_good.placeholders == {"T": "dtype_spelling"}
    assert track.dtype_spelling["fp16"]["TL"] == "tl.float16"
    assert track.dtype_spelling["bf16"]["T"] == "bfloat"
    assert track.backend_id == "metal"
    assert track.lowering_profile == "metal-bitdelta-gemm"
    assert track.opspec_path == REPO_ROOT / "tasks" / V1_TASK / "opspec.json"


def test_an_undeclared_placeholder_in_the_template_is_refused(tmp_path):
    """@brief 模板里有未声明的占位符 → 装配失败（不会渲染出 ``{{...}}`` 给模型看）。"""
    document = _document()
    document["spec_text"]["placeholders"].pop("M")
    with pytest.raises(TrackError):
        load_track(_write(tmp_path, document))


def test_a_declared_placeholder_that_is_absent_from_the_template_is_refused(tmp_path):
    """@brief 声明了模板里不存在的占位符也失败（声明只能是“实际出现”的集合）。"""
    document = _document()
    document["known_good"]["placeholders"]["NUM_WARPS"] = "backend_params"
    with pytest.raises(TrackError):
        load_track(_write(tmp_path, document))


def test_a_placeholder_that_two_sources_provide_is_refused():
    """@brief F-22：同名占位符在两个来源都有值时拒绝（不猜来源）。"""
    document = _document()
    document["dtype_spelling"]["fp16"]["M"] = "fp16-iota"
    track = TrackBinding(path=TRACK_PATH, document=document)
    task = get_task(V1_TASK, "fp16", "small")
    with pytest.raises(TrackError) as excinfo:
        track.render_spec_text(task.semantic_instance, task.specialization)
    assert "ambiguous" in str(excinfo.value)


def test_a_declared_source_without_a_value_is_refused():
    """@brief 声明来源没有值（例如把 ``{{TL}}`` 声明成 ``spec_params``）时拒绝。"""
    document = _document()
    document["spec_text"]["placeholders"]["TL"] = "spec_params"
    track = TrackBinding(path=TRACK_PATH, document=document)
    task = get_task(V1_TASK, "fp16", "small")
    with pytest.raises(TrackError) as excinfo:
        track.render_spec_text(task.semantic_instance, task.specialization)
    assert "spec_params" in str(excinfo.value)


def test_a_dtype_outside_the_spelling_table_is_refused():
    """@brief 拼写表缺该 dtype 时拒绝，而不是留下半渲染的题面。"""
    document = _document()
    document["dtype_spelling"].pop("bf16")
    track = TrackBinding(path=TRACK_PATH, document=document)
    task = get_task(V1_TASK, "bf16", "paper")
    with pytest.raises(TrackError):
        track.render_spec_text(task.semantic_instance, task.specialization)


def test_the_pack_file_set_is_relative_existing_unique_and_symlink_free():
    """@brief R4 的 ``task_pack_digest`` 依赖这个文件集：全部存在、相对路径且不重复。"""
    track = load_track(TRACK_PATH)
    files = track.pack_files
    assert len(files) == len({path.casefold() for path in files})
    for relative in files:
        assert not relative.startswith("/")
        path = REPO_ROOT / relative
        assert path.is_file(), relative
        assert not path.is_symlink(), relative
    # The semantics, the lowering, the templates and the registered implementations are covered.
    for required in (
        f"tasks/{V1_TASK}/opspec.json",
        f"tasks/{V1_TASK}/track.json",
        f"tasks/{V1_TASK}/kernel_triton.py",
        f"tasks/{V1_TASK}/naive.metal",
        "tcenv/backends/metal/lowering.py",
        "tcenv/lowering.py",
        "tcenv/opspec/model.py",
    ):
        assert required in files


def test_an_absolute_duplicate_or_missing_pack_file_is_refused(tmp_path):
    """@brief 绝对路径、大小写冲突与缺失文件都使装配失败（不得跳过）。"""
    for mutate in (
        lambda document: document["pack_files"].append("/etc/passwd"),
        lambda document: document["pack_files"].append(f"TASKS/{V1_TASK}/TASK.py"),
        lambda document: document["pack_files"].append(f"tasks/{V1_TASK}/opspec.json"),
        lambda document: document["pack_files"].append(f"tasks/{V1_TASK}/missing.txt"),
        lambda document: document["pack_files"].append("../outside.txt"),
    ):
        document = _document()
        mutate(document)
        with pytest.raises(TrackError):
            load_track(_write(tmp_path, document))


def test_registered_implementations_and_baselines_are_closed_sets(tmp_path):
    """@brief gen/reference/error_scale 与 baseline 登记必须齐备（缺登记 → R1 失败）。"""
    track = load_track(TRACK_PATH)
    implementations = track.implementations
    assert implementations["gen_inputs"]["implementation_id"] == "bitdelta_gemm.gen_inputs"
    assert implementations["reference"]["implementation_id"] == "bitdelta_gemm.reference"
    assert implementations["error_scale"] is None
    assert [entry["name"] for entry in track.baselines] == ["mlx", "mlx_compile", "torch_mps", "torch_compile"]
    for entry in track.baselines:
        assert entry["implementation_id"] and entry["implementation_version"] >= 1
        assert entry["params"]

    for mutate in (
        lambda document: document["implementations"].pop("reference"),
        lambda document: document["baselines"][0].pop("implementation_id"),
        lambda document: document["implementations"]["gen_inputs"].__setitem__("implementation_version", 0),
    ):
        document = copy.deepcopy(_document())
        mutate(document)
        with pytest.raises(TrackError):
            load_track(_write(tmp_path, document))


def _generated_document() -> dict:
    """@brief 以 preserved v1 track 为样板，构造一份 retired ``gen_*`` 形状的文档。

    @return track 文档字典。

    A5 migrated the whole ``gen_*`` family, so no shipped v1 generated track remains.  The
    generated-block *mechanism* is still exercised on a synthetic document built from the
    preserved v1 bitdelta_gemm track (same fixture shape the retired ``gen_relu`` used).
    """
    document = _document()
    document["spec_text"] = {"generator": {
        "implementation_id": "generated.elementwise.triton", "implementation_version": 1,
        "params": {"grammar_version": 1, "renderer": "pointwise-expression", "view": "triton"},
    }}
    document["known_good"] = {"generator": {
        "implementation_id": "generated.elementwise.msl", "implementation_version": 1,
        "params": {"grammar_version": 1, "renderer": "pointwise-expression", "view": "msl"},
    }}
    document["template_literals"] = {"launch": [], "semantic": []}
    return document


def test_a_generated_text_block_with_one_real_opspec_is_accepted(tmp_path):
    """@brief 生成块 + 真实 ``opspec`` 合法：``gen_*`` 每题一份语义文件，文本是它的渲染（N-31）。"""
    document = _generated_document()
    track = load_track(_write(tmp_path, document))
    assert track.spec_text_generator is not None and track.known_good_generator is not None
    assert track.opspec_path == REPO_ROOT / "tasks" / V1_TASK / "opspec.json"


def test_a_half_generated_track_is_refused(tmp_path):
    """@brief 一份生成、一份模板 → 拒绝（否则题面与 known-good 会来自两套事实）。"""
    document = _generated_document()
    document["known_good"] = {
        "template": "tasks/bitdelta_gemm/naive.metal",
        "placeholders": {"T": "dtype_spelling"},
    }
    with pytest.raises(TrackError, match="both be generated or both be templates"):
        load_track(_write(tmp_path, document))


def test_a_fully_generated_track_still_requires_both_blocks(tmp_path):
    """@brief ``opspec=null`` 只在两个块都是生成器时合法（``sem_int`` 族的形状）。"""
    document = _generated_document()
    document["opspec"] = None
    document["known_good"] = {
        "template": "tasks/bitdelta_gemm/naive.metal",
        "placeholders": {"T": "dtype_spelling"},
    }
    with pytest.raises(TrackError, match="only valid for a fully generated track"):
        load_track(_write(tmp_path, document))
