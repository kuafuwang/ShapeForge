"""V3: generated Host Program contract, restricted C99 precheck and Wasm runtime (CPU only).

These tests never touch the GPU.  They prove:

* ``host_contract.h`` is *derived* from the OpSpec/ABI (same spec from the instance and from the
  derived ``TensorMemoryContract``), so statement and build cannot disagree;
* the header's Wasm32 layout and the Python layout are locked together (the compiled fixture uses
  both, and the static assertions fail the compile if they drift);
* projection only refuses dangerous syntax; the real Clang call lives in ``compile_host_program``
  and its diagnostics are the repair signal;
* a failed host run (non-zero return, trap, fuel exhaustion, span/guard/input violation) commits
  nothing, and the valid span is the only thing the harness copies;
* large inputs can only be read through the generated whitelisted imports, and each read interval
  is recorded;
* a missing toolchain is the typed ``backend_capability_unavailable``, not a candidate failure or
  an unhandled crash.

When this machine has no Clang/lld/Wasm engine the compile-dependent tests skip with the exact
``doctor.py`` next step; the typed-refusal test still runs.
"""

from __future__ import annotations

from functools import lru_cache

import pytest

from tcenv.hostprogram.contract import (
    MAX_MAPPED_INPUT_BYTES,
    STORAGE_GUARD_BYTES,
    host_contract_spec,
    referenced_workspaces,
    storage_layout,
)
from tcenv.hostprogram.delivery import preflight_host_program
from tcenv.hostprogram.errors import HostCapabilityUnavailable, HostProgramError
from tcenv.hostprogram.precheck import static_precheck
from tcenv.hostprogram.runtime import HostRuntime
from tcenv.hostprogram.toolchain import compile_host_program, discover_toolchain, fuel_budget, memory_budget_bytes
from tcenv.taskcontract import load_task_contract
from tcenv.taskcontract.locator import REPO_ROOT
from tcenv.track.abi import derive_tensor_memory_contract
from tcenv.target import load_target
from tcenv.tensors import round_to, to_raw

from ..support.hostprogram_fixtures import PLANNER_SOURCE

TASK = REPO_ROOT / "tasks" / "varlen_kda"
TARGET = load_target("apple-m4-max")


def _contract(size: str = "mixed"):
    return load_task_contract(TASK, dtype="fp16", size=size)


def _toolchain():
    try:
        return discover_toolchain()
    except HostCapabilityUnavailable as exc:  # pragma: no cover - depends on the host
        pytest.skip(f"V3 Host Program toolchain unavailable ({exc}); run `brew install llvm lld` + `uv sync`")


def _spec(contract, *, mapped_limit: int = MAX_MAPPED_INPUT_BYTES):
    abi = derive_tensor_memory_contract(contract.instance, target=TARGET)
    return host_contract_spec(instance=contract.instance, tensors=abi.tensors, mapped_limit=mapped_limit), abi


def _raw_inputs(contract, case: str = "canonical", seed: int = 7) -> dict[str, bytes]:
    instance = contract.instance
    logical = contract.reference.generate(instance, case, seed)
    shapes = {tensor.name: tensor.shape for tensor in instance.tensors}
    dtypes = {tensor.name: tensor.dtype for tensor in instance.tensors}
    out: dict[str, bytes] = {}
    for name, value in logical.items():
        if name in shapes:
            rounded = round_to(value.reshape(tuple(int(v) for v in shapes[name])), dtypes[name])
            out[name] = to_raw(rounded, dtypes[name])
    return out


def _reader(raw: dict[str, bytes]):
    def read(name: str, byte_off: int, nbytes: int) -> bytes:
        return raw[name][byte_off : byte_off + nbytes]

    return read


@lru_cache(maxsize=32)
def _build(size: str, source: str, mapped_limit: int):
    contract = _contract(size)
    spec, _abi = _spec(contract, mapped_limit=mapped_limit)
    build = compile_host_program(source, spec=spec, entry_path="host/tiling.c", toolchain=_toolchain())
    return spec, build


def _runtime(size: str, source: str = PLANNER_SOURCE, *, mapped_limit: int = MAX_MAPPED_INPUT_BYTES):
    spec, build = _build(size, source, mapped_limit)
    runtime = HostRuntime(build, spec)
    runtime.set_target(
        {
            "max_buffer_index": int(TARGET.raw_limits.get("max_buffer_index", 0)),
            "max_threads_per_threadgroup": int(TARGET.raw_limits.get("max_threads_per_threadgroup", 0)),
            "max_threadgroup_memory_bytes": int(TARGET.raw_limits.get("max_threadgroup_memory_bytes", 0)),
        }
    )
    return spec, build, runtime


def _mutate(source: str, needle: str, replacement: str) -> str:
    assert source.count(needle) == 1, needle
    return source.replace(needle, replacement)


# ---------------------------------------------------------------------------------------------
# Contract derivation and layout
# ---------------------------------------------------------------------------------------------


def test_the_same_contract_is_derived_from_the_instance_and_from_the_derived_abi():
    """@brief 题面（instance）与 build（ABI）必须得到同一份 header/摘要。"""
    contract = _contract()
    abi = derive_tensor_memory_contract(contract.instance, target=TARGET)
    from_instance = host_contract_spec(instance=contract.instance)
    from_abi = host_contract_spec(instance=contract.instance, tensors=abi.tensors)
    assert from_instance.digest == from_abi.digest
    assert from_instance.header() == from_abi.header()
    assert "int32_t *cu_seqlens;" in from_instance.header()
    assert "uint32_t work_items;" not in from_instance.header()


def test_the_header_exposes_declared_names_types_and_capacities_only():
    """@brief 强类型字段、cap 宏和返回码来自 OpSpec/ABI，不来自候选。"""
    contract = _contract()
    spec, _abi = _spec(contract)
    header = spec.header(used_workspaces=("work_items", "item_counts"))
    symbols = dict(contract.instance.symbols)
    work_items = spec.workspace("work_items")
    assert f"#define SF_WS_WORK_ITEMS_CAPACITY_ELEMS {work_items.capacity}u" in header
    assert "uint32_t *work_items;" in header
    # The mapped/metadata split is the declared policy, not a candidate guess.
    assert spec.input("cu_seqlens").mapped is True
    assert spec.input("q").mapped is False
    assert spec.input("q").dtype == "fp16"
    # `log_decay_tab` is not allocated when the source does not name it.
    assert "#define SF_HOST_USE_LOG_DECAY_TAB 0" in header
    assert "#define SF_HOST_USE_WORK_ITEMS 1" in header
    # The full statement header (all workspaces) is still complete.
    full = spec.header()
    assert "#define SF_HOST_USE_LOG_DECAY_TAB 1" in full
    assert "SF_HOST_OK 0" in header
    assert "host_tiling(const SfHostInputs *inputs" in header
    assert work_items.capacity == symbols["num_seqs"] * symbols["H"] * 3 * 2


def test_the_python_layout_matches_the_compiled_header():
    """@brief 生成的 static assert 与 Python 偏移必须一致；编译通过即两者一致。"""
    contract = _contract()
    spec, build = _build("mixed", PLANNER_SOURCE, MAX_MAPPED_INPUT_BYTES)
    layout = spec.layout()
    assert layout.abi_bytes % 4 == 0
    assert layout.inputs_offset == len(spec.shape_fields) * 4
    assert layout.workspace_fields[0].ptr_offset == layout.workspaces_offset
    # Inputs and workspaces live in one storage array with a guard between every region; the
    # views handed to the program are disjoint by construction.
    regions = sorted(
        storage_layout(spec, ("work_items", "item_counts", "handoff_flags")), key=lambda region: region.data_offset
    )
    assert {region.kind for region in regions} == {"input", "workspace"}
    for left, right in zip(regions, regions[1:]):
        assert left.guard_hi + STORAGE_GUARD_BYTES <= right.guard_lo
    # The module was linked with the header; the planner ran with the Python-side offsets.
    runtime = HostRuntime(build, spec)
    runtime.set_target({"max_threads_per_threadgroup": 1024})
    raw = _raw_inputs(contract)
    run = runtime.execute(_reader(raw), values=None, audit=True)
    assert run.ok, run.error_detail
    assert {span.name for span in run.spans} == {"work_items", "item_counts", "handoff_flags"}


def test_referenced_workspaces_are_the_source_names_not_the_declaration_list():
    """@brief 未引用的 workspace 不分配（384 MiB 表不会因为声明存在而被实例化）。"""
    contract = _contract()
    spec, _abi = _spec(contract)
    assert referenced_workspaces(PLANNER_SOURCE, spec) == ("work_items", "item_counts", "handoff_flags")
    small = (
        "int32_t host_tiling(const SfHostInputs *i, const SfHostShapes *s, const SfHostTarget *t,\n"
        "                    SfHostWorkspaces *workspaces, SfHostResolved *r) {\n"
        "    (void)i; (void)s; (void)t; (void)r;\n"
        "    workspaces->work_items_count = 0u;\n"
        "    return SF_HOST_OK;\n"
        "}\n"
    )
    assert referenced_workspaces(small, spec) == ("work_items",)


# ---------------------------------------------------------------------------------------------
# Static precheck / compile diagnostics
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "source, needle",
    [
        ("#include <stdio.h>\nint32_t host_tiling(void){return 0;}", "#include"),
        ("int32_t host_tiling(void){void *p = malloc(4); return p != 0;}", "malloc("),
        ("int32_t host_tiling(void){int (*f)(void) = 0; return f();}", "function pointer"),
        (
            '__attribute__((import_module("env"), import_name("evil"))) extern int evil(void);\n'
            "int32_t host_tiling(void){return 0;}",
            "__attribute__",
        ),
        ("int32_t host_tiling(void){FILE *f = 0; return f != 0;}", "FILE"),
    ],
)
def test_the_projection_precheck_refuses_escape_hatches_without_a_compiler(source, needle):
    """@brief 投影只做静态预检，不调用 Clang；这些源码在编译前就被拒绝。"""
    problems = static_precheck(source)
    assert problems, source
    assert any(needle in problem for problem in problems), problems
    contract = _contract()
    spec, _abi = _spec(contract)
    preflight = preflight_host_program(
        source, spec=spec, declared_workspaces={view.name for view in spec.workspaces}
    )
    assert preflight


def test_the_fixture_passes_the_static_precheck():
    """@brief 真实候选 fixture 不应被预检误杀。"""
    contract = _contract()
    spec, _abi = _spec(contract)
    assert static_precheck(PLANNER_SOURCE) == ()
    assert (
        preflight_host_program(
            PLANNER_SOURCE, spec=spec, declared_workspaces={view.name for view in spec.workspaces}
        )
        == ()
    )


def test_a_compile_diagnostic_names_the_line_and_one_scripted_edit_repairs_it():
    """@brief 脚本化假候选根据编译诊断修复一处错误（离线，不花模型费）。"""
    broken = _mutate(
        PLANNER_SOURCE,
        "workspaces->work_items_count = item_count * 2u;",
        "workspaces->work_items_countt = item_count * 2u;",
    )
    contract = _contract()
    spec, _abi = _spec(contract)
    with pytest.raises(HostProgramError) as caught:
        compile_host_program(broken, spec=spec, entry_path="host/tiling.c", toolchain=_toolchain())
    diagnostic = str(caught.value)
    assert "work_items_countt" in diagnostic
    assert "host_program.c:" in diagnostic or "error:" in diagnostic
    # The scripted repair: rename the field the diagnostic named, nothing else.
    repaired = broken.replace("work_items_countt", "work_items_count")
    build = compile_host_program(repaired, spec=spec, entry_path="host/tiling.c", toolchain=_toolchain())
    assert build.wasm_sha256 and build.assembled_sha256


def test_a_missing_toolchain_is_a_typed_capability_refusal(monkeypatch):
    """@brief 缺 clang 是 ``backend_capability_unavailable``，不是候选错误也不是裸异常。"""
    import tcenv.hostprogram.toolchain as toolchain

    monkeypatch.setattr(toolchain, "_CLANG_CANDIDATES", ())
    monkeypatch.setattr(toolchain.shutil, "which", lambda _name: None)
    monkeypatch.delenv("SF_HOST_CLANG", raising=False)
    with pytest.raises(HostCapabilityUnavailable) as caught:
        toolchain.discover_toolchain()
    assert caught.value.code == "backend_capability_unavailable"


# ---------------------------------------------------------------------------------------------
# Atomic commit / bounds / imports
# ---------------------------------------------------------------------------------------------


def test_a_nonzero_return_commits_no_span_or_resolved_scalar():
    """@brief 返回非 0：写过的 span 也全部丢弃。"""
    source = _mutate(PLANNER_SOURCE, "return sf_plan_canonical(inputs, shapes, workspaces);", "return SF_HOST_REFUSED;")
    spec, _build_, runtime = _runtime("mixed", source)
    contract = _contract()
    run = runtime.execute(_reader(_raw_inputs(contract)), values=None, audit=True)
    assert run.status == 1
    assert run.spans == ()
    assert run.resolved_scalars == {}
    assert run.error_code == "host_program_invalid"


def test_a_span_beyond_the_declared_capacity_is_refused():
    """@brief offset+count 超过 capacity：整个 host run 丢弃。"""
    source = _mutate(
        PLANNER_SOURCE,
        "workspaces->work_items_count = item_count * 2u;",
        "workspaces->work_items_count = SF_WS_WORK_ITEMS_CAPACITY_ELEMS + 1u;",
    )
    _spec_, _build_, runtime = _runtime("mixed", source)
    contract = _contract()
    with pytest.raises(HostProgramError) as caught:
        runtime.execute(_reader(_raw_inputs(contract)), values=None, audit=True)
    assert "exceeds its declared capacity" in str(caught.value)


def test_writing_outside_the_allocated_region_hits_the_guard_sentinel():
    """@brief 跨块写会被哨兵抓到，而不是悄悄污染邻块。"""
    source = _mutate(
        PLANNER_SOURCE,
        "workspaces->item_counts[0] = (int32_t)item;",
        "workspaces->item_counts[-8] = (int32_t)item;",
    )
    _spec_, _build_, runtime = _runtime("mixed", source)
    contract = _contract()
    with pytest.raises(HostProgramError) as caught:
        runtime.execute(_reader(_raw_inputs(contract)), values=None, audit=True)
    assert "guard sentinel" in str(caught.value)


def test_a_read_only_input_cannot_be_modified():
    """@brief host program 没有输入写权限；映射输入被改写必须拒绝。"""
    source = _mutate(
        PLANNER_SOURCE,
        "int32_t *cu = inputs->cu_seqlens;\n    uint32_t chains = shapes->num_seqs * shapes->H;\n    uint32_t item = 0u;",
        "int32_t *cu = inputs->cu_seqlens; cu[0] = 1;\n    uint32_t chains = shapes->num_seqs * shapes->H;\n    uint32_t item = 0u;",
    )
    _spec_, _build_, runtime = _runtime("mixed", source)
    contract = _contract()
    with pytest.raises(HostProgramError) as caught:
        runtime.execute(_reader(_raw_inputs(contract)), values=None, audit=True)
    assert "read-only" in str(caught.value) or "modified" in str(caught.value)


def test_fuel_exhaustion_is_a_typed_candidate_failure_not_a_hang():
    """@brief 不终止的 planner 被 fuel 拒绝；span 为空。"""
    source = _mutate(
        PLANNER_SOURCE,
        "int32_t status = sf_validate(inputs, shapes);",
        "for (;;) { }\n    int32_t status = sf_validate(inputs, shapes);",
    )
    _spec_, _build_, runtime = _runtime("mixed", source)
    contract = _contract()
    run = runtime.execute(_reader(_raw_inputs(contract)), values=None, audit=True)
    assert run.status != 0
    assert "fuel" in run.error_detail
    assert run.spans == ()


def test_large_inputs_are_only_read_through_the_whitelisted_imports():
    """@brief mapped_limit=0 → 全部元数据；cu_seqlens 只能走 import，且区间被记录。"""
    read_import_source = r"""
int32_t host_tiling(const SfHostInputs *inputs, const SfHostShapes *shapes,
                    const SfHostTarget *target, SfHostWorkspaces *workspaces,
                    SfHostResolved *resolved) {
    (void)target; (void)resolved;
    int32_t bounds[32];
    int32_t status = sf_input_read_i32(inputs->cu_seqlens.id, 0u,
                                       (uint32_t)(uintptr_t)bounds, inputs->cu_seqlens.numel);
    if (status != SF_HOST_OK) {
        return SF_HOST_IMPORT_REFUSED;
    }
    if (bounds[0] != 0 || bounds[shapes->num_seqs] != (int32_t)shapes->total_tokens) {
        return SF_HOST_INVALID_BOUNDS;
    }
    workspaces->item_counts[0] = (int32_t)bounds[1];
    workspaces->item_counts_offset = 0u;
    workspaces->item_counts_count = 1u;
    return SF_HOST_OK;
}
"""
    spec, _build_, runtime = _runtime("mixed", read_import_source, mapped_limit=0)
    assert all(not view.mapped for view in spec.inputs)
    contract = _contract()
    run = runtime.execute(_reader(_raw_inputs(contract)), values=None, audit=True)
    assert run.ok, run.error_detail
    assert [read.name for read in run.input_reads] == ["cu_seqlens"]
    assert run.input_reads[0].count == contract.instance.symbols["NSP1"]
    assert run.input_reads[0].sha256 is not None


def test_an_import_refuses_a_wrong_id_dtype_and_range():
    """@brief id/dtype/范围由 harness 校验；错误 import 返回非 0。"""
    source = r"""
int32_t host_tiling(const SfHostInputs *inputs, const SfHostShapes *shapes,
                    const SfHostTarget *target, SfHostWorkspaces *workspaces,
                    SfHostResolved *resolved) {
    (void)target; (void)resolved; (void)shapes; (void)workspaces;
    int32_t value = 0;
    /* wrong range: one element past the declared numel */
    if (sf_input_read_i32(inputs->cu_seqlens.id, 1u, (uint32_t)(uintptr_t)&value, inputs->cu_seqlens.numel) == 0) {
        return 1;
    }
    /* wrong id */
    if (sf_input_read_i32(9999u, 0u, (uint32_t)(uintptr_t)&value, 1u) == 0) {
        return 2;
    }
    /* wrong dtype for this import */
    if (sf_input_read_f32(inputs->cu_seqlens.id, 0u, (uint32_t)(uintptr_t)&value, 1u) == 0) {
        return 3;
    }
    return SF_HOST_OK;
}
"""
    _spec_, _build_, runtime = _runtime("mixed", source, mapped_limit=0)
    contract = _contract()
    run = runtime.execute(_reader(_raw_inputs(contract)), values=None, audit=True)
    assert run.ok, run.error_detail
    assert run.input_reads == ()


def test_an_unreferenced_workspace_is_not_allocated_and_touching_it_traps():
    """@brief 未分配块指针为 0：引用它要么预检拒绝，要么在 Wasm 里 trap，不会越界写真实内存。"""
    source = _mutate(
        PLANNER_SOURCE,
        "workspaces->item_counts[0] = (int32_t)item;",
        "workspaces->item_counts[0] = (int32_t)item; workspaces->handoff_states[0] = 0.0f;",
    )
    contract = _contract()
    spec, _abi = _spec(contract)
    # ``compile_host_program`` derives usage from the source; force only the referenced-by-header
    # set to prove the other workspace is genuinely unallocated.
    build = compile_host_program(
        source,
        spec=spec,
        entry_path="host/tiling.c",
        toolchain=_toolchain(),
        used_workspaces=("work_items", "item_counts", "handoff_flags"),
    )
    runtime = HostRuntime(build, spec)
    runtime.set_target({"max_threads_per_threadgroup": 1024})
    run = runtime.execute(_reader(_raw_inputs(contract)), values=None, audit=True)
    assert run.status != 0
    assert run.spans == ()


def test_the_module_is_compiled_once_and_each_call_is_deterministic():
    """@brief 同一 Module 复用、每次新 Store；同一输入给出同一 span/摘要。"""
    _spec_, build, runtime = _runtime("mixed")
    contract = _contract()
    raw = _raw_inputs(contract)
    first = runtime.execute(_reader(raw), values=None, audit=True)
    second = runtime.execute(_reader(raw), values=None, audit=True)
    assert first.ok and second.ok
    assert [span.data for span in first.spans] == [span.data for span in second.spans]
    assert [span.sha256 for span in first.spans] == [span.sha256 for span in second.spans]
    assert runtime.module is runtime.module
    # A timed run skips hashing but copies the same bytes.
    timed = runtime.execute(_reader(raw), values=None, audit=False)
    assert not timed.audit
    assert [span.data for span in timed.spans] == [span.data for span in first.spans]
    assert all(span.sha256 is None for span in timed.spans)


def test_the_assembled_digest_covers_header_and_source():
    """@brief 头或源码任一字节变化都改变 ``assembled_host_sha256``。"""
    contract = _contract()
    spec, _abi = _spec(contract)
    base = compile_host_program(PLANNER_SOURCE, spec=spec, entry_path="host/tiling.c", toolchain=_toolchain())
    changed = compile_host_program(
        PLANNER_SOURCE + "\n/* one byte more */\n", spec=spec, entry_path="host/tiling.c", toolchain=_toolchain()
    )
    assert base.assembled_sha256 != changed.assembled_sha256
    assert base.program_sha256 != changed.program_sha256
    # A trailing comment changes the assembled/source identity but not the compiled code, so the
    # module digest is stable: the two digests answer different questions.
    assert base.wasm_sha256 == changed.wasm_sha256


def test_fuel_and_memory_budgets_scale_with_the_declared_capacity():
    """@brief fuel/内存是 capacity 的函数，且 V3 的 KDA 最大档有足够余量。"""
    mixed = _contract("mixed")
    fixed = _contract("fixed")
    mixed_spec, _a = _spec(mixed)
    fixed_spec, _b = _spec(fixed)
    assert fuel_budget(mixed_spec) > fuel_budget(fixed_spec)
    assert memory_budget_bytes(mixed_spec, ("work_items", "item_counts")) >= memory_budget_bytes(
        fixed_spec, ("work_items", "item_counts")
    )
    assert memory_budget_bytes(mixed_spec, ("work_items", "item_counts")) % 65536 == 0
    # The mixed tier's planner uses a tiny fraction of the frozen budget.
    _spec_, _build_, runtime = _runtime("mixed")
    run = runtime.execute(_reader(_raw_inputs(mixed)), values=None, audit=True)
    assert run.ok
    assert run.fuel_used < fuel_budget(mixed_spec) // 4


def test_the_statement_shows_the_full_header_and_return_codes_only_for_varlen():
    """@brief varlen 题面含完整 host_contract.h/子集/返回码；固定 shape 题面不变。"""
    varlen = _contract()
    assert "## Host Program (restricted C99 -> Wasm, optional)" in varlen.statement
    assert "### Generated `host_contract.h` (full ABI)" in varlen.statement
    assert "SF_HOST_IMPORT_REFUSED" in varlen.statement
    assert "int32_t *cu_seqlens;" in varlen.statement
    fixed = load_task_contract(REPO_ROOT / "tasks" / "rmsnorm")
    assert "Host Program (restricted C99" not in fixed.statement
