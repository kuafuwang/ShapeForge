"""Native64 Host Program contract, restricted C99 precheck and worker runtime (CPU only).

The C compiler owns structure layout; Python uses generated accessors.  Tests cover compile
repair, atomic span commit, guard/read-only/pointer violations, typed large-input reads, fresh
replay, native build identity and independent work-list oracles.  Runaway/crash containment is
exercised in a separate bounded worker, never in the pytest control process.
"""

from __future__ import annotations

from functools import lru_cache
import ctypes
import hashlib
import json
from pathlib import Path
import struct
from types import SimpleNamespace

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
from tcenv.hostprogram.native_abi import assemble_native_source, render_native_support
from tcenv.hostprogram.native_runtime import NativeHostRuntime
from tcenv.hostprogram.native_toolchain import (
    compile_native_host_program,
    discover_native_toolchain,
    native_memory_budget_bytes,
    recall_native_build,
    remember_native_build,
)
from tcenv.taskcontract import load_task_contract
from tcenv.taskcontract.locator import REPO_ROOT
from tcenv.track.abi import abi_limits_from_target, derive_tensor_memory_contract
from tcenv.track.geometry import resolve_geometry
from tcenv.track.plane import execution_geometry_from_document
from tcenv.target import load_target
from tcenv.tensors import round_to, to_raw

from ..support.hostprogram_fixtures import PLANNER_SOURCE

TASK = REPO_ROOT / "tasks" / "varlen_kda"
TARGET = load_target("apple-m4-max")


def _contract(size: str = "mixed"):
    return load_task_contract(TASK, dtype="fp16", size=size)


def _toolchain():
    try:
        return discover_native_toolchain()
    except HostCapabilityUnavailable as exc:  # pragma: no cover - depends on the host
        pytest.skip(f"native Host Program toolchain unavailable ({exc}); install Clang/Xcode CLT")


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
    build = compile_native_host_program(source, spec=spec, entry_path="host/tiling.c", toolchain=_toolchain())
    return spec, build


def _runtime(size: str, source: str = PLANNER_SOURCE, *, mapped_limit: int = MAX_MAPPED_INPUT_BYTES):
    spec, build = _build(size, source, mapped_limit)
    runtime = NativeHostRuntime(build, spec)
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


def _bounded_native_probe(source: str, *, timeout: float = 4.0, launch_deadline_s: float = 1.0):
    from tcenv.hostprogram.native_probe_worker import PROBE_SCHEMA, run_bounded_probe

    return run_bounded_probe(
        {
            "schema_id": PROBE_SCHEMA,
            "task": "varlen_kda",
            "dtype": "fp16",
            "size": "mixed",
            "case": "canonical",
            "seed": 7,
            "target_id": "apple-m4-max",
            "source": source,
            "entry_path": "host/tiling.c",
        },
        timeout=timeout,
        launch_deadline_s=launch_deadline_s,
    )


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
    assert render_native_support(from_instance) == render_native_support(from_abi)
    assert "int32_t *cu_seqlens;" in render_native_support(from_instance)
    assert "uint32_t work_items;" not in render_native_support(from_instance)


def test_the_header_exposes_declared_names_types_and_capacities_only():
    """@brief 强类型字段、cap 宏和返回码来自 OpSpec/ABI，不来自候选。"""
    contract = _contract()
    spec, _abi = _spec(contract)
    header = render_native_support(spec, ("work_items", "item_counts"))
    symbols = dict(contract.instance.symbols)
    work_items = spec.workspace("work_items")
    assert f"#define SF_WS_WORK_ITEMS_CAPACITY_ELEMS {work_items.capacity}u" in header
    assert "uint32_t *work_items;" in header
    # The mapped/metadata split is the declared policy, not a candidate guess.  fp16/bf16 use
    # an explicit raw-bit uint16_t view, so a 448-byte q is still directly mapped.
    assert spec.input("cu_seqlens").mapped is True
    assert spec.input("q").mapped is True
    assert spec.input("q").dtype == "fp16"
    assert spec.input("q").ctype == "uint16_t"
    assert "uint16_t *q;" in header
    # `log_decay_tab` is not allocated when the source does not name it.
    assert "#define SF_HOST_USE_LOG_DECAY_TAB 0u" in header
    assert "#define SF_HOST_USE_WORK_ITEMS 1u" in header
    # The full statement header (all workspaces) is still complete.
    full = render_native_support(spec)
    assert "#define SF_HOST_USE_LOG_DECAY_TAB 1u" in full
    assert "SF_HOST_OK 0" in header
    assert "host_tiling(const SfHostInputs *inputs" in header
    assert work_items.capacity == symbols["num_seqs"] * symbols["H"] * 3 * 2


def test_native_compiler_owns_layout_and_regions_are_disjoint():
    """@brief native64 由 C static assert 锁指针宽度；Python 只使用 accessor，不猜偏移。"""
    contract = _contract()
    spec, build = _build("mixed", PLANNER_SOURCE, MAX_MAPPED_INPUT_BYTES)
    assert "sizeof(void *) == 8u" in build.header
    assert "SF_ABI_SHAPES" not in build.header
    assert build.assembled_sha256 == hashlib.sha256(
        assemble_native_source(build.header, PLANNER_SOURCE).encode("utf-8")
    ).hexdigest()
    regions = sorted(
        storage_layout(spec, ("work_items", "item_counts", "handoff_flags")), key=lambda region: region.data_offset
    )
    assert {region.kind for region in regions} == {"input", "workspace"}
    for left, right in zip(regions, regions[1:]):
        assert left.guard_hi + STORAGE_GUARD_BYTES <= right.guard_lo
    runtime = NativeHostRuntime(build, spec)
    runtime.set_target({"max_threads_per_threadgroup": 1024})
    run = runtime.execute(_reader(_raw_inputs(contract)), values=None, audit=True)
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
        ("int32_t host_tiling(void){return sf_native_invoke();}", "harness-internal native"),
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
        compile_native_host_program(broken, spec=spec, entry_path="host/tiling.c", toolchain=_toolchain())
    diagnostic = str(caught.value)
    assert "work_items_countt" in diagnostic
    assert "host_program.c:" in diagnostic or "error:" in diagnostic
    # The scripted repair: rename the field the diagnostic named, nothing else.
    repaired = broken.replace("work_items_countt", "work_items_count")
    build = compile_native_host_program(repaired, spec=spec, entry_path="host/tiling.c", toolchain=_toolchain())
    assert build.native_sha256 and build.assembled_sha256


def test_native_link_boundary_refuses_an_ambient_libsystem_call():
    """@brief 名字扫描之外的 libSystem 调用也会被最终未解析符号检查拒绝。"""
    source = r"""
extern int getsid(int);
int32_t host_tiling(const SfHostInputs *inputs, const SfHostShapes *shapes,
                    const SfHostTarget *target, SfHostWorkspaces *workspaces,
                    SfHostResolved *resolved) {
    (void)inputs; (void)shapes; (void)target; (void)workspaces; (void)resolved;
    return getsid(0);
}
"""
    spec, _abi = _spec(_contract())
    with pytest.raises(HostProgramError, match="unresolved host symbols"):
        compile_native_host_program(source, spec=spec, entry_path="host/tiling.c", toolchain=_toolchain())


def test_native_link_boundary_refuses_a_candidate_custom_export():
    source = r"""
__attribute__((visibility("default"))) int32_t candidate_escape(void) { return 1; }
int32_t host_tiling(const SfHostInputs *inputs, const SfHostShapes *shapes,
                    const SfHostTarget *target, SfHostWorkspaces *workspaces,
                    SfHostResolved *resolved) {
    (void)inputs; (void)shapes; (void)target; (void)workspaces; (void)resolved;
    return SF_HOST_OK;
}
"""
    spec, _abi = _spec(_contract())
    with pytest.raises(HostProgramError, match="non-harness symbols"):
        compile_native_host_program(source, spec=spec, entry_path="host/tiling.c", toolchain=_toolchain())


def test_native_link_boundary_refuses_excess_static_storage():
    source = r"""
static volatile uint8_t oversized[1024u * 1024u];
int32_t host_tiling(const SfHostInputs *inputs, const SfHostShapes *shapes,
                    const SfHostTarget *target, SfHostWorkspaces *workspaces,
                    SfHostResolved *resolved) {
    (void)inputs; (void)shapes; (void)target; (void)resolved;
    oversized[0] = 1u;
    workspaces->work_items[0] = oversized[0];
    workspaces->work_items_count = 1u;
    return SF_HOST_OK;
}
"""
    spec, _abi = _spec(_contract())
    with pytest.raises(HostProgramError, match="static storage.*exceeds"):
        compile_native_host_program(source, spec=spec, entry_path="host/tiling.c", toolchain=_toolchain())


def test_a_missing_toolchain_is_a_typed_capability_refusal(monkeypatch):
    """@brief 缺 clang 是 ``backend_capability_unavailable``，不是候选错误也不是裸异常。"""
    import tcenv.hostprogram.native_toolchain as toolchain

    real_is_file = Path.is_file
    monkeypatch.setattr(toolchain.shutil, "which", lambda _name: None)
    monkeypatch.setattr(Path, "is_file", lambda self: False if self.name == "clang" else real_is_file(self))
    monkeypatch.delenv("SF_HOST_CLANG", raising=False)
    with pytest.raises(HostCapabilityUnavailable) as caught:
        toolchain.discover_native_toolchain()
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


def test_runaway_native_planner_is_killed_by_the_external_deadline():
    """@brief 不终止的 planner 由 bounded worker 杀死，不悬挂控制机、不发布 span。"""
    source = _mutate(
        PLANNER_SOURCE,
        "int32_t status = sf_validate(inputs, shapes);",
        "for (;;) { }\n    int32_t status = sf_validate(inputs, shapes);",
    )
    result = _bounded_native_probe(source)
    report = result.get("report") or result
    assert report["status"] == "timeout"
    assert "external deadline" in report.get("message", "")
    assert not report.get("workspace_writes")


def test_large_inputs_are_only_read_through_the_whitelisted_imports():
    """@brief mapped_limit=0 → 全部元数据；cu_seqlens 只能走 import，且区间被记录。"""
    read_import_source = r"""
int32_t host_tiling(const SfHostInputs *inputs, const SfHostShapes *shapes,
                    const SfHostTarget *target, SfHostWorkspaces *workspaces,
                    SfHostResolved *resolved) {
    (void)target; (void)resolved;
    int32_t bounds[32];
    int32_t status = sf_input_read_i32(inputs->cu_seqlens.id, 0u,
                                       (uintptr_t)bounds, inputs->cu_seqlens.numel);
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
    if (sf_input_read_i32(inputs->cu_seqlens.id, 1u, (uintptr_t)&value, inputs->cu_seqlens.numel) == 0) {
        return 1;
    }
    /* wrong id */
    if (sf_input_read_i32(9999u, 0u, (uintptr_t)&value, 1u) == 0) {
        return 2;
    }
    /* wrong dtype for this import */
    if (sf_input_read_f32(inputs->cu_seqlens.id, 0u, (uintptr_t)&value, 1u) == 0) {
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


def test_an_unreferenced_workspace_cannot_publish_a_span():
    """@brief 未分配块指针为 0；即使只伪造非空 span，也必须原子拒绝。"""
    source = _mutate(
        PLANNER_SOURCE,
        "workspaces->item_counts[0] = (int32_t)item;",
        "workspaces->item_counts[0] = (int32_t)item; workspaces->handoff_states_count = 1u;",
    )
    contract = _contract()
    spec, _abi = _spec(contract)
    build = compile_native_host_program(
        source,
        spec=spec,
        entry_path="host/tiling.c",
        toolchain=_toolchain(),
        used_workspaces=("work_items", "item_counts", "handoff_flags"),
    )
    runtime = NativeHostRuntime(build, spec)
    runtime.set_target({"max_threads_per_threadgroup": 1024})
    with pytest.raises(HostProgramError, match="unallocated workspace"):
        runtime.execute(_reader(_raw_inputs(contract)), values=None, audit=True)


def test_native_build_loads_dylib_once_and_reuses_it_across_calls(monkeypatch):
    """@brief build 只 CDLL 一次；audit/timed 复用同一 library，但每次重新校验 fresh 调用状态。"""
    import tcenv.hostprogram.native_toolchain as host_toolchain

    real_cdll = host_toolchain.ctypes.CDLL
    loads: list[str] = []

    def counting_cdll(path):
        loads.append(str(path))
        return real_cdll(path)

    monkeypatch.setattr(host_toolchain.ctypes, "CDLL", counting_cdll)
    _spec_, build, runtime = _runtime("mixed", PLANNER_SOURCE + "\n/* native-load-count-probe */\n")
    assert len(loads) == 1
    assert runtime._library is build.library
    raw = _raw_inputs(_contract())
    first = runtime.execute(_reader(raw), values=None, audit=True)
    second = runtime.execute(_reader(raw), values=None, audit=True)
    assert first.ok and second.ok
    assert [span.data for span in first.spans] == [span.data for span in second.spans]
    assert [span.sha256 for span in first.spans] == [span.sha256 for span in second.spans]
    timed = runtime.execute(_reader(raw), values=None, audit=False)
    assert not timed.audit
    assert [span.data for span in timed.spans] == [span.data for span in first.spans]
    assert all(span.sha256 is None for span in timed.spans)
    assert len(loads) == 1, "execution reconstructed the dylib instead of reusing the build's library"
    host_toolchain._BUILT.clear()
    _build.cache_clear()


def test_a_host_program_may_not_repoint_a_workspace_at_an_input():
    """@brief 调用后改写 workspace pointer → 原子丢弃（输入/输出不重叠）。"""
    source = r"""
int32_t host_tiling(const SfHostInputs *inputs, const SfHostShapes *shapes,
                    const SfHostTarget *target, SfHostWorkspaces *workspaces,
                    SfHostResolved *resolved) {
    (void)shapes; (void)target; (void)resolved;
    workspaces->work_items = (uint32_t *)(uintptr_t)inputs->cu_seqlens;
    workspaces->work_items_offset = 0u;
    workspaces->work_items_count = 2u;
    return SF_HOST_OK;
}
"""
    spec, _build, runtime = _runtime("mixed", source)
    with pytest.raises(HostProgramError, match="repointed|outside"):
        runtime.execute(_reader(_raw_inputs(_contract())), values=None, audit=True)


def test_an_import_may_not_write_back_into_the_abi_pointer():
    """@brief import 的 dst_off 写回 ABI pointer 字段 → 同样原子丢弃。"""
    source = r"""
int32_t host_tiling(const SfHostInputs *inputs, const SfHostShapes *shapes,
                    const SfHostTarget *target, SfHostWorkspaces *workspaces,
                    SfHostResolved *resolved) {
    (void)shapes; (void)target; (void)resolved;
    int32_t status = sf_input_read_i32(inputs->cu_seqlens.id, 0u,
                                       (uintptr_t)&workspaces->work_items, 1u);
    if (status != SF_HOST_OK) {
        return status;
    }
    workspaces->work_items_offset = 0u;
    workspaces->work_items_count = 2u;
    return SF_HOST_OK;
}
"""
    spec, _build, runtime = _runtime("mixed", source, mapped_limit=0)
    with pytest.raises(HostProgramError, match="repointed|outside"):
        runtime.execute(_reader(_raw_inputs(_contract())), values=None, audit=True)


@pytest.mark.parametrize(
    "needle",
    [
        "SfHostWorkspaces *w = workspaces; (void)w;",
        "workspaces = (SfHostWorkspaces *)(void *)workspaces;",
        "void *p = (void *)&workspaces; (void)p;",
    ],
)
def test_aliasing_or_addressing_the_workspaces_parameter_is_refused(needle):
    """@brief 名字扫描决定分配与 device-zero 所有权；别名/取地址必须显式拒绝。"""
    source = (
        "int32_t host_tiling(const SfHostInputs *i, const SfHostShapes *s, const SfHostTarget *t,\n"
        "                    SfHostWorkspaces *workspaces, SfHostResolved *r) {\n"
        "    (void)i; (void)s; (void)t; (void)r;\n"
        f"    {needle}\n"
        "    return SF_HOST_OK;\n"
        "}\n"
    )
    problems = static_precheck(source)
    assert any("workspaces" in problem for problem in problems), problems


def test_the_assembled_digest_covers_header_and_source():
    """@brief 头或源码任一字节变化都改变 ``assembled_host_sha256``。"""
    contract = _contract()
    spec, _abi = _spec(contract)
    base = compile_native_host_program(PLANNER_SOURCE, spec=spec, entry_path="host/tiling.c", toolchain=_toolchain())
    changed = compile_native_host_program(
        PLANNER_SOURCE + "\n/* one byte more */\n", spec=spec, entry_path="host/tiling.c", toolchain=_toolchain()
    )
    assert base.assembled_sha256 != changed.assembled_sha256
    assert base.program_sha256 != changed.program_sha256
    # A trailing comment changes the assembled/source identity but not the compiled code, so the
    # module digest is stable: the two digests answer different questions.
    assert base.native_sha256 == changed.native_sha256


def test_native_memory_budget_scales_with_the_declared_capacity():
    """@brief native 存储预算是 capacity 的函数；runaway 由外部 watchdog 而不是 fuel 处理。"""
    mixed = _contract("mixed")
    fixed = _contract("fixed")
    mixed_spec, _a = _spec(mixed)
    fixed_spec, _b = _spec(fixed)
    mixed_budget = native_memory_budget_bytes(mixed_spec, ("work_items", "item_counts"))
    fixed_budget = native_memory_budget_bytes(fixed_spec, ("work_items", "item_counts"))
    assert mixed_budget >= fixed_budget
    assert mixed_budget % 65536 == 0
    _spec_, build, runtime = _runtime("mixed")
    run = runtime.execute(_reader(_raw_inputs(mixed)), values=None, audit=True)
    assert run.ok
    assert "fuel_used" not in run.document()
    assert build.memory_budget_bytes == mixed_budget


def test_the_statement_shows_the_full_header_and_return_codes_only_for_varlen():
    """@brief varlen 题面含完整 host_contract.h/子集/返回码；固定 shape 题面不变。"""
    varlen = _contract()
    assert "## Host Program (restricted C99 -> native64 worker, optional)" in varlen.statement
    assert "sf_native_invoke" in varlen.statement
    assert "Wasm32 offset" in varlen.statement
    assert "### Generated `host_contract.h` (full ABI)" in varlen.statement
    assert "SF_HOST_IMPORT_REFUSED" in varlen.statement
    assert "int32_t *cu_seqlens;" in varlen.statement
    assert "### Minimal complete host+device example" in varlen.statement
    assert "kernel void example" in varlen.statement
    fixed = load_task_contract(REPO_ROOT / "tasks" / "rmsnorm")
    assert "Host Program (restricted C99" not in fixed.statement


def test_the_build_cache_is_keyed_by_the_full_identity_not_only_the_module_bytes():
    """@brief 仅加注释的 native bytes 相同、assembled/source 不同；缓存不能返回错的 build。"""
    from tcenv.hostprogram.native_toolchain import compile_native_host_program, recall_native_build, remember_native_build

    contract = _contract()
    spec, _abi = _spec(contract)
    base = compile_native_host_program(PLANNER_SOURCE, spec=spec, entry_path="host/tiling.c", toolchain=_toolchain())
    commented = PLANNER_SOURCE + "\n/* comment-only identity */\n"
    changed = compile_native_host_program(commented, spec=spec, entry_path="host/tiling.c", toolchain=_toolchain())
    assert base.native_sha256 == changed.native_sha256
    assert base.assembled_sha256 != changed.assembled_sha256
    assert base.program_sha256 != changed.program_sha256
    remember_native_build(base)
    remember_native_build(changed)
    common = dict(
        memory_budget_bytes=base.memory_budget_bytes,
        used_workspaces=base.used_workspaces,
    )
    assert (
        recall_native_build(
            base.native_sha256,
            assembled_sha256=base.assembled_sha256,
            program_sha256=base.program_sha256,
            **common,
        )
        is base
    )
    assert (
        recall_native_build(
            changed.native_sha256,
            assembled_sha256=changed.assembled_sha256,
            program_sha256=changed.program_sha256,
            **common,
        )
        is changed
    )
    # A module digest alone must not hit any of the two builds.
    assert recall_native_build(base.native_sha256) is None


def test_read_write_state_is_not_exposed_as_a_host_input():
    """@brief 只有 access=read 的 global input 进入 Host Program；read_write 由 device 独占。"""
    contract = _contract()
    spec, abi = _spec(contract)
    rw = SimpleNamespace(name="rw_state", dtype="fp32", shape=("total_tokens",), access="read_write", space="global")
    with_rw = host_contract_spec(instance=contract.instance, tensors=tuple(abi.tensors) + (rw,))
    assert all(view.name != "rw_state" for view in with_rw.inputs)
    ro = SimpleNamespace(name="ro_state", dtype="fp32", shape=("total_tokens",), access="read", space="global")
    with_ro = host_contract_spec(instance=contract.instance, tensors=tuple(abi.tensors) + (ro,))
    assert any(view.name == "ro_state" for view in with_ro.inputs)


def test_fp16_and_bf16_are_mapped_as_raw_bits_and_readable_via_u16():
    """@brief fp16/bf16 无 C 算术类型：mapped 用 uint16_t 原始位，metadata 用 sf_input_read_u16。"""
    contract = _contract()
    spec, _abi = _spec(contract)
    for name in ("q", "k", "v", "g", "beta"):
        view = spec.input(name)
        assert view.mapped is True, name
        assert view.ctype == "uint16_t", name
        assert view.dtype == "fp16", name
    header = render_native_support(spec, ("work_items", "item_counts", "handoff_flags"))
    assert "uint16_t *q;" in header
    assert "uint16_t *beta;" in header
    assert "sf_input_read_u16" in header

    bf16 = load_task_contract(TASK, dtype="bf16", size="mixed")
    bf16_spec, _ = _spec(bf16)
    assert bf16_spec.input("q").mapped is True
    assert bf16_spec.input("q").ctype == "uint16_t"
    assert bf16_spec.input("q").dtype == "bf16"

    # metadata-only mode: read the raw 16-bit words through the import and record the semantic
    # dtype (fp16) plus the raw import name.
    source = r"""
int32_t host_tiling(const SfHostInputs *inputs, const SfHostShapes *shapes,
                    const SfHostTarget *target, SfHostWorkspaces *workspaces,
                    SfHostResolved *resolved) {
    (void)shapes; (void)target; (void)resolved;
    uint16_t raw[4];
    int32_t status = sf_input_read_u16(inputs->q.id, 0u, (uintptr_t)raw, 4u);
    if (status != SF_HOST_OK) {
        return status;
    }
    workspaces->work_items[0] = (uint32_t)raw[0];
    workspaces->work_items[1] = (uint32_t)raw[3];
    workspaces->work_items_offset = 0u;
    workspaces->work_items_count = 2u;
    return SF_HOST_OK;
}
"""
    spec_m, _build_m, runtime = _runtime("mixed", source, mapped_limit=0)
    raw_inputs = _raw_inputs(contract)
    run = runtime.execute(_reader(raw_inputs), values=None, audit=True)
    assert run.ok
    read = next(item for item in run.input_reads if item.name == "q")
    assert read.dtype == "fp16"
    assert read.import_name == "sf_input_read_u16"
    words = struct.unpack("<4H", raw_inputs["q"][:8])
    span = next(item for item in run.spans if item.name == "work_items")
    assert struct.unpack("<II", span.data[:8]) == (words[0], words[3])


def test_a_mapped_fp16_input_is_a_typed_uint16_pointer():
    """@brief 小 fp16 输入直接映射后，候选可按原始位读取 `inputs->q[0]`。"""
    source = r"""
int32_t host_tiling(const SfHostInputs *inputs, const SfHostShapes *shapes,
                    const SfHostTarget *target, SfHostWorkspaces *workspaces,
                    SfHostResolved *resolved) {
    (void)shapes; (void)target; (void)resolved;
    workspaces->work_items[0] = (uint32_t)inputs->q[0];
    workspaces->work_items[1] = (uint32_t)inputs->q[inputs->q_numel - 1u];
    workspaces->work_items_offset = 0u;
    workspaces->work_items_count = 2u;
    return SF_HOST_OK;
}
"""
    contract = _contract()
    _spec_, _build_, runtime = _runtime("mixed", source)
    raw_inputs = _raw_inputs(contract)
    run = runtime.execute(_reader(raw_inputs), values=None, audit=True)
    assert run.ok
    words = struct.unpack("<%dH" % (len(raw_inputs["q"]) // 2), raw_inputs["q"])
    span = next(item for item in run.spans if item.name == "work_items")
    assert struct.unpack("<II", span.data[:8]) == (words[0], words[-1])


def test_the_statement_host_example_compiles_runs_and_its_geometry_resolves():
    """@brief 题面的最小 host 示例可编译/执行，device 示例与几何可解析。"""
    from tcenv.hostprogram.contract import (
        minimal_device_example,
        minimal_geometry_example,
        minimal_host_example,
    )

    contract = _contract()
    spec, abi = _spec(contract)
    host = minimal_host_example(spec)
    from tcenv.hostprogram.native_runtime import NativeHostRuntime
    from tcenv.hostprogram.native_toolchain import compile_native_host_program

    build = compile_native_host_program(host, spec=spec, entry_path="example/host.c")
    runtime = NativeHostRuntime(build, spec)
    runtime.set_target({"max_threads_per_threadgroup": 1024})
    run = runtime.execute(_reader(_raw_inputs(contract)), values=None, audit=True)
    assert run.ok, run.error_detail
    assert {span.name for span in run.spans} == {"work_items", "item_counts", "handoff_flags"}
    device = minimal_device_example(spec)
    assert "kernel void example" in device
    geometry = execution_geometry_from_document(json.loads(minimal_geometry_example(spec)))
    resolved = resolve_geometry(
        geometry,
        instance=contract.instance,
        contract=abi,
        limits=abi_limits_from_target(TARGET),
    )
    assert resolved.active_stages[0].kernel == "example"


# ---------------------------------------------------------------------------------------------
# Independent native64 ABI / runtime (independent V2 work-list oracle)


def _native_runtime(size: str, source: str = PLANNER_SOURCE, *, mapped_limit: int = MAX_MAPPED_INPUT_BYTES):
    from tcenv.hostprogram.native_runtime import NativeHostRuntime
    from tcenv.hostprogram.native_toolchain import compile_native_host_program

    contract = _contract(size)
    spec, _abi = _spec(contract, mapped_limit=mapped_limit)
    build = compile_native_host_program(source, spec=spec, entry_path="host/tiling.c")
    runtime = NativeHostRuntime(build, spec)
    runtime.set_target({"max_threads_per_threadgroup": 1024})
    return spec, build, runtime


def test_native_planner_matches_independent_plan_and_rebuild_is_reproducible():
    from tcenv.hostprogram.native_toolchain import compile_native_host_program
    from ..support.npu_sim.varlen_worklist import pack_items, plan_pieces

    contract = _contract()
    spec, build, native = _native_runtime("mixed")
    rebuilt = compile_native_host_program(PLANNER_SOURCE, spec=spec, entry_path="host/tiling.c")
    assert rebuilt.native_sha256 == build.native_sha256
    raw = _raw_inputs(contract)
    bounds = struct.unpack(f"<{len(raw['cu_seqlens']) // 4}i", raw["cu_seqlens"])
    lengths = [bounds[i + 1] - bounds[i] for i in range(len(bounds) - 1)]
    expected = pack_items(plan_pieces(lengths, contract.instance.symbols["H"], 3))
    actual = native.execute(_reader(raw), audit=True)
    assert actual.ok, actual.error_detail
    assert next(span.data for span in actual.spans if span.name == "work_items") == expected
    assert build.descriptor()["execution_backend"] == "native-c99"
    assert "wasm_sha256" not in build.descriptor()


def test_native_nonzero_return_discards_every_span():
    source = r"""
int32_t host_tiling(const SfHostInputs *inputs, const SfHostShapes *shapes,
                    const SfHostTarget *target, SfHostWorkspaces *workspaces,
                    SfHostResolved *resolved) {
    (void)inputs; (void)shapes; (void)target; (void)resolved;
    workspaces->work_items[0] = 123u;
    workspaces->work_items_count = 1u;
    return 9;
}
"""
    contract = _contract()
    _spec_, _build_, runtime = _native_runtime("mixed", source)
    run = runtime.execute(_reader(_raw_inputs(contract)), audit=True)
    assert run.status == 9
    assert run.spans == ()
    assert run.resolved_scalars == {}


@pytest.mark.parametrize("violation", ["span", "guard", "input", "pointer", "shape", "target"])
def test_native_span_guard_readonly_and_repoint_violations_are_refused(violation):
    statements = {
        "span": "workspaces->work_items_count = workspaces->work_items_capacity + 1u;",
        "guard": "((uint8_t *)(uintptr_t)workspaces->work_items)[-1] = 0u;",
        "input": "inputs->cu_seqlens[0] = 1; workspaces->work_items_count = 0u;",
        "pointer": "workspaces->work_items = (uint32_t *)(uintptr_t)inputs->cu_seqlens;",
        "shape": "((SfHostShapes *)(uintptr_t)shapes)->H = 99u;",
        "target": "((SfHostTarget *)(uintptr_t)target)->max_threads_per_threadgroup = 1u;",
    }
    source = f"""
int32_t host_tiling(const SfHostInputs *inputs, const SfHostShapes *shapes,
                    const SfHostTarget *target, SfHostWorkspaces *workspaces,
                    SfHostResolved *resolved) {{
    (void)shapes; (void)target; (void)resolved; (void)inputs;
    {statements[violation]}
    return SF_HOST_OK;
}}
"""
    contract = _contract()
    _spec_, _build_, runtime = _native_runtime("mixed", source)
    with pytest.raises(HostProgramError):
        runtime.execute(_reader(_raw_inputs(contract)), audit=True)


def test_native_timed_direct_commit_writes_valid_span_without_python_bytes():
    source = r"""
int32_t host_tiling(const SfHostInputs *inputs, const SfHostShapes *shapes,
                    const SfHostTarget *target, SfHostWorkspaces *workspaces,
                    SfHostResolved *resolved) {
    (void)inputs; (void)shapes; (void)target; (void)resolved;
    workspaces->work_items[0] = 11u;
    workspaces->work_items[1] = 29u;
    workspaces->work_items_offset = 0u;
    workspaces->work_items_count = 2u;
    return SF_HOST_OK;
}
"""
    _spec_, _build_, runtime = _native_runtime("mixed", source)
    output = (ctypes.c_uint32 * 36)(*([0xCDCDCDCD] * 36))
    run = runtime.execute(
        _reader(_raw_inputs(_contract())),
        workspace_addresses={"work_items": ctypes.addressof(output)},
        audit=False,
    )
    assert run.ok and runtime.last_commit
    assert run.spans == ()
    assert list(output[:2]) == [11, 29]
    assert output[2] == 0xCDCDCDCD


@pytest.mark.parametrize("overflow", [False, True])
def test_native_timed_direct_commit_validates_all_spans_before_writing(overflow):
    statement = (
        "workspaces->work_items_offset = 0xFFFFFFFFu; workspaces->work_items_count = 2u;"
        if overflow
        else "workspaces->work_items_count = workspaces->work_items_capacity + 1u;"
    )
    source = f"""
int32_t host_tiling(const SfHostInputs *inputs, const SfHostShapes *shapes,
                    const SfHostTarget *target, SfHostWorkspaces *workspaces,
                    SfHostResolved *resolved) {{
    (void)inputs; (void)shapes; (void)target; (void)resolved;
    workspaces->work_items[0] = 99u;
    {statement}
    return SF_HOST_OK;
}}
"""
    _spec_, _build_, runtime = _native_runtime("mixed", source)
    output = (ctypes.c_uint32 * 36)(*([0xCDCDCDCD] * 36))
    with pytest.raises(HostProgramError, match="exceeds its declared capacity"):
        runtime.execute(
            _reader(_raw_inputs(_contract())),
            workspace_addresses={"work_items": ctypes.addressof(output)},
            audit=False,
        )
    assert not runtime.last_commit
    assert list(output) == [0xCDCDCDCD] * 36


def test_native_large_input_read_uses_typed_callback_and_records_interval():
    source = r"""
int32_t host_tiling(const SfHostInputs *inputs, const SfHostShapes *shapes,
                    const SfHostTarget *target, SfHostWorkspaces *workspaces,
                    SfHostResolved *resolved) {
    (void)shapes; (void)target; (void)resolved;
    int32_t values[4];
    if (sf_input_read_i32(inputs->cu_seqlens.id, 0u, (uintptr_t)values, 4u) != 0) return 1;
    if (sf_input_read_f32(inputs->cu_seqlens.id, 0u, (uintptr_t)values, 1u) == 0) return 2;
    workspaces->work_items[0] = (uint32_t)values[1];
    workspaces->work_items[1] = (uint32_t)values[3];
    workspaces->work_items_offset = 0u;
    workspaces->work_items_count = 2u;
    return SF_HOST_OK;
}
"""
    contract = _contract()
    _spec_, _build_, runtime = _native_runtime("mixed", source, mapped_limit=0)
    run = runtime.execute(_reader(_raw_inputs(contract)), audit=True)
    assert run.ok, run.error_detail
    imported = [read for read in run.input_reads if read.import_name is not None]
    assert len(imported) == 1
    assert imported[0].import_name == "sf_input_read_i32"
    assert imported[0].count == 4
    assert imported[0].sha256


def test_native_metadata_only_input_cannot_be_rewritten():
    source = r"""
int32_t host_tiling(const SfHostInputs *inputs, const SfHostShapes *shapes,
                    const SfHostTarget *target, SfHostWorkspaces *workspaces,
                    SfHostResolved *resolved) {
    (void)shapes; (void)target; (void)workspaces; (void)resolved;
    ((SfHostInputs *)(uintptr_t)inputs)->q.numel = 0u;
    return SF_HOST_OK;
}
"""
    _spec_, _build_, runtime = _native_runtime("mixed", source, mapped_limit=0)
    with pytest.raises(HostProgramError, match="metadata input 'q' numel changed"):
        runtime.execute(_reader(_raw_inputs(_contract())), audit=True)


def test_native_replay_uses_fresh_segmentation_not_cached_plan():
    contract = _contract()
    _spec_, _build_, native = _native_runtime("mixed")
    first = _raw_inputs(contract)
    second = dict(first)
    raw_cu = first["cu_seqlens"]
    bounds = list(struct.unpack(f"<{len(raw_cu) // 4}i", raw_cu))
    lengths = [bounds[i + 1] - bounds[i] for i in range(len(bounds) - 1)]
    lengths = lengths[1:] + lengths[:1]
    fresh = [0]
    for length in lengths:
        fresh.append(fresh[-1] + length)
    second["cu_seqlens"] = struct.pack(f"<{len(fresh)}i", *fresh)
    one = native.execute(_reader(first), audit=True)
    two = native.execute(_reader(second), audit=True)
    assert one.ok and two.ok
    assert one.spans[0].data != two.spans[0].data
    from ..support.npu_sim.varlen_worklist import pack_items, plan_pieces

    expected = pack_items(plan_pieces(lengths, contract.instance.symbols["H"], 3))
    assert next(span.data for span in two.spans if span.name == "work_items") == expected
