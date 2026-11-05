"""代表性 native Host+Device：候选自有 range-cut/LPT、partial tail 和状态 handoff。"""

from __future__ import annotations

import ctypes
from pathlib import Path

import pytest

from tcenv.hostprogram.native_contract import NativeContract, NativeField, build_minimal


CANDIDATE = (
    Path(__file__).resolve().parents[1]
    / "support/golden/shared/nativehost/varlen_host_device.c"
).read_text(encoding="utf-8")

CONTRACT = NativeContract(
    name="representative_varlen",
    shape_fields=(
        NativeField("total_tokens", "int32_t", 1, 4),
        NativeField("num_seqs", "uint32_t", 1, 4),
    ),
    inputs=(
        NativeField("cu_seqlens", "int32_t", 7, 4),
        NativeField("tokens", "int32_t", 32, 4),
        NativeField("initial_state", "int32_t", 6, 4),
    ),
    workspaces=(
        NativeField("work_items", "uint32_t", 72, 4),
        NativeField("item_counts", "int32_t", 4, 4),
        NativeField("worker_offsets", "int32_t", 4, 4),
        NativeField("worker_loads", "int32_t", 4, 4),
        NativeField("handoff_states", "int32_t", 12, 4),
        NativeField("handoff_flags", "int32_t", 12, 4),
    ),
)

RUNTIME = r'''
#include "host_contract.h"
static SfNativeInputs inputs;
static SfNativeShapes shapes;
static SfNativeTarget target;
static SfNativeWorkspaces workspaces;
static SfNativeResolved resolved;
static int32_t cu[7];
static int32_t tokens[32];
static int32_t initial_state[6];
static uint32_t work_items[72];
static int32_t item_counts[4];
static int32_t worker_offsets[4];
static int32_t worker_loads[4];
static int32_t handoff_states[12];
static int32_t handoff_flags[12];
static int32_t output[32];
static int32_t final_state[6];
#define INIT_WS(field, array, capacity) \
    workspaces.field = array; workspaces.field##_capacity = capacity; \
    workspaces.field##_offset = 0; workspaces.field##_count = 0; \
    workspaces.field##_elem_nbytes = 4
static void init(void) {
    inputs.cu_seqlens = cu; inputs.cu_seqlens_numel = 7;
    inputs.tokens = tokens; inputs.tokens_numel = 32;
    inputs.initial_state = initial_state; inputs.initial_state_numel = 6;
    INIT_WS(work_items, work_items, 72);
    INIT_WS(item_counts, item_counts, 4);
    INIT_WS(worker_offsets, worker_offsets, 4);
    INIT_WS(worker_loads, worker_loads, 4);
    INIT_WS(handoff_states, handoff_states, 12);
    INIT_WS(handoff_flags, handoff_flags, 12);
    workspaces.handoff_states_count = 0;
}
uintptr_t sf_native_abi_address(void) { return (uintptr_t)&inputs; }
uintptr_t sf_native_storage_address(uint32_t offset) { return (uintptr_t)((uint8_t *)cu + offset); }
uintptr_t sf_native_input_abi_address(void) { return (uintptr_t)&inputs; }
uintptr_t sf_native_shape_abi_address(void) { return (uintptr_t)&shapes; }
uintptr_t sf_native_target_abi_address(void) { return (uintptr_t)&target; }
uintptr_t sf_native_workspace_abi_address(void) { return (uintptr_t)&workspaces; }
uintptr_t sf_native_resolved_abi_address(void) { return (uintptr_t)&resolved; }
uintptr_t sf_native_output_address(void) { return (uintptr_t)output; }
uintptr_t sf_native_final_state_address(void) { return (uintptr_t)final_state; }
uintptr_t sf_native_workspace_work_items_address(void) { return (uintptr_t)work_items; }
uintptr_t sf_native_workspace_item_counts_address(void) { return (uintptr_t)item_counts; }
uintptr_t sf_native_workspace_worker_offsets_address(void) { return (uintptr_t)worker_offsets; }
uintptr_t sf_native_workspace_worker_loads_address(void) { return (uintptr_t)worker_loads; }
uintptr_t sf_native_workspace_handoff_states_address(void) { return (uintptr_t)handoff_states; }
uintptr_t sf_native_workspace_handoff_flags_address(void) { return (uintptr_t)handoff_flags; }
uint32_t sf_native_workspace_work_items_offset(void) { return workspaces.work_items_offset; }
uint32_t sf_native_workspace_work_items_count(void) { return workspaces.work_items_count; }
uint32_t sf_native_workspace_item_counts_offset(void) { return workspaces.item_counts_offset; }
uint32_t sf_native_workspace_item_counts_count(void) { return workspaces.item_counts_count; }
uint32_t sf_native_workspace_worker_offsets_offset(void) { return workspaces.worker_offsets_offset; }
uint32_t sf_native_workspace_worker_offsets_count(void) { return workspaces.worker_offsets_count; }
uint32_t sf_native_workspace_worker_loads_offset(void) { return workspaces.worker_loads_offset; }
uint32_t sf_native_workspace_worker_loads_count(void) { return workspaces.worker_loads_count; }
uint32_t sf_native_workspace_handoff_states_offset(void) { return workspaces.handoff_states_offset; }
uint32_t sf_native_workspace_handoff_states_count(void) { return workspaces.handoff_states_count; }
uint32_t sf_native_workspace_handoff_flags_offset(void) { return workspaces.handoff_flags_offset; }
uint32_t sf_native_workspace_handoff_flags_count(void) { return workspaces.handoff_flags_count; }
void sf_example_set_case(const int32_t *new_cu, const int32_t *new_tokens,
                         const int32_t *new_initial, uint32_t count, int32_t total) {
    for (uint32_t i = 0; i < 7; ++i) cu[i] = new_cu[i];
    for (uint32_t i = 0; i < 32; ++i) { tokens[i] = new_tokens[i]; output[i] = -777777; }
    for (uint32_t i = 0; i < 6; ++i) { initial_state[i] = new_initial[i]; final_state[i] = -888888; }
    shapes.num_seqs = count; shapes.total_tokens = total;
    init();
    inputs.cu_seqlens_numel = count + 1u;
    inputs.tokens_numel = (uint32_t)total;
    inputs.initial_state_numel = count;
}
void sf_example_drop_first_piece(void) {
    uint32_t packed = work_items[2];
    uint32_t pieces = (packed >> 16) & 0xffu;
    work_items[2] = (packed & 0xffu) | (1u << 8) | (pieces << 16);
}
'''


def _library(tmp_path: Path, mode: int):
    lib = ctypes.CDLL(str(build_minimal(
        CONTRACT,
        RUNTIME + CANDIDATE,
        tmp_path,
        extra_cflags=(f"-DSF_PLANNER_MODE={mode}",),
    )))
    for name in (
        "sf_native_input_abi_address", "sf_native_shape_abi_address", "sf_native_target_abi_address",
        "sf_native_workspace_abi_address", "sf_native_resolved_abi_address", "sf_native_output_address",
        "sf_native_final_state_address", "sf_native_workspace_work_items_address",
        "sf_native_workspace_item_counts_address", "sf_native_workspace_worker_offsets_address",
        "sf_native_workspace_worker_loads_address", "sf_native_workspace_handoff_states_address",
        "sf_native_workspace_handoff_flags_address",
    ):
        getattr(lib, name).restype = ctypes.c_size_t
    lib.host_tiling.restype = ctypes.c_int32
    lib.host_tiling.argtypes = [ctypes.c_void_p] * 5
    lib.device_execute.restype = ctypes.c_int32
    lib.device_execute.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                                   ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32)]
    lib.sf_example_set_case.argtypes = [
        ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32),
        ctypes.c_uint32, ctypes.c_int32,
    ]
    return lib


def _reference(lengths, tokens, initial):
    output = []
    final = []
    cursor = 0
    for seq, length in enumerate(lengths):
        state = initial[seq]
        for token in tokens[cursor:cursor + length]:
            state += token
            output.append(state)
        final.append(state)
        cursor += length
    return output, final


def _run(lib, lengths, *, poison=False):
    total = sum(lengths)
    cu_values = [0]
    for length in lengths:
        cu_values.append(cu_values[-1] + length)
    cu_values += [0] * (7 - len(cu_values))
    token_values = [(index * 7 + 3) % 13 - 6 for index in range(total)]
    token_values += [0] * (32 - total)
    initial_values = [10 * (index + 1) for index in range(len(lengths))] + [0] * (6 - len(lengths))
    cu = (ctypes.c_int32 * 7)(*cu_values)
    tokens = (ctypes.c_int32 * 32)(*token_values)
    initial = (ctypes.c_int32 * 6)(*initial_values)
    lib.sf_example_set_case(cu, tokens, initial, len(lengths), total)
    host_status = lib.host_tiling(
        lib.sf_native_input_abi_address(), lib.sf_native_shape_abi_address(),
        lib.sf_native_target_abi_address(), lib.sf_native_workspace_abi_address(),
        lib.sf_native_resolved_abi_address(),
    )
    if host_status != 0:
        return {"host_status": host_status}
    if poison:
        # Consumer sees piece 1 without piece 0 having published its state.
        lib.sf_example_drop_first_piece()
    output_address = lib.sf_native_output_address()
    final_address = lib.sf_native_final_state_address()
    device_status = lib.device_execute(
        lib.sf_native_input_abi_address(), lib.sf_native_shape_abi_address(),
        lib.sf_native_workspace_abi_address(),
        ctypes.cast(output_address, ctypes.POINTER(ctypes.c_int32)),
        ctypes.cast(final_address, ctypes.POINTER(ctypes.c_int32)),
    )
    loads_address = lib.sf_native_workspace_worker_loads_address()
    work_address = lib.sf_native_workspace_work_items_address()
    return {
        "host_status": host_status,
        "device_status": device_status,
        "output": list((ctypes.c_int32 * total).from_address(output_address)),
        "final": list((ctypes.c_int32 * len(lengths)).from_address(final_address)),
        "loads": list((ctypes.c_int32 * 4).from_address(loads_address)),
        "work_items": list((ctypes.c_uint32 * 72).from_address(work_address)),
        "handoff_count": int(lib.sf_native_workspace_handoff_states_count()),
    }


CASES = {
    "canonical": [3, 5, 2],
    "skewed": [10, 1, 1, 1, 1],
    "late_long": [1, 1, 1, 1, 10],
    "permuted": [1, 7, 2, 5],
    "primes": [2, 3, 5, 7],
    "zero_length": [0, 4, 0, 6, 3],
    "fresh": [6, 1, 2, 8],
    "chunk_boundary": [4, 5, 1, 9],
}


@pytest.mark.parametrize("mode", [1, 2])
@pytest.mark.parametrize("case", sorted(CASES))
def test_candidate_strategies_match_independent_reference(tmp_path, mode, case):
    lengths = CASES[case]
    result = _run(_library(tmp_path, mode), lengths)
    assert result["host_status"] == 0
    assert result["device_status"] == 0
    tokens = [(index * 7 + 3) % 13 - 6 for index in range(sum(lengths))]
    initial = [10 * (index + 1) for index in range(len(lengths))]
    expected_output, expected_final = _reference(lengths, tokens, initial)
    assert result["output"] == expected_output
    assert result["final"] == expected_final
    # Host never publishes device-owned handoff states.
    assert result["handoff_count"] == 0


def test_cost_lpt_improves_late_long_chain_load_balance(tmp_path):
    balanced = _run(_library(tmp_path / "balanced", 1), CASES["late_long"])
    lpt = _run(_library(tmp_path / "lpt", 2), CASES["late_long"])
    assert balanced["device_status"] == lpt["device_status"] == 0
    assert max(lpt["loads"]) < max(balanced["loads"])
    assert lpt["output"] == balanced["output"]
    assert lpt["final"] == balanced["final"]


def test_missing_handoff_is_rejected_not_silently_accepted(tmp_path):
    result = _run(_library(tmp_path, 2), [5, 2], poison=True)
    assert result["host_status"] == 0
    assert result["device_status"] == 100


def test_invalid_runtime_segmentation_publishes_no_plan(tmp_path):
    lib = _library(tmp_path, 2)
    result = _run(lib, [3, -1, 8])
    assert result["host_status"] != 0
    assert set(result) == {"host_status"}
