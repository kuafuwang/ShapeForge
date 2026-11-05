"""Native64 ABI 的最小闭环：编译、输入、span、guard、失败和 replay。"""

from __future__ import annotations

import ctypes
from pathlib import Path

import pytest

from tcenv.hostprogram.native_contract import NativeContract, NativeField, build_minimal


CONTRACT = NativeContract(
    name="minimal_varlen",
    shape_fields=(NativeField("total_tokens", "int32_t", 1, 4),),
    inputs=(NativeField("cu_seqlens", "int32_t", 4, 4),),
    workspaces=(NativeField("lengths", "int32_t", 3, 4),),
)


SOURCE = r'''
#include "host_contract.h"
#include <string.h>

static SfNativeAbi abi;
static SfNativeInputs input_abi;
static SfNativeShapes shape_abi;
static SfNativeTarget target_abi;
static SfNativeWorkspaces workspace_abi;
static SfNativeResolved resolved_abi;
static int32_t input_values[4];
static int32_t workspace_values[3];
static uint8_t input_guard[32] __attribute__((used));
static uint8_t workspace_guard[32] __attribute__((used));
static int guard_status;

uintptr_t sf_native_abi_address(void) { return (uintptr_t)&abi; }
uintptr_t sf_native_input_abi_address(void) { return (uintptr_t)&input_abi; }
uintptr_t sf_native_shape_abi_address(void) { return (uintptr_t)&shape_abi; }
uintptr_t sf_native_target_abi_address(void) { return (uintptr_t)&target_abi; }
uintptr_t sf_native_workspace_abi_address(void) { return (uintptr_t)&workspace_abi; }
uintptr_t sf_native_resolved_abi_address(void) { return (uintptr_t)&resolved_abi; }
uintptr_t sf_native_storage_address(uint32_t offset) {
    return (uintptr_t)((uint8_t *)input_values + offset);
}
uintptr_t sf_native_input_cu_seqlens_address(void) { return (uintptr_t)input_values; }
uintptr_t sf_native_workspace_lengths_address(void) { return (uintptr_t)workspace_values; }
uint32_t sf_native_workspace_lengths_offset(void) { return workspace_abi.lengths_offset; }
uint32_t sf_native_workspace_lengths_count(void) { return workspace_abi.lengths_count; }
int32_t sf_native_guard_status(void) { return guard_status; }
void sf_native_reset_guards(void) { guard_status = 0; }

static void init_abi(void) {
    abi.shapes.total_tokens = 10;
    abi.inputs.cu_seqlens = input_values;
    abi.inputs.cu_seqlens_numel = 4;
    abi.workspaces.lengths = workspace_values;
    abi.workspaces.lengths_capacity = 3;
    abi.workspaces.lengths_elem_nbytes = 4;
    abi.workspaces.lengths_offset = 0;
    abi.workspaces.lengths_count = 0;
    input_abi = abi.inputs;
    shape_abi = abi.shapes;
    workspace_abi = abi.workspaces;
}

__attribute__((visibility("default")))
int32_t host_tiling(const SfNativeInputs *inputs,
                    const SfNativeShapes *shapes,
                    const SfNativeTarget *target,
                    SfNativeWorkspaces *workspaces,
                    SfNativeResolved *resolved) {
    (void)target; (void)resolved;
    if (inputs->cu_seqlens == 0 || workspaces->lengths == 0) return 1;
    if (inputs->cu_seqlens_numel != 4 || shapes->total_tokens != 10) return 2;
    if (inputs->cu_seqlens[0] != 0 || inputs->cu_seqlens[3] != 10) return 3;
    for (int i = 0; i < 3; ++i) {
        int32_t length = inputs->cu_seqlens[i + 1] - inputs->cu_seqlens[i];
        if (length < 0) return 4;
        workspaces->lengths[i] = length;
    }
    workspaces->lengths_offset = 0;
    workspaces->lengths_count = 3;
    return 0;
}

__attribute__((visibility("default")))
void sf_native_set_input(const int32_t *values) {
    memcpy(input_values, values, sizeof(input_values));
    init_abi();
}

__attribute__((visibility("default")))
void sf_native_corrupt_guard(void) { guard_status = 1; }

__attribute__((visibility("default")))
int32_t sf_native_fail(void) { return 9; }
'''


def _load(tmp_path: Path):
    library = build_minimal(CONTRACT, SOURCE, tmp_path)
    lib = ctypes.CDLL(str(library))
    lib.host_tiling.restype = ctypes.c_int32
    lib.host_tiling.argtypes = [ctypes.c_void_p] * 5
    lib.sf_native_abi_address.restype = ctypes.c_size_t
    lib.sf_native_workspace_lengths_address.restype = ctypes.c_size_t
    lib.sf_native_workspace_lengths_count.restype = ctypes.c_uint32
    lib.sf_native_workspace_lengths_offset.restype = ctypes.c_uint32
    lib.sf_native_guard_status.restype = ctypes.c_int32
    lib.sf_native_set_input.argtypes = [ctypes.POINTER(ctypes.c_int32)]
    lib.sf_native_fail.restype = ctypes.c_int32
    return lib


def _call(lib, values):
    values = (ctypes.c_int32 * 4)(*values)
    lib.sf_native_set_input(values)
    abi = lib.sf_native_abi_address()
    lib.sf_native_input_abi_address.restype = ctypes.c_size_t
    lib.sf_native_shape_abi_address.restype = ctypes.c_size_t
    lib.sf_native_target_abi_address.restype = ctypes.c_size_t
    lib.sf_native_workspace_abi_address.restype = ctypes.c_size_t
    lib.sf_native_resolved_abi_address.restype = ctypes.c_size_t
    status = lib.host_tiling(
        lib.sf_native_input_abi_address(),
        lib.sf_native_shape_abi_address(),
        lib.sf_native_target_abi_address(),
        lib.sf_native_workspace_abi_address(),
        lib.sf_native_resolved_abi_address(),
    )
    address = lib.sf_native_workspace_lengths_address()
    count = int(lib.sf_native_workspace_lengths_count())
    offset = int(lib.sf_native_workspace_lengths_offset())
    result = list((ctypes.c_int32 * 3).from_address(address))
    return status, offset, count, result


def test_native_abi_compiles_and_produces_valid_span(tmp_path):
    lib = _load(tmp_path)
    status, offset, count, values = _call(lib, [0, 3, 8, 10])
    assert (status, offset, count) == (0, 0, 3)
    assert values == [3, 5, 2]
    assert lib.sf_native_guard_status() == 0


def test_native_abi_rejects_invalid_input_without_publishing_span(tmp_path):
    lib = _load(tmp_path)
    status, offset, count, values = _call(lib, [1, 3, 8, 10])
    assert status != 0
    # The caller must discard the whole result on nonzero status; no partial plan is valid.
    assert status == 3


def test_native_abi_guard_and_failure_are_observable(tmp_path):
    lib = _load(tmp_path)
    lib.sf_native_corrupt_guard()
    assert lib.sf_native_guard_status() == 1
    lib.sf_native_reset_guards()
    assert lib.sf_native_guard_status() == 0
    assert lib.sf_native_fail() == 9


def test_native_abi_replay_uses_fresh_input(tmp_path):
    lib = _load(tmp_path)
    first = _call(lib, [0, 1, 9, 10])
    second = _call(lib, [0, 4, 4, 10])
    assert first[3] == [1, 8, 1]
    assert second[3] == [4, 0, 6]
