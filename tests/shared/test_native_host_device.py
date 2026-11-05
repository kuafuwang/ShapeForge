"""Native ABI 的最小 Host+Device 闭环（CPU device oracle）。

Host Program 生成 lengths workspace；device consumer 按有效 span 计算 sum。这里的 device 是
独立 CPU consumer，不是 Metal kernel；Metal 接入属于下一阶段。
"""

from __future__ import annotations

import ctypes
from pathlib import Path

from tcenv.hostprogram.native_contract import NativeContract, NativeField, build_minimal


SOURCE = r'''
#include "host_contract.h"
#include <string.h>
static SfNativeAbi abi;
static SfNativeInputs inputs;
static SfNativeShapes shapes;
static SfNativeTarget target;
static SfNativeWorkspaces workspaces;
static SfNativeResolved resolved;
static int32_t cu[4];
static int32_t lengths[3];
uintptr_t sf_native_abi_address(void) { return (uintptr_t)&abi; }
uintptr_t sf_native_storage_address(uint32_t offset) { return (uintptr_t)((uint8_t *)cu + offset); }
uintptr_t sf_native_input_cu_seqlens_address(void) { return (uintptr_t)cu; }
uintptr_t sf_native_workspace_lengths_address(void) { return (uintptr_t)lengths; }
uint32_t sf_native_workspace_lengths_offset(void) { return workspaces.lengths_offset; }
uint32_t sf_native_workspace_lengths_count(void) { return workspaces.lengths_count; }
uintptr_t sf_native_input_abi_address(void) { return (uintptr_t)&inputs; }
uintptr_t sf_native_shape_abi_address(void) { return (uintptr_t)&shapes; }
uintptr_t sf_native_target_abi_address(void) { return (uintptr_t)&target; }
uintptr_t sf_native_workspace_abi_address(void) { return (uintptr_t)&workspaces; }
uintptr_t sf_native_resolved_abi_address(void) { return (uintptr_t)&resolved; }
static void init(void) {
    inputs.cu_seqlens = cu; inputs.cu_seqlens_numel = 4;
    shapes.total_tokens = 10;
    workspaces.lengths = lengths; workspaces.lengths_capacity = 3;
    workspaces.lengths_elem_nbytes = 4;
}
void sf_native_set_input(const int32_t *v) { memcpy(cu, v, sizeof(cu)); init(); }
int32_t host_tiling(const SfNativeInputs *i, const SfNativeShapes *s,
                    const SfNativeTarget *t, SfNativeWorkspaces *w, SfNativeResolved *r) {
    (void)t; (void)r;
    if (i->cu_seqlens[0] != 0 || i->cu_seqlens[3] != s->total_tokens) return 1;
    for (int j = 0; j < 3; ++j) w->lengths[j] = i->cu_seqlens[j+1] - i->cu_seqlens[j];
    w->lengths_offset = 0; w->lengths_count = 3; return 0;
}
int32_t device_sum(const int32_t *valid, uint32_t count) {
    int32_t result = 0;
    for (uint32_t i = 0; i < count; ++i) result += valid[i];
    return result;
}
'''


def test_native_host_span_is_consumed_by_device(tmp_path):
    contract = NativeContract(
        name="minimal_host_device",
        shape_fields=(NativeField("total_tokens", "int32_t", 1, 4),),
        inputs=(NativeField("cu_seqlens", "int32_t", 4, 4),),
        workspaces=(NativeField("lengths", "int32_t", 3, 4),),
    )
    library = build_minimal(contract, SOURCE, tmp_path)
    lib = ctypes.CDLL(str(library))
    for name in ("sf_native_input_abi_address", "sf_native_shape_abi_address", "sf_native_target_abi_address", "sf_native_workspace_abi_address", "sf_native_resolved_abi_address", "sf_native_workspace_lengths_address"):
        getattr(lib, name).restype = ctypes.c_size_t
    lib.sf_native_workspace_lengths_count.restype = ctypes.c_uint32
    lib.sf_native_workspace_lengths_offset.restype = ctypes.c_uint32
    lib.host_tiling.restype = ctypes.c_int32
    lib.host_tiling.argtypes = [ctypes.c_void_p] * 5
    lib.device_sum.restype = ctypes.c_int32
    lib.device_sum.argtypes = [ctypes.POINTER(ctypes.c_int32), ctypes.c_uint32]
    values = (ctypes.c_int32 * 4)(0, 3, 8, 10)
    lib.sf_native_set_input(values)
    status = lib.host_tiling(
        lib.sf_native_input_abi_address(), lib.sf_native_shape_abi_address(),
        lib.sf_native_target_abi_address(), lib.sf_native_workspace_abi_address(),
        lib.sf_native_resolved_abi_address(),
    )
    assert status == 0
    address = lib.sf_native_workspace_lengths_address()
    count = int(lib.sf_native_workspace_lengths_count())
    offset = int(lib.sf_native_workspace_lengths_offset())
    valid = (ctypes.c_int32 * count).from_address(address + offset * 4)
    assert lib.device_sum(valid, count) == 10
