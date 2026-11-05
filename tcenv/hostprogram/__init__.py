"""Backend-neutral restricted C99 / Wasm Host Program plane (V3).

Import from here, not from ``tcenv.backends.metal``: the contract generator, the static precheck,
the Clang/lld build and the Wasm runtime know nothing about Metal, so the same module can be
executed by the Metal executor and by the NPU simulator.

    tcenv.hostprogram.contract   generated host_contract.h + Wasm32 layout + span rules
    tcenv.hostprogram.precheck   projection-time static refusal (never compiles)
    tcenv.hostprogram.toolchain  standard Clang/lld -> .wasm, engine identity, fuel/memory bounds
    tcenv.hostprogram.runtime    one Module, one Store per call, atomic span commit

Nothing here is imported by ``tcenv.taskcontract``; the deployment pack stays backend-free.
"""

from .contract import (
    CONTRACT_SCHEMA_ID,
    CONTRACT_SCHEMA_VERSION,
    HOST_STACK_BYTES,
    MAX_MAPPED_INPUT_BYTES,
    MAX_TOTAL_MAPPED_BYTES,
    STORAGE_GUARD_BYTES,
    STORAGE_GUARD_PATTERN,
    HostAbiLayout,
    HostContractSpec,
    HostField,
    HostInputView,
    HostStorageRegion,
    HostWorkspaceView,
    assembled_sha256,
    assembled_source,
    host_contract_spec,
    referenced_workspaces,
    storage_layout,
)
from .errors import HostCapabilityUnavailable, HostProgramError
from .precheck import require_static_precheck, static_precheck
from .runtime import HostInputRead, HostRun, HostRuntime, HostWorkspaceSpan
from .toolchain import (
    ENGINE_NAME,
    HostBuild,
    HostToolchain,
    compile_host_program,
    discover_toolchain,
    fuel_budget,
    max_items,
    memory_budget_bytes,
    recall_build,
    remember_build,
)

__all__ = [
    "CONTRACT_SCHEMA_ID",
    "CONTRACT_SCHEMA_VERSION",
    "ENGINE_NAME",
    "HOST_STACK_BYTES",
    "HostAbiLayout",
    "HostBuild",
    "HostCapabilityUnavailable",
    "HostContractSpec",
    "HostField",
    "HostInputRead",
    "HostInputView",
    "HostProgramError",
    "HostRun",
    "HostRuntime",
    "HostStorageRegion",
    "HostToolchain",
    "HostWorkspaceSpan",
    "HostWorkspaceView",
    "MAX_MAPPED_INPUT_BYTES",
    "MAX_TOTAL_MAPPED_BYTES",
    "STORAGE_GUARD_BYTES",
    "STORAGE_GUARD_PATTERN",
    "assembled_sha256",
    "assembled_source",
    "compile_host_program",
    "discover_toolchain",
    "fuel_budget",
    "host_contract_spec",
    "max_items",
    "memory_budget_bytes",
    "recall_build",
    "referenced_workspaces",
    "remember_build",
    "require_static_precheck",
    "static_precheck",
    "storage_layout",
]
