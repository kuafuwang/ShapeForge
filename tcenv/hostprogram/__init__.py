"""Backend-neutral restricted C99 / Wasm Host Program plane (V3).

Import from the concrete module, not from ``tcenv.backends.metal``: the contract generator, the
static precheck, the Clang/lld build and the Wasm runtime know nothing about Metal, so the same
module can be executed by the Metal executor and by the NPU simulator.

    tcenv.hostprogram.contract   generated host_contract.h + Wasm32 layout + span rules
    tcenv.hostprogram.precheck   projection-time static refusal (never compiles)
    tcenv.hostprogram.toolchain  standard Clang/lld -> .wasm, engine identity, fuel/memory bounds
    tcenv.hostprogram.runtime    one Module, one Store per call, atomic span commit

Only ``contract`` enters the task pack closure (``tcenv.task_presentation`` renders the generated
header for varlen statements); the runtime/toolchain planes and the Metal executor stay outside
it, so the deployment pack remains backend-free.  This module deliberately has no imports: loading
``tcenv.hostprogram.contract`` must not drag the compiler/engine into a statement-only consumer.
"""
