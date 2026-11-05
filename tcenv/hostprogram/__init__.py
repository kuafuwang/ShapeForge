"""Backend-neutral restricted-C99/native64 Host Program plane.

``contract.py`` derives names/types/capacities from OpSpec.  ``native_abi.py`` generates the
compiler-owned native64 ABI, ``native_toolchain.py`` builds a worker-local dylib, and
``native_runtime.py`` produces atomic ``results.HostRun`` values.  Metal and the NPU simulator
use the same semantics; neither imports a Wasm engine, module or Store.

Statement-only consumers can import the contract without loading a compiler or native library.
"""
