"""V3 工具链探针：标准 Clang/lld → Wasm，Wasmtime 执行与 Store/fuel 边界。

这不是候选实现，也不进入任何评测或发布链。它证明 V3 的最小基础设施在真实本机可用：

* ``host_tiling((0,3,8,10)) -> (3,5,2)``；
* 同一 Module 复用，每次调用新建 Store，输入可重放；
* 非法首项/非单调边界返回非零；
* 死循环被 fuel 耗尽拒绝（trap），不是基础设施崩溃。

构建（需要 Homebrew LLVM + lld）：

    /opt/homebrew/opt/llvm/bin/clang --target=wasm32 -O2 -nostdlib \
      -Wl,--no-entry -Wl,--export=host_tiling -Wl,--export=busy \
      tests/support/host_tiling_probe.c -o /tmp/host_tiling_probe.wasm
    wasm-validate /tmp/host_tiling_probe.wasm

执行：``uv run python tests/support/host_tiling_probe.py``（传入 wasm 路径可选）。
"""

from __future__ import annotations

import sys
from pathlib import Path

import wasmtime


def run(wasm_path: str | Path = "/tmp/host_tiling_probe.wasm") -> None:
    """@brief 用同一 Module 做多次独立 Store 调用，并验证 fuel trap。

    @param wasm_path 由标准 Clang/lld 构建的 probe module。
    """
    module_bytes = Path(wasm_path).read_bytes()
    config = wasmtime.Config()
    config.consume_fuel = True
    engine = wasmtime.Engine(config)
    module = wasmtime.Module(engine, module_bytes)

    def call(cu: tuple[int, ...]) -> tuple[int, list[int]]:
        store = wasmtime.Store(engine)
        store.set_fuel(100_000)
        store.set_limits(memory_size=1 << 20)
        instance = wasmtime.Instance(store, module, [])
        exports = instance.exports(store)
        memory = instance.exports(store).get("memory")
        assert memory is not None, "the C module must export its linear memory"
        raw = b"".join(int(value).to_bytes(4, "little", signed=True) for value in cu)
        # cu 是输入，out 是独立输出区；两者都从 0 开始会覆盖边界，得到 [0,3,8] 的假象。
        # V3 的 host_contract.h 必须为每个视图分配不同的 Wasm 地址，不能默认共享 0。
        out_offset = len(raw)
        memory.write(store, raw, 0)
        status = exports["host_tiling"](store, 0, len(cu), out_offset)
        lengths = [
            int.from_bytes(
                memory.read(store, out_offset + 4 * index, out_offset + 4 * index + 4),
                "little",
                signed=True,
            )
            for index in range(len(cu) - 1)
        ]
        return int(status), lengths

    first = call((0, 3, 8, 10))
    second = call((0, 2, 6, 10))
    replay = call((0, 3, 8, 10))
    invalid = call((1, 3, 8, 10))
    assert first == (0, [3, 5, 2]), first
    assert second == (0, [2, 4, 4]), second
    assert replay == first
    assert invalid[0] != 0
    print("host_tiling:", first, second, replay, "invalid:", invalid[0])

    store = wasmtime.Store(engine)
    store.set_fuel(100_000)
    instance = wasmtime.Instance(store, module, [])
    try:
        instance.exports(store)["busy"](store)
    except Exception as exc:  # Wasmtime Trap is intentionally caught for this probe.
        print("fuel trap:", type(exc).__name__)
    else:
        raise AssertionError("the busy export was not stopped by fuel")


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else "/tmp/host_tiling_probe.wasm")
