# MT2 失败与修正历史

本目录保留开发过程中的拒绝证据。后续成功不把这些失败改写成成功。

1. `worklist-r1-abi-error.json`：`tmp` 同时被两个阶段读写，却只声明 `write`；真实 Metal reflection/ABI 返回 `abi_error`。改为 `read_write`，没有放宽 binder。
2. `worklist-r2-poison-unsafe.json`：第一阶段读取尚未初始化的 `tmp`；`out1` poison 正确暴露 NaN/状态依赖。改为先写后读，并取消不必要的 zero 依赖。
3. 多函数 `IRModule` 仍重复固定 include/union。preamble 由 harness-owned 装配层精确去重，函数体不变；随后真实 Metal 编译通过。
4. 最初把 `/etc/passwd` 错当成未授权路径。实际它是 `tcenv/worker_isolation/hmz_fence.py` 显式声明的系统只读文件，候选返回 `ok`。当时断言在写证据前失败，因此没有原轮 raw JSON，不补造。`system-read-allowed-control.json` 是后来明确标注的当前控制观察；真正的外部读取负例改为 attempt 外 canary，r4/r6 均得到 `PermissionError` 且 canary 未变。
5. r2/r3 的 `device_ir.json` 含随机 attempt 绝对 `ir.SourceName`，使 artifact digest 不确定。修复为逻辑 `kernel.py`；这些目录留在包里作修复前历史。r4/r6 才用于可复现性验收。
6. r5 的 MSL 对照 correctness/ABI 通过，但 `noisy=true`，测试失败；`noisy-r5-msl-handoff.md` 保留该失败。r6 是后续独立安静窗口确认，不是删除或改写 r5。

`../r2/`、`../confirmation-r3/`、`../noisy-r5/` 均属于上述历史；当前主证据是 `../r4/`，独立确认是 `../confirmation-r6/`。不同轮耗时只作各自观察，不用于挑选性能结论。
