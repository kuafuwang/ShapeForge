# v1 清理后的真实 Metal 验证（2026-11-25）

## 范围与授权

用户明确要求运行 GPU。只运行真实 Metal 测试，没有付费模型、搜索或真实 NPU。最终在 MetalToolchain **27A266a** 上串行完成：

```bash
uv run pytest -q -m gpu -n 0
```

第一次 1800 秒上限只覆盖部分节点，未当成通过；修复后完整运行两次，最终结果为 **207 passed、1 skipped、2 failed**。唯一的 skip 是缺少 `hmz` 的既有环境边界，不是 v1 清理引入。

## 修复并复核的功能项

- 真实 `MetalSourceBuilder`：使用当前 `varlen_scale/mixed` 候选和 `MetalV2Executor.bind/correctness`，构建产物真实执行通过。
- Frozen Track：测试 registry 保留同一个已实现 adapter、改变 `track_version`/digest，worker 正确选择父进程冻结身份；不再把未实现的 build adapter 当成可用。
- Scratch：ping-pong 的 launch declaration、逐 stage access 和元素单位补齐。device pointer 的 `reflects readwrite` 反例仍拒绝；texture 正向路径与 Host Program 非零负例分别通过。
- TileLang：registry 默认版本按当前 `/3` 校验，显式 `/2` 构建与 work-list 继续通过；MT4 当前工具链 artifact digest 经父提交对照确认是环境身份变化，不是本轮代码回归。
- varlen：忽略分段的候选同时数值错误并依赖输出初值，按当前安全合同判 `unsafe`，不是弱化为普通 `incorrect`。
- Fresh identity：bitdelta、gemm_silu 与 conv2d 的当前 fresh subject/manifest 已重新生成并复核；旧归档不改写。
- MetalToolchain 安装后，原 `test_statement_msl_example_compiles` 和四个硬件编译分析节点全部通过。

## 最终两个失败（不接受为发布或收益）

1. **per-launch paired evaluation**：新提交方式的中位 speedup **0.9934756163**，低于 1.0；未达到 `new_median > 1.0`，更不能满足相对旧方式再快 3%。正确性、安全与 identity 均通过，但没有采用该提交方式的性能依据。
2. **shape-portfolio performance**：当前 shard `timing_uncertain=true`。正确性和 dispatch 均通过，但当前窗口不能发布。此前的诊断中 6/7 shard 也为 `timing_uncertain`；没有用追加样本去凑通过。

这两个门保持原样，没有放宽阈值、skip/xfail、反复重测求幸运值或把失败改写成环境噪声。

## 当前 fresh 证据

- `docs/可信harness/bitdelta-fresh-20261124-remove-v1.json`
- `docs/可信harness/gemm_silu-fresh-20261124-remove-v1.json`
- `docs/可信harness/conv2d-portfolio-fresh-20261124-remove-v1.json`

这些是本轮当前代码的 fresh subject/manifest，旧 P2 归档保持原字节。`publication_allowed=false`；本记录不发布 speedup，也不证明真实 NPU 正确性或性能。
