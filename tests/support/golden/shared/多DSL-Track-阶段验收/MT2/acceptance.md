# MT2：TileLang 第一条 DSL Track 的真实 Metal 垂直切片

## 状态与范围

**MT2 已完成：实现、真实 Metal 证据、只读 verifier 与独立 reviewer 均通过。**独立 DeepSeek Flash
reviewer 返回 `done=true`（`review.json`，日志 SHA-256 已绑定），但只签署下述固定人工真实 Metal 机制范围。

- Track：`tilelang-pipeline/2`，登记为 `available`，仅覆盖**固定人工候选的 candidate-owned `kernel_pipeline` 构建与评测**。
- `tilelang-pipeline/1` 继续是 `interface_only`，用于 MT1b frozen/source-only replay；合同摘要仍为 `689f8680…`。
- `/2` 合同摘要为 `03541a31…`，registry 摘要为 `596474db…`。
- 本轮获得并使用了真实 Metal GPU 授权；没有 LLM 候选生成、付费搜索、NPU 执行或性能发布。
- Graph、第二条非 MSL DSL、生产 `taic run/track-explore`、LLM prompt/搜索、fresh portfolio confirmation 和 speedup publication 均未开放。MT0 整体仍因第二条 DSL deferred 而 blocked，MT3 未开始。

## 实现边界

1. `tcenv/backends/metal/kernel_pipeline.py` 是 DSL 无关的共享 task 投影、bounded evaluator 和编译失败投影。现有 MSL 路径只保留自己的候选解析/executor；真实 MSL 对照回归通过。
2. `tilelang_build.py` 在 lower-level hmz `Profile + Proxy(())` 内运行 pinned TileLang 0.1.15。候选可声明一个 `KERNEL` 或候选自有 `KERNELS` work-list；多函数一次 lowering 为 `IRModule`。
3. `tilelang_metal.py` 将 frontend 生成的 MSL 经同一边界检查后交给 `MetalSourceBuilder`，形成真实 Apple Metal library/pipeline。frontend 输入摘要与真正进入 Apple 编译器的 assembled digest 分账。
4. `tilelang_runtime.py` 从已审计 build 读取 `kernel.metal`，把冻结几何直接投影成 runtime `LaunchSpec`。不解析/伪造 `taic-launch`，不调用 MSL Track adapter 或 HostTilingPlan。
5. 共享 `MetalV2Runtime` 继续拥有 reflection binding、buffer 生命周期、只读输入保护、output/workspace guard/poison、read_write 恢复、实际 dispatch parity、fresh rotation、CommandBuffer commit/wait 和计时。
6. `verify_evaluator_build` 增加 Track 级 generated-artifact 审计缝。控制机重新核对候选清单、冻结几何、frontend/options/attempt、生成物闭集、source-only cache/build identity 和 native entry；不是接受 worker 自报 JSON。
7. `device_ir.json` 的 `ir.SourceName` 规范化成逻辑 `kernel.py`，消除随机 attempt 路径造成的 artifact 漂移。IR 指令不变；两次独立构建的 artifact 字节/digest 固定测试通过。

## 固定任务与候选

任务 fixture：`y = 2*x + 1`，`acc = acc + y`；`acc` 是 `read_write` 并纳入最终差分。

| 路线 | 候选调度 | 用途 |
| --- | --- | --- |
| TileLang single | `scale_kernel_kernel`，4×256 | 单阶段 frontend/ABI/correctness/fresh rotation |
| TileLang work-list | `scale_kernel_kernel → bias_kernel_kernel`，8×128，`tmp` workspace | 候选自有阶段/work-list 所有权 |
| MSL single | `scale_msl`，4×256，单阶段融合 | 同语义、不同内部执行计划的现有路线对照 |

固定候选、几何、OpSpec/reference 原字节位于 `candidates/`。候选身份：

- single：`803a3686…`，bundle `62d16853…`，artifact `fbdd1da8…`；
- work-list：`3d496ec5…`，bundle `5dfa9df9…`，artifact `95e3796e…`；
- MSL：`6ee469c1…`，bundle `aae4b06d…`，artifact `669e2d91…`。

## 真实 Metal 证据

### 主证据 r4 与独立确认 r6

`evidence/r4/` 与 `evidence/confirmation-r6/` 均为当前完整 **7 个 GPU 节点，7/7 passed**。两轮所有正例：

- `status=ok`、ABI/correctness 通过、真实 dispatch 与冻结几何一致；
- fresh rotation = 16，`noisy=false`，benchmark guard violation 为空；
- `read_write acc` 真实绑定，三个 case 的 seed/scaled error 在 TileLang 与 MSL 间逐项一致；
- candidate/bundle/artifact digest 在 r4/r6 逐项一致；TileLang 新 attempt 的 native identity 正常前进；
- shipped registry 节点直接读取 `backends/metal/tracks.json`，选择 `/2`，同时保留 `/1`。

耗时只记录各自观察，不用于选轮或发布：

| 观察 | r4 median (µs) | r6 median (µs) |
| --- | ---: | ---: |
| TileLang single | 1.260219 | 1.257335 |
| TileLang work-list | 6.618529 | 6.843774 |
| MSL comparison | 1.172840 | 1.134929 |

该 fixture 没有注册 baseline；`speedup=None`，这些数字不是 DSL 性能排名或发布结论。两阶段 work-list 是机制证明，不是性能候选。

### 负例

- `KERNEL=42`：frontend 拒绝，`compile_error + partial`，候选身份匹配、无 artifact set、零 dispatch；
- attempt 外 canary 读取：Seatbelt `PermissionError`，partial、零 dispatch，canary 字节不变；
- `y[index+1]` 越界写：真实编译有 full build，但 guard/correctness 判 `unsafe`，不可发布；
- generated MSL 宿主 include：executor 在设备执行前拒绝；
- frontend/options/adapter 漂移：typed 拒绝；
- re-sealed native candidate 伪造：generic native digest 检查后，仍被 Track-specific candidate 审计拒绝；
- timeout/crash 投影：固定测试证明不得伪造 full build。

## 失败与噪声历史

`evidence/history/README.md` 逐项保存：初次 workspace ABI 错误、poison 下未初始化读取、多函数重复 preamble、错误地把 `/etc/passwd` 当成拒绝路径、attempt 路径导致的 IR 不可复现，以及 r5 MSL `noisy=true`。

- `/etc/passwd` 是明确声明的系统只读文件，不是沙箱逃逸；真正的外部读取负例使用 attempt 外 canary。
- r2/r3 是 `ir.SourceName` 未规范化的历史，不用于当前可复现性证明。
- r5 为 **6 passed + 1 noisy failure**；没有删除或改判。r6 是后续独立安静确认，不是挑选同一轮数字。
- 所有正例现在先写实际观察再断言，避免错误预期或 noisy 结果丢失原始记录。

## CPU、结构与 verifier

- MT2 自有固定 CPU/source-only 节点：**8**；
- 最终聚焦 CPU/兼容回归：**105 passed**（含旧 `/1` 显式版本负例、现有 MSL evaluator/task、registry/resume、B 证据写出边界与结构规则）；
- 完整 CPU 安全套件：**4861 passed / 26 failed / 13 skipped**；26 个失败 id 与 `MT1b/p2-epoch-drift.json` 完全一致，没有新增当前回归。首轮发现的 15 个 CLI/registry/A4 兼容问题已修复并逐个通过，环境 `run_detached` flake 单进程通过；没有改写 P2 历史来消除失败（`full-suite-r2.json`）；
- MT2 GPU：**r4 7/7，r6 7/7**；
- `verify.py` 只读重建候选/canonical identity，复核 full/partial build、真实 dispatch、负例、r4/r6 artifact 一致性、shipped registry、manifest，并执行伪造 native 自变异；
- task-pack 闭包仍为 74，示例 digest `b02ec81e…`；中立 TileLang/runner 字节已前进，P2 重录继续 deferred，未改写 P2/MT1b 历史证据。

运行：

```bash
uv run pytest -q -n 0 tests/backends/metal/test_tilelang_mt2_contract.py -m 'not gpu'
uv run pytest -q -n 0 tests/backends/metal/test_tilelang_mt2_gpu.py
uv run python 'docs/架构/多DSL-Track-阶段验收/MT2/verify.py' --check
```

## 尚未完成

- **独立 reviewer/RLAR 已签署 `done=true`**，验收身份与日志摘要见 `review.json`。其非阻断后续项也逐条保留；living-plan 头部状态已随后刷新，其余代码/测试卫生项不在本阶段暗改。
- `available` 不等于生产 LLM 搜索可用：MT4 的 DSL prompt、人工/LLM 提交链、preflight、搜索与正式 fresh confirmation 仍未实现。
- 第二条 DSL、MT0 整体、MT3、Graph、性能发布和 P2 重录均不在本轮完成范围内。
