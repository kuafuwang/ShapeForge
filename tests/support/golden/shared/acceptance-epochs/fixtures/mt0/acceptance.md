# MT0：DSL 选型与独立路线可行性审计（blocked）

## 1. 结论与边界

**MT0 本轮为 `blocked`。审计了两个具体、真实、非 MSL 的 `kernel_pipeline` DSL 候选，但当前没有两种满足全部硬条件的 DSL，不能把任何一个新 Track 标为 `available`，也不得开始 MT1b/MT2–MT7。**

| 候选 | 固定版本 / commit | 真实 frontend 与调度证据 | 实际产物与执行路线 | 本轮判定 |
| --- | --- | --- | --- | --- |
| TileLang | `0.1.15` / `a35f8ddf45eba16c21211ec8822d56ce5363036f` | `T.Kernel`、scope/layout/swizzle、Metal `SelectInst`/`GemmMetal` 和独立 codegen；TVM 子模块 `4211874e9e4b8770fd20ad0d98ab80afd1df5028` | host/device/rt_mod、逐 kernel MSL；`newLibraryWithSource` → pipeline → 真实 grid/block dispatch；`MetalRawStream` 可由外部 owner 持有 CommandBuffer | 具体候选，但 `embedded_host_code`、未安装、共享底座集成未证，**不合格 / interface_only** |
| Halide | `22.1.0` / `a8c592b056c06eeeeb064778e6f83e3c6002338f` | Halide algorithm/schedule IR、GPU block/thread/vector schedule 和 `CodeGen_Metal_Dev` | AOT host/device 产物、MSL；配置 Apple 工具链时还有 `kernel.metal.ir` 与 `kernel.metallib`；runtime 真实 block/thread dispatch | 具体候选，但 `embedded_host_code`、未安装/LLVM 本地兼容未证、stock runtime 所有权未接共享底座，**不合格 / interface_only** |

二者不是同一 frontend 改名：TileLang/TVM 的 tile/layout/pass 体系与 Halide 的 C++ algorithm/schedule 编译体系彼此独立。二者也都有真实编译与 dispatch 语义，不是 JSON/S-expression 字符串拼接玩具。

但“后端能生成 MSL/AIR/metallib”只解决下游形态，不能消除以下 release blocker：

1. 两种候选正常写法都会执行宿主 Python/C++，本轮分类均为 `embedded_host_code`。现有 bounded worker 的 timeout、进程组和环境清理不是经实证的文件写集、网络、凭据、临时 HOME/cache 和子进程隔离。没有这层证据，最高只能 `interface_only`。
2. 本机没有安装固定版本的 `tilelang` 或 `halide`，所以没有本地 frontend build、AIR/metallib 或执行产物可记录；`local_observed` 明确为空。
3. TileLang 的外部 CommandBuffer 接缝和 Halide 的 AOT 产物只是可行设计线索。尚未证明 ShapeForge 的 buffer 生命周期、只读输入保护、output/workspace guard、实际 dispatch、commit/wait、Host 计时和 fresh evidence 不被上游 runtime 绕过。
4. 本轮未生成 kernel、未进行前端编译、未运行 GPU/NPU、未调用模型或付费搜索，也没有性能结论。上游文档中的性能快照没有进入选择依据。

MT0 blocked 不阻断独立的 MT1a 与 MT5；它不授权 MT1b 的具体 DSL role/kind/schema，也不授权 MT2、MT3、MT4、MT6 或 MT7。

## 2. 证据包

- `dsl-selection.json`：完整两种候选的名称、版本、来源、license、frontend/compiler、产物、执行路线、`execution_model`、Host 调度、cache/isolation 要求、独立路线证明、阻断原因和下一授权点。
- `upstream-map.json`：14 条来源台账。固定 tag 由 GitHub API 解析到完整 commit；每条证据摘录有独立 SHA-256 与检索引用，不改写 G0。
- `capability-matrix.json`：能力/风险矩阵，区分 upstream source、local inventory、unverified 与 blocked。
- `vertical-slices.json`：两种固定人工候选的设计规格，明确正/负例；只是设计，不是 MT1b/MT2 开工许可。
- `local-toolchain.json`：本机包与工具链盘点。`tilelang`/`halide` import/distribution 均不存在；MLX 为 `0.32.3`；Apple clang 为 `21.0.0`，`xcrun` 可找到 Metal 工具。
- `verify.py` / `manifest.sha256`：结构、来源摘要、负例自检、manifest 和固定 pytest 节点。
- 共同起点：`../start-state.json` 与 `../start-worktree.sha256`；HEAD `efb8218587a3deb9752a0f9dc70035afe5fb26e2`，起始 manifest 摘要 `8ee2e3688e35700ec4a1a771720e232488a39a2ef0676866a62ab8849152ff86`。
- 全量 CPU 回归及环境失败分类：`../cpu-regression.json`。

## 3. 固定正例、负例与 node id

正例固定节点：`tests/shared/test_mt0_dsl_selection.py`，6 个正例 + 5 个负例，共 11 个测试；完整 node id 与负例清单见 `nodes.json`，覆盖：

- blocked 状态与阶段依赖；
- 两种固定 DSL 身份、frontend、产物和 dispatch；
- 不把上游产物冒充本地构建/Metal 证据；
- frontend/调度/交付独立性和不经过现有 MSL Track；
- 来源摘录哈希与性能数字排除；
- 垂直切片设计边界和玩具/保留 v1/graph/CUDA/CANN 路线拒绝。

负例：`verify.py::self_test` 固定 5 个变异，并在 `nodes.json` 的 `negative` 中给出对应的 5 个独立 pytest node id，分别验证：

1. 未经隔离把 embedded candidate 改成 `available`；
2. 偷改为复用现有 MSL Track adapter；
3. 把 blocked 改成 complete；
4. 虚构本地构建产物；
5. 在 MT0 未过时进入 MT2。

`verify.py --check` 还逐项重算 manifest，任何证据字节漂移都失败。失败时按 `tests/README.md` 先读 `logs/pytest/triage/triage_summary.md` 和对应 handoff，不删除、skip/xfail 或弱化断言。

**验证强度边界：** `verify.py` 是结构回归器，硬编码的是本次 blocked 结论和两个固定身份，防止以后把 blocked 偷改成通过；它不是独立上游验证器。`upstream-map.json` 的 excerpt hash 也只能证明本包内摘录未漂移，不能证明摘录对应的真实上游构建。独立上游/本地构建验证必须走第 5 节的下一授权点。故本包不得被描述为“MT0 已通过”，MT0 仍然 blocked。

## 4. 明确排除的凑数路线

- MLX `mx.fast.metal_kernel`：候选仍是 MSL source 字符串，未形成独立非 MSL frontend/调度语义。
- MPSGraph / MLX graph：属于后续 graph ExecutionKind，缺少 tagged ExecutionPlan/CanonicalBundle 接缝，不满足首期 `kernel_pipeline`。
- preserved Triton v1：禁止恢复生产路径。外部 `triton-msl 0.3.0` 仅记录为未验证方向，未安装、嵌入式隔离和本地 Metal 证据均不足，没有计入。
- 玩具文本模板、同 frontend 改名、CUDA/CANN/Ascend 路线、清微模拟：均没有用于增加合格数量。清微仍只能 `simulated=true/publication_allowed=false`，且 NPU-P0 不解除。

## 5. 下一授权点

1. **选型确认**：用户审阅 TileLang 与 Halide 的嵌入式宿主代码和共享底座风险后，确认是否仍以这两者为目标；本文件不冻结任何 MT1b schema。
2. **隔离授权与实证**：为 Python/C++ 候选建立独立 OS 级写集、网络、凭据、子进程隔离证据；或者证明一个只由受信 parser 接受的真正 declarative 子集。不得用 import/ctypes 黑名单代替。
3. **离线 frontend 授权**：允许安装两个固定开源 frontend，运行不产生/不执行设备 kernel 的版本、帮助与 source-only build 探测，记录真实产物和 compiler/options identity，再重新评审 MT0。
4. **后续设备授权**：只有 MT0 通过且 MT1b 单独获批后，才申请 MT2/MT3 固定候选的真实 Metal build/execute。真实 LLM、`taic run/resume` 和付费性能搜索仍需另行授权。

本包没有 reviewer 自签文件；最终 reviewer 结论只通过本轮要求的合法 `{done, notes}` JSON 返回，不伪造独立 RLAR 签名。
