# MT1a：DSL 无关去特判、动态构造与兼容锚点（CPU/协议范围完成）

## 1. 范围结论

本包只完成获授权的 **MT1a DSL 无关改造及 MT5 所需协议接缝**。没有开始 MT1b，没有冻结任何 DSL-specific role/kind/schema，没有改多版本 registry，也没有启动 MT2–MT7。MT0 仍为 blocked。

本包是 **CPU/协议验收**，不是现有两条 Track 的真实 Metal 重验收；用户明确禁止本轮 GPU/NPU 与付费搜索，因此没有运行真实 Metal anchor，也没有性能声明。

## 2. 实际实现

### 2.1 Registry 与 CLI 动态读取

- `taic/cli.py` 新增 `_run_track_choices()` / `_resume_track_choices()`。
- 新 run 从 `registry.available()` 取选项；resume 从 `registry.tracks` 取选项，既有 recorded id/version/delivery 与 availability 复核仍由原路径执行。
- 现有帮助仍显示 `--track {msl-single,msl-bundle}`；registry 顺序固定。损坏 registry 没有硬编码 fallback。
- `backends/metal/tracks.json`、`tcenv/track/schema/tracks.v1.schema.json` 和多版本语义均未修改。

### 2.2 Adapter 延迟注入与 fail-closed

- `tcenv/track/adapter.py` 是 DSL 无关的 `LazyTrackAdapter`：接收调用方已经选定/冻结的 `TrackSpec`，第一次 `resolve()` 才构造服务；不读默认 registry，不定义新 role、delivery、ExecutionKind 或 media type。
- `tcenv/backends/metal/track_factory.py` 按既有 renderer/validator/build implementation identity 构造当前实现。未知 identity/version 返回 `UnsupportedContractCapability`，不能落入 `msl-single`。
- `MetalBackend.track_adapter()` 延迟调用工厂；`search_candidate.py` 使用同一 registry 选择构造 builder/evaluator/geometry resolver；`v2_worker.py` 先从 `frozen_track` 重建选择，再交给工厂，未让 worker 读取默认 registry。
- `geometry_resolver` 只作为现有 kernel-pipeline 实现细节暴露，**没有宣称 `GeometryPair` 是未来 Track 的统一执行上限**。
- `taic/track_prompt.py` 按 renderer identity 选择现有 prompt/submission style；未知 renderer 不得收到 MSL bundle 说明。辅助选择函数已加入 `taic.trust` 的 prompt source 指纹，模型可见文字仍留在原指纹覆盖函数内。

### 2.3 现有特判清理与输出兼容

- `TaskBinding.signature_stub()` 改为读取 registry + prompt style，不再直接比较 `msl-bundle` 名字。
- `system_prompt_v2()` 使用已解析 delivery/style；`kernel_agent` 的 `_check_command`、`render_contract` 和 `collect_session` 使用同一 style，不再以 Track 名字决定 bundle 行为。
- Orchestrator 的 workspace 前置检查按 `track.delivery == "candidate_bundle"`；在生成 session 前验证 adapter，未知实现不会先烧模型轮次。
- 默认 `msl-single`、v1 限制、bundle envelope 字节读取、single 自测命令省略 `--track` 等兼容行为保留。
- MSL candidate projection、native descriptor/ExecutionKind、多版本 registry、P2 portfolio 专用内部逻辑没有在本轮泛化或伪造；这些属于 MT1b 或更后阶段。特别是 `tcenv/portfolio/candidate.py`、`tcenv/taskcontract/portfolio_runner.py`、`tcenv/backends/metal/portfolio_confirmation.py` 的 MSL-only 投影/expected 分支仍是 MT1b 边界，不能由本包的通用 registry/adapter 名称宣称支持新 DSL。`test_an_unregistered_track_is_rejected_before_any_msl_projection` 固定证明未知 Track 会在进入这些分支前 fail closed。

### 2.4 MT5 协议接缝

- `confirmation_master_seed()` 统一现有 `taic-v2-search:<run_seed>` 派生，供 run 与 MT5 共同使用。
- 仅 v2 新 run 的 `run_config.json` 记录 `confirmation_guidance_digest`；`_confirm_v2` 使用同一份 guidance。v1 run_config 不增加该字段。resume 使用记录值并在 `restore()` 按 snapshot system 重算/交叉校验，snapshot 不匹配为 `NeedsInspection`；pre-MT1a 无记录时才从恢复文本派生，不能用当前 prompt 静默改写旧 seed。
- Metal adapter 的 `confirmation` 服务以 CPU-only 方式重建现有 `PortfolioFreeze/ValidationPlan` 并返回 freeze/validation digest、实际 confirmation seed 和派生版本，供方案 A 核验；不创建第二套确认或 evaluator。
- `TAIC_TRACK_BUILD_TIMEOUT_SEC` 是可选的 per-process build deadline 注入；未设置时行为不变。设置后必须有限、正数且不超过 worker timeout。它不改变 pack 或计时口径。
- `taic/track_explore.py` 的 versioned schema、comparability/selection policy 与实际汇总在 MT5 包验收；MT1a 只提供这些通用接缝。

## 3. 指纹与历史证据

`baseline.json` 在任何 MT1a 源码修改前记录，`after.json` 在改造后独立重算：

| 项目 | 结果 |
| --- | --- |
| Track registry 文件与 document digest | 未变 |
| 35 个 v2 task deployment pack digest / locator identity | 全部未变；静态闭包仍为 73 个 code files |
| 70 个 `(task,dtype,size,track)` presentation/statement/signature 记录 | 全部未变 |
| 两条 Track 的完整 system prompt、TASK.md、quick/timed command | 全部逐字节未变 |
| `current_prompt_sha1()` | `5fb6aa1ff55e03b72d10c5dca160aacc1e95691d5245ba46f9114d9fc05d515c` → `3fc1b8356c78876f5d6401e3a3c75daca6e4a3b89a68880dc20c201f3419b892` |

源码指纹前进是预期行为：`signature_stub`、`system_prompt_v2`、`_check_command`、`render_contract` 及新 prompt-style 选择函数都属于受信 source surface；不能把“输出未变”误写为“源码指纹未变”。

`docs/架构/backend-phase0-baseline.json` 未改写。`tests/backends/metal/test_backend_phase0.py` 现在同时钉住：

1. 历史文件仍是 pre-MT1a 指纹及其原始字节；
2. live 源码等于 MT1a 新 epoch 的指纹。

这不是回填或弱化旧断言，而是新增 epoch 后保留两个方向的校验。baseline 中 19 个 `signature_stub` 对既有 `u32` dtype 的 `KeyError` 也原样记录，未借本任务做无关修复或掩盖。

完整 golden 文本位于 `golden/gemm-msl-{single,bundle}-TASK.md`。共同起点见 `../start-state.json` 与 `../start-worktree.sha256`，HEAD 为 `efb8218587a3deb9752a0f9dc70035afe5fb26e2`；其他会话已有文档改动未回退或覆盖。全量 CPU 回归及环境失败分类见 `../cpu-regression.json`。

**Prompt 授权影响：** `targets/apple-m4-max/trust.toml` 中三个既有 grant 的 `prompt_sha1` 都是 `aa71abd38a9843d67d329dd7485eff3d3efccb913e6cdc622ca305321c42a4e5`，在 MT1a 前就已相对当时的 `5fb6aa1f...` 失效（pre-existing stale）。因此没有一份“当前有效 grant”因本次 `3fc1b835...` 而新增失效；本包没有自动重签、激活或宣称这些 grant 恢复有效。`after.json` 的结构化 `grant_impact` 同步记录这一边界。

## 4. 固定正负例与 node id

新增 23 个 CPU 测试（完整 node id 见 `nodes.json`）：

- `tests/shared/test_track_mt1a.py`（11）：动态 available/resume choices、lazy resolve、未知 renderer、两条完整 golden、prompt 新 epoch、registry 不变、v1 run_config 不增加 v2 字段，以及 3 个 verifier 负变异固定 node id。
- `tests/backends/metal/test_track_factory.py`（10）：当前两类构造、无默认 registry 读取、未知 implementation version、build timeout 正负例、guidance 绑定、方案 A 两条实际派生路径、未知 Track 不得进入 MSL 投影，以及 resume 使用记录/snapshot guidance 并拒绝不匹配/非法 digest。
- `tests/backends/metal/test_backend_phase0.py`（2）：历史/新 prompt epoch 与既有 GPU 记录重算（只读，不执行 GPU）。

现有回归固定为 86 个，覆盖：

- `tests/shared/test_search_v2_bundle_prompt.py`
- `tests/shared/test_search_v2_bundle_track.py`
- `tests/shared/test_search_v2_prompt.py`
- `tests/shared/test_search_task_binding.py`
- `tests/shared/test_trust.py`
- `tests/backends/metal/test_search_v2_bundle_session.py`
- `tests/backends/metal/test_track_registry.py`
- `tests/backends/metal/test_track_a4.py`（仅 `not gpu and not llm and not slow`，24 个设备/慢节点 deselect，未 skip/xfail）

`verify.py::self_test` 另有 3 个负例：pack 篡改、prompt 指纹未前进、registry digest 篡改，且每个都有独立 pytest negative node id（见 `nodes.json` 的 `negative`）。`verify.py --check` 要求精确 86 + 23 passed，并重算 before/after/live surface 与 manifest；数量不符、skip/xfail、字节漂移或历史回填均失败。

## 5. 未做事项与下一授权点

- 未运行真实 Metal，未验证新 DSL，未运行真实 `taic run/resume/track-explore`，未调用模型或付费搜索。
- 未改六 RPC、field number、HMAC、BlobChannel，未恢复 v1 生产路径，未引入被禁的 pattern/lowering/stage/tiling registry。
- MT0 仍 blocked，所以 MT1b/MT2–MT7 不开工。若后续 MT0 重新评审通过，MT1b 仍需独立授权。
- 在声称 MT1a 的真实 Metal 兼容锚点复验完成之前，还需单独 GPU 授权，运行既有 `test_track_a4.py` 与 bundle 的真实 Metal 固定节点；本包不冒充这一步。
- 最终 reviewer 结论只通过本轮合法 `{done, notes}` JSON 返回，不伪造独立 RLAR 签名。
