# v1 正式退役：v1-track 批次

用户于 2026-11-23 明确要求正式退役 v1。本批次只处理可以独立闭合的 **v1-track 归档代码包和它的直接测试入口**，不把后续 specimen 迁移伪装成已经完成。

## 恢复点

- last commit：`bfd58fe716c90c533d948aee1fad68dbf9557263`
- `tests/support/golden/shared/v1-track` tree：`c1a1a61fdcb714ae3508a4769e3d933706b28846`
- 300 个 Git 跟踪文件，3,021,474 bytes（工作区额外的 pycache 不计入）
- 恢复命令：

  ```bash
  git checkout bfd58fe716c90c533d948aee1fad68dbf9557263 -- tests/support/golden/shared/v1-track
  ```

纯 v1 的 `tests/shared/test_track_binding.py` 同批移除，其历史 blob 为 `eb033f32db92313e6d24f75ac4d42f2431db4549`。恢复命令：

```bash
git checkout bfd58fe716c90c533d948aee1fad68dbf9557263 -- tests/shared/test_track_binding.py
```

`retired.json` 没有新增行：该文件的治理合同只接受验收包根，`v1-track` 是归档代码而不是 inventoried acceptance package root，单个测试文件也不是 tree。这里以本文作为合规的代码退役记录，不伪造清单条目。

## 活跃入口变化

- 删除纯 v1 的 `test_track_binding.py`；
- `test_opspec_v2.py`、`test_opspec_varlen.py`、`test_task_contract_v2.py` 移除只针对该 archive 的用例，其余 v2 测试保留；
- inventory/structure 不再为 `v1-track` 保留特殊兼容分支；
- 没有活跃测试或生产代码再按路径打开 `tests/support/golden/shared/v1-track`。

本批验证：**146 passed**，覆盖上述三个 v2 测试文件、structure 和完整 acceptance governance。`git diff --check` 通过。

## production remote fallback 的移除

后续批次已删除 `tcenv/remote/task_pack.py` 与 `tcenv/remote/production.py` 的 specimen/v1 分支，并移除 `grpc_service` 的 specimen 装配钩子。生产 remote 不再允许通过环境变量退回 v1；worker 只走 live v2，cache 仍限定 `disabled`。旧的 v1 cache 映射测试随退役删除，相关 remote 验证为 **106 passed**。

## v1 worker pipeline 与剩余生产 fallback 的移除

后续批次已删除 `tests/backends/metal/test_tcenv.py` 的 v1 worker pipeline 和 `tests/shared/test_triton_specs.py` 的 Triton dry-run，并移除 `taic/task_binding.py`、`tcenv/worker.py`、`tcenv/xcworker.py`、`tcenv/triton_runner.py` 的环境变量 specimen 分支。生产不再提供隐式回退；`tasks.get_task` 仍拒绝所有 live v1 任务。

共享编排测试在 `tests/conftest.py` 显式注入 `v1_task_loader`，只在该题有 specimen 时进入旧分支，不依赖环境变量。track-explore 的 fixture 已按现行发布合同同步 candidate/baseline 的 `median_us` 与 `median_s`，完整 **45 passed**。SSE 串行复验通过；MT1b 的 staged TileLang frontend 缺失单独归类为环境问题，不归入 v1 退役。

## 模块级入口的删除

`tcenv/triton_runner.py` 与 `tcenv/backends/metal/triton_baseline.py` 已删除，Triton CLI 和 backend 的可选比较入口一并移除。外层 `tcenv.evaluate.evaluate()` 与旧 `tcenv.worker.main()` 现在以 `UnsupportedContractCapability` 明确拒绝；v1 xctrace 主路径也拒绝，只有 `_main_v2` 继续服务。`evaluate_in_process()` 和 v2 bounded worker 仍保留，不能把共享报告类型和 v2 求值误删。

相关 backend/registry/profile/backend-neutral 复验为 **52 passed**。Triton runner/baseline 的专用测试、旧 xctrace 测试和直接调用外层 evaluate 的用例随退役删除；v2 分流测试继续执行。

## 共享机制测试的 v2 fixture 迁移

`tests/support/v2_specimen.py` 只装配生产 `TaskContract` / `TaskBinding`，不重建 v1 `Task`、归档 geometry 或历史 digest。结构 build fixture 使用当前 runner 的 `contract.spec.digest + contract.fingerprint()` 身份映射。`tests/support/orchestrator.py::make_orch` 已切到真实 v2 binding 和 `V2LocalEvaluator`；`make_legacy_orch` 只用于当前没有等价 v2 路径的旧逐批确认和旧远程 adapter 测试。

`taic.cli.prepare_resume` 新增可选的完整 executor 注入，生产缺省行为不变；当前 v2 resume 测试不再把 `evaluate_fn` 当成 v2 evaluator。`from_run` 测试使用完整 v2 候选和候选自己的 launch，继续执行真实 preflight。脚本 confirmation 是显式 CPU 替身，不是 fresh Metal 证据。

最终 `test_orchestrator.py + test_resume.py` 为 **136 passed**；当前/显式 legacy runtime 与 R4 组为 **34 passed**，search switches 为 **8 passed**。run-config、seed 和 v2 orchestrator 的迁移另在提交 `50e4227` 中固定。

## 尚未完成的退役工作

`tests/support/v1_tasks.py`、`v1_specimen.py` 仍服务于显式 legacy、旧报告、硬件特征和部分 fake-NPU 测试。下一阶段继续按机制迁移这些测试；只有历史格式/迁移 oracle 才保留归档 v1 输入，不能把全部引用机械改名后冒称完成。
