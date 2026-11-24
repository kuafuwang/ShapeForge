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

## 当前 fixture 迁移进展与阻塞

后续提交 `50e4227`、`c7e0c25`、`01d012b`、`c6bd3d5`、`4668cea`、`a8908f8`、`0e87dd9`、`a432bc9`、`82e378f`、`01028dd` 已迁移 run-config/seed、脚本编排与 resume、schema-4、fake-NPU/job worker、批量/委员会/target、report readers、SessionBrief、差分复核、experiment suite 和当前指纹投影。直接 import `v1_tasks`/`v1_specimen` 的文件从 60 降到 42；其余文件主要是显式 legacy、迁移 oracle 或硬件 prompt 专用测试。

完整 subject fixture `v2_subject_confirmation` 具有 candidate-bound reports、fresh seed、expected、identity 与 digest，能通过 `verify_subject_document` 和读盘发布谓词。fake transport 的轻量结构投影不声称 deployment identity，可选 evaluation identity 保持 `null`；需要 deployment/geometry 的测试使用完整 candidate/preflight fixture。

两个不能靠过滤或改名解决的边界：

1. `test_ieee.py` 的改动已撤回。`affine_ramp`/`kda` 没声明 IEEE case，varlen 没有 `small` 档，而 `sigmoid_shift` 虽声明 IEEE，v2 默认 sampling 却不注入特殊值。修改 `tasks/` 或全局 sampling 会移动任务和输入身份，需要独立授权与证据；不能把这些任务过滤后宣称全任务 IEEE 覆盖。
2. `test_hw_feedback.py`、部分 `test_hw_features.py` 仍依赖 v1 prompt 和 LoweringProfile 常量头。v2 evaluation task 的 OpSpec 常量不是那份 prompt/合同原文，不能机械替换；这些测试暂时仍是显式 legacy，除非另做 v2 专用硬件分析实现。

旧逐批确认、旧远程 evaluator adapter、pre-R2 digest 和 frozen migration oracle 继续隔离在 `make_legacy_orch` 或专门的 legacy fixture 中，不计入当前 v2 覆盖。`tests/conftest.py` 的全局 specimen hooks 只有在这些剩余消费者全部迁移或明确退役后才能删除。

## 最终退役边界（2026-11-24）

提交 `c5d25a6` 删除了全局 specimen hooks、`v1_loader`/`v1_only`/生产 specimen 分支，以及旧逐批确认、旧远程 adapter 和 v1 validation 的执行套件。当前 binding、orchestrator、resume、catalog、local provenance 和 launch deadline 都只走 v2。`v1_specimen.py` 与旧 backend-neutral driver 已删除。

`v1_tasks.py` 只剩冻结的题名常量，因为关闭的清微模拟包仍把它当报告标签 import；它没有 `v1_task()`、任务装配器或生产调用。历史候选与合同文本由只读 `v1_archive.py` 承载，不能再构造 v1 `Task`。`hw_golden.py` 也改为历史文档的只读接口，不再重新生成 v1 prompt。

79 文件闭包从 `9cac7a93…` 移到 `ac2d4894…`，prompt 不变，纪元为 `20261124-retire-v1-specimen-hooks.json`。完整安全套件的最终数字以本轮最后报告为准；MT1b/hmz 的环境边界必须单独列出，不能混入 v1 回归。

最终完整安全套件为 **5735 passed、17 skipped、7 failed**。失败分类如下：MT1b 缺 staged TileLang frontend；两个 P2 trust digest 是既有历史锚点漂移；`status_is_clean` 是当前 anchor-only 提交的 worktree-head 注册限制；observe SSE、NPU cleanup、真实 scorecard 三项均在单进程复验通过，属于并发/语料窗口问题，不是本批回归。没有把这些失败删掉或转成 skip，也没有把它们冒称全绿。
