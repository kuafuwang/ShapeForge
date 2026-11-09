# MT4：TileLang 的 LLM/人工候选生成与工具链接入

## 状态与范围

**MT4 已完成：实现、真实模型生成、真实 Metal replay、fake-agent 运行/resume、只读 verifier 与独立 reviewer 均通过。**独立 Reviewer 返回 `done=true`（`review.json`），但只签署下述 correctness-only 范围。

本轮获得 MT4 授权，新增并验证了 `tilelang-pipeline/2` 的**正确性导向**生成与评测：

- TileLang prompt、ABI→PrimFunc 签名、packer、quick check、production evaluate、Orchestrator、snapshot restore 和 resume dry-run；
- 人工固定候选与真实 DeepSeek Flash 生成候选都走同一 `kernel.py + host/execution_geometry.json` envelope；
- `/1` 继续只作 interface-only replay；现有 MSL single/bundle 的投影、生命周期、prompt 与评测保持原行为；
- 正式 fresh portfolio confirmation 与 speedup publication 仍属于后续 MT6。DSL 不会被送进 MSL confirmation。

MT0 整体仍因第二条 DSL deferred 而 blocked，MT3、Graph、批量竞速和性能发布均未开始。

## 生成与执行链

1. `taic/track_prompt.py` 只为 `/2` 注册 `tilelang-pipeline` prompt style；`/1` 仍无生成 adapter。
2. `system_prompt_v2` 与 `TaskBinding.signature_stub` 输出真实 TileLang 规则、ABI 张量形状和 `KERNEL`/`KERNELS` 说明，不再给 DSL 一份 MSL 签名。
3. `scripts/tilelang_pack.py` 只读取文本、校验闭集几何并写 envelope，**不 import/执行候选**。交付文件为 `.taic`，不伪装成 `.metal`。
4. `portfolio_for_submission` 与 `project_provider` 仅在显式选择已登记 DSL Track 时进入 DSL 分支；默认 MSL 投影仍拒绝 DSL 内容。
5. `run_portfolio_task` 复用 single-variant dispatch、schema-4 和实际执行证据。MSL 调用与 monkeypatch 的原有参数形状不变，DSL 才接收显式 `track_id`。
6. 几何-only `schedule` 不被当成额外 prepare step；`msl-bundle` 原有 `host_tiling + schedule` 生命周期保持不变。
7. `V2LocalEvaluator` 走同一 preflight/evaluate 路径；`confirmation_input` 和 `confirmation_expected` 对 DSL 明确 typed deferred to MT6。

## 真实 DeepSeek Flash 切片

运行：`relu/fp16/small`，correctness-only、`max_rounds=0`、seed 4、预算 \$1、单会话 cap \$0.50。

- 只进行一次 DeepSeek Flash 初始生成，未进入优化轮或确认阶段；
- 实际支出 **\$0.295137132**，约 1029.6 秒；
- 生成候选：`bbe388556769de007eab9089a4d6e6a445c7115830d8bd41a5062b062fd57f45`；
- artifact set：`d516c1da859c24e8012028b8060337e63d5e0a97dd3e465f5d8c57497af7a767`；
- 真实 Metal：6 个 case（含 IEEE NaN/Inf/-0）全部通过，ABI/safety 通过，full build；
- `result.json` 的 `best_speedup=1.0` 是 correctness-only 占位值，`confirmed_speedup=null`、`candidate_us=null`，**不是性能或发布结论**；
- 原始 run、报告、候选和 run_config 位于 `evidence/deepseek-relu-r1/`。未把原始模型 transcript 放进仓库。

### 无模型 replay 与 resume

`replay.json` 直接重放 `best.taic`，candidate/artifact identity 不变，新的 native identity 前进，并记录真实 dispatch：

```text
relu_kernel_kernel  grid=(64,1,1)  threads=(512,1,1)
```

普通 `taic resume` 因 run 是 `max_rounds` 停止而按既有规则拒绝（exit 3），保存为正确负例。使用 `--dry-run --lift-count-limits --max-rounds 1` 后，恢复一个候选、原 system prompt 和下一轮位置，**没有调用模型或 GPU**。

## Fake agent / 人工固定路径

- 生产 `evaluate_candidate(quick)` 使用两阶段 work-list，真实 dispatch 为 `scale → bias`；
- fake agent 只提供固定 envelope，不调用 API；真实 Orchestrator 完成 quick correctness，写出 `best.taic`；
- `load_snapshot + restore` 恢复同一候选，未启动新 agent；
- 固定 CPU 节点 **5**、fake-agent GPU 节点 **2** 全部通过。

## 回归与并行化

- 现有 MSL/portfolio/runner/prompt/preflight/evaluate_task：**114 passed**；
- 完整 CPU 安全套件：**4963 passed / 26 failed / 13 skipped**（`full-suite-r2.json`）；
  - 26 个失败 id 与 `MT1b/p2-epoch-drift.json` 完全一致，无新增当前回归；
  - 两个并行环境失败在单进程复核通过，没有改成 skip/xfail。
- `test_tolerance_verdict` 从单节点串行循环改为按 frozen `(task,dtype,size)` 参数化：
  - 组合直接来自 `snapshot.migrated_cases()`，不增加 varlen 或不存在于 R0 的任务；
  - 所有原断言、seed、case 和扰动保持不变；
  - **102 个节点通过，132 秒**，此前单节点约 360 秒，约 2.7× 加速；
  - 每个 xdist worker 内限制 Torch 单线程，避免组合并行与 BLAS 线程叠加。

## 验证

```bash
uv run pytest -q -n 0 tests/backends/metal/test_tilelang_mt4.py -m 'not gpu'
uv run pytest -q -n 0 tests/backends/metal/test_tilelang_mt4.py -m gpu
uv run pytest -q tests/backends/metal/test_tolerance_verdict.py
uv run python 'docs/架构/多DSL-Track-阶段验收/MT4/verify.py' --check
```

独立 Reviewer 已签署 `done=true`（`review.json`）。Reviewer 独立重跑 5 个 MT4 CPU 节点、15 个 A4 节点、8 个 MT2 合同节点、6 个结构节点、148 个 MSL/MT2 兼容节点和 102 个 frozen tolerance 节点，全部通过；actor 的 `track_adapter._evaluate_in_process` 回归修复确认有效。Vector reflection 与 kernel-name 疑点归为 pre-existing 限制，self-check 归为嵌套沙箱环境限制，没有拿这些旧限制冒充新回归，也没有用 agent workaround 替代生产证据。`actor-review-request.json` 保留 actor 的完整请求与摘要。

下一阶段不是自动开始 MT3；需单独决定 MT6 的确认/性能范围。`done=true` 仍只覆盖本文件声明的 correctness-only 生成、评测与 replay。
