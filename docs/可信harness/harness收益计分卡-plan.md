# Harness 收益计分卡实施计划

## 父任务

目标不是优化某个算子，而是判断 ShapeForge 的 harness 改动能否在多个任务、多个 backend 上带来可复核收益。Metal 使用真实设备证据；清微 NPU 当前仅允许 offline simulation，不能据此声明设备正确性或性能提升。

本计划不调用付费模型、不跑 GPU、不启动搜索。后续是否开展受控实验，必须根据本计分卡的结果另行授权。

## 非目标

- 不用 conv2d 或任何单一算子的结果代表 harness 收益。
- 不把不同任务、shape、dtype、测量边界下的 speedup 直接求平均。
- 不把 NPU 模拟当成真实 NPU 证据。
- 不把 search best 当发布加速比；fresh confirmation 与搜索期数字分开。
- 不为了得到“改进”而调整分母、剔除失败 run 或隐藏不可判数据。

## 数据入口

只读使用现有材料：

- `runs/**/result.json`、`state.json`、`run_config.json`
- `candidates/*/report.json`
- `sessions/*/task_delivery.json`
- `paired_comparison.json`
- `harness_proposals.jsonl`、`harness_overlay.jsonl`、路线 M 留痕
- `feedback.jsonl`、自测账本和宿主权威报告
- `targets/*/experience.md` 与 `taic harness audit-experience` 的作用域审计

不重建或修改历史 run。原始 run 缺失字段必须是 `null/unavailable`，不得写成 0。

## 计分维度

### A. 搜索效率

- 会话数、候选数、明确 ok 候选数与比例。
- 到第一颗明确 ok 候选的轮次；没有则 `null`。
- 未交核、incorrect、unsafe、timeout、crash、其他失败分列。
- 每颗 ok 候选的模型成本与墙钟时间；成本未知单列。
- 同窗口可判配对数与不可判配对数，不把 `too_noisy` 当错误。

### B. 测量可信度

- `timed_by` 分布；旧 `gpu=true` 不得冒充 GPU 时间戳。
- A/A 对照、配对比区间、`too_noisy` 与不可判覆盖。
- 候选/几何/bundle/artifact/native/evaluation 身份不匹配计数。
- fallback、不可比 speedup、跨窗口比值和混合测量边界的标记。

### C. 反馈与知识价值

- `host_process` 证据占比；模型脚本、宿主工具、人工确认分级。
- 前提降级、未验证、无前提声明分别为多少；无声明不能算 0%。
- 被否决前提及其数字是否保留，后续是否重复提出相同主张。
- `verified` 经验的有效指针、缺失指针、模拟声明、target/任务错配。

### D. 信息送达

- 静态合同省略必须为 0。
- 动态实际/省略字节及分布；overlay、host ledger、peer tips 分层。
- NOTES、证据、历史等其他块的省略，不能只报 situation delivered。
- 指定保留信息是否真正到达，不把 delivered 当作已理解或有效。

### E. 治理与后端边界

- prompt/protocol 变化后旧授权是否失效。
- Metal 专属功能与通用 harness 功能分开标记。
- NPU 只输出 `simulated=true`、`publication_allowed=false` 的协议/模拟指标。
- 任何自动停权、自动晋升、自动跨 target 传播都不在本计分卡授权内。

## 输出与分组

新增只读入口：

```bash
uv run python -m taic.cli harness scorecard
uv run python -m taic.cli harness scorecard --json
```

输出 schema：`shapeforge.harness-scorecard.1`。

按以下键分别出表，禁止只给全局均值：

- `target_id / backend / task / dtype / size`
- `provider / model`
- `protocol_version / prompt_sha1`（有则记录）
- `evidence_level` 与 `simulated`

每组包含 `runs`、`sessions`、`coverage`、`counts`、`ratios`、`notes`。样本不足的组标 `insufficient`，不与充足组混合。原始 run 路径和生成时间必须保留。

## 实施步骤

### H0：数据盘点与口径冻结

- 列出可用 run、缺失字段、协议版本、真实/模拟 backend。
- 固定 ok、未交核、first-ok、成本覆盖、配对可判、证据等级的定义。
- 若某指标无法从现有文件重算，标 `unavailable`，不发明代理指标。

### H1：只读计分卡实现

- 新模块放在通用 `taic/` 或 `observe/`，不得 import Metal/NPU backend 到 shared 测试。
- 复用现有 seed-decision、dynamic-budget、omission-content、experience-scope 审计，避免第二套口径。
- 提供 JSON 与人类摘要，真实/模拟/不可判分栏。

### H2：自然 run 分析

- 按 task/backend/model 输出当前基线。
- 只描述关联，不把自然 run 的前后差异写成因果。
- 列出最适合做后续受控实验的 1–2 个 harness 改动及理由。
- 将结果写入实验索引与 N5，父任务进度同步。

### H3：后续受控实验（不在本计划授权内）

只有 H2 找到跨任务候选后才设计：同任务、同候选、同 seed、同模型，按模型分层 AB/BA；Metal 与 NPU 分开，NPU 仍需真实设备授权才能谈设备效果。预算、停止条件、主指标预先登记。

## 测试与验收

- 合成 run 覆盖：有/无 ok、未交核、first-ok 缺失、未知成本、too_noisy、fallback、缺失回执、模拟 NPU。
- 验证比例分母、缺失值、分组不串线、speedup 不被平均。
- 真实 run 只读与内部一致性，不钉死当前 run 数。
- shared 测试不得 import backend。
- 文档完整性、结构、相关审计测试通过。
- 最终交付包括：实现提交、测试数字、真实计分卡摘要、仍不可判的项、下一步是否需要付费及理由。

## 父任务汇报要求

每个阶段结束都同时说明：父任务总目标、相关已完成项、剩余项、本阶段改变的是机制/数据质量/证据/效果中的哪一类，以及下一步和证据位置。不能只汇报改了哪些函数。
