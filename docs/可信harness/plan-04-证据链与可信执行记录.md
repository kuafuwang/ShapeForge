# plan-04：证据链与可信执行记录

状态：**冻结，出现 premise 提案后重启。** 当前只做已有 fail-closed 规则的只读维护，不新建证据框架。

## 当前决定

- **输入**：当前 run 的 proposal、probe、self-check 与 `trusted_record`。
- **现状**：结构化提案中没有 `surface=premise`，也没有 `trusted_record=true`；此时建设 E0–E4、E2H、freshness、签名或 active 降级只会产出空链。
- **本批不做**：只读审计平台建设、证据等级、宿主双写、`legacy_exception`、签名或任何 active 变化。完整设计见[归档](archive/plan-04-证据链与可信执行记录.md)。
- **保持现状**：已有 premise 必须带本场探针和匹配宿主记录，缺失继续 fail closed；不重跑 kernel、不调用模型或 GPU。

## 重启条件

出现第一条真实 premise 提案或 `trusted_record=true` 记录后，先只做该条链的只读审计：谁写的、绑定什么、是否回放、缺哪些字节。样本足以形成两条以上不同来源的证据链后，才重新评估是否启用归档中的分级与绑定设计；active 降级另走 plan-03 的具名决定，不使用第二套批准格式。
