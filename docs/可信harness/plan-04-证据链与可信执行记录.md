# plan-04：证据链与可信执行记录

状态：**冻结；重启条件已经出现（见下方「现状」）。** 当前只做已有 fail-closed 规则的只读维护，不新建证据框架。

## 当前决定

- **输入**：当前 run 的 proposal、probe、self-check 与 `trusted_record`。
- **现状**：2026-11-20 的 `runs/apple-m4-max/p1-premise-20261120/` 已出现第一条 `trusted_record=true` 的 premise（proposal key `f142f237a50ab0ef`，`evidence_level=host_process`，见 `架构与理论.md` 的「官方报告验证的 premise 可以激活并准确送达」与 `代理反馈.md` 的「P1 premise 激活与配对 smoke」节）。重启条件已满足到「先做该条链的只读审计」这一步；分级与绑定设计仍等两条以上不同来源的证据链。
- **本批不做**：只读审计平台建设、证据等级、宿主双写、`legacy_exception`、签名或任何 active 变化。完整设计见[归档](archive/plan-04-证据链与可信执行记录.md)。
- **保持现状**：已有 premise 必须带本场探针和匹配宿主记录，缺失继续 fail closed；不重跑 kernel、不调用模型或 GPU。

## 重启条件

出现第一条真实 premise 提案或 `trusted_record=true` 记录后，先只做该条链的只读审计：谁写的、绑定什么、是否回放、缺哪些字节。样本足以形成两条以上不同来源的证据链后，才重新评估是否启用归档中的分级与绑定设计；active 降级另走 plan-03 的具名决定，不使用第二套批准格式。
