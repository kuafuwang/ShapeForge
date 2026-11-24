# 清微 NPU：v2 数据与协议离线模拟

这一目录是旧清微模拟包的 v2 接替者。旧包已经退役到 Git 历史；本包重新生成协议证据，不修改旧包的任何字节。

## 边界

本包仍然没有清微编译器、SDK、驱动或真实设备。所有结果固定：

```text
simulated=true
publication_allowed=false
npu_p0.status=blocked
```

它不能证明设备正确性、设备计时或加速比，也不能解除 NPU-P0。

## 两道彼此独立的 v2 题

| | 数据题 | 协议夹具 |
| --- | --- | --- |
| 任务 | v2 `sigmoid / fp16 / small / seed 7` | v2 `sem_attention_odd_scale / fp16 / small` |
| 证明内容 | 张量摘要、ArtifactBundle 逐字节往返、单字节篡改拒绝 | 现有 fake vendor-A 的 v2 请求、签名、回执和 v2 报告投影 |
| 身份 | 真实 `TaskContract` fingerprint、contract/deployment pack digest | fake 服务登记的 v2 structural binding；不是 deployment 证据 |
| 不能推出 | sigmoid 在清微上算对 | attention 在清微上算对或更快 |

数据题和协议夹具仍然是两道独立的题，不能合并成一次设备验收。`EvaluationRequest` 是共享传输信封；本包不读取 v1 task registry、v1 specimen 或 v1 题名模块。

## 相对旧包的迁移

- 报告 schema 从 `local.qingmicro-npu-data-simulation.1` 升级为 `.2`。
- 协议夹具改成 v2 attention，候选源和请求身份重新生成。
- 报告明确记录 `protocol_task.spec_version=2` 和 structural fake-service 身份。
- 只保留一份当前 `report.json`；旧 `report.json` 与 `report.current.json` 留在已退役包的 Git 历史中。
- 六 RPC、字段号、HMAC、BlobChannel 和 remote EvalReport schema 均未修改。

旧包恢复命令：

```bash
git checkout c6df47a4e98dca836ed4d8182f23d6051e648bc9 -- docs/架构/清微NPU数据模拟
```

恢复只用于审计历史，不是现行协议的输入。

## 验收

完整检查 manifest、模拟边界和逐字段重放：

```bash
uv run python docs/架构/清微NPU数据模拟-v2/verify.py
```

只重放并与当前报告比较：

```bash
uv run python docs/架构/清微NPU数据模拟-v2/simulate.py --replay
```

如果协议或任务合同确实发生批准后的变化，应新建证据纪元，不能为迎合当前代码静默改写已关闭报告。
