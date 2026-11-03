# 清微 NPU 离线数据与协议模拟

状态：**仅模拟；`simulated=true`、`publication_allowed=false`、NPU-P0 仍 `blocked`。**

本目录不实现清微编译器、ABI、设备模型或计时，也不产生 correctness、speedup、性能或发布结论。
它只验证数据、字节、摘要、ArtifactBundle 和现有 fake-NPU 协议链。

## 两条独立证据

1. **v2 `sigmoid/fp16/small` 数据模拟**
   - 使用生产标准目录的 OpSpec、reference 和 deployment pack 身份；
   - 以 `case=normal, seed=7` 生成输入与 fp64 reference 输出；
   - 另生成 fp16 候选输入，明确它不是设备执行结果；
   - 保存完整张量的 shape/dtype/SHA-256，并把前 256 元素的确定性窗口放入 ArtifactBundle；
   - 验证 payload 往返、materialize 后逐字节一致，以及任意单字节篡改被拒绝。

2. **preserved v1 `rope` fake-NPU 协议模拟**
   - 使用现有 `NpuGrpcHarness`、真实服务核心和 fake vendor A；
   - 走 `EvaluationRequest → bind_wire_digest → gRPC → receipt/evidence`；
   - 校验 candidate digest、device group、请求摘要及同步完成；
   - 结果必须是 `measurement_kind=simulated`、`publication_allowed=false`。

**两条证据不混成同一任务或端到端 correctness 声明。** rope 只是现有协议夹具，不与 sigmoid
张量共享任务身份。报告有意不包含 host/device 耗时、speedup 或 confirmed 数值。

## 重放验收

```bash
uv run python docs/架构/清微NPU数据模拟/verify.py
```

只重跑模拟并逐字段比较 `report.json`：

```bash
uv run python docs/架构/清微NPU数据模拟/simulate.py --replay
```

重新生成报告（只有任务、协议或预期数据确实变化时才允许）：

```bash
uv run python docs/架构/清微NPU数据模拟/simulate.py \
  --output docs/架构/清微NPU数据模拟/report.json
```

`manifest.sha256` 固定脚本、说明和归档报告。临时 gRPC 状态、vendor ledger、materialize 目录均在
临时目录中创建并删除，不写入 run、confirmed、catalog 或 brief。
