# 远程评测模拟验收（历史记录）

状态：Phase 2.5 的历史验收记录，日期 2026-10-24；2026-10-27 起旧 SSH 数据面与可运行
runner 已删除。本文只保留当时的证据与边界，**不再提供可复现的部署或运行步骤**。
当前实现见[执行运行时与设备扩展](执行运行时与设备扩展.md)，当前对外协议见
[对外评测服务协议](对外评测服务协议.md)，部署见[远程评测部署与轮换](远程评测部署与轮换.md)。

## 1. 当前结论

- 旧 `RemoteEvaluatorPool`、`CommandTransport`、forced-command 服务端、SSH 自定义外部
  帧和 Phase 2.5 runner 已删除，不再提供第二套评测数据面。
- 新远程配置必须显式 `transport=grpc`；旧 `execution.json` 声明 legacy SSH 或缺少
  transport 时 resume 明确失败，不改写 manifest，也不自动换成 gRPC/local。
- 历史报告仍作为只读 artifact 保留；旧 digest/provenance/HMAC 的验证能力没有删除。
  这不表示旧服务仍然存在或旧 run 仍可 resume。
- `tcenv.remote.protocol` 中仍供一次性 worker 与 `job_worker_bridge` 使用的私有
  帧不是旧 SSH 对外协议；agent→host Unix selfcheck bridge 也不属于旧数据面。

## 2. 当时做了什么

当时环境为 macOS 控制面 → SSH forced command → Ubuntu ARM VM → Docker → evaluator
worker，后端是 CPU-only `FakeNpuBackend`。`sim-npu` 只是设备桩，不模拟 NPU ISA、SDK、
驱动或性能。

`runs/npu-sim-20261024/report.json` 记录了 41 项通过的模拟检查，主要覆盖：

- hello、formal/selfcheck、结构化请求与报告/evidence/receipt；
- 幂等、同 key 不同 body 拒绝、错误 build/provenance 拒绝；
- 帧拆包/合包、Unicode/大源码、大小上限、慢读写与有界内存；
- shell/路径/`docker exec` 注入拒绝、stdout 数据帧与 stderr 日志分离；
- transport timeout、断连清理、重启恢复、half-open/慢消费者；
- 多 run 公平/aging、worker 进程组与容器清理。

报告字段为 `split_control=true`、`container_only=false`、
`coverage=extended_split_control`、`simulated=true`、`publication_allowed=false`，
没有 `run_config.json`，最终没有遗留 evaluator 容器。VM 本地延迟只用于控制面诊断，
不能用于推导真实站点的性能或搬代理阈值。

## 3. 已删除的复现材料

以下旧实现与专用 fixture 已不存在，不能再按历史命令运行：

- `tcenv/remote/client.py`、`tcenv/remote/service.py`；
- `tests/support/npu_sim/forced_command.py`、`service_entry.py`、`phase25_runner.py`；
- 旧 forced-command/service-entry 专用测试。

保留的 Docker 基础镜像或其他通用 fixture 不构成旧服务的复现能力。当前 gRPC 模拟
证据见 `runs/grpc-sim-20261025/report.json`；当前一次性容器执行面证据见
`runs/container-sim-20261026/report.json`。三者是不同协议和不同时间的报告，不能互相
替代或合并成生产验收。

## 4. 不可外推的边界

- 当时没有运行真实 NPU/GPU、付费模型，也没有安装 NPU SDK；报告的 `not_run` 明确记录
  `real_npu`、`real_gpu`、`paid_model`、`npu_sdk_installation`。
- 历史结果只证明旧 SSH/帧链路的协议与生命周期，不是 gRPC/TLS 的外部兼容性证据，
  更不是当前生产验收。
- 当时的环境 provisioning 授权只针对那次 Phase 2.5，不是以后重新部署或删除系统的
  持续授权。
- 所有模拟数字都不得进入 confirmed、catalog、brief 或发布加速比；真实站点、真实设备、
  发布许可、HA/gVisor/Kata 仍需另行验收。
