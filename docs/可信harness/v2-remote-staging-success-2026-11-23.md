# v2 remote staging：Submit→fresh result→replay→cancel→cleanup 已跑通

本文记录最终代码的真实 VM→Mac staging 结果。前一份 [阻塞记录](real-ssh-remote-staging-2026-11-23.md) 保留为历史，不回写。**本次使用 live OpSpec v2，不使用 specimen；模型成本 $0，没有搜索或付费调用。**

## 修复内容

1. **task-pack provider**
   - 非 specimen 的生产路径改用 `load_task_contract(...).fingerprint()`；
   - specimen 仅保留给 v1 机制测试，不再把 live v2 送入退役 `tasks.get_task`。
2. **worker 求值**
   - `ProductionBackend` 在非 specimen 路径使用现有 `V2LocalEvaluator`；
   - `worker.main()` 修正为传入 `ProductionBackend()` 实例，消除 `function.evaluate` 的入口 `AttributeError`；
   - 当前仅接受 `cache_policy=disabled`。reuse/refresh 明确 typed 拒绝，不静默套用 v1 cache 语义。
3. **macOS worker 创建**
   - remote JobEngine 改用真正的 `posix_spawn`，避免 gRPC 初始化后的 fork-exec poll 状态继承；
   - dup2 后关闭所有其他有效父 FD，包含 gRPC poll pipes；`/dev/fd` 枚举逐项 `fstat`，过滤已失效目录流 FD；
   - pipe 源 FD 等于目标时跳过 `DUP2(fd,fd)` 并显式设置 inheritable；
   - 保留所有目标标准 FD，覆盖父 stdin/stdout/stderr 分别关闭的碰撞；
   - 新 session、进程组 kill、wait/poll/reap 保持，无法枚举 FD 时不回退到 fork。
4. **服务生命周期**
   - 前一提交已修复 `wait_for_termination(0)` 导致启动约一秒自行退出的问题。

独立 reviewer 对最终标准 FD 修复给出 `done=true`，认可提交。

## 最终真实 staging 身份

- client：Ubuntu 22.04.3 / aarch64 VM，经真实 SSH 网络和 mTLS 调用 Mac；
- task：live `relu/fp16/small`，`msl-single`，protocol 3；
- site/device：`v2-staging-m4` / `apple-m4-max`；worker_build=`v2-posix-spawn-staging`；
- task-pack digest `c224bd8dfacb01d737b4b5df46930a035df8a4bfe16d5aba4228ee3de08ed588`；
- task fingerprint `dbda7ad2481958269f0fa4defa469131fb2877e2638c0d4d29b25767ea843e61`；
- candidate SHA-256 `155602d00a64140c46f675d837d594acd1a4e7666c867a5b21da63188f0ebed3`；
- evaluator epoch `5be56f9e6b1ddf9347e953a134b33f4845c12642f763f84d066f189fcbeac98c`；
- `cache_policy=disabled`、`legacy_specimen=false`、`measurement_kind=measured`、`publication_allowed=false`。

## Submit 与 fresh result

最终 request `v2-relu-quick-3` 返回：

- `status=ok`、`correctness.ok=true`；
- attempt `attempt-4-3665611365b146c8`；
- receipt `9f5f9fe8be95237d54d0a78250d81dd4d2624249a141efbcd2950a1704b94141`；
- lease fence 4；signature key `v2-staging-m4/key-1`。

这是实际 Mac Metal 求值结果，不是 fake backend 或 capabilities-only 证据。

## Replay

对同一 request 再执行一次 `formal`：

- attempt、receipt、lease fence 和 evaluator epoch 均与 fresh result 完全相同；
- 没有创建新 attempt 或伪装一次 fresh 重测。

## Cancel

第二个 request `v2-relu-cancel-2`：

- Submit 返回 `queued`，Cancel 响应时为 `running`；
- 最终 durable snapshot 为 `cancelled`，`cancel_requested=true`；
- attempt `attempt-5-5021fba777ef4266`。

## Cleanup 与回归

- 服务收到 SIGTERM 后停止，端口不再监听，没有遗留 worker；
- VM client 目录已删除；本场 CA/client key/HMAC/grant 和专用 state 已删除；
- 最终相关 CPU 回归 **165 passed**；其中 spawn 聚焦 **6 passed**，覆盖 gRPC FD 身份、stdin/stdout/stderr 关闭、进程组回收；
- `git diff --check` 通过。

## 适用范围与未完成边界

- 本次只验收 **live v2 `msl-single` + `cache_policy=disabled` + process-worker 开发 staging**。
- 不把该结果外推为 bundle、Host Program、TileLang、reuse/refresh cache、容器隔离或生产部署完成。
- 没有 NPU 或第二颗真实设备证据，没有 speedup publication；`publication_allowed=false`。
