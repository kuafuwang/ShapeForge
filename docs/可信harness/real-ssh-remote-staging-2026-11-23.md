# 真实 SSH 与远程 Metal staging：通过项和阻塞项

本次在用户授权后检查真实远程执行层。模型成本 $0，没有搜索或付费调用。**真实 SSH workspace 和远程 mTLS capabilities 成功；远程 Metal quick 求值没有成功。** 不把 Docker/fake backend、模拟 NPU 或 capabilities 握手写成设备正确性证据。

## 1. 真实 SSH workspace：通过

目标是本机 VMware 中的 Ubuntu 22.04.3 LTS / aarch64 VM，通过 `kuafu@172.16.145.130` 的真实 SSH 连接。`eval@` 仍只有 `publickey denied`，未使用。

执行真实 `SshWorkspaceTransport`：

- 暂存 `TASK.md` 与 `kernel.metal`，manifest digest `de30548187fa2e98cac0c8b90df8292edc4565ffc9868a3a02f6716ff615773c`；
- receipt id `2e76813b9d5d6261243cdb78f5e5554ebb44c41d27a595a97d03036f0f5873ec`，读取后逐字段一致；
- 两个文件均按原字节回收，缺失文件不伪造；
- agent 式地把 `kernel.metal` 换成指向 `/etc/passwd` 的 symlink 后，`collect` 以 typed `WorkspaceError(kind=path)` 拒绝，没有返回区外字节；
- 只删除本次会话目录，清理后 workdir 不存在。

这条证据只覆盖 transport/receipt/read/cleanup，不覆盖真实 hmz 围栏、隔离、逃逸或 agent 调用；`agent_called=false`、`metal_called=false`、`isolation_verified=false`。

## 2. 生产部署预检：发现两个独立边界

### v2 task-pack 不能由当前 production provider 登记

先尝试现行 `relu/fp16/small`，`python -m tcenv.remote.grpc_service --check` 在指纹重算时拒绝：provider 仍调用 v1 `tasks.get_task`，而该入口明确拒绝 v2 standard entry。故 `production_v2_deployable=false`，不能把现行 v2 任务描述成已可远程生产评测。需要单独阶段迁移 production task-pack/worker 身份与执行路径，不能靠改六 RPC 或 specimen 冒充。

### 服务启动后约一秒自行退出

legacy staging 首次启动时，`grpc_service.main` 记录 listening 后约一秒无错误退出。根因是主循环把 `wait_for_termination(0)` 的立即返回当成 ingress 已终止。已修复为 signal event 拥有 shutdown，`finally` 统一关闭 ingress/engine，并在 `tests/shared/test_remote_admin.py` 增加“收到 SIGTERM 前必须存活”的回归。修复后服务能持续监听，并在测试结束时被 SIGTERM 正常停止。

## 3. legacy mTLS staging：capabilities 成功，Metal Submit 失败

为继续验证网络协议，显式使用 v1 `sem_attention_odd_scale/fp16/small` specimen 与 process-worker 开发路径；这不是生产 v2 或容器部署证据。

- site/device：`staging-m4` / `apple-m4-max`；worker_build=`d0b8c4e+process-staging`；
- pack digest `3f4e8f9adbcf053b9764a9da1a81512991ad5a537be7173b7e0194e2bd0d15ca`；
- task fingerprint `ac74773255df6a7afe4ef1af188f3315f6a4312e82ebda28756c1b152abbe201`；
- 候选 `tests/support/golden/metal/backend_phase0_kernel.metal`，SHA-256 `dd9c31df8c49ba6093b130fe7aac638710af7df266c66d9cd1e58d0a902e4005`；
- client 在 VM 上运行，remote client/contracts/conformance 四个文件与 Mac 端逐字节 SHA 相同；
- mTLS endpoint `172.16.145.1:7443`，SAN `evaluator.local`，短期 grant 和 HMAC keyring 都只用于本场。

VM 发出的真实 GetCapabilities 返回 `ok=true`：

- evaluator epoch `6366dbd6f09aa892d26c48aead8a06236c1d05c92455e84abdb62072532bb955`；
- API `tcenv.remote.v1.evaluator-api.1`，evidence `tcenv.remote.v1.evidence.1`；
- request digest `sha256-rfc8785-v1`，signature `hmac-sha256-v1`，key id `staging-m4/key-1`。

随后两个独立 request id 的 Metal quick Submit 均为 `EvaluationUnavailable(code=worker_failed)`：macOS gRPC 初始化后的 process-worker fork 报 `FD from fork parent still in poll list`。设置 `GRPC_ENABLE_FORK_SUPPORT=1` 后仍未消除，故没有继续重试。**没有有效 Metal quick 报告、设备 correctness、resume/cancel 完成证据或发布数字。**

这说明 process-worker 开发路径不能在当前 macOS+gRPC 组合下承担 staging Metal 求值。容器是另一执行面，但 Linux 容器不能据此声称 Apple Metal 可用；需要 spawn-safe worker supervisor 或真实支持的部署路径，不能静默 fallback。

## 4. 清理、回归与边界

- 服务已 SIGTERM 停止，端口不再监听，没有遗留 worker；
- VM 的 client 凭据目录和 SSH session 目录均已删除；
- 本场本地 CA/client key/HMAC/grant 和专用 state 目录均已删除；
- 相关非 GPU 回归 **97 passed**，`git diff --check` 通过；
- `publication_allowed=false`，NPU 仍 simulation-only。

下一步若要真正关闭远程 Metal 验证，必须先解决现行 v2 task-pack/worker 支持及 macOS worker 创建模型，再做一次 Submit→fresh result→replay/cancel/cleanup；当前 capabilities 成功不计入这些未完成项。
