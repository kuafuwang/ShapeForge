# 执行运行时与设备扩展计划

状态：待评审，未实现。日期：2026-10-24。

本文只规划执行隔离和设备扩展，不批准实现 NPU、CUDA、ROCm 或第二个 target，不批准付费搜索，也不批准运行 GPU。

相关文档：

- [backend 协议与 GPU 实现计划](backend协议与GPU实现-plan.md)：设备 backend、target、评测协议和 Metal 迁移。
- [hmz](hmz.md)：当前 `hmz + pi` 写核路径。
- [安全边界](../security.md)：不可信模型反馈、P2 patch 和人工 apply 的边界。

## 0. 一句话

把“不可信内循环在哪里执行”和“可信评测在哪里执行”拆开，再与“使用哪种设备 backend”分开：

```text
AgentRuntime（local / vm / remote）
    × EvaluationExecutor（host / remote）
    × Backend（metal / 未来的 npu）
```

当前只保留 `local + host + metal` 的现有行为。VM 是可选运行时，不是以后唯一执行方式；`local` 也不等于 Apple 笔记本，只表示与控制面同一台受信主机。

## 1. 背景与现状

当前搜索写内核：

```text
taic/orchestrator.py
  -> taic/kernel_agent.py
  -> 本机 popen_group(hmz exec -f taic/flows/kernel_once)
  -> 本机 hmz + pi
  -> 交卷文件
  -> host 上的 tcenv.evaluate()
```

现状约束：

- `kernel_once` 使用 `LocalEnv`；`hmz` 虽已有 `ssh`、`docker`、`apple-container` 等环境后端，本仓库尚未接入。
- 当前 active target 只有 `apple-m4-max`，backend 是 `metal`。
- 搜索、反馈、trust、预算和确认在 `taic`；真正评测和发布计时在 host 的 `tcenv`。
- P2 的 `_test_patch_in_worktree` 只是隔离 git worktree，不是安全沙箱；补丁代码可能在 host 的测试进程里执行。
- 当前 Metal 的 `TASK.md`、prompt、trust 和 GPU 行为不能被这次重构悄悄改变。

## 2. 目标

1. 同一套搜索同时支持 `local` 和 `vm` 两种 agent 执行方式。
2. 可信评测可以继续在本机，也可以以后放到远程设备主机；它不随不可信 agent 一起搬进 VM。
3. Metal 和未来 NPU 通过 backend/target 扩展，不把设备类型写进 AgentRuntime。
4. P2 的补丁测试可以选择本机快速路径或隔离 VM 路径。
5. 保留现有产物、候选报告、预算、确认门和 resume 语义；runtime 选择必须进入运行记录。
6. 让 local 成为兼容默认路径，VM 成为显式选择的安全路径；VM 不可用时不得静默退回 local。

## 3. 非目标

- 本计划不实现 NPU，不新增 NPU target，不写 NPU runtime。
- 不把 GPU/NPU 发布计时移到不可信 VM 内。
- 不改变 Metal 的编译、正确性、计时、GPU 锁和确认门。
- 不把整个仓库、`~/.pi`、API key、trust 或 control 文件挂进不可信 VM。
- 不用 VM 替代 P2 的路径黑名单、人工 diff review 或 `--confirm`。
- 不把 local 和 VM 的启动耗时当成 kernel 加速比，也不跨 backend 合并结果。
- 不让模型自己决定搜索种子或发布结果。

## 4. 三个独立维度

| 维度 | 回答的问题 | 当前/未来取值 |
| --- | --- | --- |
| `AgentRuntime` | 不可信模型和工具代码在哪里执行？ | `local`、`vm`，以后可加 `remote` |
| `EvaluationExecutor` | 可信评测在哪里执行？ | `host`，以后可加 `remote` |
| `Backend` | 候选怎样编译、启动、检查和计时？ | 当前 `metal`，未来按单独计划增加 NPU |
| `Target` | 具体设备档案和限制是什么？ | 当前 `apple-m4-max`，未来另立 target |

### 4.1 典型组合

| 组合 | 用途 |
| --- | --- |
| `local + host + metal` | 当前 Apple 开发路径，保持原样 |
| `vm + host + metal` | Apple 上隔离写核，host 仍做 Metal 评测 |
| `local + host + npu` | 未来在 NPU Linux 服务器上本机开发 |
| `vm + remote + npu` | 本机控制面，VM 写核，远程 NPU 主机评测 |

`local` 只表示 agent 与控制面同机，不限定 Apple；`vm` 只表示隔离执行环境，不限定 Metal 或 NPU。

## 5. 目标架构

```text
┌────────────────────────── 控制面 ──────────────────────────┐
│ taic：轮次、委员会、预算、反馈、trust、确认、发布           │
│ run_config / transcript / ledger / candidates              │
└───────────────┬───────────────────────────┬────────────────┘
                │                           │
      ┌─────────▼─────────┐       ┌─────────▼─────────┐
      │ AgentRuntime      │       │ EvaluationExecutor│
      │ local / vm        │       │ host / remote     │
      └─────────┬─────────┘       └─────────┬─────────┘
                │                           │
        不可信内循环                   可信评测与设备锁
                │                           │
        ┌───────▼───────┐           ┌───────▼────────┐
        │ workspace     │           │ Backend        │
        │ hmz + pi      │           │ metal / npu    │
        │ kernel + logs │           │ target device  │
        └───────────────┘           └────────────────┘
```

原则：

- 不可信 agent 永远不直接决定种子、预算、trust 或发布。
- 可信评测可以在 host 或远程设备主机，但不与不可信 agent 共用同一写入权限。
- 候选 kernel 仍是不可信输入；host 的静态检查、正确性、超时和确认门继续存在。

## 6. 接口草案

以下是计划接口，不是当前已存在的 API。

### 6.1 AgentRuntime

```python
class AgentRuntime(Protocol):
    def prepare(self, brief) -> RuntimeWorkspace: ...
    def run(self, workspace, brief) -> AgentResult: ...
    def collect(self, workspace) -> AgentArtifacts: ...
    def stop(self, workspace, reason: str) -> None: ...
```

`AgentResult` 至少包含：状态、错误类别、stdout/stderr 摘要、token/费用、attempt 信息和 runtime 诊断。

`AgentArtifacts` 至少包含：交卷源码、`NOTES.md`、`FEEDBACK.md`、session/epic 摘要和文件哈希。 orchestrator 只消费统一结果，不读取 VM 特有路径。

### 6.2 LocalRuntime

第一版只包装现有行为：

- 使用当前 `agent_work` 和 `popen_group`；
- 保持当前 `hmz exec` 参数、工作目录、超时和交卷文件语义；
- 保持 local 的 `TASK.md` 字节不变；
- 自测继续使用 host 上的 `tcenv eval --quick` / `--timed`。

### 6.3 VmRuntime

第一版只要求“能写核并交回工件”，不要求 VM 内有 GPU：

1. host 创建本场 staging 目录；
2. 只复制 `TASK.md`、seed、冻结的 guidance、skill 和交卷目录；
3. 在 VM 中运行 `hmz + pi`；
4. 取回交卷文件和日志；
5. host 重新评测；
6. 按策略销毁 VM 或恢复快照。

可复用 `hmz` 的 `ssh`/`docker`/`apple-container` 环境后端，或先用一个外部 Linux VM + SSH 做最小实现。当前 `kernel_once` 的 `LocalEnv` 需要抽象，不是简单改一个命令行开关。

## 7. 自测接口

当前 `TASK.md` 里的 `uv run --directory ... tcenv eval` 带有 host 路径，不能直接照搬到 VM。

增加设备无关的自测接口：

```python
class SelfCheck(Protocol):
    def quick(self, request) -> EvalReport: ...
    def timed(self, request) -> EvalReport: ...
```

实现：

- `LocalSelfCheck`：保留当前 host 直接调用；
- `RpcSelfCheck`：VM 内的窄客户端调用 host 或远程评测机；
- 可选的 `VmSelfCheck`：仅用于 macOS guest 等明确支持的环境，不能替代 host 发布评测。

RPC 请求只接受结构化字段：run/session、target、task、dtype、size、源码、模式和预算上下文。不得接受 shell 命令、任意路径或任意环境变量。

host 侧必须重新检查 target、源码大小、预算、GPU/设备锁、超时和报告 schema。VM 自测结果只用于 agent 反馈，不能直接进入发布门。

## 8. P2 补丁测试运行时

当前 `_test_patch_in_worktree` 会在 host 执行补丁后的测试，属于“隔离 worktree”，不是安全沙箱。

计划抽象：

```text
PatchTestRunner
  ├── LocalPatchTestRunner
  └── VmPatchTestRunner
```

### LocalPatchTestRunner

保留当前快速开发路径，适用于人已经审查过 patch 的情况。

### VmPatchTestRunner

- 将 patch 应用到隔离 worktree；
- 在无 host 凭据、无网络、无 Docker socket、只读主仓库的 VM/容器中运行离线测试；
- 不把宿主 `HOME`、API key、`~/.pi` 或主仓库写权限带进测试环境；
- 先让人查看 diff，再运行不可信 patch 的测试；
- 测试输出只作为验收证据，不能代替人工 diff review。

若用户明确选择 `vm` 而运行时不可用，必须报错，不得自动改跑 local。

## 9. 安全边界

### 9.1 VM 输入和输出

- 只复制本场需要的文件，不挂载整个 host 仓库；
- 输入可只读，输出仅限本场交卷目录；
- artifact receiver 拒绝绝对路径、`..`、符号链接、设备文件和超大文件；
- 交卷源码、日志和反馈都按不可信数据处理；
- VM 内的 `TASK.md` 不能通过文字获得 host 的额外权限。

### 9.2 凭据和网络

- 真实 API key 不放进 VM；
- 需要时由 host provider proxy 注入 Authorization；
- 代理只允许指定 provider、路径、速率和 streaming 协议；
- 默认禁止任意 egress，host eval RPC 单独放行；
- 不把 host 的 `~/.pi`、auth、ssh agent、Docker socket 或桌面会话暴露给 VM。

### 9.3 生命周期

- host 保留 wall-clock、预算和 kill 权限；
- VM 崩溃映射为 `agent_failed`，不能变成成功；
- 优先按 session 或 run 恢复快照，避免恶意状态跨 run 残留；
- hypervisor、镜像和共享文件系统仍可能有漏洞，需保持更新并减少设备暴露。

VM 是防御层，不是对候选 kernel 正确性的证明，也不替代 P2 的路径黑名单和人工 apply。

## 10. 配置、resume 与证据

拟增加独立配置（字段名为草案）：

```toml
[search]
agent_runtime = "local"       # local | vm

[runtime.vm]
driver = "ssh"                # 首个实现只选一种
endpoint = ""
image_digest = ""
network = "provider-proxy"
reset = "per_session"

[evaluation]
executor = "host"             # host | remote（后续）
```

`run_config.json` 记录：

- `agent_runtime`、driver、image digest、网络策略、reset 策略；
- `evaluation_executor`、目标设备和 backend/backend_version；
- 每场 session 的 VM 启动、复制、agent、回收耗时；
- 不记录密码、私钥、完整 Authorization 或 host 敏感路径。

规则：

- 默认 `local`，不改变现有运行；
- `resume` 默认沿用原 runtime 和 evaluator；
- 切换 runtime 不得静默发生，需显式新运行或显式授权；
- VM 临时状态不是 resume 的唯一来源，host 的候选、转录、报告和账本必须足以重建；
- local 的 `TASK.md`/prompt 字节保持不变；VM 若需要不同自测命令，视为新的 prompt surface，记录独立哈希并重新授权；
- runtime 不改变发布加速比的可比性，启动开销单独展示。

## 11. 实施阶段

### Phase 0：接口与基线（不改行为）

- 定义 `AgentRuntime`、`SelfCheck`、`PatchTestRunner` 和统一 artifact 结构；
- `LocalRuntime` 包装当前 `popen_group` 路径；
- `LocalPatchTestRunner` 包装当前 `_test_patch_in_worktree`；
- 默认配置仍是 local；
- 记录现有 local `TASK.md`、prompt hash、候选格式和 P0 基线。

验收：现有离线测试通过；local 的 `TASK.md` 和 Metal 行为不变；不启动 VM、不调用模型、不跑 GPU。

### Phase 1：VM 只写核，不做自测

- 选定一种驱动，优先 SSH 到一台 Linux VM，避免先绑定 Apple 专用 API；
- 完成 staging、输入复制、交卷回收、日志回收、超时和清理；
- VM 不可用时明确失败，不 silent fallback；
- host 继续做全部 `tcenv` 评测；
- 用 fake agent 写固定 kernel 做端到端测试，不调用真实模型。

验收：local 与 VM 产生同样的候选结构；host 能独立评测 VM 交回的 kernel；VM 无 host repo/home 写权限。

### Phase 2：受控自测与凭据代理

- 实现 `SelfCheck` 和窄 RPC；
- 增加 host provider proxy、网络 allowlist 和速率限制；
- 加入 VM 快照/销毁策略、启动耗时和 runtime 证据；
- VM 的 `TASK.md` 使用独立 prompt surface 和 trust 记录。

验收：RPC 不接受 shell/任意路径；无凭据泄露；host 预算、设备锁和超时仍生效。

### Phase 3：P2 VM 测试

- 实现 `VmPatchTestRunner`；
- 先人工看 diff，再在隔离环境运行测试；
- 保留 `apply-code --confirm` 和现有黑名单；
- 记录测试环境镜像、网络、挂载和结果。

验收：测试环境无法访问 host home、主仓库写路径、provider 网络和 Docker socket；local 快速路径仍可用。

### Phase 4：远程可信评测（为 NPU 主机做准备）

- 抽象 `EvaluationExecutor`，先让 host 实现复用现有 `tcenv`；
- 增加远程设备主机的请求/报告、认证、超时、设备锁和审计；
- 不把远程评测 RPC 暴露给不可信 agent 的任意命令能力。

验收：同一候选在 host/remote 的报告字段可区分，runtime、backend、target 和协议版本完整记录；不跨 backend 混算。

### Phase 5：未来 NPU（另行批准）

- kernel 级 NPU：按 backend 协议新增实现、target、skills、测试和基线；
- graph/IR 级 NPU：另立 graph/operator 任务、契约、IR、正确性和计时计划，不能硬套 Metal 的 dispatch/线程组模型；
- 每个 NPU 结果都带 backend、backend_version、target 和协议版本。

本计划不执行这一阶段。

## 12. 测试策略

### 不需要 VM/GPU 的测试

- local/vm fake runtime 的统一 artifact、状态和错误映射；
- staging 的路径穿越、符号链接、超大文件和残留文件处理；
- VM 不可用时不会 silent fallback；
- runtime 配置写入、resume 绑定和 prompt hash 记录；
- RPC schema、超时、预算和非法请求拒绝；
- LocalRuntime 与现有 `kernel_agent` 行为对比；
- P2 runner 的 host 环境不泄漏测试（用 fake runner/静态检查）。

### 显式授权的集成测试

- 一台临时 Linux VM 的 fake agent 写核；
- SSH/挂载/网络断开、VM 崩溃、超时和清理；
- 隔离 P2 测试无法访问 host 凭据和主仓库写路径；
- 只有用户明确批准后，才做真实模型或 GPU 对照。

## 13. 回滚

- 默认 `agent_runtime=local`，删除或禁用 VM 配置即可回到当前路径；
- VM 代码只通过统一接口接入，失败不得改变 local；
- 任何 local prompt、报告 schema 或 Metal 行为变化都视为单独变更，不能用本计划的名义合入；
- P2 VM runner 失败时保留 local runner，但不可信 patch 不得因此自动改跑 local。

## 14. 待评审问题

1. 首个 VM 驱动选 SSH + Linux VM，还是先做 Lima/Apple Container 的本机便利层？
2. VM 的快照粒度是每场 session、每次 run，还是按任务复用后强制清理？
3. 自测 RPC 采用 SSH port forwarding、vsock 还是本地 TCP？如何做认证和审计？
4. P2 测试是否必须先完成人工 diff review，再允许启动任何补丁代码？
5. NPU 未来是 kernel 级还是 graph/IR 级？这决定是否复用当前 `Task`/`Contract`。
6. 远程评测的凭据、设备锁和结果签名由哪一层负责？

在这些问题确定前，只实施 Phase 0 的接口和 local 包装，不提前引入 VM、NPU 或新的 GPU 行为。
