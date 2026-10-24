# 执行运行时与设备扩展计划

状态：未实现；已吸收多轮评审，详见 §15。日期：2026-10-24。默认保持现有 local 行为；远程评测在声称可用于生产前必须通过 Phase 2.5 的真实传输模拟验收，不做真实 NPU。

本文只规划执行隔离和设备扩展，不批准实现 NPU、CUDA、ROCm 或第二个 target，不批准付费搜索，也不批准运行 GPU。

相关文档：

- [backend 协议与 GPU 实现计划](backend协议与GPU实现-plan.md)：设备 backend、target、评测协议和 Metal 迁移。
- [hmz](hmz.md)：当前 `hmz + pi` 写核路径。
- [远程评测模拟验收计划](远程评测模拟验收-plan.md)：Phase 2.5 的环境、真实传输、故障矩阵、授权与验收细则。
- [安全边界](../security.md)：不可信模型反馈、P2 patch 和人工 apply 的边界。

## 0. 一句话

系统由两层循环组成：**外循环**负责搜索与判卷，**内循环**负责在一场会话里写核。`taic` 与 `hmz` 同属外循环：`taic` 决定轮次、预算、换种子、确认与发布，`hmz` 按外循环的安排启动一场编程代理会话、套权限围栏并回传费用与状态。编程代理是内循环，它在自己的会话里多轮调用模型、读文件、改文件、运行命令和自测，但不决定搜索的下一轮。

在这个定位下，**默认只分三件事**：内循环在哪里写核、评测请求提交到哪个受信执行端、设备差异由哪个 Backend × Target 处理。默认形态如下（设备在本机时，把「远程受信评测服务」换成本机 `LocalEvaluator`，其余不变）：

```text
┌─ control_host ─────────────────────────────────────────────┐
│  外循环  taic（唯一搜索控制面）                             │
│          hmz（单场会话启动/监督适配器）                     │
│  内循环  pi（workspace=local；provider / model / effort）   │
│          EvaluationExecutor 客户端（formal / selfcheck）    │
└──────────────────────────┬─────────────────────────────────┘
                           │  候选源码、TASK、反馈
                           │  结构化自测请求 / 正式评测请求
                           ▼
┌─ 远程受信评测服务端（设备侧部署）───────────────────────────┐
│  认证与幂等、队列、设备租约（formal 优先 + 跨 run 公平）   │
│  报告回传：绑定源码摘要、设备身份、评测构建摘要           │
└──────────────────────────┬─────────────────────────────────┘
                           ▼
┌─ 设备侧 ───────────────────────────────────────────────────┐
│  GPU / NPU worker + 设备 SDK + 设备队列                    │
│  Backend × Target（当前本机为 metal × apple-m4-max）       │
└────────────────────────────────────────────────────────────┘
```

也就是说，`taic + hmz + pi` 留在 control_host，pi 写的候选源码、结构化自测请求与正式评测请求发给远程受信评测服务，服务在其设备侧用 GPU/NPU worker 执行并回传报告。只要这条通道能做到设备侧闭合计时、独占租约、身份与配额分离、报告绑定源码/设备/构建摘要，pi 就**不需要**去 GPU/NPU 主机，不需要把 provider 登录态带到设备主机，也不需要本计划去实现 VM、设备直通或远程 agent 生命周期。

`hmz` 不需要运行在 Docker 或 VM 内，也不是 hypervisor；它与 `taic` 同属外循环，但只有 `taic` 决定轮次、总预算与停止，`hmz` 只负责把一场会话启动、围栏、超时并回传费用与状态，不做第二套搜索控制。

把代理或 workspace 搬到设备站点（colocated）只是一种**可选优化**，触发条件写死为三选一：实测远程自测延迟已影响搜索、设备 SDK/探针必须在 agent 环境内直接可用、或隔离要求代理进程离开 control_host。没有证据时不做。当前只保留 `pi + harness=local + workspace=local + evaluation=local trusted process + metal` 的现有行为。

## 1. 背景与现状

当前搜索写内核。外循环每场只启动一次内循环；内循环在会话内部可以有多次模型与工具往返：

```text
外循环 taic/orchestrator.py（control_host）
  -> taic/kernel_agent.py 组装本场 SessionBrief（现有唯一会话内容模型，不另造 SessionSpec）
  -> 本机 popen_group(hmz exec -f taic/flows/kernel_once)
  -> hmz 启动并监督内循环
       内循环 pi：读 TASK.md -> 写候选 -> 自测 -> 改候选 -> 写交卷文件
  -> hmz 回传状态、费用与会话摘要
  -> 外循环读取交卷文件
  -> host 上的 tcenv.evaluate() 判卷
  -> 外循环决定换种子、下一场由哪个委员会接手，或进入确认
```

`hmz` 不排整次搜索，也不决定下一场。它只把外循环交给它的一场会话运行完并如实回传；委员会轮换、预算停机、候选选择和确认仍在 `taic`。

现状约束：

- `kernel_once` 使用 `LocalEnv`；`hmz` 已有 `ssh`、`docker`、`apple-container`、`slurm` 等 workspace 环境后端，本仓库尚未接入。接入点是 flow 的 Env 声明与 `hmz exec -e` 映射，不是把 `hmz` 本身搬进容器或 VM。
- `hmz` 可操作的编程代理不只 pi；本仓库的 `agent_spec()` 与 preflight 当前只生成并校验 `pi/<provider>/<model>`，所以通用 agent 只是 hmz 能力，不是当前 `taic` 能力。
- 当前 active target 只有 `apple-m4-max`，backend 是 `metal`。
- 搜索、反馈、trust、预算和确认在 `taic`；真正评测和发布计时在 host 的 `tcenv`。
- P2 的 `_test_patch_in_worktree` 只是隔离 git worktree，不是安全沙箱；补丁代码可能在 host 的测试进程里执行。
- 当前 Metal 的 `TASK.md`、prompt、trust 和 GPU 行为不能被这次重构悄悄改变。

## 2. 目标

1. 外循环保持唯一控制权：`taic` 决定每一场何时开始、由哪个委员会写、预算与超时是多少、何时停；`hmz` 是单场会话适配器，只执行外循环交给它的一场会话，不复制预算与停止逻辑。
2. 内循环默认留在 control_host：pi 在本地写核，自测与正式评测都经 `EvaluationExecutor` 提交。只有 §0 的三个触发条件之一成立，才把代理/workspace 搬到设备站点；搬家是优化，不是接入远程设备的前置条件。
3. 可信评测必须具备通往目标设备的受信设备操作能力，同时与内循环分开信任：优先接入设备方已部署的受信评测服务，没有时才在设备站点部署同协议的服务端；评测控制端可与 pi 同地或远离 pi，但设备侧执行端不能共用 pi 可写的进程、账号、镜像或文件系统权限。
4. Metal 和未来 NPU 通过 backend/target 扩展，不把设备类型写进 session runner。
5. 先把 P2 拆成不执行代码的静态验收和经人工审阅后才启动的动态测试，再支持隔离动态测试。
6. 保留现有产物、候选报告、预算、确认门和 resume 语义；agent、harness 位置、workspace 环境和评测位置必须进入运行记录。
7. 让当前 local 组合成为兼容默认路径，远端或隔离组合必须显式选择；不可信 P2 不得因为隔离环境不可用而自动改跑 local。
8. 每一场内循环的输入、输出与证据都有清单：可写工作区与只读上下文分开，自测与工具证据以受信设备操作端记录为准，超时或断连后仍能回收已落盘的交卷 checkpoint。
9. 对外只暴露少数命名拓扑 Profile；底层放置维度保留给运行证据与未来扩展，任意笛卡尔积组合在配置加载时拒绝。

## 3. 非目标

- 本计划不实现 NPU，不新增 NPU target，不写 NPU runtime。
- 不把 GPU/NPU 发布计时移到不可信内循环环境内；评测控制端可以与 pi 分开，但设备侧执行必须在独立受信域中完成。
- 不改变 Metal 的编译、正确性、计时、GPU 锁和确认门。
- 不把整个仓库、控制机的 `~/.pi`、API key、trust 或 control 文件挂进不可信 workspace。
- 不用容器或 VM 替代 P2 的路径黑名单、人工 diff review 或 `--confirm`。
- 不把环境启动耗时当成 kernel 加速比，也不跨 backend 合并结果。
- 不让模型自己决定搜索种子或发布结果。
- 不把 local 当成安全边界；它只适合受信开发环境。
- 不把 workspace 隔离当成候选 kernel 的安全证明；host/远程评测仍会执行不可信 kernel。
- 不在 MVP 中发明通用的 agent 框架，也不承诺 pi 以外的编程代理；先抽 `HmzKernelSessionRunner`，`agent_kind` 只记录当前支持的 `pi`。
- 不在本计划中实现 hypervisor 或 VM 生命周期平台；默认路径根本不需要 VM，若以后启用可选的 colocated 扩展，虚拟机也必须预先存在并通过 SSH 或所选环境后端可达。

## 4. 先分清外循环与内循环，再分位置

外循环与内循环不是两个进程位置，而是两种职责：

| 层 | 谁在里面 | 一次迭代做什么 | 谁能结束它 |
| --- | --- | --- | --- |
| 外循环 | `taic` + `hmz` | `taic` 备场、定预算与委员会；`hmz` 启动一场会话并监督；结束后由 `taic` 判卷、换种子或停机 | `taic` |
| 内循环 | 编程代理（如 pi） | 在一场会话内多轮读、写、运行命令、自测和改候选，最后写交卷文件 | 本场预算先到，或代理主动结束；`hmz` 负责强制收尾 |

在此之下，**默认路径只用前两行加最后一行**；中间两行是可选扩展，没有 §0 的触发证据时不实现、不配置：

| 维度 | 回答的问题 | 当前/未来取值 |
| --- | --- | --- |
| `AgentSpec` | 哪个编程代理、哪个模型写核？ | MVP 只有 `pi`；未来值必须逐个适配费用、诊断与技能交付 |
| `EvaluationExecutor` | 评测请求提交到哪个受信执行端？ | `LocalEvaluator`（本机 `tcenv.evaluate`）/ `RemoteEvaluator`（设备侧部署的受信评测服务端）；执行端用不透明 `site_id` 标识，不把 vm/docker/passthrough 写进搜索配置 |
| `Backend × Target` | 候选怎样在什么设备上编译、检查和计时？ | 当前 `metal × apple-m4-max`，未来按单独计划增加 |
| `AgentHarnessPlacement`（可选） | 编程代理进程在哪里运行？ | 默认 `local`（控制机）；`env`/`remote` 仅在 §0 触发条件成立时启用 |
| `WorkspaceEnv`（可选） | 文件读写与命令在哪里执行？ | 默认 `local`；`ssh`、`docker`、`apple-container`、`slurm` 仅随上一行按需启用 |

术语只固定三个：`control_host` 跑 `taic`、`hmz exec` 与默认的 pi；`evaluator` 是接受结构化评测请求的受信执行端（本机进程或远程服务），用 `site_id + worker_build + trust_domain` 描述；`device` 是 evaluator 背后能直接操作的 GPU/NPU。`hmz` 本身不是 workspace，也不必进入 Docker/VM。

`local` 只描述位置与控制机相同，不描述安全等级。虚拟化方式（passthrough、virtual-function、容器、guest）是 evaluator 的部署与能力探测结果，只进运行证据，不进普通搜索配置。

### 4.1 默认形态：内循环留控制机，评测去远程服务

| 情形 | pi / workspace | 评测怎样成立 |
| --- | --- | --- |
| 设备在控制机本地（当前 Metal） | control_host / local | `LocalEvaluator` 直接包装本机受信 `tcenv.evaluate` |
| 设备在远程（含未来 NPU）且设备侧有可接入的受信评测服务 | **仍在 control_host / local（默认）** | `RemoteEvaluator` 把候选与结构化请求发给远程服务；服务在设备侧独占调度、闭合计时并回传绑摘要的报告 |
| 远程自测延迟、工具链或隔离触发 §0 条件 | 按证据搬到设备站点（可选） | 评测仍走同一 `EvaluationExecutor`，只是自测改走站点内通道 |

要分清两个问题：**评测可信靠受信设备操作能力，不靠 pi 与设备的物理距离；pi 就近只解决自测延迟与工具可用性。** pi 负责写核与发自测请求；受信评测负责正式判卷。两者可以同机也可以分开，但不能共用 pi 可写的账号、镜像、进程或目录。若 pi 在设备环境里拥有 root，而设备侧评测执行端也跑在同一个 guest/账号下，这个评测不算受信评测。

因此跨站点写核不是「降级模式」，而是远程设备的**一等默认模式**。它的代价只有一条需要实测：每次自测多一次网络往返（提交与回传，不进计时窗口）。只有实测证明这个往返已影响搜索，才升级为 colocated。

### 4.2 对外只暴露命名拓扑 Profile

底层维度用于描述证据和未来扩展，不直接暴露成任意可组合的配置。对外只支持下面三个 Profile；未列出的组合在配置加载阶段拒绝，不在运行时临时拼装。

| Profile | 固定放置 | 何时用 |
| --- | --- | --- |
| `local-integrated` | harness、workspace、设备与评测都在 control_host | 当前 Metal 开发机；本地 GPU/NPU 节点 |
| `split-control`（远程设备默认） | pi 与 workspace 在 control_host，评测经 `RemoteEvaluator` 去远程受信服务 | 远程 GPU/NPU 且设备侧有可接入的受信评测服务时 |
| `device-site-colocated`（可选优化） | harness 与 workspace 去到设备站点，评测设备侧执行端在同站点的独立受信域 | 仅 §0 三个触发条件之一成立时 |

Profile 只固定放置关系，不固定操作系统、容器运行时或设备类型。Mac、Linux GPU 节点、NPU 服务器和虚拟化加速器节点都按同一组 Profile 选择；具体 backend 与 target 仍由设备协议决定。

## 5. 目标架构

默认形态（远程设备走 `split-control`，本地设备把远程服务换成本机 `LocalEvaluator`，其余不变）：

```text
┌──────────────── control_host ────────────────┐
│ 外循环 taic：轮次、委员会、预算、反馈、确认   │
│ 外循环 hmz：启动一场 pi、围栏、epic/费用流    │
│ 内循环 pi：读 TASK → 写候选 → 自测 → 改候选   │
│ EvaluationExecutor 客户端（formal / selfcheck）│
└──────┬──────────────────────────┬────────────┘
       │ 交卷候选（正式评测）       │ 候选源码+结构化参数（自测，不同身份/配额）
       ▼                          ▼
┌──────────────── 远程受信评测服务端（设备侧部署）──────────────┐
│ 认证、幂等、队列与设备租约（formal 优先 + 跨 run 公平）      │
│ 设备侧 worker：静态检查 → 正确性 → 设备侧闭合计时 → 报告     │
│ Backend × Target → GPU / NPU                                │
└─────────────────────────────────────────────────────────────┘
```

原则：

- 外循环只有一个控制者：`taic`。`hmz` 是外循环的单场会话适配器，不是第二套搜索循环；它不轮换委员会、不换种子、不决定确认。
- 内循环只对本场负责：它可以多轮试错和自测，但不能决定本场成绩是否算数，也不能开启下一场。
- pi 与 workspace 默认留在 control_host。设备在远程时，自测与正式评测都经 `EvaluationExecutor` 发给远程受信评测服务；网络往返只在提交与回传，不进计时窗口。
- 可信评测的设备侧执行端必须能直接操作目标设备，并与内循环分开信任。评测控制端可以远离设备，前提是它握有受信设备操作能力；同机、同虚拟化宿主或同远程服务器只表示同地，不表示同信任。
- 只有 §0 的触发条件成立时，才把编程代理进程或 workspace 搬离 control_host；凭据跟随代理进程与 provider 调用路径，不跟随 workspace 名称自动移动。
- 候选 kernel 仍是不可信输入；受信评测域的静态检查、正确性、超时和确认门继续存在。workspace 隔离不证明 kernel 或驱动调用安全。

## 6. 接口草案

以下是计划接口，不是当前已存在的 API。

### 6.1 HmzKernelSessionRunner（MVP）

MVP 不做通用 `AgentRuntime`。只抽外循环调用内循环的这一层：输入是现有的 `SessionBrief`，输出仍是现有 `AgentResult` 语义。`hmz` 的启动、围栏、主回合与收尾回合、超时、费用流解析都留在这一层；内循环怎样多轮试错不归 runner 管。

**不另造 `SessionSpec`。** `SessionBrief` 已经是唯一的会话内容模型（任务、契约、seed、历史、guidance、target、语言）；再造一份字段相近的规格必然双写漂移。新增的只有执行位置与路径视图：

```python
@dataclass(frozen=True)
class HmzSessionPlacement:
    agent_kind: str                 # MVP 只有 "pi"
    harness_placement: str          # 默认 local；env | remote 仅可选扩展
    workspace_env: str              # 默认 local；ssh | docker | apple-container | slurm 仅可选扩展
    endpoint: str = ""              # 不含凭据
    image_digest: str = ""

@dataclass(frozen=True)
class SessionPaths:
    host_work: Path                 # control_host 上的 canonical agent_work/<tag>，也是工件回收位置
    agent_work: str                 # agent 实际看到的工作目录（默认与 host_work 相同）

@dataclass(frozen=True)
class SelfCheckBinding:
    mode: str                       # local_cli | host_bridge | direct_client
    endpoint_id: str                # 不含凭据；由 ExecutionConfig 解析
    token_ref: str = ""             # 秘密的引用，不是 token 本身；local_cli 为空

@dataclass(frozen=True)
class SessionExecution:
    placement: HmzSessionPlacement
    paths: SessionPaths
    prices: Prices                  # 保留当前 run_kernel_agent 的计费输入
    turn_output_tokens: int
    turn_minutes: int
    session_cost_cap_usd: float | None
    timeout_s: float | None
    hmz_binary: str | None
    selfcheck: SelfCheckBinding

class HmzKernelSessionRunner(Protocol):
    def execute(
        self,
        brief: SessionBrief,
        *,
        execution: SessionExecution,
        provider: str,
        model: str,
        effort: str | None,
    ) -> AgentResult: ...
```

`SessionExecution` 只装执行参数，不是另一份会话内容模型：它保留当前 `run_kernel_agent` 的
`prices`、主回合 token/分钟、单场费用上限、hmz 路径与墙钟兜底，并把自测绑定显式传给 runner。
`orchestrator` 现有的 `agent_runner` / `evaluate_fn` 测试注入先用兼容 adapter 包装，不要求一次性改写测试。
`SelfCheckBinding.token_ref` 只能指向 host 管理的秘密引用，不能把真实 token 写进 manifest、TASK 或日志。
`host_bridge` 表示 agent 先连 host 桥接再由 host 转发；`direct_client` 才表示 agent 直接连接远程评测服务。

MVP 的 `execute` 是同步调用：runner 内部按本场 deadline 监督并强制收尾。异步句柄与外部 cancel 不在 MVP 内，后续确有需要时再单独加 `start() -> SessionHandle`，不让一个同步接口假装可被外部取消。跨环境传输时从 `brief` 派生一份只读的序列化清单即可，那是传输信封，不是第二份可独立修改的会话模型；信封不携带 API key 或可执行命令。

输入清单分两层，不能把整个仓库或整个 run 目录默认塞进 workspace：

- **可写工作区（RW）**：本场 seed、交卷文件、`NOTES.md`、`FEEDBACK.md` 与本场新建的探针/候选文件。
- **只读上下文包（RO bundle）**：冻结的 `agent_guidance/`、本场需要的技能与参考、经摘要裁剪的前序 `NOTES.md`/反馈索引。仓库 `docs/` 只按清单提供索引与被点名子集，或提供受控的只读按需取回，不默认整仓复制。

`taic.agent_feedback check` 这类纯文本校验不要求代理在 workspace 里找到 control_host 仓库：它可以作为设备操作通道之外的本地校验 job，或由 runner 在工件回收后执行，并把结果回填本场记录。实现不能靠把整个 `taic` 仓库复制进 workspace 解决。

`AgentResult` 沿用现有语义：状态、错误类别、stdout/stderr 摘要、token/费用与费用是否完整、attempt 信息、自测计数、runtime 诊断和 `artifacts`。`artifacts` 是可选扩展字段，默认 `None`，不能让现有多数测试和 fake agent 的构造器被迫全部改写；local 的 `source`/`notes`/`feedback` 继续是兼容主字段，remote workspace 才填充新的工件结构。

`AgentArtifacts` 至少包含：交卷源码、`NOTES.md`、`FEEDBACK.md`、被反馈点名的证据文件、session/epic 摘要和文件哈希。

**remote runner 必须物化 host 侧 canonical workspace。** 即使本场在远端环境执行，runner 也要按 artifact manifest 把文件逐个回收到 `paths.host_work`（即现有 `agent_work/<tag>` 布局），再返回 `AgentResult`。现有 `_parsed_feedback`、`_record_feedback`、`_file_session_notes`、`prune_work`、`harness.record_proposals` 与 `.selfcheck` 汇总都按这个布局工作；只返回抽象 artifacts 会迫使它们一起重写。orchestrator 继续只消费统一结果，不读取某个 workspace 环境特有的远端路径。

超时、断连或 runner 强制收尾后，runner 必须在 reset 前做一次有界的 best-effort 工件回收。只有已经原子替换为正式交卷文件的内容才算 checkpoint；只剩 `.tmp` 的半成品不算交卷。回收失败要单独记为 artifact 不完整，不能把已落盘的新核静默丢弃，也不能把拉回失败伪装成代理没写核。

### 6.2 当前 local 组合的第一版包装

第一版只包装现有行为：

- 使用当前 `agent_work` 和 `popen_group` 在 control_host 上启动 `hmz exec`；
- 保持当前 `hmz exec` 参数、工作目录、超时和交卷文件语义；
- 保持 local 的 `TASK.md` 字节不变；
- 自测继续使用 host 上的 `tcenv eval --quick` / `--timed`。

这一版的验收是行为不变，不是获得隔离。

### 6.3 远端 workspace 的 runner（可选扩展，默认不做）

默认路径不需要这一节：pi 与 workspace 留在 control_host，设备在远程时只把评测发出去。只有 §0 的触发条件成立、决定搬 workspace 时，才启用下面这条路径。它不新造一个「VM runner」，仍是同一个 `HmzKernelSessionRunner` 按 placement 改变内循环的位置：

1. 外循环在 control_host 组装 `SessionBrief` 与输入清单；
2. `hmz exec` 仍由 control_host 发起，按 flow 的 Env 声明与 `-e` 映射选择 workspace 环境，并按 harness placement 决定编程代理进程在哪里运行；
3. 内循环在所选组合里写核；workspace 环境只承载它实际读写的文件与命令；
4. runner 通过工件清单取回交卷文件、笔记、反馈与被点名的证据文件，逐个校验哈希与大小，并物化到 host 侧 canonical `agent_work/<tag>`（§6.1）；
5. 外循环经 `EvaluationExecutor` 重新评测，不采信内循环自报成绩；
6. 按 placement 的 reset 策略清理本场环境状态；环境不可用时明确失败，不改跑 local。

搬家的优先顺序按触发原因决定，不按设备位置一刀切：

- 为隔离搬：先做 `harness=local + workspace=ssh/docker` 的 transport spike（§11 Phase 3），只证明文件/命令锚定与工件回收，不宣称进程隔离；真要隔离代理进程才做 `harness=env + workspace=同一隔离环境`。
- 为自测延迟或工具链搬：直接做 `harness=env + workspace=设备站点`，pi 去设备站点；自测改走站点内通道，正式评测仍经同一 `EvaluationExecutor`。

当前 `kernel_once` 的 `LocalEnv` 需要改成可配置的 hmz Env 声明，并由 `kernel_agent` 传 `-e` 映射；这不是给现有命令多加一个字符串开关就结束，还包括 TASK 命令、skill 提示面、工件回收与费用流的适配（§7、§9.4）。

### 6.4 EvaluationExecutor：Local/Remote 共用，设备操作必须可信

这是本计划的主接口之一；与之配套还需要 selfcheck bridge、EvidenceStore、帧协议和可部署服务端。可信评测的放置不由 pi 的位置决定，也不要求评测控制进程与设备同机。它分成两半：

- **评测客户端（本仓库实现）**：`EvaluationExecutor` 的 `formal` 与 `selfcheck` 两个方法。`LocalEvaluator` 包装现有 `tcenv.evaluate`；`RemoteEvaluator` 按下面的窄协议调用远程受信评测服务。正式评测与自测共用传输、认证、请求摘要、设备租约、超时与审计，但结果类型、模式白名单与配额分开；`tool_probe` 只借同一传输与租约，不并进评测结果类型。
- **设备侧执行端（优先接入设备方已部署的服务，没有时才在设备站点部署）**：在设备站点运行版本化的 `tcenv`/backend，直接操作目标 GPU/NPU。它必须不受 pi 写权限影响。设备方已有满足下面五条的服务时，本仓库只写客户端，不实现服务端、队列与 VM 生命周期。若由本项目部署，帧编解码、forced-command 入口、认证/幂等、队列/租约和 worker supervisor 是正式代码，规划放在 `tcenv/remote/`；`tests/support/npu_sim/` 只放 fake backend、fixture 和故障注入，Phase 2.5 测的是同一份可部署服务端，不是另写的测试替身。

对外请求只用结构化字段加源码，由执行端自己创建内部临时路径；**不复用**当前 `tcenv.worker` 那种带 `kernel_path / out / heartbeat / stage` 本地路径的内部请求。接口必须覆盖当前 `tcenv.evaluate()` 的语义，而不是只覆盖一个简化 demo：

```python
@dataclass(frozen=True)
class EvaluationRequest:
    kind: str                    # formal_eval | selfcheck；tool_probe 用自己的 typed request
    request_id: str              # 幂等键；同一 request_id 必须对应同一 request digest
    run_id: str
    session_id: str
    task: str
    dtype: str
    size: str
    target_id: str
    task_fingerprint: str        # 已包含 task.contract.describe()；服务端重算并比对
    candidate_sha256: str         # 与实际发送、实际编译的 source 字节一致，不做任何规范化
    source: str
    mode: str                    # quick | timed | full | validate（按身份限模式）
    seed: int | None
    reps: int
    queue_timeout_s: float
    candidate_timeout_s: float
    cases: tuple[str, ...] | None
    fast_bench: bool = False
    require_contract_launch: bool = False
    validate: bool = False
    measure_serial: bool = False
    cache_policy: str = "reuse"   # reuse | refresh | disabled；确认可强制 disabled

@dataclass(frozen=True)
class EvaluatorProvenance:
    schema_version: int
    epoch: str                     # 下列 site/device/build/task_pack/calibration/protocol 的摘要
    site_id: str
    device_id: str
    backend: str
    backend_version: str
    target_id: str
    worker_build: str
    task_pack_digest: str          # 服务端版本化任务包的摘要
    calibration_digest: str
    protocol_version: str
    request_digest: str            # hello 时为空；job/result 时必填
    measurement_kind: str          # measured | simulated
    publication_allowed: bool      # 仅作第二道校验；host allowlist 才是发布许可

@dataclass
class EvaluationResult:
    report: EvalReport
    provenance: EvaluatorProvenance
    evidence: dict | None = None   # 供 host EvidenceStore 校验后落盘的自测证据
    receipt: dict | None = None    # selfcheck 的权威回执；tool_probe 有自己的结果类型
    signature: str = ""            # 外层签名；不塞进 EvalReport 本体
```

`workdir`、`out`、`heartbeat`、`progress_path`、`self_check_dir` 是执行端内部派生路径，不进入 RPC。
`reps`、`cases`、最大 timeout 与配额受服务端策略限制；selfcheck 不能通过修改命令行放大它们。
`candidate_sha256` 对实际发送和实际编译的字节计算，不做换行转换、Unicode 规范化、trim 或二次改写。
`request_id` 就是幂等键；request digest 用规范 JSON（例如 RFC 8785）计算，候选字节摘要独立绑定。
服务端必须重算 `task_fingerprint`（当前已经包含 `task.contract.describe()`）并比对，不再另传一份冗余的
`contract_digest`。不一致时拒绝请求，记为 evaluator/protocol mismatch，不能误判成候选的 compile_error
或 incorrect。`cache_policy=disabled` 必须能表达确认阶段的 `eval_cache=False`。

**先握手，再固定 epoch，再提交 job。** 协议以 `hello` / `capabilities` 开始：客户端声明协议版本范围、一次性 `client_nonce`、session token、期望的 site_id、expected_worker_build 与 expected_task_pack_digest；服务端返回回声 nonce、自己的 `server_nonce`、不含 request_digest 的 `EvaluatorProvenance` 和能力清单（支持的 target/mode、最大源码、配额、task_pack_digest）。host 先验证已认证通道、nonce、session token 与服务身份，再校验 expected_worker_build，并把这次返回固定为 evaluator epoch；nonce 不符、旧 hello 重放、身份或 build 不匹配都拒绝连接，不能等第一份报告才取得基准。hello 的签名/通道身份必须能绑定到后续报告，不能只信第一帧里的自我声明。

任务包随服务端以只读、版本化方式部署，包含服务端重算所需的 `tasks/`、契约、容差、参考实现与基线。
`task_pack_digest` 由其规范清单计算并纳入 epoch。清单（`task_pack_manifest`）至少包含 schema_version、
backend/target、排序后的 POSIX 相对路径、每个文件的实际字节 sha256 与大小、参考/基线版本；digest 用
同一份规范 JSON 计算，不包含 mtime、宿主机绝对路径或临时文件。host 在 expected_task_pack_digest 为空时，
用同一 manifest 规则计算本地包；否则 hello/job 阶段明确报 task pack mismatch，不能分不清候选错误与任务版本漂移。
任务包的动态更新走独立的受信部署/管理流程，不放进 agent 可请求的 eval RPC；更新后形成新的 epoch，必须重新登记发布许可。

两者合起来必须提供 **TrustedDeviceOperation**，满足下面五条：

1. **独立受信执行。** 设备侧评测代码来自不可由 pi 修改的版本化安装或镜像；运行账号、进程、容器或 guest 与内循环分开。pi 不能写评测代码、任务参考实现、设备锁、报告目录或评测凭据。
2. **只收结构化评测请求。** 请求绑定候选源码与摘要、task/target/backend、模式、seed、reps、cases、run/session、调用方类型与幂等键。评测通道不接受来自 pi 的 shell 命令、任意路径或“请把这份报告改成 ok”。
3. **设备由受信侧独占调度。** 设备锁与队列归设备侧执行端所有。正式评测优先于自测；同一块设备同一时刻只允许一个计时租约。现有本机 `gpulock` 是 local 执行端的实现，远程或虚拟化站点要在该站点内提供等价的服务端锁，不能让多个评测控制端各自以为自己拿到了同一块设备。
4. **计时在设备侧闭合。** 候选启动、重复计时与设备时间戳都在设备侧完成；网络往返只发生在提交与回传阶段，不进入计时窗口。若某个“远程操作”实现把每次 launch 都变成一次网络往返，它没有可信计时能力。
5. **报告带执行端身份回传。** 除 `EvalReport` 本体外，还要记录评测代码/镜像摘要、backend 版本、target 与设备身份、标定摘要、协议版本、请求摘要与签名。签名放在报告外层信封 `{report, provenance, signature}`，不把传输证明塞进 `EvalReport` 本体。评测控制端只接受 schema、源码摘要、target/设备与协议都对得上的报告；评测缓存键必须并入 evaluator `site_id`、设备身份、评测构建摘要与标定摘要，不能把一个站点的缓存报告用到另一个站点。

**宿主默认拒绝未登记站点和 evaluator epoch 发布。** `ExecutionConfig` 在 control_host 同时维护 `publishable_site_ids` 与 `publishable_epochs`。站点白名单限制服务位置，epoch 白名单精确绑定 site/device/worker_build/task_pack/calibration/protocol；split-control 必须同时命中当前 site 与当前 epoch 才能进入 confirmed、catalog、brief 的发布加速比和种子比较。`local-integrated` 的现有 local 身份保留兼容，split-control 两个列表默认都为空，必须由人显式登记；epoch 任何字段变化都要重新登记，不能靠旧 site 白名单放行新设备或新构建。`sim-npu` 永远不在生产 allowlist；远端 `measurement_kind` / `publication_allowed` 只作第二道校验，远端漏写或伪造标记也不能绕过宿主默认拒绝。未登记站点或 epoch 的报告只能作为未验证诊断证据，不能进入候选缓存或确认门。

**一次运行固定 evaluator epoch。** 运行开始时由 host 观测并固定 site/device/backend/backend_version/target/worker_build/calibration/protocol 及其 `epoch` 摘要，写入 execution manifest。之后每份 `EvaluationResult` 都与它比对；中途重新部署、换设备或换标定时抛 `EvaluationProvenanceError`，不得进入 orchestrator 只按源码哈希键的内存缓存，也不得与旧候选比较或进入确认。该情况以 `evaluation_environment_changed` 停止；resume 默认要求同一 epoch，改变 epoch 必须显式新运行或重新建立基线。

**传输故障与候选故障分开。** SSH 断开、服务不可达、协议不兼容、报告 provenance 不匹配分别用
`EvaluationTransportError` / `EvaluationProvenanceError` 表示，不能伪装成候选的 `crash` 或 `timeout`，
也不能直接变成 `agent_failed`。设备侧真正的候选 timeout/crash 才写入 `EvalReport`；selfcheck 的暂时
不可用按 §7 非零返回、不计次数且不写权威记录，正式评测的基础设施失败进入独立的 `evaluation_unavailable` / 停止或 resume 路径。

正式评测的 `evaluation_unavailable` 不计入 `hangs`、`crashes_in_a_row` 或 `_agent_failures`，不把下一场交给另一个委员会。当前候选及其 `EvaluationRequest` 先以 `pending_evaluation` 持久化，run 以可 resume 状态停止；resume 时只在 request digest 与 evaluator epoch 都不变时，用同一 `request_id` 重查/重提。服务端必须在开始 job 前持久化幂等记录和最终结果，才能避免客户端断线后的重复设备执行；没有这层持久化时只能声称 at-least-once 执行，不能声称 exactly-once。

**超时、取消和进度分层。** 请求体只携带服务端需要的 `queue_timeout_s` 与 `candidate_timeout_s`；transport timeout 是客户端本地策略，不进入 request digest，由客户端按“排队上限 + 候选执行上限 + 协议收尾余量”计算，resume 可只调整本地余量而不改变 request_id。客户端先断线不能假设服务端已经停止。协议增加 cancel 帧、job/租约 TTL 和服务端超时清扫，取消必须终止 worker 进程组并回传最终状态。可选 progress 帧绑定 request_id 和递增序号，host 校验后写回 `eval_progress.json`，不能由远端指定 host 路径。

**缓存所有权先定再接入。** LocalEvaluator 的缓存由 local evaluator 管理；RemoteEvaluator 默认只使用
远程受信服务的缓存，host 不再用旧 key 直接复用 remote report。若 host 也缓存，键必须包含完整 request
与 provenance digest。当前 `_fast_bench_path()` / `_remembered_baseline()` 的最快基线缓存同样影响结果，
也必须按 site/device/build 隔离，或明确只存在设备侧服务内；不能只扩展 `_eval_cache_key()` 就算完成。

按设备形态落地：

| 设备形态 | 受信评测怎样成立 |
| --- | --- |
| 本地 GPU/NPU | control_host 上的受信 `tcenv` 进程直接评测；pi 只能交候选，不能改评测进程与结果 |
| 远程服务器 GPU/NPU | 评测控制端可在 control_host 或受信服务；远程服务器上的独立账号、容器或服务做设备侧执行，正式评测与受限自测用两种身份提交 |
| 虚拟机内 GPU/NPU | 优先用独立受信 guest、账号或容器持有设备；pi 所在 guest 不直接拿设备节点，自测也走受信设备操作通道 |
| 单块设备只能直通给一个 guest | 用设备租约在受信评测与自测之间串行调度，并在租约切换时做必要的设备/VM 重置；若 pi 对同一 guest 有 root，受信评测不能与它同 guest，必须换到宿主侧、另一个 guest 或可信镜像顺序启动 |

最后一条是硬约束：**能不能远程评测，只看受信设备操作能力，不看物理距离。** 若现有环境只能做到 pi 与设备侧执行端同 guest、同账号且 pi 有 root，这个组合只能记为开发模式，不能把它的计时写成可信发布结果。

### 6.5 其他 hmz 会话与实验驱动的放置

本仓库不只有 `kernel_once` 一种 hmz 会话，放置策略要逐个写清，不能默认它们自动跟随写核 runner：

| 入口 | 性质 | 放置原则 |
| --- | --- | --- |
| `kernel_once` | 写核内循环 | 默认 control_host local；搬家时按 `HmzSessionPlacement` 选择 |
| `feedback_review` | 只读反馈审查 | MVP 固定在 control_host；以后若移动，必须单独适配费用与只读输入清单 |
| `code_patch` | P2 代码生成会话 | 可与隔离测试共用环境基建（镜像、工件传输、清理），但使用独立安全策略：它要写 worktree 并访问 provider，不能套用测试环境的无网络策略 |
| `experiments/harness_runner.py` | 外循环实验驱动，不是 hmz 会话 | 留在 control_host；它只负责按已授权 plan 调度两臂与回收证据，不进内循环放置模型 |

## 7. 自测接口

自测属于内循环的工具，不属于外循环判卷。**默认形态下它跟随评测执行端，不跟随设备物理位置**：pi 在 control_host 写核，每次自测把候选发给 `EvaluationExecutor`，本地设备走 `LocalEvaluator`，远程设备走 `RemoteEvaluator`。只有 workspace 已按 §6.3 搬到设备站点时，自测才改从该环境发起并优先走站点内通道。

当前 `TASK.md` 里的 `uv run --directory ... tcenv eval` 带有 control_host 的仓库路径，设备一旦不在本机就不能照搬。

自测不另建一套协议，它就是 `EvaluationExecutor.selfcheck`，与 `formal` 共用传输、认证、请求摘要、设备租约、超时与审计：

```python
class EvaluationExecutor(Protocol):
    def formal(self, request: EvaluationRequest) -> EvaluationResult: ...
    def selfcheck(self, request: EvaluationRequest) -> EvaluationResult: ...
```

- `LocalEvaluator`：设备在 control_host 时，两者都包装现有 `tcenv.evaluate`，保持当前行为；`EvaluationResult` 由本地 adapter 补上 provenance。
- `RemoteEvaluator`：设备在远程时，两者都按 §6.4 的窄协议调用远程受信评测服务；区别只在调用方身份、模式白名单与配额。
- 可选的 `EnvSelfCheck`：仅用于 workspace 已搬到设备站点、且该环境本身就是受信评测执行端的情形，不能替代外循环的发布评测。

### 自测桥接与输出（实现前必须定）

`taic-selfcheck` 不能调用 `taic` 进程内的 Python 对象“隔空”方法。MVP 固定使用下面的 host bridge；独立 CLI 只在 host bridge 已证明无法覆盖目标环境时另做 spike。runner 必须把绑定放进 `SessionExecution.selfcheck`：

1. **host bridge（MVP 默认）**：`taic-selfcheck` 连接本场 Unix socket/本地 broker，由 host 持有 `EvaluationExecutor`，再转给 LocalEvaluator 或 RemoteEvaluator；
2. **独立 CLI（后续可选）**：`taic-selfcheck` 读取 workspace 外、agent 只读的 session 配置和短期 token，直接调用 RemoteEvaluator 服务。配置只允许引用 `SelfCheckBinding.token_ref`，不得把秘密写进 TASK 或日志。

最小的模型侧命令只允许选择模式和本场 workspace 内的候选，例如：

```bash
taic-selfcheck --mode quick --candidate <workspace-relative-path>
```

`task`、`target`、`run/session`、`reps`、`cases`、费用/次数上限、最大 timeout 和 quota 都由 host/服务端从 session 配置取得，agent 不能通过命令行改大。`--validate` 是否可用也由 selfcheck policy 决定。候选由 host 对实际发送、实际编译的字节计算 SHA-256，不做换行或 Unicode 规范化，远程请求不携带任意 host 路径。host 解析候选时必须拒绝越界、符号链接和超大文件；bridge 只暴露 selfcheck，不暴露 formal，agent 不能通过本地 socket 冒用外循环身份。

`taic-selfcheck` 还必须保留当前 `tcenv eval` 的展示语义：candidate 微秒、speedup、cv、`--validate` 行为，以及基于 `seed_report`、`min_improvement`、`noise_z` 的“是否够换种子”输出，复用现有 `tcenv` report/verdict formatter，而不是只返回一段 JSON。

“是否够换种子”由 host-side presenter 根据现有 `SessionBrief.seed_report_json`、`min_improvement`、`noise_z` 计算；这些值不进入 agent 可修改的 `EvaluationRequest`，也不要求远端服务决定搜索种子。direct_client 也只取回报告，再由 host 生成 verdict，避免 selfcheck 结果和外层选择语义分叉。

自测遇到 `evaluation_unavailable` 时，命令以非零退出并打印脱敏后的基础设施原因，**不计入自测次数或 GPU 秒数，也不写权威 selfcheck 记录**；agent 仍可交卷，外层正式评测独立决定。它不是候选 crash/timeout，也不触发 provider retry。只有设备侧真实候选 timeout/crash 才沿用现有限制后续 timed 自测的语义。

split-control 的自测配额在 `ExecutionConfig` 中解析，默认与当前 `tcenv.selfcheck` 对齐：最多 20 次 timed、累计 120 秒 GPU 时间；服务端取配置与自身硬上限的较小值。split-control 的 TASK 上限文案由 resolved 配置生成并计入 TASK hash/trust，local-integrated 仍用现有常量与原文，不能出现模型看到 20 次、服务端却按另一套拒绝的情况。

当前 `kernel_once` 的 `online=PermissionKind.NONE` 也必须纳入验收：不能假设 AF_UNIX 在 macOS/Linux 的 hmz 围栏下天然可用。Phase 2 的第一个关卡是在真实 hmz 围栏里用 fake agent 验证 host bridge，macOS 与 Linux 都通过后才继续；不调用模型/GPU。若当前 hmz 版本支持承载 typed selfcheck 的自定义工具，可以把工具通道作为备选；否则再评估 direct_client + 严格 endpoint allowlist，不能把 host bridge 当作未经验证的退路。

host bridge 每场使用独立 socket，放在 agent 不可写的 host 目录（目录只允许 traverse，socket 权限 0600），握手绑定 session token 和 request 前缀，只开放 selfcheck，会话结束立即销毁并拒绝后续连接。还要验证同机另一场会话不能冒用本场配额。

**投递方式固定如下：** runner 在启动 hmz 会话时，通过会话环境注入 `TAIC_SELFCHECK_SOCKET` 与 `TAIC_SELFCHECK_TOKEN`，不写进 argv、TASK、workspace、run_config、manifest 或日志；`taic-selfcheck` 从自己的进程环境读取。agent 及其子进程可能继承这些变量，因此不能声称秘密“只存在于 shim 进程”。同一 OS 用户下，0600 与目录权限也只是卫生措施，不是防同用户进程的边界；真正能力边界是短期、selfcheck-only、绑定 run/session/mode/quota/request 前缀的 token，以及服务端配额、审计和会话结束撤销。若未来要求 token 只暴露给工具进程，必须改用已验证的 hmz 工具投递 API，不能靠环境变量措辞假装实现。

`local-integrated` 继续显示现有 `uv run ... tcenv eval`，保持 local TASK 字节不变；只有 `split-control` 等新 prompt profile 才把模型命令替换为 `taic-selfcheck`，并单独记录 TASK/skill hash 与 trust。`taic-selfcheck` 放在 workspace 外的只读路径或由环境提供的不可替换 PATH 入口，不能由代理写入。

远端模式还要同步适配 skill 提示面：当前 `taic-tools` skill 里直接 import Metal runtime、用 `gpu_lock` 跑的探针示例，在设备不可直达时必须改走 `tool_probe` 或明确禁用，不能只换自测命令就宣称提示面已适配；本地模式的 TASK 与 skill 字节保持不变。

### 7.1 共享的设备租约层（不做巨型 Broker）

正式评测、自测与受限工具探针（如 `metal.time_msl`）共用一层薄的传输与租约，不做一个包办三种结果语义的上帝 Broker（现有 `tcenv/tools/broker.py` 是 typed tool 的本地 broker，不是设备调度器，不要与它合并）：

- 共享层只负责身份、配额、队列、设备租约与审计；每个 job 仍启动独立的评测 worker 进程，不把设备评测做成长驻同进程服务。
- job 类型至少分 `formal_eval`、`selfcheck`、`tool_probe`。三者可以共用请求信封与设备通道，但模式白名单、优先级、配额与结果语义分开；`tool_probe` 继续用它自己的 typed request 与结果 schema，只借身份、租约与配额，不伪装成 `EvalReport`。
- 多 run 共用评测端时不能只有全局 formal 优先：按 run/session 做加权公平或 aging，formal 优先限定在同 run 内或设置全局上限，避免一个 run 饿死另一个 run 的全部自测。公平性只作用于租约获取顺序，不抢占已开始的设备 launch。
- 调用方身份至少分外循环正式评测与内循环自测。自测 token 绑定 run/session、target、允许模式与过期时间；次数与 GPU 时间上限由评测端服务端策略执行，不能靠 workspace 内的本地计数文件，也不能把上限只写进 token 文案。

传输优先选站点内 Unix socket/vsock，或 SSH forced command / socket forwarding；不直接新开局域网 TCP 服务。服务端 stdout 只允许帧字节，sshd banner/motd、forced-command 杂散输出和任何非帧文本都视为协议错误；stderr 只走受限日志通道，不能混入帧解析。若走 SSH forced command，第一版可依靠通道认证暂缓报告独立签名，但身份、请求幂等与防重放第一天就要有：同一 idempotency key + 同一 request digest 只能返回原结果且不重复占设备，同 key + 不同 body 必须拒绝，过期 token 与 formal/selfcheck 身份互换必须拒绝。正式远程评测进入发布链前，报告完整性必须与执行端身份绑定（§6.4 第 5 条）。

租约不能无意改变当前 Metal 的并发语义：correctness 等未计时阶段可共享，benchmark 才升级独占。formal 的优先体现在获取租约时的排队优先级，不是抢占已经在设备上执行的 launch；是否支持更粗粒度独占要由 backend capability 明确声明。

### 7.2 premise 与自测证据的权威记录

跨环境后，P1 `premise` 的证据链按下面六条收口：

1. 不再比较 control_host 的 `session_started` 与远端文件系统 `mtime`，也不信任 agent 可写 workspace 里的时间标记。host 在 staging 前记录 session 起点，评测端为每次自测/tool_probe 签发递增 receipt；workspace 内的文件时间只作诊断，不作权威新鲜度证明。
2. 远程评测端只返回 `report + provenance + evidence + receipt`，**不直接写 control_host 的任意目录**。control_host 校验 run/session、request digest、身份与 receipt 后，由 host-side `EvidenceStore` 原子写入权威 `self_checks.jsonl` 与 `tool_evidence/`；workspace 侧只拿到只读镜像。
3. `taic` 结算 `premise` 时只读 host 权威记录与外层候选报告，不读 workspace 侧可写的自测副本。代理在 NOTES/FEEDBACK 里写的数字只是待核对文本。
4. 每次自测响应至少回传 receipt id、候选哈希、测量值与权威记录摘要，内循环据此写反馈，下一场也能复核。
5. **按 receipt 去重。** 自测记录从第一阶段就带 `receipt_id / request_digest`；host 物化 canonical workspace 时按 receipt 去重，权威记录与只读镜像不得把同一条自测的次数、GPU 时间与 premise 证据算两遍。旧的 local 记录没有 receipt 时保留原行为。
6. **premise 需要探针执行证据，不只是一个文件和一张 selfcheck receipt。** 远程 artifact 的 mtime 不能证明 probe 在本场创建或执行；若开放远程 premise，必须有 `tool_probe` receipt 绑定 session、probe_sha256、request_digest、测量值、evaluator build/device。第一版 split-control 可以明确暂不允许远程 premise 自动激活，只开放写核、普通 selfcheck 和正式评测，待 tool_probe 证据协议完成后再开放。

## 8. P2 补丁测试运行时

当前 `_test_patch_in_worktree` 会在 host 执行补丁后的测试，属于“隔离 worktree”，不是安全沙箱。P2 必须先拆成两个阶段；下面两个命令是计划新增的接口，不是当前已有命令：

```bash
harness verify-code RUN --key KEY --static
harness verify-code RUN --key KEY --execute  # 消费已存在的 review-code 记录
```

`--static` 只检查路径、`allowed_paths`、acceptance spec、patch hash 和 implementation claims，不执行补丁代码；`--execute` 必须等人看过 diff 后，才能在隔离环境中运行测试。两个阶段都要留下独立的验收记录，不能只看一个 `accepted` 布尔值。

计划抽象。动态测试的执行环境与写 patch 的编程代理位置是两回事：code agent 在哪里写 patch，不决定测试在哪里跑。

**先限定“未审阅 patch 不执行”的范围。** 当前 `taic/code_agent.py::run_code_agent` 会在 host 启动 `code_patch`，而 `taic/flows/code_patch` 使用 `LocalEnv`，prompt 还允许 agent 运行相关测试；因此人工看 diff 之前，生成阶段本身可能已经执行了未审阅代码。若本阶段只实现 `verify-code` 的静态/动态分离，就必须把承诺写成“未审阅 patch 不经过 `verify-code --execute` 执行”，并明确 code_patch 生成仍是受信开发路径。若目标是整个 P2 都不执行未审阅代码，则必须把 code_patch 生成也放进隔离环境，禁止它在生成阶段运行测试，先回收 diff、人工审阅，再进入 IsolatedPatchTestRunner。

**本计划的默认落点是前者：** Phase 1 只隔离 patch 的验收/动态执行，code_patch 生成暂按受信开发路径处理，不把“全链路未审阅代码零执行”写成已经达成的安全性质。若未来把 code_patch 生成也列为不可信，必须先另开全链路隔离阶段；在那之前不得把该 patch 流程用于不可信输入。

```text
PatchTestRunner
  ├── LocalPatchTestRunner
  └── IsolatedPatchTestRunner（ssh / docker / apple-container / 预置虚拟机）
```

### LocalPatchTestRunner

保留当前快速开发路径，适用于人已经审查过 patch 的受信开发场景；它不是不可信 patch 的默认验证环境。

### IsolatedPatchTestRunner

- 将 patch 应用到隔离环境内的临时源码树。源码来自只读快照、本地 clone 或导出包，不假设只读挂载主仓库 `.git` 就能直接建 worktree；
- 依赖层用仓库锁文件离线构建并只读复用，网络隔离由环境层提供（不是某个命令的 `--offline` 文案）；测试进程以临时目录为工作区，导入必须来自打过 patch 的源码树；
- 在无控制机凭据、无默认网络、无 Docker socket 的隔离环境中运行离线测试；
- 不把控制机 `HOME`、API key、`~/.pi` 或主仓库写权限带进测试环境；
- 先让人查看 diff，再运行不可信 patch 的测试；不可信 P2 默认使用这条路径；
- 测试输出只作为验收证据，不能代替人工 diff review。

`--execute` 的记录必须绑定人实际看过的那一份 patch：`patch_sha256`、acceptance spec 摘要、`base_commit`、审阅人与时间。patch 或 spec 在审阅后变化，旧批准自动失效，必须重新审阅。现有 `verify-code` 已有实现验收语义，不悄悄改变它的默认行为：`--static` 只新增只读检查，`--execute` 要求已存在的 static 记录与人工审阅记录，legacy `--by` 与「允许执行 patch」的批准分开记，不共用一个含义模糊的字段。

为避免“命令行带了 `--by` 就假装人看过”，增加一个独立的人工记录动作（命令名为草案）：

```bash
harness review-code RUN --key KEY --by HUMAN
```

它只记录审阅，不执行 patch；记录包含 patch/spec/base 摘要、审阅人、时间和 diff 版本。`--execute` 必须消费这条记录；`apply-code` 也必须能区分 static、人工审阅、动态执行和效果验证，不能只检查一个旧的 `accepted` 布尔值。

**测试集必须是真正的 device-free。** 当前 `_test_patch_in_worktree` 用的 `pytest -m "not gpu and not llm"` 不够：仓库里有未标 `gpu` 却直接调用 `evaluate()` 的 Metal 测试，`not gpu` 不等于不碰设备。隔离 runner 第一版用明确的纯单元测试 allowlist，或新增可信的 `device_free / offline_strict` 标记并逐步清理测试分类；在此之前不得宣称「离线测试 = 无设备、无网络、无 host 依赖」。

若用户明确选择隔离环境而它不可用，必须报错，不得自动改跑 local。

## 9. 安全边界

### 9.1 威胁模型

资产是 control_host 凭据、仓库/评测/trust/control、其他 run 的数据、GPU/NPU 和发布结果。不可信输入包括模型生成的命令和代码、P2 patch、候选 kernel、日志和反馈文本。

隔离强度由 **harness 位置与 workspace 环境的组合**决定，不能只看一个 `vm` 标签：

- `harness=local + workspace=local`：无新增隔离，只适合受信开发环境。
- `harness=local + workspace=远端/容器`：代理进程与凭据留在 control_host，文件与命令进入隔离环境。这首先只是 **workspace transport/placement**，不自动等于隔离：它不证明代理读不到 control_host 其他文件、写不到 control_host，或凭据不会经进程/工具泄露。只有先验证 hmz 的工具锚定与权限围栏确实不能绕过 workspace 访问控制机文件，才可在此之上谈隔离，且残余风险必须写清。
- `harness=env/remote + workspace=远端/容器`：代理进程也离开 control_host，隔离更强；代价是远端要安装 agent CLI 并处理登录与凭据下发。

无论哪种组合，workspace 隔离的都是内循环和工具代码，不是候选 kernel：受信评测域仍会编译、启动和计时不可信 kernel，恶意 shader 仍可能攻击驱动或设备运行时。

因此两层都要防：内循环侧限制 workspace、网络、凭据和生命周期；候选侧继续做静态筛查、ABI 检查、保护带、超时、设备锁和确认。不得把环境内的 GPU 直接 passthrough 给不可信 kernel 当作发布计时，也不得把环境内计时当作发布成绩。设备驱动、容器运行时与 hypervisor 都要保持更新。

### 9.2 本场输入和输出

- 输入按 §6.1 的清单分可写工作区与只读上下文包，不挂载整个 control_host 仓库；
- 输入可只读，输出仅限本场交卷目录；
- artifact receiver 只接受清单内路径，拒绝绝对路径、`..`、符号链接、设备文件和超大文件；
- 工件回收在 host 侧按现有 `prune_work` 的保留与归档规则收口。远端只回逐个文件与清单，不回让 host 直接解包的不可信 tar；归档由 host 用回收到的文件生成；
- 被反馈点名的证据文件先经 artifact manifest 校验相对路径与归属，再进入保留集合，不能凭 FEEDBACK 文本任意点名读取 workspace 外文件；
- 交卷源码、日志和反馈都按不可信数据处理；
- workspace 内的 `TASK.md` 不能通过文字获得 control_host 的额外权限。

### 9.3 凭据和网络

凭据放在编程代理进程实际调用 provider 的那一侧，不按 workspace 名称推断：

- `harness=local`：凭据与 `~/.pi` 留在 control_host，不下发到 workspace 环境；workspace 只能通过 hmz 工具通道读写本场文件与执行命令。
- `harness=env/remote`：远端需要 agent CLI 的登录态。优先用短期、最小权限的凭据下发；control_host 侧 provider 反向代理（经 SSH 反向隧道把 provider 请求转回 control_host 注入凭据）是优先验证的候选，它能避免远端落盘长期 key 并在隧道断开后切断后续调用，但只有在验证它能完整承载所选代理的 streaming、重试、用量回传与预算熔断后才采用，不能预先写成已解决记账。
- 默认禁止任意 egress，设备站点受信评测域的自测通道单独放行；
- 不把 control_host 的 `~/.pi`、auth、ssh agent、Docker socket 或桌面会话暴露给 workspace 环境。

### 9.4 生命周期

- 外循环保留 wall-clock、预算和 kill 权限；`hmz` 负责结束本场内循环，`taic` 负责判定这场在搜索里算什么。
- 环境崩溃、网络中断与代理进程死亡映射为明确的 agent 失败，不能变成成功；
- 优先按 session 重置 workspace 状态，避免恶意状态跨 run 残留；
- reset 或销毁远端 workspace 之前，必须先回收交卷 checkpoint、工件清单与 hmz epic 定位信息。**费用对账键必须具体化**：当前 `epic_spend_by_label` 按本地 `<run>/agent_work` 下的 workspace 路径与 basename 匹配 epic，远端 workspace 后必然失效。`AgentResult` 要带回一组可绑定标识——`taic` 的 run/session label、hmz 的 session/epic 引用、远端 workspace identity——并写清三者如何绑定、用确定性的 remote workspace 映射作后备、两个 run 用相同 label 时如何防串账；只把字段改名成 `session_id` 不算完成；
- epic 取不回或绑定不上时，本场记为费用不完整（`cost_incomplete`）。`resume` 的 epic 对账与 `--accept-cost-gap` 语义保持不变：缺失对账不能静默当作零花费；
- 外循环负责清理环境、子进程、端口转发和孤儿会话，并继续对账模型费用；
- 日志回传 control_host 后做敏感信息脱敏和保留期管理，不把远端环境当成永久秘密存储；
- hypervisor、容器运行时、镜像和共享文件系统仍可能有漏洞，需保持更新并减少设备暴露。

workspace 隔离是防御层，不是对候选 kernel 正确性的证明，也不替代 P2 的路径黑名单和人工 apply。

### 9.5 可信评测域的残余风险

受信评测域与内循环同处一个设备站点时，仍有三类残余风险，plan 不能把它们写没：

- **同机不同账号不是强隔离。** 若 pi 能拿到 sudo/root、宿主管理接口或评测目录写权限，受信域即失效。评测域至少要做到 pi 无提权、无评测安装写权限、无报告目录写权限。
- **候选仍在真设备上执行。** 受信域解决的是“谁判卷、结果能不能被改”，不解决恶意 kernel 攻击驱动、让设备挂死或污染后续计时的风险。现有静态筛查、保护带、超时杀进程与确认门仍要保留；单设备站点在 timeout/crash 后先跑一次轻量已知-good 健康探针，探针失败才隔离设备并要求处理，不把完整 naive 评测当作每次异常的默认恢复动作。
- **签名只证明来源。** 报告签名能证明结果来自登记的评测域，不能证明那台设备上的评测代码没有被站点管理员替换。评测构建摘要、设备身份与标定摘要必须一并记录，变更时按新评测环境处理，不与旧结果混算。

### 9.6 候选源码与任务数据的生命周期

远程评测会把候选源码以及服务端重现任务所需的契约/参考/基线定义带离 control_host。`split-control` 的启用记录必须明确数据接收方；设备方或第三方托管服务不能因“只是评测”就默认获得无限期保留权。

- 服务端日志默认只记录 request_id、source/task digest、状态和必要错误，不记录完整源码、Triton 规格或参考实现；调试需要原文时必须单独开启并有期限。
- 为编译/评测临时落盘的源码和任务包使用 0600 权限与 job 独立目录，job 结束、取消或 TTL 到期后删除；崩溃转储、备份和日志归档也纳入同一清理策略。
- 记录谁能读源码与任务数据、保留多久、何时删除、是否用于训练或其他任务；跨第三方服务前需要人的明确授权。
- 回传 control_host 的 evidence 只保留验收所需的最小内容；API key、provider 登录态和宿主敏感路径不得随请求或调试日志发送。

## 10. 配置、resume 与证据

用单独的 `ExecutionConfig` 解析 `[execution]`、`[agent]`、`[workspace]`、`[evaluation]`，**不塞进 `SearchConfig`**：当前 `load_config()` 只保留 `SearchConfig` 认识的字段，未知字段会被静默过滤，不能依赖它拒绝非法组合。字段名为草案，默认形态只需要 `[execution]` 与 `[evaluation]` 两段：

```toml
[execution]
profile = "local-integrated"  # local-integrated | split-control | device-site-colocated
selfcheck = "local_cli"       # local_cli | host_bridge | direct_client

[agent]
kind = "pi"                   # MVP 只有 pi
harness = "local"             # 默认 local；env | remote 仅可选扩展

[workspace]
env = "local"                 # 默认 local；ssh | docker | apple-container | slurm 仅可选扩展
endpoint = ""                 # 不含凭据
image_digest = ""
network = "deny"
reset = "per_session"

[evaluation]
transport = "local"           # local | ssh | unix-socket：到评测执行端的传输
endpoint = ""                 # 不含凭据；local 时为空
site_id = "local"             # 不透明站点标识，如 "npu-node-01"
expected_worker_build = ""    # 可选的期望/allowlist 构建摘要；不是实际观测值
expected_task_pack_digest = "" # 可选的期望任务包摘要；空时由 host 任务清单计算
trust_domain = "host-process" # host-process | service | guest
publishable_site_ids = []       # host 站点白名单；split-control 默认空，local-integrated 由受信解析器填 local
publishable_epochs = []         # 额外要求精确 evaluator epoch；split-control 默认空即不可发布
selfcheck_max_timed = 20        # 与 tcenv.selfcheck.MAX_CHECKS 对齐
selfcheck_max_gpu_s = 120.0     # 与 tcenv.selfcheck.MAX_GPU_S 对齐
```

`profile` 与 `selfcheck` 是显式请求，resolver 再派生 harness、workspace、evaluator transport 和 selfcheck binding；请求与派生冲突时在配置加载阶段拒绝，不能到运行时才拼出非法组合。`local-integrated` 默认 `local_cli`，`split-control` 默认 `host_bridge`，`direct_client` 只能在显式扩展 Profile 中启用。manifest 的 `requested` 保存原始 profile，`resolved` 保存派生结果，二者都可审计。

虚拟化方式（passthrough / virtual-function）不进这份配置：它由评测端的部署与能力探测决定，只作为观测结果写入 manifest。

execution manifest 必须区分三层，不能把用户写的配置直接当运行事实：`requested`（用户请求的 profile）、`resolved`（配置加载后确定的实际 placement/evaluator）、`observed`（启动后真实观察到的 hmz 版本、镜像、设备身份与评测构建）。`run_config` 同时保存 profile、resolved manifest 与其摘要；每次运行写一份 execution manifest，并进入 `run_config.json`、session 记录、看板和 resume 校验：

```json
{
  "hmz_version": "0.2.1",
  "agent_kind": "pi",
  "agent_impl_version": "...",
  "harness_placement": "local",
  "workspace_env": "local",
  "control_host": "local",
  "profile": "split-control",
  "evaluator_transport": "ssh",
  "evaluator_site_id": "npu-node-01",
  "device_id": "...",
  "evaluation_trust_domain": "service",
  "evaluator_build_digest": "...",
  "expected_worker_build": "...",
  "task_pack_digest": "...",
  "evaluator_epoch": "...",
  "publishable_site_ids": [],
  "publishable_epochs": [],
  "task_fingerprint": "...",
  "calibration_digest": "...",
  "network_policy": "deny-except-eval",
  "mounts": ["task-ro", "workspace-rw"],
  "fence": true,
  "selfcheck": "host_bridge",
  "task_sha256": "..."
}
```

不记录密码、私钥、完整 Authorization 或 control_host 敏感路径。`evaluator_build_digest`、`evaluator_epoch` 和设备身份只能来自 observed 证据，不能由用户配置冒充；示例 JSON 不含注释。环境启动、文件复制、内循环、评测、回收分别计时；环境计算成本是否计入搜索预算必须显式定义，不能与 kernel 加速比混合。看板显示 harness 与 workspace 的组合、镜像、reset、网络策略和孤儿清理状态。

它与 trust 的关系要分清：execution manifest 是运行证据，不是提示文本本身。只有模型实际看到的 TASK、skill 与 system prompt 变化才影响 trust；为此每场另记实际 `TASK.md` 的 SHA-256。不能因为换了 workspace 就让全部旧授权无条件失效，也不能在 TASK 命令已经变了时假装提示面没变。

### 10.1 Schema 与缓存的升级纪律

- Phase 0a–2 不改 `EvalReport` 顶层 schema。execution manifest 只进 `run_config.json` 的可选块与 session 记录；`run_config` 现有读取不因可选块拒绝旧文件，resume 必须显式校验它，不能靠未知键被忽略就假装已绑定。
- 接入远程评测时若要给报告加执行端身份，优先用报告外层信封承载 `provenance` 与签名。只有评测结果本体确需新字段时，才一次性把 `REPORT_SCHEMA_VERSION` 从 3 升到 4，并写清旧报告读取规则。
- SQLite 索引不因单 run 展示而升版。只有看板或查询确实要按 placement、evaluator site 或评测构建筛选时，才评估 `INDEX_SCHEMA_VERSION` 升版与全量重建成本。
- `eval_cache` 在接入跨站点评测前必须扩展键：并入 evaluator `site_id`、设备身份、backend 版本、评测构建摘要与标定摘要。扩展前的缓存不能被远端或虚拟化评测复用。LocalEvaluator 与 RemoteEvaluator 的缓存所有者按 §6.4 分开；remote 默认由服务端缓存，host 若缓存必须包含完整 request + provenance digest。
- `_fast_bench_path()` / `_remembered_baseline()` 的最快基线缓存也纳入同一 site/device/build 隔离规则；确认请求的 `cache_policy=disabled` 不得被任何一层缓存覆盖。

规则：

- 默认 `pi + harness=local + workspace=local`，不改变现有运行；local 是受信开发路径，不是隔离承诺；
- 不可信 P2 默认要求隔离动态测试环境，不能因不可用而自动改跑 local；
- `resume` 必须在任何模型/GPU 工作之前校验 execution manifest（profile、resolved placement 与 evaluator epoch），不一致就停；默认沿用原 placement 与 evaluator epoch，切换组合不得静默发生，需显式新运行或显式授权；
- 正式请求在首次发送前写入 run 内 `evaluation_outbox/<request_id>.json`，至少包含 `run_id`、`session_id`、`request_id`、`request_digest`、`candidate_sha256`、`evaluator_epoch`、`status`（pending/completed/unavailable）、`report_path`、创建/更新时间；不含秘密。收到结果后原子标记完成。`evaluation_unavailable` 恢复时先读 outbox；同 digest/epoch 才允许重查或重提，服务端也必须有持久化幂等记录，不能只在客户端保存 request_id。
- `RunResult.stop_reason` 使用独立的 `evaluation_unavailable`，brief/observe 从 outbox 显示 `pending_evaluation` 与最后传输错误，不把它映射成 candidate crash 或 agent failure。outbox 是 sidecar 状态，不改变 `EvalReport` schema；只有需要按 pending/epoch 过滤查询时才评估 SQLite 索引升级。
- A/B 实验的 control/treatment 两臂必须冻结同一个 placement 与 evaluator，否则实验差异会混入环境变量；环境/VM 成本单独记账，不混入模型 `CostLedger`，也不计入 kernel 加速比；
- 每场 workspace 按策略 fresh/reset，远端临时状态不是 resume 的唯一来源；control_host 的候选、转录、报告和账本必须足以重建；
- 远端 session 默认不承诺跨环境续聊，run resume 重放 control_host 证据，不伪造旧会话；
- local 的 `TASK.md`/prompt 字节保持不变；远端组合若需要不同自测命令，视为新的提示面，记录独立哈希并重新授权；
- placement 不改变发布加速比的可比性，启动开销单独展示；环境崩溃、超时、无交卷和网络中断映射为明确的 agent 失败并触发清理。

## 11. 实施阶段

顺序的原则只有三条：先证明 local 行为不变，先堵住 P2 在 host 执行不可信 patch 的洞，远程设备先接评测、最后才考虑搬代理。Phase 1a 与 Phase 1b 分别验收 static 和 isolated dynamic，不能相互代替；Phase 2A 与 Phase 2B 是同一服务端的连续交付物，不设第二条并行实现线；Phase 2.5 的 Docker/VM 集成是远程生产前的必选验收，缺环境先创建/安装，但它不阻塞 Phase 0a/0b/1a 的独立交付，Phase 1b 可用独立的隔离环境先行完成。

#### Phase 0a：local 写核 runner 抽取（不改行为）

- 把 `run_kernel_agent` 拆成 prepare / invoke / collect 三个内部步骤；定义 `HmzKernelSessionRunner`（输入 `SessionBrief + SessionExecution`，输出现有 `AgentResult`），不同时引入远端 workspace；
- 第一版 runner 只包装当前 `popen_group(hmz exec)` 路径，harness 与 workspace 都为 local；
- 默认配置仍是 `pi + harness=local + workspace=local + evaluation=local`；不新增远程配置，不改 `TASK.md`，不改 Metal，不改 provider retry；
- 保留 `agent_runner` 测试注入的兼容 adapter；先给现有 local 会话补特征测试：正常、超时、进程失败、无交卷、费用不完整（`cost_incomplete`）、checkpoint、自测计数与工件回收，再做抽取；
- 记录现有 local `TASK.md`、prompt hash、候选格式和 P0 基线。

验收：现有离线测试通过；local 的 `TASK.md` 和 Metal 行为不变；不启动远端环境、不调用模型、不跑 GPU。这一步只证明写核 runner 行为不变。

#### Phase 0b：LocalEvaluator adapter（不改行为）

- 定义完整的 `EvaluationRequest` / `EvaluationResult`，用 adapter 包住当前 `tcenv.evaluate` 的全部现有参数；`LocalEvaluator` 只包装本机 `tcenv.evaluate`；
- 保留 `evaluate_fn` 测试注入的兼容 adapter；provenance 先写 sidecar，不改 `EvalReport` 顶层 schema；
- `PatchTestRunner` 只定义最小的 local 包装，具体静态/动态状态机在 Phase 1 完成，不在写核 runner 抽取时一并改完。

验收：现有离线测试通过；formal 的 quick/full/confirm 参数与当前一致，自测输出和缓存行为不变；不启动远端环境、不调用模型、不跑 GPU。

#### Phase 1：P2 静态/动态分离（优先于任何远端 workspace）

本阶段拆成两个独立验收：**Phase 1a** 只做 static，不依赖外部环境；**Phase 1b** 做 isolated dynamic，使用独立的预置 SSH/Docker 环境或提前 provision 的隔离环境，不等待 Phase 2.5 的远程评测服务。Phase 1b 未完成前，不能把 Phase 1 整体标为完成，也不能用 local trusted path 冒充隔离验收。

- **Phase 1a**：增加 `verify-code --static`：路径、allowed_paths、spec、patch hash 和 claims，全部在不执行补丁代码的情况下检查；
- **Phase 1a**：增加 `harness review-code`（或等价的 `approve-execution`）只记录人工审阅、不执行 patch；动态执行由 Phase 1b 的 `verify-code --execute` 消费该记录，批准绑定 patch 与 spec 摘要；legacy `--by` 与执行批准分开记（§8）；
- 按 §8 已定的默认落点，本阶段只隔离 verify-code 的验收/动态执行，code_patch 生成仍标为受信开发路径；若未来把生成阶段也列为不可信，另开全链路隔离阶段，不在本阶段暗中扩大安全承诺；
- **Phase 1b**：实现 `IsolatedPatchTestRunner`，先于写核远端 workspace 落地；环境用只读依赖层与临时源码树，不假设只读 `.git` 可直接建 worktree；
- 测试集用明确的纯单元测试 allowlist 或新增 `device_free / offline_strict` 标记（§8），不用 `not gpu` 冒充无设备；
- 保留 `apply-code --confirm`、现有黑名单和 local 受信开发路径；
- 记录测试环境镜像、网络、挂载、人工审阅和结果。

验收：Phase 1a 的 static/review-code 阶段不执行 patch；Phase 1b 未审阅 patch 不经过 `verify-code --execute` 执行，且本阶段不声称 code_patch 生成阶段具有同样的隔离保证。动态测试环境无法访问 control_host home、主仓库写路径、provider 网络和 Docker socket；隔离环境不可用时不会把不可信 patch 退回 local，只能保持 Phase 1b 未完成或先 provision。

#### Phase 2：远程评测服务与客户端（两个连续交付物）

Phase 2A 先交付唯一的 `tcenv.remote` 服务端并用子进程管道验证，Phase 2B 再接客户端、桥接和远程调用；不再在 Phase 2 之外另起一条服务端实现线。先不做完整设备平台，也不搬代理。

##### Phase 2A：同一份 `tcenv.remote` 服务端 + 子进程管道

1. **正式服务端代码**：帧编解码、hello/capabilities、forced-command 入口、认证/幂等、队列/租约和 worker supervisor 放在 `tcenv/remote/`；`tests/support/npu_sim/` 只放共享 FakeNpuBackend、fixture、测试镜像与故障注入 helper。B0.5 与 Phase 2A 复用同一份 fake backend，不维护两套。
2. **测试注入不进入生产入口**：`sim-npu` 只在测试 fixture/临时 registry 中，不能注册成 production active target，也不通过改成 `active` 绕过 `require_active`。生产服务端制品不包含 sim 适配器或 `allow_simulation` 分支；FakeNpuBackend 只经测试镜像/测试 composition root 的 backend 注入点进入，2.5 要断言生产入口上模拟适配不可达。fake backend 不要求 `kernel void`、Metal import 或 GPU threadgroup limits，返回确定性的 report/provenance、task fingerprint、task_pack/site/device/build 与 receipt。
3. **子进程管道先行**：不用 Docker，先以子进程 + 管道运行同一份 `tcenv.remote` 服务端和帧编解码，形成真实字节与进程边界。帧至少含 `schema_version`、`frame_type`、`request_id`、`payload_length` 与 `payload_sha256`，限制帧/总请求大小，支持拆包、合包与有界内存读取；stdin/stdout 传数据帧，stderr 只作受限日志。forced command 只映射固定 evaluator entrypoint，`SSH_ORIGINAL_COMMAND` 只做严格协议/版本匹配，请求不能触发任意 shell、路径、环境或 `docker exec`。
4. **把问题变成回归测试**：覆盖 `SessionExecution` 参数、selfcheck bridge、request 覆盖当前 `tcenv.evaluate`、远程传输错误分类、task/task_pack fingerprint、缓存跨 site、receipt 去重、premise 执行证据与模拟发布门禁。已存在行为用普通测试；未实现接口用 `xfail(strict=True)` 或明确的 milestone-gated skip，不能让默认快速套件新增红灯。host `publishable_site_ids` 与 `publishable_epochs` 默认拒绝未登记站点/epoch，远端漏写或伪造 simulated 标记也不能进入确认、catalog 或发布数字。

验收：同一 `tcenv.remote` 服务端在子进程+管道上完成 hello、formal/selfcheck、报告/evidence/receipt、幂等和故障恢复；生产入口不包含 sim 代码；没有第二套测试专用服务端，没有 Docker、模型、GPU 或真实 NPU 依赖。

##### Phase 2B：EvaluationExecutor + selfcheck bridge + RemoteEvaluator

- **先过桥接关卡**：在真实 hmz `online=NONE` 围栏下，分别于 macOS 与 Linux 用 fake agent 验证每场 0600 socket、session token、只允许 selfcheck、跨会话不可冒用和结束销毁；不通过就先解决桥接，不进入远程联调；
- `LocalEvaluator` 仍是现有 `tcenv.evaluate` 的 adapter；桥接通过后，MVP 用 host Unix-socket bridge 打通 `taic-selfcheck`，再接 `RemoteEvaluator`（独立受保护 CLI 仅作后续可选 spike）；
- `taic-selfcheck` 只接受受限模式与 workspace 相对候选路径，复用现有 report/verdict formatter；selfcheck 的 task/target/reps/cases/quota 由 host/服务端决定；`evaluation_unavailable` 非零返回、不计次数且不写权威记录；
- `RemoteEvaluator` 先完成 hello/capabilities 与 expected_worker_build/task_pack_digest 校验，再固定 evaluator epoch，最后用 SSH forced command 或本地服务调用正式服务端；请求包含 §6.4 的完整字段、task_fingerprint 与 cache_policy，报告以 `EvaluationResult` 带回 request/source/target/build/provenance/receipt；
- 实现 host `publishable_site_ids` 与 `publishable_epochs` 默认拒绝和 `pending_evaluation` / `evaluation_unavailable` 的可 resume 状态，基础设施故障不计 hangs/crashes/agent_failures，也不换委员会；
- 实现服务端 queue/candidate 两层超时与客户端 transport 超时、cancel 帧、租约 TTL 和可选 progress 帧；请求 outbox 与服务端幂等记录先于 job 启动持久化，resume 只在 request digest 与 epoch 不变时复用 request_id；
- 正式评测与自测用不同身份与配额，默认值与本地 20 次/120 GPU 秒对齐，TASK 由 resolved 配额生成；按 run/session 做公平或 aging，formal 优先不饿死其他 run；
- host `EvidenceStore` 负责权威写入与 receipt 去重，远程服务不直接写 control_host 任意路径；认证传输第一天就绪，幂等、过期 token、身份隔离与防重放不能省；
- SSH/服务/协议/provenance 故障与候选 timeout/crash 分类处理；Local/remote 缓存所有者、fast_bench 基线缓存和 `cache_policy=disabled` 行为一并验证；全部使用 fake agent / fake backend，不调用真实模型。

验收：真实围栏下 host bridge 先通过；hello 能固定 epoch 并发现 task pack 不匹配；pi 留在 control_host 就能完成远程自测与正式评测；pi 无法伪造报告、冒用 formal 身份或抢占计时租约；RPC 不接受 shell/任意路径或杂散 stdout；自测输出与当前 quick/timed/validate/verdict 语义一致；远程基础设施故障不会误触发候选 hang/crash/agent_failure 计数；未登记站点或 epoch 不能发布；epoch 变化会停止而不是混算；pending_evaluation 可按同一 request_id 幂等恢复；progress 能安全回到看板；多 run 自测不会被单一 formal 队列饿死。

#### Phase 2.5：macOS → Linux VM → Docker 的远程服务集成（远程生产前必做）

详细环境 provisioning、真实传输用例、故障矩阵、授权开关与验收报告见[远程评测模拟验收计划](远程评测模拟验收-plan.md)。本阶段只保留四条约束：

- 它是声称远程评测可用于生产前的必做验收，不是 local/P2 的总闸门；
- 使用 Phase 2A 已交付的同一 `tcenv.remote` 服务端，不另写测试替身；真实穿过 Mac → SSH → Linux VM → Docker → worker；
- `sim-npu` 只是 CPU 假设备，不做真实 NPU，不产生性能或发布结论；
- 缺 Linux VM/Docker 时按专项文档创建/安装，不把缺环境作为跳过理由。

#### Phase 3：一个 SSH workspace 的 transport spike（可选，Phase 2 之后）

只有 §0 的隔离或工具链触发条件出现时才做。本阶段只证明传输，不宣称安全隔离：

- 把 `kernel_once` 的 `LocalEnv` 改为可配置 Env，并传 `-e` 映射；只选一种环境（优先 SSH 到预置主机）；
- 完成输入清单、artifact manifest、host 侧 canonical workspace 物化、按 receipt 去重、超时、断连与清理；
- 先验证 hmz 的工具锚定与权限围栏：内循环不能绕过 workspace 读取 control_host 文件；做不到就只记为 transport，不记为隔离；
- 本阶段不做远程自测（自测仍走 Phase 2 的 `EvaluationExecutor`）；环境不可用时明确失败，不 silent fallback；
- 同步落实 epic 的 session identity 绑定（§9.4）与远端 TASK/skill 提示面的独立哈希（§7、§10）；
- 用 fake agent 写固定 kernel 做端到端测试，不调用真实模型。

验收：远端组合与 local 产生同样的候选结构并物化到同一 canonical 布局；外循环能独立评测交回的 kernel；workspace 无 control_host repo/home 写权限；费用可按 session identity 对账，对不上记 `cost_incomplete`。

#### Phase 4：按证据决定是否搬代理（可选）

只有 Phase 2/3 的实测证据表明远程自测延迟已影响搜索、设备工具必须在 agent 环境内可用、或隔离要求代理进程离开 control_host 时才做：

- 在设备站点或隔离环境安装所选 agent CLI，处理登录、短期凭据（或先验证 provider 反向代理）、streaming、费用回传、会话清理与 orphan session；
- pi 的 workspace 不直接持有可信评测权限；自测与正式评测仍经同一 `EvaluationExecutor`。

验收：远端 agent 环境不持有长期 provider 凭据（control_host 作为受信控制面可保留本地登录态）；远端崩溃、超时和无交卷都能清理并映射为失败；pi 无法写受信评测域的代码、锁与报告。

#### Phase B0：保持 Metal host evaluator 不变（与上面并行）

- 继续使用当前 `tcenv`、Metal backend、设备锁、确认门和 P0 基线；
- 不因接入远程评测或远端 workspace 而改变 Metal 的计时或发布口径。

验收：Metal 的现有离线回归通过；真实 GPU 对照只在单独批准后运行。

#### Phase B0.5：backend-neutral cleanup + fake backend（真正 NPU 之前）

在任何 NPU 批准之前，先把当前 Metal 硬编码从通用评测路径里拆出去，并用一个 fake backend 证明拆干净了：

- 通用评测只调用 `backend.validate_launch()` 或等价钩子；可以上移通用异常类型，但 Metal 的 grid/threadgroup 检查留在 Metal backend，Metal 的 shader validation 与 Triton→Metal 编译基线也只在那里启用；
- 增加 import-boundary 验收：沿 fake backend 的通用评测路径执行时不得 import `tcenv.backends.metal.*`，并逐项处理 `tcenv.evaluate` 当前的 Metal screen/launch/sanitize/triton_baseline 直接依赖；不能只凭 fake backend 可以构造就宣布解耦；
- 任务指纹按 backend 计算参考源：新增 `task_fingerprint(task, backend_id)` 或等价接口，Metal 保持 `reference_for("metal")` 的旧行为，未来 backend 不再错误地只哈希 Metal 参考实现；
- `TargetProfile` 保留通用的 root、language、file_ext、backend_version、skills/docs 与 calibration 元数据；设备上限拆成 backend-specific limits，设备匹配交给 backend/target hook，`gpu_family` 降为 Metal 兼容字段；
- 任务指纹、候选源码提取与 worker fallback 里的 `kernel void` / Metal 假设逐个列入兼容清单并给出 backend 钩子，不要求现在全改完；kernel 级 NPU 在 B2 前必须另定 `KernelContract`（或等价非 MSL 契约），不能只靠 graph/IR 的 `TaskFamily` 概念掩盖当前 `tcenv.contract.Contract` 的 MSL 绑定；
- fake backend 与 Phase 2A 共用 `tests/support/npu_sim/` 的同一份实现和 fixture，不另建一套；测试证明：不 import Metal、不要求 `kernel void`、不要求 GPU threadgroup limits、target/device identity 可区分、eval cache 不跨站点/设备混用；现有 Metal golden 与报告保持不变。

验收：上面的 fake backend 测试离线通过；Metal 离线回归与 golden 不变。

#### Phase B2：未来 NPU（另行批准）

- kernel 级 NPU：先落实 B0.5 的非 MSL `KernelContract` 与 backend-specific task fingerprint，再按 backend 协议新增实现、target、skills、测试和基线；接入方式优先用 Phase 2 的 `RemoteEvaluator`（pi 留 control_host，候选送远程受信评测），不要求先做 Phase 4；
- graph/IR 级 NPU：另立 graph/operator 任务、契约、IR、正确性和计时计划，不能硬套 Metal 的 dispatch/线程组模型；当前只保留 `TaskFamily = kernel | graph_ir` 与工件包（primary source + auxiliary files + manifest）的概念位置，不在本计划设计完整图编译框架；
- 每个 NPU 结果都带 backend、backend_version、target 和协议版本。

本计划不执行这一阶段。

## 12. 测试策略

### 不需要远端环境/GPU 的测试

- local 与 fake placement 的 `HmzKernelSessionRunner` 统一 artifact、状态和错误映射；
- 输入清单与工件回收的路径穿越、符号链接、超大文件和残留文件处理；
- 所选 workspace 环境不可用时不会 silent fallback；
- placement 配置写入、resume 绑定、实际 TASK 哈希与 prompt hash 记录；
- RPC schema、hello/capabilities、task_pack/expected_worker_build 校验、queue/candidate/transport 三层超时、cancel、TTL、预算和非法请求拒绝；transport timeout 不进入 request digest，resume 可只调整本地余量；
- 多 run 队列公平或 aging、formal 优先不跨 run 饿死 selfcheck；selfcheck 默认 20 次/120 GPU 秒与 TASK 文案一致，evaluation_unavailable 非零返回但不计次数、不写权威记录；
- `request_id` 幂等、规范 JSON request digest、候选实际字节 digest，以及同 key 不同 body 拒绝；服务端幂等记录必须先于 job 启动持久化；
- evaluator epoch 在运行中漂移时停止并不进入源码缓存/比较；未登记 `publishable_site_ids` / `publishable_epochs` 的站点或 epoch 即使漏写 simulated 标记也不能发布；
- `pending_evaluation` 不计 hangs/crashes/agent_failures、不换委员会，resume 只在 request digest 与 epoch 不变时复用 request_id；progress 帧只能由 host 写到固定看板路径；
- 远程设备默认组合校验：设备不在本地时，pi 留在 control_host 且自测/正式评测都经 `RemoteEvaluator` 仍可达；不能因为没搬代理就宣称不可自测，也不能在评测端不可达时假装可自测；
- 命名 Profile 的合法组合与非法笛卡尔积拒绝；同一 Profile 在本地 GPU、虚拟化设备与远程 NPU 节点上只换 backend/target，不换放置语义；
- premise 新鲜度在 workspace 时钟快慢两种偏移下都不能误判；权威自测记录与 workspace 镜像不一致时以权威记录为准，且按 receipt 去重后次数与 GPU 时间不翻倍；
- 共享租约层的三类 job（formal/selfcheck/tool_probe）身份、配额与优先级互不越界；
- fake backend 不 import Metal、不要求 `kernel void` 与 GPU threadgroup limits，且 eval cache 不跨 site/设备混用（Phase B0.5）；
- 受信评测域的身份校验：pi 篡改报告、抢设备锁、写评测目录或冒用正式评测身份时必须被拒绝；
- 单设备租约切换与异常后健康检查的状态记录；
- 第一版 runner 与现有 `kernel_agent` 在正常、超时、失败、无交卷和费用不完整场景下的行为对比；
- P2 static 阶段不执行任何补丁代码；execute 阶段必须绑定人工审阅人与 patch/spec 摘要；
- P2 runner 的 control_host 环境不泄漏测试（用 fake runner/静态检查）；
- 环境崩溃、网络断开、超时、无交卷、孤儿进程和费用对账；
- 候选 kernel 仍按不可信输入处理，不能把 workspace 标记当作正确性证明。

### Docker/NPU 模拟测试（不需要真实 NPU）

完整的模拟测试矩阵、环境 provisioning、授权开关和 `runs/npu-sim-<日期>/report.json` 规范见[远程评测模拟验收计划](远程评测模拟验收-plan.md)。这里只保留测试体系约束：

- 纯 fake 测试默认进入安全 CPU 套件；Docker/VM 集成只使用 `npu_sim` marker，conftest 在未设置 `TAIC_DOCKER_NPU_SIM=1` 时先 skip；测试模块必须 import-safe，顶层不得创建 Docker client、连接 socket、启动 VM 或读取凭据，所有外部动作放在 lazy fixture 中，skip 先于 fixture 执行；
- Phase 2.5 验收的是正式 `tcenv.remote` 服务端的真实 SSH/VM/Docker 链路，不是本机 fake 函数或另一套测试服务；
- 模拟报告必须被 host 发布策略拒绝，不产生 NPU 性能、硬件正确性或发布加速比结论；
- 环境授权与执行开关只在专项文档 §5 记录，主 plan 不保留第二份口径。

### 显式授权的集成测试

- 在真实 hmz `online=NONE` 围栏下分别于 macOS 与 Linux 验证 host bridge：每场 socket 0600、目录不可写、session token 经会话环境注入而不落 argv/TASK/workspace/manifest/log、只允许 selfcheck、跨会话拒绝和结束销毁；明确同 OS 用户下文件权限不是能力边界，token 才是 selfcheck-only 能力票；不调用模型/GPU；
- 验证 Phase 2.5 镜像运行的是正式 `tcenv.remote` 服务端，而 FakeNpuBackend 只通过测试注入；生产制品不得包含 sim 适配器或 `allow_simulation` 可达路径；frame codec、forced-command、认证/幂等、队列/租约与 worker supervisor 不得存在测试专用的另一套实现；
- 在一台预置 SSH 主机或本地虚拟机中用 fake agent 写核；
- SSH/挂载/网络断开、环境崩溃、超时和清理；
- `harness=local` 时验证工具锚定不能绕过 workspace 访问 control_host 文件；
- 隔离 P2 测试无法访问 control_host 凭据和主仓库写路径；
- 只有用户明确批准后，才做真实模型或 GPU 对照。

## 13. 回滚

- 默认 `pi + harness=local + workspace=local`，删除或禁用远端 placement 配置即可回到当前路径；local 是受信开发路径，不提供隔离承诺；
- 远端组合只通过统一 runner 接口接入，失败不得改变 local；
- 任何 local prompt、报告 schema 或 Metal 行为变化都视为单独变更，不能用本计划的名义合入；
- P2 隔离 runner 失败时保留 local runner，但不可信 patch 不得因此自动改跑 local；
- `verify-code --static` 可独立回滚，不得因为动态 runner 不可用而跳过人工审阅直接执行 patch。

## 14. 已定落点与仍待评审的问题

先把跨设备通用的落点定下来，避免把 Mac 上的具体做法写成所有 GPU/NPU 节点的规则：

| 问题 | 本稿落点 |
| --- | --- |
| 对外支持几种拓扑 | 只支持 §4.2 的三个命名 Profile（`local-integrated` / `split-control` / `device-site-colocated`）；远程设备默认 `split-control`，底层维度只作证据与扩展点，非法组合在配置加载时拒绝 |
| workspace reset | 默认 `per_session`；reset 前必须完成工件与 epic 回收 |
| 自测通道 | 同站点优先本地通道或 Unix socket/vsock；跨机优先 SSH forced command 或 socket forwarding；不新开局域网 TCP |
| 自测命令 | local-integrated 保留现有 `tcenv eval`；split-control 使用稳定的 `taic-selfcheck`，由 runner 提供且不可被代理替换 |
| P2 隔离环境 | 只读依赖层 + 临时源码树离线跑测试；人工审阅记录绑定 patch/spec/base |
| 设备健康恢复 | timeout/crash 后先跑轻量已知-good 探针，失败才隔离设备 |
| Docker/NPU 模拟 | **远程生产前必选**：Phase 2A 先用同一 `tcenv.remote` 服务端和子进程管道暴露问题，Phase 2.5 再用 Mac 控制面 + Linux VM Docker 验收**可部署的远程受信评测服务端**；真实 SSH/forced command/长度前缀帧传输源码、命令、报告与证据；不做真实 NPU 测试，不发布模拟数字。细则见[远程评测模拟验收计划](远程评测模拟验收-plan.md) |
| 评测环境一致性 | 运行开始固定 evaluator epoch；每份结果校验，漂移即停止，不进入源码缓存、候选比较或确认；resume 默认要求同一 epoch |
| 发布许可 | host `publishable_site_ids` 与 `publishable_epochs` 默认拒绝未登记站点/epoch；远端 simulated/publication 标记只作第二道校验 |

仍待评审：

1. 第一个远程受信评测服务端由谁在设备侧提供——设备方托管、内部平台提供，还是本项目在设备站点部署（例如 SSH forced command）？它的身份、配额、设备锁与报告摘要是否满足 §6.4 五条？
2. 真实远程站点的自测往返延迟是多少、何时触发 Phase 4 搬代理？Phase 2.5 的本地 VM 数据只用于协议与下限诊断，不能用来定阈值；必须在真实站点另测。
3. 目标设备是单块直通、虚拟功能还是可多域共享？单设备时租约切换、重置与健康检查由谁负责？
4. 受信评测执行端用独立账号、容器还是独立 guest？pi 的最高权限到哪里为止？
5. NPU 未来是 kernel 级还是 graph/IR 级？这决定是否复用当前 `Task`/`Contract`。
6. 若以后真做 `harness=env/remote`，短期凭据下发与 provider 反向代理先验证哪一条？两者都要做小规模 spike 后再定，不阻塞 Phase 2。

实施顺序固定为 Phase 0a/0b → Phase 1a → Phase 1b → Phase 2A → Phase 2B → Phase 2.5；B0.5 可与前期离线工作并行。Phase 0a、0b、1a、1b、2A、2B 各自独立验收，不提前引入远端写核、代理搬家、真实 NPU 或新的 GPU 行为。Phase 2.5 是远程评测声称可用于生产前的门禁，不是全 plan 或 local/P2 工作的门禁；其环境授权与执行开关统一见[远程评测模拟验收计划](远程评测模拟验收-plan.md) §5，没有环境时先创建/安装。

## 15. 评审采纳记录

| 建议 | 处理 | 落点 |
| --- | --- | --- |
| P2 测试隔离先于远端 workspace | 采纳。P2 是当前直接执行不可信 patch 代码的高风险路径 | §11 Phase 1 先于 Phase 3 |
| host 评测仍执行不可信 kernel | 采纳。workspace 只隔离内循环，不能证明 kernel 或驱动安全 | §9.1 |
| `hmz` 与系统同属外循环，编程代理是内循环 | 采纳。`taic` 决定轮次与判卷，`hmz` 只启动并监督一场会话，代理在会话内多轮试错 | §0、§4、§5 |
| 代理进程与 workspace 环境分开建模 | 采纳。`hmz` 是启动器，不是 hypervisor；代理可在 control_host、环境内或第三台机器，文件与命令由 workspace 环境承载 | §4、§6 |
| MVP 收窄为 `HmzKernelSessionRunner` | 采纳。先包装现有写核 session，不发明通用 agent 框架；`agent_kind` 暂只支持 pi | §0、§6 |
| P2 分静态验收和动态执行 | 采纳并限定范围。未审阅 patch 不经过 `verify-code --execute`；若要覆盖 code_patch 生成阶段，必须另做隔离，不能把生成阶段的 host 执行忽略；测试集须是真正的 device-free，不用 `not gpu` 冒充 | §8、§11 Phase 1 |
| 区分 control/harness/workspace/evaluation 四个位置 | 采纳但收窄。默认只用 control_host 内循环 + evaluator 两处；harness/workspace 搬家降为可选扩展 | §4 |
| pi 就近设备站点（上一轮落点） | **已被本轮评审推翻，改写。** 设备位置不再绑架 pi：pi 默认留 control_host，自测与正式评测经远程受信评测服务完成；colocated 只在延迟/工具链/隔离三条件之一成立时启用 | §0、§4.1、§4.2、§11 Phase 4 |
| 不另造 `SessionSpec`，remote 物化 canonical workspace | 采纳。`SessionBrief` 是唯一会话内容模型，只新增 `SessionPaths`；remote runner 回收后物化 host 侧 `agent_work/<tag>`，兼容现有反馈与证据链 | §6.1 |
| 正式评测/自测共用客户端，不做巨型 Broker | 采纳并改写。共用 `EvaluationExecutor` 的传输、认证、租约与审计；`tool_probe` 只借租约，不并结果类型；对外请求不复用 worker 的本地路径字段 | §6.4、§7、§7.1 |
| epic session identity 与 receipt 去重 | 采纳。对账键具体绑定 taic/hmz/workspace 三方标识，对不上记 `cost_incomplete`；自测按 receipt 去重 | §7.2、§9.4 |
| execution manifest 分 requested/resolved/observed | 采纳并改写。独立 `ExecutionConfig` 解析，不塞 `SearchConfig`；resume 在模型/GPU 工作前校验；A/B 两臂冻结同一 placement/evaluator | §10 |
| 自测 RPC 收窄 | 采纳。短时 token、速率限制、源码上限、评测端计数，SSH forced command/port forwarding 优先；签名可随通道认证暂缓，幂等与防重放不可省 | §7 |
| 成本、清理、日志和 resume | 采纳。分别计时、孤儿清理、脱敏、workspace reset、control_host 证据恢复 | §9.4、§10 |
| NPU 前先 backend-neutral cleanup | 采纳。新增 Phase B0.5 拆 Metal 硬编码并用 fake backend 验证，不等真 NPU 到场才发现耦合 | §11 Phase B0.5 |
| NPU 独立设备线 | 采纳。kernel 级与 graph/IR 级分开，不和远端 workspace 实施捆绑；kernel 级优先走 `RemoteEvaluator` 接入 | §11 Phase B2 |
| local 不是安全边界 | 采纳。local 保留开发默认，不可信 P2 不得 silent fallback | §3、§10、§13 |
| 输入清单、证据权威记录与工件回收 | 采纳并改写。RW/RO 分开，premise 不再比跨机 mtime；远程端返回证据，host-side EvidenceStore 校验后落盘，超时后先 best-effort 回收 checkpoint | §6.1、§7.2、§9.2、§9.4 |
| 统一设备操作 Broker 与命名拓扑 | 采纳并改写。正式评测/自测/工具探针共用一层薄租约（不做巨型 Broker），job 与身份分开；对外只暴露三个命名 Profile，远程默认 `split-control` | §4.2、§7.1 |
| 自测 Shim、schema/cache 与 P2 配方 | 部分采纳。稳定命令名保留，但 Shim 必须不可被代理替换；签名放报告外层信封；P2 用临时源码树，不假设只读 `.git` 可建 worktree；provider 代理与健康探针先验证再定 | §7、§8、§9.3、§9.5、§10.1、§14 |
| 第二轮 runner/evaluator 接口 | 采纳。新增 `SessionExecution` 保留现有预算与 hmz 参数；`EvaluationRequest/Result` 覆盖当前 evaluate 语义并携带 provenance/receipt；保留现有测试注入的兼容 adapter | §6.1、§6.4、§11 Phase 0a/0b |
| selfcheck 桥接与输出 | 采纳。先实现 host bridge 或受保护 CLI，再接 RemoteEvaluator；复用现有 report/verdict formatter；显式处理 `online=NONE` 与 local/remote 两套提示面 | §7、§11 Phase 2 |
| 传输错误与任务/缓存边界 | 采纳。基础设施错误不等同候选 crash/timeout；请求绑定 task_fingerprint；明确 local/remote cache 与 fast_bench 基线缓存的 site/device/build 隔离 | §6.4、§10.1、§11 Phase 2 |
| P2 生成阶段与人工审阅 | 采纳并限定安全承诺。补充 `review-code`/`approve-execution` 记录；若要求全链路不执行未审阅代码，code_patch 生成也必须隔离 | §8、§11 Phase 1 |
| macOS/Linux VM/Docker NPU 模拟 | 采纳为**远程生产前必选验收**。Phase 2A 先交付并验证唯一 `tcenv.remote`，Phase 2.5 再用同一服务端穿过 SSH/forced command/长度前缀帧，验证数据传输、命令边界、报告回传和故障恢复；fake `sim-npu` 只是设备桩，不做真实 NPU，不产生性能或发布结论。环境授权与开关统一见[远程评测模拟验收计划](远程评测模拟验收-plan.md) §5 | §11 Phase 2/2.5、§12、§14 |
| 测可部署服务而非测试替身 | 采纳。frame codec、forced-command、认证/幂等、队列/租约和 worker supervisor 归入正式 `tcenv.remote`；tests/support 只提供 fake backend、fixture 与故障注入 | §6.4、§11 Phase 2.5 |
| 宿主发布许可与 evaluator epoch | 采纳。host 默认拒绝未登记站点发布；运行开始固定 evaluator epoch，中途漂移即停止，不进入缓存、比较或确认 | §6.4、§10、§14 |
| 桥接、超时与恢复 | 采纳。真实 hmz 围栏验证先行；每场 0600 socket 与 token；三层 deadline、cancel/TTL、progress 和 pending_evaluation 幂等恢复补齐 | §6.4、§7、§11 Phase 2 |
| 请求字段与数据生命周期 | 采纳并修正。task_fingerprint 不再重复 contract_digest；候选按实际字节哈希；request_id 明确为幂等键，provenance 类型化；补源码/任务数据的日志、保留和删除边界 | §6.4、§9.6 |
| 模拟测试表述与延迟口径 | 采纳。Phase 2A 用子进程+管道，完整 SSH/VM/Docker 留给 Phase 2.5；未实现接口用 strict xfail 或 milestone skip；本地 VM 延迟不用于真实站点搬代理阈值 | §11 Phase 2/2.5、§12、§14 |
| 合并 Phase 2 与模拟服务端 | 采纳。Phase 2A/2B 是同一 `tcenv.remote` 的连续交付物，Phase 2.5 只做完整 SSH/VM/Docker 集成；取消第二条并行服务端实现线，2.5 不阻塞 local/P2 独立交付 | §11 Phase 2/2.5、§14 |
| hello、任务包与 evaluator epoch | 采纳。hello/capabilities 返回首份 provenance 与能力清单；expected_worker_build 在握手时校验；任务包只读版本化部署并计算 task_pack_digest，纳入 epoch | §6.4、§11 Phase 2B |
| selfcheck 能力票与配额 | 采纳并修正。socket/token 经会话环境注入，不能声称只在 shim 进程；同 OS 用户下文件权限不是边界，selfcheck-only token 才是。配额默认 20 次/120 GPU 秒并驱动 TASK 文案，evaluation_unavailable 不计次数、不写权威记录 | §7、§10、§12 |
| 传输超时、队列公平与生产模拟隔离 | 采纳。transport timeout 移到客户端、不进 request digest；多 run 按公平/aging 排队；生产制品不包含 sim 适配器或 allow_simulation 可达路径 | §6.4、§7.1、§11 Phase 2A/2.5 |
| 授权来源与门禁范围 | 采纳并限定。plan 只记录用户明确授权的来源和范围，执行开关防默认副作用；Phase 2.5 是远程生产前门禁，不是 local/P2 的总闸门；缺环境仍先 provisioning | §11 Phase 2.5、[远程评测模拟验收计划](远程评测模拟验收-plan.md) §5、§14 |
| 拆分主计划与模拟验收 | 采纳。主 plan 保留架构、顺序和链接；环境 provisioning、真实传输、故障矩阵、授权开关与报告细则移入独立专项文档，授权口径只在专项文档 §5 保留一份 | §11 Phase 2.5、§12、[远程评测模拟验收计划](远程评测模拟验收-plan.md) |
| 配置、hello 与精确发布许可 | 采纳。显式 `[execution].profile/selfcheck`；hello 绑定 nonce/session/服务身份；发布同时要求 site 与精确 evaluator epoch，epoch 变化必须重新登记 | §6.4、§10、§14 |
| pending 数据模型与 P2 环境拆分 | 采纳。outbox 记录 request/candidate/epoch/status/report，`RunResult.stop_reason=evaluation_unavailable`，brief/observe 可读；Phase 1 拆成无环境 static 与需隔离环境的 dynamic，不互相冒充完成 | §10、§11 Phase 1 |
| 自测上下文、任务包与 import-safe | 采纳。verdict 使用 SessionBrief 的 host 上下文；task_pack_manifest 规范 digest；AgentResult.artifacts 可选；npu_sim 测试模块 import-safe，skip 先于外部副作用 | §6.1、§6.4、§7、§12、专项文档 §5 |
| NPU import boundary 与 KernelContract | 采纳。B0.5 明确通用 fake backend 路径不得 import Metal、task_fingerprint 按 backend 取参考源；kernel 级 NPU 先补非 MSL KernelContract，再谈实现 | §11 Phase B0.5/B2 |
