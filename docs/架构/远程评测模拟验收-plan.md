# 远程评测模拟验收计划

状态：container_only 预验收和 split-control 核心模拟验收已完成（2026-10-25）；扩展故障矩阵与真实 NPU 未做。是[执行运行时与设备扩展计划](执行运行时与设备扩展-plan.md) Phase 2.5 的专项文档。日期：2026-10-24。

本文只规定远程受信评测服务的模拟验收，不重复主计划的 runner、EvaluationExecutor、P2 和 backend 设计。主计划负责架构与交付顺序，本文负责环境、真实传输、故障矩阵、授权开关和验收报告。

## 1. 定位与边界

本阶段是**声称远程评测可用于生产前必做的模拟验收**，不是 Phase 0a/0b/1 或本地开发工作的总闸门。核心对象不是 NPU 硬件，而是 Phase 2A 交付的**远程受信评测服务端是否可行**：能否把候选源码和结构化命令真实地送到远端，由远端受信服务排队、启动 worker、操作设备并回传报告/证据。`sim-npu` 只是 worker 后面的假设备，不模拟或宣称任何 NPU ISA、SDK、驱动和性能。

测试必须真实穿过 Mac → SSH → Linux VM → Docker → evaluator worker 的进程与字节边界。Phase 2A 完成子进程+管道验证后，本阶段不再另写测试结构或服务端，只把同一实现放进真实远程链路，暴露部署、证据链、缓存和故障恢复问题。Mac 没有可用 NPU 时，**不使用真实 NPU、不安装 NPU SDK、不伪装硬件测量**；用 CPU 上的 FakeNpuBackend 产生确定性的模拟结果。

## 2. 实施步骤（Phase 2A/2B 完成后）

1. **准备环境（必选，允许 AI 执行可复现安装步骤）**：先检测 Mac 上已有的 Linux VM、SSH 和 Docker Engine，优先复用。缺 Linux VM 时，使用机器上已有的受支持工具（如 UTM/Lima/VirtualBox 等，按实际环境选择）创建一台；缺 Docker 时在 VM 内安装 Docker Engine，并准备 Python/uv 等测试依赖。可以编写幂等的 provisioning 脚本由 AI 执行，记录工具、版本、镜像摘要和安装日志；不自研 hypervisor，不安装 NPU SDK，不挂真实设备。授权范围、执行开关与系统交互统一见本文 §5。仅使用 Docker Desktop 而没有 Linux VM 时，结果只能标记为 `container_only`，仍需继续创建/配置 SSH 可达的 Linux VM，不能把它当作完整的 split-control 验收。
2. **构建并启动远程评测服务**：镜像使用固定版本/摘要，非 root、只读 rootfs、drop capabilities、`no-new-privileges`、无设备挂载、无 Docker socket、无 host HOME/API key、默认无网络；不把 NPU SDK 或真实设备节点带进镜像。镜像以正式 `tcenv.remote` 服务端为主体，测试只注入 FakeNpuBackend 和故障点；frame codec、forced-command、认证/幂等、队列/租约和 worker supervisor 与未来部署共用同一代码路径。容器内是实际 evaluator service：broker 加每 job 独立 worker，支持共享/独占租约、配额、receipt 和故障注入；不是一个只返回固定 JSON 的本机假函数或另写的测试替身。生产制品不得包含 sim 适配器或可达的 `allow_simulation` 路径。
3. **验证真实数据与命令传输**：Mac 上的 `RemoteEvaluator` 必须通过真实 SSH forced command 或 socket forwarding 连接该容器，不能绕过 SSH/VM 直接 import 服务。用 fake agent/固定候选调用 formal/selfcheck，验证 `taic-selfcheck` bridge、报告摘要、任务指纹、cache_policy、权威 EvidenceStore 和 receipt 去重。测试小/大/近上限源码、Unicode、分片与合包、准确 digest、最大帧/总大小、慢读写与有界内存；host workspace、repo 和文件路径不通过共享挂载提供。
4. **验证命令边界**：合法 formal/selfcheck 请求只能触发固定 worker；任意 shell、`docker exec`、绝对路径、`..`、环境变量注入、未知 frame、错误协议版本和 oversized length 全部拒绝。用一个 host/VM canary 文件证明注入字符串没有执行；stdout 数据帧与 stderr 日志不能混淆，worker 的退出码、取消和进程组清理必须回传正确。服务端 stdout 只允许帧字节，sshd banner/motd 或 forced-command 杂散输出都视为协议错误。
5. **注入真实远程故障并测量控制面开销**：在测试 proxy/service 中注入延迟、半包、连接重置、half-open、慢消费者、重复请求、服务崩溃和容器重启；重连后验证幂等、resume、artifact 回收与 orphan 清理。记录请求/回传的 p50/p95 往返时间、吞吐、内存上限、队列等待与恢复时间，作为协议和控制面诊断；这些控制面数字不得写进 candidate_us。fake backend 的模拟测量在服务端闭合，额外网络延迟不得进入模拟计时窗口。**本地 VM 的延迟只是下限，只能证明本地协议可行，不能用来决定真实远程站点是否需要搬代理；[主计划 §14](执行运行时与设备扩展-plan.md#14-已定落点与仍待评审的问题) 的阈值必须在真实站点另测。**
6. **复用 Docker 做 P2 隔离验证**：同一 Linux VM 可另起无网络、无设备、无 host 凭据的容器运行明确的 device-free patch 测试；它不是 NPU 测试，也不能替代人工 diff review。

## 3. 验收条件

Phase 2A/2B 完成后，在 Mac 不依赖真实 NPU，通过真实 SSH/VM/Docker 完成一次 `split-control` 远程评测闭环，且环境检测/创建/安装过程有可复现记录。验收必须同时满足：

- 合法 hello/formal/selfcheck 能完成，task_pack/epoch 校验生效；
- 传输前后源码/报告 digest 一致；
- 任意命令和路径注入没有执行；
- 重复请求不重复占设备，同 key 不同 body 被拒绝；
- 错误 task/build/provenance/replay 被拒绝；
- 并发两个 run 时 formal 不饿死另一 run 的 selfcheck；
- 远程故障不伪装成候选 crash/timeout，pending_evaluation 可按同一 request_id 幂等恢复；
- 容器/worker 无孤儿，生产入口的 sim 适配不可达；
- 网络延迟只进入控制面指标，不进入模拟设备计时；
- fake backend 不 import Metal、不要求 `kernel void` 或 GPU threadgroup limits；
- 任何模拟报告都不能进入发布链。

进入本阶段后，缺少 VM/Docker 就先创建/安装，不能把缺环境作为跳过或取消验收的理由。

## 4. 测试矩阵与报告

测试分两层，不能把“有 Docker”误写成“有 NPU”：

| 层 | 环境 | 能证明什么 | 不能证明什么 |
| --- | --- | --- | --- |
| fake backend 单元测试 | Mac 本机、纯 CPU | backend-neutral 接口、请求/报告、缓存键、错误映射 | NPU ISA、SDK、驱动、性能 |
| Docker/VM 集成测试 | Mac 控制面 + Linux VM 内 Docker | 镜像部署、SSH/传输、租约、配额、证据、故障恢复、P2 隔离 | 真实设备正确性、真实计时、发布加速比 |

**远程受信评测服务的可行性必须用真实传输用例判定，不能用本机 fake 函数代替：**

| 用例组 | 必须实际验证的内容 |
| --- | --- |
| 数据帧 | 长度前缀帧的拆包/合包、半帧、非法长度、最大帧/总请求、Unicode、近上限源码、payload digest、有界内存；报告/evidence 也走同一通道 |
| 命令边界 | SSH forced command 固定入口、`SSH_ORIGINAL_COMMAND` 严格匹配、禁止任意 shell/argv/路径/env/`docker exec`；canary 文件证明注入没有执行 |
| 进程与流 | stdin/stdout 数据帧与 stderr 日志分离、worker 独立进程、取消/退出码/进程组清理、服务重启后的重连 |
| 远程异常 | 延迟、慢消费者、连接重置、half-open、容器崩溃、服务不可达、协议错误、报告篡改；传输错误与候选 crash/timeout 分开 |
| 队列与身份 | formal/selfcheck 不同 token、配额与优先级；多 run 公平/aging，不能由一个 run 饿死其他 run；共享/独占租约；同 key 同 body 幂等，同 key 不同 body 拒绝，重复请求不重复占设备 |
| 可行性与开销 | 记录控制面 p50/p95 往返、吞吐、内存、队列等待和恢复时间；模拟 candidate_us 在服务端闭合，不含网络往返 |

每次 Phase 2.5 生成一份模拟验收报告（规划为 `runs/npu-sim-<日期>/report.json`），记录 VM/Docker/镜像/协议版本、上述用例结果、控制面延迟、故障恢复和未解决问题。报告只能证明“远程服务协议和生命周期是否可行”，不能证明 NPU 硬件正确性或速度，也不能用本地 VM 延迟推导真实远程站点的搬代理阈值。该目录是诊断 artifact，不是搜索 run：不写 `run_config.json`，标记 `simulated=true`，`observe` 不得把它当普通运行、best kernel 或发布证据索引。

## 5. 统一授权与执行开关（唯一口径）

授权来源是本次与用户讨论执行运行时计划的会话：用户在 2026-10-24 明确指示“环境没有就安装”，范围为 Phase 2.5 的 Linux VM/Docker 创建安装、固定镜像 build/pull 与模拟集成，不包括真实 NPU/GPU/模型。plan 只记录这项事实，不自行扩大授权。实际执行仍须显式进入该阶段或设置 `TAIC_DOCKER_NPU_SIM=1`，用于避免默认测试副作用，不是在同一范围内反复请求授权；超出该范围或需要 sudo、密码、系统扩展时，在执行点由用户完成必要交互，不读取或记录密码。缺环境先 provisioning，失败要暴露并修复；只有用户明确中止时才保留未完成状态。

- 纯 fake 测试默认可以进入安全 CPU 套件；Docker/VM 集成只注册一个 `npu_sim` marker，并在 pytest 配置中登记。conftest 在未设置 `TAIC_DOCKER_NPU_SIM=1` 时先 skip，因此无需修改 `AGENTS.md` 的标准安全命令。测试模块必须 import-safe：顶层不得创建 Docker client、连接 socket、启动 VM 或读取宿主机凭据，所有外部动作放在 lazy fixture 中，skip 在 fixture 执行前生效。授权后的计划命令形如 `TAIC_DOCKER_NPU_SIM=1 uv run pytest -q -n 0 -m npu_sim`，实际 nodeid 以实现为准。
- 普通 CPU 套件缺环境时 skip 并打印原因；进入 Phase 2.5 后按上面的统一授权先 provisioning，不得把缺环境当作验收豁免，也不得 fallback 到真实设备。
- 模拟报告在外层 provenance 标记 `measurement_kind=simulated`、`publication_allowed=false`，并使用独立 `sim-npu` 测试身份；更根本的是 host `publishable_site_ids` 与 `publishable_epochs` 默认拒绝未登记站点/评测 epoch。任何 `confirmed`、catalog、发布 brief 或加速比汇总都必须拒绝它；模拟得到的数字只能用于断言协议和状态机。
- Docker 容器断言：非 root、无设备节点、无 Docker socket、无 host HOME/API key、无默认网络、只读依赖/源码输入；网络、挂载、镜像摘要、容器退出和 orphan 清理都写入测试证据。另验证服务端日志默认不含完整源码/任务定义，临时源码目录在 job 完成、取消和 TTL 到期后都删除。
- 只有 Docker Desktop 而没有 Linux VM 时，测试名称和结果必须写 `container_only`，不能把它计为 SSH/VM 的 split-control 验收。真实 NPU 到来后另开硬件集成测试，不复用模拟结果作为硬件证据。

## 6. 本次已执行的验收

- `hmz` 0.2.2 已安装到 `.venv`，`tests/shared/test_security_boundaries.py`：2 passed。
- 使用本地 torch wheel 构建 `taic-patch-test:local`，真实 P2 Docker 验收：`tests/shared/test_docker_integration.py`：1 passed。
- 设置 `TAIC_TEST_DOCKER_IMAGE=taic-patch-test:local` 后，宿主机完整 shared 并行套件：987 passed。
- Docker Ubuntu 隔离环境中的远程/P2/selfcheck 测试：109 passed。
- 使用现有 Ubuntu ARM VM、VM 内 Docker 和 SSH forced command，运行 `tests/support/npu_sim/phase25_runner.py`：23 个验收项全部通过。覆盖 formal/selfcheck、hello、幂等、同 key 不同 body、Unicode/大帧、错误 worker build、shell/`docker exec` 注入、transport timeout、断连清理和重启恢复。
- 本次 split-control 报告为 `runs/npu-sim-20261024/report.json`，`split_control=true`、`container_only=false`、`simulated=true`；没有 `run_config.json`。最终没有遗留 evaluator 容器。
- 本次没有运行真实 NPU、真实 GPU、付费模型或安装 NPU SDK。VM 本地延迟只用于控制面诊断，不能用于决定真实远程站点是否需要搬代理。
- 本次报告的 `coverage` 为 `core_split_control`，尚未执行 half-open/慢消费者、完整帧上限和多 run 公平性扩展矩阵；不把它写成全部 Phase 2.5 条目均已完成。

复现用的构建与验收文件为 `tests/support/docker/Dockerfile.patch-test`、`tests/support/docker/build_patch_test.sh`、`tests/support/docker/Dockerfile.remote-sim`、`tests/support/npu_sim/forced_command.py` 和 `tests/support/npu_sim/phase25_runner.py`。
