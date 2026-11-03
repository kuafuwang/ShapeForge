# 模型写核与 P2 的安全边界

## 这篇讲什么

模型写 kernel 的目标是让搜索更快，但“更快”不能靠改评测、测试、信任或控制文件来换。本页记录当前边界，以及仍然存在的残余风险。

读完应带走三件事：

- 写 kernel 的 agent 只在自己的工作目录里动手，碰不到仓库的评测代码、trust、budget 或 control。
- 它自报的数字不是发布依据；候选必须回到外层 `tcenv` 重新编译、检查、做正确性测试和计时。
- 反馈文本和代码 patch 都按不可信输入处理：转换、review、apply 是分开的动作，P2 永久改动仍要人确认。

## 正常写核流程

`taic run` 通过 `hmz` 启动 `kernel_once`。写核 agent 的权限边界：

- 只能改自己的工作目录；
- 不能改仓库、评测代码、trust、budget 或 control；
- 只能读仓库和历史证据；
- 新 session 的 `_permission.online` 设为 `NONE`，不提供 web 搜索/抓取；
- `~/.pi` 和认证文件仍由 pi/hmz 自身管理，项目没有把这段 host 权限完全收回；这是残余风险。

候选仍要在外层 `tcenv` 中重新编译、检查、正确性测试和计时。写核 agent 自报的数字不进入发布门。

```mermaid
flowchart TD
    A["taic run 启动"] --> B["hmz 新建 session"]
    B --> C["kernel_once 启动写核 agent"]
    C --> D["agent 只在工作目录写候选 kernel"]
    D --> E["交回候选文件"]
    E --> F["外层 tcenv 重新编译"]
    F --> G["检查与正确性测试"]
    G --> H["计时并进入发布门"]
    D -.->|"自报数字"| X["不进入发布门"]
```

## 反馈与 review

`# 对系统的建议` 是不可信数据。自由文本不能直接变成 overlay；转换、review、apply 是三个分开的动作。

| 类别 | 是什么 | 生效需要的条件 |
| --- | --- | --- |
| `tip` | 一条可复用的经验提示 | 默认需要两个不同模型、两个委员会、active trust 和不同探针 |
| `premise` | 对任务前提的修正 | 需要本场探针、self-check/外层报告，以及 active `overlay:premise` |

其余规则：

- review 只留 advisory 记录，不自动激活 overlay；
- P2 永久改动仍要人确认；
- `harness_auto` 默认关闭。

```mermaid
flowchart LR
    A["建议自由文本"] --> B["转换成结构化提案"]
    B --> C["review 只留 advisory 记录"]
    C --> E["人确认"]
    E --> F["P2 永久改动"]
    A -.->|"不直接写入"| G["overlay"]
```

## P2 code agent

代码 patch 在 detached worktree 中生成，不 commit、不自动 apply。默认先做不执行代码的静态验收；人工看过 diff 之后，才用实名审阅记录和隔离容器执行补丁测试：

```bash
# 默认：只查路径、allowed_paths、spec、hash 和 claims，不运行补丁代码
uv run python -m taic.cli harness verify-code RUN --key KEY --static

# 人工看 diff，只记录，不执行
uv run python -m taic.cli harness review-code RUN --key KEY --by HUMAN

# 看过 diff 后：才在隔离 Docker 中执行补丁测试
uv run python -m taic.cli harness verify-code RUN --key KEY --execute --by HUMAN --docker-image IMAGE

uv run python -m taic.cli harness apply-code RUN --key KEY --check
uv run python -m taic.cli harness apply-code RUN --key KEY --confirm --by HUMAN
```

`--execute` 不接受默认的 `cli` 作为审阅人，并消费 `review-code` 记录的 patch/spec/base 摘要；三者任一变化，旧批准失效。隔离容器使用 `--network none`、`--read-only`、`--cap-drop ALL`、`--no-new-privileges`、pids/memory 限制和非 root 用户；只挂源码快照（rw）、patch/request（ro）和只读 harness，不挂 host HOME、provider key、Docker socket 或主仓库 `.git`。它不是 VM 沙箱，但比 host worktree 强；不可信 patch 仍应先人工 review。

`apply-code` 还有默认的 effect 门：它要求 implementation 记录 `accepted`/`host_verified` 且有 host checks；风险级别不是 `offline_only` 时，默认还要求同 spec/patch 的 `validate-effect` 记录。只有显式 `--legacy-unvalidated` 才豁免 effect 验证；它不是“跳过所有验收”的开关。

### 提案的接收者与可执行性（2026-11-03）

结构化提案里的 `audience` / `needs` / `peer_action` 是**模型自报的意图，不是事实**。宿主 `taic/proposal_routing.py::route_proposal` 独立计算 `recipient`、`requires_repo_change` 与 `actionable_within_session`：自报与 action 冲突时按更保守处理。只有 `overlay_candidate` 才可能进入后续 scope/probe/trust/never-auto 门；需要改仓库、加工具、改报告/评分材料、要权限或人决定的一律 `developer_queue`。不确定但没有这些硬越权的条目是 `review_candidate`：可交给后续写核 LLM 判断、改写或拒绝，但不能自动 overlay、不能写经验库。双来源 quorum 与 `review useful=true` 都不能覆盖这道门。`acceptance.kind_of` 仍只是证据分类，不决定接收者。开发者队列只记录意图，不自动实现请求。

`inherit_prompt` 已实现为**当前任务内的待复核提示**：宿主先按接收者、仓库改动需求、当前可执行性与 scope 过滤，最多给后续写核 LLM 两条候选及 advisory review 摘要；后续 LLM 自行 adopt/reject/unknown 并在 NOTES 留 `tip-review`，已处置或已有 overlay 的 key 不重复送。它不激活 overlay、不升级证据、不写长期经验，也不额外调用 reviewer。`peer_tips` 与 NOTES 仍是不经 never-auto 的常驻注入面，不能因为它们出现在 TASK 里就当成已通过上述路由。**单来源 review 只作 advisory**：2026-11-03 起 `settle` 不再生成新的 `reviewed_active`，`harness_review=false`，默认不自动调用 reviewer；历史行只作证据。写核围栏由 hmz 提供（Landlock/Seatbelt，架不起围栏会拒绝启动），但本仓库只断言声明，未做 OS 级围栏验证，也未记录实际 hmz 版本。

### 永远禁止自动 patch 的路径

`taic/code_agent.py::SECURITY_CRITICAL` 现在拒绝：

- `tests/`
- `tcenv/evaluate.py`、`tcenv/schema.py`、`tcenv/bench.py`、`tcenv/backends/`、`tcenv/gpulock.py`
- `tcenv/task.py`、`tcenv/distributions.py`、`tcenv/taskcontract/`、`tcenv/opspec/`（v1/v2 的任务语义、输入分布、容差与基线同样是评分材料；2026-11-03 补入）
- `taic/prompts.py`、`taic/kernel_agent.py`、`taic/harness.py`、`taic/review.py`、`taic/trust.py`、`taic/code_agent.py`、`taic/acceptance.py`、`taic/acceptance_checks.py`、`taic/orchestrator.py`、`taic/policy.py`、`taic/control.py`、`taic/llm.py`、`taic/artifacts.py`、`taic/harness_experiment.py`、`taic/resume.py`、`taic/committees.py`、`taic/cli.py`、`taic/flows/`
- `experiments/common.py`、`experiments/harness_runner.py`、`experiments/harness_cell.py`
- `observe/`、`targets/`、`backends/`、`.pi/`、`.git/`、`.github/`
- `conftest.py`、`sitecustomize.py`、`pytest.ini`、`tox.ini`、`setup.cfg`
- `taic.toml`、`taic.paper.toml`、`control.json`、`pyproject.toml`、`uv.lock`

这些文件如果确实要改，应走单独的人工安全 review，而不是 P2 自动 patch。另：`forbidden_path()` 还拒绝 `tasks/`、仓库外 `..`、凭据/私钥文件（`.env`、`credentials*`、`id_rsa*`、`*.pem`、`*.key` 等），这些不在 `SECURITY_CRITICAL` 常量里，但同样不能经 P2 自动改。

补丁的 `allowed_paths` 现在必须非空，并覆盖补丁触及的每个路径。没有验收契约、或契约里允许列表为空的旧 patch，`apply-code` 会拒绝落地；不再用空列表表示“不限制”。

### 仍存在的风险

- code agent 补丁未触及上述路径时，仍可能改变行为；必须看 diff，不能只看 tests。
- 没有非空 `allowed_paths` 的 patch 会被拒绝；新提案必须显式列出允许路径。
- 动态 `verify-code --execute` 在隔离 Docker 中执行补丁代码；`apply-code --run-tests` 仍在 host 的 worktree 中执行。worktree 隔离不是安全沙箱；两者都必须先人工看 diff。
- `code_patch` 生成阶段仍使用 `LocalEnv`，属于受信开发路径；“未审阅 patch 不经过 `verify-code --execute` 执行”不等于生成阶段零执行。
- `--run-tests` 使用补丁后的工作树，测试被禁改，但仍应检查测试命令和覆盖范围。
- 人和 reviewer 都可能被提案文本社会工程；不要照着建议里给的命令直接执行。

## 远程评测与自测

远程评测把“写核”和“判卷”分在两个信任域：`pi` 在 control_host，受信评测服务在设备侧。边界靠结构而不是物理距离：

- `pi` 只能提交候选和结构化自测请求；control_host 到 evaluator 只走 gRPC/TLS/Protobuf，任何 shell、`docker exec`、任意路径或报告改写入口都不在合约内。旧 SSH 数据面已删除。
- `hmz`/`pi` 不持有设备锁、评测代码、任务参考实现或报告目录的写权限；生产 worker 每次 attempt 使用一个一次性容器。formal mTLS 私钥、formal grant 和 HMAC verifier keyring 由 fence 外的 host evaluator/client 持有，同 UID 的 0600 不能单独当作能力边界。
- formal 与 selfcheck 使用独立 grant、模式白名单、配额和审计，每个 RPC 都重新授权；agent 侧 bridge token 只授权 host 侧 selfcheck，不能冒用 formal。
- 每个 device 只有一个 single-active frontend 和一个持久 `JobEngine`；formal 优先并按 run/session aging，旧 worker/fence 未验证清理前不启动新 attempt。
- 报告带 provenance（site/device/backend/build/task pack/calibration/协议）与 `hmac-sha256-v1`；可发布结果必须由 host keyring 验签，且 `measurement_kind=measured`、`publication_allowed=true`、site/epoch 在 allowlist 内。模拟结果永远不进入候选或发布。
- split-control 自测经每场一个 host-owned Unix socket，短期 token 文件在 host-owned 的 `${XDG_CACHE_HOME:-~/.cache}/taic-selfcheck/`（0600），通过会话环境注入而不进入 argv/TASK/workspace；配额由 host 与服务端各记一份。timed 结果会写 `.selfcheck/self_checks.jsonl` 和 host 权威日志，可参与 `premise`；quick 和基础设施错误不写。仍缺 `tool_probe` 执行证据。
- 同机不同账号不是强隔离：pi 若有 sudo/root、宿主管理接口或评测目录写权限，受信域即失效。候选 kernel 仍可能攻击驱动；timeout/crash 后应先跑轻量 known-good 健康探针，失败才隔离设备，不直接恢复正式计时。
- split-control 启用时必须登记数据接收方、可读者、保留/删除时间及训练或其他二次用途；发给第三方前需要人的明确授权。日志不记完整源码或凭据，临时源码按 job 清理，回传 evidence 只保留验收所需最小内容。

完整协议、outbox 和字段见 [架构/执行运行时与设备扩展.md](架构/执行运行时与设备扩展.md)。

## 可选远端 workspace（Phase 3，transport-only）

默认仍是 `workspace=local`。显式传 `--remote-workspace-config` 后，写核会话的文件与命令放在一台预置 SSH 主机上，但**这首先只是 transport**，不是安全隔离：

- 配置只写 hmz 已保存的 ssh runtime 名、远端根目录和期望的 `harness`；主机/用户/端口/identity 文件留在 hmz 的 `settings.yaml`，不进本项目配置、manifest、TASK 或日志。启动前校验 runtime 的 `harness` 与配置一致，不一致就拒绝，不静默改变代理位置。
- 输入与回收都按白名单 manifest：拒绝绝对路径、`..`、符号链接、设备文件、超大文件和总量超限；host 侧只把回收文件写进 `agent_work/<tag>`，并再次校验目标没有逃出该目录。
- 远端只执行 transport 自己的固定命令；agent 不能通过它运行任意 shell、argv 或环境变量。会话目录先写 receipt（绑定 run/session/runtime/workdir/manifest/prompt hash），同 receipt 幂等，不同 receipt 的目录不复用也不删除；清理前重读 receipt。
- 本阶段远端 workspace **没有**自测通道：TASK 使用独立 profile，不打印 control_host 路径，也不提供 `tcenv eval`/`taic-selfcheck`；runner 拒绝 host bridge 绑定。自测仍属于 Phase 2 的 `EvaluationExecutor`。
- 费用按 `began.workspace + envs/used + 开始时间` 绑定 hmz epic；绑定不上时记 `cost_incomplete`，不借别的会话账单。
- **未证明**：hmz 围栏能否阻止内循环读取/改写 control_host 文件。真机逃逸测试与真实 SSH 验收都还没跑（本机 VM 可连通但 SSH 私钥缺失），所以不能把这条路径写成“隔离”或“不可信 P2 的默认环境”。
- 远端 workspace 不持有 provider 凭据：凭据仍只在 control_host 的 pi/hmz 一侧；也不把 `~/.pi`、ssh agent、Docker socket 或仓库挂进远端工作目录。


## 可选隔离 agent（Phase 4，离线/模拟）

只有 `workspace.harness=env` 且显式配置 `isolation` 时，agent CLI 才离开 control_host。这一层同样是可选的，默认不启用：

- **短期凭据**：control_host 在内存里 mint 绑定 run/session/runtime 的随机 token（TTL、使用上限），只保存摘要；token 不进 TASK、workspace、manifest、argv 或日志，`repr` 也隐藏。成功、超时、崩溃或无交卷都在 `finally` 里 revoke，并记录清理证据。
- **凭据 seam**：`loopback-fake` 只在 `127.0.0.1` 提供确定性 provider，必须显式 `simulated=true`，并记录 `provisioned_to_agent=false`：它只交付给同进程测试替身，不证明真实远端 agent 拿到凭据。`external-helper` 由 operator 提供，token 经 stdin 而不是 argv 传入，provision/revoke 由它负责，记录 `provisioned_to_agent=true`；helper 在远端是否真的只写短期凭据是 **operator-attested**，本项目不能证明。
- **受信评测围栏**：启动时拒绝任何让 `agent_work` 与 evaluator 候选目录、outbox、manifest/ledger、`tcenv`/`taic`/`tests` 或 selfcheck authority 目录发生包含关系的配置；回收写入仍限制在 `agent_work/<tag>`。它是布局检查，不是沙箱，不能替代远端账号/权限和 hmz 围栏。
- **放置证据**：`harness=env` 要求 epic 的 `opened.harness` 恰为 `env`，缺失也按不一致处理；`harness=local` 接受 `local` 或空。不一致时该场 `cost_usd=null`、`cost_incomplete=true`，并记 `harness_mismatch`，不把跑错位置的会话账单算进搜索。
- **失败与孤儿**：凭据 mint/provision 失败是 `credential_unavailable`，不会先启动 agent 再补账；crash/timeout/no-submission 走 typed 失败并撤销凭据；run 结束按 run token 只清扫本 run 的非活跃会话目录。
- **Docker 隔离模拟已通过（2026-10-26）**：真实 SSH transport + `harness=env` 在 Ubuntu 24.04 容器内运行确定性 fake agent，host 侧 helper 经 `docker exec` 交付短期 token，agent 调 control-host fake provider；13 项必需 check 全过，报告为 `runs/remote-agent-sim-20261026/report.json`（`simulated=true`、`publication_allowed=false`，不进入搜索/发布）。
- **仍未证明**：真实 pi CLI 与真实 provider、真实 Ubuntu VM、生产 `online=NONE` 围栏（模拟 fixture 放宽为 `ALL`）、远端进程/围栏逃逸和真实 orphan 清理。本机 VM 可达但缺 SSH 私钥，这些验收是 blocked。

## 操作规则

- 普通搜索不重跑 `taic run`/`taic resume` 以外的 GPU。
- 修改 trust、budget、control、测试、评测或 prompt 必须单独人工确认。
- 任何 commit 前先看提交正文和 `git diff --cached`。
- 不要把 run 生成的脚本、命令、URL 或路径直接执行。
