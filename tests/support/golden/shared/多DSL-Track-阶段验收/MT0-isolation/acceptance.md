# MT0-isolation：hmz fence 适配与静态合同验证

## 1. 范围结论

**本轮只完成“复用 hmz fence 的适配 + 静态合同与真实 CPU shim/watchdog 验证（不启动 fence）”。MT0 整体仍为 `blocked`；真实攻击证据为 `pending_outer_probe`；嵌入式 DSL 仍不能标为生产可用，MT1b/MT2–MT7 未开工。**

用户明确保留嵌入式 DSL 的表达自由，并决定复用 hmz 的 macOS Seatbelt/fence，而不是维护第二套隔离实现。本轮据此替换了上一轮被 `/flow stop` 中断、从未验收的草稿：

- 删除草稿 `tcenv/worker_isolation/macos_seatbelt.py` 中的自写 `(deny default)` renderer、`sandbox-exec` probe 和 exact-profile preflight；
- 新增 `tcenv/worker_isolation/hmz_fence.py`，只使用源码审计过的 `hmz internal fence --policy=@FILE -- ...` 启动协议；
- 保留并重述 `child.py` 的 ShapeForge 责任：它在 hmz fence 内应用必需 rlimits、明确记录 unsupported/not-configured 项，通过控制 FD 回报后再 `exec` 受控 Python/Clang frontend；
- 没有 import 另一个 uv-tool Python 环境的 `hmz.*` 内部模块，没有直接执行 `sandbox-exec`，没有启动真实 hmz/fence/攻击探针，也没有无隔离 fallback。

历史 `MT0`、`MT1a`、`MT5` 包未改写；`verify.py::verify_history` 逐字节固定其封闭锚点。`start-state.json` / `start-worktree.sha256` 保留上一轮开始时的原始记录，本文件是新的决策与验收范围，不回填旧 MT0 blocked 结论。

## 2. hmz 协议与责任分界

### 2.1 只读源码审计事实

本机发现的可执行文件为 `/Users/kuafu/.local/bin/hmz`，解析到 uv tool 中的 hmz **0.2.2**。`hmz-installation.json` 固定 executable、site-packages、协议 ID、协议摘要及以下七份源码的 SHA-256：

- `hmz/cli/__init__.py`（internal 命令路由）；
- `hmz/cli/fence.py`；
- `hmz/coganchor/fence/__init__.py`；
- `hmz/coganchor/fence/proxy.py`（空 host allowlist 的网络语义）；
- `hmz/coganchor/fence/wrap.py`；
- `hmz/coganchor/darwin/seatbelt.py`；
- `hmz/coganchor/serve/fencing.py`。

由这些源码确定的受控接口是：

```text
hmz internal fence --policy=@ATTEMPT/hmz-policy.json --
  PYTHON -m tcenv.worker_isolation.child LIMITS... -- FRONTEND ARGS...
```

policy 是 `Fence.dumps()/loads()` 对应的 `read/write/online/hosts/tmp/scopes/listen` 字段，不是 ShapeForge 自造的 Seatbelt profile。适配层在启动前检查固定版本与上述源码字节；缺 hmz、uv-tool 布局不明、版本漂移、协议源码漂移、缺 Seatbelt 均 typed `backend_capability_unavailable`。返回 126 只有 stderr 含 `hmz internal fence:` 才按 hmz 能力失败归类，frontend 自己 126 是 `process_failed`；该前缀分类可以被伪造，但无论哪一类都不会被接受为成功。

`seatbelt.available()` 的源码也说明：进程已经处于 sandbox 内时不能再次 `sandbox_apply`。上一轮 `Operation not permitted` 只证明当前 RLAR/工具上下文不能嵌套应用 Seatbelt；它不证明 hmz 没有隔离，也不证明外层控制机已经通过隔离验收。真实 probe 必须由 sandbox 外的控制机单独执行。

### 2.2 ShapeForge 保留的 attempt 合同

| 责任 | 固定行为 |
| --- | --- |
| 环境 | 从空显式 allowlist 构造，不继承宿主环境；secret-shaped 名称、凭据文件变量、代理覆盖和凭据形态的值均拒绝 |
| HOME/cache/TMP | 每次 attempt 新建，全部位于 `attempt/build`；hmz 离线 proxy 变量只由 hmz 添加 |
| 读集 | 声明的输入、repo/frontend/Python/Clang/SDK toolchain 只读；不把宿主 HOME 整体加入读集 |
| 写集 | policy 中持久写根仅为 `attempt/build` 下的 work/home/cache/config/data/tmp/pycache/artifacts；另有非持久 OS device nodes |
| 网络 | `online=false`、`hosts=[]`、`listen=[]`；不把 proxy/loopback 的存在写成所有网络/本地 IPC 攻击已经通过 |
| 子进程 | 受控 rlimit shim、Python/Clang frontend 及其后代都在同一 hmz wrapper 下，不粗暴禁止 fork/exec |
| 资源 | 默认 wall/CPU 均 60 秒；CPU/FSIZE/NOFILE/CORE 为必需项，缺失/失败由 shim 125 + 可信报告拒绝。Darwin AS 只记 unsupported、不声称内存上限；NPROC 默认不设，因为它是 per-UID 而非 per-attempt。ShapeForge 外层 wall timeout 与真实进程组回收；尚无外层 RSS 上限 |
| artifact | 精确声明集合，POSIX 相对路径，拒绝逃逸、symlink、hardlink、特殊文件、缺失/额外文件、单文件/总量超限，并记录大小与 SHA-256 |
| 证据 | hmz/source/policy/spec/artifact identity 由 trusted worker 记录；候选自报不构成隔离证据 |

### 2.3 当前 hmz 0.2.2 的额外阻断

只读审计发现 `wrap._seatbelted()` 会在调用方 policy 之外自动把：

```text
os.confstr(_CS_DARWIN_USER_CACHE_DIR) / mds
```

加入 Seatbelt 写根。本机对应 `/private/var/folders/p0/mvdp0k2527g1xv_s_0_tkbc80000gn/C/mds`。这不是 `attempt/build` 下的路径，且候选继承该权限，因此不能用“ShapeForge 写出的 policy 没列它”来宣称满足严格写集。

所以当前默认生产入口会在这条兼容性检查上 **typed fail-closed，且不启动候选**。mock 正例使用没有额外隐式写根的兼容 installation 替身，只证明 argv/policy/watchdog/artifact 适配逻辑，不证明本机 hmz 已可运行。

owner 后续必须明确二选一，ShapeForge 不得自行放宽：

1. **上游 hmz 修正协议**：让 `C/mds` 可关闭，或允许配置到本次 `attempt/build` 内；ShapeForge 再固定新版本/源码摘要并执行外层 probe。
2. **owner 显式批准合同例外**：逐项记录例外路径、可读写字节范围、持久性、凭据风险和到期/复审条件，形成新的明确合同后再执行外层 probe。本轮没有该批准，当前严格写集和 fail-closed 保持不变。

hmz 的 `online=false` 还会启动一个空 host allowlist 的 loopback proxy；其源码同时明确记录 Unix socket/Mach service 等本地 IPC 边界。它们都保留在外层正负例中，不能把 `online=false` 简化成数学意义上的“所有 socket 均不存在”。

## 3. 静态合同与真实 CPU shim/watchdog 验证（无 fence）

固定节点完整列于 `nodes.json`：

- `tests/shared/test_worker_isolation_hmz.py`：44 个节点，其中 4 个直接执行真实 CPU shim 或真实 `popen_group+supervise`；
- `tests/shared/test_mt0_isolation.py`：9 个节点；
- 合计 **53 passed，0 skip/xfail**（最终由 `verify.py --check` 重跑并核对精确数量）。

实际命令、既有 MT0/MT1a/MT5 复验数量、结构测试与 scoped source SHA-256 另记录在 `verification.json`；该文件不把其他阶段的 CPU/fixture 通过扩大为真实 Metal 或隔离证据。

覆盖：

1. hmz argv、`@policy` 文件、读写根、离线字段、临时 HOME/cache；
2. 宿主 secret 不继承，显式 secret/凭据/代理注入拒绝；
3. 外部路径、`..`/绝对路径/反斜杠、symlink、hardlink、额外/缺失/超限 artifact；
4. 正例 artifact manifest 的路径、大小、SHA 和输入篡改拒绝；
5. mock timeout/kill/forget 状态映射，以及 frontend 自己 125/126 与 hmz 能力失败的区分；
6. **真实 shim `/bin/echo ok` 成功**，frontend 直接读到 CPU=30、FSIZE=64 MiB、NOFILE=128，且能再创建打印 `grandchild-ok` 的孙进程；
7. 显式请求 Darwin AS=2 GiB 时仅记录 unsupported、不阻断 frontend；必需 NOFILE 无法应用时真实退出 125，`SHOULD_NOT_PRINT` 未执行；
8. **真实 `popen_group+supervise`** 在 0.2 秒 timeout 后杀死普通后代，孙进程的延迟 sentinel 未出现；
9. 缺 hmz、版本漂移、协议源码漂移、Seatbelt 缺失、隐式写根不兼容均 fail-closed；
10. 受控 Clang frontend 参数保留，没有第二套 renderer 或 hmz Python import；
11. verifier 拒绝 production-ready、攻击通过、MT0 完成、无隔离 fallback 和 MT1b 开工五种变异。

真实 CPU 测试只执行 shim/普通子进程，没有调用真实 hmz，也没有应用 Seatbelt。**mock 返回成功、CPU shim 成功、hmz 源码存在、`/usr/bin/sandbox-exec` 存在都不是隔离通过证据。**

复验命令（从仓库根运行）：

```bash
uv run python 'docs/架构/多DSL-Track-阶段验收/MT0-isolation/verify.py' --check
```

## 4. 外层真实 probe 仍为 pending

`outer-probe.json` 冻结了后续外层命令与正负例；`outer_probe.py` 是有显式 `outside-seatbelt` ack 的入口，本轮未执行，也没有 `outer-probe-result.json`。后续只能在 sandbox 外的控制机单独授权运行：

```bash
uv run python 'docs/架构/多DSL-Track-阶段验收/MT0-isolation/outer_probe.py' \
  --outer-control-ack outside-seatbelt \
  --output 'docs/架构/多DSL-Track-阶段验收/MT0-isolation/outer-probe-result.json'
```

计划包括：人工 outside read/write canary、真实 mds 写根 canary、outer-owned loopback listener、宿主环境 canary、Python/Clang 子进程、普通 timeout 进程组、`setsid` 脱离进程组、artifact symlink，以及缺能力/版本漂移负例。所有攻击目标都是临时人工文件或 outer-owned listener，不使用真实凭据、外部网络、GPU/NPU 或模型。`setsid` 后代由外层清理。

当前 hmz 的隐式写根兼容性门若仍拒绝，runner 只能输出 `backend_capability_unavailable` / `pending_compatible_hmz_and_outer_probe`，不能将其写成攻击通过。即使以后得到 `outer_probe_observed`，也只是待独立复核的 raw outer observation，不会自动解除 MT0 或开放生产 Track。

## 5. 未完成事项

- 真实文件写集、网络、凭据、普通/脱离进程组与 artifact 攻击证据：**pending_outer_probe**；
- 当前 hmz 0.2.2 隐式 `mds` 写根与严格 attempt 写集的兼容性：**blocked / backend_capability_unavailable**；owner 必须按 §2.3 明确选择“上游可关闭/可配置”或“显式批准记录在案的合同例外”，在此之前不得运行生产候选或外层攻击 probe；
- TileLang/Halide frontend 安装、编译、真实 Metal、性能、模型/付费搜索：未授权、未执行；
- MT0 整体、MT1b、MT2–MT7：未通过/未开工；MT1a 与 MT5 既有各自 CPU/fixture 范围不被本轮扩大。

本包没有自签 reviewer 文件。最终 reviewer 的 `done=true` 只能表示上述 **hmz 复用适配、静态合同与真实 CPU shim/watchdog 验证** 范围完成，不能表示真实 Seatbelt 攻击证据完成或嵌入式 DSL 已可用于生产。
