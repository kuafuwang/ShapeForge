# hmz OS 围栏与逃逸补验（2026-11-25）

## 结论与边界

本轮用真实 hmz **0.2.2**、真实 Seatbelt 和 Landlock/seccomp，以及人工 canary/loopback listener，补上了此前只有静态/CPU mock 的缺口：

- **Seatbelt：18/18 检查通过**，见 [seatbelt-result.json](seatbelt-result.json)。
- **Landlock：16/16 实质性检查通过**，见 [landlock-result.json](landlock-result.json)。

这些结果证明**所测攻击面**，不是所有本地 IPC 或所有可能逃逸都被消除，也不是生产完成或 speedup 发布。

### 已实测

- 严格 attempt profile 下，区外读取、区外写入均拒绝；允许写和负例控制同时成立。
- 宿主 canary 环境变量不进入严格 attempt，HOME/TMPDIR 位于 attempt 内。
- 非代理 loopback、UDP 均拒绝；空 hosts 的代理 CONNECT 返回 403，没有上游连接。
- Python 子进程继承同一围栏；普通子进程由 outer process-group kill 清理（Landlock 不冒称生产 watchdog timeout）；Seatbelt 的真实 `HmzFenceExecutor` timeout 通过；artifact symlink 在 Seatbelt 路径被拒绝。
- Seatbelt 的 `start_new_session` 后代区外文件写入被拒；Landlock double-fork daemon 的区外文件写入、非代理 TCP 和 UDP 均被拒。两类 detached 后代都由外层按记录 PGID SIGKILL，探针结束后无存活进程或 canary 目录。这里说 outer kill，不把 zombie 判死称作生产自动 wait/reap。
- 实际 hmz 版本、Python、Landlock ABI 4、seccomp user notification、pidfd supervision 和 hmz 源码摘要均记录；VM 上八份 hmz 源码与生产 pin 一致，实际使用的 `landlock.py`、`seccomp.py`、`loopback.py` 也绑定在执行结果中，见 [installation.json](installation.json)。区外读取负例先在外层真实读取成功，见 [seatbelt-negative-control.json](seatbelt-negative-control.json) 和 Landlock 的 `negative_controls`。

### 不能据此宣称的事项

1. **`kernel_once` 不是 control-host 不可读模型。** 它的实际权限是 `local=ALL,user=READ,system=READ,online=NONE`，允许读取仓库/历史/control-host。严格 DSL attempt profile 才是这里验证的窄边界，见 [kernel-once-permission.json](kernel-once-permission.json)。
2. **detached 自动回收仍未闭环。** `detached_process_group_gap_closed=false`。当前证据是 detached 后代存活但被外层清理，不是围栏或生产 supervisor 自动回收全部 orphan。若要求生产自动闭环，仍需单独实现和验收。
3. **Unix socket/Mach service 等本地 IPC 不宣称全部消失。** `online=false` 不等于所有 loopback/本地服务不可达。
4. **不是生产发布。** `production_ready=false`、`publication_allowed=false`，无 GPU、NPU、付费模型或搜索。

## 环境与版本

### macOS Seatbelt

直接执行现有 MT0 探针，旧归档未改写：

```bash
uv run python tests/support/golden/shared/acceptance-epochs/fixtures/mt0-isolation-profile/outer_probe.py \
  --outer-control-ack outside-seatbelt \
  --output docs/可信harness/hmz-fence-2026-11-25/seatbelt-result.json
```

hmz 0.2.2、CPython 3.14.4，协议 `hmz-lower-level-seatbelt-profile-proxy-1`。

### Ubuntu Landlock

Ubuntu 22.04.3 ARM VM 的实际预检：

- Landlock ABI 4；
- seccomp `user_notif` 可用；
- Yama `ptrace_scope=1`；
- `os.pidfd_open` 与 `pidfd_getfd` 可用。

uv standalone Python 没有 `os.pidfd_open`，因此没有把网络围栏降级成文件系统围栏。改为在 VM 安装 build-essential/libssl-dev/zlib1g-dev/libffi-dev，并从 python.org 源码在隔离 `/tmp` prefix 构建 CPython 3.14.4。hmz 仍严格为 0.2.2，源码 hash 与生产 pin 一致。

执行：

```bash
PYTHONPATH=/tmp/shapeforge-hmz-linux/uv-tools/hmz/lib/python3.14/site-packages \
  /tmp/shapeforge-hmz-linux/cpython/bin/python3.14 \
  /tmp/shapeforge-hmz-linux/landlock_probe.py \
  --ack landlock --output /tmp/shapeforge-hmz-linux/landlock-result.json
```

探针源在 `tests/support/landlock_probe.py`。它显式要求 `--ack=landlock`，只用人工 canary 和 loopback，不使用真实凭据或外部网络。

## 清理

- detached 进程组已由 outer control 清理；double-fork daemon 不再存活。
- 历次 outside canary 目录已删除，进程表无探针残留；VM 已停止。
- VM 客户端目录、CA、token、grant 等远程 staging 凭据不存在于本证据目录。
- 没有新增生产代码，旧 MT0/MT1b 证据和已关闭验收包未改写。
- 独立只读复核已返回 `done=true`（[review.json](review.json)）；清理字段见 [cleanup.json](cleanup.json)。APT 编译依赖在 VM 中的安装单独记录，不冒称系统未改。
