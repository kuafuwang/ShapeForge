# MT0-isolation-profile：lower-level hmz Profile 静态验收与真实外层观测独立复核

## 1. 当前结论与严格范围

本包现在包含两个前后相继、语义不同的 epoch：

| epoch | 范围与结论 |
| --- | --- |
| `mt0-hmz-lower-level-profile-1` | 单一 lower-level hmz Profile 生产适配、静态/CPU mock、真实普通 shim/watchdog；该 epoch 已完成，其 `static-contract.json` 仍如实记录当时 `pending_outer_probe`，没有回填成真实攻击证据 |
| `mt0-hmz-lower-level-profile-outer-review-1` | 保留第一次真实外层 r1 失败与第二次 r2 observed，并对 raw、探针修正、lower-level Profile/空 host Proxy、negative controls、身份和残余边界做只读独立复核；该限定范围已完成，未重跑 probe |

**本次 `done=true` 的语义仅限第二行：真实 lower-level hmz Profile 外层隔离 observation 已独立复核，且本包 verifier 通过。**它不是正式 production pass，不表示 MT0、DSL 选型、frontend、MT1b、真实 Metal、性能或整个多 DSL 计划完成。

仍然固定：

- `mt0_status=blocked`；
- `production_ready=false`、`publication_allowed=false`；
- DSL 选型仍是 0 个 qualified candidate，TileLang/Halide 仍为 `interface_only`；
- frontend 未安装、未验收，MT1b 及依赖它的后续 DSL Track 阶段未完成；
- 真实 Metal DSL 构建执行与 fresh 性能证据未做。

本轮没有嵌套调用 hmz/hmz_flow/Agent/subagent，没有重跑攻击 probe，没有修改 hmz，没有接受 Xcode license，没有安装 frontend，没有运行 GPU/NPU/真实模型/付费搜索，也没有 Git 写操作。

## 2. 唯一 lower-level 路线与源码复核

最终路线仍是：

```text
ShapeForge trusted worker（sandbox 外，popen_group + supervise）
  → HMZ_UV_TOOL/bin/python -I hmz_profile_driver.py --request STRICT_REQUEST
      ├─ 验证 interpreter/layout/version/source/driver/shim identity
      ├─ Proxy(()) start，取得实际 127.0.0.1 port
      ├─ Profile(read=..., write=..., port=actual_port)
      └─ Profile.command(PROJECT_PYTHON -I child.py LIMITS -- FRONTEND_ARGS...)
           → sandbox-exec 应用 Profile 后才执行 trusted rlimit shim
                → exec Python/Clang frontend 及其后代
```

独立源码复核确认：

1. 受信 driver 只导入经身份固定的 `hmz.coganchor.fence.proxy.Proxy` 与 `hmz.coganchor.darwin.seatbelt.Profile`；项目 Python 不跨环境 import hmz。
2. 空 host `Proxy(())` 在任何候选进程之前启动，实际端口传给 `Profile(port=port)`；没有使用会留下网络不受限的 `port=None`。
3. `Profile.command()` 是唯一候选-bearing 启动点，trusted rlimit shim、Python/Clang 与后代继承同一个 Profile。
4. 不调用 `wrap._seatbelted()`。只读对照该 wrapper，确认它才追加 per-user `C/mds`；lower-level 路线没有这项写根，也没有扩大 Profile 写集。
5. hmz Profile 源码同时规范化授予路径及其 realpath；它在离线规则中只允许实际 proxy 的 TCP outbound，并保留 loopback bind、Unix-socket 与 Mach-service 文档边界。故 `online=false` 不能解释为“所有本地 IPC 数学上消失”。
6. 空 host Proxy 在 `passes()` 失败时先回 403，只有 allowlist 命中才会创建 upstream connection；r2 的人工 CONNECT 403 与这一源码顺序一致。

固定身份未漂移：hmz 0.2.2、hmz uv-tool Python 3.14.4、8 份 hmz 源码、entrypoint、driver、trusted shim、adapter 和协议
`hmz-lower-level-seatbelt-profile-proxy-1` / `c6a6a61052d8e7a19b30a9674b685b75378d8999791eff0ef1c6c570cb762c83`。
缺能力或身份漂移仍 typed `backend_capability_unavailable`，没有无隔离 fallback。

## 3. r1 与 r2 都保留，失败没有被洗掉

两份 raw JSON 的字节没有为本轮复核修改，均进入 manifest 和 verifier：

| raw | SHA-256 | 原始结论 | 必须保留的事实 |
| --- | --- | --- | --- |
| `outer-probe-result.json`（r1） | `a8c50f471d04ac3418cdf83c0441cdab4d2a781fae9197c607ad7c349f0cf88d` | `outer_probe_failed` | `real_os_fence_executed=true`；18 个 checks 中 16 true、2 false；失败恰为 `clang_child_usable` 与 `temporary_home`；Clang returncode=69；没有 `clang_stdout`/`clang_stderr` 字段 |
| `outer-probe-result-r2.json`（r2） | `69f21a5da32a4051396adca94702195533e716bf071c1d117e5da53b0d164229` | `outer_probe_observed` | `real_os_fence_executed=true`；同一精确 18 键集合全部 true；Clang returncode=0；保存 Apple Clang 21.0.0 stdout 与空 stderr；仍要求独立复核 |

**计数以 raw 为准：两份 `checks` 都实际有 18 项，不是早期文字所述的 17 项。**本包不会删除任一键、把 r2 改成 17 项或修改 raw 来迎合摘要。`outer-probe-review.json` 和固定测试明确记录这一差异。

r1 没保存 Clang stderr，不能补写或虚构。主会话另行诊断时观察到 xcrun_db/license 文本，只用于解释探针修正，不冒充 r1 JSON 字段。当前复核也不重新运行 compiler 或 probe。

## 4. 仅限两项探针修正

r2 对应的 `outer_probe.py` SHA-256 为
`5a7f63ea24d06c1b4e17d114ce70a8836ea0b426ee4106fe15f0f72bdf365ef6`。复核的修正只有：

1. **probe root resolve**：临时目录先 `Path(directory).resolve()`，再做 HOME 的 `is_relative_to` 比较，消除 `/var` 与 `/private/var` 别名导致的假失败。r1 的 `temporary_home=false` 仍原样保留；独立测试只用修正后的纯路径比较重算路径关系，不改写历史布尔值。
2. **真实 compiler path**：通过 `xcrun -n -f clang` 取得直接 compiler path，经 `PROBE_CLANG` 传入，并在 r2 保存 child 的 stdout/stderr。r2 实测 `Apple clang version 21.0.0 (clang-2100.3.34.2)`、returncode=0、stderr 为空。

没有新增 Profile 写根、没有改成 `/usr/bin/clang` 包装路径、没有修改 hmz、没有接受 license。`outer-probe.json` 中原计划仍属于静态 epoch；其中 compiler 预期文字已校正为上述真实路径机制。

## 5. 对 r2 的 18 项逐键复核

Verifier 固定精确键集、键值类型、对应 raw observation 和 negative controls，不是只计算 `all(checks.values())`。

| check | r2 raw/源码对应的限定结论 |
| --- | --- |
| `positive_artifact` | 主 probe 返回成功，声明的普通 `report.json` 能由受信收集器接受 |
| `temporary_home` | HOME 位于 resolve 后的 `main/attempt/build/home`；宿主环境没有成为 HOME |
| `host_environment_not_inherited` | 人工 `SHAPEFORGE_OUTER_PROBE_CANARY` 在候选中为 null |
| `outside_read_denied` | 人工 outside secret read 为 `PermissionError` |
| `outside_write_denied` | 人工 outside file write 为 `PermissionError` |
| `mds_write_denied` | 真实 per-user `C/mds` 下唯一人工 canary write 为 `PermissionError` |
| `mds_parent_negative_control` | 外层已在同一 mds 父目录执行 write/unlink 预检，拒绝不是因父目录天然不可写 |
| `non_proxy_loopback_denied` | 连接 outer-owned 非 proxy 127.0.0.1 listener 为 `PermissionError` |
| `empty_host_proxy_refused_upstream` | 候选能连实际 Proxy，但人工 CONNECT 得到 403；空 host allowlist 不建立 upstream |
| `actual_proxy_port_matches_evidence` | 候选观察端口、driver proxy 端口和 Profile 端口一致 |
| `python_child_still_fenced` | Python child 试图 outside write 后以约定 13 退出 |
| `clang_child_usable` | 直接 xcrun 解析的 Clang 在同一 Profile 内 returncode=0，stdout/stderr 已保存 |
| `single_lower_level_driver_identity` | 单一 lower-level 协议、隔离 driver argv、无 internal wrap、无隐式 mds 写根 |
| `ordinary_child_reaped` | 普通 timeout 后代被进程组回收，attempt-local 延迟 sentinel 未出现 |
| `setsid_outside_write_denied` | `start_new_session=True` 的后代仍不能 outside write |
| `setsid_fence_inherited` | detached child 自记 `denied`，证明上项不是单纯漏执行；其 attempt-local outcome 可读 |
| `setsid_process_group_gap_explicitly_measured` | detached child 在普通进程组回收后仍写出 alive，**明确证明脱离进程组缺口存在** |
| `artifact_symlink_rejected` | 指向 outside 的声明 artifact symlink 被 trusted collector 拒绝 |

五个 negative-control 字段也按精确字典校验：人工 secret 可读、outside 目录可写、mds 父目录可写、outer loopback listener 已 bind/accept 均为 true；`real_credentials_or_external_network_used=false`。探针只使用临时人工目标和 outer-owned listener。

## 6. 不可消去的残余边界

真实 r2 observed 不扩大为下列结论：

- **不宣称所有 loopback/Unix socket/Mach service 或所有本地 IPC 消失。**非 proxy loopback deny 和空 Proxy 403 只覆盖本次实际测量的路径；hmz 的 Unix-socket/Mach-service 文档边界继续保留。
- **普通后代回收不等于 detached 后代回收。**`setsid` 后代继承 Profile 并拒绝 tested outside write，但确实脱离 ShapeForge-owned process group；outer runner 的 `killpg` 清理仍是必要边界。
- 本包没有建立外层 RSS memory cap，也没有宣称完整资源隔离。
- 本包没有证明任意 compiler/kernel/driver 漏洞不可能被利用，只复核所记录的人工攻击面。
- 上述隔离 observation 不替代 DSL 选型、frontend 构建、MT1b、Metal correctness/performance 或正式发布证据。

## 7. 静态/普通 CPU 证据与新固定测试

原静态 epoch 的证据不回填：4 个真实普通进程节点仍是 `/bin/echo`、真实 rlimit+grandchild、必需 NOFILE 失败不执行 frontend、普通进程组 timeout 回收；它们自身仍不是 Seatbelt 攻击证据。

新 epoch 的 `nodes.json` 固定 **100 个节点，0 skip/xfail**：

- `tests/shared/test_worker_isolation_hmz.py`：**55** 个适配、静态、mock 与普通 CPU 节点；
- `tests/shared/test_mt0_isolation_profile.py`：**45** 个静态边界、raw/独立复核与 verifier 变异节点。

新增固定覆盖包括：r1 两个失败及缺失诊断字段不被隐藏、r2 精确 18 键、`/var`/`/private/var` resolve 比较、xcrun 真实 Clang path 与 r2 streams、mds/outside/non-proxy deny、精确 negative controls、IPC 残余、`setsid` 缺口、MT0/DSL/frontend/Metal 不得完成，以及 raw/review/source/digest/status 的拒绝变异。

Verifier 还逐项重算 raw SHA、probe/adapter/driver/shim/hmz 身份、protocol digest、历史封闭包、精确 manifest 文件集和自身变异负例。真实 outer observation 只从保存的 raw 文件读取；`--check` 不调用 `outer_probe.py` 的 `main()`。

## 8. 复核与回归命令

从仓库根运行：

```bash
uv run python 'docs/架构/多DSL-Track-阶段验收/MT0-isolation-profile/verify.py' --check
uv run pytest -q -n 0 -m 'not gpu and not llm and not slow' \
  tests/shared/test_failure_codes.py tests/shared/test_structure.py \
  tests/shared/test_mt0_isolation.py tests/shared/test_mt0_dsl_selection.py
```

`verification.json` 记录本 epoch 的实际节点计数、只读复核边界、源码摘要和回归结果。失败必须按 `logs/pytest/triage/triage_summary.md` 分类处理，不 skip/xfail、不删 raw、不弱化断言。

## 9. 历史纪律与 reviewer 结论边界

- 旧 `MT0-isolation` internal-fence 包全部 11 个文件及 MT0/MT1a/MT5 封闭锚点由 `verify_history()` 固定，逐字节未改。
- R0–R6/G0/A1–A6/P0–P2 未改；六 RPC/field number/HMAC/BlobChannel 未改。
- 没有新增 pattern/tiling registry，没有候选算法或调度限制。
- `outer-probe-result.json` 与 `outer-probe-result-r2.json` 都是证据，不允许只保留后者。
- `outer-probe-review.json` 的 `review_done=true` 与 reviewer 最终 `done=true` 只覆盖本文件第 1 节所述的限定范围。任何 MT0、frontend、生产 DSL、Metal、性能或整个计划完成的解释都属于证据越界。
