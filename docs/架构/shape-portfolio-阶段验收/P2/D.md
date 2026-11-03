# P2-D：portfolio sidecar、代次地板、恢复与失效账本

状态：**completed（独立 reviewer ACCEPT；身份校验与 10 个恢复节点通过）**。

## 范围

- `portfolio_run.json` 绑定 freeze、ValidationPlan、run_config digest、DispatchManifest digest、generation、
  subjects 与 `invalidated_artifacts`；不改 schema-3 `run_config.json`；
- `taic/shape_portfolio.py` 提供专用 `write_checkpoint / resume_checkpoint / revoke_checkpoint`；
- `taic/resume.py::inspect_run` 对 sidecar 返回 `resumable=false, kind=portfolio_run`，`load_snapshot`
  明确 `ResumeError`；`cli.prepare_resume` 转为 `ResumeRefused(code=3)`，旧 schema 1–3 runs 不变；
- `observe/shape_portfolio.py` 显式读取 per-shard provider/credit/coverage/invalidation，不做 SQLite migration；
- `ActivePortfolioPointer.read` 支持 `minimum_generation`，持久化 revoked sidecar 不能恢复执行。

## 恢复语义

- checkpoint 分别原子写 `portfolio_run.json` 与 `active_portfolio_pointer.json`（不是成对事务）；
  已有 revoked sidecar 时拒绝再次 checkpoint，防止**通过公开 checkpoint 路径**回滚代次和清空失效账本；
- resume 重新校验 freeze/plan/run_config digest、manifest digest/generation/status 以及 subjects 是否逐一对应当前
  manifest，返回已完成的 instance_keys，调用方不得重复 fresh batch；
- revoked sidecar 带 `invalidated_artifacts: [{activation_generation, reason}]`，旧 pointer 只作 stale 证据；
  active sidecar 不得携带失效记录；revoke 先原子写持久 `portfolio_revocation.json` 墓碑，再写 revoked
  sidecar，resume/write 遇到墓碑均拒绝，observe 以墓碑为准，因此只回写旧 active sidecar 不能复活运行；
- 文档-only manifest 不能激活；恢复必须回到 sidecar 重放，不能只信 manifest+pointer。

## 固定 CPU 证据

`tests/backends/metal/test_shape_portfolio_resume.py` 10 个节点：

1. checkpoint/resume 返回 14 个完成实例，observe 一致，旧 `inspect_run/load_snapshot/prepare_resume` 全部拒绝；
2. revoked sidecar 带 reason，resume 拒绝，observe 保留失效账本；
3. generation floor 与 freeze 漂移拒绝；
4. ghost/重复 subjects 与 run_config digest 漂移拒绝；
5. 文档-only 或部分覆盖 manifest 不得 checkpoint；
6. revoked 代次不得再次 checkpoint 或清空失效账本；
7. manifest 必须重新绑定调用方 freeze/plan，伪造 run 的 freeze/plan 不得 resume；
8. revoked manifest 在 pointer 防御之前即被拒绝；
9. active checkpoint 不得被同代次 run_config 接管，同身份幂等重写仍允许；
10. 持久撤销墓碑阻止回写旧 active sidecar，observe 保留 revoked 状态与失效原因。

默认验收还在临时目录实际执行一次：写 checkpoint → resume 完成集 → observe → revoke → resume 拒绝。
没有 GPU/model/search 参与这些检查。

## 尚未覆盖

- 多进程崩溃中途中断的完整故障注入与重放矩阵；
- 生产 master seed/HMAC 回执持有与远程设备证明；
- P3 性能治理与第二任务；
- 持久墓碑覆盖只删除/回写 `portfolio_run.json` 的攻击；若攻击者还能删除墓碑或回滚整个目录快照，
  仍需上层的持久撤销日志/权限边界，本阶段不声称解决整目录回滚。公开 API 只允许同一 active manifest
  与同一 `run_config_digest` 的幂等重写，revoked、代次回滚以及 manifest/run_config 变化均被拒绝；
- `write_checkpoint` 不重新计算 stable confirmation digest，进程内 `dataclasses.replace` 仍属于调用方信任边界；
- `minimum_generation` 仍是调用方传入值；持久墓碑是跨进程拒用证据，但不是带自摘要的追加式单调撤销日志；
- `PortfolioRun.read` 的 run_config digest 已改为必传，公开 resume 不再提供默认跳过路径。
