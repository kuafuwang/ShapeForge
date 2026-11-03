# P2-E：跨阶段接口、信任根与证据链最终验收

状态：**completed（RLAR 最终独立 reviewer `done=true`，主会话复验通过）**。本文记录第二轮实际发现、固定负例与修复；不进入 P3，也不发布原始计时或 speedup。

## 1. 旧状态默认验收

第一轮完整运行：

```bash
uv run python -u docs/架构/shape-portfolio-阶段验收/P2/verify.py
```

旧状态退出 0：36 个 P2 CPU 节点 + 5 个真实 Mac 节点通过，14/14 active manifest
`activation_generation=1`，checkpoint/resume/observe/revoke 生命周期与 live==archived stable identity
均通过。该结果只证明旧验收面，不能覆盖下述生产门绕行。

## 2. 独立发现与先红负例

第二轮先只新增 5 个固定负例，再对旧生产代码串行运行，实测 **5 failed**，分别复现：

1. **跨 pack 一致重钉激活**：`activation_expectation` 未绑定 freeze 的 task pack。复制 fixture A，
   只向 `reference.py` 追加注释得到 pack B，再用 locator B 重建 14 份 expectations，协同替换 sidecar
   expected/两份报告身份并重算 R5 identity 与 confirmation digest 后，freeze A/plan A 可激活 14 subjects。
   新负例：`test_p2_b_activation_expectation_rejects_a_consistent_cross_pack_repin`。
2. **非法确认计数**：真实 confirmed sidecar 的 `confirmation_attempts` 改为 `True`、`1.5`、`3.0`
   并重算 digest，旧 `verify_subject_document` 仍接受 confirmed 语义。
   新负例：`test_p2_b_confirmation_attempts_require_a_strict_bounded_integer`。
3. **document-only active pointer**：`DispatchManifest.from_document` 得到的 active manifest
   （`_derived_from_confirmations=False`）与匹配 pointer 可直接通过生产 reader。
   新负例：`test_p2_c_document_only_active_manifest_cannot_have_an_active_pointer`。
4. **resume 跨 freeze/plan 串用**：伪造 run sidecar 的 freeze/plan 为调用方 A2，但继续传 manifest A；
   旧 `PortfolioRun.read` 只检查 run 自身字段，公开 resume 可返回 14 个 completed subjects。
   新负例：`test_p2_d_resume_binds_the_manifest_to_the_callers_freeze_and_plan`。
5. **revoked manifest 防御层缺失**：把 run 的 manifest digest/generation 配平到 revoked manifest 后，
   旧 `PortfolioRun.read` 接受 status=active 的 run；公开 resume 原先还依赖后续 pointer 拒绝。
   新负例：`test_p2_d_resume_rejects_a_revoked_manifest_before_pointer_fallback`。

这些攻击只修改临时 fixture/临时 run 或内存副本，没有改真实证据数字、删除测试、skip/xfail 或降低既有断言。

## 3. 最小生产修复

- `tcenv/backends/metal/portfolio_confirmation.py`：在 expectation 缓存之前同时校验 locator 的
  deployment pack 与 op_name 属于 freeze 权威任务；
- `taic/portfolio_confirmation.py`：`confirmation_attempts` 必须满足 `type(value) is int` 且在既有
  `1..BENCH_ATTEMPTS` 闭区间；
- `taic/portfolio_manifest.py`：active pointer reader 拒绝 document-only active manifest；
- `taic/portfolio_store.py`：resume 直接要求 active manifest，并绑定 manifest 与调用方的 freeze/plan；
  另外重放 `validation.validate_for(freeze)`，不再只信 run sidecar 自报字段或后续 pointer 防御。

没有新增 dispatch、HMAC 重构、P3 性能治理或模型/GPU 搜索路径。

## 4. 修复后聚焦证据

同一组 5 个新负例在生产修复后实测 **5 passed**。随后运行：

```bash
uv run pytest -q -n 0 -p no:cacheprovider \
  tests/shared/test_shape_portfolio_validation.py \
  tests/backends/metal/test_shape_portfolio_confirmation.py \
  tests/backends/metal/test_shape_portfolio_manifest.py \
  tests/backends/metal/test_shape_portfolio_resume.py
```

实测 **41 passed**。默认验收器相应更新为 P2-A 9 + P2-B 8 + P2-C 16 + P2-D 8，共 41 个
CPU 节点；真实 Mac 节点仍为 5 个。E.md 已加入 `DELIVERABLES`，最终文件字节由 manifest 钉住。

## 5. 完整默认复验

修复后执行：

```bash
uv run python docs/架构/shape-portfolio-阶段验收/P2/verify.py --write-manifest
uv run python -u docs/架构/shape-portfolio-阶段验收/P2/verify.py
```

实测 manifest 更新为 27 个文件，完整默认验收退出 0：P0/P1 anchors 54、结构 6、P2-0 anchor 1，
P2-A/B/C/D 分别 9/8/16/8（合计 41）个 CPU 节点与 5 个真实 Mac 节点全部通过且无 skip；
archived 12 unseen + 2 tune 全部 confirmed，live 12 unseen + 2 tune 也全部 confirmed，
live==archived stable identity 通过；active DispatchManifest 为 14 subjects、generation=1，
checkpoint resume 返回 14 个完成实例，revoked generation=2 后拒绝 resume。没有模型搜索、`taic run/resume`
或性能发布。

上述攻击已固化为仓库内的固定负例，不依赖仓库外临时脚本；正常 14-subject activation 保持成功，
其余伪造与串用路径均由测试断言拒绝。

## 6. 追加独立复核：expectation 权威与 checkpoint 接管

后续 reviewer 又发现两处旧负例未覆盖的跨阶段缺口。本轮先新增两个固定负例并在旧代码上实测 **2 failed**：

1. `test_p2_b_manifest_rejects_expectations_rebuilt_under_another_freeze`：locator B 与 freeze B/plan B
   完全合法地重建 14 份 expectations，再协同重钉 sidecars；随后即使激活时提供 locator A，旧 manifest 门仍把
   调用方 expectation 当作权威并接受 freeze A/plan A。这覆盖了只检查 `activation_expectation` 参数或只检查
   随调用附带的 locator 都无法关闭的路径。
2. `test_p2_d_active_checkpoint_cannot_be_taken_over_by_another_run_config`：同一 active manifest、同一代次下，
   旧 `write_checkpoint` 允许用 run_config B 覆盖 A 的 sidecar，使 B 继承 A 的 14 个完成实例。

最小修复：`DispatchManifest.from_confirmations` 现在必须接收权威 locator，并在 manifest 门内部调用唯一的
`activation_expectation` 对当前 freeze/plan/locator 重放 dispatch/preflight/expected；调用方 expectations
只作交叉核对，必须与这次内部重建的 record/expected 严格一致。当时 locator 只比较自报的 pack/op_name，
真实字节绑定仍有第 8 节记录的缺口。`write_checkpoint` 对已有 active sidecar 额外拒绝 manifest 或
run_config 身份变化，只允许二者都相同的幂等重写；D.md 的旧“重复 checkpoint 已被阻止”表述已改成这一
精确语义。`verify_manifest_document` docstring 也明确其 canonical reader 不是 sidecar-backed 激活证据。

两个新负例修复后实测 **2 passed**；相关四个测试文件实测 **43 passed**。默认验收计数相应更新为
P2-A/B/C/D = 9/9/16/9（合计 43），真实 Mac 节点仍为 5。最终完整默认验收结果在下节记录；P2-E 继续保持
in progress，不自签 completed。

## 7. 最终字节复验

追加修复后执行 `--write-manifest` 与完整默认验收，实测 manifest 仍为 27 个文件，验收退出 0：
P0/P1 anchors 54、结构 6、P2-0 anchor 1；P2-A/B/C/D = 9/9/16/9，共 43 个 CPU 节点，另有
5 个真实 Mac 节点，全部通过且无 skip/deselect。archived 与 live 均为 12 unseen + 2 tune 全 confirmed，
live==archived stable identity 通过；active manifest 为 14 subjects、generation=1，checkpoint 返回 14 个
完成实例，revoked generation=2 后拒绝 resume。没有模型搜索、`taic run/resume`、P3 或性能发布。

本文写入上述实测结果后再次重写 manifest，并对最终字节重跑同一完整默认验收。P2-E 仍待独立 reviewer，
保持 in progress。

## 8. 自报 locator 与撤销回写攻击

最新独立攻击再次先固化为两个负例，并在旧代码上分别实测失败：

1. `test_p2_b_lying_locator_cannot_bind_pack_b_bytes_to_freeze_a`：构造
   `lying = replace(changed, deployment_pack_digest=freeze_A.task_pack_digest)`。它的 locator identity 与真实 A
   相同，但 `task_dir` 指向只追加一行注释的 pack B。旧门只比较自报 digest，能把 B 的 14 份 expectations
   配平后激活 freeze A。`test_p2_d_revocation_marker_blocks_restoring_the_old_active_sidecar` 则复现 revoke 后
   回写旧 active `portfolio_run.json`，旧 resume 返回 14 个完成实例且 observe 显示 active。
2. 修复在 `activation_expectation`、`DispatchManifest.from_confirmations` 与 `confirm_subject` 三个入口都调用
   `locator.load()`，由唯一 TaskLocator reader 重算 task_dir/code_root 闭包并比对两个 digest；op_name 的既有
   稳定错误门保持优先。这样自报 identity 相同也不能把 B 字节绑到 freeze A，且不依赖后续执行路径补救。
3. revoke 现在先原子写持久 `portfolio_revocation.json` 墓碑，再写 revoked sidecar。resume/write 遇到墓碑
   一律拒绝，observe 以墓碑为准；只恢复旧 active sidecar 或旧 pointer 均不能清除撤销。墓碑不是 HMAC 或
   追加式外部日志：若攻击者能删除墓碑或回滚整个目录，仍需上层权限/持久撤销边界，D.md 已明确该限制。

两个新负例修复后实测 **2 passed**，wrong-op 既有负例与 lying-locator 负例共同复跑也通过。C/D 文档已
同步为 16 个 Manifest 节点与 10 个恢复节点并补齐清单。默认验收计数更新为 P2-A/B/C/D = 9/10/16/10，
合计 45 个 CPU 节点，真实 Mac 节点仍为 5；最终完整结果记录于第 9 节。

## 9. 最终 45 节点复验

执行 `--write-manifest` 与完整默认验收，实测 manifest 为 27 个文件，验收退出 0：P0/P1 anchors 54、
结构 6、P2-0 anchor 1；P2-A/B/C/D = 9/10/16/10，共 45 个 CPU 节点，另有 5 个真实 Mac 节点，
全部通过且无 skip/deselect。archived/live 的 12 unseen + 2 tune 本次均全 confirmed，live==archived stable
identity 通过；active manifest 为 14 subjects、generation=1，checkpoint 返回 14 个完成实例，revoke 写出
持久墓碑与 generation=2 失效账本后拒绝 resume。无模型搜索、`taic run/resume`、P3 或性能发布。

本文记录实测结果后再次重写 manifest，并对最终字节重跑同一完整默认验收。P2-E 继续保持 in progress，
不自签 completed。

## 10. 文档证据清单校正与本阶段残余边界

最终文档复核发现 B.md 仍保留旧节点数。现已将三处统一为 **10 个 CPU 负例 + 5 个真实 Mac/GPU 节点**，
并在真实 Mac 证据清单补入 `test_p2_c_tune_instances_have_independent_fresh_confirmation_for_activation`：
该节点生成 2 份 tune fresh-confirmation sidecars，归档位置为 `P2/evidence/tune-confirmations.json`，与
12 份 unseen sidecars 一起支撑 14/14 激活。C/D 已分别为 16/10 节点，最终默认计数保持 45 CPU + 5 Mac。

以下是有意保留并已定位的残余边界，不在 P2-E 继续加功能：

1. sidecar 自身不是外部信任根。持有者可选择同一 freeze 的另一个 plan 重签 seed，或重新 freeze 到 pack B
   后用 CPU 重建 expected 并重签 archive；这可以零 GPU 构造自洽文档，但不构成真实设备回执。当前阶段不借
   HMAC 重构消除该边界，archive 语义投影与 live fresh confirmation 不得被表述为硬件真实性证明。
2. 持久墓碑只防止保留墓碑时的 sidecar/pointer 回写。删除 `portfolio_revocation.json` 后再恢复旧 active
   sidecar 仍可绕过本地撤销状态；整目录回滚需要上层权限或外部持久撤销日志，D.md 已明确声明。
3. 当前固定负例覆盖状态转换与回写攻击，但没有真正注入进程在 checkpoint 原子写步骤中途崩溃的故障矩阵；
   这项仍是 D.md 的未覆盖范围，不能由普通 resume/revoke 正例冒充。
