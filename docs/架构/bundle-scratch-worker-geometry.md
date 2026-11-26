# bundle + Host Program + scratch 的 worker 几何身份

P2-R 只修一件事：`msl-bundle` 带 Host Program、并且几何里有非空 scratch 时，worker 重建的 `ResolvedGeometry` 必须和控制机冻结的文档一致。不做 Plan 3，不把 SEQ 的 `handoff_states` 搬进 scratch，不改 OpSpec workspace 协议。

`publication_allowed=false`。NPU 仍是 simulation-only。提示面和 trust 没有改。

## 根因

控制机按求值后的几何登记 `CandidateLifecycle`。`ResolvedGeometry.document()` 只在 scratch 非空时写出 `scratch` 键，每个 stage 同样。worker 收到这份冻结文档后，`tcenv/v2_worker.py::_lifecycle_from_document` 原来重建 `ResolvedGeometry` 和 `ResolvedStage` 时不读这个键。

`tcenv/taskcontract/runner.py` 接着比较两份文档：

```text
lifecycle.geometry.document() != chosen.geometry.document()
```

不相等就抛 `TaskContractError: the lifecycle was registered for a different resolved geometry`。比较本身是对的。P2-Q 的两块 `uninitialized_poison` 乒乓和 `device_zero` 单块都过了生命周期预检，然后在这里失败。空 scratch 的 KDA bundle 和 `msl-single` 的 conv2d scratch 不走这条带 Host Program 的重建，所以本来就能跑。

修复前的重建留在 `tests/support/bundle_scratch_identity.py::lifecycle_from_document_before_p2r`。用当前 P2-Q 乒乓候选重算：冻结文档有 `scratch`，旧重建没有，两份 `document()` 不相等。现行重建相等。

## 修改

生产代码只有两处。

`tcenv/v2_worker.py` 增加三个读取函数。缺省、`null` 和空列表都当成空，重建后的 `document()` 不写 `scratch` 键。非空列表用 `ScratchUse.from_document` 和 `ResolvedScratch.from_document` 还原。身份比较的那一行没有删，guard、coverage、`host_written` 的检查没有放宽。

`tcenv/scratch/spec.py` 的 `ResolvedScratch.from_document` 要求正好六个字段，并且保存冻结文档里的 `numel` 和 `nbytes`，不按 size term 重算。重算会让一份被改过的冻结文档在比较时悄悄变成一致。改大 `nbytes` 的文档因此仍然和对侧求值几何不相等。

没有 scratch 字段的旧对象继续走 `getattr(..., "scratch", ())`。`ResolvedGeometry.scratch` 的缺省仍是空元组。

## 闭包

P2-R 当时的任务包是 `tcenv.taskcontract` 和 `tcenv.opspec` 的 import 闭包，79 个文件。`tcenv/v2_worker.py` 是 worker 入口，不在这 79 个文件里。`tcenv/scratch/spec.py` 在里面。加上 `from_document` 之后，文件集合不变，摘要从

`5512395852ea7608a6075e2b6b45dca9bd85f112210af336cf6f8f77f5fd811f`

变成

`b5b74185710e5c64cf55ed20cc472f3258e0fe2b1c631bb7f850eeb8dcbb069e`。

相对 HEAD，闭包里只有 `tcenv/scratch/spec.py` 的字节变了。`tcenv/v2_worker.py` 也改了，但它不在这 79 个文件里。

P2-I 的 manifest 把当时的 `tcenv/v2_worker.py` 记成 `39ce59320d35066489d88db4eac213c65764e6c57f8932b16fbd6e84a01c370c`。P2-G 和 P2-H 用的是同一个摘要。这份 manifest 没有改写。P2-R 当时的文件是 `a083a36272e31bb9f9edaa60f5df9a57a927facf150582e7b51f087ac65cbe33`。现场字节已经不是这份摘要，所以在这棵树上重跑 P2-I 的 verifier，仍会停在这一个文件上。P2-J 到 P2-Q 的 manifest 不含这个文件，其余条目仍然逐字节相符。P2-N/O/P/Q 里钉住旧闭包摘要 `55123958…` 的测试也会失败。这和 P2-I 之后重跑 P2-H 是同一类纪元边界：旧 manifest 保留原字节，不把新修复写回去。

P2-Q 的 GPU 节点 `test_a_two_buffer_pingpong_satisfies_the_scratch_lifecycle` 当时接受两种结局：整段 `ok`，或者身份检查失败。修复后，那份 device 指针候选不再撞身份检查，反射为 `readwrite`，状态是 `abi_error`。那个节点没有改，重跑会失败。归档的 `seq-handoff-scratch-attempt.json` 仍是当时的 crash 记录。

## 空与非空

| 几何 | worker 文档 |
| --- | --- |
| 没有 `scratch` 键的历史 resolution、空的 KDA bundle | 与冻结文档相等，键仍然不存在 |
| P2-Q 两块 poison 乒乓、`device_zero`、`host_written` | 与求值几何和整份 lifecycle 文档都相等 |
| 同一份候选字节，只把已求值几何换成非空 scratch | execution、lifecycle、artifact、native，以及 `TaskEvaluationResult.identity` 都变 |

`TaskEvaluationResult.identity` 是 `fingerprint(document())`。比较用空 scratch 候选的同一份字节：scratch 一侧只注入已经求值的乒乓几何，然后两边都走 `evaluate_task`。`candidate_sha256` 相同。`task_pack_digest` 也相同，都是 `94f63e548ec4066f762238bf705eb09f1e59c96a28847c84d3ef1b1bee4a21aa`。变的顶层字段是 `geometry_identity`、`bundle_digest`、`artifact_set_digest`、`native_digest` 和 `report`。

报告壳两边都是 `rejected`。这份 CPU 报告把已经附上的构建绑进身份，不代表设备上的正确性或计时。报告里不同的叶是冻结几何产物 `execution_geometry.json` 的 sha256 和长度（4707 字节到 5647 字节），以及由此重算的 artifact、provenance 和 native descriptor。`empty_evaluation` 是 `2f7ead7bf7942ebd4dc699e77e212b918b55ed8518cb023bd50f5aa256a039ff`，`scratch_evaluation` 是 `c9350833fae927586b2f665cc9cd630b212c82ec1763d1d94fc99ca1be1275e7`，`evaluation_changed=true`。逐叶差异在 `tests/support/golden/shared/acceptance-epochs/fixtures/p2r/evidence/identity-delta.json`。

相对上一纪元，代码闭包摘要从 `55123958…` 变成 `b5b74185…`，因为闭包里的 `tcenv/scratch/spec.py` 变了。那不是这两次求值之间的差异。

单块 `uninitialized_poison` 的第一次使用如果是 `read_write` 或 `read`，仍然是 `GeometryError`，文案仍要求第一次必须 write-only。

## 真实 Metal

`tests/backends/metal/test_bundle_scratch_geometry_gpu.py` 有 5 个 gpu-marked 节点。前两个不在设备上求值：行宽只算 `scratch_texture_layout`，单块 `read_write` 只走 `preflight_submission`。后三个才调用 `evaluate_candidate`。

可执行候选在 `tests/backends/metal/bundle_scratch_seq.py`。它不改 `tasks/kda`，也不改仓库里的 KDA fixture。两块 scratch 各是 `H*D*D` 个 fp32，`uninitialized_poison`。stage 0 用 `texture2d<float, access::write>` 写 `state_a`，stage 1 读 `state_a` 并写 `state_b`，stage 2 读 `state_b` 并写 `final_state`。`kda small` 的 Host Program 发布 3 个 piece，所以两块 poison 缓冲都会被自己的 producer 整段写完。纹理行宽是 4：登记的 single、small、tail（128 个元素）和 paper（32768 个元素）用 `scratch_texture_layout` 算出来都是 4。线性下标 `i` 对应 `uint2(i % 4, i / 4)`。

launch 块只声明 `work_items` 和 `item_counts`。`handoff_states` 仍是 OpSpec workspace，这份候选不绑定它。

`kda small`、`mode=quick`、reps 1：

| 候选 | 结果 |
| --- | --- |
| 纹理乒乓 | `ok`。`normal` / `wide_range` / `outlier` 都过。`safety.ok`，没有 guard、只读或 coverage 违规。dispatch 是 `kda_piece0/1/2`。HostRun `host_status` 为 0 |
| 同一份纹理核，Host Program `return 1` | `incorrect`。文案含 `host program failed` 和 `host_tiling returned 1`，不是身份检查 |
| P2-Q device 指针乒乓 | `abi_error`。`state_a` 声明 write、反射 `readwrite`。不是 `different resolved geometry` |
| 单块 `read_write` poison | 预检 `geometry` 拒绝，`first used as read_write`，要求 write-only |

## 验收

验收目录已移出工作区。恢复命令在 `tests/support/golden/shared/retired.json`。身份差和任务包闭包的副本在 `tests/support/golden/shared/acceptance-epochs/fixtures/p2r`。

11 个 CPU 节点，2 个 backend/geometry 节点，3 个真实 Metal 求值节点。parent HEAD `a9c1dccce636107c2852995578b05611bf19865b`。

## 还没做的

这份纹理候选只证明合法的 scratch bundle 能跑完。它不是 SEQ 的新实现，也没有速度数字。P2-Q 的结论不变：`handoff_states` 留在 workspace。device 指针仍然不能当 poison 的第一个 producer。P2-I 到 P2-Q 里钉住旧闭包摘要的测试，在这棵树上会失败，不能靠改那些文件或放宽断言来消掉。
