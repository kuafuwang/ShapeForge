# P2-M：ScratchSpec 提示面

- epoch：`p2-m-scratch-prompt-1`
- parent HEAD：`2383bae8a3f56c5dec317e819b46c38b755aa737`
- previous prompt SHA-256：`27e2bd2c2dfe3c009b065e1e83c9e2051bed9381fce79aa36b92618c2ac0fe3c`
- current prompt SHA-256：`e9d28ff61819a041610d763f5c1464ea860265be2d414f78c15a1f8aea1fdbcf`
- pack：79 文件，`5512395852ea7608a6075e2b6b45dca9bd85f112210af336cf6f8f77f5fd811f`
- `publication_allowed=false`

本阶段把 ScratchSpec 写入 `system_prompt_v2`、`msl_reference.md` 和 `taic-tools/SKILL.md`。msl-single 与 msl-bundle 都有候选自有 scratch 的声明方式；bundle geometry 从“只有三个 key”修正为三个必需 key 加可选 `scratch`。三种 init policy、反射 write、受限 fp32 texture、每轮重算、guard/probe、独立 fallback 和同协议 A/B 都进入提示。

特别强调 correctness out1/out3 对 `device_zero` scratch 也填 poison，out2 才填零；`zero` 只决定 timed fill。禁止借用输出、prepare 缓存、未清全局 flag、跨 TG 自旋和无序浮点 atomics。全局 U 的 conv2d 正例、small 两阶段负例和 GEMM Split-K 负结果用于指导选择，不构成新性能发布。

`prompt-epoch.json` 登记新旧摘要；`rendered-prompts.json` 登记 gemm small 的 single/bundle 实际渲染摘要。MT1a 的 TASK.md 仍逐字节相同，system prompt 只前进到 P2-M。四个旧 live prompt 锚点改为校验这个明确的新 epoch，不做 hash 别名或动态自证。

`verify.py` 重算 79 文件闭包、当前 prompt SHA、single/bundle 渲染摘要，并逐文件核对 P2-I 71、P2-J 22、P2-K 13、P2-L 12 个历史交付文件。`hw_off.json` 也按生成器记录新的 off baseline，旧文件保留为 `hw_off-previous.json`；只改变 32 个包含新增 MSL scratch 章节的 system-prompt hash，其余 user/feedback/outbound/orchestrator 区段不变。验收运行 88 个 prompt/scratch/旧锚点节点、5 个恢复后的 TileLang 节点与 10 个结构/文档节点，共 103 个 CPU 节点，不需要 GPU 或模型搜索。旧 overlay grant 随 prompt 变化自然 stale，本阶段不重签、不修改 trust.toml，也不激活或发布。此前的 5 个 TileLang 失败来自本机恢复安装只剩 4 个文件/75MB；现有完整本地 clean 安装已切到 `-recovered` 路径，损坏目录另存，未下载、未改 pinned identity。5 个节点已全部通过，完整快速套件最终为 **5203 passed、13 skipped、0 failed**。恢复事实见 `frontend-recovery-note.json`。
