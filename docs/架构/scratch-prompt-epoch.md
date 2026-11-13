# ScratchSpec 提示面 epoch（P2-M）

P2-M 把已在 P2-I 实现的 ScratchSpec 合同送进写核代理实际可见的提示面。此前 `system_prompt_v2` 仍把 bundle geometry 写成“只有三个 key”，`msl_reference` 还说 contract 没有 scratch，这两个描述都已过时。

## 修改面

- `taic/prompts.py`：msl-single 的 launch scratch 语法、msl-bundle 的可选顶层 `scratch` 与 stage uses、三种 init policy、每轮重算、barrier、fallback 和同协议 A/B。
- `backends/metal/skills/msl_reference.md`：完整生命周期、反射 write、受限 fp32 texture、guard/probe、禁止输出借用/prepare 缓存/跨 TG 自旋/无序浮点 atomics。
- `taic/flows/kernel_once/skills/taic-tools/SKILL.md`：写核会话中的测量与正确性检查方法。
- `docs/agent-skills.md` 与 `docs/写核代理读什么.md`：把“提示面尚未引用”改为真实落点。

提示特别强调：**correctness out1/out3 即使对 `device_zero` scratch 也填 poison，out2 才填零**。声明 `zero` 只决定计时窗内的 fill，不允许全局 flag 依赖初值为零。每个计时迭代都重新生成中间结果，prepare 缓存不属于合法优化。

## 摘要与历史边界

- previous prompt SHA-256：`27e2bd2c2dfe3c009b065e1e83c9e2051bed9381fce79aa36b92618c2ac0fe3c`
- current prompt SHA-256：`e9d28ff61819a041610d763f5c1464ea860265be2d414f78c15a1f8aea1fdbcf`
- parent HEAD：`2383bae8a3f56c5dec317e819b46c38b755aa737`
- pack closure 仍为 79 文件，digest `5512395852ea…`

`prompt-epoch.json` 是新旧摘要的权威登记。旧测试继续钉住各自的历史值；live 锚点只前进到上述新值，没有 old→new hash 别名，也没有把 `current_prompt_sha1()` 改成动态自证。提示变化会使旧 overlay grant 变成 stale，本阶段不重签、不修改 `trust.toml`，也不扩大权限。

P2-I/J/K/L 的候选、证据和 manifest 保持原字节。没有改 `experience.md`、六 RPC、字段号、HMAC 或 BlobChannel，没有启动模型搜索。性能示例只说明如何做同协议消融，不发布新的 speedup；GEMM Split-K 的负结果与 small 两阶段负结果同样进入指导。

`tests/support/golden/metal/hw_off.json` 是允许随提示更新重新生成的 live off baseline，本阶段通过 `tests.support.hw_golden --record` 重录，旧文件保存在 P2-M `hw_off-previous.json`。此前 5 个 TileLang staged-install identity 失败来自恢复安装损坏；现有完整本地 clean 安装已切到 `-recovered` 路径，未下载、未修改 pinned frontend，5 个节点已纳入本阶段验收。最终完整快速套件为 **5203 passed、13 skipped、0 failed**，恢复记录见 P2-M `frontend-recovery-note.json`。
