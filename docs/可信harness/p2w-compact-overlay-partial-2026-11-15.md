# P2-W 紧凑 overlay 实弹 run 的 partial 证据

这份记录只登记盘上已经存在、但此前没有进入实验索引的真实执行。**本次没有启动新搜索、GPU 或付费调用。** P2-V 文档描述的是当时仅做 render-only 的阶段，保持原字节；这里记录的是其后单独批准的 P2-W，不能把冻结计划里的 `executed=false` 当成当前执行状态，也不能把它改写成已完成实验。

## 冻结配置

- 任务：`gemm_silu` / fp16 / small / `apple-m4-max`；run_seed 与搜索评测字面量均为 7。
- 委员会：first = DeepSeek `deepseek-flash`，second = `kimi-code/k3.1-pro`，effort=high。
- 两对 AB/BA；control 无 overlay，treatment 使用紧凑 overlay，SHA-256 `fd72abf078d9d5a7cd2118ee5263a3c83e5695c044c0bc0cc6ca86758a0aa3d1`。
- 两组（对照组是不加这项改动的一侧，实验组是加上要验证的改动的一侧）其余开关相同：`init_samples=1`、`plateau_after=0`、`sweep_tool=false`、`task_ledger=false`。
- 每格软预算 $3，是否启动后续格的承诺上限 $12；soft budget 不是账单硬上限。
- P2-W 有独立批准记录 `shapeforge.p2w-compact-search-approval.1`，不沿用 P2-U 批准；`publication_allowed=false`。
- 原始计划：`runs/p2w-compact-overlay-paid-20261115/plan.json`，parent HEAD `d3388497aabaf2a687f3f1d65801c87edd9fdd9c`。

## 盘上实际结果

| cell | 候选 / confirmation SHA | 实际成本 | search best | fresh confirmation |
| --- | --- | ---: | ---: | --- |
| control rep1 | `1e5c530a172a7b05f07db14194f91d0574d14b7ddf08652e6318370767c058dd` | $2.874923508 | 1.75681080343184 | ok / 1.6097813118723656 |
| treatment rep1 | 未形成已绑定 confirmation | $3.2983582480000004 | 1.6748727233980327 | **timing_uncertain**，数值 unavailable |
| treatment rep2 | `1bcd12044baa7f47aec2947b3c9008c28997f7174a21b579cf05d66a874a118c` | $2.986434316 | 1.8454684805290096 | ok / 1.899193482812056 |
| control rep2 | 缺失 | 未发生 / unavailable | unavailable | unavailable |

三格实际账本成本合计 **$9.159716072**。原始 run 位于 `runs/p2w-compact-overlay-paid-20261115/`；上述数字已复制到本文，不只依赖 gitignored run 目录。

## 结论与停止边界

- 只有一对半的格子，control rep2 缺失，treatment rep1 又是 `timing_uncertain`，所以**没有完整配对，不计算 lift，不给效果 verdict**。
- 状态固定为 `partial / inconclusive`，`adopt_compact_overlay=false`、`publication_allowed=false`。不能拿 treatment rep2 的单格 confirmed 与 control rep1 直接配对，也不能用 search best 代替 fresh confirmation。
- 本次仅补证据登记，不续跑、不补确认、不追加付费。若未来要完成实验，必须重新审批剩余 cell、预算与停止规则。
