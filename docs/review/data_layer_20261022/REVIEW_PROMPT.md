# Review prompt（把这段连同目录一起交给审核 agent）

你是独立审核者。目标：验证 `docs/review/data_layer_20261022/` 里的数据、结论和建议是否成立，并区分「统计上还不够」与「结构上已经错了」。

## 输入
- `README.md`：主审核包（§0 阅读说明、§5 F1–F15、§7 12 条挑战、§9 输出模板）。
- `findings.json`：机器可读的指标、发现、建议、挑战。
- `repro.sh`：只读复现脚本，打印基线数字。

## 硬约束
- **不要**运行 `taic run` / `taic resume`；**不要**启动 GPU kernel；**不要**调用 LLM/API（会花钱）。
- `taic brief` 和 `repro.sh` 是只读的，允许运行。
- 需要写文件时只用 `/tmp`；不要修改仓库。
- 仓库：`/Users/kuafu/CLionProjects/ai_lowerling`。

## 步骤
1. 读 `README.md` 的 §0、§1、§3、§4。
2. 运行：
   ```bash
   cd /Users/kuafu/CLionProjects/ai_lowerling
   bash docs/review/data_layer_20261022/repro.sh
   # 可选：SLOW=1 会多测一次 run_detail（某个 run 可能约 10s）
   ```
3. 对 §5 的 **F1–F15** 逐条给 `PASS / FAIL / UNKNOWN`，附证据（命令输出、文件行号、数字）。
   - 特别优先：F1（计费）、F3/F8（每 run/每场固定成本）、F4/F5（规模化检索）。
   - 对每个结论标注：`S0`（与样本量无关）、`S1`（随规模恶化）、`S2`（需要更多数据）。
4. 对 §7 的 **12 条挑战**逐条回答；能推翻就推翻。
5. 检查 §8 的「已知分歧」：特别是「文件数中位 398 不可复现，实测全集中位 40、当前大 run 17–32/场」这条修正是否正确。
6. 按 §9 模板输出；另加：
   - 被推翻/降级/升级的结论；
   - 新发现（带 `file:line`、测量值、影响）；
   - 对 P0/试点/P1 优先级的调整建议。

## 判定口径
- **PASS**：证据复现，结论成立。
- **FAIL**：证据不成立、口径错误、或结论过度外推。
- **UNKNOWN**：需要更多数据/环境，当前无法判定（说明缺什么）。
- 不要把 legacy chat 协议和当前 hmz 协议混算；不要把 `runs.model` 当成会话模型；不要跨协议 epoch 聚合速度/成本/confirmed。

## 输出位置
把审核结果写到 `/tmp/data_layer_review_by_<your-name>.md`（或直接 stdout），不要写进仓库。
