# 真实 Metal fresh 观察历史

本目录保留当前仍可读取的三次完整 benchmark sidecar，不挑选较快一轮作为结论。

- `pytest-194`：同一实现的一次早期 benchmark 返回 `rejected`。当时测试在断言前未落盘，只有
  `logs/pytest/triage/` 中的失败记录，reason 未保存；该观察不用于性能或稳定性结论。
- `pytest-195`：一次 benchmark/golden `confirmed`、identity `match`，控制台摘要为
  candidate `1.0215285458668535 µs`、MLX `2.1575835249093793 µs`、speedup `2.11211280745805`。
  原 temporary sidecar 后续被 pytest 临时目录轮转，未纳入本包，也不用于稳定性结论。
- `r1-confirmed`（源 `pytest-196`）：完整 benchmark/golden 已归档，均 confirmed。
- `r2-timing-uncertain`（源 `pytest-197`）：benchmark 三次确认后 `timing_uncertain`；测试当时在
  benchmark 断言处停止，因此没有 golden sidecar。该缺失不是一次 golden 通过或失败。
- `r3-timing-uncertain`（源 `pytest-198`）：benchmark `timing_uncertain`，golden `confirmed`，均已归档。

三次归档观察中两次 quiet timing 未通过，所以 `stable_fresh_performance=false`、
`performance_gate_passed=false`、`publication_allowed=false`。identity/native/artifact 核对与已执行的
golden correctness 均保留原始证据；本包只支持真实 Metal 机制结论，不支持生产 available 或性能发布。
