## gemm_silu fp16 small

overlay_kind: compact
task: gemm_silu
dtype: fp16
size: small
target_id: apple-m4-max
Confidence: verified
这份文字是 P2-S 的冻结叠加层，还没有写入 targets/apple-m4-max/experience.md。
观察: 一次性 SiLU 中间结果留在寄存器。GLOBAL scratch 不省 FLOPs，还增加 fp32 写读、barrier、producer/consumer 和 dispatch。没有写 GLOBAL 候选。REG fresh 1.8252 µs，两阶段 no-fill floor 8.7713 µs，not_a_candidate=true。这是已测结构边界，不是发布 speedup。候选自有算法与 tiling。正确性与 fresh identity 仍按原合同。
来源: `docs/架构/gemm-silu-scratch-headroom.md`；P2-O `reg-small-confirmation.json`、`scratch-probe.json`、`decision.json`
本机实测: Apple M4 Max，fp16 small，REG fresh 1.8252 µs，两阶段 no-fill floor 8.7713 µs。
边界: 只覆盖 gemm_silu、fp16、small。不指定候选 tiling。不写其他任务合同。publication_allowed=false。
