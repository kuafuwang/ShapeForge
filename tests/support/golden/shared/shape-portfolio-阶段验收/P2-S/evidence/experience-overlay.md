## conv2d paper：全局 U 只在这套 MMA 上划算

Keywords: conv2d scratch shared winograd paper mma
Confidence: verified
观察: 这份文字是 P2-S 的冻结叠加层，还没有写入 targets/apple-m4-max/experience.md。 标量三臂，以及后文里标成 probe 下界或 paired probe 的数字，作为反例上下文保留，但不是独立 teachable lesson，也不是 confirmed speedup。kda tail 的决策测量、kda single 的 N=1 自检，以及 P2-R 的 harness 修复，出现在后文里只作边界：数值不采用，不作为独立经验。occupied retry 的墙钟和搜索最优倍数不进入本文。 fp16/paper 上，同一套 F(4×4) simdgroup MMA，全局 U 的 fresh 墙钟是 617.29 µs，每个 threadgroup 重算 U 是 1474.42 µs。重算除以全局 U 等于 2.3885 倍。两份都是 fresh subject（confirmed，identity match，confirmation seed 不是搜索种子 7）。这是同算法复用，不是发布的加速比。
同一形状上更早的逐线程标量三臂也是 fresh subject，但都远慢于库基线：recompute-f2 17622.82 µs, scratch-f2 19480.31 µs, scratch-f4 17473.25 µs。那三臂不能当成「scratch 一定更快」。它们留在这里只作反例上下文，不是独立 teachable lesson，也不是 confirmed speedup。
来源: `tests/support/golden/shared/shape-portfolio-阶段验收/P2-I/evidence/conv2d-mma-f4-confirmation.json`、`tests/support/golden/shared/shape-portfolio-阶段验收/P2-I/evidence/conv2d-ablation.json`、`docs/架构/中间结果复用与Scratch Workspace.md`
本机实测: Apple M4 Max，fp16 paper，全局 U 617.29 µs，重算 1474.42 µs，比 2.3885。标量三臂是反例，不是独立经验。
边界: 只覆盖 Apple M4 Max 的 fp16 paper 和这份 MMA。不要把 producer 的十几微秒说成整个 kernel 的时间。标量三臂的数字不作为独立经验。publication_allowed=false。

## conv2d small：单阶段重算，不要照搬 paper 的两阶段

Keywords: conv2d small single-stage recompute floor
Confidence: verified
观察: fp16/small 的单阶段 fresh 墙钟是 3.2373 µs，相对最快基线的比值是 0.9858，没有快过最快基线。两阶段、无 fill 的启动下限是 6.1287 µs，而且 `not_a_candidate=true`；这是 probe 下界，不是 candidate 时间，也不是 confirmed speedup。它已经落在 1.10 门外面。
随后把权重缓进 lane 私有数组的候选是正确的，但墙钟 4.0056 µs，比这份 fresh 慢 23.7%。P2-N 因此停止继续做 small 的微优化。这个百分比是变慢，不是加速比。
来源: `tests/support/golden/shared/shape-portfolio-阶段验收/P2-J/evidence/small-confirmation.json`、`tests/support/golden/shared/shape-portfolio-阶段验收/P2-J/evidence/launch-floor.json`、`tests/support/golden/shared/shape-portfolio-阶段验收/P2-N/evidence/decision.json`、`docs/架构/conv2d-small-mma-headroom.md`
本机实测: Apple M4 Max，fp16 small，单阶段 fresh 3.2373 µs，两阶段下限 6.1287 µs，not_a_candidate=true。
边界: 只覆盖登记的 fp16 small。不要为了使用 ScratchSpec 再写一个两阶段 small。publication_allowed=false。

## gemm_silu：SiLU 留在片段里，不要写全局 scratch

Keywords: gemm_silu reg silu epilogue scratch
Confidence: verified
观察: `C = silu(A @ B)` 的中间结果只属于一个输出元素。fp16/small 的 REG fresh 是 1.8252 µs，相对最快基线 1.6721 倍；fp16/paper 的 REG fresh 是 1160.87 µs，相对最快基线 1.0071 倍。两份都是 fresh subject。paper 只比最快基线快不到 1%，不要把它写成稳定的生产加速。
没有实现 GLOBAL 候选。small 的两阶段启动下限是 8.7713 µs，约 4.8056 倍于 REG；这是 probe 下界，不是 confirmed speedup。paper 的保守流量下界比 REG 高 4.4%，依据是 stage-2 必须读回 fp32 再写输出，不是一份 GLOBAL 核的 fresh 时间。
来源: `tests/support/golden/shared/shape-portfolio-阶段验收/P2-O/evidence/reg-small-confirmation.json`、`tests/support/golden/shared/shape-portfolio-阶段验收/P2-O/evidence/reg-paper-confirmation.json`、`tests/support/golden/shared/shape-portfolio-阶段验收/P2-O/evidence/decision.json`、`tests/support/golden/shared/shape-portfolio-阶段验收/P2-O/evidence/scratch-probe.json`、`docs/架构/gemm-silu-scratch-headroom.md`
本机实测: Apple M4 Max，fp16，REG fresh small 1.8252 µs、paper 1160.87 µs。GLOBAL 没有实现。
边界: 只覆盖这两个 fp16 形状。不要把 conv2d 的全局 U 套到这个逐元素 epilogue 上。publication_allowed=false。

## bitdelta_gemm：在 tile 里解包，不要先展开成全局权重

Keywords: bitdelta unpack reg shared
Confidence: verified
观察: fp16/small 的 REG fresh 是 2.1044 µs，fp16/paper 是 1219.06 µs。两份都是 fresh subject。没有写 SHARED 候选。paper 的同轮配对中位比是 1.0259 倍，可以保留。small 的两个最省合法形态在归档、重放和评审重跑之间翻过，只能教大约一个数量级，不能教精确倍数或排序。这是 paired probe，不是 fresh candidate，也不是 confirmed speedup。paper 上 REG 高于硬件下限的可回收比例是 4.6%，这是同一份报告里的比值，不是另一套算法的上限。
来源: `tests/support/golden/shared/shape-portfolio-阶段验收/P2-P/evidence/reg-small-confirmation.json`、`tests/support/golden/shared/shape-portfolio-阶段验收/P2-P/evidence/reg-paper-confirmation.json`、`tests/support/golden/shared/shape-portfolio-阶段验收/P2-P/evidence/decision.json`、`tests/support/golden/shared/shape-portfolio-阶段验收/P2-P/evidence/unpack-probe.json`、`docs/架构/bitdelta-unpack-headroom.md`、`tests/support/golden/shared/shape-portfolio-阶段验收/P2-P/P.md`、`tests/support/golden/shared/shape-portfolio-阶段验收/P2-P/review.json`
本机实测: Apple M4 Max，fp16。REG fresh small 2.1044 µs、paper 1219.06 µs。paper 配对中位比 1.0259。small 只记大约一个数量级。
边界: 只覆盖这两个 fp16 形状。small 不教精确倍数，也不教两个最省形态的排序。展开省下的是位运算，换来的是更大的全局流量和一个 stage。publication_allowed=false。

## GEMM Split-K：这两个形状不要做

Keywords: gemm split-k reduction scratch
Confidence: verified
观察: fp16/small 的两阶段无 fill 下限是 7.0390 µs，约 3.7514 倍于归档的既有 candidate 锚点。这个下限 `not_a_candidate=true`，不是 candidate 时间。fp16/paper 已有 1024 个输出 threadgroup。10.98% 不是字节流量比。它是峰值带宽时间等价：`split2_partial_write_read_us_at_peak` / `existing_candidate_us` = 127.42 µs / 1160.09 µs。没有新增 Split-K 源码。
来源: `tests/support/golden/shared/shape-portfolio-阶段验收/P2-L/evidence/analysis.json`、`tests/support/golden/shared/shape-portfolio-阶段验收/P2-L/evidence/launch-floor.json`、`docs/架构/gemm-splitk-headroom.md`
本机实测: Apple M4 Max，fp16。small 两阶段下限 7.0390 µs。paper 的峰值带宽时间等价比是 10.98%，分子是 split2_partial_write_read_us_at_peak。
边界: 既有 candidate 微秒数是归档锚点，不是本阶段新的 fresh subject。这个百分比的单位是峰值带宽时间，不是字节比。不要因为 conv2d 的 SHARED 成立就做 Split-K。publication_allowed=false。

## 因果状态：串行 stage 下切链不会变并发

Keywords: kda seq handoff serial stage
Confidence: verified
观察: KDA 的六份 fresh subject 里，`device/host_piece_handoff.metal` 都是 124c7485…。切链改的是 Host Program `host/tiling.c`：三 piece 为 369e3b44…，两个单 piece 臂为 e66406d7…。stage 数改的是 `host/execution_geometry.json`：三 stage 为 686ce257…，单 stage 为 0b8cfe07…。varlen mixed 的两份 fresh subject 没有 Host Program。`device/candidate.metal` 不是同一份字节：pieces_three 是 0d214416…，pieces_one 是 aaa42e0a…。`varlen_source` 的唯一文本差别在 `kernel void varlen_plan` 内，把 `uint pieces = min(MAX_SPLIT, len);` 换成 `uint pieces = 1u;`。这不是 Host Program 改切链。 八个 fresh subject 都经过确认读取：kda small 与 paper 各有 A/B/C，varlen mixed 有两臂。
kda small：A seq_three_piece 246.64 µs（confirmation seed 8854856279691335），B single_piece_three_stage 224.81 µs（confirmation seed 6676742521560864），C single_piece_single_stage 205.51 µs（confirmation seed 6589246940285159）。A−B 是 21.8319 µs，B−C 是 19.2936 µs。A−C 相对 C 是 20.0%。三份 confirmation seed 不同，这不是同一次输入的 paired run。A−C 同时改了 piece 数和 stage 数，是跨 confirmation 的总开销，不能解释为隔离的 handoff 成本。A−B 与 B−C 也是跨 confirmation 的臂差，不能和决策样本的 handoff、stage floor 加成一个数。
kda paper：A seq_three_piece 188422.04 µs（confirmation seed 4123769460687530），B single_piece_three_stage 187407.45 µs（confirmation seed 4861717864745990），C single_piece_single_stage 187275.13 µs（confirmation seed 7842705585200777）。A−B 是 1014.58 µs，B−C 是 132.33 µs。A−C 相对 C 是 0.6%。三份 confirmation seed 不同，这不是同一次输入的 paired run。A−C 同时改了 piece 数和 stage 数，是跨 confirmation 的总开销，不能解释为隔离的 handoff 成本。A−B 与 B−C 也是跨 confirmation 的臂差，不能和决策样本的 handoff、stage floor 加成一个数。 paper 的 A−C 0.6% 只说明相对一次慢递推的总开销很小，不能外推成 SEQ 便宜。
varlen mixed 只覆盖 L1：pieces_three 54.9266 µs（confirmation seed 3618969398026108）对 pieces_one 33.9083 µs（confirmation seed 1801390490111273），相对是 62.0%。两份 seed 不同，dispatch 的 stage 数相同，device 字节不同。不含 L2 capacity。
决策测量是另一份样本：small A−C 23.0%，其中 handoff 字段 26.0449 µs、stage floor 20.2645 µs；paper 0.5%；varlen 56.4%。不要把决策样本和 fresh 加成一个数。
kda tail 在 decision.sizes 里，A−C 是 19.1%，handoff 字段 25.2026 µs，stage floor 15.4361 µs。它不是八个 fresh subject 之一。数值不采用，不作为独立经验。
kda single 不在 decision.sizes。state_algebra 的 pieces_per_chain 是 1，boundaries_used 是 0。测量文件里的 N=1 自检，A−B 相对 B 是 0.8%。这不是决策行。数值不采用，不作为独立经验。
两块 scratch 乒乓的容量等于声明的 workspace。P2-R 修的是 worker 几何身份，evaluation_device_verdict=false。数值不采用，不作为独立经验。handoff_states 继续留在 OpSpec workspace。允许的同步是串行 encoder 加 buffer barrier，切链不会让同一条链的两段同时跑。
来源: `tests/support/golden/shared/shape-portfolio-阶段验收/P2-Q/evidence/confirmations.json`、`tests/support/golden/shared/shape-portfolio-阶段验收/P2-Q/evidence/decision.json`、`tests/support/golden/shared/shape-portfolio-阶段验收/P2-Q/evidence/seq-handoff-measurements.json`、`tests/support/seq_handoff.py`、`tests/support/golden/shared/shape-portfolio-阶段验收/P2-R/evidence/identity-delta.json`、`docs/架构/seq-state-handoff-headroom.md`、`docs/架构/bundle-scratch-worker-geometry.md`
本机实测: Apple M4 Max，fp16。kda small fresh A/B/C 246.64/224.81/205.51 µs；kda paper fresh A/B/C 188422.04/187407.45/187275.13 µs；varlen L1 mixed 54.9266 µs 对 33.9083 µs。这些不是同一次输入的 paired run。
边界: 只覆盖 Apple M4 Max、fp16、kda 的 small 与 paper，以及 varlen_kda 的 L1 mixed。不含 L2 capacity。paper 上跨 confirmation 的 A−C 0.6% 不能外推成 SEQ 便宜。kda tail 的决策测量和 kda single 的 N=1 自检数值不采用、不作为独立经验。P2-R 的 harness 修复同样不作为独立经验。不要为了并发写跨 threadgroup 自旋。publication_allowed=false。NPU 仍是 simulation-only。
