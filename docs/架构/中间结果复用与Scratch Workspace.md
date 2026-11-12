# 中间结果复用与 Scratch Workspace

候选可以在一次启动里写出一块设备缓冲，供后面的 stage 再读。这块缓冲叫 scratch。它不是 OpSpec 里的 workspace 张量，也不进入参考输入。harness 负责容量、生命周期、计时边界和身份；算法、切分和要不要用 scratch 仍由候选自己决定。

权威实现在 `tcenv/scratch/`。Metal 的解析、绑定和计时在 `tcenv/backends/metal/`。本文记录 Plan 1 已经验收的合同。跨算子推广、提示面和 NPU 模拟映射分别留在后续计划，不在这里提前改。

## 问题从哪来

conv2d 的 Winograd `F(m×m, 3×3)` 先把每个滤波器变成 `U = G g Gᵀ`，再对每个输出 tile 算 `V = Bᵀ d B`，最后 `Y = Aᵀ (U ⊙ V) A`。`U` 只依赖权重。同一组 `(c, oc)` 的 `U` 会被所有输出 tile 重复使用。

如果每个 tile 都从 `w` 重算 `U`，重复的是同一份变换。如果把 `U` 写进全局缓冲再读回来，省下的计算必须盖过这次额外流量。paper 形状是 `N=32, H=W=56, C=64, OC=128, KH=KW=3, OH=OW=54`。`F(2×2)` 的 `U` 是 `16×C×OC` 个 fp32，共 524288 字节。`F(4×4)` 是 `36×C×OC` 个 fp32，共 1179648 字节。`OH=OW=54` 对 4 的余数是 2，尾部 tile 用零填充的输入做变换，不是另一条算法。

借用输出 `y` 当中间缓冲会把结果和临时值叠在同一块内存上。旧状态、漏写和越界都会变成正确性事故。prepare 里只算一次 `U`、后面的计时迭代直接读，会把候选的工作挪到计时窗外。

## 什么时候复用

先看中间结果能不能留在寄存器或片上内存，并且别的执行单元不必再算一遍。

| 分型 | 数据依赖 | 存储 | 本阶段的例子 |
| --- | --- | --- | --- |
| `REG` | 一个执行单元内部就能用完 | 寄存器或 threadgroup，零全局 scratch | 融合类算子。本阶段没有新的实证 |
| `SHARED` | 许多 tile 读同一份输入或权重变换 | 全局 scratch，每轮一个生产者写完，屏障之后多个消费者只读 | conv2d 的 `U` |
| `SPLIT` | 输出并行度不够，要切开归约维再合并 | 每片一块 scratch，最后归并 | Split-K / Split-KV。本阶段没有新的实证 |
| `SEQ` | 序列或时间步必须按因果顺序传递状态 | 候选自己的 handoff 缓冲，stage 串行，屏障建立顺序 | `varlen_kda` 的 piece handoff |

净收益只在同一协议下比较：生产者耗时 + 消费者耗时 + scratch 流量，对上「每个消费者自己重算」的耗时。scratch 更大、或者消费者仍然支配总时间时，复用可以更慢。下面的 conv2d 朴素核就是这个结果。预测的微秒数不是完成条件。

## 字段

`ScratchSpec` 的字段是闭集：

| 字段 | 含义 |
| --- | --- |
| `name` | C/MSL 标识符。不能与 OpSpec 张量、符号、attrs、stage 的 `derived_scalars` 或另一块 scratch 重名 |
| `dtype` | 仓库已注册的标量类型，例如 `fp32` |
| `size_terms` | `tcenv/track/expr.py` 的整数表达式。`F(4×4)` 写成 `36*C*OC`，求值后再乘 `dtype` 的字节数 |
| `init_policy` | `uninitialized_poison`、`device_zero`、`host_written` 三选一 |

目标上限放在 `TargetAbiLimits`：`max_scratch_buffers` 默认 8，`max_total_scratch_bytes` 在 `apple-m4-max` 上是 67108864。这两个键不进 ABI `document()`，也不进 `target_generation_digest`，所以旧的 target 指纹和 P4 模拟重放保持原字节。超限在 CPU 预检给出 `GeometryError`。

`msl-single` 在 `/* taic-launch */` 里写：

```text
scratch uwin fp32 36*C*OC
launch conv2d_wgrid grid=128,1,1 threads=64,1,1 scratch=uwin:write
launch conv2d_kernel grid=12544,1,1 threads=64,1,1 scratch=uwin:read
```

`zero` 表示 `device_zero`，`host` 表示 `host_written`，两个都不写就是 `uninitialized_poison`。`msl-bundle` 在 `execution_geometry.json` 的可选顶层键 `scratch` 里放同一份规格。键不出现时，几何文档与改动前逐键相同。显式空数组会在规范化时消失，旧文件本身没有这个键。

OpSpec workspace 仍是任务合同里的命名张量，容量进 ABI。scratch 是候选几何的一部分，容量进几何身份。同一块物理缓冲不能既报 workspace 又报 scratch；字节只计一次。

## 生命周期

`check_scratch_lifecycle` 在编译前做这些检查：

1. 缓冲个数和总字节不超过目标上限。
2. 名字不与保留名、workspace 或另一块 scratch 冲突。借用输出 `y` 落在这一条。
3. `host_written` 要求候选真有 Host Program。没有 host 入口就拒绝。
4. 每个 stage 引用的名字必须在规格里。
5. `uninitialized_poison` 的第一处激活使用必须声明 `access=write`。先读、或第一处就是 `read_write`，都拒绝。
6. 没有任何激活 stage 使用的 scratch 拒绝。

## 借用 y 的五阶段

把输出 `y` 借来当 `U`，会走过五步，每一步都和真正的输出抢同一块缓冲：

1. Producer 把变换后的权重写入 `y`。此时 `y` 还不是卷积结果。
2. Consumer 把 `y` 的这块区域读成 `U`。这是对同一次迭代里 producer 的 RAW，但读的是输出张量。
3. Consumer 再把卷积结果写回 `y`，盖住刚才的 `U`。这是同一块缓冲上的 WAW，也是对第 2 步的 WAR。
4. 下一次计时迭代如果不再重算，读到的是上一轮留下的输出，而不是这一轮的 `U`。
5. 评测要的最终 `y` 和中间 `U` 已经无法区分，漏写的 `U` 会被下一轮的输出伪装成写过。

这五步在编译前就停住。scratch 的名字与输出张量 `y`、输入张量、符号或 workspace 相撞时，`resolve_geometry` 报 `GeometryError`，匹配 `collides`。harness 不会给 `y` 再做一块别名，也不会把输出槽改成可复用的中间纹理。

合法的 producer/consumer 用一块独立的 scratch，不借用 `y`：

```text
iteration i
  poison / device_zero / host_written     这一轮的初值，只作用于 scratch
           |
           v
  stage 0  producer  write U              RAW 的写端
           |  barrier：buffer；有纹理时再加 texture
           v
  stage 1  consumer  read U, write y      RAW 的读端；y 是另一块缓冲
           |
           v
iteration i+1
  新的 scratch 缓冲或同一轮转组上的重新写入
  不读取 iteration i 留下的 U
```

依赖关系：

| 关系 | 在这条路径上是什么 | 实际拒绝点 |
| --- | --- | --- |
| RAW | stage 1 读 stage 0 刚写的 `U` | 两段之间的 barrier 必须覆盖写过的存储。只有 buffer 时是 buffer barrier；scratch 是纹理时，barrier 同时覆盖 buffer 和纹理。先读后写在编译前拒绝，匹配 `first used as read` |
| WAW | 同一轮里两个 stage 都写同一块 scratch，后写盖住前写 | 第一处 `uninitialized_poison` 必须声明并且反射为 `write`。反射成 `readwrite` 的 device 指针被拒绝，不能改写成 `write` |
| WAR | 后一个 stage 写、前一个 stage 已经读过同一块缓冲；借用 `y` 时输出和 `U` 互相覆盖 | 名字与 `y` 相撞直接拒绝。部分写入三轮结果一致但数值不对，判 `incorrect`。看见哨兵就跳过后续元素的 producer，在整段探针里留下未改的字，correctness 和 benchmark 都拒绝 |

这台机器上，非 const 的 `device float*` 即使只做存储，反射也是 `readwrite`。`texture2d<float, access::write>` 反射 `write`，`access::read` 反射 `read`。因此 `uninitialized_poison` 的第一次使用必须同时满足：几何声明 `write`，并且反射恰好是 `write`。声明 `write` 但反射 `readwrite` 会被拒绝，不能靠扫源码把 `readwrite` 改写成 `write`。合法的 conv2d producer 用写纹理，consumer 用读纹理；纹理多留一整行，这一行的第一个 texel 落在尾保护带上，所以越界一格仍然改到 `0xCD`。`device_zero` 和 `host_written` 的 buffer 写仍接受反射 `write` 或 `readwrite`，因为那条只写反射只约束 poison 的第一次使用。

三种初值：

| 策略 | 谁写入 | 计时窗 |
| --- | --- | --- |
| `uninitialized_poison` | 生产者每个计时迭代覆盖全部元素 | benchmark 不插入清零。参数全是 buffer，或再加一块受约束的 fp32 scratch 纹理时，空 fill 走 `native_batch`。没有纹理时 barrier 只覆盖 buffer；有纹理时同一条 barrier 同时覆盖 buffer 和纹理。correctness 的 out1/out3 填毒，out2 填零 |
| `device_zero` | 计时命令缓冲里的 fill，发生在该轮 stage 之前 | 清零算进候选时间。prepare 里免费清零不成立 |
| `host_written` | 候选 Host Program 真的执行，返回 0 才提交 span，然后才 dispatch | 非零返回失败，不留下 span，也不发射 stage。`evaluate_task` 和 portfolio 预检都要带上候选是否真有 Host Program |

正确性与安全共用一次运行。输出随 poison/zero 初值变化、改输入、或越过保护带，记 `unsafe`。三轮结果一致但和参考不符，才记 `incorrect`。conv2d 的三个负例按这个顺序归档在 `tests/backends/metal/test_conv2d_scratch_gpu.py`：

- Stage 0 在最后一个 OC 直接返回。out1/out3 的 `U` 留着 `0xFF`，out2 留着 0，`y` 跟着变。报告是 `unsafe`，原因是结果依赖写输出的初值。M2 的计划文本把这一例写成 quick 下的 `incorrect`。评测的实际顺序是先看安全：结果随初值变化就停在 `unsafe`，不会为了凑计划里的词再改判。安全门保持原样，不把这一例放宽成 `incorrect`。
- 每个 `U` 的最后一个元素写成 0。三轮一致，数值不对，报告是 `incorrect`。
- 在 `uwin` 末尾多写一个 float。保护带 `0xCD` 抓到，报告是 `unsafe`。

`F(2×2)` 与 `F(4×4)` 的合法两阶段核在 paper 的 `normal / wide_range / large / tiny / outlier / ieee` 六个 case 上都是 `ok`，包括 image 0、image 31 和 `54 = 13×4+2` 的尾部。输入或变换出现非有限值时，该 tile 改走直接卷积；`U` 本身非有限时不从 `w` 把漏写补成正确答案。

## 每轮都重算

计时循环每个迭代都发射全部激活 stage。编码一个 stage 只说明命令被提交，不能说明中间结果重新生成。

`uninitialized_poison` 在计时结束后还要过两道探针，而且 quick 模式在 correctness 全部通过、安全和非确定都为空时也会跑同一套，所以只跑 correctness 的坏核藏不过去。已经是 `incorrect` 或 `unsafe` 的结果不再探针，避免把漏写最后一个元素和越界改判成另一种状态。

1. 残留探针留下第 0 个字，把后面每个 4 字节字涂成互不相同的 signaling NaN，再发射一轮。看见 scratch 里的哨兵就跳过后续计算的 producer，会把这些字留成探针毒。算术静音后的 quiet NaN 和探针逐位不同，所以 IEEE 输入写出的 NaN 不会被当成没写。
2. 全轮探针按本轮计时的迭代数回放，至少两轮。每一轮把包括第 0 个字在内的每个字涂上本轮独有的 signaling NaN，再要求每一个字都变掉。空写和只写一部分都会留下探针毒。

探针每次都会再发射候选 stage。全部探针结束之后，correctness 对探针用过的 argset 再调用 `safety_violations()`，guard 和只读输入都在内，新的违例记 `unsafe`。benchmark 在探针结束后对每个轮转组调用未被替换的 `safety_violations()`，并保留 `guard_violations()` 和覆盖探针的结果。只读输入对照的是 `load_initial` 写下的 `_read_snapshot`，不是探针前后临时抄下的字节。计时 launch 已经改过 `x`/`w`、探针本身不再写这两个输入时，前后字节相同，仍然要报 `input was modified by the kernel`。不要在探针之后重拍这份快照。计时样本已经取完，这次检查不进入中位数。只在看见残留探针的 NaN 时才写穿尾保护带的 stage，普通轮次的保护带是完好的；检查放在探针前面会把它报成 `ok`。

不足 8 字节的缓冲没有可藏的尾部，残留探针无字可查；全轮探针仍要求那一个字发生变化。探针在计时中位数取完之后运行，也不改写已经摘下的 `workspace_probe`。`device_zero` 和 `host_written` 不参与这两道探针。

每个轮转组有自己的 `GuardedBuffer`。组与组不共享同一块 `uwin`。CPU 负例在 `tests/backends/metal/test_candidate_scratch_cpu.py`：哨兵偷懒在 correctness 和 benchmark 都被拒绝；只在残留探针出现时写穿尾保护带的 stage 同样被这两条路径拒绝；普通计时 launch 改 `x`/`w` 而覆盖探针不写这两个输入时，两个轮转组都报只读输入被改；每次都整段重写、且不碰保护带和只读输入的假核通过。

`device_zero` 的 fill 留在计时命令缓冲内。`host_written` 走真实的 native64 Host Program，返回 0 才把字节提交给设备。

## 身份链

空 scratch 不出现在文档里。因此单核、bundle、TileLang 和既有几何的 canonical 字节、`GeometryPair.identity`、bundle digest、artifact digest、native digest 保持原值。证明在 `tests/shared/test_scratch_spec.py`。P2-0、P2-H、P2-G、P3、P4 的历史 manifest 文件没有被改写。

非空 scratch 进入下面每一环，改一个字段就会得到另一个摘要，旧缓存不能命中：

1. `ExecutionGeometry.document()` 在 scratch 非空时写入 `scratch`。
2. `resolve_geometry` 做生命周期检查，结果放进 `ResolvedGeometry`。
3. `GeometryPair.candidate_resolution` 保存这份已解析几何，`GeometryPair.identity` 覆盖它。
4. `CanonicalBundle.bundle_digest` 覆盖几何文档。
5. `attach_execution_geometry` 重算 `artifact_set_digest`，native descriptor 的 `execution_geometry.sha256` 一起变。

`dispatch_signature` 在 workspace 名字后面接上 scratch 名字。空 scratch 时这个元组与原来相同。

任务包闭包因此先从 74 个文件变成 76 个：新增 `tcenv/scratch/__init__.py` 和 `tcenv/scratch/spec.py`。生产求值要判断候选有没有 Host Program，`tcenv/taskcontract/runner.py` 因此导入 `host_tiling_entry`，闭包再收入 `tcenv/hostprogram/delivery.py`、`errors.py` 和 `precheck.py`，成为 79 个文件。`tcenv/evaluate.py` 把选中那次计时的融合 GPU 中位数写入 `candidate_gpu_us_median` 之后，文件数仍是 79，摘要变为 `5512395852ea7608a6075e2b6b45dca9bd85f112210af336cf6f8f77f5fd811f`。P2-H 的 verifier 仍钉住 74 和它自己的 parent HEAD；在这棵树上重跑它会停在闭包漂移，这和 P2-F 保留 start-surface 锚点是同一类决定。新纪元是 `tests/support/golden/shared/shape-portfolio-阶段验收/P2-I/`，epoch `p2-i-scratch-1`，parent HEAD `72a9c3a08986b51a366bfc489e34038b5983dec6`。`current_prompt_sha1` 仍是 `27e2bd2c2dfe3c009b065e1e83c9e2051bed9381fce79aa36b92618c2ac0fe3c`。提示面、`experience.md` 和六 RPC / 字段号 / HMAC / BlobChannel 都没有改。

`msl-bundle` 在搜索启动时还没有候选，scratch 未知，没有 workspace 的题仍会被拒绝。投影和评测看到候选几何之后，OpSpec 声明了 workspace，或者候选 scratch 非空，二者任一成立就允许 bundle。两者都空的空壳继续拒绝。

## Metal 与 NPU

生产路径是 Apple GPU / MSL，双通道都接通：`msl-single` 的 launch 块，和 `msl-bundle` 的几何文档。清微 NPU 仍只有模拟器，`simulated=true`，`publication_allowed=false`。本阶段没有改 NPU 模拟代码，也没有跑真实 NPU。同一份 `ScratchSpec` 以后能不能映射进模拟协议，留给单独的计划。

## 向 KDA / CAKE2 借什么

可以借的只有边界：

- 候选拥有算法、切分和 stage 顺序。harness 不内置 Winograd 或 KDA 调度器。
- stage 之间用串行编码加 barrier 建立读写顺序。只有 buffer 时范围是 buffer；scratch 纹理在同一条 barrier 上加上纹理。不在 threadgroup 之间自旋。
- Host Program 返回 0 才提交。`host_written` 沿用这条。
- 形状组合仍是候选的 ShapePlanner 加独立 fallback。scratch 改变的是变体身份，不改变「没覆盖到的点走 fallback」这条。

不照搬的：

- `varlen_kda` 的 work list、`handoff_states` 和 `handoff_flags` 解决的是变长序列的因果状态。conv2d 的 `U` 是同一轮里的权重变换，不回答分段、capacity 或 continuous batching。变长合同仍以 [varlen支持计划.md](varlen支持计划.md) 为准。
- KDA 的多阶段 bundle 示例在 [fixed-kda-bundle示例.md](fixed-kda-bundle示例.md)。那份示例的 handoff workspace 是任务自己的 OpSpec workspace，不是本文的候选 scratch。
- CAKE2 的 Host Tiling 和 TilingKey 不整份搬进来。对照写在 [KDA-CAKE2-与ShapeForge的ABI设计深度对比.md](../可信harness/KDA-CAKE2-与ShapeForge的ABI设计深度对比.md)、[KDA-CAKE2-与ShapeForge的Host调度成本对比.md](../可信harness/KDA-CAKE2-与ShapeForge的Host调度成本对比.md) 和 [KDA-CAKE2-与ShapeForge的多Track交付对比.md](../可信harness/KDA-CAKE2-与ShapeForge的多Track交付对比.md)。scratch 只增加一块候选拥有的设备缓冲，不新增第二套 tiling 主键。

## 不进入合同的做法

| 做法 | 会坏在哪 | 现在的拒绝点 |
| --- | --- | --- |
| 借用输出 `y` 当 scratch | 结果和临时值叠在一起，漏写看起来像正确性噪声 | 名字与结果张量冲突，预检拒绝 |
| prepare 里只算一次 `U`，或首轮写哨兵后跳过 | 生产者耗时留在窗外，抽查一个字节变化看不出来 | correctness 和 benchmark 之后的残留探针加全轮毒化；留下探针毒，或探针执行时写穿保护带、改只读输入，即拒绝。计时阶段改过的只读输入，即使探针不再写，也对照原始快照拒绝 |
| 跨 CTA 自旋等上一块写完 | 设备上可以死锁；Metal 证据不接受这种等待 | stage 之间是同一命令缓冲里的 barrier：纯 buffer 只挡 buffer，scratch 走纹理时再挡纹理 |
| 写一块全局中间矩阵再在同一 stage 读回 | 多一次往返，常常比寄存器里融合更慢 | `REG` 路径不分配 scratch。本阶段没有把这种写法当成加速 |
| 没有收益仍分配大块 scratch | 占配额、改变身份、消费者仍支配时间 | 配额上限拒绝超额；有没有收益由同协议消融决定，harness 不替候选删掉一块合法 scratch |

## P2-I 与 conv2d 实测

验收目录是 [P2-I/verify.py](../../tests/support/golden/shared/shape-portfolio-阶段验收/P2-I/verify.py)，文件清单是 [manifest.sha256](../../tests/support/golden/shared/shape-portfolio-阶段验收/P2-I/manifest.sha256)。`verify.py` 按 P2-H 的同一把公开 master key 重录了 79 文件闭包、14 个默认域实例和 softmax 的 per-launch 证据。记录当时的配对 fresh 中位数是 0.967099× 到 1.012472×。`publication_allowed` 保持 false。历史的 P2-H、P2-G、P2-F、P3、P4 文件没有改。

conv2d 三臂都走 `confirm_candidate`，搜索种子 7，fresh 种子由这把 key 和当次题面派生，三次都不是 7。paper、fp16、benchmark，reps 15，第一轮确认即为 `confirmed`，identity 为 `match`。每臂 JSON 保留完整 subject（`search_report`、`confirmation_report`、`identity`、`expected`）。`verify.py` 用 `verify_subject_document` 重算身份，并要求 `candidate_sha256` 等于重新生成的源码摘要。搜索和确认的 `candidate_gpu_us_median` 都是同一次命令缓冲里的融合 GPU 中位数，verifier 拿摘要和 subject 逐项核对。下表的 stage 时间是同核、同网格、各自一条命令缓冲的 GPU 时钟，5 次 `gpu_s/8` 的中位数，用来拆开 stage 0 和 stage 1。它不是确认中位数。确认中位数仍是 harness 的墙钟。两阶段臂的 `uwin` 是 `texture2d<float, access::write/read>`，因为这台机器上的 device 指针反射不出 `write`。原生编码器按 buffer 槽和纹理槽分开绑定，barrier 同时覆盖两者。原始数字在 [conv2d-ablation.json](../../tests/support/golden/shared/shape-portfolio-阶段验收/P2-I/evidence/conv2d-ablation.json)。先前只留手抄摘要、丢掉 subject 的那一版留在 `conv2d-ablation-r1-summary.json`，不作为本表的证据。

| 臂 | fresh 种子 | 墙钟中位数 µs | 融合 GPU µs | cv | Torch-MPS µs | MLX µs | 相对 Torch-MPS | stage0 GPU µs | stage1 GPU µs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 单阶段重算 `U` 的 `F(2×2)` | 4067513467619784 | 17622.82 | 17576.44 | 0.00252 | 776.57 | 1012.81 | 0.04407 | — | 17450.17 |
| 两阶段 scratch `F(2×2)` | 1220837283154464 | 19480.31 | 19425.36 | 0.03674 | 772.70 | 996.24 | 0.03967 | 5.69 | 19655.26 |
| 两阶段 scratch `F(4×4)` | 2163249565235418 | 17473.25 | 17416.40 | 0.00605 | 771.26 | 978.03 | 0.04414 | 15.46 | 17304.52 |

三臂都比同一次测到的 Torch-MPS 和 MLX 慢大约 19–23 倍。生产者大约 6 到 15 微秒，消费者是全部时间。在这颗逐线程标量核上，把 `U` 写入全局再读回，比每个 tile 自己重算 `U` 更慢。`F(4×4)` 的 tile 更少，墙钟低于两臂 `F(2×2)`，仍然远慢于厂商卷积。这是机制验收和同协议消融的记录，不是 speedup 发布，也不拿来证明「scratch 一定更快」。

正确性烟囱在 `tests/backends/metal/test_conv2d_scratch_gpu.py`，矩阵与直接卷积的对照在 `tests/backends/metal/test_conv2d_winograd_cpu.py`。

上述负结果只适用于当时的逐线程标量 consumer，不能据此推出「全局 U 复用一定没有收益」。随后保留同一 ScratchSpec 生命周期，把 consumer 换成历史 interleaved-U 布局的 `simdgroup_float8x8` F(4×4) 实现后，paper 六 case（含 IEEE 和三轮初值）全部通过。关键修正是检测完整 6×6 输入 patch 和完整 `K_TOTAL` filter，而不是依赖一个会被 out1/out3 poison 的全局 flag。候选 fixture 为 `tests/support/golden/metal/fixtures/conv2d_f4_mma_scratch.metal`，正式 fresh confirmation 的搜索/fresh 墙钟分别为 602.93/617.29 µs，相对各自 Torch-MPS 为 1.2924×/1.2461×，identity `match`。独立 stage 诊断中 producer 约 17.65 µs、融合 GPU 约 552 µs，消费者仍占绝大部分；这些 stage 数字不是发布中位数。完整 subject 在 P2-I 的 `evidence/conv2d-mma-f4-confirmation.json`，旧三臂证据原样保留。同一 helper 还生成严格 A/B control：consumer 的 tiling、MMA、输出 epilogue 和 IEEE 回退不变，只把 144 次 interleaved-U 读取替换为每个 threadgroup 重算，并删除 producer/ScratchSpec。control 同样六 case 通过、identity `match`。fresh 墙钟为 1474.42 µs，对全局 U 的 617.29 µs，全局复用快 2.3885×；fresh 融合 GPU 为 1427.08 µs 对 561.25 µs，约 2.54×。control 相对自己的 Torch-MPS 只有 0.5206×。两份完整 subject 都在 P2-I 的 `evidence/conv2d-mma-f4-confirmation.json`。因此在 Apple M4 Max 的 conv2d paper 形状和这套 F(4×4) MMA 算法上，全局 U 复用有明确净收益；该结论仍不外推到其它形状、算法或生产发布。

## small 与 paper 的候选 portfolio

paper 的 `simdgroup_float8x8` F(4×4) 源码仍是 `tests/support/golden/metal/fixtures/conv2d_f4_mma_scratch.metal`，SHA-256 `5fbfea6c501f6439e5abc69a18633d1d1f6c8205759d4add39ce723c749b5be2`，227864 字节。候选树 [conv2d_portfolio/coverage.json](../../tests/support/golden/metal/conv2d_portfolio/coverage.json) 把这份源码原样放在 `variants/paper/kernel.metal`。coverage plan 只有这一个变体，guard 要求 N、H、W、C、OC、KH、KW、OH、OW 九个符号同时等于 paper。small 对不上这组符号，exact dispatch 选择独立 general fallback，原因是 `outside_candidate_coverage`。fallback 的正确性和墙钟不记成变体加速。`fallback_credits_variant` 保持 false。

small 没有专用 MMA 变体。已确认的 producer 把 C=64 放进线程号低 3 位，把 OC=128 放进随后 7 位，consumer 以 stride 2048 展开 144 个 U panel。small 是 N=4、H=W=18、C=32、OC=64、KH=KW=3、OH=OW=16。这组 panel 和 stride 常量属于另一份核。阻塞说明在 [small-mma-blocker.json](../../tests/support/golden/metal/conv2d_portfolio/small-mma-blocker.json)。

通用 fallback 自己声明 `scratch uwin fp32 36*C*OC zero`，不借用 `y`。有限 tile 读取已经写入的 U；输入或变换非有限时，该 tile 改走 fp32 直接卷积。producer 网格是 `C*OC/64`，consumer 网格是 `N*OC/64`。small 为 (32,1,1) 与 (4,1,1)，paper 为 (128,1,1) 与 (64,1,1)。把最后一个 U 元素写成 0 之后，small 的 normal case 为 `incorrect`。直接卷积没有把这条有限路径补成正确。

真实 Metal 正确性在 `tests/backends/metal/test_conv2d_shape_portfolio_gpu.py`，种子 7。portfolio 的 paper 点执行 MMA 变体，六个声明 case 和 out1/out2/out3 通过；OH=54 余 2。portfolio 的 small 点执行 fallback，六个 case 和三轮初值通过；OH=16 余 0。同一份 fallback 源码再在 paper 上单独跑一遍，六个 case 通过，覆盖 2 宽尾块。CPU 侧的 exact dispatch、重叠 guard、拿掉 fallback 和单字节改动 variant identity 在 `tests/shared/test_conv2d_shape_portfolio.py`。

搜索种子 7、reps 15、benchmark 的可重放记录是 [conv2d-portfolio-small.json](../../tests/support/golden/shared/shape-portfolio-阶段验收/P2-I/evidence/conv2d-portfolio-small.json) 和 [conv2d-portfolio-paper.json](../../tests/support/golden/shared/shape-portfolio-阶段验收/P2-I/evidence/conv2d-portfolio-paper.json)。[full-eval.json](../../tests/support/golden/metal/conv2d_portfolio/full-eval.json) 由这两份 run 派生，`publication_allowed` 为 false，`fresh_confirmation` 为 false。P2-I verifier 把候选树、helper、CPU/GPU 测试和这两份 run 列入 manifest，并重跑对应节点。重跑核对 provider、variant/fallback 身份、六案、安全和 1.10 门的方向，不要求墙钟微秒逐值相等。归档 fresh confirmation 仍是 617.29 µs、相对 Torch-MPS 1.2461×。

| 域点 | provider | 墙钟 µs | 融合 GPU µs | cv | Torch-MPS µs | MLX µs | 相对 Torch-MPS | 相对 MLX |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| small | fallback | 1264.88 | 1246.59 | 0.01775 | 19.27 | 3.18 | 0.01524 | 0.00251 |
| paper | candidate | 608.83 | 560.35 | 0.00792 | 769.97 | 984.59 | 1.2647 | 1.6172 |

paper 这次执行的 `candidate_sha256` 等于上面的 fixture。相对 Torch-MPS 的延迟比是 0.791，低于 1.10。同一协议的前一次样本是 small 1255.56 µs、延迟比 64.73，paper 612.69 µs、1.2628×；该次没有归档 run 文件，只作参考。两份样本都把 small 留在 1.10 门外、把 paper 留在门内。small fallback 相对 MLX 的延迟比是 397.94。small 的专用 MMA 性能变体仍然没有。r4 最终验收已通过：`done=true` 记录在 [P2-I/review.json](../../tests/support/golden/shared/shape-portfolio-阶段验收/P2-I/review.json)，manifest 覆盖 71 个 `DELIVERABLES`，完整 `verify.py` 为 139 CPU + 23 真实 Metal 节点，真实 Metal 重跑 paper 612.28 µs（相对 Torch-MPS 0.7897×）、small 1272.53 µs fallback（62.78×），与归档 `stable_identity` 逐字段相等。这些数字仍不是生产 speedup 发布，`publication_allowed=false`，small 没有专用 MMA，性能推广未完成。
