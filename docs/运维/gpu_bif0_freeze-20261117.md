# GPU BIF0 页错误反复 reset GPU，最终冻结整机

日期：2026-11-17

> 时间线复核（2026-11-26）：[gpu_bif0_freeze-timeline-20261126.md](gpu_bif0_freeze-timeline-20261126.md)。原文「等了约 10 分钟」与「14:48 重启」不能同时成立；复核只改这一句的读法，因果链与处置结论不变。

## 1. 现象

2026-11-17 14:45 前后，这台机器**整个图形界面冻结**：鼠标不动、键盘无响应、SSH 之外的本地操作全部失效。等了约 10 分钟没有恢复，只能长按电源强制关机。

冻结前机器已连续运行 3 天 15 小时 41 分 47 秒（`Time Awake Since Boot: 315707s`）。

重启后（14:48）4 分钟内 load average 升到 `85.11 94.73 45.12`，因为同一批任务被立刻重新拉起。

事后回看，**这天 GPU 一共崩了 5 次，机器不是"突然"卡住的**：

| 时间 | restart_reason | signature |
| --- | --- | --- |
| 10:03:41 | BIF0 page fault | 563 |
| 11:55:51 | BIF0 page fault | 563 |
| 12:51:01 | BIF0 page fault | 563 |
| 14:28:07 | BIF0 page fault | 563 |
| 14:29:54 | BIF0 page fault | 563 |
| **14:45:11** | WindowServer 看门狗超时 | — |
| **14:45:21** | WindowServer 看门狗超时 | — |
| **14:45:51** | WindowServer 崩溃 | — |

往前追还有 3 次（`/Library/Logs/DiagnosticReports/Retired/`），累计 8 次，**全部由 `Python` 进程触发**：

| 时间 | restart_reason | signature |
| --- | --- | --- |
| 11-14 20:18:30 | CDM Kill timeout | 659 |
| 11-16 07:47:55 | CDM Kill timeout | 659 |
| 11-16 09:25:16 | BIF0 page fault | 563 |

## 2. 证据

### 2.1 WindowServer 是被看门狗杀掉的，不是自己崩的

```
/Library/Logs/DiagnosticReports/
  WindowServer-2026-11-17-144511.ips
  WindowServer-2026-11-17-144551.ips
  WindowServer_2026-11-17-144521_<host>.userspace_watchdog_timeout.spin
  WindowServer_2026-11-17-144554_<host>.userspace_watchdog_timeout.spin
```

`WindowServer-2026-11-17-144551.ips` 的 `termination` 字段原文：

```
flags: 0
indicator: monitoring timed out for service
code: 1
namespace: WATCHDOG
details:
  (1 monitored services unresponsive): checkin with service: WindowServer
  (1 induced crashes) returned not alive with context:
  unresponsive work processor(s): WindowServer main thread
  80 seconds since last successful checkin,
  31546 total successful checkins since 315544 seconds ago,
  39 seconds since last crashed by watchdogd
```

读法：**WindowServer 主线程连续 80 秒没有向看门狗报到**；它在 39 秒前已经被看门狗杀过一次（`1 induced crashes`），这次是第二次。两次超时对应上面 14:45:11 和 14:45:21 两个 spin 文件。

### 2.2 主线程卡在 GPU 驱动里，不是在等某个 app

`WindowServer_2026-11-17-144554_*.userspace_watchdog_timeout.spin` 里，WindowServer 的 `ws_main_thread` **12/12 采样全部落在同一条栈上**：

```
Thread 0x1535  DispatchQueue "com.apple.SkyLight.mtl_submit"  Thread name "ws_main_thread"
  dyld
  WindowServer
  SkyLight → SkyLight → SkyLight → SkyLight → SkyLight → SkyLight
  SkyLight → SkyLight → SkyLight
  Metal → libdispatch → libdispatch → Metal → Metal
  IOGPU → IOGPU → IOKit
  kernel.release.t6041 → kernel.release.t6041 → kernel.release.t6041 → kernel.release.t6041
  IOGPUFamily → IOGPUFamily → IOGPUFamily → IOGPUFamily → IOGPUFamily
  AGXG16X → AGXG16X                    ← Apple GPU 驱动，卡在这里
  IOGPUFamily → kernel.release.t6041 → kernel.release.t6041
```

即：一个 Metal command buffer 提交下去，**GPU 侧不返回**，主线程在内核态等 GPU 完成，等了 80 秒。同一份 spindump 里其他线程（CoreAnimation display / cursor）都正常在等 mach_msg，只有主线程被 GPU 钉住。

### 2.3 GPU 侧真正发生的事：读了地址 0

`gpuEvent-Python-2026-11-17-142954.ips`（`bug_type: 284`，GPU 事件报告）关键字段：

```json
"process_name": "Python",
"analysis": {
  "restart_reason": 3,
  "restart_reason_desc": "BIF0 page fault",
  "signature": 563,
  "guilty_dm": 3,
  "bif0_fault": {
    "pm_protect": false,
    "is_read": true,
    "sideband": 82,
    "level": 2,
    "requestor": 113,
    "address": 0
  },
  "command_buffer_trace_id": 63160953943
}
```

`BIF0` 是 GPU 的总线接口（Bus Interface）。这条记录的含义是：

> **GPU 执行某条命令时，去读地址 `0x0`，触发总线页错误，GPU 硬件层被 reset。**

`address: 0` + `is_read: true` 是这类故障里信息量最大的两个字段。GPU 读 0 地址基本只有几种可能：

1. 某个 `[[buffer(n)]]` 参数在 host 侧根本没绑定，或绑成了 `nil`；
2. host 侧绑定的 buffer 长度算小了，kernel 索引越界到 0 附近；
3. 自动生成的 MSL 里 `[[buffer(n)]]` 的 n 和 host 侧 `setBuffer(_:offset:index:)` 的 index 错位——**这是 `conv2d-fp16-small-init.metal` 这种自动生成 kernel 最容易犯的错**；
4. fp16 下 `device const half*` 的偏移/长度按元素数还是字节数算混了。

另一类故障 `CDM Kill timeout`（signature 659，2 次）是 compute 队列层面的超时，属于 kernel 自己跑太久/不退出，性质不同但同样会导致 GPU reset。

## 3. 谁在触发

`gpuEvent` 只记录 `process_name: Python`，不记录命令行。事后按时间点核对进程表，当时在跑的是本工程的 kernel 搜索：

```
uv run python -m taic.cli run conv2d --dtype fp16 --size small --budget 5 \
    --config /tmp/taic-control.toml
```

上级是 hmz flow：

```
hmz exec -f ai_lowerling/taic/flows/kernel_once \
    -a assistant=pi/deepseek/deepseek-flash:high \
    -p submit_name=conv2d-fp16-small-init.metal \
    -p seed_name=kernel.metal \
    -p budget.duration=30m \
    -p turn_output_tokens=1666666 -p turn_minutes=25
```

事发时**并发 20 个** `taic.cli run conv2d`（`ps -Ao command= | grep -c "taic.cli run conv2d"` 返回 20），21 个 hmz 进程。

## 4. 原因（区分已证实与推断）

**已证实：**

1. GPU 发生 `BIF0 page fault`（读地址 0），触发 GPU 硬件 reset。8 次故障里 6 次是这一类，指纹完全一致（signature 563）。
2. 全部故障的 guilty process 是 `Python`，即本工程的 kernel 执行路径。
3. GPU reset 期间 WindowServer 主线程阻塞在 `IOGPU / AGXG16X` 提交路径上 80 秒。
4. 看门狗判定 WindowServer 失联，杀掉它 → 整个 GUI 冻结 → 只能强制关机。
5. `CDM Kill timeout`（2 次）是另一类故障，成因不同（kernel 超时未退出）。

**未证实（需要复现才能定位）：**

- 具体是哪个候选、哪个 buffer 导致的 0 地址访问。可用 `command_buffer_trace_id` 对账：最后一次 BIF0 是 `63160953943`，如果搜索日志里有按 trace_id 记录的提交记录，可以直接定位到具体轮次。
- 是否与 macOS 升级有关。冻结前系统是 **macOS 27.2.1 (26B131)**，而 2026-10-28 的记录里还是 26.7.1 (25G241)。无法排除驱动回归，但 signature 稳定复现更像是 kernel 本身的问题。
- 8 次 GPU reset 是否造成了不可逆硬件损伤。目前没有证据，但反复 reset 不应该被当成常态。

## 5. 对本工程的硬约束

以下是这次事故推出的、**写核与本机执行时必须遵守**的约束：

1. **本机首次执行任何新候选 kernel，必须开 Metal 验证层。** 验证层会在 CPU 侧拦截非法 buffer 访问并报错，不会打崩 GPU：

   ```bash
   MTL_DEBUG_LAYER=1 MTL_DEBUG_LAYER_ERROR_MODE=0 \
   uv run python -m taic.cli run conv2d --dtype fp16 --size small --budget 1
   ```

   代价是慢，但换来的是「报错」而不是「整机冻结」。

2. **本机并发压到 1–2。** 事发时 20 路并发，等于把单次故障概率放大 20 倍，也让一次 reset 同时波及 20 个在跑的进程。

3. **`budget.duration` 先压到 1–2 分钟。** 30 分钟对一个跑飞的 kernel 来说太长，能连着触发多次 GPU reset。

4. **每次 GPU 故障后先停，不要立刻重跑。** 这台机器在重启后 4 分钟内就被重新拉满 load 90，等于刚恢复就再次进入故障窗口。

5. **候选 kernel 侧至少要检查这四件事**，尤其 fp16 路径：
   - 每个 `[[buffer(n)]]` 都有对应的 `setBuffer(_:offset:index:)`，没有漏绑；
   - MSL 里的 n 与 host 侧 index 一一对应，没有错位；
   - buffer 长度按字节算，fp16 下元素数要乘 2；
   - 越界索引被 clamp 或显式拒绝，而不是让 GPU 去读。

6. **`CDM Kill timeout` 是独立问题**：kernel 必须有可终止的循环条件和边界检查，不能指望外部超时兜底。

## 6. 自查命令

```bash
# 有没有新的 GPU 故障
ls -1t /Library/Logs/DiagnosticReports/gpuEvent-*.ips | head -3

# 看某一份的判定原因
python3 -c "
import json,sys
s=open(sys.argv[1],encoding='utf-8').read(); d=json.loads(s[s.index(chr(10)):])
a=d['analysis']
print('process :', d.get('process_name'))
print('reason  :', a.get('restart_reason_desc'), '| signature', a.get('signature'))
print('bif0    :', a.get('bif0_fault'))
" /Library/Logs/DiagnosticReports/gpuEvent-Python-*.ips | tail -20

# 这台机器有没有被看门狗杀过 WindowServer
ls -1 /Library/Logs/DiagnosticReports/*watchdog_timeout.spin 2>/dev/null

# 当前并发
ps -Ao command= | grep -c "[t]aic.cli run"
```

判据：**只要 `gpuEvent-*.ips` 出现新的、且 `process_name` 是 `Python`，就说明本工程的 kernel 又把 GPU 打崩了，应立即停止当前搜索。**

## 7. 这次事故没解释的现象

一并记录，避免下次误判：

- **企业微信 IPCHelper 有线程卡了 17.6 小时。** spindump（同一份 14:45:21 的 spin）里，`88L2Q4487U.com.tencent.WeWorkMac.IPCHelper [19911]` 的主线程停在
  `ExtensionFoundation → BoardServices → libxpc → xpc reply → mach_msg`，
  状态是 `last ran 63539.192s ago`，即 17.6 小时没被调度过。是同步 XPC 调用不返回。与本次冻结无因果关系（时间对不上），但企业微信该重启。
- **2026-11-17 08:54 有一次 JetsamEvent**（`JetsamEvent-2026-11-17-085414.ips`），内存压力下的进程查杀。128GB 机器上出现 Jetsam 本身就值得单独看一眼，本次未深究。
- **重启后 `mds_stores` 长时间占 90%+**：macOS 升级到 27.2.1 后 Spotlight 重建索引，属预期，会自动结束，不要当成故障处理。
