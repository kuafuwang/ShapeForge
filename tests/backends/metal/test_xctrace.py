"""xctrace parsing is tested on small synthetic exports (the real ones are hundreds of MB); a live capture is opt-in."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from tcenv.backends.metal import xctrace as xt

SCHEMA = "<schema name='t'>{cols}</schema>"


def table(cols: list[str], rows: str) -> str:
    """@brief 拼一个最小可解析的 trace-query-result XML，模拟 xctrace 导出的表。

    @param cols 列助记符，按导出顺序排列。
    @param rows 已拼好的 row XML 片段。
    @return 完整的 XML 文本。
    """
    c = "".join(f"<col><mnemonic>{m}</mnemonic></col>" for m in cols)
    return f"<?xml version='1.0'?><trace-query-result><node><schema name='t'>{c}</schema>{rows}</node></trace-query-result>"


def test_iter_rows_resolves_id_ref_sharing(tmp_path):
    """@brief 验证导出里首次出现的 id 单元格会被后续 ref 行复用，逐行列解析不缺值。

    @param tmp_path pytest 提供的临时目录。
    """
    rows = (
        '<row><start-time id="1" fmt="a">100</start-time><gpu-counter-name id="2" fmt="ALU">1.ALU</gpu-counter-name><percent id="3" fmt="10%">10</percent></row>'
        '<row><start-time id="4" fmt="b">200</start-time><gpu-counter-name ref="2"/><percent ref="3"/></row>'
        '<row><start-time id="5" fmt="c">300</start-time><gpu-counter-name id="6" fmt="Occ">2.Occ</gpu-counter-name><percent id="7" fmt="50%">50</percent></row>'
    )
    p = tmp_path / "t.xml"
    p.write_text(table(["start", "name", "percent-value"], rows))
    got = list(xt.iter_rows(p))
    assert [r["name"][1] for r in got] == ["ALU", "ALU", "Occ"]
    assert [r["percent-value"][0] for r in got] == ["10", "10", "50"]


def test_compute_window_ignores_render_channels(tmp_path):
    """@brief 验证计算窗口只统计 Compute 通道的区间，忽略渲染通道，返回起止时间与各段时长。

    @param tmp_path pytest 提供的临时目录。
    """
    rows = (
        '<row><start-time id="1" fmt="x">1000</start-time><duration id="2" fmt="d">500</duration><gpu-channel-name id="3" fmt="Vertex">Vertex</gpu-channel-name></row>'
        '<row><start-time id="4" fmt="x">5000</start-time><duration id="5" fmt="d">2000</duration><gpu-channel-name id="6" fmt="Compute">Compute</gpu-channel-name></row>'
        '<row><start-time id="7" fmt="x">9000</start-time><duration id="8" fmt="d">1000</duration><gpu-channel-name ref="6"/></row>'
    )
    p = tmp_path / "iv.xml"
    p.write_text(table(["start", "duration", "channel-name"], rows))
    lo, hi, durs = xt.compute_window(p)
    assert (lo, hi, durs) == (5000.0, 10000.0, [2000.0, 1000.0])


def test_counter_summary_is_time_weighted_and_window_restricted(tmp_path):
    """@brief 验证计数器汇总按持续时间加权求均值，并只统计落在给定窗口内的样本。

    @param tmp_path pytest 提供的临时目录。
    """
    rows = (
        '<row><start-time id="1" fmt="x">0</start-time><duration id="2" fmt="d">10</duration><gpu-counter-name id="3" fmt="ALU">1.ALU Utilization</gpu-counter-name><fixed-decimal id="4" fmt="v">90</fixed-decimal><percent id="5" fmt="p">90</percent></row>'
        '<row><start-time id="6" fmt="x">100</start-time><duration id="7" fmt="d">10</duration><gpu-counter-name ref="3"/><fixed-decimal id="8" fmt="v">10</fixed-decimal><percent id="9" fmt="p">10</percent></row>'
        '<row><start-time id="10" fmt="x">110</start-time><duration id="11" fmt="d">30</duration><gpu-counter-name ref="3"/><fixed-decimal id="12" fmt="v">50</fixed-decimal><percent id="13" fmt="p">50</percent></row>'
    )
    p = tmp_path / "c.xml"
    p.write_text(table(["start", "duration", "name", "value", "percent-value"], rows))
    st = xt.summarize_counters(p, (100.0, 200.0))["ALU Utilization"]  # the first row is outside the window
    assert st.samples == 2 and st.mean == pytest.approx((10 * 10 + 30 * 50) / 40)


def test_format_summary_flags_spills_and_throttling():
    """@brief 验证摘要文本会指出寄存器溢出、着色器代码体积与降频，并声明数值仅供参考。
    """
    s = xt.TraceSummary(window_s=1.0, n_dispatches=10, dispatch_us_median=5.0, dispatch_us_p10=4.0, dispatch_us_p90=6.0, gpu_active_fraction=0.9, spilled_bytes=4096, spill_events=2, shader_code_bytes=1234, thermal_states=["Fair"])
    out = xt.format_summary(s)
    assert "spilled registers" in out and "1234 bytes" in out and "throttling" in out and "indicative only" in out


def test_compute_processes_reads_the_exported_column_and_is_empty_when_missing(tmp_path):
    """@brief 验证计算通道的进程名取自导出列，缺进程列时返回空集而不是假称是自己。

    @param tmp_path pytest 提供的临时目录。
    """
    rows = (
        '<row><start-time id="1" fmt="x">1</start-time><duration id="2" fmt="d">10</duration>'
        '<gpu-channel-name id="3" fmt="Compute">Compute</gpu-channel-name><process-name id="4" fmt="p">python</process-name></row>'
        '<row><start-time id="5" fmt="x">20</start-time><duration id="6" fmt="d">10</duration>'
        '<gpu-channel-name ref="3"/><process-name id="7" fmt="p">Cursor Helper</process-name></row>'
    )
    p = tmp_path / "iv.xml"
    p.write_text(table(["start", "duration", "channel-name", "process"], rows))
    assert xt.compute_processes(p) == {"python", "Cursor Helper"}
    # no process column at all: unknown, not "ours"
    p2 = tmp_path / "iv2.xml"
    p2.write_text(table(
        ["start", "duration", "channel-name"],
        '<row><start-time id="1" fmt="x">1</start-time><duration id="2" fmt="d">10</duration>'
        '<gpu-channel-name id="3" fmt="Compute">Compute</gpu-channel-name></row>',
    ))
    assert xt.compute_processes(p2) == set()


def test_an_absent_table_is_unknown_rather_than_a_negative():
    """@brief 验证模板未记录的表不会被说成无溢出或无降频，而是明确标为未记录。

    默认模板可能不记录溢出、代码体积与热状态。缺失的表不能在模型读到的文本里变成
    “无溢出 / 未降频”。
    """
    s = xt.TraceSummary(
        window_s=1.0, n_dispatches=10, dispatch_us_median=5.0, gpu_active_fraction=None,
        spilled_bytes=0, spill_events=0, shader_code_bytes=None, thermal_states=[],
        tables={"spills": False, "gpu_state": False, "shaders": False, "thermal": False}, window_processes=[],
    )
    out = xt.format_summary(s)
    assert "not recorded by this trace template" in out
    assert "machine-wide" in out
    text = "\n".join(xt.advice(s)).lower()
    assert "no register spills" not in text and "not throttled" not in text
    assert "unknown rather than cleared" in text


def test_a_recorded_empty_table_means_none_happened():
    """@brief 验证表确实被记录但内容为空时，摘要直接写 0 事件，不再加未记录或范围存疑的限定。
    """
    s = xt.TraceSummary(
        window_s=1.0, n_dispatches=10, dispatch_us_median=5.0, gpu_active_fraction=0.9,
        spilled_bytes=0, spill_events=0, shader_code_bytes=4096, thermal_states=["Nominal"],
        tables={"spills": True, "gpu_state": True, "shaders": True, "thermal": True}, window_processes=["python"],
    )
    out = xt.format_summary(s)
    assert "register spills: 0 events, 0 bytes" in out
    assert "not recorded" not in out
    assert "scope:" not in out


def test_record_refuses_unbounded_loops_without_starting_anything(tmp_path):
    """@brief 验证带无界循环的 kernel 在任何采集启动之前就被拒绝。

    @param tmp_path pytest 提供的临时目录。
    """
    with pytest.raises(xt.TraceError, match="refused"):
        xt.record("relu", "fp16", "small", "kernel void relu_kernel() { while (true) {} }", tmp_path)


# ---- GPU counter samples (gpu-counter-info / gpu-counter-value of a trace recorded with a counter template) --------------------------------


def instruments_export(cols: list[str], rows: list[str]) -> str:
    """@brief 生成形状接近真实导出的 XML：首行紧跟 schema，其余每行单独一行。

    @param cols 列助记符。
    @param rows 已拼好的 row XML 片段列表。
    @return 合法的 trace-query-result XML 文本。
    """
    c = "".join(f"<col><mnemonic>{m}</mnemonic></col>" for m in cols)
    return f"<?xml version='1.0'?>\n<trace-query-result>\n<node><schema name='t'>{c}</schema>" + "\n".join(rows) + "\n</node>\n</trace-query-result>\n"


INFO_COLS = ["timestamp", "counter-id", "name", "max-value", "accelerator-id", "description", "group-index", "type"]


def info_row(cid: int, name: str, kind: str, desc: str) -> str:
    """@brief 拼一行 gpu-counter-info 数据，描述一个硬件计数器。

    @param cid 计数器 id。
    @param name 计数器名称。
    @param kind 计数器类型，例如 Percentage 或 Value。
    @param desc Apple 提供的描述文本，单位由它推导。
    @return 单行 counter-info 的 XML 片段。
    """
    return (
        f'<row><event-time fmt="t">0</event-time><uint32 fmt="{cid}">{cid}</uint32><gpu-counter-name fmt="{name}">{name}</gpu-counter-name><uint64>100</uint64>'
        f'<uint64>1</uint64><string>{desc}</string><uint32>1</uint32><string>{kind}</string></row>'
    )


def write_counter_fixture(tmp_path):
    """@brief 写一套合成的计数器导出（定义、样本、区间），含 id/ref 共享与空闲样本等边界情况。

    @param tmp_path pytest 提供的临时目录。
    @return (counter-info, counter-value, interval) 三个 XML 路径。
    """
    info = tmp_path / "info.xml"
    info.write_text(instruments_export(INFO_COLS, [
        info_row(0, "RT Unit Active", "Percentage", "Percentage of cores where the raytracing unit is active."),
        info_row(3, "Kernel Occupancy", "Percentage", "Percentage of shader core resources occupied by the compute kernel"),
        info_row(54, "Compute SIMD Groups Inflight", "Value", "The average number of simdgroups running concurrently per shader core."),
        info_row(57, "Threadgroup Memory L1 Read Bandwidth", "Value", "Threadgroup Memory L1 read bandwidth in GiB/s"),
        info_row(77, "GPU Read Bandwidth", "Value", "Measures how much memory, in gigabytes per second, are read by the GPU"),
    ]))
    # the first row defines ids that later rows refer to; id 5 is defined in a column that is never read directly (sample index) and is later used as the counter id 0
    rows = [
        '<row><event-time id="1" fmt="t">1000</event-time><uint32 id="2" fmt="3">3</uint32><fixed-decimal id="3" fmt="v">10</fixed-decimal><uint64 id="4" fmt="a">7</uint64><uint32 id="5" fmt="0">0</uint32></row>',
        '<row><event-time ref="1"/><uint32 id="6" fmt="77">77</uint32><fixed-decimal id="7" fmt="v">200</fixed-decimal><uint64 ref="4"/><uint32 ref="5"/></row>',
        '<row><event-time ref="1"/><uint32 ref="5"/><fixed-decimal id="8" fmt="v">4</fixed-decimal><uint64 ref="4"/><uint32 ref="5"/></row>',
        '<row><event-time id="9" fmt="t">2000</event-time><uint32 ref="2"/><fixed-decimal id="10" fmt="v">30</fixed-decimal><uint64 ref="4"/><uint32 ref="5"/></row>',
        '<row><event-time id="11" fmt="t">3000</event-time><uint32 ref="2"/><fixed-decimal id="12" fmt="v">50</fixed-decimal><uint64 ref="4"/><uint32 ref="5"/></row>',
        '<row><event-time id="13" fmt="t">5000</event-time><uint32 ref="2"/><fixed-decimal id="14" fmt="v">99</fixed-decimal><uint64 ref="4"/><uint32 ref="5"/></row>',  # idle GPU
    ]
    vals = tmp_path / "vals.xml"
    vals.write_text(instruments_export(["timestamp", "counter-id", "value", "accelerator-id", "sample-index"], rows))
    ivs = tmp_path / "iv.xml"
    ivs.write_text(instruments_export(["start", "duration", "channel-name"], [
        '<row><start-time id="1" fmt="x">900</start-time><duration id="2" fmt="d">1200</duration><gpu-channel-name id="3" fmt="Compute">Compute</gpu-channel-name></row>',
        '<row><start-time id="4" fmt="x">2000</start-time><duration id="5" fmt="d">500</duration><gpu-channel-name ref="3"/></row>',  # overlaps the first: merged
        '<row><start-time id="6" fmt="x">2900</start-time><duration id="7" fmt="d">200</duration><gpu-channel-name ref="3"/></row>',
        '<row><start-time id="8" fmt="x">4900</start-time><duration id="9" fmt="d">300</duration><gpu-channel-name id="10" fmt="Vertex">Vertex</gpu-channel-name></row>',  # not our kernel
    ]))
    return info, vals, ivs


def test_dispatch_spans_merge_overlaps_and_keep_compute_only(tmp_path):
    """@brief 验证 dispatch 区间会合并重叠段，且只保留 Compute 通道。

    @param tmp_path pytest 提供的临时目录。
    """
    _, _, ivs = write_counter_fixture(tmp_path)
    assert xt.dispatch_spans(ivs) == [(900.0, 2500.0), (2900.0, 3100.0)]


def test_counter_defs_take_the_unit_from_apples_description(tmp_path):
    """@brief 验证计数器单位从 Apple 的描述文本推导，而不是写死。

    @param tmp_path pytest 提供的临时目录。
    """
    info, _, _ = write_counter_fixture(tmp_path)
    d = xt.counter_defs(info)
    assert d[3] == ("Kernel Occupancy", "%") and d[0] == ("RT Unit Active", "%")
    assert d[54] == ("Compute SIMD Groups Inflight", " simdgroups/core")
    assert d[57][1] == " GiB/s" and d[77][1] == " GB/s"


def test_counter_values_are_read_only_inside_our_dispatches_and_resolve_shared_cells(tmp_path):
    """@brief 验证计数器样本只取 dispatch 区间内的值，并正确解析只在未读列定义过的共享 id 单元。

    @param tmp_path pytest 提供的临时目录。
    """
    info, vals, ivs = write_counter_fixture(tmp_path)
    cs = xt.summarize_counter_values(vals, xt.counter_defs(info), xt.dispatch_spans(ivs))
    occ = cs["Kernel Occupancy"]  # 10, 30, 50 inside the spans; the 99 at t=5000 is the idle GPU
    assert (occ.samples, occ.p50, occ.mean, occ.max, occ.unit) == (3, 30.0, 30.0, 50.0, "%")
    assert cs["GPU Read Bandwidth"].p50 == 200.0 and cs["GPU Read Bandwidth"].unit == " GB/s"
    assert cs["RT Unit Active"].p50 == 4.0  # the counter id comes from a cell that was only defined in an unused column of the first row


def test_typical_value_is_the_median_not_the_mean_with_ramp_samples():
    """@brief 验证带爬升与回落样本时典型值取中位数，不会被两端斜坡拉低均值。
    """
    # a dispatch of 12 samples: 0.3, 1.3 ramp up, ten at 1.6, 1.0, 0.3 ramp down
    xs = sorted([0.3, 1.3] + [1.6] * 10 + [1.0, 0.3])
    c = xt.CounterStat(mean=sum(xs) / len(xs), max=xs[-1], p90=xs[int(0.9 * (len(xs) - 1))], samples=len(xs), p50=xs[(len(xs) - 1) // 2])
    assert c.p50 == 1.6 and c.mean < 1.4


def named(d):
    """@brief 把“名字到数值”的字典转成名字带空格、各分位数相等的计数器统计表。

    @param d 以双下划线分隔单词的名字到数值的映射。
    @return 计数器名到 CounterStat 的映射，供建议文本测试使用。
    """
    return {n.replace("__", " "): xt.CounterStat(mean=v, max=v, p90=v, samples=10, p50=v) for n, v in d.items()}


def test_advice_names_the_saturated_limiter_and_the_idle_subpipelines():
    """@brief 验证建议文本点出最高的限制器及其对应管线，并说明其他 ALU 子管线还有空闲产能。
    """
    c = named({"F32 Limiter": 95.0, "Instruction Throughput Limiter": 40.0, "Integer and Conditional Limiter": 10.0, "Kernel Occupancy": 60.0})
    out = " ".join(xt.advice(xt.TraceSummary(dispatch_us_median=500.0, gpu_active_fraction=0.99, counters=c)))
    assert "highest limiter is F32 Limiter at 95%" in out and "FP32 ALUs" in out
    assert "other ALU sub-pipelines" in out and "idle capacity" in out
    assert "latency" not in out


def test_advice_reads_all_low_limiters_as_latency_bound_and_mentions_the_fixed_grid():
    """@brief 验证所有限制器都很低且占用率很低时，建议判为延迟受限并指出 launch 契约固定了网格。
    """
    c = named({"Instruction Throughput Limiter": 4.0, "F32 Limiter": 4.0, "Kernel Occupancy": 1.6})
    out = " ".join(xt.advice(xt.TraceSummary(dispatch_us_median=578.0, gpu_active_fraction=0.99, counters=c)))
    assert "no limiter is saturated" in out and "kernel occupancy is 2%" in out and "latency" in out and "launch contract fixes it" in out


def test_advice_flags_l1_eviction_and_blurred_short_dispatches():
    """@brief 验证建议会指出 L1 逐出率偏高、占用率目标值，以及过短 dispatch 使指标模糊不清。
    """
    c = named({"Last Level Cache Limiter": 70.0, "L1 Eviction Rate": 80.0, "Occupancy Manager Target": 45.0, "Kernel Occupancy": 40.0})
    out = xt.advice(xt.TraceSummary(dispatch_us_median=60.0, gpu_active_fraction=0.99, counters=c))
    joined = " ".join(out)
    assert "L1 eviction rate is 80%" in joined and "target of 45%" in joined
    assert "only 60 us" in joined and "blurred" in joined


def test_format_summary_puts_limiters_first_with_units_and_typical_values():
    """@brief 验证摘要把限制器排在其他计数器之前，并带上单位与典型值。
    """
    c = named({"F32 Limiter": 70.0, "Instruction Throughput Limiter": 30.0, "Kernel Occupancy": 25.0, "GPU Read Bandwidth": 300.0})
    c["GPU Read Bandwidth"].unit = " GB/s"
    out = xt.format_summary(xt.TraceSummary(window_s=1.0, n_dispatches=10, dispatch_us_median=500.0, dispatch_us_p10=450.0, dispatch_us_p90=550.0, gpu_active_fraction=0.9, counters=c))
    lines = out.splitlines()
    first = next(i for i, l in enumerate(lines) if "F32 Limiter" in l)
    assert first < next(i for i, l in enumerate(lines) if "Kernel Occupancy" in l)
    assert "typical    70.0%" in lines[first] and "300.0 GB/s" in out


@pytest.mark.slow
@pytest.mark.skipif(os.environ.get("TCENV_RUN_XCTRACE") != "1", reason="live Instruments capture (tens of seconds, heavy); set TCENV_RUN_XCTRACE=1")
def test_live_capture_of_a_safe_kernel(tmp_path):
    """@brief 验证一次真实 Instruments 采集能拿到 dispatch 与代码体积且无寄存器溢出（默认按环境变量跳过）。

    @param tmp_path pytest 提供的临时目录。
    """
    from ...support.v1_tasks import V1_TASK, v1_task

    task = v1_task()
    summ = xt.profile_kernel(V1_TASK, "fp16", "small", task.naive_msl, workdir=tmp_path)
    assert summ.n_dispatches > 10 and summ.shader_code_bytes and summ.spilled_bytes == 0
    assert "register spills" in xt.format_summary(summ)
