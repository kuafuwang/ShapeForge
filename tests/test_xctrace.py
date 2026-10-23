"""xctrace parsing is tested on small synthetic exports (the real ones are hundreds of MB); a live capture is opt-in."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from tcenv.backends.metal import xctrace as xt

SCHEMA = "<schema name='t'>{cols}</schema>"


def table(cols: list[str], rows: str) -> str:
    c = "".join(f"<col><mnemonic>{m}</mnemonic></col>" for m in cols)
    return f"<?xml version='1.0'?><trace-query-result><node><schema name='t'>{c}</schema>{rows}</node></trace-query-result>"


def test_iter_rows_resolves_id_ref_sharing(tmp_path):
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
    s = xt.TraceSummary(window_s=1.0, n_dispatches=10, dispatch_us_median=5.0, dispatch_us_p10=4.0, dispatch_us_p90=6.0, gpu_active_fraction=0.9, spilled_bytes=4096, spill_events=2, shader_code_bytes=1234, thermal_states=["Fair"])
    out = xt.format_summary(s)
    assert "spilled registers" in out and "1234 bytes" in out and "throttling" in out and "indicative only" in out


def test_record_refuses_unbounded_loops_without_starting_anything(tmp_path):
    with pytest.raises(xt.TraceError, match="refused"):
        xt.record("relu", "fp16", "small", "kernel void relu_kernel() { while (true) {} }", tmp_path)


# ---- GPU counter samples (gpu-counter-info / gpu-counter-value of a trace recorded with a counter template) --------------------------------


def instruments_export(cols: list[str], rows: list[str]) -> str:
    """Shaped like a real export: the first row follows the schema on the same line, every other row has its own line."""
    c = "".join(f"<col><mnemonic>{m}</mnemonic></col>" for m in cols)
    return f"<?xml version='1.0'?>\n<trace-query-result>\n<node><schema name='t'>{c}</schema>" + "\n".join(rows) + "\n</node>\n</trace-query-result>\n"


INFO_COLS = ["timestamp", "counter-id", "name", "max-value", "accelerator-id", "description", "group-index", "type"]


def info_row(cid: int, name: str, kind: str, desc: str) -> str:
    return (
        f'<row><event-time fmt="t">0</event-time><uint32 fmt="{cid}">{cid}</uint32><gpu-counter-name fmt="{name}">{name}</gpu-counter-name><uint64>100</uint64>'
        f'<uint64>1</uint64><string>{desc}</string><uint32>1</uint32><string>{kind}</string></row>'
    )


def write_counter_fixture(tmp_path):
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
    _, _, ivs = write_counter_fixture(tmp_path)
    assert xt.dispatch_spans(ivs) == [(900.0, 2500.0), (2900.0, 3100.0)]


def test_counter_defs_take_the_unit_from_apples_description(tmp_path):
    info, _, _ = write_counter_fixture(tmp_path)
    d = xt.counter_defs(info)
    assert d[3] == ("Kernel Occupancy", "%") and d[0] == ("RT Unit Active", "%")
    assert d[54] == ("Compute SIMD Groups Inflight", " simdgroups/core")
    assert d[57][1] == " GiB/s" and d[77][1] == " GB/s"


def test_counter_values_are_read_only_inside_our_dispatches_and_resolve_shared_cells(tmp_path):
    info, vals, ivs = write_counter_fixture(tmp_path)
    cs = xt.summarize_counter_values(vals, xt.counter_defs(info), xt.dispatch_spans(ivs))
    occ = cs["Kernel Occupancy"]  # 10, 30, 50 inside the spans; the 99 at t=5000 is the idle GPU
    assert (occ.samples, occ.p50, occ.mean, occ.max, occ.unit) == (3, 30.0, 30.0, 50.0, "%")
    assert cs["GPU Read Bandwidth"].p50 == 200.0 and cs["GPU Read Bandwidth"].unit == " GB/s"
    assert cs["RT Unit Active"].p50 == 4.0  # the counter id comes from a cell that was only defined in an unused column of the first row


def test_typical_value_is_the_median_not_the_mean_with_ramp_samples():
    # a dispatch of 12 samples: 0.3, 1.3 ramp up, ten at 1.6, 1.0, 0.3 ramp down
    xs = sorted([0.3, 1.3] + [1.6] * 10 + [1.0, 0.3])
    c = xt.CounterStat(mean=sum(xs) / len(xs), max=xs[-1], p90=xs[int(0.9 * (len(xs) - 1))], samples=len(xs), p50=xs[(len(xs) - 1) // 2])
    assert c.p50 == 1.6 and c.mean < 1.4


def named(d):
    return {n.replace("__", " "): xt.CounterStat(mean=v, max=v, p90=v, samples=10, p50=v) for n, v in d.items()}


def test_advice_names_the_saturated_limiter_and_the_idle_subpipelines():
    c = named({"F32 Limiter": 95.0, "Instruction Throughput Limiter": 40.0, "Integer and Conditional Limiter": 10.0, "Kernel Occupancy": 60.0})
    out = " ".join(xt.advice(xt.TraceSummary(dispatch_us_median=500.0, gpu_active_fraction=0.99, counters=c)))
    assert "highest limiter is F32 Limiter at 95%" in out and "FP32 ALUs" in out
    assert "other ALU sub-pipelines" in out and "idle capacity" in out
    assert "latency" not in out


def test_advice_reads_all_low_limiters_as_latency_bound_and_mentions_the_fixed_grid():
    c = named({"Instruction Throughput Limiter": 4.0, "F32 Limiter": 4.0, "Kernel Occupancy": 1.6})
    out = " ".join(xt.advice(xt.TraceSummary(dispatch_us_median=578.0, gpu_active_fraction=0.99, counters=c)))
    assert "no limiter is saturated" in out and "kernel occupancy is 2%" in out and "latency" in out and "launch contract fixes it" in out


def test_advice_flags_l1_eviction_and_blurred_short_dispatches():
    c = named({"Last Level Cache Limiter": 70.0, "L1 Eviction Rate": 80.0, "Occupancy Manager Target": 45.0, "Kernel Occupancy": 40.0})
    out = xt.advice(xt.TraceSummary(dispatch_us_median=60.0, gpu_active_fraction=0.99, counters=c))
    joined = " ".join(out)
    assert "L1 eviction rate is 80%" in joined and "target of 45%" in joined
    assert "only 60 us" in joined and "blurred" in joined


def test_format_summary_puts_limiters_first_with_units_and_typical_values():
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
    from tasks import get_task

    summ = xt.profile_kernel("relu", "fp16", "small", get_task("relu", "fp16", "small").naive_msl, workdir=tmp_path)
    assert summ.n_dispatches > 10 and summ.shader_code_bytes and summ.spilled_bytes == 0
    assert "register spills" in xt.format_summary(summ)
