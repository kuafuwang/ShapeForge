"""The baseline read probe reports its six counters and a cheaper stamp-short-circuited warm path."""

from __future__ import annotations

from experiments.read_probe import measure


def test_read_probe_reports_counters_and_stamp_short_circuit():
    """@brief 探针给出六类计数，且带 stamp 的暖路径确实比冷路径便宜。"""
    rows = measure(5)
    by_key = {(row["scenario"], row["temperature"]): row for row in rows}
    expected = {
        ("first_screen", "cold"), ("first_screen", "warm"),
        ("open_run", "cold"), ("open_run", "warm"),
        ("open_reviews", "cold"), ("open_reviews", "warm"),
        ("open_turns_older", "cold"), ("ai_bytes", "cold"),
    }
    assert expected <= set(by_key)

    for row in rows:
        for field in ("requests", "endpoint_bytes", "file_reads", "bytes_read", "stat_calls", "json_decode_calls"):
            assert row[field] >= 0, (row["scenario"], field)

    cold = by_key[("open_run", "cold")]
    warm = by_key[("open_run", "warm")]
    assert cold["requests"] == warm["requests"] == 4
    assert cold["file_reads"] > 0
    assert warm["endpoint_bytes"] < cold["endpoint_bytes"]
    assert warm["file_reads"] < cold["file_reads"]

    # §5.6/§5.7：首屏的预算与总览读窄投影，不再解码 ``row_json``。
    assert by_key[("first_screen", "cold")]["row_json_rows"] == 0
    assert by_key[("first_screen", "warm")]["row_json_rows"] == 0
    assert by_key[("open_reviews", "warm")]["file_reads"] == 0


def test_read_probe_counts_open_based_reads_by_bytes(tmp_path):
    """@brief 探针覆盖 ``Path.open`` + read/readline/迭代：已知字节数的读入必须被精确计入。

    @param tmp_path pytest 提供的临时目录。
    """
    import json

    from experiments.read_probe import Counters, _instrument
    from taic.reads import iter_jsonl, tail_text

    path = tmp_path / "probe.jsonl"
    path.write_text("".join(json.dumps({"t": index, "what": "y" * 100}) + "\n" for index in range(100)),
                    encoding="utf-8")
    size = path.stat().st_size
    counters = Counters()
    restore = _instrument(counters)
    try:
        seen = sum(1 for _ in iter_jsonl(path))
        tail = tail_text(path, limit=200, trim_partial=False)
    finally:
        restore()
    assert seen == 100 and len(tail.encode("utf-8")) == 200
    assert counters.file_reads >= 2, "open 型读取也必须计入 file_reads"
    assert counters.bytes_read == size + 200
