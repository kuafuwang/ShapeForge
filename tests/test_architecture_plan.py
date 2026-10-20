"""Plan ledger, experience hints, contract screen, search_ab skip, and bootstrap interval. No GPU and no model."""

from __future__ import annotations

from pathlib import Path

from experiments.committee_ab import matrix, nominal_usd
from experiments.common import suite_metrics
from experiments.contract_sensitivity import static_screen_contracts
from experiments.search_ab import jobs_for, run_selected
from taic.experience import build_experience_index, retrieve_experience_hints
from taic.llm import LLMConfig
from taic.plan_ledger import classify_outcome, format_ledger_for_planner, nodes_from_trace
from taic.policy import feature_signature, select_diverse_parent
from tcenv.bench import paired_bootstrap_speedup_ci


class _Cand:
    def __init__(self, source: str, speedup: float, label: str):
        self.source = source
        self.speedup = speedup
        self.label = label
        self.eligible = True
        self.recheck_rejected = False


def test_an_abi_failure_is_not_a_falsified_direction():
    kind, hint = classify_outcome("abi_error", "static threadgroup memory 40960 B exceeds the device limit of 32768 B", improved=False, within_noise=False, speedup=None, parent_speedup=1.0)
    assert kind == "untested_impl_fail"
    assert "32768" in hint
    kind, _hint = classify_outcome("ok", "not better", improved=False, within_noise=False, speedup=0.9, parent_speedup=1.0)
    assert kind == "falsified"
    kind, _hint = classify_outcome("ok", "new best", improved=True, within_noise=False, speedup=1.2, parent_speedup=1.0)
    assert kind == "validated_win"
    nodes = nodes_from_trace(
        [{"label": "r1.p0", "status": "abi_error", "speedup": None, "improved": False, "within_noise": False, "best_speedup": 1.0}],
        ["round 1: pad the tile -> abi: threadgroup 40960"],
    )
    text = format_ledger_for_planner(nodes)
    assert "untested_impl_fail" in text
    assert "falsified" not in text.split("untested")[0]


def test_a_same_round_winner_replaced_by_a_sibling_is_not_falsified():
    nodes = nodes_from_trace(
        [{"label": "r2.p0", "status": "ok", "speedup": 1.4, "improved": True, "within_noise": False, "best_speedup": 1.8}],
        ["round 2: vectorize the loads -> 1.400x (not better)"],
    )
    assert nodes[0].outcome_kind == "validated_win"


def test_a_recheck_failure_is_not_a_timed_loss():
    nodes = nodes_from_trace(
        [{"label": "r2.p0", "status": "ok", "speedup": 1.4, "improved": False, "within_noise": False, "best_speedup": 1.0, "differential_recheck": "incorrect"}],
        ["round 2: widen the tile -> 1.400x (not better)"],
    )
    assert nodes[0].outcome_kind == "untested_impl_fail"


def test_diverse_parent_is_none_until_features_differ():
    same = "kernel void k() { o[0] = 1; }\n"
    other = "kernel void k() { simdgroup_matrix<float, 8, 8> acc[4]; simdgroup_multiply_accumulate(acc[0], acc[0], acc[0], acc[0]); }\n"
    best = _Cand(same, 1.0, "seed")
    close = _Cand(same + "\n", 0.99, "near")
    assert feature_signature(same)[0] is False
    parent = select_diverse_parent([best, close, _Cand(other, 0.9, "alt")], best)
    assert parent is not None and parent.label == "alt"


def test_static_screen_drops_a_32_fragment_gemm_contract():
    rows = static_screen_contracts("gemm", ["64x64x32t128", "64x64x32t64", "128x128x32t128"], "fp16", "paper")
    by = {row["variant"]: row for row in rows}
    assert by["64x64x32t128"]["kept"] is True
    assert by["64x64x32t64"]["kept"] is False
    assert by["64x64x32t64"]["acc_tiles"] >= 32
    assert by["128x128x32t128"]["kept"] is False
    assert by["128x128x32t128"]["acc_tiles"] >= 64


def test_search_ab_skips_an_occupied_directory(tmp_path: Path):
    dest = tmp_path / "fa_on_1"
    dest.mkdir()
    (dest / "result.json").write_text("{}")
    seen = []
    notes = run_selected(tmp_path, jobs_for("fa", "on", 1), budget=3, max_total_usd=3, runner=lambda *a, **k: seen.append(a))
    assert seen == []
    assert notes[0]["skipped"] == "occupied"


def test_suite_metrics_count_only_published_speedups():
    rows = [
        {"published_speedup": 1.01, "naive_speedup": 0.5, "spent_usd": 2.0},
        {"published_speedup": 0.9, "spent_usd": 1.0},
        {"published_speedup": None, "best_speedup": 2.0, "spent_usd": 1.0},
    ]
    got = suite_metrics(rows, thresholds=(1.0,))
    assert got["fast"]["1"]["n"] == 1
    assert got["published_n"] == 2
    assert got["n"] == 3


def test_bootstrap_interval_needs_two_samples():
    assert paired_bootstrap_speedup_ci([1.0], [1.0]) is None
    lo, hi = paired_bootstrap_speedup_ci([1.0, 1.0, 1.0, 1.0], [2.0, 2.0, 2.0, 2.0], n_boot=50, seed=1)
    assert lo == hi == 2.0


def test_experience_index_names_an_added_feature(tmp_path: Path):
    run = tmp_path / "relu_one"
    for idx, (body, speed) in enumerate((
        ("kernel void k() { o[0] = 1; }\n", 1.0),
        ("kernel void k() { simdgroup_matrix<float, 8, 8> acc[4]; simdgroup_multiply_accumulate(acc[0], acc[0], acc[0], acc[0]); }\n", 1.2),
    )):
        folder = run / "candidates" / f"{idx:03d}_c"
        folder.mkdir(parents=True)
        (folder / "kernel.metal").write_text(body)
        (folder / "report.json").write_text('{"status": "ok", "speedup": %s, "task": "relu", "roofline": {"bound": "memory"}}' % speed)
    index = build_experience_index([tmp_path])
    text = retrieve_experience_hints(index, "relu", "memory")
    assert "matrix_unit" in text
    assert index["wins"] == 1
    assert retrieve_experience_hints(index, "relu", "memory", exclude_task=True) == ""
    other = retrieve_experience_hints(index, "gemm", "memory", exclude_task=True)
    assert "matrix_unit" in other
    leaked = {"entries": index["entries"]}
    assert retrieve_experience_hints(leaked, "gemm", "memory", exclude_task=True) == ""


def test_repeatability_drops_an_experience_index(tmp_path: Path, monkeypatch):
    from experiments.repeatability import repeatability_overrides, run_repeatability

    seen = {}

    def fake(task, dtype, size, rd, **kw):
        seen["experience_index"] = kw.get("experience_index")
        rd.mkdir(parents=True)
        (rd / "result.json").write_text(
            '{"schema_version": 1, "task": "%s", "dtype": "%s", "stop_reason": "max_rounds", "spent_usd": 0.0}' % (task, dtype)
        )

    monkeypatch.setattr("experiments.repeatability.run_one", fake)
    assert repeatability_overrides({"experience_index": "/tmp/leak.json", "preset": "base"}) == {"preset": "base", "experience_index": ""}
    run_repeatability(["swiglu"], "fp16", "small", tmp_path, trials=1, budget=0.01, resume=False, experience_index="/tmp/leak.json")
    assert seen["experience_index"] == ""
    from tcenv.evaluate import selected_bench_names

    assert selected_bench_names(["mlx", "torch_mps"], fast_bench=False, remembered="mlx") == ["candidate", "candidate_serial", "mlx", "torch_mps"]
    assert selected_bench_names(["mlx", "torch_mps"], fast_bench=True, remembered="mlx") == ["candidate", "mlx"]
    assert selected_bench_names(["mlx", "torch_mps"], fast_bench=True, remembered=None) == ["candidate", "mlx", "torch_mps"]
    doc = matrix(n=3, budget=3, max_total_usd=20)
    assert doc["dry_run"] is True
    assert nominal_usd(3, 3) == 27
    assert doc["fits"] is False
    assert LLMConfig(base_url="http://x", api_key="k", model="m").truncation_retry_effort == "low"
    assert LLMConfig(base_url="http://x", api_key="k", model="m").plan_reasoning_effort is None
