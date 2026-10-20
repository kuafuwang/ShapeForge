"""The contract menu driver, with a stand-in for the search. No model calls and no GPU."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from experiments.contract_menu import judge, resolve_menu, run_menu, schedule

SOURCE = "kernel void k() {}\n"


def _published(run_dir: Path, task: str, speedup: float, *, best: float, spent: float) -> None:
    text = SOURCE + f"{speedup}"
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    label = "r1.p0"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "best.metal").write_text(text)
    cand = run_dir / "candidates" / f"001_{label}"
    cand.mkdir(parents=True)
    (cand / "kernel.metal").write_text(text)
    (run_dir / "confirm_report.json").write_text(json.dumps({
        "status": "ok",
        "benchmark": {"noisy": False, "candidate": {"cv": 0.01, "reps": 15}, "baselines": {"mlx": {"cv": 0.01, "reps": 15}}, "best_baseline": "mlx"},
    }))
    (run_dir / "result.json").write_text(json.dumps({
        "task": task, "dtype": "fp16", "model": "fake", "stop_reason": "budget_exhausted", "rounds": 1, "evaluations": 1,
        "spent_usd": spent, "budget_usd": spent, "best_speedup": best, "confirmed_speedup": speedup,
        "confirmed_status": "ok", "confirm_winner_label": label, "confirmation_source_sha256": digest,
    }))


def _unconfirmed(run_dir: Path, task: str, best: float, spent: float) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "result.json").write_text(json.dumps({
        "task": task, "dtype": "fp16", "model": "fake", "stop_reason": "budget_exhausted", "rounds": 1, "evaluations": 1,
        "spent_usd": spent, "budget_usd": spent, "best_speedup": best, "confirmed_speedup": best,
        "confirmed_status": "incorrect", "confirmation_source_sha256": "unbound",
    }))


def test_a_menu_without_default_is_refused_before_any_search(tmp_path):
    calls = []

    def run_fn(*_a, **_k):
        calls.append(1)

    with pytest.raises(ValueError, match="default"):
        run_menu("flash_attention", ["32x16t128", "64x32t256"], tmp_path / "m", trials=1, budget=3, max_total_usd=9, run_fn=run_fn, log=lambda _s: None)
    assert calls == []


def test_a_task_without_variants_is_refused_before_any_search(tmp_path):
    calls = []

    def run_fn(*_a, **_k):
        calls.append(1)

    with pytest.raises(ValueError, match="no contract variants"):
        run_menu("relu", ["default", "32x16t128"], tmp_path / "m", trials=1, budget=3, run_fn=run_fn, log=lambda _s: None)
    assert calls == []
    assert not (tmp_path / "m" / "menu.json").exists()


def test_trials_are_interleaved_and_a_second_pass_skips_finished_runs(tmp_path):
    calls = []

    def run_fn(task, dtype, size, run_dir, **kw):
        assert kw["require_contract_launch"] is True
        calls.append((task, kw["preset"], kw["budget"]))
        _unconfirmed(Path(run_dir), task, best=9.0, spent=kw["budget"])
        return type("R", (), {"stop_reason": "budget_exhausted"})()

    out = tmp_path / "m"
    report = run_menu(
        "flash_attention", ["default", "32x16t128", "64x32t256"], out,
        trials=3, budget=3, preset="base", max_total_usd=100, run_fn=run_fn, log=lambda _s: None,
    )
    assert [task for task, _preset, _budget in calls] == [
        "flash_attention", "flash_attention@32x16t128", "flash_attention@64x32t256",
        "flash_attention", "flash_attention@32x16t128", "flash_attention@64x32t256",
        "flash_attention", "flash_attention@32x16t128", "flash_attention@64x32t256",
    ]
    assert {preset for _t, preset, _b in calls} == {"base"}
    assert report["verdict"] == "无结论" and report["winner"] is None
    n = len(calls)
    again = run_menu(
        "flash_attention", ["default", "32x16t128", "64x32t256"], out,
        trials=3, budget=3, preset="base", max_total_usd=100, run_fn=run_fn, log=lambda _s: None,
    )
    assert len(calls) == n and again["verdict"] == "无结论"


def test_a_different_budget_cannot_reuse_the_same_directory(tmp_path):
    def run_fn(task, dtype, size, run_dir, **kw):
        _unconfirmed(Path(run_dir), task, best=1.0, spent=kw["budget"])
        return type("R", (), {"stop_reason": "budget_exhausted"})()

    out = tmp_path / "m"
    run_menu("flash_attention", ["default", "32x16t128"], out, trials=1, budget=3, max_total_usd=9, run_fn=run_fn, log=lambda _s: None)
    with pytest.raises(ValueError, match="different menu"):
        run_menu("flash_attention", ["default", "32x16t128"], out, trials=1, budget=15, max_total_usd=9, run_fn=run_fn, log=lambda _s: None)


def test_the_cap_skips_the_runs_that_do_not_fit(tmp_path):
    calls = []

    def run_fn(task, dtype, size, run_dir, **kw):
        calls.append(Path(run_dir).parts[-2:])
        _unconfirmed(Path(run_dir), task, best=1.0, spent=3.0)
        return type("R", (), {"stop_reason": "budget_exhausted"})()

    # 3 contracts x $3. A cap of $21 admits 7 children and refuses the 8th and 9th.
    report = run_menu(
        "flash_attention", ["default", "32x16t128", "64x32t256"], tmp_path / "m",
        trials=3, budget=3, max_total_usd=21, run_fn=run_fn, log=lambda _s: None,
    )
    assert len(calls) == 7
    assert report["skipped"] == [
        {"slug": "32x16t128", "task": "flash_attention@32x16t128", "trial": 3, "reason": "over_cap"},
        {"slug": "64x32t256", "task": "flash_attention@64x32t256", "trial": 3, "reason": "over_cap"},
    ]
    cap = json.loads((tmp_path / "m" / "cap.json").read_text())
    assert cap["max_total_usd"] == 21


def test_only_published_numbers_can_win_and_a_hang_stops_the_queue(tmp_path):
    # default publishes 1.20 and 1.30. The middle contract publishes 1.00 and 1.05, and one unconfirmed child searched at 9.
    # The last contract publishes 0.90 and 0.95. Ranking by the search best would pick the middle contract.
    plan = {
        ("default", 1): ("pub", 1.20, 1.20),
        ("default", 2): ("pub", 1.30, 1.30),
        ("default", 3): ("raw", None, 1.25),
        ("32x16t128", 1): ("pub", 1.00, 9.0),
        ("32x16t128", 2): ("pub", 1.05, 1.05),
        ("32x16t128", 3): ("raw", None, 9.0),
        ("64x32t256", 1): ("pub", 0.90, 0.90),
        ("64x32t256", 2): ("pub", 0.95, 0.95),
        ("64x32t256", 3): ("raw", None, 0.95),
    }

    def run_fn(task, dtype, size, run_dir, **kw):
        slug = Path(run_dir).parent.name
        trial = int(Path(run_dir).name.split("_")[1])
        kind, speed, best = plan[(slug, trial)]
        if kind == "pub":
            _published(Path(run_dir), task, speed, best=best, spent=1.0)
        else:
            _unconfirmed(Path(run_dir), task, best=best, spent=1.0)
        return type("R", (), {"stop_reason": "budget_exhausted"})()

    report = run_menu(
        "flash_attention", ["default", "32x16t128", "64x32t256"], tmp_path / "rank",
        trials=3, budget=3, max_total_usd=100, run_fn=run_fn, log=lambda _s: None,
    )
    assert report["winner"] == "default" and report["verdict"] == "胜出"
    assert "9" not in report["reason"]
    text = (tmp_path / "rank" / "summary.md").read_text()
    assert "Winner: default" in text and "9.000" not in text


def test_overlapping_published_ranges_are_inconclusive():
    rows = [
        {"slug": "default", "trial": 1, "published_speedup": 1.0, "best_speedup": 1.0},
        {"slug": "default", "trial": 2, "published_speedup": 1.4, "best_speedup": 1.4},
        {"slug": "other", "trial": 1, "published_speedup": 1.2, "best_speedup": 5.0},
        {"slug": "other", "trial": 2, "published_speedup": 1.3, "best_speedup": 5.0},
    ]
    got = judge(rows, ["default", "other"])
    assert got["verdict"] == "无结论" and got["winner"] is None
    assert "重叠" in got["reason"]


def test_one_published_number_is_an_anecdote():
    rows = [
        {"slug": "default", "trial": 1, "published_speedup": 2.0, "best_speedup": 2.0},
        {"slug": "other", "trial": 1, "published_speedup": 1.0, "best_speedup": 1.0},
        {"slug": "other", "trial": 2, "published_speedup": 1.1, "best_speedup": 1.1},
    ]
    got = judge(rows, ["default", "other"])
    assert got["verdict"] == "无结论" and "个例" in got["reason"] and "default" in got["reason"]


def test_a_hang_does_not_start_the_remaining_contracts(tmp_path):
    calls = []

    def run_fn(task, dtype, size, run_dir, **kw):
        calls.append(task)
        _unconfirmed(Path(run_dir), task, best=1.0, spent=0.1)
        reason = "gpu_hang: metal" if len(calls) == 2 else "budget_exhausted"
        return type("R", (), {"stop_reason": reason})()

    report = run_menu(
        "flash_attention", ["default", "32x16t128", "64x32t256"], tmp_path / "hang",
        trials=2, budget=3, max_total_usd=100, run_fn=run_fn, log=lambda _s: None,
    )
    assert calls == ["flash_attention", "flash_attention@32x16t128"]
    assert report["aborted"].startswith("gpu_hang")


def test_schedule_and_resolve_keep_the_default_contract_first():
    items = resolve_menu("flash_attention", ["64x32t256", "default", "32x16t128"])
    assert [slug for slug, _task in items] == ["64x32t256", "default", "32x16t128"]
    assert schedule(items, 2)[0][0] == "64x32t256"
    assert schedule(items, 2)[1] == ("default", "flash_attention", 1)


def test_report_reprints_the_menu_without_starting_a_search(tmp_path):
    from experiments.contract_menu import main

    out = tmp_path / "menu"
    out.mkdir()
    (out / "menu.json").write_text(json.dumps({
        "task": "gemm", "contracts": ["default"], "trials": 1, "budget_usd": 3.0,
        "preset": "base", "dtype": "fp16", "size": "paper", "max_total_usd": None,
    }))
    assert main(["gemm", "--contracts", "default", "--out", str(out), "--report"]) == 0
    assert (out / "summary.md").is_file()
