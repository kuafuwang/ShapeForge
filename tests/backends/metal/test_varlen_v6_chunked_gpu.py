"""V6: real simdgroup-matrix G-form KDA on Metal.

The small D=128 probe exercises every L1 distribution and pins the oracle's decay-before-residual
order.  The six shipped V6 instances then cover D=128, H=64/96 and fixed/mixed/uniform canonical
inputs.  These tests are correctness gates. The recorded fresh speedups are summarized in
AGENTS.md and are only versus the registered naive Torch-MPS baseline.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from taic.task_binding import bind_task
from tcenv.backends.metal.search_candidate import evaluate_candidate
from tcenv.target import load_target
from tcenv.taskcontract.locator import REPO_ROOT, discover_standard_tasks

from ...support.varlen_probe import write_probe_task

pytestmark = pytest.mark.gpu

SOURCE_PATH = REPO_ROOT / "tests/support/golden/metal/varlen/chunked_gform.metal"
SOURCE = SOURCE_PATH.read_text(encoding="utf-8")
V6_SIZES = (
    "v6_h64_fixed",
    "v6_h64_mixed",
    "v6_h64_uniform",
    "v6_h96_fixed",
    "v6_h96_mixed",
    "v6_h96_uniform",
)


def _evaluate(locator, source: str, *, cases: tuple[str, ...] | None = None):
    return evaluate_candidate(
        locator,
        source,
        target=load_target("apple-m4-max"),
        mode="quick",
        reps=3,
        cases=cases,
    )


def _small_probe(tmp_path):
    """Keep the shipped ABI/reference, changing only the probe shape to H=2, D=128."""
    task = tmp_path / "varlen_kda"
    write_probe_task(REPO_ROOT / "tasks/varlen_kda", task, {"mixed": [16, 8, 4, 1]})
    document = json.loads((task / "opspec.json").read_text(encoding="utf-8"))
    document["sizes"]["mixed"]["H"] = 2
    document["sizes"]["mixed"]["D"] = 128
    (task / "opspec.json").write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    return discover_standard_tasks(tmp_path, code_root=REPO_ROOT)[0]


def test_gform_passes_every_l1_distribution_and_sequence_final_state(tmp_path):
    run = _evaluate(_small_probe(tmp_path), SOURCE)
    report = run.result.report
    assert report["status"] == "ok", report.get("correctness")
    assert report["correctness"]["ok"] is True
    assert len(report["correctness"]["cases"]) == 9
    assert all(case["scaled_err"] < 0.05 for case in report["correctness"]["cases"])


def test_gform_rejects_residuals_that_omit_the_current_token_decay(tmp_path):
    """The former L_prev implementation passed tiny probes but failed mixed final_state."""
    previous_decay = SOURCE.replace(
        "kd[t][j] = kv * exp(ld[t][j]);", "kd[t][j] = kv * exp(previous);"
    ).replace(
        "qt[t][j] = qv * exp(ld[t][j]);", "qt[t][j] = qv * exp(previous);"
    )
    assert previous_decay != SOURCE
    run = _evaluate(_small_probe(tmp_path), previous_decay, cases=("canonical",))
    report = run.result.report
    assert report["status"] == "incorrect"
    assert report["correctness"]["ok"] is False
    assert report["correctness"]["cases"][0]["scaled_err"] > 1.0


@pytest.mark.parametrize("size", V6_SIZES)
def test_gform_real_scale_d128_h64_h96_canonical(size):
    binding = bind_task("varlen_kda", "fp16", size)
    assert binding.contract.instance.symbols["D"] == 128
    assert binding.contract.instance.symbols["H"] in (64, 96)
    assert binding.contract.instance.symbols["total_tokens"] == 8192
    run = _evaluate(binding.locator, SOURCE, cases=("canonical",))
    report = run.result.report
    assert report["status"] == "ok", report.get("correctness")
    assert report["correctness"]["ok"] is True
    assert report["correctness"]["cases"][0]["scaled_err"] < 0.05
