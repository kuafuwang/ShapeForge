"""Real-Metal replay of the incomplete, non-publishable Split-K launch floor."""

from __future__ import annotations

import pytest

from experiments.gemm_splitk_probe import run_probe
from tests.support.gemm_splitk import analysis_document

pytestmark = [pytest.mark.gpu, pytest.mark.slow]


def test_two_stage_floor_still_rejects_small_splitk(tmp_path):
    result = run_probe(tmp_path / "floor.json")
    assert result["not_a_candidate"] is True
    assert result["two_stage_floor_exceeds_existing_candidate"] is True
    assert result["two_stage_floor_exceeds_mlx_baseline"] is True
    analysis = analysis_document(result)
    assert analysis["decision"] == "do_not_implement"
    assert analysis["paper"]["output_threadgroups"] == 1024
    assert analysis["paper"]["split2_partial_write_read_us_at_peak"] > 100.0
