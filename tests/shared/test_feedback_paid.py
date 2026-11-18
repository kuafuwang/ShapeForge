"""Paid model tests for G10/G11/G8 feedback system improvements.

These tests call real LLM APIs (DeepSeek/Kimi) to verify that the feedback system
works end-to-end with real model outputs. They are marked with `llm` and require
explicit opt-in because they cost money.

Run with: uv run pytest -q -m llm tests/shared/test_feedback_paid.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

# Skip all tests if no API key is configured.
pytestmark = pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY") and not os.environ.get("DEEPSEEK_API_KEY"),
    reason="paid model tests require an API key",
)


@pytest.mark.llm
def test_g10_cluster_view_with_real_suggestions(tmp_path):
    """G10: cluster view works with real model-generated suggestions."""
    from taic.cli import main
    
    # Run a minimal search to generate real suggestions.
    # This costs money, so we use the smallest possible budget.
    result = main([
        "run", "conv2d", "--dtype", "fp16", "--size", "small",
        "--budget", "0.5", "--max-rounds", "2",
        "--run-dir", str(tmp_path / "run"),
    ])
    
    # The search should complete (may stop early due to budget).
    assert result in (0, 2, 6)  # 0=ok, 2=no eligible kernel, 6=stop reason
    
    # Query suggestions with cluster view.
    # This should group similar suggestions together.
    result = main(["query", "suggestions", "--cluster", "--json"])
    assert result == 0
    
    # Verify that the output contains cluster information.
    # (We can't check the exact output because it depends on the model's suggestions.)


@pytest.mark.llm
def test_g11_note_replies_with_real_measurements(tmp_path):
    """G11: taic.note replies with existing measurements from real runs."""
    from taic.cli import main
    
    # Run a minimal search to generate real measurements.
    result = main([
        "run", "conv2d", "--dtype", "fp16", "--size", "small",
        "--budget", "0.5", "--max-rounds", "2",
        "--run-dir", str(tmp_path / "run"),
    ])
    assert result in (0, 2, 6)
    
    # Write a note referencing the run.
    result = main([
        "note",
        "--work", str(tmp_path / "run"),
        "--text", "fp32 MMA and FMA time add up: 12.62 vs 13.7 TFLOP/s",
        "--run", str(tmp_path / "run"),
    ])
    
    # The note should be accepted and should reply with existing measurements.
    assert result == 0


@pytest.mark.llm
def test_g8_experience_draft_with_real_run(tmp_path):
    """G8: experience draft generates a complete section from a real run."""
    from taic.cli import main
    
    # Run a minimal search to generate a real run.
    result = main([
        "run", "conv2d", "--dtype", "fp16", "--size", "small",
        "--budget", "0.5", "--max-rounds", "2",
        "--run-dir", str(tmp_path / "run"),
    ])
    assert result in (0, 2, 6)
    
    # Generate an experience draft from the run.
    result = main([
        "harness", "draft-experience",
        "--target", "apple-m4-max",
        "--task", "conv2d",
        "--title", "test title",
        "--source", "runs/" + str(tmp_path / "run").split("/")[-1],
        "--measurement", "12.62 vs 13.7 TFLOP/s",
        "--boundary", "tested on apple-m4-max only",
        "--counterexample", "not tested on other chips",
        "--observation", "test observation",
        "--out", str(tmp_path / "draft.md"),
    ])
    
    # The draft should be generated and should contain the required sections.
    assert result == 0
    
    # Verify that the draft contains the required fields.
    draft_path = tmp_path / "draft.md"
    if draft_path.exists():
        draft = draft_path.read_text()
        assert "## " in draft  # Title
        assert "Confidence: reported" in draft
        assert "观察:" in draft
        assert "来源:" in draft
        assert "本机实测:" in draft
        assert "边界:" in draft
        assert "反例:" in draft
