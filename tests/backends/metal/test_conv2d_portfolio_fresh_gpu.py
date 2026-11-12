"""Independent real-Metal replay of the P2-K four-subject fresh suite."""

from __future__ import annotations

import pytest

from taic.portfolio_manifest import DispatchManifest
from tests.backends.metal.conv2d_portfolio_fresh_record import confirm_portfolio_subjects, validated_manifest
from tests.support.conv2d_portfolio_fresh import VALIDATED_MANIFEST, load_json

pytestmark = [pytest.mark.gpu, pytest.mark.slow]


def test_portfolio_fresh_replay_rebuilds_the_same_validated_manifest():
    subjects = confirm_portfolio_subjects()
    rebuilt = validated_manifest(subjects)
    archived = DispatchManifest.from_document(load_json(VALIDATED_MANIFEST))
    assert rebuilt.status == "validated" and rebuilt.activation_generation == 0
    assert rebuilt.document() == archived.document()
    for subject in subjects:
        if subject["usage"] == "benchmark":
            report = subject["confirmation_report"]
            assert report["candidate_us"] <= 1.10 * min(report["baseline_us"].values())
