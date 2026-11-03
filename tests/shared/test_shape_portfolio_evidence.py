"""P2-0 shared core-API extension; no stage verifier or backend/GPU recursion."""

from __future__ import annotations

import pytest

from tcenv.contract_errors import TaskContractError
from tcenv.portfolio_evidence.projection import (
    STRICT_PARSE,
    EvidenceRejection,
    RawPortfolioObservation,
    loads_document,
    strict_json_equal,
    verify_json,
)


def test_p2_0_core_api_strict_json_and_numeric_boundaries(tmp_path):
    assert not strict_json_equal((1,), (True,))
    assert not strict_json_equal({1: "x"}, {True: "x"})
    assert not strict_json_equal(-0.0, 0.0)
    assert strict_json_equal({"a": [1, False]}, {"a": [1, False]})

    for elapsed in (10**400, float("inf"), -1):
        with pytest.raises(TaskContractError):
            RawPortfolioObservation(None, None, None, "fp16/below/golden", elapsed)

    duplicate = '{"schema_id":"first","schema_id":"second"}'
    with pytest.raises(EvidenceRejection) as text_error:
        loads_document(duplicate, expected_freeze_digest="0" * 64)
    assert text_error.value.stage == STRICT_PARSE
    assert text_error.value.rejected_by == "loads_strict_json"
    assert "repeats the key 'schema_id'" in str(text_error.value)

    path = tmp_path / "duplicate-evidence.json"
    path.write_text(duplicate, encoding="utf-8")
    with pytest.raises(EvidenceRejection) as file_error:
        verify_json(path, expected_freeze_digest="0" * 64)
    assert file_error.value.stage == STRICT_PARSE
    assert file_error.value.rejected_by == "load_strict_json"
