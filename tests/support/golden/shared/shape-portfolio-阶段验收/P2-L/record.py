#!/usr/bin/env python3
"""Re-record the GEMM Split-K launch floor and derive the negative analysis."""

import json

from experiments.gemm_splitk_probe import run_probe
from tests.support.gemm_splitk import ANALYSIS, FLOOR, analysis_document, load_json


def main() -> None:
    run_probe(FLOOR)
    ANALYSIS.write_text(json.dumps(analysis_document(load_json(FLOOR)), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("OK: recorded GEMM Split-K negative analysis; no candidate implementation")


if __name__ == "__main__":
    main()
