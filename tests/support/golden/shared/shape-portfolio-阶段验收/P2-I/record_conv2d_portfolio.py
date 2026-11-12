#!/usr/bin/env python3
"""Record the conv2d small+paper portfolio search-seed runs.

This writes the two run documents and regenerates ``full-eval.json`` from them.
It does not replace the archived paper fresh confirmation and does not publish.
"""

from __future__ import annotations

from tests.backends.metal.conv2d_portfolio_record import record_search_seed_runs


def main() -> None:
    record_search_seed_runs()
    print("OK: recorded conv2d portfolio small and paper runs")


if __name__ == "__main__":
    main()
