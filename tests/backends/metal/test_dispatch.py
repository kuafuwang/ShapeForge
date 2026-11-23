"""Candidate dispatch mode (serial vs concurrent) on a real Metal device.

The three v1-candidate cases that used to live here are retired with the v1 task assembly and
the retired ``tcenv.evaluate.evaluate`` entry point.  Current dispatch coverage runs through the
v2 executor: ``tcenv/backends/metal/runtime.py::MetalV2Executor`` records the dispatch mode and
the v2 GPU suites exercise concurrent batches on real candidates.  The CPU half of the old
module (batch-size probe and interleaved timing loop) lives in ``tests/shared/test_bench.py``.
"""
