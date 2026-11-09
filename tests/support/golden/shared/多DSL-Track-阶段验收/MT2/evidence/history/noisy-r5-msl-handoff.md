# Failure triage: `tests/backends/metal/test_tilelang_mt2_gpu.py::test_mt2_same_semantics_msl_single_stage_comparison`

## One-shot reproduce

```bash
cd '/Users/kuafu/CLionProjects/ai_lowerling'
uv run pytest -q --tb=short 'tests/backends/metal/test_tilelang_mt2_gpu.py::test_mt2_same_semantics_msl_single_stage_comparison'
```

## For AI assistants

The fenced JSON is the ground truth for this failure.

````
---BEGIN_AI_FAILURE_HANDOFF---
{
  "schema": "ai_lowerling_failure_handoff/v1",
  "git_short": "efb8218",
  "worker": "master",
  "nodeid": "tests/backends/metal/test_tilelang_mt2_gpu.py::test_mt2_same_semantics_msl_single_stage_comparison",
  "source": {
    "file": "tests/backends/metal/test_tilelang_mt2_gpu.py",
    "lineno": 168
  },
  "markers": [
    "gpu",
    "xdist_group"
  ],
  "error_kind": "assertion",
  "pytest_short": "assert (1.0930427733347217 > 0 and True is False)",
  "exception": {
    "type": "AssertionError",
    "message_lines": [
      "assert (1.0930427733347217 > 0 and True is False)"
    ]
  },
  "params": {},
  "feedback": "",
  "minimal_repro_cmd": "cd '/Users/kuafu/CLionProjects/ai_lowerling'\nuv run pytest -q --tb=short 'tests/backends/metal/test_tilelang_mt2_gpu.py::test_mt2_same_semantics_msl_single_stage_comparison'",
  "per_test_log": {
    "path": "logs/pytest/tests/master/tests_backends_metal_test_tilelang_mt2_gpu.py__test_mt2_same_semantics_msl_single_stage_comparison.log",
    "tail": "2026-11-08 00:23:09,417 | INFO     | ai_lowerling.tests | ========== begin tests/backends/metal/test_tilelang_mt2_gpu.py::test_mt2_same_semantics_msl_single_stage_comparison ==========\n2026-11-08 00:23:11,633 | INFO     | ai_lowerling.tests | ========== end tests/backends/metal/test_tilelang_mt2_gpu.py::test_mt2_same_semantics_msl_single_stage_comparison duration_s=2.217 ==========\n2026-11-08 00:25:15,964 | INFO     | ai_lowerling.tests | ========== begin tests/backends/metal/test_tilelang_mt2_gpu.py::test_mt2_same_semantics_msl_single_stage_comparison ==========\n2026-11-08 00:25:18,183 | INFO     | ai_lowerling.tests | ========== end tests/backends/metal/test_tilelang_mt2_gpu.py::test_mt2_same_semantics_msl_single_stage_comparison duration_s=2.220 ==========\n2026-11-08 00:47:57,401 | INFO     | ai_lowerling.tests | ========== begin tests/backends/metal/test_tilelang_mt2_gpu.py::test_mt2_same_semantics_msl_single_stage_comparison ==========\n2026-11-08 00:47:59,581 | INFO     | ai_lowerling.tests | ========== end tests/backends/metal/test_tilelang_mt2_gpu.py::test_mt2_same_semantics_msl_single_stage_comparison duration_s=2.180 ==========\n2026-11-08 00:52:35,658 | INFO     | ai_lowerling.tests | ========== begin tests/backends/metal/test_tilelang_mt2_gpu.py::test_mt2_same_semantics_msl_single_stage_comparison ==========\n2026-11-08 00:52:37,830 | INFO     | ai_lowerling.tests | ========== end tests/backends/metal/test_tilelang_mt2_gpu.py::test_mt2_same_semantics_msl_single_stage_comparison duration_s=2.172 ==========\n2026-11-08 00:53:21,872 | INFO     | ai_lowerling.tests | ========== begin tests/backends/metal/test_tilelang_mt2_gpu.py::test_mt2_same_semantics_msl_single_stage_comparison ==========\n"
  },
  "environment": {
    "TCENV_ALLOW_GPU_HANG_TESTS": "",
    "TCENV_RUN_XCTRACE": "",
    "MTL_SHADER_VALIDATION": "",
    "TCENV_GPU_LOCK_FILE": ""
  }
}
---END_AI_FAILURE_HANDOFF---
````
