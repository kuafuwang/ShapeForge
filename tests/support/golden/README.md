# Golden data

Golden files are grouped by backend because prompts, fixtures and expected
artefacts are backend-specific:

```text
golden/
└── metal/
    ├── backend_phase0_kernel.metal
    ├── hw_off.json
    └── fixtures/
```

A future backend adds `golden/<id>/` beside `metal/`; do not put backend
artefacts directly under `golden/` again.

Recording and placement rules are in `../../README.md` (the `--record --reason`
command, and "golden 改动要和生成它的代码改动放在同一提交"). The reader/writer is
`tests.support.hw_golden`.
