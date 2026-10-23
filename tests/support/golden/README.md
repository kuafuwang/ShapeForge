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
