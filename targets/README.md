# Targets

A target is one chip. It names a backend and holds what was measured on that chip: `device.md`, `experience.md`, `calibration.json`, and the limits in `target.toml`.

`apple-m4-max` is the machine this repository runs on. `status = "active"` is required before `taic run` or `tcenv eval` will start. `_template` is `reserved` and cannot run.

A second Apple GPU, such as an M3, is a new directory here with `backend = "metal"`. It reuses `backends/metal` and brings its own measurements.

`trust.toml` is the target's model-trust registry. It is empty by default; a person grants a model for one task and scope, and `trust_audit.jsonl` records every grant, suspension, and revocation. A prompt or measurement-protocol change makes an old grant stale. See `docs/代理反馈.md`.
