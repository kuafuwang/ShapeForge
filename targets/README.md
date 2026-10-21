# Targets

A target is one chip. It names a backend and holds what was measured on that chip: `device.md`, `experience.md`, `calibration.json`, and the limits in `target.toml`.

`apple-m4-max` is the machine this repository runs on. `status = "active"` is required before `taic run` or `tcenv eval` will start. `_template` is `reserved` and cannot run.

A second Apple GPU, such as an M3, is a new directory here with `backend = "metal"`. It reuses `backends/metal` and brings its own measurements.
