# Backends

A backend is a language and a runtime. It is not a chip.

`metal` is the Apple GPU path: Metal Shading Language, the skill notes, and Apple's PDFs. Chip measurements do not live here.

A new accelerator is a new directory next to `metal`, copied from `_template`, with its own `backend.toml`. A target under `targets/` then names that backend. `backend.toml` must name a repository `entrypoint` under `tcenv.backends.`; the registry checks the id and code version. Skills and wiki alone do not make the backend runnable; the entrypoint has to implement the backend protocol.

The shared `Contract` and the verification interface live at `tcenv.contract` and `tcenv.verify`. The old top-level `tcenv/*.py` backend shims were removed after Phase 4; backend-specific code is imported from `tcenv.backends.<id>.*`.
