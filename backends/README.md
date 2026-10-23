# Backends

A backend is a language and a runtime. It is not a chip.

`metal` is the Apple GPU path: Metal Shading Language, the skill notes, and Apple's PDFs. Chip measurements do not live here.

A new accelerator is a new directory next to `metal`, copied from `_template`, with its own `backend.toml`. A target under `targets/` then names that backend. `backend.toml` must name a repository `entrypoint` under `tcenv.backends.`; the registry (`tcenv/backends/registry.py`) checks the id and code version. The protocol itself is `tcenv.backends.base.Backend`. Skills and wiki alone do not make the backend runnable; the entrypoint has to implement the protocol.

The shared `Contract` and the verification interface live at `tcenv.contract` and `tcenv.verify`. Backend-specific code is imported from `tcenv.backends.<id>.*`; the old top-level `tcenv/*.py` shims were removed after Phase 4.
