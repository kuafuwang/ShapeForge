# Backends

A backend is a language and a runtime. It is not a chip.

| concept | what it is | where it lives |
| --- | --- | --- |
| backend | a language plus a runtime, implemented against a protocol | `backends/<id>/` (`backend.toml`, skill notes, wiki) |
| target | one chip: names its backend and holds what was measured on that chip | `targets/<id>/target.toml` |

`metal` is the Apple GPU path: Metal Shading Language, the skill notes, and Apple's PDFs. Chip measurements do not live here.

## Adding a new accelerator

A new accelerator is a new directory next to `metal`, copied from `_template`, with its own `backend.toml`. A target under `targets/` then names that backend.

```mermaid
flowchart LR
  tpl["backends/_template"] -->|copy| new["backends/&lt;id&gt;"]
  new --> toml["backend.toml<br/>id + entrypoint"]
  toml --> reg["registry checks<br/>tcenv/backends/registry.py<br/>id and code version"]
  reg -->|ok| tgt["targets/&lt;id&gt;/target.toml<br/>names the backend"]
  tgt -->|active| run["tcenv loads tcenv.backends.&lt;id&gt;.*"]
```

`backend.toml` must name a repository `entrypoint` under `tcenv.backends.`; the registry (`tcenv/backends/registry.py`) checks the id and code version. The protocol itself is `tcenv.backends.base.Backend`. Skills and wiki alone do not make the backend runnable; the entrypoint has to implement the protocol.

The shared `Contract` and the verification interface live at `tcenv.contract` and `tcenv.verify`. Backend-specific code is imported from `tcenv.backends.<id>.*`; the old top-level `tcenv/*.py` shims were removed after Phase 4.
