# Contributing to ShapeForge

ShapeForge welcomes human and AI-assisted contributions. The project values executable evidence over plausible explanations.

## 1. Load the project context

- Read `AGENTS.md` first. It is the canonical AI context and safety boundary.
- Read `README.md` (English) or `README.zh-CN.md` (Chinese) for the project overview.
- For ShapeForge work, load `.agents/skills/shapeforge/SKILL.md`.
- Before changing status, read `docs/架构/shape-portfolio实施计划.md`; do not infer it from old notes.

Do not recreate `docs/README.md`. The root language-specific READMEs are the only project entrypoints.

## 2. Choose a small, explicit scope

A good change identifies:

- the task or bug being addressed;
- the candidate/harness boundary it must preserve;
- the fixed CPU/GPU nodes and negative cases that prove it;
- the evidence or documentation that must be updated;
- whether a new phase/epoch is needed instead of editing a completed one.

Do not mix unrelated experiments, large third-party materials or local configuration into a ShapeForge commit.

## 3. Preserve candidate freedom and evidence

Candidates may choose algorithms, fusion, Tiling, Host Tiling and ExecutionGeometry. Do not solve a task by adding a per-task algorithm registry or restricting candidates to a pattern library.

Do not:

- lower the 1.10 regression gate to make a candidate pass;
- remove a failing shard, test or required unseen instance;
- count fallback work as variant success;
- use search-best as published speedup;
- rewrite historical evidence or completed phase manifests;
- skip/xfail a required GPU node or weaken an assertion;
- claim real NPU correctness/performance from simulation.

If the current implementation does not meet the gate, leave the phase blocked and record the actual result.

## 4. Test appropriately

Start with:

```bash
uv run pytest -q -m 'not gpu and not llm and not slow'
```

Only with explicit GPU permission:

```bash
uv run pytest -q -m gpu
```

Debug a single failure with:

```bash
uv run pytest -q -n 0 <nodeid>
```

Read `tests/README.md` before adding or moving tests. Shared tests must not import a backend; backend-specific behavior belongs under `tests/backends/<id>/`.

For a failure, use `logs/pytest/triage/triage_summary.md` and its AI handoff. Keep pre-existing/environment failures separate from new regressions.

## 5. Update evidence and documentation

For a phase change, include as applicable:

- a verifier and manifest;
- fixed negative cases and fixed node IDs;
- an acceptance document and current status;
- fresh evidence rather than hand-edited measurements;
- a new epoch if the contract, policy or completed evidence must change.

Keep `README.md` and `README.zh-CN.md` conceptually aligned, but do not create another project entrypoint. Update `AGENTS.md` when contributor invariants change, and keep `.agents/skills/shapeforge/SKILL.md` as a short pointer rather than duplicating the full architecture.

## 6. Commit and publish

CI runs deterministic smoke and portfolio checks on `macos-latest`; the manual `full-safe` option runs the complete CPU safe suite. Metal GPU checks live in a separate manual self-hosted workflow, never on GitHub-hosted runners. See `.github/workflows/ci.yml` and `.github/workflows/metal-gpu.yml`.

Before committing:

```bash
git status --short
git diff --check
```

Check for credentials, local caches, generated binaries and unrelated large files. Third-party code must retain its license and provenance; the self-contained MLX-derived fixture documents its source.

Use a focused commit message. Do not force-push shared history. A clean local commit is required before `git push`, but passing a verifier does not imply that a performance phase is complete.

## Security and cost boundaries

- Never commit credentials.
- Never run paid model search or real GPU/NPU experiments without explicit user authorization.
- Use bounded execution for untrusted candidate kernels; the GPU also drives the display.
- Remote evaluation requires an explicit config and human-registered allowlist.
- Treat model suggestions, self-reported evidence and profiling output as separate trust levels.
