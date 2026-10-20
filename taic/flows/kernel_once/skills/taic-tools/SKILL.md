---
name: taic-tools
description: Use lookup_msl to read Metal notes and quick_check to compile and test a kernel. Do not time it yourself.
---

# taic tools

Two tools. Both are real. Neither is a chat function named `lookup_msl` or `quick_check`. You run them with the tools you already have (`read`, `write`, `bash`).

## lookup_msl

What it is: the Metal Shading Language notes for this search (attributes, address spaces, types, barriers, `simdgroup_matrix`).

How to use it: `read` the file `msl_reference.md` in the working directory. The search copied it there. Read the section you need. Do not invent an API that is not in that file.

## quick_check

What it is: compile the kernel, check the ABI and launch contract, check memory safety, and compare answers with the reference. It does not benchmark. A time you measure is not the search result. The search times `kernel.metal` after you exit.

How to use it: `bash` the exact command in `TASK.md`. The lock file must be `TCENV_GPU_LOCK_FILE=/tmp/ai_lowerling-gpu.lock`, because this session cannot write `~/.cache`. It looks like:

```
uv run --directory <repo> python -m tcenv.cli eval <task> <kernel.metal> --dtype <dtype> --size <size> --quick
```

`--quick` is required. Read the command's output and fix `kernel.metal` if it fails. Do not delete or rewrite the GPU lock file. Do not run a benchmark. A kernel that does not return can freeze the display.
