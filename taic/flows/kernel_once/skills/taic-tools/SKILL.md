---
name: taic-tools
description: Read the Metal notes, check a kernel with the commands in TASK.md, and leave NOTES.md.
---

# taic tools

These are the commands in `TASK.md`. You run them with the tools you already have (`read`, `write`, `bash`).

## Metal notes

`read` the file `msl_reference.md` in the working directory. That file is the Metal reference for this compiler. Do not invent an API that is not in it.

Apple's own PDFs are in the repository named by the eval command's `--directory`, under `backends/metal/wiki/`. `INDEX.md` there says which file answers which question. The shading-language specification is the language. The feature-set tables say what each Apple GPU family has. A number measured on this chip, in `device.md` or on the scorecard, outranks both PDFs. A blog does not.

## Checks

`TASK.md` gives two commands.

- The validation command answers only whether the kernel compiles and matches the reference. It does not time the kernel.
- The evaluation command answers correctness and timing. It prints microseconds, the speedup, cv, and whether this version would replace the seed under the outer rule. That line says the self-check does not count. The search reads the submission file named in `TASK.md` after you exit. It does not read `kernel.metal`.

Run the commands from this directory. Do not change the GPU lock path. Do not delete the lock file.

A `self_checks.jsonl` in this directory records every timed check. After a timeout or a crash, another timed check is refused. Leave the kernel as it is and stop.

## Probes

A claim about this machine — an operand dtype runs at half rate in the matrix unit, an instruction does not exist, a limit is such-and-such — is a hypothesis until you measure it. The scorecard prints matrix peaks measured per operand dtype; anything finer (accumulator width, a specific instruction, a specific shape) nobody has measured for you. When such a claim decides whether an algorithm is worth writing, measure that exact combination first with a small kernel through the same runtime the harness uses:

```python
import sys
sys.path.insert(0, "<repo>")  # the --directory of the eval command in TASK.md
from tcenv.metal_runtime import BufferArg, get_context
from tcenv.gpulock import gpu_lock

SRC = r"""#include <metal_stdlib>
using namespace metal;
kernel void probe(device float* out [[buffer(0)]]) { /* the operation under test */ }
"""
ctx = get_context()
k = ctx.compile(SRC, "probe")
out = ctx.new_buffer(4096)
with gpu_lock("probe"):
    tm = ctx.run(k, [[(0, BufferArg(out))]], (8192, 1, 1), (32, 1, 1), iters=4)
print(tm.gpu_per_iter_s)
```

Put the measured number and the probe file name in `NOTES.md`. An unmeasured limit claim is not a reason to drop an algorithm; if you drop one on a claim you did not measure, copy the claim into `NOTES.md` marked unverified, so the next session can re-open it.

## 测量

Read this section before comparing two versions or deleting a piece to see what it costs.

- Time two versions in one process, alternating, and read the median. One `--timed` number drifts by a few percent. Do not rank two versions by two separate runs.
- After you delete a piece, check that the compiler did not delete the work upstream of it. A sink that lives only in a branch that never runs takes the whole region with it.
- If you split the time into parts, the parts have to add up to the total. When they do not, check the measurement before you explain the kernel.

## Notes

Keep `NOTES.md` current as you go: what each version changed, the self-check microseconds and cv, and whether you kept it. Append a line when you finish a version. Hardware-limit claims need the probe file and the measured number, or the word unverified next to them. The next session reads this file and does not see this conversation.

Keep the submission file named in `TASK.md` equal to the best version so far. When a self-check is correct and faster, `cp` that file onto the submission name. If nothing has passed yet, `cp` the latest version, not the seed. The file is absent until you copy it. Leaving it absent means this session returned no kernel. `kernel.metal` is the seed when one was placed. Do not copy an older kernel into the submission file to make a command pass.

Append `FEEDBACK.md` the same way. One line in the matching section when you reject a premise or hit a pitfall. Do not save that file for the end.
