Hardware-feature hints: the "Hardware-feature analysis" block in the evaluation feedback

When the feedback of the current best kernel ends with this block, read it as follows.

- It is STATIC analysis: the harness read your source, compiled it, and counted which Apple-GPU features the kernel uses (simdgroup_matrix and
  its accumulator fragments, threadgroup staging and padding, double buffering or register prefetch, wide device loads and stores, SIMD
  reductions and shuffles, online softmax, an epilogue on accumulator registers). It matches patterns: it is not a measurement and it can be wrong.
- "Uses" lists what the kernel already has. Do not rebuild it.
- "Expected for this kind of kernel but not found" lists features that kernels of this kind usually have and that may explain a gap to the
  roofline. Each item is a hint, never a requirement. Take the one that fits the bottleneck you diagnosed from the roofline, the pipeline
  statistics and the code; ignore the others. If a hint does not fit your diagnosis, say so instead of following it.
- "Measured" shows how far the kernel is from the roofline. A kernel at about 90% of its roofline or above is done as far as hardware features
  go: nothing can be gained beyond the roofline, the list of missing features is empty on purpose, and adding features to it for their own sake
  only adds risk. Memory-bound elementwise kernels are usually in this state: scalar loads already reach most of the bandwidth.
- A warning about accumulator fragments or maxTotalThreadsPerThreadgroup means register pressure. More live state (more fragments, more
  unrolling, deeper prefetch) can make a kernel many times slower through spills, so shrink before you grow.
- Only the measured time decides. A change that adds a listed feature but is not faster is discarded like any other change.
