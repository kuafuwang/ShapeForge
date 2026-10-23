# Target device: Apple M4 Max (Apple9 GPU family)

This search is running on `apple-m4-max`, backend `metal`. The GPU also drives the display. A kernel that does not return can freeze the screen.

Numbers below were measured on this machine with `tcenv calibrate` and the harness. A different chip has its own `targets/<id>/device.md`. The scorecard's roofline uses this machine's calibration, not a data sheet.

## Measured peaks
Keywords: bandwidth tflops roofline m4 max measured

- Peak read bandwidth is about 527 GB/s. Copy (read and write) is about 449 GB/s.
- Vendor GEMM throughput is about 14.8 TFLOP/s for fp16 and bf16, and about 12.8 TFLOP/s for fp32.
- A naive scalar-FMA GEMM reaches about 5.5 TFLOP/s. `simdgroup_matrix` is required to get close to the matrix peak.
- Threadgroup memory on this chip is 32768 bytes. A threadgroup has at most 1024 threads. Re-read the device when the chip changes. Do not hard-code these as if they were the language.
- Static threadgroup memory also sets residency: 9216 B fits 3 resident threadgroups, while 17408 B fits only 1. A same-output pad control on the flash-attention kernel slowed wall time by 12.0% and the GPU timestamp by 14.3%; this is one measured shape, not a universal percentage. Detailed evidence is in `experience.md`.
