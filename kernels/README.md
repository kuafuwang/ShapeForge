# Kernels

Confirmed kernels, one directory per target, task, dtype and size.

```text
kernels/<target_id>/<task>/<dtype>/<size>/best.metal
kernels/<target_id>/<task>/<dtype>/<size>/meta.json
kernels/<target_id>/<task>/<dtype>/<size>/history.jsonl
```

A kernel is copied here only when the publication gate says `confirmed`. `taic brief --kernels` lists them. `taic run TASK --from @catalog` starts from `best.metal` for the same target.
