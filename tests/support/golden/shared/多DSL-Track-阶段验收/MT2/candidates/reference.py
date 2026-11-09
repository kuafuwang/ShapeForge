"""MT2 fixed elementwise oracle: y=2*x+1 and read_write acc += y."""

import torch


def gen_inputs(case, seed, instance):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    shape = tuple(instance.tensor("x").shape)
    x = torch.randn(shape, generator=generator, dtype=torch.float32)
    acc = torch.randn(shape, generator=generator, dtype=torch.float32) * 0.25
    if case == "wide_range":
        x = x * 100.0
    elif case == "outlier":
        x = torch.where(torch.rand(shape, generator=generator) < 0.01, x * 1000.0, x)
    return {"x": x, "acc": acc}


def reference(inputs, attrs):
    x = inputs["x"].to(torch.float64)
    y = x * 2.0 + 1.0
    return {"y": y, "acc": inputs["acc"].to(torch.float64) + y}
