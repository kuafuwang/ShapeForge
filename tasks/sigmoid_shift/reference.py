"""sigmoid(x) + 1 in fp64, one rounding at the output."""

import torch


def reference(inputs, attrs):
    x = inputs["x"].to(torch.float64)
    return {"y": (torch.sigmoid(x) + 1.0).to(torch.float32)}
