"""Haze D_(beta,A)(x)=x*exp(-beta*d)+A*(1-exp(-beta*d)).

Depth d is observed/fixed for this synthetic operator, never inferred internally.
Unit depth gives a controlled global attenuation experiment, not realistic
depth-dependent atmospheric haze. A is a scalar achromatic airlight.
"""
import torch
from .common import parameters


def haze(x, xi, *, depth):
    beta, airlight = parameters(x, xi, 2).unbind()
    if depth.device != x.device or depth.dtype != x.dtype:
        raise ValueError("depth must have the same dtype/device as x.")
    if depth.ndim not in (2, 3, 4) or depth.shape[-2:] != x.shape[-2:]:
        raise ValueError("depth must share x's spatial dimensions.")
    try:
        output_shape = torch.broadcast_shapes(x.shape, depth.shape)
    except RuntimeError as exc:
        raise ValueError("depth cannot broadcast to x.") from exc
    if output_shape != x.shape:
        raise ValueError("depth must broadcast to exactly x.shape.")
    transmission = torch.exp(-beta * depth)
    return x * transmission + airlight * (1 - transmission)
