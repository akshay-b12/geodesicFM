"""Reparameterized additive noise: epsilon is explicit and fixed per derivative."""
import torch
from .common import parameters, check_image


def make_epsilon(x, seed):
    check_image(x)
    # Generate on CPU for the same realization on CPU and CUDA with this build.
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn(x.shape, generator=generator, dtype=x.dtype).to(x.device)


def gaussian_noise(x, xi, *, epsilon):
    sigma = parameters(x, xi, 1)[0]
    if epsilon.shape != x.shape or epsilon.dtype != x.dtype or epsilon.device != x.device:
        raise ValueError("epsilon must match the image's shape, dtype and device.")
    return x + sigma * epsilon
