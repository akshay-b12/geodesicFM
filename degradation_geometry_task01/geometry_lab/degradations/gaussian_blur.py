"""Isotropic sampled Gaussian: sigma is measured in original-image pixels."""
import torch
from .common import parameters, convolve, check_kernel_size


def gaussian_kernel1d(x, sigma, kernel_size=33):
    check_kernel_size(kernel_size)
    sigma = parameters(x, sigma, 1)[0]
    # Mathematical domain: sigma > 0; fixed support across the entire study.
    q = torch.arange(-(kernel_size // 2), kernel_size // 2 + 1,
                     dtype=x.dtype, device=x.device)
    weights = torch.exp(-0.5 * (q / sigma).square())
    return weights / weights.sum()


def gaussian_blur(x, xi, kernel_size=33):
    kernel = gaussian_kernel1d(x, xi, kernel_size)
    return convolve(convolve(x, kernel[None, :]), kernel[:, None])
