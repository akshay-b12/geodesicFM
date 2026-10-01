"""Elliptical Gaussian: xi=(sigma_x, sigma_y, theta), theta in radians.

Image axes are x=columns (right), y=rows (down). Positive theta rotates toward
the downward y-axis. Equivalent axes and periodic angles make this chart
non-identifiable globally; sigma_x=sigma_y makes theta unidentifiable locally.
"""
import torch
from .common import parameters, convolve, check_kernel_size


def anisotropic_kernel(x, xi, kernel_size=33):
    check_kernel_size(kernel_size)
    sx, sy, theta = parameters(x, xi, 3).unbind()
    q = torch.arange(-(kernel_size // 2), kernel_size // 2 + 1,
                     dtype=x.dtype, device=x.device)
    yy, xx = torch.meshgrid(q, q, indexing="ij")
    c, s = theta.cos(), theta.sin()
    # R(theta)^T applied to each spatial coordinate.
    u, v = c * xx + s * yy, -s * xx + c * yy
    weights = torch.exp(-0.5 * ((u / sx).square() + (v / sy).square()))
    return weights / weights.sum()


def anisotropic_blur(x, xi, kernel_size=33):
    return convolve(x, anisotropic_kernel(x, xi, kernel_size))
