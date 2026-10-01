"""Small tensor helpers; parameters stay connected to the autograd graph."""
import torch
import torch.nn.functional as F


def check_image(x):
    if x.ndim not in (3, 4) or not x.is_floating_point():
        raise ValueError("Expected floating CHW or NCHW tensor.")


def parameters(x, xi, count):
    check_image(x)
    # as_tensor preserves an existing tensor graph; avoid torch.tensor(xi).
    p = torch.as_tensor(xi, dtype=x.dtype, device=x.device)
    if p.ndim == 0 and count == 1:
        p = p.reshape(1)
    if p.shape != (count,):
        raise ValueError(f"Expected parameter vector of shape ({count},), got {tuple(p.shape)}")
    return p


def check_kernel_size(kernel_size):
    if type(kernel_size) is not int or kernel_size < 3 or kernel_size % 2 == 0:
        raise ValueError("Kernel size must be a fixed odd integer >= 3.")


def convolve(x, kernel):
    check_image(x)
    unbatched = x.ndim == 3
    xb = x.unsqueeze(0) if unbatched else x
    channels = xb.shape[1]
    h, w = kernel.shape
    py, px = h // 2, w // 2
    if xb.shape[-2] <= py or xb.shape[-1] <= px:
        raise ValueError("Reflection padding requires image dimensions greater than kernel radii.")
    weights = kernel[None, None].expand(channels, 1, h, w).contiguous()
    padded = F.pad(xb, (px, px, py, py), mode="reflect")
    y = F.conv2d(padded, weights, groups=channels)
    return y.squeeze(0) if unbatched else y
